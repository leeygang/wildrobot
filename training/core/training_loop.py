"""JIT-compiled PPO trainer.

Architecture:
    train_iteration(state, env_state) -> new_state, metrics
        └── collect_rollout()
        └── compute_gae()
        └── ppo_update_scan()

Usage:
    config = load_training_config("configs/ppo_walking_v0201_smoke.yaml")
    config.freeze()
    train(env_step_fn, env_reset_fn, config)
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from training.configs.training_config import get_robot_config, TrainingConfig

from training.algos.ppo.ppo_core import (
    compute_gae,
    PPOLossMetrics,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)
from training.core.rollout import (
    collect_rollout,
    TrajectoryBatch,
)
from training.core.metrics_registry import (
    aggregate_metrics,
    METRIC_INDEX,
    METRICS_VEC_KEY,
    unpack_metrics,
)
from training.envs.env_info import WR_INFO_KEY
from training.policy_spec_utils import build_policy_spec_from_training_config


# =============================================================================
# Training State
# =============================================================================


class TrainingState(NamedTuple):
    """Complete training state - all JAX arrays, no Python objects."""

    # Network parameters
    policy_params: Any
    value_params: Any
    processor_params: Any

    # Optimizer states
    policy_opt_state: Any
    value_opt_state: Any

    # Training progress
    iteration: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.Array


class IterationMetrics(NamedTuple):
    """Metrics from one training iteration."""

    # PPO losses (computed in training loop)
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    total_loss: jnp.ndarray
    clip_fraction: jnp.ndarray
    approx_kl: jnp.ndarray

    # Episode-level metrics (computed from trajectory)
    episode_reward: jnp.ndarray
    task_reward_mean: jnp.ndarray
    episode_length: jnp.ndarray
    success_rate: jnp.ndarray

    # v0.10.4: Aggregated environment metrics from metrics_registry
    env_metrics: Dict[str, jnp.ndarray]


def _linear_schedule_factor(progress: float, end_factor: float) -> float:
    """Linear factor schedule from 1.0 -> end_factor over progress in [0, 1]."""
    clamped = min(max(progress, 0.0), 1.0)
    return 1.0 + (float(end_factor) - 1.0) * clamped


def _effective_ppo_epochs(
    base_epochs: int,
    previous_approx_kl: float,
    target_kl: float,
    early_stop_multiplier: float,
) -> int:
    """Choose active PPO epochs for this iteration using previous KL signal."""
    if target_kl <= 0.0:
        return max(1, int(base_epochs))
    if previous_approx_kl > float(target_kl) * float(early_stop_multiplier):
        return 1
    return max(1, int(base_epochs))


def _is_eval_better(
    success_rate: float,
    episode_length: float,
    best_success_rate: float,
    best_episode_length: float,
    eps: float = 1e-6,
) -> bool:
    """Lexicographic compare: success_rate first, then episode_length."""
    if success_rate > best_success_rate + eps:
        return True
    if abs(success_rate - best_success_rate) <= eps and episode_length > best_episode_length + eps:
        return True
    return False


def _should_trigger_rollback(
    current_success_rate: float,
    best_success_rate: float,
    threshold: float,
) -> bool:
    """Return True when eval success-rate regressed enough to consider rollback."""
    return current_success_rate < (best_success_rate - threshold)


# =============================================================================
# PPO Update
# =============================================================================


def ppo_update_scan(
    policy_params: Any,
    value_params: Any,
    processor_params: Any,
    policy_opt_state: Any,
    value_opt_state: Any,
    ppo_network: Any,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    obs: jnp.ndarray,
    critic_obs: Optional[jnp.ndarray],
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    rng: jax.Array,
    num_epochs: int,
    active_epochs: int,
    num_minibatches: int,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    update_scale: float,
    target_kl: float,
    kl_early_stop_multiplier: float,
) -> Tuple[Any, Any, Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PPO update using jax.lax.scan over epochs and minibatches."""
    batch_size = obs.shape[0]
    minibatch_size = batch_size // num_minibatches
    num_updates = num_epochs * num_minibatches
    active_epochs = jnp.clip(jnp.asarray(active_epochs, dtype=jnp.int32), 1, num_epochs)
    update_scale = jnp.asarray(update_scale, dtype=jnp.float32)

    rng, perm_rng = jax.random.split(rng)
    epoch_rngs = jax.random.split(perm_rng, num_epochs)
    epoch_perms = jax.vmap(lambda key: jax.random.permutation(key, batch_size))(epoch_rngs)
    target_kl = jnp.asarray(float(target_kl), dtype=jnp.float32)
    kl_stop_threshold = target_kl * jnp.asarray(
        float(kl_early_stop_multiplier), dtype=jnp.float32
    )

    def ppo_step(carry, update_idx):
        policy_p, value_p, policy_opt, value_opt, rng, stop_updates, active_count = carry
        rng, step_rng = jax.random.split(rng)

        # Compute epoch and minibatch indices
        epoch_idx = update_idx // num_minibatches
        mb_idx = update_idx % num_minibatches

        def _active_update(_):
            perm = epoch_perms[epoch_idx]

            start_idx = mb_idx * minibatch_size
            mb_indices = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))

            mb_obs = obs[mb_indices]
            mb_critic_obs = None if critic_obs is None else critic_obs[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            mb_advantages = (mb_advantages - jnp.mean(mb_advantages)) / (
                jnp.std(mb_advantages) + 1e-8
            )

            def loss_fn(policy_p, value_p):
                return compute_ppo_loss(
                    processor_params=processor_params,
                    policy_params=policy_p,
                    value_params=value_p,
                    ppo_network=ppo_network,
                    obs=mb_obs,
                    value_obs=mb_critic_obs,
                    actions=mb_actions,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    rng=step_rng,
                    clip_epsilon=clip_epsilon,
                    value_loss_coef=value_loss_coef,
                    entropy_coef=entropy_coef,
                )

            (_total_loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(policy_p, value_p)

            policy_updates, next_policy_opt = policy_optimizer.update(
                policy_grads, policy_opt
            )
            policy_updates = jax.tree_util.tree_map(
                lambda u: u * update_scale, policy_updates
            )
            next_policy_p = optax.apply_updates(policy_p, policy_updates)

            value_updates, next_value_opt = value_optimizer.update(
                value_grads, value_opt
            )
            value_updates = jax.tree_util.tree_map(
                lambda u: u * update_scale, value_updates
            )
            next_value_p = optax.apply_updates(value_p, value_updates)

            return (
                (
                    next_policy_p,
                    next_value_p,
                    next_policy_opt,
                    next_value_opt,
                    rng,
                    stop_updates,
                    active_count + jnp.asarray(1, dtype=jnp.int32),
                ),
                (metrics, jnp.asarray(1.0, dtype=jnp.float32)),
            )

        def _inactive_update(_):
            zero = jnp.array(0.0, dtype=jnp.float32)
            metrics = PPOLossMetrics(
                total_loss=zero,
                policy_loss=zero,
                value_loss=zero,
                entropy_loss=zero,
                clip_fraction=zero,
                approx_kl=zero,
            )
            return (
                (
                    policy_p,
                    value_p,
                    policy_opt,
                    value_opt,
                    rng,
                    stop_updates,
                    active_count,
                ),
                (metrics, jnp.asarray(0.0, dtype=jnp.float32)),
            )

        is_active = jnp.logical_and(epoch_idx < active_epochs, jnp.logical_not(stop_updates))
        (next_carry, (metrics, active_flag)) = jax.lax.cond(
            is_active, _active_update, _inactive_update, operand=None
        )
        stop_now = jnp.logical_and(
            target_kl > 0.0,
            metrics.approx_kl > kl_stop_threshold,
        )
        updated_stop = jnp.logical_or(next_carry[5], jnp.logical_and(is_active, stop_now))
        next_carry = (
            next_carry[0],
            next_carry[1],
            next_carry[2],
            next_carry[3],
            next_carry[4],
            updated_stop,
            next_carry[6],
        )
        return next_carry, (metrics, active_flag)

    # Run scan
    update_indices = jnp.arange(num_updates)
    init_carry = (
        policy_params,
        value_params,
        policy_opt_state,
        value_opt_state,
        rng,
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final_carry, scan_output = jax.lax.scan(ppo_step, init_carry, update_indices)

    new_policy_params, new_value_params, new_policy_opt, new_value_opt, _, _, total_active = final_carry
    all_metrics, active_mask = scan_output

    active_count = jnp.maximum(jnp.sum(active_mask), 1.0)
    epochs_used = jnp.maximum(
        jnp.ceil(total_active.astype(jnp.float32) / float(num_minibatches)),
        1.0,
    )

    def _active_mean(values):
        return jnp.sum(values * active_mask) / active_count

    return (
        new_policy_params,
        new_value_params,
        new_policy_opt,
        new_value_opt,
        _active_mean(all_metrics.policy_loss),
        _active_mean(all_metrics.value_loss),
        _active_mean(all_metrics.entropy_loss),
        _active_mean(all_metrics.total_loss),
        _active_mean(all_metrics.clip_fraction),
        _active_mean(all_metrics.approx_kl),
        epochs_used,
        total_active.astype(jnp.float32),
    )


# =============================================================================
# Training Iteration
# =============================================================================


def make_train_iteration_fn(
    env_step_fn: Callable,
    ppo_network: Any,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    config: TrainingConfig,
):
    """Create JIT-compiled training iteration function."""

    @jax.jit
    def train_iteration(
        state: TrainingState,
        env_state: Any,
        active_ppo_epochs: int,
        entropy_coef: float,
        learning_rate_scale: float,
    ) -> Tuple[TrainingState, Any, IterationMetrics]:
        """Execute one training iteration (fully on GPU)."""

        rng = state.rng
        rng, rollout_rng, ppo_rng = jax.random.split(rng, 3)

        # ================================================================
        # Step 1: Collect rollout
        # ================================================================
        new_env_state, trajectory = collect_rollout(
            env_step_fn=env_step_fn,
            env_state=env_state,
            policy_params=state.policy_params,
            value_params=state.value_params,
            processor_params=state.processor_params,
            ppo_network=ppo_network,
            rng=rollout_rng,
            num_steps=config.ppo.rollout_steps,
            use_privileged_critic=config.ppo.critic_privileged_enabled,
        )

        total_rewards = trajectory.task_rewards

        # ================================================================
        # Step 2: Compute GAE advantages
        # ================================================================
        advantages, returns = compute_gae(
            rewards=total_rewards,
            values=trajectory.values,
            dones=trajectory.dones,
            bootstrap_value=trajectory.bootstrap_value,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )

        # ================================================================
        # Step 3: PPO update
        # ================================================================
        batch_size = config.ppo.rollout_steps * config.ppo.num_envs

        flat_obs = trajectory.obs.reshape(batch_size, -1)
        flat_critic_obs = None
        if config.ppo.critic_privileged_enabled:
            flat_critic_obs = trajectory.critic_obs.reshape(batch_size, -1)
        flat_actions = trajectory.actions.reshape(batch_size, -1)
        flat_log_probs = trajectory.log_probs.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        (
            new_policy_params,
            new_value_params,
            new_policy_opt,
            new_value_opt,
            policy_loss,
            value_loss,
            entropy_loss,
            total_loss,
            clip_fraction,
            approx_kl,
            epochs_used,
            active_updates,
        ) = ppo_update_scan(
            policy_params=state.policy_params,
            value_params=state.value_params,
            processor_params=state.processor_params,
            policy_opt_state=state.policy_opt_state,
            value_opt_state=state.value_opt_state,
            ppo_network=ppo_network,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            obs=flat_obs,
            critic_obs=flat_critic_obs,
            actions=flat_actions,
            old_log_probs=flat_log_probs,
            advantages=flat_advantages,
            returns=flat_returns,
            rng=ppo_rng,
            num_epochs=config.ppo.epochs,
            active_epochs=active_ppo_epochs,
            num_minibatches=config.ppo.num_minibatches,
            clip_epsilon=config.ppo.clip_epsilon,
            value_loss_coef=config.ppo.value_loss_coef,
            entropy_coef=entropy_coef,
            update_scale=learning_rate_scale,
            target_kl=config.ppo.target_kl,
            kl_early_stop_multiplier=config.ppo.kl_early_stop_multiplier,
        )

        # ================================================================
        # Step 4: Build new state and metrics
        # ================================================================
        env_steps = config.ppo.rollout_steps * config.ppo.num_envs

        # v0.10.4: Use registry-based metrics aggregation
        # Aggregate all metrics from packed vector using registry reducers
        agg_metrics = aggregate_metrics(trajectory.metrics_vec, trajectory.dones)

        # Episode metrics (computed from trajectory, not from metrics_vec)
        episode_reward = jnp.mean(jnp.sum(trajectory.task_rewards, axis=0))
        task_reward_mean = jnp.mean(trajectory.task_rewards)

        # v0.10.4: Episode length = mean step count of COMPLETED episodes only
        # Use episode_step_count from metrics_vec (preserved through auto-reset)
        # This is more reliable than wr_info.step_count which resets to 0
        ep_step_idx = METRIC_INDEX["episode_step_count"]
        episode_step_counts = trajectory.metrics_vec[..., ep_step_idx]  # (T, N)
        completed_episode_lengths = episode_step_counts * trajectory.dones  # Zero out non-terminal steps
        total_completed_length = jnp.sum(completed_episode_lengths)
        total_done = jnp.sum(trajectory.dones)
        episode_length = jnp.where(
            total_done > 0,
            total_completed_length / total_done,
            0.0,  # No episodes completed this rollout
        )

        # Success rate
        total_truncated = jnp.sum(trajectory.truncations)
        success_rate = jnp.where(
            total_done > 0,
            total_truncated / total_done,
            0.0,
        )

        # v0.10.2: Termination diagnostics - compute fraction of episodes ending due to each cause
        # Access term/* metrics from aggregated dict and normalize by total done episodes
        # Note: aggregate_metrics uses MEAN reducer, but we need SUM then normalize
        # So we access the raw metrics_vec and compute manually
        term_idx_height_low = METRIC_INDEX["term/height_low"]
        term_idx_height_high = METRIC_INDEX["term/height_high"]
        term_idx_pitch = METRIC_INDEX["term/pitch"]
        term_idx_roll = METRIC_INDEX["term/roll"]
        term_idx_truncated = METRIC_INDEX["term/truncated"]

        total_term_height_low = jnp.sum(trajectory.metrics_vec[..., term_idx_height_low])
        total_term_height_high = jnp.sum(trajectory.metrics_vec[..., term_idx_height_high])
        total_term_pitch = jnp.sum(trajectory.metrics_vec[..., term_idx_pitch])
        total_term_roll = jnp.sum(trajectory.metrics_vec[..., term_idx_roll])
        total_term_truncated = jnp.sum(trajectory.metrics_vec[..., term_idx_truncated])

        # Normalize by total done episodes (fraction of terminations by cause)
        term_height_low_frac = jnp.where(total_done > 0, total_term_height_low / total_done, 0.0)
        term_height_high_frac = jnp.where(total_done > 0, total_term_height_high / total_done, 0.0)
        term_pitch_frac = jnp.where(total_done > 0, total_term_pitch / total_done, 0.0)
        term_roll_frac = jnp.where(total_done > 0, total_term_roll / total_done, 0.0)
        term_truncated_frac = jnp.where(total_done > 0, total_term_truncated / total_done, 0.0)

        # v0.11.4: Failure-mode debug metrics (condition on pitch terminations)
        def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            denom = jnp.sum(mask)
            return jnp.where(denom > 0, jnp.sum(values * mask) / denom, 0.0)

        def _masked_max(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            masked = jnp.where(mask > 0, values, -jnp.inf)
            m = jnp.max(masked)
            return jnp.where(jnp.isfinite(m), m, 0.0)

        pitch_term_mask = trajectory.metrics_vec[..., term_idx_pitch]
        pitch_done_mask = trajectory.dones * pitch_term_mask
        pitch_done_count = jnp.sum(pitch_done_mask)

        pitch_val = trajectory.metrics_vec[..., METRIC_INDEX["term/pitch_val"]]
        pitch_abs = jnp.abs(pitch_val)
        pitch_rate_abs = jnp.abs(
            trajectory.metrics_vec[..., METRIC_INDEX["debug/pitch_rate"]]
        )
        action_abs_max = trajectory.metrics_vec[..., METRIC_INDEX["debug/action_abs_max"]]
        action_sat_frac = trajectory.metrics_vec[..., METRIC_INDEX["debug/action_sat_frac"]]
        raw_action_abs_max = trajectory.metrics_vec[
            ..., METRIC_INDEX["debug/raw_action_abs_max"]
        ]
        raw_action_sat_frac = trajectory.metrics_vec[
            ..., METRIC_INDEX["debug/raw_action_sat_frac"]
        ]
        max_torque_ratio = trajectory.metrics_vec[..., METRIC_INDEX["tracking/max_torque"]]
        torque_abs_max = trajectory.metrics_vec[..., METRIC_INDEX["debug/torque_abs_max"]]
        torque_sat_frac = trajectory.metrics_vec[..., METRIC_INDEX["debug/torque_sat_frac"]]

        agg_metrics["debug/pitch_term/count"] = pitch_done_count
        agg_metrics["debug/pitch_term/pitch_abs_mean"] = _masked_mean(
            pitch_abs, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/pitch_abs_max"] = _masked_max(
            pitch_abs, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/pitch_rate_abs_mean"] = _masked_mean(
            pitch_rate_abs, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/pitch_rate_abs_max"] = _masked_max(
            pitch_rate_abs, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/max_torque_mean"] = _masked_mean(
            max_torque_ratio, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/max_torque_max"] = _masked_max(
            max_torque_ratio, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/action_abs_max_mean"] = _masked_mean(
            action_abs_max, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/action_abs_max_max"] = _masked_max(
            action_abs_max, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/action_sat_frac_mean"] = _masked_mean(
            action_sat_frac, pitch_done_mask
        )

        # Contact state at pitch termination (toe/heel switches averaged over pitch-done steps)
        lt = trajectory.metrics_vec[..., METRIC_INDEX["debug/left_toe_switch"]]
        lh = trajectory.metrics_vec[..., METRIC_INDEX["debug/left_heel_switch"]]
        rt = trajectory.metrics_vec[..., METRIC_INDEX["debug/right_toe_switch"]]
        rh = trajectory.metrics_vec[..., METRIC_INDEX["debug/right_heel_switch"]]
        agg_metrics["debug/pitch_term/left_toe_switch_mean"] = _masked_mean(
            lt, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/left_heel_switch_mean"] = _masked_mean(
            lh, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/right_toe_switch_mean"] = _masked_mean(
            rt, pitch_done_mask
        )
        agg_metrics["debug/pitch_term/right_heel_switch_mean"] = _masked_mean(
            rh, pitch_done_mask
        )

        # v0.11.4: "last-step" pitch termination metrics (no dones mask)
        # These reflect the step where term/pitch is true (pre-reset snapshot).
        agg_metrics["debug/pitch_term/last_pitch_abs"] = _masked_max(
            pitch_abs, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_pitch_rate_abs"] = _masked_max(
            pitch_rate_abs, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_max_torque"] = _masked_max(
            max_torque_ratio, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_action_abs_max"] = _masked_max(
            action_abs_max, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_action_sat_frac"] = _masked_max(
            action_sat_frac, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_raw_action_abs_max"] = _masked_max(
            raw_action_abs_max, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_raw_action_sat_frac"] = _masked_max(
            raw_action_sat_frac, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_torque_abs_max"] = _masked_max(
            torque_abs_max, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_torque_sat_frac"] = _masked_max(
            torque_sat_frac, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_left_toe_switch"] = _masked_max(
            lt, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_left_heel_switch"] = _masked_max(
            lh, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_right_toe_switch"] = _masked_max(
            rt, pitch_term_mask
        )
        agg_metrics["debug/pitch_term/last_right_heel_switch"] = _masked_max(
            rh, pitch_term_mask
        )

        # v0.10.4: Enrich agg_metrics with computed episode-level metrics
        # This replaces individual field unpacking with a single dict
        agg_metrics["episode_length"] = episode_length
        agg_metrics["success_rate"] = success_rate
        agg_metrics["term_height_low_frac"] = term_height_low_frac
        agg_metrics["term_height_high_frac"] = term_height_high_frac
        agg_metrics["term_pitch_frac"] = term_pitch_frac
        agg_metrics["term_roll_frac"] = term_roll_frac
        agg_metrics["term_truncated_frac"] = term_truncated_frac
        agg_metrics["ppo/epochs_used"] = epochs_used
        agg_metrics["ppo/early_stop_epoch"] = jnp.where(
            epochs_used < float(config.ppo.epochs),
            epochs_used,
            0.0,
        )
        agg_metrics["ppo/active_updates"] = active_updates

        new_state = TrainingState(
            policy_params=new_policy_params,
            value_params=new_value_params,
            processor_params=state.processor_params,
            policy_opt_state=new_policy_opt,
            value_opt_state=new_value_opt,
            iteration=state.iteration + 1,
            total_steps=state.total_steps + env_steps,
            rng=rng,
        )

        metrics = IterationMetrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            total_loss=total_loss,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
            episode_reward=episode_reward,
            task_reward_mean=task_reward_mean,
            episode_length=episode_length,
            success_rate=success_rate,
            # v0.10.4: All env metrics in single dict
            env_metrics=agg_metrics,
        )

        return new_state, new_env_state, metrics

    return train_iteration


# =============================================================================
# Resume Validation
# =============================================================================


def _validate_resume_checkpoint(
    ckpt_metrics: Dict[str, Any],
    ckpt_config: Dict[str, Any],
    config: TrainingConfig,
    *,
    current_policy_spec_hash: str,
) -> None:
    """Log checkpoint info and validate compatibility with current config.

    Raises ValueError on hard mismatches.
    """
    hard_mismatch = []
    warn_mismatch = []

    def _format_metric(value: Any, fmt: str, suffix: str = "") -> str:
        if isinstance(value, (int, float)):
            return f"{value:{fmt}}{suffix}"
        return f"N/A{suffix}"

    if ckpt_metrics:
        print(
            f"  Checkpoint reward: "
            f"{_format_metric(ckpt_metrics.get('episode_reward'), '.2f')}"
        )
        print(
            f"  Checkpoint velocity: "
            f"{_format_metric(ckpt_metrics.get('forward_velocity'), '.2f', ' m/s')}"
        )
        print(
            f"  Checkpoint height: "
            f"{_format_metric(ckpt_metrics.get('robot_height'), '.3f', ' m')}"
        )
        print(
            f"  Checkpoint ep_len: "
            f"{_format_metric(ckpt_metrics.get('episode_length'), '.0f')}"
        )
    if ckpt_config:
        print(
            f"  Original config: num_envs={ckpt_config.get('num_envs', 'N/A')}, "
            f"lr={ckpt_config.get('learning_rate', 'N/A')}"
        )

    def _check_mismatch(key: str, ckpt_value: Any, current_value: Any) -> None:
        if ckpt_value != current_value:
            hard_mismatch.append((key, ckpt_value, current_value))

    def _warn_if_mismatch(key: str, ckpt_value: Any, current_value: Any) -> None:
        if ckpt_value != current_value:
            warn_mismatch.append(
                f"{key}: ckpt={ckpt_value}, current={current_value}"
            )

    _check_mismatch("num_envs", ckpt_config.get("num_envs"), config.ppo.num_envs)
    _check_mismatch(
        "rollout_steps", ckpt_config.get("rollout_steps"), config.ppo.rollout_steps
    )
    _check_mismatch(
        "num_minibatches", ckpt_config.get("num_minibatches"), config.ppo.num_minibatches
    )
    _check_mismatch("epochs", ckpt_config.get("epochs"), config.ppo.epochs)
    _check_mismatch(
        "critic_privileged_enabled",
        ckpt_config.get("critic_privileged_enabled", False),
        config.ppo.critic_privileged_enabled,
    )
    _warn_if_mismatch(
        "learning_rate", ckpt_config.get("learning_rate"), config.ppo.learning_rate
    )
    _check_mismatch(
        "clip_epsilon", ckpt_config.get("clip_epsilon"), config.ppo.clip_epsilon
    )
    _check_mismatch(
        "value_loss_coef", ckpt_config.get("value_loss_coef"), config.ppo.value_loss_coef
    )
    _warn_if_mismatch(
        "entropy_coef", ckpt_config.get("entropy_coef"), config.ppo.entropy_coef
    )
    _check_mismatch(
        "max_grad_norm", ckpt_config.get("max_grad_norm"), config.ppo.max_grad_norm
    )
    _check_mismatch("gamma", ckpt_config.get("gamma"), config.ppo.gamma)
    _check_mismatch("gae_lambda", ckpt_config.get("gae_lambda"), config.ppo.gae_lambda)
    _check_mismatch(
        "actor_hidden_sizes",
        ckpt_config.get("actor_hidden_sizes"),
        config.networks.actor.hidden_sizes,
    )
    _check_mismatch(
        "critic_hidden_sizes",
        ckpt_config.get("critic_hidden_sizes"),
        config.networks.critic.hidden_sizes,
    )
    if ckpt_config:
        if (
            ckpt_config.get("actor_hidden_sizes") is None
            or ckpt_config.get("critic_hidden_sizes") is None
        ):
            warn_mismatch.append("network_arch_missing_in_ckpt")

    # Policy contract / observation-action semantics guard:
    # If the policy contract changed, resuming will silently break behavior.
    ckpt_policy_spec_hash = ckpt_config.get("policy_spec_hash")
    if not ckpt_policy_spec_hash:
        raise ValueError(
            "Resume checkpoint is missing policy_contract fingerprint (policy_spec_hash).\n"
            "This checkpoint was created before the policy_contract migration, so resuming is unsafe.\n"
            "Start a fresh run (no --resume) to train under the current contract."
        )
    if ckpt_policy_spec_hash != current_policy_spec_hash:
        raise ValueError(
            "Resume checkpoint policy_contract mismatch (hard):\n"
            f"  - policy_spec_hash: ckpt={ckpt_policy_spec_hash}, current={current_policy_spec_hash}\n"
            "Start a fresh run (no --resume) to train under the current contract."
        )

    if hard_mismatch:
        mismatch_lines = [
            f"  - {key}: ckpt={ckpt_value}, current={current_value}"
            for key, ckpt_value, current_value in hard_mismatch
        ]
        raise ValueError(
            "Resume checkpoint config mismatch (hard):\n"
            + "\n".join(mismatch_lines)
            + "\nUse a compatible config or start from scratch."
        )
    if warn_mismatch:
        print(f"  ⚠️  Resume config warning: {', '.join(warn_mismatch)}")


# =============================================================================
# Main Training Function
# =============================================================================


def train(
    env_step_fn: Callable,
    env_reset_fn: Callable,
    config: TrainingConfig,
    callback: Optional[
        Callable[[int, TrainingState, IterationMetrics, float], None]
    ] = None,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
    eval_env_step_fn: Optional[Callable] = None,
    eval_env_step_fn_no_push: Optional[Callable] = None,
    eval_env_reset_fn: Optional[Callable] = None,
    *,
    policy_init_action: Optional[jnp.ndarray] = None,
) -> TrainingState:
    """Main PPO training function.

    Args:
        env_step_fn: Batched environment step function
        env_reset_fn: Batched environment reset function
        config: Training configuration
        callback: Optional callback for logging/checkpointing
        resume_checkpoint: Optional checkpoint data to resume from (from load_checkpoint)
        eval_env_step_fn_no_push: Optional eval step fn with disturbances disabled

    Returns:
        Final training state
    """
    robot_config = get_robot_config()
    spec = build_policy_spec_from_training_config(
        training_cfg=config,
        robot_cfg=robot_config,
    )
    obs_dim = spec.model.obs_dim
    action_dim = spec.model.action_dim
    critic_obs_dim = obs_dim
    if bool(config.ppo.critic_privileged_enabled):
        from training.envs.env_info import PRIVILEGED_OBS_DIM

        critic_obs_dim = PRIVILEGED_OBS_DIM
    from policy_contract.spec import policy_spec_hash

    current_spec_hash = policy_spec_hash(spec)

    print("=" * 60)
    print("PPO Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  critic_obs_dim: {critic_obs_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  num_envs: {config.ppo.num_envs}")
    print(f"  rollout_steps: {config.ppo.rollout_steps}")
    print(f"  iterations: {config.ppo.iterations}")
    print("=" * 60)

    # Initialize RNG
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # Create networks.  Brax's make_ppo_networks takes a single shared
    # activation; the v0.20.1 spec (walking_training.md §"Smoke policy
    # initialisation", was `G6`) uses ELU for
    # both actor and critic, so we require them to match in the YAML
    # rather than silently picking one.
    actor_activation = str(config.networks.actor.activation).lower()
    critic_activation = str(config.networks.critic.activation).lower()
    if actor_activation != critic_activation:
        raise ValueError(
            f"actor.activation ({actor_activation!r}) and "
            f"critic.activation ({critic_activation!r}) must match — "
            "Brax's make_ppo_networks shares one activation across both."
        )
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=config.networks.actor.hidden_sizes,
        value_hidden_dims=config.networks.critic.hidden_sizes,
        critic_obs_dim=critic_obs_dim,
        activation=actor_activation,
    )

    # Initialize network parameters.  policy_init_std is exp(log_std_init)
    # so the actor's softplus-parameterized scale starts at the spec's
    # exploration pressure (smoke policy init spec, was `G6`:
    # walking_training.md v0.20.1 §, log_std_init
    # = -1.0 → std ≈ 0.368).  Without this thread the scale would default
    # to 0.10 (~ log_std=-2.30) regardless of the YAML.
    policy_init_std = float(jnp.exp(jnp.float32(config.networks.actor.log_std_init)))
    processor_params, policy_params, value_params = init_network_params(
        ppo_network,
        obs_dim,
        action_dim,
        seed=int(init_rng[0]),
        policy_init_action=policy_init_action,
        policy_init_std=policy_init_std,
    )

    # Create optimizers
    total_schedule_updates = max(
        1,
        int(config.ppo.iterations)
        * int(config.ppo.epochs)
        * int(config.ppo.num_minibatches),
    )
    lr_schedule = optax.linear_schedule(
        init_value=float(config.ppo.learning_rate),
        end_value=float(config.ppo.learning_rate) * float(config.ppo.lr_schedule_end_factor),
        transition_steps=total_schedule_updates,
    )
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(lr_schedule),
    )
    value_optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(lr_schedule),
    )

    # Initialize optimizer states
    policy_opt_state = policy_optimizer.init(policy_params)
    value_opt_state = value_optimizer.init(value_params)

    # Create initial training state or restore from checkpoint
    start_iteration = 0
    if resume_checkpoint is not None:
        # Log checkpoint info
        ckpt_iteration = resume_checkpoint.get("iteration", 0)
        ckpt_steps = resume_checkpoint.get("total_steps", 0)
        ckpt_metrics = resume_checkpoint.get("metrics", {})
        ckpt_config = resume_checkpoint.get("config", {})

        print("=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        print(f"  Checkpoint iteration: {ckpt_iteration}")
        print(f"  Checkpoint steps: {ckpt_steps:,}")
        _validate_resume_checkpoint(
            ckpt_metrics,
            ckpt_config,
            config,
            current_policy_spec_hash=current_spec_hash,
        )
        print("=" * 60)

        # Restore state from checkpoint
        state = TrainingState(
            policy_params=resume_checkpoint["policy_params"],
            value_params=resume_checkpoint["value_params"],
            processor_params=resume_checkpoint.get("processor_params", processor_params),
            policy_opt_state=resume_checkpoint["policy_opt_state"],
            value_opt_state=resume_checkpoint["value_opt_state"],
            iteration=jnp.array(ckpt_iteration, dtype=jnp.int32),
            total_steps=jnp.array(ckpt_steps, dtype=jnp.int32),
            rng=resume_checkpoint.get("rng", rng),
        )
        start_iteration = ckpt_iteration
        print(f"✓ Restored training state from iteration {ckpt_iteration}")
    else:
        # Create fresh training state
        state = TrainingState(
            policy_params=policy_params,
            value_params=value_params,
            processor_params=processor_params,
            policy_opt_state=policy_opt_state,
            value_opt_state=value_opt_state,
            iteration=jnp.array(0, dtype=jnp.int32),
            total_steps=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )

    # Initialize environment
    env_state = env_reset_fn(env_rng)
    print(f"✓ Environment initialized")

    # Create JIT-compiled training function
    train_iteration_fn = make_train_iteration_fn(
        env_step_fn=env_step_fn,
        ppo_network=ppo_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        config=config,
    )

    eval_step_push_fn = eval_env_step_fn or env_step_fn
    eval_step_clean_fn = eval_env_step_fn_no_push or eval_step_push_fn
    eval_reset_fn = eval_env_reset_fn or env_reset_fn
    eval_enabled = config.ppo.eval.enabled and config.ppo.eval.interval > 0
    eval_steps = config.ppo.eval.num_steps or config.ppo.rollout_steps
    use_eval_survival_metric = int(eval_steps) < int(config.env.max_episode_steps)
    eval_base_rng = jax.random.PRNGKey(config.seed + config.ppo.eval.seed_offset)

    # Note: Training-time eval runs two passes for efficiency:
    # - eval_push: Uses training-configured push settings
    # - eval_clean: No pushes
    # For comprehensive ladder evaluation (clean, easy, medium, hard, hard_long),
    # use the standalone script: training/eval/eval_ladder_v0170.py

    def _make_eval_runner(step_fn: Callable):
        @jax.jit
        def _runner(
            policy_params: Any,
            processor_params: Any,
            eval_rng: jax.Array,
        ) -> Tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ]:
            eval_env_state = eval_reset_fn(eval_rng)
            # ``state.metrics`` is now a packed metrics_vec (see
            # metrics_registry.METRICS_VEC_KEY); the legacy code path read
            # a per-key dict that no longer exists.  Slice the height
            # column from the packed vec.
            height_idx = METRIC_INDEX["height"]
            reset_h = eval_env_state.metrics[METRICS_VEC_KEY][..., height_idx]
            reset_h_mean = jnp.mean(reset_h)
            reset_h_min = jnp.min(reset_h)

            def eval_step(carry, _):
                env_state, rng = carry
                rng, action_rng = jax.random.split(rng)
                obs = env_state.obs
                action, _, _ = sample_actions(
                    processor_params,
                    policy_params,
                    ppo_network,
                    obs,
                    action_rng,
                    deterministic=config.ppo.eval.deterministic,
                )
                next_env_state = step_fn(env_state, action)
                wr_info = next_env_state.info[WR_INFO_KEY]
                step_data = {
                    "reward": next_env_state.reward,
                    "done": next_env_state.done,
                    "truncation": wr_info.truncated,
                    "metrics_vec": next_env_state.metrics[METRICS_VEC_KEY],
                }
                return (next_env_state, rng), step_data

            (_, _), eval_rollout = jax.lax.scan(
                eval_step, (eval_env_state, eval_rng), None, length=eval_steps
            )

            episode_reward = jnp.mean(jnp.sum(eval_rollout["reward"], axis=0))
            term_idx_height_low = METRIC_INDEX["term/height_low"]
            term_idx_height_high = METRIC_INDEX["term/height_high"]
            term_idx_pitch = METRIC_INDEX["term/pitch"]
            term_idx_roll = METRIC_INDEX["term/roll"]
            unnecessary_step_idx = METRIC_INDEX["unnecessary_step_rate"]
            recovery_completed_idx = METRIC_INDEX["recovery/completed"]
            recovery_no_touchdown_idx = METRIC_INDEX["recovery/no_touchdown_frac"]
            recovery_touchdown_then_fail_idx = METRIC_INDEX["recovery/touchdown_then_fail_frac"]
            recovery_touchdown_count_idx = METRIC_INDEX["recovery/touchdown_count"]
            recovery_first_liftoff_idx = METRIC_INDEX["recovery/first_liftoff_latency"]
            recovery_first_touchdown_idx = METRIC_INDEX["recovery/first_touchdown_latency"]
            recovery_pitch_reduction_idx = METRIC_INDEX["recovery/pitch_rate_reduction_10t"]
            recovery_capture_reduction_idx = METRIC_INDEX["recovery/capture_error_reduction_10t"]
            recovery_touchdown_to_term_idx = METRIC_INDEX["recovery/touchdown_to_term_steps"]
            recovery_visible_step_idx = METRIC_INDEX["recovery/visible_step_rate"]
            visible_step_rate_hard_idx = METRIC_INDEX["visible_step_rate_hard"]
            recovery_min_height_idx = METRIC_INDEX["recovery/min_height"]
            recovery_max_knee_flex_idx = METRIC_INDEX["recovery/max_knee_flex"]
            recovery_first_step_dist_abs_idx = METRIC_INDEX["recovery/first_step_dist_abs"]
            teacher_clean_step_count_idx = METRIC_INDEX["teacher/clean_step_count"]
            teacher_push_step_count_idx = METRIC_INDEX["teacher/push_step_count"]
            teacher_whole_body_active_frac_idx = METRIC_INDEX["teacher/whole_body_active_frac"]
            teacher_whole_body_active_count_idx = METRIC_INDEX["teacher/whole_body_active_count"]
            teacher_whole_body_active_during_clean_frac_idx = METRIC_INDEX[
                "teacher/whole_body_active_during_clean_frac"
            ]
            teacher_whole_body_active_during_push_frac_idx = METRIC_INDEX[
                "teacher/whole_body_active_during_push_frac"
            ]
            teacher_recovery_height_target_mean_idx = METRIC_INDEX[
                "teacher/recovery_height_target_mean"
            ]
            teacher_recovery_height_error_idx = METRIC_INDEX["teacher/recovery_height_error"]
            teacher_recovery_height_in_band_frac_idx = METRIC_INDEX[
                "teacher/recovery_height_in_band_frac"
            ]
            teacher_com_velocity_target_mean_idx = METRIC_INDEX[
                "teacher/com_velocity_target_mean"
            ]
            teacher_com_velocity_error_idx = METRIC_INDEX["teacher/com_velocity_error"]
            teacher_com_velocity_target_hit_frac_idx = METRIC_INDEX[
                "teacher/com_velocity_target_hit_frac"
            ]
            reward_teacher_recovery_height_idx = METRIC_INDEX["reward/teacher_recovery_height"]
            reward_teacher_com_velocity_reduction_idx = METRIC_INDEX[
                "reward/teacher_com_velocity_reduction"
            ]

            total_done = jnp.sum(eval_rollout["done"])
            done_by_env = jnp.any(eval_rollout["done"] > 0, axis=0)
            done_env_frac = jnp.mean(done_by_env.astype(jnp.float32))
            # "Survival" means: did not end due to failure during this eval horizon.
            # We treat truncation (max steps) as success, and any other done as failure.
            trunc_by_env = jnp.any(eval_rollout["truncation"] > 0, axis=0)
            failure_done_by_env = done_by_env & (~trunc_by_env)
            survival_rate = 1.0 - jnp.mean(failure_done_by_env.astype(jnp.float32))

            # Survival steps: first done step (1-indexed), or eval_steps if never done.
            done_mask = eval_rollout["done"] > 0  # (T, N)
            first_done_idx = jnp.argmax(done_mask.astype(jnp.int32), axis=0)  # 0 if none
            survived_steps = jnp.where(
                done_by_env,
                first_done_idx.astype(jnp.float32) + 1.0,
                jnp.asarray(float(eval_steps), dtype=jnp.float32),
            )
            survival_steps = jnp.mean(survived_steps)
            completed_lengths = (
                eval_rollout["metrics_vec"][..., METRIC_INDEX["episode_step_count"]]
                * eval_rollout["done"]
            )
            episode_length = jnp.where(
                total_done > 0,
                jnp.sum(completed_lengths) / total_done,
                0.0,
            )

            total_truncated = jnp.sum(eval_rollout["truncation"])
            truncated_by_env = jnp.any(eval_rollout["truncation"] > 0, axis=0)
            trunc_env_frac = jnp.mean(truncated_by_env.astype(jnp.float32))
            success_rate = jnp.where(total_done > 0, total_truncated / total_done, 0.0)
            term_height_low_frac = jnp.where(
                total_done > 0,
                jnp.sum(eval_rollout["metrics_vec"][..., term_idx_height_low]) / total_done,
                0.0,
            )
            term_height_high_frac = jnp.where(
                total_done > 0,
                jnp.sum(eval_rollout["metrics_vec"][..., term_idx_height_high]) / total_done,
                0.0,
            )
            term_pitch_frac = jnp.where(
                total_done > 0,
                jnp.sum(eval_rollout["metrics_vec"][..., term_idx_pitch]) / total_done,
                0.0,
            )
            term_roll_frac = jnp.where(
                total_done > 0,
                jnp.sum(eval_rollout["metrics_vec"][..., term_idx_roll]) / total_done,
                0.0,
            )
            unnecessary_step_rate = jnp.mean(
                eval_rollout["metrics_vec"][..., unnecessary_step_idx]
            )
            recovery_completed = jnp.sum(
                eval_rollout["metrics_vec"][..., recovery_completed_idx]
            )
            recovery_completed_safe = jnp.maximum(recovery_completed, 1.0)
            recovery_has_completed = recovery_completed > 0.0
            recovery_no_touchdown_frac = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_no_touchdown_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_touchdown_then_fail_frac = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_touchdown_then_fail_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_touchdown_count = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_touchdown_count_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_first_liftoff_latency = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_first_liftoff_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_first_touchdown_latency = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_first_touchdown_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_pitch_rate_reduction_10t = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_pitch_reduction_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_capture_error_reduction_10t = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_capture_reduction_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_touchdown_to_term_steps = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_touchdown_to_term_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_visible_step_rate = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_visible_step_idx])
                / recovery_completed_safe,
                0.0,
            )
            visible_step_rate_hard = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., visible_step_rate_hard_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_min_height = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_min_height_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_max_knee_flex = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_max_knee_flex_idx])
                / recovery_completed_safe,
                0.0,
            )
            recovery_first_step_dist_abs = jnp.where(
                recovery_has_completed,
                jnp.sum(eval_rollout["metrics_vec"][..., recovery_first_step_dist_abs_idx])
                / recovery_completed_safe,
                0.0,
            )
            teacher_clean_step_count = jnp.sum(
                eval_rollout["metrics_vec"][..., teacher_clean_step_count_idx]
            )
            teacher_push_step_count = jnp.sum(
                eval_rollout["metrics_vec"][..., teacher_push_step_count_idx]
            )
            teacher_whole_body_active_count = jnp.sum(
                eval_rollout["metrics_vec"][..., teacher_whole_body_active_count_idx]
            )
            teacher_clean_step_count_safe = jnp.maximum(teacher_clean_step_count, 1.0)
            teacher_push_step_count_safe = jnp.maximum(teacher_push_step_count, 1.0)
            teacher_whole_body_active_count_safe = jnp.maximum(
                teacher_whole_body_active_count, 1.0
            )
            teacher_whole_body_active_frac = jnp.mean(
                eval_rollout["metrics_vec"][..., teacher_whole_body_active_frac_idx]
            )
            teacher_whole_body_active_during_clean_frac = jnp.where(
                teacher_clean_step_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][
                        ..., teacher_whole_body_active_during_clean_frac_idx
                    ]
                )
                / teacher_clean_step_count_safe,
                0.0,
            )
            teacher_whole_body_active_during_push_frac = jnp.where(
                teacher_push_step_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][
                        ..., teacher_whole_body_active_during_push_frac_idx
                    ]
                )
                / teacher_push_step_count_safe,
                0.0,
            )
            teacher_recovery_height_target_mean = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][..., teacher_recovery_height_target_mean_idx]
                )
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            teacher_recovery_height_error = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(eval_rollout["metrics_vec"][..., teacher_recovery_height_error_idx])
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            teacher_recovery_height_in_band_frac = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][..., teacher_recovery_height_in_band_frac_idx]
                )
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            teacher_com_velocity_target_mean = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][..., teacher_com_velocity_target_mean_idx]
                )
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            teacher_com_velocity_error = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(eval_rollout["metrics_vec"][..., teacher_com_velocity_error_idx])
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            teacher_com_velocity_target_hit_frac = jnp.where(
                teacher_whole_body_active_count > 0.0,
                jnp.sum(
                    eval_rollout["metrics_vec"][..., teacher_com_velocity_target_hit_frac_idx]
                )
                / teacher_whole_body_active_count_safe,
                0.0,
            )
            reward_teacher_recovery_height = jnp.mean(
                eval_rollout["metrics_vec"][..., reward_teacher_recovery_height_idx]
            )
            reward_teacher_com_velocity_reduction = jnp.mean(
                eval_rollout["metrics_vec"][..., reward_teacher_com_velocity_reduction_idx]
            )

            # ----- Walking eval block (v0.20.1 ToddlerBot-style) -----
            # Deterministic eval rollout already runs above; we just pull the
            # walking-relevant slices from the same metrics_vec and report
            # them under the ``Evaluate/*`` namespace (matches ToddlerBot
            # train_mjx.py log_metrics convention -- no success_rate concept;
            # checkpoint quality reads off mean eval reward + ep length +
            # per-term tracking errors).  Indices resolved at trace time.
            walking_idx = {
                "forward_velocity": METRIC_INDEX["forward_velocity"],
                "cmd_vs_achieved_forward": METRIC_INDEX[
                    "tracking/cmd_vs_achieved_forward"
                ],
                "forward_velocity_cmd_ratio": METRIC_INDEX[
                    "tracking/forward_velocity_cmd_ratio"
                ],
                "step_length_touchdown_event_m": METRIC_INDEX[
                    "tracking/step_length_touchdown_event_m"
                ],
                # Per-reward-term tracking signals (ToddlerBot pattern: each
                # ``Evaluate/<term>`` directly carries the deterministic eval
                # rollout's exp(-α·error²) value for that term).  These are
                # what the v0.20.1 G4 promotion-horizon gate floors on.
                "ref_q_track": METRIC_INDEX["reward/ref_q_track"],
                "ref_body_quat_track": METRIC_INDEX["reward/ref_body_quat_track"],
                "torso_pos_xy": METRIC_INDEX["reward/torso_pos_xy"],
                # v0.20.2 smoke6: TB-aligned continuous phase signals.
                "lin_vel_z": METRIC_INDEX["reward/lin_vel_z"],
                "ang_vel_xy": METRIC_INDEX["reward/ang_vel_xy"],
                "cmd_forward_velocity_track": METRIC_INDEX[
                    "reward/cmd_forward_velocity_track"
                ],
                # Raw error diagnostics (paired with the reward terms above).
                "ref_q_track_err_rmse": METRIC_INDEX["ref/q_track_err_rmse"],
                "ref_body_quat_err_deg": METRIC_INDEX["ref/body_quat_err_deg"],
                "ref_feet_pos_err_l2": METRIC_INDEX["ref/feet_pos_err_l2"],
                "ref_feet_pos_track_raw": METRIC_INDEX["ref/feet_pos_track_raw"],
                "ref_torso_pos_xy_err_m": METRIC_INDEX["ref/torso_pos_xy_err_m"],
                # v0.20.2 smoke6: TB-aligned phase-signal diagnostics.
                "ref_lin_vel_z_err_m_s": METRIC_INDEX["ref/lin_vel_z_err_m_s"],
                "ref_ang_vel_xy_err_rad_s": METRIC_INDEX[
                    "ref/ang_vel_xy_err_rad_s"
                ],
                "ref_contact_phase_match": METRIC_INDEX["ref/contact_phase_match"],
            }
            walking_metrics = {
                key: jnp.mean(eval_rollout["metrics_vec"][..., idx])
                for key, idx in walking_idx.items()
            }
            return (
                episode_reward,
                success_rate,
                episode_length,
                term_height_low_frac,
                term_height_high_frac,
                term_pitch_frac,
                term_roll_frac,
                done_env_frac,
                trunc_env_frac,
                survival_rate,
                survival_steps,
                reset_h_mean,
                reset_h_min,
                unnecessary_step_rate,
                recovery_completed,
                recovery_no_touchdown_frac,
                recovery_touchdown_then_fail_frac,
                recovery_touchdown_count,
                recovery_first_liftoff_latency,
                recovery_first_touchdown_latency,
                recovery_pitch_rate_reduction_10t,
                recovery_capture_error_reduction_10t,
                recovery_touchdown_to_term_steps,
                recovery_visible_step_rate,
                visible_step_rate_hard,
                recovery_min_height,
                recovery_max_knee_flex,
                recovery_first_step_dist_abs,
                teacher_whole_body_active_frac,
                teacher_whole_body_active_during_clean_frac,
                teacher_whole_body_active_during_push_frac,
                teacher_recovery_height_target_mean,
                teacher_recovery_height_error,
                teacher_recovery_height_in_band_frac,
                teacher_com_velocity_target_mean,
                teacher_com_velocity_error,
                teacher_com_velocity_target_hit_frac,
                reward_teacher_recovery_height,
                reward_teacher_com_velocity_reduction,
                walking_metrics,
            )

        return _runner

    run_eval_push = _make_eval_runner(eval_step_push_fn)
    eval_has_clean_pass = eval_step_clean_fn is not eval_step_push_fn
    run_eval_clean = (
        _make_eval_runner(eval_step_clean_fn) if eval_has_clean_pass else run_eval_push
    )

    if eval_enabled and use_eval_survival_metric:
        print(
            "WARNING: ppo.eval.num_steps < env.max_episode_steps. "
            "eval_*/success_rate is truncation-based, so it will be ~0 unless episodes end early. "
            "Use eval_*/survival_rate and eval_*/survival_steps for short-horizon probes. "
            f"(eval.num_steps={int(config.ppo.eval.num_steps)}, max_episode_steps={int(config.env.max_episode_steps)})"
        )

    print("=" * 60)
    print("Pre-compiling JIT functions...")
    compile_start = time.time()
    _ = train_iteration_fn(
        state,
        env_state,
        config.ppo.epochs,
        config.ppo.entropy_coef,
        1.0,
    )
    jax.block_until_ready(_)
    print(f"  ✓ train_iteration_fn compiled ({time.time() - compile_start:.1f}s)")
    if eval_enabled:
        # Eval compilation can be very expensive (full rollout scan + metrics), so
        # compile it lazily on first eval call instead of blocking startup.
        print("  eval runners will JIT-compile on first eval call")
    print("=" * 60)

    # Calculate target iteration (resume adds to checkpoint iteration)
    target_iteration = start_iteration + config.ppo.iterations

    if start_iteration > 0:
        print(f"Resuming training from iteration {start_iteration}...")
        print(f"  Target iterations: {target_iteration:,}")
        print(f"  Iterations to run: {config.ppo.iterations:,}")
        remaining_steps = (
            config.ppo.iterations * config.ppo.num_envs * config.ppo.rollout_steps
        )
        print(f"  Remaining steps: {remaining_steps:,}")
    else:
        print("Starting training...")
        print(f"  Total iterations: {config.ppo.iterations:,}")
    total_expected_steps = (
        target_iteration * config.ppo.num_envs * config.ppo.rollout_steps
    )
    print(f"  Total steps target: {total_expected_steps:,}")
    print()

    # Training loop
    start_time = time.time()
    last_approx_kl = 0.0
    lr_backoff_scale = 1.0
    last_eval_metrics: Dict[str, float] = {}
    rollback_bad_count = 0
    best_eval = {
        "success_rate": -1.0,
        "episode_length": 0.0,
        "state": None,
        "iteration": 0,
    }

    def _extract_opt_count(opt_state: Any) -> int:
        counts = []
        for leaf in jax.tree_util.tree_leaves(jax.device_get(opt_state)):
            if isinstance(leaf, (int, np.integer)):
                counts.append(int(leaf))
            elif isinstance(leaf, np.ndarray) and leaf.shape == () and np.issubdtype(
                leaf.dtype, np.integer
            ):
                counts.append(int(leaf.item()))
        return max(counts) if counts else 0

    cumulative_ppo_updates = _extract_opt_count(state.policy_opt_state)

    for iteration in range(start_iteration + 1, target_iteration + 1):
        iter_start = time.time()

        local_iter = iteration - start_iteration - 1
        total_local_iters = max(config.ppo.iterations - 1, 1)
        progress = local_iter / total_local_iters
        entropy_schedule_factor = _linear_schedule_factor(
            progress, config.ppo.entropy_schedule_end_factor
        )
        learning_rate_scale = lr_backoff_scale
        entropy_coef = float(config.ppo.entropy_coef) * entropy_schedule_factor
        active_epochs = _effective_ppo_epochs(
            base_epochs=config.ppo.epochs,
            previous_approx_kl=last_approx_kl,
            target_kl=float(config.ppo.target_kl),
            early_stop_multiplier=float(config.ppo.kl_early_stop_multiplier),
        )

        state, env_state, metrics = train_iteration_fn(
            state,
            env_state,
            active_epochs,
            entropy_coef,
            learning_rate_scale,
        )
        jax.block_until_ready(state.total_steps)

        current_approx_kl = float(metrics.approx_kl)
        kl_backoff_applied = False
        if (
            config.ppo.target_kl > 0.0
            and current_approx_kl
            > float(config.ppo.target_kl) * float(config.ppo.kl_lr_backoff_multiplier)
        ):
            lr_backoff_scale = max(
                lr_backoff_scale * float(config.ppo.kl_lr_backoff_factor),
                1e-3,
            )
            kl_backoff_applied = True
        last_approx_kl = current_approx_kl

        env_metrics = metrics.env_metrics
        active_updates = int(float(env_metrics.get("ppo/active_updates", 0.0)))
        cumulative_ppo_updates += active_updates
        base_lr_now = float(lr_schedule(cumulative_ppo_updates))
        env_metrics["ppo/lr"] = jnp.asarray(
            base_lr_now * lr_backoff_scale,
            dtype=jnp.float32,
        )
        env_metrics["ppo/target_kl"] = jnp.asarray(config.ppo.target_kl, dtype=jnp.float32)
        env_metrics["ppo/lr_backoff_scale"] = jnp.asarray(lr_backoff_scale, dtype=jnp.float32)
        env_metrics["ppo/lr_backoff_applied"] = jnp.asarray(
            1.0 if kl_backoff_applied else 0.0,
            dtype=jnp.float32,
        )
        env_metrics["ppo/entropy_coef"] = jnp.asarray(entropy_coef, dtype=jnp.float32)
        env_metrics["ppo/rollback_triggered"] = jnp.asarray(0.0, dtype=jnp.float32)

        if eval_enabled and (iteration == 1 or iteration % config.ppo.eval.interval == 0):
            eval_push_reward, eval_push_success, eval_push_ep_len, eval_push_term_h_low, eval_push_term_h_high, eval_push_term_pitch, eval_push_term_roll, eval_push_done_env, eval_push_trunc_env, eval_push_survival_rate, eval_push_survival_steps, eval_push_reset_h_mean, eval_push_reset_h_min, eval_push_unnecessary_step_rate, eval_push_recovery_completed, eval_push_recovery_no_touchdown_frac, eval_push_recovery_touchdown_then_fail_frac, eval_push_recovery_touchdown_count, eval_push_recovery_first_liftoff_latency, eval_push_recovery_first_touchdown_latency, eval_push_recovery_pitch_rate_reduction_10t, eval_push_recovery_capture_error_reduction_10t, eval_push_recovery_touchdown_to_term_steps, eval_push_recovery_visible_step_rate, eval_push_visible_step_rate_hard, eval_push_recovery_min_height, eval_push_recovery_max_knee_flex, eval_push_recovery_first_step_dist_abs, eval_push_teacher_whole_body_active_frac, eval_push_teacher_whole_body_active_during_clean_frac, eval_push_teacher_whole_body_active_during_push_frac, eval_push_teacher_recovery_height_target_mean, eval_push_teacher_recovery_height_error, eval_push_teacher_recovery_height_in_band_frac, eval_push_teacher_com_velocity_target_mean, eval_push_teacher_com_velocity_error, eval_push_teacher_com_velocity_target_hit_frac, eval_push_reward_teacher_recovery_height, eval_push_reward_teacher_com_velocity_reduction, eval_push_walking = (
                run_eval_push(
                    state.policy_params,
                    state.processor_params,
                    eval_base_rng,
                )
            )
            if eval_has_clean_pass:
                eval_clean_reward, eval_clean_success, eval_clean_ep_len, eval_clean_term_h_low, eval_clean_term_h_high, eval_clean_term_pitch, eval_clean_term_roll, eval_clean_done_env, eval_clean_trunc_env, eval_clean_survival_rate, eval_clean_survival_steps, eval_clean_reset_h_mean, eval_clean_reset_h_min, eval_clean_unnecessary_step_rate, eval_clean_recovery_completed, eval_clean_recovery_no_touchdown_frac, eval_clean_recovery_touchdown_then_fail_frac, eval_clean_recovery_touchdown_count, eval_clean_recovery_first_liftoff_latency, eval_clean_recovery_first_touchdown_latency, eval_clean_recovery_pitch_rate_reduction_10t, eval_clean_recovery_capture_error_reduction_10t, eval_clean_recovery_touchdown_to_term_steps, eval_clean_recovery_visible_step_rate, eval_clean_visible_step_rate_hard, eval_clean_recovery_min_height, eval_clean_recovery_max_knee_flex, eval_clean_recovery_first_step_dist_abs, eval_clean_teacher_whole_body_active_frac, eval_clean_teacher_whole_body_active_during_clean_frac, eval_clean_teacher_whole_body_active_during_push_frac, eval_clean_teacher_recovery_height_target_mean, eval_clean_teacher_recovery_height_error, eval_clean_teacher_recovery_height_in_band_frac, eval_clean_teacher_com_velocity_target_mean, eval_clean_teacher_com_velocity_error, eval_clean_teacher_com_velocity_target_hit_frac, eval_clean_reward_teacher_recovery_height, eval_clean_reward_teacher_com_velocity_reduction, eval_clean_walking = (
                    run_eval_clean(
                        state.policy_params,
                        state.processor_params,
                        eval_base_rng,
                    )
                )
            else:
                eval_clean_reward = eval_push_reward
                eval_clean_success = eval_push_success
                eval_clean_ep_len = eval_push_ep_len
                eval_clean_term_h_low = eval_push_term_h_low
                eval_clean_term_h_high = eval_push_term_h_high
                eval_clean_term_pitch = eval_push_term_pitch
                eval_clean_term_roll = eval_push_term_roll
                eval_clean_done_env = eval_push_done_env
                eval_clean_trunc_env = eval_push_trunc_env
                eval_clean_survival_rate = eval_push_survival_rate
                eval_clean_survival_steps = eval_push_survival_steps
                eval_clean_reset_h_mean = eval_push_reset_h_mean
                eval_clean_reset_h_min = eval_push_reset_h_min
                eval_clean_unnecessary_step_rate = eval_push_unnecessary_step_rate
                eval_clean_recovery_completed = eval_push_recovery_completed
                eval_clean_recovery_no_touchdown_frac = eval_push_recovery_no_touchdown_frac
                eval_clean_recovery_touchdown_then_fail_frac = (
                    eval_push_recovery_touchdown_then_fail_frac
                )
                eval_clean_recovery_touchdown_count = eval_push_recovery_touchdown_count
                eval_clean_recovery_first_liftoff_latency = (
                    eval_push_recovery_first_liftoff_latency
                )
                eval_clean_recovery_first_touchdown_latency = (
                    eval_push_recovery_first_touchdown_latency
                )
                eval_clean_recovery_pitch_rate_reduction_10t = (
                    eval_push_recovery_pitch_rate_reduction_10t
                )
                eval_clean_recovery_capture_error_reduction_10t = (
                    eval_push_recovery_capture_error_reduction_10t
                )
                eval_clean_recovery_touchdown_to_term_steps = (
                    eval_push_recovery_touchdown_to_term_steps
                )
                eval_clean_recovery_visible_step_rate = eval_push_recovery_visible_step_rate
                eval_clean_visible_step_rate_hard = eval_push_visible_step_rate_hard
                eval_clean_recovery_min_height = eval_push_recovery_min_height
                eval_clean_recovery_max_knee_flex = eval_push_recovery_max_knee_flex
                eval_clean_recovery_first_step_dist_abs = eval_push_recovery_first_step_dist_abs
                eval_clean_teacher_whole_body_active_frac = eval_push_teacher_whole_body_active_frac
                eval_clean_teacher_whole_body_active_during_clean_frac = (
                    eval_push_teacher_whole_body_active_during_clean_frac
                )
                eval_clean_teacher_whole_body_active_during_push_frac = (
                    eval_push_teacher_whole_body_active_during_push_frac
                )
                eval_clean_teacher_recovery_height_target_mean = (
                    eval_push_teacher_recovery_height_target_mean
                )
                eval_clean_teacher_recovery_height_error = eval_push_teacher_recovery_height_error
                eval_clean_teacher_recovery_height_in_band_frac = (
                    eval_push_teacher_recovery_height_in_band_frac
                )
                eval_clean_teacher_com_velocity_target_mean = (
                    eval_push_teacher_com_velocity_target_mean
                )
                eval_clean_teacher_com_velocity_error = eval_push_teacher_com_velocity_error
                eval_clean_teacher_com_velocity_target_hit_frac = (
                    eval_push_teacher_com_velocity_target_hit_frac
                )
                eval_clean_reward_teacher_recovery_height = eval_push_reward_teacher_recovery_height
                eval_clean_reward_teacher_com_velocity_reduction = (
                    eval_push_reward_teacher_com_velocity_reduction
                )
                eval_clean_walking = eval_push_walking
            jax.block_until_ready(eval_push_success)

            eval_push_success_f = float(eval_push_success)
            eval_push_ep_len_f = float(eval_push_ep_len)
            eval_push_survival_rate_f = float(eval_push_survival_rate)
            eval_push_survival_steps_f = float(eval_push_survival_steps)
            last_eval_metrics = {
                "eval_push/reward": float(eval_push_reward),
                "eval_push/success_rate": eval_push_success_f,
                "eval_push/episode_length": eval_push_ep_len_f,
                "eval_push/term_height_low_frac": float(eval_push_term_h_low),
                "eval_push/term_height_high_frac": float(eval_push_term_h_high),
                "eval_push/term_pitch_frac": float(eval_push_term_pitch),
                "eval_push/term_roll_frac": float(eval_push_term_roll),
                "eval_push/done_env_frac": float(eval_push_done_env),
                "eval_push/trunc_env_frac": float(eval_push_trunc_env),
                "eval_push/survival_rate": eval_push_survival_rate_f,
                "eval_push/survival_steps": eval_push_survival_steps_f,
                "eval_push/reset_height_mean": float(eval_push_reset_h_mean),
                "eval_push/reset_height_min": float(eval_push_reset_h_min),
                "eval_push/unnecessary_step_rate": float(eval_push_unnecessary_step_rate),
                "eval_push/recovery_completed": float(eval_push_recovery_completed),
                "eval_push/recovery_no_touchdown_frac": float(
                    eval_push_recovery_no_touchdown_frac
                ),
                "eval_push/recovery_touchdown_then_fail_frac": float(
                    eval_push_recovery_touchdown_then_fail_frac
                ),
                "eval_push/recovery_touchdown_count": float(eval_push_recovery_touchdown_count),
                "eval_push/recovery_first_liftoff_latency": float(
                    eval_push_recovery_first_liftoff_latency
                ),
                "eval_push/recovery_first_touchdown_latency": float(
                    eval_push_recovery_first_touchdown_latency
                ),
                "eval_push/recovery_pitch_rate_reduction_10t": float(
                    eval_push_recovery_pitch_rate_reduction_10t
                ),
                "eval_push/recovery_capture_error_reduction_10t": float(
                    eval_push_recovery_capture_error_reduction_10t
                ),
                "eval_push/recovery_touchdown_to_term_steps": float(
                    eval_push_recovery_touchdown_to_term_steps
                ),
                "eval_push/recovery_visible_step_rate": float(
                    eval_push_recovery_visible_step_rate
                ),
                "eval_push/visible_step_rate_hard": float(eval_push_visible_step_rate_hard),
                "eval_push/recovery_min_height": float(eval_push_recovery_min_height),
                "eval_push/recovery_max_knee_flex": float(eval_push_recovery_max_knee_flex),
                "eval_push/recovery_first_step_dist_abs": float(
                    eval_push_recovery_first_step_dist_abs
                ),
                "eval_push/teacher_whole_body_active_frac": float(
                    eval_push_teacher_whole_body_active_frac
                ),
                "eval_push/teacher_whole_body_active_during_clean_frac": float(
                    eval_push_teacher_whole_body_active_during_clean_frac
                ),
                "eval_push/teacher_whole_body_active_during_push_frac": float(
                    eval_push_teacher_whole_body_active_during_push_frac
                ),
                "eval_push/teacher_recovery_height_target_mean": float(
                    eval_push_teacher_recovery_height_target_mean
                ),
                "eval_push/teacher_recovery_height_error": float(
                    eval_push_teacher_recovery_height_error
                ),
                "eval_push/teacher_recovery_height_in_band_frac": float(
                    eval_push_teacher_recovery_height_in_band_frac
                ),
                "eval_push/teacher_com_velocity_target_mean": float(
                    eval_push_teacher_com_velocity_target_mean
                ),
                "eval_push/teacher_com_velocity_error": float(
                    eval_push_teacher_com_velocity_error
                ),
                "eval_push/teacher_com_velocity_target_hit_frac": float(
                    eval_push_teacher_com_velocity_target_hit_frac
                ),
                "eval_push/reward_teacher_recovery_height": float(
                    eval_push_reward_teacher_recovery_height
                ),
                "eval_push/reward_teacher_com_velocity_reduction": float(
                    eval_push_reward_teacher_com_velocity_reduction
                ),
                "eval_clean/reward": float(eval_clean_reward),
                "eval_clean/success_rate": float(eval_clean_success),
                "eval_clean/episode_length": float(eval_clean_ep_len),
                "eval_clean/term_height_low_frac": float(eval_clean_term_h_low),
                "eval_clean/term_height_high_frac": float(eval_clean_term_h_high),
                "eval_clean/term_pitch_frac": float(eval_clean_term_pitch),
                "eval_clean/term_roll_frac": float(eval_clean_term_roll),
                "eval_clean/done_env_frac": float(eval_clean_done_env),
                "eval_clean/trunc_env_frac": float(eval_clean_trunc_env),
                "eval_clean/survival_rate": float(eval_clean_survival_rate),
                "eval_clean/survival_steps": float(eval_clean_survival_steps),
                "eval_clean/reset_height_mean": float(eval_clean_reset_h_mean),
                "eval_clean/reset_height_min": float(eval_clean_reset_h_min),
                "eval_clean/unnecessary_step_rate": float(eval_clean_unnecessary_step_rate),
                "eval_clean/recovery_completed": float(eval_clean_recovery_completed),
                "eval_clean/recovery_no_touchdown_frac": float(
                    eval_clean_recovery_no_touchdown_frac
                ),
                "eval_clean/recovery_touchdown_then_fail_frac": float(
                    eval_clean_recovery_touchdown_then_fail_frac
                ),
                "eval_clean/recovery_touchdown_count": float(
                    eval_clean_recovery_touchdown_count
                ),
                "eval_clean/recovery_first_liftoff_latency": float(
                    eval_clean_recovery_first_liftoff_latency
                ),
                "eval_clean/recovery_first_touchdown_latency": float(
                    eval_clean_recovery_first_touchdown_latency
                ),
                "eval_clean/recovery_pitch_rate_reduction_10t": float(
                    eval_clean_recovery_pitch_rate_reduction_10t
                ),
                "eval_clean/recovery_capture_error_reduction_10t": float(
                    eval_clean_recovery_capture_error_reduction_10t
                ),
                "eval_clean/recovery_touchdown_to_term_steps": float(
                    eval_clean_recovery_touchdown_to_term_steps
                ),
                "eval_clean/recovery_visible_step_rate": float(
                    eval_clean_recovery_visible_step_rate
                ),
                "eval_clean/visible_step_rate_hard": float(eval_clean_visible_step_rate_hard),
                "eval_clean/recovery_min_height": float(eval_clean_recovery_min_height),
                "eval_clean/recovery_max_knee_flex": float(eval_clean_recovery_max_knee_flex),
                "eval_clean/recovery_first_step_dist_abs": float(
                    eval_clean_recovery_first_step_dist_abs
                ),
                "eval_clean/teacher_whole_body_active_frac": float(
                    eval_clean_teacher_whole_body_active_frac
                ),
                "eval_clean/teacher_whole_body_active_during_clean_frac": float(
                    eval_clean_teacher_whole_body_active_during_clean_frac
                ),
                "eval_clean/teacher_whole_body_active_during_push_frac": float(
                    eval_clean_teacher_whole_body_active_during_push_frac
                ),
                "eval_clean/teacher_recovery_height_target_mean": float(
                    eval_clean_teacher_recovery_height_target_mean
                ),
                "eval_clean/teacher_recovery_height_error": float(
                    eval_clean_teacher_recovery_height_error
                ),
                "eval_clean/teacher_recovery_height_in_band_frac": float(
                    eval_clean_teacher_recovery_height_in_band_frac
                ),
                "eval_clean/teacher_com_velocity_target_mean": float(
                    eval_clean_teacher_com_velocity_target_mean
                ),
                "eval_clean/teacher_com_velocity_error": float(
                    eval_clean_teacher_com_velocity_error
                ),
                "eval_clean/teacher_com_velocity_target_hit_frac": float(
                    eval_clean_teacher_com_velocity_target_hit_frac
                ),
                "eval_clean/reward_teacher_recovery_height": float(
                    eval_clean_reward_teacher_recovery_height
                ),
                "eval_clean/reward_teacher_com_velocity_reduction": float(
                    eval_clean_reward_teacher_com_velocity_reduction
                ),
            }
            # ToddlerBot-style walking eval block (v0.20.1).  Pulls per-term
            # walking aggregates from the same deterministic eval rollout.
            # Mirrors toddlerbot/locomotion/train_mjx.py log_metrics —
            # checkpoint quality reads off ``Evaluate/episode_length`` and
            # the ``Evaluate/<term>`` tracking errors, NOT a success_rate.
            for src, dst_prefix in (
                (eval_push_walking, "eval_push"),
                (eval_clean_walking, "eval_clean"),
            ):
                for key, value in src.items():
                    last_eval_metrics[f"{dst_prefix}/{key}"] = float(value)
            # Also expose the canonical ``Evaluate/*`` namespace (matches
            # ToddlerBot exactly) — uses the clean-pass values so eval
            # numbers always describe a non-disturbed deterministic walk.
            last_eval_metrics["Evaluate/mean_reward"] = float(eval_clean_reward)
            last_eval_metrics["Evaluate/mean_episode_length"] = float(eval_clean_ep_len)
            for key, value in eval_clean_walking.items():
                last_eval_metrics[f"Evaluate/{key}"] = float(value)
            for key, value in last_eval_metrics.items():
                env_metrics[key] = jnp.asarray(value, dtype=jnp.float32)

            # Use survival metrics for short-horizon probes (eval_steps < max_episode_steps).
            gate_success = eval_push_survival_rate_f if use_eval_survival_metric else eval_push_success_f
            gate_len = eval_push_survival_steps_f if use_eval_survival_metric else eval_push_ep_len_f

            if _is_eval_better(
                success_rate=gate_success,
                episode_length=gate_len,
                best_success_rate=best_eval["success_rate"],
                best_episode_length=best_eval["episode_length"],
            ):
                best_eval["success_rate"] = gate_success
                best_eval["episode_length"] = gate_len
                best_eval["state"] = jax.device_get(state)
                best_eval["iteration"] = iteration
                rollback_bad_count = 0
            elif config.ppo.rollback.enabled and best_eval["state"] is not None:
                if _should_trigger_rollback(
                    current_success_rate=gate_success,
                    best_success_rate=float(best_eval["success_rate"]),
                    threshold=float(config.ppo.rollback.success_rate_drop_threshold),
                ):
                    rollback_bad_count += 1
                else:
                    rollback_bad_count = 0

                if rollback_bad_count >= int(config.ppo.rollback.patience):
                    snap = best_eval["state"]
                    state = TrainingState(
                        policy_params=snap.policy_params,
                        value_params=snap.value_params,
                        processor_params=snap.processor_params,
                        policy_opt_state=snap.policy_opt_state,
                        value_opt_state=snap.value_opt_state,
                        iteration=state.iteration,
                        total_steps=state.total_steps,
                        rng=state.rng,
                    )
                    cumulative_ppo_updates = _extract_opt_count(state.policy_opt_state)
                    env_state = env_reset_fn(jax.random.fold_in(state.rng, iteration))
                    lr_backoff_scale = max(
                        lr_backoff_scale * float(config.ppo.rollback.lr_factor),
                        1e-4,
                    )
                    rollback_bad_count = 0
                    env_metrics["ppo/rollback_triggered"] = jnp.asarray(
                        1.0, dtype=jnp.float32
                    )
                    print(
                        f"  ↩ rollback at iter {iteration}: restored params from iter "
                        f"{best_eval['iteration']}, new_lr_scale={lr_backoff_scale:.4f}"
                    )

        iter_time = time.time() - iter_start
        steps_per_sec = (config.ppo.rollout_steps * config.ppo.num_envs) / iter_time

        # Logging
        if iteration % config.ppo.log_interval == 0 or iteration == 1:
            total_steps = int(state.total_steps)
            progress_pct = (total_steps / total_expected_steps) * 100

            # Access env metrics from dict
            env = metrics.env_metrics
            for key, value in last_eval_metrics.items():
                if key not in env:
                    env[key] = jnp.asarray(value, dtype=jnp.float32)

            # ----- v0.20.1 walking-aware console output -----
            # Drops the legacy ``success_rate`` (truncation-based, always 0
            # for v0.20.1 walking) and ``vel_err`` (env/velocity_error,
            # also stuck at 0).  Surfaces the v0.20.1 G4/G5 gates inline so
            # the operator can see pass/fail per iter; replaces eval_push /
            # eval_clean success rows with the ToddlerBot-style
            # ``Evaluate/*`` block.

            def _g(key: str, default: float = 0.0) -> float:
                v = env.get(key, default)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            def _mark(ok: bool) -> str:
                return "\u2713" if ok else "\u2717"

            fwd = _g("forward_velocity")
            cmd_vx = _g("velocity_command")
            cmd_err = _g("tracking/cmd_vs_achieved_forward")
            step_len = _g("tracking/step_length_touchdown_event_m")
            vx_cmd_ratio = _g("tracking/forward_velocity_cmd_ratio")
            ep_len = float(metrics.episode_length)
            res_hpL = _g("tracking/residual_hip_pitch_left_abs")
            res_hpR = _g("tracking/residual_hip_pitch_right_abs")
            res_knL = _g("tracking/residual_knee_left_abs")
            res_knR = _g("tracking/residual_knee_right_abs")
            # ankle_roll added by the v20 ankle_roll merge — must be in
            # res_max so the console G5 line matches the env's
            # authoritative g5_residuals tally (which already includes
            # ankle_roll in wildrobot_env.py:1838-1841).
            res_arL = _g("tracking/residual_ankle_roll_left_abs")
            res_arR = _g("tracking/residual_ankle_roll_right_abs")
            res_max = max(res_hpL, res_hpR, res_knL, res_knR, res_arL, res_arR)
            term_h_low = _g("term_height_low_frac")
            ref_q_rmse = _g("ref/q_track_err_rmse")
            ref_quat_deg = _g("ref/body_quat_err_deg")
            ref_feet_l2 = _g("ref/feet_pos_err_l2")
            ref_contact = _g("ref/contact_phase_match")

            # G4 / G5 floors are pulled directly from
            # walking_training.md v0.20.1 §.  ``ep_len`` G4 floor is 95% of
            # ``max_episode_steps`` so single-cmd horizon changes don't
            # break the gate.
            #
            # Phase 9D-aware (2026-05-09): forward_velocity and stride
            # floors are command-scaled from the eval cmd, not hardcoded
            # to vx=0.15 values.  Without this, a Phase 9D run at
            # vx=0.20 would PASS the velocity gate at 38% tracking
            # (0.075/0.20) instead of the doc-intended 50%, and the
            # stride gate would only require 32% of nominal stride
            # instead of the intended ~50%.  Thresholds:
            #   forward_velocity ≥ 0.50 × eval_vx
            #     — equivalent to ≥0.075 m/s at vx=0.15, ≥0.10 at vx=0.20
            #   cmd_vs_achieved_forward ≤ 0.50 × eval_vx
            #     — same scaling as the velocity floor
            #   step_length ≥ max(0.030, 0.50 × vx × cycle/2)
            #     — equivalent to ≥0.030 m at vx=0.15 cycle=0.72 (the old
            #     calibration), ≥0.048 m at Phase 9D vx=0.20 cycle=0.96
            # The 0.030 lower bound is retained as the "absolute
            # shuffle floor" — any operating point where the scaled
            # value is below 0.030 m is below the shuffling regime.
            max_ep = float(getattr(config.env, "max_episode_steps", 500) or 500)
            # Two separate variables to keep sentinel semantics intact:
            #   raw_eval_vx : the literal config value (negative ⇒ sentinel
            #                 / sampling mode, ratio gates undefined)
            #   g4_floor_vx : the vx used for floor/ceiling calculations.
            #                 Falls back to historical vx=0.15 calibration
            #                 in sentinel mode so the train console still
            #                 has well-defined floors.
            # Collapsing them (the previous shape) made the G5-eval
            # ratio's ``n/a`` branch unreachable in sentinel mode.
            raw_eval_vx = float(getattr(config.env, "eval_velocity_cmd", -1.0))
            g4_floor_vx = raw_eval_vx if raw_eval_vx > 0 else 0.15
            from control.zmp.zmp_walk import ZMPWalkConfig as _ZMPCfg
            cycle_time_s = float(_ZMPCfg.cycle_time_s)
            g4_vel_floor = 0.50 * g4_floor_vx
            g4_cmd_err_ceiling = 0.50 * g4_floor_vx
            g4_step_len_floor = max(0.030, 0.50 * g4_floor_vx * cycle_time_s / 2.0)
            g4_vel_ok = fwd >= g4_vel_floor
            g4_cmd_err_ok = cmd_err <= g4_cmd_err_ceiling
            g4_step_len_ok = step_len >= g4_step_len_floor
            g4_ep_len_ok = ep_len >= 0.95 * max_ep
            g5_residual_ok = res_max <= 0.20
            g5_ratio_ok = 0.6 <= vx_cmd_ratio <= 1.5

            # Main line.  Reward + ep_len + throughput.  ``success`` deleted
            # — it was the truncation-based env/success_rate which is always
            # 0 for v0.20.1 walking.
            main_line = (
                f"#{iteration:<4} Steps: {total_steps:>10} ({progress_pct:>5.1f}%): "
                f"reward={float(metrics.episode_reward):>8.2f} | "
                f"ep_len={ep_len:>5.0f} | "
                f"steps/s={steps_per_sec:>8.0f}"
            )

            YELLOW = "\x1b[33m"
            RESET = "\x1b[0m"
            try:
                print(f"{YELLOW}{main_line}{RESET}")
            except Exception:
                print(main_line)

            # Train-rollout walking signals.
            print(
                f"  └─ vel={fwd:+5.3f} (cmd={cmd_vx:+5.3f} ratio={vx_cmd_ratio:5.2f}) | "
                f"cmd_err={cmd_err:.3f} | step_len={step_len:+.4f} | "
                f"h={_g('height'):.2f}m"
            )

            # G4 promotion-horizon gate (pass/fail per iter).
            print(
                f"  └─ G4-train (eval_vx={g4_floor_vx:.3f}, cycle={cycle_time_s:.2f}s): "
                f"vel\u2265{g4_vel_floor:.3f} {_mark(g4_vel_ok)} | "
                f"cmd_err\u2264{g4_cmd_err_ceiling:.3f} {_mark(g4_cmd_err_ok)} | "
                f"step_len\u2265{g4_step_len_floor:.3f} {_mark(g4_step_len_ok)} | "
                f"ep_len\u2265{int(0.95 * max_ep)} {_mark(g4_ep_len_ok)}"
            )

            # G5 anti-exploit gate (pass/fail per iter).
            # Train-side: residual magnitudes are the authoritative
            # promotion gate (env emits g5_residual_ok over both train +
            # eval terminal metrics).  The train-rollout vx/cmd ratio is
            # **diagnostic only** under the smoke7 multi-cmd curriculum:
            # sampled near-zero commands (zero_chance + deadzone) make
            # the train mean numerically unstable.  The promotion ratio
            # gate uses the pinned eval rollout — see the G5-eval line
            # in the Evaluate/* block below.
            print(
                f"  └─ G5-train: "
                f"|\u0394q|hipL/R={res_hpL:.2f}/{res_hpR:.2f} "
                f"kneeL/R={res_knL:.2f}/{res_knR:.2f} "
                f"ankle_rollL/R={res_arL:.2f}/{res_arR:.2f} "
                f"\u22640.20 {_mark(g5_residual_ok)} | "
                f"vx/cmd\u2208[0.6,1.5] {_mark(g5_ratio_ok)} (diagnostic \u2014 "
                f"promotion ratio is on eval, see G5-eval below)"
            )

            # Imitation diagnostics — surfaces dead-gradient (large
            # ref_*_err_* alongside near-zero reward/ref_*_track) and gait
            # drift (ref/contact_phase_match falling).
            print(
                f"  └─ ref: "
                f"q_rmse={ref_q_rmse:.3f} | "
                f"quat={ref_quat_deg:5.1f}\u00b0 | "
                f"feet_l2={ref_feet_l2:.2f} | "
                f"contact={ref_contact:.2f}"
            )

            # Termination flag — under relaxed termination only h_low
            # actually terminates; pitch / roll are soft-band penalties so
            # their per-step "term_*_frac" values are NOT bounded to [0,1]
            # (sum-across-steps), and printing them as percentages is
            # misleading (the legacy line showed "pitch=415.3%").  Print
            # h_low only, and only when nonzero.
            if term_h_low > 0.001:
                print(f"  └─ term: h_low={term_h_low:>5.1%} (relaxed termination: only height terminates)")

            # PPO internals — only on anomaly (early stop or rollback).
            ppo_early_stop = int(_g("ppo/early_stop_epoch"))
            ppo_rollback = int(_g("ppo/rollback_triggered"))
            if ppo_early_stop > 0 or ppo_rollback > 0:
                print(
                    f"  └─ ppo: kl={float(metrics.approx_kl):>6.4f} "
                    f"(target={_g('ppo/target_kl'):>6.4f}) | "
                    f"epochs={int(_g('ppo/epochs_used')):>2} | "
                    f"lr={_g('ppo/lr'):>8.6f}"
                    f"{' | EARLY_STOP' if ppo_early_stop > 0 else ''}"
                    f"{' | ROLLBACK' if ppo_rollback > 0 else ''}"
                )

            # Evaluate/* block — ToddlerBot pattern (no success_rate
            # concept; checkpoint quality reads off mean_reward +
            # mean_episode_length + per-term tracking).  Only printed when
            # the eval rollout actually ran this iter.
            #
            # Eval-side G4 marks (separate from train G4 above) are the
            # promotion-horizon gate per walking_training.md G4: the
            # smoke ships only if BOTH train-side and eval-side G4 pass
            # at the same horizon.  Train and eval can disagree (eval
            # is deterministic and pinned to ``eval_velocity_cmd``;
            # train rollouts sample the cmd range under the curriculum)
            # so a "G4 ✓" on train alone is not the smoke promotion
            # signal.
            if "Evaluate/mean_reward" in env:
                eval_r = _g("Evaluate/mean_reward")
                eval_L = _g("Evaluate/mean_episode_length")
                eval_v = _g("Evaluate/forward_velocity")
                eval_e = _g("Evaluate/cmd_vs_achieved_forward")
                # Same command-scaled floors as train G4 (single source
                # of truth: ``eval_velocity_cmd`` + cycle_time_s).
                # Eval is always pinned to eval_velocity_cmd, so the
                # floor is well-defined here regardless of sentinel
                # mode (which only affects train rollouts).
                eval_g4_vel_ok = eval_v >= g4_vel_floor
                eval_g4_cmd_err_ok = eval_e <= g4_cmd_err_ceiling
                eval_g4_ep_len_ok = eval_L >= 0.95 * max_ep
                # G5 ratio gate (walking_training.md Phase 9D):
                # ``0.6 ≤ Evaluate/forward_velocity / eval_velocity_cmd
                # ≤ 1.5`` on the pinned eval rollout.  This is the
                # promotion gate; the train-rollout
                # ``tracking/forward_velocity_cmd_ratio`` mean is
                # numerically unstable when sampled commands are near
                # zero (zero_chance + deadzone) and is therefore
                # diagnostic-only — the train console line below labels
                # it as such.  When ``eval_velocity_cmd <= 0`` (sentinel
                # / sampling mode) the eval ratio is undefined; report
                # n/a rather than dividing by zero.
                # Sentinel-aware: only gate the ratio when the eval cmd
                # is actually pinned (raw_eval_vx > 0).  In sampling
                # mode the per-iter ratio is meaningless because the
                # eval rollout's sampled cmd disagrees with the floor's
                # fallback vx=0.15.
                if raw_eval_vx > 0:
                    eval_vx_cmd_ratio = float(eval_v) / float(raw_eval_vx)
                    eval_g5_ratio_ok = 0.6 <= eval_vx_cmd_ratio <= 1.5
                    eval_ratio_str = (
                        f"vx/cmd={eval_vx_cmd_ratio:5.2f}∈[0.6,1.5] "
                        f"{_mark(eval_g5_ratio_ok)}"
                    )
                else:
                    eval_ratio_str = "vx/cmd=n/a (eval cmd not pinned)"
                print(
                    f"  └─ Evaluate: reward={eval_r:>7.2f} | "
                    f"ep_len={eval_L:>5.0f} | "
                    f"vel={eval_v:+5.3f} | "
                    f"cmd_err={eval_e:.3f}"
                )
                print(
                    f"  └─ G4-eval (vx={g4_floor_vx:.3f}, cycle={cycle_time_s:.2f}s): "
                    f"vel≥{g4_vel_floor:.3f} {_mark(eval_g4_vel_ok)} | "
                    f"cmd_err≤{g4_cmd_err_ceiling:.3f} {_mark(eval_g4_cmd_err_ok)} | "
                    f"ep_len≥{int(0.95 * max_ep)} {_mark(eval_g4_ep_len_ok)}"
                )
                # G5-eval anti-exploit ratio (the smoke promotion gate
                # per walking_training.md Phase 9D).  Residual-magnitude
                # G5 (``g5_residual_ok``) is computed once over both
                # train + eval terminal metrics in the env (see
                # wildrobot_env.py:1827-1841); the ratio is the only
                # piece that splits train/eval.
                print(
                    f"  └─ G5-eval (promotion gate): {eval_ratio_str}"
                )

            if callback is not None:
                callback(iteration, state, metrics, steps_per_sec)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training complete!")
    print(f"  Total steps: {int(state.total_steps):,}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Average steps/sec: {int(state.total_steps) / elapsed:,.0f}")
    print("=" * 60)

    return state
