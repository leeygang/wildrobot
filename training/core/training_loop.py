"""Unified JIT-compiled trainer for PPO and AMP+PPO.

This module provides a single training loop that works for both:
- Stage 1: PPO-only (amp_weight=0)
- Stage 3: AMP+PPO (amp_weight>0)

Key Design Principles:
1. Single code path for both stages (no separate trainers)
2. lax.cond branching for AMP vs no-AMP (no recompilation needed)
3. Identical PPO behavior whether AMP is enabled or disabled
4. All Brax components reused: networks, distributions, optimizers

Architecture:
    train_iteration(state, env_state, amp_weight) -> new_state, metrics
        ├── collect_rollout() - unified rollout collector
        ├── lax.cond(amp_weight > 0):
        │   ├── True: extract_amp_features -> disc_reward -> train_disc
        │   └── False: skip (no RNG, no compute, no grads)
        ├── compute_reward_total() - task + amp_weight * amp
        ├── compute_gae() - GAE advantage estimation
        └── ppo_update_scan() - PPO policy/value update

Usage:
    # Stage 1: PPO-only
    config = load_training_config("configs/ppo_walking.yaml")
    config.freeze()
    train(env_step_fn, env_reset_fn, config, ref_motion_data=None)

    # Stage 3: AMP+PPO
    config = load_training_config("configs/ppo_amp.yaml")
    config.freeze()
    train(env_step_fn, env_reset_fn, config, ref_motion_data=ref_data)
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from training.amp.discriminator import (
    AMPDiscriminator,
    compute_amp_reward,
    create_discriminator,
    discriminator_loss,
)
from training.amp.policy_features import (
    extract_amp_features_batched,
    FeatureConfig,
)
from training.configs.feature_config import get_feature_config
from training.configs.training_config import get_robot_config, TrainingConfig

from training.algos.ppo.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)
from training.core.rollout import (
    collect_rollout,
    compute_reward_total,
    TrajectoryBatch,
)
from training.core.metrics_registry import (
    aggregate_metrics,
    METRIC_INDEX,
    unpack_metrics,
)


# =============================================================================
# Training State (unified for both PPO-only and AMP)
# =============================================================================


class TrainingState(NamedTuple):
    """Complete training state - all JAX arrays, no Python objects.

    This state is used for both PPO-only and AMP+PPO training.
    When amp_weight=0, discriminator fields are still present but unused.
    """

    # Network parameters (always used)
    policy_params: Any
    value_params: Any
    processor_params: Any

    # Optimizer states (always used)
    policy_opt_state: Any
    value_opt_state: Any

    # Discriminator (present but unused when amp_weight=0)
    disc_params: Any
    disc_opt_state: Any

    # Feature normalization (computed from ref data, or dummy when no AMP)
    feature_mean: jnp.ndarray
    feature_var: jnp.ndarray

    # Training progress
    iteration: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.Array


class IterationMetrics(NamedTuple):
    """Metrics from one training iteration.

    v0.10.4: Simplified to use aggregated env_metrics dict instead of
    individual fields. Training losses remain explicit.

    All fields are present for both PPO-only and AMP modes.
    AMP-specific fields are zero when amp_weight=0.
    """

    # PPO losses (computed in training loop)
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    total_loss: jnp.ndarray
    clip_fraction: jnp.ndarray
    approx_kl: jnp.ndarray

    # AMP metrics (zero when amp_weight=0)
    disc_loss: jnp.ndarray
    disc_accuracy: jnp.ndarray
    amp_reward_mean: jnp.ndarray

    # Episode-level metrics (computed from trajectory)
    episode_reward: jnp.ndarray
    task_reward_mean: jnp.ndarray
    episode_length: jnp.ndarray
    success_rate: jnp.ndarray

    # v0.10.4: Aggregated environment metrics from metrics_registry
    # Contains all metrics from METRIC_SPECS (forward_velocity, height, etc.)
    # Access via: env_metrics["forward_velocity"], env_metrics["tracking/vel_error"], etc.
    env_metrics: Dict[str, jnp.ndarray]


# =============================================================================
# Feature Normalization
# =============================================================================


def compute_normalization_stats(
    data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute normalization statistics from reference data."""
    mean = jnp.mean(data, axis=0)
    var = jnp.var(data, axis=0)
    var = jnp.maximum(var, 1e-8)
    return mean, var


def normalize_features(
    features: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Normalize features using fixed statistics."""
    return (features - mean) / jnp.sqrt(var + epsilon)


# =============================================================================
# Discriminator Training
# =============================================================================


def train_discriminator_scan(
    disc_params: Any,
    disc_opt_state: Any,
    disc_model: AMPDiscriminator,
    disc_optimizer: optax.GradientTransformation,
    agent_features: jnp.ndarray,
    ref_buffer_data: jnp.ndarray,
    rng: jax.Array,
    num_updates: int,
    batch_size: int,
    r1_gamma: float,
) -> Tuple[Any, Any, jnp.ndarray, jnp.ndarray]:
    """Train discriminator using jax.lax.scan."""
    num_ref_samples = ref_buffer_data.shape[0]
    num_agent_samples = agent_features.shape[0]

    def disc_update_step(carry, rng):
        params, opt_state = carry
        rng, agent_rng, expert_rng, loss_rng = jax.random.split(rng, 4)

        # Sample batches
        agent_indices = jax.random.choice(
            agent_rng, num_agent_samples, shape=(batch_size,), replace=True
        )
        agent_batch = agent_features[agent_indices]

        expert_indices = jax.random.choice(
            expert_rng, num_ref_samples, shape=(batch_size,), replace=True
        )
        expert_batch = ref_buffer_data[expert_indices]

        # Compute loss and gradients
        def loss_fn(p):
            return discriminator_loss(
                params=p,
                model=disc_model,
                real_obs=expert_batch,
                fake_obs=agent_batch,
                rng_key=loss_rng,
                r1_gamma=r1_gamma,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update parameters
        updates, new_opt_state = disc_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state), (loss, metrics["discriminator_accuracy"])

    # Run scan
    rngs = jax.random.split(rng, num_updates)
    (final_params, final_opt_state), (losses, accuracies) = jax.lax.scan(
        disc_update_step, (disc_params, disc_opt_state), rngs
    )

    return final_params, final_opt_state, jnp.mean(losses), jnp.mean(accuracies)


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
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    rng: jax.Array,
    num_epochs: int,
    num_minibatches: int,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> Tuple[Any, Any, Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PPO update using jax.lax.scan over epochs and minibatches."""
    batch_size = obs.shape[0]
    minibatch_size = batch_size // num_minibatches
    num_updates = num_epochs * num_minibatches

    def ppo_step(carry, update_idx):
        policy_p, value_p, policy_opt, value_opt, rng = carry
        rng, shuffle_rng, step_rng = jax.random.split(rng, 3)

        # Compute epoch and minibatch indices
        epoch_idx = update_idx // num_minibatches
        mb_idx = update_idx % num_minibatches

        # Generate permutation for this epoch
        epoch_rng = jax.random.fold_in(shuffle_rng, epoch_idx)
        perm = jax.random.permutation(epoch_rng, batch_size)

        # Get minibatch
        start_idx = mb_idx * minibatch_size
        mb_indices = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))

        mb_obs = obs[mb_indices]
        mb_actions = actions[mb_indices]
        mb_old_log_probs = old_log_probs[mb_indices]
        mb_advantages = advantages[mb_indices]
        mb_returns = returns[mb_indices]

        # Normalize advantages
        mb_advantages = (mb_advantages - jnp.mean(mb_advantages)) / (
            jnp.std(mb_advantages) + 1e-8
        )

        # Compute loss and gradients
        def loss_fn(policy_p, value_p):
            return compute_ppo_loss(
                processor_params=processor_params,
                policy_params=policy_p,
                value_params=value_p,
                ppo_network=ppo_network,
                obs=mb_obs,
                actions=mb_actions,
                old_log_probs=mb_old_log_probs,
                advantages=mb_advantages,
                returns=mb_returns,
                rng=step_rng,
                clip_epsilon=clip_epsilon,
                value_loss_coef=value_loss_coef,
                entropy_coef=entropy_coef,
            )

        (total_loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(policy_p, value_p)

        # Update networks
        policy_updates, new_policy_opt = policy_optimizer.update(
            policy_grads, policy_opt
        )
        new_policy_p = optax.apply_updates(policy_p, policy_updates)

        value_updates, new_value_opt = value_optimizer.update(value_grads, value_opt)
        new_value_p = optax.apply_updates(value_p, value_updates)

        return (new_policy_p, new_value_p, new_policy_opt, new_value_opt, rng), metrics

    # Run scan
    update_indices = jnp.arange(num_updates)
    init_carry = (policy_params, value_params, policy_opt_state, value_opt_state, rng)
    final_carry, all_metrics = jax.lax.scan(ppo_step, init_carry, update_indices)

    new_policy_params, new_value_params, new_policy_opt, new_value_opt, _ = final_carry

    return (
        new_policy_params,
        new_value_params,
        new_policy_opt,
        new_value_opt,
        jnp.mean(all_metrics.policy_loss),
        jnp.mean(all_metrics.value_loss),
        jnp.mean(all_metrics.entropy_loss),
        jnp.mean(all_metrics.total_loss),
        jnp.mean(all_metrics.clip_fraction),
        jnp.mean(all_metrics.approx_kl),
    )


# =============================================================================
# Unified Training Iteration (with lax.cond for AMP branching)
# =============================================================================


def make_train_iteration_fn(
    env_step_fn: Callable,
    ppo_network: Any,
    disc_model: AMPDiscriminator,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    disc_optimizer: optax.GradientTransformation,
    config: TrainingConfig,
    ref_buffer_data: Optional[jnp.ndarray],
):
    """Create JIT-compiled training iteration function.

    This function handles both PPO-only (amp_weight=0) and AMP+PPO (amp_weight>0)
    using lax.cond for zero-overhead branching.
    """
    amp_feature_config = get_feature_config()
    amp_enabled = config.amp.enabled

    @jax.jit
    def train_iteration(
        state: TrainingState,
        env_state: Any,
    ) -> Tuple[TrainingState, Any, IterationMetrics]:
        """Execute one training iteration (fully on GPU)."""

        rng = state.rng
        rng, rollout_rng, disc_rng, ppo_rng = jax.random.split(rng, 4)

        # ================================================================
        # Step 1: Collect rollout (unified for both modes)
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
            collect_amp_features=amp_enabled,
        )

        # ================================================================
        # Step 2-4: AMP branch (skipped when amp_weight=0)
        # ================================================================

        def amp_branch(_):
            """AMP enabled: extract features, compute rewards, train disc."""
            # Extract AMP features
            amp_features = extract_amp_features_batched(
                trajectory.obs,
                amp_feature_config,
                trajectory.foot_contacts,
                trajectory.root_heights,
                trajectory.prev_joint_positions,
                dt=0.02,
                use_estimated_contacts=config.amp.feature_config.use_estimated_contacts,
                use_finite_diff_vel=config.amp.feature_config.use_finite_diff_vel,
                contact_threshold_angle=0.1,
                contact_knee_scale=0.5,
                contact_min_confidence=0.3,
            )

            # Flatten and normalize
            flat_features = amp_features.reshape(-1, amp_features.shape[-1])
            norm_features = normalize_features(
                flat_features, state.feature_mean, state.feature_var
            )
            norm_ref = normalize_features(
                ref_buffer_data, state.feature_mean, state.feature_var
            )

            # Compute AMP rewards (using OLD disc_params)
            norm_features_shaped = norm_features.reshape(amp_features.shape)

            def compute_reward_single(feat):
                return compute_amp_reward(state.disc_params, disc_model, feat)

            amp_rewards = jax.vmap(jax.vmap(compute_reward_single))(
                norm_features_shaped
            )

            # Train discriminator
            new_disc_params, new_disc_opt, disc_loss, disc_acc = (
                train_discriminator_scan(
                    disc_params=state.disc_params,
                    disc_opt_state=state.disc_opt_state,
                    disc_model=disc_model,
                    disc_optimizer=disc_optimizer,
                    agent_features=norm_features,
                    ref_buffer_data=norm_ref,
                    rng=disc_rng,
                    num_updates=config.amp.discriminator.updates_per_ppo_update,
                    batch_size=config.amp.discriminator.batch_size,
                    r1_gamma=config.amp.discriminator.r1_gamma,
                )
            )

            return amp_rewards, new_disc_params, new_disc_opt, disc_loss, disc_acc

        def no_amp_branch(_):
            """AMP disabled: return zeros, keep params unchanged."""
            amp_rewards = jnp.zeros_like(trajectory.task_rewards)
            return (
                amp_rewards,
                state.disc_params,
                state.disc_opt_state,
                jnp.array(0.0),
                jnp.array(0.5),  # Neutral accuracy
            )

        # Use lax.cond for zero-overhead branching
        # When amp_enabled is a compile-time constant, the unused branch is optimized out
        if amp_enabled:
            amp_rewards, new_disc_params, new_disc_opt, disc_loss, disc_acc = (
                amp_branch(None)
            )
        else:
            amp_rewards, new_disc_params, new_disc_opt, disc_loss, disc_acc = (
                no_amp_branch(None)
            )

        # ================================================================
        # Step 5: Combine rewards
        # ================================================================
        total_rewards = compute_reward_total(
            trajectory.task_rewards, amp_rewards, config.amp.weight
        )

        # ================================================================
        # Step 6: Compute GAE advantages
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
        # Step 7: PPO update
        # ================================================================
        batch_size = config.ppo.rollout_steps * config.ppo.num_envs

        flat_obs = trajectory.obs.reshape(batch_size, -1)
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
            actions=flat_actions,
            old_log_probs=flat_log_probs,
            advantages=flat_advantages,
            returns=flat_returns,
            rng=ppo_rng,
            num_epochs=config.ppo.epochs,
            num_minibatches=config.ppo.num_minibatches,
            clip_epsilon=config.ppo.clip_epsilon,
            value_loss_coef=config.ppo.value_loss_coef,
            entropy_coef=config.ppo.entropy_coef,
        )

        # ================================================================
        # Step 8: Build new state and metrics
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

        new_state = TrainingState(
            policy_params=new_policy_params,
            value_params=new_value_params,
            processor_params=state.processor_params,
            policy_opt_state=new_policy_opt,
            value_opt_state=new_value_opt,
            disc_params=new_disc_params,
            disc_opt_state=new_disc_opt,
            feature_mean=state.feature_mean,
            feature_var=state.feature_var,
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
            disc_loss=disc_loss,
            disc_accuracy=disc_acc,
            amp_reward_mean=jnp.mean(amp_rewards),
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
            f"lr={ckpt_config.get('learning_rate', 'N/A')}, "
            f"amp_weight={ckpt_config.get('amp_weight', 'N/A')}"
        )

    def _check_mismatch(key: str, ckpt_value: Any, current_value: Any) -> None:
        if ckpt_value != current_value:
            hard_mismatch.append((key, ckpt_value, current_value))

    _check_mismatch("num_envs", ckpt_config.get("num_envs"), config.ppo.num_envs)
    _check_mismatch(
        "rollout_steps", ckpt_config.get("rollout_steps"), config.ppo.rollout_steps
    )
    _check_mismatch(
        "num_minibatches", ckpt_config.get("num_minibatches"), config.ppo.num_minibatches
    )
    _check_mismatch("epochs", ckpt_config.get("epochs"), config.ppo.epochs)
    _check_mismatch(
        "learning_rate", ckpt_config.get("learning_rate"), config.ppo.learning_rate
    )
    _check_mismatch(
        "clip_epsilon", ckpt_config.get("clip_epsilon"), config.ppo.clip_epsilon
    )
    _check_mismatch(
        "value_loss_coef", ckpt_config.get("value_loss_coef"), config.ppo.value_loss_coef
    )
    _check_mismatch(
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
    _check_mismatch("amp_weight", ckpt_config.get("amp_weight"), config.amp.weight)
    _check_mismatch(
        "disc_hidden_sizes",
        ckpt_config.get("disc_hidden_sizes"),
        config.networks.discriminator.hidden_sizes,
    )
    _check_mismatch(
        "disc_learning_rate",
        ckpt_config.get("disc_learning_rate"),
        config.amp.discriminator.learning_rate,
    )
    _check_mismatch(
        "disc_updates_per_ppo_update",
        ckpt_config.get("disc_updates_per_ppo_update"),
        config.amp.discriminator.updates_per_ppo_update,
    )
    _check_mismatch(
        "disc_batch_size",
        ckpt_config.get("disc_batch_size"),
        config.amp.discriminator.batch_size,
    )
    _check_mismatch(
        "disc_r1_gamma",
        ckpt_config.get("disc_r1_gamma"),
        config.amp.discriminator.r1_gamma,
    )
    _check_mismatch(
        "disc_input_noise_std",
        ckpt_config.get("disc_input_noise_std"),
        config.amp.discriminator.input_noise_std,
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
    ref_motion_data: Optional[jnp.ndarray] = None,
    callback: Optional[
        Callable[[int, TrainingState, IterationMetrics, float], None]
    ] = None,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
    *,
    policy_init_action: Optional[jnp.ndarray] = None,
) -> TrainingState:
    """Main training function for both PPO-only and AMP+PPO.

    This is the single entry point for all training modes:
    - Stage 1: config.amp.reward_weight=0, ref_motion_data=None
    - Stage 3: config.amp.reward_weight>0, ref_motion_data=<data>

    Args:
        env_step_fn: Batched environment step function
        env_reset_fn: Batched environment reset function
        config: Training configuration
        ref_motion_data: Reference motion features (required if amp_weight>0)
        callback: Optional callback for logging/checkpointing
        resume_checkpoint: Optional checkpoint data to resume from (from load_checkpoint)

    Returns:
        Final training state
    """
    amp_enabled = config.amp.enabled
    mode_str = "AMP+PPO" if amp_enabled else "PPO-only"

    from policy_contract.spec_builder import build_policy_spec

    robot_config = get_robot_config()
    spec = build_policy_spec(
        robot_name=robot_config.robot_name,
        actuated_joint_specs=robot_config.actuated_joints,
        action_filter_alpha=float(config.env.action_filter_alpha),
    )
    obs_dim = spec.model.obs_dim
    action_dim = spec.model.action_dim
    from policy_contract.spec import policy_spec_hash

    current_spec_hash = policy_spec_hash(spec)

    print("=" * 60)
    print(f"Unified Training ({mode_str})")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  num_envs: {config.ppo.num_envs}")
    print(f"  rollout_steps: {config.ppo.rollout_steps}")
    print(f"  iterations: {config.ppo.iterations}")
    print(f"  amp_weight: {config.amp.weight}")
    print("=" * 60)

    # Validate inputs
    if amp_enabled and ref_motion_data is None:
        raise ValueError(
            f"amp.weight={config.amp.weight} > 0 but ref_motion_data is None. "
            "Either set amp.reward_weight=0 or provide ref_motion_data."
        )

    # Initialize RNG
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng, disc_rng = jax.random.split(rng, 4)

    # Create networks
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=config.networks.actor.hidden_sizes,
        value_hidden_dims=config.networks.critic.hidden_sizes,
    )

    # Initialize network parameters
    processor_params, policy_params, value_params = init_network_params(
        ppo_network,
        obs_dim,
        action_dim,
        seed=int(init_rng[0]),
        policy_init_action=policy_init_action,
    )

    # Create discriminator (always created, may be unused)
    amp_feature_dim = get_feature_config().feature_dim
    disc_model, disc_params = create_discriminator(
        obs_dim=amp_feature_dim,
        hidden_dims=config.networks.discriminator.hidden_sizes,
        seed=int(disc_rng[0]),
    )

    # Create optimizers
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(float(config.ppo.learning_rate)),
    )
    value_optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(float(config.ppo.learning_rate)),
    )
    disc_optimizer = optax.adam(float(config.amp.discriminator.learning_rate))

    # Initialize optimizer states
    policy_opt_state = policy_optimizer.init(policy_params)
    value_opt_state = value_optimizer.init(value_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    # Feature normalization stats
    if amp_enabled:
        ref_buffer_data = jnp.asarray(ref_motion_data, dtype=jnp.float32)
        feature_mean, feature_var = compute_normalization_stats(ref_buffer_data)
        print(f"✓ Reference buffer: {ref_buffer_data.shape[0]} samples on GPU")
    else:
        # Dummy stats for PPO-only mode
        ref_buffer_data = None
        feature_mean = jnp.zeros(amp_feature_dim)
        feature_var = jnp.ones(amp_feature_dim)

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
            disc_params=resume_checkpoint.get("disc_params", disc_params),
            disc_opt_state=resume_checkpoint.get("disc_opt_state", disc_opt_state),
            feature_mean=resume_checkpoint.get("feature_mean", feature_mean),
            feature_var=resume_checkpoint.get("feature_var", feature_var),
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
            disc_params=disc_params,
            disc_opt_state=disc_opt_state,
            feature_mean=feature_mean,
            feature_var=feature_var,
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
        disc_model=disc_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        disc_optimizer=disc_optimizer,
        config=config,
        ref_buffer_data=ref_buffer_data,
    )

    print("=" * 60)
    print("Pre-compiling JIT functions...")
    compile_start = time.time()
    _ = train_iteration_fn(state, env_state)
    jax.block_until_ready(_)
    print(f"  ✓ train_iteration_fn compiled ({time.time() - compile_start:.1f}s)")
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

    for iteration in range(start_iteration + 1, target_iteration + 1):
        iter_start = time.time()

        state, env_state, metrics = train_iteration_fn(state, env_state)
        jax.block_until_ready(state.total_steps)

        iter_time = time.time() - iter_start
        steps_per_sec = (config.ppo.rollout_steps * config.ppo.num_envs) / iter_time

        # Logging
        if iteration % config.ppo.log_interval == 0 or iteration == 1:
            total_steps = int(state.total_steps)
            progress_pct = (total_steps / total_expected_steps) * 100

            # Access env metrics from dict
            env = metrics.env_metrics

            # Main metrics line - v0.10.3: Include velocity command for walking
            main_line = (
                f"#{iteration:<4} Steps: {total_steps:>10} ({progress_pct:>5.1f}%): "
                f"reward={float(metrics.episode_reward):>8.2f} | "
                f"vel={float(env['forward_velocity']):>5.2f}m/s (cmd={float(env['velocity_command']):>4.2f}) | "
                f"steps/s={steps_per_sec:>8.0f}"
            )

            # Print the main metrics line in yellow to make it stand out in logs
            YELLOW = "\x1b[33m"
            RESET = "\x1b[0m"
            try:
                print(f"{YELLOW}{main_line}{RESET}")
            except Exception:
                # Fallback to normal print if terminal doesn't support ANSI sequences
                print(main_line)

            # Second line: episode metrics + v0.10.3 tracking metrics
            print(
                f"  └─ ep_len={float(metrics.episode_length):>5.0f} | "
                f"height={float(env['height']):>4.2f}m | "
                f"success={float(metrics.success_rate):>4.1%} | "
                f"vel_err={float(env['tracking/vel_error']):>5.3f} | "
                f"torque={float(env['tracking/max_torque']):>4.1%}"
            )

            # v0.10.2: Third line - termination diagnostics (what's causing failures?)
            # Always show termination breakdown for debugging
            print(
                f"  └─ term: h_low={float(env['term_height_low_frac']):>4.1%} | "
                f"h_high={float(env['term_height_high_frac']):>4.1%} | "
                f"pitch={float(env['term_pitch_frac']):>4.1%} | "
                f"roll={float(env['term_roll_frac']):>4.1%}"
            )

            if amp_enabled:
                print(
                    f" | amp={float(metrics.amp_reward_mean):>6.4f} | "
                    f"disc_acc={float(metrics.disc_accuracy):>5.2f}"
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
