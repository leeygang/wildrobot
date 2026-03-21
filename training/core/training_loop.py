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
import numpy as np
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
    PPOLossMetrics,
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
    METRICS_VEC_KEY,
    unpack_metrics,
)
from training.envs.env_info import WR_INFO_KEY


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
        active_ppo_epochs: int,
        entropy_coef: float,
        learning_rate_scale: float,
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
            use_privileged_critic=config.ppo.critic_privileged_enabled,
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
    eval_env_step_fn: Optional[Callable] = None,
    eval_env_step_fn_no_push: Optional[Callable] = None,
    eval_env_reset_fn: Optional[Callable] = None,
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
        eval_env_step_fn_no_push: Optional eval step fn with disturbances disabled

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
        layout_id=str(config.env.actor_obs_layout_id),
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
    print(f"Unified Training ({mode_str})")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  critic_obs_dim: {critic_obs_dim}")
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
        critic_obs_dim=critic_obs_dim,
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
        ]:
            eval_env_state = eval_reset_fn(eval_rng)
            reset_h = eval_env_state.metrics["height"]
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
            eval_push_reward, eval_push_success, eval_push_ep_len, eval_push_term_h_low, eval_push_term_h_high, eval_push_term_pitch, eval_push_term_roll, eval_push_done_env, eval_push_trunc_env, eval_push_survival_rate, eval_push_survival_steps, eval_push_reset_h_mean, eval_push_reset_h_min = (
                run_eval_push(
                    state.policy_params,
                    state.processor_params,
                    eval_base_rng,
                )
            )
            if eval_has_clean_pass:
                eval_clean_reward, eval_clean_success, eval_clean_ep_len, eval_clean_term_h_low, eval_clean_term_h_high, eval_clean_term_pitch, eval_clean_term_roll, eval_clean_done_env, eval_clean_trunc_env, eval_clean_survival_rate, eval_clean_survival_steps, eval_clean_reset_h_mean, eval_clean_reset_h_min = (
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
            }
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
                        disc_params=snap.disc_params,
                        disc_opt_state=snap.disc_opt_state,
                        feature_mean=snap.feature_mean,
                        feature_var=snap.feature_var,
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
            print(
                f"  └─ stress: torque_sat={float(env['debug/torque_sat_frac']):>4.1%} | "
                f"torque_max={float(env['debug/torque_abs_max']):>4.1%} | "
                f"action_sat={float(env['debug/action_sat_frac']):>4.1%}"
            )

            print(
                f"  └─ ppo: kl={float(metrics.approx_kl):>6.4f} "
                f"(target={float(env['ppo/target_kl']):>6.4f}) | "
                f"epochs={int(float(env['ppo/epochs_used'])):>2} | "
                f"lr={float(env['ppo/lr']):>8.6f}"
            )

            if "eval_push/success_rate" in env:
                print(
                    f"  └─ eval_push: success={float(env['eval_push/success_rate']):>5.1%} | "
                    f"ep_len={float(env['eval_push/episode_length']):>5.1f} | "
                    f"term_h_low={float(env['eval_push/term_height_low_frac']):>4.1%} | "
                    f"term_h_high={float(env['eval_push/term_height_high_frac']):>4.1%} | "
                    f"term_pitch={float(env['eval_push/term_pitch_frac']):>4.1%} | "
                    f"term_roll={float(env['eval_push/term_roll_frac']):>4.1%} | "
                    f"done_env={float(env.get('eval_push/done_env_frac', 0.0)):>4.1%} | "
                    f"trunc_env={float(env.get('eval_push/trunc_env_frac', 0.0)):>4.1%} | "
                    f"survive={float(env.get('eval_push/survival_rate', 0.0)):>5.1%} "
                    f"@{float(env.get('eval_push/survival_steps', 0.0)):>5.1f} | "
                    f"reset_h={float(env.get('eval_push/reset_height_mean', 0.0)):>4.2f}m"
                )
            if "eval_clean/success_rate" in env:
                print(
                    f"  └─ eval_clean: success={float(env['eval_clean/success_rate']):>5.1%} | "
                    f"ep_len={float(env['eval_clean/episode_length']):>5.1f} | "
                    f"term_h_low={float(env['eval_clean/term_height_low_frac']):>4.1%} | "
                    f"term_h_high={float(env['eval_clean/term_height_high_frac']):>4.1%} | "
                    f"term_pitch={float(env['eval_clean/term_pitch_frac']):>4.1%} | "
                    f"term_roll={float(env['eval_clean/term_roll_frac']):>4.1%} | "
                    f"done_env={float(env.get('eval_clean/done_env_frac', 0.0)):>4.1%} | "
                    f"trunc_env={float(env.get('eval_clean/trunc_env_frac', 0.0)):>4.1%} | "
                    f"survive={float(env.get('eval_clean/survival_rate', 0.0)):>5.1%} "
                    f"@{float(env.get('eval_clean/survival_steps', 0.0)):>5.1f} | "
                    f"reset_h={float(env.get('eval_clean/reset_height_mean', 0.0)):>4.2f}m"
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
