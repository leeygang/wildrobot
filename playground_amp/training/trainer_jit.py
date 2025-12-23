"""Fully JIT-compiled AMP+PPO trainer for maximum GPU utilization.

This module implements a training pipeline where the ENTIRE training loop
runs inside JAX/XLA with no Python for-loops in the hot path. This achieves
1000x+ speedup compared to Python-loop based training.

Key optimizations:
- All loops use jax.lax.scan instead of Python for-loops
- Reference motion buffer is GPU-resident (no CPU→GPU transfers)
- Single JIT compilation for entire training
- Minimal Python overhead (only for logging callbacks)

Architecture:
    jax.lax.scan(train_epoch, num_epochs)
        └── train_epoch (pure JAX)
              ├── jax.lax.scan(env.step, unroll_length)  # rollout
              ├── jax.lax.scan(disc_update, num_disc_updates)
              └── jax.lax.scan(ppo_update, num_minibatches * epochs)

Usage:
    from playground_amp.training.trainer_jit import train_amp_ppo_jit

    final_state, metrics = train_amp_ppo_jit(
        env=env,
        config=config,
        ref_motion_data=reference_features,  # Pre-loaded numpy array
    )
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple

import flax.linen as nn

import jax
import jax.numpy as jnp
import optax

from playground_amp.amp.amp_features import (
    AMPFeatureConfig,
    get_default_amp_config,
    extract_amp_features,
)
from playground_amp.amp.discriminator import (
    AMPDiscriminator,
    compute_amp_reward,
    create_discriminator,
    discriminator_loss,
)
from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class AMPPPOConfigJit:
    """Configuration for JIT-compiled AMP+PPO training.

    All values must be static (known at compile time) for JIT.
    """

    # Environment
    obs_dim: int
    action_dim: int
    num_envs: int
    num_steps: int  # Unroll length per iteration

    # PPO hyperparameters
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

    # PPO training
    num_minibatches: int
    update_epochs: int

    # Network architecture
    policy_hidden_dims: Tuple[int, ...]
    value_hidden_dims: Tuple[int, ...]

    # AMP configuration
    amp_reward_weight: float
    disc_learning_rate: float
    disc_updates_per_iter: int
    disc_batch_size: int
    gradient_penalty_weight: float
    disc_hidden_dims: Tuple[int, ...]
    disc_input_noise_std: float  # Gaussian noise std for discriminator inputs (hides simulation artifacts)

    # Training
    total_iterations: int
    seed: int
    log_interval: int


# =============================================================================
# Training State
# =============================================================================


class RunningMeanStdState(NamedTuple):
    """State for online mean/std computation."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class TrainingState(NamedTuple):
    """Complete training state - all JAX arrays, no Python objects."""

    # Network parameters
    policy_params: Any
    value_params: Any
    processor_params: Any
    disc_params: Any

    # Optimizer states
    policy_opt_state: Any
    value_opt_state: Any
    disc_opt_state: Any

    # Feature normalization
    feature_mean: jnp.ndarray
    feature_var: jnp.ndarray
    feature_count: jnp.ndarray

    # Training progress
    iteration: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.Array


class Transition(NamedTuple):
    """Single transition for rollout storage."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    next_obs: jnp.ndarray


class IterationMetrics(NamedTuple):
    """Metrics from one training iteration."""

    # PPO losses
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    total_loss: jnp.ndarray
    # AMP metrics
    disc_loss: jnp.ndarray
    disc_accuracy: jnp.ndarray
    amp_reward_mean: jnp.ndarray
    # Episode metrics
    episode_reward: jnp.ndarray
    # Task reward breakdown (for debugging)
    task_reward_mean: jnp.ndarray  # Mean task reward per step
    forward_velocity: jnp.ndarray  # Actual forward velocity (m/s)
    robot_height: jnp.ndarray  # Robot base height (m)
    episode_length: jnp.ndarray  # Mean episode length (steps)


# =============================================================================
# Reference Motion Buffer (GPU-resident)
# =============================================================================


class JaxReferenceBuffer:
    """GPU-resident reference motion buffer for efficient sampling.

    All data is stored as JAX arrays on GPU. Sampling is done with
    jax.random.choice which stays entirely on GPU.
    """

    def __init__(self, data: jnp.ndarray):
        """Initialize buffer with reference motion features.

        Args:
            data: Reference motion features, shape (num_samples, feature_dim)
        """
        self.data = jnp.asarray(data, dtype=jnp.float32)
        self.num_samples = self.data.shape[0]
        self.feature_dim = self.data.shape[1]

    def sample(self, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        """Sample batch of reference features (pure JAX, stays on GPU).

        Args:
            rng: JAX random key
            batch_size: Number of samples

        Returns:
            Sampled features, shape (batch_size, feature_dim)
        """
        indices = jax.random.choice(
            rng, self.num_samples, shape=(batch_size,), replace=True
        )
        return self.data[indices]


# =============================================================================
# Feature Normalization (Fixed from Reference Data)
# =============================================================================


def compute_normalization_stats(
    data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute normalization statistics from reference data.

    These stats are computed ONCE from reference motion data and used
    throughout training. This prevents distribution drift where the
    running statistics adapt to the policy's motion, causing the
    reference data to appear "out of distribution".

    Args:
        data: Reference motion features, shape (num_samples, feature_dim)

    Returns:
        (mean, var): Fixed normalization statistics
    """
    mean = jnp.mean(data, axis=0)
    var = jnp.var(data, axis=0)
    # Ensure variance is at least some minimum to avoid division by zero
    var = jnp.maximum(var, 1e-8)
    return mean, var


def normalize_features(
    features: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Normalize features using running statistics."""
    return (features - mean) / jnp.sqrt(var + epsilon)


# =============================================================================
# Rollout Collection (jax.lax.scan)
# =============================================================================


def collect_rollout_scan(
    env_step_fn: Callable,
    env_state: Any,
    policy_params: Any,
    value_params: Any,
    processor_params: Any,
    ppo_network: Any,
    rng: jax.Array,
    num_steps: int,
) -> Tuple[Any, Transition, jnp.ndarray]:
    """Collect rollout using jax.lax.scan (fully on GPU).

    Args:
        env_step_fn: Batched environment step function
        env_state: Current environment state
        policy_params: Policy network parameters
        value_params: Value network parameters
        processor_params: Observation processor parameters
        ppo_network: PPO network module
        rng: Random key
        num_steps: Number of steps to collect

    Returns:
        (final_env_state, transitions, bootstrap_value)
    """

    def step_fn(carry, _):
        env_state, rng = carry
        rng, action_rng = jax.random.split(rng)

        obs = env_state.obs

        # Sample action from policy
        action, raw_action, log_prob = sample_actions(
            processor_params, policy_params, ppo_network, obs, action_rng
        )

        # Get value estimate
        value = compute_values(processor_params, value_params, ppo_network, obs)

        # Step environment
        next_env_state = env_step_fn(env_state, action)

        # Create transition
        transition = Transition(
            obs=obs,
            action=raw_action,
            reward=next_env_state.reward,
            done=next_env_state.done,
            log_prob=log_prob,
            value=value,
            next_obs=next_env_state.obs,
        )

        return (next_env_state, rng), transition

    # Run scan
    init_carry = (env_state, rng)
    (final_env_state, _), transitions = jax.lax.scan(
        step_fn, init_carry, None, length=num_steps
    )

    # Get bootstrap value for final state
    bootstrap_value = compute_values(
        processor_params, value_params, ppo_network, final_env_state.obs
    )

    return final_env_state, transitions, bootstrap_value


# =============================================================================
# AMP Feature Extraction (Batched)
# =============================================================================


def extract_amp_features_batched(
    obs: jnp.ndarray,
    config: AMPFeatureConfig,
) -> jnp.ndarray:
    """Extract AMP features from observations (batched).

    Args:
        obs: Observations, shape (num_steps, num_envs, obs_dim)
        config: Feature extraction configuration (REQUIRED)

    Returns:
        AMP features, shape (num_steps, num_envs, feature_dim)
    """
    # Vectorize over steps and envs
    # functools.partial to bind config for vmap
    extract_fn = functools.partial(extract_amp_features, config=config)
    return jax.vmap(jax.vmap(extract_fn))(obs)


# =============================================================================
# Discriminator Training (jax.lax.scan)
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
    gradient_penalty_weight: float,
    input_noise_std: float = 0.0,
) -> Tuple[Any, Any, jnp.ndarray, jnp.ndarray]:
    """Train discriminator using jax.lax.scan.

    Args:
        disc_params: Discriminator parameters
        disc_opt_state: Optimizer state
        disc_model: Discriminator module
        disc_optimizer: Optimizer
        agent_features: Flattened agent features from rollout
        ref_buffer_data: Reference motion data (GPU-resident)
        rng: Random key
        num_updates: Number of discriminator updates
        batch_size: Batch size per update
        gradient_penalty_weight: WGAN-GP penalty weight
        input_noise_std: Gaussian noise std to add to inputs (hides simulation artifacts)

    Returns:
        (new_params, new_opt_state, mean_loss, mean_accuracy)
    """
    num_ref_samples = ref_buffer_data.shape[0]
    num_agent_samples = agent_features.shape[0]

    def disc_update_step(carry, rng):
        params, opt_state = carry
        rng, agent_rng, expert_rng, loss_rng, noise_rng1, noise_rng2 = jax.random.split(
            rng, 6
        )

        # Sample agent features
        agent_indices = jax.random.choice(
            agent_rng, num_agent_samples, shape=(batch_size,), replace=True
        )
        agent_batch = agent_features[agent_indices]

        # Sample expert features
        expert_indices = jax.random.choice(
            expert_rng, num_ref_samples, shape=(batch_size,), replace=True
        )
        expert_batch = ref_buffer_data[expert_indices]

        # Add input noise to both batches (hides simulation artifacts)
        if input_noise_std > 0:
            agent_batch = (
                agent_batch
                + jax.random.normal(noise_rng1, agent_batch.shape) * input_noise_std
            )
            expert_batch = (
                expert_batch
                + jax.random.normal(noise_rng2, expert_batch.shape) * input_noise_std
            )

        # Compute loss and gradients
        def loss_fn(p):
            return discriminator_loss(
                params=p,
                model=disc_model,
                real_obs=expert_batch,
                fake_obs=agent_batch,
                rng_key=loss_rng,
                gradient_penalty_weight=gradient_penalty_weight,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Update parameters
        updates, new_opt_state = disc_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return (new_params, new_opt_state), (loss, metrics["discriminator_accuracy"])

    # Run scan over updates
    rngs = jax.random.split(rng, num_updates)
    (final_params, final_opt_state), (losses, accuracies) = jax.lax.scan(
        disc_update_step, (disc_params, disc_opt_state), rngs
    )

    return final_params, final_opt_state, jnp.mean(losses), jnp.mean(accuracies)


# =============================================================================
# PPO Update (jax.lax.scan)
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
    """PPO update using jax.lax.scan over epochs and minibatches.

    Returns:
        Updated params, opt_states, and mean metrics
    """
    batch_size = obs.shape[0]
    minibatch_size = batch_size // num_minibatches
    num_updates = num_epochs * num_minibatches

    def ppo_step(carry, update_idx):
        policy_p, value_p, policy_opt, value_opt, rng = carry
        rng, shuffle_rng, step_rng = jax.random.split(rng, 3)

        # Compute which epoch and minibatch we're in
        epoch_idx = update_idx // num_minibatches
        mb_idx = update_idx % num_minibatches

        # Generate a fresh permutation for each epoch
        # Use epoch_idx to generate consistent permutation within an epoch
        epoch_rng = jax.random.fold_in(shuffle_rng, epoch_idx)
        perm = jax.random.permutation(epoch_rng, batch_size)

        # Get minibatch indices using dynamic_slice
        start_idx = mb_idx * minibatch_size
        mb_indices = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))

        # Extract minibatch using advanced indexing
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

        # Update policy
        policy_updates, new_policy_opt = policy_optimizer.update(
            policy_grads, policy_opt
        )
        new_policy_p = optax.apply_updates(policy_p, policy_updates)

        # Update value
        value_updates, new_value_opt = value_optimizer.update(value_grads, value_opt)
        new_value_p = optax.apply_updates(value_p, value_updates)

        return (new_policy_p, new_value_p, new_policy_opt, new_value_opt, rng), metrics

    # Run scan over update indices
    update_indices = jnp.arange(num_updates)

    init_carry = (policy_params, value_params, policy_opt_state, value_opt_state, rng)
    final_carry, all_metrics = jax.lax.scan(ppo_step, init_carry, update_indices)

    new_policy_params, new_value_params, new_policy_opt, new_value_opt, _ = final_carry

    # Average metrics
    mean_policy_loss = jnp.mean(all_metrics.policy_loss)
    mean_value_loss = jnp.mean(all_metrics.value_loss)
    mean_entropy_loss = jnp.mean(all_metrics.entropy_loss)
    mean_total_loss = jnp.mean(all_metrics.total_loss)

    return (
        new_policy_params,
        new_value_params,
        new_policy_opt,
        new_value_opt,
        mean_policy_loss,
        mean_value_loss,
        mean_entropy_loss,
        mean_total_loss,
    )


# =============================================================================
# Single Training Iteration (JIT-compiled)
# =============================================================================


def make_train_iteration_fn(
    env_step_fn: Callable,
    ppo_network: Any,
    disc_model: AMPDiscriminator,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    disc_optimizer: optax.GradientTransformation,
    config: AMPPPOConfigJit,
):
    """Create JIT-compiled training iteration function.

    This is a factory that creates the train_iteration function with
    static arguments baked in for maximum JIT efficiency.
    """
    # Get AMP feature config once (before JIT) and capture in closure
    amp_feature_config = get_default_amp_config()

    @jax.jit
    def train_iteration(
        state: TrainingState,
        env_state: Any,
        ref_buffer_data: jnp.ndarray,
    ) -> Tuple[TrainingState, Any, IterationMetrics]:
        """Execute one training iteration (fully on GPU)."""

        rng = state.rng
        rng, rollout_rng, disc_rng, ppo_rng = jax.random.split(rng, 4)

        # ====================================================================
        # Step 1: Collect rollout using jax.lax.scan
        # ====================================================================
        new_env_state, transitions, bootstrap_value = collect_rollout_scan(
            env_step_fn=env_step_fn,
            env_state=env_state,
            policy_params=state.policy_params,
            value_params=state.value_params,
            processor_params=state.processor_params,
            ppo_network=ppo_network,
            rng=rollout_rng,
            num_steps=config.num_steps,
        )

        # transitions shape: (num_steps, num_envs, ...)

        # ====================================================================
        # Step 2: Extract and normalize AMP features
        # ====================================================================
        amp_features = extract_amp_features_batched(transitions.obs, amp_feature_config)
        # Shape: (num_steps, num_envs, feature_dim)

        # Flatten for discriminator training
        flat_amp_features = amp_features.reshape(-1, amp_features.shape[-1])

        # Use FIXED normalization stats from reference data (prevents distribution drift)
        # These were computed once at initialization from ref_motion_data
        # Both policy and reference features use the same fixed stats
        norm_amp_features = normalize_features(
            flat_amp_features, state.feature_mean, state.feature_var
        )

        # Reference data also uses same fixed stats (computed from itself)
        norm_ref_data = normalize_features(
            ref_buffer_data, state.feature_mean, state.feature_var
        )

        # ====================================================================
        # Step 3: Train discriminator using jax.lax.scan
        # ====================================================================
        new_disc_params, new_disc_opt_state, disc_loss, disc_accuracy = (
            train_discriminator_scan(
                disc_params=state.disc_params,
                disc_opt_state=state.disc_opt_state,
                disc_model=disc_model,
                disc_optimizer=disc_optimizer,
                agent_features=norm_amp_features,
                ref_buffer_data=norm_ref_data,
                rng=disc_rng,
                num_updates=config.disc_updates_per_iter,
                batch_size=config.disc_batch_size,
                gradient_penalty_weight=config.gradient_penalty_weight,
                input_noise_std=config.disc_input_noise_std,
            )
        )

        # ====================================================================
        # Step 4: Compute AMP rewards
        # ====================================================================
        # Reshape normalized features back to (num_steps, num_envs, feature_dim)
        norm_amp_features_shaped = norm_amp_features.reshape(amp_features.shape)

        # Compute AMP reward for each transition
        def compute_amp_reward_single(feat):
            return compute_amp_reward(new_disc_params, disc_model, feat)

        amp_rewards = jax.vmap(jax.vmap(compute_amp_reward_single))(
            norm_amp_features_shaped
        )
        # Shape: (num_steps, num_envs)

        amp_reward_mean = jnp.mean(amp_rewards)

        # Shape rewards: task_reward + amp_weight * amp_reward
        shaped_rewards = transitions.reward + config.amp_reward_weight * amp_rewards

        # ====================================================================
        # Step 5: Compute GAE advantages
        # ====================================================================
        advantages, returns = compute_gae(
            rewards=shaped_rewards,
            values=transitions.value,
            dones=transitions.done,
            bootstrap_value=bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # ====================================================================
        # Step 6: PPO update using jax.lax.scan
        # ====================================================================
        # Flatten batch: (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        batch_size = config.num_steps * config.num_envs

        flat_obs = transitions.obs.reshape(batch_size, -1)
        flat_actions = transitions.action.reshape(batch_size, -1)
        flat_log_probs = transitions.log_prob.reshape(batch_size)
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
            num_epochs=config.update_epochs,
            num_minibatches=config.num_minibatches,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
        )

        # ====================================================================
        # Step 7: Update state and collect metrics
        # ====================================================================
        env_steps = config.num_steps * config.num_envs

        # Task reward breakdown (mean per step)
        task_reward_mean = jnp.mean(transitions.reward)

        # Episode metrics
        episode_reward = jnp.mean(jnp.sum(transitions.reward, axis=0))

        # Extract environment metrics from the final env state
        # In vmapped context, metrics values are arrays of shape (num_envs,)
        # Direct dict access - keys are guaranteed to exist in WildRobotEnv
        forward_velocity = jnp.mean(new_env_state.metrics["forward_velocity"])
        robot_height = jnp.mean(new_env_state.metrics["height"])

        # Episode length: actual step count since last reset (from environment)
        # This is the TRUE episode length, max = max_episode_steps (500)
        episode_length = jnp.mean(new_env_state.info["step_count"])

        new_state = TrainingState(
            policy_params=new_policy_params,
            value_params=new_value_params,
            processor_params=state.processor_params,
            disc_params=new_disc_params,
            policy_opt_state=new_policy_opt,
            value_opt_state=new_value_opt,
            disc_opt_state=new_disc_opt_state,
            feature_mean=state.feature_mean,  # Fixed - never changes
            feature_var=state.feature_var,    # Fixed - never changes
            feature_count=state.feature_count,  # Not used anymore, kept for API compat
            iteration=state.iteration + 1,
            total_steps=state.total_steps + env_steps,
            rng=rng,
        )

        metrics = IterationMetrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            total_loss=total_loss,
            disc_loss=disc_loss,
            disc_accuracy=disc_accuracy,
            amp_reward_mean=amp_reward_mean,
            episode_reward=episode_reward,
            task_reward_mean=task_reward_mean,
            forward_velocity=forward_velocity,
            robot_height=robot_height,
            episode_length=episode_length,
        )

        return new_state, new_env_state, metrics

    return train_iteration


# =============================================================================
# Batch Training (Multiple iterations in one JIT call)
# =============================================================================


def make_train_n_iterations_fn(
    train_iteration_fn: Callable,
    n_iterations: int,
):
    """Create function that runs N training iterations in a single JIT call.

    This further reduces Python overhead by batching multiple iterations.
    """

    @jax.jit
    def train_n_iterations(
        state: TrainingState,
        env_state: Any,
        ref_buffer_data: jnp.ndarray,
    ) -> Tuple[TrainingState, Any, IterationMetrics]:
        """Run N training iterations (all on GPU)."""

        def iteration_body(carry, _):
            state, env_state = carry
            new_state, new_env_state, metrics = train_iteration_fn(
                state, env_state, ref_buffer_data
            )
            return (new_state, new_env_state), metrics

        (final_state, final_env_state), all_metrics = jax.lax.scan(
            iteration_body, (state, env_state), None, length=n_iterations
        )

        # Return last metrics (or could aggregate)
        last_metrics = jax.tree.map(lambda x: x[-1], all_metrics)

        return final_state, final_env_state, last_metrics

    return train_n_iterations


# =============================================================================
# Main Training Function
# =============================================================================


def train_amp_ppo_jit(
    env_step_fn: Callable,
    env_reset_fn: Callable,
    config: AMPPPOConfigJit,
    ref_motion_data: jnp.ndarray,
    callback: Optional[
        Callable[[int, TrainingState, IterationMetrics, float], None]
    ] = None,
) -> TrainingState:
    """Main training function with fully JIT-compiled training loop.

    Args:
        env_step_fn: Batched environment step function (state, action) -> new_state
        env_reset_fn: Batched environment reset function (rng) -> state
        config: Training configuration
        ref_motion_data: Reference motion features, shape (num_samples, feature_dim)
        callback: Optional callback for logging and checkpoint management,
                  called every log_interval with (iteration, state, metrics, steps_per_sec)

    Returns:
        Final training state
    """
    print("=" * 60)
    print("JIT-Compiled AMP+PPO Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  obs_dim: {config.obs_dim}")
    print(f"  action_dim: {config.action_dim}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  total_iterations: {config.total_iterations}")
    print(f"  amp_reward_weight: {config.amp_reward_weight}")
    print("=" * 60)

    # Initialize RNG
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng, disc_rng = jax.random.split(rng, 4)

    # Create networks
    ppo_network = create_networks(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        policy_hidden_dims=config.policy_hidden_dims,
        value_hidden_dims=config.value_hidden_dims,
    )

    # Initialize network parameters
    processor_params, policy_params, value_params = init_network_params(
        ppo_network, config.obs_dim, config.action_dim, seed=int(init_rng[0])
    )

    # Create discriminator
    amp_feature_dim = get_default_amp_config().feature_dim
    disc_model, disc_params = create_discriminator(
        obs_dim=amp_feature_dim,
        hidden_dims=config.disc_hidden_dims,
        seed=int(disc_rng[0]),
    )

    # Create optimizers
    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(float(config.learning_rate)),
    )
    value_optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(float(config.learning_rate)),
    )
    disc_optimizer = optax.adam(float(config.disc_learning_rate))

    # Initialize optimizer states
    policy_opt_state = policy_optimizer.init(policy_params)
    value_opt_state = value_optimizer.init(value_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    # Move reference data to GPU first (needed for normalization stats)
    ref_buffer_data = jnp.asarray(ref_motion_data, dtype=jnp.float32)
    print(f"✓ Reference buffer: {ref_buffer_data.shape[0]} samples on GPU")

    # Compute FIXED normalization stats from reference data
    # This prevents distribution drift during training
    feature_mean, feature_var = compute_normalization_stats(ref_buffer_data)
    feature_count = jnp.array(float(ref_buffer_data.shape[0]))  # Not used, kept for compat
    print(f"✓ Fixed normalization stats from reference data")

    # Create initial training state
    state = TrainingState(
        policy_params=policy_params,
        value_params=value_params,
        processor_params=processor_params,
        disc_params=disc_params,
        policy_opt_state=policy_opt_state,
        value_opt_state=value_opt_state,
        disc_opt_state=disc_opt_state,
        feature_mean=feature_mean,
        feature_var=feature_var,
        feature_count=feature_count,
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
    )

    # Optionally create batch training function for even more efficiency
    batch_size = min(config.log_interval, 10)  # Run this many iterations per JIT call
    if batch_size > 1:
        train_batch_fn = make_train_n_iterations_fn(train_iteration_fn, batch_size)
    else:
        train_batch_fn = None

    # Calculate total expected steps for progress tracking
    total_expected_steps = config.total_iterations * config.num_envs * config.num_steps

    print("=" * 60)
    print("Pre-compiling JIT functions...")

    # Trigger JIT compilation before training starts
    # This prevents a 30s pause after iteration #1 when switching to batch mode
    compile_start = time.time()
    _ = train_iteration_fn(state, env_state, ref_buffer_data)
    jax.block_until_ready(_)
    print(f"  ✓ train_iteration_fn compiled ({time.time() - compile_start:.1f}s)")

    if train_batch_fn is not None:
        compile_start = time.time()
        _ = train_batch_fn(state, env_state, ref_buffer_data)
        jax.block_until_ready(_)
        print(f"  ✓ train_batch_fn compiled ({time.time() - compile_start:.1f}s)")

    print("=" * 60)
    print("Starting training...")
    print(f"  Total iterations: {config.total_iterations:,}")
    print(f"  Total steps: {total_expected_steps:,}")
    print()

    # Training loop
    start_time = time.time()
    iteration = 0

    while iteration < config.total_iterations:
        iter_start = time.time()

        if (
            train_batch_fn is not None
            and iteration + batch_size <= config.total_iterations
        ):
            # Run batch of iterations
            state, env_state, metrics = train_batch_fn(
                state, env_state, ref_buffer_data
            )
            iteration += batch_size
        else:
            # Run single iteration
            state, env_state, metrics = train_iteration_fn(
                state, env_state, ref_buffer_data
            )
            iteration += 1

        # Force sync to get accurate timing
        jax.block_until_ready(state.total_steps)

        iter_time = time.time() - iter_start
        steps_per_sec = (
            (config.num_steps * config.num_envs * batch_size) / iter_time
            if train_batch_fn
            else (config.num_steps * config.num_envs) / iter_time
        )

        # Logging and callback (handles W&B logging and checkpoint saving)
        if iteration % config.log_interval == 0 or iteration == 1:
            total_steps = int(state.total_steps)
            progress_pct = (total_steps / total_expected_steps) * 100

            # Main metrics line
            print(
                f"#{iteration:<4} Steps: {total_steps:>10} ({progress_pct:>5.1f}%): "
                f"reward={float(metrics.episode_reward):>8.2f} | "
                f"amp={float(metrics.amp_reward_mean):>7.4f} | "
                f"disc_acc={float(metrics.disc_accuracy):>5.2f} | "
                f"steps/s={steps_per_sec:>8.0f}"
            )
            # Task breakdown line (indented)
            print(
                f"  └─ task_r={float(metrics.task_reward_mean):>6.3f}/step | "
                f"vel={float(metrics.forward_velocity):>5.2f}m/s | "
                f"height={float(metrics.robot_height):>5.3f}m | "
                f"ep_len={float(metrics.episode_length):>5.1f}"
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
