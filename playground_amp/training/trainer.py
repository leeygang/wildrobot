"""Custom PPO+AMP training loop for WildRobot.

This module implements a complete training pipeline that integrates:
- PPO policy optimization using our building blocks
- AMP discriminator for natural motion learning
- Custom rollout collection with AMP features
- Separate discriminator training between rollout and PPO update

The training loop owns all state and provides explicit hooks for:
- Rollout collection with AMP feature extraction
- Discriminator training with reference motion
- AMP reward shaping
- PPO policy/value updates

This solves the Brax v2 impedance mismatch by not using ppo.train().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax

import jax.numpy as jnp
import numpy as np
import optax
from playground_amp.amp.amp_features import (
    create_running_stats,
    DEFAULT_AMP_CONFIG,
    extract_amp_features,
    normalize_features,
    RunningMeanStd,
    update_running_stats,
)
from playground_amp.amp.discriminator import (
    AMPDiscriminator,
    compute_amp_reward,
    create_discriminator,
    create_discriminator_optimizer,
    discriminator_loss,
)
from playground_amp.amp.ref_buffer import create_reference_buffer, ReferenceMotionBuffer
from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    create_optimizer,
    init_network_params,
    PPOLossMetrics,
    sample_actions,
)

from playground_amp.training.transitions import (
    AMPTransition,
    RolloutBatch,
    stack_transitions,
)


@dataclass
class AMPPPOConfig:
    """Configuration for AMP+PPO training.

    All parameters must be provided explicitly - no defaults.
    Load from wildrobot_phase3_training.yaml and pass to this config.
    """

    # Environment (required - from env section)
    obs_dim: int
    action_dim: int
    num_envs: int
    num_steps: int  # Steps per rollout before update
    max_episode_steps: int

    # PPO hyperparameters (required - from trainer section)
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

    # PPO training (required - from trainer section)
    num_minibatches: int
    update_epochs: int

    # Network architecture (required - from networks section)
    policy_hidden_dims: Tuple[int, ...]
    value_hidden_dims: Tuple[int, ...]
    log_std_min: float
    log_std_max: float

    # AMP configuration (required - from amp section)
    enable_amp: bool
    amp_reward_weight: float
    disc_learning_rate: float
    disc_updates_per_iter: int
    disc_batch_size: int
    gradient_penalty_weight: float
    disc_input_noise_std: float  # Gaussian noise std for discriminator inputs

    # AMP discriminator architecture (required - from amp section)
    disc_hidden_dims: Tuple[int, ...]

    # Reference motion (required - from amp section)
    ref_buffer_size: int
    ref_seq_len: int
    ref_motion_mode: str  # "walking", "standing", or "file"
    ref_motion_path: Optional[str]

    # Feature normalization
    normalize_amp_features: bool

    # Training (required - from trainer section)
    total_iterations: int
    seed: int

    # Logging (required - from trainer section)
    log_interval: int
    save_interval: int
    checkpoint_dir: str


class AMPPPOState(NamedTuple):
    """Complete training state for AMP+PPO."""

    # Processor params for observation normalization
    processor_params: Any

    # Policy/Value parameters and optimizers
    policy_params: Any
    value_params: Any
    policy_opt_state: Any
    value_opt_state: Any

    # Discriminator parameters and optimizer
    disc_params: Any
    disc_opt_state: Any

    # Feature normalization statistics
    amp_feature_stats: RunningMeanStd

    # Training progress
    iteration: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.Array


class TrainingMetrics(NamedTuple):
    """Metrics from one training iteration."""

    # PPO metrics
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    clip_fraction: float
    approx_kl: float

    # AMP metrics
    disc_loss: float
    disc_accuracy: float
    amp_reward_mean: float
    amp_reward_std: float

    # Environment metrics
    episode_reward: float
    episode_length: float
    env_steps_per_sec: float


def create_amp_ppo_state(
    config: AMPPPOConfig,
    rng: jax.Array,
) -> Tuple[AMPPPOState, Any, AMPDiscriminator, optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation]:
    """Create initial training state.

    Args:
        config: Training configuration
        rng: Random key

    Returns:
        (state, ppo_network, disc_model, policy_optimizer, value_optimizer, disc_optimizer):
        State, network modules, and optimizers
    """
    rng, policy_rng, value_rng, disc_rng = jax.random.split(rng, 4)

    # Create PPO networks using Brax's factory
    ppo_network = create_networks(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        policy_hidden_dims=config.policy_hidden_dims,
        value_hidden_dims=config.value_hidden_dims,
    )

    # Initialize parameters (returns processor_params, policy_params, value_params)
    processor_params, policy_params, value_params = init_network_params(
        ppo_network, config.obs_dim, config.action_dim, seed=int(policy_rng[0])
    )

    # Create discriminator
    amp_feature_dim = DEFAULT_AMP_CONFIG.feature_dim
    disc_model, disc_params = create_discriminator(
        obs_dim=amp_feature_dim,
        seed=int(disc_rng[0]),
    )

    # Create optimizers (create ONCE and return them)
    policy_optimizer = create_optimizer(
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
    )
    value_optimizer = create_optimizer(
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
    )
    disc_optimizer = create_discriminator_optimizer(
        learning_rate=config.disc_learning_rate,
    )

    # Initialize optimizer states
    policy_opt_state = policy_optimizer.init(policy_params)
    value_opt_state = value_optimizer.init(value_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    # Initialize feature normalization stats
    amp_feature_stats = create_running_stats(amp_feature_dim)

    state = AMPPPOState(
        processor_params=processor_params,
        policy_params=policy_params,
        value_params=value_params,
        policy_opt_state=policy_opt_state,
        value_opt_state=value_opt_state,
        disc_params=disc_params,
        disc_opt_state=disc_opt_state,
        amp_feature_stats=amp_feature_stats,
        iteration=jnp.array(0, dtype=jnp.int32),
        total_steps=jnp.array(0, dtype=jnp.int32),  # Use int32 instead of int64
        rng=rng,
    )

    return state, ppo_network, disc_model, policy_optimizer, value_optimizer, disc_optimizer


def collect_rollout(
    env_step_fn: Callable,
    env_state: Any,
    ppo_network: Any,
    processor_params: Any,
    policy_params: Any,
    value_params: Any,
    rng: jax.Array,
    num_steps: int,
    num_envs: int,
) -> Tuple[Any, AMPTransition, jnp.ndarray]:
    """Collect rollout data from environment.

    Args:
        env_step_fn: Function to step environment (state, action) -> (new_state, obs, reward, done, info)
        env_state: Current environment state
        ppo_network: PPO networks from create_networks()
        processor_params: Observation normalization parameters (empty tuple if not used)
        policy_params: Policy parameters
        value_params: Value parameters
        rng: Random key
        num_steps: Number of steps to collect
        num_envs: Number of parallel environments

    Returns:
        (final_env_state, transitions, bootstrap_value):
            Final state, collected transitions, and value estimate for final state
    """
    transitions_list = []

    current_state = env_state
    current_obs = env_state.obs if hasattr(env_state, "obs") else env_state

    for step in range(num_steps):
        rng, action_rng, step_rng = jax.random.split(rng, 3)

        # Get action from policy using Brax's network API
        # sample_actions returns (postprocessed_actions, raw_actions, log_probs)
        postprocessed_action, raw_action, log_prob = sample_actions(
            processor_params, policy_params, ppo_network, current_obs, action_rng
        )

        # Get value estimate
        value = compute_values(processor_params, value_params, ppo_network, current_obs)

        # Step environment with postprocessed actions
        new_state, new_obs, reward, done, info = env_step_fn(current_state, postprocessed_action)

        # Extract AMP features
        amp_feat = extract_amp_features(current_obs, new_obs)

        # Store transition
        # NOTE: We store RAW actions for PPO loss computation (log_prob is computed on raw actions)
        transition = AMPTransition(
            obs=current_obs,
            next_obs=new_obs,
            action=raw_action,  # Store raw actions for PPO
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            amp_feat=amp_feat,
        )
        transitions_list.append(transition)

        # Update state
        current_state = new_state
        current_obs = new_obs

    # Stack transitions
    transitions = stack_transitions(transitions_list)

    # Get bootstrap value for final state
    bootstrap_value = compute_values(processor_params, value_params, ppo_network, current_obs)

    return current_state, transitions, bootstrap_value


def train_discriminator_step(
    disc_params: Any,
    disc_opt_state: Any,
    disc_model: AMPDiscriminator,
    disc_optimizer: optax.GradientTransformation,
    agent_features: jnp.ndarray,
    expert_features: jnp.ndarray,
    rng: jax.Array,
    gradient_penalty_weight: float = 5.0,
    input_noise_std: float = 0.0,
) -> Tuple[Any, Any, Dict[str, float]]:
    """Single discriminator training step.

    Args:
        disc_params: Discriminator parameters
        disc_opt_state: Optimizer state
        disc_model: Discriminator module
        disc_optimizer: Optimizer
        agent_features: Features from policy rollout (batch, feat_dim)
        expert_features: Features from reference motion (batch, feat_dim)
        rng: Random key
        gradient_penalty_weight: Weight for gradient penalty
        input_noise_std: Std of Gaussian noise to add to inputs

    Returns:
        (new_params, new_opt_state, metrics): Updated state and metrics
    """
    # Ensure features are float32 (not traced or float64)
    agent_features = jnp.asarray(agent_features, dtype=jnp.float32)
    expert_features = jnp.asarray(expert_features, dtype=jnp.float32)

    def loss_fn(params):
        return discriminator_loss(
            params=params,
            model=disc_model,
            real_obs=expert_features,
            fake_obs=agent_features,
            rng_key=rng,
            gradient_penalty_weight=gradient_penalty_weight,
            input_noise_std=input_noise_std,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_params)

    # Use optax directly to update - avoid potential issues with optimizer state
    updates, new_opt_state = disc_optimizer.update(grads, disc_opt_state, disc_params)
    new_params = optax.apply_updates(disc_params, updates)

    return new_params, new_opt_state, metrics


def ppo_update_step(
    processor_params: Any,
    policy_params: Any,
    value_params: Any,
    policy_opt_state: Any,
    value_opt_state: Any,
    ppo_network: Any,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    batch: RolloutBatch,
    rng: jax.Array,
    config: AMPPPOConfig,
) -> Tuple[Any, Any, Any, Any, PPOLossMetrics]:
    """Single PPO update step on a minibatch.

    Args:
        processor_params: Observation normalization parameters (empty tuple if not used)
        policy_params: Policy parameters
        value_params: Value parameters
        policy_opt_state: Policy optimizer state
        value_opt_state: Value optimizer state
        ppo_network: PPO networks from create_networks()
        policy_optimizer: Policy optimizer
        value_optimizer: Value optimizer
        batch: Minibatch of rollout data
        rng: Random key for entropy computation
        config: Training configuration

    Returns:
        Updated parameters, optimizer states, and metrics
    """

    def loss_fn(policy_p, value_p):
        return compute_ppo_loss(
            processor_params=processor_params,
            policy_params=policy_p,
            value_params=value_p,
            ppo_network=ppo_network,
            obs=batch.obs,
            actions=batch.actions,
            old_log_probs=batch.log_probs,
            advantages=batch.advantages,
            returns=batch.returns,
            rng=rng,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
        )

    # Compute gradients
    (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_params, value_params)

    # Update policy
    policy_updates, new_policy_opt_state = policy_optimizer.update(
        policy_grads, policy_opt_state
    )
    new_policy_params = optax.apply_updates(policy_params, policy_updates)

    # Update value
    value_updates, new_value_opt_state = value_optimizer.update(
        value_grads, value_opt_state
    )
    new_value_params = optax.apply_updates(value_params, value_updates)

    return (
        new_policy_params,
        new_value_params,
        new_policy_opt_state,
        new_value_opt_state,
        metrics,
    )


def train_iteration(
    state: AMPPPOState,
    env_step_fn: Callable,
    env_state: Any,
    ppo_network: Any,
    disc_model: AMPDiscriminator,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    disc_optimizer: optax.GradientTransformation,
    ref_buffer: ReferenceMotionBuffer,
    config: AMPPPOConfig,
) -> Tuple[AMPPPOState, Any, TrainingMetrics]:
    """Execute one training iteration.

    This is the core training loop that:
    1. Collects rollout with AMP features
    2. Trains discriminator
    3. Computes AMP-shaped rewards
    4. Updates policy/value with PPO

    Args:
        state: Current training state
        env_step_fn: Environment step function
        env_state: Current environment state
        ppo_network: PPO networks from create_networks()
        disc_model: Discriminator module
        policy_optimizer: Policy optimizer
        value_optimizer: Value optimizer
        disc_optimizer: Discriminator optimizer
        ref_buffer: Reference motion buffer
        config: Training configuration

    Returns:
        (new_state, new_env_state, metrics): Updated state and metrics
    """
    start_time = time.time()

    rng = state.rng
    rng, rollout_rng, disc_rng, ppo_rng = jax.random.split(rng, 4)

    # ========================================================================
    # Step 1: Collect rollout with AMP features
    # ========================================================================

    new_env_state, transitions, bootstrap_value = collect_rollout(
        env_step_fn=env_step_fn,
        env_state=env_state,
        ppo_network=ppo_network,
        processor_params=state.processor_params,
        policy_params=state.policy_params,
        value_params=state.value_params,
        rng=rollout_rng,
        num_steps=config.num_steps,
        num_envs=config.num_envs,
    )

    # ========================================================================
    # Step 2: Train discriminator (if AMP enabled)
    # ========================================================================

    disc_params = state.disc_params
    disc_opt_state = state.disc_opt_state
    amp_feature_stats = state.amp_feature_stats

    disc_loss = 0.0
    disc_accuracy = 0.5
    amp_reward_mean = 0.0
    amp_reward_std = 0.0

    if config.enable_amp:
        # Flatten AMP features from rollout
        agent_features = transitions.amp_feat.reshape(
            -1, transitions.amp_feat.shape[-1]
        )

        # Update feature normalization stats
        amp_feature_stats = update_running_stats(amp_feature_stats, agent_features)

        # Normalize features
        if config.normalize_amp_features:
            agent_features_norm = normalize_features(agent_features, amp_feature_stats)
        else:
            agent_features_norm = agent_features

        # Train discriminator multiple times per iteration
        for disc_step in range(config.disc_updates_per_iter):
            rng, disc_step_rng, sample_rng = jax.random.split(rng, 3)

            # Sample expert features from reference buffer
            # The buffer stores 29-dim AMP features directly (not observations)
            expert_features = jnp.array(ref_buffer.sample_single_frames(config.disc_batch_size))

            if config.normalize_amp_features:
                expert_features = normalize_features(expert_features, amp_feature_stats)

            # Sample agent features
            batch_indices = jax.random.choice(
                sample_rng,
                agent_features_norm.shape[0],
                shape=(config.disc_batch_size,),
                replace=True,
            )
            agent_batch = agent_features_norm[batch_indices]

            # Update discriminator
            disc_params, disc_opt_state, disc_metrics = train_discriminator_step(
                disc_params=disc_params,
                disc_opt_state=disc_opt_state,
                disc_model=disc_model,
                disc_optimizer=disc_optimizer,
                agent_features=agent_batch,
                expert_features=expert_features,
                rng=disc_step_rng,
                gradient_penalty_weight=config.gradient_penalty_weight,
                input_noise_std=config.disc_input_noise_std,
            )

            disc_loss = float(disc_metrics["discriminator_loss"])
            disc_accuracy = float(disc_metrics["discriminator_accuracy"])

        # Compute AMP reward for all transitions
        if config.normalize_amp_features:
            amp_features_for_reward = normalize_features(
                transitions.amp_feat.reshape(-1, transitions.amp_feat.shape[-1]),
                amp_feature_stats,
            ).reshape(transitions.amp_feat.shape)
        else:
            amp_features_for_reward = transitions.amp_feat

        amp_rewards = jax.vmap(
            lambda feat: compute_amp_reward(disc_params, disc_model, feat)
        )(amp_features_for_reward.reshape(-1, amp_features_for_reward.shape[-1]))
        amp_rewards = amp_rewards.reshape(transitions.reward.shape)

        amp_reward_mean = float(jnp.mean(amp_rewards))
        amp_reward_std = float(jnp.std(amp_rewards))

        # Shape rewards: task_reward + amp_weight * amp_reward
        shaped_rewards = transitions.reward + config.amp_reward_weight * amp_rewards
    else:
        shaped_rewards = transitions.reward

    # ========================================================================
    # Step 3: Compute advantages with shaped rewards
    # ========================================================================

    advantages, returns = compute_gae(
        rewards=shaped_rewards,
        values=transitions.value,
        dones=transitions.done,
        bootstrap_value=bootstrap_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    # ========================================================================
    # Step 4: PPO update
    # ========================================================================

    # Flatten data for minibatch training
    batch_size = config.num_steps * config.num_envs

    def flatten(x):
        if x is None:
            return None
        return x.reshape((batch_size,) + x.shape[2:])

    flat_batch = RolloutBatch(
        obs=flatten(transitions.obs),
        actions=flatten(transitions.action),
        rewards=flatten(shaped_rewards),
        dones=flatten(transitions.done),
        log_probs=flatten(transitions.log_prob),
        values=flatten(transitions.value),
        advantages=flatten(advantages),
        returns=flatten(returns),
        amp_feats=flatten(transitions.amp_feat),
    )

    # Multiple PPO update epochs
    policy_params = state.policy_params
    value_params = state.value_params
    policy_opt_state = state.policy_opt_state
    value_opt_state = state.value_opt_state

    ppo_metrics = None

    for epoch in range(config.update_epochs):
        rng, shuffle_rng = jax.random.split(rng)

        # Shuffle data
        perm = jax.random.permutation(shuffle_rng, batch_size)
        shuffled_batch = RolloutBatch(
            obs=flat_batch.obs[perm],
            actions=flat_batch.actions[perm],
            rewards=flat_batch.rewards[perm],
            dones=flat_batch.dones[perm],
            log_probs=flat_batch.log_probs[perm],
            values=flat_batch.values[perm],
            advantages=flat_batch.advantages[perm],
            returns=flat_batch.returns[perm],
            amp_feats=(
                flat_batch.amp_feats[perm] if flat_batch.amp_feats is not None else None
            ),
        )

        # Process minibatches
        minibatch_size = batch_size // config.num_minibatches

        for mb_idx in range(config.num_minibatches):
            start_idx = mb_idx * minibatch_size
            end_idx = start_idx + minibatch_size

            minibatch = RolloutBatch(
                obs=shuffled_batch.obs[start_idx:end_idx],
                actions=shuffled_batch.actions[start_idx:end_idx],
                rewards=shuffled_batch.rewards[start_idx:end_idx],
                dones=shuffled_batch.dones[start_idx:end_idx],
                log_probs=shuffled_batch.log_probs[start_idx:end_idx],
                values=shuffled_batch.values[start_idx:end_idx],
                advantages=shuffled_batch.advantages[start_idx:end_idx],
                returns=shuffled_batch.returns[start_idx:end_idx],
                amp_feats=(
                    shuffled_batch.amp_feats[start_idx:end_idx]
                    if shuffled_batch.amp_feats is not None
                    else None
                ),
            )

            rng, ppo_step_rng = jax.random.split(rng)

            (
                policy_params,
                value_params,
                policy_opt_state,
                value_opt_state,
                ppo_metrics,
            ) = ppo_update_step(
                processor_params=state.processor_params,
                policy_params=policy_params,
                value_params=value_params,
                policy_opt_state=policy_opt_state,
                value_opt_state=value_opt_state,
                ppo_network=ppo_network,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                batch=minibatch,
                rng=ppo_step_rng,
                config=config,
            )

    # ========================================================================
    # Step 5: Update state and metrics
    # ========================================================================

    elapsed_time = time.time() - start_time
    env_steps = config.num_steps * config.num_envs

    new_state = AMPPPOState(
        processor_params=state.processor_params,
        policy_params=policy_params,
        value_params=value_params,
        policy_opt_state=policy_opt_state,
        value_opt_state=value_opt_state,
        disc_params=disc_params,
        disc_opt_state=disc_opt_state,
        amp_feature_stats=amp_feature_stats,
        iteration=state.iteration + 1,
        total_steps=state.total_steps + env_steps,
        rng=rng,
    )

    metrics = TrainingMetrics(
        policy_loss=float(ppo_metrics.policy_loss) if ppo_metrics else 0.0,
        value_loss=float(ppo_metrics.value_loss) if ppo_metrics else 0.0,
        entropy_loss=float(ppo_metrics.entropy_loss) if ppo_metrics else 0.0,
        total_loss=float(ppo_metrics.total_loss) if ppo_metrics else 0.0,
        clip_fraction=float(ppo_metrics.clip_fraction) if ppo_metrics else 0.0,
        approx_kl=float(ppo_metrics.approx_kl) if ppo_metrics else 0.0,
        disc_loss=disc_loss,
        disc_accuracy=disc_accuracy,
        amp_reward_mean=amp_reward_mean,
        amp_reward_std=amp_reward_std,
        episode_reward=float(jnp.mean(jnp.sum(transitions.reward, axis=0))),
        episode_length=float(config.num_steps),
        env_steps_per_sec=env_steps / elapsed_time if elapsed_time > 0 else 0.0,
    )

    return new_state, new_env_state, metrics


def train_amp_ppo(
    env_step_fn: Callable,
    env_reset_fn: Callable,
    config: AMPPPOConfig,
    callback: Optional[Callable[[int, AMPPPOState, TrainingMetrics], None]] = None,
) -> AMPPPOState:
    """Main training function for AMP+PPO.

    Args:
        env_step_fn: Function to step environment (state, action) -> (new_state, obs, reward, done, info)
        env_reset_fn: Function to reset environment (rng) -> env_state
        config: Training configuration
        callback: Optional callback called after each iteration

    Returns:
        Final training state
    """
    print("=" * 60)
    print("AMP+PPO Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  obs_dim: {config.obs_dim}")
    print(f"  action_dim: {config.action_dim}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  total_iterations: {config.total_iterations}")
    print(f"  enable_amp: {config.enable_amp}")
    print(f"  disc_batch_size: {config.disc_batch_size}")  # DEBUG: print actual batch size
    if config.enable_amp:
        print(f"  amp_reward_weight: {config.amp_reward_weight}")
        print(f"  disc_updates_per_iter: {config.disc_updates_per_iter}")
    print("=" * 60)

    # Initialize random key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, env_rng = jax.random.split(rng, 3)

    # Create training state and networks (with optimizers)
    state, ppo_network, disc_model, policy_optimizer, value_optimizer, disc_optimizer = create_amp_ppo_state(config, init_rng)

    # Create reference motion buffer
    # Use feature_dim=29 for AMP features (not obs_dim=38)
    amp_feature_dim = DEFAULT_AMP_CONFIG.feature_dim  # 29
    ref_buffer = create_reference_buffer(
        obs_dim=amp_feature_dim,  # Use 29-dim features, not 38-dim observations
        seq_len=config.ref_seq_len,
        max_size=config.ref_buffer_size,
        mode=config.ref_motion_mode,
        filepath=config.ref_motion_path,
    )

    print(f"✓ Reference buffer: {len(ref_buffer)} sequences")

    # Initialize environment
    env_state = env_reset_fn(env_rng)

    print(f"✓ Environment initialized")

    # Re-initialize discriminator optimizer state after env reset
    # This is a workaround for MJX overflow warnings affecting JAX state
    disc_optimizer = create_discriminator_optimizer(learning_rate=config.disc_learning_rate)
    disc_opt_state = disc_optimizer.init(state.disc_params)
    state = state._replace(disc_opt_state=disc_opt_state)

    print("=" * 60)
    print("Starting training...")
    print()

    # Training loop
    for iteration in range(config.total_iterations):
        state, env_state, metrics = train_iteration(
            state=state,
            env_step_fn=env_step_fn,
            env_state=env_state,
            ppo_network=ppo_network,
            disc_model=disc_model,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            disc_optimizer=disc_optimizer,
            ref_buffer=ref_buffer,
            config=config,
        )

        # Logging
        if iteration % config.log_interval == 0:
            print(
                f"Iter {iteration:5d} | "
                f"Reward: {metrics.episode_reward:8.2f} | "
                f"PPO Loss: {metrics.total_loss:8.4f} | "
                f"Disc Loss: {metrics.disc_loss:6.4f} | "
                f"Disc Acc: {metrics.disc_accuracy:5.2f} | "
                f"AMP Rew: {metrics.amp_reward_mean:7.4f} | "
                f"Steps/s: {metrics.env_steps_per_sec:6.0f}"
            )

        # Callback
        if callback is not None:
            callback(iteration, state, metrics)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Total steps: {state.total_steps}")
    print("=" * 60)

    return state


if __name__ == "__main__":
    # Test with dummy environment
    print("Testing AMP+PPO training loop...")

    # Test config with all required parameters
    config = AMPPPOConfig(
        # Environment
        obs_dim=44,
        action_dim=11,
        num_envs=4,
        num_steps=5,
        max_episode_steps=500,
        # PPO hyperparameters
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        # PPO training
        num_minibatches=4,
        update_epochs=4,
        # Network architecture
        policy_hidden_dims=(512, 256, 128),
        value_hidden_dims=(512, 256, 128),
        log_std_min=-20.0,
        log_std_max=2.0,
        # AMP configuration
        enable_amp=True,
        amp_reward_weight=1.0,
        disc_learning_rate=1e-4,
        disc_updates_per_iter=2,
        disc_batch_size=512,
        gradient_penalty_weight=5.0,
        # AMP discriminator architecture
        disc_hidden_dims=(1024, 512, 256),
        # Reference motion
        ref_buffer_size=1000,
        ref_seq_len=32,
        ref_motion_mode="walking",
        ref_motion_path=None,
        # Feature normalization
        normalize_amp_features=True,
        # Training
        total_iterations=3,
        seed=0,
        # Logging
        log_interval=1,
        save_interval=100,
        checkpoint_dir="checkpoints/amp_ppo",
    )

    # Dummy environment functions
    def dummy_step_fn(state, action):
        obs = state
        reward = jnp.zeros(config.num_envs)
        done = jnp.zeros(config.num_envs)
        new_obs = obs + jax.random.normal(jax.random.PRNGKey(0), obs.shape) * 0.01
        return new_obs, new_obs, reward, done, {}

    def dummy_reset_fn(rng):
        return jax.random.normal(rng, (config.num_envs, config.obs_dim))

    final_state = train_amp_ppo(
        env_step_fn=dummy_step_fn,
        env_reset_fn=dummy_reset_fn,
        config=config,
    )

    print("\n✅ AMP+PPO training loop test passed!")
