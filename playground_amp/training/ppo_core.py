"""PPO building blocks using Brax's implementations.

This module provides a thin wrapper around Brax's PPO components with
explicit interfaces that allow injection of AMP reward shaping and
discriminator training.

Key components (from Brax):
- make_ppo_networks: Create policy and value networks
- make_inference_fn: Create inference function for trained policy
- compute_gae: Generalized Advantage Estimation (custom impl for flexibility)
- ppo_loss: Clipped surrogate objective (custom impl for AMP integration)

The custom GAE and PPO loss implementations allow us to:
1. Shape rewards with AMP discriminator output between rollout and update
2. Access intermediate values for logging
3. Customize loss computation if needed
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.training.agents.ppo import networks as ppo_networks


# ============================================================================
# Re-export Brax's network factory
# ============================================================================

# Use Brax's network factory directly
make_ppo_networks = ppo_networks.make_ppo_networks
make_inference_fn = ppo_networks.make_inference_fn


def create_networks(
    obs_dim: int,
    action_dim: int,
    policy_hidden_dims: Tuple[int, ...],
    value_hidden_dims: Tuple[int, ...],
    preprocess_observations_fn: Optional[Callable] = None,
):
    """Create PPO networks using Brax's factory.

    This is a convenience wrapper around Brax's make_ppo_networks that
    uses our config naming conventions.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        policy_hidden_dims: Hidden layer sizes for policy network
        value_hidden_dims: Hidden layer sizes for value network
        preprocess_observations_fn: Optional observation preprocessing

    Returns:
        PPONetworks: Brax's PPO network container with policy and value networks
    """
    return ppo_networks.make_ppo_networks(
        observation_size=obs_dim,
        action_size=action_dim,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_dims,
        value_hidden_layer_sizes=value_hidden_dims,
    )


def init_network_params(
    ppo_network,
    obs_dim: int,
    action_dim: int,
    seed: int = 0,
):
    """Initialize network parameters.

    Args:
        ppo_network: PPO networks from create_networks()
        obs_dim: Observation dimension
        action_dim: Action dimension
        seed: Random seed

    Returns:
        Initialized parameters for policy and value networks
    """
    rng = jax.random.PRNGKey(seed)
    rng, policy_rng, value_rng = jax.random.split(rng, 3)

    # Initialize policy
    policy_params = ppo_network.policy_network.init(policy_rng)

    # Initialize value
    value_params = ppo_network.value_network.init(value_rng)

    return policy_params, value_params


# ============================================================================
# Generalized Advantage Estimation (GAE)
# ============================================================================
# Custom implementation to allow AMP reward shaping between rollout and GAE


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    This is a custom implementation (not using Brax's) to allow:
    1. AMP reward shaping between rollout collection and GAE computation
    2. Explicit access to intermediate values for logging

    Args:
        rewards: Rewards (num_steps, num_envs) - can include AMP-shaped rewards
        values: Value estimates (num_steps, num_envs)
        dones: Episode termination flags (num_steps, num_envs)
        bootstrap_value: Value estimate for final state (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages (num_steps, num_envs)
        returns: Discounted returns (num_steps, num_envs)
    """
    num_steps = rewards.shape[0]

    # Append bootstrap value
    values_with_bootstrap = jnp.concatenate(
        [values, bootstrap_value[None, ...]], axis=0
    )

    def gae_step(carry, t):
        gae = carry

        # TD error
        delta = (
            rewards[t]
            + gamma * values_with_bootstrap[t + 1] * (1 - dones[t])
            - values_with_bootstrap[t]
        )

        # GAE
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        return gae, gae

    # Compute GAE backwards
    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros_like(bootstrap_value),
        jnp.arange(num_steps - 1, -1, -1),
    )

    # Reverse to get correct order
    advantages = advantages[::-1]

    # Compute returns
    returns = advantages + values

    return advantages, returns


# ============================================================================
# PPO Loss Function
# ============================================================================
# Custom implementation to expose all metrics for AMP integration


class PPOLossMetrics(NamedTuple):
    """Metrics from PPO loss computation."""

    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    clip_fraction: jnp.ndarray
    approx_kl: jnp.ndarray


def compute_ppo_loss(
    policy_params: Any,
    value_params: Any,
    ppo_network,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, PPOLossMetrics]:
    """Compute PPO loss with clipped objective.

    This uses Brax's network interface but computes the loss explicitly
    to expose all metrics for logging and AMP integration.

    Args:
        policy_params: Policy network parameters
        value_params: Value network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch_size, obs_dim)
        actions: Actions (batch_size, action_dim)
        old_log_probs: Log probs from rollout (batch_size,)
        advantages: GAE advantages (batch_size,)
        returns: Discounted returns (batch_size,)
        clip_epsilon: PPO clipping parameter
        value_loss_coef: Value loss weight
        entropy_coef: Entropy bonus weight
        normalize_advantages: Whether to normalize advantages

    Returns:
        (total_loss, PPOLossMetrics): Scalar loss and metrics
    """
    # Normalize advantages
    if normalize_advantages:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Get policy distribution from Brax network
    policy_network = ppo_network.policy_network
    value_network = ppo_network.value_network

    # Compute new log probs using Brax's policy network
    # Brax's policy network returns a distribution
    dist = policy_network.apply(policy_params, obs)
    new_log_probs = dist.log_prob(actions)

    # Compute value estimates
    values = value_network.apply(value_params, obs)

    # Policy loss (clipped surrogate objective)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Value loss (squared error)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # Entropy bonus (from distribution)
    entropy = jnp.mean(dist.entropy())
    entropy_loss = -entropy

    # Total loss
    total_loss = (
        policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    )

    # Metrics
    clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_epsilon)
    approx_kl = jnp.mean(old_log_probs - new_log_probs)

    return total_loss, PPOLossMetrics(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
    )


# ============================================================================
# Optimizer Creation
# ============================================================================


def create_optimizer(
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    """Create optimizer for policy/value networks.

    Args:
        learning_rate: Base learning rate
        max_grad_norm: Gradient clipping threshold

    Returns:
        Optax optimizer
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


# ============================================================================
# Sampling utilities for custom training loop
# ============================================================================


def sample_actions(
    policy_params: Any,
    ppo_network,
    obs: jnp.ndarray,
    rng: jax.Array,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample actions from policy.

    Args:
        policy_params: Policy network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch, obs_dim)
        rng: Random key
        deterministic: If True, return mode of distribution

    Returns:
        actions: Sampled actions (batch, action_dim)
        log_probs: Log probabilities of actions (batch,)
    """
    dist = ppo_network.policy_network.apply(policy_params, obs)

    if deterministic:
        actions = dist.mode()
    else:
        actions = dist.sample(seed=rng)

    log_probs = dist.log_prob(actions)

    return actions, log_probs


def compute_values(
    value_params: Any,
    ppo_network,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute value estimates.

    Args:
        value_params: Value network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch, obs_dim)

    Returns:
        values: Value estimates (batch,)
    """
    return ppo_network.value_network.apply(value_params, obs)


# ============================================================================
# Training State Container
# ============================================================================


class TrainingState(NamedTuple):
    """Container for all training state.

    Designed to be a valid JAX pytree for JIT compatibility.
    """

    policy_params: Any
    value_params: Any
    policy_opt_state: Any
    value_opt_state: Any
    step: jnp.ndarray


# ============================================================================
# Test
# ============================================================================


if __name__ == "__main__":
    print("Testing PPO building blocks (using Brax networks)...")

    obs_dim = 44
    action_dim = 11
    batch_size = 64
    num_steps = 10
    num_envs = 16

    # Network config (would come from training YAML in production)
    policy_hidden_dims = (512, 256, 128)
    value_hidden_dims = (512, 256, 128)

    # Create networks using Brax
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=policy_hidden_dims,
        value_hidden_dims=value_hidden_dims,
    )
    print(f"✓ PPO networks created (Brax)")

    # Initialize parameters
    policy_params, value_params = init_network_params(
        ppo_network, obs_dim, action_dim, seed=0
    )
    print(f"✓ Network parameters initialized")

    # Test policy sampling
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (batch_size, obs_dim))
    actions, log_probs = sample_actions(policy_params, ppo_network, obs, rng)
    print(
        f"✓ Policy sample: action shape {actions.shape}, log_prob shape {log_probs.shape}"
    )

    # Test value computation
    values = compute_values(value_params, ppo_network, obs)
    print(f"✓ Value: shape {values.shape}")

    # Test GAE
    rewards = jax.random.normal(rng, (num_steps, num_envs))
    step_values = jax.random.normal(rng, (num_steps, num_envs))
    dones = jnp.zeros((num_steps, num_envs))
    bootstrap = jax.random.normal(rng, (num_envs,))

    advantages, returns = compute_gae(rewards, step_values, dones, bootstrap)
    print(f"✓ GAE: advantages shape {advantages.shape}, returns shape {returns.shape}")

    # Test PPO loss
    obs_batch = jax.random.normal(rng, (batch_size, obs_dim))
    action_batch = jax.random.normal(rng, (batch_size, action_dim))
    log_prob_batch = jax.random.normal(rng, (batch_size,))
    adv_batch = jax.random.normal(rng, (batch_size,))
    ret_batch = jax.random.normal(rng, (batch_size,))

    loss, metrics = compute_ppo_loss(
        policy_params,
        value_params,
        ppo_network,
        obs_batch,
        action_batch,
        log_prob_batch,
        adv_batch,
        ret_batch,
    )
    print(f"✓ PPO loss: {loss:.4f}")
    print(f"  Policy loss: {metrics.policy_loss:.4f}")
    print(f"  Value loss: {metrics.value_loss:.4f}")
    print(f"  Entropy loss: {metrics.entropy_loss:.4f}")

    # Test optimizer
    optimizer = create_optimizer()
    opt_state = optimizer.init(policy_params)
    print(f"✓ Optimizer created")

    # Test inference function
    inference_fn = make_inference_fn(ppo_network)
    policy_fn = inference_fn(policy_params, deterministic=True)
    test_obs = jax.random.normal(rng, (1, obs_dim))
    test_action = policy_fn(test_obs, rng)
    print(f"✓ Inference function: action shape {test_action.shape}")

    print("\n✅ All PPO building blocks tests passed (using Brax)!")
