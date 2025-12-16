"""PPO building blocks for custom training loop.

This module provides PPO components that can be composed into a custom
training loop. Components are designed to be JAX-friendly and JIT-compatible.

Based on Brax's PPO implementation but with explicit interfaces that
allow injection of AMP reward shaping and discriminator training.

Key components:
- compute_gae: Generalized Advantage Estimation
- ppo_loss: Clipped surrogate objective
- Policy/Value networks with Gaussian policy
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


# ============================================================================
# Network Architectures
# ============================================================================


class PolicyNetwork(nn.Module):
    """Gaussian policy network for continuous control.

    Outputs mean and log_std for diagonal Gaussian distribution.
    Architecture follows 2024 best practices for locomotion.
    """

    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool = False):
        """Forward pass.

        Args:
            obs: Observations (batch, obs_dim)
            deterministic: If True, return mean action

        Returns:
            action: Sampled or mean action (batch, action_dim)
            log_prob: Log probability of action (batch,)
        """
        x = obs.astype(jnp.float32)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
            )(x)
            x = nn.elu(x)

        # Output mean
        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        )(x)

        # Output log_std (learned per-action-dim)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self,
        params: Any,
        obs: jnp.ndarray,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample action from policy.

        Args:
            params: Network parameters
            obs: Observations (batch, obs_dim)
            rng: Random key
            deterministic: If True, return mean action

        Returns:
            action: Sampled action (batch, action_dim)
            log_prob: Log probability of action (batch,)
        """
        mean, log_std = self.apply(params, obs)
        std = jnp.exp(log_std)

        if deterministic:
            action = mean
        else:
            noise = jax.random.normal(rng, mean.shape)
            action = mean + std * noise

        # Compute log probability
        log_prob = gaussian_log_prob(action, mean, log_std)

        return action, log_prob

    def log_prob(
        self,
        params: Any,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action.

        Args:
            params: Network parameters
            obs: Observations (batch, obs_dim)
            action: Actions (batch, action_dim)

        Returns:
            log_prob: Log probability (batch,)
        """
        mean, log_std = self.apply(params, obs)
        return gaussian_log_prob(action, mean, log_std)


class ValueNetwork(nn.Module):
    """Value function network for PPO.

    Outputs scalar value estimate for given observation.
    """

    hidden_dims: Tuple[int, ...] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: Observations (batch, obs_dim)

        Returns:
            value: Value estimate (batch,)
        """
        x = obs.astype(jnp.float32)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)),
            )(x)
            x = nn.elu(x)

        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
        )(x)

        return value.squeeze(-1)


def gaussian_log_prob(
    action: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log probability of Gaussian distribution.

    Args:
        action: Action (batch, action_dim)
        mean: Mean (batch, action_dim)
        log_std: Log standard deviation (action_dim,)

    Returns:
        log_prob: Log probability (batch,)
    """
    std = jnp.exp(log_std)
    var = std**2

    # Log probability of Gaussian
    log_prob = -0.5 * (
        jnp.sum(((action - mean) ** 2) / var, axis=-1)
        + jnp.sum(2 * log_std)
        + action.shape[-1] * jnp.log(2 * jnp.pi)
    )

    return log_prob


def create_policy_network(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Tuple[int, ...] = (512, 256, 128),
    seed: int = 0,
) -> Tuple[PolicyNetwork, Any]:
    """Create and initialize policy network.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer sizes
        seed: Random seed

    Returns:
        (network, params): Network module and initialized parameters
    """
    network = PolicyNetwork(action_dim=action_dim, hidden_dims=hidden_dims)
    rng = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = network.init(rng, dummy_obs)
    return network, params


def create_value_network(
    obs_dim: int,
    hidden_dims: Tuple[int, ...] = (512, 256, 128),
    seed: int = 0,
) -> Tuple[ValueNetwork, Any]:
    """Create and initialize value network.

    Args:
        obs_dim: Observation dimension
        hidden_dims: Hidden layer sizes
        seed: Random seed

    Returns:
        (network, params): Network module and initialized parameters
    """
    network = ValueNetwork(hidden_dims=hidden_dims)
    rng = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = network.init(rng, dummy_obs)
    return network, params


# ============================================================================
# Generalized Advantage Estimation (GAE)
# ============================================================================


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Rewards (num_steps, num_envs)
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


class PPOLossOutput(NamedTuple):
    """Output of PPO loss computation."""

    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    clip_fraction: jnp.ndarray
    approx_kl: jnp.ndarray


def ppo_loss(
    policy_params: Any,
    value_params: Any,
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, PPOLossOutput]:
    """Compute PPO loss with clipped objective.

    Args:
        policy_params: Policy network parameters
        value_params: Value network parameters
        policy_network: Policy network module
        value_network: Value network module
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
        (total_loss, PPOLossOutput): Scalar loss and metrics
    """
    # Normalize advantages
    if normalize_advantages:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Compute new log probs
    new_log_probs = policy_network.log_prob(policy_params, obs, actions)

    # Compute value estimates
    values = value_network.apply(value_params, obs)

    # Policy loss (clipped surrogate objective)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Value loss (squared error)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # Entropy bonus
    mean, log_std = policy_network.apply(policy_params, obs)
    std = jnp.exp(log_std)
    entropy = (
        jnp.mean(0.5 * (1.0 + jnp.log(2 * jnp.pi)) + log_std) * mean.shape[-1]
    )  # Sum over action dims
    entropy_loss = -entropy

    # Total loss
    total_loss = (
        policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    )

    # Metrics
    clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_epsilon)
    approx_kl = jnp.mean(old_log_probs - new_log_probs)

    return total_loss, PPOLossOutput(
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
    schedule: str = "constant",
    total_steps: int = 1000000,
) -> optax.GradientTransformation:
    """Create optimizer for policy/value networks.

    Args:
        learning_rate: Base learning rate
        max_grad_norm: Gradient clipping threshold
        schedule: Learning rate schedule ("constant" or "cosine")
        total_steps: Total training steps (for cosine schedule)

    Returns:
        Optax optimizer
    """
    if schedule == "cosine":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha=0.1,  # Final LR = 0.1 * initial
        )
    else:
        lr_schedule = learning_rate

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr_schedule),
    )


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


def create_training_state(
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    policy_params: Any,
    value_params: Any,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Create initial training state.

    Args:
        policy_network: Policy network module
        value_network: Value network module
        policy_params: Initial policy parameters
        value_params: Initial value parameters
        policy_optimizer: Policy optimizer
        value_optimizer: Value optimizer

    Returns:
        TrainingState
    """
    return TrainingState(
        policy_params=policy_params,
        value_params=value_params,
        policy_opt_state=policy_optimizer.init(policy_params),
        value_opt_state=value_optimizer.init(value_params),
        step=jnp.array(0, dtype=jnp.int32),
    )


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_observations(
    obs: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    clip: float = 10.0,
) -> jnp.ndarray:
    """Normalize observations.

    Args:
        obs: Raw observations
        mean: Running mean
        std: Running std
        clip: Clipping range

    Returns:
        Normalized observations
    """
    normalized = (obs - mean) / (std + 1e-8)
    return jnp.clip(normalized, -clip, clip)


if __name__ == "__main__":
    # Test PPO building blocks
    print("Testing PPO building blocks...")

    obs_dim = 44
    action_dim = 11
    batch_size = 64
    num_steps = 10
    num_envs = 16

    # Create networks
    policy_net, policy_params = create_policy_network(obs_dim, action_dim)
    value_net, value_params = create_value_network(obs_dim)
    print(f"✓ Networks created")

    # Test policy forward pass
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (batch_size, obs_dim))
    action, log_prob = policy_net.sample(policy_params, obs, rng)
    print(
        f"✓ Policy sample: action shape {action.shape}, log_prob shape {log_prob.shape}"
    )

    # Test value forward pass
    value = value_net.apply(value_params, obs)
    print(f"✓ Value: shape {value.shape}")

    # Test GAE
    rewards = jax.random.normal(rng, (num_steps, num_envs))
    values = jax.random.normal(rng, (num_steps, num_envs))
    dones = jnp.zeros((num_steps, num_envs))
    bootstrap = jax.random.normal(rng, (num_envs,))

    advantages, returns = compute_gae(rewards, values, dones, bootstrap)
    print(f"✓ GAE: advantages shape {advantages.shape}, returns shape {returns.shape}")

    # Test PPO loss
    obs_batch = jax.random.normal(rng, (batch_size, obs_dim))
    action_batch = jax.random.normal(rng, (batch_size, action_dim))
    log_prob_batch = jax.random.normal(rng, (batch_size,))
    adv_batch = jax.random.normal(rng, (batch_size,))
    ret_batch = jax.random.normal(rng, (batch_size,))

    loss, metrics = ppo_loss(
        policy_params,
        value_params,
        policy_net,
        value_net,
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

    print("\n✅ All PPO building blocks tests passed!")
