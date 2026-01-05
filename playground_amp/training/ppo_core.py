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

IMPORTANT: Brax Network API Notes
---------------------------------
Brax's FeedForwardNetwork has a specific apply signature:
  network.apply(processor_params, network_params, obs)

Where:
  - processor_params: Parameters for observation normalization (running statistics)
                      Can be empty tuple () if no normalization is used
  - network_params: The actual neural network weights
  - obs: Observation tensor

The policy network returns LOGITS (raw output), not a distribution.
Use ppo_network.parametric_action_distribution to convert logits to actions/log_probs.
"""

from __future__ import annotations

import importlib.util
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

_HAS_FLAX = importlib.util.find_spec("flax") is not None
if _HAS_FLAX:
    from flax.core import freeze as _flax_freeze  # type: ignore[import-not-found]
    from flax.core import unfreeze as _flax_unfreeze  # type: ignore[import-not-found]
else:
    _flax_freeze = None
    _flax_unfreeze = None


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
        preprocess_observations_fn: Optional observation preprocessing.
            If None, uses Brax's identity preprocessor.

    Returns:
        PPONetworks: Brax's PPO network container with policy and value networks
    """
    # Build kwargs - only include preprocess_observations_fn if explicitly provided
    # If None, Brax will use its default identity_observation_preprocessor
    kwargs = {
        "observation_size": obs_dim,
        "action_size": action_dim,
        "policy_hidden_layer_sizes": policy_hidden_dims,
        "value_hidden_layer_sizes": value_hidden_dims,
    }

    if preprocess_observations_fn is not None:
        kwargs["preprocess_observations_fn"] = preprocess_observations_fn

    return ppo_networks.make_ppo_networks(**kwargs)


def init_network_params(
    ppo_network,
    obs_dim: int,
    action_dim: int,
    seed: int = 0,
    *,
    policy_init_action: Optional[jnp.ndarray] = None,
    policy_init_std: float = 0.10,
):
    """Initialize network parameters.

    Brax's networks require TWO sets of parameters:
    1. processor_params: For observation normalization (empty tuple if not used)
    2. network_params: The actual network weights

    Args:
        ppo_network: PPO networks from create_networks()
        obs_dim: Observation dimension
        action_dim: Action dimension
        seed: Random seed

    Returns:
        Tuple of (processor_params, policy_params, value_params)
        - processor_params: Empty tuple (no normalization) or running statistics state
        - policy_params: Policy network weights
        - value_params: Value network weights
    """
    rng = jax.random.PRNGKey(seed)
    rng, policy_rng, value_rng = jax.random.split(rng, 3)

    # Initialize policy
    policy_params = ppo_network.policy_network.init(policy_rng)
    if policy_init_action is not None:
        policy_params = _init_policy_output_bias(
            ppo_network=ppo_network,
            policy_params=policy_params,
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_init_action=policy_init_action,
            policy_init_std=policy_init_std,
        )

    # Initialize value
    value_params = ppo_network.value_network.init(value_rng)

    # No observation normalization by default (empty tuple as processor_params)
    processor_params = ()

    return processor_params, policy_params, value_params


def _inv_softplus(x: jnp.ndarray) -> jnp.ndarray:
    # Numerically stable inverse of softplus for x>0.
    return jnp.log(jnp.expm1(x))


def _init_policy_output_bias(
    *,
    ppo_network: Any,
    policy_params: Any,
    obs_dim: int,
    action_dim: int,
    policy_init_action: jnp.ndarray,
    policy_init_std: float,
) -> Any:
    """Bias-initialize policy outputs so tanh(loc) ~= policy_init_action.

    This prevents a freshly initialized policy (which typically outputs near 0)
    from immediately driving the robot away from a stable reset pose.

    Brax uses NormalTanhDistribution with parameters split as:
      parameters = concat([loc, scale_param])
      scale = softplus(scale_param) + min_std
      action = tanh(sample Normal(loc, scale))
    """
    init_action = jnp.asarray(policy_init_action, dtype=jnp.float32).reshape((action_dim,))
    eps = jnp.float32(1e-5)
    init_action = jnp.clip(init_action, -1.0 + eps, 1.0 - eps)
    loc_bias = jnp.arctanh(init_action)

    dist = getattr(ppo_network, "parametric_action_distribution", None)
    min_std = jnp.float32(getattr(dist, "_min_std", 0.001)) if dist is not None else jnp.float32(0.001)
    var_scale = jnp.float32(getattr(dist, "_var_scale", 1.0)) if dist is not None else jnp.float32(1.0)

    target_scale = jnp.float32(policy_init_std)
    # Solve: target_scale = (softplus(param) + min_std) * var_scale
    softplus_target = jnp.maximum(target_scale / var_scale - min_std, jnp.float32(1e-6))
    scale_param_bias = _inv_softplus(softplus_target) * jnp.ones((action_dim,), dtype=jnp.float32)

    desired_logits = jnp.concatenate([loc_bias, scale_param_bias], axis=-1)

    # Compute the current output at a reference observation and shift the final
    # layer bias so the reference output matches desired_logits exactly.
    ref_obs = jnp.zeros((int(obs_dim),), dtype=jnp.float32)
    base_logits = ppo_network.policy_network.apply((), policy_params, ref_obs)
    base_logits = jnp.asarray(base_logits, dtype=jnp.float32).reshape((2 * action_dim,))
    delta = desired_logits - base_logits

    params = policy_params.get("params") if isinstance(policy_params, dict) else None
    if params is None:
        raise ValueError("Unexpected policy_params structure: expected dict with key 'params'")

    freezer = None
    params_mut = params
    if _flax_unfreeze is not None and _flax_freeze is not None:
        # Flax commonly wraps params in FrozenDict; unwrap to a plain dict for mutation.
        # Use feature-detection (not exceptions) to keep control flow explicit.
        if hasattr(params, "unfreeze") and callable(getattr(params, "unfreeze")):
            params_mut = _flax_unfreeze(params)  # type: ignore[misc]
            freezer = _flax_freeze

    # Prefer a structured update: Brax names MLP layers hidden_0..hidden_k, where
    # hidden_k is the output layer.
    if isinstance(params_mut, dict):
        hidden_keys = [k for k in params_mut.keys() if isinstance(k, str) and k.startswith("hidden_")]
    else:
        hidden_keys = []

    updated = False
    if hidden_keys:
        try:
            last_idx = max(int(k.split("_", 1)[1]) for k in hidden_keys)
            out_key = f"hidden_{last_idx}"
            out_layer = params_mut.get(out_key)
            if isinstance(out_layer, dict) and "bias" in out_layer:
                bias = jnp.asarray(out_layer["bias"], dtype=jnp.float32)
                if tuple(bias.shape) != (2 * action_dim,):
                    raise ValueError(
                        f"Unexpected output bias shape at params['{out_key}']['bias']: "
                        f"got {tuple(bias.shape)}, expected ({2 * action_dim},)"
                    )
                out_layer["bias"] = bias + delta
                params_mut[out_key] = out_layer
                updated = True
        except Exception:
            updated = False

    if not updated:
        raise ValueError(
            "Failed to bias-initialize policy output layer. "
            "Unexpected params structure (expected params['hidden_k']['bias'])."
        )

    return {**policy_params, "params": freezer(params_mut) if freezer is not None else params_mut}


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

    # Ensure consistent float32 dtype
    rewards = rewards.astype(jnp.float32)
    values = values.astype(jnp.float32)
    dones = dones.astype(jnp.float32)
    bootstrap_value = bootstrap_value.astype(jnp.float32)
    gamma = jnp.float32(gamma)
    gae_lambda = jnp.float32(gae_lambda)

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

    # Initialize with float32
    init_gae = jnp.zeros_like(bootstrap_value, dtype=jnp.float32)

    # Compute GAE backwards
    _, advantages = jax.lax.scan(
        gae_step,
        init_gae,
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
    processor_params: Any,
    policy_params: Any,
    value_params: Any,
    ppo_network,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    rng: jax.Array,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, PPOLossMetrics]:
    """Compute PPO loss with clipped objective.

    This uses Brax's network interface but computes the loss explicitly
    to expose all metrics for logging and AMP integration.

    IMPORTANT: Brax's network API:
    - policy_network.apply(processor_params, policy_params, obs) -> logits
    - parametric_action_distribution.log_prob(logits, raw_actions) -> log_probs
    - parametric_action_distribution.entropy(logits, rng) -> entropy

    Args:
        processor_params: Observation normalization parameters (empty tuple if not used)
        policy_params: Policy network parameters
        value_params: Value network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch_size, obs_dim)
        actions: Raw/pre-tanh actions (batch_size, action_dim) - NOT postprocessed
        old_log_probs: Log probs from rollout (batch_size,)
        advantages: GAE advantages (batch_size,)
        returns: Discounted returns (batch_size,)
        rng: Random key for entropy computation
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

    # Get logits from policy network (NOT a distribution!)
    # Brax API: apply(processor_params, network_params, obs) -> logits
    logits = ppo_network.policy_network.apply(processor_params, policy_params, obs)

    # Use parametric_action_distribution to compute log_probs
    # NOTE: We use raw actions (before postprocessing) for log_prob
    new_log_probs = ppo_network.parametric_action_distribution.log_prob(logits, actions)

    # Compute value estimates
    # Brax API: apply(processor_params, network_params, obs) -> values
    values = ppo_network.value_network.apply(processor_params, value_params, obs)

    # Policy loss (clipped surrogate objective)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Value loss (squared error)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # Entropy bonus
    entropy = jnp.mean(ppo_network.parametric_action_distribution.entropy(logits, rng))
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
    # Ensure learning_rate and max_grad_norm are Python scalars, not JAX/numpy arrays
    # This is critical because optax expects scalar learning rates
    learning_rate = float(learning_rate)
    max_grad_norm = float(max_grad_norm)
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


# ============================================================================
# Sampling utilities for custom training loop
# ============================================================================


def sample_actions(
    processor_params: Any,
    policy_params: Any,
    ppo_network,
    obs: jnp.ndarray,
    rng: jax.Array,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample actions from policy using Brax's network API.

    IMPORTANT: Brax's network flow:
    1. policy_network.apply(processor_params, policy_params, obs) -> logits
    2. parametric_action_distribution.sample_no_postprocessing(logits, rng) -> raw_actions
    3. parametric_action_distribution.postprocess(raw_actions) -> postprocessed_actions (for environment)
    4. parametric_action_distribution.log_prob(logits, raw_actions) -> log_probs

    For PPO updates, we need RAW actions (before postprocessing) for log_prob computation.
    For environment stepping, we use POSTPROCESSED actions.

    Args:
        processor_params: Observation normalization parameters (empty tuple if not used)
        policy_params: Policy network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch, obs_dim)
        rng: Random key
        deterministic: If True, return mode of distribution

    Returns:
        postprocessed_actions: Actions to send to environment (batch, action_dim)
        raw_actions: Pre-postprocessing actions for log_prob computation (batch, action_dim)
        log_probs: Log probabilities of raw actions (batch,)
    """
    # Get logits from policy network
    # Brax API: apply(processor_params, network_params, obs) -> logits
    logits = ppo_network.policy_network.apply(processor_params, policy_params, obs)

    dist = ppo_network.parametric_action_distribution

    if deterministic:
        # Mode returns postprocessed actions
        postprocessed_actions = dist.mode(logits)
        # For deterministic, raw = inverse_postprocess(postprocessed)
        raw_actions = dist.inverse_postprocess(postprocessed_actions)
    else:
        # Sample raw actions (before postprocessing)
        raw_actions = dist.sample_no_postprocessing(logits, rng)
        # Postprocess for environment
        postprocessed_actions = dist.postprocess(raw_actions)

    # Log prob is computed on raw actions
    log_probs = dist.log_prob(logits, raw_actions)

    return postprocessed_actions, raw_actions, log_probs


def compute_values(
    processor_params: Any,
    value_params: Any,
    ppo_network,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute value estimates using Brax's network API.

    Args:
        processor_params: Observation normalization parameters (empty tuple if not used)
        value_params: Value network parameters
        ppo_network: PPO networks from create_networks()
        obs: Observations (batch, obs_dim)

    Returns:
        values: Value estimates (batch,)
    """
    # Brax API: apply(processor_params, network_params, obs) -> values
    return ppo_network.value_network.apply(processor_params, value_params, obs)


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
