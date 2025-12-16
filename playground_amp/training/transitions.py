"""Transition data structures for AMP+PPO training.

This module defines the data structures used to store rollout data
during training, including AMP-specific features for discriminator training.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp


class AMPTransition(NamedTuple):
    """Transition data with AMP features for discriminator training.

    This extends the standard RL transition with:
    - next_obs: For computing transition-based AMP features
    - amp_feat: Pre-computed motion features for discriminator
    - log_prob: Log probability of action (for PPO)
    - value: Value function estimate (for GAE)

    All arrays should have shape (num_envs,) or (num_envs, dim) for
    batched training.
    """

    obs: jnp.ndarray  # Current observation (num_envs, obs_dim)
    next_obs: jnp.ndarray  # Next observation (num_envs, obs_dim)
    action: jnp.ndarray  # Action taken (num_envs, act_dim)
    reward: jnp.ndarray  # Environment reward (num_envs,)
    done: jnp.ndarray  # Episode termination (num_envs,)
    log_prob: jnp.ndarray  # Log probability of action (num_envs,)
    value: jnp.ndarray  # Value function estimate (num_envs,)
    amp_feat: Optional[jnp.ndarray] = (
        None  # Motion features for AMP (num_envs, feat_dim)
    )


class RolloutBatch(NamedTuple):
    """Batch of rollout data for PPO training.

    Contains data from multiple timesteps across multiple environments,
    reshaped for minibatch training.

    Shapes: (batch_size, dim) where batch_size = num_envs * num_steps
    """

    obs: jnp.ndarray  # (batch_size, obs_dim)
    actions: jnp.ndarray  # (batch_size, act_dim)
    rewards: jnp.ndarray  # (batch_size,)
    dones: jnp.ndarray  # (batch_size,)
    log_probs: jnp.ndarray  # (batch_size,)
    values: jnp.ndarray  # (batch_size,)
    advantages: jnp.ndarray  # (batch_size,)
    returns: jnp.ndarray  # (batch_size,)
    amp_feats: Optional[jnp.ndarray] = None  # (batch_size, feat_dim)


def flatten_transitions(
    transitions: AMPTransition, bootstrap_value: jnp.ndarray
) -> RolloutBatch:
    """Flatten transitions from (num_steps, num_envs, ...) to (batch_size, ...).

    Also computes advantages and returns using GAE.

    Args:
        transitions: AMPTransition with stacked arrays from rollout
        bootstrap_value: Value estimate for final state (num_envs,)

    Returns:
        RolloutBatch ready for minibatch training
    """
    # Import GAE computation
    from playground_amp.training.ppo_building_blocks import compute_gae

    # transitions arrays have shape (num_steps, num_envs, ...)
    num_steps = transitions.obs.shape[0]
    num_envs = transitions.obs.shape[1]
    batch_size = num_steps * num_envs

    # Compute advantages and returns
    advantages, returns = compute_gae(
        rewards=transitions.reward,
        values=transitions.value,
        dones=transitions.done,
        bootstrap_value=bootstrap_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Flatten all arrays: (num_steps, num_envs, ...) -> (batch_size, ...)
    def flatten(x):
        if x is None:
            return None
        return x.reshape((batch_size,) + x.shape[2:])

    return RolloutBatch(
        obs=flatten(transitions.obs),
        actions=flatten(transitions.action),
        rewards=flatten(transitions.reward),
        dones=flatten(transitions.done),
        log_probs=flatten(transitions.log_prob),
        values=flatten(transitions.value),
        advantages=flatten(advantages),
        returns=flatten(returns),
        amp_feats=flatten(transitions.amp_feat),
    )


def stack_transitions(transitions_list: list[AMPTransition]) -> AMPTransition:
    """Stack a list of transitions into a single AMPTransition.

    Args:
        transitions_list: List of AMPTransition, one per timestep

    Returns:
        AMPTransition with stacked arrays (num_steps, num_envs, ...)
    """
    return AMPTransition(
        obs=jnp.stack([t.obs for t in transitions_list]),
        next_obs=jnp.stack([t.next_obs for t in transitions_list]),
        action=jnp.stack([t.action for t in transitions_list]),
        reward=jnp.stack([t.reward for t in transitions_list]),
        done=jnp.stack([t.done for t in transitions_list]),
        log_prob=jnp.stack([t.log_prob for t in transitions_list]),
        value=jnp.stack([t.value for t in transitions_list]),
        amp_feat=(
            jnp.stack([t.amp_feat for t in transitions_list])
            if transitions_list[0].amp_feat is not None
            else None
        ),
    )


# Register AMPTransition as a JAX pytree for JIT compatibility
def _amp_transition_flatten(t: AMPTransition):
    """Flatten AMPTransition for JAX pytree."""
    children = (
        t.obs,
        t.next_obs,
        t.action,
        t.reward,
        t.done,
        t.log_prob,
        t.value,
        t.amp_feat,
    )
    aux = None
    return children, aux


def _amp_transition_unflatten(aux, children):
    """Unflatten AMPTransition from JAX pytree."""
    return AMPTransition(*children)


# Register with JAX
jax.tree_util.register_pytree_node(
    AMPTransition,
    _amp_transition_flatten,
    _amp_transition_unflatten,
)
