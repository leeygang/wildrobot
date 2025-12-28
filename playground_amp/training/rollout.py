"""Unified rollout collector for PPO and AMP training.

This module provides a single rollout collector that works for both:
- Stage 1: PPO-only (collect_amp_features=False)
- Stage 3: AMP+PPO (collect_amp_features=True)

The key design principle is that the rollout semantics (obs, actions, logp,
values, task_reward) are IDENTICAL regardless of whether AMP features are
collected. This ensures Stage 1 is a true subset of Stage 3.

v0.10.4: Registry-based metrics
- Replaced individual metric fields with single `metrics_vec` tensor
- Uses metrics_registry for schema enforcement
- No more .get() calls with silent fallbacks
- Adding new metrics only requires registry + env changes

Usage:
    # Stage 1: PPO-only
    traj = collect_rollout(env_step_fn, env_state, policy_params, ...,
                           collect_amp_features=False)

    # Stage 3: AMP+PPO
    traj = collect_rollout(env_step_fn, env_state, policy_params, ...,
                           collect_amp_features=True)

Invariant Tests:
    When collect_amp_features changes, these fields MUST be identical:
    - obs, actions, log_probs, values, task_rewards, dones, truncations
    Only these may differ:
    - amp_features (present or None)
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from playground_amp.training.ppo_core import (
    compute_values,
    sample_actions,
)
from playground_amp.envs.env_types import WR_INFO_KEY
from playground_amp.training.metrics_registry import METRICS_VEC_KEY, NUM_METRICS


class TrajectoryBatch(NamedTuple):
    """Container for collected trajectory data.

    This is the single source of truth for rollout data, used by both
    PPO-only and AMP+PPO training paths.

    Shapes:
        All tensors have shape (num_steps, num_envs, ...)

    Fields:
        obs: Observations at each step
        actions: Raw actions (pre-postprocessing, for log_prob computation)
        log_probs: Log probabilities of actions under policy
        values: Value estimates at each step
        task_rewards: Task rewards from environment (NOT shaped with AMP)
        dones: Episode termination flags (1.0 if done)
        truncations: Truncation flags (1.0 if episode ended due to time limit)
        next_obs: Next observations (for bootstrapping)
        bootstrap_value: Value estimate for final state (num_envs,)

        # v0.10.4: Packed metrics vector (replaces individual metric fields)
        # Shape: (num_steps, num_envs, NUM_METRICS)
        # Use metrics_registry.unpack_metrics() to access by name
        metrics_vec: All logging metrics in registry order

        # Algorithm-critical fields (kept separate for explicit access)
        step_counts: Episode step count at each step (for ep_len calculation)

        # Optional AMP fields (None if collect_amp_features=False)
        foot_contacts: Foot contact states for AMP features
        root_heights: Root heights for AMP features
        prev_joint_positions: Previous joint positions for finite diff velocity
    """
    # Core PPO fields (always populated)
    obs: jnp.ndarray
    actions: jnp.ndarray  # Raw actions for log_prob
    log_probs: jnp.ndarray
    values: jnp.ndarray
    task_rewards: jnp.ndarray
    dones: jnp.ndarray
    truncations: jnp.ndarray
    next_obs: jnp.ndarray
    bootstrap_value: jnp.ndarray

    # v0.10.4: Packed metrics vector (replaces individual metric fields)
    # Shape: (num_steps, num_envs, NUM_METRICS)
    metrics_vec: jnp.ndarray

    # Algorithm-critical fields (kept separate for explicit access)
    step_counts: jnp.ndarray

    # AMP feature fields (None if not collecting)
    foot_contacts: Optional[jnp.ndarray]
    root_heights: Optional[jnp.ndarray]
    prev_joint_positions: Optional[jnp.ndarray]


def collect_rollout(
    env_step_fn: Callable,
    env_state: Any,
    policy_params: Any,
    value_params: Any,
    processor_params: Any,
    ppo_network: Any,
    rng: jax.Array,
    num_steps: int,
    collect_amp_features: bool = False,
) -> Tuple[Any, TrajectoryBatch]:
    """Collect rollout using jax.lax.scan (fully on GPU).

    This is the unified rollout collector for both PPO-only and AMP+PPO.
    The key invariant is that when collect_amp_features changes:
    - Core PPO fields (obs, actions, log_probs, values, task_rewards) are IDENTICAL
    - Only AMP-specific fields differ

    v0.10.4: Uses state.metrics[METRICS_VEC_KEY] instead of .get() calls.
    This ensures:
    - Fixed pytree structure for lax.scan
    - No silent fallbacks to zeros
    - Schema enforcement via metrics_registry

    Args:
        env_step_fn: Batched environment step function
        env_state: Current environment state
        policy_params: Policy network parameters
        value_params: Value network parameters
        processor_params: Observation processor parameters
        ppo_network: PPO network module
        rng: Random key
        num_steps: Number of steps to collect
        collect_amp_features: Whether to collect AMP feature fields

    Returns:
        (final_env_state, trajectory_batch)
    """

    def step_fn(carry, _):
        env_state, rng, prev_joint_pos = carry
        rng, action_rng = jax.random.split(rng)

        obs = env_state.obs

        # Sample action from policy (deterministic RNG consumption)
        action, raw_action, log_prob = sample_actions(
            processor_params, policy_params, ppo_network, obs, action_rng
        )

        # Get value estimate
        value = compute_values(processor_params, value_params, ppo_network, obs)

        # Step environment
        next_env_state = env_step_fn(env_state, action)

        # Extract current joint positions for next iteration
        # Joint positions are at indices 9-18 in observation (38-dim layout)
        current_joint_pos = obs[..., 9:18]

        # Access typed WildRobotInfo namespace directly (fail loudly if missing)
        wr_info = next_env_state.info[WR_INFO_KEY]

        # Build step data - minimal fields, metrics come from metrics[METRICS_VEC_KEY]
        step_data = {
            # Core PPO fields
            "obs": obs,
            "action": raw_action,
            "log_prob": log_prob,
            "value": value,
            "task_reward": next_env_state.reward,
            "done": next_env_state.done,
            "truncation": wr_info.truncated,  # Required field from typed namespace
            "next_obs": next_env_state.obs,
            # v0.10.4: Packed metrics vector from metrics dict
            "metrics_vec": next_env_state.metrics[METRICS_VEC_KEY],
            # Algorithm-critical fields (kept explicit)
            "step_count": wr_info.step_count,
            # AMP fields - required fields from typed namespace
            "foot_contact": wr_info.foot_contacts,  # Required field (4,) per env
            "root_height": wr_info.root_height,  # Required field scalar per env
            "prev_joint_pos": prev_joint_pos,
        }

        return (next_env_state, rng, current_joint_pos), step_data

    # Initialize prev_joint_pos from initial observation
    initial_joint_pos = env_state.obs[..., 9:18]

    # Run scan
    init_carry = (env_state, rng, initial_joint_pos)
    (final_env_state, _, _), step_data = jax.lax.scan(
        step_fn, init_carry, None, length=num_steps
    )

    # Get bootstrap value for final state
    bootstrap_value = compute_values(
        processor_params, value_params, ppo_network, final_env_state.obs
    )

    # Build TrajectoryBatch
    trajectory = TrajectoryBatch(
        obs=step_data["obs"],
        actions=step_data["action"],
        log_probs=step_data["log_prob"],
        values=step_data["value"],
        task_rewards=step_data["task_reward"],
        dones=step_data["done"],
        truncations=step_data["truncation"],
        next_obs=step_data["next_obs"],
        bootstrap_value=bootstrap_value,
        # v0.10.4: Packed metrics vector
        metrics_vec=step_data["metrics_vec"],
        # Algorithm-critical fields
        step_counts=step_data["step_count"],
        # AMP fields - set to None if not collecting (saves memory)
        foot_contacts=step_data["foot_contact"] if collect_amp_features else None,
        root_heights=step_data["root_height"] if collect_amp_features else None,
        prev_joint_positions=step_data["prev_joint_pos"] if collect_amp_features else None,
    )

    return final_env_state, trajectory


def compute_reward_total(
    task_rewards: jnp.ndarray,
    amp_rewards: Optional[jnp.ndarray],
    amp_weight: float,
) -> jnp.ndarray:
    """Compute total reward = task_reward + amp_weight * amp_reward.

    This is the single source of truth for reward combination.

    Implementation guarantees:
    - When amp_weight == 0.0, returns task_rewards exactly (no modification)
    - When amp_rewards is None and amp_weight > 0, raises error

    Args:
        task_rewards: Task rewards from environment (num_steps, num_envs)
        amp_rewards: AMP rewards from discriminator (num_steps, num_envs) or None
        amp_weight: Weight for AMP rewards (0.0 = PPO-only)

    Returns:
        total_rewards: Combined rewards (num_steps, num_envs)
    """
    if amp_weight == 0.0:
        return task_rewards

    if amp_rewards is None:
        raise ValueError(
            f"amp_weight={amp_weight} > 0 but amp_rewards is None. "
            "Either set amp_weight=0 or provide amp_rewards."
        )

    return task_rewards + amp_weight * amp_rewards
