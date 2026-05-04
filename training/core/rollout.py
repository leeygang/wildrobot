"""Rollout collector for PPO training.

v0.10.4: Registry-based metrics
- Replaced individual metric fields with single `metrics_vec` tensor
- Uses metrics_registry for schema enforcement
- No more .get() calls with silent fallbacks
- Adding new metrics only requires registry + env changes
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from training.envs.env_info import WR_INFO_KEY
from training.core.metrics_registry import METRICS_VEC_KEY, NUM_METRICS

from training.algos.ppo.ppo_core import compute_values, sample_actions


class TrajectoryBatch(NamedTuple):
    """Container for collected trajectory data.

    Shapes:
        All tensors have shape (num_steps, num_envs, ...)

    Fields:
        obs: Observations at each step
        actions: Raw actions (pre-postprocessing, for log_prob computation)
        log_probs: Log probabilities of actions under policy
        values: Value estimates at each step
        task_rewards: Task rewards from environment
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
    """

    # Core PPO fields
    obs: jnp.ndarray
    critic_obs: jnp.ndarray
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


def collect_rollout(
    env_step_fn: Callable,
    env_state: Any,
    policy_params: Any,
    value_params: Any,
    processor_params: Any,
    ppo_network: Any,
    rng: jax.Array,
    num_steps: int,
    use_privileged_critic: bool = False,
) -> Tuple[Any, TrajectoryBatch]:
    """Collect rollout using jax.lax.scan (fully on GPU).

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

    Returns:
        (final_env_state, trajectory_batch)
    """

    def step_fn(carry, _):
        env_state, rng = carry
        rng, action_rng = jax.random.split(rng)

        obs = env_state.obs

        # Sample action from policy (deterministic RNG consumption)
        action, raw_action, log_prob = sample_actions(
            processor_params, policy_params, ppo_network, obs, action_rng
        )

        # Get value estimate
        if use_privileged_critic:
            critic_obs = env_state.info[WR_INFO_KEY].critic_obs
        else:
            critic_obs = obs
        value = compute_values(
            processor_params, value_params, ppo_network, obs, critic_obs
        )

        # Step environment
        next_env_state = env_step_fn(env_state, action)

        # Access typed WildRobotInfo namespace directly (fail loudly if missing)
        wr_info = next_env_state.info[WR_INFO_KEY]

        # Build step data - minimal fields, metrics come from metrics[METRICS_VEC_KEY]
        step_data = {
            # Core PPO fields
            "obs": obs,
            "critic_obs": critic_obs,
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
        }

        return (next_env_state, rng), step_data

    # Run scan
    init_carry = (env_state, rng)
    (final_env_state, _), step_data = jax.lax.scan(
        step_fn, init_carry, None, length=num_steps
    )

    # Get bootstrap value for final state
    if use_privileged_critic:
        bootstrap_critic_obs = final_env_state.info[WR_INFO_KEY].critic_obs
    else:
        bootstrap_critic_obs = final_env_state.obs

    bootstrap_value = compute_values(
        processor_params,
        value_params,
        ppo_network,
        final_env_state.obs,
        bootstrap_critic_obs,
    )

    # Build TrajectoryBatch
    trajectory = TrajectoryBatch(
        obs=step_data["obs"],
        critic_obs=step_data["critic_obs"],
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
    )

    return final_env_state, trajectory
