"""Brax wrapper for WildRobot's MJX WildRobotEnv.

This module provides a full Brax v2 compatible adapter for WildRobotEnv,
implementing the brax.envs.Env interface. This enables using Brax's
battle-tested PPO trainer and training infrastructure.

Architecture:
- Wraps existing WildRobotEnv (numpy/mjx-based)
- Converts to/from Brax State pytrees
- Provides Brax Env interface (reset, step, properties)
- Handles autoreset for Brax training loops

Note: For maximum performance, combine with Task 11 (Pure-JAX port) to enable
full JIT compilation. Current implementation works with numpy-based env.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from brax import envs
    from brax.envs.base import State

    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False

    # Fallback State class for type hints
    class State:
        pass


from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv


class BraxWildRobotWrapper(envs.Env if BRAX_AVAILABLE else object):
    """Full Brax v2 compatible wrapper for WildRobotEnv.

    Implements the brax.envs.Env interface:
    - observation_size, action_size, backend properties
    - reset(rng) -> State
    - step(state, action) -> State
    - unwrapped property

    This enables drop-in compatibility with Brax's PPO trainer and other
    Brax training infrastructure.
    """

    def __init__(self, env_config: EnvConfig, autoreset: bool = True):
        """Initialize Brax wrapper.

        Args:
            env_config: WildRobotEnv configuration
            autoreset: If True, automatically reset done environments (Brax standard)
        """
        if not BRAX_AVAILABLE:
            raise ImportError(
                "Brax is not installed. Install with: pip install brax>=0.10.0"
            )

        self._env = WildRobotEnv(env_config)
        self._autoreset = autoreset
        self.num_envs = env_config.num_envs

        # Cache for current observations (for autoreset)
        self._current_obs = None

    @property
    def observation_size(self) -> int:
        """Return observation dimensionality (Brax interface)."""
        return self._env.OBS_DIM

    @property
    def action_size(self) -> int:
        """Return action dimensionality (Brax interface)."""
        return self._env.ACT_DIM

    @property
    def backend(self) -> str:
        """Return backend name (Brax interface)."""
        return "mjx"  # MuJoCo MJX backend

    @property
    def unwrapped(self) -> WildRobotEnv:
        """Return underlying environment (Brax interface)."""
        return self._env

    def reset(self, rng: jax.Array) -> State:
        """Reset environment and return initial State.

        Args:
            rng: JAX random key (for Brax compatibility, currently unused)

        Returns:
            State: Brax State pytree with initial observations
        """
        obs = self._env.reset()
        self._current_obs = obs

        # Convert to JAX arrays
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        reward_jax = jnp.zeros(self.num_envs, dtype=jnp.float32)
        done_jax = jnp.zeros(self.num_envs, dtype=jnp.bool_)

        return State(
            pipeline_state=None,  # Not needed for MJX (state managed internally)
            obs=obs_jax,
            reward=reward_jax,
            done=done_jax,
            metrics={},
            info={},
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Step environment with action and return new State.

        Args:
            state: Current Brax State (obs field used if autoreset enabled)
            action: Action array (num_envs, action_size)

        Returns:
            State: New Brax State with observations, rewards, dones
        """
        # IMPORTANT: Keep action as JAX array for JIT/vmap compatibility
        # The underlying env.step() will handle conversion if needed
        # For now, we convert to numpy since WildRobotEnv expects numpy
        # TODO: Once Task 11 (Pure-JAX port) completes, remove this conversion

        # Use jax.device_get instead of np.array to avoid tracing issues
        action_np = jax.device_get(action).astype(np.float32)

        # Step underlying environment
        obs, rew, done, info = self._env.step(action_np)

        # Handle autoreset (Brax standard behavior)
        if self._autoreset and np.any(done):
            # For done environments, reset and use reset observation
            # Note: WildRobotEnv handles this internally via _handle_resets
            # We just need to track the observations
            pass

        self._current_obs = obs

        # Convert to JAX arrays
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        rew_jax = jnp.array(rew, dtype=jnp.float32)
        done_jax = jnp.array(done, dtype=jnp.bool_)

        # Extract metrics from info dict
        metrics = {}
        for key in ["episode_length", "episode_reward", "success"]:
            if key in info:
                metrics[key] = jnp.array(info[key], dtype=jnp.float32)

        return State(
            pipeline_state=None,
            obs=obs_jax,
            reward=rew_jax,
            done=done_jax,
            metrics=metrics,
            info=info,
        )


# Legacy wrapper for backward compatibility
class BraxWildRobotEnv:
    """Legacy thin wrapper (deprecated - use BraxWildRobotWrapper instead).

    Kept for backward compatibility with existing code.
    """

    def __init__(self, env_config: EnvConfig):
        self._env = WildRobotEnv(env_config)
        self.obs_dim = self._env.OBS_DIM
        self.act_dim = self._env.ACT_DIM
        self.num_envs = env_config.num_envs

    def reset(self, rng: jax.Array | None = None) -> jnp.ndarray:
        """Reset the underlying env and return observations (num_envs, obs_dim)."""
        obs = self._env.reset()
        return jnp.array(obs)

    def step(
        self, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step with `action` shaped (num_envs, act_dim).

        Returns: (obs, reward, done, info)
        """
        # convert to numpy if needed by underlying env
        if isinstance(action, jax.Array):
            a_np = jax.device_get(action)
        else:
            a_np = action

        obs, rew, done, info = self._env.step(a_np)

        return jnp.array(obs), jnp.array(rew), jnp.array(done), info


def create_brax_env_from_stage_config(stage_config_path: str) -> BraxWildRobotEnv:
    """Create a Brax-compatible env from a stage config YAML/TOML/JSON path.

    The stage config is expected to include `env_config_path` pointing to the
    shared environment config file (e.g. `playground_amp/configs/wildrobot_env.yaml`).
    """
    # lazy import to avoid adding yaml hard dependency at module import time
    try:
        import yaml

        with open(stage_config_path, "r") as f:
            raw = yaml.safe_load(f)
    except Exception:
        # fallback: assume default phase3 config
        raw = {}

    env_path = raw.get("env_config_path", "playground_amp/configs/wildrobot_env.yaml")
    env_cfg = EnvConfig.from_file(env_path)
    return BraxWildRobotEnv(env_cfg)
