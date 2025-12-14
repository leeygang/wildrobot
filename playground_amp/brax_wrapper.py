"""Brax wrapper for WildRobot's MJX WildRobotEnv.

This module provides a thin adapter so Brax training code can call into
the existing `WildRobotEnv` (which uses mujoco.mjx). It purposely keeps the
surface minimal so it can be adapted to the exact Brax training API in use.

Note: Brax expects pure JAX functions and pytrees for maximal performance.
This adapter focuses on compatibility and a clear migration path; for full
performance you should port the env fully to Brax/native brax.envs patterns.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv


class BraxWildRobotEnv:
    """Thin wrapper exposing reset/step compatible with Brax-style training loops.

    It provides:
    - reset(rng) -> obs
    - step(obs, action) -> next_obs, reward, done, info

    This is intentionally small: the training script can adapt the loop for
    batching/vectorization using jax.vmap/jit as needed.
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

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step with `action` shaped (num_envs, act_dim).

        Returns: (obs, reward, done, info)
        All arrays are JAX arrays for compatibility; they may be host-backed if
        WildRobotEnv returns numpy arrays.
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

        with open(stage_config_path, 'r') as f:
            raw = yaml.safe_load(f)
    except Exception:
        # fallback: assume default phase3 config
        raw = {}

    env_path = raw.get('env_config_path', 'playground_amp/configs/wildrobot_env.yaml')
    env_cfg = EnvConfig.from_file(env_path)
    return BraxWildRobotEnv(env_cfg)
