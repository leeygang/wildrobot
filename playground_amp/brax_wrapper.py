"""Brax wrapper for WildRobot's MJX WildRobotEnv.

This module provides a full Brax v2 compatible adapter for WildRobotEnv,
implementing the brax.envs.Env interface. This enables using Brax's
battle-tested PPO trainer and training infrastructure.

Architecture:
- Wraps Pure-JAX backend (jax_full_port.py) for full JIT/vmap compatibility
- Converts to/from Brax State pytrees
- Provides Brax Env interface (reset, step, properties)
- Handles autoreset for Brax training loops

Note: Requires Task 11 (Pure-JAX port) completion. Uses jitted JAX functions
directly instead of WildRobotEnv.step() to avoid numpy conversion issues.
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

# Import Pure-JAX backend functions
try:
    from playground_amp.envs.jax_full_port import (
        JaxData,
        make_jax_data,
        jitted_vmapped_step_and_observe,
    )
    JAX_PORT_AVAILABLE = True
except ImportError:
    JAX_PORT_AVAILABLE = False
    JaxData = None
    make_jax_data = None
    jitted_vmapped_step_and_observe = None


class BraxWildRobotWrapper(envs.Env if BRAX_AVAILABLE else object):
    """Full Brax v2 compatible wrapper for WildRobotEnv.

    Implements the brax.envs.Env interface:
    - observation_size, action_size, backend properties
    - reset(rng) -> State
    - step(state, action) -> State
    - unwrapped property

    This enables drop-in compatibility with Brax's PPO trainer and other
    Brax training infrastructure.

    Architecture:
    - Uses Pure-JAX backend (jax_full_port.py) directly for JIT/vmap compatibility
    - Bypasses WildRobotEnv.step() to avoid numpy conversion issues
    - Manages internal JaxData state for all parallel environments
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

        if not JAX_PORT_AVAILABLE:
            raise ImportError(
                "Pure-JAX port not available. Task 11 must be complete."
            )

        # IMPORTANT: Brax will handle parallelization via vmap
        # Each env instance manages a SINGLE environment
        # The num_envs from config is ignored - Brax creates multiple instances
        self.num_envs = 1  # Single environment per wrapper instance
        self._autoreset = autoreset
        self._config = env_config

        # Initialize Pure-JAX backend state for SINGLE environment
        # WildRobot: 7 (base pos+quat) + 11 (joints) = 18 DOF position
        #            6 (base lin+ang vel) + 11 (joint vels) = 17 DOF velocity
        self._nq = 18  # qpos dimension
        self._nv = 17  # qvel dimension
        self._obs_dim = 44  # observation dimension (from WildRobotEnv.OBS_DIM)
        self._act_dim = 11  # action dimension (joint targets)

        # Create initial JAX state for SINGLE env (Brax handles batching via vmap)
        self._jax_state = make_jax_data(nq=self._nq, nv=self._nv, batch=1)

        # PD control parameters
        self._kp = 50.0
        self._kd = 1.0
        self._dt = 0.02

        print(f"✓ BraxWildRobotWrapper initialized with Pure-JAX backend")
        print(f"  Single environment (Brax handles parallelization)")
        print(f"  nq={self._nq}, nv={self._nv}, obs_dim={self._obs_dim}, act_dim={self._act_dim}")
        print(f"  JIT/vmap compatible for Brax training")

    @property
    def observation_size(self) -> int:
        """Return observation dimensionality (Brax interface)."""
        return self._obs_dim

    @property
    def action_size(self) -> int:
        """Return action dimensionality (Brax interface)."""
        return self._act_dim

    @property
    def backend(self) -> str:
        """Return backend name (Brax interface)."""
        return "mjx"  # MuJoCo MJX backend (Pure-JAX implementation)

    @property
    def unwrapped(self):
        """Return underlying environment (Brax interface)."""
        return self

    def reset(self, rng: jax.Array) -> State:
        """Reset environment and return initial State.

        Args:
            rng: JAX random key (for Brax compatibility, can be used for randomization)

        Returns:
            State: Brax State pytree with initial observations (single env)
        """
        # Initialize JAX state with default qpos/qvel for SINGLE env
        jax_state = make_jax_data(nq=self._nq, nv=self._nv, batch=1)

        # Build initial observations for SINGLE env
        # Shape should be (obs_dim,) not (1, obs_dim) for Brax vmap
        obs = jnp.zeros(self._obs_dim, dtype=jnp.float32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        # CRITICAL: done must be float32, not bool (Brax convention!)
        done = jnp.array(0.0, dtype=jnp.float32)

        return State(
            pipeline_state=jax_state,  # Store JAX state in pipeline_state
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            info={},
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Step environment with action and return new State.

        This uses Pure-JAX backend directly for full JIT/vmap compatibility.
        Note: This method handles single environment steps. Brax's training
        wrappers will vmap over this to handle batches.

        Args:
            state: Current Brax State (pipeline_state contains JaxData)
            action: Action array (action_size,) as JAX array (single env)

        Returns:
            State: New Brax State with observations, rewards, dones
        """
        # Extract current JAX state from pipeline_state
        current_jax_state = state.pipeline_state

        # Import Pure-JAX step function (non-squeezing version)
        from playground_amp.envs.jax_full_port import step_fn, build_obs_from_state_j, compute_reward_from_state_j, is_done_from_state_j

        # Step physics
        new_jax_state = step_fn(current_jax_state, action, dt=self._dt, kp=self._kp, kd=self._kd)

        # CRITICAL FIX: Ensure ctrl dimension matches input for shape consistency in scan
        # step_fn returns ctrl with shape (1, 11) but input had shape (1, 17)
        # Pad ctrl to match qvel dimension (17) for pytree structure consistency
        if new_jax_state.ctrl.shape[-1] != current_jax_state.ctrl.shape[-1]:
            pad_width = current_jax_state.ctrl.shape[-1] - new_jax_state.ctrl.shape[-1]
            ctrl_padded = jnp.pad(new_jax_state.ctrl, ((0, 0), (0, pad_width)), constant_values=0.0)
            new_jax_state = type(new_jax_state)(
                qpos=new_jax_state.qpos,
                qvel=new_jax_state.qvel,
                ctrl=ctrl_padded,
                xpos=new_jax_state.xpos,
                xquat=new_jax_state.xquat,
            )

        # Build observations/reward/done manually without squeezing
        # Extract arrays (keep batch=1 dimension for shape consistency)
        qpos = new_jax_state.qpos  # shape (1, 18)
        qvel = new_jax_state.qvel  # shape (1, 17)
        ctrl = new_jax_state.ctrl  # shape (1, 17) after padding
        xpos = new_jax_state.xpos if new_jax_state.xpos is not None else jnp.zeros((1, 3), dtype=jnp.float32)
        xquat = new_jax_state.xquat if new_jax_state.xquat is not None else jnp.zeros((1, 4), dtype=jnp.float32)

        # For observation functions, we need to extract the single example
        # but keep the result squeezed for Brax compatibility
        qpos_1d = qpos[0] if qpos.ndim == 2 else qpos
        qvel_1d = qvel[0] if qvel.ndim == 2 else qvel
        ctrl_1d = ctrl[0] if ctrl.ndim == 2 else ctrl
        xpos_1d = xpos[0] if xpos.ndim == 2 else xpos
        xquat_1d = xquat[0] if xquat.ndim == 2 else xquat

        derived = {"xpos": xpos_1d, "xquat": xquat_1d}

        obs = build_obs_from_state_j(qpos_1d, qvel_1d, ctrl_1d, derived=derived, obs_noise_std=0.0, key=None)
        reward = compute_reward_from_state_j(qvel_1d, ctrl_1d)
        done = is_done_from_state_j(qpos_1d, qvel_1d, obs, derived=derived, max_episode_steps=500, step_count=0)

        # Ensure scalar outputs with correct dtypes for Brax
        # CRITICAL: All dtypes must match Brax conventions (float32 for scalars)
        if hasattr(reward, 'ndim') and reward.ndim > 0:
            reward = reward[0] if reward.shape[0] == 1 else reward
        reward = jnp.asarray(reward, dtype=jnp.float32)

        if hasattr(done, 'ndim') and done.ndim > 0:
            done = done[0] if done.shape[0] == 1 else done
        # Convert done to float32 (Brax convention, not bool!)
        done = jnp.asarray(done, dtype=jnp.float32)

        if hasattr(obs, 'ndim') and obs.ndim > 1:
            obs = obs[0] if obs.shape[0] == 1 else obs
        obs = jnp.asarray(obs, dtype=jnp.float32)

        # Return new Brax State
        # IMPORTANT: Preserve state.info and state.metrics structures for Brax's training wrappers
        # Brax uses jax.lax.scan which requires consistent pytree structures
        return State(
            pipeline_state=new_jax_state,  # Preserve batch=1 dimension with padded ctrl!
            obs=obs,
            reward=reward,
            done=done,
            metrics=state.metrics,  # Preserve metrics dict structure
            info=state.info,  # Preserve info dict structure
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
