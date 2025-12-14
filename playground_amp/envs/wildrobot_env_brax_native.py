"""Brax-native scaffold: subclass of `WildRobotEnv` prepared for porting to a
pure-JAX, Brax-friendly environment.

This file provides a minimally-invasive subclass that (1) ensures the
underlying model is an MJX model (via `mjx.put_model` when necessary), (2)
creates a per-instance `mjx.Data` object, and (3) exposes `init_state` and
`step_state` methods which are the starting point for converting to a
pure functional, jittable API.

This is a staged approach: it keeps existing functionality while making the
code paths to a full port explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

import mujoco
import mujoco.mjx as mjx

from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig


class BraxNativeWildRobotEnv(WildRobotEnv):
    """Subclass of `WildRobotEnv` that prepares MJX-native model/data and
    exposes stateful step functions intended for later conversion to pure
    JAX functions.

    Notes:
    - This class still uses host-backed mutable `mjx.Data` for correctness
      and ease of migration. Later the goal is to make `step_state` a
      pure function `step_fn(model, data, action) -> (new_data, obs, reward)`
      and JIT/VMAP it for full speed.
    """

    def __init__(self, env_cfg: EnvConfig):
        # initialize base implementation (keeps current behavior)
        super().__init__(env_cfg)

        # ensure the model has an mjx 'impl' (native mjx model) for mjx.step
        # If the loaded model lacks `.impl`, convert it using mjx.put_model.
        model = getattr(self, 'model', None)
        if model is None:
            # fallback: try to load from path
            model = mujoco.MjModel.from_xml_path(env_cfg.model_path)

        if not hasattr(model, 'impl'):
            model = mjx.put_model(model)

        self.model = model

        # create mjx.Data for per-instance stepping
        try:
            self._data = mjx.make_data(self.model)
        except Exception:
            # mjx.make_data may fail if model is unexpected; re-raise with context
            raise

    def init_state(self) -> Dict[str, np.ndarray]:
        """Return a host-backed state dict (qpos, qvel) for the env's reset."""
        # Use current data qpos/qvel as initial state
        return {
            'qpos': np.array(self._data.qpos, copy=True),
            'qvel': np.array(self._data.qvel, copy=True),
        }

    def step_state(self, state: Dict[str, np.ndarray], action: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step from explicit state with `action` and return (next_state, obs, rew, done, info).

        This method purposely mirrors a functional `step_fn` signature but uses
        the host-backed `mjx.Data` for now. The migration plan is to convert
        this to a pure `jax` function that takes and returns pytrees.
        """
        # write state into data buffers
        self._data.qpos[:] = state['qpos']
        self._data.qvel[:] = state['qvel']

        # set control (assumes actions are PD targets or torques depending on env)
        # convert action shape if necessary
        a = np.asarray(action)
        if a.shape != self._data.ctrl.shape:
            # try to broadcast or reshape
            a = a.reshape(self._data.ctrl.shape)
        self._data.ctrl[:] = a

        # perform an MJX step
        mjx.step(self.model, self._data)

        # extract next state
        next_state = {
            'qpos': np.array(self._data.qpos, copy=True),
            'qvel': np.array(self._data.qvel, copy=True),
        }

        # build observation using parent helper if available
        try:
            obs = self._build_obs_from_data(self._data)
        except Exception:
            # fallback: use parent `step` to get obs/reward — but avoid recursion
            obs = np.zeros((self.OBS_DIM,))

        # compute reward and done using existing utilities when possible
        try:
            rew = np.array(self._compute_reward_from_data(self._data))
            done = np.array(self._is_done_from_data(self._data))
        except Exception:
            # simple placeholders
            rew = np.array(0.0)
            done = np.array(False)

        info = {}
        return next_state, obs, rew, done, info

    # Helper wrappers: attempt to call parent data-based helpers if present
    def _build_obs_from_data(self, data) -> np.ndarray:
        if hasattr(self, '_build_obs'):
            return self._build_obs()
        raise NotImplementedError()

    def _compute_reward_from_data(self, data) -> float:
        if hasattr(self, '_compute_reward'):
            return self._compute_reward()
        raise NotImplementedError()

    def _is_done_from_data(self, data) -> bool:
        if hasattr(self, '_is_done'):
            return self._is_done()
        raise NotImplementedError()

    # TODOs for a full port:
    # - Convert `step_state` into a pure function: `step_fn(model, data, action)`
    # - Make `data` a pytree and avoid host-side mutation
    # - Support jax.jit/jax.vmap on `step_fn` for large-scale vectorization
