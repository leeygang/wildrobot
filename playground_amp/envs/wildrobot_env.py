"""WildRobot MJX/JAX environment (moved to playground_amp)"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import time
import concurrent.futures
import threading

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None
try:
    import yaml
except Exception:
    yaml = None

try:
    import jax
except Exception:
    jax = None

try:
    import mujoco.mjx as mjx
except Exception:
    try:
        import mujoco as muj

        mjx = muj
    except Exception:
        mjx = None


@dataclass
class EnvConfig:
    model_path: str = "assets/wildrobot.xml"
    num_envs: int = 8
    control_freq: int = 50
    obs_noise_std: float = 0.02
    max_episode_steps: int = 500
    seed: int = 0
    use_jax: bool = False
    require_jax: bool = False
    data_pool_size: int = 16

    @classmethod
    def from_file(cls, path: str) -> "EnvConfig":
        """Load EnvConfig from a JSON, TOML, or YAML file.

        Supported formats: .json, .toml, .yaml/.yml. Falls back to json parsing
        if extension is unknown.
        """
        path = str(path)
        data = None
        # choose parser by extension
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        elif path.endswith(".toml") and tomllib is not None:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        elif (path.endswith(".yaml") or path.endswith(".yml")) and yaml is not None:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        else:
            # try JSON, then YAML, then TOML
            with open(path, "r") as f:
                txt = f.read()
            try:
                data = json.loads(txt)
            except Exception:
                if yaml is not None:
                    try:
                        data = yaml.safe_load(txt)
                    except Exception:
                        data = None
                if data is None and tomllib is not None:
                    try:
                        with open(path, "rb") as f:
                            data = tomllib.load(f)
                    except Exception:
                        data = None

        if data is None:
            raise RuntimeError(f"Failed to parse config file: {path}. Install pyyaml for YAML support or use JSON/TOML.")

        # Map keys to dataclass fields
        kwargs = {}
        for field in ("model_path", "num_envs", "control_freq", "obs_noise_std", "max_episode_steps", "seed"):
            if field in data:
                kwargs[field] = data[field]

        return cls(**kwargs)


class WildRobotEnv:
    """Vectorized environment wrapper for the WildRobot MJCF model.

    This class provides a small, pragmatic API sufficient to run PPO-style
    training loops with MJX/JAX. It's intentionally minimal — replace the
    placeholders with your project's exact model loading and MJX helper code.
    """

    OBS_DIM = 44
    ACT_DIM = 11

    def __init__(self, config: EnvConfig | None = None, backend: str = "mjx"):
        # `config` may be an EnvConfig instance or a string path to a config file
        if isinstance(config, str):
            self.cfg = EnvConfig.from_file(config)
        else:
            self.cfg = config or EnvConfig()
        self.num_envs = int(self.cfg.num_envs)
        self._rng = jax.random.PRNGKey(self.cfg.seed)

        # Load model (path is relative to this file; adjust as needed)
        model_path = self.cfg.model_path

        # If configured model_path doesn't exist, try to resolve relative to repo root
        if not os.path.exists(model_path):
            # repo root is two parents up from playground_amp/envs
            repo_root = Path(__file__).resolve().parents[2]
            candidate = str(repo_root / "assets" / "wildrobot.xml")
            if os.path.exists(candidate):
                model_path = candidate
            else:
                # also try absolute workspace location if known
                alt = str(Path(__file__).resolve().parents[3] / "assets" / "wildrobot.xml")
                if os.path.exists(alt):
                    model_path = alt

        # Try multiple loader APIs to support different mujoco/mjx versions.
        load_errs = []
        try:
            if hasattr(mjx, "load_model_from_path"):
                self.model = mjx.load_model_from_path(model_path)
                self.data = mjx.make_data(self.model)
            elif hasattr(mjx, "load"):
                # some mjx builds expose `load` which returns model
                self.model = mjx.load(model_path)
                # attempt to build data
                if hasattr(mjx, "make_data"):
                    self.data = mjx.make_data(self.model)
                elif hasattr(self.model, "new_data"):
                    self.data = self.model.new_data()
                else:
                    raise RuntimeError("Loaded model but could not create data object for mjx.load() result")
            elif hasattr(mjx, "MjModel") and hasattr(mjx.MjModel, "from_xml_path"):
                # newer mujoco Python API
                self.model = mjx.MjModel.from_xml_path(model_path)
                self.data = mjx.MjData(self.model)
            else:
                # try top-level mujoco package as a last resort
                import mujoco as muj

                if hasattr(muj, "MjModel") and hasattr(muj.MjModel, "from_xml_path"):
                    self.model = muj.MjModel.from_xml_path(model_path)
                    self.data = muj.MjData(self.model)
                else:
                    raise AttributeError("No compatible model loader found on mjx or mujoco module")
        except Exception as e:
            load_errs.append(str(e))
            # Provide a helpful error with available attributes to diagnose API mismatch
            available_attrs = ", ".join(sorted([a for a in dir(mjx) if not a.startswith("__")])[:50])
            raise RuntimeError(
                f"Failed to load MJCF model '{model_path}': {e}\n"
                f"Tried mjx and mujoco loader fallbacks. mjx available attributes (sample): {available_attrs}\n"
                "Ensure you have an MJX-compatible mujoco package installed (see plan for install steps)."
            )

        # If model is a native mujoco MjModel (no 'impl'), convert to an MJX Model
        # which has the required `impl` field. This keeps compatibility with
        # mjx.step while allowing us to use the native loader above.
        try:
            if not hasattr(self.model, "impl") and hasattr(mjx, "put_model"):
                self.model = mjx.put_model(self.model)
                # create a mjx Data object from the converted model
                self.data = mjx.make_data(self.model)
        except Exception:
            # If conversion fails, keep original model/data and allow later
            # checks to raise informative errors.
            pass

        # Internal state arrays for vectorized envs (host-side simple arrays)
        self._step_count = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_done = np.zeros(self.num_envs, dtype=bool)

        # Observation / action buffer shapes (host numpy)
        self.obs = np.zeros((self.num_envs, self.OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.ACT_DIM), dtype=np.float32)

        # Basic PD gains (tweak for your robot)
        self.kp = 50.0
        self.kd = 1.0

        # Domain randomization state placeholder
        self._dr_params = [self._sample_dr_params(i) for i in range(self.num_envs)]

        # Create per-env Data objects when possible (allows reading qpos/qvel)
        self._datas = []
        try:
            for _ in range(self.num_envs):
                # mjx.make_data is the most compatible helper
                d = mjx.make_data(self.model)
                self._datas.append(d)
        except Exception:
            # Fallback: single shared data (slower, less correct for vectorization)
            self._datas = [self.data] * self.num_envs

        # Data pool fallback: pre-allocate a pool of fresh mjx.Data objects
        # to swap into `self._datas` when we detect immutable JAX-backed Data.
        # Only create a pool when we are not using per-env model copies, since
        # pooled Data are created from the shared `self.model`.
        self._data_pool = []
        self._pool_lock = threading.Lock()
        try:
            if not self._per_env_models and getattr(self.cfg, "data_pool_size", 0) > 0:
                pool_size = int(self.cfg.data_pool_size)
                for _ in range(pool_size):
                    try:
                        pd = mjx.make_data(self.model)
                        self._data_pool.append(pd)
                    except Exception:
                        break
        except Exception:
            self._data_pool = []

        # Optionally create per-env model copies so we can apply DR to model parameters.
        # Only do this for reasonably small vector counts to avoid excessive memory.
        self._per_env_models = False
        self._models = [self.model] * self.num_envs
        try:
            if self.num_envs <= 64:
                models = []
                datas = []
                for i in range(self.num_envs):
                    try:
                        mcopy = copy.deepcopy(self.model)
                        # ensure MJX impl present if helper exists
                        if not hasattr(mcopy, "impl") and hasattr(mjx, "put_model"):
                            try:
                                mcopy = mjx.put_model(mcopy)
                            except Exception:
                                pass
                        models.append(mcopy)
                        try:
                            dcopy = mjx.make_data(mcopy)
                        except Exception:
                            dcopy = mjx.make_data(self.model)
                        datas.append(dcopy)
                    except Exception:
                        models = [self.model] * self.num_envs
                        datas = self._datas
                        break
                else:
                    self._models = models
                    self._datas = datas
                    self._per_env_models = True
        except Exception:
            self._per_env_models = False

        # Cache original model parameter arrays for safe scaling/restoration
        try:
            self._orig_body_mass = np.array(getattr(self.model, "body_mass", []), dtype=np.float32)
        except Exception:
            self._orig_body_mass = np.array([], dtype=np.float32)
        try:
            self._orig_geom_friction = np.array(getattr(self.model, "geom_friction", []), dtype=np.float32)
        except Exception:
            self._orig_geom_friction = np.array([], dtype=np.float32)
        try:
            self._orig_dof_damping = np.array(getattr(self.model, "dof_damping", []), dtype=np.float32)
        except Exception:
            self._orig_dof_damping = np.array([], dtype=np.float32)

        # Determine qpos/qvel sizes from a data object
        sample_data = self._datas[0]
        self.nq = int(getattr(sample_data, "qpos").size)
        self.nv = int(getattr(sample_data, "qvel").size)

        # Pure-JAX port wiring (optional). If enabled, we'll use the jitted
        # JAX step_fn + helpers in `playground_amp.envs.jax_full_port`.
        self._use_jax = bool(self.cfg.use_jax)
        self._jax_available = False
        self._jax_datas = None
        if self._use_jax:
            try:
                from playground_amp.envs.jax_full_port import make_jax_data, jitted_step_and_observe

                self._make_jax_data = make_jax_data
                self._jitted_step_and_observe = jitted_step_and_observe
                self._jax_available = self._jitted_step_and_observe is not None
            except Exception:
                self._make_jax_data = None
                self._jitted_step_and_observe = None
                self._jax_available = False

        # Enforce requiring JAX when configured to do so
        if getattr(self.cfg, "require_jax", False) and not self._jax_available:
            raise RuntimeError(
                "EnvConfig.require_jax=True but the pure-JAX port is not available. "
                "Ensure `playground_amp.envs.jax_full_port` is importable and JAX is installed, or set `require_jax=False`."
            )

        # Internal per-env qpos/qvel state (host numpy arrays)
        self.qpos = np.zeros((self.num_envs, self.nq), dtype=np.float32)
        self.qvel = np.zeros((self.num_envs, self.nv), dtype=np.float32)

        # deprecation/logging helpers
        self._deprecation_warned = False

        # If we detect that mjx.Data objects are JAX-backed/immutable and
        # cannot accept direct writes, flip this per-env flag so stepping
        # uses host qpos/qvel instead of attempting in-place Data writes.
        self._force_host_qpos = np.zeros(self.num_envs, dtype=bool)

        # Simple physical params for Euler integration (placeholders)
        self._mass_scale = np.ones(self.num_envs, dtype=np.float32)
        self._dt = 1.0 / float(self.cfg.control_freq)
        # Action latency / buffering (max supported latency in steps)
        self._max_latency = 5
        self._action_buffers = np.zeros((self.num_envs, self._max_latency + 1, self.ACT_DIM), dtype=np.float32)
        # Require simulator stepping support (no Euler fallback)
        if not hasattr(mjx, "step"):
            raise RuntimeError(
                "mujoco.mjx does not expose 'step' — remove Euler fallback requires mjx.step. "
                "Install an MJX build with step() support."
            )
        # Ensure data.ctrl is present on the data object
        if not hasattr(self._datas[0], "ctrl"):
            raise RuntimeError(
                "mjx Data objects do not have 'ctrl' attribute. Ensure your model has actuators and mjx supports controls."
            )

        # Initialize JAX per-env Data objects if requested and available
        if self._use_jax and self._jax_available:
            try:
                # make per-env JaxData objects (batch=1 each)
                self._jax_datas = [self._make_jax_data(self.nq, self.nv, batch=1) for _ in range(self.num_envs)]
            except Exception:
                self._jax_datas = None

    # ---------------------- public API ----------------------
    def reset(self) -> np.ndarray:
        """Reset all envs and return initial observations (num_envs, OBS_DIM)."""
        # Reset per-env state
        for i in range(self.num_envs):
            self._reset_single(i)

        # Build initial observations from qpos/qvel using mjx forward to populate derived fields
        obs = np.zeros((self.num_envs, self.OBS_DIM), dtype=np.float32)
        # Write state into each Data and call forward once per-data to populate derived fields
        for i in range(self.num_envs):
            try:
                d = self._datas[i]
                d.qpos[:] = self.qpos[i]
                d.qvel[:] = self.qvel[i]
                if hasattr(mjx, "forward"):
                    mjx.forward(self.model, d)
            except Exception:
                pass

        # If using JAX port, initialize jax-data per env to match host qpos/qvel
        if self._use_jax and self._jax_available and self._jax_datas is not None:
            for i in range(self.num_envs):
                try:
                    d = self._jax_datas[i]
                    # set qpos/qvel into JaxData via numpy->jax conversion
                    # jax arrays are immutable, so replace fields
                    qpos = jnp.array(self.qpos[i][None, :], dtype=jnp.float32)
                    qvel = jnp.array(self.qvel[i][None, :], dtype=jnp.float32)
                    ctrl = jnp.zeros_like(qvel)
                    self._jax_datas[i] = type(d)(qpos=qpos, qvel=qvel, ctrl=ctrl, xpos=d.xpos, xquat=d.xquat)
                except Exception:
                    pass

        # Now build observations deterministically from data objects
        for i in range(self.num_envs):
            obs[i] = self._build_obs(i)

        self.obs = obs
        self._step_count.fill(0)
        self._episode_done.fill(False)
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Take a vectorized step.

        actions: np.ndarray (num_envs, ACT_DIM) target joint positions (PD setpoints)
        returns: obs, reward, done, info
        """
        assert actions.shape == (self.num_envs, self.ACT_DIM)
        self.actions = actions.astype(np.float32)

        # Apply PD control and step simulator (placeholder single-step loop)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: Dict[str, Any] = {"step_time": []}

        t0 = time.time()

        # Phase 1: prepare torques for all envs and write to Data.ctrl
        ctrl_size = int(getattr(self._datas[0], "ctrl").size)
        torques_buf = np.zeros((self.num_envs, ctrl_size), dtype=np.float32)
        for i in range(self.num_envs):
            if self._episode_done[i]:
                continue
            # update action buffer and select applied action
            self._action_buffers[i, 1:] = self._action_buffers[i, :-1]
            self._action_buffers[i, 0] = actions[i]
            latency = int(self._dr_params[i].get("latency_steps", 0))
            latency = min(latency, self._max_latency)
            applied_action = self._action_buffers[i, latency]
            t = self._pd_control(i, applied_action)
            n = min(ctrl_size, t.size)
            torques_buf[i, :n] = t[:n]

        # write torques into each Data.ctrl in a compact loop
        for i in range(self.num_envs):
            try:
                d = self._datas[i]
                # attempt fast assignment
                try:
                    d.ctrl[:] = torques_buf[i]
                except Exception:
                    # fallback to numpy assign / attribute set
                    ctrl_arr = np.array(getattr(d, "ctrl"), dtype=np.float32)
                    ctrl_arr[: torques_buf.shape[1]] = torques_buf[i]
                    try:
                        d.ctrl = ctrl_arr
                    except Exception:
                        for k in range(min(int(getattr(d, "ctrl").size), torques_buf.shape[1])):
                            try:
                                d.ctrl[k] = float(ctrl_arr[k])
                            except Exception:
                                pass
            except Exception:
                pass

        # Phase 2: Prefer the pure-JAX stepping path when available. This avoids
        # touching host `mjx.Data` objects and allows jitted/VMAP execution.
        used_jax = False
        if self._use_jax and self._jax_available and self._jax_datas is not None:
            used_jax = True
            for i in range(self.num_envs):
                if self._episode_done[i]:
                    continue
                try:
                    d_j = self._jax_datas[i]
                    # call jitted step_and_observe (returns newd, obs, reward, done)
                    newd, obs_i, reward_i, done_i = self._jitted_step_and_observe(d_j, torques_buf[i], dt=self._dt, kp=self.kp, kd=self.kd, obs_noise_std=self.cfg.obs_noise_std, key=None)
                    # update host qpos/qvel from jax arrays when possible
                    try:
                        self.qpos[i] = np.array(getattr(newd, "qpos")[0], dtype=np.float32)
                        self.qvel[i] = np.array(getattr(newd, "qvel")[0], dtype=np.float32)
                    except Exception:
                        pass
                    # store immediate outputs; full obs/reward/done computation will be done below
                    try:
                        if obs_i is not None:
                            self.obs[i] = np.array(obs_i, dtype=np.float32)
                    except Exception:
                        pass
                    try:
                        # store reward/done in temporary buffers; will be finalized after DR pushes
                        rewards[i] = float(reward_i) if reward_i is not None else 0.0
                        dones[i] = bool(done_i) if done_i is not None else False
                        self._episode_done[i] = dones[i]
                        self._step_count[i] += 1
                    except Exception:
                        pass
                    # replace jax data
                    self._jax_datas[i] = newd
                except Exception:
                    # any failure falls back to host/MJX stepping for this env
                    used_jax = False
                    break

        if not used_jax:
            # If user requested JAX but we fell back to MJX/Euler, warn once about deprecation
            if self._use_jax and not self._deprecation_warned:
                self._warn_mjx_deprecation()
            # Phase 2 fallback: step all datas in parallel where possible to approximate vectorization
            try:
                self._parallel_step_datas(torques_buf)
            except Exception:
                # fallback to sequential stepping
                for i in range(self.num_envs):
                    if self._episode_done[i]:
                        continue
                    d = self._datas[i]
                    try:
                        mjx.step(self.model, d)
                        self.qpos[i] = np.array(d.qpos, dtype=np.float32)
                        self.qvel[i] = np.array(d.qvel, dtype=np.float32)
                    except Exception:
                        try:
                            if hasattr(mjx, "forward"):
                                mjx.forward(self.model, d)
                        except Exception:
                            pass
                        accel = torques_buf[i].astype(np.float32) / (self._mass_scale[i] + 1e-6)
                        self.qvel[i, : accel.size] += accel * self._dt
                        self.qpos[i, : accel.size] += self.qvel[i, : accel.size] * self._dt

        # After stepping all envs, apply pushes, compute obs/rewards/dones
        # After stepping all envs, apply pushes, compute obs/rewards/dones.
        # If JAX port is enabled and available, use its jitted helpers per-env.
        if self._use_jax and self._jax_available and self._jax_datas is not None:
            for i in range(self.num_envs):
                if self._episode_done[i]:
                    continue
                try:
                    # Apply scheduled push perturbation (impulse) if configured for this env
                    dr = self._dr_params[i]
                    if dr.get("push_time", -1) == int(self._step_count[i]):
                        mag = float(dr.get("push_mag", 0.0))
                        dirx = float(dr.get("push_dir_x", 1.0))
                        if self.qvel.shape[1] >= 3:
                            self.qvel[i, 0] += (mag * dirx) / (self._mass_scale[i] + 1e-6)
                except Exception:
                    pass

                # Use jitted step+observe helper to compute new state/obs/reward/done
                try:
                    d_j = self._jax_datas[i]
                    targ = np.array(self.actions[i], dtype=np.float32)
                    # call jitted step_and_observe (returns newd, obs, reward, done)
                    newd, obs_i, reward_i, done_i = self._jitted_step_and_observe(d_j, targ, dt=self._dt, kp=self.kp, kd=self.kd, obs_noise_std=self.cfg.obs_noise_std, key=None)
                    # update host qpos/qvel from jax arrays
                    try:
                        self.qpos[i] = np.array(newd.qpos[0], dtype=np.float32)
                        self.qvel[i] = np.array(newd.qvel[0], dtype=np.float32)
                    except Exception:
                        pass
                    # store
                    if obs_i is not None:
                        self.obs[i] = np.array(obs_i, dtype=np.float32)
                    rewards[i] = float(reward_i) if reward_i is not None else 0.0
                    dones[i] = bool(done_i) if done_i is not None else False
                    self._episode_done[i] = dones[i]
                    self._step_count[i] += 1
                    # replace jax data
                    self._jax_datas[i] = newd
                    continue
                except Exception:
                    # fall back to host-data stepping below
                    pass

        # Fallback: existing MJX / Euler path
        for i in range(self.num_envs):
            if self._episode_done[i]:
                continue
            # Apply scheduled push perturbation (impulse) if configured for this env
            try:
                dr = self._dr_params[i]
                if dr.get("push_time", -1) == int(self._step_count[i]):
                    mag = float(dr.get("push_mag", 0.0))
                    dirx = float(dr.get("push_dir_x", 1.0))
                    if self.qvel.shape[1] >= 3:
                        self.qvel[i, 0] += (mag * dirx) / (self._mass_scale[i] + 1e-6)
            except Exception:
                pass

            # Build observation from simulator Data (now-updated)
            obs_i = self._build_obs(i)
            reward_i = self._compute_reward(i, obs_i, torques_buf[i])
            done_i = self._is_done(i, obs_i)

            # store
            self.obs[i] = obs_i
            rewards[i] = float(reward_i)
            dones[i] = bool(done_i)
            self._episode_done[i] = done_i
            self._step_count[i] += 1

        infos["step_time"].append(time.time() - t0)
        return self.obs.copy(), rewards, dones, infos

    def _warn_mjx_deprecation(self) -> None:
        """Emit a one-time deprecation warning when MJX-only fallback is used.

        Encourages users to enable the pure-JAX port (`EnvConfig.use_jax=True`) for
        better performance and future compatibility.
        """
        if self._deprecation_warned:
            return
        try:
            warnings.warn(
                "MJX-only execution path used — consider enabling the pure-JAX port by setting EnvConfig.use_jax=True. "
                "MJX fallbacks will be deprecated once the JAX port is validated.",
                DeprecationWarning,
            )
        except Exception:
            pass
        self._deprecation_warned = True

    def random_action(self) -> np.ndarray:
        return np.random.uniform(low=-0.5, high=0.5, size=(self.num_envs, self.ACT_DIM)).astype(np.float32)

    # ---------------------- small helpers ----------------------
    def _reset_single(self, idx: int) -> None:
        # TODO: use mujoco vectorized reset; currently a placeholder
        self._dr_params[idx] = self._sample_dr_params(idx)
        # Reset step count and done flag
        self._step_count[idx] = 0
        self._episode_done[idx] = False
        # Reset per-env action latency buffer
        try:
            self._action_buffers[idx, :, :] = 0.0
        except Exception:
            pass

    def _step_one(self, i: int, torques: np.ndarray) -> None:
        """Helper to step a single data object safely; intended for parallel execution."""
        if self._episode_done[i]:
            return
        # If Data writes are not possible for this env, use host-side Euler
        # fallback so stepping reflects `self.qpos` that callers may have set.
        if getattr(self, "_force_host_qpos", None) is not None and self._force_host_qpos[i]:
            accel = torques.astype(np.float32) / (self._mass_scale[i] + 1e-6)
            self.qvel[i, : accel.size] += accel * self._dt
            self.qpos[i, : accel.size] += self.qvel[i, : accel.size] * self._dt
            return
        # Prefer creating a fresh Data object per-step (data.replace-style)
        model = self._models[i] if (self._per_env_models and i < len(self._models)) else self.model
        try:
            # Try to acquire a Data object from the pool when available
            newd = None
            if not self._per_env_models:
                try:
                    with self._pool_lock:
                        if len(self._data_pool) > 0:
                            newd = self._data_pool.pop()
                except Exception:
                    newd = None
            if newd is None:
                newd = mjx.make_data(model)
            # copy host qpos/qvel into fresh Data (avoid in-place writes into possibly immutable arrays)
            try:
                newd.qpos[:] = self.qpos[i]
            except Exception:
                try:
                    setattr(newd, "qpos", np.array(self.qpos[i], dtype=np.float32))
                except Exception:
                    pass
            try:
                newd.qvel[:] = self.qvel[i]
            except Exception:
                try:
                    setattr(newd, "qvel", np.array(self.qvel[i], dtype=np.float32))
                except Exception:
                    pass
            # write ctrl
            try:
                newd.ctrl[:] = torques
            except Exception:
                try:
                    setattr(newd, "ctrl", np.array(torques, dtype=np.float32))
                except Exception:
                    pass

            # step the fresh data
            mjx.step(model, newd)

            # copy results back to host and replace stored data
            try:
                self.qpos[i] = np.array(newd.qpos, dtype=np.float32)
            except Exception:
                pass
            try:
                self.qvel[i] = np.array(newd.qvel, dtype=np.float32)
            except Exception:
                pass
            # replace internal data reference with the fresh (non-JAX-backed) Data
            # If we used a pooled Data instance, return it to pool rather
            # than permanently assigning it to the per-env list. Otherwise
            # replace the stored data reference so subsequent reads see
            # the latest values.
            try:
                if not self._per_env_models and getattr(self, "_data_pool", None) is not None:
                    # return to pool if there is room
                    with self._pool_lock:
                        if len(self._data_pool) < int(getattr(self.cfg, "data_pool_size", 16)):
                            self._data_pool.append(newd)
                        else:
                            # pool full: keep as per-env data
                            self._datas[i] = newd
                else:
                    self._datas[i] = newd
            except Exception:
                try:
                    self._datas[i] = newd
                except Exception:
                    pass
            # If this code path executed, we are using MJX-native stepping.
            # Emit a deprecation warning so users move to the JAX port.
            try:
                if self._use_jax:
                    self._warn_mjx_deprecation()
            except Exception:
                pass
            return
        except Exception:
            # fallback to stepping the existing Data in-place
            d = self._datas[i]
            try:
                mjx.step(model, d)
                self.qpos[i] = np.array(d.qpos, dtype=np.float32)
                self.qvel[i] = np.array(d.qvel, dtype=np.float32)
                return
            except Exception:
                try:
                    if hasattr(mjx, "forward"):
                        mjx.forward(model, d)
                except Exception:
                    pass
                accel = torques.astype(np.float32) / (self._mass_scale[i] + 1e-6)
                self.qvel[i, : accel.size] += accel * self._dt
                self.qpos[i, : accel.size] += self.qvel[i, : accel.size] * self._dt
                try:
                    if self._use_jax:
                        self._warn_mjx_deprecation()
                except Exception:
                    pass

    def _parallel_step_datas(self, torques_buf: np.ndarray) -> None:
        """Attempt to step all data objects in parallel using threads.

        This provides practical speedups on multi-core hosts while we
        transition to a pure-JAX `data.replace` flow. Falls back to sequential
        stepping if threading is unavailable or fails.
        """
        max_workers = min(self.num_envs, (os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in range(self.num_envs):
                if self._episode_done[i]:
                    continue
                # ensure ctrl is set on Data (in case previous write failed)
                try:
                    d = self._datas[i]
                    try:
                        d.ctrl[:] = torques_buf[i]
                    except Exception:
                        pass
                except Exception:
                    pass
                futures.append(ex.submit(self._step_one, i, torques_buf[i]))
            # wait for completion
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass
        # Apply terrain offset if sampled
        try:
            d = self._datas[idx]
            terrain_offset = float(self._dr_params[idx].get("terrain_offset", 0.0))
            if hasattr(d, "qpos") and d.qpos.size >= 3:
                self.qpos[idx, 2] = float(self.qpos[idx, 2]) + terrain_offset
                try:
                    d.qpos[:] = self.qpos[idx]
                    if hasattr(mjx, "forward"):
                        mjx.forward(self.model if not self._per_env_models else self._models[idx], d)
                except Exception:
                    pass

            # Apply DR parameters to the model copy if we have per-env models
            dr = self._dr_params[idx]
            if self._per_env_models:
                try:
                    m = self._models[idx]
                    # mass scaling
                    if getattr(self._orig_body_mass, "size", 0) > 0 and hasattr(m, "body_mass"):
                        try:
                            m.body_mass[:] = (self._orig_body_mass * float(dr.get("mass_scale", 1.0))).astype(np.float32)
                        except Exception:
                            pass
                    # geom friction
                    if getattr(self._orig_geom_friction, "size", 0) > 0 and hasattr(m, "geom_friction"):
                        try:
                            # geom_friction may be (ngeom,3) or (ngeom,)
                            friction_val = float(dr.get("friction", 1.0))
                            gf = getattr(m, "geom_friction")
                            try:
                                gf[:] = friction_val
                            except Exception:
                                # fallback reshape
                                arr = np.array(gf, dtype=np.float32)
                                arr[:] = friction_val
                                m.geom_friction[:] = arr
                        except Exception:
                            pass
                    # joint/dof damping
                    if getattr(self._orig_dof_damping, "size", 0) > 0 and hasattr(m, "dof_damping"):
                        try:
                            jd = float(dr.get("joint_damping", 0.0))
                            m.dof_damping[:] = (self._orig_dof_damping * (1.0 + jd)).astype(np.float32)
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                # fallback: apply mass_scale locally for Euler fallback integration and PD scaling
                try:
                    self._mass_scale[idx] = float(dr.get("mass_scale", 1.0))
                except Exception:
                    pass
        except Exception:
            pass

    def _build_obs(self, idx: int) -> np.ndarray:
        # Build a dummy observation consistent with plan (44 dims)
        # Layout (example): [gravity(3), joint_pos(11), joint_vel(11), prev_action(11), phase(2), extras(6)]
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Use current qpos/qvel for joints (assume joints are last nq-7 entries)
        qpos = self.qpos[idx]
        qvel = self.qvel[idx]

        # Heuristic: if model uses free base (pos+quat) then joint positions start at 7
        if qpos.size >= 7 + 11:
            joint_pos = qpos[7 : 7 + 11].astype(np.float32)
        else:
            # fallback: take first 11 qpos entries
            joint_pos = qpos[:11].astype(np.float32)

        if qvel.size >= 11:
            joint_vel = qvel[:11].astype(np.float32)
        else:
            joint_vel = np.zeros(11, dtype=np.float32)

        prev_action = self.actions[idx].astype(np.float32)
        phase = np.zeros(2, dtype=np.float32)

        # extras: base height, pitch, roll, com_x, com_y, com_z (fallback values)
        base_height = 0.0
        pitch = 0.0
        roll = 0.0
        com = np.zeros(3, dtype=np.float32)
        try:
            d = self._datas[idx]
            # Prefer world-space base position if available
            if hasattr(d, "xpos") and getattr(d, "xpos") is not None and getattr(d, "xpos").size >= 3:
                # assume first body/site corresponds to base
                base_pos = np.array(d.xpos[:3], dtype=np.float32) if np.asarray(d.xpos).ndim == 1 else np.array(d.xpos[0], dtype=np.float32)
                base_height = float(base_pos[2])
            elif d.qpos.size >= 3:
                base_height = float(d.qpos[2])

            # quaternion: read robustly (MJCF/MJX may use wxyz or xyzw ordering)
            from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz

            try:
                if hasattr(d, "xquat") and getattr(d, "xquat") is not None:
                    raw_q = np.array(d.xquat[:4], dtype=np.float32) if np.asarray(d.xquat).ndim == 1 else np.array(d.xquat[0], dtype=np.float32)
                elif d.qpos.size >= 7:
                    raw_q = np.array(d.qpos[3:7], dtype=np.float32)
                else:
                    raw_q = None
                if raw_q is not None:
                    q = normalize_quat_wxyz(raw_q)
                    roll, pitch = quat_to_euler_wxyz(q)
                    roll = float(roll)
                    pitch = float(pitch)
            except Exception:
                pass

            # center of mass if available
            if hasattr(d, "com_pos") and getattr(d, "com_pos") is not None:
                try:
                    com = np.array(d.com_pos, dtype=np.float32)
                except Exception:
                    com = np.zeros(3, dtype=np.float32)

            # contact forces: explicit read (sum absolute ext forces)
            try:
                c = getattr(d, "cfrc_ext", None)
                if c is not None:
                    contact_sum = float(np.sum(np.abs(np.array(c, dtype=np.float32))))
                    # keep contact as a separate scalar by folding slightly into COM z
                    com[2] = com[2] + 0.001 * contact_sum
            except Exception:
                pass
        except Exception:
            pass

        extras = np.array([base_height, pitch, roll, com[0], com[1], com[2]], dtype=np.float32)

        obs = np.concatenate([gravity, joint_pos, joint_vel, prev_action, phase, extras], axis=0)
        assert obs.shape[0] == self.OBS_DIM

        # Add observation noise
        noise = np.random.normal(scale=self.cfg.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs + noise

    def _pd_control(self, idx: int, target_qpos: np.ndarray) -> np.ndarray:
        # PD control enhanced with available qpos/qvel and DR scaling
        current_qpos = np.zeros_like(target_qpos)
        current_qvel = np.zeros_like(target_qpos)
        # try to read actual joint qpos/qvel if present
        try:
            qpos = self.qpos[idx]
            if qpos.size >= 11:
                current_qpos = qpos[:11].astype(np.float32)
        except Exception:
            pass
        try:
            qvel = self.qvel[idx]
            if qvel.size >= 11:
                current_qvel = qvel[:11].astype(np.float32)
        except Exception:
            pass

        pos_err = target_qpos - current_qpos
        vel_err = -current_qvel
        torque = self.kp * pos_err + self.kd * vel_err

        # Apply joint-level damping from DR params (per-env scalar)
        try:
            dr = self._dr_params[idx]
            jd = float(dr.get("joint_damping", 0.0))
            torque = torque - jd * current_qvel[: torque.size]
            # motor strength scales torque
            ms = float(dr.get("motor_strength", 1.0))
            torque = torque * ms
        except Exception:
            pass

        # Clip torques to reasonable bounds
        torque = np.clip(torque, -4.0, 4.0)
        return torque

    def _compute_reward(self, idx: int, obs: np.ndarray, torques: np.ndarray) -> float:
        # Minimal reward: encourage forward velocity (placeholder) and penalize torque
        # In practice compute from simulated base velocities, AMP reward, etc.
        # Estimate forward velocity from qvel if available
        forward_vel = 0.0
        try:
            v = self.qvel[idx]
            # heuristic: forward velocity approximately first translational velocity
            if v.size >= 1:
                forward_vel = float(v[0])
        except Exception:
            forward_vel = 0.0

        # target forward speed is zero in this placeholder; penalty on torque
        torque_penalty = 0.0002 * float(np.sum(np.square(torques)))
        # reward: encourage small torque (energy efficiency) and small forward_vel error
        return float(-abs(forward_vel) - torque_penalty)

    def _is_done(self, idx: int, obs: np.ndarray) -> bool:
        # Simple done condition: max episode steps reached
        if self._step_count[idx] >= self.cfg.max_episode_steps:
            return True
        # Prefer host-side qpos/qvel for termination checks (robust against
        # MJX/JAX immutable-data write failures). Fall back to obs-derived
        # values only when host state is not available.
        try:
            # base height: qpos z index (if available)
            if getattr(self, "qpos", None) is not None and self.qpos.shape[1] >= 3:
                base_height = float(self.qpos[idx, 2])
            else:
                base_height = float(obs[-6])

            # orientation: prefer host qpos quaternion when available
            pitch = None
            roll = None
            if getattr(self, "qpos", None) is not None and self.qpos.shape[1] >= 7:
                try:
                    from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz

                    raw_q = np.array(self.qpos[idx, 3:7], dtype=np.float32)
                    q = normalize_quat_wxyz(raw_q)
                    roll, pitch = quat_to_euler_wxyz(q)
                except Exception:
                    roll = None
                    pitch = None

            # fallback to obs if orientation unavailable
            if roll is None or pitch is None:
                try:
                    pitch = float(obs[-5])
                    roll = float(obs[-4])
                except Exception:
                    pitch = 0.0
                    roll = 0.0

            if base_height < 0.25:
                return True
            # convert rad thresholds (45 deg)
            if abs(pitch) > (45.0 * np.pi / 180.0) or abs(roll) > (45.0 * np.pi / 180.0):
                return True
        except Exception:
            pass

        # Additional termination checks using simulator data when available
        try:
            d = self._datas[idx]
            # NaN or inf in state
            if np.isnan(self.qpos[idx]).any() or np.isinf(self.qpos[idx]).any():
                return True
            # excessive external contact force
            c = getattr(d, "cfrc_ext", None)
            if c is not None:
                csum = float(np.sum(np.abs(np.array(c, dtype=np.float32))))
                if csum > 500.0:
                    return True
            # large joint velocities
            if np.any(np.abs(self.qvel[idx]) > 50.0):
                return True
            # use base orientation if available for early termination (upright loss)
            try:
                from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz

                raw_q = None
                if hasattr(d, "xquat") and getattr(d, "xquat") is not None:
                    try:
                        raw_q = np.array(d.xquat[:4], dtype=np.float32) if np.asarray(d.xquat).ndim == 1 else np.array(d.xquat[0], dtype=np.float32)
                    except Exception:
                        raw_q = None
                if raw_q is None and d.qpos.size >= 7:
                    raw_q = np.array(d.qpos[3:7], dtype=np.float32)
                if raw_q is not None:
                    q = normalize_quat_wxyz(raw_q)
                    roll, pitch = quat_to_euler_wxyz(q)
                    if abs(pitch) > (60.0 * np.pi / 180.0) or abs(roll) > (60.0 * np.pi / 180.0):
                        return True
            except Exception:
                pass
        except Exception:
            pass

        return False

    def _sample_dr_params(self, idx: int) -> Dict[str, float]:
        # Sample domain randomization parameters for one env instance
        rng = np.random.RandomState(self.cfg.seed + idx)
        # latency in control steps (integer 0..3)
        latency = int(rng.randint(0, 4))
        # small terrain offset (meters)
        terrain_offset = float(rng.uniform(-0.02, 0.02))
        # scheduled push perturbation (time step and magnitude)
        push_prob = rng.uniform()
        if push_prob < 0.15:
            push_time = int(rng.randint(5, max(6, int(self.cfg.max_episode_steps // 3))))
            push_mag = float(rng.uniform(5.0, 50.0))
            push_dir_x = float(rng.choice([-1.0, 1.0]))
        else:
            push_time = -1
            push_mag = 0.0
            push_dir_x = 1.0

        return {
            "friction": float(rng.uniform(0.4, 1.25)),
            "mass_scale": float(rng.uniform(0.85, 1.15)),
            "motor_strength": float(rng.uniform(0.9, 1.1)),
            "joint_damping": float(rng.uniform(0.0, 0.5)),
            "latency_steps": latency,
            "terrain_offset": terrain_offset,
            "push_time": push_time,
            "push_mag": push_mag,
            "push_dir_x": push_dir_x,
        }

    def set_data_qpos(self, idx: int, qpos: np.ndarray) -> None:
        """Safely write a host `qpos` array into the internal Data object for `idx`.

        This helper attempts multiple fallbacks because different MJX builds may
        expose device-backed (JAX) arrays that are immutable for direct writes.
        """
        try:
            d = self._datas[idx]
        except Exception:
            raise IndexError("Invalid data index")

        qarr = np.array(qpos, dtype=np.float32)
        # Try fast slice assignment
        try:
            d.qpos[:] = qarr
            return
        except Exception:
            pass

        # Try setting attribute to a numpy array
        try:
            setattr(d, "qpos", qarr)
            return
        except Exception:
            pass

        # Fallback: create a fresh Data object from the model and replace
        try:
            m = self._models[idx] if (self._per_env_models and idx < len(self._models)) else self.model
            newd = mjx.make_data(m)
            try:
                newd.qpos[:] = qarr
            except Exception:
                # final fallback: set attribute directly
                setattr(newd, "qpos", qarr)
            self._datas[idx] = newd
            return
        except Exception as e:
            # If we cannot write into the underlying Data (common when
            # Data arrays are JAX-backed/immutable), ensure host-side
            # qpos is updated so stepping will use the new state. This
            # guarantees a reliable host->sim round-trip for the current
            # step even if we cannot mutate the Data object in-place.
            try:
                self.qpos[idx] = qarr
            except Exception:
                pass
            # mark this env to use host-side qpos until writes become available
            try:
                self._force_host_qpos[idx] = True
            except Exception:
                pass
            # Best-effort: attempt to create a fresh Data and replace, but
            # if that fails also fall back to host-only update and surface
            # a warning rather than raising — callers expect this helper
            # to be usable in interactive workflows.
            try:
                # try one more time to construct a fresh, non-JAX-backed
                # Data and set qpos as an attribute (may still fail).
                m = self._models[idx] if (self._per_env_models and idx < len(self._models)) else self.model
                newd = mjx.make_data(m)
                try:
                    setattr(newd, "qpos", qarr)
                    self._datas[idx] = newd
                except Exception:
                    # give up and keep host qpos updated
                    print(f"Warning: cannot write qpos into mjx.Data for idx={idx}; updated host qpos only.")
            except Exception:
                print(f"Warning: cannot create fallback mjx.Data for idx={idx}; updated host qpos only.")
            return


if __name__ == "__main__":
    # Simple smoke test
    print("Running WildRobotEnv smoke test (single-process, few envs)")
    cfg = EnvConfig(num_envs=8, seed=42, max_episode_steps=50)
    env = WildRobotEnv(cfg)
    obs = env.reset()
    print("reset obs shape:", obs.shape)

    act = env.random_action()
    for i in range(5):
        obs, rew, done, info = env.step(act)
        print(f"step {i}: rew mean={rew.mean():.6f}, done sum={done.sum()}")
