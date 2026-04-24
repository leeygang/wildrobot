"""WildRobot training environment — v0.20.1 v3-only rewrite.

Single-file ``mjx_env.MjxEnv`` for the v0.20.1 PPO smoke
(``training/configs/ppo_walking_v0201_smoke.yaml``).  The actor sees
the offline ``ReferenceLibrary`` window plus a 3-frame past-proprio
stack (``wr_obs_v6_offline_ref_history`` layout) and emits a bounded
residual on top of the library's ``q_ref``; the env composes
``q_target = q_ref + clip(action) * scale_per_joint`` (absolute
mode), runs MJX physics, and emits the imitation-dominant reward.

Design refs:
  - ``training/docs/v0201_env_wiring.md``        env design (§3-§5, §10)
  - ``training/docs/v0201_env_rewrite_audit.md`` reuse audit (§B-§D)
  - ``training/docs/walking_training.md``        v0.20.1 smoke contract

Notable v0.19.5c subsystems intentionally absent (audit §C):
  - parametric reference (v1/v2), the IK / startup / brake / support
    helpers that fed it, and the v0.19.5c reward family
  - M3 step FSM + MPC standing controller + teacher targets +
    recovery metrics
  - support-posture B precomputation (v3 starts from the MJCF keyframe)
  - the M2 base+residual mix path

The placeholder reward is ``reward_weights.alive`` per step until
termination.  Task #49 lands the imitation-dominant reward family on
top of this skeleton.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from flax import struct
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from policy_contract.calib import JaxCalibOps
from policy_contract.jax import frames as jax_frames
from policy_contract.jax.action import postprocess_action
from policy_contract.jax.obs import build_observation
from policy_contract.jax.state import PolicyState
from policy_contract.spec_builder import build_policy_spec

from training.cal.cal import ControlAbstractionLayer
from training.cal.types import CoordinateFrame
from training.configs.training_config import (
    TrainingConfig,
    get_robot_config,
    load_robot_config,
)
from training.core.experiment_tracking import get_initial_env_metrics_jax
from training.core.metrics_registry import METRICS_VEC_KEY, build_metrics_vec
from training.envs.disturbance import (
    DisturbanceSchedule,
    apply_push,
    sample_push_schedule,
)
from training.envs.domain_randomize import (
    nominal_domain_rand_params,
    sample_domain_rand_params,
)
from training.envs.env_info import (
    IMU_HIST_LEN,
    IMU_MAX_LATENCY,
    PRIVILEGED_OBS_DIM,
    PROPRIO_HISTORY_FRAMES,
    WR_INFO_KEY,
    WildRobotInfo,
)
from training.policy_spec_utils import clamp_home_ctrl, get_home_ctrl_from_mj_model
from training.sim_adapter.mjx_signals import MjxSignalsAdapter
from training.utils.ctrl_order import CtrlOrderMapper


__all__ = [
    "WildRobotEnv",
    "WildRobotEnvState",
    "get_assets",
]


# =============================================================================
# Module-level helpers
# =============================================================================


def get_assets(root_path: Path) -> Dict[str, bytes]:
    """Load all assets from the WildRobot assets directory.

    Walks ``root_path`` for XML files plus ``root_path/assets/`` for STL
    meshes (recursively, to support ``convex_decomposition/*.stl``
    references emitted by the post-processor).  Returns the dict
    ``mujoco.MjModel.from_xml_string`` consumes.
    """
    assets: Dict[str, bytes] = {}
    mjx_env.update_assets(assets, root_path, "*.xml")
    meshes_path = root_path / "assets"
    if meshes_path.exists():
        mjx_env.update_assets(assets, meshes_path, "*.stl")
        for stl_path in meshes_path.rglob("*.stl"):
            rel = stl_path.relative_to(meshes_path).as_posix()
            assets[rel] = stl_path.read_bytes()
    return assets


def _apply_imu_noise_and_delay(signals, rng, cfg, prev_hist_quat, prev_hist_gyro):
    """Inject Gaussian IMU noise + N-step latency on the policy signals.

    The delay buffer is fixed-size (``IMU_HIST_LEN`` = ``IMU_MAX_LATENCY +
    1``); shapes must match the ``WildRobotInfo`` schema.  Returns the
    overridden signals plus the new buffers and the next RNG.
    """
    imu_std = float(cfg.env.imu_gyro_noise_std)
    quat_deg = float(cfg.env.imu_quat_noise_deg)
    latency = int(cfg.env.imu_latency_steps)
    hist_len = IMU_HIST_LEN

    curr_quat = signals.quat_xyzw
    curr_gyro = signals.gyro_rad_s

    use_noise = (imu_std != 0.0) or (quat_deg != 0.0)
    rng_out = rng

    if use_noise:
        rng_out, rng_noise, rng_a, rng_axis = jax.random.split(rng, 4)
        gyro_noise = jax.random.normal(rng_noise, shape=curr_gyro.shape) * imu_std
        noisy_gyro = curr_gyro + gyro_noise

        angle_std = quat_deg * (jp.pi / 180.0)
        angle = jax.random.normal(rng_a, shape=()) * angle_std
        axis = jax.random.normal(rng_axis, shape=(3,))
        noise_quat = jax_frames.axis_angle_to_quat(axis, angle)
        noisy_quat = jax_frames.quat_mul(noise_quat, curr_quat)
        noisy_quat = noisy_quat / (jp.linalg.norm(noisy_quat) + 1e-12)
    else:
        noisy_gyro = curr_gyro
        noisy_quat = curr_quat

    if prev_hist_quat is None:
        new_q_hist = jp.repeat(noisy_quat[None, :], hist_len, axis=0)
        new_g_hist = jp.repeat(noisy_gyro[None, :], hist_len, axis=0)
    else:
        q_head = noisy_quat[None, :]
        g_head = noisy_gyro[None, :]
        q_rest = prev_hist_quat[: hist_len - 1]
        g_rest = prev_hist_gyro[: hist_len - 1]
        new_q_hist = jp.concatenate([q_head, q_rest], axis=0)
        new_g_hist = jp.concatenate([g_head, g_rest], axis=0)

    latency_clipped = int(np.clip(latency, 0, IMU_MAX_LATENCY))
    delayed_quat = new_q_hist[latency_clipped]
    delayed_gyro = new_g_hist[latency_clipped]

    signals_override = signals.replace(
        quat_xyzw=delayed_quat.astype(jp.float32),
        gyro_rad_s=delayed_gyro.astype(jp.float32),
    )
    return signals_override, new_q_hist, new_g_hist, rng_out


# =============================================================================
# State
# =============================================================================


@struct.dataclass
class WildRobotEnvState(mjx_env.State):
    """Brax-compatible env state.  Adds ``pipeline_state`` (the
    underlying ``mjx.Data`` aliased for Brax wrappers) and ``rng`` (the
    per-env RNG carried across steps)."""

    pipeline_state: Any = None
    rng: jp.ndarray = None


# =============================================================================
# Environment
# =============================================================================


class WildRobotEnv(mjx_env.MjxEnv):
    """Single-command, v3-only PPO env for the v0.20.1 smoke."""

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        config: TrainingConfig,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        # MjxEnv parent expects a ConfigDict with .ctrl_dt / .sim_dt.
        parent_config = config_dict.ConfigDict()
        parent_config.ctrl_dt = config.env.ctrl_dt
        parent_config.sim_dt = config.env.sim_dt
        super().__init__(parent_config, config_overrides)

        self._config = config

        if int(self._config.env.imu_max_latency_steps) != int(IMU_MAX_LATENCY):
            raise ValueError(
                f"env.imu_max_latency_steps ({self._config.env.imu_max_latency_steps}) "
                f"must equal IMU_MAX_LATENCY ({IMU_MAX_LATENCY}) compiled into env_info.py."
            )
        if int(getattr(self._config.env, "action_delay_steps", 0)) not in (0, 1):
            raise ValueError("Only action_delay_steps in {0, 1} is supported.")
        self._action_delay_enabled = (
            int(getattr(self._config.env, "action_delay_steps", 0)) == 1
        )

        loc_ref_version = str(getattr(self._config.env, "loc_ref_version", "")).lower()
        if loc_ref_version != "v3_offline_library":
            raise ValueError(
                "v0.20.1 WildRobotEnv requires env.loc_ref_version='v3_offline_library'. "
                f"Got '{loc_ref_version}'.  v1/v2 paths were removed at v0.20.1."
            )

        layout_id = str(self._config.env.actor_obs_layout_id)
        if layout_id != "wr_obs_v6_offline_ref_history":
            raise ValueError(
                "v0.20.1 WildRobotEnv requires env.actor_obs_layout_id="
                "'wr_obs_v6_offline_ref_history'.  v5 was deprecated along "
                "with the high-confidence prep (proprio history is now "
                "always wired); older layouts depended on v1/v2 reference "
                "state and were removed by the v3 rewrite."
            )

        if not self._config.env.scene_xml_path:
            raise ValueError("env.scene_xml_path is required.")
        self._model_path = Path(self._config.env.scene_xml_path)

        self._load_model()
        self._init_residual_scale()
        self._init_offline_service()

        print("WildRobotEnv (v0.20.1 v3-only) initialized:")
        print(f"  Actuators:    {self._mj_model.nu}")
        print(f"  obs_dim:      {self._policy_spec.model.obs_dim}")
        print(f"  ctrl_dt:      {self.dt}s")
        print(f"  sim_dt:       {self.sim_dt}s")
        print(f"  ref n_steps:  {self._offline_service.n_steps} "
              f"(vx={self._config.env.loc_ref_offline_command_vx})")

    # --------------------------------------------------------------- _load_model

    def _load_model(self) -> None:
        project_root = Path(__file__).parent.parent.parent
        xml_path = project_root / self._model_path
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {xml_path} (model_path={self._model_path})"
            )
        root_path = xml_path.parent

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=get_assets(root_path)
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        # Robot config: prefer the global singleton (train.py preloads it
        # before constructing the env), but fall back to loading from the
        # YAML-configured path so the env is self-contained for tests,
        # eval, and ad-hoc construction.
        try:
            self._robot_config = get_robot_config()
        except RuntimeError:
            robot_config_path = Path(self._config.env.robot_config_path)
            if not robot_config_path.is_absolute():
                robot_config_path = (
                    Path(__file__).parent.parent.parent / robot_config_path
                )
            if not robot_config_path.exists():
                raise FileNotFoundError(
                    f"robot_config not loaded and "
                    f"env.robot_config_path does not exist: {robot_config_path}"
                )
            self._robot_config = load_robot_config(robot_config_path)
        self._signals_adapter = MjxSignalsAdapter(
            self._mj_model,
            self._robot_config,
            foot_switch_threshold=self._config.env.foot_switch_threshold,
        )
        self._cal = ControlAbstractionLayer(self._mj_model, self._robot_config)

        if self._mj_model.nkey > 0:
            self._init_qpos = jp.array(self._mj_model.key_qpos[0])
        else:
            self._init_qpos = jp.array(self._mj_model.qpos0)

        self._default_joint_qpos = self._cal.get_ctrl_for_default_pose()

        home_ctrl_list = clamp_home_ctrl(
            home_ctrl=get_home_ctrl_from_mj_model(
                mj_model=self._mj_model,
                actuator_names=[
                    str(item["name"]) for item in self._robot_config.actuated_joints
                ],
            ),
            actuated_joint_specs=self._robot_config.actuated_joints,
            actuator_names=[
                str(item["name"]) for item in self._robot_config.actuated_joints
            ],
        )
        self._policy_spec = build_policy_spec(
            robot_name=self._robot_config.robot_name,
            actuated_joint_specs=self._robot_config.actuated_joints,
            action_filter_alpha=float(self._config.env.action_filter_alpha),
            layout_id=str(self._config.env.actor_obs_layout_id),
            mapping_id=str(self._config.env.action_mapping_id),
            home_ctrl_rad=home_ctrl_list,
        )
        self._actuator_name_to_index = {
            name: i for i, name in enumerate(self._policy_spec.robot.actuator_names)
        }

        # Default-pose action (used for filter init; matches ctrl at reset).
        self._default_pose_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec,
            ctrl_rad=self._default_joint_qpos,
        ).astype(jp.float32)

        # Per-joint range arrays (for residual clipping).
        self._joint_range_mins = jp.asarray(
            [float(item["range"][0]) for item in self._robot_config.actuated_joints],
            dtype=jp.float32,
        )
        self._joint_range_maxs = jp.asarray(
            [float(item["range"][1]) for item in self._robot_config.actuated_joints],
            dtype=jp.float32,
        )
        self._joint_half_spans = 0.5 * (self._joint_range_maxs - self._joint_range_mins)

        # Joint qpos / dof addrs in PolicySpec actuator order.
        actuator_qpos_addrs: List[int] = []
        actuator_dof_addrs: List[int] = []
        for name in self._policy_spec.robot.actuator_names:
            act_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = int(self._mj_model.actuator_trnid[act_id][0])
            actuator_qpos_addrs.append(int(self._mj_model.jnt_qposadr[joint_id]))
            actuator_dof_addrs.append(int(self._mj_model.jnt_dofadr[joint_id]))
        self._actuator_qpos_addrs = jp.asarray(actuator_qpos_addrs, dtype=jp.int32)
        self._actuator_dof_addrs = jp.asarray(actuator_dof_addrs, dtype=jp.int32)

        # PolicySpec → MJ ctrl-order bridge.  Use this for ALL ctrl writes.
        # Touch the JAX permutation eagerly so its lazy-init doesn't leak a
        # tracer when first accessed inside a jit'd reset (the cached
        # tracer would then escape into self._ctrl_mapper._perm_jax).
        self._ctrl_mapper = CtrlOrderMapper(
            self._mj_model, list(self._policy_spec.robot.actuator_names)
        )
        _ = self._ctrl_mapper.policy_to_mj_order_jax

        # Cached base values for domain-randomized model construction.
        self._base_geom_friction = self._mjx_model.geom_friction
        self._base_body_mass = self._mjx_model.body_mass
        self._base_actuator_gainprm = self._mjx_model.actuator_gainprm
        self._base_actuator_biasprm = self._mjx_model.actuator_biasprm
        self._base_dof_frictionloss = self._mjx_model.dof_frictionloss

        # Push body ids (sentinel [-1] when push disabled — apply_push gates on this).
        self._push_body_ids = jp.asarray([-1], dtype=jp.int32)
        if self._config.env.push_enabled:
            push_body_names = list(self._config.env.push_bodies) or [
                self._config.env.push_body
            ]
            push_body_ids: List[int] = []
            for body_name in push_body_names:
                body_id = mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                if body_id < 0:
                    raise ValueError(f"Push body '{body_name}' not found in model.")
                push_body_ids.append(int(body_id))
            self._push_body_ids = jp.asarray(push_body_ids, dtype=jp.int32)

    # --------------------------------------------------------- residual scale

    def _init_residual_scale(self) -> None:
        """Per-joint residual scale (absolute-mode rad bounds).

        The smoke runs with ``loc_ref_residual_mode = 'absolute'``: the
        scale value IS the rad clip directly.  Joints not listed in
        ``loc_ref_residual_scale_per_joint`` fall back to the scalar
        ``loc_ref_residual_scale``.  See v0201_env_wiring.md §4.1.
        """
        mode = str(getattr(self._config.env, "loc_ref_residual_mode", "absolute")).lower()
        if mode != "absolute":
            raise ValueError(
                "v0.20.1 v3 env requires env.loc_ref_residual_mode='absolute'. "
                f"Got '{mode}'.  half_span mode is the v1/v2 legacy contract."
            )
        per_joint_overrides = dict(
            getattr(self._config.env, "loc_ref_residual_scale_per_joint", {}) or {}
        )
        scalar_default = float(getattr(self._config.env, "loc_ref_residual_scale", 0.18))
        per_joint_arr = np.array(
            [
                float(per_joint_overrides.get(name, scalar_default))
                for name in self._policy_spec.robot.actuator_names
            ],
            dtype=np.float32,
        )
        self._residual_q_scale_per_joint = jp.asarray(per_joint_arr, dtype=jp.float32)

        # G5 anti-exploit hard gate (walking_training.md v0.20.1 §, line 980).
        # Pre-resolve hip-pitch + knee actuator indices in PolicySpec order so
        # the env can emit per-joint residual magnitudes each step.  Fail
        # loudly if the WildRobot v2 leg joints are missing from the policy
        # spec — that would mean the smoke can't be evaluated against G5.
        actuator_names_list = list(self._policy_spec.robot.actuator_names)
        g5_joint_names = (
            "left_hip_pitch", "right_hip_pitch",
            "left_knee_pitch", "right_knee_pitch",
        )
        missing = [n for n in g5_joint_names if n not in actuator_names_list]
        if missing:
            raise ValueError(
                f"G5 anti-exploit gate requires {g5_joint_names} in the policy spec; "
                f"missing: {missing}"
            )
        self._g5_residual_idx = jp.asarray(
            [actuator_names_list.index(n) for n in g5_joint_names], dtype=jp.int32
        )

    # -------------------------------------------------- offline reference svc

    def _init_offline_service(self) -> None:
        """Build the runtime reference service + pre-staged JAX arrays.

        Threads the dict returned by ``service.to_jax_arrays()`` through
        the JIT'd step (the env stores it as a plain attribute; JAX
        treats the leaves as constants and folds the indexing).
        """
        from control.references.runtime_reference_service import (
            RuntimeReferenceService,
        )

        offline_path = getattr(self._config.env, "loc_ref_offline_library_path", None)
        offline_vx = float(getattr(self._config.env, "loc_ref_offline_command_vx", 0.15))

        if offline_path:
            from control.references.reference_library import ReferenceLibrary
            lib = ReferenceLibrary.load(offline_path)
        else:
            from control.zmp.zmp_walk import ZMPWalkGenerator
            lib = ZMPWalkGenerator().build_library()

        traj = lib.lookup(offline_vx)
        self._offline_service = RuntimeReferenceService(traj, n_anchor=2)
        self._offline_jax_arrays = self._offline_service.to_jax_arrays()
        self._offline_n_steps = int(self._offline_service.n_steps)

        if self._config.env.max_episode_steps > self._offline_n_steps:
            raise ValueError(
                f"max_episode_steps ({self._config.env.max_episode_steps}) "
                f"exceeds offline trajectory length ({self._offline_n_steps}). "
                "Lower max_episode_steps or extend the trajectory; the "
                "RuntimeReferenceService clamps at the terminal frame."
            )

    # ------------------------------------------------------ domain rand wiring

    def _sample_domain_rand_params(self, rng: jax.Array) -> Dict[str, jax.Array]:
        if not bool(getattr(self._config.env, "domain_randomization_enabled", False)):
            return nominal_domain_rand_params(
                num_bodies=self._mj_model.nbody,
                num_actuators=self.action_size,
            )
        return sample_domain_rand_params(
            rng,
            num_bodies=self._mj_model.nbody,
            num_actuators=self.action_size,
            friction_range=tuple(self._config.env.domain_rand_friction_range),
            mass_scale_range=tuple(self._config.env.domain_rand_mass_scale_range),
            kp_scale_range=tuple(self._config.env.domain_rand_kp_scale_range),
            frictionloss_scale_range=tuple(
                self._config.env.domain_rand_frictionloss_scale_range
            ),
            joint_offset_rad=float(self._config.env.domain_rand_joint_offset_rad),
        )

    def _get_randomized_mjx_model(self, dr_params: Dict[str, jax.Array]) -> mjx.Model:
        if not bool(getattr(self._config.env, "domain_randomization_enabled", False)):
            return self._mjx_model
        friction_scale = dr_params["friction_scale"]
        mass_scales = dr_params["mass_scales"]
        kp_scales = dr_params["kp_scales"]
        frictionloss_scales = dr_params["frictionloss_scales"]

        geom_friction = self._base_geom_friction * friction_scale
        body_mass = self._base_body_mass * mass_scales
        actuator_gainprm = self._base_actuator_gainprm.at[:, 0].set(
            self._base_actuator_gainprm[:, 0] * kp_scales
        )
        actuator_biasprm = self._base_actuator_biasprm.at[:, 1].set(
            self._base_actuator_biasprm[:, 1] * kp_scales
        )
        dof_frictionloss = self._base_dof_frictionloss.at[self._actuator_dof_addrs].set(
            self._base_dof_frictionloss[self._actuator_dof_addrs] * frictionloss_scales
        )
        return self._mjx_model.replace(
            geom_friction=geom_friction,
            body_mass=body_mass,
            actuator_gainprm=actuator_gainprm,
            actuator_biasprm=actuator_biasprm,
            dof_frictionloss=dof_frictionloss,
        )

    # ------------------------------------------------------------- ctrl write

    def _to_mj_ctrl(self, ctrl_policy_order: jax.Array) -> jax.Array:
        return self._ctrl_mapper.to_mj_jax(ctrl_policy_order)

    # ----------------------------------------------- residual action composition

    def _compose_target_q_from_residual(
        self,
        *,
        policy_action: jax.Array,
        nominal_q_ref: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Absolute-mode residual compose: ``target_q = clip(q_ref + clip(a) * scale)``.

        ``policy_action`` is the residual command in PolicySpec [-1, 1]
        space (post-filter, post-delay).  Scaling is per-joint
        (``self._residual_q_scale_per_joint``) and the result is
        clipped to the joint range.  Returns
        ``(target_q_rad, residual_delta_q_rad)``.

        G6 contract: with ``policy_action == 0`` (and the filter's
        ``prev_action == 0``, set up by ``_make_initial_state``),
        ``residual_delta_q == 0`` and ``target_q == q_ref`` exactly,
        independent of ``action_filter_alpha``.  The legacy path
        filtered the composed target in policy-action space, so iter-0
        bare-q_ref replay only held when ``alpha == 0``.
        """
        residual_delta_q = (
            jp.clip(jp.asarray(policy_action, dtype=jp.float32), -1.0, 1.0)
            * self._residual_q_scale_per_joint
        )
        target_q = jp.clip(
            jp.asarray(nominal_q_ref, dtype=jp.float32) + residual_delta_q,
            self._joint_range_mins,
            self._joint_range_maxs,
        ).astype(jp.float32)
        return target_q, residual_delta_q

    # --------------------------------------------------- offline window helpers

    def _lookup_offline_window(self, step_idx: jax.Array) -> Dict[str, jax.Array]:
        """Pure JAX lookup into the pre-staged service arrays."""
        return self._offline_service.lookup_jax(step_idx, self._offline_jax_arrays)

    def _v4_compat_channels_from_window(
        self, win: Dict[str, jax.Array], prev_history: Optional[jax.Array]
    ) -> Dict[str, jax.Array]:
        """Populate the v4 ``loc_ref_*`` channel slots from the offline
        window so the v5 layout (which reuses v4's channels) stays
        well-defined.  Foothold = swing-side foot; pelvis_targets =
        ``[height, roll(=0), pitch(=0)]``; history rolls
        ``[phase_sin, phase_cos]`` from the previous step."""
        phase_sin = win["phase_sin"]
        phase_cos = win["phase_cos"]
        phase_sin_cos = jp.stack([phase_sin, phase_cos]).astype(jp.float32)

        stance_id = win["stance_foot_id"].astype(jp.float32)
        # Swing-side foot pos / vel.  stance_id: 0=left → swing right; 1=right → swing left.
        is_left_stance = win["stance_foot_id"] == 0
        next_foothold_xyz = jp.where(
            is_left_stance, win["right_foot_pos"], win["left_foot_pos"]
        )
        swing_pos_xyz = next_foothold_xyz
        swing_vel_xyz = jp.where(
            is_left_stance, win["right_foot_vel"], win["left_foot_vel"]
        )

        # next_foothold is documented as 2D (x, y in stance frame).  v3
        # emits world-frame positions; the env passes the (x, y) pair,
        # which is fine for the smoke since the trajectory is heading-
        # aligned.  Stance-frame conversion is a Task #49 nicety.
        next_foothold_xy = next_foothold_xyz[:2].astype(jp.float32)
        swing_pos = swing_pos_xyz[:3].astype(jp.float32)
        swing_vel = swing_vel_xyz[:3].astype(jp.float32)
        pelvis_targets = jp.stack(
            [win["pelvis_pos"][2].astype(jp.float32), jp.float32(0.0), jp.float32(0.0)]
        )

        if prev_history is None:
            history = jp.tile(phase_sin_cos, 2)
        else:
            history = jp.concatenate([phase_sin_cos, prev_history[:2]]).astype(jp.float32)

        return dict(
            phase_sin_cos=phase_sin_cos,
            stance=jp.atleast_1d(stance_id).astype(jp.float32),
            next_foothold=next_foothold_xy,
            swing_pos=swing_pos,
            swing_vel=swing_vel,
            pelvis_targets=pelvis_targets.astype(jp.float32),
            history=history,
        )

    # ------------------------------------------------------------------ obs

    def _compute_proprio_bundle(
        self,
        *,
        signals,
        prev_action: jax.Array,
    ) -> jax.Array:
        """Per-frame proprio bundle for ``wr_obs_v6_offline_ref_history``.

        Channels: ``[gyro(3), foot_switches(4), joint_pos_norm(N),
        joint_vel_norm(N), prev_action(N)]``.  Total size = ``3 + 4 + 3*N``.
        """
        from policy_contract.calib import JaxCalibOps as _JaxCalibOps
        joint_pos_norm = _JaxCalibOps.normalize_joint_pos(
            spec=self._policy_spec, joint_pos_rad=signals.joint_pos_rad
        ).astype(jp.float32)
        joint_vel_norm = _JaxCalibOps.normalize_joint_vel(
            spec=self._policy_spec, joint_vel_rad_s=signals.joint_vel_rad_s
        ).astype(jp.float32)
        return jp.concatenate(
            [
                signals.gyro_rad_s.astype(jp.float32),
                signals.foot_switches.astype(jp.float32),
                joint_pos_norm,
                joint_vel_norm,
                prev_action.astype(jp.float32),
            ]
        )

    @staticmethod
    def _roll_proprio_history(
        history: jax.Array, new_bundle: jax.Array
    ) -> jax.Array:
        """Drop the oldest frame and append the new one.

        ``history`` has shape ``(PROPRIO_HISTORY_FRAMES, bundle_size)``;
        index 0 is the OLDEST, index -1 is the NEWEST.  After rolling,
        ``history[-1] == new_bundle``.
        """
        return jp.concatenate([history[1:], new_bundle[None, :]], axis=0)

    def _get_obs(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
        win: Dict[str, jax.Array],
        v4_compat: Dict[str, jax.Array],
        signals=None,
        proprio_history: Optional[jax.Array] = None,
    ) -> jax.Array:
        """v5 layout dispatch.  All v4 + v5 ``loc_ref_*`` slots are
        populated from the offline window."""
        if signals is None:
            signals = self._signals_adapter.read(data)
        policy_state = PolicyState(prev_action=action)
        # Flatten the (PROPRIO_HISTORY_FRAMES, bundle) buffer for the
        # v6 layout; v5 callers pass None and the layout branch ignores it.
        flat_history = (
            None if proprio_history is None
            else proprio_history.astype(jp.float32).reshape(-1)
        )
        return build_observation(
            spec=self._policy_spec,
            state=policy_state,
            signals=signals,
            velocity_cmd=velocity_cmd,
            loc_ref_phase_sin_cos=v4_compat["phase_sin_cos"],
            loc_ref_stance_foot=v4_compat["stance"],
            loc_ref_next_foothold=v4_compat["next_foothold"],
            loc_ref_swing_pos=v4_compat["swing_pos"],
            loc_ref_swing_vel=v4_compat["swing_vel"],
            loc_ref_pelvis_targets=v4_compat["pelvis_targets"],
            loc_ref_history=v4_compat["history"],
            loc_ref_q_ref=win["q_ref"].astype(jp.float32),
            loc_ref_pelvis_pos=win["pelvis_pos"].astype(jp.float32),
            loc_ref_pelvis_vel=win["pelvis_vel"].astype(jp.float32),
            loc_ref_left_foot_pos=win["left_foot_pos"].astype(jp.float32),
            loc_ref_right_foot_pos=win["right_foot_pos"].astype(jp.float32),
            loc_ref_left_foot_vel=win["left_foot_vel"].astype(jp.float32),
            loc_ref_right_foot_vel=win["right_foot_vel"].astype(jp.float32),
            loc_ref_contact_mask=win["contact_mask"].astype(jp.float32),
            proprio_history=flat_history,
        )

    def _get_privileged_critic_obs(
        self, data: mjx.Data, root_vel_h
    ) -> jax.Array:
        """Privileged critic obs: linear vel + angular vel + per-foot contact
        forces.  Sim-only fields the actor doesn't see (asymmetric
        actor-critic)."""
        lin = root_vel_h.linear.astype(jp.float32)
        ang = root_vel_h.angular.astype(jp.float32)
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        contacts = jp.stack([left_force, right_force]).astype(jp.float32)
        # Pad / shape to PRIVILEGED_OBS_DIM so the trainer's allocator matches.
        head = jp.concatenate([lin, ang, contacts])
        if head.shape[0] >= PRIVILEGED_OBS_DIM:
            return head[:PRIVILEGED_OBS_DIM]
        pad = jp.zeros(PRIVILEGED_OBS_DIM - head.shape[0], dtype=jp.float32)
        return jp.concatenate([head, pad])

    # ----------------------------------------------------------- reward terms

    def _compute_reward_terms(
        self,
        *,
        data: mjx.Data,
        win: Dict[str, jax.Array],
        nominal_q_ref: jax.Array,
        applied_action: jax.Array,
        prev_applied_action: jax.Array,
        forward_velocity: jax.Array,
        lin_vel_z: jax.Array,
        prev_ref_pelvis_z: jax.Array,
        velocity_cmd: jax.Array,
        left_force: jax.Array,
        right_force: jax.Array,
        root_pose,
        prev_left_foot_pos: jax.Array,
        prev_right_foot_pos: jax.Array,
        gyro_rad_s: jax.Array,
        # ToddlerBot-aligned reward inputs (Appendix A).  Pre-update
        # air-time / air-dist mirror walk_env._reward_feet_air_time /
        # _reward_feet_clearance read semantics; first_contact /
        # stance_mask are the touchdown event mask + current-step
        # contact (per-foot, [left, right]).
        feet_air_time: jax.Array,
        feet_air_dist: jax.Array,
        first_contact: jax.Array,
        stance_mask: jax.Array,
        left_foot_pos: jax.Array,
        right_foot_pos: jax.Array,
    ) -> Dict[str, jax.Array]:
        """v0.20.1 imitation-dominant reward family.

        DeepMimic-style ``r = exp(-alpha * sum_of_squares)`` for each
        tracking term; raw squared-sum penalties for the regularizers.
        Sigmas/alphas are configurable via ``reward_weights.*`` and the
        defaults are ToddlerBot-canonical.  See:

          - Peng et al. 2018 "DeepMimic" (eq. 9)
          - ``toddlerbot/locomotion/mjx_env.py:1501-1806`` — reward shapes
          - ``toddlerbot/locomotion/mjx_config.py:104-118`` — sigmas
          - ``loco-mujoco/loco_mujoco/core/reward/trajectory_based.py`` —
            MimicReward (alpha-numerator form)
          - ``mujoco_playground/_src/locomotion/g1/joystick.py:663-803``

        All terms return scalars in [0, 1] (Gaussian) or unbounded
        non-negative (regularizers) before the weight is applied.
        """
        weights = self._config.reward_weights
        # ---- ref/q_ref_track --------------------------------------------------
        q_actual = data.qpos[self._actuator_qpos_addrs]
        q_err = q_actual - nominal_q_ref
        q_err_sq_sum = jp.sum(q_err * q_err)
        r_q_track = jp.exp(-jp.float32(weights.ref_q_track_alpha) * q_err_sq_sum)
        q_track_rmse = jp.sqrt(jp.mean(q_err * q_err))

        # ---- ref/body_quat_track (geodesic angle vs identity) ----------------
        # Prior emits pelvis_rpy = zeros (yaw-stationary, no roll/pitch),
        # so the reference quat is identity [w=1, x=0, y=0, z=0].
        # angle = 2 * arccos(|qw|).  Unit-norm input from MuJoCo qpos.
        quat_wxyz = root_pose.orientation
        qw_abs = jp.abs(quat_wxyz[0])
        # Numerical guard: arccos arg must be in [0, 1].
        body_quat_angle = 2.0 * jp.arccos(jp.clip(qw_abs, 0.0, 1.0))
        r_body_quat = jp.exp(
            -jp.float32(weights.ref_body_quat_alpha) * body_quat_angle * body_quat_angle
        )

        # ---- ref/feet_pos_track_raw (diagnostic-only) ------------------------
        # Keep the legacy root-relative WORLD-frame feet tracking probe as a
        # debug signal only (not weighted into total reward).
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.WORLD
        )
        root_pos_xyz = root_pose.position
        ref_pelvis_pos = win["pelvis_pos"]
        left_rel = (left_foot_pos - root_pos_xyz) - (
            win["left_foot_pos"] - ref_pelvis_pos
        )
        right_rel = (right_foot_pos - root_pos_xyz) - (
            win["right_foot_pos"] - ref_pelvis_pos
        )
        feet_err_l2 = jp.sum(left_rel * left_rel) + jp.sum(right_rel * right_rel)
        r_feet_track_raw = jp.exp(-jp.float32(200.0) * feet_err_l2)

        # ---- torso_pos_xy (ToddlerBot parity) --------------------------------
        # Match ToddlerBot _reward_torso_pos_xy:
        # exp(-alpha * ||torso_xy - torso_xy_ref||^2).
        torso_pos_xy_err = root_pos_xyz[:2] - ref_pelvis_pos[:2]
        r_torso_pos_xy = jp.exp(
            -jp.float32(weights.torso_pos_xy_alpha)
            * jp.sum(torso_pos_xy_err * torso_pos_xy_err)
        )

        # ---- lin_vel_z (v0.20.2 TB-aligned vertical body velocity) -----------
        # Match ToddlerBot _reward_lin_vel_z (toddlerbot/locomotion/walk_env.py):
        # exp(-alpha * (lin_vel_z - ref_lin_vel_z)^2).
        # The prior's ReferenceLibrary stores positions only, so the reference
        # vertical velocity is computed via finite-diff of pelvis_pos[2] across
        # consecutive offline frames (Appendix A.3 G2 decision).  At the prior's
        # ~32 ctrl-step gait cycle, vertical pelvis bobbing is small (~1-2 cm
        # amplitude) so ref_lin_vel_z is small per-step (~0.10 m/s peak); a
        # well-balanced policy will track it tightly.
        ctrl_dt_inv = jp.float32(1.0 / self.dt)
        ref_lin_vel_z = (win["pelvis_pos"][2] - prev_ref_pelvis_z) * ctrl_dt_inv
        lin_vel_z_err = lin_vel_z - ref_lin_vel_z
        r_lin_vel_z = jp.exp(
            -jp.float32(weights.lin_vel_z_alpha) * lin_vel_z_err * lin_vel_z_err
        )

        # ---- ang_vel_xy (v0.20.2 TB-aligned body angular velocity, xy axes) --
        # Match ToddlerBot _reward_ang_vel_xy:
        # exp(-alpha * sum((ang_vel_xy - ref_ang_vel_xy)^2)).
        # The prior is yaw-stationary with no pitch/roll oscillation
        # (pelvis_rpy = 0 throughout), so ref_ang_vel_xy = (0, 0) and the
        # reward reduces to penalizing body roll/pitch rate magnitude.
        # gyro_rad_s is body-frame angular velocity: [0]=roll rate (around X),
        # [1]=pitch rate (around Y).  ang_vel_z (yaw rate) is the [2] slot
        # and rewarded separately by ang_vel_z (deferred — smoke has no yaw cmd).
        ang_vel_xy_sq_sum = (
            gyro_rad_s[0] * gyro_rad_s[0] + gyro_rad_s[1] * gyro_rad_s[1]
        )
        r_ang_vel_xy = jp.exp(
            -jp.float32(weights.ang_vel_xy_alpha) * ang_vel_xy_sq_sum
        )

        # ---- ref/contact_match (smooth Gaussian per spec G3) -----------------
        contact_thresh = self._config.env.contact_threshold_force
        left_actual = (left_force > contact_thresh).astype(jp.float32)
        right_actual = (right_force > contact_thresh).astype(jp.float32)
        left_cmd = win["contact_mask"][0].astype(jp.float32)
        right_cmd = win["contact_mask"][1].astype(jp.float32)
        sigma = jp.float32(weights.ref_contact_match_sigma)
        denom = 2.0 * sigma * sigma + 1e-8
        gauss_l = jp.exp(-(left_cmd - left_actual) ** 2 / denom)
        gauss_r = jp.exp(-(right_cmd - right_actual) ** 2 / denom)
        r_contact = 0.5 * (gauss_l + gauss_r)

        # ---- cmd/forward_velocity_track --------------------------------------
        vx_err = forward_velocity - velocity_cmd
        r_vx = jp.exp(
            -jp.float32(weights.cmd_forward_velocity_alpha) * vx_err * vx_err
        )

        # ---- regularizers (raw squared-sums; weights are negative) ----------
        delta_action = applied_action - prev_applied_action
        penalty_action_rate = jp.sum(delta_action * delta_action)
        penalty_torque = jp.sum(data.actuator_force * data.actuator_force)
        joint_vel = data.qvel[self._actuator_dof_addrs]
        penalty_joint_vel = jp.sum(joint_vel * joint_vel)

        # ---- slip + pitch_rate (M1 fail-mode hooks; default weight 0) -------
        # Always computed and logged; YAML weight 0 means no contribution to
        # the total reward.  The M1 "balance issue" branch turns these on
        # without an env edit.  See training/docs/walking_training.md
        # v0.20.1 § (M1 fail-mode decision tree) and ToddlerBot's
        # _reward_feet_slip / _reward_torso_pitch.
        ctrl_dt = jp.float32(self.dt)
        left_foot_vel_xy = (left_foot_pos[:2] - prev_left_foot_pos[:2]) / ctrl_dt
        right_foot_vel_xy = (right_foot_pos[:2] - prev_right_foot_pos[:2]) / ctrl_dt
        # Mask by stance contact (only stance-side slip matters).  ``left_actual``
        # / ``right_actual`` are the measured contact bools from the contact
        # match block above.
        penalty_slip = (
            jp.sum(left_foot_vel_xy * left_foot_vel_xy) * left_actual
            + jp.sum(right_foot_vel_xy * right_foot_vel_xy) * right_actual
        )
        # Pitch rate = gyro Y in body frame (right-handed: X fwd, Y left, Z up).
        pitch_rate = gyro_rad_s[1]
        penalty_pitch_rate = pitch_rate * pitch_rate

        # ---- ToddlerBot-aligned shaping terms (Appendix A) -------------------
        # cmd_active gates feet_air_time / feet_clearance: ToddlerBot
        # zeros these on standstill (mjx_env.py walk_env.py:303 +330)
        # so the policy doesn't pay the air-time bonus for marching in
        # place at zero command.  ``velocity_cmd`` is a scalar here; v3
        # only commands forward vx.  The Gauss-band terms (torso_pitch
        # / torso_roll / feet_distance) are NOT gated since they are
        # posture-keeping rewards independent of cmd magnitude.
        cmd_active = (jp.abs(velocity_cmd) > jp.float32(1e-6)).astype(jp.float32)

        # ``feet_air_time`` (toddlerbot/locomotion/walk_env.py:283-303).
        # Σ_per_foot (air_time * first_contact), only paid on touchdown.
        r_feet_air_time = jp.sum(feet_air_time * first_contact) * cmd_active

        # ``feet_clearance`` (walk_env.py:305-329).  Σ_per_foot peak air
        # excursion above feet_height_init at touchdown.  Same gate.
        r_feet_clearance = jp.sum(feet_air_dist * first_contact) * cmd_active

        # ``feet_distance`` (walk_env.py:331-358).  Penalize lateral
        # foot spacing outside [min_feet_y_dist, max_feet_y_dist].
        # Match ToddlerBot exactly: rotate the world-frame foot delta
        # into the FULL torso frame (not just yaw) using the inverse
        # of the root quaternion, then take |y|.  Yaw-only rotation is
        # invariant to roll/pitch only when the torso is upright; under
        # the off-nominal poses where this reward is supposed to keep
        # spacing healthy, a yaw-only frame would feed pose-dependent
        # noise into the reward.  ``rotate_vec_by_quat`` expects xyzw,
        # so convert from MuJoCo's wxyz convention and conjugate for
        # the world->body inverse rotation.
        feet_vec = left_foot_pos - right_foot_pos
        root_quat_xyzw = jp.concatenate(
            [quat_wxyz[1:], quat_wxyz[:1]]
        ).astype(jp.float32)
        root_quat_xyzw = jax_frames.normalize_quat_xyzw(root_quat_xyzw)
        # quat conjugate for the inverse: (x, y, z, w) -> (-x, -y, -z, w).
        root_quat_inv_xyzw = jp.array(
            [-root_quat_xyzw[0], -root_quat_xyzw[1], -root_quat_xyzw[2],
             root_quat_xyzw[3]],
            dtype=jp.float32,
        )
        feet_vec_torso = jax_frames.rotate_vec_by_quat(
            root_quat_inv_xyzw, feet_vec.astype(jp.float32)
        )
        feet_dist = jp.abs(feet_vec_torso[1])
        d_min_clip = jp.clip(
            feet_dist - jp.float32(self._config.env.min_feet_y_dist), max=0.0
        )
        d_max_clip = jp.clip(
            feet_dist - jp.float32(self._config.env.max_feet_y_dist), min=0.0
        )
        r_feet_distance = 0.5 * (
            jp.exp(-jp.abs(d_min_clip) * 100.0)
            + jp.exp(-jp.abs(d_max_clip) * 100.0)
        )

        # ``torso_pitch_soft`` / ``torso_roll_soft``
        # (walk_env.py:229-281).  Smooth band penalty: reward = 1 inside
        # the band, drops as exp(-100*|deviation|) outside.  Replaces
        # the v0.20.1 hard pitch/roll termination with a soft signal so
        # PPO sees a gradient before the hard limit trips.
        roll, pitch_val, _ = root_pose.euler_angles()
        pitch_min_clip = jp.clip(
            pitch_val - jp.float32(self._config.env.torso_pitch_soft_min_rad),
            max=0.0,
        )
        pitch_max_clip = jp.clip(
            pitch_val - jp.float32(self._config.env.torso_pitch_soft_max_rad),
            min=0.0,
        )
        r_torso_pitch_soft = 0.5 * (
            jp.exp(-jp.abs(pitch_min_clip) * 100.0)
            + jp.exp(-jp.abs(pitch_max_clip) * 100.0)
        )
        roll_min_clip = jp.clip(
            roll - jp.float32(self._config.env.torso_roll_soft_min_rad), max=0.0
        )
        roll_max_clip = jp.clip(
            roll - jp.float32(self._config.env.torso_roll_soft_max_rad), min=0.0
        )
        r_torso_roll_soft = 0.5 * (
            jp.exp(-jp.abs(roll_min_clip) * 100.0)
            + jp.exp(-jp.abs(roll_max_clip) * 100.0)
        )

        return dict(
            r_q_track=r_q_track,
            r_body_quat_track=r_body_quat,
            r_torso_pos_xy=r_torso_pos_xy.astype(jp.float32),
            # v0.20.2 smoke6: TB-aligned continuous phase signals.
            r_lin_vel_z=r_lin_vel_z.astype(jp.float32),
            r_ang_vel_xy=r_ang_vel_xy.astype(jp.float32),
            r_contact_match=r_contact,
            r_cmd_forward_velocity_track=r_vx,
            penalty_action_rate=penalty_action_rate,
            penalty_torque=penalty_torque,
            penalty_joint_vel=penalty_joint_vel,
            penalty_slip=penalty_slip.astype(jp.float32),
            penalty_pitch_rate=penalty_pitch_rate.astype(jp.float32),
            r_feet_air_time=r_feet_air_time.astype(jp.float32),
            r_feet_clearance=r_feet_clearance.astype(jp.float32),
            r_feet_distance=r_feet_distance.astype(jp.float32),
            r_torso_pitch_soft=r_torso_pitch_soft.astype(jp.float32),
            r_torso_roll_soft=r_torso_roll_soft.astype(jp.float32),
            # Diagnostic scalars (not weighted into the reward sum):
            q_track_rmse=q_track_rmse.astype(jp.float32),
            body_quat_err_deg=(body_quat_angle * (180.0 / jp.pi)).astype(jp.float32),
            feet_pos_err_l2=jp.sqrt(feet_err_l2).astype(jp.float32),
            feet_pos_track_raw=r_feet_track_raw.astype(jp.float32),
            torso_pos_xy_err_m=jp.sqrt(jp.sum(torso_pos_xy_err * torso_pos_xy_err)).astype(
                jp.float32
            ),
            # v0.20.2 smoke6: TB-aligned phase-signal diagnostics.
            lin_vel_z_err_m_s=jp.abs(lin_vel_z_err).astype(jp.float32),
            ang_vel_xy_err_rad_s=jp.sqrt(ang_vel_xy_sq_sum).astype(jp.float32),
            feet_distance_torso_m=feet_dist.astype(jp.float32),
        )

    def _aggregate_reward(
        self, terms: Dict[str, jax.Array], terminated: jax.Array
    ) -> Dict[str, jax.Array]:
        """Apply ``reward_weights`` and the ToddlerBot ``* dt`` scale to
        per-term values; return total + the dt-scaled per-term
        contributions used for logging.

        ``alive`` is a one-shot **negative-on-done** penalty mirroring
        ToddlerBot ``_reward_survival = -done`` at weight 10
        (toddlerbot/locomotion/mjx_env.py:1897-1914 + walk.gin
        ``RewardScales.survival = 10.0``).  PRE-fix this term paid a
        dense +alive_w bonus every surviving step, which biased PPO
        toward any long-lived behavior independently of the imitation
        objective.  Now: 0 while alive, ``-alive_w * dt`` on the
        terminating step (after the global ``* dt`` scaling below).

        ``* dt`` matches ``mjx_env.py:1048``
        (``reward = sum(reward_dict.values()) * self.dt``).  This is
        load-bearing for ToddlerBot's published weights to be in the
        right scale: WR's per-step weights here MUST be the same
        numbers as ToddlerBot's ``RewardScales.*`` for the audit in
        walking_training.md Appendix A.3 to hold.  Without ``* dt`` the
        WR effective per-step magnitudes were ~50x ToddlerBot's at
        ``ctrl_dt = 0.02`` and event-based terms (notably
        ``feet_air_time`` whose seconds-airborne values are ~0.3-0.5)
        dominated the imitation gradient.
        """
        w = self._config.reward_weights
        alive_w = jp.float32(w.alive)
        # Pre-dt weighted contributions (ToddlerBot's
        # ``state.info["rewards"]`` stores these unscaled values).
        pre_dt = dict(
            # ToddlerBot semantics: -alive_w on the terminal step, 0
            # while alive.  Matches ``_reward_survival = -done`` *
            # ``RewardScales.survival = 10`` at weight 10 in walk.gin.
            alive=-alive_w * terminated,
            ref_q_track=jp.float32(w.ref_q_track) * terms["r_q_track"],
            ref_body_quat_track=jp.float32(w.ref_body_quat_track)
            * terms["r_body_quat_track"],
            torso_pos_xy=jp.float32(w.torso_pos_xy) * terms["r_torso_pos_xy"],
            # v0.20.2 smoke6: TB-aligned continuous phase signals.
            lin_vel_z=jp.float32(w.lin_vel_z) * terms["r_lin_vel_z"],
            ang_vel_xy=jp.float32(w.ang_vel_xy) * terms["r_ang_vel_xy"],
            ref_contact_match=jp.float32(w.ref_contact_match)
            * terms["r_contact_match"],
            cmd_forward_velocity_track=jp.float32(w.cmd_forward_velocity_track)
            * terms["r_cmd_forward_velocity_track"],
            action_rate=jp.float32(w.action_rate) * terms["penalty_action_rate"],
            torque=jp.float32(w.torque) * terms["penalty_torque"],
            joint_velocity=jp.float32(w.joint_velocity) * terms["penalty_joint_vel"],
            slip=jp.float32(w.slip) * terms["penalty_slip"],
            pitch_rate=jp.float32(w.pitch_rate) * terms["penalty_pitch_rate"],
            # ToddlerBot-aligned shaping (Appendix A).  Defaults are 0
            # so existing v0.19.x / v0.20.0 configs are unaffected.
            feet_air_time=jp.float32(w.feet_air_time) * terms["r_feet_air_time"],
            feet_clearance=jp.float32(w.feet_clearance) * terms["r_feet_clearance"],
            feet_distance=jp.float32(w.feet_distance) * terms["r_feet_distance"],
            torso_pitch_soft=jp.float32(w.torso_pitch_soft)
            * terms["r_torso_pitch_soft"],
            torso_roll_soft=jp.float32(w.torso_roll_soft)
            * terms["r_torso_roll_soft"],
        )
        # Apply the * dt rescale uniformly so the per-term contributions
        # logged at reward/* are exactly what each term contributes to
        # the PPO reward.  Equivalent to ToddlerBot's
        # ``reward = sum(reward_dict.values()) * self.dt`` but pushed
        # into the per-term contributions for log fidelity.
        dt = jp.float32(self.dt)
        contrib = {k: v * dt for k, v in pre_dt.items()}
        total = sum(contrib.values())
        contrib["total"] = total
        return contrib

    # ----------------------------------------------------------- termination

    def _get_termination(
        self, data: mjx.Data, step_count
    ) -> tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]:
        root_pose = self._cal.get_root_pose(data)
        height = root_pose.height
        roll, pitch, _ = root_pose.euler_angles()

        height_too_low = height < self._config.env.min_height
        height_too_high = height > self._config.env.max_height
        height_fail = height_too_low | height_too_high
        pitch_fail = jp.abs(pitch) > self._config.env.max_pitch
        roll_fail = jp.abs(roll) > self._config.env.max_roll

        if self._config.env.use_relaxed_termination:
            terminated = height_fail
        else:
            terminated = height_fail | (pitch_fail | roll_fail)

        truncated = (step_count >= self._config.env.max_episode_steps) & ~terminated
        done = terminated | truncated

        info = {
            "term/height_low": jp.where(height_too_low, 1.0, 0.0),
            "term/height_high": jp.where(height_too_high, 1.0, 0.0),
            "term/pitch": jp.where(pitch_fail, 1.0, 0.0),
            "term/roll": jp.where(roll_fail, 1.0, 0.0),
            "term/truncated": jp.where(truncated, 1.0, 0.0),
            "term/pitch_val": pitch,
            "term/roll_val": roll,
            "term/height_val": height,
        }
        return (
            jp.where(done, 1.0, 0.0),
            jp.where(terminated, 1.0, 0.0),
            jp.where(truncated, 1.0, 0.0),
            info,
        )

    # ------------------------------------------------------------------ reset

    def reset(self, rng: jax.Array) -> WildRobotEnvState:
        """Sample velocity_cmd / push schedule / DR params; build initial
        WildRobotInfo at offline step 0."""
        rng, key_vel, key_qnoise, key_push, key_dr, key_imu, key_cmd = (
            jax.random.split(rng, 7)
        )

        velocity_cmd = self._sample_velocity_cmd(key_vel)

        dr_params = self._sample_domain_rand_params(key_dr)

        # Always start from MJCF keyframe (v3 doesn't use support-posture B).
        joint_noise = jax.random.uniform(
            key_qnoise, shape=self._default_joint_qpos.shape, minval=-0.05, maxval=0.05
        )
        joint_qpos = jp.clip(
            self._default_joint_qpos + joint_noise + dr_params["joint_offsets"],
            self._joint_range_mins,
            self._joint_range_maxs,
        )
        qpos = self._init_qpos.at[self._actuator_qpos_addrs].set(joint_qpos)

        push_schedule = sample_push_schedule(
            key_push, self._config.env, self._push_body_ids
        )

        return self._make_initial_state(
            rng=rng,
            qpos=qpos,
            velocity_cmd=velocity_cmd,
            push_schedule=push_schedule,
            dr_params=dr_params,
            imu_init_rng=key_imu,
            cmd_rng=key_cmd,
        )

    # ----------------------------------------------------- cmd resampling

    def _sample_velocity_cmd(self, rng: jax.Array) -> jax.Array:
        """ToddlerBot-style multi-command sampler (`_sample_command`,
        `walk_env.py:140-226`).  Returns a scalar forward-velocity
        command in m/s.

        Behavior:
          - draw uniform on ``[min_velocity, max_velocity]``
          - with probability ``cmd_zero_chance`` zero it (stand-still)
          - apply ``cmd_deadzone``: any |cmd| below the deadzone snaps
            to 0 (matches ToddlerBot's deadzone[0] for vx)
        ``cmd_turn_chance`` is reserved for the v0.20.4 yaw upgrade
        and is ignored on the single-axis cmd path.

        Degenerate range (min == max) collapses to a deterministic
        constant, so the v0.20.1 smoke (vx pinned at 0.15) sees no
        change in behavior even when this sampler runs.
        """
        rng, k_vel, k_zero = jax.random.split(rng, 3)
        cmd = jax.random.uniform(
            k_vel,
            shape=(),
            minval=self._config.env.min_velocity,
            maxval=self._config.env.max_velocity,
        )
        zero_chance = jp.float32(self._config.env.cmd_zero_chance)
        cmd = jp.where(
            jax.random.uniform(k_zero, shape=()) < zero_chance,
            jp.float32(0.0),
            cmd,
        )
        deadzone = jp.float32(self._config.env.cmd_deadzone)
        cmd = jp.where(jp.abs(cmd) < deadzone, jp.float32(0.0), cmd)
        return cmd.astype(jp.float32)

    def _make_initial_state(
        self,
        *,
        rng: jax.Array,
        qpos: jax.Array,
        velocity_cmd: jax.Array,
        push_schedule: DisturbanceSchedule,
        dr_params: Dict[str, jax.Array],
        imu_init_rng: jax.Array,
        cmd_rng: jax.Array,
    ) -> WildRobotEnvState:
        qvel = jp.zeros(self._mj_model.nv)
        randomized_model = self._get_randomized_mjx_model(dr_params)

        # Seed ctrl with the offline trajectory's frame-0 q_ref.  Under
        # the residual-only filter contract (G6) the policy's residual
        # command starts at zero, so the composed target on iter-0 is
        # exactly q_ref0 — matching the ctrl we set here.
        win0 = self._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
        q_ref0 = win0["q_ref"].astype(jp.float32)
        ctrl_init = q_ref0
        data = mjx.make_data(randomized_model)
        data = data.replace(
            qpos=qpos, qvel=qvel, ctrl=self._to_mj_ctrl(ctrl_init)
        )
        data = mjx.forward(randomized_model, data)

        # Filter / pending-action init: zero in policy [-1, 1] space.
        # The filter operates on the residual; iter-0 policy hasn't
        # acted, so prev_residual_command = 0.  This is what makes G6
        # hold: filtered_action = α·0 + (1-α)·action stays 0 for any
        # alpha when action == 0.  The legacy default_action was
        # ctrl_to_policy_action(q_ref0); under the new compose path
        # that warm-start is unnecessary (the filter is already at
        # the residual's natural zero).
        default_action = jp.zeros(self.action_size, dtype=jp.float32)

        signals_raw = self._signals_adapter.read(data)
        signals_override, imu_quat_hist, imu_gyro_hist, _ = _apply_imu_noise_and_delay(
            signals_raw, imu_init_rng, self._config, None, None
        )

        root_pose = self._cal.get_root_pose(data)
        root_vel_h = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.WORLD
        )
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        foot_contacts = jp.stack(
            [
                left_force, jp.float32(0.0),  # toe / heel split unused under placeholder
                right_force, jp.float32(0.0),
            ]
        ).astype(jp.float32)

        v4_compat = self._v4_compat_channels_from_window(win0, prev_history=None)
        critic_obs = self._get_privileged_critic_obs(data, root_vel_h)

        # v6 actor proprio history.  Zero-filled at reset per the
        # WildRobotInfo schema contract (env_info.py): the buffer
        # holds PAST proprio bundles only, never duplicates the
        # current frame.  At reset there is no past, so all slots
        # are zero; step 1's obs sees zeros in the history channel
        # and the current bundle in the standard proprio channels.
        # The buffer fills in over the first PROPRIO_HISTORY_FRAMES
        # steps as new_bundle gets rolled in (oldest dropped, newest
        # appended).
        proprio_bundle_size = 3 + 4 + 3 * self.action_size
        proprio_history_init = jp.zeros(
            (PROPRIO_HISTORY_FRAMES, proprio_bundle_size), dtype=jp.float32
        )

        wr_info = WildRobotInfo(
            step_count=jp.zeros((), dtype=jp.int32),
            prev_action=default_action,
            pending_action=default_action,
            truncated=jp.zeros(()),
            velocity_cmd=velocity_cmd,
            prev_root_pos=root_pose.position.astype(jp.float32),
            prev_root_quat=root_pose.orientation.astype(jp.float32),
            prev_left_foot_pos=left_foot_pos.astype(jp.float32),
            prev_right_foot_pos=right_foot_pos.astype(jp.float32),
            imu_quat_hist=imu_quat_hist,
            imu_gyro_hist=imu_gyro_hist,
            foot_contacts=foot_contacts,
            root_height=root_pose.height.astype(jp.float32),
            prev_left_loaded=(left_force > self._config.env.contact_threshold_force).astype(jp.float32),
            prev_right_loaded=(right_force > self._config.env.contact_threshold_force).astype(jp.float32),
            last_left_touchdown_x=left_foot_pos[0].astype(jp.float32),
            last_right_touchdown_x=right_foot_pos[0].astype(jp.float32),
            last_step_length=jp.float32(0.0),
            critic_obs=critic_obs,
            # v3 offline playback state — primary read/write fields.
            loc_ref_offline_step_idx=jp.zeros((), dtype=jp.int32),
            loc_ref_offline_command_id=jp.zeros((), dtype=jp.int32),
            # v4-compat / v5 obs feed (populated from offline window).
            loc_ref_stance_foot=win0["stance_foot_id"].astype(jp.int32),
            loc_ref_gait_phase_sin=v4_compat["phase_sin_cos"][0],
            loc_ref_gait_phase_cos=v4_compat["phase_sin_cos"][1],
            loc_ref_next_foothold=v4_compat["next_foothold"],
            loc_ref_swing_pos=v4_compat["swing_pos"],
            loc_ref_swing_vel=v4_compat["swing_vel"],
            loc_ref_pelvis_height=v4_compat["pelvis_targets"][0],
            loc_ref_pelvis_roll=v4_compat["pelvis_targets"][1],
            loc_ref_pelvis_pitch=v4_compat["pelvis_targets"][2],
            loc_ref_history=v4_compat["history"],
            nominal_q_ref=q_ref0,
            push_schedule=push_schedule,
            domain_rand_friction_scale=dr_params["friction_scale"],
            domain_rand_mass_scales=dr_params["mass_scales"],
            domain_rand_kp_scales=dr_params["kp_scales"],
            domain_rand_frictionloss_scales=dr_params["frictionloss_scales"],
            domain_rand_joint_offsets=dr_params["joint_offsets"],
            proprio_history=proprio_history_init,
            # ToddlerBot-aligned per-foot air-time / clearance bookkeeping
            # (env_info.WildRobotInfo).  At reset both feet are assumed
            # in stance: air_time / air_dist = 0; feet_height_init pins
            # the spawn-pose floor reference for clearance.
            feet_air_time=jp.zeros((2,), dtype=jp.float32),
            feet_air_dist=jp.zeros((2,), dtype=jp.float32),
            feet_height_init=jp.stack(
                [left_foot_pos[2], right_foot_pos[2]]
            ).astype(jp.float32),
            cmd_rng=cmd_rng.astype(jp.uint32),
        )

        obs = self._get_obs(
            data=data,
            action=default_action,
            velocity_cmd=velocity_cmd,
            win=win0,
            v4_compat=v4_compat,
            signals=signals_override,
            proprio_history=proprio_history_init,
        )

        roll_init, pitch_init, _ = root_pose.euler_angles()
        # Reset reward = 0 under ToddlerBot survival semantics (alive
        # only pays -w on the terminating step; reset is not a
        # terminating step so total reward at iter-0 is 0).
        reset_reward = jp.float32(0.0)
        contact_thresh = self._config.env.contact_threshold_force
        left_switch_init = (left_force > contact_thresh).astype(jp.float32)
        right_switch_init = (right_force > contact_thresh).astype(jp.float32)
        metrics_dict = get_initial_env_metrics_jax(
            velocity_cmd=velocity_cmd,
            height=root_pose.height,
            pitch=pitch_init.astype(jp.float32),
            roll=roll_init.astype(jp.float32),
            left_force=left_force,
            right_force=right_force,
            left_toe_switch=left_switch_init,
            left_heel_switch=left_switch_init,
            right_toe_switch=right_switch_init,
            right_heel_switch=right_switch_init,
            forward_reward=jp.float32(0.0),
            healthy_reward=jp.float32(0.0),
            action_rate=jp.float32(0.0),
            total_reward=reset_reward,
        )
        # Live diagnostics at reset.  episode_step_count is 0; phase
        # progress is 0; loc_ref tracking fields read from the window
        # at step 0.  Single-mode (offline playback) → mode_id=0,
        # progression_permission=1.
        metrics_dict["forward_velocity"] = root_vel_h.linear[0].astype(jp.float32)
        metrics_dict["episode_step_count"] = jp.float32(0.0)
        metrics_dict["tracking/loc_ref_phase_progress"] = jp.float32(0.0)
        metrics_dict["tracking/loc_ref_stance_foot"] = (
            win0["stance_foot_id"].astype(jp.float32)
        )
        metrics_dict["tracking/loc_ref_mode_id"] = jp.float32(0.0)
        metrics_dict["tracking/loc_ref_progression_permission"] = jp.float32(1.0)
        metrics_dict["tracking/loc_ref_left_reachable"] = jp.float32(1.0)
        metrics_dict["tracking/loc_ref_right_reachable"] = jp.float32(1.0)
        metrics_dict["tracking/nominal_q_abs_mean"] = jp.mean(jp.abs(q_ref0)).astype(jp.float32)
        metrics_dict["tracking/residual_q_abs_mean"] = jp.float32(0.0)
        metrics_dict["tracking/residual_q_abs_max"] = jp.float32(0.0)
        metrics_dict["tracking/cmd_vs_achieved_forward"] = jp.abs(
            root_vel_h.linear[0] - velocity_cmd
        ).astype(jp.float32)
        # G5 anti-exploit metrics — zeros at reset (no residual yet).
        metrics_dict["tracking/residual_hip_pitch_left_abs"] = jp.float32(0.0)
        metrics_dict["tracking/residual_hip_pitch_right_abs"] = jp.float32(0.0)
        metrics_dict["tracking/residual_knee_left_abs"] = jp.float32(0.0)
        metrics_dict["tracking/residual_knee_right_abs"] = jp.float32(0.0)
        metrics_dict["tracking/forward_velocity_cmd_ratio"] = (
            root_vel_h.linear[0]
            / jp.maximum(jp.abs(velocity_cmd), jp.float32(1e-3))
        ).astype(jp.float32)
        metrics_dict["tracking/step_length_touchdown_event_m"] = jp.float32(0.0)
        metrics = {
            METRICS_VEC_KEY: build_metrics_vec(metrics_dict),
        }

        info: Dict[str, Any] = {WR_INFO_KEY: wr_info}
        return WildRobotEnvState(
            data=data,
            obs=obs,
            reward=reset_reward,
            done=jp.float32(0.0),
            metrics=metrics,
            info=info,
            pipeline_state=data,
            rng=rng,
        )

    # ------------------------------------------------------------------ step

    def step(
        self,
        state: WildRobotEnvState,
        action: jax.Array,
        disable_pushes: bool = False,
    ) -> WildRobotEnvState:
        """One control step.  v3 flow:

            1. read wr.loc_ref_offline_step_idx, advance by 1
            2. lookup the offline window at the new step_idx
            3. filter raw policy_action (residual command in [-1, 1])
            4. optional 1-step action delay
            5. compose target_q = clip(q_ref + filtered_residual * scale)
            6. set ctrl + push xfrc; scan mjx.step n_substeps times
            7. v5 obs from the new data + window
            8. termination + imitation reward
            9. build new WildRobotInfo, advance step_count
        """
        wr = state.info[WR_INFO_KEY]
        velocity_cmd = wr.velocity_cmd
        pending_action = wr.pending_action

        next_step_idx = (wr.loc_ref_offline_step_idx + 1).astype(jp.int32)
        win = self._lookup_offline_window(next_step_idx)
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        # Prev-frame window for finite-diff reference velocity (lin_vel_z).
        # Clamp at 0 so step 0's "previous" is itself (zero velocity for the
        # very first step; subsequent steps see real bobbing).
        prev_step_idx = jp.maximum(next_step_idx - 1, 0).astype(jp.int32)
        prev_win = self._lookup_offline_window(prev_step_idx)
        prev_ref_pelvis_z = prev_win["pelvis_pos"][2].astype(jp.float32)

        # G6 contract: filter the residual ONLY (in policy [-1, 1] space).
        # ``pending_action`` carries last step's filtered residual command;
        # at reset it is zero (see ``_make_initial_state``), so iter-0 with
        # ``action == 0`` keeps ``filtered_action == 0`` and the composed
        # target_q is exactly q_ref, independent of action_filter_alpha.
        # The legacy path filtered the composed target in policy-space,
        # which low-passed q_ref over time and broke G6 for any alpha > 0.
        policy_state = PolicyState(prev_action=pending_action)
        filtered_action, policy_state = postprocess_action(
            spec=self._policy_spec,
            state=policy_state,
            action_raw=action,
        )
        applied_action = (
            pending_action if self._action_delay_enabled else filtered_action
        )

        applied_target_q, _residual_delta = self._compose_target_q_from_residual(
            policy_action=applied_action,
            nominal_q_ref=nominal_q_ref,
        )
        ctrl_mj = self._to_mj_ctrl(applied_target_q)

        # Apply push (gated by schedule + disable_pushes flag).
        push_enabled = jp.logical_not(jp.asarray(disable_pushes))
        data = apply_push(
            state.data, wr.push_schedule, wr.step_count, enabled=push_enabled
        )
        data = data.replace(ctrl=ctrl_mj)

        # Inner physics scan with the (possibly randomized) model.
        randomized_model = self._get_randomized_mjx_model(
            dict(
                friction_scale=wr.domain_rand_friction_scale,
                mass_scales=wr.domain_rand_mass_scales,
                kp_scales=wr.domain_rand_kp_scales,
                frictionloss_scales=wr.domain_rand_frictionloss_scales,
                joint_offsets=wr.domain_rand_joint_offsets,
            )
        )

        def _physics_step(d, _):
            d = d.replace(ctrl=ctrl_mj)
            d = mjx.step(randomized_model, d)
            return d, None

        data, _ = jax.lax.scan(_physics_step, data, (), self.n_substeps)

        # Post-step state.
        rng_step, rng_imu = jax.random.split(state.rng)
        signals_raw = self._signals_adapter.read(data)
        signals_override, imu_quat_hist, imu_gyro_hist, _ = _apply_imu_noise_and_delay(
            signals_raw, rng_imu, self._config, wr.imu_quat_hist, wr.imu_gyro_hist
        )

        root_pose = self._cal.get_root_pose(data)
        root_vel_h = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        roll_post, pitch_post, _yaw_post = root_pose.euler_angles()
        forward_velocity = root_vel_h.linear[0].astype(jp.float32)
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.WORLD
        )
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        contact_thresh = self._config.env.contact_threshold_force
        left_toe_switch = (left_force > contact_thresh).astype(jp.float32)
        right_toe_switch = (right_force > contact_thresh).astype(jp.float32)

        # G4 step-length-on-touchdown (walking_training.md v0.20.1 §, line 979).
        # Touchdown event = loaded transition (prev=0 → curr=1).  Step length =
        # current_foot_x - last_touchdown_foot_x of THAT foot.  Carries
        # ``last_step_length`` between events so the rollout MEAN is a usable
        # proxy for the spec's "touchdown step length mean".
        left_touchdown = ((wr.prev_left_loaded < 0.5) & (left_toe_switch > 0.5))
        right_touchdown = ((wr.prev_right_loaded < 0.5) & (right_toe_switch > 0.5))
        left_step_len = left_foot_pos[0] - wr.last_left_touchdown_x
        right_step_len = right_foot_pos[0] - wr.last_right_touchdown_x
        # Prefer right-foot value when both touch down on the same step (rare but
        # possible at flight-phase transitions); fall back to last_step_length
        # when neither foot touched down this step.
        new_step_length = jp.where(
            right_touchdown,
            right_step_len,
            jp.where(left_touchdown, left_step_len, wr.last_step_length),
        ).astype(jp.float32)
        new_last_left_touchdown_x = jp.where(
            left_touchdown, left_foot_pos[0], wr.last_left_touchdown_x
        ).astype(jp.float32)
        new_last_right_touchdown_x = jp.where(
            right_touchdown, right_foot_pos[0], wr.last_right_touchdown_x
        ).astype(jp.float32)

        # Termination + imitation reward family (v0.20.1).
        new_step_count = wr.step_count + 1
        done, terminated, truncated, term_info = self._get_termination(
            data, new_step_count
        )

        # ToddlerBot-style per-foot air-time / clearance bookkeeping.
        # ``stance`` = boolean foot-in-contact this step (post-step).
        # ``last_stance`` = previous-step boolean.  ``first_contact`` =
        # foot was airborne at any point of the swing AND is on the
        # ground now.  Reward is paid on ``first_contact``; air-time
        # and air-dist accumulators wipe to 0 on stance.  Mirrors
        # toddlerbot/locomotion/mjx_env.py:1052-1062 and
        # walk_env.py:_reward_feet_air_time / _reward_feet_clearance.
        ctrl_dt_f = jp.float32(self.dt)
        stance_mask = jp.stack([left_toe_switch, right_toe_switch]).astype(jp.float32)
        last_stance_mask = jp.stack(
            [wr.prev_left_loaded, wr.prev_right_loaded]
        ).astype(jp.float32)
        first_contact = jp.logical_or(
            stance_mask > 0.5, last_stance_mask > 0.5
        ).astype(jp.float32) * (wr.feet_air_time > 0).astype(jp.float32)
        new_feet_air_time = (wr.feet_air_time + ctrl_dt_f) * (1.0 - stance_mask)
        feet_z = jp.stack([left_foot_pos[2], right_foot_pos[2]]).astype(jp.float32)
        feet_z_delta = feet_z - wr.feet_height_init
        new_feet_air_dist = (wr.feet_air_dist + feet_z_delta) * (1.0 - stance_mask)

        reward_terms = self._compute_reward_terms(
            data=data,
            win=win,
            nominal_q_ref=nominal_q_ref,
            applied_action=applied_action,
            prev_applied_action=wr.prev_action,
            forward_velocity=forward_velocity,
            # v0.20.2 smoke6: TB-aligned vertical body velocity reward
            # needs current vz + prev-frame prior pelvis z for finite-diff
            # reference velocity.
            lin_vel_z=root_vel_h.linear[2].astype(jp.float32),
            prev_ref_pelvis_z=prev_ref_pelvis_z,
            velocity_cmd=velocity_cmd,
            left_force=left_force,
            right_force=right_force,
            root_pose=root_pose,
            prev_left_foot_pos=wr.prev_left_foot_pos,
            prev_right_foot_pos=wr.prev_right_foot_pos,
            # Use the RAW gyro for the reward (not the IMU-noisy override
            # the policy sees) so the regularizer signal is clean.
            gyro_rad_s=signals_raw.gyro_rad_s,
            # Pre-update air-time / air-dist values match ToddlerBot's
            # read-side semantics (mjx_env.py:1047 reads
            # ``info["feet_air_time"]`` BEFORE the += dt / *(1-stance)
            # update at lines 1053-1054).  ``wr.feet_air_time`` is what
            # was written at the end of the previous step, so by the
            # time the reward reads it on the touchdown step it already
            # holds the full accumulated swing time.
            feet_air_time=wr.feet_air_time,
            feet_air_dist=wr.feet_air_dist,
            first_contact=first_contact,
            stance_mask=stance_mask,
            left_foot_pos=left_foot_pos,
            right_foot_pos=right_foot_pos,
        )
        reward_contrib = self._aggregate_reward(reward_terms, terminated)
        reward = reward_contrib["total"]
        # Action-rate diagnostic (mean absolute filter delta — kept for
        # parity with the legacy v0.19.5c metric; the *reward* uses the
        # squared sum from reward_terms["penalty_action_rate"]).
        action_rate = jp.mean(jp.abs(applied_action - wr.prev_action)).astype(jp.float32)

        # v5 obs from the new state.
        v4_compat = self._v4_compat_channels_from_window(
            win, prev_history=wr.loc_ref_history
        )
        critic_obs = self._get_privileged_critic_obs(data, root_vel_h)

        # v6 actor proprio history (env_info.py schema): the buffer
        # holds PAST bundles only, never the current frame (which is
        # already in the obs's joint_pos / joint_vel / gyro /
        # foot_switches / prev_action channels).  So:
        #   1. obs at THIS step reads ``wr.proprio_history`` — the
        #      pre-roll buffer, past 3 bundles (zero-padded at reset,
        #      fully populated after step >= PROPRIO_HISTORY_FRAMES).
        #   2. Compute the new bundle from the post-step signals +
        #      this step's applied_action.
        #   3. Roll the buffer (drop oldest, append new) and store
        #      the rolled buffer in new_wr — that buffer becomes the
        #      "past" for the NEXT step.
        new_bundle = self._compute_proprio_bundle(
            signals=signals_override, prev_action=applied_action
        )
        new_proprio_history = self._roll_proprio_history(wr.proprio_history, new_bundle)

        # ToddlerBot-style cmd resampling.  ``cmd_resample_steps == 0``
        # disables resampling (episode-constant cmd, the v0.19.x and
        # v0.20.1 smoke contract).  When > 0 the cmd is redrawn every N
        # ticks via ``_sample_velocity_cmd`` (which honors zero_chance
        # and deadzone).  See toddlerbot/locomotion/mjx_env.py:1068-1077.
        # The smoke YAML pins min_velocity == max_velocity so the redraw
        # collapses to the same vx; the plumbing exists for v0.20.4
        # multi-command work without a future env edit.
        cmd_period = jp.int32(self._config.env.cmd_resample_steps)
        new_cmd_rng_carry, sample_rng = jax.random.split(wr.cmd_rng.astype(jp.uint32))
        should_resample = jp.logical_and(
            cmd_period > 0,
            jp.equal(jp.mod(new_step_count, jp.maximum(cmd_period, 1)), 0),
        )
        resampled_cmd = self._sample_velocity_cmd(sample_rng)
        new_velocity_cmd = jp.where(
            should_resample, resampled_cmd, velocity_cmd
        ).astype(jp.float32)
        new_cmd_rng = jp.where(
            should_resample, new_cmd_rng_carry, wr.cmd_rng
        ).astype(jp.uint32)

        new_wr = wr.replace(
            step_count=new_step_count,
            prev_action=applied_action,
            pending_action=policy_state.prev_action,
            truncated=truncated,
            velocity_cmd=new_velocity_cmd,
            prev_root_pos=root_pose.position.astype(jp.float32),
            prev_root_quat=root_pose.orientation.astype(jp.float32),
            prev_left_foot_pos=left_foot_pos.astype(jp.float32),
            prev_right_foot_pos=right_foot_pos.astype(jp.float32),
            imu_quat_hist=imu_quat_hist,
            imu_gyro_hist=imu_gyro_hist,
            foot_contacts=jp.stack(
                [left_force, jp.float32(0.0), right_force, jp.float32(0.0)]
            ).astype(jp.float32),
            root_height=root_pose.height.astype(jp.float32),
            prev_left_loaded=left_toe_switch,
            prev_right_loaded=right_toe_switch,
            last_left_touchdown_x=new_last_left_touchdown_x,
            last_right_touchdown_x=new_last_right_touchdown_x,
            last_step_length=new_step_length,
            critic_obs=critic_obs,
            loc_ref_offline_step_idx=next_step_idx,
            loc_ref_gait_phase_sin=v4_compat["phase_sin_cos"][0],
            loc_ref_gait_phase_cos=v4_compat["phase_sin_cos"][1],
            loc_ref_stance_foot=win["stance_foot_id"].astype(jp.int32),
            loc_ref_next_foothold=v4_compat["next_foothold"],
            loc_ref_swing_pos=v4_compat["swing_pos"],
            loc_ref_swing_vel=v4_compat["swing_vel"],
            loc_ref_pelvis_height=v4_compat["pelvis_targets"][0],
            loc_ref_pelvis_roll=v4_compat["pelvis_targets"][1],
            loc_ref_pelvis_pitch=v4_compat["pelvis_targets"][2],
            loc_ref_history=v4_compat["history"],
            nominal_q_ref=nominal_q_ref,
            proprio_history=new_proprio_history,
            feet_air_time=new_feet_air_time.astype(jp.float32),
            feet_air_dist=new_feet_air_dist.astype(jp.float32),
            cmd_rng=new_cmd_rng,
        )

        obs = self._get_obs(
            data=data,
            action=applied_action,
            velocity_cmd=velocity_cmd,
            win=win,
            v4_compat=v4_compat,
            signals=signals_override,
            # PRE-roll buffer: history = past 3 bundles, current frame
            # is supplied by the standard proprio channels.  See the
            # comment on new_proprio_history above for why this isn't
            # the rolled buffer.
            proprio_history=wr.proprio_history,
        )

        # Live diagnostics.  forward_velocity / phase_progress / pitch /
        # roll were previously hardcoded to zero, which masked real env
        # behavior and broke downstream eval (loc_ref probe phase
        # progress std).  The reward terms (m3_*, posture, slip, …) stay
        # zero under the placeholder-alive contract; Task #49 fills them
        # in with the imitation reward family.
        #
        # Residual is measured against what the env *actually applied*
        # (post action-filter and post action-delay), not the raw policy
        # action.  Otherwise tracking/residual_q_abs_* would lie when
        # action_filter_alpha > 0 or action_delay_steps == 1.
        residual_q_abs = jp.abs(applied_target_q - nominal_q_ref)
        terminal_metrics_dict = get_initial_env_metrics_jax(
            velocity_cmd=velocity_cmd,
            height=root_pose.height,
            pitch=pitch_post.astype(jp.float32),
            roll=roll_post.astype(jp.float32),
            left_force=left_force,
            right_force=right_force,
            left_toe_switch=left_toe_switch,
            left_heel_switch=left_toe_switch,
            right_toe_switch=right_toe_switch,
            right_heel_switch=right_toe_switch,
            forward_reward=jp.float32(0.0),
            healthy_reward=jp.float32(0.0),
            action_rate=action_rate,
            total_reward=reward,
        )
        terminal_metrics_dict.update(term_info)
        terminal_metrics_dict["forward_velocity"] = forward_velocity
        terminal_metrics_dict["episode_step_count"] = new_step_count.astype(jp.float32)
        terminal_metrics_dict["tracking/loc_ref_phase_progress"] = (
            next_step_idx.astype(jp.float32) / jp.float32(self._offline_n_steps)
        )
        terminal_metrics_dict["tracking/loc_ref_stance_foot"] = (
            win["stance_foot_id"].astype(jp.float32)
        )
        # Single-mode in v3 (offline playback); progression always permitted.
        terminal_metrics_dict["tracking/loc_ref_mode_id"] = jp.float32(0.0)
        terminal_metrics_dict["tracking/loc_ref_progression_permission"] = jp.float32(1.0)
        terminal_metrics_dict["tracking/nominal_q_abs_mean"] = jp.mean(
            jp.abs(nominal_q_ref)
        ).astype(jp.float32)
        terminal_metrics_dict["tracking/residual_q_abs_mean"] = jp.mean(residual_q_abs).astype(jp.float32)
        terminal_metrics_dict["tracking/residual_q_abs_max"] = jp.max(residual_q_abs).astype(jp.float32)
        terminal_metrics_dict["tracking/loc_ref_left_reachable"] = jp.float32(1.0)
        terminal_metrics_dict["tracking/loc_ref_right_reachable"] = jp.float32(1.0)
        # G4 promotion-horizon gate: |achieved_vx - cmd_vx|.  The gate is
        # ``<= 0.075 m/s`` per walking_training.md v0.20.1 §.
        terminal_metrics_dict["tracking/cmd_vs_achieved_forward"] = jp.abs(
            forward_velocity - velocity_cmd
        ).astype(jp.float32)
        # G5 anti-exploit hard gate (walking_training.md v0.20.1 §, line 980).
        # Per-joint |residual_delta_q| for hip_pitch L+R and knee L+R; the
        # spec calls for p50 ≤ 0.20 rad post-rollout, the registry's MEAN
        # reducer is the closest aggregator we have without adding a
        # MEDIAN reducer (the mean is a slightly weaker but still useful
        # signal for "policy uses too much leg authority").
        g5_residuals = residual_q_abs[self._g5_residual_idx]
        terminal_metrics_dict["tracking/residual_hip_pitch_left_abs"] = g5_residuals[0]
        terminal_metrics_dict["tracking/residual_hip_pitch_right_abs"] = g5_residuals[1]
        terminal_metrics_dict["tracking/residual_knee_left_abs"] = g5_residuals[2]
        terminal_metrics_dict["tracking/residual_knee_right_abs"] = g5_residuals[3]
        # Realized-vs-commanded forward speed ratio.  Spec gate: 0.6 ≤ ratio
        # ≤ 1.5 to catch both undershoot and v0.19.5 "lean and skate"
        # overshoot.  Guard against cmd≈0 to avoid log explosions.
        terminal_metrics_dict["tracking/forward_velocity_cmd_ratio"] = (
            forward_velocity / jp.maximum(jp.abs(velocity_cmd), jp.float32(1e-3))
        ).astype(jp.float32)
        # G4 step-length-on-touchdown.  Carries the most recent value
        # between events; rollout MEAN approximates the spec's
        # "touchdown step length mean ≥ 0.03 m" gate.
        terminal_metrics_dict["tracking/step_length_touchdown_event_m"] = new_step_length
        # v0.20.1 imitation reward terms (weighted contributions + diagnostics).
        terminal_metrics_dict["reward/total"] = reward
        terminal_metrics_dict["reward/alive"] = reward_contrib["alive"]
        terminal_metrics_dict["reward/ref_q_track"] = reward_contrib["ref_q_track"]
        terminal_metrics_dict["reward/ref_body_quat_track"] = reward_contrib[
            "ref_body_quat_track"
        ]
        terminal_metrics_dict["reward/torso_pos_xy"] = reward_contrib["torso_pos_xy"]
        # v0.20.2 smoke6: TB-aligned continuous phase signals.
        terminal_metrics_dict["reward/lin_vel_z"] = reward_contrib["lin_vel_z"]
        terminal_metrics_dict["reward/ang_vel_xy"] = reward_contrib["ang_vel_xy"]
        terminal_metrics_dict["reward/ref_contact_match"] = reward_contrib[
            "ref_contact_match"
        ]
        terminal_metrics_dict["reward/cmd_forward_velocity_track"] = reward_contrib[
            "cmd_forward_velocity_track"
        ]
        terminal_metrics_dict["reward/action_rate"] = reward_contrib["action_rate"]
        terminal_metrics_dict["reward/torque"] = reward_contrib["torque"]
        terminal_metrics_dict["reward/joint_vel"] = reward_contrib["joint_velocity"]
        terminal_metrics_dict["reward/slip"] = reward_contrib["slip"]
        terminal_metrics_dict["reward/pitch_rate"] = reward_contrib["pitch_rate"]
        terminal_metrics_dict["reward/feet_air_time"] = reward_contrib["feet_air_time"]
        terminal_metrics_dict["reward/feet_clearance"] = reward_contrib["feet_clearance"]
        terminal_metrics_dict["reward/feet_distance"] = reward_contrib["feet_distance"]
        terminal_metrics_dict["reward/torso_pitch_soft"] = reward_contrib[
            "torso_pitch_soft"
        ]
        terminal_metrics_dict["reward/torso_roll_soft"] = reward_contrib[
            "torso_roll_soft"
        ]
        terminal_metrics_dict["ref/q_track_err_rmse"] = reward_terms["q_track_rmse"]
        terminal_metrics_dict["ref/body_quat_err_deg"] = reward_terms["body_quat_err_deg"]
        terminal_metrics_dict["ref/feet_pos_err_l2"] = reward_terms["feet_pos_err_l2"]
        terminal_metrics_dict["ref/feet_pos_track_raw"] = reward_terms[
            "feet_pos_track_raw"
        ]
        terminal_metrics_dict["ref/torso_pos_xy_err_m"] = reward_terms[
            "torso_pos_xy_err_m"
        ]
        # v0.20.2 smoke6: TB-aligned phase-signal diagnostics.
        terminal_metrics_dict["ref/lin_vel_z_err_m_s"] = reward_terms[
            "lin_vel_z_err_m_s"
        ]
        terminal_metrics_dict["ref/ang_vel_xy_err_rad_s"] = reward_terms[
            "ang_vel_xy_err_rad_s"
        ]
        terminal_metrics_dict["ref/contact_phase_match"] = reward_terms["r_contact_match"]
        # Raw penalty values (pre-weight) so the M1 fail-mode tree can
        # pick a sensible weight when ``slip`` / ``pitch_rate`` are
        # promoted to nonzero in the YAML.
        terminal_metrics_dict["reward/penalty_slip_raw"] = reward_terms["penalty_slip"]
        terminal_metrics_dict["reward/penalty_pitch_rate_raw"] = reward_terms[
            "penalty_pitch_rate"
        ]

        # ----- Per-foot stride / swing-time diagnostics (v0.20.1-smoke2) -----
        # Smoke1 found ``tracking/step_length_touchdown_event_m`` (a single
        # last-touchdown carry across both feet) saturated below the
        # G4 0.030 m gate at ~0.022 m, but the dispatched-carry signal can't
        # tell us whether one foot is doing all the stepping or the gait
        # really is short on both sides.  Log per-foot raw values *only on
        # the touchdown step of that foot*, plus the touchdown event mask.
        # All MEAN-reduced (see metrics_registry).  Per-event mean step
        # length is then ``step_length_left_event_m / touchdown_rate_left``
        # post-aggregation.
        left_event_f = left_touchdown.astype(jp.float32)
        right_event_f = right_touchdown.astype(jp.float32)
        terminal_metrics_dict["tracking/touchdown_rate_left"] = left_event_f
        terminal_metrics_dict["tracking/touchdown_rate_right"] = right_event_f
        terminal_metrics_dict["tracking/swing_air_time_left_event_s"] = (
            wr.feet_air_time[0] * left_event_f
        ).astype(jp.float32)
        terminal_metrics_dict["tracking/swing_air_time_right_event_s"] = (
            wr.feet_air_time[1] * right_event_f
        ).astype(jp.float32)
        terminal_metrics_dict["tracking/step_length_left_event_m"] = (
            left_step_len * left_event_f
        ).astype(jp.float32)
        terminal_metrics_dict["tracking/step_length_right_event_m"] = (
            right_step_len * right_event_f
        ).astype(jp.float32)

        terminal_metrics = {METRICS_VEC_KEY: build_metrics_vec(terminal_metrics_dict)}

        # Auto-reset: on done, the next-episode starting data + obs +
        # info[WR_INFO_KEY] + rng come from a fresh reset; reward, done,
        # and metrics carry the TERMINAL step's values so PPO sees the
        # right reward, done flag (for GAE bootstrap), and termination
        # diagnostics on the rollout.  Without this, term/* and
        # forward_velocity at the terminal step would be replaced by
        # the reset's initial-metric zeros.
        def _do_reset(_):
            return self.reset(rng_step)

        def _no_reset(_):
            return WildRobotEnvState(
                data=data,
                obs=obs,
                reward=reward,
                done=done,
                metrics=terminal_metrics,
                info={WR_INFO_KEY: new_wr},
                pipeline_state=data,
                rng=rng_step,
            )

        next_state = jax.lax.cond(done > 0.5, _do_reset, _no_reset, operand=None)
        return next_state.replace(
            reward=reward, done=done, metrics=terminal_metrics
        )

    # ------------------------------------------------------------------ props

    @property
    def xml_path(self) -> str:
        return str(self._model_path)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def cal(self) -> ControlAbstractionLayer:
        return self._cal

    @property
    def observation_size(self) -> int:
        return self._policy_spec.model.obs_dim
