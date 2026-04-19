"""WildRobot training environment — v0.20.1 v3-only rewrite.

Single-file ``mjx_env.MjxEnv`` for the v0.20.1 PPO smoke
(``training/configs/ppo_walking_v0201_smoke.yaml``).  The actor sees
the offline ``ReferenceLibrary`` window (``wr_obs_v5_offline_ref``
layout) and emits a bounded residual on top of the library's
``q_ref``; the env composes ``q_target = q_ref + clip(action) *
scale_per_joint`` (absolute mode), runs MJX physics, and emits the
placeholder ``alive`` reward.

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
from training.configs.training_config import TrainingConfig, get_robot_config
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
        if layout_id != "wr_obs_v5_offline_ref":
            raise ValueError(
                "v0.20.1 WildRobotEnv requires env.actor_obs_layout_id="
                "'wr_obs_v5_offline_ref'.  Other layouts are unsupported in the "
                "v3-only rewrite (older layouts depended on v1/v2 reference state)."
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

        self._robot_config = get_robot_config()
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

    def _compose_loc_ref_residual_action(
        self,
        *,
        policy_action: jax.Array,
        nominal_q_ref: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Absolute-mode residual: target_q = q_ref + clip(action) * scale_per_joint.

        Returns ``(raw_action, residual_delta_q)`` where ``raw_action``
        is in PolicySpec [-1, 1] action space (so the action filter
        sees policy-space and the env still goes through
        ``JaxCalibOps`` for sign / span correction)."""
        residual_delta_q = (
            jp.clip(jp.asarray(policy_action, dtype=jp.float32), -1.0, 1.0)
            * self._residual_q_scale_per_joint
        )
        target_q = jp.clip(
            jp.asarray(nominal_q_ref, dtype=jp.float32) + residual_delta_q,
            self._joint_range_mins,
            self._joint_range_maxs,
        )
        raw_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec, ctrl_rad=target_q
        ).astype(jp.float32)
        return raw_action, residual_delta_q

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

    def _get_obs(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
        win: Dict[str, jax.Array],
        v4_compat: Dict[str, jax.Array],
        signals=None,
    ) -> jax.Array:
        """v5 layout dispatch.  All v4 + v5 ``loc_ref_*`` slots are
        populated from the offline window."""
        if signals is None:
            signals = self._signals_adapter.read(data)
        policy_state = PolicyState(prev_action=action)
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
        rng, key_vel, key_qnoise, key_push, key_dr, key_imu = jax.random.split(rng, 6)

        velocity_cmd = jax.random.uniform(
            key_vel,
            shape=(),
            minval=self._config.env.min_velocity,
            maxval=self._config.env.max_velocity,
        )

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
        )

    def _make_initial_state(
        self,
        *,
        rng: jax.Array,
        qpos: jax.Array,
        velocity_cmd: jax.Array,
        push_schedule: DisturbanceSchedule,
        dr_params: Dict[str, jax.Array],
        imu_init_rng: jax.Array,
    ) -> WildRobotEnvState:
        qvel = jp.zeros(self._mj_model.nv)
        randomized_model = self._get_randomized_mjx_model(dr_params)

        # Seed ctrl with the offline trajectory's frame-0 q_ref so the
        # action filter has no transient when policy_action == 0.
        win0 = self._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
        q_ref0 = win0["q_ref"].astype(jp.float32)
        ctrl_init = q_ref0
        data = mjx.make_data(randomized_model)
        data = data.replace(
            qpos=qpos, qvel=qvel, ctrl=self._to_mj_ctrl(ctrl_init)
        )
        data = mjx.forward(randomized_model, data)

        # Action filter init: policy-space action that maps to ctrl_init.
        default_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec, ctrl_rad=ctrl_init
        ).astype(jp.float32)

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
            cycle_start_forward_x=jp.float32(0.0),
            last_touchdown_root_pos=root_pose.position.astype(jp.float32),
            last_touchdown_foot=jp.float32(-1.0),
            prev_touchdown_foot_x=jp.float32(0.0),
            critic_obs=critic_obs,
            # v3-unused parametric-ref state — sentinels (kept on schema
            # so unrelated trainer code doesn't break; cleanup is §10 step 4).
            loc_ref_phase_time=jp.float32(0.0),
            loc_ref_stance_foot=win0["stance_foot_id"].astype(jp.int32),
            loc_ref_switch_count=jp.zeros((), dtype=jp.int32),
            loc_ref_mode_id=jp.zeros((), dtype=jp.int32),
            loc_ref_mode_time=jp.float32(0.0),
            loc_ref_startup_route_progress=jp.float32(1.0),
            loc_ref_startup_route_ceiling=jp.float32(1.0),
            loc_ref_startup_route_stage_id=jp.asarray(4, dtype=jp.int32),
            loc_ref_startup_route_transition_reason=jp.zeros((), dtype=jp.int32),
            loc_ref_com_x0_at_stance_start=jp.float32(0.0),
            loc_ref_com_vx0_at_stance_start=jp.float32(0.0),
            loc_ref_stance_margin_smooth=jp.float32(0.0),
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
            # v3 offline playback state — primary read/write fields.
            loc_ref_offline_step_idx=jp.zeros((), dtype=jp.int32),
            loc_ref_offline_command_id=jp.zeros((), dtype=jp.int32),
            push_schedule=push_schedule,
            # FSM / MPC / teacher / recovery — sentinels (drop in §10 step 4 cleanup).
            fsm_phase=jp.zeros((), dtype=jp.int32),
            fsm_swing_foot=jp.zeros((), dtype=jp.int32),
            fsm_phase_ticks=jp.zeros((), dtype=jp.int32),
            fsm_frozen_tx=jp.float32(0.0),
            fsm_frozen_ty=jp.float32(0.0),
            fsm_swing_sx=jp.float32(0.0),
            fsm_swing_sy=jp.float32(0.0),
            fsm_touch_hold=jp.zeros((), dtype=jp.int32),
            fsm_trigger_hold=jp.zeros((), dtype=jp.int32),
            recovery_active=jp.float32(0.0),
            recovery_age=jp.zeros((), dtype=jp.int32),
            recovery_last_support_foot=jp.asarray(-1, dtype=jp.int32),
            recovery_first_touchdown_recorded=jp.float32(0.0),
            recovery_first_step_latency=jp.zeros((), dtype=jp.int32),
            recovery_touchdown_count=jp.zeros((), dtype=jp.int32),
            recovery_support_foot_changes=jp.zeros((), dtype=jp.int32),
            recovery_first_liftoff_recorded=jp.float32(0.0),
            recovery_first_liftoff_latency=jp.zeros((), dtype=jp.int32),
            recovery_first_touchdown_age=jp.zeros((), dtype=jp.int32),
            recovery_visible_step_recorded=jp.float32(0.0),
            recovery_pitch_rate_at_push_end=jp.float32(0.0),
            recovery_pitch_rate_at_touchdown=jp.float32(0.0),
            recovery_pitch_rate_after_10t=jp.float32(0.0),
            recovery_capture_error_at_push_end=jp.float32(0.0),
            recovery_capture_error_at_touchdown=jp.float32(0.0),
            recovery_capture_error_after_10t=jp.float32(0.0),
            recovery_first_step_dx=jp.float32(0.0),
            recovery_first_step_dy=jp.float32(0.0),
            recovery_first_step_target_err_x=jp.float32(0.0),
            recovery_first_step_target_err_y=jp.float32(0.0),
            recovery_min_height=root_pose.height.astype(jp.float32),
            recovery_max_knee_flex=jp.float32(0.0),
            teacher_active=jp.float32(0.0),
            teacher_step_required_soft=jp.float32(0.0),
            teacher_step_required_hard=jp.float32(0.0),
            teacher_swing_foot=jp.asarray(-1, dtype=jp.float32),
            teacher_target_step_x=jp.float32(0.0),
            teacher_target_step_y=jp.float32(0.0),
            teacher_target_reachable=jp.float32(0.0),
            mpc_planner_active=jp.float32(0.0),
            mpc_controller_active=jp.float32(0.0),
            mpc_target_com_x=jp.float32(0.0),
            mpc_target_com_y=jp.float32(0.0),
            mpc_target_step_x=jp.float32(0.0),
            mpc_target_step_y=jp.float32(0.0),
            mpc_support_state=jp.float32(0.0),
            mpc_step_requested=jp.float32(0.0),
            domain_rand_friction_scale=dr_params["friction_scale"],
            domain_rand_mass_scales=dr_params["mass_scales"],
            domain_rand_kp_scales=dr_params["kp_scales"],
            domain_rand_frictionloss_scales=dr_params["frictionloss_scales"],
            domain_rand_joint_offsets=dr_params["joint_offsets"],
        )

        obs = self._get_obs(
            data=data,
            action=default_action,
            velocity_cmd=velocity_cmd,
            win=win0,
            v4_compat=v4_compat,
            signals=signals_override,
        )

        alive_weight = jp.float32(self._config.reward_weights.alive)
        metrics_dict = get_initial_env_metrics_jax(
            velocity_cmd=velocity_cmd,
            height=root_pose.height,
            pitch=jp.float32(0.0),
            roll=jp.float32(0.0),
            left_force=left_force,
            right_force=right_force,
            left_toe_switch=jp.float32(0.0),
            left_heel_switch=jp.float32(0.0),
            right_toe_switch=jp.float32(0.0),
            right_heel_switch=jp.float32(0.0),
            forward_reward=jp.float32(0.0),
            healthy_reward=jp.float32(0.0),
            action_rate=jp.float32(0.0),
            total_reward=alive_weight,
        )
        metrics = {
            METRICS_VEC_KEY: build_metrics_vec(metrics_dict),
        }

        info: Dict[str, Any] = {WR_INFO_KEY: wr_info}
        return WildRobotEnvState(
            data=data,
            obs=obs,
            reward=alive_weight,
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
            3. compose residual action around q_ref (absolute mode)
            4. action filter + optional 1-step delay
            5. set ctrl + push xfrc; scan mjx.step n_substeps times
            6. v5 obs from the new data + window
            7. termination + alive reward
            8. build new WildRobotInfo, advance step_count
        """
        wr = state.info[WR_INFO_KEY]
        velocity_cmd = wr.velocity_cmd
        pending_action = wr.pending_action

        next_step_idx = (wr.loc_ref_offline_step_idx + 1).astype(jp.int32)
        win = self._lookup_offline_window(next_step_idx)
        nominal_q_ref = win["q_ref"].astype(jp.float32)

        raw_action, _residual_delta = self._compose_loc_ref_residual_action(
            policy_action=action,
            nominal_q_ref=nominal_q_ref,
        )

        policy_state = PolicyState(prev_action=pending_action)
        filtered_action, policy_state = postprocess_action(
            spec=self._policy_spec,
            state=policy_state,
            action_raw=raw_action,
        )
        applied_action = (
            pending_action if self._action_delay_enabled else filtered_action
        )

        applied_target_q = JaxCalibOps.action_to_ctrl(
            spec=self._policy_spec, action=applied_action
        ).astype(jp.float32)
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
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.WORLD
        )
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        contact_thresh = self._config.env.contact_threshold_force

        # Termination + alive reward.
        new_step_count = wr.step_count + 1
        done, terminated, truncated, term_info = self._get_termination(
            data, new_step_count
        )
        alive_weight = jp.float32(self._config.reward_weights.alive)
        reward = alive_weight * (1.0 - terminated)

        # v5 obs from the new state.
        v4_compat = self._v4_compat_channels_from_window(
            win, prev_history=wr.loc_ref_history
        )
        critic_obs = self._get_privileged_critic_obs(data, root_vel_h)

        new_wr = wr.replace(
            step_count=new_step_count,
            prev_action=applied_action,
            pending_action=policy_state.prev_action,
            truncated=truncated,
            velocity_cmd=velocity_cmd,
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
            prev_left_loaded=(left_force > contact_thresh).astype(jp.float32),
            prev_right_loaded=(right_force > contact_thresh).astype(jp.float32),
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
        )

        obs = self._get_obs(
            data=data,
            action=applied_action,
            velocity_cmd=velocity_cmd,
            win=win,
            v4_compat=v4_compat,
            signals=signals_override,
        )

        # Auto-reset on done so Brax PPO sees a fresh trajectory after
        # termination.  Reuses current rng_step to keep the per-env
        # randomness flowing.
        def _do_reset(_):
            return self.reset(rng_step)

        def _no_reset(_):
            metrics_dict = get_initial_env_metrics_jax(
                velocity_cmd=velocity_cmd,
                height=root_pose.height,
                pitch=jp.float32(0.0),
                roll=jp.float32(0.0),
                left_force=left_force,
                right_force=right_force,
                left_toe_switch=jp.float32(0.0),
                left_heel_switch=jp.float32(0.0),
                right_toe_switch=jp.float32(0.0),
                right_heel_switch=jp.float32(0.0),
                forward_reward=jp.float32(0.0),
                healthy_reward=jp.float32(0.0),
                action_rate=jp.float32(0.0),
                total_reward=reward,
            )
            metrics_dict.update(term_info)
            new_metrics = {METRICS_VEC_KEY: build_metrics_vec(metrics_dict)}
            return WildRobotEnvState(
                data=data,
                obs=obs,
                reward=reward,
                done=done,
                metrics=new_metrics,
                info={WR_INFO_KEY: new_wr},
                pipeline_state=data,
                rng=rng_step,
            )

        next_state = jax.lax.cond(done > 0.5, _do_reset, _no_reset, operand=None)
        # Preserve done / reward on auto-reset so the trainer still
        # observes the terminal step's reward and the done flag for
        # GAE bootstrapping.
        return next_state.replace(reward=reward, done=done)

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
