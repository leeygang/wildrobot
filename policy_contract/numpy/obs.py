from __future__ import annotations

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.frames import angvel_heading_local, gravity_local_from_quat
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicySpec


def build_observation_from_components(
    *,
    spec: PolicySpec,
    gravity_local: np.ndarray,
    angvel_heading_local: np.ndarray,
    joint_pos_normalized: np.ndarray,
    joint_vel_normalized: np.ndarray,
    foot_switches: np.ndarray,
    prev_action: np.ndarray,
    velocity_cmd: np.ndarray,
    # v0.21.0 P11 (wr_obs_v8_cmd3d): numpy mirror of policy_contract/jax/obs.py.
    # NEW size-2 actor channel carrying (vy_cmd, wz_cmd).  The shared
    # ``velocity_cmd`` slot above stays scalar (sliced to ``[..., :1]``) so
    # v1-v7 layouts are byte-identical regardless of whether a (1,) or (3,)
    # ``velocity_cmd`` is passed.  The v8 layout appends this 2-vec
    # immediately after ``proprio_history`` (see body below).
    velocity_cmd_lateral_yaw: np.ndarray | None = None,
    capture_point_error: np.ndarray | None = None,
    gait_clock: np.ndarray | None = None,
    loc_ref_phase_sin_cos: np.ndarray | None = None,
    loc_ref_stance_foot: np.ndarray | None = None,
    loc_ref_next_foothold: np.ndarray | None = None,
    loc_ref_swing_pos: np.ndarray | None = None,
    loc_ref_swing_vel: np.ndarray | None = None,
    loc_ref_pelvis_targets: np.ndarray | None = None,
    loc_ref_history: np.ndarray | None = None,
    # v0.20.1 wr_obs_v6_offline_ref_history reference-window kwargs
    # (mirror policy_contract/jax/obs.py).
    loc_ref_q_ref: np.ndarray | None = None,
    loc_ref_pelvis_pos: np.ndarray | None = None,
    loc_ref_pelvis_vel: np.ndarray | None = None,
    loc_ref_left_foot_pos: np.ndarray | None = None,
    loc_ref_right_foot_pos: np.ndarray | None = None,
    loc_ref_left_foot_vel: np.ndarray | None = None,
    loc_ref_right_foot_vel: np.ndarray | None = None,
    loc_ref_contact_mask: np.ndarray | None = None,
    proprio_history: np.ndarray | None = None,
) -> np.ndarray:
    if spec.observation.layout_id not in {
        "wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_v4",
        "wr_obs_v6_offline_ref_history",
        "wr_obs_v7_phase_proprio",
        # v0.21.0 P11: strict superset of v7 — same base + phase +
        # proprio_history, plus a 2-dim ``velocity_cmd_lateral_yaw`` slot.
        "wr_obs_v8_cmd3d",
    }:
        raise ValueError(f"Unsupported layout_id: {spec.observation.layout_id}")

    cp_error = (
        np.zeros((2,), dtype=np.float32)
        if capture_point_error is None
        else np.asarray(capture_point_error, dtype=np.float32).reshape(2)
    )
    clock = (
        np.zeros((4,), dtype=np.float32)
        if gait_clock is None
        else np.asarray(gait_clock, dtype=np.float32).reshape(4)
    )
    phase = (
        np.zeros((2,), dtype=np.float32)
        if loc_ref_phase_sin_cos is None
        else np.asarray(loc_ref_phase_sin_cos, dtype=np.float32).reshape(2)
    )
    stance = (
        np.zeros((1,), dtype=np.float32)
        if loc_ref_stance_foot is None
        else np.asarray(loc_ref_stance_foot, dtype=np.float32).reshape(1)
    )
    foothold = (
        np.zeros((2,), dtype=np.float32)
        if loc_ref_next_foothold is None
        else np.asarray(loc_ref_next_foothold, dtype=np.float32).reshape(2)
    )
    swing_pos = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_swing_pos is None
        else np.asarray(loc_ref_swing_pos, dtype=np.float32).reshape(3)
    )
    swing_vel = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_swing_vel is None
        else np.asarray(loc_ref_swing_vel, dtype=np.float32).reshape(3)
    )
    pelvis = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_pelvis_targets is None
        else np.asarray(loc_ref_pelvis_targets, dtype=np.float32).reshape(3)
    )
    history = (
        np.zeros((4,), dtype=np.float32)
        if loc_ref_history is None
        else np.asarray(loc_ref_history, dtype=np.float32).reshape(4)
    )

    # v0.20.1 wr_obs_v6_offline_ref_history reference-window channels —
    # mirrors policy_contract/jax/obs.py.  Defaults are zero arrays so
    # the v6 branch below can extend ``parts`` without conditionals.
    n_actuators = len(spec.robot.actuator_names)
    v5_q_ref = (
        np.zeros((n_actuators,), dtype=np.float32)
        if loc_ref_q_ref is None
        else np.asarray(loc_ref_q_ref, dtype=np.float32).reshape(n_actuators)
    )
    v5_pelvis_pos = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_pelvis_pos is None
        else np.asarray(loc_ref_pelvis_pos, dtype=np.float32).reshape(3)
    )
    v5_pelvis_vel = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_pelvis_vel is None
        else np.asarray(loc_ref_pelvis_vel, dtype=np.float32).reshape(3)
    )
    v5_left_foot_pos = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_left_foot_pos is None
        else np.asarray(loc_ref_left_foot_pos, dtype=np.float32).reshape(3)
    )
    v5_right_foot_pos = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_right_foot_pos is None
        else np.asarray(loc_ref_right_foot_pos, dtype=np.float32).reshape(3)
    )
    v5_left_foot_vel = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_left_foot_vel is None
        else np.asarray(loc_ref_left_foot_vel, dtype=np.float32).reshape(3)
    )
    v5_right_foot_vel = (
        np.zeros((3,), dtype=np.float32)
        if loc_ref_right_foot_vel is None
        else np.asarray(loc_ref_right_foot_vel, dtype=np.float32).reshape(3)
    )
    v5_contact_mask = (
        np.zeros((2,), dtype=np.float32)
        if loc_ref_contact_mask is None
        else np.asarray(loc_ref_contact_mask, dtype=np.float32).reshape(2)
    )

    parts = [
        np.asarray(gravity_local, dtype=np.float32).reshape(3),
        np.asarray(angvel_heading_local, dtype=np.float32).reshape(3),
        np.asarray(joint_pos_normalized, dtype=np.float32).reshape(-1),
        np.asarray(joint_vel_normalized, dtype=np.float32).reshape(-1),
        np.asarray(foot_switches, dtype=np.float32).reshape(4),
        np.asarray(prev_action, dtype=np.float32).reshape(-1),
        # v0.21.0 P11: mirror policy_contract/jax/obs.py.  ``velocity_cmd``
        # may arrive as scalar, (1,), or (3,) — the env now always emits
        # (3,) [vx, vy, wz] post-P3, but legacy callers still feed scalars
        # / (1,).  Slice the first element to preserve v1-v7 byte-identical
        # behavior (vx_cmd lives at index 0 of the 3-vec).  The v8 layout
        # dispatches (vy, wz) separately via ``velocity_cmd_lateral_yaw``.
        np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)[..., :1].reshape(1),
    ]
    if spec.observation.layout_id == "wr_obs_v2":
        parts.append(cp_error)
    if spec.observation.layout_id == "wr_obs_v3":
        parts.append(clock)
    if spec.observation.layout_id == "wr_obs_v4":
        parts.extend([phase, stance, foothold, swing_pos, swing_vel, pelvis, history])
    if spec.observation.layout_id == "wr_obs_v6_offline_ref_history":
        # v6 = v4 base + offline-ref window + flat proprio_history.
        # See policy_contract/jax/obs.py for the matching JAX branch.
        # Order locked by v0201_env_wiring.md §5.2.
        parts.extend([phase, stance, foothold, swing_pos, swing_vel, pelvis, history])
        parts.extend([
            v5_q_ref,
            v5_pelvis_pos, v5_pelvis_vel,
            v5_left_foot_pos, v5_right_foot_pos,
            v5_left_foot_vel, v5_right_foot_vel,
            v5_contact_mask,
        ])
        if proprio_history is None:
            raise ValueError(
                "wr_obs_v6_offline_ref_history requires proprio_history; "
                "got None.  Env must roll a per-step proprio buffer and "
                "pass it through build_observation."
            )
        parts.append(np.asarray(proprio_history, dtype=np.float32).reshape(-1))
    if spec.observation.layout_id == "wr_obs_v7_phase_proprio":
        # smoke11 de-hybridized actor — see policy_contract/jax/obs.py
        # for the matching JAX branch.  Order locked by
        # policy_contract/spec_builder.py.
        parts.append(phase)
        if proprio_history is None:
            raise ValueError(
                "wr_obs_v7_phase_proprio requires proprio_history; got None."
            )
        parts.append(np.asarray(proprio_history, dtype=np.float32).reshape(-1))
    if spec.observation.layout_id == "wr_obs_v8_cmd3d":
        # v0.21.0 P11: strict superset of v7 — same base + phase +
        # proprio_history, with the new 2-dim ``velocity_cmd_lateral_yaw``
        # slot APPENDED before the trailing padding.  The shared scalar
        # ``velocity_cmd`` slot above already carries vx_cmd (sliced from
        # the 3-vec at the parts builder), so v8 only needs to add
        # (vy_cmd, wz_cmd) here.  Mirrors policy_contract/jax/obs.py.
        parts.append(phase)
        if proprio_history is None:
            raise ValueError(
                "wr_obs_v8_cmd3d requires proprio_history; got None."
            )
        parts.append(np.asarray(proprio_history, dtype=np.float32).reshape(-1))
        if velocity_cmd_lateral_yaw is None:
            raise ValueError(
                "wr_obs_v8_cmd3d requires velocity_cmd_lateral_yaw=(vy, wz); "
                "got None.  Env must pass velocity_cmd[1:] for v8."
            )
        parts.append(
            np.asarray(velocity_cmd_lateral_yaw, dtype=np.float32).reshape(2),
        )
    parts.append(np.zeros((1,), dtype=np.float32))

    obs = np.concatenate(parts)
    return obs.astype(np.float32)


def build_observation(
    *,
    spec: PolicySpec,
    state: PolicyState,
    signals: Signals,
    velocity_cmd: np.ndarray,
    # v0.21.0 P11 — see ``build_observation_from_components`` docstring.
    velocity_cmd_lateral_yaw: np.ndarray | None = None,
    capture_point_error: np.ndarray | None = None,
    gait_clock: np.ndarray | None = None,
    loc_ref_phase_sin_cos: np.ndarray | None = None,
    loc_ref_stance_foot: np.ndarray | None = None,
    loc_ref_next_foothold: np.ndarray | None = None,
    loc_ref_swing_pos: np.ndarray | None = None,
    loc_ref_swing_vel: np.ndarray | None = None,
    loc_ref_pelvis_targets: np.ndarray | None = None,
    loc_ref_history: np.ndarray | None = None,
    # v0.20.1 wr_obs_v6_offline_ref_history reference-window kwargs:
    loc_ref_q_ref: np.ndarray | None = None,
    loc_ref_pelvis_pos: np.ndarray | None = None,
    loc_ref_pelvis_vel: np.ndarray | None = None,
    loc_ref_left_foot_pos: np.ndarray | None = None,
    loc_ref_right_foot_pos: np.ndarray | None = None,
    loc_ref_left_foot_vel: np.ndarray | None = None,
    loc_ref_right_foot_vel: np.ndarray | None = None,
    loc_ref_contact_mask: np.ndarray | None = None,
    proprio_history: np.ndarray | None = None,
) -> np.ndarray:
    gravity = gravity_local_from_quat(signals.quat_xyzw)
    angvel = angvel_heading_local(signals.gyro_rad_s, signals.quat_xyzw)

    joint_pos_norm = NumpyCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=signals.joint_pos_rad)
    joint_vel_norm = NumpyCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=signals.joint_vel_rad_s)

    return build_observation_from_components(
        spec=spec,
        gravity_local=gravity,
        angvel_heading_local=angvel,
        joint_pos_normalized=joint_pos_norm,
        joint_vel_normalized=joint_vel_norm,
        foot_switches=signals.foot_switches,
        prev_action=state.prev_action,
        velocity_cmd=velocity_cmd,
        velocity_cmd_lateral_yaw=velocity_cmd_lateral_yaw,
        capture_point_error=capture_point_error,
        gait_clock=gait_clock,
        loc_ref_phase_sin_cos=loc_ref_phase_sin_cos,
        loc_ref_stance_foot=loc_ref_stance_foot,
        loc_ref_next_foothold=loc_ref_next_foothold,
        loc_ref_swing_pos=loc_ref_swing_pos,
        loc_ref_swing_vel=loc_ref_swing_vel,
        loc_ref_pelvis_targets=loc_ref_pelvis_targets,
        loc_ref_history=loc_ref_history,
        loc_ref_q_ref=loc_ref_q_ref,
        loc_ref_pelvis_pos=loc_ref_pelvis_pos,
        loc_ref_pelvis_vel=loc_ref_pelvis_vel,
        loc_ref_left_foot_pos=loc_ref_left_foot_pos,
        loc_ref_right_foot_pos=loc_ref_right_foot_pos,
        loc_ref_left_foot_vel=loc_ref_left_foot_vel,
        loc_ref_right_foot_vel=loc_ref_right_foot_vel,
        loc_ref_contact_mask=loc_ref_contact_mask,
        proprio_history=proprio_history,
    )
