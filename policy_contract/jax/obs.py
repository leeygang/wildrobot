from __future__ import annotations

import jax.numpy as jnp

from policy_contract.calib import JaxCalibOps
from policy_contract.jax.frames import angvel_heading_local, gravity_local_from_quat
from policy_contract.jax.signals import Signals
from policy_contract.jax.state import PolicyState
from policy_contract.spec import PolicySpec


def build_observation_from_components(
    *,
    spec: PolicySpec,
    gravity_local: jnp.ndarray,
    angvel_heading_local: jnp.ndarray,
    joint_pos_normalized: jnp.ndarray,
    joint_vel_normalized: jnp.ndarray,
    foot_switches: jnp.ndarray,
    prev_action: jnp.ndarray,
    velocity_cmd: jnp.ndarray,
    capture_point_error: jnp.ndarray | None = None,
    gait_clock: jnp.ndarray | None = None,
    teacher_phase: jnp.ndarray | None = None,
    teacher_joint_pos_target: jnp.ndarray | None = None,
    teacher_root_lin_vel_target: jnp.ndarray | None = None,
    teacher_root_height_target: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if spec.observation.layout_id not in {"wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_teacher"}:
        raise ValueError(f"Unsupported layout_id: {spec.observation.layout_id}")

    cp_error = (
        jnp.zeros((2,), dtype=jnp.float32)
        if capture_point_error is None
        else capture_point_error.reshape(2)
    )
    clock = (
        jnp.zeros((4,), dtype=jnp.float32)
        if gait_clock is None
        else gait_clock.reshape(4)
    )
    teacher_phase_feat = (
        jnp.zeros((2,), dtype=jnp.float32)
        if teacher_phase is None
        else teacher_phase.reshape(2)
    )
    action_dim = spec.model.action_dim
    teacher_joint_target = (
        jnp.zeros((action_dim,), dtype=jnp.float32)
        if teacher_joint_pos_target is None
        else teacher_joint_pos_target.reshape(action_dim)
    )
    teacher_root_vel_target = (
        jnp.zeros((2,), dtype=jnp.float32)
        if teacher_root_lin_vel_target is None
        else teacher_root_lin_vel_target.reshape(2)
    )
    teacher_root_h_target = (
        jnp.zeros((1,), dtype=jnp.float32)
        if teacher_root_height_target is None
        else teacher_root_height_target.reshape(1)
    )

    parts = [
        gravity_local.reshape(3),
        angvel_heading_local.reshape(3),
        joint_pos_normalized.reshape(-1),
        joint_vel_normalized.reshape(-1),
        foot_switches.reshape(4),
        prev_action.reshape(-1),
        jnp.atleast_1d(velocity_cmd).reshape(1),
    ]
    if spec.observation.layout_id == "wr_obs_v2":
        parts.append(cp_error)
    if spec.observation.layout_id == "wr_obs_v3":
        parts.append(clock)
    if spec.observation.layout_id == "wr_obs_teacher":
        parts.extend(
            [
                teacher_phase_feat,
                teacher_joint_target,
                teacher_root_vel_target,
                teacher_root_h_target,
            ]
        )
    parts.append(jnp.zeros((1,), dtype=jnp.float32))

    obs = jnp.concatenate(parts)
    return obs.astype(jnp.float32)


def build_observation(
    *,
    spec: PolicySpec,
    state: PolicyState,
    signals: Signals,
    velocity_cmd: jnp.ndarray,
    capture_point_error: jnp.ndarray | None = None,
    gait_clock: jnp.ndarray | None = None,
    teacher_phase: jnp.ndarray | None = None,
    teacher_joint_pos_target: jnp.ndarray | None = None,
    teacher_root_lin_vel_target: jnp.ndarray | None = None,
    teacher_root_height_target: jnp.ndarray | None = None,
) -> jnp.ndarray:
    gravity = gravity_local_from_quat(signals.quat_xyzw)
    angvel = angvel_heading_local(signals.gyro_rad_s, signals.quat_xyzw)

    joint_pos_norm = JaxCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=signals.joint_pos_rad)
    joint_vel_norm = JaxCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=signals.joint_vel_rad_s)

    return build_observation_from_components(
        spec=spec,
        gravity_local=gravity,
        angvel_heading_local=angvel,
        joint_pos_normalized=joint_pos_norm,
        joint_vel_normalized=joint_vel_norm,
        foot_switches=signals.foot_switches,
        prev_action=state.prev_action,
        velocity_cmd=velocity_cmd,
        capture_point_error=capture_point_error,
        gait_clock=gait_clock,
        teacher_phase=teacher_phase,
        teacher_joint_pos_target=teacher_joint_pos_target,
        teacher_root_lin_vel_target=teacher_root_lin_vel_target,
        teacher_root_height_target=teacher_root_height_target,
    )
