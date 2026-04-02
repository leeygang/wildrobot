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
    loc_ref_phase_sin_cos: jnp.ndarray | None = None,
    loc_ref_stance_foot: jnp.ndarray | None = None,
    loc_ref_next_foothold: jnp.ndarray | None = None,
    loc_ref_swing_pos: jnp.ndarray | None = None,
    loc_ref_swing_vel: jnp.ndarray | None = None,
    loc_ref_pelvis_targets: jnp.ndarray | None = None,
    loc_ref_history: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if spec.observation.layout_id not in {"wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_v4"}:
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
    phase = (
        jnp.zeros((2,), dtype=jnp.float32)
        if loc_ref_phase_sin_cos is None
        else loc_ref_phase_sin_cos.reshape(2)
    )
    stance = (
        jnp.zeros((1,), dtype=jnp.float32)
        if loc_ref_stance_foot is None
        else jnp.asarray(loc_ref_stance_foot, dtype=jnp.float32).reshape(1)
    )
    foothold = (
        jnp.zeros((2,), dtype=jnp.float32)
        if loc_ref_next_foothold is None
        else loc_ref_next_foothold.reshape(2)
    )
    swing_pos = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_swing_pos is None
        else loc_ref_swing_pos.reshape(3)
    )
    swing_vel = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_swing_vel is None
        else loc_ref_swing_vel.reshape(3)
    )
    pelvis = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_pelvis_targets is None
        else loc_ref_pelvis_targets.reshape(3)
    )
    history = (
        jnp.zeros((4,), dtype=jnp.float32)
        if loc_ref_history is None
        else loc_ref_history.reshape(4)
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
    if spec.observation.layout_id == "wr_obs_v4":
        parts.extend([phase, stance, foothold, swing_pos, swing_vel, pelvis, history])
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
    loc_ref_phase_sin_cos: jnp.ndarray | None = None,
    loc_ref_stance_foot: jnp.ndarray | None = None,
    loc_ref_next_foothold: jnp.ndarray | None = None,
    loc_ref_swing_pos: jnp.ndarray | None = None,
    loc_ref_swing_vel: jnp.ndarray | None = None,
    loc_ref_pelvis_targets: jnp.ndarray | None = None,
    loc_ref_history: jnp.ndarray | None = None,
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
        loc_ref_phase_sin_cos=loc_ref_phase_sin_cos,
        loc_ref_stance_foot=loc_ref_stance_foot,
        loc_ref_next_foothold=loc_ref_next_foothold,
        loc_ref_swing_pos=loc_ref_swing_pos,
        loc_ref_swing_vel=loc_ref_swing_vel,
        loc_ref_pelvis_targets=loc_ref_pelvis_targets,
        loc_ref_history=loc_ref_history,
    )
