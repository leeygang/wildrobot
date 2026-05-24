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
    # v0.21.0 P7 (wr_obs_v8_cmd3d): NEW size-2 actor channel carrying
    # ``(vy_cmd, wz_cmd)`` in the heading frame.  v1-v7 do NOT read
    # this kwarg (None passthrough is fine), so it stays optional.
    # Only v8 requires it (the v8 dispatch raises if missing).  The
    # shared scalar ``velocity_cmd`` slot still carries vx_cmd; the
    # v8 layout appends ``velocity_cmd_lateral_yaw`` after the
    # proprio_history block (see _build_obs_layout).
    velocity_cmd_lateral_yaw: jnp.ndarray | None = None,
    capture_point_error: jnp.ndarray | None = None,
    gait_clock: jnp.ndarray | None = None,
    loc_ref_phase_sin_cos: jnp.ndarray | None = None,
    loc_ref_stance_foot: jnp.ndarray | None = None,
    loc_ref_next_foothold: jnp.ndarray | None = None,
    loc_ref_swing_pos: jnp.ndarray | None = None,
    loc_ref_swing_vel: jnp.ndarray | None = None,
    loc_ref_pelvis_targets: jnp.ndarray | None = None,
    loc_ref_history: jnp.ndarray | None = None,
    # v0.20.1 wr_obs_v6_offline_ref_history reference-window kwargs
    # (None defaults so v1-v4 callsites remain unchanged):
    loc_ref_q_ref: jnp.ndarray | None = None,
    loc_ref_pelvis_pos: jnp.ndarray | None = None,
    loc_ref_pelvis_vel: jnp.ndarray | None = None,
    loc_ref_left_foot_pos: jnp.ndarray | None = None,
    loc_ref_right_foot_pos: jnp.ndarray | None = None,
    loc_ref_left_foot_vel: jnp.ndarray | None = None,
    loc_ref_right_foot_vel: jnp.ndarray | None = None,
    loc_ref_contact_mask: jnp.ndarray | None = None,
    # v0.20.1 wr_obs_v6_offline_ref_history addition: 3 past proprio
    # bundles (oldest -> newest), flattened.  See spec_builder.py for
    # the layout.  Required for v6, ignored for v1-v5.
    proprio_history: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if spec.observation.layout_id not in {
        "wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_v4",
        "wr_obs_v6_offline_ref_history",
        "wr_obs_v7_phase_proprio",
        # v0.21.0 P7: see spec.py SUPPORTED_LAYOUT_IDS for the
        # full motivation.  v8 is a strict superset of v7 that
        # appends a 2-dim ``velocity_cmd_lateral_yaw`` slot.
        "wr_obs_v8_cmd3d",
    }:
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

    # v0.20.1 wr_obs_v6_offline_ref_history reference-window channels.
    # Defaults are zero arrays of the right shape so the v6 branch
    # below can extend ``parts`` without conditionals.
    n_actuators = len(spec.robot.actuator_names)
    v5_q_ref = (
        jnp.zeros((n_actuators,), dtype=jnp.float32)
        if loc_ref_q_ref is None
        else loc_ref_q_ref.reshape(n_actuators)
    )
    v5_pelvis_pos = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_pelvis_pos is None
        else loc_ref_pelvis_pos.reshape(3)
    )
    v5_pelvis_vel = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_pelvis_vel is None
        else loc_ref_pelvis_vel.reshape(3)
    )
    v5_left_foot_pos = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_left_foot_pos is None
        else loc_ref_left_foot_pos.reshape(3)
    )
    v5_right_foot_pos = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_right_foot_pos is None
        else loc_ref_right_foot_pos.reshape(3)
    )
    v5_left_foot_vel = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_left_foot_vel is None
        else loc_ref_left_foot_vel.reshape(3)
    )
    v5_right_foot_vel = (
        jnp.zeros((3,), dtype=jnp.float32)
        if loc_ref_right_foot_vel is None
        else loc_ref_right_foot_vel.reshape(3)
    )
    v5_contact_mask = (
        jnp.zeros((2,), dtype=jnp.float32)
        if loc_ref_contact_mask is None
        else loc_ref_contact_mask.reshape(2)
    )

    parts = [
        gravity_local.reshape(3),
        angvel_heading_local.reshape(3),
        joint_pos_normalized.reshape(-1),
        joint_vel_normalized.reshape(-1),
        foot_switches.reshape(4),
        prev_action.reshape(-1),
        # v0.21.0 P7: ``velocity_cmd`` may arrive as scalar, (1,), or
        # (3,) — the env now always emits (3,) [vx, vy, wz] post-P3,
        # but legacy callers (export tooling, parity tests) still feed
        # scalars / (1,).  Take the first element to preserve v1-v7
        # byte-identical behavior (vx_cmd lives at index 0 of the
        # 3-vec).  The v8 layout dispatches (vy, wz) separately via
        # the ``velocity_cmd_lateral_yaw`` kwarg.
        jnp.atleast_1d(jnp.asarray(velocity_cmd, dtype=jnp.float32))[..., :1].reshape(1),
    ]
    if spec.observation.layout_id == "wr_obs_v2":
        parts.append(cp_error)
    if spec.observation.layout_id == "wr_obs_v3":
        parts.append(clock)
    if spec.observation.layout_id == "wr_obs_v4":
        parts.extend([phase, stance, foothold, swing_pos, swing_vel, pelvis, history])
    if spec.observation.layout_id == "wr_obs_v6_offline_ref_history":
        # v6 = v4 base + offline-ref window + flat proprio_history.
        # History is required at v6 — caller must supply it; missing
        # history would be a wiring bug.  Order matches
        # v0201_env_wiring.md §5.2; do not reorder without updating
        # the design note and the parity test.
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
        parts.append(jnp.asarray(proprio_history, dtype=jnp.float32).reshape(-1))
    if spec.observation.layout_id == "wr_obs_v7_phase_proprio":
        # smoke11 de-hybridized actor: TB-aligned proprio + command +
        # 2-dim phase clock + proprio_history.  All other loc_ref_*
        # kwargs passed by the env are intentionally ignored here so
        # the env's build_observation call site can stay layout-
        # agnostic (it always populates every reference channel; the
        # layout decides what reaches the actor).
        parts.append(phase)
        if proprio_history is None:
            raise ValueError(
                "wr_obs_v7_phase_proprio requires proprio_history; got None."
            )
        parts.append(jnp.asarray(proprio_history, dtype=jnp.float32).reshape(-1))
    if spec.observation.layout_id == "wr_obs_v8_cmd3d":
        # v0.21.0 P7: strict superset of v7 — same base + phase +
        # proprio_history, with the new 2-dim ``velocity_cmd_lateral_yaw``
        # slot APPENDED before the trailing padding.  The shared
        # scalar ``velocity_cmd`` slot above already carries vx_cmd
        # (sliced from the 3-vec at the parts builder), so v8 only
        # needs to add (vy_cmd, wz_cmd) here.
        parts.append(phase)
        if proprio_history is None:
            raise ValueError(
                "wr_obs_v8_cmd3d requires proprio_history; got None."
            )
        parts.append(jnp.asarray(proprio_history, dtype=jnp.float32).reshape(-1))
        if velocity_cmd_lateral_yaw is None:
            raise ValueError(
                "wr_obs_v8_cmd3d requires velocity_cmd_lateral_yaw=(vy, wz); "
                "got None.  Env must pass velocity_cmd[1:] for v8."
            )
        parts.append(
            jnp.asarray(velocity_cmd_lateral_yaw, dtype=jnp.float32).reshape(2)
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
    # v0.21.0 P7 — see ``build_observation_from_components`` docstring.
    velocity_cmd_lateral_yaw: jnp.ndarray | None = None,
    capture_point_error: jnp.ndarray | None = None,
    gait_clock: jnp.ndarray | None = None,
    loc_ref_phase_sin_cos: jnp.ndarray | None = None,
    loc_ref_stance_foot: jnp.ndarray | None = None,
    loc_ref_next_foothold: jnp.ndarray | None = None,
    loc_ref_swing_pos: jnp.ndarray | None = None,
    loc_ref_swing_vel: jnp.ndarray | None = None,
    loc_ref_pelvis_targets: jnp.ndarray | None = None,
    loc_ref_history: jnp.ndarray | None = None,
    # v0.20.1 wr_obs_v6_offline_ref_history reference-window kwargs:
    loc_ref_q_ref: jnp.ndarray | None = None,
    loc_ref_pelvis_pos: jnp.ndarray | None = None,
    loc_ref_pelvis_vel: jnp.ndarray | None = None,
    loc_ref_left_foot_pos: jnp.ndarray | None = None,
    loc_ref_right_foot_pos: jnp.ndarray | None = None,
    loc_ref_left_foot_vel: jnp.ndarray | None = None,
    loc_ref_right_foot_vel: jnp.ndarray | None = None,
    loc_ref_contact_mask: jnp.ndarray | None = None,
    proprio_history: jnp.ndarray | None = None,
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
