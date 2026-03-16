"""Teacher tracking reward helpers for v0.16.x bootstrap."""

from __future__ import annotations

import jax.numpy as jp
import jax


def _safe_sigma(sigma: jax.Array) -> jax.Array:
    return jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)


def wrapped_phase_error(current_phase: jax.Array, reference_phase: jax.Array) -> jax.Array:
    """Shortest wrapped phase error in [0, 0.5]."""
    delta = jp.mod(current_phase - reference_phase + 0.5, 1.0) - 0.5
    return jp.abs(delta)


def phase_consistency_reward(
    current_phase: jax.Array,
    reference_phase: jax.Array,
    sigma: jax.Array = 0.10,
) -> tuple[jax.Array, jax.Array]:
    """Gaussian phase-consistency reward and raw phase error."""
    err = wrapped_phase_error(current_phase, reference_phase)
    sig = _safe_sigma(sigma)
    reward = jp.exp(-jp.square(err / sig))
    return reward, err


def root_pose_tracking_reward(
    root_pos: jax.Array,
    root_quat_wxyz: jax.Array,
    ref_root_pos: jax.Array,
    ref_root_quat_wxyz: jax.Array,
    *,
    pos_sigma: jax.Array,
    quat_sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Track root translation and orientation with smooth Gaussian shaping."""
    pos_err = jp.linalg.norm(root_pos - ref_root_pos)
    pos_reward = jp.exp(-jp.square(pos_err / _safe_sigma(pos_sigma)))

    dot = jp.abs(jp.sum(root_quat_wxyz * ref_root_quat_wxyz))
    dot = jp.clip(dot, 0.0, 1.0)
    quat_err = 1.0 - dot
    quat_reward = jp.exp(-jp.square(quat_err / _safe_sigma(quat_sigma)))

    return pos_reward * quat_reward, pos_err, quat_err


def root_velocity_tracking_reward(
    root_lin_vel: jax.Array,
    root_ang_vel: jax.Array,
    ref_root_lin_vel: jax.Array,
    ref_root_ang_vel: jax.Array,
    *,
    lin_sigma: jax.Array,
    ang_sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Track root linear and angular velocities."""
    lin_err = jp.linalg.norm(root_lin_vel - ref_root_lin_vel)
    ang_err = jp.linalg.norm(root_ang_vel - ref_root_ang_vel)
    lin_reward = jp.exp(-jp.square(lin_err / _safe_sigma(lin_sigma)))
    ang_reward = jp.exp(-jp.square(ang_err / _safe_sigma(ang_sigma)))
    return lin_reward * ang_reward, lin_err, ang_err


def joint_pose_tracking_reward(
    joint_pos: jax.Array,
    ref_joint_pos: jax.Array,
    *,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Track actuator-space joint targets."""
    mse = jp.mean(jp.square(joint_pos - ref_joint_pos))
    rmse = jp.sqrt(mse + 1e-12)
    reward = jp.exp(-jp.square(rmse / _safe_sigma(sigma)))
    return reward, rmse


def foot_position_contact_reward(
    left_foot_pos: jax.Array,
    right_foot_pos: jax.Array,
    ref_left_foot_pos: jax.Array,
    ref_right_foot_pos: jax.Array,
    left_contact: jax.Array,
    right_contact: jax.Array,
    ref_left_contact: jax.Array,
    ref_right_contact: jax.Array,
    *,
    pos_sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Track foot placement and contact timing agreement."""
    left_err = jp.linalg.norm(left_foot_pos - ref_left_foot_pos)
    right_err = jp.linalg.norm(right_foot_pos - ref_right_foot_pos)
    pos_err = 0.5 * (left_err + right_err)
    pos_reward = jp.exp(-jp.square(pos_err / _safe_sigma(pos_sigma)))

    left_agree = 1.0 - jp.abs(left_contact - ref_left_contact)
    right_agree = 1.0 - jp.abs(right_contact - ref_right_contact)
    contact_agreement = 0.5 * (left_agree + right_agree)
    contact_reward = jp.clip(contact_agreement, 0.0, 1.0)
    return pos_reward * contact_reward, pos_err, contact_reward, contact_agreement


def upright_reward(
    pitch: jax.Array,
    roll: jax.Array,
    *,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reward upright posture with smooth Gaussian shaping."""
    tilt = jp.sqrt(jp.square(pitch) + jp.square(roll))
    reward = jp.exp(-jp.square(tilt / _safe_sigma(sigma)))
    return reward, tilt

