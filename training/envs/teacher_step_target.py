from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class TeacherStepTarget(NamedTuple):
    """Training-time teacher outputs for disturbed-window step guidance."""

    teacher_active: jax.Array
    active_count: jax.Array
    step_required_soft: jax.Array
    step_required_hard: jax.Array
    raw_step_required_soft: jax.Array
    raw_step_required_hard: jax.Array
    swing_foot: jax.Array
    target_step_x: jax.Array
    target_step_y: jax.Array
    target_reachable: jax.Array


def compute_teacher_step_target(
    *,
    teacher_enabled: bool,
    push_active: jax.Array,
    recovery_active: jax.Array,
    need_step: jax.Array,
    pitch: jax.Array,
    roll: jax.Array,
    pitch_rate: jax.Array,
    forward_vel: jax.Array,
    lateral_vel: jax.Array,
    capture_error_xy: jax.Array,
    hard_threshold: float,
    x_min: float,
    x_max: float,
    left_y_min: float,
    left_y_max: float,
    right_y_min: float,
    right_y_max: float,
) -> TeacherStepTarget:
    """Compute bounded heuristic teacher targets for recovery stepping."""
    push_or_recovery = jp.logical_or(
        jp.asarray(push_active, dtype=jp.bool_),
        jp.asarray(recovery_active, dtype=jp.bool_),
    )
    disturbed = jp.asarray(teacher_enabled, dtype=jp.bool_) & push_or_recovery

    capture_x = jp.asarray(capture_error_xy[0], dtype=jp.float32)
    capture_y = jp.asarray(capture_error_xy[1], dtype=jp.float32)
    capture_norm = jp.sqrt(capture_x * capture_x + capture_y * capture_y)

    step_required_soft_raw = jp.clip(
        0.55 * jp.asarray(need_step, dtype=jp.float32)
        + 0.20 * (jp.abs(pitch) / 0.35)
        + 0.10 * (jp.abs(roll) / 0.35)
        + 0.10 * (jp.abs(pitch_rate) / 1.0)
        + 0.05 * (capture_norm / 0.08),
        0.0,
        1.0,
    )
    hard_threshold_arr = jp.asarray(hard_threshold, dtype=jp.float32)
    step_required_hard_raw = step_required_soft_raw >= hard_threshold_arr

    lateral_fall_signal = roll + 0.4 * lateral_vel + 1.5 * capture_y
    swing_foot_raw = jp.where(
        lateral_fall_signal > 0.0,
        jp.asarray(1, dtype=jp.int32),  # right
        jp.asarray(0, dtype=jp.int32),  # left
    )

    target_x_raw = (
        0.60 * (-capture_x)
        + 0.04 * jp.asarray(forward_vel, dtype=jp.float32)
        + 0.05 * jp.asarray(pitch, dtype=jp.float32)
        + 0.03 * jp.asarray(pitch_rate, dtype=jp.float32)
    )
    target_y_mag_raw = (
        0.04
        + 0.02 * jp.clip(jp.abs(lateral_fall_signal), 0.0, 1.0)
        + 0.01 * jp.clip(jp.abs(capture_y), 0.0, 1.0)
    )
    target_y_raw = jp.where(swing_foot_raw == 0, target_y_mag_raw, -target_y_mag_raw)

    left_reachable = (
        (target_x_raw >= x_min)
        & (target_x_raw <= x_max)
        & (target_y_raw >= left_y_min)
        & (target_y_raw <= left_y_max)
    )
    right_reachable = (
        (target_x_raw >= x_min)
        & (target_x_raw <= x_max)
        & (target_y_raw >= right_y_min)
        & (target_y_raw <= right_y_max)
    )
    target_reachable_raw = jp.where(swing_foot_raw == 0, left_reachable, right_reachable)

    target_x_clipped = jp.clip(target_x_raw, x_min, x_max)
    target_y_clipped = jp.where(
        swing_foot_raw == 0,
        jp.clip(target_y_raw, left_y_min, left_y_max),
        jp.clip(target_y_raw, right_y_min, right_y_max),
    )

    teacher_active = disturbed.astype(jp.float32)
    return TeacherStepTarget(
        teacher_active=teacher_active,
        active_count=teacher_active,
        step_required_soft=jp.where(disturbed, step_required_soft_raw, 0.0).astype(jp.float32),
        step_required_hard=jp.where(disturbed, step_required_hard_raw, False).astype(jp.float32),
        raw_step_required_soft=step_required_soft_raw.astype(jp.float32),
        raw_step_required_hard=step_required_hard_raw.astype(jp.float32),
        swing_foot=jp.where(
            disturbed, swing_foot_raw, jp.asarray(-1, dtype=jp.int32)
        ).astype(jp.int32),
        target_step_x=jp.where(disturbed, target_x_clipped, 0.0).astype(jp.float32),
        target_step_y=jp.where(disturbed, target_y_clipped, 0.0).astype(jp.float32),
        target_reachable=jp.where(disturbed, target_reachable_raw, False).astype(jp.float32),
    )


def compute_teacher_target_step_xy_reward(
    *,
    touchdown_x: jax.Array,
    touchdown_y: jax.Array,
    target_x: jax.Array,
    target_y: jax.Array,
    teacher_active: jax.Array,
    first_recovery_touchdown: jax.Array,
    sigma: float,
) -> tuple[jax.Array, jax.Array]:
    """Soft agreement reward for first recovery touchdown vs teacher XY target."""
    dx = jp.asarray(touchdown_x, dtype=jp.float32) - jp.asarray(target_x, dtype=jp.float32)
    dy = jp.asarray(touchdown_y, dtype=jp.float32) - jp.asarray(target_y, dtype=jp.float32)
    err_sq = dx * dx + dy * dy
    err_norm = jp.sqrt(err_sq)
    sigma_safe = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    reward = jp.exp(-(err_sq / (sigma_safe * sigma_safe)))
    gate = (
        jp.asarray(teacher_active, dtype=jp.float32)
        * jp.asarray(first_recovery_touchdown, dtype=jp.float32)
    )
    return gate * reward, err_norm


def compute_teacher_step_required_reward(
    *,
    teacher_active: jax.Array,
    step_required_soft: jax.Array,
    first_recovery_liftoff: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reward first recovery liftoff only when teacher says stepping is needed."""
    event = jp.asarray(first_recovery_liftoff, dtype=jp.float32)
    gate = jp.asarray(teacher_active, dtype=jp.float32) * event
    demand = jp.clip(jp.asarray(step_required_soft, dtype=jp.float32), 0.0, 1.0)
    return gate * demand, gate


def compute_teacher_swing_foot_reward(
    *,
    touchdown_foot: jax.Array,
    teacher_swing_foot: jax.Array,
    teacher_active: jax.Array,
    teacher_step_required_hard: jax.Array,
    first_recovery_touchdown: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reward touchdown-foot agreement with teacher's suggested swing foot."""
    gate = (
        jp.asarray(first_recovery_touchdown, dtype=jp.float32)
        * jp.asarray(teacher_active, dtype=jp.float32)
        * jp.asarray(teacher_step_required_hard, dtype=jp.float32)
    )
    match = (
        jp.asarray(touchdown_foot, dtype=jp.int32)
        == jp.asarray(teacher_swing_foot, dtype=jp.int32)
    ).astype(jp.float32)
    return gate * match, gate
