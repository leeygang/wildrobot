from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class TeacherWholeBodyTarget(NamedTuple):
    """Training-time whole-body teacher outputs for disturbed-window recovery."""

    whole_body_active: jax.Array
    whole_body_active_count: jax.Array
    recovery_height_target: jax.Array
    recovery_height_error: jax.Array
    recovery_height_in_band: jax.Array
    com_velocity_target: jax.Array
    com_velocity_error: jax.Array
    com_velocity_target_hit: jax.Array


def compute_teacher_whole_body_target(
    *,
    whole_body_teacher_enabled: bool,
    push_active: jax.Array,
    recovery_active: jax.Array,
    step_required_hard: jax.Array,
    height: jax.Array,
    horizontal_speed: jax.Array,
    height_target_min: float,
    height_target_max: float,
    height_hard_gate: bool,
    com_vel_target: float,
) -> TeacherWholeBodyTarget:
    """Compute compact whole-body teacher targets under disturbed-window gating."""
    disturbed = jp.logical_or(
        jp.asarray(push_active, dtype=jp.bool_),
        jp.asarray(recovery_active, dtype=jp.bool_),
    )
    hard_gate_ok = (
        jp.asarray(step_required_hard, dtype=jp.float32) > 0.5
        if bool(height_hard_gate)
        else jp.asarray(True, dtype=jp.bool_)
    )
    active = jp.asarray(whole_body_teacher_enabled, dtype=jp.bool_) & disturbed & hard_gate_ok
    active_f = active.astype(jp.float32)

    target_min = jp.asarray(height_target_min, dtype=jp.float32)
    target_max = jp.asarray(height_target_max, dtype=jp.float32)
    target_mid = 0.5 * (target_min + target_max)

    h = jp.asarray(height, dtype=jp.float32)
    below = target_min - h
    above = h - target_max
    height_error = jp.maximum(jp.maximum(below, above), 0.0)
    in_band = (height_error <= 1e-6).astype(jp.float32)

    speed = jp.asarray(horizontal_speed, dtype=jp.float32)
    speed_target = jp.asarray(com_vel_target, dtype=jp.float32)
    speed_error = jp.maximum(speed - speed_target, 0.0)
    speed_hit = (speed <= speed_target).astype(jp.float32)

    return TeacherWholeBodyTarget(
        whole_body_active=active_f,
        whole_body_active_count=active_f,
        recovery_height_target=jp.where(active, target_mid, 0.0).astype(jp.float32),
        recovery_height_error=jp.where(active, height_error, 0.0).astype(jp.float32),
        recovery_height_in_band=jp.where(active, in_band, 0.0).astype(jp.float32),
        com_velocity_target=jp.where(active, speed_target, 0.0).astype(jp.float32),
        com_velocity_error=jp.where(active, speed_error, 0.0).astype(jp.float32),
        com_velocity_target_hit=jp.where(active, speed_hit, 0.0).astype(jp.float32),
    )


def compute_teacher_recovery_height_reward(
    *,
    height_error: jax.Array,
    teacher_active: jax.Array,
    sigma: float,
) -> jax.Array:
    """Band-target reward: 1.0 inside band, smoothly decays with outside-band error."""
    err = jp.asarray(height_error, dtype=jp.float32)
    sigma_safe = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    base = jp.exp(-(err * err) / (sigma_safe * sigma_safe))
    return jp.asarray(teacher_active, dtype=jp.float32) * base


def compute_teacher_com_velocity_reduction_reward(
    *,
    com_velocity_error: jax.Array,
    horizontal_speed: jax.Array,
    teacher_active: jax.Array,
    active_speed_min: float,
    sigma: float,
) -> jax.Array:
    """Teacher reward for reducing horizontal CoM speed after disturbance."""
    err = jp.asarray(com_velocity_error, dtype=jp.float32)
    sigma_safe = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    speed = jp.asarray(horizontal_speed, dtype=jp.float32)
    active_speed_gate = (speed >= jp.asarray(active_speed_min, dtype=jp.float32)).astype(jp.float32)
    base = jp.exp(-(err * err) / (sigma_safe * sigma_safe))
    return jp.asarray(teacher_active, dtype=jp.float32) * active_speed_gate * base
