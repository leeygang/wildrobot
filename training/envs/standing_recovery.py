"""JAX recovery planner used by the v0.22.6 standing task."""

from __future__ import annotations

import jax
import jax.numpy as jnp


HOLD = 0
SWING = 1
SETTLE = 2


def encode_recovery_command(
    *,
    phase: jax.Array,
    swing_foot: jax.Array,
    phase_step: jax.Array,
    target_xy: jax.Array,
    swing_duration_steps: int,
    max_step_m: float,
) -> jax.Array:
    active = (phase != HOLD).astype(jnp.float32)
    swing = jnp.stack(
        [(swing_foot == 0).astype(jnp.float32), (swing_foot == 1).astype(jnp.float32)]
    ) * active
    progress = jnp.where(
        phase == SWING,
        jnp.minimum(
            1.0,
            phase_step.astype(jnp.float32) / jnp.float32(max(1, swing_duration_steps)),
        ),
        jnp.where(phase == SETTLE, 1.0, 0.0),
    )
    angle = jnp.pi * progress
    target = jnp.clip(
        target_xy / jnp.float32(max(max_step_m, 1e-6)), -1.0, 1.0
    ) * active
    return jnp.concatenate(
        [active[None], swing, jnp.stack([jnp.sin(angle), jnp.cos(angle)]), target]
    ).astype(jnp.float32)


def advance_recovery_planner(
    *,
    phase: jax.Array,
    swing_foot: jax.Array,
    phase_step: jax.Array,
    settle_count: jax.Array,
    target_xy: jax.Array,
    swing_start_pos: jax.Array,
    step_count: jax.Array,
    roll_rad: jax.Array,
    pitch_rad: jax.Array,
    roll_rate_rad_s: jax.Array,
    pitch_rate_rad_s: jax.Array,
    left_foot_pos: jax.Array,
    right_foot_pos: jax.Array,
    left_loaded: jax.Array,
    right_loaded: jax.Array,
    trigger_angle_rad: float,
    lookahead_s: float,
    com_height_m: float,
    capture_gain: float,
    max_step_m: float,
    swing_duration_steps: int,
    settle_min_steps: int,
    settle_max_steps: int,
    settle_angle_rad: float,
    settle_rate_rad_s: float,
) -> tuple[jax.Array, ...]:
    predicted_roll = roll_rad + jnp.float32(lookahead_s) * roll_rate_rad_s
    predicted_pitch = pitch_rad + jnp.float32(lookahead_s) * pitch_rate_rad_s
    severity = jnp.maximum(jnp.abs(predicted_roll), jnp.abs(predicted_pitch))
    stable = (
        (jnp.abs(roll_rad) < jnp.float32(settle_angle_rad))
        & (jnp.abs(pitch_rad) < jnp.float32(settle_angle_rad))
        & (jnp.abs(roll_rate_rad_s) < jnp.float32(settle_rate_rad_s))
        & (jnp.abs(pitch_rate_rad_s) < jnp.float32(settle_rate_rad_s))
        & left_loaded
        & right_loaded
    )
    start = (phase == HOLD) & (severity >= jnp.float32(trigger_angle_rad))
    retry = (
        (phase == SETTLE)
        & (phase_step >= jnp.int32(settle_max_steps))
        & ~stable
    )

    def start_step(_):
        omega0 = jnp.sqrt(jnp.float32(9.81 / max(com_height_m, 1e-6)))
        dx = jnp.float32(capture_gain * com_height_m) * (
            jnp.tan(pitch_rad) + pitch_rate_rad_s / omega0
        )
        dy = jnp.float32(capture_gain * com_height_m) * (
            jnp.tan(roll_rad) + roll_rate_rad_s / omega0
        )
        new_target = jnp.clip(
            jnp.stack([dx, dy]), -jnp.float32(max_step_m), jnp.float32(max_step_m)
        )
        lateral = jnp.abs(dy) > jnp.abs(dx)
        lateral_foot = jnp.where(dy >= 0.0, jnp.int32(0), jnp.int32(1))
        forward_foot = jnp.where(
            dx >= 0.0,
            jnp.where(left_foot_pos[0] <= right_foot_pos[0], 0, 1),
            jnp.where(left_foot_pos[0] >= right_foot_pos[0], 0, 1),
        ).astype(jnp.int32)
        new_swing = jnp.where(lateral, lateral_foot, forward_foot)
        start_pos = jnp.where(
            new_swing == 0, left_foot_pos, right_foot_pos
        ).astype(jnp.float32)
        return (
            jnp.int32(SWING), new_swing, jnp.int32(0), jnp.int32(0),
            new_target.astype(jnp.float32), start_pos, step_count + jnp.int32(1),
        )

    def continue_state(_):
        def swing_state(_):
            next_step = phase_step + jnp.int32(1)
            finished = next_step >= jnp.int32(swing_duration_steps)
            return (
                jnp.where(finished, jnp.int32(SETTLE), jnp.int32(SWING)),
                swing_foot,
                jnp.where(finished, jnp.int32(0), next_step),
                jnp.int32(0), target_xy, swing_start_pos, step_count,
            )

        def not_swing(_):
            new_settle = jnp.where(stable, settle_count + 1, jnp.int32(0))
            finished = (phase == SETTLE) & (
                new_settle >= jnp.int32(settle_min_steps)
            )
            return (
                jnp.where(finished, jnp.int32(HOLD), phase),
                jnp.where(finished, jnp.int32(-1), swing_foot),
                jnp.where(finished, jnp.int32(0), phase_step + (phase == SETTLE)),
                jnp.where(finished, jnp.int32(0), new_settle),
                jnp.where(finished, jnp.zeros(2, dtype=jnp.float32), target_xy),
                swing_start_pos,
                step_count,
            )

        return jax.lax.cond(phase == SWING, swing_state, not_swing, operand=None)

    return jax.lax.cond(start | retry, start_step, continue_state, operand=None)


def desired_swing_foot_position(
    *,
    swing_start_pos: jax.Array,
    target_xy: jax.Array,
    phase_step: jax.Array,
    swing_duration_steps: int,
    swing_height_m: float,
) -> jax.Array:
    progress = jnp.clip(
        (phase_step.astype(jnp.float32) + 1.0)
        / jnp.float32(max(1, swing_duration_steps)),
        0.0,
        1.0,
    )
    smooth = progress * progress * (3.0 - 2.0 * progress)
    xy = swing_start_pos[:2] + smooth * target_xy
    z = swing_start_pos[2] + jnp.float32(swing_height_m) * jnp.sin(jnp.pi * progress)
    return jnp.concatenate([xy, z[None]]).astype(jnp.float32)
