"""v0.17.3 standing bring-up controller scaffold.

This module provides the smallest honest slice of an MPC-inspired standing path:
- conservative upright feedback in action space
- lightweight step-request signal (for inspectability)
- explicit debug outputs for planner/controller state

It is NOT a full MPC and does not claim hard-push recovery.
"""

from __future__ import annotations

import jax
import jax.numpy as jp

from control.interfaces import ControllerDebugSignals


def compute_mpc_standing_action(
    *,
    pitch: jax.Array,
    roll: jax.Array,
    pitch_rate: jax.Array,
    roll_rate: jax.Array,
    height_error: jax.Array,
    forward_vel: jax.Array,
    lateral_vel: jax.Array,
    default_action: jax.Array,
    policy_action: jax.Array,
    hip_pitch_gain: jax.Array,
    ankle_pitch_gain: jax.Array,
    hip_roll_gain: jax.Array,
    idx_left_hip_pitch: int,
    idx_right_hip_pitch: int,
    idx_left_ankle_pitch: int,
    idx_right_ankle_pitch: int,
    idx_left_hip_roll: int,
    idx_right_hip_roll: int,
    pitch_kp: float,
    pitch_kd: float,
    roll_kp: float,
    roll_kd: float,
    height_kp: float,
    residual_scale: float,
    action_clip: float,
    step_trigger_threshold: float,
) -> tuple[jax.Array, ControllerDebugSignals]:
    """Compute conservative standing action and inspectable debug signals."""
    clip = jp.maximum(jp.asarray(action_clip, dtype=jp.float32), 0.0)

    pitch_delta = -(
        jp.asarray(pitch_kp, dtype=jp.float32) * pitch
        + jp.asarray(pitch_kd, dtype=jp.float32) * pitch_rate
    )
    roll_delta = -(
        jp.asarray(roll_kp, dtype=jp.float32) * roll
        + jp.asarray(roll_kd, dtype=jp.float32) * roll_rate
    )
    # Keep height correction tiny and symmetric in pitch joints.
    height_delta = jp.asarray(height_kp, dtype=jp.float32) * height_error

    pitch_delta = jp.clip(pitch_delta + height_delta, -clip, clip)
    roll_delta = jp.clip(roll_delta, -clip, clip)

    action = default_action
    if idx_left_hip_pitch >= 0:
        action = action.at[idx_left_hip_pitch].add(hip_pitch_gain * pitch_delta)
    if idx_right_hip_pitch >= 0:
        action = action.at[idx_right_hip_pitch].add(hip_pitch_gain * pitch_delta)
    if idx_left_ankle_pitch >= 0:
        action = action.at[idx_left_ankle_pitch].add(ankle_pitch_gain * pitch_delta)
    if idx_right_ankle_pitch >= 0:
        action = action.at[idx_right_ankle_pitch].add(ankle_pitch_gain * pitch_delta)
    if idx_left_hip_roll >= 0:
        action = action.at[idx_left_hip_roll].add(hip_roll_gain * roll_delta)
    if idx_right_hip_roll >= 0:
        action = action.at[idx_right_hip_roll].add(hip_roll_gain * roll_delta)

    residual = jp.asarray(residual_scale, dtype=jp.float32) * policy_action
    mixed = jp.clip(action + residual, -1.0, 1.0).astype(jp.float32)

    need_step = jp.clip((jp.abs(roll) + jp.abs(lateral_vel)) / 2.0, 0.0, 1.0)
    step_requested = (need_step > jp.asarray(step_trigger_threshold, dtype=jp.float32)).astype(
        jp.float32
    )
    support_state = jp.where(lateral_vel >= 0.0, jp.asarray(1.0, dtype=jp.float32), 0.0)

    # Debug-only placeholders representing planner intent (not real MPC output).
    target_com_x = -0.05 * forward_vel
    target_com_y = -0.05 * lateral_vel
    target_step_x = 0.12 * forward_vel
    target_step_y = 0.18 * lateral_vel

    debug = ControllerDebugSignals(
        planner_active=jp.asarray(1.0, dtype=jp.float32),
        controller_active=jp.asarray(1.0, dtype=jp.float32),
        target_com_x=target_com_x.astype(jp.float32),
        target_com_y=target_com_y.astype(jp.float32),
        target_step_x=target_step_x.astype(jp.float32),
        target_step_y=target_step_y.astype(jp.float32),
        support_state=support_state.astype(jp.float32),
        step_requested=step_requested.astype(jp.float32),
    )
    return mixed, debug
