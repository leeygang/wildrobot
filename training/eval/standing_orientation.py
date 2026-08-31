"""Standing-orientation metrics shared by deterministic evaluators."""

from __future__ import annotations

import math
from typing import Dict

import jax.numpy as jnp


STANDING_FINAL_WINDOW_S = 0.5
WALKING_STABLE_START_S = 2.0
WALKING_PRE_FALL_WINDOW_S = 0.5


def quaternion_tilt_rad(quat_wxyz: jnp.ndarray) -> jnp.ndarray:
    """Return roll/pitch tilt from upright, independent of yaw."""
    quat = quat_wxyz / jnp.maximum(
        jnp.linalg.norm(quat_wxyz, axis=-1, keepdims=True),
        jnp.float32(1e-8),
    )
    body_z_world_z = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
    return jnp.arccos(jnp.clip(body_z_world_z, -1.0, 1.0)).astype(jnp.float32)


def quaternion_yaw_rad(quat_wxyz: jnp.ndarray) -> jnp.ndarray:
    """Return wrapped yaw relative to the identity standing reference."""
    quat = quat_wxyz / jnp.maximum(
        jnp.linalg.norm(quat_wxyz, axis=-1, keepdims=True),
        jnp.float32(1e-8),
    )
    w, x, y, z = jnp.moveaxis(quat, -1, 0)
    return jnp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ).astype(jnp.float32)


def roll_pitch_tilt_rad(
    roll_rad: jnp.ndarray,
    pitch_rad: jnp.ndarray,
) -> jnp.ndarray:
    """Return yaw-independent torso tilt from Euler roll and pitch."""
    body_z_world_z = jnp.cos(roll_rad) * jnp.cos(pitch_rad)
    return jnp.arccos(jnp.clip(body_z_world_z, -1.0, 1.0)).astype(jnp.float32)


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    count = jnp.sum(mask)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(mask, values, 0.0)) / count,
        jnp.float32(0.0),
    )


def _masked_max(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(
        jnp.any(mask),
        jnp.max(jnp.where(mask, values, -jnp.inf)),
        jnp.float32(0.0),
    )


def _masked_percentile(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    percentile: float,
) -> jnp.ndarray:
    """Return a JIT-safe nearest-rank percentile over masked values."""
    flat_values = jnp.ravel(values)
    flat_mask = jnp.ravel(mask)
    count = jnp.sum(flat_mask.astype(jnp.int32))
    sorted_values = jnp.sort(jnp.where(flat_mask, flat_values, jnp.inf))
    rank = jnp.ceil(jnp.float32(percentile / 100.0) * count).astype(jnp.int32) - 1
    rank = jnp.clip(rank, 0, sorted_values.size - 1)
    return jnp.where(count > 0, sorted_values[rank], jnp.float32(0.0))


def summarize_walking_orientation_rollout(
    roll_rad: jnp.ndarray,
    pitch_rad: jnp.ndarray,
    dones: jnp.ndarray,
    truncations: jnp.ndarray,
    *,
    ctrl_dt: float,
    stable_start_s: float = WALKING_STABLE_START_S,
    pre_fall_window_s: float = WALKING_PRE_FALL_WINDOW_S,
) -> Dict[str, jnp.ndarray]:
    """Separate stable walking posture from first-episode fall telemetry.

    Walking environments auto-reset after ``done``.  The metric arrays retain
    the terminal roll/pitch values, unlike the returned state quaternion, which
    already belongs to the reset episode.  Only each environment's first
    episode is used so post-reset samples cannot dilute a failure.
    """
    if ctrl_dt <= 0.0:
        raise ValueError("ctrl_dt must be positive")
    if stable_start_s < 0.0:
        raise ValueError("stable_start_s must be non-negative")
    if pre_fall_window_s <= 0.0:
        raise ValueError("pre_fall_window_s must be positive")
    if roll_rad.shape != pitch_rad.shape:
        raise ValueError("roll_rad and pitch_rad must have matching shapes")
    if roll_rad.shape != dones.shape or dones.shape != truncations.shape:
        raise ValueError("orientation, done, and truncation arrays must match")
    if roll_rad.ndim != 2:
        raise ValueError("walking rollout arrays must have shape (T, N)")

    num_steps, num_envs = roll_rad.shape
    done = dones > 0.5
    truncated = truncations > 0.5
    first_episode = (jnp.cumsum(done, axis=0) - done.astype(jnp.int32)) == 0
    terminal_event = first_episode & done
    failure_event = terminal_event & ~truncated
    failed_env = jnp.any(failure_event, axis=0)
    survivor_env = ~failed_env

    tilt_deg = jnp.rad2deg(roll_pitch_tilt_rad(roll_rad, pitch_rad))
    step_index = jnp.arange(num_steps, dtype=jnp.int32)[:, None]
    stable_start_steps = int(math.ceil(stable_start_s / ctrl_dt))
    stable_mask = (
        first_episode
        & survivor_env[None, :]
        & (step_index >= stable_start_steps)
    )

    has_terminal_event = jnp.any(terminal_event, axis=0)
    first_terminal_index = jnp.argmax(terminal_event, axis=0)
    final_index = jnp.where(
        has_terminal_event,
        first_terminal_index,
        jnp.int32(num_steps - 1),
    )
    final_tilt_deg = tilt_deg[final_index, jnp.arange(num_envs)]

    first_failure_index = jnp.argmax(failure_event, axis=0)
    pre_fall_steps = max(1, int(math.ceil(pre_fall_window_s / ctrl_dt)))
    pre_fall_mask = (
        failed_env[None, :]
        & (step_index < first_failure_index[None, :])
        & (step_index >= first_failure_index[None, :] - pre_fall_steps)
    )
    terminal_fall_tilt_deg = tilt_deg[
        first_failure_index,
        jnp.arange(num_envs),
    ]
    time_to_fall_s = (
        first_failure_index.astype(jnp.float32) + jnp.float32(1.0)
    ) * jnp.float32(ctrl_dt)
    legacy_final_steps = min(
        max(1, int(math.ceil(WALKING_PRE_FALL_WINDOW_S / ctrl_dt))),
        num_steps,
    )
    return {
        # Backward-compatible global metrics, now sourced from preserved
        # roll/pitch telemetry instead of the post-reset state quaternion.
        "body_tilt_deg": jnp.mean(tilt_deg),
        "body_tilt_deg_peak": jnp.max(tilt_deg),
        "body_tilt_deg_final_max": jnp.max(tilt_deg[-legacy_final_steps:]),
        # Stable posture is measured only on first-episode survivors after
        # startup; failed environments receive their own pre-fall metrics.
        "walking_survivor_env_count": jnp.sum(
            survivor_env.astype(jnp.float32)
        ),
        "walking_fall_env_count": jnp.sum(failed_env.astype(jnp.float32)),
        "walking_fall_env_frac": jnp.mean(failed_env.astype(jnp.float32)),
        "walking_stable_body_tilt_deg_mean": _masked_mean(
            tilt_deg, stable_mask
        ),
        "walking_stable_body_tilt_deg_p95": _masked_percentile(
            tilt_deg, stable_mask, 95.0
        ),
        "walking_stable_body_tilt_deg_max": _masked_max(
            tilt_deg, stable_mask
        ),
        "walking_survivor_final_body_tilt_deg_max": _masked_max(
            final_tilt_deg, survivor_env
        ),
        "walking_pre_fall_body_tilt_deg_max": _masked_max(
            tilt_deg, pre_fall_mask
        ),
        "walking_fall_terminal_body_tilt_deg_max": _masked_max(
            terminal_fall_tilt_deg, failed_env
        ),
        "walking_time_to_fall_s_mean": _masked_mean(
            time_to_fall_s, failed_env
        ),
        "walking_time_to_fall_s_min": jnp.where(
            jnp.any(failed_env),
            jnp.min(jnp.where(failed_env, time_to_fall_s, jnp.inf)),
            jnp.float32(0.0),
        ),
    }


def summarize_orientation_rollout(
    quat_wxyz: jnp.ndarray,
    *,
    final_window_steps: int,
) -> Dict[str, jnp.ndarray]:
    """Aggregate yaw-independent tilt and yaw over a ``(T, N, 4)`` rollout."""
    if final_window_steps <= 0:
        raise ValueError("final_window_steps must be positive")

    tilt_deg = jnp.rad2deg(quaternion_tilt_rad(quat_wxyz))
    yaw_abs_deg = jnp.abs(jnp.rad2deg(quaternion_yaw_rad(quat_wxyz)))
    final_window_steps = min(int(final_window_steps), int(quat_wxyz.shape[0]))
    final_tilt = tilt_deg[-final_window_steps:]
    final_yaw = yaw_abs_deg[-final_window_steps:]
    return {
        "body_tilt_deg": jnp.mean(tilt_deg),
        "body_tilt_deg_peak": jnp.max(tilt_deg),
        "body_tilt_deg_final_max": jnp.max(final_tilt),
        "body_tilt_deg_peak_per_env": jnp.max(tilt_deg, axis=0),
        "body_tilt_deg_final_max_per_env": jnp.max(final_tilt, axis=0),
        "yaw_error_deg": jnp.mean(yaw_abs_deg),
        "yaw_error_deg_peak": jnp.max(yaw_abs_deg),
        "yaw_error_deg_final_max": jnp.max(final_yaw),
        "yaw_error_deg_peak_per_env": jnp.max(yaw_abs_deg, axis=0),
        "yaw_error_deg_final_max_per_env": jnp.max(final_yaw, axis=0),
    }
