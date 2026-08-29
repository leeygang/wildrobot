"""Standing-orientation metrics shared by deterministic evaluators."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp


STANDING_FINAL_WINDOW_S = 0.5


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
