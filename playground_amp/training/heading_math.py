"""Shared heading-local math for training and visualization."""

from __future__ import annotations

import jax.numpy as jp
import numpy as np


def heading_sin_cos_from_quat_jax(quat_wxyz: jp.ndarray) -> jp.ndarray:
    """Return [sin(yaw), cos(yaw)] from quaternion (wxyz)."""
    quat = quat_wxyz / (jp.linalg.norm(quat_wxyz) + 1e-8)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    norm = jp.sqrt(siny_cosp * siny_cosp + cosy_cosp * cosy_cosp + 1e-8)
    sin_yaw = siny_cosp / norm
    cos_yaw = cosy_cosp / norm
    return jp.array([sin_yaw, cos_yaw])


def heading_local_from_world_jax(
    vec_world: jp.ndarray, sin_yaw: jp.ndarray, cos_yaw: jp.ndarray
) -> jp.ndarray:
    """Rotate world-frame vec into heading-local frame (R_z(-yaw))."""
    vx, vy, vz = vec_world[0], vec_world[1], vec_world[2]
    vx_heading = cos_yaw * vx + sin_yaw * vy
    vy_heading = -sin_yaw * vx + cos_yaw * vy
    vz_heading = vz
    return jp.array([vx_heading, vy_heading, vz_heading])


def heading_local_linvel_fd_from_pos_jax(
    curr_root_pos: jp.ndarray,
    prev_root_pos: jp.ndarray,
    quat_wxyz: jp.ndarray,
    dt: float,
) -> jp.ndarray:
    """Finite-diff linear velocity in heading-local frame."""
    linvel_world = (curr_root_pos - prev_root_pos) / dt
    sin_yaw, cos_yaw = heading_sin_cos_from_quat_jax(quat_wxyz)
    return heading_local_from_world_jax(linvel_world, sin_yaw, cos_yaw)


def heading_local_angvel_fd_from_quat_jax(
    curr_quat: jp.ndarray,
    prev_quat: jp.ndarray,
    dt: float,
) -> jp.ndarray:
    """Finite-diff angular velocity in heading-local frame."""
    w_prev, x_prev, y_prev, z_prev = (
        prev_quat[0],
        prev_quat[1],
        prev_quat[2],
        prev_quat[3],
    )
    q_prev_conj = jp.array([w_prev, -x_prev, -y_prev, -z_prev])

    w_curr, x_curr, y_curr, z_curr = (
        curr_quat[0],
        curr_quat[1],
        curr_quat[2],
        curr_quat[3],
    )

    w_delta = (
        w_curr * q_prev_conj[0]
        - x_curr * q_prev_conj[1]
        - y_curr * q_prev_conj[2]
        - z_curr * q_prev_conj[3]
    )
    x_delta = (
        w_curr * q_prev_conj[1]
        + x_curr * q_prev_conj[0]
        + y_curr * q_prev_conj[3]
        - z_curr * q_prev_conj[2]
    )
    y_delta = (
        w_curr * q_prev_conj[2]
        - x_curr * q_prev_conj[3]
        + y_curr * q_prev_conj[0]
        + z_curr * q_prev_conj[1]
    )
    z_delta = (
        w_curr * q_prev_conj[3]
        + x_curr * q_prev_conj[2]
        - y_curr * q_prev_conj[1]
        + z_curr * q_prev_conj[0]
    )

    vec = jp.array([x_delta, y_delta, z_delta])
    vec_norm = jp.linalg.norm(vec) + 1e-8
    angle = 2.0 * jp.arctan2(vec_norm, jp.abs(w_delta))
    angle = jp.where(w_delta < 0, -angle, angle)
    axis = vec / vec_norm
    rotvec = angle * axis
    angvel_world = rotvec / dt

    sin_yaw, cos_yaw = heading_sin_cos_from_quat_jax(curr_quat)
    return heading_local_from_world_jax(angvel_world, sin_yaw, cos_yaw)


def pitch_roll_from_quat_jax(quat_wxyz: jp.ndarray) -> jp.ndarray:
    """Return [pitch, roll] from quaternion (wxyz)."""
    quat = quat_wxyz / (jp.linalg.norm(quat_wxyz) + 1e-8)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    sinp = 2.0 * (w * y - z * x)
    pitch = jp.arcsin(jp.clip(sinp, -1.0, 1.0))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    return jp.array([pitch, roll])


def heading_sin_cos_from_quat_np(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Return (sin(yaw), cos(yaw)) from quaternion (wxyz)."""
    quat = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    norm = np.sqrt(siny_cosp * siny_cosp + cosy_cosp * cosy_cosp + 1e-8)
    sin_yaw = siny_cosp / norm
    cos_yaw = cosy_cosp / norm
    return float(sin_yaw), float(cos_yaw)


def heading_local_from_world_np(
    vec_world: np.ndarray, sin_yaw: float, cos_yaw: float
) -> np.ndarray:
    """Rotate world-frame vec into heading-local frame (R_z(-yaw))."""
    vx, vy, vz = vec_world
    return np.array(
        [
            cos_yaw * vx + sin_yaw * vy,
            -sin_yaw * vx + cos_yaw * vy,
            vz,
        ],
        dtype=np.float32,
    )


def heading_local_linvel_fd_from_pos_np(
    curr_root_pos: np.ndarray,
    prev_root_pos: np.ndarray,
    quat_wxyz: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Finite-diff linear velocity in heading-local frame."""
    linvel_world = (curr_root_pos - prev_root_pos) / dt
    sin_yaw, cos_yaw = heading_sin_cos_from_quat_np(quat_wxyz)
    return heading_local_from_world_np(linvel_world, sin_yaw, cos_yaw)


def heading_local_angvel_fd_from_quat_np(
    curr_quat: np.ndarray,
    prev_quat: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Finite-diff angular velocity in heading-local frame."""
    w_prev, x_prev, y_prev, z_prev = prev_quat
    q_prev_conj = np.array([w_prev, -x_prev, -y_prev, -z_prev], dtype=np.float32)

    w_curr, x_curr, y_curr, z_curr = curr_quat
    w_delta = (
        w_curr * q_prev_conj[0]
        - x_curr * q_prev_conj[1]
        - y_curr * q_prev_conj[2]
        - z_curr * q_prev_conj[3]
    )
    x_delta = (
        w_curr * q_prev_conj[1]
        + x_curr * q_prev_conj[0]
        + y_curr * q_prev_conj[3]
        - z_curr * q_prev_conj[2]
    )
    y_delta = (
        w_curr * q_prev_conj[2]
        - x_curr * q_prev_conj[3]
        + y_curr * q_prev_conj[0]
        + z_curr * q_prev_conj[1]
    )
    z_delta = (
        w_curr * q_prev_conj[3]
        + x_curr * q_prev_conj[2]
        - y_curr * q_prev_conj[1]
        + z_curr * q_prev_conj[0]
    )

    vec = np.array([x_delta, y_delta, z_delta], dtype=np.float32)
    vec_norm = np.linalg.norm(vec) + 1e-8
    angle = 2.0 * np.arctan2(vec_norm, np.abs(w_delta))
    if w_delta < 0:
        angle = -angle
    axis = vec / vec_norm
    rotvec = angle * axis
    angvel_world = rotvec / dt

    sin_yaw, cos_yaw = heading_sin_cos_from_quat_np(curr_quat)
    return heading_local_from_world_np(angvel_world, sin_yaw, cos_yaw)


def pitch_roll_from_quat_np(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Return (pitch, roll) from quaternion (wxyz)."""
    quat = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)
    w, x, y, z = quat
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return float(pitch), float(roll)
