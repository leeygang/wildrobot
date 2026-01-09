from __future__ import annotations

import jax.numpy as jnp

from policy_contract.frames import FrameOps


class JaxFrameOps(FrameOps):
    @staticmethod
    def normalize_quat_xyzw(quat_xyzw):
        return normalize_quat_xyzw(quat_xyzw)

    @staticmethod
    def quat_mul(quat_a, quat_b):
        return quat_mul(quat_a, quat_b)

    @staticmethod
    def axis_angle_to_quat(axis, angle):
        return axis_angle_to_quat(axis, angle)

    @staticmethod
    def rotate_vec_by_quat(quat_xyzw, vec):
        return rotate_vec_by_quat(quat_xyzw, vec)

    @staticmethod
    def gravity_local_from_quat(quat_xyzw):
        return gravity_local_from_quat(quat_xyzw)

    @staticmethod
    def angvel_heading_local(gyro_body, quat_xyzw):
        return angvel_heading_local(gyro_body, quat_xyzw)


def normalize_quat_xyzw(quat_xyzw: jnp.ndarray) -> jnp.ndarray:
    quat = jnp.asarray(quat_xyzw, dtype=jnp.float32).reshape(4)
    n = jnp.linalg.norm(quat)
    return jnp.where(
        n <= 1e-8,
        jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        quat / n,
    )


def _quat_conjugate(quat_xyzw: jnp.ndarray) -> jnp.ndarray:
    x, y, z, w = quat_xyzw
    return jnp.array([-x, -y, -z, w], dtype=jnp.float32)


def quat_mul(quat_a: jnp.ndarray, quat_b: jnp.ndarray) -> jnp.ndarray:
    """Quaternion multiply (wxyz)."""
    w1, x1, y1, z1 = quat_a[0], quat_a[1], quat_a[2], quat_a[3]
    w2, x2, y2, z2 = quat_b[0], quat_b[1], quat_b[2], quat_b[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z], dtype=jnp.float32)


def axis_angle_to_quat(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle (axis, angle) to quaternion (wxyz)."""
    axis_norm = axis / (jnp.linalg.norm(axis) + 1e-12)
    half = angle / 2.0
    w = jnp.cos(half)
    xyz = axis_norm * jnp.sin(half)
    return jnp.concatenate([w[None], xyz]).astype(jnp.float32)


def rotate_vec_by_quat(quat_xyzw: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    x, y, z, w = quat_xyzw
    vx, vy, vz = vec

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return jnp.array([vpx, vpy, vpz], dtype=jnp.float32)


def gravity_local_from_quat(quat_xyzw: jnp.ndarray) -> jnp.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    quat_inv = _quat_conjugate(quat)
    gravity_world = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    return rotate_vec_by_quat(quat_inv, gravity_world)


def angvel_heading_local(gyro_body: jnp.ndarray, quat_xyzw: jnp.ndarray) -> jnp.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    gyro_body = jnp.asarray(gyro_body, dtype=jnp.float32).reshape(3)
    gyro_world = rotate_vec_by_quat(quat, gyro_body)
    yaw = _yaw_from_quat_xyzw(quat)
    return _world_to_heading_local(gyro_world, yaw)


def _yaw_from_quat_xyzw(quat_xyzw: jnp.ndarray) -> jnp.ndarray:
    x, y, z, w = quat_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)


def _world_to_heading_local(vec_world: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    vx, vy, vz = vec_world
    c = jnp.cos(-yaw)
    s = jnp.sin(-yaw)
    hx = c * vx - s * vy
    hy = s * vx + c * vy
    return jnp.array([hx, hy, vz], dtype=jnp.float32)
