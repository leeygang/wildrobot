from __future__ import annotations

import math

import numpy as np

from policy_contract.frames import FrameOps


class NumpyFrameOps(FrameOps):
    @staticmethod
    def normalize_quat_xyzw(quat_xyzw):
        return normalize_quat_xyzw(quat_xyzw)

    @staticmethod
    def rotate_vec_by_quat(quat_xyzw, vec):
        return rotate_vec_by_quat(quat_xyzw, vec)

    @staticmethod
    def gravity_local_from_quat(quat_xyzw):
        return gravity_local_from_quat(quat_xyzw)

    @staticmethod
    def angvel_heading_local(gyro_body, quat_xyzw):
        return angvel_heading_local(gyro_body, quat_xyzw)

def normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(quat))
    if n <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / n).astype(np.float32)


def _quat_conjugate(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.array([-x, -y, -z, w], dtype=np.float32)


def rotate_vec_by_quat(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    vx, vy, vz = [float(v) for v in vec]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return np.array([vpx, vpy, vpz], dtype=np.float32)


def gravity_local_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    quat_inv = _quat_conjugate(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return rotate_vec_by_quat(quat_inv, gravity_world)


def angvel_heading_local(gyro_body: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    gyro_body = np.asarray(gyro_body, dtype=np.float32).reshape(3)
    gyro_world = rotate_vec_by_quat(quat, gyro_body)
    yaw = _yaw_from_quat_xyzw(quat)
    return _world_to_heading_local(gyro_world, yaw)


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _world_to_heading_local(vec_world: np.ndarray, yaw: float) -> np.ndarray:
    vx, vy, vz = [float(v) for v in vec_world]
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    hx = c * vx - s * vy
    hy = s * vx + c * vy
    return np.array([hx, hy, vz], dtype=np.float32)
