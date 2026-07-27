from __future__ import annotations

from typing import Protocol


class FrameOps(Protocol):
    def normalize_quat_wxyz(self, quat_wxyz):
        ...

    def quat_mul(self, quat_a, quat_b):
        ...

    def axis_angle_to_quat(self, axis, angle):
        ...

    def rotate_vec_by_quat(self, quat_wxyz, vec):
        ...

    def gravity_local_from_quat(self, quat_wxyz):
        ...

    def angvel_heading_local(self, gyro_body, quat_wxyz):
        ...
