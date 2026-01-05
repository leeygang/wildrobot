from __future__ import annotations

from typing import Protocol


class FrameOps(Protocol):
    def normalize_quat_xyzw(self, quat_xyzw):
        ...

    def rotate_vec_by_quat(self, quat_xyzw, vec):
        ...

    def gravity_local_from_quat(self, quat_xyzw):
        ...

    def angvel_heading_local(self, gyro_body, quat_xyzw):
        ...
