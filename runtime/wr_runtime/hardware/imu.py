from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ImuSample:
        """IMU reading in the robot body frame.

        Invariants:
            - quat_xyzw: orientation as (x, y, z, w), body frame → world
            - gyro_rad_s: angular velocity in body frame (rad/s)
            - timestamp_s: monotonic seconds (optional)
            - valid: True when the reading is considered good; False signals the caller to fail fast
        """

        quat_xyzw: np.ndarray  # (4,) float32
        gyro_rad_s: np.ndarray  # (3,) float32
        timestamp_s: Optional[float]
        valid: bool = True


@runtime_checkable
class Imu(Protocol):
    """Minimal IMU protocol used by HardwareRobotIO and control loop."""

    def read(self) -> ImuSample:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...
