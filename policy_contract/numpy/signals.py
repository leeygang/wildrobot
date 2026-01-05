from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Signals:
    quat_xyzw: np.ndarray
    gyro_rad_s: np.ndarray
    joint_pos_rad: np.ndarray
    joint_vel_rad_s: np.ndarray
    foot_switches: np.ndarray
    timestamp_s: Optional[float] = None
