from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np

from policy_contract.io import RobotIO
from policy_contract.numpy.frames import normalize_quat_xyzw
from policy_contract.numpy.signals import Signals

from .actuators import Actuators
from .foot_switches import FootSwitches
from .imu import Imu


@dataclass
class HardwareRobotIO(RobotIO[Signals]):
    actuator_names: List[str]
    control_dt: float
    actuators: Actuators
    imu: Imu
    foot_switches: FootSwitches
    max_consecutive_invalid_imu: int = 5

    _imu_invalid_consecutive: int = 0
    _last_valid_imu_sample: Optional[object] = None
    _last_imu_warn_time_s: float = 0.0

    def read(self) -> Signals:
        imu_sample = self.imu.read()
        if not getattr(imu_sample, "valid", True):
            self._imu_invalid_consecutive += 1
            if self._last_valid_imu_sample is None:
                raise RuntimeError("IMU reported invalid sample before any valid sample was observed; aborting for safety")
            if self._imu_invalid_consecutive > int(self.max_consecutive_invalid_imu):
                extra = ""
                if hasattr(self.imu, "error_count") and hasattr(self.imu, "last_error"):
                    try:
                        extra = f" (imu_error_count={getattr(self.imu, 'error_count')}, last_error={getattr(self.imu, 'last_error')})"
                    except Exception:
                        extra = ""
                raise RuntimeError(
                    "IMU reported invalid sample for too long; aborting for safety. "
                    f"invalid_consecutive={self._imu_invalid_consecutive} max={self.max_consecutive_invalid_imu}{extra}"
                )

            now = time.monotonic()
            if now - self._last_imu_warn_time_s > 1.0:
                self._last_imu_warn_time_s = now
                print(
                    "Warning: IMU sample invalid; reusing last valid sample "
                    f"(invalid_consecutive={self._imu_invalid_consecutive}/{self.max_consecutive_invalid_imu}).",
                    flush=True,
                )
            imu_sample = self._last_valid_imu_sample
        else:
            self._imu_invalid_consecutive = 0
            self._last_valid_imu_sample = imu_sample

        joint_pos = self.actuators.get_positions_rad()
        if joint_pos is None:
            port = getattr(self.actuators, "port", "unknown")
            baudrate = getattr(self.actuators, "baudrate", "unknown")
            last_error = getattr(self.actuators, "_last_error", None)
            err_msg = f"; last_error={repr(last_error)}" if last_error is not None else ""
            raise RuntimeError(
                f"Failed to read joint positions for {self.actuator_names} on port {port} baud {baudrate}{err_msg}"
            )
        joint_vel = self.actuators.estimate_velocities_rad_s(self.control_dt)
        foot_sample = self.foot_switches.read()
        foot = np.array(foot_sample.switches, dtype=np.float32)
        quat_xyzw = normalize_quat_xyzw(np.asarray(imu_sample.quat_xyzw, dtype=np.float32))

        return Signals(
            quat_xyzw=quat_xyzw,
            gyro_rad_s=np.asarray(imu_sample.gyro_rad_s, dtype=np.float32),
            joint_pos_rad=np.asarray(joint_pos, dtype=np.float32),
            joint_vel_rad_s=np.asarray(joint_vel, dtype=np.float32),
            foot_switches=foot,
            timestamp_s=float(getattr(imu_sample, "timestamp_s", 0.0) or 0.0),
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        self.actuators.set_targets_rad(np.asarray(ctrl_targets_rad, dtype=np.float32), move_time_ms=None)

    def close(self) -> None:
        try:
            self.foot_switches.close()
        finally:
            try:
                self.imu.close()
            finally:
                self.actuators.disable()
                self.actuators.close()
