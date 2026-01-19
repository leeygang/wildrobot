from __future__ import annotations

from dataclasses import dataclass
from typing import List

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

    def read(self) -> Signals:
        imu_sample = self.imu.read()
        if not getattr(imu_sample, "valid", True):
            raise RuntimeError("IMU reported invalid sample (valid=False); aborting for safety")
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
