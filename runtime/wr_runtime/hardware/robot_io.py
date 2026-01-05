from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from policy_contract.io import RobotIO
from policy_contract.numpy.signals import Signals

from .bno085 import BNO085IMU
from .foot_switches import FootSwitches
from .hiwonder_actuators import HiwonderBoardActuators


@dataclass
class HardwareRobotIO(RobotIO[Signals]):
    actuator_names: List[str]
    control_dt: float
    actuators: HiwonderBoardActuators
    imu: BNO085IMU
    foot_switches: FootSwitches

    def read(self) -> Signals:
        imu_sample = self.imu.read()
        joint_pos = self.actuators.get_positions_rad()
        if joint_pos is None:
            joint_pos = np.zeros(len(self.actuator_names), dtype=np.float32)

        joint_vel = self.actuators.estimate_velocities_rad_s(self.control_dt)
        foot_sample = self.foot_switches.read()
        foot = np.array(foot_sample.switches, dtype=np.float32)

        return Signals(
            quat_xyzw=np.asarray(imu_sample.quat_xyzw, dtype=np.float32),
            gyro_rad_s=np.asarray(imu_sample.gyro_rad_s, dtype=np.float32),
            joint_pos_rad=np.asarray(joint_pos, dtype=np.float32),
            joint_vel_rad_s=np.asarray(joint_vel, dtype=np.float32),
            foot_switches=foot,
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        self.actuators.set_targets_rad(np.asarray(ctrl_targets_rad, dtype=np.float32))

    def close(self) -> None:
        try:
            self.foot_switches.close()
        finally:
            try:
                self.imu.close()
            finally:
                self.actuators.disable()
                self.actuators.close()
