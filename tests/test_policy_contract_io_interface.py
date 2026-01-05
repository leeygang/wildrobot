from __future__ import annotations

import numpy as np

from policy_contract.io import RobotIO
from policy_contract.numpy.signals import Signals


class _DummyRobotIO(RobotIO[Signals]):
    actuator_names = ["joint"]
    control_dt = 0.02

    def read(self) -> Signals:
        return Signals(
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            joint_pos_rad=np.zeros(1, dtype=np.float32),
            joint_vel_rad_s=np.zeros(1, dtype=np.float32),
            foot_switches=np.zeros(4, dtype=np.float32),
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        _ = np.asarray(ctrl_targets_rad, dtype=np.float32)


def test_robot_io_interface() -> None:
    io = _DummyRobotIO()
    sig = io.read()
    assert sig.quat_xyzw.shape == (4,)
    io.write_ctrl(np.zeros(1, dtype=np.float32))
