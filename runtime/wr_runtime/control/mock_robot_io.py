"""Hardware-free ``RobotIO`` for dry-run / CI smoke tests.

Returns a deterministic upright home-pose ``Signals`` every read and records the
control targets written.  Lets ``wildrobot-run-policy`` exercise the full
read -> obs -> predict -> compose -> write loop with no servos/IMU/foot
switches attached.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from policy_contract.numpy.signals import Signals


class MockRobotIO:
    """Minimal ``RobotIO``-compatible stub.

    ``joint_pos_rad`` reports the last written ctrl target (or the home pose
    before the first write), so the loop sees a robot that perfectly tracks
    commands — adequate for a dry run / shape + plumbing smoke test.
    """

    def __init__(
        self,
        *,
        actuator_names: List[str],
        control_dt: float,
        home_q_rad: Optional[np.ndarray] = None,
    ) -> None:
        self.actuator_names = list(actuator_names)
        self.control_dt = float(control_dt)
        n = len(self.actuator_names)
        self._home = (
            np.zeros(n, dtype=np.float32)
            if home_q_rad is None
            else np.asarray(home_q_rad, dtype=np.float32).reshape(n)
        )
        self._last_ctrl = self._home.copy()
        self._prev_ctrl = self._home.copy()
        self.written: List[np.ndarray] = []
        self.closed = False

    def read(self) -> Signals:
        # Upright torso: identity quat (xyzw), no rotation.
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        joint_pos = self._last_ctrl.copy()
        joint_vel = (self._last_ctrl - self._prev_ctrl) / max(self.control_dt, 1e-6)
        foot = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # both feet planted
        return Signals(
            quat_xyzw=quat_xyzw,
            gyro_rad_s=gyro,
            joint_pos_rad=joint_pos.astype(np.float32),
            joint_vel_rad_s=joint_vel.astype(np.float32),
            foot_switches=foot,
            timestamp_s=0.0,
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        target = np.asarray(ctrl_targets_rad, dtype=np.float32).reshape(-1)
        self._prev_ctrl = self._last_ctrl
        self._last_ctrl = target.copy()
        self.written.append(target.copy())

    def close(self) -> None:
        self.closed = True
