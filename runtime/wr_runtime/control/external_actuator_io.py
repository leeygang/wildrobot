"""Policy-to-hardware bridge for actuators owned by another controller."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from policy_contract.numpy.signals import Signals


class ExternalActuatorRobotIO:
    """Present full policy signals while reading and writing a hardware subset.

    Externally managed joints are reported at fixed positions with zero velocity.
    Policy targets for those joints are intentionally not forwarded to the wrapped
    locomotion ``RobotIO``.
    """

    def __init__(
        self,
        *,
        robot_io,
        policy_actuator_names: Sequence[str],
        external_home_rad: Mapping[str, float],
    ) -> None:
        self._robot_io = robot_io
        self.actuator_names = [str(name) for name in policy_actuator_names]
        self.hardware_actuator_names = [
            str(name) for name in getattr(robot_io, "actuator_names", ())
        ]
        self.control_dt = float(getattr(robot_io, "control_dt", 0.0))
        self._external_home_rad = {
            str(name): float(value) for name, value in external_home_rad.items()
        }

        if len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("policy actuator names must be unique")
        if len(set(self.hardware_actuator_names)) != len(self.hardware_actuator_names):
            raise ValueError("hardware actuator names must be unique")

        policy_set = set(self.actuator_names)
        hardware_set = set(self.hardware_actuator_names)
        external_set = set(self._external_home_rad)
        overlap = sorted(hardware_set & external_set)
        missing = sorted(policy_set - hardware_set - external_set)
        unknown = sorted((hardware_set | external_set) - policy_set)
        if overlap or missing or unknown:
            raise ValueError(
                "invalid external-actuator mapping: "
                f"overlap={overlap} missing={missing} unknown={unknown}"
            )

        policy_index = {name: idx for idx, name in enumerate(self.actuator_names)}
        self._hardware_to_policy = np.asarray(
            [policy_index[name] for name in self.hardware_actuator_names], dtype=np.intp
        )
        self._external_policy_indices = {
            policy_index[name]: np.float32(value)
            for name, value in self._external_home_rad.items()
        }

    def read(self) -> Signals:
        signals = self._robot_io.read()
        hardware_pos = np.asarray(signals.joint_pos_rad, dtype=np.float32).reshape(-1)
        hardware_vel = np.asarray(signals.joint_vel_rad_s, dtype=np.float32).reshape(-1)
        hardware_count = len(self.hardware_actuator_names)
        if hardware_pos.size != hardware_count or hardware_vel.size != hardware_count:
            raise ValueError(
                "hardware signal size does not match actuator names: "
                f"pos={hardware_pos.size} vel={hardware_vel.size} "
                f"actuators={hardware_count}"
            )

        joint_pos = np.zeros(len(self.actuator_names), dtype=np.float32)
        joint_vel = np.zeros(len(self.actuator_names), dtype=np.float32)
        joint_pos[self._hardware_to_policy] = hardware_pos
        joint_vel[self._hardware_to_policy] = hardware_vel
        for idx, value in self._external_policy_indices.items():
            joint_pos[idx] = value

        return Signals(
            quat_xyzw=np.asarray(signals.quat_xyzw, dtype=np.float32),
            gyro_rad_s=np.asarray(signals.gyro_rad_s, dtype=np.float32),
            joint_pos_rad=joint_pos,
            joint_vel_rad_s=joint_vel,
            foot_switches=np.asarray(signals.foot_switches, dtype=np.float32),
            timestamp_s=signals.timestamp_s,
        )

    def write_ctrl(self, ctrl_targets_rad) -> None:
        target = np.asarray(ctrl_targets_rad, dtype=np.float32).reshape(-1)
        if target.size != len(self.actuator_names):
            raise ValueError(
                f"policy target has {target.size} elements, expected "
                f"{len(self.actuator_names)}"
            )
        self._robot_io.write_ctrl(target[self._hardware_to_policy])

    def close(self) -> None:
        self._robot_io.close()

    def __getattr__(self, name: str):
        return getattr(self._robot_io, name)
