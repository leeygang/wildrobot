"""Shared locomotion log schema for sim-vs-real replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from control.references.locomotion_contract import (
    LocomotionActionContract,
    LocomotionCommand,
    LocomotionReferenceState,
)


@dataclass(frozen=True)
class RobotReplayState:
    """Robot state fields needed for replay-oriented sim-vs-real comparison."""

    quat_xyzw: Tuple[float, float, float, float]
    gyro_rad_s: Tuple[float, float, float]
    joint_pos_rad: Tuple[float, ...]
    joint_vel_rad_s: Tuple[float, ...]
    foot_switches: Tuple[float, float, float, float]
    pelvis_height_m: float

    def __post_init__(self) -> None:
        if len(self.quat_xyzw) != 4:
            raise ValueError("quat_xyzw must have length 4")
        if len(self.gyro_rad_s) != 3:
            raise ValueError("gyro_rad_s must have length 3")
        if len(self.foot_switches) != 4:
            raise ValueError("foot_switches must have length 4")
        if len(self.joint_pos_rad) != len(self.joint_vel_rad_s):
            raise ValueError("joint_pos_rad and joint_vel_rad_s lengths must match")


@dataclass(frozen=True)
class LocomotionLogRecord:
    """Single-step locomotion log schema shared by sim and runtime."""

    command: LocomotionCommand
    reference: LocomotionReferenceState
    action: LocomotionActionContract
    replay_state: RobotReplayState
    velocity_cmd: float
    yaw_rate_cmd: float

    def __post_init__(self) -> None:
        if abs(float(self.velocity_cmd) - float(self.command.forward_speed_mps)) > 1e-6:
            raise ValueError("velocity_cmd must match command.forward_speed_mps")
        if abs(float(self.yaw_rate_cmd) - float(self.command.yaw_rate_rps)) > 1e-6:
            raise ValueError("yaw_rate_cmd must match command.yaw_rate_rps")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def required_replay_log_fields() -> Tuple[str, ...]:
    """Required array keys for replay datasets."""
    return (
        "quat_xyzw",
        "gyro_rad_s",
        "joint_pos_rad",
        "joint_vel_rad_s",
        "foot_switches",
        "velocity_cmd",
        "yaw_rate_cmd",
    )
