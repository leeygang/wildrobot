"""Protocol stubs for hardware IO.

These are placeholders to align runtime integration with the policy contract.
Concrete implementations live in runtime hardware drivers (IMU, actuators, GPIO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SignalPacket:
    """Raw physical signals in actuator order.

    TODO: Replace with policy_contract Signals type once shared across runtime.
    """

    quat_xyzw: Sequence[float]
    gyro_rad_s: Sequence[float]
    joint_pos_rad: Sequence[float]
    joint_vel_rad_s: Sequence[float]
    foot_switches: Sequence[float]


class HardwareIO(Protocol):
    """Protocol-compatible hardware IO interface.

    Implementations should read sensors and apply control targets (radians)
    in spec.robot.actuator_names order.
    """

    actuator_names: Sequence[str]
    control_dt: float

    def read(self) -> SignalPacket:
        """Return latest raw signals in physical units."""

    def write_ctrl(self, ctrl_targets_rad: Sequence[float]) -> None:
        """Apply control targets (radians) in actuator_names order."""

    def close(self) -> None:
        """Release resources; safe to call multiple times."""
