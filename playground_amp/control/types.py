"""Type definitions for the Control Abstraction Layer (CAL).

This module defines enums used across CAL components.

Version: v0.11.0
"""

from enum import Enum


class ActuatorType(Enum):
    """Types of actuators supported by CAL.

    Currently only POSITION is implemented (WildRobot's servo motors).
    Future types (VELOCITY, TORQUE) can be added when needed.

    POSITION actuators:
    - ctrl = target angle in radians
    - MuJoCo's PD controller computes torque
    - Default for hobby servos (Dynamixel, etc.)
    """

    POSITION = "position"  # ctrl = target angle (rad), MuJoCo PD controller
