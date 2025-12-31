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


class CoordinateFrame(Enum):
    """Coordinate frame for position/velocity queries.

    WORLD: Global world frame (inertial, fixed)
        - Origin at world center
        - Z-up, X-forward convention
        - Used for: absolute positions, trajectory logging

    LOCAL: Body-attached frame (rotates with root body)
        - Origin at body center
        - Rotates with the body
        - Used for: body-relative sensing (accelerometers, gyros)

    HEADING_LOCAL: Root-relative frame rotated by -yaw (heading-invariant)
        - Origin at root body
        - XY plane aligned with heading direction
        - Z remains world-up
        - Used for: RL observations, AMP features, velocity commands
        - Most common for locomotion policies

    Example:
        # Get root velocity in heading-local frame (for policy observation)
        vel = cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        vx, vy, vz = vel.linear_xyz  # Tuple unpacking
    """

    WORLD = "world"
    LOCAL = "local"
    HEADING_LOCAL = "heading_local"
