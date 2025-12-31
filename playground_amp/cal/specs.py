"""Specification dataclasses for the Control Abstraction Layer (CAL).

This module defines:
- JointSpec: Joint specification (range, default, symmetry)
- ActuatorSpec: Actuator specification (ctrl range, force limits, motor specs)
- ControlCommand: Control command with trajectory support for Sim2Real
- Pose3D: 3D pose with position and orientation (JAX, with NumPy factory)
- Velocity3D: 3D velocity with linear and angular components (JAX, with NumPy factory)
- FootSpec: Foot body/geom specification for contact detection
- RootSpec: Root body specification with IMU sensors (BNO085)

Array Type Design (Factory Pattern):
    Pose3D and Velocity3D are JAX-native for JIT-compiled training.
    For visualization with NumPy arrays, use the from_numpy() factory methods
    which explicitly convert to JAX arrays.

    Training (JAX - default):
        pose = cal.get_root_pose(data)  # Returns JAX arrays

    Visualization (NumPy → JAX):
        pose = Pose3D.from_numpy(
            position=mj_data.qpos[:3],
            orientation=mj_data.qpos[3:7],
            frame=CoordinateFrame.WORLD,
        )

Version: v0.11.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from playground_amp.cal.types import ActuatorType, CoordinateFrame


@dataclass(frozen=True)
class JointSpec:
    """Joint specification - single source of truth for joint properties.

    This replaces scattered joint range access throughout the codebase.

    Attributes:
        name: Joint name (e.g., "left_hip_pitch")
        joint_id: MuJoCo joint ID
        qpos_addr: Address in qpos array
        qvel_addr: Address in qvel array (dof address)
        range_min: Minimum joint angle in radians
        range_max: Maximum joint angle in radians
        default_pos: Default position (from keyframe or qpos0)
        mirror_sign: +1.0 for same-sign joints, -1.0 for inverted (mirrored) ranges
        symmetry_pair: Name of paired joint (e.g., "right_hip_pitch" for "left_hip_pitch")
    """

    name: str
    joint_id: int
    qpos_addr: int
    qvel_addr: int

    # Physical range (from URDF/XML)
    range_min: float  # radians
    range_max: float  # radians

    # Default position (from keyframe or qpos0)
    default_pos: float

    # Symmetry handling
    mirror_sign: float = 1.0  # +1.0 for same-sign, -1.0 for inverted
    symmetry_pair: Optional[str] = None  # e.g., "left_hip_pitch" ↔ "right_hip_pitch"

    @property
    def range_center(self) -> float:
        """Center of joint range."""
        return (self.range_min + self.range_max) / 2

    @property
    def range_span(self) -> float:
        """Half-width of joint range (for normalization)."""
        return (self.range_max - self.range_min) / 2


@dataclass(frozen=True)
class ActuatorSpec:
    """Actuator specification - defines control semantics.

    Naming Convention:
    - action: Normalized policy output in [-1, 1] (what NN produces)
    - ctrl: MuJoCo control signal in physical units (what simulator consumes)

    For POSITION actuators (WildRobot's exclusive type):
    - ctrl_range == joint_range (target angle in radians)
    - Use joint_spec.range_* for all normalization

    Attributes:
        name: Actuator name (usually same as joint name)
        actuator_id: MuJoCo actuator ID (index in ctrl array)
        joint_spec: Associated JointSpec (single source of truth for ranges)
        actuator_type: Type of actuator (POSITION, etc.)
        force_range: Force/torque limits in Nm
        max_velocity: Maximum angular velocity in rad/s (from motor datasheet)
        kp: Position gain (for position actuators)
        kv: Velocity gain (for position actuators)
    """

    name: str
    actuator_id: int
    joint_spec: JointSpec

    # Actuator type determines how ctrl is interpreted
    actuator_type: ActuatorType = ActuatorType.POSITION

    # Force/torque limits (for reward normalization, safety)
    force_range: Tuple[float, float] = (-10.0, 10.0)

    # Velocity limit (from motor datasheet)
    # Default: 10.0 rad/s (~95 RPM) - conservative estimate for small servos
    max_velocity: float = 10.0  # rad/s

    # PD gains (only applicable for POSITION type)
    kp: Optional[float] = None
    kv: Optional[float] = None  # Note: kv not kd (MuJoCo convention)

    @property
    def ctrl_range(self) -> Tuple[float, float]:
        """Control range - derived from joint_spec for position actuators."""
        return (self.joint_spec.range_min, self.joint_spec.range_max)

    @property
    def ctrl_center(self) -> float:
        """Center of control range - delegates to joint_spec."""
        return self.joint_spec.range_center

    @property
    def ctrl_span(self) -> float:
        """Half-width of control range - delegates to joint_spec."""
        return self.joint_spec.range_span

    @property
    def force_limit(self) -> float:
        """Maximum absolute force/torque (for normalization)."""
        return max(abs(self.force_range[0]), abs(self.force_range[1]))


@dataclass
class ControlCommand:
    """Complete control command with trajectory information.

    CRITICAL FOR SIM2REAL:
    In simulation, MuJoCo's position actuator instantly starts moving toward
    the target at whatever speed the PD dynamics allow. But real motors need:
    - Target position (where to go)
    - Target velocity (how fast to get there)
    - Duration/time (when to arrive)

    This dataclass bridges the sim-real gap by explicitly specifying trajectory.

    Simulation vs Real Hardware:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ SIMULATION (MuJoCo)                                                     │
    │ - ctrl = target_position only                                           │
    │ - PD controller computes: torque = kp*(target - qpos) - kv*qvel        │
    │ - Velocity is emergent from dynamics (not commanded)                   │
    │ - Time to reach target depends on dynamics, not specified              │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ REAL HARDWARE (Servo Motors)                                            │
    │ - Need target_position + target_velocity (or duration)                  │
    │ - Motor controller interpolates trajectory                              │
    │ - Velocity profile: trapezoidal, S-curve, or linear                    │
    │ - Time to reach is explicitly controlled                               │
    └─────────────────────────────────────────────────────────────────────────┘

    Usage:
        # In simulation - velocity is optional (computed from dynamics)
        cmd = cal.policy_action_to_command(action)
        ctrl = cmd.target_position  # Only this goes to MuJoCo

        # In real hardware - velocity is required
        cmd = cal.policy_action_to_command(action, include_velocity=True)
        motor.set_position(cmd.target_position, velocity=cmd.target_velocity)

    Attributes:
        target_position: Target positions in radians, shape (num_actuators,)
        target_velocity: Target velocities in rad/s (for Sim2Real), optional
        duration: Time to reach target in seconds (alternative to velocity)
        max_acceleration: Acceleration limits in rad/s² (for smooth motion)
    """

    # Target position (always required)
    target_position: jnp.ndarray  # shape (num_actuators,), in radians

    # Target velocity (for Sim2Real trajectory control)
    target_velocity: Optional[jnp.ndarray] = None  # shape (num_actuators,), rad/s

    # Duration to reach target (alternative to velocity)
    duration: Optional[float] = None  # seconds

    # Acceleration limits (for smooth motion)
    max_acceleration: Optional[jnp.ndarray] = None  # rad/s²


# =============================================================================
# Generic 3D Types (Reusable for root, feet, any body)
# =============================================================================


@dataclass(frozen=True)
class Pose3D:
    """General 3D pose with coordinate frame (JAX-native).

    Self-documenting: frame is stored with the data.
    Reusable for root, feet, or any body.

    Array Type Design (Factory Pattern):
        - Default: JAX arrays for JIT-compiled training
        - Use from_numpy() for visualization with MuJoCo's NumPy arrays

    Attributes:
        position: (3,) [x, y, z] in meters (JAX array)
        orientation: (4,) quaternion [qw, qx, qy, qz] (JAX array)
        frame: Coordinate frame this pose is expressed in

    Example (JAX - training):
        pose = cal.get_root_pose(data, frame=CoordinateFrame.WORLD)
        height = pose.height
        roll, pitch, yaw = pose.euler_angles()

    Example (NumPy - visualization):
        pose = Pose3D.from_numpy(
            position=mj_data.qpos[:3],
            orientation=mj_data.qpos[3:7],
            frame=CoordinateFrame.WORLD,
        )
        sin_yaw, cos_yaw = pose.heading_sincos
    """

    position: jnp.ndarray  # (3,) [x, y, z] in meters
    orientation: jnp.ndarray  # (4,) quaternion [qw, qx, qy, qz]
    frame: CoordinateFrame  # Frame this pose is expressed in

    @classmethod
    def from_numpy(
        cls,
        position: np.ndarray,
        orientation: np.ndarray,
        frame: CoordinateFrame,
    ) -> Pose3D:
        """Create Pose3D from NumPy arrays (for visualization).

        Explicitly converts NumPy arrays to JAX arrays. Use this when working
        with MuJoCo's native NumPy arrays in visualization code.

        Args:
            position: (3,) position array (NumPy)
            orientation: (4,) quaternion array (NumPy)
            frame: Coordinate frame

        Returns:
            Pose3D with JAX arrays

        Example:
            pose = Pose3D.from_numpy(
                position=mj_data.qpos[:3],
                orientation=mj_data.qpos[3:7],
                frame=CoordinateFrame.WORLD,
            )
        """
        return cls(
            position=jnp.array(position),
            orientation=jnp.array(orientation),
            frame=frame,
        )

    @property
    def height(self) -> float:
        """Z position (height above ground)."""
        return self.position[2]

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Position as (x, y, z) tuple for unpacking."""
        return self.position[0], self.position[1], self.position[2]

    def euler_angles(self) -> tuple[float, float, float]:
        """Extract Euler angles (ZYX convention) from orientation quaternion.

        Returns:
            (roll, pitch, yaw) in radians
            - roll: rotation around X-axis (body lean left/right)
            - pitch: rotation around Y-axis (body lean forward/backward)
            - yaw: rotation around Z-axis (heading direction)

        Raises:
            ValueError: If pose is not in WORLD frame

        Note:
            WORLD frame is required because pitch/roll have physical meaning
            only when measured relative to gravity (world Z-up). The math works
            on any quaternion, but the semantic interpretation ("how tilted is
            the robot?") requires world-relative orientation.

        Example:
            roll, pitch, yaw = pose.euler_angles()
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"euler_angles requires WORLD frame, got {self.frame}. "
                f"Pitch/roll are only meaningful relative to gravity (world Z-up)."
            )
        quat = self.orientation / (jnp.linalg.norm(self.orientation) + 1e-8)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Roll (X-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))

        # Yaw (Z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @property
    def heading_sincos(self) -> tuple[float, float]:
        """Heading (yaw) angle as (sin, cos) components.

        Returns sin/cos instead of angle because:
        - Continuous (no discontinuity at ±π)
        - Better for NN observations and frame transformations

        Naming: "heading" for consistency with HEADING_LOCAL frame
        and to_heading_local() method.

        Returns:
            (sin_heading, cos_heading) - heading direction components

        Raises:
            ValueError: If pose is not in WORLD frame (via euler_angles)

        Example:
            sin_yaw, cos_yaw = pose.heading_sincos
        """
        _, _, yaw = self.euler_angles()
        return jnp.sin(yaw), jnp.cos(yaw)

    @property
    def gravity_vector(self) -> jnp.ndarray:
        """Get gravity vector in body (local) frame.

        Rotates world gravity [0, 0, -1] into the local body frame using
        the pose's orientation quaternion. This matches CAL.get_gravity_vector().

        Returns:
            (3,) gravity direction in local frame, normalized

        Raises:
            ValueError: If pose is not in WORLD frame

        Example:
            pose = Pose3D.from_numpy(mj_data.qpos[:3], mj_data.qpos[3:7], CoordinateFrame.WORLD)
            gravity = pose.gravity_vector  # e.g., [0.01, 0.0, -0.99]
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"gravity_vector requires WORLD frame pose, got {self.frame}. "
                f"Gravity is only meaningful relative to world Z-up."
            )
        # Rotate world gravity [0, 0, -1] into local frame using quat inverse
        quat = self.orientation
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Rotation matrix from quaternion (transposed for inverse rotation)
        # This matches CAL._rotate_vector_by_quat_inverse()
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y + w * z)
        r02 = 2 * (x * z - w * y)
        r10 = 2 * (x * y - w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z + w * x)
        r20 = 2 * (x * z + w * y)
        r21 = 2 * (y * z - w * x)
        r22 = 1 - 2 * (x * x + y * y)

        # world_gravity = [0, 0, -1], so we only need the third column
        return jnp.array([r20 * (-1), r21 * (-1), r22 * (-1)])

    def to_heading_local(self, vec_world: jnp.ndarray) -> jnp.ndarray:
        """Transform a world-frame vector to heading-local frame.

        Heading-local orientation: world frame rotated by -yaw (heading direction is +X).
        Note: This is a vector transform only. For positions, subtract the root
        position before applying heading-local rotation.

        Args:
            vec_world: 3D vector in world frame (3,)

        Returns:
            Vector in heading-local frame (3,)

        Raises:
            ValueError: If pose is not in WORLD frame

        Example:
            pose = cal.get_root_pose(data)
            world_vel = jnp.array([1.0, 0.5, 0.0])
            local_vel = pose.to_heading_local(world_vel)
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"to_heading_local requires WORLD frame pose, got {self.frame}"
            )
        sin_yaw, cos_yaw = self.heading_sincos
        vx, vy, vz = vec_world[0], vec_world[1], vec_world[2]
        return jnp.array(
            [
                cos_yaw * vx + sin_yaw * vy,
                -sin_yaw * vx + cos_yaw * vy,
                vz,
            ]
        )

    def linvel_fd(
        self,
        prev: Pose3D,
        dt: float,
        frame: CoordinateFrame,  # REQUIRED - no default to force explicit intent
    ) -> jnp.ndarray:
        """Compute finite-difference linear velocity.

        Calculates velocity from position change between two poses.

        Args:
            prev: Previous pose (at time t - dt)
            dt: Time step in seconds
            frame: REQUIRED. Coordinate frame for output velocity

        Returns:
            Linear velocity (3,) [vx, vy, vz] m/s in specified frame

        Raises:
            ValueError: If either pose is not in WORLD frame

        Example:
            curr_pose = Pose3D.from_numpy(curr_pos, curr_quat, CoordinateFrame.WORLD)
            prev_pose = Pose3D.from_numpy(prev_pos, prev_quat, CoordinateFrame.WORLD)
            linvel = curr_pose.linvel_fd(prev_pose, dt=0.02, frame=CoordinateFrame.HEADING_LOCAL)
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"linvel_fd requires WORLD frame for current pose, got {self.frame}"
            )
        if prev.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"linvel_fd requires WORLD frame for previous pose, got {prev.frame}"
            )
        linvel_world = (self.position - prev.position) / dt
        if frame == CoordinateFrame.HEADING_LOCAL:
            return self.to_heading_local(linvel_world)
        return linvel_world

    def angvel_fd(
        self,
        prev: Pose3D,
        dt: float,
        frame: CoordinateFrame,  # REQUIRED - no default to force explicit intent
    ) -> jnp.ndarray:
        """Compute finite-difference angular velocity.

        Calculates angular velocity from quaternion change between two poses.
        Uses quaternion derivative: ω = 2 * q_delta / dt, where q_delta = q_curr * q_prev^-1

        Args:
            prev: Previous pose (at time t - dt)
            dt: Time step in seconds
            frame: REQUIRED. Coordinate frame for output velocity

        Returns:
            Angular velocity (3,) [wx, wy, wz] rad/s in specified frame

        Raises:
            ValueError: If either pose is not in WORLD frame

        Example:
            curr_pose = Pose3D.from_numpy(curr_pos, curr_quat, CoordinateFrame.WORLD)
            prev_pose = Pose3D.from_numpy(prev_pos, prev_quat, CoordinateFrame.WORLD)
            angvel = curr_pose.angvel_fd(prev_pose, dt=0.02, frame=CoordinateFrame.HEADING_LOCAL)
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"angvel_fd requires WORLD frame for current pose, got {self.frame}"
            )
        if prev.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"angvel_fd requires WORLD frame for previous pose, got {prev.frame}"
            )
        # Quaternion conjugate of prev (inverse for unit quaternions)
        w_prev, x_prev, y_prev, z_prev = (
            prev.orientation[0],
            prev.orientation[1],
            prev.orientation[2],
            prev.orientation[3],
        )
        q_prev_conj = jnp.array([w_prev, -x_prev, -y_prev, -z_prev])

        # Quaternion multiplication: q_delta = q_curr * q_prev_conj
        w_curr, x_curr, y_curr, z_curr = (
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
            self.orientation[3],
        )
        w_delta = (
            w_curr * q_prev_conj[0]
            - x_curr * q_prev_conj[1]
            - y_curr * q_prev_conj[2]
            - z_curr * q_prev_conj[3]
        )
        x_delta = (
            w_curr * q_prev_conj[1]
            + x_curr * q_prev_conj[0]
            + y_curr * q_prev_conj[3]
            - z_curr * q_prev_conj[2]
        )
        y_delta = (
            w_curr * q_prev_conj[2]
            - x_curr * q_prev_conj[3]
            + y_curr * q_prev_conj[0]
            + z_curr * q_prev_conj[1]
        )
        z_delta = (
            w_curr * q_prev_conj[3]
            + x_curr * q_prev_conj[2]
            - y_curr * q_prev_conj[1]
            + z_curr * q_prev_conj[0]
        )

        # Convert quaternion delta to axis-angle, then to angular velocity
        vec = jnp.array([x_delta, y_delta, z_delta])
        vec_norm = jnp.linalg.norm(vec) + 1e-8
        angle = 2.0 * jnp.arctan2(vec_norm, jnp.abs(w_delta))
        # Handle negative w (quaternion double-cover)
        angle = jnp.where(w_delta < 0, -angle, angle)
        axis = vec / vec_norm
        rotvec = angle * axis
        angvel_world = rotvec / dt

        if frame == CoordinateFrame.HEADING_LOCAL:
            return self.to_heading_local(angvel_world)
        return angvel_world

    def velocity_fd(
        self,
        prev: Pose3D,
        dt: float,
        frame: CoordinateFrame = CoordinateFrame.HEADING_LOCAL,
    ) -> Velocity3D:
        """Compute finite-difference velocity between two poses.

        Combines linvel_fd and angvel_fd into a single Velocity3D.
        This is the primary method for computing velocity from pose history,
        matching reference data computation for AMP parity.

        Args:
            prev: Previous pose (at time t - dt)
            dt: Time step in seconds
            frame: Coordinate frame for output velocity (default HEADING_LOCAL for AMP)

        Returns:
            Velocity3D with linear and angular velocity in specified frame

        Raises:
            ValueError: If either pose is not in WORLD frame

        Example:
            curr_pose = cal.get_root_pose(data)
            prev_pose = Pose3D(prev_pos, prev_quat, CoordinateFrame.WORLD)
            vel = curr_pose.velocity_fd(prev_pose, dt=0.02)
        """
        return Velocity3D(
            linear=self.linvel_fd(prev, dt, frame),
            angular=self.angvel_fd(prev, dt, frame),
            frame=frame,
        )


@dataclass(frozen=True)
class Velocity3D:
    """General 3D velocity with coordinate frame (JAX-native).

    Self-documenting: frame is stored with the data.
    Reusable for root, feet, or any body.

    Array Type Design (Factory Pattern):
        - Default: JAX arrays for JIT-compiled training
        - Use from_numpy() for visualization with MuJoCo's NumPy arrays

    Coordinate Conversion (OOD):
        Use to_heading_local(pose) to transform velocity to heading-local frame.
        The pose provides the heading direction for the rotation.

    Attributes:
        linear: (3,) [vx, vy, vz] in m/s (JAX array)
        angular: (3,) [wx, wy, wz] in rad/s (JAX array)
        frame: Coordinate frame this velocity is expressed in

    Example (JAX - training):
        vel = cal.get_root_velocity(data, frame=CoordinateFrame.WORLD)
        vel_local = vel.to_heading_local(pose)
        assert vel_local.frame == CoordinateFrame.HEADING_LOCAL

    Example (NumPy - visualization):
        vel = Velocity3D.from_numpy(
            linear=mj_data.qvel[:3],
            angular=mj_data.qvel[3:6],
            frame=CoordinateFrame.WORLD,
        )
        vel_local = vel.to_heading_local(pose)
    """

    linear: jnp.ndarray  # (3,) [vx, vy, vz] in m/s
    angular: jnp.ndarray  # (3,) [wx, wy, wz] in rad/s
    frame: CoordinateFrame  # Frame this velocity is expressed in

    @classmethod
    def from_numpy(
        cls,
        linear: np.ndarray,
        angular: np.ndarray,
        frame: CoordinateFrame,
    ) -> Velocity3D:
        """Create Velocity3D from NumPy arrays (for visualization).

        Explicitly converts NumPy arrays to JAX arrays. Use this when working
        with MuJoCo's native NumPy arrays in visualization code.

        Args:
            linear: (3,) linear velocity array (NumPy)
            angular: (3,) angular velocity array (NumPy)
            frame: Coordinate frame

        Returns:
            Velocity3D with JAX arrays

        Example:
            vel = Velocity3D.from_numpy(
                linear=mj_data.qvel[:3],
                angular=mj_data.qvel[3:6],
                frame=CoordinateFrame.WORLD,
            )
        """
        return cls(
            linear=jnp.array(linear),
            angular=jnp.array(angular),
            frame=frame,
        )

    # Tuple accessors for easy unpacking
    @property
    def linear_xyz(self) -> tuple[float, float, float]:
        """Linear velocity as (vx, vy, vz) tuple for unpacking."""
        return self.linear[0], self.linear[1], self.linear[2]

    @property
    def angular_xyz(self) -> tuple[float, float, float]:
        """Angular velocity as (wx, wy, wz) tuple for unpacking."""
        return self.angular[0], self.angular[1], self.angular[2]

    def to_heading_local(self, pose: Pose3D) -> Velocity3D:
        """Transform velocity from world frame to heading-local frame.

        Heading-local orientation: world frame rotated by -yaw (heading direction is +X).
        Uses the pose's orientation to determine the heading direction.
        Note: This is a vector transform only; position origin does not apply.

        Args:
            pose: Pose3D providing the heading direction (must have orientation)

        Returns:
            New Velocity3D in heading-local frame

        Raises:
            ValueError: If velocity is not in WORLD frame

        Example:
            pose = cal.get_root_pose(data)
            vel_world = cal.get_root_velocity(data, frame=CoordinateFrame.WORLD)
            vel_local = vel_world.to_heading_local(pose)
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"to_heading_local requires WORLD frame velocity, got {self.frame}"
            )
        return Velocity3D(
            linear=pose.to_heading_local(self.linear),
            angular=pose.to_heading_local(self.angular),
            frame=CoordinateFrame.HEADING_LOCAL,
        )


# =============================================================================
# Internal Specs (Built from robot_config at CAL init)
# =============================================================================


@dataclass(frozen=True)
class FootSpec:
    """Internal spec built from robot_config.feet at CAL init.

    Attributes:
        name: Foot identifier (e.g., "left_foot", "right_foot")
        body_name: MuJoCo body name
        body_id: Resolved MuJoCo body ID
        toe_geom_id: Resolved MuJoCo geom ID for toe contact
        heel_geom_id: Resolved MuJoCo geom ID for heel contact
        symmetry_pair: Name of paired foot (e.g., "right_foot" for "left_foot")
    """

    name: str  # e.g., "left_foot"
    body_name: str  # MuJoCo body name
    body_id: int  # Resolved via mj_name2id
    toe_geom_id: int  # Resolved from robot_config.feet.left_toe
    heel_geom_id: int  # Resolved from robot_config.feet.left_heel
    symmetry_pair: Optional[str] = None


@dataclass(frozen=True)
class RootSpec:
    """Internal spec built from robot_config.floating_base at CAL init.

    Uses chest_imu (BNO085) sensors which exist on the real robot:
    - chest_imu_quat: framequat for orientation
    - chest_imu_gyro: gyroscope for angular velocity
    - chest_imu_accel: accelerometer (can derive gravity direction)

    BNO085 is a 9-DOF IMU (gyro + accel + magnetometer) with onboard sensor fusion:
    - Outputs stable framequat (orientation quaternion)
    - Unlike 6-DOF IMUs (ICM45686), has magnetometer for absolute heading
    - Onboard processor handles sensor fusion - no drift in orientation

    SIMULATION-ONLY Sensors:
        local_linvel_sensor (pelvis_local_linvel) is a MuJoCo velocimeter that
        does NOT exist on real hardware. The BNO085 cannot measure linear velocity:
        - IMUs measure acceleration, not velocity
        - Velocity from accelerometer integration drifts rapidly (unusable)
        - Real hardware uses finite-difference from position (via Pose3D.velocity_fd)

        For Sim2Real: Use Pose3D.velocity_fd() which computes velocity from
        position changes (same as reference data, works on real hardware).

    IMU Mounting Position Note:
        The BNO085 is mounted on the chest, not at the center of the root body.
        For WildRobot: ~16.7cm above chest body origin (see wildrobot.xml site pos)

        Effect on readings:
        - Orientation (framequat): NOT affected
          (orientation is same for all points on rigid body)
        - Angular velocity (gyro): NOT affected
          (angular velocity is body-invariant, same for all points)
        - Linear acceleration (accel): SLIGHTLY affected by offset
          - Centripetal: ω²r ≈ (5 rad/s)² × 0.167m ≈ 4.2 m/s²
          - Tangential: α*r ≈ (50 rad/s²) × 0.167m ≈ 8.4 m/s²
          - Compare to gravity (~10 m/s²) and motion (~20 m/s²)
        - Gravity vector: NOT affected (derived from orientation)

        Why it's OK for training:
        1. Orientation/gyro: Body-invariant, no effect from offset
        2. Gravity vector: Derived from orientation, not accelerometer
        3. Accelerometer: Only used for gravity direction (via orientation),
           not for velocity integration (which would accumulate drift)
        4. Sim2Real: MuJoCo sensor site matches real hardware position

    Attributes:
        body_name: Root body name (e.g., "waist")
        body_id: Resolved MuJoCo body ID
        qpos_addr: Address in qpos for root position/orientation
        qvel_addr: Address in qvel for root linear/angular velocity
        orientation_sensor: Chest IMU framequat sensor name (real hardware)
        gyro_sensor: Chest IMU gyro sensor name (real hardware)
        accel_sensor: Chest IMU accelerometer sensor name (real hardware)
        local_linvel_sensor: MuJoCo velocimeter (SIMULATION-ONLY, not on real robot)
    """

    body_name: str  # From robot_config.floating_base.root_body
    body_id: int  # Resolved via mj_name2id
    qpos_addr: int  # From mj_model.jnt_qposadr
    qvel_addr: int  # From mj_model.jnt_dofadr
    # Chest IMU sensors (BNO085 - real hardware)
    orientation_sensor: str  # chest_imu_quat (framequat)
    gyro_sensor: str  # chest_imu_gyro (angular velocity)
    accel_sensor: str  # chest_imu_accel (can derive gravity)
    # SIMULATION-ONLY: MuJoCo velocimeter (not on real robot)
    local_linvel_sensor: (
        str  # pelvis_local_linvel (use Pose3D.velocity_fd for Sim2Real)
    )
