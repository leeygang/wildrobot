"""Specification dataclasses for the Control Abstraction Layer (CAL).

This module defines:
- JointSpec: Joint specification (range, default, symmetry)
- ActuatorSpec: Actuator specification (ctrl range, force limits, motor specs)
- ControlCommand: Control command with trajectory support for Sim2Real

Version: v0.11.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp

from playground_amp.control.types import ActuatorType


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
