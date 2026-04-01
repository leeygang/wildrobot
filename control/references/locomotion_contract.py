"""Typed locomotion contracts for the v0.19.0 platform pivot.

These contracts are schema-only and must remain lightweight/stable in Milestone 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class LocomotionCommand:
    """Primary locomotion command space for Milestone 0."""

    forward_speed_mps: float
    yaw_rate_rps: float

    def __post_init__(self) -> None:
        for name, value in (
            ("forward_speed_mps", self.forward_speed_mps),
            ("yaw_rate_rps", self.yaw_rate_rps),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value)!r}")


@dataclass(frozen=True)
class LocomotionReferenceState:
    """Shared locomotion reference state used by training/runtime logs."""

    gait_phase_sin: float
    gait_phase_cos: float
    stance_foot_id: int
    desired_next_foothold_stance_frame: Vec2
    desired_swing_foot_position: Vec3
    desired_swing_foot_velocity: Vec3
    desired_pelvis_height_m: float
    desired_pelvis_roll_rad: float
    desired_pelvis_pitch_rad: float

    def __post_init__(self) -> None:
        if self.stance_foot_id not in (0, 1):
            raise ValueError("stance_foot_id must be 0 (left) or 1 (right)")
        if len(self.desired_next_foothold_stance_frame) != 2:
            raise ValueError("desired_next_foothold_stance_frame must be length 2")
        if len(self.desired_swing_foot_position) != 3:
            raise ValueError("desired_swing_foot_position must be length 3")
        if len(self.desired_swing_foot_velocity) != 3:
            raise ValueError("desired_swing_foot_velocity must be length 3")


@dataclass(frozen=True)
class LocomotionObservationContract:
    """Shared locomotion observation schema (Milestone 0 contract only)."""

    imu_orientation_quat_xyzw: Tuple[float, float, float, float]
    imu_angular_velocity_rad_s: Vec3
    joint_pos_rad: Tuple[float, ...]
    joint_vel_rad_s: Tuple[float, ...]
    previous_action: Tuple[float, ...]
    foot_switches: Tuple[float, float, float, float]
    command: LocomotionCommand
    reference: LocomotionReferenceState
    history_length: int

    def __post_init__(self) -> None:
        if len(self.imu_orientation_quat_xyzw) != 4:
            raise ValueError("imu_orientation_quat_xyzw must be length 4")
        if len(self.imu_angular_velocity_rad_s) != 3:
            raise ValueError("imu_angular_velocity_rad_s must be length 3")
        if len(self.foot_switches) != 4:
            raise ValueError("foot_switches must be length 4")
        n = len(self.joint_pos_rad)
        if n == 0:
            raise ValueError("joint_pos_rad must be non-empty")
        if len(self.joint_vel_rad_s) != n:
            raise ValueError("joint_vel_rad_s size must match joint_pos_rad")
        if len(self.previous_action) != n:
            raise ValueError("previous_action size must match joint_pos_rad")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")


@dataclass(frozen=True)
class LocomotionActionContract:
    """Residual action around nominal targets.

    Runtime output remains a learned policy action; this schema only records
    how residual and nominal targets combine.
    """

    residual_joint_delta: Tuple[float, ...]
    nominal_joint_target: Tuple[float, ...]
    final_joint_target: Tuple[float, ...]
    policy_output_is_learned: bool = True

    def __post_init__(self) -> None:
        n = len(self.residual_joint_delta)
        if n == 0:
            raise ValueError("residual_joint_delta must be non-empty")
        if len(self.nominal_joint_target) != n:
            raise ValueError("nominal_joint_target size must match residual_joint_delta")
        if len(self.final_joint_target) != n:
            raise ValueError("final_joint_target size must match residual_joint_delta")
        if not self.policy_output_is_learned:
            raise ValueError("policy_output_is_learned must remain True in locomotion contract")
        for idx, (residual, nominal, final) in enumerate(
            zip(self.residual_joint_delta, self.nominal_joint_target, self.final_joint_target)
        ):
            expected = float(nominal) + float(residual)
            if abs(float(final) - expected) > 1e-6:
                raise ValueError(
                    f"final_joint_target[{idx}] must equal nominal_joint_target + residual_joint_delta"
                )


def validate_locomotion_contract(
    command: LocomotionCommand,
    reference: LocomotionReferenceState,
    action: LocomotionActionContract,
) -> None:
    """Validate cross-object contract consistency."""
    del command
    del reference
    if len(action.residual_joint_delta) != len(action.final_joint_target):
        raise ValueError("Action contract shape mismatch")
    for idx, (residual, nominal, final) in enumerate(
        zip(action.residual_joint_delta, action.nominal_joint_target, action.final_joint_target)
    ):
        expected = float(nominal) + float(residual)
        if abs(float(final) - expected) > 1e-6:
            raise ValueError(
                f"Action contract invariant violated at index {idx}: final != nominal + residual"
            )
