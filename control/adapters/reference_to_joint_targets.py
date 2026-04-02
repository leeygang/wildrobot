"""Convert walking reference + IK into nominal joint targets q_ref (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Dict, Tuple

from assets.robot_config import RobotConfig

from control.kinematics.leg_ik import LegIkConfig, LegIkResult, solve_leg_sagittal_ik
from control.references.locomotion_contract import LocomotionReferenceState


@dataclass(frozen=True)
class NominalTargetOutput:
    q_ref: Tuple[float, ...]
    left_leg_reachable: bool
    right_leg_reachable: bool
    left_ik: LegIkResult
    right_ik: LegIkResult


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _joint_limit_map(robot_cfg: RobotConfig) -> Dict[str, tuple[float, float]]:
    return dict(robot_cfg.joint_limits)


def _home_pose_map(
    *,
    robot_cfg: RobotConfig,
    joint_limits: Dict[str, tuple[float, float]],
    actuator_order: list[str],
) -> Dict[str, float]:
    """Resolve per-joint home defaults; fallback is 0 rad (home-centered contract)."""
    home_by_name: Dict[str, float] = {name: 0.0 for name in actuator_order}
    for joint in robot_cfg.actuated_joints:
        name = str(joint.get("name", ""))
        if not name or name not in home_by_name:
            continue
        if "home_rad" in joint:
            raw_home = float(joint["home_rad"])
        elif "home_deg" in joint:
            raw_home = radians(float(joint["home_deg"]))
        elif "home" in joint:
            raw_home = float(joint["home"])
        else:
            raw_home = 0.0
        lo, hi = joint_limits.get(name, (-3.14, 3.14))
        home_by_name[name] = _clip(raw_home, lo, hi)
    return home_by_name


def _apply_joint(
    *,
    q_ref: Dict[str, float],
    joint_limits: Dict[str, tuple[float, float]],
    name: str,
    value: float,
) -> None:
    lo, hi = joint_limits.get(name, (-3.14, 3.14))
    q_ref[name] = _clip(value, lo, hi)


def reference_to_nominal_joint_targets(
    *,
    reference: LocomotionReferenceState,
    robot_cfg: RobotConfig,
    leg_ik_config: LegIkConfig,
) -> NominalTargetOutput:
    actuator_order = list(robot_cfg.actuator_names)
    limits = _joint_limit_map(robot_cfg)
    q_by_name = _home_pose_map(
        robot_cfg=robot_cfg,
        joint_limits=limits,
        actuator_order=actuator_order,
    )

    # Split swing target by stance.
    swing_x = float(reference.desired_swing_foot_position[0])
    swing_z = float(reference.desired_swing_foot_position[2])
    # Keep a tiny knee bend margin so nominal targets avoid hard limit hugging.
    stance_leg_extension_margin_m = 0.015
    nominal_stance_z = -max(0.0, float(reference.desired_pelvis_height_m) - stance_leg_extension_margin_m)
    stance_target = (0.0, nominal_stance_z)
    swing_target = (swing_x, nominal_stance_z + swing_z)

    if reference.stance_foot_id == 0:  # left stance, right swing
        left_target = stance_target
        right_target = swing_target
    else:
        left_target = swing_target
        right_target = stance_target

    left_ik = solve_leg_sagittal_ik(
        target_x_m=float(left_target[0]),
        target_z_m=float(left_target[1]),
        config=leg_ik_config,
    )
    right_ik = solve_leg_sagittal_ik(
        target_x_m=float(right_target[0]),
        target_z_m=float(right_target[1]),
        config=leg_ik_config,
    )

    # Hip pitch axes are mirrored across legs in this robot config.
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="left_hip_pitch", value=-left_ik.hip_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="left_knee_pitch", value=left_ik.knee_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="left_ankle_pitch", value=left_ik.ankle_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="right_hip_pitch", value=right_ik.hip_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="right_knee_pitch", value=right_ik.knee_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="right_ankle_pitch", value=right_ik.ankle_pitch_rad)

    # Light nominal posture cues.
    roll = float(reference.desired_pelvis_roll_rad)
    pitch = float(reference.desired_pelvis_pitch_rad)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="left_hip_roll", value=-roll)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="right_hip_roll", value=roll)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="waist_yaw", value=0.0)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="left_shoulder_pitch", value=-0.15 * pitch)
    _apply_joint(q_ref=q_by_name, joint_limits=limits, name="right_shoulder_pitch", value=-0.15 * pitch)

    q_ref = tuple(float(q_by_name.get(name, 0.0)) for name in actuator_order)
    return NominalTargetOutput(
        q_ref=q_ref,
        left_leg_reachable=left_ik.reachable,
        right_leg_reachable=right_ik.reachable,
        left_ik=left_ik,
        right_ik=right_ik,
    )
