"""Nominal IK adapter — converts walking reference targets to joint angles.

Pure NumPy implementation shared by training (via JAX wrapper in env) and
runtime.  Uses the same sagittal IK solver as the training env.

This module extracts the core IK logic from wildrobot_env.py's
_compute_nominal_q_ref_from_loc_ref() into a standalone, testable function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sin
from typing import Dict, Tuple

import numpy as np

from control.kinematics.leg_ik import LegIkConfig, solve_leg_sagittal_ik


@dataclass(frozen=True)
class NominalIkConfig:
    """Configuration for the nominal IK adapter."""

    leg_ik: LegIkConfig = field(default_factory=LegIkConfig)

    # Two-height support system
    support_margin_m: float = 0.020
    walking_crouch_extra_m: float = 0.045

    # COM compensation
    com_forward_gain: float = 0.20
    com_depth_fade_start_z: float = -0.36
    com_depth_fade_end_z: float = -0.32

    # Swing foot tracking
    swing_target_blend: float = 1.0
    swing_max_delta_per_step_m: float = 0.02

    # Hip roll coupling
    swing_y_to_hip_roll_gain: float = 0.8
    base_hip_roll_rad: float = 0.0

    # Arm swing (waist pitch compensation)
    arm_swing_gain: float = -0.10

    # Joint limits (radians) — per-joint min/max
    # Default: wide enough for any pose; caller should tighten from robot config
    joint_range_min: Tuple[float, ...] = (-1.5,) * 9
    joint_range_max: Tuple[float, ...] = (1.5,) * 9

    # Actuator names in PolicySpec order (for readable diagnostics)
    actuator_names: Tuple[str, ...] = (
        "left_hip_pitch",
        "right_hip_pitch",
        "left_hip_roll",
        "right_hip_roll",
        "left_knee_pitch",
        "right_knee_pitch",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "waist",
    )


# Default joint index map (PolicySpec order from mujoco_robot_config.json)
_DEFAULT_IDX: Dict[str, int] = {
    "left_hip_pitch": 0,
    "right_hip_pitch": 1,
    "left_hip_roll": 2,
    "right_hip_roll": 3,
    "left_knee_pitch": 4,
    "right_knee_pitch": 5,
    "left_ankle_pitch": 6,
    "right_ankle_pitch": 7,
    "waist": 8,
}


@dataclass(frozen=True)
class NominalIkResult:
    """Output from the nominal IK adapter."""

    q_ref: np.ndarray  # (9,) joint angles in PolicySpec order
    swing_reachable: bool
    stance_reachable: bool


def compute_nominal_q_ref(
    *,
    config: NominalIkConfig,
    pelvis_height_m: float,
    pelvis_roll_rad: float,
    pelvis_pitch_rad: float,
    stance_foot_id: int,
    swing_pos: Tuple[float, float, float],
    com_x_planned: float = 0.0,
    mode_id: int = 1,
    home_pose: np.ndarray | None = None,
    idx: Dict[str, int] | None = None,
) -> NominalIkResult:
    """Compute nominal joint targets from walking reference task-space targets.

    Args:
        config: IK configuration.
        pelvis_height_m: Desired pelvis height from reference.
        pelvis_roll_rad: Desired pelvis roll from reference.
        pelvis_pitch_rad: Desired pelvis pitch from reference.
        stance_foot_id: 0=left stance, 1=right stance.
        swing_pos: (x, y, z) swing foot position in stance frame.
        com_x_planned: COM forward offset from DCM trajectory (m).
        mode_id: Walking reference mode (1=SUPPORT, 2=SWING, etc.).
        home_pose: (9,) home joint positions; defaults to zeros.
        idx: Joint name → index map; defaults to PolicySpec order.

    Returns:
        NominalIkResult with q_ref array and reachability flags.
    """
    if idx is None:
        idx = _DEFAULT_IDX
    if home_pose is None:
        home_pose = np.zeros(9, dtype=np.float32)

    n_joints = len(config.actuator_names)
    q_ref = np.array(home_pose, dtype=np.float32).copy()

    # --- Determine stance height ---
    # Support/startup modes use shallow squat; walking modes use deeper squat
    is_walking_mode = mode_id >= 2  # SWING_RELEASE, TOUCHDOWN, SETTLE
    if is_walking_mode:
        stance_z = -(pelvis_height_m - config.walking_crouch_extra_m)
    else:
        stance_z = -(pelvis_height_m - config.support_margin_m)

    # --- Stance leg IK ---
    # Sagittal offset: COM compensation + DCM trajectory
    stance_hip_pitch = q_ref[idx["left_hip_pitch"] if stance_foot_id == 0
                             else idx["right_hip_pitch"]]
    depth = -stance_z  # positive depth
    depth_fade = np.clip(
        (depth - (-config.com_depth_fade_end_z))
        / max((-config.com_depth_fade_start_z) - (-config.com_depth_fade_end_z), 1e-6),
        0.0,
        1.0,
    )
    com_offset = config.com_forward_gain * depth_fade * sin(float(stance_hip_pitch))

    # Add DCM COM trajectory offset (stance foot moves behind as COM advances)
    stance_x = com_offset - com_x_planned

    stance_ik = solve_leg_sagittal_ik(
        target_x_m=stance_x,
        target_z_m=stance_z,
        config=config.leg_ik,
    )

    # --- Swing leg IK ---
    swing_x = float(swing_pos[0])
    swing_z_height = float(swing_pos[2])
    # Swing foot at pelvis_height offset minus ground clearance
    swing_z = stance_z + swing_z_height

    swing_ik = solve_leg_sagittal_ik(
        target_x_m=swing_x,
        target_z_m=swing_z,
        config=config.leg_ik,
    )

    # --- Assign joint angles ---
    if stance_foot_id == 0:  # left stance
        # Stance (left) leg
        q_ref[idx["left_hip_pitch"]] = stance_ik.hip_pitch_rad
        q_ref[idx["left_knee_pitch"]] = stance_ik.knee_pitch_rad
        q_ref[idx["left_ankle_pitch"]] = stance_ik.ankle_pitch_rad
        # Swing (right) leg
        q_ref[idx["right_hip_pitch"]] = swing_ik.hip_pitch_rad
        q_ref[idx["right_knee_pitch"]] = swing_ik.knee_pitch_rad
        q_ref[idx["right_ankle_pitch"]] = swing_ik.ankle_pitch_rad
    else:  # right stance
        # Stance (right) leg
        q_ref[idx["right_hip_pitch"]] = stance_ik.hip_pitch_rad
        q_ref[idx["right_knee_pitch"]] = stance_ik.knee_pitch_rad
        q_ref[idx["right_ankle_pitch"]] = stance_ik.ankle_pitch_rad
        # Swing (left) leg
        q_ref[idx["left_hip_pitch"]] = swing_ik.hip_pitch_rad
        q_ref[idx["left_knee_pitch"]] = swing_ik.knee_pitch_rad
        q_ref[idx["left_ankle_pitch"]] = swing_ik.ankle_pitch_rad

    # --- Hip roll from pelvis roll reference ---
    roll_sign = -1.0 if stance_foot_id == 0 else 1.0
    swing_y = float(swing_pos[1])
    hip_roll_from_swing = config.swing_y_to_hip_roll_gain * swing_y

    q_ref[idx["left_hip_roll"]] = (
        config.base_hip_roll_rad + pelvis_roll_rad + hip_roll_from_swing * (1.0 if stance_foot_id == 1 else 0.0)
    )
    q_ref[idx["right_hip_roll"]] = (
        config.base_hip_roll_rad - pelvis_roll_rad + hip_roll_from_swing * (1.0 if stance_foot_id == 0 else 0.0)
    )

    # --- Waist (arm swing compensation) ---
    if "waist" in idx:
        q_ref[idx["waist"]] = config.arm_swing_gain * pelvis_pitch_rad

    # --- Clip to joint limits ---
    range_min = np.array(config.joint_range_min[:n_joints], dtype=np.float32)
    range_max = np.array(config.joint_range_max[:n_joints], dtype=np.float32)
    q_ref = np.clip(q_ref, range_min, range_max)

    return NominalIkResult(
        q_ref=q_ref,
        swing_reachable=swing_ik.reachable,
        stance_reachable=stance_ik.reachable,
    )
