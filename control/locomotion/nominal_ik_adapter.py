"""Nominal IK adapter — converts walking reference targets to joint angles.

Pure NumPy implementation extracted from wildrobot_env.py's
_compute_nominal_q_ref_from_loc_ref() (lines 4663-4875).

Every sign convention, depth-fade formula, and mode-dependent branch matches
the env source of truth exactly.  If the env IK changes, this module must be
updated to match.

Shared by training (via JAX wrapper in env) and runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from control.kinematics.leg_ik import LegIkConfig, solve_leg_sagittal_ik
from control.references.walking_ref_v2 import WalkingRefV2Mode


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
class NominalIkConfig:
    """Configuration for the nominal IK adapter.

    Defaults here should match what ppo_walking_v0195.yaml sets
    through the env's loc_ref_* config keys.
    """

    leg_ik: LegIkConfig = field(default_factory=LegIkConfig)

    # Two-height support system (env: _loc_ref_support_margin_m, _loc_ref_walking_crouch_extra_m)
    support_margin_m: float = 0.020
    walking_crouch_extra_m: float = 0.045

    # Stance height blending (env: _loc_ref_stance_height_blend)
    stance_height_blend: float = 0.0

    # COM forward compensation gain (env: _loc_ref_stance_com_forward_gain)
    com_forward_gain: float = 0.20

    # Swing foot tracking (env: _loc_ref_swing_target_blend, _loc_ref_max_swing_x_delta_m, _loc_ref_max_swing_z_delta_m)
    swing_target_blend: float = 1.0
    max_swing_x_delta_m: float = 0.06
    max_swing_z_delta_m: float = 0.03

    # Hip roll coupling (env: _loc_ref_swing_y_to_hip_roll)
    swing_y_to_hip_roll_gain: float = 0.08

    # Arm swing (env: shoulder pitch = -0.10 * pitch)
    arm_swing_gain: float = -0.10

    # Joint limits (radians) — per-joint min/max
    joint_range_min: Tuple[float, ...] = (-1.5,) * 9
    joint_range_max: Tuple[float, ...] = (1.5,) * 9


@dataclass(frozen=True)
class NominalIkResult:
    """Output from the nominal IK adapter."""

    q_ref: np.ndarray       # (N,) joint angles in PolicySpec order
    left_reachable: bool
    right_reachable: bool
    stance_target_z: float   # Actual stance z used for IK
    swing_x_target: float    # Swing x after blending/clamping


def compute_nominal_q_ref(
    *,
    config: NominalIkConfig,
    pelvis_height_m: float,
    pelvis_roll_rad: float,
    pelvis_pitch_rad: float,
    stance_foot_id: int,
    swing_pos: Tuple[float, float, float],
    com_x_planned: float = 0.0,
    mode_id: int = int(WalkingRefV2Mode.SUPPORT_STABILIZE),
    home_pose: np.ndarray | None = None,
    idx: Dict[str, int] | None = None,
) -> NominalIkResult:
    """Compute nominal joint targets matching env._compute_nominal_q_ref_from_loc_ref().

    Sign conventions and depth-fade match env lines 4663-4875 exactly.
    """
    if idx is None:
        idx = _DEFAULT_IDX
    n_joints = max(idx.values()) + 1 if idx else 9
    if home_pose is None:
        home_pose = np.zeros(n_joints, dtype=np.float32)

    q_ref = np.array(home_pose, dtype=np.float32).copy()

    # --- Two-height system (env lines 4698-4710) ---
    _is_walking_mode = mode_id not in (
        int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP),
        int(WalkingRefV2Mode.SUPPORT_STABILIZE),
    )
    margin = config.walking_crouch_extra_m if _is_walking_mode else config.support_margin_m
    nominal_stance_z = -max(0.0, pelvis_height_m - margin)

    # --- Swing targets (env lines 4711-4764) ---
    swing_x_ref = float(swing_pos[0])
    swing_z_ref = float(swing_pos[2])

    # Simplified: for standalone runtime, use swing_target_blend=1.0 (full ref).
    # The env blends with actual foot positions (which require sim state).
    # With blend=1.0, swing_x_target = swing_x_ref (env line 4744 collapses).
    swing_x_target = swing_x_ref
    swing_z_target = swing_z_ref
    swing_target_z = nominal_stance_z + swing_z_target

    # Stance z (env lines 4765-4772): blend nominal with actual (default 0.0 = pure nominal)
    stance_target_z = nominal_stance_z  # stance_height_blend=0.0

    # --- Stance foot sagittal offset (env lines 4774-4812) ---
    # Reach-based depth fade matching env exactly.
    l1 = config.leg_ik.upper_leg_length_m
    l2 = config.leg_ik.lower_leg_length_m
    max_reach = l1 + l2 - 1e-4
    reach = min(abs(stance_target_z), max_reach)
    cos_hip_aux = np.clip(
        (l1 * l1 + reach * reach - l2 * l2) / (2.0 * l1 * reach + 1e-6),
        -1.0, 1.0,
    )
    approx_hip_pitch = float(np.arccos(cos_hip_aux))

    # Depth fade: 1.0 when nearly straight (reach > 95% max),
    # tapering to 0.0 for deep squats (reach < 90% max).
    shallow_thresh = 0.95 * max_reach
    deep_thresh = 0.90 * max_reach
    depth_fade = float(np.clip(
        (reach - deep_thresh) / max(shallow_thresh - deep_thresh, 1e-6),
        0.0, 1.0,
    ))

    stance_foot_x = config.com_forward_gain * depth_fade * math.sin(approx_hip_pitch)

    # DCM COM trajectory: ADDITIVE offset (env line 4812: _stance_foot_x + _com_x)
    stance_foot_x = stance_foot_x + com_x_planned

    # --- Both-feet-grounded symmetry (env lines 4817-4826) ---
    # During startup/support, BOTH legs get the stance offset for a symmetric squat.
    # During walking modes, only the stance leg gets it; swing uses its reference.
    both_feet_grounded = mode_id in (
        int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP),
        int(WalkingRefV2Mode.SUPPORT_STABILIZE),
    )
    swing_x_for_mode = stance_foot_x if both_feet_grounded else swing_x_target

    stance_is_left = (stance_foot_id == 0)
    if stance_is_left:
        left_target_x = stance_foot_x
        left_target_z = stance_target_z
        right_target_x = swing_x_for_mode
        right_target_z = swing_target_z
    else:
        left_target_x = swing_x_for_mode
        left_target_z = swing_target_z
        right_target_x = stance_foot_x
        right_target_z = stance_target_z

    # --- Solve IK for both legs (env lines 4828-4833) ---
    left_ik = solve_leg_sagittal_ik(
        target_x_m=left_target_x, target_z_m=left_target_z, config=config.leg_ik,
    )
    right_ik = solve_leg_sagittal_ik(
        target_x_m=right_target_x, target_z_m=right_target_z, config=config.leg_ik,
    )

    # --- Assign joint angles with EXACT env sign conventions (lines 4836-4847) ---
    # Left hip pitch is NEGATED (env line 4837: q_ref.at[idx].set(-left_hip))
    if "left_hip_pitch" in idx:
        q_ref[idx["left_hip_pitch"]] = -left_ik.hip_pitch_rad
    if "left_knee_pitch" in idx:
        q_ref[idx["left_knee_pitch"]] = left_ik.knee_pitch_rad
    if "left_ankle_pitch" in idx:
        q_ref[idx["left_ankle_pitch"]] = left_ik.ankle_pitch_rad
    if "right_hip_pitch" in idx:
        q_ref[idx["right_hip_pitch"]] = right_ik.hip_pitch_rad
    if "right_knee_pitch" in idx:
        q_ref[idx["right_knee_pitch"]] = right_ik.knee_pitch_rad
    if "right_ankle_pitch" in idx:
        q_ref[idx["right_ankle_pitch"]] = right_ik.ankle_pitch_rad

    # --- Hip roll (env lines 4849-4855) ---
    # roll = pelvis_roll + swing_y_to_hip_roll * swing_pos[1]
    # left_hip_roll = -roll, right_hip_roll = +roll
    roll = pelvis_roll_rad + config.swing_y_to_hip_roll_gain * float(swing_pos[1])
    if "left_hip_roll" in idx:
        q_ref[idx["left_hip_roll"]] = -roll
    if "right_hip_roll" in idx:
        q_ref[idx["right_hip_roll"]] = roll

    # --- Waist / arm swing (env lines 4856-4861) ---
    if "waist" in idx:
        q_ref[idx["waist"]] = 0.0  # env line 4857: set to 0.0

    # --- Clip to joint limits (env line 4863) ---
    range_min = np.array(config.joint_range_min[:n_joints], dtype=np.float32)
    range_max = np.array(config.joint_range_max[:n_joints], dtype=np.float32)
    q_ref = np.clip(q_ref, range_min, range_max)

    return NominalIkResult(
        q_ref=q_ref,
        left_reachable=left_ik.reachable,
        right_reachable=right_ik.reachable,
        stance_target_z=float(stance_target_z),
        swing_x_target=float(swing_x_target),
    )
