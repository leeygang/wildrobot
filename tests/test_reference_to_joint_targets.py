from __future__ import annotations

from math import isclose

from assets.robot_config import load_robot_config
from control.adapters import reference_to_nominal_joint_targets
from control.kinematics import LegIkConfig
from control.references import LocomotionReferenceState


def test_nominal_adapter_output_shape_and_contract() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    ref = LocomotionReferenceState(
        gait_phase_sin=0.0,
        gait_phase_cos=1.0,
        stance_foot_id=0,
        desired_next_foothold_stance_frame=(0.08, -0.09),
        desired_swing_foot_position=(0.03, -0.04, 0.03),
        desired_swing_foot_velocity=(0.1, 0.0, 0.0),
        desired_pelvis_height_m=0.40,
        desired_pelvis_roll_rad=0.02,
        desired_pelvis_pitch_rad=0.01,
    )
    out = reference_to_nominal_joint_targets(
        reference=ref,
        robot_cfg=robot_cfg,
        leg_ik_config=LegIkConfig(),
    )
    assert len(out.q_ref) == len(robot_cfg.actuator_names)
    assert isinstance(out.left_leg_reachable, bool)
    assert isinstance(out.right_leg_reachable, bool)


def test_nominal_adapter_is_home_centered_for_unsolved_joints() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    ref = LocomotionReferenceState(
        gait_phase_sin=0.0,
        gait_phase_cos=1.0,
        stance_foot_id=0,
        desired_next_foothold_stance_frame=(0.06, -0.09),
        desired_swing_foot_position=(0.02, -0.03, 0.02),
        desired_swing_foot_velocity=(0.1, 0.0, 0.0),
        desired_pelvis_height_m=0.40,
        desired_pelvis_roll_rad=0.0,
        desired_pelvis_pitch_rad=0.0,
    )
    out = reference_to_nominal_joint_targets(
        reference=ref,
        robot_cfg=robot_cfg,
        leg_ik_config=LegIkConfig(),
    )
    q_by_name = dict(zip(robot_cfg.actuator_names, out.q_ref))
    for joint_name in ("left_elbow_pitch", "right_elbow_pitch", "waist_yaw"):
        assert isclose(q_by_name[joint_name], 0.0, abs_tol=1e-6), joint_name


def test_nominal_adapter_symmetric_hips_for_symmetric_pose() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    ref = LocomotionReferenceState(
        gait_phase_sin=0.0,
        gait_phase_cos=1.0,
        stance_foot_id=0,
        desired_next_foothold_stance_frame=(0.0, -0.09),
        desired_swing_foot_position=(0.0, 0.09, 0.0),
        desired_swing_foot_velocity=(0.0, 0.0, 0.0),
        desired_pelvis_height_m=0.36,
        desired_pelvis_roll_rad=0.0,
        desired_pelvis_pitch_rad=0.0,
    )
    out = reference_to_nominal_joint_targets(
        reference=ref,
        robot_cfg=robot_cfg,
        leg_ik_config=LegIkConfig(),
    )
    q_by_name = dict(zip(robot_cfg.actuator_names, out.q_ref))
    assert isclose(
        abs(q_by_name["left_hip_pitch"]),
        abs(q_by_name["right_hip_pitch"]),
        rel_tol=1e-4,
        abs_tol=1e-4,
    )
    left_lo, left_hi = robot_cfg.joint_limits["left_hip_pitch"]
    right_lo, right_hi = robot_cfg.joint_limits["right_hip_pitch"]
    assert (q_by_name["left_hip_pitch"] - left_lo) > 0.01
    assert (right_hi - q_by_name["right_hip_pitch"]) > 0.01
