from __future__ import annotations

from control.adapters import compose_final_joint_target
from control.references import (
    LocomotionActionContract,
    LocomotionCommand,
    LocomotionObservationContract,
    LocomotionReferenceState,
    validate_locomotion_contract,
)
from runtime.wr_runtime.logging import (
    LocomotionLogRecord,
    RobotReplayState,
    required_replay_log_fields,
)


def test_locomotion_command_schema_construction() -> None:
    cmd = LocomotionCommand(forward_speed_mps=0.3, yaw_rate_rps=0.2)
    assert cmd.forward_speed_mps == 0.3
    assert cmd.yaw_rate_rps == 0.2


def test_locomotion_reference_schema_validation() -> None:
    ref = LocomotionReferenceState(
        gait_phase_sin=0.0,
        gait_phase_cos=1.0,
        stance_foot_id=0,
        desired_next_foothold_stance_frame=(0.1, 0.03),
        desired_swing_foot_position=(0.0, 0.0, 0.05),
        desired_swing_foot_velocity=(0.0, 0.0, 0.0),
        desired_pelvis_height_m=0.42,
        desired_pelvis_roll_rad=0.02,
        desired_pelvis_pitch_rad=0.01,
    )
    assert ref.stance_foot_id in (0, 1)


def test_action_contract_shape_and_fields() -> None:
    nominal = (0.1, -0.2, 0.3)
    residual = (0.01, 0.02, -0.01)
    final = compose_final_joint_target(nominal_joint_target=nominal, residual_joint_delta=residual)
    act = LocomotionActionContract(
        residual_joint_delta=residual,
        nominal_joint_target=nominal,
        final_joint_target=final,
    )
    assert act.policy_output_is_learned is True
    expected = (0.11, -0.18, 0.29)
    assert all(abs(a - b) < 1e-9 for a, b in zip(act.final_joint_target, expected))


def test_shared_log_schema_serialization_friendly() -> None:
    cmd = LocomotionCommand(forward_speed_mps=0.2, yaw_rate_rps=0.0)
    ref = LocomotionReferenceState(
        gait_phase_sin=0.5,
        gait_phase_cos=0.86,
        stance_foot_id=1,
        desired_next_foothold_stance_frame=(0.08, -0.03),
        desired_swing_foot_position=(0.03, -0.01, 0.07),
        desired_swing_foot_velocity=(0.0, 0.0, 0.0),
        desired_pelvis_height_m=0.41,
        desired_pelvis_roll_rad=-0.01,
        desired_pelvis_pitch_rad=0.02,
    )
    act = LocomotionActionContract(
        residual_joint_delta=(0.01, 0.02),
        nominal_joint_target=(0.1, -0.1),
        final_joint_target=(0.11, -0.08),
    )
    replay = RobotReplayState(
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        gyro_rad_s=(0.0, 0.0, 0.0),
        joint_pos_rad=(0.1, -0.1),
        joint_vel_rad_s=(0.0, 0.0),
        foot_switches=(1.0, 1.0, 1.0, 1.0),
        pelvis_height_m=0.42,
    )
    record = LocomotionLogRecord(
        command=cmd,
        reference=ref,
        action=act,
        replay_state=replay,
        velocity_cmd=cmd.forward_speed_mps,
        yaw_rate_cmd=cmd.yaw_rate_rps,
    )
    payload = record.to_dict()
    assert "command" in payload
    assert "reference" in payload
    assert "action" in payload
    assert "replay_state" in payload
    assert "velocity_cmd" in payload
    assert "yaw_rate_cmd" in payload
    assert set(required_replay_log_fields()) == {
        "quat_xyzw",
        "gyro_rad_s",
        "joint_pos_rad",
        "joint_vel_rad_s",
        "foot_switches",
        "velocity_cmd",
        "yaw_rate_cmd",
    }
    validate_locomotion_contract(cmd, ref, act)


def test_locomotion_observation_contract_schema() -> None:
    obs = LocomotionObservationContract(
        imu_orientation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        imu_angular_velocity_rad_s=(0.01, -0.02, 0.03),
        joint_pos_rad=(0.1, -0.1, 0.2),
        joint_vel_rad_s=(0.0, 0.0, 0.0),
        previous_action=(0.01, 0.01, -0.01),
        foot_switches=(1.0, 0.0, 1.0, 0.0),
        command=LocomotionCommand(forward_speed_mps=0.25, yaw_rate_rps=0.1),
        reference=LocomotionReferenceState(
            gait_phase_sin=0.0,
            gait_phase_cos=1.0,
            stance_foot_id=0,
            desired_next_foothold_stance_frame=(0.1, 0.0),
            desired_swing_foot_position=(0.0, 0.0, 0.05),
            desired_swing_foot_velocity=(0.0, 0.0, 0.0),
            desired_pelvis_height_m=0.42,
            desired_pelvis_roll_rad=0.0,
            desired_pelvis_pitch_rad=0.0,
        ),
        history_length=3,
    )
    assert obs.history_length == 3


def test_locomotion_reference_invalid_stance_raises() -> None:
    raised = False
    try:
        LocomotionReferenceState(
            gait_phase_sin=0.0,
            gait_phase_cos=1.0,
            stance_foot_id=2,
            desired_next_foothold_stance_frame=(0.1, 0.0),
            desired_swing_foot_position=(0.0, 0.0, 0.05),
            desired_swing_foot_velocity=(0.0, 0.0, 0.0),
            desired_pelvis_height_m=0.42,
            desired_pelvis_roll_rad=0.0,
            desired_pelvis_pitch_rad=0.0,
        )
    except ValueError:
        raised = True
    assert raised


def test_action_contract_invariant_rejects_invalid_final_target() -> None:
    raised = False
    try:
        LocomotionActionContract(
            residual_joint_delta=(0.1, 0.2),
            nominal_joint_target=(0.2, 0.3),
            final_joint_target=(0.1, 0.1),
        )
    except ValueError:
        raised = True
    assert raised


def test_log_record_rejects_command_scalar_mismatch() -> None:
    cmd = LocomotionCommand(forward_speed_mps=0.2, yaw_rate_rps=0.1)
    ref = LocomotionReferenceState(
        gait_phase_sin=0.0,
        gait_phase_cos=1.0,
        stance_foot_id=0,
        desired_next_foothold_stance_frame=(0.1, 0.0),
        desired_swing_foot_position=(0.0, 0.0, 0.05),
        desired_swing_foot_velocity=(0.0, 0.0, 0.0),
        desired_pelvis_height_m=0.42,
        desired_pelvis_roll_rad=0.0,
        desired_pelvis_pitch_rad=0.0,
    )
    act = LocomotionActionContract(
        residual_joint_delta=(0.01,),
        nominal_joint_target=(0.1,),
        final_joint_target=(0.11,),
    )
    replay = RobotReplayState(
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        gyro_rad_s=(0.0, 0.0, 0.0),
        joint_pos_rad=(0.1,),
        joint_vel_rad_s=(0.0,),
        foot_switches=(1.0, 1.0, 1.0, 1.0),
        pelvis_height_m=0.42,
    )
    raised = False
    try:
        LocomotionLogRecord(
            command=cmd,
            reference=ref,
            action=act,
            replay_state=replay,
            velocity_cmd=0.19,
            yaw_rate_cmd=0.1,
        )
    except ValueError:
        raised = True
    assert raised
