from pathlib import Path
from types import SimpleNamespace

import numpy as np

from runtime.wr_runtime.logging.policy_telemetry import PolicyTelemetryRecorder
from runtime.wr_runtime.validation.inspect_log import inspect_log


def _info(*, previous: list[float], observed: list[float]) -> dict:
    signals = SimpleNamespace(
        timestamp_s=12.5,
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gyro_rad_s=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        joint_pos_rad=np.asarray(observed, dtype=np.float32),
        joint_vel_rad_s=np.array([0.4, -0.5], dtype=np.float32),
        foot_switches=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    return {
        "step_idx": 4,
        "obs": np.arange(6, dtype=np.float32),
        "raw_action": np.array([0.2, -0.3], dtype=np.float32),
        "applied_action": np.array([0.1, -0.2], dtype=np.float32),
        "target_q_rad": np.array([0.4, -0.4], dtype=np.float32),
        "commanded_q_rad": np.array([0.3, -0.3], dtype=np.float32),
        "previous_commanded_q_rad": np.asarray(previous, dtype=np.float32),
        "signals": signals,
        "footswitch_available": False,
        "control_mode": "policy",
        "action_scale": 0.5,
        "command_ramp_scale": 0.75,
        "obs_debug": {
            "velocity_cmd": np.array([0.065, 0.0, 0.0], dtype=np.float32),
            "reference_bin_idx": 2,
            "phase_sin_cos": np.array([0.0, 1.0], dtype=np.float32),
        },
        "timing_s": {"work": 0.01, "policy": 0.002},
        "servo_metrics": {"servo_read_fail_count": 1, "ignored": "text"},
        "servo_diagnostics": {
            "position_units": np.array([510.0, 490.0], dtype=np.float32),
            "velocity_units_s": np.array([4.0, -5.0], dtype=np.float32),
            "position_age_s": np.array([0.02, 0.04], dtype=np.float32),
            "read_fail_count": np.array([0, 1], dtype=np.int32),
        },
    }


def test_policy_telemetry_records_aligned_command_and_feedback(tmp_path: Path) -> None:
    output = tmp_path / "trial.npz"
    recorder = PolicyTelemetryRecorder(
        output,
        actuator_names=["left_hip_roll", "right_hip_roll"],
        ctrl_dt=0.02,
        bundle_path=tmp_path / "bundle",
        hardware_config_path=tmp_path / "hardware.json",
    )
    recorder.record(
        _info(previous=[0.25, -0.15], observed=[0.20, -0.10]),
        phase="walking",
        loop_step=3,
        requested_velocity_cmd=np.array([0.07, 0.0, 0.0], dtype=np.float32),
    )

    assert recorder.save(outcome="aborted", error="fall cutoff") == output

    with np.load(output) as data:
        assert int(data["schema_version"]) == 1
        assert str(data["outcome"]) == "aborted"
        assert str(data["error"]) == "fall cutoff"
        assert data["phase"].tolist() == ["walking"]
        assert data["actuator_names"].tolist() == [
            "left_hip_roll",
            "right_hip_roll",
        ]
        np.testing.assert_allclose(
            data["joint_tracking_error_rad"], [[0.05, -0.05]], atol=1e-6
        )
        np.testing.assert_allclose(
            data["requested_velocity_cmd"], [[0.07, 0.0, 0.0]], atol=1e-6
        )
        assert data["timing_work"].tolist() == [0.01]
        assert data["servo_servo_read_fail_count"].tolist() == [1.0]
        assert "servo_ignored" not in data.files
        np.testing.assert_allclose(data["servo_position_age_s"], [[0.02, 0.04]])


def test_inspect_log_reports_walking_tracking_and_missing_footswitches(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "trial.npz"
    recorder = PolicyTelemetryRecorder(
        output,
        actuator_names=["left_hip_roll", "right_hip_roll"],
        ctrl_dt=0.02,
        bundle_path=tmp_path / "bundle",
        hardware_config_path=tmp_path / "hardware.json",
    )
    recorder.record(
        _info(previous=[0.25, -0.15], observed=[0.20, -0.10]),
        phase="walking",
        loop_step=0,
    )
    recorder.save(outcome="completed")

    inspect_log(output)

    text = capsys.readouterr().out
    assert "Outcome: completed" in text
    assert "Phases: walking=1" in text
    assert "Previous-command vs feedback error top 2" in text
    assert "left_hip_roll=2.865/2.865" in text
    assert "Servo feedback cache age top 2" in text
    assert "foot switches were disabled or unavailable" in text
