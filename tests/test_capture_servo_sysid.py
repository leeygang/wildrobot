from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from runtime.configs.config import ServoConfig
from runtime.scripts.capture_servo_sysid import (
    ProfileSegment,
    _empty_capture_arrays,
    add_standard_capture_arrays,
    build_fixture_servo_config,
    build_profile_segments,
    capture_profile,
    evaluate_manual_horizontal_fixture,
    estimate_delay_metrics,
    estimate_step_response_metrics,
    load_mujoco_fixture,
    prepare_servo_center,
    summarize_capture,
    validate_health,
    validate_profile,
    wait_for_cooldown,
    write_capture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _servo() -> ServoConfig:
    return ServoConfig(
        id=6,
        servo_offset_unit=34,
        motor_unit_direction=-1.0,
        joint_angle_at_servo_center_deg=0.0,
        rad_range=(-math.pi / 2.0, math.pi / 2.0),
    )


def test_fixture_servo_uses_raw_centered_htd_coordinates() -> None:
    servo = build_fixture_servo_config(100)

    assert servo.id == 100
    assert servo.joint_target_rad_to_elect_unit(0.0) == 500
    assert servo.joint_target_rad_to_elect_unit(math.radians(8.0)) == 533
    assert servo.servo_elect_units_to_joint_target_rad(500) == 0.0
    assert servo.rad_range == pytest.approx(
        (-ServoConfig.RANGE_RAD / 2.0, ServoConfig.RANGE_RAD / 2.0)
    )


def test_profile_contains_symmetric_steps_and_toddlerbot_style_chirps() -> None:
    segments = build_profile_segments(
        center_rad=0.0,
        amplitudes_rad=np.deg2rad([2.0, 5.0]),
        sample_hz=50.0,
        settle_s=1.0,
        step_hold_s=1.0,
        chirp_duration_s=10.0,
        chirp_start_hz=0.1,
        chirp_end_hz=2.0,
        chirp_decay_rate=0.05,
    )

    assert len(segments) == 13
    assert segments[1].name == "step_positive_2deg"
    assert segments[3].name == "step_negative_2deg"
    assert segments[9].name == "chirp_2deg"
    assert len(segments[9].targets_rad) == 500
    validate_profile(
        segments,
        servo=_servo(),
        joint_limit_margin_rad=math.radians(2.0),
        sample_hz=50.0,
        max_chirp_speed_rad_s=2.5,
    )


def test_profile_validation_rejects_guarded_joint_limit_violation() -> None:
    servo = ServoConfig(
        id=6,
        motor_unit_direction=-1.0,
        rad_range=(math.radians(-10.0), math.radians(90.0)),
    )
    segments = build_profile_segments(
        center_rad=0.0,
        amplitudes_rad=[math.radians(9.0)],
        sample_hz=50.0,
        settle_s=0.1,
        step_hold_s=0.1,
        chirp_duration_s=1.0,
        chirp_start_hz=0.1,
        chirp_end_hz=1.0,
        chirp_decay_rate=0.05,
    )

    with pytest.raises(ValueError, match="guarded joint range"):
        validate_profile(
            segments,
            servo=servo,
            joint_limit_margin_rad=math.radians(2.0),
            sample_hz=50.0,
            max_chirp_speed_rad_s=2.5,
        )


def test_delay_estimate_recovers_known_sample_lag() -> None:
    sample_hz = 50.0
    t = np.arange(500, dtype=np.float64) / sample_hz
    target = np.sin(2.0 * np.pi * (0.4 * t + 0.15 * t * t))
    lag = 4
    position = np.concatenate((np.full(lag, target[0]), target[:-lag]))

    metrics = estimate_delay_metrics(
        target,
        position,
        sample_hz=sample_hz,
        max_delay_s=0.3,
    )

    assert metrics["delay_samples"] == lag
    assert metrics["delay_s"] == pytest.approx(lag / sample_hz)
    assert float(metrics["correlation"]) > 0.99


def test_capture_records_transmitted_command_after_write_deadband() -> None:
    servo = ServoConfig(id=1, rad_range=(-1.0, 1.0))

    class FakeBus:
        position = 500
        writes: list[int] = []

        def move_time_write(self, _servo_id, position, _move_time_ms):
            self.position = int(position)
            self.writes.append(int(position))

        def read_position(self, _servo_id):
            return self.position

    bus = FakeBus()
    arrays = capture_profile(
        bus,
        servo_id=1,
        servo=servo,
        segments=(
            ProfileSegment(
                name="test",
                kind="chirp",
                targets_rad=(0.0, 0.004, 0.008, 0.020),
            ),
        ),
        sample_hz=1000.0,
        move_time_ms=20,
        write_deadband_units=3,
        max_position_error_rad=1.0,
        read_retries=1,
        read_retry_sleep_s=0.0,
    )

    assert arrays["command_written"].tolist() == [True, False, False, True]
    assert arrays["applied_servo_units"].tolist() == [500, 500, 500, 505]
    assert np.isfinite(arrays["command_write_elapsed_s"][[0, 3]]).all()
    assert np.isnan(arrays["command_write_elapsed_s"][[1, 2]]).all()
    assert np.all(arrays["command_age_at_read_s"] >= 0.0)
    assert arrays["scheduler_wait_s"].shape == (4,)
    assert bus.writes == [500, 505]

    summary = summarize_capture(
        arrays,
        segments=(
            ProfileSegment(
                name="test",
                kind="chirp",
                targets_rad=(0.0, 0.004, 0.008, 0.020),
            ),
        ),
        sample_hz=1000.0,
        max_delay_s=0.1,
    )
    assert summary["command_write_fraction"] == pytest.approx(0.5)
    assert summary["timing_ms"]["position_read"]["max"] >= 0.0
    assert summary["segments"]["test"]["command_write_fraction"] == pytest.approx(0.5)


def test_step_response_metrics_report_gain_and_wait_times() -> None:
    metrics = estimate_step_response_metrics(
        np.full(10, 1.0),
        np.asarray([0.0, 0.1, 0.4, 0.7, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9]),
        baseline_position_rad=np.zeros(5),
        sample_hz=10.0,
    )

    assert metrics["steady_gain"] == pytest.approx(0.9)
    assert metrics["response_t10_s"] == pytest.approx(0.1)
    assert metrics["response_t50_s"] == pytest.approx(0.3)
    assert metrics["response_t90_s"] == pytest.approx(0.4)
    assert metrics["response_rise_10_90_s"] == pytest.approx(0.3)


def test_summary_handles_failure_before_profile_samples() -> None:
    summary = summarize_capture(
        _empty_capture_arrays(),
        segments=(
            ProfileSegment(name="step_positive_2deg", kind="hold", targets_rad=(0.1,)),
        ),
        sample_hz=50.0,
        max_delay_s=0.6,
    )

    assert summary["samples"] == 0
    assert summary["segments"]["step_positive_2deg"]["samples"] == 0


def test_standard_arrays_preserve_legacy_sysid_contract_and_load_torque() -> None:
    arrays = {
        "position_elapsed_s": np.asarray([0.0, 0.02, 0.04]),
        "target_joint_rad": np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        "applied_joint_rad": np.asarray([0.0, 0.08, 0.18], dtype=np.float32),
        "position_joint_rad": np.asarray([0.0, 0.05, 0.15], dtype=np.float32),
    }

    result = add_standard_capture_arrays(
        arrays,
        fixture_qpos_rad=np.asarray([0.0, 0.05, 0.15], dtype=np.float32),
        estimated_hold_torque_nm=np.asarray([0.980665, 0.9, 0.8]),
        estimated_load_inertia_kg_m2=np.asarray([0.01, 0.01, 0.01]),
    )

    np.testing.assert_array_equal(result["timestamps_s"], arrays["position_elapsed_s"])
    np.testing.assert_array_equal(result["command_rad"], arrays["applied_joint_rad"])
    np.testing.assert_array_equal(
        result["requested_command_rad"], arrays["target_joint_rad"]
    )
    np.testing.assert_array_equal(
        result["measured_position_rad"], arrays["position_joint_rad"]
    )
    np.testing.assert_allclose(result["measured_velocity_rad_s"], [0.0, 2.5, 5.0])
    assert result["estimated_gravity_torque_nm"][0] == pytest.approx(0.980665)


def test_manual_horizontal_fixture_matches_center_torque() -> None:
    qpos, torque, inertia = evaluate_manual_horizontal_fixture(
        np.asarray([0.0, math.radians(8.0)]),
        center_rad=0.0,
        signed_load_moment_kg_m=0.1,
        load_inertia_kg_m2=0.02,
    )

    np.testing.assert_allclose(qpos, [0.0, math.radians(8.0)])
    assert torque[0] == pytest.approx(0.980665)
    assert torque[1] == pytest.approx(0.980665 * math.cos(math.radians(8.0)))
    np.testing.assert_allclose(inertia, [0.02, 0.02])


def test_bam_fixture_is_fixed_one_dof_and_loads_the_weighted_arm() -> None:
    fixture = load_mujoco_fixture(
        REPO_ROOT / "assets" / "bam" / "robot.xml",
        joint_name="pitch",
        direction=1,
        qpos_offset_rad=0.0,
    )

    qpos, torque, inertia = fixture.evaluate(
        np.deg2rad(np.asarray([0.0, 8.0], dtype=np.float64))
    )

    assert fixture.root_body_name == "fixed_top"
    assert fixture.moving_body_name == "innor_support"
    assert fixture.moving_subtree_mass_kg == pytest.approx(0.612761)
    np.testing.assert_allclose(qpos, np.deg2rad([0.0, 8.0]), atol=1e-7)
    assert torque[0] == pytest.approx(-0.0035567, abs=1e-6)
    assert torque[1] == pytest.approx(0.0941045, abs=1e-6)
    np.testing.assert_allclose(inertia, [0.00933525, 0.00933525], atol=1e-7)


def test_write_capture_writes_npz_and_manifest_without_overwrite(tmp_path) -> None:
    output = tmp_path / "capture.npz"
    arrays = {
        "timestamps_s": np.asarray([0.0, 0.02]),
        "command_rad": np.asarray([0.0, 0.1], dtype=np.float32),
        "measured_position_rad": np.asarray([0.0, 0.08], dtype=np.float32),
        "measured_velocity_rad_s": np.asarray([0.0, 4.0], dtype=np.float32),
    }

    npz_path, json_path = write_capture(
        output,
        arrays=arrays,
        metadata={"outcome": "completed"},
        summary={"samples": 2},
    )

    with np.load(npz_path) as payload:
        assert set(arrays).issubset(payload.files)
        assert int(payload["schema_version"]) == 4
    assert json.loads(json_path.read_text())["summary"]["samples"] == 2
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_capture(
            output,
            arrays=arrays,
            metadata={},
            summary={},
        )


def test_health_validation_requires_telemetry_and_applies_limits() -> None:
    with pytest.raises(RuntimeError, match="telemetry is unavailable"):
        validate_health(
            {"voltage_v": None, "temperature_c": 25.0},
            min_voltage_v=9.0,
            max_temperature_c=60.0,
        )
    with pytest.raises(RuntimeError, match="below"):
        validate_health(
            {"voltage_v": 8.9, "temperature_c": 25.0},
            min_voltage_v=9.0,
            max_temperature_c=60.0,
        )


def test_cooldown_records_temperature_voltage_and_actual_wait() -> None:
    class FakeBus:
        temperatures = iter((42, 38, 34))

        def read_voltage_v(self, _servo_id):
            return 11.6

        def read_temperature_c(self, _servo_id):
            return next(self.temperatures)

    now = [0.0]

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    samples, waited_s = wait_for_cooldown(
        FakeBus(),
        servo_id=100,
        target_temperature_c=35.0,
        timeout_s=30.0,
        poll_s=5.0,
        min_voltage_v=9.0,
        max_temperature_c=60.0,
        monotonic_fn=monotonic,
        sleep_fn=sleep,
    )

    assert waited_s == pytest.approx(10.0)
    assert [sample["temperature_c"] for sample in samples] == [42, 38, 34]
    assert [sample["cooldown_elapsed_s"] for sample in samples] == [0.0, 5.0, 10.0]
    assert all(sample["voltage_v"] == pytest.approx(11.6) for sample in samples)


def test_center_preparation_retries_and_records_diagnostics() -> None:
    class FakeBus:
        positions = iter((562, 625))
        writes: list[tuple[int, int]] = []

        def move_time_write(self, _servo_id, position, move_time_ms):
            self.writes.append((int(position), int(move_time_ms)))

        def read_move_time(self, _servo_id):
            return self.writes[-1]

        def read_position(self, _servo_id):
            return next(self.positions)

        def read_voltage_v(self, _servo_id):
            return 11.6

        def read_temperature_c(self, _servo_id):
            return 35

        def read_loaded(self, _servo_id):
            return True

    now = [0.0]

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    attempts: list[dict[str, object]] = []
    centered_units, centered_rad, commanded_s, actual_s = prepare_servo_center(
        FakeBus(),
        servo_id=100,
        servo=build_fixture_servo_config(100),
        center_rad=math.radians(30.0),
        initial_units=500,
        prepare_speed_deg_s=20.0,
        move_time_ms=20,
        settle_s=1.0,
        center_tolerance_deg=2.0,
        max_attempts=3,
        min_voltage_v=9.0,
        max_temperature_c=60.0,
        read_retries=1,
        read_retry_sleep_s=0.0,
        attempts=attempts,
        monotonic_fn=monotonic,
        sleep_fn=sleep,
    )

    assert centered_units == 625
    assert math.degrees(centered_rad) == pytest.approx(30.0)
    assert commanded_s == pytest.approx(3.0)
    assert actual_s == pytest.approx(5.0)
    assert len(attempts) == 2
    assert attempts[0]["accepted_target_units"] == 625
    assert attempts[0]["measured_position_units"] == 562
    assert attempts[0]["progress_deg"] == pytest.approx(14.88)
    assert attempts[1]["error_deg"] == pytest.approx(0.0)


def test_center_preparation_stops_after_bounded_retries() -> None:
    class StalledBus:
        def move_time_write(self, _servo_id, _position, _move_time_ms):
            return None

        def read_move_time(self, _servo_id):
            return 625, 1500

        def read_position(self, _servo_id):
            return 562

        def read_voltage_v(self, _servo_id):
            return 11.6

        def read_temperature_c(self, _servo_id):
            return 36

        def read_loaded(self, _servo_id):
            return True

    attempts: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        prepare_servo_center(
            StalledBus(),
            servo_id=100,
            servo=build_fixture_servo_config(100),
            center_rad=math.radians(30.0),
            initial_units=500,
            prepare_speed_deg_s=20.0,
            move_time_ms=20,
            settle_s=0.1,
            center_tolerance_deg=2.0,
            max_attempts=3,
            min_voltage_v=9.0,
            max_temperature_c=60.0,
            read_retries=1,
            read_retry_sleep_s=0.0,
            attempts=attempts,
            monotonic_fn=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )

    assert len(attempts) == 3
    assert attempts[-1]["measured_position_units"] == 562
