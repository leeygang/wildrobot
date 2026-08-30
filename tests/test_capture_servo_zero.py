from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import runtime.scripts.capture_servo_zero as capture_mod
from runtime.configs.config import ServoConfig
from runtime.scripts.calibrate import DetectedServoBoard
from runtime.scripts.capture_servo_zero import (
    LEG_PITCH_JOINTS,
    analyze_zero_samples,
    build_candidate_config,
    build_controller_from_discovered_ports,
    collect_position_samples,
    resolve_joint_selection,
    training_pitch_bias_summary,
    validate_output_paths,
)


def test_resolve_joint_selection_supports_leg_pitch_and_explicit_lists() -> None:
    available = [*LEG_PITCH_JOINTS, "waist_yaw"]
    assert resolve_joint_selection("leg-pitch", available) == list(
        LEG_PITCH_JOINTS
    )
    assert resolve_joint_selection(
        "left_hip_pitch, right_hip_pitch", available
    ) == ["left_hip_pitch", "right_hip_pitch"]
    with pytest.raises(ValueError, match="missing"):
        resolve_joint_selection("missing_joint", available)


def test_collect_position_samples_requires_every_sample() -> None:
    class FakeController:
        responses = [
            [(1, 479), (3, 540)],
            [(1, 480), (3, 540)],
            [(1, 479), (3, 541)],
        ]

        def read_servo_positions(self, _servo_ids):
            return self.responses.pop(0)

    samples = collect_position_samples(
        FakeController(),
        servo_ids=[1, 3],
        sample_count=3,
        sample_interval_s=0.0,
    )
    assert samples == {1: [479, 480, 479], 3: [540, 540, 541]}


def test_analyze_zero_samples_computes_offsets_and_stability() -> None:
    servos = {
        "left_hip_pitch": ServoConfig(
            id=1,
            servo_offset_unit=-22,
            motor_unit_direction=-1,
            joint_angle_at_servo_center_deg=0.0,
        ),
        "left_knee_pitch": ServoConfig(
            id=3,
            servo_offset_unit=42,
            motor_unit_direction=-1,
            joint_angle_at_servo_center_deg=0.0,
        ),
    }
    captures = analyze_zero_samples(
        joint_names=servos,
        servo_cfgs=servos,
        samples_by_servo_id={1: [479, 479, 480], 3: [539, 540, 543]},
        max_spread_units=2,
    )
    left_hip, left_knee = captures
    assert left_hip.median_raw_units == 479
    assert left_hip.suggested_offset_unit == -21
    assert left_hip.offset_change_unit == 1
    assert left_hip.stable is True
    assert left_hip.current_deg_at_physical_zero == pytest.approx(-0.24)
    assert left_hip.implied_physical_target_error_deg == pytest.approx(0.24)
    assert left_knee.spread_units == 4
    assert left_knee.stable is False


def test_build_controller_uses_discovered_ports_for_selected_boards(
    monkeypatch,
) -> None:
    detected = [
        DetectedServoBoard("left_leg_board", "/dev/new-left", (1, 3, 4)),
        DetectedServoBoard("right_leg_board", "/dev/new-right", (5, 7, 8)),
    ]
    built_configs = []
    controller = object()
    monkeypatch.setattr(
        capture_mod,
        "discover_servo_boards",
        lambda **_kwargs: detected,
    )

    def _build(config):
        built_configs.append(config)
        return controller

    monkeypatch.setattr(capture_mod, "build_calibration_controller", _build)
    servo_cfgs = {
        joint: ServoConfig(id=servo_id)
        for joint, servo_id in zip(LEG_PITCH_JOINTS, (1, 3, 4, 5, 7, 8))
    }

    result, boards = build_controller_from_discovered_ports(
        servo_controller_config=SimpleNamespace(
            type="hiwonder_ttl_bus",
            port="/dev/stale",
            baudrate=115200,
        ),
        servo_cfgs=servo_cfgs,
        joint_names=LEG_PITCH_JOINTS,
    )

    assert result is controller
    assert boards == detected
    assert [board.port for board in built_configs[0].boards] == [
        "/dev/new-left",
        "/dev/new-right",
    ]


def test_build_controller_rejects_missing_selected_board(monkeypatch) -> None:
    monkeypatch.setattr(
        capture_mod,
        "discover_servo_boards",
        lambda **_kwargs: [
            DetectedServoBoard("left_leg_board", "/dev/new-left", (1, 3, 4))
        ],
    )
    servo_cfgs = {
        joint: ServoConfig(id=servo_id)
        for joint, servo_id in zip(LEG_PITCH_JOINTS, (1, 3, 4, 5, 7, 8))
    }

    with pytest.raises(RuntimeError, match="right_leg_board"):
        build_controller_from_discovered_ports(
            servo_controller_config=SimpleNamespace(
                type="hiwonder_ttl_bus",
                port="/dev/stale",
                baudrate=115200,
            ),
            servo_cfgs=servo_cfgs,
            joint_names=LEG_PITCH_JOINTS,
        )


def test_main_does_not_retry_unload_after_initial_unload_failure(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "hardware_config.json"
    config_path.write_text("{}")
    servo = ServoConfig(id=1)
    config = SimpleNamespace(
        hiwonder_controller=SimpleNamespace(
            servos={"left_hip_pitch": servo},
        )
    )

    class FailingController:
        def __init__(self):
            self.unload_calls = 0
            self.closed = False

        def unload_servos(self, _servo_ids):
            self.unload_calls += 1
            raise RuntimeError("primary unload failure")

        def close(self):
            self.closed = True

    controller = FailingController()
    monkeypatch.setattr(
        capture_mod,
        "_parse_args",
        lambda: SimpleNamespace(
            samples=7,
            sample_interval_s=0.0,
            max_spread_units=2,
            config=config_path,
            joints="left_hip_pitch",
            report=tmp_path / "report.json",
            output_config=None,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(capture_mod.WrRuntimeConfig, "load", lambda _path: config)
    monkeypatch.setattr("builtins.input", lambda _prompt: "UNLOAD")
    monkeypatch.setattr(
        capture_mod,
        "build_controller_from_discovered_ports",
        lambda **_kwargs: (controller, []),
    )

    with pytest.raises(RuntimeError, match="primary unload failure"):
        capture_mod.main()

    assert controller.unload_calls == 1
    assert controller.closed is True


def test_training_pitch_bias_summary_matches_training_sign_pattern() -> None:
    servos = {
        joint: ServoConfig(
            id=index + 1,
            servo_offset_unit=0,
            motor_unit_direction=-1,
            joint_angle_at_servo_center_deg=0.0,
        )
        for index, joint in enumerate(LEG_PITCH_JOINTS)
    }
    samples = {
        1: [501],
        2: [498],
        3: [498],
        4: [498],
        5: [498],
        6: [498],
    }
    captures = analyze_zero_samples(
        joint_names=LEG_PITCH_JOINTS,
        servo_cfgs=servos,
        samples_by_servo_id=samples,
        max_spread_units=0,
    )
    summary = training_pitch_bias_summary(captures)
    assert summary is not None
    assert summary["left_deg"] == pytest.approx(-0.24)
    assert summary["right_deg"] == pytest.approx(-0.48)
    assert summary["average_deg"] == pytest.approx(-0.36)


def test_build_candidate_config_changes_only_captured_offsets() -> None:
    raw_config = {
        "servo_controller": {
            "boards": [
                {
                    "name": "left_leg_board",
                    "port": "/dev/old-left",
                    "servo_ids": [1, 3],
                }
            ],
            "servos": {
                "left_hip_pitch": {"id": 1, "servo_offset_unit": -22},
                "left_knee_pitch": {"id": 3, "servo_offset_unit": 42},
            }
        },
        "untouched": {"value": 7},
    }
    servos = {
        "left_hip_pitch": ServoConfig(
            id=1,
            servo_offset_unit=-22,
            motor_unit_direction=-1,
        )
    }
    captures = analyze_zero_samples(
        joint_names=servos,
        servo_cfgs=servos,
        samples_by_servo_id={1: [479, 479, 479]},
        max_spread_units=0,
    )

    candidate = build_candidate_config(
        raw_config,
        captures,
        [DetectedServoBoard("left_leg_board", "/dev/new-left", (1,))],
    )

    assert candidate["servo_controller"]["servos"]["left_hip_pitch"][
        "servo_offset_unit"
    ] == -21
    assert candidate["servo_controller"]["servos"]["left_knee_pitch"][
        "servo_offset_unit"
    ] == 42
    assert candidate["untouched"] == {"value": 7}
    assert candidate["servo_controller"]["boards"][0]["port"] == "/dev/new-left"
    assert raw_config["servo_controller"]["servos"]["left_hip_pitch"][
        "servo_offset_unit"
    ] == -22
    assert raw_config["servo_controller"]["boards"][0]["port"] == "/dev/old-left"


def test_validate_output_paths_protects_active_config_and_separates_outputs() -> None:
    config = Path("/tmp/hardware_config.json")
    report = Path("/tmp/zero_report.json")
    candidate = Path("/tmp/hardware_config.zero_candidate.json")

    validate_output_paths(
        config_path=config,
        report_path=report,
        output_config=candidate,
    )
    with pytest.raises(ValueError, match="report"):
        validate_output_paths(
            config_path=config,
            report_path=config,
            output_config=candidate,
        )
    with pytest.raises(ValueError, match="active hardware config"):
        validate_output_paths(
            config_path=config,
            report_path=report,
            output_config=config,
        )
    with pytest.raises(ValueError, match="different paths"):
        validate_output_paths(
            config_path=config,
            report_path=report,
            output_config=report,
        )
