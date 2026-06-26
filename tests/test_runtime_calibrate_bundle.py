from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from runtime.scripts.calibrate import (
    CALIBRATION_IMU_SAMPLING_HZ,
    _axis_label,
    _capture_imu_series,
    _format_footswitch_status_line,
    _infer_rotation_axis_from_gyro,
    _init_calibration_bno085,
    _load_axis_map_measurements,
    _wait_for_imu_stream,
    calibrate_imu_axis_map_full,
    calibrate_imu_axis_sign_only,
    compose_policy_action_target_rad,
    evaluate_policy_action,
    JointAxisMetadata,
    JointState,
    load_joint_axis_metadata,
    load_bundle_spec,
    load_policy_action_setup,
    motor_sign_prompt,
    PolicyActionSetup,
    prompt_select_footswitch_targets,
    resolve_config_path,
    resolve_home_ctrl,
    resolve_joint_names,
    resolve_robot_config_path,
    resolve_robot_xml_path,
)
from runtime.configs.config import ServoConfig, WrRuntimeConfig


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WALKING_BUNDLE = _REPO_ROOT / "runtime" / "bundles" / "walking_v0210_smoke6_ckpt1650"


def test_resolve_config_path_defaults_to_bundle_runtime_config() -> None:
    args = argparse.Namespace(config=None, bundle=str(_WALKING_BUNDLE))
    assert resolve_config_path(args) == _WALKING_BUNDLE / "wildrobot_config.json"


def test_resolve_joint_names_uses_bundle_actuator_order() -> None:
    cfg = WrRuntimeConfig.load(_WALKING_BUNDLE / "wildrobot_config.json")
    joint_names = resolve_joint_names(args=argparse.Namespace(bundle=str(_WALKING_BUNDLE)), servo_cfgs=cfg.hiwonder_controller.servos)
    bundle_names, _ = load_bundle_spec(_WALKING_BUNDLE)
    assert joint_names == bundle_names
    assert joint_names != list(cfg.hiwonder_controller.servos.keys())


def test_resolve_home_ctrl_reorders_bundle_home_to_requested_joint_names() -> None:
    args = argparse.Namespace(bundle=str(_WALKING_BUNDLE), scene_xml=None, keyframes_xml=None)

    cfg_order = list(
        json.loads((_WALKING_BUNDLE / "wildrobot_config.json").read_text())["servo_controller"]["servos"].keys()
    )
    home_in_cfg_order = resolve_home_ctrl(args, cfg_order)

    bundle_names, bundle_home = load_bundle_spec(_WALKING_BUNDLE)
    home_by_name = dict(zip(bundle_names, bundle_home, strict=True))
    expected = [home_by_name[name] for name in cfg_order]

    assert home_in_cfg_order == expected
    assert home_in_cfg_order != bundle_home


def test_prompt_select_footswitch_targets_raw_all(monkeypatch) -> None:
    inputs = iter(["all footswitches", "#7"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    assert prompt_select_footswitch_targets() == [
        "left_toe",
        "left_heel",
        "right_toe",
        "right_heel",
    ]
    assert prompt_select_footswitch_targets() == [
        "left_toe",
        "left_heel",
        "right_toe",
        "right_heel",
    ]


def test_format_footswitch_status_line_raw_switches() -> None:
    status = {
        "left_toe": True,
        "left_heel": False,
        "right_toe": True,
        "right_heel": False,
    }
    assert (
        _format_footswitch_status_line(
            status,
            ["left_toe", "left_heel", "right_toe", "right_heel"],
        )
        == "left_toe=1, left_heel=0, right_toe=1, right_heel=0"
    )


def test_wait_for_imu_stream_error_includes_diagnostics() -> None:
    class _StaleImu:
        polling_mode = True
        error_count = 2
        last_error = "stale"
        diag = {"quat_status": "missing"}

        def read(self):
            return SimpleNamespace(
                timestamp_s=0.0,
                valid=False,
                quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                gyro_rad_s=[0.0, 0.0, 0.0],
            )

    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_imu_stream(_StaleImu(), timeout_s=0.0)

    msg = str(excinfo.value)
    assert "timestamp_s not advancing" in msg
    assert "polling_mode=True" in msg
    assert "valid=False" in msg
    assert "last_error=stale" in msg
    assert "diag={'quat_status': 'missing'}" in msg


def test_wait_for_imu_stream_accepts_slow_first_valid_sample() -> None:
    class _SlowFirstValidImu:
        polling_mode = True

        def read(self):
            time.sleep(0.02)
            return SimpleNamespace(
                timestamp_s=123.0,
                valid=True,
                quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                gyro_rad_s=[0.0, 0.0, 0.0],
            )

    assert _wait_for_imu_stream(_SlowFirstValidImu(), timeout_s=0.001) == 123.0


def test_capture_imu_series_keeps_only_fresh_valid_samples(monkeypatch) -> None:
    import runtime.scripts.calibrate as calibrate_mod

    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.001
        return clock["t"]

    monkeypatch.setattr(calibrate_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(calibrate_mod.time, "sleep", lambda _seconds: None)

    def sample(ts: float, valid: bool, gyro_x: float):
        return SimpleNamespace(
            timestamp_s=ts,
            valid=valid,
            quat_xyzw=[gyro_x, 0.0, 0.0, 1.0],
            gyro_rad_s=[gyro_x, 0.0, 0.0],
        )

    class _SequenceImu:
        polling_mode = False

        def __init__(self):
            self.samples = [
                sample(1.0, True, 1.0),
                sample(1.0, True, 2.0),   # duplicate timestamp
                sample(2.0, False, 3.0),  # invalid fresh report
                sample(3.0, True, 4.0),
            ]

        def read(self):
            if self.samples:
                return self.samples.pop(0)
            return sample(3.0, True, 5.0)

    gyro, quat, last_ts = _capture_imu_series(
        _SequenceImu(),
        duration_s=0.03,
        last_ts=0.0,
        sample_dt_s=0.0,
        stall_timeout_s=1.0,
        progress_every_s=0.0,
    )

    assert last_ts == 3.0
    np.testing.assert_allclose(gyro[:, 0], [1.0, 4.0])
    np.testing.assert_allclose(quat[:, 0], [1.0, 4.0])


def test_imu_axis_calibration_uses_background_reader() -> None:
    full_src = inspect.getsource(calibrate_imu_axis_map_full)
    sign_src = inspect.getsource(calibrate_imu_axis_sign_only)
    init_src = inspect.getsource(_init_calibration_bno085)

    assert CALIBRATION_IMU_SAMPLING_HZ == 200
    assert "sampling_hz" in init_src
    assert "_init_calibration_bno085" in full_src
    assert "_init_calibration_bno085" in sign_src
    assert "polling_mode\": False" in init_src
    assert "enable_rotation_vector\": False" in init_src
    assert "polling_mode=True" not in full_src
    assert "polling_mode=True" not in sign_src


def test_infer_rotation_axis_from_gyro_detects_yaw_axis() -> None:
    baseline = np.zeros((20, 3), dtype=np.float32)
    motion = np.zeros((100, 3), dtype=np.float32)
    motion[20:80, 2] = -1.0

    result = _infer_rotation_axis_from_gyro(
        baseline_gyro=baseline,
        motion_gyro=motion,
        duration_s=1.0,
    )

    assert result is not None
    angle_rad, axis = result
    assert angle_rad == pytest.approx(0.6, abs=0.05)
    np.testing.assert_allclose(axis, [0.0, 0.0, -1.0], atol=1e-5)


def test_load_axis_map_measurements_normalizes_valid_axes() -> None:
    raw_config = {
        "bno085": {
            "axis_map_measurements": {
                "body_z": [0.0, 0.0, -2.0],
                "body_y": [0.0, -3.0, 0.0],
                "bad": [1.0, 0.0, 0.0],
                "body_x": [0.0, 0.0],
            }
        }
    }

    axes = _load_axis_map_measurements(raw_config)

    assert sorted(axes) == ["body_y", "body_z"]
    np.testing.assert_allclose(axes["body_z"], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(axes["body_y"], [0.0, -1.0, 0.0])


def test_full_axis_calibration_is_resumable() -> None:
    src = inspect.getsource(calibrate_imu_axis_map_full)

    assert "saved_axes_by_body = _load_axis_map_measurements" in src
    assert "saved measurement to" in src
    assert "reusing saved measurement" in src


def test_axis_metadata_reads_local_and_init_world_axes() -> None:
    cfg = WrRuntimeConfig.load(_WALKING_BUNDLE / "wildrobot_config.json")
    raw_config = json.loads((_WALKING_BUNDLE / "wildrobot_config.json").read_text())
    robot_cfg_path = resolve_robot_config_path(
        raw_config,
        _WALKING_BUNDLE / "wildrobot_config.json",
        bundle_dir=_WALKING_BUNDLE,
    )
    robot_xml_path = resolve_robot_xml_path(
        robot_config_path=robot_cfg_path,
        config=cfg,
        bundle_dir=_WALKING_BUNDLE,
    )

    axes = load_joint_axis_metadata(
        robot_config_path=robot_cfg_path,
        robot_xml_path=robot_xml_path,
    )

    assert _axis_label(axes["left_hip_pitch"].local_axis) == "+Z"
    assert _axis_label(axes["left_hip_pitch"].init_world_axis) == "-Y"
    assert _axis_label(axes["right_hip_pitch"].init_world_axis) == "+Y"


def test_direction_prompt_includes_axis_labels_and_right_hand_rule() -> None:
    prompt = motor_sign_prompt(
        "left_hip_pitch",
        "left leg swings forward",
        0.5,
        JointAxisMetadata(
            local_axis=(0.0, 0.0, 1.0),
            init_world_axis=(0.0, -1.0, 0.0),
        ),
    )

    assert "local_axis: +Z" in prompt
    assert "init_world_axis: -Y" in prompt
    assert "right-hand rule" in prompt
    assert "expected positive motion: left leg swings forward" in prompt


def test_policy_action_setup_uses_runtime_bundle_residual_scale() -> None:
    args = argparse.Namespace(bundle=str(_WALKING_BUNDLE))
    cfg = WrRuntimeConfig.load(_WALKING_BUNDLE / "wildrobot_config.json")
    joint_names = ["left_hip_pitch", "left_hip_roll"]

    setup = load_policy_action_setup(args=args, config=cfg, joint_names=joint_names)

    assert setup.residual_base == "home"
    assert setup.residual_scale_by_joint["left_hip_pitch"] == 0.64
    assert setup.residual_scale_by_joint["left_hip_roll"] == 0.375
    assert "runtime_policy_config.json" in setup.source


def test_policy_action_setup_infers_bundle_from_policy_path() -> None:
    args = argparse.Namespace(bundle=None)
    cfg = WrRuntimeConfig.load(_WALKING_BUNDLE / "wildrobot_config.json")

    setup = load_policy_action_setup(
        args=args,
        config=cfg,
        joint_names=["left_hip_pitch"],
    )

    assert setup.residual_scale_by_joint["left_hip_pitch"] == 0.64
    assert "runtime_policy_config.json" in setup.source


def test_compose_policy_action_target_uses_home_residual_and_joint_clip() -> None:
    eval_result = compose_policy_action_target_rad(
        base_rad=0.2,
        action=2.0,
        residual_scale_rad=0.64,
        rad_range=(-0.5, 0.7),
    )

    assert eval_result.action_applied == 1.0
    assert eval_result.residual_rad == 0.64
    assert abs(eval_result.target_rad_unclipped - 0.84) < 1e-12
    assert eval_result.target_rad == 0.7
    assert eval_result.clipped_by_joint_range is True


def test_policy_action_evaluator_moves_home_then_action(monkeypatch) -> None:
    class FakeController:
        def __init__(self) -> None:
            self.pos = 500
            self.moves: list[tuple[list[tuple[int, int]], int]] = []

        def read_servo_positions(self, servo_ids):
            return [(int(servo_ids[0]), int(self.pos))]

        def move_servos(self, commands, move_ms):
            self.moves.append((commands, int(move_ms)))
            self.pos = int(commands[0][1])

    servo = ServoConfig(
        id=1,
        servo_offset_unit=0,
        motor_unit_direction=1,
        joint_angle_at_zero_unit_deg=0,
        rad_range=(-1.0, 1.0),
    )
    state = JointState(offset=10, motor_sign=1)
    setup = PolicyActionSetup(
        base_rad_by_joint={"left_hip_pitch": 0.2},
        residual_scale_by_joint={"left_hip_pitch": 0.4},
        residual_base="home",
        residual_mode="absolute",
        action_delay_steps=1,
        action_filter_alpha=0.0,
        source="test",
    )
    controller = FakeController()
    inputs = iter(["0.5", "y"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("runtime.scripts.calibrate.time.sleep", lambda seconds: None)

    evaluate_policy_action(
        controller,
        joint="left_hip_pitch",
        servo=servo,
        state=state,
        policy_setup=setup,
        move_ms_fallback=300,
    )

    home_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        0.2,
        motor_sign=1,
        offset=10,
    )
    target_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        0.4,
        motor_sign=1,
        offset=10,
    )

    assert [move[0][0] for move in controller.moves] == [
        (1, home_units),
        (1, target_units),
    ]
