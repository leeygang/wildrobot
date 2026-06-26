from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.scripts.calibrate import (
    _axis_label,
    _format_footswitch_status_line,
    _wait_for_imu_stream,
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
