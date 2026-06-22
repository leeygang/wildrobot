from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.scripts.calibrate import (
    _axis_label,
    compose_policy_action_target_rad,
    load_joint_axis_metadata,
    load_bundle_spec,
    load_policy_action_setup,
    resolve_config_path,
    resolve_home_ctrl,
    resolve_joint_names,
    resolve_robot_config_path,
    resolve_robot_xml_path,
)
from runtime.configs.config import WrRuntimeConfig


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
