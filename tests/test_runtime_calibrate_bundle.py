from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.scripts.calibrate import (
    load_bundle_spec,
    resolve_config_path,
    resolve_home_ctrl,
    resolve_joint_names,
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
