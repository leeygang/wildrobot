from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import load_training_config_from_dict
from training.scripts.run_stance_pose_training_screen import (
    ARM_SPECS,
    build_arm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    PROJECT_ROOT
    / "training/configs/ppo_walking_v0210_tb11_reference_geometry_com_lever.yaml"
)


def _base() -> dict:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


def _without_allowed_differences(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("version")
    comparable.pop("version_name")
    comparable.pop("seed")
    comparable["env"].pop("home_joint_offsets_rad")
    comparable["env"].pop("loc_ref_default_stance_width_m")
    comparable["env"].pop("close_feet_threshold")
    comparable["checkpoints"].pop("dir")
    comparable["wandb"].pop("tags")
    return comparable


@pytest.mark.parametrize("arm_name", tuple(ARM_SPECS))
def test_stance_pose_arm_config_is_loadable_and_exact(arm_name: str) -> None:
    base = _base()
    config = build_arm_config(base, arm_name=arm_name, seed=43, iterations=5)
    parsed = load_training_config_from_dict(config)
    spec = ARM_SPECS[arm_name]

    assert parsed.seed == 43
    assert parsed.ppo.iterations == 5
    assert parsed.ppo.eval.interval == 5
    assert parsed.ppo.eval.post_training_top_k == 1
    assert parsed.env.loc_ref_residual_base == "home"
    assert parsed.env.loc_ref_default_stance_width_m == pytest.approx(
        spec["stance_half_width_m"]
    )
    assert parsed.env.close_feet_threshold == pytest.approx(
        spec["close_feet_threshold_m"]
    )
    roll = float(spec["roll_offset_rad"])
    assert parsed.env.home_joint_offsets_rad == pytest.approx(
        {
            "left_hip_roll": roll,
            "right_hip_roll": -roll,
            "left_ankle_roll": -roll,
            "right_ankle_roll": roll,
        }
    )
    assert "tb11" not in parsed.wandb.tags
    assert "diagnostic_410k" not in parsed.wandb.tags
    assert "tb12_pose_ab" in parsed.wandb.tags
    assert f"pose_{arm_name}" in parsed.wandb.tags


def test_stance_pose_arms_differ_only_by_whitelisted_fields() -> None:
    base = _base()
    configs = [
        build_arm_config(base, arm_name=name, seed=42, iterations=5)
        for name in ARM_SPECS
    ]

    expected = _without_allowed_differences(configs[0])
    assert all(_without_allowed_differences(config) == expected for config in configs)
    assert base == _base(), "building arms must not mutate the source config"
