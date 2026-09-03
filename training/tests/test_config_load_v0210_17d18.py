from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import load_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d17_safety_continuation.yaml"
)
CONFIG = Path("training/configs/ppo_walking_v0210_17d18_early_torque_margin.yaml")


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["env"].pop("torque_saturation_soft_limit_ratio")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d18_changes_only_saturation_onset_and_metadata() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    cfg = load_training_config(CONFIG)
    assert cfg.version == "0.21.0-17d18"
    assert cfg.env.torque_saturation_soft_limit_ratio == pytest.approx(0.7)
    assert cfg.env.torque_saturation_weights_per_joint == {"left_hip_roll": 1.0}
    assert cfg.reward_weights.saturation == pytest.approx(-0.025)
    assert cfg.reward_weights.ang_vel_xy == pytest.approx(1.0)
    assert cfg.ppo.iterations == 10
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.rollback.enabled is True
