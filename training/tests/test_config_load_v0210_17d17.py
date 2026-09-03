from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import load_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d15_stance_width_retreat.yaml"
)
CONFIG = Path("training/configs/ppo_walking_v0210_17d17_safety_continuation.yaml")


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d17_preserves_17d15_training_contract_for_resume() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    cfg = load_training_config(CONFIG)
    assert cfg.version == "0.21.0-17d17"
    assert cfg.reward_weights.ang_vel_xy == pytest.approx(1.0)
    assert cfg.env.loc_ref_penalty_ang_vel_xy_form == "tb_neg_squared"
    assert cfg.env.loc_ref_default_stance_width_m == pytest.approx(0.0535)
    assert cfg.ppo.iterations == 10
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.reward_weights.saturation == pytest.approx(-0.025)
