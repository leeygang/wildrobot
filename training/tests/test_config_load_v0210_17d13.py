from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from training.configs.training_config import load_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d12_stance_width_retreat.yaml"
)
CONFIG = Path("training/configs/ppo_walking_v0210_17d13_stance_width_retreat.yaml")


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["env"].pop("loc_ref_default_stance_width_m")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d13_changes_only_stance_width_and_metadata() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    cfg = load_training_config(CONFIG)
    assert cfg.version == "0.21.0-17d13"
    assert cfg.env.loc_ref_default_stance_width_m == pytest.approx(0.0515)
    assert cfg.ppo.iterations == 10
    assert cfg.ppo.critic_warmup_iterations == 2
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.rollback.enabled is True
    assert cfg.reward_weights.saturation == pytest.approx(-0.025)
