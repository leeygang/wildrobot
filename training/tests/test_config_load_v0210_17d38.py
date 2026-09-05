from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d37_contact_free_cautious_resume.yaml"
)
CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d38_contact_free_source_anchor.yaml"
)


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["ppo"].pop("source_policy_kl_coef")
    raw["ppo"].pop("source_policy_kl_limit")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d38_changes_only_source_policy_guard_and_metadata() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.21.0-17d38"
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert cfg.ppo.iterations == 10
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-7)
    assert cfg.ppo.critic_warmup_iterations == 2
    assert cfg.ppo.rollback.enabled is True
    assert cfg.ppo.rollback.patience == 1
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_limit == pytest.approx(0.003)
    assert cfg.reward_weights.ang_vel_xy == pytest.approx(1.0)
    assert cfg.reward_weights.saturation == pytest.approx(-0.025)
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17
