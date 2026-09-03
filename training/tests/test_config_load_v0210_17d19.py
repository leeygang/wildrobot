from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d18_early_torque_margin.yaml"
)
CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d19_contact_free_proprio.yaml"
)


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["env"].pop("actor_obs_layout_id")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d19_changes_only_actor_observation_contract_and_metadata() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.21.0-17d19"
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17
    assert "foot_switches" not in {
        field.name for field in spec.observation.layout
    }
