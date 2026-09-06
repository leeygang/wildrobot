from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d46_knee_torque_headroom.yaml"
)
CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d48_targeted_failure_state_replay.yaml"
)


def _normalized_training_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw.pop("bootstrap")
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d48_changes_only_targeted_replay_bootstrap() -> None:
    assert _normalized_training_contract(CONFIG) == _normalized_training_contract(
        BASE_CONFIG
    )

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bootstrap = raw["bootstrap"]
    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.21.0-17d48"
    assert bootstrap["mode"] == "contact_observed_to_proprio"
    assert bootstrap["student_checkpoint"].endswith("checkpoint_3_61440.pkl")
    assert bootstrap["student_checkpoint_sha256"] == (
        "42dc6f42ac77dc4495140e5f714918987516ed697434476dde41088fd8919e96"
    )
    assert "auto-16-eval-failure_diagnostic-04b23eb0" in bootstrap["failure_trace"]
    assert bootstrap["failure_trace_sha256"] == (
        "e21660563071ae844ca94bcd6e4e4ffc2022ddfb5948540da9d4f1c77f2e8962"
    )
    assert bootstrap["failure_replay_repeats"] == 16
    assert bootstrap["failure_replay_max_pitch_rad"] == pytest.approx(-0.10)
    assert bootstrap["epochs"] == 10
    assert bootstrap["learning_rate"] == pytest.approx(1.0e-5)
    assert bootstrap["max_validation_rmse"] == pytest.approx(0.02)
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert cfg.ppo.iterations == 5
    assert cfg.ppo.eval.post_training_top_k == 5
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-7)
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_limit == pytest.approx(0.003)
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17
