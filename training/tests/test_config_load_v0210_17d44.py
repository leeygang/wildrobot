from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d38_contact_free_source_anchor.yaml"
)
CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d44_backpitch_recovery_curriculum.yaml"
)


def _normalized_training_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["ppo"].pop("iterations")
    raw["ppo"]["eval"].pop("post_training_top_k")
    for field in (
        "reset_torso_pitch_range",
        "reset_torso_roll_rate_range",
        "reset_torso_pitch_rate_range",
        "reset_foot_stagger_range_m",
        "standing_recovery_enabled",
    ):
        raw["env"].pop(field, None)
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d44_changes_only_measured_recovery_resets_and_budget() -> None:
    assert _normalized_training_contract(CONFIG) == _normalized_training_contract(
        BASE_CONFIG
    )

    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.21.0-17d44"
    assert cfg.env.reset_torso_roll_range == pytest.approx([-0.1, 0.1])
    assert cfg.env.reset_torso_pitch_range == pytest.approx([-0.20, -0.10])
    assert cfg.env.reset_torso_roll_rate_range == pytest.approx([-0.40, 0.25])
    assert cfg.env.reset_torso_pitch_rate_range == pytest.approx([-1.10, -0.35])
    assert cfg.env.reset_foot_stagger_range_m == pytest.approx([0.0, 0.0])
    assert cfg.env.standing_recovery_enabled is True
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert cfg.ppo.iterations == 5
    assert cfg.ppo.eval.post_training_top_k == 5
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-7)
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_limit == pytest.approx(0.003)
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17


def test_17d44_training_reset_applies_rates_but_clean_eval_does_not() -> None:
    env = WildRobotEnv(config=load_training_config(CONFIG))

    train_state = env.reset(jax.random.PRNGKey(0))
    train_qvel = np.asarray(train_state.data.qvel)
    assert -0.40 <= train_qvel[3] <= 0.25
    assert -1.10 <= train_qvel[4] <= -0.35

    eval_state = env.reset_for_eval(jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(eval_state.data.qvel[3:6]), 0.0, atol=1e-6)
