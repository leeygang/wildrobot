from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d46_knee_torque_headroom.yaml"
)
CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d47_non_rsi_backpitch_recovery.yaml"
)


def _normalized_training_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["env"].pop("reset_torso_pitch_rate_range", None)
    raw["env"].pop("standing_recovery_enabled", None)
    raw["env"].pop("standing_recovery_reset_non_rsi_only", None)
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d47_changes_only_non_rsi_backpitch_recovery() -> None:
    assert _normalized_training_contract(CONFIG) == _normalized_training_contract(
        BASE_CONFIG
    )

    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.21.0-17d47"
    assert cfg.env.reset_torso_roll_range == pytest.approx([-0.1, 0.1])
    assert cfg.env.reset_torso_pitch_range == pytest.approx([-0.1, 0.1])
    assert cfg.env.reset_torso_roll_rate_range == pytest.approx([0.0, 0.0])
    assert cfg.env.reset_torso_pitch_rate_range == pytest.approx([-0.80, -0.35])
    assert cfg.env.reset_foot_stagger_range_m == pytest.approx([0.0, 0.0])
    assert cfg.env.standing_recovery_enabled is True
    assert cfg.env.standing_recovery_reset_non_rsi_only is True
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert cfg.ppo.iterations == 5
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-7)
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17


def test_17d47_recovery_rate_only_changes_non_rsi_resets() -> None:
    env = WildRobotEnv(config=load_training_config(CONFIG))
    base_env = WildRobotEnv(config=load_training_config(BASE_CONFIG))

    home_state = env.reset(jax.random.PRNGKey(0))
    assert float(np.asarray(home_state.info[WR_INFO_KEY].reset_is_rsi)) == 0.0
    assert float(home_state.data.qvel[3]) == pytest.approx(0.0)
    assert -0.80 <= float(home_state.data.qvel[4]) <= -0.35

    rsi_state = env.reset(jax.random.PRNGKey(2))
    base_rsi_state = base_env.reset(jax.random.PRNGKey(2))
    assert float(np.asarray(rsi_state.info[WR_INFO_KEY].reset_is_rsi)) == 1.0
    np.testing.assert_allclose(
        np.asarray(rsi_state.data.qvel),
        np.asarray(base_rsi_state.data.qvel),
        atol=1e-7,
    )

    eval_state = env.reset_for_eval(jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(eval_state.data.qvel[3:6]), 0.0, atol=1e-6)
