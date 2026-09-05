from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import (
    WildRobotEnv,
    projected_gravity_orientation_penalty,
)
from training.policy_spec_utils import build_policy_spec_from_training_config


BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_17d39_contact_free_source_anchor_refresh.yaml"
)
CONFIG = Path("training/configs/ppo_walking_v0210_17d40_tilt_penalty.yaml")


def _normalized_contract(path: Path) -> dict:
    raw = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
    raw.pop("version")
    raw.pop("version_name")
    raw["reward_weights"].pop("orientation", None)
    raw["checkpoints"].pop("dir")
    raw["wandb"].pop("tags")
    return raw


def test_17d40_changes_only_orientation_reward_and_metadata() -> None:
    assert _normalized_contract(CONFIG) == _normalized_contract(BASE_CONFIG)

    source = load_training_config(BASE_CONFIG)
    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert source.reward_weights.orientation == pytest.approx(0.0)
    assert cfg.reward_weights.orientation == pytest.approx(-0.5)
    assert cfg.env.actor_obs_layout_id == "wr_obs_v11_cmd3d_proprio"
    assert cfg.ppo.iterations == 10
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-7)
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(1.0)
    assert cfg.ppo.source_policy_kl_limit == pytest.approx(0.003)
    assert spec.model.obs_dim == 873
    assert spec.model.action_dim == 17


def test_projected_gravity_orientation_penalty_is_yaw_invariant() -> None:
    identity = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    yaw_90 = jnp.asarray([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=jnp.float32)
    pitch_30 = jnp.asarray(
        [jnp.cos(jnp.pi / 12), 0.0, jnp.sin(jnp.pi / 12), 0.0],
        dtype=jnp.float32,
    )

    assert float(projected_gravity_orientation_penalty(identity)) == pytest.approx(0.0)
    assert float(projected_gravity_orientation_penalty(yaw_90)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(projected_gravity_orientation_penalty(pitch_30)) == pytest.approx(
        0.25, abs=1e-6
    )


def test_orientation_penalty_is_weighted_and_included_in_total() -> None:
    env = WildRobotEnv(config=load_training_config(CONFIG))
    terms = defaultdict(lambda: jnp.float32(0.0))
    terms["penalty_orientation"] = jnp.float32(0.25)

    contrib = env._aggregate_reward(terms, jnp.float32(0.0))

    expected = -0.5 * 0.25 * env.dt
    assert float(contrib["orientation"]) == pytest.approx(expected)
    assert float(contrib["total"]) == pytest.approx(
        float(contrib["alive"]) + expected
    )
