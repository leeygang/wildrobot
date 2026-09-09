"""Tests for the measured single-support COM-lever intervention."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.core.experiment_tracking import REWARD_TERM_KEYS
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.wildrobot_env import (
    WildRobotEnv,
    single_support_com_lateral_penalty,
)
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb6_single_support_com_lever.yaml")
CHAMPION_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb4_hip_roll_margin_resume.yaml"
)


@pytest.mark.parametrize(
    ("left_loaded", "right_loaded", "expected"),
    [
        (True, False, 1.0),
        (False, True, 1.0),
        (True, True, 0.0),
        (False, False, 0.0),
    ],
)
def test_com_lever_penalty_is_gated_to_measured_single_support(
    left_loaded: bool,
    right_loaded: bool,
    expected: float,
) -> None:
    penalty = single_support_com_lateral_penalty(
        whole_body_com=jnp.asarray([0.0, 0.0, 0.5]),
        left_foot_pos=jnp.asarray([0.0, 0.1, 0.0]),
        right_foot_pos=jnp.asarray([0.0, -0.1, 0.0]),
        root_quat_wxyz=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        left_loaded=jnp.asarray(left_loaded),
        right_loaded=jnp.asarray(right_loaded),
    )

    assert float(penalty) == pytest.approx(expected)


def test_com_lever_penalty_uses_torso_lateral_axis() -> None:
    half_sqrt_two = 2.0**-0.5
    penalty = single_support_com_lateral_penalty(
        whole_body_com=jnp.asarray([0.0, 0.0, 0.5]),
        left_foot_pos=jnp.asarray([0.05, 0.0, 0.0]),
        right_foot_pos=jnp.asarray([0.0, 0.0, 0.0]),
        root_quat_wxyz=jnp.asarray(
            [half_sqrt_two, 0.0, 0.0, half_sqrt_two]
        ),
        left_loaded=jnp.asarray(True),
        right_loaded=jnp.asarray(False),
    )

    assert float(penalty) == pytest.approx(0.25)


def test_com_lever_reward_is_weighted_and_included_in_total() -> None:
    cfg = load_training_config(CONFIG)
    env_stub = SimpleNamespace(_config=cfg, dt=cfg.env.ctrl_dt)
    terms = defaultdict(lambda: jnp.float32(0.0))
    terms["penalty_single_support_com_lateral"] = jnp.float32(1.0)

    contrib = WildRobotEnv._aggregate_reward(
        env_stub,
        terms,
        jnp.float32(0.0),
    )

    expected = -0.05 * cfg.env.ctrl_dt
    assert float(contrib["single_support_com_lateral"]) == pytest.approx(expected)
    assert float(contrib["total"]) == pytest.approx(
        float(contrib["alive"]) + expected
    )
    assert "reward/single_support_com_lateral" in METRIC_INDEX
    assert "reward/single_support_com_lateral" in REWARD_TERM_KEYS


def test_com_lever_metric_emits_on_environment_step() -> None:
    cfg = load_training_config(CONFIG)
    env = WildRobotEnv(config=cfg)
    state = env.reset(jnp.asarray([0, 0], dtype=jnp.uint32))

    next_state = env.step(
        state,
        jnp.zeros(env.action_size, dtype=jnp.float32),
    )
    metric_index = METRIC_INDEX["reward/single_support_com_lateral"]

    assert float(next_state.metrics[METRICS_VEC_KEY][metric_index]) <= 0.0


def test_tb6_changes_only_com_lever_reward_and_run_bookkeeping() -> None:
    champion = load_training_config(CHAMPION_CONFIG)
    cfg = load_training_config(CONFIG)

    assert cfg.version == "0.21.0-tb6"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 819_200
    assert cfg.ppo.eval.interval == 10
    assert cfg.checkpoints.interval == 10
    assert cfg.ppo.eval.post_training_top_k == 4
    assert cfg.reward_weights.single_support_com_lateral == pytest.approx(-0.05)
    assert cfg.reward_weights.saturation == champion.reward_weights.saturation

    champion_raw = copy.deepcopy(champion.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (champion_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["ppo"].pop("iterations")
        raw["ppo"]["eval"].pop("interval")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["reward_weights"].pop("single_support_com_lateral", None)
        raw["checkpoints"].pop("dir")
        raw["checkpoints"].pop("interval")
        raw["wandb"].pop("tags")
    assert cfg_raw == champion_raw


def test_tb6_preserves_champion_actor_contract() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    champion = load_training_config(CHAMPION_CONFIG)
    cfg = load_training_config(CONFIG)

    champion_spec = build_policy_spec_from_training_config(
        training_cfg=champion,
        robot_cfg=robot_cfg,
    )
    cfg_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg_spec.to_json_dict() == champion_spec.to_json_dict()
