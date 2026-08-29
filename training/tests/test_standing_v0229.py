from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


_CFG = Path("training/configs/ppo_standing_stabilizer_v0229_stance_health.yaml")


def test_v0229_is_short_actor_only_stance_health_diagnostic() -> None:
    cfg = load_training_config(str(_CFG))
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg.version == "0.22.9"
    assert spec.observation.layout_id == "wr_obs_v9_standing"
    assert spec.model.obs_dim == 59
    assert spec.model.action_dim == 17
    assert cfg.ppo.iterations == 40
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-5)
    assert cfg.reward_weights.feet_distance == pytest.approx(1.0)
    assert cfg.reward_weights.penalty_close_feet_xy == pytest.approx(10.0)
    assert cfg.env.close_feet_threshold == pytest.approx(0.146)
    assert cfg.env.min_feet_y_dist == pytest.approx(0.171)
    assert cfg.env.max_feet_y_dist == pytest.approx(0.317)
    assert cfg.env.domain_rand_persistent_torso_pitch_error_range == pytest.approx(
        (-0.218166, 0.218166)
    )
    assert cfg.ppo.eval.post_training_num_envs == 128
    assert cfg.ppo.eval.post_training_num_steps == 1000


def test_v0229_stance_health_metrics_are_registered() -> None:
    from training.core.metrics_registry import METRIC_NAMES

    assert "support/feet_lateral_distance_m" in METRIC_NAMES
    assert "support/feet_too_close_frac" in METRIC_NAMES


def test_v0229_emits_exact_close_feet_condition() -> None:
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(str(_CFG))
    load_robot_config(cfg.env.robot_config_path)
    env = WildRobotEnv(config=cfg)
    state = jax.jit(env.reset_for_eval)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(
        state,
        jnp.zeros(env.action_size, dtype=jnp.float32),
    )
    metrics = state.metrics[METRICS_VEC_KEY]
    distance = float(metrics[METRIC_INDEX["support/feet_lateral_distance_m"]])
    too_close = float(metrics[METRIC_INDEX["support/feet_too_close_frac"]])

    assert distance >= 0.0
    assert too_close in (0.0, 1.0)
    assert too_close == float(distance < cfg.env.close_feet_threshold)
