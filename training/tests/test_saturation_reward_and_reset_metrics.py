"""Regression coverage for torque saturation shaping and reset-origin metrics."""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from training.configs.training_config import load_training_config
from training.core.metrics_registry import (
    METRIC_INDEX,
    METRICS_VEC_KEY,
    NUM_METRICS,
    unpack_metrics,
)
from training.core.training_loop import _aggregate_reset_origin_metrics
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import (
    WildRobotEnv,
    torque_saturation_penalty,
)


_SOURCE_CONFIG = "training/configs/ppo_walking_v0210_17d4_startup_mix.yaml"
_FINETUNE_CONFIG = "training/configs/ppo_walking_v0210_17d5_saturation_finetune.yaml"


def test_torque_saturation_penalty_uses_normalized_soft_limit_excess() -> None:
    penalty = torque_saturation_penalty(
        actuator_force=jnp.asarray([0.0, -9.5, 9.75, 10.0]),
        force_limits=jnp.asarray([10.0, 10.0, 10.0, 10.0]),
    )

    # Ratios [0, .95, .975, 1] map to normalized excess [0, 0, .5, 1].
    assert float(penalty) == pytest.approx(1.25)


def test_saturation_reward_is_weighted_and_included_in_total() -> None:
    cfg = load_training_config(_SOURCE_CONFIG)
    cfg = dataclasses.replace(
        cfg,
        reward_weights=dataclasses.replace(cfg.reward_weights, saturation=-0.1),
    )
    env = WildRobotEnv(config=cfg)
    terms = defaultdict(lambda: jnp.float32(0.0))
    terms["penalty_saturation"] = jnp.float32(1.25)

    contrib = env._aggregate_reward(terms, jnp.float32(0.0))

    expected = -0.1 * 1.25 * env.dt
    assert float(contrib["saturation"]) == pytest.approx(expected)
    assert float(contrib["total"]) == pytest.approx(
        float(contrib["alive"]) + expected
    )


def test_saturation_metric_emits_on_environment_step() -> None:
    env = WildRobotEnv(config=load_training_config(_FINETUNE_CONFIG))
    state = env.reset(jax.random.PRNGKey(0))

    next_state = env.step(state, jnp.zeros(env.action_size, dtype=jnp.float32))
    metrics = unpack_metrics(next_state.metrics[METRICS_VEC_KEY])

    assert "reward/saturation" in metrics
    assert float(metrics["reward/saturation"]) <= 0.0


def test_saturation_is_opt_in_and_17d5_is_the_only_optimization_change() -> None:
    source = load_training_config(_SOURCE_CONFIG)
    finetune = load_training_config(_FINETUNE_CONFIG)

    assert source.reward_weights.saturation == pytest.approx(0.0)
    assert finetune.reward_weights.saturation == pytest.approx(-0.1)
    assert finetune.reward_weights.torque == source.reward_weights.torque
    assert finetune.env.loc_ref_rsi_probability == pytest.approx(0.75)
    assert finetune.reward_weights.cmd_lateral_velocity_track == pytest.approx(1.0)
    assert finetune.ppo.iterations == 60
    assert finetune.ppo.eval.post_training_top_k == 6
    assert finetune.ppo.eval.post_training_strict_lateral_drift is True
    assert finetune.ppo.eval.post_training_strict_walking_safety is True

    source_raw = copy.deepcopy(source.raw_config)
    finetune_raw = copy.deepcopy(finetune.raw_config)
    for raw in (source_raw, finetune_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["ppo"].pop("iterations")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["reward_weights"].pop("saturation", None)
        raw["checkpoints"].pop("dir")
        raw["wandb"].pop("tags")
    assert finetune_raw == source_raw


@pytest.mark.parametrize("rsi_probability,expected_rsi", [(0.0, 0.0), (1.0, 1.0)])
def test_reset_records_whether_episode_started_from_rsi(
    rsi_probability: float, expected_rsi: float
) -> None:
    cfg = load_training_config(_SOURCE_CONFIG)
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            loc_ref_rsi_probability=rsi_probability,
        ),
    )
    env = WildRobotEnv(config=cfg)

    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    metrics = unpack_metrics(state.metrics[METRICS_VEC_KEY])

    assert float(wr.reset_is_rsi) == pytest.approx(expected_rsi)
    assert float(metrics["reset/is_rsi"]) == pytest.approx(expected_rsi)


def test_reset_origin_aggregation_splits_exposure_and_episode_outcomes() -> None:
    metrics = jnp.zeros((4, 2, NUM_METRICS), dtype=jnp.float32)
    metrics = metrics.at[..., METRIC_INDEX["reset/is_rsi"]].set(
        jnp.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
    )
    metrics = metrics.at[..., METRIC_INDEX["reset/event"]].set(
        jnp.asarray(
            [
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
    )
    metrics = metrics.at[..., METRIC_INDEX["episode_step_count"]].set(
        jnp.asarray(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [1.0, 4.0],
            ]
        )
    )
    dones = jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    truncations = jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = _aggregate_reset_origin_metrics(metrics, dones, truncations)

    assert float(result["reset/home_reset_count"]) == pytest.approx(1.0)
    assert float(result["reset/rsi_reset_count"]) == pytest.approx(2.0)
    assert float(result["reset/home_reset_frac"]) == pytest.approx(1.0 / 3.0)
    assert float(result["reset/rsi_reset_frac"]) == pytest.approx(2.0 / 3.0)
    assert float(result["reset/home_episode_count"]) == pytest.approx(1.0)
    assert float(result["reset/rsi_episode_count"]) == pytest.approx(1.0)
    assert float(result["reset/home_episode_length"]) == pytest.approx(3.0)
    assert float(result["reset/rsi_episode_length"]) == pytest.approx(4.0)
    assert float(result["reset/home_success_rate"]) == pytest.approx(0.0)
    assert float(result["reset/rsi_success_rate"]) == pytest.approx(1.0)
    assert float(result["reset/home_failure_rate"]) == pytest.approx(1.0)
    assert float(result["reset/rsi_failure_rate"]) == pytest.approx(0.0)


def test_reset_origin_metrics_are_forwarded_to_wandb() -> None:
    from training.core.experiment_tracking import build_wandb_metrics

    env_metrics = {
        "forward_velocity": 0.12,
        "tracking/avg_torque": 0.7,
        "velocity_command": 0.13,
        "tracking/vel_error": 0.01,
        "tracking/max_torque": 0.98,
        "height": 0.46,
        "reset/home_reset_frac": 0.25,
        "reset/rsi_reset_frac": 0.75,
        "reset/home_failure_rate": 0.10,
        "reset/rsi_failure_rate": 0.02,
    }
    iteration_metrics = SimpleNamespace(
        episode_reward=1.0,
        total_loss=0.1,
        policy_loss=0.05,
        value_loss=0.05,
        entropy_loss=0.01,
        clip_fraction=0.1,
        approx_kl=0.01,
        success_rate=0.9,
        episode_length=900.0,
        task_reward_mean=0.05,
        env_metrics=env_metrics,
    )

    emitted, _ = build_wandb_metrics(
        iteration=1,
        metrics=iteration_metrics,
        steps_per_sec=100.0,
        reward_terms=[],
    )

    assert emitted["reset/home_reset_frac"] == pytest.approx(0.25)
    assert emitted["reset/rsi_reset_frac"] == pytest.approx(0.75)
    assert emitted["reset/home_failure_rate"] == pytest.approx(0.10)
    assert emitted["reset/rsi_failure_rate"] == pytest.approx(0.02)
