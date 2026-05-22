from __future__ import annotations

import math
from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import RewardWeightsConfig
from training.core.metrics_registry import METRIC_SPEC_BY_NAME, Reducer
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")


def test_cmd_velocity_track_dim_defaults_to_scalar_mode() -> None:
    assert RewardWeightsConfig().cmd_velocity_track_dim == 1


def test_smoke13_sets_2d_cmd_velocity_tracking_mode() -> None:
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    cfg = load_training_config(str(_SMOKE13))
    assert cfg.reward_weights.cmd_velocity_track_dim == 2


def test_scalar_cmd_tracking_ignores_lateral_velocity() -> None:
    reward, xy_err, lat_abs = WildRobotEnv._cmd_forward_velocity_track_reward(
        forward_velocity=0.1333333333,
        lateral_velocity=0.1,
        velocity_cmd=0.1333333333,
        alpha=562.5,
        track_dim=1,
    )
    assert float(reward) == pytest.approx(1.0)
    assert float(xy_err) == pytest.approx(0.1)
    assert float(lat_abs) == pytest.approx(0.1)


def test_2d_cmd_tracking_penalizes_lateral_velocity() -> None:
    reward, _xy_err, _lat_abs = WildRobotEnv._cmd_forward_velocity_track_reward(
        forward_velocity=0.1333333333,
        lateral_velocity=0.1,
        velocity_cmd=0.1333333333,
        alpha=562.5,
        track_dim=2,
    )
    expected = math.exp(-562.5 * 0.01)
    assert float(reward) == pytest.approx(expected)


def test_metrics_registry_has_cmd_velocity_xy_and_lateral_abs() -> None:
    xy = METRIC_SPEC_BY_NAME["tracking/cmd_velocity_xy_err"]
    lat = METRIC_SPEC_BY_NAME["tracking/lateral_velocity_abs"]
    assert xy.reducer == Reducer.MEAN
    assert lat.reducer == Reducer.MEAN

