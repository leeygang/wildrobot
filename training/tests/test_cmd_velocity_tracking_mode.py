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
    """Dim=1 reward must be ``exp(-alpha * (vx_actual - velocity_cmd)^2)``
    and ignore both ``lateral_velocity`` and the reference velocity
    inputs.  ``cmd_velocity_xy_err`` is the actual-vs-commanded
    diagnostic (kept under both dims); ``ref_velocity_xy_err`` is
    always 0 under dim=1."""
    reward, xy_err, lat_abs, ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.1333333333,
            lateral_velocity=0.1,
            velocity_cmd=0.1333333333,
            # Adversarial: huge ref velocity that would crater dim=2
            # but is irrelevant to dim=1.
            ref_forward_velocity=99.0,
            ref_lateral_velocity=-99.0,
            alpha=562.5,
            track_dim=1,
        )
    )
    assert float(reward) == pytest.approx(1.0)
    assert float(xy_err) == pytest.approx(0.1)
    assert float(lat_abs) == pytest.approx(0.1)
    assert float(ref_err) == pytest.approx(0.0)


def test_2d_cmd_tracking_against_reference_velocity() -> None:
    """Dim=2 reward is TB-style: it compares actual (vx, vy) to the
    reference's (ref_vx, ref_vy), NOT to (velocity_cmd, 0).  Pin this
    by feeding actual = reference but ≠ (velocity_cmd, 0) and
    asserting the reward is 1.0 (zero error against the reference)."""
    reward, xy_err, lat_abs, ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.10,
            lateral_velocity=0.02,
            # Sampled command for this episode is 0.1333 — DIFFERENT
            # from the actual / reference pair.  If the reward still
            # used (velocity_cmd, 0) like the pre-smoke13 dim=2 code,
            # the reward would be far below 1.
            velocity_cmd=0.1333333333,
            ref_forward_velocity=0.10,
            ref_lateral_velocity=0.02,
            alpha=562.5,
            track_dim=2,
        )
    )
    assert float(reward) == pytest.approx(1.0), (
        "dim=2 reward must be 1.0 when actual == reference, regardless "
        "of velocity_cmd — proves it tracks the cmd-conditioned ref, "
        "not (velocity_cmd, 0)."
    )
    assert float(ref_err) == pytest.approx(0.0)
    # cmd-distance diagnostic is unaffected by the reward path.
    expected_cmd_err = math.sqrt((0.10 - 0.1333333333) ** 2 + 0.02 ** 2)
    assert float(xy_err) == pytest.approx(expected_cmd_err)
    assert float(lat_abs) == pytest.approx(0.02)


def test_2d_cmd_tracking_penalizes_lateral_velocity_against_zero_ref() -> None:
    """Sanity check that the dim=2 reward still penalizes lateral
    velocity when the reference's lateral velocity is zero (the
    typical near-frame-0 case).  ``ref_vx`` set to velocity_cmd so
    the forward axis contributes zero error — only the lateral
    component drives the loss."""
    reward, _xy_err, _lat_abs, ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.1333333333,
            lateral_velocity=0.1,
            velocity_cmd=0.1333333333,
            ref_forward_velocity=0.1333333333,
            ref_lateral_velocity=0.0,
            alpha=562.5,
            track_dim=2,
        )
    )
    expected = math.exp(-562.5 * 0.01)
    assert float(reward) == pytest.approx(expected)
    assert float(ref_err) == pytest.approx(0.1)


def test_metrics_registry_has_cmd_and_ref_velocity_err() -> None:
    xy = METRIC_SPEC_BY_NAME["tracking/cmd_velocity_xy_err"]
    lat = METRIC_SPEC_BY_NAME["tracking/lateral_velocity_abs"]
    ref = METRIC_SPEC_BY_NAME["tracking/ref_velocity_xy_err"]
    assert xy.reducer == Reducer.MEAN
    assert lat.reducer == Reducer.MEAN
    assert ref.reducer == Reducer.MEAN

