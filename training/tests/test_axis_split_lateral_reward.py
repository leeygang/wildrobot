"""v0.21.0 smoke10 — axis-split standalone lateral command reward.

Covers the new ``reward_weights.cmd_lateral_velocity_track`` term:
  * the lateral FACTOR is non-neutral under ``cmd_velocity_track_dim == 1`` when
    the standalone weight > 0, and stays NEUTRAL 1.0 when the weight == 0
    (legacy forward-only behavior).
  * the weighted dt-scaled contribution is EMITTED end-to-end and reflects the
    weight × factor (catches the silent-drop signature).
  * the new ``reward/cmd_lateral_velocity_track_contrib`` metric is registered.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.core.metrics_registry import METRIC_SPEC_BY_NAME, Reducer
from training.envs.wildrobot_env import WildRobotEnv

_SMOKE10 = Path("training/configs/ppo_walking_v0210_smoke10_axis_split_lateral_vy.yaml")


def _smoke10_cfg():
    if not _SMOKE10.exists():
        pytest.skip(f"{_SMOKE10.name} not found")
    return load_training_config(str(_SMOKE10))


# ---------------------------------------------------------------------------
# Gate: dim==1 + weight>0 activates the lateral factor; weight==0 stays neutral.
# ---------------------------------------------------------------------------
def test_lateral_track_active_under_dim1_with_weight() -> None:
    env = WildRobotEnv(config=_smoke10_cfg())
    assert env._cmd_velocity_track_dim == 1
    assert env._cmd_lateral_track_active is True


def test_lateral_track_neutral_under_dim1_without_weight() -> None:
    cfg = _smoke10_cfg()
    rw = dataclasses.replace(cfg.reward_weights, cmd_lateral_velocity_track=0.0)
    env = WildRobotEnv(config=dataclasses.replace(cfg, reward_weights=rw))
    assert env._cmd_velocity_track_dim == 1
    assert env._cmd_lateral_track_active is False


# ---------------------------------------------------------------------------
# E2E: factor varies with (vy - cmd_vy) and the weighted contribution emits.
# ---------------------------------------------------------------------------
def _step_with_vy_cmd(env, vy_cmd: float):
    import jax
    import jax.numpy as jp

    from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
    from training.envs.env_info import WR_INFO_KEY

    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(
        velocity_cmd=jp.array([0.13, vy_cmd, 0.0], dtype=jp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    state2 = env.step(state, jp.zeros(env.action_size))
    return unpack_metrics(state2.metrics[METRICS_VEC_KEY])


def test_lateral_factor_nonneutral_and_contrib_emits_dim1_weight() -> None:
    """dim==1 + cmd_lateral_velocity_track=1.0: forcing a nonzero vy_cmd with
    vy_actual~0 (first step) drives the RAW factor below 1.0 and emits a
    positive weighted contribution."""
    env = WildRobotEnv(config=_smoke10_cfg())
    metrics = _step_with_vy_cmd(env, vy_cmd=0.065)

    assert "reward/cmd_lateral_velocity_track" in metrics
    assert "reward/cmd_lateral_velocity_track_contrib" in metrics
    raw = float(metrics["reward/cmd_lateral_velocity_track"])
    contrib = float(metrics["reward/cmd_lateral_velocity_track_contrib"])

    # vy_actual ~ 0 at first step, vy_err ~ -0.065 -> factor = exp(-120*0.065^2)
    # ~ 0.60, clearly non-neutral.
    assert raw < 0.95, f"lateral factor not engaged (raw={raw})"
    # Weighted contribution = weight(1.0) * factor * dt(0.02) ~ 0.012 > 0.
    assert contrib > 1e-3, f"lateral contribution silently zero (contrib={contrib})"


def test_lateral_factor_neutral_when_weight_zero_dim1() -> None:
    """dim==1 + weight 0: factor stays NEUTRAL 1.0 and contribution stays 0
    (legacy forward-only behavior preserved)."""
    cfg = _smoke10_cfg()
    rw = dataclasses.replace(cfg.reward_weights, cmd_lateral_velocity_track=0.0)
    env = WildRobotEnv(config=dataclasses.replace(cfg, reward_weights=rw))
    metrics = _step_with_vy_cmd(env, vy_cmd=0.065)

    raw = float(metrics["reward/cmd_lateral_velocity_track"])
    contrib = float(metrics["reward/cmd_lateral_velocity_track_contrib"])
    assert raw == pytest.approx(1.0), f"expected neutral factor, got {raw}"
    assert contrib == pytest.approx(0.0, abs=1e-9)


def test_lateral_factor_decays_with_error() -> None:
    """Sharper: a larger |vy - cmd_vy| must give a SMALLER factor (the term is
    a live function of the lateral command error, not a constant)."""
    env = WildRobotEnv(config=_smoke10_cfg())
    raw_small = float(
        _step_with_vy_cmd(env, vy_cmd=0.03)["reward/cmd_lateral_velocity_track"]
    )
    env2 = WildRobotEnv(config=_smoke10_cfg())
    raw_big = float(
        _step_with_vy_cmd(env2, vy_cmd=0.065)["reward/cmd_lateral_velocity_track"]
    )
    # |err| at vy_cmd=0.065 > |err| at vy_cmd=0.03 (vy_actual~0 both) -> smaller.
    assert raw_big < raw_small


def test_contrib_metric_registered_with_mean_reducer() -> None:
    spec = METRIC_SPEC_BY_NAME["reward/cmd_lateral_velocity_track_contrib"]
    assert spec.reducer == Reducer.MEAN
