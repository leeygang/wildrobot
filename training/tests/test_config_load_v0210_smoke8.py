"""Pin the v0.21.0 smoke8 (2D signed-vy) contract + the new lateral diagnostics.

smoke8 = smoke7 (home+RSI, anisotropic 2D reward) with the SIGNED lateral signal
strengthened so the policy tracks both +vy and -vy (smoke7 learned a one-sided
+y bias).  Levers vs smoke7:
  * env.min/max_velocity_y: ±0.03 -> ±0.065   (command exceeds the ~0.05 bias)
  * reward_weights.cmd_forward_velocity_alpha_y: 30 -> 120
  * eval_velocity_cmd_probes: [0.13,±0.03,0] -> [0.13,±0.065,0]
  * NEW diagnostics: reward/cmd_lateral_velocity_track + tracking/cmd_vy_err

Reference: /tmp/wildrobot_smoke8_prompt.md, ppo_walking_v0210_smoke8_2d_signed_vy.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from training.configs.training_config import load_training_config
from training.core.metrics_registry import (
    METRIC_NAMES,
    METRICS_VEC_KEY,
    unpack_metrics,
)
from training.envs.wildrobot_env import WildRobotEnv

_CFG = Path("training/configs/ppo_walking_v0210_smoke8_2d_signed_vy.yaml")
_SMOKE7 = Path("training/configs/ppo_walking_v0210_smoke7_2d_tiny_vy.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


@pytest.fixture(scope="module")
def env(cfg):
    return WildRobotEnv(config=cfg)


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------
def test_smoke8_keeps_contract(cfg) -> None:
    e, rw = cfg.env, cfg.reward_weights
    assert e.loc_ref_residual_base == "home"
    assert e.loc_ref_rsi_enabled is True
    assert rw.cmd_velocity_track_dim == 2
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)   # alpha_x kept
    assert rw.cmd_yaw_rate_track == pytest.approx(0.0)             # yaw off
    assert e.max_yaw_rate == pytest.approx(0.0)
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True
    # coverage-1.1 scales unchanged
    s = e.loc_ref_residual_scale_per_joint
    assert s["left_knee_pitch"] >= 0.889
    assert s["left_hip_pitch"] == pytest.approx(s["right_hip_pitch"])


def test_smoke8_signed_lateral_levers(cfg) -> None:
    e, rw = cfg.env, cfg.reward_weights
    assert e.min_velocity_y == pytest.approx(-0.065)
    assert e.max_velocity_y == pytest.approx(0.065)
    assert rw.cmd_forward_velocity_alpha_y == pytest.approx(120.0)
    grid = [round(float(v), 4) for v in e.loc_ref_offline_command_vy_grid]
    assert 0.065 in grid and -0.065 in grid


def test_smoke8_probes_at_pm_0065(cfg) -> None:
    probes = [list(p) for p in cfg.env.eval_velocity_cmd_probes]
    vys = sorted(round(float(p[1]), 4) for p in probes)
    assert vys == [-0.065, 0.065], f"expected ±0.065 probes, got {probes}"
    # both above the Appendix C min-cmd floor (0.02) so neither is skipped
    assert all(abs(float(p[1])) >= 0.02 for p in probes)


# ---------------------------------------------------------------------------
# Lateral diagnostics — registration + emission (reset and step)
# ---------------------------------------------------------------------------
def test_lateral_diagnostics_registered() -> None:
    assert "reward/cmd_lateral_velocity_track" in METRIC_NAMES
    assert "tracking/cmd_vy_err" in METRIC_NAMES


def test_lateral_diagnostics_emitted_and_consistent(env, cfg) -> None:
    """Both diagnostics are emitted at reset AND after a step (exercises both
    emission paths), and the factor matches exp(-alpha_y * cmd_vy_err^2)."""
    alpha_y = cfg.reward_weights.cmd_forward_velocity_alpha_y

    st = env.reset(jax.random.PRNGKey(0))
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    assert "reward/cmd_lateral_velocity_track" in m
    assert "tracking/cmd_vy_err" in m

    act = jp.zeros(env.action_size)
    for _ in range(3):
        st = env.step(st, act)
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    vy_err = float(m["tracking/cmd_vy_err"])
    factor = float(m["reward/cmd_lateral_velocity_track"])
    assert vy_err >= 0.0
    assert 0.0 <= factor <= 1.0
    # factor must equal the lateral Gaussian of the error (single source of truth)
    assert factor == pytest.approx(math.exp(-alpha_y * vy_err * vy_err), rel=1e-3, abs=1e-4)


def test_smoke7_lateral_diagnostics_present_backward_compat() -> None:
    """The diagnostics are registered globally, so older 2D configs (smoke7)
    also surface them — registration is config-independent."""
    assert "reward/cmd_lateral_velocity_track" in METRIC_NAMES
    assert "tracking/cmd_vy_err" in METRIC_NAMES
    if _SMOKE7.exists():
        c7 = load_training_config(str(_SMOKE7))
        # smoke7 stays at its own values (unchanged by smoke8)
        assert c7.reward_weights.cmd_forward_velocity_alpha_y == pytest.approx(30.0)
        assert c7.env.max_velocity_y == pytest.approx(0.03)
