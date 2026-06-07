"""Pin the v0.21.0 smoke10 (axis-split lateral command reward) contract.

smoke10 = smoke9 (home base + RSI + coverage-1.1 scales + exact signed-vy bins)
RESUMED from smoke6 ckpt_1900, with the velocity reward AXIS-SPLIT:
  * cmd_velocity_track_dim: 2 -> 1   (forward term tracks vx ONLY)
  * NEW reward_weights.cmd_lateral_velocity_track: 0.0 -> 1.0 (standalone
    lateral term exp(-alpha_y*(vy - cmd_vy)^2), alpha_y=120 reused)

Everything else (exact signed-vy curriculum, vy range, probes, gates, yaw off)
is kept identical to smoke9.

Reference: /tmp/wildrobot_smoke10_axis_split_lateral_reward_prompt.md,
ppo_walking_v0210_smoke10_axis_split_lateral_vy.yaml,
CHANGELOG [v0.21.0-smoke9-result + smoke10-plan].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import RewardWeightsConfig

_CFG = Path("training/configs/ppo_walking_v0210_smoke10_axis_split_lateral_vy.yaml")
_SMOKE9 = Path("training/configs/ppo_walking_v0210_smoke9_exact_signed_vy.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


def test_smoke10_loads(cfg) -> None:
    assert cfg is not None


def test_smoke10_axis_split_levers(cfg) -> None:
    """THE smoke10 levers: forward vx-only + standalone lateral term."""
    rw = cfg.reward_weights
    assert rw.cmd_velocity_track_dim == 1
    assert rw.cmd_lateral_velocity_track == pytest.approx(1.0)
    # Forward tracker unchanged from smoke9.
    assert rw.cmd_forward_velocity_track == pytest.approx(2.0)
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)
    assert rw.cmd_forward_velocity_alpha_y == pytest.approx(120.0)
    assert rw.cmd_yaw_rate_track == pytest.approx(0.0)


def test_smoke10_preserves_exact_signed_vy_curriculum(cfg) -> None:
    e = cfg.env
    assert e.cmd_sampler_walk_vy_exact_signed_bins is True
    assert e.min_velocity_y == pytest.approx(-0.065)
    assert e.max_velocity_y == pytest.approx(0.065)
    assert e.max_yaw_rate == pytest.approx(0.0)
    # home base + RSI contract kept.
    assert e.loc_ref_residual_base == "home"
    assert e.loc_ref_rsi_enabled is True


def test_smoke10_keeps_signed_probes_and_gate(cfg) -> None:
    probes = [list(p) for p in cfg.env.eval_velocity_cmd_probes]
    assert probes == [[0.13, 0.065, 0.0], [0.13, -0.065, 0.0]]
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True


# ---------------------------------------------------------------------------
# Regression: smoke9 and defaults remain unchanged.
# ---------------------------------------------------------------------------
def test_smoke9_unchanged() -> None:
    if not _SMOKE9.exists():
        pytest.skip("smoke9 config not found")
    c9 = load_training_config(str(_SMOKE9))
    assert c9.reward_weights.cmd_velocity_track_dim == 2
    assert c9.reward_weights.cmd_lateral_velocity_track == pytest.approx(0.0)


def test_cmd_lateral_velocity_track_default_is_zero() -> None:
    """Default keeps every legacy / dim==2 config byte-equivalent."""
    assert RewardWeightsConfig().cmd_lateral_velocity_track == pytest.approx(0.0)
