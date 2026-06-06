"""Pin the v0.21.0 smoke9 (exact signed-vy curriculum) contract.

smoke9 = smoke8 (home base + RSI + coverage-1.1 scales + anisotropic 2D
reward alpha_x=562.5 / alpha_y=120, vy range +-0.065, probes +-0.065)
RESUMED from smoke6 ckpt_1900, with ONE lever:
  * env.cmd_sampler_walk_vy_exact_signed_bins: false -> True
    de-projects vy into the balanced exact signed bins {-0.065, 0, +0.065}
    so the train-side lateral command matches the +-0.065 probes (smoke8
    cos(theta)-projected it down to ~0.026, below the inherited +y bias).

Everything else is kept identical to smoke8.

Reference: /tmp/wildrobot_smoke9_exact_vy_prompt.md,
ppo_walking_v0210_smoke9_exact_signed_vy.yaml,
CHANGELOG [v0.21.0-smoke8-result + smoke9-plan].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config

_CFG = Path("training/configs/ppo_walking_v0210_smoke9_exact_signed_vy.yaml")
_SMOKE8 = Path("training/configs/ppo_walking_v0210_smoke8_2d_signed_vy.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


def test_smoke9_loads(cfg) -> None:
    assert cfg is not None


def test_smoke9_keeps_home_rsi_contract(cfg) -> None:
    """smoke6/8 contract preserved: home base + RSI + coverage-1.1 scales."""
    e = cfg.env
    assert e.loc_ref_residual_base == "home"
    assert e.loc_ref_rsi_enabled is True
    s = e.loc_ref_residual_scale_per_joint
    assert s["left_knee_pitch"] >= 0.889
    assert s["left_hip_pitch"] >= 0.601
    assert s["left_hip_pitch"] == pytest.approx(s["right_hip_pitch"])  # symmetric


def test_smoke9_the_only_lever_is_vy_exact_bins(cfg) -> None:
    """THE smoke9 lever: vy is de-projected into exact signed bins."""
    e = cfg.env
    assert e.cmd_sampler_walk_vy_exact_signed_bins is True
    # vx de-projection (smoke8) stays on; vx is unaffected by the vy lever.
    assert e.cmd_sampler_walk_vx_positive_only is True
    assert e.cmd_sampler_3d_branched is True


def test_smoke9_keeps_smoke8_2d_reward(cfg) -> None:
    """alpha_x/alpha_y, track_dim, and vy range are UNCHANGED from smoke8."""
    e = cfg.env
    rw = cfg.reward_weights
    assert rw.cmd_velocity_track_dim == 2
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)   # alpha_x tight
    assert rw.cmd_forward_velocity_alpha_y == pytest.approx(120.0)  # alpha_y (smoke8)
    assert rw.cmd_forward_velocity_track > 0.0
    assert e.min_velocity_y == pytest.approx(-0.065)
    assert e.max_velocity_y == pytest.approx(0.065)
    assert e.max_yaw_rate == pytest.approx(0.0)               # NO yaw command


def test_smoke9_yaw_reward_off(cfg) -> None:
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(0.0)


def test_smoke9_lateral_drift_is_hard_gate(cfg) -> None:
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True


def test_smoke9_signed_probes_at_pm_0p065(cfg) -> None:
    """Eval probes stay at the +-0.065 signed pair so the train-side exact
    bins match the authoritative probe magnitude exactly."""
    probes = [list(p) for p in cfg.env.eval_velocity_cmd_probes]
    lat = sorted(round(float(p[1]), 4) for p in probes if abs(float(p[1])) >= 0.02)
    assert lat == [-0.065, 0.065], f"expected +-0.065 signed probes; got {lat}"


def test_smoke9_reference_grid_includes_pm_0p065(cfg) -> None:
    """Each exact bin {-0.065, 0, +0.065} must map to a real reference row
    (nearest-key L2 must snap +-0.065 -> itself, not the 0.0 bin)."""
    grid = [round(float(v), 4) for v in cfg.env.loc_ref_offline_command_vy_grid]
    assert 0.065 in grid and -0.065 in grid, f"vy grid missing +-0.065: {grid}"
    assert min(grid, key=lambda g: abs(g - 0.065)) == pytest.approx(0.065)
    assert min(grid, key=lambda g: abs(g + 0.065)) == pytest.approx(-0.065)


def test_smoke8_does_not_set_vy_exact_bins() -> None:
    """Backward-compat: smoke8 keeps the cos(theta) ellipse (flag default
    False), so smoke8 reproducibility from disk is unchanged."""
    if not _SMOKE8.exists():
        pytest.skip("smoke8 config not found")
    c8 = load_training_config(str(_SMOKE8))
    assert c8.env.cmd_sampler_walk_vy_exact_signed_bins is False
