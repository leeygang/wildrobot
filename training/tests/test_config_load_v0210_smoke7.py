"""Pin the v0.21.0 smoke7 (2D tiny-vy) contract.

smoke7 = smoke6 (home base + RSI + coverage-1.1 scales) RESUMED, plus:
  * anisotropic 2D velocity reward: cmd_velocity_track_dim 1->2,
    cmd_forward_velocity_alpha_y 0->30 (forward alpha_x=562.5 kept)
  * branched (vx, vy, 0) sampler with tiny vy ±0.03, yaw off
  * lateral drift promoted to a HARD promotion gate

Reference: /tmp/wildrobot_smoke7_2d_prompt.md, ppo_walking_v0210_smoke7_2d_tiny_vy.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config

_CFG = Path("training/configs/ppo_walking_v0210_smoke7_2d_tiny_vy.yaml")
_SMOKE6 = Path("training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


def test_smoke7_loads(cfg) -> None:
    assert cfg is not None


def test_smoke7_keeps_home_rsi_contract(cfg) -> None:
    """smoke6 contract is preserved: home base + RSI + the coverage-1.1 scales."""
    e = cfg.env
    assert e.loc_ref_residual_base == "home"
    assert e.loc_ref_rsi_enabled is True
    s = e.loc_ref_residual_scale_per_joint
    # coverage-1.1 scales unchanged from smoke6 (full-range max sizing)
    assert s["left_knee_pitch"] >= 0.889
    assert s["left_hip_pitch"] >= 0.601
    assert s["left_hip_pitch"] == pytest.approx(s["right_hip_pitch"])  # symmetric


def test_smoke7_enables_branched_tiny_vy_sampler(cfg) -> None:
    e = cfg.env
    assert e.cmd_sampler_3d_branched is True
    assert e.min_velocity_y == pytest.approx(-0.03)
    assert e.max_velocity_y == pytest.approx(0.03)
    assert e.max_yaw_rate == pytest.approx(0.0)  # NO yaw command


def test_smoke7_anisotropic_2d_reward(cfg) -> None:
    rw = cfg.reward_weights
    assert rw.cmd_velocity_track_dim == 2
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)        # alpha_x tight
    assert 20.0 <= rw.cmd_forward_velocity_alpha_y <= 40.0              # alpha_y loose
    assert rw.cmd_forward_velocity_track > 0.0                          # term active


def test_smoke7_yaw_reward_off(cfg) -> None:
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(0.0)


def test_smoke7_lateral_drift_is_hard_gate(cfg) -> None:
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True


def test_smoke7_no_reference_imitation_terms(cfg) -> None:
    """Out of scope for smoke7: reference/path imitation stays OFF."""
    rw = cfg.reward_weights
    assert rw.ref_q_track == pytest.approx(0.0)
    assert rw.ref_contact_match == pytest.approx(0.0)
    assert rw.torso_pos_xy == pytest.approx(0.0)
    assert rw.lin_vel_z == pytest.approx(0.0)


def test_smoke6_isotropic_unchanged() -> None:
    """Backward-compat: smoke6 keeps dim=1, alpha_y unset (isotropic/legacy)."""
    if not _SMOKE6.exists():
        pytest.skip("smoke6 config not found")
    c6 = load_training_config(str(_SMOKE6))
    assert c6.reward_weights.cmd_velocity_track_dim == 1
    assert c6.reward_weights.cmd_forward_velocity_alpha_y == pytest.approx(0.0)
    assert c6.ppo.eval.post_training_strict_lateral_drift is False


def test_smoke7_reference_grid_includes_tiny_vy(cfg) -> None:
    """Finding-1 fix: the offline vy grid must include ±0.03 so the tiny-vy
    command maps to a MATCHING reference bin (nearest-key L2 would otherwise
    snap ±0.03 -> the 0.0 bin -> forward-only reference)."""
    grid = [round(float(v), 4) for v in cfg.env.loc_ref_offline_command_vy_grid]
    assert 0.03 in grid and -0.03 in grid, f"vy grid missing ±0.03: {grid}"
    # the sampled command edge (±0.03) must snap to itself, not 0.0
    nearest = min(grid, key=lambda g: abs(g - 0.03))
    assert nearest == pytest.approx(0.03), f"vy=0.03 snaps to {nearest}, not 0.03"


def test_smoke7_has_nonzero_vy_eval_probes(cfg) -> None:
    """Finding-2 fix: a nonzero-vy probe must exist so the deterministic eval
    measures 2D tracking (not just forward+drift)."""
    probes = [list(p) for p in cfg.env.eval_velocity_cmd_probes]
    assert probes, "smoke7 must configure nonzero-vy eval probes"
    lateral = [p for p in probes if abs(float(p[1])) >= 0.02]
    assert lateral, f"no probe with |vy|>=0.02 (Appendix C min): {probes}"
    # probe |vy| must be at/above the criterion floor and within the cmd range
    for p in lateral:
        assert abs(float(p[1])) == pytest.approx(0.03)
