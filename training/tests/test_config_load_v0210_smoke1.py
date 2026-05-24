"""Pin the v0.21.0 lateral+yaw smoke1 YAML contract.

``ppo_walking_v0210_smoke1_lateral_yaw.yaml`` is the FIRST v0.21.0 config
that opts into all four lateral+yaw gates simultaneously:

  1. ``env.actor_obs_layout_id == "wr_obs_v8_cmd3d"``     (P7 — 3-axis cmd obs)
  2. ``env.cmd_sampler_3d_branched is True``              (P4 — TB-branched sampler)
  3. ``env.loc_ref_command_axes_3d is True``              (P5 — 3D library lookup)
  4. ``reward_weights.cmd_yaw_rate_track > 0``            (P6 — yaw-rate term on)

Plus the WR-scaled command range (vy ±0.13 m/s, wz ±0.25 rad/s), the
bumped lateral residual scales (H6: hip_roll + ankle_roll × 1.5; WR has
NO hip_yaw joints), and the explicit ``eval_velocity_cmd`` 3-tuple at
(0.18, 0.04, 0.10) so eval probes a mixed (vx, vy, wz) point instead of
defaulting to forward-only via the H3 scalar broadcast rule.

These tests pin the YAML contract so a later edit can't silently turn
off one of the four gates or drift the vy/wz range and break the run
attribution.

Reference: training/docs/v0210_lateral_yaw_prior_plan-final-r2.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_SMOKE1 = Path("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
_SMOKE14 = Path("training/configs/ppo_walking_v0201_smoke14.yaml")


@pytest.fixture(scope="module")
def cfg_smoke1():
    if not _SMOKE1.exists():
        pytest.skip(f"{_SMOKE1.name} not found")
    return load_training_config(str(_SMOKE1))


@pytest.fixture(scope="module")
def cfg_smoke14():
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")
    return load_training_config(str(_SMOKE14))


# ---------------------------------------------------------------------------
# Core contract: all four v0.21 gates wired + WR-scaled lateral/yaw ranges
# ---------------------------------------------------------------------------
def test_smoke1_yaml_loads_and_has_3d_axes_enabled(cfg_smoke1) -> None:
    env = cfg_smoke1.env
    rw = cfg_smoke1.reward_weights
    # Gate 1: 3-axis cmd actor obs layout (P7).
    assert env.actor_obs_layout_id == "wr_obs_v8_cmd3d"
    # Gate 2: TB-branched 3D command sampler (P4).
    assert env.cmd_sampler_3d_branched is True
    # Gate 3: 3D reference library lookup (P5).
    assert env.loc_ref_command_axes_3d is True
    # Gate 4: yaw-rate reward term on (P6).
    assert rw.cmd_yaw_rate_track == pytest.approx(1.5)
    assert rw.cmd_yaw_rate_alpha == pytest.approx(0.25)
    # 2D velocity tracking (vx + vy against (cmd_vx, cmd_vy)).
    assert rw.cmd_velocity_track_dim == 2
    # WR-scaled lateral / yaw ranges.
    assert env.min_velocity_y == pytest.approx(-0.13)
    assert env.max_velocity_y == pytest.approx(0.13)
    assert env.max_yaw_rate == pytest.approx(0.25)
    # Branched sampler probabilities.
    assert env.cmd_zero_chance == pytest.approx(0.20)
    assert env.cmd_turn_chance == pytest.approx(0.20)
    # Per-axis deadzone (was a scalar before v0.21).
    assert tuple(env.cmd_deadzone) == pytest.approx((0.02, 0.02, 0.05))
    # H3: explicit 3-tuple eval cmd (avoid scalar→(s, 0, 0) broadcast).
    assert tuple(env.eval_velocity_cmd) == pytest.approx((0.18, 0.04, 0.10))


# ---------------------------------------------------------------------------
# vy / wz grids populated with the documented 5-bin values
# ---------------------------------------------------------------------------
def test_smoke1_yaml_vy_and_wz_grids_present(cfg_smoke1) -> None:
    env = cfg_smoke1.env
    vy = tuple(env.loc_ref_offline_command_vy_grid)
    wz = tuple(env.loc_ref_offline_command_yaw_rate_grid)
    assert vy == pytest.approx((-0.10, -0.05, 0.0, 0.05, 0.10))
    assert wz == pytest.approx((-0.20, -0.10, 0.0, 0.10, 0.20))


# ---------------------------------------------------------------------------
# H2: the 3D library factory force-unions vx=0.0 (control/zmp/zmp_walk.py:1685),
# so the static (0, 0, 0) bin is guaranteed without an explicit
# loc_ref_offline_command_vx_grid override.  The vx grid itself is derived
# from {min_velocity, max_velocity, loc_ref_command_grid_interval}; smoke1
# uses interval=0.04 over [0.18, 0.26] → {0.18, 0.22, 0.26}, then the
# factory unions 0.0 → effective stored vx set {0.0, 0.18, 0.22, 0.26}.
# ---------------------------------------------------------------------------
def test_smoke1_vx_grid_inputs_yield_static_bin_via_factory(cfg_smoke1) -> None:
    env = cfg_smoke1.env
    # The vx grid is derived; pin the inputs that drive it.
    assert env.min_velocity == pytest.approx(0.18)
    assert env.max_velocity == pytest.approx(0.26)
    assert env.loc_ref_command_grid_interval == pytest.approx(0.04)
    # The library factory unconditionally appends vx=0.0 for the pure-yaw
    # bins (see ``build_library_for_3d_values`` round-2 follow-up), so the
    # static (0, 0, 0) bin is guaranteed without needing a YAML override.


# ---------------------------------------------------------------------------
# H6: hip_roll + ankle_roll bumped 1.5×; WR has no *_hip_yaw joints so
# those keys must NOT appear in the residual scale dict.
# ---------------------------------------------------------------------------
def test_smoke1_has_bumped_lateral_residual_scales_no_hip_yaw(
    cfg_smoke1, cfg_smoke14
) -> None:
    scales = cfg_smoke1.env.loc_ref_residual_scale_per_joint
    base = cfg_smoke14.env.loc_ref_residual_scale_per_joint

    for j in (
        "left_hip_roll",
        "right_hip_roll",
        "left_ankle_roll",
        "right_ankle_roll",
    ):
        assert j in scales, f"smoke1 missing lateral DoF {j!r}"
        assert j in base, f"smoke14 missing lateral DoF {j!r}"
        assert scales[j] == pytest.approx(base[j] * 1.5), (
            f"smoke1.{j}={scales[j]} should be 1.5× smoke14 ({base[j]})"
        )

    # Non-lateral joints unchanged from smoke14.
    for j in (
        "left_hip_pitch",
        "left_knee_pitch",
        "left_ankle_pitch",
        "right_hip_pitch",
        "right_knee_pitch",
        "right_ankle_pitch",
    ):
        assert j in scales, f"smoke1 missing pitch DoF {j!r}"
        assert scales[j] == pytest.approx(base[j]), (
            f"smoke1.{j}={scales[j]} drifted from smoke14 ({base[j]}); "
            "only hip_roll + ankle_roll should change for v0.21 lateral bump"
        )

    # H6 guard: WR has no hip_yaw joints (verified at assets/v2/wildrobot.xml).
    assert "left_hip_yaw" not in scales
    assert "right_hip_yaw" not in scales


# ---------------------------------------------------------------------------
# End-to-end env construction: prove all four opt-in gates wire together
# without runtime error.  Builds the 3D library (25 bins: 4 vx × 5 vy
# linear + 5 pure-yaw, no dedup because env vx grid excludes 0.0 — see
# smoke1 YAML header block for the full math) so it's not free, but it's
# the only way to catch a wiring regression that would only manifest at
# training start.
# ---------------------------------------------------------------------------
def test_smoke1_env_init_succeeds(cfg_smoke1) -> None:
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg_smoke1)
    # Pin the actual library size: 4 vx (arange 0.18/0.22/0.26 ∪ 0.20)
    # × 5 vy linear-walk bins + 5 pure-yaw bins = 25 stored trajectories.
    # No dedup happens because the env's linear vx grid excludes 0.0
    # (the meta.command_range_vx 0.0 force-union from P2.5c is
    # metadata-only and does NOT inject a (0, 0, 0) linear bin).
    cmd_keys = list(env._offline_cmd_keys)
    assert len(cmd_keys) == 25, (
        f"smoke1 library bin count regressed: expected 25 trajectories "
        f"(20 linear + 5 pure-yaw, no dedup), got {len(cmd_keys)}.  "
        "Either env._init_offline_service vx-union logic changed or "
        "build_library_for_3d_values dedup semantics drifted."
    )
    has_static = any(
        abs(k[0]) < 1e-9 and abs(k[1]) < 1e-9 and abs(k[2]) < 1e-9
        for k in cmd_keys
    )
    assert has_static, (
        "Expected the 3D library factory to force-union the static "
        "(0, 0, 0) bin (control/zmp/zmp_walk.py:1685); got cmd_keys="
        f"{cmd_keys}"
    )
    has_lateral = any(abs(k[1]) > 1e-6 for k in cmd_keys)
    has_yaw = any(abs(k[2]) > 1e-6 for k in cmd_keys)
    assert has_lateral, f"Expected at least one vy != 0 bin; got {cmd_keys}"
    assert has_yaw, f"Expected at least one wz != 0 bin; got {cmd_keys}"
