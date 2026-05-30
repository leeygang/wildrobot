"""Pin the v0.21.0 smoke3 YAML contract.

``ppo_walking_v0210_smoke3.yaml`` is the dead-gradient fix for smoke2
(run 5v4j5j7w, 13.5M env-steps), which converged to a tap-in-place /
slight-backward-drift minimum: ``reward/cmd_forward_velocity_track``
stayed at 0.0001 the entire run because the α=562.5 tracker
``exp(-α·err²)`` has a live-gradient window of only ``cmd ± 0.042 m/s``
and smoke2 sampled the narrow high band ``vx ∈ [0.18, 0.26]`` — so from
a standing cold start (err ≈ 0.22 → exp(-35) ≈ 0) no gradient ever
pulled the policy toward walking.

Smoke3 raises the walk floor and re-grids the library to keep 0.26:

  env.min_velocity:                  0.18 -> 0.065  (walk floor)
  env.loc_ref_command_grid_interval: 0.04 -> 0.065  (keep 0.26 ref bin)

Floor 0.065 = 1.58 x the α=562.5 live-window (1/sqrt(562.5)=0.042) — the
size/α-normalized ToddlerBot deadzone/window ratio (TB: 0.05/0.032=1.58).
The positive-only walk branch then samples ``vx ∈ [0.065, 0.26]``, where
standing pays exp(-562.5*0.065^2)=0.093 (clearly suboptimal) — restoring
the reachable-target zone smoke12b (identical α=562.5) walked in from
iteration 0.  interval=0.065 makes arange land on both 0.065 and 0.26, so
the top-speed reference bin is retained (smoke3's first draft used
min_velocity=0 + interval=0.04, which silently dropped the 0.26 bin).
α and all G5 bounds are UNCHANGED (lowering α would reopen the smoke1
standing-leak; this is a floor/sampling fix, not a reward reform).

These tests pin the lever so a later edit can't silently revert it, and
assert the rest of the smoke2 basin-break recipe is carried over intact.

Reference: training/CHANGELOG.md v0.21.0-smoke3-plan + v0.21.0-smoke2-result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_SMOKE3 = Path("training/configs/ppo_walking_v0210_smoke3.yaml")
_SMOKE2 = Path("training/configs/ppo_walking_v0210_smoke2.yaml")


@pytest.fixture(scope="module")
def cfg_smoke3():
    if not _SMOKE3.exists():
        pytest.skip(f"{_SMOKE3.name} not found")
    return load_training_config(str(_SMOKE3))


@pytest.fixture(scope="module")
def cfg_smoke2():
    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    return load_training_config(str(_SMOKE2))


# ---------------------------------------------------------------------------
# Smoke3 lever: raise the walk floor to 0.065 (gradient sweet spot)
# ---------------------------------------------------------------------------
def test_smoke3_walk_floor_is_0065(cfg_smoke3) -> None:
    """The dead-gradient lever.  Floor 0.065 = 1.58 x the alpha=562.5
    live-window (1/sqrt(562.5)=0.042) — the size/alpha-normalized TB
    deadzone/window ratio.  At 0.065 standing pays exp(-562.5*0.065^2)=0.093
    (clearly suboptimal); at 0.02 it paid 0.80 (standing nearly satisfies),
    at >=0.10 the v=0 gradient collapses (recreates the smoke2 dead zone)."""
    assert cfg_smoke3.env.min_velocity == pytest.approx(0.065)


def test_smoke3_max_velocity_unchanged_target_speed_preserved(cfg_smoke3) -> None:
    """max_velocity stays 0.26 so the target speed is still trained;
    smoke3 raises the floor, it does not lower the ceiling."""
    assert cfg_smoke3.env.max_velocity == pytest.approx(0.26)


def test_smoke3_effective_walk_vx_floor(cfg_smoke3) -> None:
    """Realized sampler floor: positive-only walk branch uses
    ``x_min = max(min_velocity, cmd_deadzone[0]) = max(0.065, 0.02) = 0.065``.
    cmd_deadzone stays at smoke2's 0.02 (subsumed by min_velocity)."""
    env = cfg_smoke3.env
    assert env.cmd_sampler_walk_vx_positive_only is True
    assert env.cmd_deadzone[0] == pytest.approx(0.02)
    assert max(env.min_velocity, env.cmd_deadzone[0]) == pytest.approx(0.065)


def test_smoke3_interval_keeps_top_speed_bin(cfg_smoke3) -> None:
    """interval 0.04 -> 0.065 so arange(0.065, 0.26+eps, 0.065) lands
    EXACTLY on both the 0.065 floor and the 0.26 top-speed reference bin.
    smoke3's earlier interval=0.04 over [0,0.26] silently dropped 0.26
    (0.28 > 0.26) -> 0.26 cmds snapped to a 0.24 prior."""
    assert cfg_smoke3.env.loc_ref_command_grid_interval == pytest.approx(0.065)


# ---------------------------------------------------------------------------
# Unchanged from smoke2: α / G5 / reward block are NOT reformed
# ---------------------------------------------------------------------------
def test_smoke3_alpha_unchanged_from_smoke2(cfg_smoke3) -> None:
    """smoke3 is a floor/sampling fix.  Lowering α would reopen the smoke1
    standing-leak (at α=40 standing paid exp(-40·0.18²)=0.27)."""
    assert cfg_smoke3.reward_weights.cmd_forward_velocity_alpha == pytest.approx(562.5)


def test_smoke3_carries_smoke2_basin_break_recipe(cfg_smoke3) -> None:
    """The rest of the smoke2 recipe is inherited verbatim."""
    env = cfg_smoke3.env
    rw = cfg_smoke3.reward_weights
    assert rw.action_rate == pytest.approx(-1.0)
    assert env.cmd_zero_chance == pytest.approx(0.05)
    assert env.cmd_turn_chance == pytest.approx(0.05)
    assert env.max_episode_steps == 1000
    assert env.loc_ref_residual_base == "home"
    # prior-free reward block: imitation terms stay gated off
    assert rw.ref_q_track == pytest.approx(0.0)
    assert rw.ref_contact_match == pytest.approx(0.0)
    assert rw.feet_phase == pytest.approx(7.5)


def test_smoke3_diff_vs_smoke2_is_floor_and_interval(cfg_smoke3, cfg_smoke2) -> None:
    """Lock the blast radius: smoke3 must differ from smoke2 ONLY in
    min_velocity (the floor) and loc_ref_command_grid_interval (which keeps
    0.26 an exact library bin given the new floor).  All other env
    command-sampler and reward knobs identical."""
    e3, e2 = cfg_smoke3.env, cfg_smoke2.env
    assert e3.min_velocity != e2.min_velocity  # the floor change
    assert e3.loc_ref_command_grid_interval != e2.loc_ref_command_grid_interval
    assert e3.max_velocity == pytest.approx(e2.max_velocity)
    assert e3.min_velocity_y == pytest.approx(e2.min_velocity_y)
    assert e3.max_velocity_y == pytest.approx(e2.max_velocity_y)
    assert e3.max_yaw_rate == pytest.approx(e2.max_yaw_rate)
    assert e3.cmd_zero_chance == pytest.approx(e2.cmd_zero_chance)
    assert e3.cmd_turn_chance == pytest.approx(e2.cmd_turn_chance)
    assert tuple(e3.cmd_deadzone) == tuple(e2.cmd_deadzone)
    assert (
        cfg_smoke3.reward_weights.cmd_forward_velocity_alpha
        == pytest.approx(cfg_smoke2.reward_weights.cmd_forward_velocity_alpha)
    )


# ---------------------------------------------------------------------------
# Real blast radius: instantiate the env and pin the actual library bins
# (config-field asserts above do NOT catch the min_velocity -> grid coupling;
# this test would have caught the dropped-0.26 regression in the first draft)
# ---------------------------------------------------------------------------
def test_smoke3_env_init_pins_reference_vx_bins(cfg_smoke3) -> None:
    """min_velocity also anchors the reference vx grid (arange in
    _init_offline_service), so the floor change reshapes the library.
    Pin the actual ``_offline_cmd_keys`` so a future floor/interval edit
    can't silently drop the 0.26 top-speed bin or the 0.065 floor bin."""
    import numpy as np
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg_smoke3)
    keys = np.asarray(env._offline_cmd_keys)  # (n_bins, 3) = (vx, vy, wz)
    vx = sorted({round(float(v), 4) for v in keys[:, 0]})
    assert vx == [0.0, 0.065, 0.13, 0.195, 0.2, 0.26]
    assert len(keys) == 30  # 26 linear (5 nonzero-vx x 5 vy + static) + 4 pure-yaw

    def has(v):
        return any(np.allclose(k, v, atol=1e-6) for k in keys)

    assert has([0.065, 0.0, 0.0])   # walk floor bin present
    assert has([0.26, 0.0, 0.0])    # top-speed bin retained (the regression guard)
    assert has([0.20, 0.065, 0.0])  # primary eval cmd is exact-bin
