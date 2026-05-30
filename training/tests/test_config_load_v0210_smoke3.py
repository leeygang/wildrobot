"""Pin the v0.21.0 smoke3 YAML contract.

``ppo_walking_v0210_smoke3.yaml`` is the dead-gradient fix for smoke2
(run 5v4j5j7w, 13.5M env-steps), which converged to a tap-in-place /
slight-backward-drift minimum: ``reward/cmd_forward_velocity_track``
stayed at 0.0001 the entire run because the α=562.5 tracker
``exp(-α·err²)`` has a live-gradient window of only ``cmd ± 0.042 m/s``
and smoke2 sampled the narrow high band ``vx ∈ [0.18, 0.26]`` — so from
a standing cold start (err ≈ 0.22 → exp(-35) ≈ 0) no gradient ever
pulled the policy toward walking.

Smoke3 makes ONE functional change:

  env.min_velocity: 0.18 -> 0.0   (max_velocity stays 0.26)

With the positive-only walk branch (``x_min = max(min_velocity,
cmd_deadzone[0])``) this samples ``vx ∈ [0.02, 0.26]`` — restoring the
reachable-target zone that smoke12b (identical α=562.5) walked in from
iteration 0.  α and all G5 bounds are UNCHANGED (lowering α would reopen
the smoke1 standing-leak; this is a sampling fix, not a reward reform).

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
# Smoke3 lever: widen vx sampling down to the deadzone
# ---------------------------------------------------------------------------
def test_smoke3_min_velocity_widened_to_zero(cfg_smoke3) -> None:
    """The single functional change vs smoke2.  With the positive-only
    walk branch this makes vx sample from [cmd_deadzone[0], max_velocity]
    = [0.02, 0.26], spanning the reachable-target zone the α=562.5
    tracker needs for a live gradient."""
    assert cfg_smoke3.env.min_velocity == pytest.approx(0.0)


def test_smoke3_max_velocity_unchanged_target_speed_preserved(cfg_smoke3) -> None:
    """max_velocity stays 0.26 so the target speed is still trained;
    smoke3 widens the floor, it does not lower the ceiling."""
    assert cfg_smoke3.env.max_velocity == pytest.approx(0.26)


def test_smoke3_effective_walk_vx_floor_is_deadzone(cfg_smoke3) -> None:
    """Guard the realized lower bound: positive-only walk branch uses
    ``x_min = max(min_velocity, cmd_deadzone[0])``.  With min_velocity=0
    the floor is the vx deadzone (0.02), not 0 — keep them consistent so
    the realized range is [0.02, 0.26]."""
    env = cfg_smoke3.env
    assert env.cmd_sampler_walk_vx_positive_only is True
    assert env.cmd_deadzone[0] == pytest.approx(0.02)
    realized_floor = max(env.min_velocity, env.cmd_deadzone[0])
    assert realized_floor == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Unchanged from smoke2: α / G5 / reward block are NOT reformed
# ---------------------------------------------------------------------------
def test_smoke3_alpha_unchanged_from_smoke2(cfg_smoke3) -> None:
    """smoke3 is a SAMPLING fix.  Lowering α would reopen the smoke1
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


def test_smoke3_diff_vs_smoke2_is_only_min_velocity(cfg_smoke3, cfg_smoke2) -> None:
    """Lock the blast radius: smoke3 must differ from smoke2 ONLY in
    min_velocity (the dead-gradient lever).  All other env command-sampler
    and reward knobs identical."""
    e3, e2 = cfg_smoke3.env, cfg_smoke2.env
    assert e3.min_velocity != e2.min_velocity  # the change
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
