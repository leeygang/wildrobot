"""Pin smoke12's basin-break deltas (cmd_zero_chance, action_rate, feet_phase).

These three deltas are documented in
``training/configs/ppo_walking_v0201_smoke12.yaml`` (see header) and in
``training/docs/tmp_next_training_interventions.md``:

  (1) ``env.cmd_zero_chance: 0.2 -> 0.0``
      Every bootstrap episode now requires forward motion.  smoke9c/10/11
      gave PPO a perfect-tracking standstill mode on 20% of episodes;
      that was a contributor to the smoke11 standstill basin.

  (2) ``reward_weights.action_rate: -2.0 -> -1.0``
      Halve the action-rate smoothness penalty for bootstrap so early
      exploration can produce stride attempts without unpenalized
      action thrash.

  (3) ``_feet_phase_reward`` helper formula: raw -> max(0, raw - flat_foot_baseline)
      Global code change.  smoke11 measured ``reward/feet_phase ≈ 0.1088``
      on a near-flat-foot policy because TB's raw formula paid ~0.73 for
      keeping feet on the ground during walking phase.  smoke12 subtracts
      that baseline so flat-foot walking pays zero and only real swing
      tracking pays positive.  Standing branch is hard-zeroed.

The residual-widening (the fourth smoke12 delta) is pinned separately
in ``tests/test_smoke12_residual_scales.py``.  This file covers ONLY
the basin-break deltas + the runtime feet_phase numeric contract.
"""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE11_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke11.yaml"
_SMOKE12_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12.yaml"


def _load_cfg(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from training.configs.training_config import load_training_config
    return load_training_config(str(path))


def _maybe_build_env(cfg_path: Path):
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not found")
    try:
        from assets.robot_config import load_robot_config
        from training.configs.training_config import load_training_config
        from training.envs.wildrobot_env import WildRobotEnv
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"env deps unavailable: {exc}")
    load_robot_config(str(_REPO_ROOT / "assets/v2/mujoco_robot_config.json"))
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()
    return WildRobotEnv(cfg)


# -----------------------------------------------------------------------------
# 1. yaml: cmd_zero_chance + action_rate deltas
# -----------------------------------------------------------------------------


def test_smoke12_cmd_zero_chance_disabled() -> None:
    """smoke12 must pin cmd_zero_chance=0.0 (delta #1 of the basin-break)."""
    cfg = _load_cfg(_SMOKE12_CFG)
    assert cfg.env.cmd_zero_chance == 0.0, (
        f"smoke12 cmd_zero_chance = {cfg.env.cmd_zero_chance}; expected 0.0 "
        "so every bootstrap episode requires forward motion (no "
        "perfect-tracking standstill mode)."
    )


def test_smoke12_action_rate_halved_vs_smoke11() -> None:
    """smoke12 must halve action_rate from smoke11's -2.0 to -1.0 (delta #2).

    Rationale (also captured in the smoke12 yaml header): smoke11's
    -2.0 (matching TB walk.gin:119 in magnitude) combined with the
    flat-foot feet_phase exploit produced a "do less" attractor.  -1.0
    weakens that pressure during bootstrap; defer back to -2.0 once
    smoke12 shows the policy is exploring stride space.
    """
    s11 = _load_cfg(_SMOKE11_CFG)
    s12 = _load_cfg(_SMOKE12_CFG)
    assert s11.reward_weights.action_rate == -2.0, (
        f"smoke11 baseline action_rate changed to "
        f"{s11.reward_weights.action_rate}; smoke12 rationale assumes "
        "smoke11=-2.0."
    )
    assert s12.reward_weights.action_rate == -1.0, (
        f"smoke12 action_rate = {s12.reward_weights.action_rate}; "
        "expected -1.0 (halved smoke11's -2.0 for basin-break bootstrap)."
    )


# -----------------------------------------------------------------------------
# 2. _feet_phase_reward helper formula — smoke12 basin-break
# -----------------------------------------------------------------------------
# The detailed numeric tests live in tests/test_v0201_env_rewards_tb.py
# (under the "smoke12 basin-break formula" section).  We add two
# minimal sanity checks here so the smoke12 test suite is
# self-contained and a regression in the helper trips this file as
# well as the cross-cutting one.


def _call_helper(left_z, right_z, phase, is_standing, swing_h=0.05, alpha=914.304):
    """Call the static helper at smoke12 production params."""
    from training.envs.wildrobot_env import WildRobotEnv
    return float(WildRobotEnv._feet_phase_reward(
        left_foot_z_rel=jp.float32(left_z),
        right_foot_z_rel=jp.float32(right_z),
        phase_sin=jp.float32(math.sin(phase)),
        phase_cos=jp.float32(math.cos(phase)),
        is_standing=jp.bool_(is_standing),
        swing_height=jp.float32(swing_h),
        alpha=jp.float32(alpha),
    ))


def test_smoke12_feet_phase_flat_walking_returns_zero() -> None:
    """smoke12 contract: walking command + flat feet = 0 (no exploit)."""
    for phase in (0.0, math.pi / 4.0, math.pi / 2.0, math.pi, 3 * math.pi / 2.0):
        r = _call_helper(0.0, 0.0, phase, False)
        assert r == 0.0, (
            f"smoke12: flat-foot walking @ phase={phase:.3f} returned {r}; "
            "expected exactly 0 under the baseline-subtract formula."
        )


def test_smoke12_feet_phase_perfect_swing_pays_positive() -> None:
    """smoke12 contract: walking command + perfect swing tracking still
    pays meaningful positive reward (not just 0).  Exact value at
    smoke12 prod params (swing_h=0.05, α=914.304):
        raw = 1.0 (perfect) * 2.0 (bonus) = 2.0
        baseline = exp(-914.304 * 0.05²) * 2.0 = exp(-2.286) * 2 ≈ 0.2034
        reward  = max(0, 2.0 - 0.2034) ≈ 1.7966
    """
    r = _call_helper(0.05, 0.0, math.pi / 2.0, False, swing_h=0.05, alpha=914.304)
    raw = 2.0
    baseline = math.exp(-914.304 * 0.05 ** 2) * 2.0
    expected = max(0.0, raw - baseline)
    assert r == pytest.approx(expected, abs=1e-3), (
        f"perfect swing reward {r:.4f} differs from expected "
        f"{expected:.4f}; the smoke12 formula or production "
        "swing/alpha params drifted."
    )
    assert r > 1.5, f"perfect swing should give clear positive signal; got {r}"


def test_smoke12_feet_phase_standing_returns_zero() -> None:
    """smoke12 contract: standing (||cmd||≈0) always returns 0,
    regardless of foot positions.  This branch should be unreachable
    under smoke12's cmd_zero_chance=0 but the helper still hard-zeroes
    it for back-compat with on-policy zero-cmd episodes."""
    for lz, rz in ((0.0, 0.0), (0.05, 0.0), (0.034, 0.034)):
        r = _call_helper(lz, rz, math.pi / 2.0, True)
        assert r == 0.0, (
            f"standing + ({lz}, {rz}) returned {r}; expected 0 under "
            "smoke12 hard-zero-on-standing."
        )


# -----------------------------------------------------------------------------
# 3. End-to-end env: under the smoke12 yaml, iter-1 feet_phase contrib ≈ 0
# -----------------------------------------------------------------------------


def test_smoke12_env_iter1_feet_phase_near_zero() -> None:
    """End-to-end check that under the smoke12 yaml + the smoke12
    formula, the iter-1 ``reward/feet_phase`` contribution is near
    zero (the policy at home pose has flat feet, walking cmd → 0).

    Pre-smoke12 raw formula gave ~0.16 per step in the same state;
    measured ~5e-5 under the smoke12 formula.  Threshold 0.05 is
    well above the measured floor and well below the smoke11
    standstill basin's late-run ~0.109.
    """
    env = _maybe_build_env(_SMOKE12_CFG)
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY

    reset_fn = jax.jit(env.reset_for_eval)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    state = step_fn(state, zero_action)
    assert int(state.done) == 0

    idx = METRIC_INDEX["reward/feet_phase"]
    feet_phase = float(state.metrics[METRICS_VEC_KEY][idx])
    assert 0.0 <= feet_phase < 0.05, (
        f"smoke12 iter-1 reward/feet_phase = {feet_phase:.5f}; expected "
        "in [0.0, 0.05).  >= 0.05 would mean the smoke11 standstill basin "
        "(~0.109) has returned."
    )
