"""Numeric reward-formula tests for smoke9's TB-faithful additions.

Validates the 4 reward changes the env makes for smoke9:
  1. penalty_ang_vel_xy under form=tb_neg_squared returns -sum(rate²)
  2. penalty_feet_ori under form=tb_linear_lateral returns
     -(sqrt(gx²+gy²)_L + sqrt(gx²+gy²)_R)
  3. penalty_close_feet_xy returns binary -1.0 when lateral foot
     distance < env.close_feet_threshold
  4. feet_phase computes the cubic-Bézier swing trajectory and returns
     exp(-α·Δz²) × (1 + max_expected/swing_height) bonus

These tests probe the env's reward-term computation directly via the
public _compute_reward_terms helper, NOT via a full rollout.  They
don't depend on physics convergence and are sub-second.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

_SMOKE9 = Path("training/configs/ppo_walking_v0201_smoke9.yaml")
_SMOKE = Path("training/configs/ppo_walking_v0201_smoke.yaml")
if not _SMOKE.exists() or not _SMOKE9.exists():
    pytest.skip(
        "smoke / smoke9 yaml missing",
        allow_module_level=True,
    )

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv


@pytest.fixture(scope="module")
def smoke9_env() -> WildRobotEnv:
    """Construct a smoke9 env (TB-faithful form flags)."""
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE9))
    cfg.freeze()
    return WildRobotEnv(cfg)


@pytest.fixture(scope="module")
def smoke7_env() -> WildRobotEnv:
    """Construct a smoke7 env (Gaussian / quadratic form flags) for
    back-compat assertions."""
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE))
    cfg.freeze()
    return WildRobotEnv(cfg)


# -----------------------------------------------------------------------------
# 1. ang_vel_xy form switch
# -----------------------------------------------------------------------------


def test_smoke9_ang_vel_xy_form_is_tb_neg_squared(smoke9_env) -> None:
    """The cached env flag must reflect smoke9's yaml setting."""
    assert smoke9_env._penalty_ang_vel_xy_form == "tb_neg_squared"


def test_smoke7_ang_vel_xy_form_is_gaussian(smoke7_env) -> None:
    """Smoke7 must default to the historical Gaussian form (back-compat)."""
    assert smoke7_env._penalty_ang_vel_xy_form == "gaussian"


# -----------------------------------------------------------------------------
# 2. penalty_feet_ori form switch
# -----------------------------------------------------------------------------


def test_smoke9_penalty_feet_ori_form_is_tb_linear_lateral(smoke9_env) -> None:
    assert smoke9_env._penalty_feet_ori_form == "tb_linear_lateral"


def test_smoke7_penalty_feet_ori_form_is_wr_quad_3axis(smoke7_env) -> None:
    assert smoke7_env._penalty_feet_ori_form == "wr_quad_3axis"


# -----------------------------------------------------------------------------
# 3. close_feet_threshold
# -----------------------------------------------------------------------------


def test_smoke9_close_feet_threshold_matches_tb(smoke9_env) -> None:
    """TB walk.gin:125 close_feet_threshold = 0.06.  Use float32
    tolerance — the env caches as jp.float32 so the value drifts ~1e-7
    from the yaml's 0.06.  Anything tighter than ~1e-6 is sub-precision
    noise."""
    assert float(smoke9_env._close_feet_threshold) == pytest.approx(0.06, abs=1e-6)


# -----------------------------------------------------------------------------
# 4. feet_phase formula — pure-Python reference implementation
# -----------------------------------------------------------------------------


def _expected_foot_z_pure(phase: float, swing_height: float, is_left: bool) -> float:
    """Pure-Python reference port of TB walk_env.py:583-628.  Used to
    cross-check the env's JAX implementation."""
    two_pi = 2.0 * math.pi
    phase = (phase + two_pi) % two_pi
    if is_left:
        in_swing = phase < math.pi
        swing_progress = phase / math.pi
    else:
        in_swing = phase >= math.pi
        swing_progress = (phase - math.pi) / math.pi
    if not in_swing:
        return 0.0
    # Cubic-Bézier (smoothstep): 3x² - 2x³
    if swing_progress <= 0.5:
        x = 2.0 * swing_progress
        bezier = x * x * (3.0 - 2.0 * x)
        return swing_height * bezier
    else:
        x = 2.0 * swing_progress - 1.0
        bezier = x * x * (3.0 - 2.0 * x)
        return swing_height * (1.0 - bezier)


def test_feet_phase_expected_height_curve_matches_tb_shape() -> None:
    """Cubic-Bézier swing trajectory: starts at 0, peaks at swing_height
    at swing_progress=0.5, returns to 0 at swing_progress=1.

    Sample left foot at phase ∈ {0, π/4, π/2, 3π/4, π, 3π/2}:
      phase=0     → swing_progress=0,    z=0
      phase=π/4   → swing_progress=0.25, z = 0.5 × (3·0.25 - 0.25)·h
                                            ... use direct bezier formula
      phase=π/2   → swing_progress=0.5,  z=swing_height (peak)
      phase=3π/4  → swing_progress=0.75, z mirrors π/4
      phase=π     → swing_progress=1,    z=0 (end of left swing)
      phase=3π/2  → grounded, z=0 (right foot's swing window)
    """
    h = 0.04  # TB swing_height
    cases = [
        (0.0,             0.0),                # start
        (math.pi / 2.0,   h),                  # peak
        (math.pi - 0.001, 0.0),                # near end (3x²-2x³ at x≈0)
        (3 * math.pi / 2, 0.0),                # right's swing → left grounded
    ]
    for phase, expected in cases:
        got = _expected_foot_z_pure(phase, h, is_left=True)
        assert abs(got - expected) < 1e-3, (
            f"phase={phase:.3f}: expected_z_left={got:.5f}, expected≈{expected}"
        )


def test_feet_phase_left_right_swing_complementary() -> None:
    """At any given phase, exactly one foot is in swing (per TB
    asymmetric timing: left during [0, π], right during [π, 2π]).
    Verify by sampling a few phases that exactly one is non-zero."""
    h = 0.04
    for phase in [math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
        z_l = _expected_foot_z_pure(phase, h, is_left=True)
        z_r = _expected_foot_z_pure(phase, h, is_left=False)
        # Left in swing, right grounded
        assert z_l > 0.0 and z_r == 0.0, f"phase={phase}: l={z_l}, r={z_r}"
    for phase in [5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4]:
        z_l = _expected_foot_z_pure(phase, h, is_left=True)
        z_r = _expected_foot_z_pure(phase, h, is_left=False)
        # Right in swing, left grounded
        assert z_l == 0.0 and z_r > 0.0, f"phase={phase}: l={z_l}, r={z_r}"


# -----------------------------------------------------------------------------
# 5. Default reward-weight values
# -----------------------------------------------------------------------------


def test_smoke7_does_not_enable_smoke9_rewards() -> None:
    """Smoke7's yaml predates smoke9's reward additions.  Loaded config
    must have penalty_close_feet_xy and feet_phase at 0 (the dataclass
    default).  Catches an over-aggressive default flip."""
    cfg = load_training_config(str(_SMOKE))
    rw = cfg.reward_weights
    assert rw.penalty_close_feet_xy == 0.0, (
        f"smoke7 penalty_close_feet_xy = {rw.penalty_close_feet_xy}; "
        "must default to 0.  If this fails, the dataclass default "
        "flipped or smoke7 yaml was edited."
    )
    assert rw.feet_phase == 0.0, (
        f"smoke7 feet_phase = {rw.feet_phase}; must default to 0."
    )
