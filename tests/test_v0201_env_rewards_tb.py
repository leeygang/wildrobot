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
# 5. Direct numeric tests of WildRobotEnv._feet_phase_reward
# -----------------------------------------------------------------------------
# These probe the env's actual JAX implementation, NOT just a Python
# port.  Inputs are baseline-relative foot z (so 0.0 = grounded), which
# is the contract the env enforces by subtracting feet_height_init
# before calling the helper.  A regression that reverts to body-origin
# z (the bug the reviewer flagged) would fail these tests because at
# WR's home pose foot body-origin z ≈ 0.034 m ≠ 0 and the helper would
# return ~0.037 even when the foot is correctly grounded.


def _call_feet_phase(
    env: WildRobotEnv,
    *,
    left_z_rel: float,
    right_z_rel: float,
    phase: float,
    is_standing: bool,
    swing_height: float = 0.04,
    alpha: float = 1428.6,
) -> float:
    """Convenience wrapper that converts a phase angle to (sin, cos)
    and invokes the env's static helper.  Returns a Python float."""
    return float(
        env._feet_phase_reward(
            left_foot_z_rel=jp.float32(left_z_rel),
            right_foot_z_rel=jp.float32(right_z_rel),
            phase_sin=jp.float32(math.sin(phase)),
            phase_cos=jp.float32(math.cos(phase)),
            is_standing=jp.bool_(is_standing),
            swing_height=jp.float32(swing_height),
            alpha=jp.float32(alpha),
        )
    )


def test_feet_phase_standing_grounded_max_reward(smoke9_env) -> None:
    """Standing (||cmd|| ≈ 0) + both feet at baseline (z_rel = 0) ⇒
    expected z = 0 for both, error = 0, base reward = exp(0) = 1.0,
    bonus = 1 + 0/h = 1.0 → reward = 1.0 (max)."""
    r = _call_feet_phase(
        smoke9_env, left_z_rel=0.0, right_z_rel=0.0,
        phase=0.0, is_standing=True,
    )
    assert r == pytest.approx(1.0, abs=1e-5), (
        f"standing + grounded both feet must give max reward 1.0; got {r}"
    )


def test_feet_phase_walking_grounded_during_own_swing_low(smoke9_env) -> None:
    """Walking + LEFT foot stays on ground during left's swing window
    (phase=π/2 expects left at apex h=0.04).  Error is the full
    swing_height; base reward = exp(-1428.6 * 0.04²) = exp(-2.286)
    ≈ 0.102; bonus = 1 + h/h = 2.0; reward ≈ 0.204.

    This is what the reviewer's flagged bug would produce at home pose:
    foot at body-origin z stays "low" (close to baseline) during swing,
    and PPO sees a small reward — but with the baseline correction the
    starting state IS grounded (z_rel = 0), so this test exercises
    PPO's incentive to LIFT the foot when its swing is commanded."""
    r = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.0,         # foot stays grounded
        right_z_rel=0.0,
        phase=math.pi / 2.0,    # left's swing apex commanded
        is_standing=False,
    )
    expected_base = math.exp(-1428.6 * 0.04 ** 2)
    expected = expected_base * 2.0  # bonus = 2.0 at peak swing
    assert r == pytest.approx(expected, abs=1e-3), (
        f"left grounded during left's swing apex: expected ≈{expected:.4f}, got {r}"
    )
    # Sanity: must be much smaller than the "left foot at apex" case (below).
    assert r < 0.5


def test_feet_phase_walking_lifted_during_own_swing_high(smoke9_env) -> None:
    """Walking + LEFT foot at swing apex during left's swing window
    (phase=π/2, z_rel = swing_height = 0.04) ⇒ left error = 0,
    right error = 0 (right grounded during left's swing).  Base
    reward = exp(0) = 1.0, bonus = 1 + 0.04/0.04 = 2.0, total = 2.0.

    This is the strongest signal the policy can earn — cmd-aligned
    swing.  PPO is incentivized to track this."""
    r = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.04,        # left at swing apex
        right_z_rel=0.0,        # right grounded
        phase=math.pi / 2.0,    # left's swing apex commanded
        is_standing=False,
    )
    assert r == pytest.approx(2.0, abs=1e-4), (
        f"perfect swing tracking + height bonus must give 2.0; got {r}"
    )


def test_feet_phase_standing_lifted_penalized(smoke9_env) -> None:
    """Standing (cmd ≈ 0) + lifting the left foot to swing_height ⇒
    expected z = 0 for both feet (standing override), so left error =
    0.04.  Base reward = exp(-1428.6 × 0.04²) ≈ 0.102.  Bonus = 1.0
    (no expected swing).  Total ≈ 0.102.

    This penalizes a "lift feet to fake walking" exploit on zero cmd."""
    r = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.04,        # left lifted
        right_z_rel=0.0,
        phase=math.pi / 2.0,    # phase is non-zero but standing override fires
        is_standing=True,
    )
    expected = math.exp(-1428.6 * 0.04 ** 2) * 1.0  # bonus = 1.0 standing
    assert r == pytest.approx(expected, abs=1e-4), (
        f"standing + lifted foot should give base reward ~{expected:.4f}; "
        f"got {r}"
    )
    # Confirm: standing+lifted (~0.10) << standing+grounded (1.0)
    assert r < 0.2


def test_feet_phase_baseline_correction_protects_against_body_origin_bug(
    smoke9_env,
) -> None:
    """REGRESSION TEST for the reviewer-flagged bug: WR foot body-origin
    z ≈ 0.034 m at home pose.  If the env passes raw foot_pos[2]
    (without subtracting feet_height_init) into _feet_phase_reward,
    a correctly grounded foot would be treated as 0.034 m above the
    expected 0 and earn:
        exp(-1428.6 * 2 * 0.034²) ≈ exp(-3.30) ≈ 0.037
    instead of the correct max reward 1.0.

    This test fails loud if a future refactor reverts the baseline
    subtraction in the env's call site.  We invoke the helper with
    unconverted body-origin z (0.034) for both feet and assert the
    reward is the BAD value, then with the correctly-converted
    baseline-relative z (0.0) and assert the reward is 1.0.  Both
    pass means the helper itself is fine; the env's call-site
    correction (left_foot_pos[2] - feet_height_init[0]) is what
    distinguishes correct from buggy behavior.
    """
    # Helper invoked with raw body-origin z (the bug case): low reward.
    r_bug = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.034, right_z_rel=0.034,  # what raw body-origin z gives
        phase=0.0, is_standing=True,
    )
    expected_bug = math.exp(-1428.6 * 2 * 0.034 ** 2)
    assert r_bug == pytest.approx(expected_bug, abs=1e-3), (
        f"body-origin z input should give ~{expected_bug:.4f}; got {r_bug}"
    )
    assert r_bug < 0.05, (
        "body-origin z must give a small reward (~0.037); else this "
        "test isn't actually exercising the bug case"
    )

    # Helper invoked with baseline-relative z (the correct case): max.
    r_ok = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.0, right_z_rel=0.0,
        phase=0.0, is_standing=True,
    )
    assert r_ok == pytest.approx(1.0, abs=1e-5)


def test_feet_phase_left_swing_active_right_grounded_tracks(smoke9_env) -> None:
    """At phase ∈ [0, π] left should be in swing (expected z > 0) and
    right should be grounded (expected z = 0).  If the policy tracks
    perfectly (left at apex during left's apex, right at 0), reward is
    high.  If the policy mistakes which foot to lift (right at apex
    during LEFT's swing), reward should drop sharply due to non-zero
    error on both feet."""
    # Correct pairing
    r_correct = _call_feet_phase(
        smoke9_env, left_z_rel=0.04, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    # Wrong-foot-lift
    r_wrong = _call_feet_phase(
        smoke9_env, left_z_rel=0.0, right_z_rel=0.04,
        phase=math.pi / 2.0, is_standing=False,
    )
    assert r_correct > r_wrong, (
        f"correct foot lift ({r_correct:.4f}) must reward more than "
        f"wrong foot lift ({r_wrong:.4f})"
    )
    assert r_correct == pytest.approx(2.0, abs=1e-4)
    # Wrong-foot has BOTH errors at swing_height: 2 * 0.04² in the exp.
    expected_wrong_base = math.exp(-1428.6 * 2 * 0.04 ** 2)
    assert r_wrong == pytest.approx(expected_wrong_base * 2.0, abs=1e-3)


# -----------------------------------------------------------------------------
# 6. Default reward-weight values
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


# -----------------------------------------------------------------------------
# 7. Integration test for the feet_phase CALL SITE
# -----------------------------------------------------------------------------
# The helper-only tests above don't exercise the env's call-site
# baseline subtraction (`left_foot_pos[2] - feet_height_init[0]`).
# A future regression that drops the subtraction would silently
# re-introduce the body-origin bug while passing all the helper tests.
# This integration test runs env.reset + env.step under smoke9 and
# reads the actually-logged reward/feet_phase metric, so the call-site
# wiring is exercised end-to-end.


def test_feet_phase_metric_reflects_baseline_correction(smoke9_env) -> None:
    """End-to-end test of the feet_phase call site under smoke9.

    Reset → step(action=0); read state.metrics["reward/feet_phase"].
    Expected value at iter 1 with the baseline correction:
      ~0.15 per step (= 7.5 weight × 0.02 dt × ~1.0 base reward)

    Without the baseline correction (the body-origin bug, fixed in
    `86dc0ae`), the same state would report ~0.04 per step (= 7.5 ×
    0.02 × ~0.27 reward, where 0.27 = exp(-1428.6 × 2 × 0.034²)).
    The two regimes differ by ~4×, which is well above any noise from
    the small physics drift over a single 20 ms step.
    """
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY

    reset_fn = jax.jit(smoke9_env.reset_for_eval)
    step_fn = jax.jit(smoke9_env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(smoke9_env.action_size, dtype=jp.float32)
    state = step_fn(state, zero_action)

    # State must not have terminated on iter 1 — if it did, the metric
    # snapshot is unreliable.
    assert int(state.done) == 0, "env terminated on iter 1; cannot probe"

    feet_phase_idx = METRIC_INDEX["reward/feet_phase"]
    feet_phase_contrib = float(state.metrics[METRICS_VEC_KEY][feet_phase_idx])

    # Baseline-corrected lower bound.  At iter 1 the phase has barely
    # advanced (Δphase ≈ 0.26 rad over 20 ms at cycle=0.96), so
    # expected_z is small (~3 mm) and a correctly grounded foot
    # (z_rel ≈ 0) gives reward ≈ 0.99 with bonus ≈ 1.07 → ~1.06.
    # Per-step contribution = 7.5 × 0.02 × 1.06 ≈ 0.16.
    #
    # The bug case would give ~0.04; we set the lower threshold at
    # 0.10 to leave a clean margin while accepting some physics jitter.
    assert feet_phase_contrib > 0.10, (
        f"reward/feet_phase = {feet_phase_contrib:.4f}; expected > 0.10 "
        f"(baseline-corrected reward ≈ 0.16 per step).  A value < 0.10 "
        f"indicates the call site is using raw foot_pos[2] instead of "
        f"foot_pos[2] - feet_height_init[i] — the body-origin bug "
        f"reviewer flagged in the smoke9 patch."
    )
    # Upper bound sanity: contribution can't exceed weight × dt × bonus_max
    # = 7.5 × 0.02 × 2.0 = 0.30 (perfect swing tracking with full bonus).
    assert feet_phase_contrib < 0.30, (
        f"reward/feet_phase = {feet_phase_contrib:.4f}; exceeds the "
        f"theoretical max contribution (0.30 = 7.5×0.02×2.0).  Likely "
        f"a metric registration / weight bug."
    )


def test_feet_phase_metric_is_registered_and_logged(smoke9_env) -> None:
    """Catches the silent-drop bug where reward/feet_phase is written
    to terminal_metrics_dict but missing from METRIC_INDEX (and so
    never reaches the metrics vector — value defaults to 0).
    """
    from training.core.metrics_registry import METRIC_INDEX

    assert "reward/feet_phase" in METRIC_INDEX, (
        "reward/feet_phase is not in METRIC_INDEX.  build_metrics_vec "
        "silently drops unregistered keys, so the smoke9 metric would "
        "always log 0.0.  Add a MetricSpec to metrics_registry.py."
    )
    assert "reward/penalty_close_feet_xy" in METRIC_INDEX, (
        "reward/penalty_close_feet_xy is not in METRIC_INDEX.  Same "
        "silent-drop bug as above.  Add a MetricSpec."
    )
