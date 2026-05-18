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

# smoke9.yaml was removed after smoke9c landed; use smoke9c as the
# TB-faithful-form-flags fixture.  The form flags (tb_neg_squared /
# tb_linear_lateral), the feet_phase formula structure, and the
# _feet_phase_reward static helper are identical between smoke9 and
# smoke9c.  smoke9c additionally rescales close_feet_threshold from
# the TB literal 0.06 to WR-normalized 0.146 (smoke9c spatial pass);
# the threshold test below reflects that.
_SMOKE9C = Path("training/configs/ppo_walking_v0201_smoke9c.yaml")
_SMOKE = Path("training/configs/ppo_walking_v0201_smoke.yaml")
if not _SMOKE.exists() or not _SMOKE9C.exists():
    pytest.skip(
        "smoke / smoke9c yaml missing",
        allow_module_level=True,
    )

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv


@pytest.fixture(scope="module")
def smoke9_env() -> WildRobotEnv:
    """Construct a smoke9c env (TB-faithful form flags).  Fixture name
    kept as ``smoke9_env`` for git-blame continuity with the original
    smoke9 tests; the underlying yaml is smoke9c."""
    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(_SMOKE9C))
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


def test_smoke9_close_feet_threshold_matches_wr_normalized(smoke9_env) -> None:
    """TB walk.gin:125 sets the TB literal close_feet_threshold = 0.06
    at TB stance width 0.074 m.  smoke9c's spatial normalization pass
    preserves the fraction-of-stance on WR's 0.18056 m stance:
        threshold = 0.06 / 0.074 * 0.18056 ≈ 0.146
    Use float32 tolerance — the env caches as jp.float32.
    """
    assert float(smoke9_env._close_feet_threshold) == pytest.approx(0.146, abs=5e-4)


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
    zero_on_standing: bool = False,
) -> float:
    """Convenience wrapper that converts a phase angle to (sin, cos)
    and invokes the env's static helper.  Returns a Python float.

    ``zero_on_standing`` defaults to False (the helper's own default —
    pre-smoke12 standing payout).  Pass ``True`` to exercise the
    smoke12 bootstrap standing-zero branch.
    """
    return float(
        env._feet_phase_reward(
            left_foot_z_rel=jp.float32(left_z_rel),
            right_foot_z_rel=jp.float32(right_z_rel),
            phase_sin=jp.float32(math.sin(phase)),
            phase_cos=jp.float32(math.cos(phase)),
            is_standing=jp.bool_(is_standing),
            swing_height=jp.float32(swing_height),
            alpha=jp.float32(alpha),
            zero_on_standing=zero_on_standing,
        )
    )


# -----------------------------------------------------------------------------
# 5b. smoke12 basin-break formula (2026-05-17)
# -----------------------------------------------------------------------------
# The helper changed from the raw TB-style formula to:
#
#     walking_reward = max(0, raw - flat_foot_baseline)
#     standing       = 0
#
# Rationale recorded in WildRobotEnv._feet_phase_reward docstring and in
# training/configs/ppo_walking_v0201_smoke12.yaml header.  The tests
# below pin the NEW values (the old numerics are recorded in each test
# docstring as "was" for git-archaeology).


def test_feet_phase_standing_branch_is_configurable(smoke9_env) -> None:
    """Standing-branch behavior is selected by ``zero_on_standing``
    (Python bool, ``env.loc_ref_feet_phase_zero_on_standing`` in
    config).  Two branches:

      - ``zero_on_standing=False`` (default; pre-smoke12 behavior):
        standing returns ``exp(-α·(lz²+rz²)) × 1.0`` — pays max when
        both feet are at baseline, falls off as feet lift.  Pays
        ~1.0 at grounded, ~0.102 at lift=swing_h, ~0.037 at the
        body-origin-bug input.

      - ``zero_on_standing=True`` (smoke12 bootstrap):  standing
        returns 0 unconditionally.  Used when ``cmd_zero_chance: 0.0``
        makes the standing branch unreachable on the training
        distribution and we want no residual feet_phase payout on any
        ||cmd||≈0 frame.

    Walking branch (||cmd|| > 0) is independent of this flag — it is
    always the smoke12 baseline-subtract form (see the other tests
    in this section for that contract).
    """
    cases = (
        (0.0,   0.0,   1.0,                                "grounded"),
        (0.04,  0.0,   math.exp(-1428.6 * 0.04 ** 2),      "left-lifted"),
        (0.04,  0.04,  math.exp(-1428.6 * 2 * 0.04 ** 2),  "both-lifted"),
        (0.034, 0.034, math.exp(-1428.6 * 2 * 0.034 ** 2), "body-origin-z (pre-correction bug input)"),
    )
    for left_z, right_z, expected_default, label in cases:
        # Default (zero_on_standing=False): pre-smoke12 standing reward.
        r_default = _call_feet_phase(
            smoke9_env, left_z_rel=left_z, right_z_rel=right_z,
            phase=math.pi / 2.0, is_standing=True,
            zero_on_standing=False,
        )
        assert r_default == pytest.approx(expected_default, abs=1e-3), (
            f"standing + {label} with zero_on_standing=False: "
            f"expected ≈{expected_default:.4f}, got {r_default:.4f}"
        )
        # smoke12 (zero_on_standing=True): hard zero regardless of feet.
        r_smoke12 = _call_feet_phase(
            smoke9_env, left_z_rel=left_z, right_z_rel=right_z,
            phase=math.pi / 2.0, is_standing=True,
            zero_on_standing=True,
        )
        assert r_smoke12 == 0.0, (
            f"standing + {label} with zero_on_standing=True: "
            f"expected exactly 0.0 (smoke12 bootstrap); got {r_smoke12}"
        )


def test_feet_phase_walking_flat_feet_returns_zero(smoke9_env) -> None:
    """The whole point of the smoke12 baseline subtraction: walking
    command + both feet flat must give exactly 0, no matter what the
    phase says.  Catches a regression that re-introduces the
    flat-foot walking payout (which produced the smoke11 standstill
    basin at ``reward/feet_phase ≈ 0.109``).

    Was (pre-smoke12 raw formula):
      walking + flat @ phase=π/2 = exp(-1428.6·0.04²)·2.0 ≈ 0.203
    """
    for phase in (0.0, math.pi / 4.0, math.pi / 2.0, math.pi, 3 * math.pi / 2.0):
        r = _call_feet_phase(
            smoke9_env, left_z_rel=0.0, right_z_rel=0.0,
            phase=phase, is_standing=False,
        )
        assert r == 0.0, (
            f"walking + flat feet @ phase={phase:.3f}: expected 0.0 under "
            f"smoke12 baseline-subtract formula; got {r}"
        )


def test_feet_phase_walking_perfect_swing_pays_after_baseline(smoke9_env) -> None:
    """Walking + LEFT foot at swing apex during left's swing window
    still pays positive reward after the baseline subtraction.
    Exact value: raw=2.0, baseline=exp(-1428.6·0.04²)·2.0≈0.2034,
    reward=max(0, 2.0 − 0.2034)=1.7966.

    Was (pre-smoke12 raw formula): 2.0
    """
    r = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.04, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    raw = 2.0  # base=1.0, bonus=2.0
    baseline = math.exp(-1428.6 * 0.04 ** 2) * 2.0
    expected = max(0.0, raw - baseline)
    assert r == pytest.approx(expected, abs=1e-3), (
        f"perfect swing - flat baseline: expected ≈{expected:.4f}, got {r}"
    )
    # Still the strongest signal the policy can earn (clear lead over flat).
    assert r > 1.5


def test_feet_phase_walking_partial_swing_pays_partial(smoke9_env) -> None:
    """A 75%-of-target lift (3 cm vs 4 cm swing target) earns
    something between 0 and the perfect-swing value — confirms the
    gradient is smooth toward higher lift.  This is the "smoke12
    encourages more lift" property; if a regression breaks it, the
    policy loses signal to increase swing height during training.
    """
    r_full = _call_feet_phase(
        smoke9_env, left_z_rel=0.04, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    r_partial = _call_feet_phase(
        smoke9_env, left_z_rel=0.03, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    r_flat = _call_feet_phase(
        smoke9_env, left_z_rel=0.0, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    assert r_flat == 0.0
    assert r_full > r_partial > 0.0, (
        f"smoke12 must reward partial swing between flat (0) and full "
        f"({r_full:.4f}); got partial={r_partial:.4f}"
    )


def test_feet_phase_walking_wrong_foot_lift_clamps_to_zero(smoke9_env) -> None:
    """Walking + lifting the WRONG foot (right at apex during LEFT's
    swing window) ⇒ raw < baseline ⇒ ``max(0, ...)`` clamps to 0.
    Confirms that under smoke12, a policy that lifts both feet (or the
    wrong foot) at random earns nothing — only correct phase-aligned
    lifts pay.

    Was (pre-smoke12 raw formula): exp(-1428.6·2·0.04²)·2.0 ≈ 0.021
    """
    r_correct = _call_feet_phase(
        smoke9_env, left_z_rel=0.04, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    r_wrong = _call_feet_phase(
        smoke9_env, left_z_rel=0.0, right_z_rel=0.04,
        phase=math.pi / 2.0, is_standing=False,
    )
    assert r_wrong == 0.0, (
        f"wrong-foot lift during left's swing should clamp to 0 under "
        f"smoke12 baseline-subtract; got {r_wrong}"
    )
    assert r_correct > 1.5  # correct lift still pays (see test above)


def test_feet_phase_body_origin_bug_caller_still_needed(smoke9_env) -> None:
    """Regression: the env's call-site MUST still subtract
    ``feet_height_init`` before passing foot_pos[2] into the helper.
    The body-origin z ≈ 0.034 m at home would otherwise be treated as
    a small lift in the WALKING branch — and after smoke12's baseline
    subtraction it doesn't clamp to zero (the bug input is "lifted"
    relative to baseline, so raw > flat_baseline in some phases).
    The standing branch is hard-zeroed by smoke12 so it no longer
    distinguishes correct from buggy at standing; we exercise the
    walking case here instead.

    With body-origin z=0.034 fed into both feet at phase=π/2 (left's
    swing apex h=0.04), the helper sees left_err=|0.034 - 0.04|=0.006,
    right_err=0.034, so the apparent lift looks plausible — exactly the
    buggy outcome the caller-side subtraction prevents.  We pin: raw
    body-origin input must produce a numerically different reward
    from the correctly-baseline-subtracted input (z_rel=0).
    """
    r_bug = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.034, right_z_rel=0.034,
        phase=math.pi / 2.0, is_standing=False,
    )
    r_correct = _call_feet_phase(
        smoke9_env,
        left_z_rel=0.0, right_z_rel=0.0,
        phase=math.pi / 2.0, is_standing=False,
    )
    # smoke12 contract: feet truly flat (z_rel=0,0) under walking gives 0.
    assert r_correct == 0.0
    # Body-origin input gives a NONZERO value because the helper sees
    # "lifted" feet that aren't really lifted.  This is the bug surface
    # the caller-side feet_height_init subtraction blocks.
    assert r_bug != r_correct, (
        "body-origin input (0.034, 0.034) and baseline-corrected input "
        "(0.0, 0.0) must differ — otherwise the helper-side baseline "
        "subtraction in smoke12 has accidentally masked the env-side "
        "feet_height_init bug surface."
    )


# -----------------------------------------------------------------------------
# 6. Default reward-weight values
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 8. penalty_feet_ori baseline correction (smoke9b round-2 audit)
# -----------------------------------------------------------------------------
# WR's MJCF foot body has a 90° rotation built in (onshape mate convention),
# so foot body z-axis is forward-pointing, not sole-up.  Without baseline
# correction, TB's `sqrt(gx² + gy²)` formula returns 1.0 per foot at home
# pose (constant ~-0.20/step penalty regardless of actual tilt).  Fix:
# subtract the home-pose g-in-foot baseline before computing tilt magnitude.


def test_foot_ori_baseline_cached_at_init(smoke9_env) -> None:
    """The two baseline arrays must exist after env init.  Each is a
    3-vector representing gravity in the foot body frame at home pose."""
    assert hasattr(smoke9_env, "_foot_ori_baseline_left")
    assert hasattr(smoke9_env, "_foot_ori_baseline_right")
    L = np.asarray(smoke9_env._foot_ori_baseline_left)
    R = np.asarray(smoke9_env._foot_ori_baseline_right)
    assert L.shape == (3,)
    assert R.shape == (3,)
    # Each baseline should be a unit vector (gravity has unit magnitude).
    assert abs(np.linalg.norm(L) - 1.0) < 1e-3
    assert abs(np.linalg.norm(R) - 1.0) < 1e-3


def test_foot_ori_baseline_matches_wr_mjcf_convention(smoke9_env) -> None:
    """WR foot body has 90° rotation: at home pose, gravity expressed
    in foot frame is along the BODY X-axis (sole-up direction in the
    body frame), not the body Z-axis.  Verify the cached baselines
    reflect this:
      LEFT  baseline ≈ [+1, 0, 0]  (foot body x = sole-up)
      RIGHT baseline ≈ [-1, 0, 0]  (mirrored)

    If a future MJCF re-export changes the foot body convention so
    body z-axis becomes sole-up (TB-style), these baselines would shift
    to ≈ [0, 0, -1] and TB's original `sqrt(gx² + gy²)` formula would
    work directly — but the baseline-subtracted form would still give
    0 at home, so this test would also need updating.
    """
    L = np.asarray(smoke9_env._foot_ori_baseline_left)
    R = np.asarray(smoke9_env._foot_ori_baseline_right)
    # Allow some tolerance for the MJCF's near-but-not-exactly-90° quat.
    # The probe showed L ≈ [+1.0, 0, 0.002], R ≈ [-1.0, 0, 0].
    assert abs(L[0] - 1.0) < 0.01, f"LEFT baseline x={L[0]}, expected ~+1.0"
    assert abs(R[0] + 1.0) < 0.01, f"RIGHT baseline x={R[0]}, expected ~-1.0"


def test_penalty_feet_ori_zero_at_home_after_baseline_fix(smoke9_env) -> None:
    """End-to-end: at iter 1 with zero policy_action (robot near home),
    `reward/penalty_feet_ori` magnitude should be SMALL (close to 0),
    not ~-0.20 like the smoke9b run before the baseline fix.

    Pre-fix: -0.20/step (constant 90° MJCF baseline tilt)
    Post-fix: ~0 at home; grows as PPO actually tilts the foot.

    Pass threshold: |contribution| < 0.05 per step at iter 1 (the
    feet may have drifted slightly during the 20-substep physics roll;
    full zero requires identity action AND zero foot motion).
    """
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY

    reset_fn = jax.jit(smoke9_env.reset_for_eval)
    step_fn = jax.jit(smoke9_env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(smoke9_env.action_size, dtype=jp.float32)
    state = step_fn(state, zero_action)
    if int(state.done) > 0:
        pytest.skip("env terminated on iter 1; cannot probe")

    feet_ori_idx = METRIC_INDEX["reward/penalty_feet_ori"]
    contrib = float(state.metrics[METRICS_VEC_KEY][feet_ori_idx])

    assert abs(contrib) < 0.05, (
        f"reward/penalty_feet_ori = {contrib:+.4f} at iter 1; expected "
        f"|contrib| < 0.05 after baseline correction.  Pre-fix value "
        f"was ~-0.20 (constant 90° MJCF rotation overwhelming actual "
        f"foot tilt signal).  If this fails: either the baseline "
        f"subtraction in _compute_reward_terms regressed, or the env "
        f"code at the call site was refactored away."
    )


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


def test_feet_phase_metric_reflects_smoke12_basin_break(smoke9_env) -> None:
    """End-to-end test of the feet_phase call site under the smoke12
    baseline-subtract formula.

    smoke12 contract (replaces the pre-smoke12 raw-formula contract):
    walking command + nearly-flat feet should give a contribution
    ≈ 0 (the whole point of the basin-break is to kill the smoke11
    flat-foot exploit at ``reward/feet_phase ≈ 0.109``).

    Reset → step(action=0); read state.metrics["reward/feet_phase"].
    At iter 1 the policy is at home pose with both feet planted, so
    the helper sees walking + flat → 0 → contribution ≈ 7.5 × 0.02 ×
    0 = 0.  Measured empirically at ≈ 5e-5 (essentially zero;
    leftover from microscopic physics drift in 20 ms).

    Pre-smoke12 raw formula reported ~0.16 per step for the same
    state, because flat-foot walking paid ~1.0 raw reward.  If that
    threshold returns this test fails loud — the basin-break was
    reverted globally.

    Note on the body-origin bug regression coverage: the helper-side
    baseline subtraction in smoke12 changes both regimes (correct
    callsite AND body-origin-bug callsite) to ~0 in this state, so
    this test no longer separately catches a buggy callsite.  See
    ``test_feet_phase_body_origin_bug_caller_still_needed`` above for
    the surviving regression coverage of the caller-side bug surface.
    """
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY

    reset_fn = jax.jit(smoke9_env.reset_for_eval)
    step_fn = jax.jit(smoke9_env.step)
    state = reset_fn(jax.random.PRNGKey(0))
    zero_action = jp.zeros(smoke9_env.action_size, dtype=jp.float32)
    state = step_fn(state, zero_action)

    assert int(state.done) == 0, "env terminated on iter 1; cannot probe"

    feet_phase_idx = METRIC_INDEX["reward/feet_phase"]
    feet_phase_contrib = float(state.metrics[METRICS_VEC_KEY][feet_phase_idx])

    # smoke12 contract upper bound: at flat feet the contribution
    # must be near 0.  We allow up to 0.05 to absorb physics drift
    # and bonus from microscopic foot lift during the first ctrl step.
    # If the pre-smoke12 raw formula is re-introduced globally this
    # value snaps back to ~0.16, well above the 0.05 ceiling.
    assert feet_phase_contrib < 0.05, (
        f"reward/feet_phase = {feet_phase_contrib:.5f} at iter 1 with "
        "flat feet; expected < 0.05 under the smoke12 baseline-subtract "
        "formula.  Values ≥ 0.05 likely mean the pre-smoke12 raw "
        "formula (which paid ~0.16 for this state) has been "
        "re-introduced globally — the smoke11 standstill basin returns."
    )
    # Lower bound: a finite non-negative value.  Catches a NaN regression
    # or accidental large negative penalty.
    assert feet_phase_contrib >= 0.0, (
        f"reward/feet_phase = {feet_phase_contrib} is negative; the "
        "helper's max(0, ...) clamp was bypassed."
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
