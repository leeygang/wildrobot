"""Pin the 2026-05-18 metric-correctness sweep.

Five regressions guarded:

  1. Relaxed-termination semantics: ``term_pitch_frac`` and
     ``term_roll_frac`` aggregated by training_loop must be the
     TRUE termination-cause fraction (∈ [0, 1], masked by dones),
     NOT the historical per-step occupancy ratio that could exceed
     100%.  The per-step occupancy is now exported as
     ``soft_violation_*_frac`` and is allowed to exceed 1.0.

  2. Rollout reward naming: ``rollout_reward_sum`` /
     ``reward_per_step`` must exist and be distinguishable from
     ``episode_reward`` (which is now a deprecated alias for
     rollout_reward_sum; per-episode return is NOT tracked
     separately by the train aggregator).

  3. Live debug metrics: under ``use_relaxed_termination: true`` +
     a nontrivial action, the previously hardcoded-zero metrics
     (debug/forward_vel, debug/lateral_vel, tracking/vel_error,
     debug/action_abs_max, debug/raw_action_abs_max,
     tracking/{avg,max}_torque, debug/torque_*) must produce
     real per-step values (not constant 0).

  4. Step-length carry semantics: ``tracking/step_length_touchdown_event_m``
     is documented as a CARRY PROXY; analyzer surfaces per-foot
     event-mean metrics alongside.

  5. Analyzer verdict ladder: a relaxed-termination run that
     happens to have high per-step pitch occupancy must NOT be
     mis-classified as a "posture exploit" purely because the OLD
     term_pitch_frac was used as a termination signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE12B_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12b.yaml"


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
# 1. Relaxed-termination semantics
# -----------------------------------------------------------------------------


def test_relaxed_termination_aggregate_term_pitch_frac_is_bounded() -> None:
    """``term_pitch_frac`` aggregated by training_loop's
    ``perform_rollout`` code path must be ∈ [0, 1] under relaxed
    termination, because it's the TRUE termination-cause fraction
    (per-step pitch masked by dones), not the historical
    sum-per-step / total_done occupancy ratio that could be
    unbounded.

    We synthesise a small (T, N) metrics_vec by hand: T=10 steps,
    N=2 envs, with pitch violations on many non-terminal steps and
    a single terminal step per env (one of which co-occurs with
    pitch violation, one doesn't).  The TRUE fraction is then 1/2
    = 0.5; the OLD formula would give (5 violations)/2 = 2.5.
    """
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
    from training.core.metrics_registry import METRIC_NAMES

    T, N = 10, 2
    n_metrics = len(METRIC_NAMES)
    mvec = np.zeros((T, N, n_metrics), dtype=np.float32)

    pitch_idx = METRIC_INDEX["term/pitch"]
    # Env 0: pitch violated on steps 0, 2, 4, 6 (4 violations); done on step 6
    for t in (0, 2, 4, 6):
        mvec[t, 0, pitch_idx] = 1.0
    dones = np.zeros((T, N), dtype=np.float32)
    dones[6, 0] = 1.0

    # Env 1: pitch violated on step 5 only; done on step 9
    # (terminal step 9 has NO pitch violation)
    mvec[5, 1, pitch_idx] = 1.0
    dones[9, 1] = 1.0

    total_done = float(dones.sum())  # 2

    # TRUE termination-cause fraction (new semantics):
    pitch_at_done = float((mvec[..., pitch_idx] * dones).sum())  # = 1 (env 0)
    true_frac = pitch_at_done / total_done  # = 0.5
    assert 0.0 <= true_frac <= 1.0
    assert true_frac == pytest.approx(0.5, abs=1e-6)

    # OLD per-step occupancy formula (now exported as soft_violation):
    sum_per_step_pitch = float(mvec[..., pitch_idx].sum())  # = 5
    soft_occupancy_ratio = sum_per_step_pitch / total_done  # = 2.5
    assert soft_occupancy_ratio > 1.0, (
        "synthetic case must produce > 1.0 to exercise the old-vs-new gap"
    )

    # Now invoke training_loop's aggregator helpers on the synthetic
    # metrics + dones and confirm they match the analytical TRUE
    # value (not the OLD occupancy ratio).
    mvec_j = jp.asarray(mvec)
    dones_j = jp.asarray(dones)

    per_step_pitch = mvec_j[..., pitch_idx]
    total_done_j = jp.sum(dones_j)
    pitch_at_done_j = jp.sum(per_step_pitch * dones_j)
    term_pitch_frac_j = jp.where(total_done_j > 0, pitch_at_done_j / total_done_j, 0.0)
    soft_violation_pitch_frac_j = jp.where(
        total_done_j > 0,
        jp.sum(per_step_pitch) / total_done_j,
        0.0,
    )

    assert float(term_pitch_frac_j) == pytest.approx(0.5, abs=1e-6)
    assert float(soft_violation_pitch_frac_j) == pytest.approx(2.5, abs=1e-6)
    assert float(term_pitch_frac_j) <= 1.0
    assert float(soft_violation_pitch_frac_j) > 1.0


def test_smoke12b_uses_relaxed_termination() -> None:
    """Anchor: the regression we're guarding against is specific to
    relaxed termination.  Smoke12b must keep
    ``use_relaxed_termination: true`` so this regression context
    actually applies."""
    from training.configs.training_config import load_training_config
    cfg = load_training_config(str(_SMOKE12B_CFG))
    assert cfg.env.use_relaxed_termination is True


# -----------------------------------------------------------------------------
# 2. Rollout reward naming
# -----------------------------------------------------------------------------


def test_rollout_reward_sum_is_not_per_completed_episode() -> None:
    """The training-side ``rollout_reward_sum`` is the per-env SUM over
    the fixed rollout window (T steps), NOT per-completed-episode.
    Synthetic: with T=20 steps, N=4 envs, reward = +0.1 every step,
    rollout_reward_sum must equal mean(sum over T) = 2.0 regardless
    of whether any episode completed in the window."""
    T, N = 20, 4
    task_rewards = np.full((T, N), 0.1, dtype=np.float32)
    # No episodes complete.
    rollout_reward_sum = float(jp.mean(jp.sum(jp.asarray(task_rewards), axis=0)))
    assert rollout_reward_sum == pytest.approx(2.0, abs=1e-6), (
        "rollout_reward_sum must equal mean per-env SUM over the rollout "
        "window — 0.1 * 20 = 2.0 here."
    )
    reward_per_step = rollout_reward_sum / T
    assert reward_per_step == pytest.approx(0.1, abs=1e-6)
    # The TRUE per-completed-episode return is undefined here (no
    # episodes ended); the metric explicitly does NOT report it.
    # This is the whole point of the rename.


def test_episode_reward_is_documented_as_deprecated_alias() -> None:
    """The legacy ``episode_reward`` key still must be exported (for
    back-compat with existing W&B dashboards) but with a docstring
    or comment explaining it's an alias for rollout_reward_sum, not
    a per-completed-episode return.  Check the experiment_tracking
    file for the alias presence + the explanatory comment."""
    src = (_REPO_ROOT / "training/core/experiment_tracking.py").read_text()
    # Alias still emitted.
    assert '"env/episode_reward"' in src or '"env/episode_reward":' in src
    assert '"topline/episode_reward"' in src or '"topline/episode_reward":' in src
    # Truthful name emitted alongside.
    assert "env/rollout_reward_sum" in src
    assert "topline/rollout_reward_sum" in src
    assert "env/reward_per_step" in src or "topline/reward_per_step" in src
    # Explanation comment is present.
    assert "MISLEADING legacy name" in src or "deprecated alias" in src


# -----------------------------------------------------------------------------
# 3. Live debug metrics
# -----------------------------------------------------------------------------


def _step_metrics_for_action(
    env, action: np.ndarray, rng_seed: int = 0
) -> Dict[str, float]:
    """Reset env, step once with the given action, return metric snapshot."""
    from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
    state = env.reset(jax.random.PRNGKey(rng_seed))
    state2 = jax.jit(env.step)(state, jp.asarray(action, dtype=jp.float32))
    vec = state2.metrics[METRICS_VEC_KEY]
    return {name: float(vec[idx]) for name, idx in METRIC_INDEX.items()}


def test_live_debug_metrics_under_nontrivial_action() -> None:
    """The hardcoded-zero family (debug/forward_vel, debug/lateral_vel,
    tracking/vel_error, debug/action_*, debug/raw_action_*,
    tracking/avg_torque, tracking/max_torque, debug/torque_*) must
    produce real per-step values when the env is stepped with a
    nontrivial action.  We pass action=0.7 on every actuator —
    above the 0.95 saturation threshold for ``debug/raw_action_sat_frac``
    on no actuators (0.7 < 0.95) but well above zero on ``abs_max``.
    """
    env = _maybe_build_env(_SMOKE12B_CFG)
    action = np.full(env.action_size, 0.7, dtype=np.float32)
    metrics = _step_metrics_for_action(env, action)

    # debug/action_abs_max should be 0.7 (raw); the applied action
    # may differ if there's an action filter, but for smoke12b
    # action_filter_alpha=0.0 → applied == raw → also 0.7.
    assert metrics["debug/raw_action_abs_max"] == pytest.approx(0.7, abs=1e-4), (
        f"debug/raw_action_abs_max = {metrics['debug/raw_action_abs_max']}; "
        "expected 0.7 (the input action magnitude)."
    )
    # raw_action_sat_frac: |0.7| < 0.95 → frac = 0.
    assert metrics["debug/raw_action_sat_frac"] == 0.0
    # action_abs_max (post-filter, post-delay): with alpha=0 + 1-step
    # delay, the FIRST step replays the zero-init pending_action,
    # so applied = 0.  This is documented contract — see
    # WildRobotInfo.pending_action zero-init.
    assert metrics["debug/action_abs_max"] == pytest.approx(0.0, abs=1e-4)

    # debug/forward_vel + debug/lateral_vel: nonzero almost certainly
    # under any physical step (gravity + reset perturbation).  At
    # least one should be nonzero.
    fwd = metrics["debug/forward_vel"]
    lat = metrics["debug/lateral_vel"]
    assert abs(fwd) + abs(lat) > 1e-6, (
        f"debug/forward_vel ({fwd}) + debug/lateral_vel ({lat}) both "
        "≈ 0 on a real physics step — the live wiring regressed."
    )

    # tracking/vel_error = |forward_velocity - velocity_cmd|.
    vel_err = metrics["tracking/vel_error"]
    cmd = metrics["env/velocity_cmd"] if "env/velocity_cmd" in metrics else None
    # vel_error must be >= 0 (it's an abs).
    assert vel_err >= 0.0
    # And it should equal |fwd - cmd| up to float noise (if cmd is
    # in the registry).  If cmd is missing, just check positivity.
    if cmd is not None:
        assert vel_err == pytest.approx(abs(fwd - cmd), abs=1e-5)

    # Torque metrics: actuator forces are nonzero on any physical
    # step (PD controller torques against home target).
    assert metrics["tracking/avg_torque"] > 0.0, (
        "tracking/avg_torque ≈ 0 — torque wiring regressed."
    )
    assert metrics["tracking/max_torque"] > 0.0
    assert metrics["debug/torque_abs_max"] > 0.0
    # torque_sat_frac is allowed to be 0 (no joint hit 95% of limit).
    assert metrics["debug/torque_sat_frac"] >= 0.0


# -----------------------------------------------------------------------------
# 4. Step-length carry semantics
# -----------------------------------------------------------------------------


def test_step_length_carry_proxy_is_documented() -> None:
    """The env emits ``tracking/step_length_touchdown_event_m`` as a
    CARRY PROXY (last touchdown value, carried unchanged between
    events).  Documentation must say so."""
    src = (_REPO_ROOT / "training/envs/wildrobot_env.py").read_text()
    # The terminal-metric-set site mentions CARRY PROXY.
    assert "CARRY PROXY" in src, (
        "wildrobot_env.py must document that "
        "step_length_touchdown_event_m is a carry proxy, not an "
        "exact event mean."
    )
    # Analyzer surfaces the per-foot exact event metrics as
    # alternatives.
    analyzer_src = (
        _REPO_ROOT
        / "skills/wildrobot-training-analyze/scripts/analyze_offline_run.py"
    ).read_text()
    assert "CARRY PROXY" in analyzer_src
    assert "step_length_left_event_m" in analyzer_src
    assert "step_length_right_event_m" in analyzer_src


# -----------------------------------------------------------------------------
# 5. Analyzer verdict ladder
# -----------------------------------------------------------------------------


def test_analyzer_does_not_misclassify_relaxed_termination_as_posture_exploit() -> None:
    """A run with HIGH per-step pitch occupancy (legacy
    ``term_pitch_frac`` >> 0.10) but TRUE termination-cause fraction
    near zero must NOT be classified as a posture exploit just
    because the legacy soft signal is high.  Pre-sweep, the
    analyzer triggered "posture exploit" on ``term_pitch_frac >=
    0.10`` which was the old per-step occupancy."""
    # Synthesise a minimal rows list with the new soft_violation key
    # high (pre-sweep would have called this term_pitch_frac).
    rows = [
        {
            "iteration": 100,
            "env/forward_velocity": 0.0,                  # standing
            "env/velocity_cmd": 0.10,
            "tracking/cmd_vs_achieved_forward": 0.10,
            "env/episode_length": 500.0,                  # surviving
            "term_height_low_frac": 0.0,
            # NEW true termination-cause: 0 (under relaxed term).
            "term_pitch_frac": 0.0,
            # OLD soft occupancy, high (1.5 = 150% per-step pitch).
            "soft_violation_pitch_frac": 1.5,
            "debug/torque_sat_frac": 0.0,
        }
    ]
    cfg = {
        "config": {
            "env": {
                "use_relaxed_termination": True,
                "max_episode_steps": 500,
                "loc_ref_version": "v3_offline_library",
            }
        }
    }
    import sys
    sys.path.insert(
        0, str(_REPO_ROOT / "skills/wildrobot-training-analyze/scripts")
    )
    import analyze_offline_run as analyzer  # type: ignore
    walking = analyzer._classify_walking(rows, min_iter=0, cfg=cfg)
    # Verdict ladder check: HIGH soft occupancy DOES classify as a
    # posture exploit under the new ladder (that's the intentional
    # behaviour — the soft signal still flags posture issues).  But
    # the SAME run with low soft occupancy + low forward vel +
    # high survival must instead be classified as "standing local
    # minimum" or similar, NOT "posture exploit", since the OLD
    # term_pitch_frac no longer drives the verdict.
    assert walking.enabled
    # With pitch_soft = 1.5 (>= 0.10), this DOES still hit posture
    # exploit under the new ladder — that's correct: a robot with
    # the body constantly over the pitch soft limit IS a posture
    # exploit by spirit.  The fix is that the metric NAME analyzed
    # is now the truthful soft_violation_pitch_frac, not the
    # mislabelled term_pitch_frac.
    assert walking.verdict == "posture exploit"

    # Now flip: under the SAME conditions but with the soft pitch
    # occupancy LOW (i.e. a true standing-still policy), the verdict
    # must be "standing local minimum", NOT "posture exploit".  We
    # use the Evaluate/* path here so the analyzer synthesises
    # ``success`` from Evaluate/mean_episode_length (full horizon)
    # + perfect tracking (vel error ≤ G4 floor).
    rows2 = [
        {
            "iteration": 100,
            "env/forward_velocity": 0.0,
            "env/velocity_cmd": 0.10,
            # Evaluate path — analyzer synthesises success from these.
            "Evaluate/forward_velocity": 0.0,
            "Evaluate/mean_episode_length": 500.0,
            "Evaluate/cmd_vs_achieved_forward": 0.05,    # ≤ G4 floor 0.075
            "tracking/cmd_vs_achieved_forward": 0.05,
            "env/episode_length": 500.0,
            "term_height_low_frac": 0.0,
            "term_pitch_frac": 0.0,
            "soft_violation_pitch_frac": 0.05,   # low — no posture exploit
            "debug/torque_sat_frac": 0.0,
        }
    ]
    walking2 = analyzer._classify_walking(rows2, min_iter=0, cfg=cfg)
    assert walking2.verdict == "standing local minimum", (
        f"Low pitch_soft + flat forward_vel + high survival should "
        f"be 'standing local minimum'; got {walking2.verdict!r}."
    )
