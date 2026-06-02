"""Post-training checkpoint ranking and deterministic promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from control.zmp.zmp_walk import ZMPWalkConfig


@dataclass(frozen=True)
class CheckpointMetricCandidate:
    """Raw checkpoint candidate with merged train-side metrics."""

    checkpoint_path: str
    iteration: int
    total_steps: int
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class RankedCheckpointCandidate:
    """Ranked candidate used for post-training deterministic evaluation."""

    checkpoint_path: str
    iteration: int
    total_steps: int
    train_score: float
    train_reward: Optional[float]
    train_forward_velocity: Optional[float]
    train_cmd_err: Optional[float]
    train_step_length: Optional[float]
    train_episode_length: Optional[float]
    train_cmd_ratio: Optional[float]
    rich_metric_count: int
    used_reward_fallback: bool
    passes_filters: bool
    filter_fail_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DeterministicEvalDecision:
    """Deterministic promotion gate decision.

    ``passed`` reflects the HARD gates only — currently the v0.20.1 G4
    set (forward_velocity, cmd_vs_achieved_forward, mean_episode_length,
    step_length_touchdown_event_m, forward_velocity_cmd_ratio).

    ``soft_signals`` is a separate dict of deploy-facing pass/fail
    flags that are computed and logged but do NOT block promotion.
    Lateral velocity, yaw drift, and world-y drift go here.  Track
    these for SEV-style regressions without blocking the run.
    """

    passed: bool
    step_metric_available: bool
    ratio_gate_applied: bool
    forward_velocity_cmd_ratio: Optional[float]
    gates: Mapping[str, bool]
    soft_signals: Mapping[str, bool] = field(default_factory=dict)


# Deploy-facing soft thresholds — report-only.  Picked to be lenient
# enough that a healthy walking policy at WR's vx range comfortably
# clears them; tighten once we have a passing baseline.
#
# Sources:
#   LATERAL_VELOCITY_SOFT_CAP_MPS: 0.10 m/s ≈ 50% of WR cmd vx upper
#     bound (0.20).  Smoke14's tail mean was 0.20 m/s — well above.
#     A healthy forward gait should be below half of cmd.
#   YAW_DRIFT_SOFT_CAP_RAD: 0.40 rad ≈ 23 deg over a 10 s rollout
#     (500 steps × 0.02 s).  Catches material heading drift.
#   WORLD_Y_DRIFT_SOFT_CAP_M: 0.30 m over a 10 s rollout — must be
#     less than the policy's forward progress for any reasonable gait.
LATERAL_VELOCITY_SOFT_CAP_MPS = 0.10
YAW_DRIFT_SOFT_CAP_RAD = 0.40
WORLD_Y_DRIFT_SOFT_CAP_M = 0.30

# walking_training.md Appendix C (v0.21.0+) lateral / yaw cmd
# tracking pass-criterion thresholds.  Both require the SIGNED ratio
# ``achieved / commanded`` to clear 0.5 at the probe cmd point, with
# the signs matching.  Active only when the probe selects the
# relevant axis (vy != 0 for lateral, wz != 0 for yaw).  Smoke1's
# eval probes are configured per
# ``training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml``.
LATERAL_YAW_PASS_RATIO_MIN = 0.5
# Below this absolute cmd magnitude we skip the criterion entirely —
# tiny commands sit within the cmd_deadzone region anyway, and
# dividing by them inflates noise on the signed ratio.
LATERAL_YAW_PROBE_MIN_CMD_ABS = 0.02

# G4 touchdown-stride gate.  The absolute floor is the historical
# v0.20.1 anti-shuffle sanity check; the vx-scaled branch requires at
# least half of the nominal per-touchdown reference stride
# (cmd_vx * cycle_time / 2).
STEP_LENGTH_TOUCHDOWN_ABS_FLOOR_M = 0.030
STEP_LENGTH_TOUCHDOWN_FRACTION_OF_NOMINAL = 0.50


def step_length_touchdown_floor_m(
    eval_velocity_cmd: Optional[float],
    *,
    cycle_time_s: float = ZMPWalkConfig.cycle_time_s,
) -> float:
    """Return the G4 touchdown step-length floor for a commanded vx."""
    if eval_velocity_cmd is None or float(eval_velocity_cmd) <= 0.0:
        return STEP_LENGTH_TOUCHDOWN_ABS_FLOOR_M
    nominal_step = float(eval_velocity_cmd) * float(cycle_time_s) / 2.0
    return max(
        STEP_LENGTH_TOUCHDOWN_ABS_FLOOR_M,
        STEP_LENGTH_TOUCHDOWN_FRACTION_OF_NOMINAL * nominal_step,
    )


@dataclass(frozen=True)
class LateralYawPassDecision:
    """Pass / fail result of a single Appendix C probe.

    Each probe is either a pure-lateral cmd ``(vx, vy, 0)`` with
    ``|vy| >= LATERAL_YAW_PROBE_MIN_CMD_ABS`` or a pure-yaw cmd
    ``(0, 0, wz)`` with ``|wz| >= LATERAL_YAW_PROBE_MIN_CMD_ABS``.
    The decision returns the signed achieved/commanded ratio along
    the active axis and whether it clears the 0.5 floor with matching
    sign.
    """

    axis: str  # "lateral" | "yaw" | "unknown"
    cmd: Tuple[float, float, float]
    achieved: Optional[float]
    commanded: Optional[float]
    signed_ratio: Optional[float]
    passed: bool
    skip_reason: Optional[str] = None


def evaluate_lateral_yaw_pass_criterion(
    probe_cmd: Tuple[float, float, float],
    eval_metrics: Mapping[str, Any],
) -> LateralYawPassDecision:
    """Evaluate the v0.21.0+ Appendix C lateral / yaw pass criterion
    for a single eval probe.

    The probe axis is auto-detected from the cmd shape:

      - Lateral if ``|vy| >= LATERAL_YAW_PROBE_MIN_CMD_ABS`` AND
        ``|wz| < LATERAL_YAW_PROBE_MIN_CMD_ABS``.  Reads
        ``eval_metrics["lateral_velocity_signed_m_s"]`` and compares
        to ``probe_cmd[1]``.
      - Yaw if ``|wz| >= LATERAL_YAW_PROBE_MIN_CMD_ABS`` AND
        ``|vy| < LATERAL_YAW_PROBE_MIN_CMD_ABS``.  Reads
        ``eval_metrics["ang_vel_z_signed_rad_s"]`` and compares to
        ``probe_cmd[2]``.
      - Otherwise marks the probe as ``"unknown"`` axis and skips
        the criterion (mixed probes are not what Appendix C grades).

    Pass when ``signed_ratio = achieved / commanded`` is
    ``>= LATERAL_YAW_PASS_RATIO_MIN`` (0.5).  ``signed_ratio < 0``
    (sign mismatch) fails by construction.  Missing eval metric reads
    as a skip (legacy log back-compat), NOT a pass.
    """
    cmd_tuple = (
        float(probe_cmd[0]),
        float(probe_cmd[1]),
        float(probe_cmd[2]),
    )
    vy_cmd = cmd_tuple[1]
    wz_cmd = cmd_tuple[2]
    lateral_active = abs(vy_cmd) >= LATERAL_YAW_PROBE_MIN_CMD_ABS
    yaw_active = abs(wz_cmd) >= LATERAL_YAW_PROBE_MIN_CMD_ABS

    if lateral_active and not yaw_active:
        axis = "lateral"
        achieved = _metric(eval_metrics, "lateral_velocity_signed_m_s")
        commanded: float = vy_cmd
    elif yaw_active and not lateral_active:
        axis = "yaw"
        achieved = _metric(eval_metrics, "ang_vel_z_signed_rad_s")
        commanded = wz_cmd
    else:
        return LateralYawPassDecision(
            axis="unknown",
            cmd=cmd_tuple,
            achieved=None,
            commanded=None,
            signed_ratio=None,
            passed=False,
            skip_reason=(
                "probe is not a pure-lateral or pure-yaw cmd (Appendix C "
                "criterion requires one axis isolated above "
                f"|{LATERAL_YAW_PROBE_MIN_CMD_ABS}|)"
            ),
        )

    if achieved is None:
        return LateralYawPassDecision(
            axis=axis,
            cmd=cmd_tuple,
            achieved=None,
            commanded=commanded,
            signed_ratio=None,
            passed=False,
            skip_reason=(
                f"eval payload missing {axis}-axis signed metric "
                "(legacy log; signed metrics added 2026-05-24)"
            ),
        )

    ratio = float(achieved) / float(commanded)
    return LateralYawPassDecision(
        axis=axis,
        cmd=cmd_tuple,
        achieved=float(achieved),
        commanded=commanded,
        signed_ratio=ratio,
        passed=(ratio >= LATERAL_YAW_PASS_RATIO_MIN),
        skip_reason=None,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _metric(metrics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics:
            parsed = _safe_float(metrics[key])
            if parsed is not None:
                return parsed
    return None


def _norm_non_negative(value: Optional[float], ref: float, cap: float) -> float:
    if value is None:
        return 0.0
    if ref <= 0.0:
        return 0.0
    return max(0.0, min(float(value) / ref, cap))


def _train_candidate_filter_failures(
    forward_velocity: Optional[float],
    cmd_err: Optional[float],
    step_length: Optional[float],
    episode_length: Optional[float],
    cmd_ratio: Optional[float],
) -> list[str]:
    failures: list[str] = []
    inferred_cmd_vx: Optional[float] = None
    if (
        forward_velocity is not None
        and cmd_ratio is not None
        and abs(float(cmd_ratio)) > 1e-6
    ):
        inferred_cmd_vx = abs(float(forward_velocity) / float(cmd_ratio))
    step_floor = step_length_touchdown_floor_m(inferred_cmd_vx)
    if forward_velocity is not None and forward_velocity < 0.075:
        failures.append("forward_velocity<0.075")
    if cmd_err is not None and cmd_err > 0.075:
        failures.append("cmd_vs_achieved_forward>0.075")
    if episode_length is not None and episode_length < 475.0:
        failures.append("episode_length<475")
    if step_length is not None and step_length < step_floor:
        failures.append(f"step_length<{step_floor:.3f}")
    if cmd_ratio is not None and not (0.6 <= cmd_ratio <= 1.5):
        failures.append("forward_velocity_cmd_ratio∉[0.6,1.5]")
    return failures


def _train_walking_score(
    forward_velocity: Optional[float],
    cmd_err: Optional[float],
    step_length: Optional[float],
    episode_length: Optional[float],
    reward: Optional[float],
) -> tuple[float, bool]:
    reward_tiebreak = 0.05 * math.tanh((reward or 0.0) / 100.0)
    rich_count = sum(
        metric is not None
        for metric in (forward_velocity, cmd_err, step_length, episode_length)
    )
    if rich_count == 0:
        return reward_tiebreak, True

    score = (
        _norm_non_negative(forward_velocity, ref=0.15, cap=2.0)
        - _norm_non_negative(cmd_err, ref=0.15, cap=2.0)
        + _norm_non_negative(step_length, ref=0.06, cap=2.0)
        + _norm_non_negative(episode_length, ref=500.0, cap=1.2)
        + reward_tiebreak
    )
    return score, False


def rank_checkpoint_candidates(
    candidates: Sequence[CheckpointMetricCandidate],
    top_k: int,
) -> tuple[list[RankedCheckpointCandidate], bool]:
    """Rank checkpoints by train-side walking score with metric-aware fallback."""
    ranked_all: list[RankedCheckpointCandidate] = []
    for candidate in candidates:
        metrics = candidate.metrics
        reward = _metric(metrics, "episode_reward")
        forward_velocity = _metric(metrics, "forward_velocity")
        cmd_err = _metric(metrics, "tracking/cmd_vs_achieved_forward")
        step_length = _metric(metrics, "tracking/step_length_touchdown_event_m")
        episode_length = _metric(metrics, "episode_length")
        cmd_ratio = _metric(metrics, "tracking/forward_velocity_cmd_ratio")

        failures = _train_candidate_filter_failures(
            forward_velocity=forward_velocity,
            cmd_err=cmd_err,
            step_length=step_length,
            episode_length=episode_length,
            cmd_ratio=cmd_ratio,
        )
        score, used_reward_fallback = _train_walking_score(
            forward_velocity=forward_velocity,
            cmd_err=cmd_err,
            step_length=step_length,
            episode_length=episode_length,
            reward=reward,
        )
        rich_metric_count = sum(
            metric is not None
            for metric in (
                forward_velocity,
                cmd_err,
                step_length,
                episode_length,
                cmd_ratio,
            )
        )

        ranked_all.append(
            RankedCheckpointCandidate(
                checkpoint_path=candidate.checkpoint_path,
                iteration=int(candidate.iteration),
                total_steps=int(candidate.total_steps),
                train_score=float(score),
                train_reward=reward,
                train_forward_velocity=forward_velocity,
                train_cmd_err=cmd_err,
                train_step_length=step_length,
                train_episode_length=episode_length,
                train_cmd_ratio=cmd_ratio,
                rich_metric_count=rich_metric_count,
                used_reward_fallback=used_reward_fallback,
                passes_filters=len(failures) == 0,
                filter_fail_reasons=tuple(failures),
            )
        )

    def _sort_key(candidate: RankedCheckpointCandidate) -> tuple:
        reward = candidate.train_reward if candidate.train_reward is not None else float("-inf")
        return (
            1 if candidate.passes_filters else 0,
            candidate.rich_metric_count,
            candidate.train_score,
            reward,
            candidate.iteration,
        )

    ranked_all.sort(key=_sort_key, reverse=True)
    top_k = max(1, int(top_k))
    filtered = [candidate for candidate in ranked_all if candidate.passes_filters]
    if filtered:
        return filtered[:top_k], False
    return ranked_all[:top_k], True


def lateral_probe_gate_passed(probe_results: Sequence[Mapping[str, Any]]) -> bool:
    """smoke7 2D-tracking acceptance gate over a candidate's lateral/yaw probes.

    Every NON-skipped probe must pass its Appendix C signed-ratio criterion
    (``evaluate_lateral_yaw_pass_criterion``), AND at least one probe must have
    been evaluated — a configured-but-all-skipped probe set cannot validate 2D
    tracking, so it fails closed.  Used only when
    ``ppo.eval.post_training_strict_lateral_drift`` is on AND probes are
    configured; otherwise probes stay report-only.
    """
    evaluated = [p for p in probe_results if p.get("skip_reason") is None]
    return bool(evaluated) and all(bool(p.get("passed")) for p in evaluated)


def apply_lateral_probe_gate(row: MutableMapping[str, Any]) -> bool:
    """Fold the smoke7 lateral-probe 2D-tracking gate into a candidate eval row,
    keeping ``passed`` / ``gates`` / ``fail_reasons`` CONSISTENT.

    Reads ``row["lateral_yaw_probes"]``, records the result as a first-class
    ``gates["lateral_probe_tracking"]`` entry, and on failure flips
    ``row["passed"]`` to False AND appends ``"lateral_probe_tracking"`` to
    ``row["fail_reasons"]`` — so a probe-failed candidate can never read
    ``passed=False`` with every gate green and no reason (which would mislead
    the no-passing-candidate message + downstream analyzer).  Idempotent.
    Returns the probe-gate result.
    """
    probes_ok = lateral_probe_gate_passed(row.get("lateral_yaw_probes", ()))
    row["lateral_probe_gate_passed"] = probes_ok
    row.setdefault("gates", {})["lateral_probe_tracking"] = probes_ok
    row.setdefault("fail_reasons", [])
    if not probes_ok:
        row["passed"] = False
        if "lateral_probe_tracking" not in row["fail_reasons"]:
            row["fail_reasons"].append("lateral_probe_tracking")
    return probes_ok


def deterministic_eval_gate(
    eval_metrics: Mapping[str, Any],
    eval_velocity_cmd: float,
    *,
    eval_num_steps: int = 500,
    strict_lateral_drift: bool = False,
) -> DeterministicEvalDecision:
    """Apply deterministic post-training promotion gates.

    HARD gates (block promotion) — v0.20.1 G4 set:
      forward_velocity, cmd_vs_achieved_forward, mean_episode_length,
      step_length_touchdown_event_m, forward_velocity_cmd_ratio.

    Horizon-aware gates (smoke2 follow-up): the mean_episode_length
    floor scales with ``eval_num_steps`` (= ``post_training_num_steps``,
    which the caller MUST pass when it differs from the legacy 500).
    A 1000-step horizon evaluated for 1000 steps gives the same 95%
    survival floor (950) that 500 used to (475).  Callers that
    pre-date this kwarg keep the historical 475/500 contract via the
    ``eval_num_steps=500`` default.

    The touchdown step-length floor is vx-scaled:
    ``max(0.030, 0.50 * eval_velocity_cmd * ZMP_cycle_time / 2)``.
    The absolute 30 mm floor keeps old low-vx anti-shuffle behavior;
    smoke2's vx=0.20 with WR's 0.96 s cycle requires 48 mm.

    SOFT signals (report-only, do not block promotion) — smoke15
    deploy-facing diagnostics.  Each gets a
    ``soft_signals[name]`` boolean that is True when the metric is
    missing or within the documented cap; promoted checkpoints can
    still have these flagged.  Soft-signal payload keys:

      - ``lateral_velocity_abs`` — per-step mean of |vy|.
      - ``yaw_drift_abs_rad`` — per-episode-terminal mean of
        |yaw drift since spawn| (NOT the signed mean).  Falls back
        to ``abs(yaw_drift_signed_rad)`` for legacy logs that
        predate the abs-aggregation fix.
      - ``world_y_drift_abs_m`` — per-episode-terminal mean of
        |world-y drift since spawn|.  Falls back to
        ``abs(world_y_drift_signed_m)`` for legacy logs.

    The signed-variant fallback exists for back-compat only; under
    cross-env sign cancellation the signed mean understates drift
    (half the envs at +0.5 m and half at -0.5 m read as ~0 mean
    even though every env is bad), which is why new runs always
    emit the ``*_abs_*`` variants from per-episode aggregation
    BEFORE cross-env reduction.

    The post-training summary surfaces both gates + soft_signals
    so a regression on sideways drift is visible even when the G4
    gates pass.
    """
    forward_velocity = _metric(eval_metrics, "forward_velocity")
    cmd_err = _metric(eval_metrics, "cmd_vs_achieved_forward")
    episode_length = _metric(eval_metrics, "mean_episode_length")
    step_length = _metric(eval_metrics, "step_length_touchdown_event_m")

    step_floor = step_length_touchdown_floor_m(eval_velocity_cmd)
    step_metric_available = step_length is not None
    step_ok = True if step_length is None else step_length >= step_floor

    ratio_gate_applied = float(eval_velocity_cmd) > 0.0
    ratio_value: Optional[float] = None
    ratio_ok = True
    if ratio_gate_applied:
        if forward_velocity is None:
            ratio_ok = False
        else:
            ratio_value = float(forward_velocity) / float(eval_velocity_cmd)
            ratio_ok = 0.6 <= ratio_value <= 1.5

    # smoke2 follow-up: scale the mean_episode_length floor with the
    # eval horizon.  Historical 500-step horizon used 475 (=0.95*500);
    # 1000-step horizon at the same survival ratio yields 950.
    # Keeping the 0.95 ratio means the gate semantics ("policy must
    # survive 95% of the rollout") are invariant to horizon choice.
    episode_length_floor = 0.95 * float(eval_num_steps)
    gates: Dict[str, bool] = {
        "forward_velocity": (forward_velocity is not None and forward_velocity >= 0.075),
        "cmd_vs_achieved_forward": (cmd_err is not None and cmd_err <= 0.075),
        "mean_episode_length": (
            episode_length is not None and episode_length >= episode_length_floor
        ),
        "step_length_touchdown_event_m": step_ok,
        "forward_velocity_cmd_ratio": ratio_ok,
    }
    passed = all(gates.values())

    # Soft signals — report-only.  Read the per-episode-abs aggregated
    # variants (``*_abs_*``) so cross-env sign cancellation cannot
    # silently mask a bad policy where half the envs drift +0.5 and
    # half drift -0.5 (mean ≈ 0 but every env is bad).  Backwards-
    # compat: old eval payloads that carry only the signed variant
    # fall back to ``abs(signed)`` to avoid breaking historical
    # summaries, but the smoke15 train.py + training_loop.py paths
    # always emit the ``*_abs_*`` variant so new runs use the
    # invariant-correct value.  Missing entirely => signal OK.
    lateral = _metric(eval_metrics, "lateral_velocity_abs")
    yaw_drift_abs = _metric(
        eval_metrics, "yaw_drift_abs_rad", "yaw_drift_signed_rad"
    )
    world_y_drift_abs = _metric(
        eval_metrics, "world_y_drift_abs_m", "world_y_drift_signed_m"
    )
    soft_signals: Dict[str, bool] = {
        "lateral_velocity_abs": (
            lateral is None or abs(float(lateral)) <= LATERAL_VELOCITY_SOFT_CAP_MPS
        ),
        "yaw_drift_abs_rad": (
            yaw_drift_abs is None
            or abs(float(yaw_drift_abs)) <= YAW_DRIFT_SOFT_CAP_RAD
        ),
        "world_y_drift_abs_m": (
            world_y_drift_abs is None
            or abs(float(world_y_drift_abs)) <= WORLD_Y_DRIFT_SOFT_CAP_M
        ),
    }

    # smoke7 — config-gated strict mode: promote lateral_velocity_abs and
    # world_y_drift_abs_m from report-only soft signals to HARD gates, so a
    # forward+tiny-vy run cannot promote a directionally-drifting checkpoint.
    # Uses the same documented soft caps; missing metric => FAIL (a hard gate
    # cannot be cleared by an absent measurement).  Default off => historical
    # behavior (these remain soft signals) is byte-identical.
    if strict_lateral_drift:
        gates = {
            **gates,
            "lateral_velocity_abs": (
                lateral is not None
                and abs(float(lateral)) <= LATERAL_VELOCITY_SOFT_CAP_MPS
            ),
            "world_y_drift_abs_m": (
                world_y_drift_abs is not None
                and abs(float(world_y_drift_abs)) <= WORLD_Y_DRIFT_SOFT_CAP_M
            ),
        }
        passed = all(gates.values())

    return DeterministicEvalDecision(
        passed=passed,
        step_metric_available=step_metric_available,
        ratio_gate_applied=ratio_gate_applied,
        forward_velocity_cmd_ratio=ratio_value,
        gates=gates,
        soft_signals=soft_signals,
    )
