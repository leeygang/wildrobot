"""Post-training checkpoint ranking and deterministic promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


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
    if forward_velocity is not None and forward_velocity < 0.075:
        failures.append("forward_velocity<0.075")
    if cmd_err is not None and cmd_err > 0.075:
        failures.append("cmd_vs_achieved_forward>0.075")
    if episode_length is not None and episode_length < 475.0:
        failures.append("episode_length<475")
    if step_length is not None and step_length < 0.030:
        failures.append("step_length<0.030")
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


def deterministic_eval_gate(
    eval_metrics: Mapping[str, Any],
    eval_velocity_cmd: float,
) -> DeterministicEvalDecision:
    """Apply deterministic post-training promotion gates.

    HARD gates (block promotion) — v0.20.1 G4 set:
      forward_velocity, cmd_vs_achieved_forward, mean_episode_length,
      step_length_touchdown_event_m, forward_velocity_cmd_ratio.

    SOFT signals (report-only, do not block promotion) — smoke15
    deploy-facing diagnostics:
      lateral_velocity_abs, yaw_drift_signed_rad, world_y_drift_signed_m.
    Each gets a ``soft_signals[name]`` boolean that is True when the
    metric is missing or within the documented cap; promoted
    checkpoints can still have these flagged.  The post-training
    summary surfaces both gates + soft_signals so a regression on
    sideways drift is visible even when the G4 gates pass.
    """
    forward_velocity = _metric(eval_metrics, "forward_velocity")
    cmd_err = _metric(eval_metrics, "cmd_vs_achieved_forward")
    episode_length = _metric(eval_metrics, "mean_episode_length")
    step_length = _metric(eval_metrics, "step_length_touchdown_event_m")

    step_metric_available = step_length is not None
    step_ok = True if step_length is None else step_length >= 0.030

    ratio_gate_applied = float(eval_velocity_cmd) > 0.0
    ratio_value: Optional[float] = None
    ratio_ok = True
    if ratio_gate_applied:
        if forward_velocity is None:
            ratio_ok = False
        else:
            ratio_value = float(forward_velocity) / float(eval_velocity_cmd)
            ratio_ok = 0.6 <= ratio_value <= 1.5

    gates: Dict[str, bool] = {
        "forward_velocity": (forward_velocity is not None and forward_velocity >= 0.075),
        "cmd_vs_achieved_forward": (cmd_err is not None and cmd_err <= 0.075),
        "mean_episode_length": (episode_length is not None and episode_length >= 475.0),
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

    return DeterministicEvalDecision(
        passed=passed,
        step_metric_available=step_metric_available,
        ratio_gate_applied=ratio_gate_applied,
        forward_velocity_cmd_ratio=ratio_value,
        gates=gates,
        soft_signals=soft_signals,
    )
