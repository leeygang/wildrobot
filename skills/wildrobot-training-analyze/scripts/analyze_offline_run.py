#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EvalKeySet:
    prefix: str  # "eval_push/" or "eval/" or "env/"
    success: str
    ep_len: str


@dataclass(frozen=True)
class WalkingSummary:
    enabled: bool
    forward_vel_mean: Optional[float]
    velocity_cmd_mean: Optional[float]
    velocity_error_mean: Optional[float]
    verdict: str
    tracking_status: str
    stability: str


@dataclass(frozen=True)
class RunAnalysis:
    run_dir: Path
    run_id: str
    version: str
    version_name: str
    tags: Tuple[str, ...]
    cfg: Dict[str, Any]
    rows: List[Dict[str, Any]]
    keys: EvalKeySet
    best_it: int
    best_row: Dict[str, Any]
    ckpt_dir: Optional[Path]
    ckpt_file: Optional[Path]
    walking: WalkingSummary


@dataclass(frozen=True)
class RegressionMetricSpec:
    label: str
    key: str
    mode: str  # "higher", "lower", or "target"
    digits: int
    tol: float
    group: str
    gate_kind: Optional[str] = None  # "min", "max", or "band"
    gate_low: Optional[float] = None
    gate_high: Optional[float] = None
    target: Optional[float] = None


@dataclass(frozen=True)
class RegressionMetric:
    label: str
    previous: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    digits: int
    status: str
    group: str
    previous_gate: str
    current_gate: str


@dataclass(frozen=True)
class RegressionSummary:
    previous: RunAnalysis
    current_it: int
    previous_it: int
    overall: str
    metrics: List[RegressionMetric]


REGRESSION_SPECS: Tuple[RegressionMetricSpec, ...] = (
    RegressionMetricSpec(
        label="env/forward_velocity",
        key="env/forward_velocity",
        mode="higher",
        digits=3,
        tol=0.005,
        group="gate",
        gate_kind="min",
        gate_low=0.075,
    ),
    RegressionMetricSpec(
        label="env/episode_length (info ≥ 475)",
        key="env/episode_length",
        mode="higher",
        digits=1,
        tol=2.0,
        group="gate",
        gate_kind="min",
        gate_low=475.0,
    ),
    RegressionMetricSpec(
        label="tracking/cmd_vs_achieved_forward",
        key="tracking/cmd_vs_achieved_forward",
        mode="lower",
        digits=3,
        tol=0.003,
        group="gate",
        gate_kind="max",
        gate_high=0.075,
    ),
    RegressionMetricSpec(
        label="tracking/step_length_touchdown_event_m",
        key="tracking/step_length_touchdown_event_m",
        mode="higher",
        digits=4,
        tol=0.001,
        group="gate",
        gate_kind="min",
        gate_low=0.030,
    ),
    RegressionMetricSpec(
        label="tracking/forward_velocity_cmd_ratio",
        key="tracking/forward_velocity_cmd_ratio",
        mode="target",
        digits=3,
        tol=0.010,
        group="gate",
        gate_kind="band",
        gate_low=0.6,
        gate_high=1.5,
        target=1.0,
    ),
    RegressionMetricSpec(
        label="tracking/residual_hip_pitch_left_abs",
        key="tracking/residual_hip_pitch_left_abs",
        mode="lower",
        digits=3,
        tol=0.005,
        group="gate",
        gate_kind="max",
        gate_high=0.20,
    ),
    RegressionMetricSpec(
        label="tracking/residual_hip_pitch_right_abs",
        key="tracking/residual_hip_pitch_right_abs",
        mode="lower",
        digits=3,
        tol=0.005,
        group="gate",
        gate_kind="max",
        gate_high=0.20,
    ),
    RegressionMetricSpec(
        label="tracking/residual_knee_left_abs",
        key="tracking/residual_knee_left_abs",
        mode="lower",
        digits=3,
        tol=0.005,
        group="gate",
        gate_kind="max",
        gate_high=0.20,
    ),
    RegressionMetricSpec(
        label="tracking/residual_knee_right_abs",
        key="tracking/residual_knee_right_abs",
        mode="lower",
        digits=3,
        tol=0.005,
        group="gate",
        gate_kind="max",
        gate_high=0.20,
    ),
    RegressionMetricSpec(
        label="ref/feet_pos_err_l2",
        key="ref/feet_pos_err_l2",
        mode="lower",
        digits=3,
        tol=0.010,
        group="diagnostic",
    ),
    RegressionMetricSpec(
        label="ref/contact_phase_match",
        key="ref/contact_phase_match",
        mode="higher",
        digits=3,
        tol=0.005,
        group="diagnostic",
    ),
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _get_iter(row: Dict[str, Any]) -> Optional[int]:
    for key in ("progress/iteration", "iteration"):
        if key in row:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def _get_float(row: Dict[str, Any], key: str) -> Optional[float]:
    if key not in row:
        return None
    try:
        v = float(row[key])
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _push_enabled(cfg: Dict[str, Any]) -> bool:
    env = cfg.get("config", {}).get("env", {})
    return bool(env.get("push_enabled", False))


def _pick_eval_keys(rows: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> EvalKeySet:
    # v0.20.1-smoke2 (ToddlerBot-aligned): walking runs now log
    # ``Evaluate/mean_reward`` + ``Evaluate/mean_episode_length`` from a
    # deterministic eval rollout (mirrors
    # toddlerbot/locomotion/train_mjx.py log_metrics).  Prefer those over
    # the legacy success_rate keys for walking — those are now permanently
    # zero (truncation-based, removed from the smoke topline).  The
    # ``success`` field of EvalKeySet becomes "ToddlerBot mean_reward"
    # in this branch; both fields are higher-is-better so the existing
    # _lexi_better / _find_best_row sort works unchanged.
    if _is_walking_run(cfg, rows):
        if any("Evaluate/mean_reward" in r for r in rows):
            return EvalKeySet(
                prefix="Evaluate/",
                success="Evaluate/mean_reward",
                ep_len="Evaluate/mean_episode_length",
            )

    # For walking runs without pushes, eval_push is not informative.
    if _is_walking_run(cfg, rows) and not _push_enabled(cfg):
        if any("eval_clean/success_rate" in r for r in rows):
            return EvalKeySet(
                prefix="eval_clean/",
                success="eval_clean/success_rate",
                ep_len="eval_clean/episode_length",
            )
        if any("eval/success_rate" in r for r in rows):
            return EvalKeySet(prefix="eval/", success="eval/success_rate", ep_len="eval/episode_length")
        if any("env/success_rate" in r for r in rows):
            return EvalKeySet(prefix="env/", success="env/success_rate", ep_len="env/episode_length")
        return EvalKeySet(prefix="", success="success_rate", ep_len="episode_length")

    # Prefer dual-eval keys if present.
    if any("eval_push/success_rate" in r for r in rows):
        return EvalKeySet(
            prefix="eval_push/",
            success="eval_push/success_rate",
            ep_len="eval_push/episode_length",
        )
    if any("eval/success_rate" in r for r in rows):
        return EvalKeySet(prefix="eval/", success="eval/success_rate", ep_len="eval/episode_length")
    # Fallback to training-rollout metrics
    if any("env/success_rate" in r for r in rows):
        return EvalKeySet(prefix="env/", success="env/success_rate", ep_len="env/episode_length")
    # Older logs sometimes use bare names
    return EvalKeySet(prefix="", success="success_rate", ep_len="episode_length")


def _lexi_better(
    s: float,
    L: float,
    best_s: float,
    best_L: float,
    eps: float = 1e-9,
) -> bool:
    if s > best_s + eps:
        return True
    return abs(s - best_s) <= eps and L > best_L + eps


def _find_best_row(rows: List[Dict[str, Any]], keys: EvalKeySet) -> Tuple[int, Dict[str, Any]]:
    best_it = -1
    best_row: Dict[str, Any] = {}
    best_s = float("-inf")
    best_L = float("-inf")
    for r in rows:
        it = _get_iter(r)
        if it is None or it < 0:
            continue
        s = _get_float(r, keys.success)
        L = _get_float(r, keys.ep_len)
        if s is None or L is None:
            continue
        if _lexi_better(s, L, best_s, best_L) or (s == best_s and L == best_L and it > best_it):
            best_s, best_L = s, L
            best_it, best_row = it, r
    if best_it < 0:
        raise SystemExit(f"No rows contained both {keys.success!r} and {keys.ep_len!r}.")
    return best_it, best_row


def _find_checkpoint_dir(run_id: str, checkpoints_root: Path) -> Optional[Path]:
    candidates = sorted(checkpoints_root.glob(f"**/*{run_id}*"))
    dirs = [p for p in candidates if p.is_dir()]
    if not dirs:
        return None
    # Prefer shallowest match with the run_id suffix; otherwise newest mtime.
    def score(p: Path) -> Tuple[int, float]:
        depth = len(p.parts)
        mtime = p.stat().st_mtime
        return (depth, -mtime)

    return sorted(dirs, key=score)[0]


def _find_checkpoint_file(
    ckpt_dir: Path, iteration: int, step: Optional[int]
) -> Optional[Path]:
    if step is not None:
        exact = ckpt_dir / f"checkpoint_{iteration}_{step}.pkl"
        if exact.exists():
            return exact
    # Fallback: any checkpoint for this iteration.
    matches = sorted(ckpt_dir.glob(f"checkpoint_{iteration}_*.pkl"))
    return matches[0] if matches else None


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _first_present(*candidates: Optional[float]) -> Optional[float]:
    """Return the first ``candidate`` that is not ``None``.

    Use this instead of Python's ``or`` chain when the values may
    legitimately be ``0.0``: ``0.0 or x`` evaluates to ``x``, which would
    silently swap a perfectly-tracked metric for the next fallback (and
    later get treated as "missing").
    """
    for c in candidates:
        if c is not None:
            return c
    return None


def _collect_series(
    rows: List[Dict[str, Any]], key: str, *, min_iter: int = 10
) -> List[float]:
    out: List[float] = []
    for r in rows:
        it = _get_iter(r)
        if it is None or it < min_iter:
            continue
        v = _get_float(r, key)
        if v is not None:
            out.append(v)
    return out


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.2f}%"


def _format_f(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _is_walking_run(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    env = cfg.get("config", {}).get("env", {})
    try:
        if float(env.get("max_velocity", 0.0)) > 0.0:
            return True
    except Exception:
        pass
    walking_keys = (
        "env/forward_velocity",
        "env/velocity_cmd",
        "Evaluate/forward_velocity",
        "tracking/cmd_vs_achieved_forward",
    )
    return any(any(k in r for k in walking_keys) for r in rows)


def _classify_walking(
    rows: List[Dict[str, Any]],
    *,
    min_iter: int,
    cfg: Dict[str, Any],
) -> WalkingSummary:
    if not _is_walking_run(cfg, rows):
        return WalkingSummary(
            enabled=False,
            forward_vel_mean=None,
            velocity_cmd_mean=None,
            velocity_error_mean=None,
            verdict="not_applicable",
            tracking_status="not_applicable",
            stability="not_applicable",
        )

    # v0.20.1-smoke2: prefer Evaluate/* (deterministic eval rollout, ToddlerBot
    # pattern) over the now-degenerate env/success_rate / env/velocity_error
    # path.  Walking runs that retired success_rate / velocity_error from the
    # topline still log forward_velocity at env/* (unchanged), so use the
    # eval-rollout forward_velocity when present and fall back gracefully.
    has_evaluate = any("Evaluate/forward_velocity" in r for r in rows)
    forward_vel_key = (
        "Evaluate/forward_velocity" if has_evaluate else "env/forward_velocity"
    )
    forward_vel_mean = _mean(_collect_series(rows, forward_vel_key, min_iter=min_iter))
    velocity_cmd_mean = _mean(_collect_series(rows, "env/velocity_cmd", min_iter=min_iter))
    # Use the v0.20.1 walking-meaningful tracking signal
    # (|achieved_vx - cmd_vx|) when present; legacy env/velocity_error is
    # always 0 for the v0.20.1 smoke.  IMPORTANT: walk the candidates by
    # ``is not None`` rather than Python's ``or`` truthiness, otherwise a
    # perfectly-tracked run with ``Evaluate/cmd_vs_achieved_forward == 0.0``
    # falls through to the next fallback and is later treated as "missing
    # → infinity → tracking bad".
    velocity_error_mean = _first_present(
        _mean(_collect_series(rows, "Evaluate/cmd_vs_achieved_forward", min_iter=min_iter)),
        _mean(_collect_series(rows, "tracking/cmd_vs_achieved_forward", min_iter=min_iter)),
        _mean(_collect_series(rows, "env/velocity_error", min_iter=min_iter)),
    )
    # ``success`` for v0.20.1 walking is "did the deterministic eval rollout
    # last most of the horizon AND track command velocity?".  Synthesize from
    # Evaluate/mean_episode_length + the cmd-tracking error already pulled
    # above; legacy success_rate is always 0 for these runs.
    if has_evaluate:
        ep_len_mean = _mean(_collect_series(rows, "Evaluate/mean_episode_length", min_iter=min_iter))
        max_ep = float(cfg.get("config", {}).get("env", {}).get("max_episode_steps", 500) or 500)
        survival_frac = (ep_len_mean / max_ep) if (ep_len_mean is not None and max_ep > 0) else 0.0
        # Tracking is "ok" only if we actually have a tracking signal AND
        # it's within the G4 floor.  ``None`` (missing field) is treated
        # the same as "bad" so we don't claim success on incomplete logs.
        track_ok = velocity_error_mean is not None and velocity_error_mean <= 0.075
        success_mean = survival_frac if track_ok else min(survival_frac, 0.5)
    else:
        success_mean = _mean(_collect_series(rows, "env/success_rate", min_iter=min_iter))
    term_low_mean = _mean(_collect_series(rows, "term_height_low_frac", min_iter=min_iter))
    term_pitch_mean = _mean(_collect_series(rows, "term_pitch_frac", min_iter=min_iter))
    torque_sat_mean = _mean(_collect_series(rows, "debug/torque_sat_frac", min_iter=min_iter))

    fwd = forward_vel_mean or 0.0
    vel_err = velocity_error_mean if velocity_error_mean is not None else float("inf")
    success = success_mean or 0.0
    term_pitch = term_pitch_mean or 0.0
    torque_sat = torque_sat_mean or 0.0

    # Walking-rate gates (m/s).  v0.20.1 smoke targets vx=0.15 — the legacy
    # 0.20 / 0.10 thresholds were calibrated for vx=0.30+ standing-era
    # branches and would mis-classify the smoke as "needs review" even when
    # tracking is good.
    fwd_emerging = 0.075  # G4 promotion-horizon floor
    fwd_low = 0.05
    vel_err_ok = 0.075    # G4 promotion-horizon floor

    # v0.20.1 verdict ladder (matches walking_training.md G4 + the
    # v0.19.5 / v0.20.1 failure-signature catalogue in SKILL.md).
    if fwd < fwd_low and term_pitch >= 0.10:
        verdict = "posture exploit"
    elif fwd < fwd_low and success >= 0.80:
        verdict = "standing local minimum"
    elif fwd < fwd_emerging and torque_sat >= 0.03:
        verdict = "shuffle exploit"
    elif fwd >= fwd_emerging and vel_err <= vel_err_ok:
        verdict = "locomotion emerging"
    else:
        verdict = "needs review"

    if fwd >= fwd_emerging and vel_err <= vel_err_ok:
        tracking_status = "improving"
    elif fwd < fwd_low and vel_err >= 0.20:
        tracking_status = "flat"
    else:
        tracking_status = "partial"

    if success >= 0.90 and (term_low_mean or 0.0) < 0.05 and term_pitch < 0.05:
        stability = "good"
    elif success >= 0.70:
        stability = "acceptable"
    else:
        stability = "poor"

    return WalkingSummary(
        enabled=True,
        forward_vel_mean=forward_vel_mean,
        velocity_cmd_mean=velocity_cmd_mean,
        velocity_error_mean=velocity_error_mean,
        verdict=verdict,
        tracking_status=tracking_status,
        stability=stability,
    )


def _load_run_analysis(
    run_dir: Path,
    *,
    checkpoints_root: Path,
    min_iter: int,
) -> RunAnalysis:
    files_dir = run_dir / "files"
    metrics_path = files_dir / "metrics.jsonl"
    config_path = files_dir / "config.json"

    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics: {metrics_path}")
    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")

    run_id = run_dir.name.split("-")[-1]
    cfg = _read_json(config_path)
    version = str(cfg.get("config", {}).get("version", "unknown"))
    version_name = str(cfg.get("config", {}).get("version_name", ""))
    tags = tuple(sorted(str(tag) for tag in cfg.get("tags", []) if tag))

    rows = _read_jsonl(metrics_path)
    rows = [r for r in rows if _get_iter(r) is not None]
    rows.sort(key=lambda r: _get_iter(r) or -1)

    keys = _pick_eval_keys(rows, cfg=cfg)
    best_it, best_row = _find_best_row(rows, keys)
    best_step = None
    for key in ("time/step", "_step"):
        if key in best_row:
            try:
                best_step = int(best_row[key])
                break
            except Exception:
                best_step = None

    ckpt_dir = _find_checkpoint_dir(run_id, checkpoints_root)
    ckpt_file = _find_checkpoint_file(ckpt_dir, best_it, best_step) if ckpt_dir else None
    walking = _classify_walking(rows, min_iter=min_iter, cfg=cfg)

    return RunAnalysis(
        run_dir=run_dir,
        run_id=run_id,
        version=version,
        version_name=version_name,
        tags=tags,
        cfg=cfg,
        rows=rows,
        keys=keys,
        best_it=best_it,
        best_row=best_row,
        ckpt_dir=ckpt_dir,
        ckpt_file=ckpt_file,
        walking=walking,
    )


def _find_previous_run_dir(current: RunAnalysis, *, wandb_root: Path) -> Optional[Path]:
    if not wandb_root.exists():
        return None

    candidates = sorted(
        p for p in wandb_root.glob("offline-run-*")
        if p.is_dir() and p.name < current.run_dir.name
    )
    for candidate in reversed(candidates):
        config_path = candidate / "files" / "config.json"
        metrics_path = candidate / "files" / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue
        try:
            cfg = _read_json(config_path)
            rows = _read_jsonl(metrics_path)
        except Exception:
            continue
        rows = [r for r in rows if _get_iter(r) is not None]
        rows.sort(key=lambda r: _get_iter(r) or -1)
        if not rows:
            continue

        version = str(cfg.get("config", {}).get("version", "unknown"))
        version_name = str(cfg.get("config", {}).get("version_name", ""))
        tags = tuple(sorted(str(tag) for tag in cfg.get("tags", []) if tag))

        if version != current.version:
            continue
        if current.version_name and version_name and version_name != current.version_name:
            continue
        if current.tags and tags and tags != current.tags:
            continue
        if _is_walking_run(cfg, rows) != current.walking.enabled:
            continue
        return candidate
    return None


def _best_regression_row(analysis: RunAnalysis) -> Tuple[int, Dict[str, Any]]:
    if not analysis.walking.enabled:
        return analysis.best_it, analysis.best_row

    best_it = analysis.best_it
    best_row = analysis.best_row
    best_score: Optional[Tuple[float, ...]] = None
    for row in analysis.rows:
        it = _get_iter(row)
        if it is None or it < 0:
            continue

        env_fwd = _get_float(row, "env/forward_velocity")
        env_ep_len = _get_float(row, "env/episode_length")
        cmd_err = _get_float(row, "tracking/cmd_vs_achieved_forward")
        step_len = _get_float(row, "tracking/step_length_touchdown_event_m")
        ratio = _get_float(row, "tracking/forward_velocity_cmd_ratio")
        residuals = [
            _get_float(row, "tracking/residual_hip_pitch_left_abs"),
            _get_float(row, "tracking/residual_hip_pitch_right_abs"),
            _get_float(row, "tracking/residual_knee_left_abs"),
            _get_float(row, "tracking/residual_knee_right_abs"),
        ]

        pass_count = 0
        if env_fwd is not None and env_fwd >= 0.075:
            pass_count += 1
        if env_ep_len is not None and env_ep_len >= 475.0:
            pass_count += 1
        if cmd_err is not None and cmd_err <= 0.075:
            pass_count += 1
        if step_len is not None and step_len >= 0.030:
            pass_count += 1
        if ratio is not None and 0.6 <= ratio <= 1.5:
            pass_count += 1
        for residual in residuals:
            if residual is not None and residual <= 0.20:
                pass_count += 1

        score = (
            float(pass_count),
            step_len if step_len is not None else float("-inf"),
            -(cmd_err if cmd_err is not None else float("inf")),
            -(abs(ratio - 1.0) if ratio is not None else float("inf")),
            env_ep_len if env_ep_len is not None else float("-inf"),
            env_fwd if env_fwd is not None else float("-inf"),
            float(it),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_it = it
            best_row = row
    return best_it, best_row


def _gate_status(spec: RegressionMetricSpec, value: Optional[float]) -> str:
    if value is None or spec.gate_kind is None:
        return "N/A"
    if spec.gate_kind == "min":
        return "pass" if value >= float(spec.gate_low) else "fail"
    if spec.gate_kind == "max":
        return "pass" if value <= float(spec.gate_high) else "fail"
    if spec.gate_kind == "band":
        return "pass" if float(spec.gate_low) <= value <= float(spec.gate_high) else "fail"
    return "N/A"


def _compare_regression_metric(
    spec: RegressionMetricSpec,
    *,
    current_row: Dict[str, Any],
    previous_row: Dict[str, Any],
) -> RegressionMetric:
    current = _get_float(current_row, spec.key)
    previous = _get_float(previous_row, spec.key)
    delta = None
    status = "N/A"

    if current is not None and previous is not None:
        delta = current - previous
        if spec.mode == "higher":
            improvement = current - previous
        elif spec.mode == "lower":
            improvement = previous - current
        elif spec.mode == "target":
            target = float(spec.target)
            improvement = abs(previous - target) - abs(current - target)
        else:
            raise ValueError(f"Unknown regression comparison mode: {spec.mode}")

        if improvement > spec.tol:
            status = "improved"
        elif improvement < -spec.tol:
            status = "regressed"
        else:
            status = "flat"

    return RegressionMetric(
        label=spec.label,
        previous=previous,
        current=current,
        delta=delta,
        digits=spec.digits,
        status=status,
        group=spec.group,
        previous_gate=_gate_status(spec, previous),
        current_gate=_gate_status(spec, current),
    )


def _summarize_regression_overall(metrics: List[RegressionMetric]) -> str:
    gate_drop = any(
        metric.group == "gate"
        and metric.previous_gate == "pass"
        and metric.current_gate == "fail"
        for metric in metrics
    )
    improved = any(metric.status == "improved" for metric in metrics)
    regressed = any(metric.status == "regressed" for metric in metrics)

    if gate_drop:
        return "mixed" if improved else "regression"
    if improved and regressed:
        return "mixed"
    if regressed:
        return "regression"
    if improved:
        return "improvement"
    return "flat"


def _build_regression_summary(
    current: RunAnalysis,
    previous: RunAnalysis,
) -> RegressionSummary:
    current_it, current_row = _best_regression_row(current)
    previous_it, previous_row = _best_regression_row(previous)
    metrics = [
        _compare_regression_metric(
            spec,
            current_row=current_row,
            previous_row=previous_row,
        )
        for spec in REGRESSION_SPECS
    ]
    return RegressionSummary(
        previous=previous,
        current_it=current_it,
        previous_it=previous_it,
        overall=_summarize_regression_overall(metrics),
        metrics=metrics,
    )


def _format_delta(delta: Optional[float], digits: int) -> str:
    if delta is None:
        return "N/A"
    return f"{delta:+.{digits}f}"


def _format_gate_transition(previous_gate: str, current_gate: str) -> str:
    if previous_gate == "N/A" and current_gate == "N/A":
        return "N/A"
    return f"{previous_gate}->{current_gate}"


def _changelog_block(
    *,
    version: str,
    run_dir: Path,
    ckpt_dir: Optional[Path],
    best_ckpt: Optional[Path],
    keys: EvalKeySet,
    best_it: int,
    best_row: Dict[str, Any],
    walking: WalkingSummary,
    regression: Optional[RegressionSummary],
) -> str:
    def g(key: str) -> Optional[float]:
        return _get_float(best_row, key)

    eval_s = g(keys.success)
    eval_L = g(keys.ep_len)
    train_s = _first_present(g("env/success_rate"), g("success_rate"))
    train_L = _first_present(g("env/episode_length"), g("episode_length"))

    term_low = g("term_height_low_frac")
    term_pitch = g("term_pitch_frac")
    term_roll = g("term_roll_frac")
    max_torque = _first_present(g("env/max_torque"), g("tracking/max_torque"))
    torque_sat = g("debug/torque_sat_frac")
    approx_kl = _first_present(g("ppo/approx_kl"), g("approx_kl"))
    clip_frac = _first_present(g("ppo/clip_fraction"), g("clip_fraction"))

    lines = [
        f"### Results (v{version})",
        f"- Run: `{run_dir.as_posix()}`",
    ]
    if ckpt_dir is not None:
        lines.append(f"- Checkpoints: `{ckpt_dir.as_posix()}`")
    if best_ckpt is not None:
        lines.append(f"- Best checkpoint ({keys.prefix or 'metric'}): `{best_ckpt.as_posix()}`")
    # ``keys.success`` is a probability in [0,1] for the legacy
    # ``*/success_rate`` keys (format as %), but a raw mean reward (often
    # tens) for ``Evaluate/mean_reward``.  Format accordingly so we don't
    # print "3400.00%" for a mean_reward = 34 row.
    success_is_rate = keys.success.endswith("/success_rate") or keys.success == "success_rate"
    fmt_success = (lambda v: _format_pct(v)) if success_is_rate else (lambda v: _format_f(v, 3))
    lines += [
        f"- Best @ iter {best_it}: {keys.success}={fmt_success(eval_s)}, {keys.ep_len}={_format_f(eval_L, 1)}",
        f"- Train @ iter {best_it}: success={_format_pct(train_s)}, ep_len={_format_f(train_L, 1)}",
    ]
    if walking.enabled:
        lines += [
            f"- Walking verdict: {walking.verdict}",
            f"- Walking tracking: {walking.tracking_status}",
            f"- Walking stability: {walking.stability}",
        ]
    lines += [
        "",
        "| Signal | Value |",
        "|---|---:|",
        f"| term_height_low_frac | {_format_pct(term_low)} |",
        f"| term_pitch_frac | {_format_pct(term_pitch)} |",
        f"| term_roll_frac | {_format_pct(term_roll)} |",
        f"| tracking/max_torque | {_format_pct(max_torque)} |",
        f"| debug/torque_sat_frac | {_format_pct(torque_sat)} |",
        f"| ppo/approx_kl | {_format_f(approx_kl, 4)} |",
        f"| ppo/clip_fraction | {_format_f(clip_frac, 4)} |",
    ]
    if walking.enabled:
        # Labels reflect the keys actually consumed by ``_summarize_walking``
        # (Evaluate/* preferred for v0.20.1-smoke2; falls back to env/* and
        # tracking/* for older runs).
        fwd_label = "Evaluate/forward_velocity (eval) or env/forward_velocity"
        verr_label = (
            "Evaluate/cmd_vs_achieved_forward (eval) or "
            "tracking/cmd_vs_achieved_forward (train)"
        )
        lines += [
            f"| {fwd_label} | {_format_f(walking.forward_vel_mean, 3)} |",
            f"| env/velocity_cmd | {_format_f(walking.velocity_cmd_mean, 3)} |",
            f"| {verr_label} | {_format_f(walking.velocity_error_mean, 3)} |",
        ]

        # v0.20.1 G4 / G5 diagnostics — print only when the relevant keys
        # are present in the best row (silently skipped on standing-era runs).
        v0201_g4_rows = [
            ("Evaluate/forward_velocity (G4 ≥ 0.075)", "Evaluate/forward_velocity", 3, _format_f),
            ("Evaluate/mean_episode_length (G4 ≥ 475)", "Evaluate/mean_episode_length", 1, _format_f),
            ("Evaluate/cmd_vs_achieved_forward (G4 ≤ 0.075)", "Evaluate/cmd_vs_achieved_forward", 3, _format_f),
            ("tracking/step_length_touchdown_event_m (G4 ≥ 0.030)", "tracking/step_length_touchdown_event_m", 4, _format_f),
            ("tracking/forward_velocity_cmd_ratio (G5 0.6..1.5)", "tracking/forward_velocity_cmd_ratio", 3, _format_f),
            ("tracking/residual_hip_pitch_left_abs (G5 ≤ 0.20)", "tracking/residual_hip_pitch_left_abs", 3, _format_f),
            ("tracking/residual_hip_pitch_right_abs (G5 ≤ 0.20)", "tracking/residual_hip_pitch_right_abs", 3, _format_f),
            ("tracking/residual_knee_left_abs (G5 ≤ 0.20)", "tracking/residual_knee_left_abs", 3, _format_f),
            ("tracking/residual_knee_right_abs (G5 ≤ 0.20)", "tracking/residual_knee_right_abs", 3, _format_f),
        ]
        for label, key, nd, fmt in v0201_g4_rows:
            v = g(key)
            if v is not None:
                lines.append(f"| {label} | {fmt(v, nd)} |")

        # v0.20.1 imitation-dominant reward family — only print if the
        # ToddlerBot-aligned terms appear in this row.  Surfaces dead
        # gradients (large ref/<X>_err_* paired with near-zero
        # reward/ref_<X>_track) at a glance.
        v0201_reward_rows = [
            ("reward/ref_q_track (w=5.0)", "reward/ref_q_track"),
            ("ref/q_track_err_rmse", "ref/q_track_err_rmse"),
            ("reward/ref_body_quat_track (w=5.0)", "reward/ref_body_quat_track"),
            ("ref/body_quat_err_deg", "ref/body_quat_err_deg"),
            ("reward/torso_pos_xy (w=2.0, α=200)", "reward/torso_pos_xy"),
            ("ref/torso_pos_xy_err_m", "ref/torso_pos_xy_err_m"),
            ("reward/ref_feet_pos_track (legacy)", "reward/ref_feet_pos_track"),
            ("ref/feet_pos_track_raw (debug-only, α=200)", "ref/feet_pos_track_raw"),
            ("ref/feet_pos_err_l2 (root-relative L2 m)", "ref/feet_pos_err_l2"),
            ("reward/ref_contact_match (w=0 gated)", "reward/ref_contact_match"),
            ("ref/contact_phase_match", "ref/contact_phase_match"),
            ("reward/cmd_forward_velocity_track (w=5.0, α=200)", "reward/cmd_forward_velocity_track"),
            ("reward/alive (w=10, -done)", "reward/alive"),
            ("reward/feet_air_time (w=500)", "reward/feet_air_time"),
            ("reward/feet_clearance (w=1.0)", "reward/feet_clearance"),
            ("reward/feet_distance (w=1.0)", "reward/feet_distance"),
            ("reward/torso_pitch_soft (w=0.5)", "reward/torso_pitch_soft"),
            ("reward/torso_roll_soft (w=0.5)", "reward/torso_roll_soft"),
            ("reward/slip (w=0.05)", "reward/slip"),
            ("reward/action_rate (w=-1.0)", "reward/action_rate"),
        ]
        for label, key in v0201_reward_rows:
            v = g(key)
            if v is not None:
                lines.append(f"| {label} | {_format_f(v, 4)} |")

        # Per-foot stride / swing-time diagnostics (v0.20.1-smoke2).
        per_foot_rows = [
            ("tracking/touchdown_rate_left", "tracking/touchdown_rate_left"),
            ("tracking/touchdown_rate_right", "tracking/touchdown_rate_right"),
            ("tracking/swing_air_time_left_event_s (mean over rollout)", "tracking/swing_air_time_left_event_s"),
            ("tracking/swing_air_time_right_event_s (mean over rollout)", "tracking/swing_air_time_right_event_s"),
            ("tracking/step_length_left_event_m (mean over rollout)", "tracking/step_length_left_event_m"),
            ("tracking/step_length_right_event_m (mean over rollout)", "tracking/step_length_right_event_m"),
        ]
        for label, key in per_foot_rows:
            v = g(key)
            if v is not None:
                lines.append(f"| {label} | {_format_f(v, 5)} |")
    if regression is not None:
        lines += [
            "",
            "### Regression vs previous comparable run",
            f"- Previous run: `{regression.previous.run_dir.as_posix()}`",
            f"- Common-metric comparison iters: current={regression.current_it}, previous={regression.previous_it}",
            f"- Overall: {regression.overall}",
            "",
            "| Regression signal | Previous | Current | Delta | Status | Gate |",
            "|---|---:|---:|---:|---|---|",
        ]
        for metric in regression.metrics:
            lines.append(
                "| {label} | {previous} | {current} | {delta} | {status} | {gate} |".format(
                    label=metric.label,
                    previous=_format_f(metric.previous, metric.digits),
                    current=_format_f(metric.current, metric.digits),
                    delta=_format_delta(metric.delta, metric.digits),
                    status=metric.status,
                    gate=_format_gate_transition(metric.previous_gate, metric.current_gate),
                )
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--previous-run-dir", type=Path, default=None)
    ap.add_argument("--wandb-root", type=Path, default=Path("training/wandb"))
    ap.add_argument("--checkpoints-root", type=Path, default=Path("training/checkpoints"))
    ap.add_argument("--min-iter", type=int, default=10)
    args = ap.parse_args()

    current = _load_run_analysis(
        args.run_dir,
        checkpoints_root=args.checkpoints_root,
        min_iter=args.min_iter,
    )
    previous_run_dir = args.previous_run_dir
    if previous_run_dir is None:
        previous_run_dir = _find_previous_run_dir(current, wandb_root=args.wandb_root)

    regression = None
    if previous_run_dir is not None:
        previous = _load_run_analysis(
            previous_run_dir,
            checkpoints_root=args.checkpoints_root,
            min_iter=args.min_iter,
        )
        regression = _build_regression_summary(current, previous)

    # Aggregate stability stats (post-warmup iters)
    term_low = _mean(_collect_series(current.rows, "term_height_low_frac", min_iter=args.min_iter))
    term_pitch = _mean(_collect_series(current.rows, "term_pitch_frac", min_iter=args.min_iter))
    term_roll = _mean(_collect_series(current.rows, "term_roll_frac", min_iter=args.min_iter))
    torque_sat = _mean(_collect_series(current.rows, "debug/torque_sat_frac", min_iter=args.min_iter))
    max_torque = _first_present(
        _mean(_collect_series(current.rows, "env/max_torque", min_iter=args.min_iter)),
        _mean(_collect_series(current.rows, "tracking/max_torque", min_iter=args.min_iter)),
    )

    print(f"Run: {current.run_dir}")
    print(f"Run ID: {current.run_id}")
    print(f"Version: {current.version}")
    print(f"Eval keyset: {current.keys.success}, {current.keys.ep_len}")
    if current.ckpt_dir:
        print(f"Checkpoint dir: {current.ckpt_dir}")
    if current.ckpt_file:
        print(f"Best checkpoint: {current.ckpt_file}")
    print(f"Best iter: {current.best_it}")
    print()
    print("Stability summary (iters >= %d):" % args.min_iter)
    print(f"- mean term_height_low_frac: {_format_pct(term_low)}")
    print(f"- mean term_pitch_frac: {_format_pct(term_pitch)}")
    print(f"- mean term_roll_frac: {_format_pct(term_roll)}")
    print(f"- mean tracking/max_torque: {_format_pct(max_torque)}")
    print(f"- mean debug/torque_sat_frac: {_format_pct(torque_sat)}")
    if current.walking.enabled:
        print()
        print("Walking summary:")
        print(f"- verdict: {current.walking.verdict}")
        print(f"- tracking: {current.walking.tracking_status}")
        print(f"- stability: {current.walking.stability}")
        # Labels reflect the actual key path used by ``_summarize_walking``.
        print(
            f"- mean forward_velocity (Evaluate or env): {_format_f(current.walking.forward_vel_mean, 3)}"
        )
        print(f"- mean env/velocity_cmd: {_format_f(current.walking.velocity_cmd_mean, 3)}")
        print(
            "- mean cmd_vs_achieved_forward (Evaluate or tracking): "
            f"{_format_f(current.walking.velocity_error_mean, 3)}"
        )
    if regression is not None:
        print()
        print("Regression checklist vs previous comparable run:")
        print(f"- previous run: {regression.previous.run_dir}")
        print(
            f"- common-metric comparison iters: current={regression.current_it}, "
            f"previous={regression.previous_it}"
        )
        print(f"- overall: {regression.overall}")
        for metric in regression.metrics:
            print(
                "- {label}: {current} vs {previous} ({delta}); {status}; gate {gate}".format(
                    label=metric.label,
                    current=_format_f(metric.current, metric.digits),
                    previous=_format_f(metric.previous, metric.digits),
                    delta=_format_delta(metric.delta, metric.digits),
                    status=metric.status,
                    gate=_format_gate_transition(metric.previous_gate, metric.current_gate),
                )
            )
    print()
    print("CHANGELOG block (paste into training/CHANGELOG.md):")
    print()
    print(
        _changelog_block(
            version=current.version,
            run_dir=current.run_dir,
            ckpt_dir=current.ckpt_dir,
            best_ckpt=current.ckpt_file,
            keys=current.keys,
            best_it=current.best_it,
            best_row=current.best_row,
            walking=current.walking,
            regression=regression,
        )
    )


if __name__ == "__main__":
    main()
