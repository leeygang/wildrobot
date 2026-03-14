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
class FsmSummary:
    enabled: bool
    phase_mean: Optional[float]
    phase_ticks_mean: Optional[float]
    swing_occupancy: Optional[float]
    step_event_mean: Optional[float]
    foot_place_mean: Optional[float]
    touchdown_mean: Optional[float]
    need_step_mean: Optional[float]
    posture_mean: Optional[float]
    verdict: str
    touchdown_style: str
    upright_recovery: str


@dataclass(frozen=True)
class WalkingSummary:
    enabled: bool
    forward_vel_mean: Optional[float]
    velocity_cmd_mean: Optional[float]
    velocity_error_mean: Optional[float]
    gait_periodicity_mean: Optional[float]
    clearance_mean: Optional[float]
    hip_swing_mean: Optional[float]
    knee_swing_mean: Optional[float]
    verdict: str
    tracking_status: str
    stability: str


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


def _pick_eval_keys(rows: List[Dict[str, Any]]) -> EvalKeySet:
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


def _is_fsm_run(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    env = cfg.get("config", {}).get("env", {})
    if bool(env.get("fsm_enabled", False)):
        return True
    return any("debug/bc_phase" in r for r in rows)


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
        "env/velocity_error",
        "reward/gait_periodicity",
    )
    return any(any(k in r for k in walking_keys) for r in rows)


def _classify_fsm(
    rows: List[Dict[str, Any]],
    *,
    min_iter: int,
    cfg: Dict[str, Any],
) -> FsmSummary:
    if not _is_fsm_run(cfg, rows):
        return FsmSummary(
            enabled=False,
            phase_mean=None,
            phase_ticks_mean=None,
            swing_occupancy=None,
            step_event_mean=None,
            foot_place_mean=None,
            touchdown_mean=None,
            need_step_mean=None,
            posture_mean=None,
            verdict="not_applicable",
            touchdown_style="not_applicable",
            upright_recovery="not_applicable",
        )

    phase_vals = _collect_series(rows, "debug/bc_phase", min_iter=min_iter)
    in_swing_vals = _collect_series(rows, "debug/bc_in_swing", min_iter=min_iter)
    phase_ticks_vals = _collect_series(rows, "debug/bc_phase_ticks", min_iter=min_iter)
    step_event_vals = _collect_series(rows, "reward/step_event", min_iter=min_iter)
    foot_place_vals = _collect_series(rows, "reward/foot_place", min_iter=min_iter)
    touchdown_left = _collect_series(rows, "debug/touchdown_left", min_iter=min_iter)
    touchdown_right = _collect_series(rows, "debug/touchdown_right", min_iter=min_iter)
    need_step_vals = _collect_series(rows, "debug/need_step", min_iter=min_iter)
    posture_vals = _collect_series(rows, "reward/posture", min_iter=min_iter)

    phase_mean = _mean(phase_vals)
    phase_ticks_mean = _mean(phase_ticks_vals)
    step_event_mean = _mean(step_event_vals)
    foot_place_mean = _mean(foot_place_vals)
    touchdown_mean = _mean((touchdown_left or []) + (touchdown_right or []))
    need_step_mean = _mean(need_step_vals)
    posture_mean = _mean(posture_vals)
    swing_occupancy = _mean(in_swing_vals) if in_swing_vals else None
    if swing_occupancy is None and phase_vals:
        swing_occupancy = _mean([max(0.0, min(1.0, v)) for v in phase_vals])

    if swing_occupancy is None or swing_occupancy < 0.02:
        verdict = "not meaningfully engaged"
    elif step_event_mean is None or step_event_mean < 0.01:
        verdict = "weakly engaged"
    else:
        verdict = "engaged"

    timeout_ticks = float(cfg.get("config", {}).get("env", {}).get("fsm_swing_timeout_ticks", 12))
    if step_event_mean is None or touchdown_mean is None:
        touchdown_style = "unknown"
    elif step_event_mean >= 0.01 and touchdown_mean >= 0.005:
        touchdown_style = "touchdown-driven"
    elif phase_ticks_mean is not None and phase_ticks_mean > 0.8 * timeout_ticks:
        touchdown_style = "timeout-driven"
    else:
        touchdown_style = "mixed"

    if posture_mean is None:
        upright_recovery = "unknown"
    elif posture_mean >= 0.10:
        upright_recovery = "good"
    elif posture_mean >= 0.03:
        upright_recovery = "partial"
    else:
        upright_recovery = "poor"

    return FsmSummary(
        enabled=True,
        phase_mean=phase_mean,
        phase_ticks_mean=phase_ticks_mean,
        swing_occupancy=swing_occupancy,
        step_event_mean=step_event_mean,
        foot_place_mean=foot_place_mean,
        touchdown_mean=touchdown_mean,
        need_step_mean=need_step_mean,
        posture_mean=posture_mean,
        verdict=verdict,
        touchdown_style=touchdown_style,
        upright_recovery=upright_recovery,
    )


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
            gait_periodicity_mean=None,
            clearance_mean=None,
            hip_swing_mean=None,
            knee_swing_mean=None,
            verdict="not_applicable",
            tracking_status="not_applicable",
            stability="not_applicable",
        )

    forward_vel_mean = _mean(_collect_series(rows, "env/forward_velocity", min_iter=min_iter))
    velocity_cmd_mean = _mean(_collect_series(rows, "env/velocity_cmd", min_iter=min_iter))
    velocity_error_mean = _mean(_collect_series(rows, "env/velocity_error", min_iter=min_iter))
    gait_periodicity_mean = _mean(_collect_series(rows, "reward/gait_periodicity", min_iter=min_iter))
    clearance_mean = _mean(_collect_series(rows, "reward/clearance", min_iter=min_iter))
    hip_swing_mean = _mean(_collect_series(rows, "reward/hip_swing", min_iter=min_iter))
    knee_swing_mean = _mean(_collect_series(rows, "reward/knee_swing", min_iter=min_iter))
    success_mean = _mean(_collect_series(rows, "env/success_rate", min_iter=min_iter))
    term_low_mean = _mean(_collect_series(rows, "term_height_low_frac", min_iter=min_iter))
    term_pitch_mean = _mean(_collect_series(rows, "term_pitch_frac", min_iter=min_iter))
    torque_sat_mean = _mean(_collect_series(rows, "debug/torque_sat_frac", min_iter=min_iter))

    fwd = forward_vel_mean or 0.0
    vel_err = velocity_error_mean if velocity_error_mean is not None else float("inf")
    success = success_mean or 0.0
    term_pitch = term_pitch_mean or 0.0
    torque_sat = torque_sat_mean or 0.0

    if fwd < 0.05 and term_pitch >= 0.10:
        verdict = "posture exploit"
    elif fwd < 0.05 and success >= 0.80:
        verdict = "trapped in standing"
    elif fwd < 0.10 and torque_sat >= 0.03:
        verdict = "shuffle exploit"
    elif fwd >= 0.20 and vel_err <= 0.20:
        verdict = "locomotion emerging"
    else:
        verdict = "needs review"

    if fwd >= 0.20 and vel_err <= 0.20:
        tracking_status = "improving"
    elif fwd < 0.10 and vel_err >= 0.20:
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
        gait_periodicity_mean=gait_periodicity_mean,
        clearance_mean=clearance_mean,
        hip_swing_mean=hip_swing_mean,
        knee_swing_mean=knee_swing_mean,
        verdict=verdict,
        tracking_status=tracking_status,
        stability=stability,
    )


def _changelog_block(
    *,
    version: str,
    run_dir: Path,
    ckpt_dir: Optional[Path],
    best_ckpt: Optional[Path],
    keys: EvalKeySet,
    best_it: int,
    best_row: Dict[str, Any],
    fsm: FsmSummary,
    walking: WalkingSummary,
) -> str:
    def g(key: str) -> Optional[float]:
        return _get_float(best_row, key)

    eval_s = g(keys.success)
    eval_L = g(keys.ep_len)
    train_s = g("env/success_rate") or g("success_rate")
    train_L = g("env/episode_length") or g("episode_length")

    term_low = g("term_height_low_frac")
    term_pitch = g("term_pitch_frac")
    term_roll = g("term_roll_frac")
    max_torque = g("env/max_torque") or g("tracking/max_torque")
    torque_sat = g("debug/torque_sat_frac")
    approx_kl = g("ppo/approx_kl") or g("approx_kl")
    clip_frac = g("ppo/clip_fraction") or g("clip_fraction")

    lines = [
        f"### Results (v{version})",
        f"- Run: `{run_dir.as_posix()}`",
    ]
    if ckpt_dir is not None:
        lines.append(f"- Checkpoints: `{ckpt_dir.as_posix()}`")
    if best_ckpt is not None:
        lines.append(f"- Best checkpoint ({keys.prefix or 'metric'}): `{best_ckpt.as_posix()}`")
    lines += [
        f"- Best @ iter {best_it}: {keys.success}={_format_pct(eval_s)}, {keys.ep_len}={_format_f(eval_L, 1)}",
        f"- Train @ iter {best_it}: success={_format_pct(train_s)}, ep_len={_format_f(train_L, 1)}",
    ]
    if walking.enabled:
        lines += [
            f"- Walking verdict: {walking.verdict}",
            f"- Walking tracking: {walking.tracking_status}",
            f"- Walking stability: {walking.stability}",
        ]
    if fsm.enabled:
        lines += [
            f"- FSM verdict: {fsm.verdict}",
            f"- FSM touchdown style: {fsm.touchdown_style}",
            f"- FSM upright recovery: {fsm.upright_recovery}",
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
        lines += [
            f"| env/forward_velocity | {_format_f(walking.forward_vel_mean, 3)} |",
            f"| env/velocity_cmd | {_format_f(walking.velocity_cmd_mean, 3)} |",
            f"| env/velocity_error | {_format_f(walking.velocity_error_mean, 3)} |",
            f"| reward/gait_periodicity | {_format_f(walking.gait_periodicity_mean, 4)} |",
            f"| reward/clearance | {_format_f(walking.clearance_mean, 4)} |",
            f"| reward/hip_swing | {_format_f(walking.hip_swing_mean, 4)} |",
            f"| reward/knee_swing | {_format_f(walking.knee_swing_mean, 4)} |",
        ]
    if fsm.enabled:
        lines += [
            f"| debug/bc_phase | {_format_f(fsm.phase_mean, 3)} |",
            f"| debug/bc_phase_ticks | {_format_f(fsm.phase_ticks_mean, 3)} |",
            f"| fsm/swing_occupancy | {_format_pct(fsm.swing_occupancy)} |",
            f"| reward/step_event | {_format_f(fsm.step_event_mean, 4)} |",
            f"| reward/foot_place | {_format_f(fsm.foot_place_mean, 4)} |",
            f"| debug/need_step | {_format_f(fsm.need_step_mean, 4)} |",
            f"| reward/posture | {_format_f(fsm.posture_mean, 4)} |",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoints-root", type=Path, default=Path("training/checkpoints"))
    ap.add_argument("--min-iter", type=int, default=10)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
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

    rows = _read_jsonl(metrics_path)
    rows = [r for r in rows if _get_iter(r) is not None]
    rows.sort(key=lambda r: _get_iter(r) or -1)

    keys = _pick_eval_keys(rows)
    best_it, best_row = _find_best_row(rows, keys)
    best_step = None
    for k in ("time/step", "_step"):
        if k in best_row:
            try:
                best_step = int(best_row[k])
                break
            except Exception:
                best_step = None

    ckpt_dir = _find_checkpoint_dir(run_id, args.checkpoints_root)
    ckpt_file = _find_checkpoint_file(ckpt_dir, best_it, best_step) if ckpt_dir else None
    fsm = _classify_fsm(rows, min_iter=args.min_iter, cfg=cfg)
    walking = _classify_walking(rows, min_iter=args.min_iter, cfg=cfg)

    # Aggregate stability stats (post-warmup iters)
    term_low = _mean(_collect_series(rows, "term_height_low_frac", min_iter=args.min_iter))
    term_pitch = _mean(_collect_series(rows, "term_pitch_frac", min_iter=args.min_iter))
    term_roll = _mean(_collect_series(rows, "term_roll_frac", min_iter=args.min_iter))
    torque_sat = _mean(_collect_series(rows, "debug/torque_sat_frac", min_iter=args.min_iter))
    max_torque = _mean(_collect_series(rows, "env/max_torque", min_iter=args.min_iter)) or _mean(
        _collect_series(rows, "tracking/max_torque", min_iter=args.min_iter)
    )

    print(f"Run: {run_dir}")
    print(f"Run ID: {run_id}")
    print(f"Version: {version}")
    print(f"Eval keyset: {keys.success}, {keys.ep_len}")
    if ckpt_dir:
        print(f"Checkpoint dir: {ckpt_dir}")
    if ckpt_file:
        print(f"Best checkpoint: {ckpt_file}")
    print(f"Best iter: {best_it}")
    print()
    print("Stability summary (iters >= %d):" % args.min_iter)
    print(f"- mean term_height_low_frac: {_format_pct(term_low)}")
    print(f"- mean term_pitch_frac: {_format_pct(term_pitch)}")
    print(f"- mean term_roll_frac: {_format_pct(term_roll)}")
    print(f"- mean tracking/max_torque: {_format_pct(max_torque)}")
    print(f"- mean debug/torque_sat_frac: {_format_pct(torque_sat)}")
    if walking.enabled:
        print()
        print("Walking summary:")
        print(f"- verdict: {walking.verdict}")
        print(f"- tracking: {walking.tracking_status}")
        print(f"- stability: {walking.stability}")
        print(f"- mean env/forward_velocity: {_format_f(walking.forward_vel_mean, 3)}")
        print(f"- mean env/velocity_cmd: {_format_f(walking.velocity_cmd_mean, 3)}")
        print(f"- mean env/velocity_error: {_format_f(walking.velocity_error_mean, 3)}")
        print(f"- mean reward/gait_periodicity: {_format_f(walking.gait_periodicity_mean, 4)}")
        print(f"- mean reward/clearance: {_format_f(walking.clearance_mean, 4)}")
        print(f"- mean reward/hip_swing: {_format_f(walking.hip_swing_mean, 4)}")
        print(f"- mean reward/knee_swing: {_format_f(walking.knee_swing_mean, 4)}")
    if fsm.enabled:
        print()
        print("FSM / M3 summary:")
        print(f"- verdict: {fsm.verdict}")
        print(f"- touchdown style: {fsm.touchdown_style}")
        print(f"- upright recovery: {fsm.upright_recovery}")
        print(f"- mean debug/bc_phase: {_format_f(fsm.phase_mean, 3)}")
        print(f"- mean debug/bc_phase_ticks: {_format_f(fsm.phase_ticks_mean, 3)}")
        print(f"- mean swing occupancy: {_format_pct(fsm.swing_occupancy)}")
        print(f"- mean reward/step_event: {_format_f(fsm.step_event_mean, 4)}")
        print(f"- mean reward/foot_place: {_format_f(fsm.foot_place_mean, 4)}")
        print(f"- mean debug/need_step: {_format_f(fsm.need_step_mean, 4)}")
        print(f"- mean reward/posture: {_format_f(fsm.posture_mean, 4)}")
    print()
    print("CHANGELOG block (paste into training/CHANGELOG.md):")
    print()
    print(_changelog_block(
        version=version,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        best_ckpt=ckpt_file,
        keys=keys,
        best_it=best_it,
        best_row=best_row,
        fsm=fsm,
        walking=walking,
    ))


if __name__ == "__main__":
    main()
