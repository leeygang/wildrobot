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


def _changelog_block(
    *,
    version: str,
    run_dir: Path,
    ckpt_dir: Optional[Path],
    best_ckpt: Optional[Path],
    keys: EvalKeySet,
    best_it: int,
    best_row: Dict[str, Any],
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
    ))


if __name__ == "__main__":
    main()

