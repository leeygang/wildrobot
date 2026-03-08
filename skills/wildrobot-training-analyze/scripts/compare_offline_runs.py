#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    run_id: str
    version: str
    best_iter: int
    best_success: float
    best_ep_len: float
    keyset: str
    best_checkpoint: Optional[Path]
    fsm_enabled: bool
    fsm_verdict: str
    fsm_touchdown_style: str
    fsm_upright_recovery: str
    fsm_swing_occupancy: Optional[float]
    fsm_step_event_mean: Optional[float]
    fsm_foot_place_mean: Optional[float]


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
        return float(row[key])
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _collect_series(rows: List[Dict[str, Any]], key: str, *, min_iter: int = 10) -> List[float]:
    out: List[float] = []
    for r in rows:
        it = _get_iter(r)
        if it is None or it < min_iter:
            continue
        v = _get_float(r, key)
        if v is not None:
            out.append(v)
    return out


def _classify_fsm(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> Tuple[bool, str, str, str, Optional[float], Optional[float], Optional[float]]:
    env = cfg.get("config", {}).get("env", {})
    enabled = bool(env.get("fsm_enabled", False)) or any("debug/bc_phase" in r for r in rows)
    if not enabled:
        return False, "not_applicable", "not_applicable", "not_applicable", None, None, None

    phase_vals = _collect_series(rows, "debug/bc_phase")
    phase_ticks_vals = _collect_series(rows, "debug/bc_phase_ticks")
    step_event_vals = _collect_series(rows, "reward/step_event")
    foot_place_vals = _collect_series(rows, "reward/foot_place")
    touchdown_vals = _collect_series(rows, "debug/touchdown_left") + _collect_series(rows, "debug/touchdown_right")
    posture_vals = _collect_series(rows, "reward/posture")

    swing_occupancy = _mean([1.0 if abs(v - 1.0) < 1e-6 else 0.0 for v in phase_vals]) if phase_vals else None
    step_event_mean = _mean(step_event_vals)
    foot_place_mean = _mean(foot_place_vals)
    touchdown_mean = _mean(touchdown_vals)
    posture_mean = _mean(posture_vals)
    phase_ticks_mean = _mean(phase_ticks_vals)
    timeout_ticks = float(env.get("fsm_swing_timeout_ticks", 12))

    if swing_occupancy is None or swing_occupancy < 0.02:
        verdict = "not meaningfully engaged"
    elif step_event_mean is None or step_event_mean < 0.01:
        verdict = "weakly engaged"
    else:
        verdict = "engaged"

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

    return enabled, verdict, touchdown_style, upright_recovery, swing_occupancy, step_event_mean, foot_place_mean


def _pick_keys(rows: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    if any("eval_push/success_rate" in r for r in rows):
        return ("eval_push", "eval_push/success_rate", "eval_push/episode_length")
    if any("eval/success_rate" in r for r in rows):
        return ("eval", "eval/success_rate", "eval/episode_length")
    if any("env/success_rate" in r for r in rows):
        return ("env", "env/success_rate", "env/episode_length")
    return ("train", "success_rate", "episode_length")


def _find_best(rows: List[Dict[str, Any]], success_k: str, len_k: str) -> Tuple[int, float, float, Dict[str, Any]]:
    best_it = -1
    best_s = float("-inf")
    best_L = float("-inf")
    best_row: Dict[str, Any] = {}
    for r in rows:
        it = _get_iter(r)
        if it is None or it < 0:
            continue
        s = _get_float(r, success_k)
        L = _get_float(r, len_k)
        if s is None or L is None:
            continue
        if s > best_s or (s == best_s and L > best_L) or (s == best_s and L == best_L and it > best_it):
            best_it, best_s, best_L, best_row = it, s, L, r
    if best_it < 0:
        raise SystemExit(f"No rows contained {success_k} and {len_k}.")
    return best_it, best_s, best_L, best_row


def _find_ckpt_dir(run_id: str, root: Path) -> Optional[Path]:
    dirs = [p for p in root.glob(f"*{run_id}*") if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_ckpt_file(ckpt_dir: Path, it: int, step: Optional[int]) -> Optional[Path]:
    if step is not None:
        exact = ckpt_dir / f"checkpoint_{it}_{step}.pkl"
        if exact.exists():
            return exact
    matches = sorted(ckpt_dir.glob(f"checkpoint_{it}_*.pkl"))
    return matches[0] if matches else None


def summarize(run_dir: Path, checkpoints_root: Path) -> RunSummary:
    run_id = run_dir.name.split("-")[-1]
    cfg = _read_json(run_dir / "files" / "config.json")
    version = str(cfg.get("config", {}).get("version", "unknown"))
    rows = _read_jsonl(run_dir / "files" / "metrics.jsonl")
    rows = [r for r in rows if _get_iter(r) is not None]
    rows.sort(key=lambda r: _get_iter(r) or -1)

    keyset, s_k, l_k = _pick_keys(rows)
    it, s, L, row = _find_best(rows, s_k, l_k)
    step = None
    for k in ("time/step", "_step"):
        if k in row:
            try:
                step = int(row[k])
                break
            except Exception:
                step = None

    ckpt_dir = _find_ckpt_dir(run_id, checkpoints_root)
    ckpt_file = _find_ckpt_file(ckpt_dir, it, step) if ckpt_dir else None
    (
        fsm_enabled,
        fsm_verdict,
        fsm_touchdown_style,
        fsm_upright_recovery,
        fsm_swing_occupancy,
        fsm_step_event_mean,
        fsm_foot_place_mean,
    ) = _classify_fsm(cfg, rows)

    return RunSummary(
        run_dir=run_dir,
        run_id=run_id,
        version=version,
        best_iter=it,
        best_success=s,
        best_ep_len=L,
        keyset=keyset,
        best_checkpoint=ckpt_file,
        fsm_enabled=fsm_enabled,
        fsm_verdict=fsm_verdict,
        fsm_touchdown_style=fsm_touchdown_style,
        fsm_upright_recovery=fsm_upright_recovery,
        fsm_swing_occupancy=fsm_swing_occupancy,
        fsm_step_event_mean=fsm_step_event_mean,
        fsm_foot_place_mean=fsm_foot_place_mean,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir-a", type=Path, required=True)
    ap.add_argument("--run-dir-b", type=Path, required=True)
    ap.add_argument("--checkpoints-root", type=Path, default=Path("training/checkpoints"))
    args = ap.parse_args()

    a = summarize(args.run_dir_a, args.checkpoints_root)
    b = summarize(args.run_dir_b, args.checkpoints_root)

    def fmt(r: RunSummary) -> str:
        ck = r.best_checkpoint.as_posix() if r.best_checkpoint else "N/A"
        base = (
            f"{r.version} ({r.run_id}) | keyset={r.keyset} | "
            f"best: it={r.best_iter} succ={r.best_success*100:.2f}% len={r.best_ep_len:.1f} | "
            f"ckpt={ck}"
        )
        if r.fsm_enabled:
            base += (
                f" | fsm={r.fsm_verdict}"
                f" | touchdown={r.fsm_touchdown_style}"
                f" | upright={r.fsm_upright_recovery}"
            )
            if r.fsm_swing_occupancy is not None:
                base += f" | swing_occ={r.fsm_swing_occupancy*100:.2f}%"
        return base

    print("A:", fmt(a))
    print("B:", fmt(b))
    print()

    if (b.best_success, b.best_ep_len) > (a.best_success, a.best_ep_len):
        winner = b
        loser = a
    else:
        winner = a
        loser = b

    print("Recommended deploy candidate:")
    print(f"- Winner: v{winner.version} ({winner.run_id})")
    if winner.best_checkpoint:
        print(f"- Checkpoint: {winner.best_checkpoint}")
    print()
    print("Delta vs other run:")
    print(f"- success: {(winner.best_success - loser.best_success)*100:+.2f}%")
    print(f"- ep_len: {winner.best_ep_len - loser.best_ep_len:+.1f}")
    if winner.fsm_enabled or loser.fsm_enabled:
        print(f"- fsm verdict: {winner.fsm_verdict} vs {loser.fsm_verdict}")
        print(f"- touchdown style: {winner.fsm_touchdown_style} vs {loser.fsm_touchdown_style}")
        print(f"- upright recovery: {winner.fsm_upright_recovery} vs {loser.fsm_upright_recovery}")
        if winner.fsm_step_event_mean is not None and loser.fsm_step_event_mean is not None:
            print(f"- reward/step_event: {winner.fsm_step_event_mean - loser.fsm_step_event_mean:+.4f}")
        if winner.fsm_foot_place_mean is not None and loser.fsm_foot_place_mean is not None:
            print(f"- reward/foot_place: {winner.fsm_foot_place_mean - loser.fsm_foot_place_mean:+.4f}")


if __name__ == "__main__":
    main()
