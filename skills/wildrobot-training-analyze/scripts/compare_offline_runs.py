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

    return RunSummary(
        run_dir=run_dir,
        run_id=run_id,
        version=version,
        best_iter=it,
        best_success=s,
        best_ep_len=L,
        keyset=keyset,
        best_checkpoint=ckpt_file,
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
        return (
            f"{r.version} ({r.run_id}) | keyset={r.keyset} | "
            f"best: it={r.best_iter} succ={r.best_success*100:.2f}% len={r.best_ep_len:.1f} | "
            f"ckpt={ck}"
        )

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


if __name__ == "__main__":
    main()

