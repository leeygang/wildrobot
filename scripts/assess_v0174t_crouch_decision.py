#!/usr/bin/env python3
"""Decision helper for v0.17.4t-crouch screening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _load_metrics(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _get(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


def _has_visible_crouch_step(metrics: Dict[str, float], threshold: float) -> bool:
    return _get(metrics, "eval_push/recovery_visible_step_rate") >= threshold


def decide_branch(
    *,
    metrics: Dict[str, float],
    visible_step_threshold: float,
    clean_unnecessary_step_max: float,
) -> str:
    eval_hard = _get(metrics, "eval_push/success_rate")
    has_mechanism = _has_visible_crouch_step(metrics, visible_step_threshold)
    clean_step_rate = _get(metrics, "eval_clean/unnecessary_step_rate")
    clean_gate_ok = clean_step_rate <= clean_unnecessary_step_max

    if eval_hard >= 0.70 and has_mechanism and clean_gate_ok:
        return "stay_rl"
    if eval_hard >= 0.65:
        return "go_crouch_teacher"
    return "restore_baseline_then_crouch_teacher"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assess v0.17.4t-crouch screening decision from metrics JSON.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Metrics JSON (e.g., eval/training metrics export).",
    )
    parser.add_argument(
        "--visible-step-threshold",
        type=float,
        default=0.10,
        help="Minimum eval_push/recovery_visible_step_rate considered visible crouch+step.",
    )
    parser.add_argument(
        "--clean-unnecessary-step-max",
        type=float,
        default=0.10,
        help="Maximum allowed eval_clean/unnecessary_step_rate for stay_rl.",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    metrics = _load_metrics(args.metrics)
    decision = decide_branch(
        metrics=metrics,
        visible_step_threshold=float(args.visible_step_threshold),
        clean_unnecessary_step_max=float(args.clean_unnecessary_step_max),
    )

    summary = {
        "decision": decision,
        "eval_hard": _get(metrics, "eval_push/success_rate"),
        "eval_clean/unnecessary_step_rate": _get(
            metrics, "eval_clean/unnecessary_step_rate"
        ),
        "eval_clean/unnecessary_step_gate_ok": _get(
            metrics, "eval_clean/unnecessary_step_rate"
        )
        <= float(args.clean_unnecessary_step_max),
        "recovery/visible_step_rate": _get(
            metrics, "eval_push/recovery_visible_step_rate"
        ),
        "recovery/no_touchdown_frac": _get(
            metrics, "eval_push/recovery_no_touchdown_frac"
        ),
        "recovery/touchdown_then_fail_frac": _get(
            metrics, "eval_push/recovery_touchdown_then_fail_frac"
        ),
        "recovery/pitch_rate_reduction_10t": _get(
            metrics, "eval_push/recovery_pitch_rate_reduction_10t"
        ),
        "recovery/min_height": _get(metrics, "eval_push/recovery_min_height"),
        "recovery/max_knee_flex": _get(metrics, "eval_push/recovery_max_knee_flex"),
        "recovery/first_step_dist_abs": _get(
            metrics, "eval_push/recovery_first_step_dist_abs"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
