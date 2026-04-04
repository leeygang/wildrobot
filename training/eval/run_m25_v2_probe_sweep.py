#!/usr/bin/env python3
"""Run the M2.5 nominal-only probe sweep and summarize results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.eval.eval_loc_ref_probe import run_nominal_probe


DEFAULT_SPEEDS = (0.06, 0.10, 0.14)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M2.5 walking_ref_v2 probe sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking_v0193a.yaml",
        help="Training config to load.",
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPEEDS),
        help="Forward command speeds to sweep.",
    )
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/m25_v2_probe_sweep",
        help="Directory for per-speed JSON/trace artifacts and aggregate summary.",
    )
    parser.add_argument(
        "--trace-format",
        type=str,
        choices=("json", "npz", "none"),
        default="json",
        help="Format for per-speed trace outputs.",
    )
    parser.add_argument(
        "--stop-on-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop each probe when any env reaches done.",
    )
    return parser.parse_args()


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _probe_status(summary: dict[str, Any]) -> str:
    done_reached = bool(summary.get("done_reached", False))
    done_step = int(summary.get("done_step", 0))
    term = str(summary.get("dominant_termination", "none"))
    fwd_mean = float(summary.get("forward_velocity_mean", 0.0))
    cmd = float(summary.get("forward_cmd", 0.0))
    swing_reward = float(summary.get("reward_m3_swing_foot_tracking", 0.0))
    foothold = float(summary.get("reward_m3_foothold_consistency", 0.0))

    if done_reached and term == "term/pitch" and done_step < 60:
        return "pitch-fail"
    if not done_reached and cmd > 0.0 and fwd_mean < 0.5 * cmd:
        return "too-conservative"
    if not done_reached and swing_reward > 1e-4 and foothold > 0.05:
        return "promising"
    return "unclear"


def _gate_assessment(results: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if not results:
        return ["No results collected."]

    pitch_failures = [
        r for r in results
        if bool(r.get("done_reached", False)) and str(r.get("dominant_termination")) == "term/pitch"
    ]
    if pitch_failures:
        notes.append("Nominal-only still reaches pitch termination in at least one sweep point.")
    else:
        notes.append("No pitch termination observed in the sampled sweep horizon.")

    overspeed_bad = [
        r for r in results
        if float(r.get("forward_cmd", 0.0)) > 0.0
        and float(r.get("forward_velocity_mean", 0.0)) > 1.5 * float(r.get("forward_cmd", 0.0))
    ]
    if overspeed_bad:
        notes.append("At least one sweep point still overspeeds materially relative to command.")
    else:
        notes.append("Mean forward speed stays within a bounded multiple of command across the sweep.")

    swing_weak = all(float(r.get("reward_m3_swing_foot_tracking", 0.0)) < 1e-5 for r in results)
    foothold_weak = all(float(r.get("reward_m3_foothold_consistency", 0.0)) < 0.05 for r in results)
    if swing_weak:
        notes.append("Swing-foot tracking remains near numerical noise across the sweep.")
    if foothold_weak:
        notes.append("Foothold consistency remains weak across the sweep.")

    if not pitch_failures and not overspeed_bad and not swing_weak and not foothold_weak:
        notes.append("The M2.5 resume-PPO gate looks plausible, but should still be checked against trace-level behavior.")
    else:
        notes.append("The M2.5 resume-PPO gate is not met yet.")
    return notes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for speed in args.speeds:
        speed_tag = f"{speed:.2f}".replace(".", "")
        json_path = output_dir / f"probe_{speed_tag}.json"
        trace_path: str | None
        if args.trace_format == "none":
            trace_path = None
        else:
            trace_suffix = ".json" if args.trace_format == "json" else ".npz"
            trace_path = str(output_dir / f"probe_{speed_tag}_trace{trace_suffix}")

        summary = run_nominal_probe(
            config_path=args.config,
            forward_cmd=float(speed),
            horizon=int(args.horizon),
            seed=int(args.seed),
            residual_scale_override=float(args.residual_scale),
            num_envs=int(args.num_envs),
            trace_output=trace_path,
            stop_on_done=bool(args.stop_on_done),
        )
        _write_json(json_path, summary)
        results.append(summary)

    aggregate = {
        "config": args.config,
        "horizon": int(args.horizon),
        "num_envs": int(args.num_envs),
        "residual_scale": float(args.residual_scale),
        "speeds": [float(s) for s in args.speeds],
        "results": results,
        "analysis": _gate_assessment(results),
    }
    _write_json(output_dir / "summary.json", aggregate)

    print("M2.5 nominal-only sweep")
    header = (
        "cmd", "done", "step", "term", "fwd_mean", "fwd_last",
        "swing", "foothold", "support", "phase_active", "swing_x_active", "q_gap", "status",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for result in results:
        row = (
            _format_float(result["forward_cmd"], 2),
            str(bool(result["done_reached"])).lower(),
            str(int(result["done_step"])),
            str(result["dominant_termination"]),
            _format_float(result["forward_velocity_mean"]),
            _format_float(result["forward_velocity_last"]),
            _format_float(result["reward_m3_swing_foot_tracking"], 6),
            _format_float(result["reward_m3_foothold_consistency"]),
            _format_float(result["debug_loc_ref_support_health_mean"]),
            _format_float(result["debug_loc_ref_phase_scale_active_mean"]),
            _format_float(result["debug_loc_ref_swing_x_scale_active_mean"]),
            _format_float(result["debug_loc_ref_nominal_vs_applied_q_l1_mean"]),
            _probe_status(result),
        )
        print(" | ".join(row))

    print("\nAssessment")
    for note in aggregate["analysis"]:
        print(f"- {note}")

    print(f"\nArtifacts: {output_dir}")


if __name__ == "__main__":
    main()
