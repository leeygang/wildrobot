#!/usr/bin/env python3
"""Run independent deterministic walking evaluations for one checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.core.post_training_eval import deterministic_eval_gate


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def _gate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Translate eval_policy output names to deterministic gate names."""
    mapped = dict(metrics)
    aliases = {
        "mean_episode_length": "episode_length",
        "cmd_vs_achieved_forward": "tracking/cmd_vs_achieved_forward",
        "step_length_touchdown_event_m": "tracking/step_length_touchdown_event_m",
    }
    for target, source in aliases.items():
        if target not in mapped and source in metrics:
            mapped[target] = metrics[source]
    return mapped


def _aggregate_results(
    seed_results: list[dict[str, Any]], *, num_envs: int
) -> dict[str, Any]:
    total_envs = int(num_envs) * len(seed_results)
    total_falls = int(
        sum(
            float(result["eval_metrics"].get("walking_fall_env_count", 0.0))
            for result in seed_results
        )
    )
    fail_reasons = sorted(
        {
            reason
            for result in seed_results
            for reason in result.get("fail_reasons", [])
        }
    )

    def worst(name: str) -> float | None:
        values = [
            float(result["eval_metrics"][name])
            for result in seed_results
            if result["eval_metrics"].get(name) is not None
        ]
        return max(values) if values else None

    return {
        "passed": bool(seed_results) and all(result["passed"] for result in seed_results),
        "total_envs": total_envs,
        "total_falls": total_falls,
        "fall_free": total_falls == 0,
        "zero_failure_probability_upper_95": (
            1.0 - math.pow(0.05, 1.0 / total_envs)
            if total_envs > 0 and total_falls == 0
            else None
        ),
        "fail_reasons": fail_reasons,
        "worst_stable_tilt_deg": worst("walking_stable_body_tilt_deg_max"),
        "worst_survivor_final_tilt_deg": worst(
            "walking_survivor_final_body_tilt_deg_max"
        ),
        "worst_stable_actuator_torque_sat_frac": worst(
            "walking_stable_max_actuator_torque_sat_frac"
        ),
        "worst_pre_fall_tilt_deg": worst("walking_pre_fall_body_tilt_deg_max"),
        "worst_pre_fall_actuator_torque_sat_frac": worst(
            "walking_pre_fall_max_actuator_torque_sat_frac"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("confirmation", "failure_diagnostic"), required=True)
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num-envs and num-steps must be positive")
    config_payload = yaml.safe_load(args.config.read_text())
    eval_velocity_cmd = float(config_payload["env"]["eval_velocity_cmd"][0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        metrics_path = args.output.parent / f"seed_{seed}.json"
        trace_path = args.output.parent / f"seed_{seed}_failure_trace.npz"
        command = [
            sys.executable,
            "training/eval/eval_policy.py",
            "--checkpoint",
            str(args.checkpoint),
            "--config",
            str(args.config),
            "--num-envs",
            str(args.num_envs),
            "--num-steps",
            str(args.num_steps),
            "--seed",
            str(seed),
            "--no-push",
            "--output",
            str(metrics_path),
            "--failure-trace-output",
            str(trace_path),
        ]
        print(f"Running walking {args.purpose} seed {seed}...", flush=True)
        completed = subprocess.run(command, cwd=_REPO_ROOT, check=False)
        if completed.returncode:
            return int(completed.returncode)
        raw_metrics = json.loads(metrics_path.read_text())
        eval_metrics = _gate_metrics(raw_metrics)
        decision = deterministic_eval_gate(
            eval_metrics=eval_metrics,
            eval_velocity_cmd=eval_velocity_cmd,
            eval_num_steps=args.num_steps,
            strict_lateral_drift=False,
            strict_walking_safety=True,
        )
        seed_results.append(
            {
                "seed": seed,
                "passed": bool(decision.passed),
                "gates": dict(decision.gates),
                "fail_reasons": [
                    name for name, passed in decision.gates.items() if not passed
                ],
                "eval_metrics": eval_metrics,
                "metrics_path": str(metrics_path),
                "failure_trace_path": str(trace_path),
            }
        )

    summary = {
        "schema_version": 1,
        "purpose": args.purpose,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seeds": args.seeds,
        "num_envs_per_seed": args.num_envs,
        "num_steps": args.num_steps,
        "aggregate": _aggregate_results(seed_results, num_envs=args.num_envs),
        "seed_results": seed_results,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    aggregate = summary["aggregate"]
    print(
        f"Walking {args.purpose}: passed={aggregate['passed']} "
        f"falls={aggregate['total_falls']}/{aggregate['total_envs']} "
        f"fail_reasons={aggregate['fail_reasons']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
