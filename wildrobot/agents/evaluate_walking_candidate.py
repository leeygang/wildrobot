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
from training.policy_migration.contact_free import SOURCE_LAYOUT_ID
from training.scripts.distill_contact_observed_to_proprio import (
    DEFAULT_TEACHER_CHECKPOINT,
    _resolve_teacher_checkpoint,
)


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


def _source_failure_indices(report_path: Path) -> dict[int, set[int]]:
    report = json.loads(report_path.read_text())
    return {
        int(result["seed"]): {
            int(case["env_index"])
            for case in result.get("eval_metrics", {}).get("failure_cases", [])
        }
        for result in report.get("seed_results", [])
    }


def _teacher_eval_config(config_path: Path, output_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("env"), dict):
        raise ValueError(f"training config has no env mapping: {config_path}")
    payload["env"]["actor_obs_layout_id"] = SOURCE_LAYOUT_ID
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return output_path


def _teacher_recoverability_aggregate(
    aggregate: dict[str, Any],
    seed_results: list[dict[str, Any]],
    source_failure_indices: dict[int, set[int]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    source_failures = {
        int(seed): sorted(source_failure_indices.get(int(seed), set()))
        for seed in seeds
    }
    teacher_failures = {
        int(result["seed"]): sorted(
            int(case["env_index"])
            for case in result["eval_metrics"].get("failure_cases", [])
        )
        for result in seed_results
    }
    unrecovered = {
        int(seed): sorted(
            set(source_failures[int(seed)])
            & set(teacher_failures.get(int(seed), []))
        )
        for seed in seeds
    }
    source_failure_count = sum(len(values) for values in source_failures.values())
    passed = bool(source_failure_count > 0 and aggregate["total_falls"] == 0)
    return {
        **aggregate,
        "deployment_gates_passed": aggregate["passed"],
        "source_failure_env_count": source_failure_count,
        "source_failure_env_indices": source_failures,
        "teacher_failure_env_indices": teacher_failures,
        "unrecovered_source_failure_env_indices": unrecovered,
        "teacher_recoverability_passed": passed,
        "passed": passed,
        "fail_reasons": [] if passed else ["teacher_walking_fall_env_frac"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--source-evaluation-report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=(
            "confirmation",
            "failure_diagnostic",
            "teacher_recoverability",
        ),
        required=True,
    )
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num-envs and num-steps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.purpose == "teacher_recoverability":
        if args.source_evaluation_report is None:
            raise ValueError(
                "teacher_recoverability requires --source-evaluation-report"
            )
        checkpoint = _resolve_teacher_checkpoint(
            args.teacher_checkpoint or DEFAULT_TEACHER_CHECKPOINT
        )
        config = _teacher_eval_config(
            args.config,
            args.output.parent / "teacher_recoverability_config.yaml",
        )
        source_failure_indices = _source_failure_indices(
            args.source_evaluation_report
        )
    else:
        if args.checkpoint is None:
            raise ValueError(f"{args.purpose} requires --checkpoint")
        checkpoint = args.checkpoint
        config = args.config
        source_failure_indices = {}
    config_payload = yaml.safe_load(args.config.read_text())
    eval_velocity_cmd = float(config_payload["env"]["eval_velocity_cmd"][0])

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        metrics_path = args.output.parent / f"seed_{seed}.json"
        trace_path = args.output.parent / f"seed_{seed}_failure_trace.npz"
        command = [
            sys.executable,
            "training/eval/eval_policy.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
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

    aggregate = _aggregate_results(seed_results, num_envs=args.num_envs)
    if args.purpose == "teacher_recoverability":
        aggregate = _teacher_recoverability_aggregate(
            aggregate,
            seed_results,
            source_failure_indices,
            args.seeds,
        )

    summary = {
        "schema_version": 1,
        "purpose": args.purpose,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "source_evaluation_report": (
            str(args.source_evaluation_report)
            if args.source_evaluation_report is not None
            else None
        ),
        "seeds": args.seeds,
        "num_envs_per_seed": args.num_envs,
        "num_steps": args.num_steps,
        "aggregate": aggregate,
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
