#!/usr/bin/env python3
"""Sweep scalar actuator force capacity for one deterministic walking policy.

This is a sensitivity test of the existing MuJoCo force limit. It does not
claim that an HTD-45H can sustain the tested torque, and it does not replace a
measured torque-speed or thermal envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_training_config
from training.core.post_training_eval import (
    deterministic_eval_gate,
    walking_safety_gates,
)


DEFAULT_TORQUE_LIMITS_NM = (4.4129925, 4.8, 5.2, 5.6, 6.0)
DEFAULT_FORWARD_COMMANDS_M_S = (0.066667, 0.133333)
TRACKED_LOAD_JOINTS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
)
TRACKED_LOAD_PAIRS = (
    "hip_pitch",
    "hip_roll",
    "knee_pitch",
    "ankle_pitch",
    "ankle_roll",
)


def _positive_finite_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    parsed = tuple(sorted({float(value) for value in values}))
    if not parsed or any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(f"{label} must contain finite positive values")
    return parsed


def _normalize_eval_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Add the aliases expected by the shared deterministic gate."""
    normalized = dict(metrics)
    aliases = {
        "mean_episode_length": "episode_length",
        "cmd_vs_achieved_forward": "tracking/cmd_vs_achieved_forward",
        "step_length_touchdown_event_m": (
            "tracking/step_length_touchdown_event_m"
        ),
        "lateral_velocity_abs": "tracking/lateral_velocity_abs",
        "yaw_drift_signed_rad": "tracking/yaw_drift_signed_rad",
        "world_y_drift_signed_m": "tracking/world_y_drift_signed_m",
    }
    for target, source in aliases.items():
        if target not in normalized and source in normalized:
            normalized[target] = normalized[source]
    return normalized


def _worst_stable_actuator(
    metrics: Mapping[str, Any],
) -> tuple[str | None, float | None]:
    prefix = "walking_stable_torque/"
    suffix = "/sat_frac"
    candidates = {
        key[len(prefix) : -len(suffix)]: float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and key.endswith(suffix)
    }
    if not candidates:
        return None, None
    name = max(candidates, key=candidates.get)
    return name, candidates[name]


def _stable_saturation(metrics: Mapping[str, Any]) -> dict[str, float]:
    prefix = "walking_stable_torque/"
    suffix = "/sat_frac"
    return {
        key[len(prefix) : -len(suffix)]: float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and key.endswith(suffix)
    }


def _tracked_rms(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        name: (
            None
            if metrics.get(f"walking_stable_torque/{name}/rms_nm") is None
            else float(metrics[f"walking_stable_torque/{name}/rms_nm"])
        )
        for name in TRACKED_LOAD_JOINTS
    }


def _pair_rms_summary(
    joint_rms_nm: Mapping[str, float | None], suffix: str
) -> dict[str, float | None]:
    left = joint_rms_nm.get(f"left_{suffix}")
    right = joint_rms_nm.get(f"right_{suffix}")
    if left is None or right is None:
        return {"sum_nm": None, "relative_imbalance": None}
    larger = max(left, right)
    return {
        "sum_nm": left + right,
        "relative_imbalance": abs(left - right) / larger if larger > 1e-9 else 0.0,
    }


def _summarize_case(
    metrics: Mapping[str, Any],
    *,
    torque_limit_nm: float,
    velocity_cmd_m_s: float,
    num_steps: int,
    metrics_path: Path,
) -> dict[str, Any]:
    normalized = _normalize_eval_metrics(metrics)
    safety_gates = walking_safety_gates(normalized)
    full_decision = deterministic_eval_gate(
        normalized,
        velocity_cmd_m_s,
        eval_num_steps=num_steps,
        strict_walking_safety=True,
    )
    worst_name, worst_sat = _worst_stable_actuator(normalized)
    joint_rms = _tracked_rms(normalized)
    return {
        "torque_limit_nm": torque_limit_nm,
        "velocity_cmd_m_s": velocity_cmd_m_s,
        "metrics_path": str(metrics_path),
        "safety_passed": all(safety_gates.values()),
        "safety_gates": safety_gates,
        "full_gate_passed": bool(full_decision.passed),
        "full_gate_failures": [
            name for name, passed in full_decision.gates.items() if not passed
        ],
        "fall_env_count": float(normalized.get("walking_fall_env_count", 0.0)),
        "fall_env_frac": float(normalized.get("walking_fall_env_frac", 0.0)),
        "stable_tilt_mean_deg": normalized.get(
            "walking_stable_body_tilt_deg_mean"
        ),
        "stable_tilt_max_deg": normalized.get(
            "walking_stable_body_tilt_deg_max"
        ),
        "survivor_final_tilt_max_deg": normalized.get(
            "walking_survivor_final_body_tilt_deg_max"
        ),
        "worst_stable_saturation_joint": worst_name,
        "worst_stable_saturation_frac": worst_sat,
        "stable_actuator_saturation_frac": _stable_saturation(normalized),
        "forward_velocity_m_s": normalized.get("forward_velocity"),
        "command_error_m_s": normalized.get("cmd_vs_achieved_forward"),
        "touchdown_step_length_m": normalized.get(
            "step_length_touchdown_event_m"
        ),
        "stable_joint_torque_rms_nm": joint_rms,
        "stable_bilateral_torque_rms": {
            suffix: _pair_rms_summary(joint_rms, suffix)
            for suffix in TRACKED_LOAD_PAIRS
        },
    }


def _minimum_limit(
    cases: Sequence[Mapping[str, Any]],
    velocity_cmds_m_s: Sequence[float],
    pass_key: str,
) -> float | None:
    commands = set(velocity_cmds_m_s)
    for limit in sorted({float(case["torque_limit_nm"]) for case in cases}):
        matching = [
            case for case in cases if float(case["torque_limit_nm"]) == limit
        ]
        if (
            {float(case["velocity_cmd_m_s"]) for case in matching} == commands
            and all(bool(case[pass_key]) for case in matching)
        ):
            return limit
    return None


def _slug(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one home-base walking checkpoint across scalar actuator "
            "force limits and forward commands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--torque-limits-nm",
        nargs="+",
        type=float,
        default=DEFAULT_TORQUE_LIMITS_NM,
    )
    parser.add_argument(
        "--velocity-cmds-m-s",
        nargs="+",
        type=float,
        default=DEFAULT_FORWARD_COMMANDS_M_S,
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("num-envs and num-steps must be positive")

    torque_limits = _positive_finite_values(
        args.torque_limits_nm, "torque-limits-nm"
    )
    velocity_cmds = _positive_finite_values(
        args.velocity_cmds_m_s, "velocity-cmds-m-s"
    )
    training_cfg = load_training_config(config)
    residual_base = str(training_cfg.env.loc_ref_residual_base)
    if residual_base != "home":
        raise ValueError(
            "actuator-capacity sweep requires loc_ref_residual_base=home; "
            f"got {residual_base!r}"
        )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "_eval" / f"actuator_capacity_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Home-base scalar actuator-capacity sweep", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  config={config}", flush=True)
    print(f"  limits_nm={list(torque_limits)}", flush=True)
    print(f"  forward_commands_m_s={list(velocity_cmds)}", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    print(
        "  interpretation=MuJoCo scalar force-limit sensitivity only; "
        "not an HTD-45H continuous or torque-speed rating",
        flush=True,
    )

    cases: list[dict[str, Any]] = []
    total_cases = len(torque_limits) * len(velocity_cmds)
    case_index = 0
    for torque_limit_nm in torque_limits:
        for velocity_cmd_m_s in velocity_cmds:
            case_index += 1
            metrics_path = output_dir / (
                f"limit_{_slug(torque_limit_nm)}nm_"
                f"vx_{_slug(velocity_cmd_m_s)}.json"
            )
            print(
                f"\n[{case_index}/{total_cases}] limit={torque_limit_nm:.4f}Nm "
                f"vx={velocity_cmd_m_s:.6f}m/s",
                flush=True,
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "training/eval/eval_policy.py"),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
                "--num-envs",
                str(args.num_envs),
                "--num-steps",
                str(args.num_steps),
                "--seed",
                str(args.seed),
                "--velocity-cmd",
                str(velocity_cmd_m_s),
                "0",
                "0",
                "--actuator-force-limit-nm",
                str(torque_limit_nm),
                "--no-push",
                "--output",
                str(metrics_path),
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            reported_limit = metrics.get("eval/actuator_force_limit_nm")
            if reported_limit is None or not math.isclose(
                float(reported_limit), torque_limit_nm, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RuntimeError(
                    "evaluator did not report the requested actuator force limit: "
                    f"requested={torque_limit_nm}, reported={reported_limit}"
                )
            cases.append(
                _summarize_case(
                    metrics,
                    torque_limit_nm=torque_limit_nm,
                    velocity_cmd_m_s=velocity_cmd_m_s,
                    num_steps=args.num_steps,
                    metrics_path=metrics_path,
                )
            )

    summary = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "residual_base": residual_base,
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "torque_limits_nm": list(torque_limits),
        "velocity_cmds_m_s": list(velocity_cmds),
        "model_scope": (
            "scalar MuJoCo actuator-force sensitivity; excludes measured "
            "torque-speed, thermal derating, and continuous-duty qualification"
        ),
        "minimum_all_command_safety_limit_nm": _minimum_limit(
            cases, velocity_cmds, "safety_passed"
        ),
        "minimum_all_command_full_gate_limit_nm": _minimum_limit(
            cases, velocity_cmds, "full_gate_passed"
        ),
        "cases": cases,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nCapacity sweep summary", flush=True)
    print(
        "limit Nm | vx m/s | falls | peak tilt | worst stable saturation | safety",
        flush=True,
    )
    for case in cases:
        worst_sat = case["worst_stable_saturation_frac"]
        worst_text = (
            "n/a"
            if worst_sat is None
            else (
                f"{100.0 * float(worst_sat):5.1f}% "
                f"{case['worst_stable_saturation_joint']}"
            )
        )
        print(
            f"{case['torque_limit_nm']:8.4f} | "
            f"{case['velocity_cmd_m_s']:6.3f} | "
            f"{int(case['fall_env_count']):2d}/{args.num_envs:<2d} | "
            f"{float(case['stable_tilt_max_deg']):9.2f} | "
            f"{worst_text:>28s} | "
            f"{'PASS' if case['safety_passed'] else 'FAIL'}",
            flush=True,
        )
    print(
        "minimum all-command safety limit: "
        f"{summary['minimum_all_command_safety_limit_nm']}",
        flush=True,
    )
    print(
        "minimum all-command full-gate limit: "
        f"{summary['minimum_all_command_full_gate_limit_nm']}",
        flush=True,
    )
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
