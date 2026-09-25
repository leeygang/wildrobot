#!/usr/bin/env python3
"""Run a fixed-policy A/B of the current and narrower canonical stance poses.

The experiment changes no hardware dimensions, mass, actuator model, gait
timing, checkpoint, command, or seed.  Each pose uses a matching reference
half-width and a ToddlerBot-normalized close-feet reward threshold.  The
threshold affects evaluation reward bookkeeping only; it does not feed the
fixed actor or alter the physics.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_training_config
from training.eval.sweep_actuator_capacity import (
    _positive_finite_values,
    _slug,
    _summarize_case,
)
from training.eval.verify_walking_stance_geometry import (
    OFFSET_SIGNS,
    analyze_stance_candidate,
    load_stance_inputs,
)


DEFAULT_NARROW_HOME_ROLL_OFFSET_RAD = 0.0635
DEFAULT_FORWARD_COMMANDS_M_S = (0.066667, 0.133333)
TB_NOMINAL_FOOT_SEPARATION_M = 0.074
TB_CLOSE_FEET_THRESHOLD_M = 0.060


def _configured_symmetric_home_roll_offset(env_cfg: Any) -> float:
    offsets = dict(getattr(env_cfg, "home_joint_offsets_rad", {}) or {})
    magnitudes = [
        float(offsets.get(joint_name, 0.0)) / sign
        for joint_name, sign in OFFSET_SIGNS.items()
    ]
    if not all(math.isfinite(value) and value >= 0.0 for value in magnitudes):
        raise ValueError("configured symmetric home-roll offsets must be non-negative")
    if max(magnitudes) - min(magnitudes) > 1e-6:
        raise ValueError(
            "pose A/B requires symmetric hip/ankle-roll home offsets; "
            f"resolved magnitudes={magnitudes}"
        )
    return float(sum(magnitudes) / len(magnitudes))


def _pose_scaled_close_feet_threshold(foot_separation_m: float) -> float:
    """Preserve ToddlerBot's 60/74 close-feet fraction for one stance."""
    return (
        float(foot_separation_m)
        * TB_CLOSE_FEET_THRESHOLD_M
        / TB_NOMINAL_FOOT_SEPARATION_M
    )


def _geometry_cases(
    config: Path,
    *,
    baseline_home_roll_offset_rad: float,
    narrow_home_roll_offset_rad: float,
) -> list[dict[str, Any]]:
    (
        model,
        robot_config,
        home_qpos,
        home_foot_rotations,
        configured_close_feet_threshold_m,
    ) = load_stance_inputs(config)
    cases = []
    for label, total_offset in (
        ("baseline", baseline_home_roll_offset_rad),
        ("narrow", narrow_home_roll_offset_rad),
    ):
        additional_offset = total_offset - baseline_home_roll_offset_rad
        if additional_offset < -1e-9:
            raise ValueError(
                "pose A/B currently supports narrowing from the configured "
                "baseline, not widening"
            )
        geometry = analyze_stance_candidate(
            model=model,
            robot_config=robot_config,
            home_qpos=home_qpos,
            home_foot_rotations=home_foot_rotations,
            offset_rad=additional_offset,
            close_feet_threshold_m=configured_close_feet_threshold_m,
            max_support_torque_ratio=0.8,
            max_foot_orientation_delta_deg=1.0,
            max_sole_height_delta_m=0.002,
        )
        physical_gates = {
            name: passed
            for name, passed in geometry["gates"].items()
            if name != "foot_separation"
        }
        if not all(physical_gates.values()):
            failures = [name for name, passed in physical_gates.items() if not passed]
            raise RuntimeError(
                f"{label} pose failed static physical gates: {', '.join(failures)}"
            )
        separation_m = float(geometry["foot_center_separation_m"])
        cases.append(
            {
                "label": label,
                "home_roll_offset_rad": float(total_offset),
                "additional_roll_offset_rad": float(additional_offset),
                "reference_stance_half_width_m": separation_m / 2.0,
                "pose_scaled_close_feet_threshold_m": (
                    _pose_scaled_close_feet_threshold(separation_m)
                ),
                "configured_close_feet_threshold_m": float(
                    configured_close_feet_threshold_m
                ),
                "physical_gates": physical_gates,
                "geometry": geometry,
            }
        )
    return cases


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def _summarize_pose_eval(
    metrics: Mapping[str, Any],
    *,
    pose: Mapping[str, Any],
    torque_limit_nm: float,
    velocity_cmd_m_s: float,
    num_steps: int,
    metrics_path: Path,
) -> dict[str, Any]:
    summary = _summarize_case(
        metrics,
        torque_limit_nm=torque_limit_nm,
        velocity_cmd_m_s=velocity_cmd_m_s,
        num_steps=num_steps,
        metrics_path=metrics_path,
    )
    summary.update(
        {
            "pose": str(pose["label"]),
            "home_roll_offset_rad": float(pose["home_roll_offset_rad"]),
            "reference_stance_half_width_m": float(
                pose["reference_stance_half_width_m"]
            ),
            "pose_scaled_close_feet_threshold_m": float(
                pose["pose_scaled_close_feet_threshold_m"]
            ),
            "observed_feet_lateral_distance_m": _metric(
                metrics, "support/feet_lateral_distance_m"
            ),
            "lateral_velocity_abs_m_s": _metric(
                metrics, "tracking/lateral_velocity_abs"
            ),
            "world_y_drift_signed_m": _metric(
                metrics, "tracking/world_y_drift_signed_m"
            ),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current canonical walking stance with a pose-only "
            "narrower stance using one fixed checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--narrow-home-roll-offset-rad",
        type=float,
        default=DEFAULT_NARROW_HOME_ROLL_OFFSET_RAD,
        help="Absolute symmetric roll correction for the narrow arm.",
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

    velocity_cmds = _positive_finite_values(
        args.velocity_cmds_m_s, "velocity-cmds-m-s"
    )
    training_cfg = load_training_config(config)
    if str(training_cfg.env.loc_ref_residual_base) != "home":
        raise ValueError("pose A/B requires loc_ref_residual_base=home")
    baseline_offset = _configured_symmetric_home_roll_offset(training_cfg.env)
    narrow_offset = float(args.narrow_home_roll_offset_rad)
    if not math.isfinite(narrow_offset) or narrow_offset <= baseline_offset:
        raise ValueError(
            "narrow-home-roll-offset-rad must be finite and greater than the "
            f"configured baseline {baseline_offset:.6f} rad"
        )
    torque_limit_nm = float(training_cfg.env.actuator_force_limit_nm)
    poses = _geometry_cases(
        config,
        baseline_home_roll_offset_rad=baseline_offset,
        narrow_home_roll_offset_rad=narrow_offset,
    )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "_eval" / f"stance_pose_ab_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fixed-policy canonical stance-pose A/B", flush=True)
    print(f"  checkpoint={checkpoint}", flush=True)
    print(f"  config={config}", flush=True)
    print(f"  torque_limit_nm={torque_limit_nm:.6f}", flush=True)
    print(f"  velocity_commands_m_s={list(velocity_cmds)}", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    for pose in poses:
        geometry = pose["geometry"]
        max_support_ratio = max(
            float(side["quasi_static_support_ratio"])
            for side in geometry["support"].values()
        )
        print(
            f"  {pose['label']}: roll={pose['home_roll_offset_rad']:.4f}rad "
            f"feet={geometry['foot_center_separation_m']:.4f}m "
            f"inner_clearance={geometry['inner_foot_clearance_m']:.4f}m "
            f"static_support_ratio={max_support_ratio:.3f} "
            f"close_threshold={pose['pose_scaled_close_feet_threshold_m']:.4f}m",
            flush=True,
        )

    cases: list[dict[str, Any]] = []
    total_cases = len(poses) * len(velocity_cmds)
    case_index = 0
    for pose in poses:
        for velocity_cmd_m_s in velocity_cmds:
            case_index += 1
            metrics_path = output_dir / (
                f"{pose['label']}_vx_{_slug(velocity_cmd_m_s)}.json"
            )
            print(
                f"\n[{case_index}/{total_cases}] pose={pose['label']} "
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
                "--home-roll-offset-rad",
                str(pose["home_roll_offset_rad"]),
                "--stance-width-m",
                str(pose["reference_stance_half_width_m"]),
                "--close-feet-threshold-m",
                str(pose["pose_scaled_close_feet_threshold_m"]),
                "--actuator-force-limit-nm",
                str(torque_limit_nm),
                "--no-push",
                "--output",
                str(metrics_path),
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            reported_offset = metrics.get("eval/home_roll_offset_rad")
            if reported_offset is None or not math.isclose(
                float(reported_offset),
                float(pose["home_roll_offset_rad"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "evaluator did not report the requested home-roll offset: "
                    f"requested={pose['home_roll_offset_rad']}, "
                    f"reported={reported_offset}"
                )
            cases.append(
                _summarize_pose_eval(
                    metrics,
                    pose=pose,
                    torque_limit_nm=torque_limit_nm,
                    velocity_cmd_m_s=velocity_cmd_m_s,
                    num_steps=args.num_steps,
                    metrics_path=metrics_path,
                )
            )

    summary = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "torque_limit_nm": torque_limit_nm,
        "velocity_cmds_m_s": list(velocity_cmds),
        "fixed_variables": [
            "checkpoint",
            "robot mass and inertias",
            "actuator model and force limit",
            "gait cycle time",
            "forward commands",
            "seed",
            "pushes disabled",
        ],
        "pose_scaled_close_feet_note": (
            "The close-feet threshold preserves ToddlerBot's 60/74 stance "
            "fraction. It changes reward bookkeeping only during this fixed-"
            "policy evaluation and does not affect actions or physics."
        ),
        "poses": poses,
        "cases": cases,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nPose A/B summary", flush=True)
    print(
        "pose     | vx m/s | falls | peak tilt | forward | lateral | "
        "worst stable saturation",
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
        lateral = case["lateral_velocity_abs_m_s"]
        print(
            f"{case['pose']:<8s} | "
            f"{case['velocity_cmd_m_s']:6.3f} | "
            f"{int(case['fall_env_count']):2d}/{args.num_envs:<2d} | "
            f"{float(case['stable_tilt_max_deg']):9.2f} | "
            f"{float(case['forward_velocity_m_s']):7.3f} | "
            f"{('n/a' if lateral is None else f'{lateral:.3f}'):>7s} | "
            f"{worst_text}",
            flush=True,
        )
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
