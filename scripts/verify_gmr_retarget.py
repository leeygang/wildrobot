#!/usr/bin/env python3
"""Verify a GMR retargeted WildRobot motion before teacher/export use."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.debug_gmr_physics import run_physics_test
from training.reference_motion.verify import compute_gmr_kinematic_verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        type=str,
        required=True,
        help="Path to GMR motion pickle.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/v2/scene_flat_terrain.xml",
        help="Path to MuJoCo scene XML.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="training/reference_motion/verification/gmr_motion_report.json",
        help="Path to write the JSON verification report.",
    )
    return parser.parse_args()


def _load_motion_metadata(motion_path: Path) -> dict[str, Any]:
    with motion_path.open("rb") as f:
        motion = pickle.load(f)
    return {
        "source_file": motion.get("source_file", "unknown"),
        "robot": motion.get("robot", "unknown"),
        "fps": float(motion.get("fps", 0.0)),
        "num_frames": int(motion.get("num_frames", len(motion.get("dof_pos", [])))),
        "duration_sec": float(motion.get("duration_sec", 0.0)),
    }


def verify_gmr_retarget(args: argparse.Namespace) -> dict[str, Any]:
    motion_path = Path(args.motion)
    meta = _load_motion_metadata(motion_path)
    kinematic = compute_gmr_kinematic_verification(
        {
            **pickle.loads(motion_path.read_bytes()),
        },
        model_path=args.model,
    )
    full = run_physics_test(
        motion_path=str(motion_path),
        model_path=args.model,
        harness_mode="full",
        harness_cap=0.15,
        loop=False,
        z_offset=0.0,
        render=False,
        return_summary=True,
    )
    capped = run_physics_test(
        motion_path=str(motion_path),
        model_path=args.model,
        harness_mode="capped",
        harness_cap=0.15,
        loop=False,
        z_offset=0.0,
        render=False,
        return_summary=True,
    )

    checks = {
        "kinematic_grounding": bool(
            abs(kinematic["lowest_support_z_m"]["median"]) <= 0.01
            and kinematic["lowest_support_z_m"]["p95"] <= 0.02
            and kinematic["lowest_support_z_m"]["min"] >= -0.01
        ),
        "kinematic_posture": bool(
            kinematic["max_abs_pitch_deg"] <= 20.0 and kinematic["max_abs_roll_deg"] <= 10.0
        ),
        "kinematic_knees": bool(
            kinematic["left_knee_deg"]["min"] >= -1.0
            and kinematic["right_knee_deg"]["min"] >= -1.0
            and kinematic["left_knee_deg"]["max"] <= 81.0
            and kinematic["right_knee_deg"]["max"] <= 81.0
            and kinematic["left_knee_deg"]["mean"] >= 5.0
            and kinematic["right_knee_deg"]["mean"] >= 5.0
        ),
        "full_contact_support": bool(
            full["load_support_mean_ratio"] >= 0.80 and full["max_pitch_deg"] <= 20.0
        ),
        "capped_dynamic_tracking": bool(
            capped["frame_completion_rate"] >= 0.90
            and capped["harness_p95_ratio"] <= 0.15
            and capped["max_pitch_deg"] <= 25.0
            and capped["max_roll_deg"] <= 25.0
        ),
    }

    notes: list[str] = []
    if meta["robot"] == "wildrobot":
        notes.append(
            "Source motion is already a WildRobot GMR output; this gate validates whether that retarget is physically usable in MuJoCo."
        )
    if not checks["kinematic_grounding"]:
        notes.append(
            "Kinematic replay does not keep the support foot close enough to the floor. Fix foot-ground alignment before teacher export."
        )
    if not checks["full_contact_support"]:
        notes.append(
            "Full kinematic replay does not generate enough ground reaction support; the motion likely floats or under-contacts the floor."
        )
    if not checks["capped_dynamic_tracking"]:
        notes.append(
            "Physics-assisted replay is not stable enough; even if the pose is plausible, the motion is not dynamically trackable yet."
        )

    report = {
        "motion": str(motion_path),
        "metadata": meta,
        "kinematic": kinematic,
        "physics_full": full,
        "physics_capped": capped,
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "verdict": "pass" if all(checks.values()) else "fail",
        "notes": notes,
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("GMR retarget verification summary")
    print(f"  motion: {report['motion']}")
    print(f"  verdict: {report['verdict']}")
    if report["failed_checks"]:
        print(f"  failed_checks: {', '.join(report['failed_checks'])}")
    print(
        "  kinematic lowest support z:"
        f" median={report['kinematic']['lowest_support_z_m']['median']:.4f} m"
        f" p95={report['kinematic']['lowest_support_z_m']['p95']:.4f} m"
        f" min={report['kinematic']['lowest_support_z_m']['min']:.4f} m"
    )
    print(
        "  full replay load support:"
        f" mean={report['physics_full']['load_support_mean_ratio']*100:.0f}% mg"
        f" max pitch={report['physics_full']['max_pitch_deg']:.1f}°"
    )
    print(
        "  capped replay:"
        f" completion={report['physics_capped']['frame_completion_rate']*100:.0f}%"
        f" p95 harness={report['physics_capped']['harness_p95_ratio']*100:.0f}% mg"
        f" max pitch={report['physics_capped']['max_pitch_deg']:.1f}°"
        f" max roll={report['physics_capped']['max_roll_deg']:.1f}°"
    )
    if report["notes"]:
        print("  notes:")
        for note in report["notes"]:
            print(f"    - {note}")


def main() -> None:
    args = parse_args()
    report = verify_gmr_retarget(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
