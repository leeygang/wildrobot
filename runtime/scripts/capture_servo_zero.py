#!/usr/bin/env python3
"""Capture servo zero offsets from a physically aligned calibration fixture.

This tool never commands servo motion.  The operator must support the robot and
align every selected joint to its MuJoCo-zero reference before confirming the
capture.  Multiple raw-position samples are used to reject unstable readings.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from configs.config import ServoConfig, WrRuntimeConfig  # noqa: E402
from runtime.scripts.calibrate import (  # noqa: E402
    build_calibration_controller,
    offset_from_reference_pose_units,
    read_servo_positions_best_effort,
    write_config,
)


LEG_PITCH_JOINTS = (
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
)


@dataclass(frozen=True)
class JointZeroCapture:
    joint: str
    servo_id: int
    samples_raw_units: tuple[int, ...]
    median_raw_units: int
    spread_units: int
    stable: bool
    current_offset_unit: int
    suggested_offset_unit: int
    offset_change_unit: int
    current_rad_at_physical_zero: float
    current_deg_at_physical_zero: float
    implied_physical_target_error_rad: float
    implied_physical_target_error_deg: float


def resolve_joint_selection(
    selector: str,
    available_joints: Iterable[str],
) -> list[str]:
    available = list(available_joints)
    available_set = set(available)
    normalized = str(selector).strip().lower().replace("-", "_")
    if normalized in {"leg_pitch", "pitch", "standing"}:
        selected = list(LEG_PITCH_JOINTS)
    elif normalized == "all":
        selected = available
    else:
        selected = [
            token.strip()
            for token in normalized.replace(",", " ").split()
            if token.strip()
        ]
    if not selected:
        raise ValueError("No joints selected")
    missing = [joint for joint in selected if joint not in available_set]
    if missing:
        raise ValueError(f"Selected joints are missing from hardware config: {missing}")
    return list(dict.fromkeys(selected))


def collect_position_samples(
    controller,
    *,
    servo_ids: Iterable[int],
    sample_count: int,
    sample_interval_s: float,
) -> dict[int, list[int]]:
    ids = [int(servo_id) for servo_id in servo_ids]
    samples = {servo_id: [] for servo_id in ids}
    for sample_index in range(int(sample_count)):
        response = read_servo_positions_best_effort(controller, ids)
        for servo_id, raw_units in response:
            servo_id = int(servo_id)
            if servo_id in samples:
                samples[servo_id].append(int(raw_units))
        if sample_index + 1 < sample_count and sample_interval_s > 0.0:
            time.sleep(float(sample_interval_s))
    missing = {
        servo_id: len(values)
        for servo_id, values in samples.items()
        if len(values) != sample_count
    }
    if missing:
        raise RuntimeError(
            "Incomplete servo readback during zero capture; "
            f"received sample counts {missing}, expected {sample_count}"
        )
    return samples


def analyze_zero_samples(
    *,
    joint_names: Iterable[str],
    servo_cfgs: dict[str, ServoConfig],
    samples_by_servo_id: dict[int, list[int]],
    max_spread_units: int,
) -> list[JointZeroCapture]:
    captures: list[JointZeroCapture] = []
    for joint in joint_names:
        servo = servo_cfgs[joint]
        values = tuple(int(value) for value in samples_by_servo_id[int(servo.id)])
        median_units = int(round(statistics.median(values)))
        spread_units = max(values) - min(values)
        current_rad = servo.servo_elect_units_to_joint_target_rad(median_units)
        suggested_offset = offset_from_reference_pose_units(
            servo,
            median_units,
            motor_sign=int(servo.motor_unit_direction),
            target_rad=0.0,
        )
        # If a physical-zero pose reads as +delta, commanding logical zero with
        # the current calibration produces approximately -delta physically.
        implied_physical_error_rad = -float(current_rad)
        captures.append(
            JointZeroCapture(
                joint=joint,
                servo_id=int(servo.id),
                samples_raw_units=values,
                median_raw_units=median_units,
                spread_units=spread_units,
                stable=spread_units <= int(max_spread_units),
                current_offset_unit=int(servo.servo_offset_unit),
                suggested_offset_unit=int(suggested_offset),
                offset_change_unit=int(suggested_offset)
                - int(servo.servo_offset_unit),
                current_rad_at_physical_zero=float(current_rad),
                current_deg_at_physical_zero=float(
                    current_rad * 180.0 / 3.141592653589793
                ),
                implied_physical_target_error_rad=implied_physical_error_rad,
                implied_physical_target_error_deg=float(
                    implied_physical_error_rad * 180.0 / 3.141592653589793
                ),
            )
        )
    return captures


def training_pitch_bias_summary(
    captures: Iterable[JointZeroCapture],
) -> dict[str, float] | None:
    by_joint = {capture.joint: capture for capture in captures}
    if any(joint not in by_joint for joint in LEG_PITCH_JOINTS):
        return None
    physical_error = {
        joint: by_joint[joint].implied_physical_target_error_rad
        for joint in LEG_PITCH_JOINTS
    }
    # Keep this projection identical to the training-domain-randomization signs
    # in training/envs/wildrobot_env.py.
    left = (
        -physical_error["left_hip_pitch"]
        + physical_error["left_knee_pitch"]
        - physical_error["left_ankle_pitch"]
    )
    right = (
        physical_error["right_hip_pitch"]
        - physical_error["right_knee_pitch"]
        + physical_error["right_ankle_pitch"]
    )
    average = 0.5 * (left + right)
    rad_to_deg = 180.0 / 3.141592653589793
    return {
        "left_rad": float(left),
        "left_deg": float(left * rad_to_deg),
        "right_rad": float(right),
        "right_deg": float(right * rad_to_deg),
        "average_rad": float(average),
        "average_deg": float(average * rad_to_deg),
    }


def build_candidate_config(
    raw_config: dict,
    captures: Iterable[JointZeroCapture],
) -> dict:
    candidate = copy.deepcopy(raw_config)
    servos = candidate["servo_controller"]["servos"]
    for capture in captures:
        servos[capture.joint]["servo_offset_unit"] = int(
            capture.suggested_offset_unit
        )
    return candidate


def validate_output_paths(
    *,
    config_path: Path,
    report_path: Path,
    output_config: Path | None,
) -> None:
    if report_path == config_path:
        raise ValueError("Refusing to overwrite the active hardware config with report")
    if output_config == config_path:
        raise ValueError(
            "Refusing to overwrite the active hardware config; choose a separate "
            "--output-config path"
        )
    if output_config is not None and output_config == report_path:
        raise ValueError("--report and --output-config must use different paths")


def print_capture_table(captures: Iterable[JointZeroCapture]) -> None:
    print("\nServo zero capture results:")
    print(
        "joint                    id  median  spread  delta_deg  old_offset  "
        "new_offset  change  status"
    )
    for capture in captures:
        status = "stable" if capture.stable else "UNSTABLE"
        print(
            f"{capture.joint:24s} {capture.servo_id:3d}  "
            f"{capture.median_raw_units:6d}  {capture.spread_units:6d}  "
            f"{capture.current_deg_at_physical_zero:+9.3f}  "
            f"{capture.current_offset_unit:+10d}  "
            f"{capture.suggested_offset_unit:+10d}  "
            f"{capture.offset_change_unit:+6d}  {status}"
        )


def _default_report_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        _REPO_ROOT
        / "runtime"
        / "calibration"
        / f"servo_zero_capture_{timestamp}.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture servo offsets from a physically aligned MuJoCo-zero fixture; "
            "this tool never commands servo motion."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "runtime" / "configs" / "hardware_config.json",
    )
    parser.add_argument(
        "--joints",
        default="leg_pitch",
        help="leg_pitch, all, or a comma/space-separated joint list",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--max-spread-units", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Optional separate candidate config; in-place writes are refused",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected joints and planned outputs without connecting to hardware",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.samples <= 0 or args.samples % 2 == 0:
        raise ValueError("--samples must be a positive odd integer")
    if args.sample_interval_s < 0.0:
        raise ValueError("--sample-interval-s must be non-negative")
    if args.max_spread_units < 0:
        raise ValueError("--max-spread-units must be non-negative")

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {config_path}")
    raw_config = json.loads(config_path.read_text())
    config = WrRuntimeConfig.load(config_path)
    servo_cfgs = config.hiwonder_controller.servos
    joint_names = resolve_joint_selection(args.joints, servo_cfgs)
    report_path = (args.report or _default_report_path()).expanduser().resolve()
    output_config = (
        None
        if args.output_config is None
        else args.output_config.expanduser().resolve()
    )
    validate_output_paths(
        config_path=config_path,
        report_path=report_path,
        output_config=output_config,
    )

    print("Servo zero capture never commands joint motion.")
    print("Selected joints:")
    for joint in joint_names:
        print(f"  #{int(servo_cfgs[joint].id):2d} {joint}")
    print(f"Samples per joint: {args.samples}")
    print(f"Maximum accepted spread: {args.max_spread_units} servo units")
    print(f"Report: {report_path}")
    print(f"Candidate config: {output_config or '(report only)'}")
    if args.dry_run:
        return 0

    print(
        "\nSupport the robot now. The selected servos will have torque disabled, "
        "so an unsupported robot can fall or pinch."
    )
    confirmation = input(
        "Type UNLOAD to disable torque on the selected servos: "
    ).strip()
    if confirmation != "UNLOAD":
        print("Calibration cancelled; no hardware command or file write occurred.")
        return 2

    servo_ids = [int(servo_cfgs[joint].id) for joint in joint_names]
    controller = build_calibration_controller(config.hiwonder_controller)
    try:
        if not controller.unload_servos(servo_ids):
            raise RuntimeError("Failed to disable torque on all selected servos")
        print("Selected servos are unloaded.")
        if joint_names == list(LEG_PITCH_JOINTS):
            print(
                "Leg-pitch zero reference: upright straight hip-knee-ankle "
                "chains with flat soles."
            )
        print(
            "Install the calibration fixture or manually align every selected "
            "joint to its MuJoCo-zero reference. Encoder readings alone cannot "
            "establish physical zero."
        )
        confirmation = input(
            "Type ZERO only after the selected joints are physically aligned: "
        ).strip()
        if confirmation != "ZERO":
            print("Capture cancelled; no files were written.")
            return 2
        samples = collect_position_samples(
            controller,
            servo_ids=servo_ids,
            sample_count=int(args.samples),
            sample_interval_s=float(args.sample_interval_s),
        )
    finally:
        try:
            controller.unload_servos(servo_ids)
        finally:
            controller.close()

    captures = analyze_zero_samples(
        joint_names=joint_names,
        servo_cfgs=servo_cfgs,
        samples_by_servo_id=samples,
        max_spread_units=int(args.max_spread_units),
    )
    print_capture_table(captures)
    pitch_summary = training_pitch_bias_summary(captures)
    if pitch_summary is not None:
        print(
            "\nTraining-model equivalent torso-pitch bias: "
            f"left={pitch_summary['left_deg']:+.3f} deg, "
            f"right={pitch_summary['right_deg']:+.3f} deg, "
            f"average={pitch_summary['average_deg']:+.3f} deg"
        )

    unstable = [capture.joint for capture in captures if not capture.stable]
    candidate_config_written = False
    if output_config is not None and not unstable:
        candidate = build_candidate_config(raw_config, captures)
        write_config(candidate, output_config, {})
        candidate_config_written = True
        print(f"\nWrote candidate config: {output_config}")
        print("The active hardware config was not changed.")

    report = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(),
        "source_config": str(config_path),
        "physical_reference": "operator-confirmed MuJoCo joint zero",
        "servo_motion_commanded": False,
        "servo_torque_disabled_during_capture": True,
        "samples_per_joint": int(args.samples),
        "sample_interval_s": float(args.sample_interval_s),
        "max_spread_units": int(args.max_spread_units),
        "all_stable": all(capture.stable for capture in captures),
        "candidate_config": str(output_config) if output_config is not None else None,
        "candidate_config_written": candidate_config_written,
        "training_model_equivalent_torso_pitch_bias": pitch_summary,
        "joints": [asdict(capture) for capture in captures],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote report: {report_path}")

    if output_config is not None and unstable:
        print(
            "Candidate config was not written because these joints were unstable: "
            + ", ".join(unstable)
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
