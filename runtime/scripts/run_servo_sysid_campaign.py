#!/usr/bin/env python3
"""Run HTD-45H fixture safety characterization or SysID campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAPTURE_SCRIPT = _REPO_ROOT / "runtime" / "scripts" / "capture_servo_sysid.py"


def _yellow(text: str) -> str:
    if not sys.stderr.isatty() or "NO_COLOR" in os.environ:
        return text
    if os.environ.get("TERM", "") in {"", "dumb"}:
        return text
    return f"\x1b[33m{text}\x1b[0m"


@dataclass(frozen=True)
class CampaignRun:
    run_id: str
    description: str
    center_deg: float
    amplitudes_deg: str = "2,5,8"
    chirp_start_hz: float = 0.1
    chirp_end_hz: float = 2.0
    chirp_duration_s: float = 10.0
    prepare_only: bool = False
    prepare_speed_deg_s: float | None = None
    prepare_monitor_hz: float | None = None
    center_max_attempts: int | None = None
    settle_s: float | None = None
    normalize_start_pose: bool = True
    return_speed_deg_s: float | None = 5.0
    write_deadband_units: int = 0


CAMPAIGN_RUNS = (
    CampaignRun(
        run_id="A1_bandwidth",
        description="zero-load high-frequency bandwidth",
        center_deg=0.0,
        amplitudes_deg="2",
        chirp_end_hz=10.0,
    ),
    CampaignRun(
        run_id="B1_fit_plus30",
        description="positive known-load fit condition",
        center_deg=30.0,
    ),
    CampaignRun(
        run_id="B2_fit_minus30",
        description="negative known-load fit condition",
        center_deg=-30.0,
    ),
    CampaignRun(
        run_id="V1_validate_plus45",
        description="held-out positive-load validation",
        center_deg=45.0,
    ),
    CampaignRun(
        run_id="V2_validate_minus45",
        description="held-out negative-load validation",
        center_deg=-45.0,
    ),
    CampaignRun(
        run_id="R1_repeat_zero",
        description="final zero-load repeatability check",
        center_deg=0.0,
    ),
)


DEPLOYMENT_RUNS = (
    CampaignRun(
        run_id="D1_deployment_deadband_plus45",
        description="held-out loaded profile with deployment command deadband",
        center_deg=45.0,
        write_deadband_units=3,
    ),
)


def _limit_run(
    run_id: str,
    description: str,
    center_deg: float,
    speed_deg_s: float,
    *,
    settle_s: float | None = None,
) -> CampaignRun:
    return CampaignRun(
        run_id=run_id,
        description=description,
        center_deg=center_deg,
        amplitudes_deg="2",
        prepare_only=True,
        prepare_speed_deg_s=speed_deg_s,
        prepare_monitor_hz=50.0,
        center_max_attempts=1,
        settle_s=settle_s,
    )


LIMIT_CAMPAIGN_RUNS = (
    _limit_run(
        "L1_load_10deg_5dps",
        "slow load sweep at 10 degrees",
        10.0,
        5.0,
        settle_s=3.0,
    ),
    _limit_run(
        "L2_load_20deg_5dps",
        "slow load sweep at 20 degrees",
        20.0,
        5.0,
        settle_s=3.0,
    ),
    _limit_run(
        "L3_load_30deg_5dps",
        "slow load sweep at 30 degrees",
        30.0,
        5.0,
        settle_s=3.0,
    ),
    _limit_run(
        "S1_speed_10deg_20dps",
        "low-load speed sweep at 20 degrees/s",
        10.0,
        20.0,
    ),
    _limit_run(
        "S2_speed_10deg_50dps",
        "low-load speed sweep at 50 degrees/s",
        10.0,
        50.0,
    ),
    _limit_run(
        "S3_speed_10deg_100dps",
        "low-load speed sweep at 100 degrees/s",
        10.0,
        100.0,
    ),
    _limit_run(
        "C1_combined_30deg_10dps",
        "loaded speed verification at 10 degrees/s",
        30.0,
        10.0,
    ),
    _limit_run(
        "C2_combined_30deg_15dps",
        "loaded speed verification at 15 degrees/s",
        30.0,
        15.0,
    ),
    _limit_run(
        "C3_combined_30deg_20dps",
        "loaded speed verification at 20 degrees/s",
        30.0,
        20.0,
    ),
)


CAMPAIGN_PLANS = {
    "limits": LIMIT_CAMPAIGN_RUNS,
    "sysid": CAMPAIGN_RUNS,
    "complete": LIMIT_CAMPAIGN_RUNS + CAMPAIGN_RUNS + DEPLOYMENT_RUNS,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run separated HTD-45H load/speed characterization or the full "
            "SysID follow-up campaign with automatic cooldown and safe unload."
        )
    )
    parser.add_argument(
        "--plan",
        choices=tuple(CAMPAIGN_PLANS),
        default="limits",
        help=(
            "Use 'limits' before chirps to separate load and speed effects; "
            "use 'sysid' after the limits plan passes; 'complete' preflights "
            "and runs both plus a deployment-deadband validation."
        ),
    )
    parser.add_argument("--servo-id", type=int, required=True)
    parser.add_argument(
        "--board-port",
        "--board_port",
        dest="board_port",
        required=True,
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--fixture-mjcf",
        type=Path,
        default=Path("assets/bam/robot.xml"),
    )
    parser.add_argument("--fixture-joint", default="pitch")
    parser.add_argument("--fixture-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--fixture-qpos-offset-deg", type=float, default=0.0)
    parser.add_argument("--servo-label", default=None)
    parser.add_argument("--fixture-label", default="bam-v1")
    parser.add_argument("--cooldown-target-c", type=float, default=35.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--cooldown-poll-s", type=float, default=5.0)
    parser.add_argument(
        "--min-voltage-v",
        type=float,
        default=9.6,
        help="Minimum HTD-45H operating voltage from the vendor specification.",
    )
    parser.add_argument("--max-temperature-c", type=float, default=60.0)
    parser.add_argument(
        "--profile-health-poll-hz",
        type=float,
        default=6.0,
        help=(
            "Total extra in-profile bus reads per second, staggered across "
            "voltage, temperature, and torque-enable state."
        ),
    )
    parser.add_argument("--max-position-error-deg", type=float, default=12.0)
    parser.add_argument(
        "--max-static-torque-nm",
        type=float,
        default=2.5,
        help=(
            "Hard preflight ceiling for modeled fixture holding torque. "
            "Increasing it does not assert that the servo can safely supply it."
        ),
    )
    parser.add_argument("--center-max-attempts", type=int, default=3)
    parser.add_argument("--startup-delay-s", type=float, default=3.0)
    parser.add_argument("--unload-pose-deg", type=float, default=0.0)
    parser.add_argument("--max-unload-static-torque-nm", type=float, default=0.05)
    parser.add_argument(
        "--start-at",
        choices=tuple(
            run.run_id for runs in CAMPAIGN_PLANS.values() for run in runs
        ),
        default=None,
        help="Resume the campaign at this condition in a new output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/calibration/servo_sysid"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.profile_health_poll_hz < 0.0:
        parser.error("--profile-health-poll-hz must be non-negative")
    if args.profile_health_poll_hz > 50.0:
        parser.error("--profile-health-poll-hz must not exceed the 50 Hz profile")
    if args.max_position_error_deg <= 0.0:
        parser.error("--max-position-error-deg must be positive")
    if args.max_static_torque_nm <= 0.0:
        parser.error("--max-static-torque-nm must be positive")
    selected_runs = CAMPAIGN_PLANS[str(args.plan)]
    if args.start_at is None:
        args.start_at = selected_runs[0].run_id
    elif args.start_at not in {run.run_id for run in selected_runs}:
        parser.error(f"--start-at {args.start_at!r} is not part of --plan {args.plan}")
    return args


def build_capture_command(
    args: argparse.Namespace,
    campaign_run: CampaignRun,
    *,
    output_path: Path,
) -> list[str]:
    servo_label = args.servo_label or f"htd45h-sysid-id{int(args.servo_id)}"
    command = [
        sys.executable,
        str(_CAPTURE_SCRIPT),
        "--servo-id",
        str(int(args.servo_id)),
        "--board-port",
        str(args.board_port),
        "--baudrate",
        str(int(args.baudrate)),
        "--center-deg",
        str(float(campaign_run.center_deg)),
        "--fixture-mjcf",
        str(args.fixture_mjcf),
        "--fixture-joint",
        str(args.fixture_joint),
        "--fixture-direction",
        str(int(args.fixture_direction)),
        "--fixture-qpos-offset-deg",
        str(float(args.fixture_qpos_offset_deg)),
        "--amplitudes-deg",
        campaign_run.amplitudes_deg,
        "--chirp-start-hz",
        str(float(campaign_run.chirp_start_hz)),
        "--chirp-end-hz",
        str(float(campaign_run.chirp_end_hz)),
        "--chirp-duration-s",
        str(float(campaign_run.chirp_duration_s)),
        "--write-deadband-units",
        str(int(campaign_run.write_deadband_units)),
        "--profile-health-poll-hz",
        str(float(args.profile_health_poll_hz)),
        "--cooldown-target-c",
        str(float(args.cooldown_target_c)),
        "--cooldown-timeout-s",
        str(float(args.cooldown_timeout_s)),
        "--cooldown-poll-s",
        str(float(args.cooldown_poll_s)),
        "--min-voltage-v",
        str(float(args.min_voltage_v)),
        "--max-temperature-c",
        str(float(args.max_temperature_c)),
        "--max-position-error-deg",
        str(float(args.max_position_error_deg)),
        "--max-static-torque-nm",
        str(float(args.max_static_torque_nm)),
        "--center-max-attempts",
        str(
            int(
                campaign_run.center_max_attempts
                if campaign_run.center_max_attempts is not None
                else args.center_max_attempts
            )
        ),
        "--startup-delay-s",
        str(float(args.startup_delay_s)),
        "--unload-pose-deg",
        str(float(args.unload_pose_deg)),
        "--max-unload-static-torque-nm",
        str(float(args.max_unload_static_torque_nm)),
        "--servo-label",
        servo_label,
        "--fixture-label",
        str(args.fixture_label),
        "--notes",
        (
            "limits-servo-campaign"
            if args.plan == "limits"
            else (
                "standard-sysid-campaign"
                if args.plan == "sysid"
                else "complete-servo-campaign"
            )
        )
        + f":{campaign_run.run_id}",
        "--output",
        str(output_path),
    ]
    if campaign_run.prepare_speed_deg_s is not None:
        command.extend(
            ["--prepare-speed-deg-s", str(float(campaign_run.prepare_speed_deg_s))]
        )
    if campaign_run.prepare_monitor_hz is not None:
        command.extend(
            ["--prepare-monitor-hz", str(float(campaign_run.prepare_monitor_hz))]
        )
    if campaign_run.settle_s is not None:
        command.extend(["--settle-s", str(float(campaign_run.settle_s))])
    if campaign_run.normalize_start_pose:
        command.append("--normalize-start-pose")
    if campaign_run.return_speed_deg_s is not None:
        command.extend(
            ["--return-speed-deg-s", str(float(campaign_run.return_speed_deg_s))]
        )
    if campaign_run.prepare_only:
        command.append("--prepare-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_campaign_manifest(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist progress so an interrupted campaign remains usable."""

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _capture_record(
    campaign_run: CampaignRun,
    *,
    index: int,
    output_path: Path,
    returncode: int,
) -> dict[str, object]:
    json_path = output_path.with_suffix(".json")
    payload: dict[str, object] = {}
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return {
        "index": int(index),
        "run_id": campaign_run.run_id,
        "description": campaign_run.description,
        "returncode": int(returncode),
        "outcome": payload.get("outcome"),
        "error": payload.get("error"),
        "center_deg": float(campaign_run.center_deg),
        "prepare_speed_deg_s": campaign_run.prepare_speed_deg_s,
        "write_deadband_units": int(campaign_run.write_deadband_units),
        "npz_path": str(output_path),
        "json_path": str(json_path),
        "npz_sha256": _sha256(output_path),
        "json_sha256": _sha256(json_path),
        "static_torque_at_center_nm": payload.get("static_torque_at_center_nm"),
        "max_abs_profile_hold_torque_nm": payload.get(
            "max_abs_profile_hold_torque_nm"
        ),
        "thermal_summary": payload.get("thermal_summary"),
        "preparation_monitor": (
            payload.get("servo_diagnostics", {}).get("preparation_monitor")
            if isinstance(payload.get("servo_diagnostics"), dict)
            else None
        ),
        "capture_summary": summary,
    }


def run_campaign(args: argparse.Namespace) -> int:
    campaign_runs = CAMPAIGN_PLANS[str(args.plan)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_label = "limits" if args.plan == "limits" else "campaign"
    campaign_dir = (
        args.output_dir.expanduser().resolve()
        / f"htd45h_servo_{int(args.servo_id)}_{directory_label}_{timestamp}"
    )
    start_index = next(
        index
        for index, campaign_run in enumerate(campaign_runs)
        if campaign_run.run_id == args.start_at
    )
    selected_runs = campaign_runs[start_index:]
    titles = {
        "limits": "HTD-45H separated load/speed characterization",
        "sysid": "HTD-45H standard SysID follow-up campaign",
        "complete": "HTD-45H comprehensive deployment characterization",
    }
    title = titles[str(args.plan)]
    print(title, flush=True)
    print(
        f"  runs={len(selected_runs)} servo_id={int(args.servo_id)} "
        f"start_at={args.start_at}",
        flush=True,
    )
    print(f"  output_dir={campaign_dir}", flush=True)
    if args.plan in {"limits", "complete"}:
        print(
            "  Preparation-only runs vary one factor at a time: load at 5deg/s, "
            "speed on the same 10deg path, then speed at 30deg load.",
            flush=True,
        )
        print(
            "  Every run normalizes to the zero pose first and returns at a "
            "fixed 5deg/s, independent of the tested outbound speed.",
            flush=True,
        )
        print(
            "  Every condition gets one attempt. Any voltage/protection failure "
            "stops the campaign; power-cycle before resuming.",
            flush=True,
        )
    if args.plan in {"sysid", "complete"}:
        print(
            "  Each run independently cools the unloaded servo, normalizes to "
            "zero, verifies center with bounded retries, returns at 5deg/s, "
            "and disables torque.",
            flush=True,
        )
    if args.plan == "complete":
        print(
            "  The complete plan preflights every condition before opening the "
            "bus, then runs limits, SysID, and deployment-deadband validation.",
            flush=True,
        )

    manifest_path = campaign_dir / "campaign_manifest.json"
    campaign_manifest: dict[str, object] | None = None
    if not args.dry_run:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        campaign_manifest = {
            "schema_version": 1,
            "status": "preflight" if args.plan == "complete" else "running",
            "plan": str(args.plan),
            "started_at": datetime.now().astimezone().isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
            "servo_id": int(args.servo_id),
            "board_port": str(args.board_port),
            "baudrate": int(args.baudrate),
            "fixture_mjcf": str(args.fixture_mjcf.expanduser().resolve()),
            "fixture_joint": str(args.fixture_joint),
            "fixture_direction": int(args.fixture_direction),
            "fixture_qpos_offset_deg": float(args.fixture_qpos_offset_deg),
            "software": {
                "python": sys.version,
                "campaign_script_sha256": _sha256(Path(__file__)),
                "capture_script_sha256": _sha256(_CAPTURE_SCRIPT),
            },
            "safety_limits": {
                "min_voltage_v": float(args.min_voltage_v),
                "max_temperature_c": float(args.max_temperature_c),
                "max_position_error_deg": float(args.max_position_error_deg),
                "max_static_torque_nm": float(args.max_static_torque_nm),
                "max_unload_static_torque_nm": float(
                    args.max_unload_static_torque_nm
                ),
            },
            "profile_health_poll_hz_total": float(args.profile_health_poll_hz),
            "preflight_completed": False if args.plan == "complete" else None,
            "runs": [],
        }
        _write_campaign_manifest(manifest_path, campaign_manifest)

    if args.plan == "complete" and not args.dry_run:
        print("\nPreflighting all selected conditions without hardware IO...", flush=True)
        for index, campaign_run in enumerate(selected_runs, start=start_index + 1):
            output_path = campaign_dir / f"{index:02d}_{campaign_run.run_id}.npz"
            command = build_capture_command(args, campaign_run, output_path=output_path)
            result = subprocess.run([*command, "--dry-run"], check=False)
            if result.returncode != 0:
                assert campaign_manifest is not None
                campaign_manifest["status"] = "preflight_failed"
                campaign_manifest["failed_run_id"] = campaign_run.run_id
                campaign_manifest["updated_at"] = datetime.now().astimezone().isoformat()
                _write_campaign_manifest(manifest_path, campaign_manifest)
                print(
                    _yellow(
                        f"Campaign preflight failed at {campaign_run.run_id}; "
                        "no hardware IO was started."
                    ),
                    file=sys.stderr,
                )
                return int(result.returncode)
        assert campaign_manifest is not None
        campaign_manifest["preflight_completed"] = True
        campaign_manifest["status"] = "running"
        campaign_manifest["updated_at"] = datetime.now().astimezone().isoformat()
        _write_campaign_manifest(manifest_path, campaign_manifest)

    for index, campaign_run in enumerate(selected_runs, start=start_index + 1):
        output_path = campaign_dir / f"{index:02d}_{campaign_run.run_id}.npz"
        command = build_capture_command(
            args,
            campaign_run,
            output_path=output_path,
        )
        print()
        print(
            f"[{index}/{len(campaign_runs)}] {campaign_run.run_id}: "
            f"{campaign_run.description}",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if campaign_manifest is not None:
            runs = campaign_manifest["runs"]
            assert isinstance(runs, list)
            runs.append(
                _capture_record(
                    campaign_run,
                    index=index,
                    output_path=output_path,
                    returncode=int(result.returncode),
                )
            )
            campaign_manifest["updated_at"] = datetime.now().astimezone().isoformat()
            _write_campaign_manifest(manifest_path, campaign_manifest)
        if result.returncode != 0:
            if campaign_manifest is not None:
                campaign_manifest["status"] = (
                    "aborted" if int(result.returncode) == 130 else "failed"
                )
                campaign_manifest["failed_run_id"] = campaign_run.run_id
                _write_campaign_manifest(manifest_path, campaign_manifest)
            print(
                _yellow(
                    f"Campaign stopped at {campaign_run.run_id}; capture exited "
                    f"with status {result.returncode}."
                ),
                file=sys.stderr,
            )
            print(
                _yellow(f"Completed artifacts remain in: {campaign_dir}"),
                file=sys.stderr,
            )
            return int(result.returncode)

    if args.dry_run:
        print("\nCampaign dry run complete; no hardware was opened or files written.")
    else:
        assert campaign_manifest is not None
        campaign_manifest["status"] = "completed"
        campaign_manifest["completed_at"] = datetime.now().astimezone().isoformat()
        campaign_manifest["updated_at"] = campaign_manifest["completed_at"]
        _write_campaign_manifest(manifest_path, campaign_manifest)
        print(f"\nCampaign complete. Artifacts: {campaign_dir}")
        print(f"Campaign manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
