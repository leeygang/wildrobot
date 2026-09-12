#!/usr/bin/env python3
"""Run the standard HTD-45H fixture SysID follow-up campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAPTURE_SCRIPT = _REPO_ROOT / "runtime" / "scripts" / "capture_servo_sysid.py"


@dataclass(frozen=True)
class CampaignRun:
    run_id: str
    description: str
    center_deg: float
    amplitudes_deg: str = "2,5,8"
    chirp_start_hz: float = 0.1
    chirp_end_hz: float = 2.0
    chirp_duration_s: float = 10.0


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the six HTD-45H captures that follow an accepted zero-load "
            "baseline with automatic cooldown, preparation, and safe unload."
        )
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
    parser.add_argument("--min-voltage-v", type=float, default=9.0)
    parser.add_argument("--max-temperature-c", type=float, default=60.0)
    parser.add_argument("--center-max-attempts", type=int, default=3)
    parser.add_argument("--startup-delay-s", type=float, default=3.0)
    parser.add_argument("--unload-pose-deg", type=float, default=0.0)
    parser.add_argument("--max-unload-static-torque-nm", type=float, default=0.05)
    parser.add_argument(
        "--start-at",
        choices=tuple(run.run_id for run in CAMPAIGN_RUNS),
        default=CAMPAIGN_RUNS[0].run_id,
        help="Resume the campaign at this condition in a new output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/calibration/servo_sysid"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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
        "0",
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
        "--center-max-attempts",
        str(int(args.center_max_attempts)),
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
        f"standard-sysid-campaign:{campaign_run.run_id}",
        "--output",
        str(output_path),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_campaign(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = (
        args.output_dir.expanduser().resolve()
        / f"htd45h_servo_{int(args.servo_id)}_campaign_{timestamp}"
    )
    start_index = next(
        index
        for index, campaign_run in enumerate(CAMPAIGN_RUNS)
        if campaign_run.run_id == args.start_at
    )
    selected_runs = CAMPAIGN_RUNS[start_index:]
    print("HTD-45H standard SysID follow-up campaign", flush=True)
    print(
        f"  runs={len(selected_runs)} servo_id={int(args.servo_id)} "
        f"start_at={args.start_at}",
        flush=True,
    )
    print(f"  output_dir={campaign_dir}", flush=True)
    print(
        "  Each run independently cools the unloaded servo, automatically "
        "verifies center with bounded retries, returns to the gravity-neutral "
        "unload pose, and disables torque.",
        flush=True,
    )

    for index, campaign_run in enumerate(selected_runs, start=start_index + 1):
        output_path = campaign_dir / f"{index:02d}_{campaign_run.run_id}.npz"
        command = build_capture_command(
            args,
            campaign_run,
            output_path=output_path,
        )
        print()
        print(
            f"[{index}/{len(CAMPAIGN_RUNS)}] {campaign_run.run_id}: "
            f"{campaign_run.description}",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                f"Campaign stopped at {campaign_run.run_id}; capture exited "
                f"with status {result.returncode}.",
                file=sys.stderr,
            )
            print(f"Completed artifacts remain in: {campaign_dir}", file=sys.stderr)
            return int(result.returncode)

    if args.dry_run:
        print("\nCampaign dry run complete; no hardware was opened or files written.")
    else:
        print(f"\nCampaign complete. Artifacts: {campaign_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
