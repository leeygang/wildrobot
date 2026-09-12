#!/usr/bin/env python3
"""Run HTD-45H fixture safety characterization or SysID campaigns."""

from __future__ import annotations

import argparse
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
            "use 'sysid' after the limits plan passes."
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
            else "standard-sysid-campaign"
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
    if campaign_run.prepare_only:
        command.append("--prepare-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


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
    title = (
        "HTD-45H separated load/speed characterization"
        if args.plan == "limits"
        else "HTD-45H standard SysID follow-up campaign"
    )
    print(title, flush=True)
    print(
        f"  runs={len(selected_runs)} servo_id={int(args.servo_id)} "
        f"start_at={args.start_at}",
        flush=True,
    )
    print(f"  output_dir={campaign_dir}", flush=True)
    if args.plan == "limits":
        print(
            "  Preparation-only runs vary one factor at a time: load at 5deg/s, "
            "speed on the same 10deg path, then speed at 30deg load.",
            flush=True,
        )
        print(
            "  Every condition gets one attempt. Any voltage/protection failure "
            "stops the campaign; power-cycle before resuming.",
            flush=True,
        )
    else:
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
            f"[{index}/{len(campaign_runs)}] {campaign_run.run_id}: "
            f"{campaign_run.description}",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
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
        print(f"\nCampaign complete. Artifacts: {campaign_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_campaign(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
