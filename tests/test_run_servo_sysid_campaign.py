from __future__ import annotations

import subprocess
from pathlib import Path

from runtime.scripts.run_servo_sysid_campaign import (
    CAMPAIGN_RUNS,
    build_capture_command,
    parse_args,
    run_campaign,
)


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--servo-id",
            "100",
            "--board-port",
            "/dev/test-servo",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_standard_campaign_has_fit_validation_and_repeatability_runs() -> None:
    assert [run.run_id for run in CAMPAIGN_RUNS] == [
        "A1_bandwidth",
        "B1_fit_plus30",
        "B2_fit_minus30",
        "V1_validate_plus45",
        "V2_validate_minus45",
        "R1_repeat_zero",
    ]
    assert [run.center_deg for run in CAMPAIGN_RUNS] == [0, 30, -30, 45, -45, 0]
    assert CAMPAIGN_RUNS[0].amplitudes_deg == "2"
    assert CAMPAIGN_RUNS[0].chirp_end_hz == 10.0
    assert all(run.chirp_end_hz == 2.0 for run in CAMPAIGN_RUNS[1:])


def test_capture_command_forces_identification_deadband_and_labels_run(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, "--dry-run")
    output = tmp_path / "01_A1_bandwidth.npz"

    command = build_capture_command(args, CAMPAIGN_RUNS[0], output_path=output)

    assert _option(command, "--servo-id") == "100"
    assert _option(command, "--board-port") == "/dev/test-servo"
    assert _option(command, "--center-deg") == "0.0"
    assert _option(command, "--amplitudes-deg") == "2"
    assert _option(command, "--chirp-end-hz") == "10.0"
    assert _option(command, "--write-deadband-units") == "0"
    assert _option(command, "--center-max-attempts") == "3"
    assert _option(command, "--startup-delay-s") == "3.0"
    assert _option(command, "--unload-pose-deg") == "0.0"
    assert _option(command, "--max-unload-static-torque-nm") == "0.05"
    assert _option(command, "--notes") == "standard-sysid-campaign:A1_bandwidth"
    assert _option(command, "--output") == str(output)
    assert command[-1] == "--dry-run"


def test_campaign_runs_all_conditions_in_order(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is False
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_campaign(_args(tmp_path, "--dry-run")) == 0
    assert [_option(command, "--notes") for command in commands] == [
        f"standard-sysid-campaign:{run.run_id}" for run in CAMPAIGN_RUNS
    ]
    assert len({_option(command, "--output") for command in commands}) == len(
        CAMPAIGN_RUNS
    )


def test_campaign_stops_after_first_failed_capture(
    monkeypatch, tmp_path: Path
) -> None:
    return_codes = iter((0, 7))
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, next(return_codes))

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_campaign(_args(tmp_path)) == 7
    assert len(commands) == 2


def test_campaign_can_resume_at_failed_condition(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        run_campaign(_args(tmp_path, "--start-at", "B1_fit_plus30", "--dry-run"))
        == 0
    )
    assert [_option(command, "--notes") for command in commands] == [
        f"standard-sysid-campaign:{run.run_id}" for run in CAMPAIGN_RUNS[1:]
    ]
