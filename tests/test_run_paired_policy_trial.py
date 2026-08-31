from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "scripts"
    / "run_paired_policy_trial.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_paired_policy_trial_test", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_trial_commands_use_runtime_frame_cutoff_and_diagnostics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    bundle = tmp_path / "standing_v0227_ckpt200"
    config = tmp_path / "hardware_config.json"
    policy_log = tmp_path / "v0227_ckpt200_policy_trial01.log"

    home = module._home_command(
        config=config,
        bundle=bundle,
        seconds=60.0,
        max_tilt_deg=15.0,
    )
    home_diagnostics = module._home_command(
        config=config,
        bundle=bundle,
        seconds=60.0,
        max_tilt_deg=15.0,
        home_state_diagnostics=True,
    )
    policy = module._policy_command(
        config=config,
        bundle=bundle,
        home_seconds=10.0,
        home_max_tilt_deg=15.0,
        fall_tilt_deg=20.0,
        log_steps=5,
        log_path=policy_log,
    )

    assert module._bundle_log_prefix(bundle) == "v0227_ckpt200"
    assert "--runtime-frame" in home
    assert "--background" in home
    assert home[home.index("--home-move-ms") + 1] == "2000"
    assert home[home.index("--max-tilt-deg") + 1] == "15.0"
    assert "--home-state-diagnostics" in home_diagnostics
    assert "--stable-only" in policy
    assert "--diagnostic-log-policy" in policy
    assert policy[policy.index("--startup-home-hold-s") + 1] == "10.0"
    assert policy[policy.index("--startup-pose-blend-s") + 1] == "2.0"
    assert policy[policy.index("--startup-stability-max-tilt-deg") + 1] == "15.0"
    assert policy[policy.index("--fall-tilt-deg") + 1] == "20.0"
    assert policy[policy.index("--log") + 1] == str(policy_log)


def test_streaming_runner_stops_after_first_policy_step() -> None:
    module = _load_module()
    command = [
        sys.executable,
        "-u",
        "-c",
        "import time; print('[step     0]', flush=True); time.sleep(30)",
    ]

    start = time.monotonic()
    result = module._run_streaming(command, stop_after_first_step_s=0.05)

    assert time.monotonic() - start < 3.0
    assert result.timed_stop
    assert result.returncode != 0
    assert not result.fall_abort_seen


def test_streaming_runner_drains_all_output_after_child_exit(tmp_path: Path) -> None:
    module = _load_module()
    output_log = tmp_path / "stream.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        "for i in range(5000): print(f'ROW {i:04d} ' + 'x' * 500)",
    ]

    result = module._run_streaming(
        command,
        output_log=output_log,
        quiet_line_prefixes=("ROW ",),
    )

    assert result.returncode == 0
    assert sum(line.startswith("ROW ") for line in output_log.read_text().splitlines()) == 5000


def test_streaming_runner_exposes_runtime_and_repo_packages(tmp_path: Path) -> None:
    module = _load_module()
    output_log = tmp_path / "imports.log"
    command = [
        sys.executable,
        "-c",
        "import policy_contract; import wr_runtime; print('imports ok')",
    ]

    result = module._run_streaming(command, output_log=output_log)

    assert result.returncode == 0
    assert "imports ok" in output_log.read_text()


def test_main_runs_home_and_policy_in_one_process_for_one_confirmed_trial(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    bundle = tmp_path / "standing_v0227_ckpt200"
    bundle.mkdir()
    (bundle / "policy_spec.json").write_text("{}")
    config = tmp_path / "hardware_config.json"
    config.write_text("{}")
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("builtins.input", lambda _prompt: "READY")
    monkeypatch.setattr(module, "_timestamp", lambda: "20260802_120000_000000")

    calls = []

    def _fake_run(
        command,
        *,
        output_log=None,
        stop_after_first_step_s=None,
        quiet_line_prefixes=(),
    ):
        calls.append((command, output_log, stop_after_first_step_s))
        return module.ProcessResult(
            returncode=0,
            timed_stop=stop_after_first_step_s is not None,
            fall_abort_seen=False,
        )

    monkeypatch.setattr(module, "_run_streaming", _fake_run)

    result = module.main(
        [
            "--trials",
            "1",
            "--bundle",
            str(bundle),
            "--hardware-config",
            str(config),
            "--log-dir",
            str(log_dir),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert "wr_runtime.control.run_policy" in calls[0][0]
    assert calls[0][0][calls[0][0].index("--startup-home-hold-s") + 1] == "60.0"
    assert calls[0][1] is None
    assert calls[0][2] == 60.0
    log_path = Path(calls[0][0][calls[0][0].index("--log") + 1])
    assert log_path.name.startswith("v0227_ckpt200_paired_trial01_")


def test_main_session_log_captures_complete_console(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    bundle = tmp_path / "standing_v0227_ckpt200"
    bundle.mkdir()
    (bundle / "policy_spec.json").write_text("{}")
    config = tmp_path / "hardware_config.json"
    config.write_text("{}")
    session_log = tmp_path / "paired_session.log"
    monkeypatch.setattr("builtins.input", lambda _prompt: "READY")
    monkeypatch.setattr(module, "_preflight_policy_runtime_imports", lambda: None)

    def _fake_run(
        command,
        *,
        output_log=None,
        stop_after_first_step_s=None,
        quiet_line_prefixes=(),
    ):
        print("CHILD OUTPUT", flush=True)
        print("CHILD STDERR", file=sys.stderr, flush=True)
        return module.ProcessResult(
            returncode=0,
            timed_stop=stop_after_first_step_s is not None,
            fall_abort_seen=False,
        )

    monkeypatch.setattr(module, "_run_streaming", _fake_run)

    result = module.main(
        [
            "--trials",
            "1",
            "--bundle",
            str(bundle),
            "--hardware-config",
            str(config),
            "--log-dir",
            str(tmp_path / "logs"),
            "--session-log",
            str(session_log),
        ]
    )

    assert result == 0
    session_text = session_log.read_text()
    assert f"Session log: {session_log}" in session_text
    assert "Continuous hardware test:" in session_text
    assert session_text.count("CHILD OUTPUT") == 1
    assert session_text.count("CHILD STDERR") == 1
    assert "All paired trials complete." in session_text


def _write_home_diagnostic_log(
    path: Path,
    *,
    status: str = "complete",
    footswitch_available: bool = True,
) -> None:
    meta = {
        "schema_version": 1,
        "sample_hz": 1.0,
        "home_after_s": 2.0,
        "home_move_ms": 2000,
        "max_tilt_deg": 15.0,
        "actuator_names": ["left_hip_pitch", "right_hip_pitch"],
        "home_target_rad": [0.0, 0.0],
        "footswitch_available": footswitch_available,
        "footswitch_order": [
            "left_toe",
            "left_heel",
            "right_toe",
            "right_heel",
        ],
    }
    samples = []
    for index, pitch in enumerate((1.0, 2.0, 3.0, 4.0), start=4):
        samples.append(
            {
                "sample": index,
                "elapsed_s": float(index),
                "imu_timestamp_s": float(index),
                "imu_fresh": True,
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gyro_rad_s": [0.0, 0.1, 0.0],
                "rpy_deg": [0.0, pitch, 0.0],
                "tilt_deg": pitch,
                "joint_pos_rad": [0.0, 0.0],
                "joint_error_deg": [0.5, -0.5],
                "footswitches": (
                    [1, index % 2, 0, 0] if footswitch_available else None
                ),
                "servo_cache_age_max_s": 0.02,
                "servo_cache_age_leg_max_s": 0.02,
                "servo_read_fail_count": 0,
            }
        )
    result = {
        "status": status,
        "samples": len(samples),
        "elapsed_s": 7.0,
        "servos_unloaded": True,
    }
    lines = [f"HOME_DIAGNOSTIC_META {json.dumps(meta)}"]
    lines.extend(
        f"HOME_DIAGNOSTIC_SAMPLE {json.dumps(sample)}" for sample in samples
    )
    lines.append(f"HOME_DIAGNOSTIC_RESULT {json.dumps(result)}")
    path.write_text("\n".join(lines) + "\n")


def test_home_diagnostic_summary_reports_distribution(tmp_path: Path) -> None:
    module = _load_module()
    trial_log = tmp_path / "v0227_ckpt200_home_trial01.log"
    _write_home_diagnostic_log(trial_log)
    module._timestamp = lambda: "20260802_120000_000000"

    trial = module._load_home_diagnostic_log(trial_log)
    summary_path = module._write_home_characterization_summary(
        log_paths=[trial_log],
        log_dir=tmp_path,
        prefix="v0227_ckpt200",
    )
    summary = json.loads(summary_path.read_text())

    assert trial["sample_rate_hz"] == 1.0
    assert trial["final_pitch_deg"] == 4.0
    assert trial["pitch_drift_slope_deg_s"] == 1.0
    assert trial["final_window_footswitch_pressed_ratio"]["left_toe"] == 1.0
    assert summary["aggregate"]["trial_count"] == 1
    assert summary["aggregate"]["final_pitch_deg"]["p95"] == 4.0
    assert summary["aggregate"]["final_window_abs_pitch_rate_p95_rad_s"]["p95"] == 0.1

    dropped_log = tmp_path / "v0227_ckpt200_home_trial02.log"
    dropped_log.write_text(
        "\n".join(
            line
            for line in trial_log.read_text().splitlines()
            if not (line.startswith("HOME_DIAGNOSTIC_SAMPLE ") and '"sample": 5' in line)
        )
        + "\n"
    )
    dropped = module._load_home_diagnostic_log(dropped_log)
    assert dropped["sample_rate_hz"] == 1.0
    assert dropped["sample_capture_ratio"] == 0.75


def test_home_diagnostic_summary_marks_disabled_footswitches_unavailable(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trial_log = tmp_path / "disabled_footswitches.log"
    _write_home_diagnostic_log(trial_log, footswitch_available=False)
    module._timestamp = lambda: "20260802_120000_000000"

    trial = module._load_home_diagnostic_log(trial_log)
    summary_path = module._write_home_characterization_summary(
        log_paths=[trial_log],
        log_dir=tmp_path,
        prefix="standing",
    )
    summary = json.loads(summary_path.read_text())

    assert trial["final_window_footswitch_pressed_ratio"] is None
    assert summary["aggregate"]["final_window_footswitch_pressed_ratio"] is None


def test_main_home_characterization_runs_home_only_and_writes_summary(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    bundle = tmp_path / "standing_v0227_ckpt200"
    bundle.mkdir()
    (bundle / "policy_spec.json").write_text("{}")
    config = tmp_path / "hardware_config.json"
    config.write_text("{}")
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("builtins.input", lambda _prompt: "READY")
    monkeypatch.setattr(module, "_timestamp", lambda: "20260802_120000_000000")
    calls = []

    def _fake_run(
        command,
        *,
        output_log=None,
        stop_after_first_step_s=None,
        quiet_line_prefixes=(),
    ):
        calls.append((command, output_log, quiet_line_prefixes))
        assert output_log is not None
        _write_home_diagnostic_log(output_log)
        return module.ProcessResult(
            returncode=0,
            timed_stop=False,
            fall_abort_seen=False,
        )

    monkeypatch.setattr(module, "_run_streaming", _fake_run)

    result = module.main(
        [
            "--home-characterization",
            "--trials",
            "1",
            "--bundle",
            str(bundle),
            "--hardware-config",
            str(config),
            "--log-dir",
            str(log_dir),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert "--home-state-diagnostics" in calls[0][0]
    assert calls[0][2] == ("HOME_DIAGNOSTIC_SAMPLE ",)
    assert list(log_dir.glob("v0227_ckpt200_home_characterization_summary_*.log"))
