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


def test_main_runs_home_then_policy_for_one_confirmed_trial(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    bundle = tmp_path / "standing_v0227_ckpt200"
    bundle.mkdir()
    (bundle / "policy_spec.json").write_text("{}")
    config = tmp_path / "hardware_config.json"
    config.write_text("{}")
    log_dir = tmp_path / "logs"
    responses = iter(("READY", "RUN"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))
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
    assert len(calls) == 2
    assert calls[0][1].name.startswith("v0227_ckpt200_home_trial01_")
    assert calls[0][2] is None
    assert "wr_runtime.control.run_policy" in calls[1][0]
    assert calls[1][1] is None
    assert calls[1][2] == 60.0


def _write_home_diagnostic_log(path: Path, *, status: str = "complete") -> None:
    meta = {
        "schema_version": 1,
        "sample_hz": 1.0,
        "home_after_s": 2.0,
        "home_move_ms": 2000,
        "max_tilt_deg": 15.0,
        "actuator_names": ["left_hip_pitch", "right_hip_pitch"],
        "home_target_rad": [0.0, 0.0],
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
                "footswitches": [1, index % 2, 0, 0],
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
