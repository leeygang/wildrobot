from __future__ import annotations

import importlib.util
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

    def _fake_run(command, *, output_log=None, stop_after_first_step_s=None):
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
