#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _RUNTIME_ROOT.parent
_PROBE_SCRIPT = _RUNTIME_ROOT / "scripts" / "probe_bno085.py"
_DEFAULT_BUNDLE = _RUNTIME_ROOT / "bundles" / "standing_v0227_ckpt200"
_DEFAULT_CONFIG = _RUNTIME_ROOT / "configs" / "hardware_config.json"
_DEFAULT_LOG_DIR = _REPO_ROOT / "_run_policy_logs"
_STEP_LINE_RE = re.compile(r"^\[step\s+\d+\]")


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_stop: bool
    fall_abort_seen: bool


def _bundle_log_prefix(bundle: Path) -> str:
    match = re.search(r"(v\d+).*?(ckpt\d+)", bundle.name.lower())
    if match is not None:
        return f"{match.group(1)}_{match.group(2)}"
    return re.sub(r"[^a-z0-9]+", "-", bundle.name.lower()).strip("-") or "bundle"


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile) / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _linear_slope(points: list[tuple[float, float]]) -> float:
    finite = [
        (float(x), float(y))
        for x, y in points
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(finite) < 2:
        return float("nan")
    mean_x = sum(x for x, _ in finite) / len(finite)
    mean_y = sum(y for _, y in finite) / len(finite)
    denominator = sum((x - mean_x) ** 2 for x, _ in finite)
    if denominator <= 0.0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in finite) / denominator


def _load_home_diagnostic_log(log_path: Path) -> dict[str, object]:
    meta: dict[str, object] = {}
    result: dict[str, object] = {}
    samples: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("HOME_DIAGNOSTIC_META "):
            meta = json.loads(line.partition(" ")[2])
        elif line.startswith("HOME_DIAGNOSTIC_SAMPLE "):
            samples.append(json.loads(line.partition(" ")[2]))
        elif line.startswith("HOME_DIAGNOSTIC_RESULT "):
            result = json.loads(line.partition(" ")[2])
    if not meta or not samples:
        raise ValueError(f"Home diagnostic log is incomplete: {log_path}")

    last_elapsed_s = float(samples[-1]["elapsed_s"])
    settle_start_s = float(meta["home_after_s"]) + float(meta["home_move_ms"]) / 1000.0
    final_window_start_s = max(settle_start_s, last_elapsed_s - 10.0)
    drift_window_start_s = max(settle_start_s, last_elapsed_s - 30.0)
    final_window = [
        sample
        for sample in samples
        if float(sample["elapsed_s"]) >= final_window_start_s
    ]
    drift_window = [
        sample
        for sample in samples
        if float(sample["elapsed_s"]) >= drift_window_start_s
    ]
    if not final_window:
        final_window = samples[-1:]
    if not drift_window:
        drift_window = samples[-1:]

    pitch = [float(sample["rpy_deg"][1]) for sample in final_window]
    tilt = [float(sample["tilt_deg"]) for sample in final_window]
    gyro_norm = [
        math.sqrt(sum(float(value) ** 2 for value in sample["gyro_rad_s"]))
        for sample in final_window
    ]
    abs_pitch_rate = [
        abs(float(sample["gyro_rad_s"][1])) for sample in final_window
    ]
    joint_error_abs = [
        abs(float(value))
        for sample in final_window
        for value in sample["joint_error_deg"]
    ]
    footswitch_order = [str(name) for name in meta["footswitch_order"]]
    footswitch_pressed_ratio = {
        name: sum(int(sample["footswitches"][index]) for sample in final_window)
        / len(final_window)
        for index, name in enumerate(footswitch_order)
    }
    elapsed = [float(sample["elapsed_s"]) for sample in samples]
    sample_rate_hz = (
        (len(elapsed) - 1) / (elapsed[-1] - elapsed[0])
        if len(elapsed) > 1 and elapsed[-1] > elapsed[0]
        else float("nan")
    )
    pitch_drift_slope = _linear_slope(
        [
            (float(sample["elapsed_s"]), float(sample["rpy_deg"][1]))
            for sample in drift_window
        ]
    )
    final_sample = samples[-1]
    return {
        "log": str(log_path),
        "status": str(result.get("status", "incomplete")),
        "sample_count": len(samples),
        "duration_s": last_elapsed_s,
        "sample_rate_hz": sample_rate_hz if math.isfinite(sample_rate_hz) else None,
        "fresh_imu_ratio": sum(bool(sample["imu_fresh"]) for sample in samples)
        / len(samples),
        "final_pitch_deg": float(final_sample["rpy_deg"][1]),
        "final_tilt_deg": float(final_sample["tilt_deg"]),
        "final_window_pitch_mean_deg": sum(pitch) / len(pitch),
        "final_window_pitch_p95_deg": _percentile(pitch, 95.0),
        "final_window_tilt_max_deg": max(tilt),
        "final_window_gyro_p95_rad_s": _percentile(gyro_norm, 95.0),
        "final_window_abs_pitch_rate_p50_rad_s": _percentile(
            abs_pitch_rate, 50.0
        ),
        "final_window_abs_pitch_rate_p95_rad_s": _percentile(
            abs_pitch_rate, 95.0
        ),
        "pitch_drift_slope_deg_s": (
            pitch_drift_slope if math.isfinite(pitch_drift_slope) else None
        ),
        "final_window_joint_error_rms_deg": math.sqrt(
            sum(value * value for value in joint_error_abs) / len(joint_error_abs)
        ),
        "final_window_joint_error_max_deg": max(joint_error_abs),
        "final_window_footswitch_pressed_ratio": footswitch_pressed_ratio,
    }


def _write_home_characterization_summary(
    *,
    log_paths: list[Path],
    log_dir: Path,
    prefix: str,
) -> Path:
    trials = [_load_home_diagnostic_log(path) for path in log_paths]
    final_pitch = [float(trial["final_pitch_deg"]) for trial in trials]
    final_tilt = [float(trial["final_tilt_deg"]) for trial in trials]
    slopes = [
        float(trial["pitch_drift_slope_deg_s"])
        for trial in trials
        if trial["pitch_drift_slope_deg_s"] is not None
    ]
    sample_rates = [
        float(trial["sample_rate_hz"])
        for trial in trials
        if trial["sample_rate_hz"] is not None
    ]
    pitch_rate_p95 = [
        float(trial["final_window_abs_pitch_rate_p95_rad_s"])
        for trial in trials
    ]
    footswitch_names = list(
        trials[0]["final_window_footswitch_pressed_ratio"].keys()
    )
    aggregate = {
        "trial_count": len(trials),
        "tilt_abort_count": sum(trial["status"] == "tilt_abort" for trial in trials),
        "final_pitch_deg": {
            "min": min(final_pitch),
            "p50": _percentile(final_pitch, 50.0),
            "p95": _percentile(final_pitch, 95.0),
            "max": max(final_pitch),
        },
        "final_tilt_deg": {
            "p50": _percentile(final_tilt, 50.0),
            "p95": _percentile(final_tilt, 95.0),
            "max": max(final_tilt),
        },
        "pitch_drift_slope_deg_s": {
            "p50": _percentile(slopes, 50.0) if slopes else None,
            "p95": _percentile(slopes, 95.0) if slopes else None,
            "max": max(slopes) if slopes else None,
        },
        "sample_rate_hz": {
            "min": min(sample_rates) if sample_rates else None,
            "p50": _percentile(sample_rates, 50.0) if sample_rates else None,
        },
        "final_window_abs_pitch_rate_p95_rad_s": {
            "p50": _percentile(pitch_rate_p95, 50.0),
            "p95": _percentile(pitch_rate_p95, 95.0),
            "max": max(pitch_rate_p95),
        },
        "final_window_footswitch_pressed_ratio": {
            name: sum(
                float(trial["final_window_footswitch_pressed_ratio"][name])
                for trial in trials
            )
            / len(trials)
            for name in footswitch_names
        },
        "final_window_joint_error_max_deg": max(
            float(trial["final_window_joint_error_max_deg"]) for trial in trials
        ),
    }
    summary = {
        "schema_version": 1,
        "description": "Natural-placement home-state characterization",
        "aggregate": aggregate,
        "trials": trials,
    }
    summary_path = (
        log_dir / f"{prefix}_home_characterization_summary_{_timestamp()}.log"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\nHome characterization summary:", flush=True)
    print(
        "  trial status       final_pitch final_tilt drift_deg_s "
        "joint_err_max LT   LH   RT   RH",
        flush=True,
    )
    for index, trial in enumerate(trials, start=1):
        ratios = trial["final_window_footswitch_pressed_ratio"]
        slope = trial["pitch_drift_slope_deg_s"]
        slope_text = "n/a" if slope is None else f"{float(slope):+11.3f}"
        print(
            f"  {index:02d}    {str(trial['status']):<12} "
            f"{float(trial['final_pitch_deg']):+10.2f} "
            f"{float(trial['final_tilt_deg']):10.2f} "
            f"{slope_text:>11} "
            f"{float(trial['final_window_joint_error_max_deg']):13.2f} "
            f"{float(ratios['left_toe']):.2f} "
            f"{float(ratios['left_heel']):.2f} "
            f"{float(ratios['right_toe']):.2f} "
            f"{float(ratios['right_heel']):.2f}",
            flush=True,
        )
    drift_p50 = aggregate["pitch_drift_slope_deg_s"]["p50"]
    drift_p95 = aggregate["pitch_drift_slope_deg_s"]["p95"]
    drift_summary = (
        "n/a"
        if drift_p50 is None or drift_p95 is None
        else f"{float(drift_p50):+.3f}/{float(drift_p95):+.3f}deg/s"
    )
    print(
        f"  aggregate: tilt_aborts={aggregate['tilt_abort_count']}/{len(trials)} "
        f"final_pitch_p50/p95={aggregate['final_pitch_deg']['p50']:+.2f}/"
        f"{aggregate['final_pitch_deg']['p95']:+.2f}deg "
        f"drift_p50/p95={drift_summary} "
        f"abs_pitch_rate_p95="
        f"{aggregate['final_window_abs_pitch_rate_p95_rad_s']['p95']:.3f}rad/s",
        flush=True,
    )
    print(f"  summary log: {summary_path}", flush=True)
    return summary_path


def _home_command(
    *,
    config: Path,
    bundle: Path,
    seconds: float,
    max_tilt_deg: float,
    home_state_diagnostics: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(_PROBE_SCRIPT),
        "--config",
        str(config),
        "--bundle",
        str(bundle),
        "--runtime-frame",
        "--background",
        "--seconds",
        str(seconds),
        "--dt",
        "0.02",
        "--print-every",
        "5",
        "--hold-home",
        "--home-after-s",
        "2",
        "--home-move-ms",
        "2000",
        "--max-tilt-deg",
        str(max_tilt_deg),
    ]
    if home_state_diagnostics:
        command.append("--home-state-diagnostics")
    return command


def _policy_command(
    *,
    config: Path,
    bundle: Path,
    fall_tilt_deg: float,
    log_steps: int,
    log_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "wr_runtime.control.run_policy",
        "--bundle",
        str(bundle),
        "--hardware-config",
        str(config),
        "--stable-only",
        "--fall-tilt-deg",
        str(fall_tilt_deg),
        "--diagnostic-log-policy",
        "--log-steps",
        str(log_steps),
        "--log",
        str(log_path),
    ]


def _run_streaming(
    command: list[str],
    *,
    output_log: Path | None = None,
    stop_after_first_step_s: float | None = None,
    quiet_line_prefixes: tuple[str, ...] = (),
) -> ProcessResult:
    print(f"Command: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=_RUNTIME_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    first_step = threading.Event()
    fall_abort = threading.Event()

    def _copy_output() -> None:
        log_file = (
            output_log.open("w", encoding="utf-8") if output_log is not None else None
        )
        try:
            if log_file is not None:
                log_file.write(f"Command: {shlex.join(command)}\n")
                log_file.flush()
            for line in process.stdout:
                if not line.startswith(quiet_line_prefixes):
                    print(line, end="", flush=True)
                if log_file is not None:
                    log_file.write(line)
                    log_file.flush()
                if _STEP_LINE_RE.match(line):
                    first_step.set()
                if "Fall safety abort" in line:
                    fall_abort.set()
        finally:
            if log_file is not None:
                log_file.close()

    output_thread = threading.Thread(target=_copy_output, daemon=True)
    output_thread.start()

    deadline: float | None = None
    timed_stop = False
    slow_stop_warning_s: float | None = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            if (
                stop_after_first_step_s is not None
                and deadline is None
                and not timed_stop
                and first_step.is_set()
            ):
                deadline = now + float(stop_after_first_step_s)
                print(
                    f"Policy control detected; stopping in {float(stop_after_first_step_s):.1f}s.",
                    flush=True,
                )
            if deadline is not None and now >= deadline:
                print(
                    "Policy trial duration reached; requesting graceful unload.",
                    flush=True,
                )
                process.send_signal(signal.SIGINT)
                timed_stop = True
                deadline = None
                slow_stop_warning_s = now + 10.0
            if slow_stop_warning_s is not None and now >= slow_stop_warning_s:
                print(
                    "WARNING: policy has not exited after SIGINT; use the servo power switch if needed.",
                    flush=True,
                )
                slow_stop_warning_s = now + 10.0
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted; requesting child cleanup.", flush=True)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait()
        raise
    finally:
        process.wait()
        output_thread.join(timeout=2.0)

    return ProcessResult(
        returncode=int(process.returncode),
        timed_stop=timed_stop,
        fall_abort_seen=fall_abort.is_set(),
    )


def _unload_all_servos(config_path: Path) -> None:
    if str(_RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(_RUNTIME_ROOT))
    from configs.config import WrRuntimeConfig
    from wr_runtime.hardware.ttl_servo_controller import build_ttl_servo_controller

    print("Unloading all configured servos...", flush=True)
    config = WrRuntimeConfig.load(config_path)
    servo_ids = sorted(
        {int(servo.id) for servo in config.servo_controller.servos.values()}
    )
    controller = build_ttl_servo_controller(config.servo_controller)
    try:
        if not controller.unload_servos(servo_ids):
            raise RuntimeError("servo controller reported unload failure")
    finally:
        controller.close()
    print("Servos unloaded.", flush=True)


def _abort_after_home(config_path: Path, message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    try:
        _unload_all_servos(config_path)
    except Exception as exc:
        print(
            f"ERROR: automatic unload failed: {type(exc).__name__}: {exc}. "
            "Use the servo power switch.",
            file=sys.stderr,
            flush=True,
        )
    return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired home-hold/policy trials or repeated natural-placement "
            "home-state characterization."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Number of trials. Defaults to 3 for paired mode and 10 for "
            "--home-characterization."
        ),
    )
    parser.add_argument(
        "--home-characterization",
        action="store_true",
        help=(
            "Run home-only natural-placement trials, capture 50 Hz IMU/joint/"
            "footswitch diagnostics, and write an aggregate summary."
        ),
    )
    parser.add_argument(
        "--home-seconds",
        type=float,
        default=60.0,
        help="Home-hold duration per trial (default: 60).",
    )
    parser.add_argument(
        "--policy-seconds",
        type=float,
        default=60.0,
        help="Policy duration after the first logged control step (default: 60).",
    )
    parser.add_argument(
        "--fall-tilt-deg",
        type=float,
        default=20.0,
        help="Diagnostic policy cutoff in degrees (default: 20).",
    )
    parser.add_argument(
        "--home-max-tilt-deg",
        type=float,
        default=15.0,
        help="Home-phase tilt cutoff in degrees (default: 15).",
    )
    parser.add_argument(
        "--log-steps", type=int, default=5, help="Policy log interval (default: 5)."
    )
    parser.add_argument("--bundle", type=Path, default=_DEFAULT_BUNDLE)
    parser.add_argument("--hardware-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)
    if args.trials is None:
        args.trials = 10 if args.home_characterization else 3
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.home_seconds <= 0.0 or args.policy_seconds <= 0.0:
        parser.error("--home-seconds and --policy-seconds must be positive")
    if not 0.0 < args.fall_tilt_deg <= 180.0:
        parser.error("--fall-tilt-deg must be in (0, 180]")
    if not 0.0 < args.home_max_tilt_deg <= 180.0:
        parser.error("--home-max-tilt-deg must be in (0, 180]")
    if args.log_steps <= 0:
        parser.error("--log-steps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    config = args.hardware_config.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    if not (bundle / "policy_spec.json").is_file():
        raise FileNotFoundError(f"Policy bundle is invalid: {bundle}")
    if not config.is_file():
        raise FileNotFoundError(f"Hardware config not found: {config}")
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = _bundle_log_prefix(bundle)

    if args.home_characterization:
        print(
            "Home-state characterization: use the normal placement process within "
            "the marked deployment footprint. Exact foot placement should vary. "
            "Keep the slack tether attached.",
            flush=True,
        )
    else:
        print(
            "Paired hardware test: keep the slack tether attached and do not touch "
            "the robot between home hold and policy.",
            flush=True,
        )
    if args.home_characterization:
        print(
            f"trials={args.trials} home_seconds={args.home_seconds:.1f} "
            f"home_max_tilt_deg={args.home_max_tilt_deg:.1f}",
            flush=True,
        )
    else:
        print(
            f"trials={args.trials} home_seconds={args.home_seconds:.1f} "
            f"policy_seconds={args.policy_seconds:.1f} "
            f"home_max_tilt_deg={args.home_max_tilt_deg:.1f} "
            f"fall_tilt_deg={args.fall_tilt_deg:.1f}",
            flush=True,
        )

    home_characterization_logs: list[Path] = []
    try:
        for trial in range(1, int(args.trials) + 1):
            trial_label = f"{trial:02d}"
            ready = (
                input(
                    f"\nTrial {trial_label}: place the robot, attach the slack tether, "
                    "take top/side photos, then type READY: "
                )
                .strip()
                .upper()
            )
            if ready != "READY":
                print("Test stopped before loading servos.", flush=True)
                return 0

            home_log = log_dir / f"{prefix}_home_trial{trial_label}_{_timestamp()}.log"
            print(f"Home log: {home_log}", flush=True)
            home_result = _run_streaming(
                _home_command(
                    config=config,
                    bundle=bundle,
                    seconds=args.home_seconds,
                    max_tilt_deg=args.home_max_tilt_deg,
                    home_state_diagnostics=bool(args.home_characterization),
                ),
                output_log=home_log,
                quiet_line_prefixes=(
                    ("HOME_DIAGNOSTIC_SAMPLE ",)
                    if args.home_characterization
                    else ()
                ),
            )
            if args.home_characterization:
                if home_result.returncode not in (0, 6):
                    return _abort_after_home(
                        config,
                        f"Home diagnostics failed with exit code {home_result.returncode}.",
                    )
                home_characterization_logs.append(home_log)
                outcome = (
                    "tilt cutoff" if home_result.returncode == 6 else "completed"
                )
                print(
                    f"Home trial {trial_label} {outcome}; servos are unloaded.\n"
                    f"  home: {home_log}",
                    flush=True,
                )
                continue
            if home_result.returncode != 0:
                return _abort_after_home(
                    config,
                    f"Home hold failed with exit code {home_result.returncode}; policy will not start.",
                )

            handoff = (
                input(
                    "Home hold complete. Do not touch the robot. Confirm the tether is "
                    "slack and no foot slid; type RUN to start policy, or ABORT: "
                )
                .strip()
                .upper()
            )
            if handoff != "RUN":
                return _abort_after_home(config, "Policy handoff aborted.")

            policy_log = (
                log_dir / f"{prefix}_policy_trial{trial_label}_{_timestamp()}.log"
            )
            print(f"Policy log: {policy_log}", flush=True)
            policy_result = _run_streaming(
                _policy_command(
                    config=config,
                    bundle=bundle,
                    fall_tilt_deg=args.fall_tilt_deg,
                    log_steps=args.log_steps,
                    log_path=policy_log,
                ),
                stop_after_first_step_s=args.policy_seconds,
            )
            clean_timed_stop = (
                policy_result.timed_stop
                and policy_result.returncode
                in (
                    0,
                    130,
                    -signal.SIGINT,
                )
            )
            if not clean_timed_stop and not policy_result.fall_abort_seen:
                return _abort_after_home(
                    config,
                    f"Policy failed unexpectedly with exit code {policy_result.returncode}.",
                )
            outcome = "fall cutoff" if policy_result.fall_abort_seen else "timed stop"
            print(
                f"Trial {trial_label} complete ({outcome}); runtime unloaded the servos.",
                flush=True,
            )
            print(f"  home:   {home_log}", flush=True)
            print(f"  policy: {policy_log}", flush=True)
    except KeyboardInterrupt:
        return _abort_after_home(config, "Paired test interrupted.")

    if args.home_characterization:
        _write_home_characterization_summary(
            log_paths=home_characterization_logs,
            log_dir=log_dir,
            prefix=prefix,
        )
        print("\nAll home characterization trials complete.", flush=True)
        return 0

    print("\nAll paired trials complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
