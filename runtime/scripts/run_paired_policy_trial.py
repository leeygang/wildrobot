#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
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


def _home_command(
    *,
    config: Path,
    bundle: Path,
    seconds: float,
) -> list[str]:
    return [
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
        "800",
    ]


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
            "Run paired home-hold and standing-policy hardware trials without "
            "changing foot placement between phases."
        )
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of paired trials (default: 3)."
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
        "--log-steps", type=int, default=5, help="Policy log interval (default: 5)."
    )
    parser.add_argument("--bundle", type=Path, default=_DEFAULT_BUNDLE)
    parser.add_argument("--hardware-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.home_seconds <= 0.0 or args.policy_seconds <= 0.0:
        parser.error("--home-seconds and --policy-seconds must be positive")
    if not 0.0 < args.fall_tilt_deg <= 180.0:
        parser.error("--fall-tilt-deg must be in (0, 180]")
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

    print(
        "Paired hardware test: keep the slack tether attached and do not touch "
        "the robot between home hold and policy.",
        flush=True,
    )
    print(
        f"trials={args.trials} home_seconds={args.home_seconds:.1f} "
        f"policy_seconds={args.policy_seconds:.1f} "
        f"fall_tilt_deg={args.fall_tilt_deg:.1f}",
        flush=True,
    )

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
                _home_command(config=config, bundle=bundle, seconds=args.home_seconds),
                output_log=home_log,
            )
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

    print("\nAll paired trials complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
