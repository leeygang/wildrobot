#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from configs.config import WrRuntimeConfig  # noqa: E402
from wr_runtime.hardware.bno085 import BNO085IMU  # noqa: E402


def _parse_i2c_address(value: str) -> int:
    return int(str(value).strip(), 0)


def _fmt_vec(values, *, digits: int = 5) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(v):+.{digits}f}" for v in arr) + "]"


def _quat_angle_delta_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    a = np.asarray(q0, dtype=np.float32).reshape(4)
    b = np.asarray(q1, dtype=np.float32).reshape(4)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    a = a / na
    b = b / nb
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _default_config_path() -> Path:
    bundle_cfg = _RUNTIME_ROOT / "bundles" / "walking_v0210_smoke6_ckpt1650" / "wildrobot_config.json"
    if bundle_cfg.exists():
        return bundle_cfg
    return _RUNTIME_ROOT / "configs" / "runtime_config_v2.json"


@contextlib.contextmanager
def _alarm_timeout(seconds: float, *, message: str):
    seconds_i = int(max(0, round(float(seconds))))
    if seconds_i <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(message)

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds_i)
        yield
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass


def _start_init_heartbeat(done: threading.Event, *, start_s: float) -> threading.Thread:
    def _run() -> None:
        if done.wait(2.0):
            return
        while not done.is_set():
            elapsed = time.monotonic() - start_s
            print(f"Still initializing BNO085... elapsed_s={elapsed:.1f}", flush=True)
            if done.wait(5.0):
                return

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe BNO085/BNO08x IMU samples and report whether quat/gyro payloads are changing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Runtime config path used for i2c_address/upside_down settings.",
    )
    parser.add_argument("--address", type=_parse_i2c_address, default=None, help="Override I2C address, e.g. 0x4B.")
    parser.add_argument("--samples", type=int, default=120, help="Number of samples to print.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Run duration in seconds; overrides --samples.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Seconds between samples. Defaults to 0.05.",
    )
    parser.add_argument("--sampling-hz", type=int, default=50, help="BNO08x feature report rate.")
    parser.add_argument(
        "--i2c-frequency-hz",
        type=int,
        default=None,
        help="Override I2C bus frequency from config, e.g. 400000.",
    )
    parser.add_argument(
        "--init-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for BNO08x initialization before failing; use 0 to disable.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N samples; use 0 to print only summary.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Use the runtime background reader instead of direct polling.",
    )
    parser.add_argument(
        "--calibration-mode",
        action="store_true",
        help=(
            "Use the same BNO08x setup as IMU calibration: background reader, "
            "20 Hz reports, and rotation-vector fallback enabled."
        ),
    )
    parser.add_argument(
        "--runtime-frame",
        action="store_true",
        help="Apply bno085.axis_map from the config. Default prints the raw configured mount frame.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show verbose Adafruit BNO08x packet debug output.",
    )
    args = parser.parse_args()

    cfg = WrRuntimeConfig.load(args.config)
    address = int(args.address if args.address is not None else cfg.bno085.i2c_address)
    axis_map = cfg.bno085.axis_map if args.runtime_frame else None
    calibration_mode = bool(args.calibration_mode)
    polling_mode = False if calibration_mode else not bool(args.background)
    sampling_hz = 20 if calibration_mode else max(1, int(args.sampling_hz))
    enable_rotation_vector = (
        True if calibration_mode else bool(cfg.bno085.enable_rotation_vector)
    )
    i2c_frequency_hz = int(args.i2c_frequency_hz or cfg.bno085.i2c_frequency_hz)
    dt_s = float(args.dt) if args.dt is not None else 0.05
    total = (
        max(1, int(round(float(args.seconds) / max(1e-6, dt_s))))
        if args.seconds is not None
        else max(1, int(args.samples))
    )

    print(
        "BNO085/BNO08x probe: "
        f"config={args.config} address=0x{address:02X} upside_down={cfg.bno085.upside_down} "
        f"axis_map={axis_map} polling_mode={polling_mode} sampling_hz={sampling_hz} "
        f"enable_rotation_vector={enable_rotation_vector} i2c_frequency_hz={i2c_frequency_hz} "
        f"samples={total} dt={dt_s:.3f}",
        flush=True,
    )
    print("Rotate the robot while samples print. Quat/gyro should change during motion.", flush=True)

    print("Initializing BNO085...", flush=True)
    init_start_s = time.monotonic()
    init_done = threading.Event()
    _start_init_heartbeat(init_done, start_s=init_start_s)
    try:
        with _alarm_timeout(
            args.init_timeout,
            message=(
                "Timed out initializing BNO085. The process is likely blocked inside "
                "BNO08x enable_feature()/packet processing or the I2C bus."
            ),
        ):
            imu = BNO085IMU(
                i2c_address=address,
                upside_down=cfg.bno085.upside_down,
                sampling_hz=sampling_hz,
                axis_map=axis_map,
                suppress_debug=not bool(args.debug),
                i2c_frequency_hz=i2c_frequency_hz,
                init_retries=cfg.bno085.init_retries,
                polling_mode=polling_mode,
                enable_rotation_vector=enable_rotation_vector,
            )
    except TimeoutError as exc:
        init_done.set()
        print(f"BNO085 init timeout after {time.monotonic() - init_start_s:.1f}s: {exc}", flush=True)
        print("Try power-cycling the IMU/robot, then rerun i2cdetect and this probe.", flush=True)
        return 4
    except Exception:
        init_done.set()
        raise
    init_done.set()
    print(f"BNO085 init complete in {time.monotonic() - init_start_s:.2f}s", flush=True)

    prev_q: np.ndarray | None = None
    prev_g: np.ndarray | None = None
    first_q: np.ndarray | None = None
    valid_count = 0
    stale_count = 0
    quat_changes = 0
    gyro_changes = 0
    timestamp_changes = 0
    max_gyro_norm = 0.0
    max_quat_angle_from_first = 0.0
    max_read_s = 0.0
    slow_read_count = 0
    very_slow_read_count = 0
    last_ts = None
    start_s = time.monotonic()

    try:
        for i in range(total):
            read_t0 = time.monotonic()
            sample = imu.read()
            read_s = time.monotonic() - read_t0
            q = np.asarray(sample.quat_xyzw, dtype=np.float32)
            g = np.asarray(sample.gyro_rad_s, dtype=np.float32)
            ts = getattr(sample, "timestamp_s", None)
            diag = getattr(imu, "diag", {})

            sample_valid = bool(getattr(sample, "valid", True))
            sample_fresh = bool(getattr(sample, "fresh", True))
            if sample_valid and sample_fresh:
                valid_count += 1
            if diag.get("payload_status") == "stale" or not sample_valid or not sample_fresh:
                stale_count += 1
            if prev_q is not None and not np.array_equal(q, prev_q):
                quat_changes += 1
            if prev_g is not None and not np.array_equal(g, prev_g):
                gyro_changes += 1
            if ts is not None and last_ts is not None and ts != last_ts:
                timestamp_changes += 1
            if first_q is None:
                first_q = q.copy()
            max_gyro_norm = max(max_gyro_norm, float(np.linalg.norm(g)))
            max_quat_angle_from_first = max(max_quat_angle_from_first, _quat_angle_delta_rad(first_q, q))
            max_read_s = max(max_read_s, read_s)
            if read_s > 0.2:
                slow_read_count += 1
            if read_s > 1.0:
                very_slow_read_count += 1

            print_every = int(args.print_every)
            if print_every > 0 and i % print_every == 0:
                print(
                    f"{i:03d} read_s={read_s:.3f} valid={sample_valid} fresh={sample_fresh} ts={ts} "
                    f"quat={_fmt_vec(q, digits=5)} gyro={_fmt_vec(g, digits=5)} diag={diag}",
                    flush=True,
                )

            prev_q = q.copy()
            prev_g = g.copy()
            last_ts = ts
            time.sleep(max(0.0, dt_s))
    finally:
        imu.close()

    elapsed_s = time.monotonic() - start_s
    print("", flush=True)
    print("Summary:", flush=True)
    print(f"  elapsed_s: {elapsed_s:.3f}", flush=True)
    print(f"  fresh_valid_samples: {valid_count}/{total}", flush=True)
    print(f"  nonfresh_or_invalid_samples: {stale_count}/{total}", flush=True)
    print(f"  timestamp_changes: {timestamp_changes}", flush=True)
    print(f"  quat_payload_changes: {quat_changes}", flush=True)
    print(f"  gyro_payload_changes: {gyro_changes}", flush=True)
    print(f"  max_gyro_norm_rad_s: {max_gyro_norm:.6f}", flush=True)
    print(f"  max_quat_angle_from_first_rad: {max_quat_angle_from_first:.6f}", flush=True)
    print(f"  max_read_s: {max_read_s:.6f}", flush=True)
    print(f"  slow_reads_over_0.2s: {slow_read_count}/{total}", flush=True)
    print(f"  very_slow_reads_over_1.0s: {very_slow_read_count}/{total}", flush=True)

    valid_ratio = valid_count / float(total)
    if valid_count == 0 or timestamp_changes == 0:
        print("Result: IMU did not produce a fresh valid stream.", flush=True)
        return 2
    if quat_changes == 0 and gyro_changes == 0:
        print("Result: IMU payload did not change. BNO08x reports are frozen or not being refreshed.", flush=True)
        return 2
    if valid_ratio < 0.5:
        print(
            f"Result: IMU payload changed, but fresh valid sample rate is low ({valid_ratio:.1%}).",
            flush=True,
        )
        return 3
    print("Result: IMU payload changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
