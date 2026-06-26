#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
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
    parser.add_argument("--dt", type=float, default=0.05, help="Seconds between samples.")
    parser.add_argument("--sampling-hz", type=int, default=50, help="BNO08x feature report rate.")
    parser.add_argument(
        "--i2c-frequency-hz",
        type=int,
        default=None,
        help="Override I2C bus frequency from config, e.g. 400000.",
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
    polling_mode = not bool(args.background)
    i2c_frequency_hz = int(args.i2c_frequency_hz or cfg.bno085.i2c_frequency_hz)

    print(
        "BNO085/BNO08x probe: "
        f"config={args.config} address=0x{address:02X} upside_down={cfg.bno085.upside_down} "
        f"axis_map={axis_map} polling_mode={polling_mode} sampling_hz={args.sampling_hz} "
        f"i2c_frequency_hz={i2c_frequency_hz}",
        flush=True,
    )
    print("Rotate the robot while samples print. Quat/gyro should change during motion.", flush=True)

    imu = BNO085IMU(
        i2c_address=address,
        upside_down=cfg.bno085.upside_down,
        sampling_hz=max(1, int(args.sampling_hz)),
        axis_map=axis_map,
        suppress_debug=not bool(args.debug),
        i2c_frequency_hz=i2c_frequency_hz,
        init_retries=cfg.bno085.init_retries,
        polling_mode=polling_mode,
    )

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
        for i in range(max(1, int(args.samples))):
            read_t0 = time.monotonic()
            sample = imu.read()
            read_s = time.monotonic() - read_t0
            q = np.asarray(sample.quat_xyzw, dtype=np.float32)
            g = np.asarray(sample.gyro_rad_s, dtype=np.float32)
            ts = getattr(sample, "timestamp_s", None)
            diag = getattr(imu, "diag", {})

            if bool(getattr(sample, "valid", True)):
                valid_count += 1
            if diag.get("payload_status") == "stale" or not bool(getattr(sample, "valid", True)):
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
                    f"{i:03d} read_s={read_s:.3f} valid={bool(getattr(sample, 'valid', True))} ts={ts} "
                    f"quat={_fmt_vec(q, digits=5)} gyro={_fmt_vec(g, digits=5)} diag={diag}",
                    flush=True,
                )

            prev_q = q.copy()
            prev_g = g.copy()
            last_ts = ts
            time.sleep(max(0.0, float(args.dt)))
    finally:
        imu.close()

    total = max(1, int(args.samples))
    elapsed_s = time.monotonic() - start_s
    print("", flush=True)
    print("Summary:", flush=True)
    print(f"  elapsed_s: {elapsed_s:.3f}", flush=True)
    print(f"  valid_samples: {valid_count}/{total}", flush=True)
    print(f"  stale_or_invalid_samples: {stale_count}/{total}", flush=True)
    print(f"  timestamp_changes: {timestamp_changes}", flush=True)
    print(f"  quat_payload_changes: {quat_changes}", flush=True)
    print(f"  gyro_payload_changes: {gyro_changes}", flush=True)
    print(f"  max_gyro_norm_rad_s: {max_gyro_norm:.6f}", flush=True)
    print(f"  max_quat_angle_from_first_rad: {max_quat_angle_from_first:.6f}", flush=True)
    print(f"  max_read_s: {max_read_s:.6f}", flush=True)
    print(f"  slow_reads_over_0.2s: {slow_read_count}/{total}", flush=True)
    print(f"  very_slow_reads_over_1.0s: {very_slow_read_count}/{total}", flush=True)

    if quat_changes == 0 and gyro_changes == 0:
        print("Result: IMU payload did not change. BNO08x reports are frozen or not being refreshed.", flush=True)
        return 2
    print("Result: IMU payload changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
