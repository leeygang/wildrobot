#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
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
from wr_runtime.hardware.ttl_servo_controller import (  # noqa: E402
    TtlServoController,
    build_ttl_servo_controller,
)


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


def _quat_tilt_rad(quat_wxyz: np.ndarray) -> float | None:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return None
    _, x, y, _ = (quat / norm).tolist()
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arccos(np.clip(body_z_world_z, -1.0, 1.0)))


def _quat_rpy_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9 or not np.all(np.isfinite(quat)):
        return np.full(3, np.nan, dtype=np.float64)
    w, x, y, z = (quat / norm).tolist()
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(np.asarray([roll, pitch, yaw], dtype=np.float64))


def _default_config_path() -> Path:
    return _RUNTIME_ROOT / "configs" / "hardware_config.json"


def _resolve_bundle_dir(config_path: Path, bundle_path: Path | None) -> Path:
    if bundle_path is not None:
        bundle_dir = bundle_path.expanduser()
    else:
        bundle_dir = config_path.expanduser().resolve().parent
    policy_spec = bundle_dir / "policy_spec.json"
    if not policy_spec.exists():
        raise FileNotFoundError(
            "--hold-home needs a policy bundle containing policy_spec.json; "
            f"could not find {policy_spec}. Pass --bundle explicitly."
        )
    return bundle_dir


def _load_bundle_home(bundle_dir: Path) -> tuple[list[str], list[float]]:
    policy_spec_path = bundle_dir / "policy_spec.json"
    data = json.loads(policy_spec_path.read_text())
    robot = data.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"{policy_spec_path} missing robot block")
    actuator_names = robot.get("actuator_names")
    home_ctrl_rad = robot.get("home_ctrl_rad")
    if not isinstance(actuator_names, list) or not isinstance(home_ctrl_rad, list):
        raise ValueError(f"{policy_spec_path} robot block must include actuator_names and home_ctrl_rad lists")
    if len(actuator_names) != len(home_ctrl_rad):
        raise ValueError(
            f"{policy_spec_path} actuator_names/home_ctrl_rad length mismatch: "
            f"{len(actuator_names)} != {len(home_ctrl_rad)}"
        )
    return [str(name) for name in actuator_names], [float(rad) for rad in home_ctrl_rad]


def _home_servo_commands(cfg: WrRuntimeConfig, bundle_dir: Path) -> list[tuple[int, int]]:
    actuator_names, home_ctrl_rad = _load_bundle_home(bundle_dir)
    missing = [name for name in actuator_names if name not in cfg.servo_controller.servos]
    if missing:
        raise ValueError(
            "Runtime config is missing servo entries required by policy home pose: "
            f"{missing}"
        )

    commands: list[tuple[int, int]] = []
    for name, target_rad in zip(actuator_names, home_ctrl_rad, strict=True):
        servo = cfg.servo_controller.servos[name]
        commands.append((int(servo.id), servo.joint_target_rad_to_elect_unit(target_rad)))
    return commands


def _start_home_command_thread(
    controller: TtlServoController,
    commands: list[tuple[int, int]],
    *,
    start_s: float,
    home_after_s: float,
    home_move_ms: int,
    repeat_home_hz: float,
    stop_event: threading.Event,
) -> tuple[threading.Thread, dict[str, object]]:
    stats: dict[str, object] = {
        "sent": 0,
        "first_sent_s": None,
        "last_sent_s": None,
        "error": None,
    }

    def _run() -> None:
        delay_s = max(0.0, home_after_s - (time.monotonic() - start_s))
        if stop_event.wait(delay_s):
            return

        interval_s = 1.0 / repeat_home_hz if repeat_home_hz > 0.0 else None
        next_send_s = time.monotonic()
        while not stop_event.is_set():
            now_s = time.monotonic()
            elapsed_s = now_s - start_s
            sent = int(stats["sent"])
            if sent == 0:
                if interval_s is None:
                    mode = "once"
                else:
                    mode = f"at {repeat_home_hz:.1f} Hz"
                print(
                    f"[t={elapsed_s:.3f}] Commanding bundle home pose "
                    f"({len(commands)} servos, move_ms={home_move_ms}, {mode})",
                    flush=True,
                )

            try:
                controller.move_servos(commands, home_move_ms)
            except Exception as exc:
                stats["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[t={elapsed_s:.3f}] Home command loop failed: {stats['error']}", flush=True)
                return

            stats["sent"] = sent + 1
            if stats["first_sent_s"] is None:
                stats["first_sent_s"] = elapsed_s
            stats["last_sent_s"] = elapsed_s
            if interval_s is None:
                return

            next_send_s += interval_s
            wait_s = max(0.0, next_send_s - time.monotonic())
            stop_event.wait(wait_s)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, stats


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


def _run_home_state_diagnostics(
    *,
    config_path: Path,
    cfg: WrRuntimeConfig,
    bundle_dir: Path,
    total: int,
    dt_s: float,
    home_after_s: float,
    home_move_ms: int,
    max_tilt_deg: float | None,
    print_every: int,
) -> int:
    """Capture synchronized home-state observations through the runtime IO path."""
    from wr_runtime.control.run_policy import _build_hardware_robot_io

    actuator_names, home_ctrl_rad_list = _load_bundle_home(bundle_dir)
    home_ctrl_rad = np.asarray(home_ctrl_rad_list, dtype=np.float32)
    robot_io = _build_hardware_robot_io(
        runtime_config_path=config_path,
        actuator_names=actuator_names,
        control_dt=dt_s,
        loaded_runtime_config=cfg,
    )
    footswitch_available = bool(
        getattr(robot_io, "footswitch_available", True)
    )
    meta = {
        "schema_version": 1,
        "sample_hz": 1.0 / dt_s,
        "home_after_s": home_after_s,
        "home_move_ms": home_move_ms,
        "max_tilt_deg": max_tilt_deg,
        "actuator_names": actuator_names,
        "home_target_rad": home_ctrl_rad.astype(float).tolist(),
        "footswitch_available": footswitch_available,
        "footswitch_order": [
            "left_toe",
            "left_heel",
            "right_toe",
            "right_heel",
        ],
    }
    print(f"HOME_DIAGNOSTIC_META {json.dumps(meta, separators=(',', ':'))}", flush=True)
    print(
        "Home-state diagnostics use the runtime IMU, cached servo readback, and "
        "footswitch path. Footswitches are observations only.",
        flush=True,
    )

    status = "complete"
    abort_sample: int | None = None
    abort_tilt_deg: float | None = None
    home_sent_s: float | None = None
    previous_imu_timestamp: float | None = None
    start_s = time.monotonic()
    try:
        robot_io.wait_for_valid_imu_sample(timeout_s=3.0)
        wait_for_cache = getattr(robot_io.actuators, "wait_for_initial_cache", None)
        if callable(wait_for_cache) and not wait_for_cache(timeout_s=3.0):
            raise RuntimeError("servo cache did not become ready for home diagnostics")

        start_s = time.monotonic()
        for i in range(total):
            elapsed_s = time.monotonic() - start_s
            if home_sent_s is None and elapsed_s >= home_after_s:
                robot_io.actuators.set_targets_rad(
                    home_ctrl_rad,
                    move_time_ms=home_move_ms,
                )
                home_sent_s = elapsed_s
                print(
                    f"[t={elapsed_s:.3f}] Commanding bundle home pose "
                    f"({len(actuator_names)} servos, move_ms={home_move_ms}, once)",
                    flush=True,
                )

            signals = robot_io.read()
            quat = np.asarray(signals.quat_wxyz, dtype=np.float32)
            gyro = np.asarray(signals.gyro_rad_s, dtype=np.float32)
            joint_pos = np.asarray(signals.joint_pos_rad, dtype=np.float32)
            footswitches = (
                np.asarray(signals.foot_switches, dtype=np.int8)
                if footswitch_available
                else None
            )
            rpy_deg = _quat_rpy_deg(quat)
            tilt_rad = _quat_tilt_rad(quat)
            tilt_deg = (
                float(np.degrees(tilt_rad)) if tilt_rad is not None else float("nan")
            )
            joint_err_deg = np.degrees(joint_pos - home_ctrl_rad)
            imu_timestamp_s = float(signals.timestamp_s)
            imu_fresh = (
                previous_imu_timestamp is None
                or imu_timestamp_s != previous_imu_timestamp
            )
            previous_imu_timestamp = imu_timestamp_s
            servo_metrics = dict(getattr(robot_io, "last_servo_metrics", {}) or {})
            payload = {
                "sample": i,
                "elapsed_s": elapsed_s,
                "imu_timestamp_s": imu_timestamp_s,
                "imu_fresh": imu_fresh,
                "quat_wxyz": quat.astype(float).tolist(),
                "gyro_rad_s": gyro.astype(float).tolist(),
                "rpy_deg": rpy_deg.astype(float).tolist(),
                "tilt_deg": tilt_deg,
                "joint_pos_rad": joint_pos.astype(float).tolist(),
                "joint_error_deg": joint_err_deg.astype(float).tolist(),
                "footswitches": (
                    footswitches.astype(int).tolist()
                    if footswitches is not None
                    else None
                ),
                "servo_cache_age_max_s": servo_metrics.get(
                    "servo_cache_age_max_s"
                ),
                "servo_cache_age_leg_max_s": servo_metrics.get(
                    "servo_cache_age_leg_max_s"
                ),
                "servo_read_fail_count": servo_metrics.get(
                    "servo_read_fail_count"
                ),
            }
            print(
                f"HOME_DIAGNOSTIC_SAMPLE {json.dumps(payload, separators=(',', ':'))}",
                flush=True,
            )
            if print_every > 0 and i % print_every == 0:
                print(
                    f"[{i:04d}] t={elapsed_s:.2f}s "
                    f"rpy_deg={_fmt_vec(rpy_deg, digits=2)} tilt_deg={tilt_deg:.2f} "
                    f"gyro={_fmt_vec(gyro, digits=4)} "
                    f"joint_err_max_deg={float(np.max(np.abs(joint_err_deg))):.2f} "
                    "fs="
                    + (
                        str(footswitches.astype(int).tolist())
                        if footswitches is not None
                        else "disabled"
                    ),
                    flush=True,
                )

            if (
                max_tilt_deg is not None
                and np.isfinite(tilt_deg)
                and tilt_deg > max_tilt_deg
            ):
                status = "tilt_abort"
                abort_sample = i
                abort_tilt_deg = tilt_deg
                print(
                    f"Safety abort: IMU tilt={tilt_deg:.1f}deg exceeds "
                    f"max_tilt_deg={max_tilt_deg:.1f} at sample {i}; "
                    "unloading servos.",
                    flush=True,
                )
                break

            next_tick_s = start_s + (i + 1) * dt_s
            time.sleep(max(0.0, next_tick_s - time.monotonic()))
    finally:
        robot_io.close()

    result = {
        "status": status,
        "samples": (abort_sample + 1) if abort_sample is not None else total,
        "elapsed_s": time.monotonic() - start_s,
        "home_command_sent_s": home_sent_s,
        "tilt_abort_sample": abort_sample,
        "tilt_abort_deg": abort_tilt_deg,
        "servos_unloaded": True,
    }
    print(f"HOME_DIAGNOSTIC_RESULT {json.dumps(result, separators=(',', ':'))}", flush=True)
    print(
        "Result: home-state diagnostics tilt safety abort."
        if status == "tilt_abort"
        else "Result: home-state diagnostics complete.",
        flush=True,
    )
    return 6 if status == "tilt_abort" else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe BNO085/BNO08x IMU samples and report whether quat/gyro payloads are changing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Runtime config path used for BNO085 transport and mounting settings.",
    )
    parser.add_argument("--address", type=_parse_i2c_address, default=None, help="Override I2C address, e.g. 0x4B.")
    parser.add_argument(
        "--transport",
        choices=("i2c", "spi"),
        default=None,
        help="Override BNO08x transport from config.",
    )
    parser.add_argument(
        "--spi-baudrate",
        type=int,
        default=None,
        help="Override SPI baudrate from config, e.g. 1000000.",
    )
    parser.add_argument(
        "--spi-read-skip-bytes",
        type=int,
        default=None,
        help="Skip this many leading bytes on each SPI read before parsing BNO08x SHTP data.",
    )
    parser.add_argument(
        "--spi-cs-pin",
        type=str,
        default=None,
        help="Override SPI chip-select Blinka pin from config, e.g. D8.",
    )
    parser.add_argument(
        "--spi-int-pin",
        type=str,
        default=None,
        help="Override SPI interrupt Blinka pin from config, e.g. D17.",
    )
    parser.add_argument(
        "--spi-reset-pin",
        type=str,
        default=None,
        help="Override SPI reset Blinka pin from config, e.g. D27.",
    )
    parser.add_argument(
        "--spi-wake-pin",
        type=str,
        default=None,
        help="Override SPI wake/PS0 Blinka pin from config, e.g. D25.",
    )
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
        "--bundle",
        type=Path,
        default=None,
        help="Policy bundle directory used by --hold-home. Inferred from --config parent when possible.",
    )
    parser.add_argument(
        "--hold-home",
        action="store_true",
        help="During the probe, command all policy servos to the bundle home pose once.",
    )
    parser.add_argument(
        "--home-after-s",
        type=float,
        default=2.0,
        help="Seconds after probe start before --hold-home sends the home command.",
    )
    parser.add_argument(
        "--home-move-ms",
        type=int,
        default=800,
        help="Hiwonder move duration for --hold-home, in milliseconds.",
    )
    parser.add_argument(
        "--repeat-home-hz",
        type=float,
        default=0.0,
        help=(
            "After --home-after-s, keep re-sending the bundle home pose at this rate. "
            "Use 50 with --home-move-ms 20 to mimic policy servo command traffic. "
            "A positive value implies --hold-home."
        ),
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=None,
        help=(
            "Stop the probe and unload home servos when a fresh IMU sample "
            "exceeds this tilt; disabled by default."
        ),
    )
    parser.add_argument(
        "--home-state-diagnostics",
        action="store_true",
        help=(
            "Capture 50 Hz runtime-frame IMU, cached servo readback, home target, "
            "and footswitch observations. Implies --hold-home and unloads on exit."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show verbose Adafruit BNO08x packet debug output.",
    )
    args = parser.parse_args()
    if args.dt is not None and not (
        np.isfinite(float(args.dt)) and float(args.dt) > 0.0
    ):
        parser.error("--dt must be positive")
    if args.max_tilt_deg is not None and not (
        np.isfinite(float(args.max_tilt_deg))
        and 0.0 < float(args.max_tilt_deg) <= 180.0
    ):
        parser.error("--max-tilt-deg must be in (0, 180]")

    cfg = WrRuntimeConfig.load(args.config)
    transport = str(args.transport or cfg.bno085.transport).strip().lower()
    address = int(args.address if args.address is not None else cfg.bno085.i2c_address)
    axis_map = cfg.bno085.axis_map if args.runtime_frame else None
    calibration_mode = bool(args.calibration_mode)
    polling_mode = False if calibration_mode else not bool(args.background)
    sampling_hz = 20 if calibration_mode else max(1, int(args.sampling_hz))
    enable_rotation_vector = (
        True if calibration_mode else bool(cfg.bno085.enable_rotation_vector)
    )
    i2c_frequency_hz = int(args.i2c_frequency_hz or cfg.bno085.i2c_frequency_hz)
    spi_baudrate = int(args.spi_baudrate or cfg.bno085.spi_baudrate)
    spi_read_skip_bytes = int(
        args.spi_read_skip_bytes
        if args.spi_read_skip_bytes is not None
        else cfg.bno085.spi_read_skip_bytes
    )
    spi_cs_pin = str(args.spi_cs_pin or cfg.bno085.spi_cs_pin)
    spi_int_pin = str(args.spi_int_pin or cfg.bno085.spi_int_pin)
    spi_reset_pin = str(args.spi_reset_pin or cfg.bno085.spi_reset_pin)
    spi_wake_pin = str(args.spi_wake_pin or cfg.bno085.spi_wake_pin)
    dt_s = float(args.dt) if args.dt is not None else 0.05
    total = (
        max(1, int(round(float(args.seconds) / max(1e-6, dt_s))))
        if args.seconds is not None
        else max(1, int(args.samples))
    )
    repeat_home_hz = max(0.0, float(args.repeat_home_hz))
    hold_home_requested = (
        bool(args.hold_home)
        or repeat_home_hz > 0.0
        or bool(args.home_state_diagnostics)
    )
    bundle_dir = _resolve_bundle_dir(args.config, args.bundle) if hold_home_requested else None
    home_move_ms = max(0, int(args.home_move_ms))
    home_after_s = max(0.0, float(args.home_after_s))

    if args.home_state_diagnostics:
        assert bundle_dir is not None
        return _run_home_state_diagnostics(
            config_path=args.config,
            cfg=cfg,
            bundle_dir=bundle_dir,
            total=total,
            dt_s=dt_s,
            home_after_s=home_after_s,
            home_move_ms=home_move_ms,
            max_tilt_deg=(
                None
                if args.max_tilt_deg is None
                else float(args.max_tilt_deg)
            ),
            print_every=max(0, int(args.print_every)),
        )

    print(
        "BNO085/BNO08x probe: "
        f"config={args.config} transport={transport} address=0x{address:02X} "
        f"spi_baudrate={spi_baudrate} spi_read_skip_bytes={spi_read_skip_bytes} "
        f"spi_cs={spi_cs_pin} spi_int={spi_int_pin} "
        f"spi_reset={spi_reset_pin} spi_wake={spi_wake_pin} "
        f"upside_down={cfg.bno085.upside_down} "
        f"axis_map={axis_map} polling_mode={polling_mode} sampling_hz={sampling_hz} "
        f"enable_rotation_vector={enable_rotation_vector} i2c_frequency_hz={i2c_frequency_hz} "
        f"samples={total} dt={dt_s:.3f} hold_home={hold_home_requested} "
        f"repeat_home_hz={repeat_home_hz:.1f} max_tilt_deg={args.max_tilt_deg}",
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
                "BNO08x enable_feature()/packet processing or the configured IMU transport."
            ),
        ):
            imu = BNO085IMU(
                transport=transport,
                i2c_address=address,
                upside_down=cfg.bno085.upside_down,
                sampling_hz=sampling_hz,
                axis_map=axis_map,
                suppress_debug=not bool(args.debug),
                i2c_frequency_hz=i2c_frequency_hz,
                spi_baudrate=spi_baudrate,
                spi_read_skip_bytes=spi_read_skip_bytes,
                spi_cs_pin=spi_cs_pin,
                spi_int_pin=spi_int_pin,
                spi_reset_pin=spi_reset_pin,
                spi_wake_pin=spi_wake_pin,
                init_retries=cfg.bno085.init_retries,
                polling_mode=polling_mode,
                enable_rotation_vector=enable_rotation_vector,
            )
    except TimeoutError as exc:
        init_done.set()
        print(f"BNO085 init timeout after {time.monotonic() - init_start_s:.1f}s: {exc}", flush=True)
        print("Try power-cycling the IMU/robot, then verify wiring and rerun this probe.", flush=True)
        return 4
    except Exception:
        init_done.set()
        raise
    init_done.set()
    print(f"BNO085 init complete in {time.monotonic() - init_start_s:.2f}s", flush=True)

    home_controller: TtlServoController | None = None
    home_commands: list[tuple[int, int]] = []
    home_stop_event: threading.Event | None = None
    home_thread: threading.Thread | None = None
    home_stats: dict[str, object] | None = None
    if hold_home_requested:
        try:
            assert bundle_dir is not None
            home_commands = _home_servo_commands(cfg, bundle_dir)
            home_controller = build_ttl_servo_controller(cfg.servo_controller)
        except Exception:
            imu.close()
            raise
        print(
            "Hold-home enabled: "
            f"bundle={bundle_dir} servos={len(home_commands)} "
            f"home_after_s={home_after_s:.3f} home_move_ms={home_move_ms} "
            f"repeat_home_hz={repeat_home_hz:.1f}",
            flush=True,
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
    max_gyro_sample: dict[str, object] | None = None
    max_quat_angle_from_first = 0.0
    max_quat_angle_sample: dict[str, object] | None = None
    max_read_s = 0.0
    slow_read_count = 0
    very_slow_read_count = 0
    tilt_safety_abort: tuple[int, float] | None = None
    safety_unload_ok: bool | None = None
    last_ts = None
    start_s = time.monotonic()

    try:
        for i in range(total):
            read_t0 = time.monotonic()
            sample = imu.read()
            read_s = time.monotonic() - read_t0
            q = np.asarray(sample.quat_wxyz, dtype=np.float32)
            g = np.asarray(sample.gyro_rad_s, dtype=np.float32)
            ts = getattr(sample, "timestamp_s", None)
            diag = getattr(imu, "diag", {})

            sample_valid = bool(getattr(sample, "valid", True))
            sample_fresh = bool(getattr(sample, "fresh", True))
            if sample_valid and sample_fresh:
                valid_count += 1
            if diag.get("payload_status") == "stale" or not sample_valid or not sample_fresh:
                stale_count += 1
            tilt_rad = _quat_tilt_rad(q)
            if (
                args.max_tilt_deg is not None
                and sample_valid
                and sample_fresh
                and tilt_rad is not None
                and float(np.degrees(tilt_rad)) > float(args.max_tilt_deg)
            ):
                tilt_deg = float(np.degrees(tilt_rad))
                tilt_safety_abort = (i, tilt_deg)
                print(
                    f"Safety abort: IMU tilt={tilt_deg:.1f}deg exceeds "
                    f"max_tilt_deg={float(args.max_tilt_deg):.1f} at sample {i}; "
                    "cancelling home command and unloading servos.",
                    flush=True,
                )
                break
            if (
                home_controller is not None
                and home_thread is None
                and sample_valid
                and sample_fresh
            ):
                home_stop_event = threading.Event()
                home_thread, home_stats = _start_home_command_thread(
                    home_controller,
                    home_commands,
                    start_s=start_s,
                    home_after_s=home_after_s,
                    home_move_ms=home_move_ms,
                    repeat_home_hz=repeat_home_hz,
                    stop_event=home_stop_event,
                )
            if prev_q is not None and not np.array_equal(q, prev_q):
                quat_changes += 1
            if prev_g is not None and not np.array_equal(g, prev_g):
                gyro_changes += 1
            if ts is not None and last_ts is not None and ts != last_ts:
                timestamp_changes += 1
            if first_q is None:
                first_q = q.copy()
            gyro_norm = float(np.linalg.norm(g))
            if gyro_norm > max_gyro_norm:
                max_gyro_norm = gyro_norm
                max_gyro_sample = {
                    "idx": i,
                    "ts": ts,
                    "valid": sample_valid,
                    "fresh": sample_fresh,
                    "gyro": g.copy(),
                    "diag": dict(diag),
                }
            quat_angle = _quat_angle_delta_rad(first_q, q)
            if quat_angle > max_quat_angle_from_first:
                max_quat_angle_from_first = quat_angle
                max_quat_angle_sample = {
                    "idx": i,
                    "ts": ts,
                    "valid": sample_valid,
                    "fresh": sample_fresh,
                    "quat": q.copy(),
                    "diag": dict(diag),
                }
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
        if home_stop_event is not None:
            home_stop_event.set()
        if home_thread is not None:
            home_thread.join(timeout=1.0)
        if tilt_safety_abort is not None and home_controller is not None:
            try:
                safety_unload_ok = bool(
                    home_controller.unload_servos(
                        [int(servo_id) for servo_id, _ in home_commands]
                    )
                )
            except Exception as exc:
                safety_unload_ok = False
                print(
                    f"ERROR: tilt safety unload failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        imu.close()
        if home_controller is not None:
            home_controller.close()

    elapsed_s = time.monotonic() - start_s
    print("", flush=True)
    print("Summary:", flush=True)
    print(f"  elapsed_s: {elapsed_s:.3f}", flush=True)
    if hold_home_requested:
        home_sent_count = int((home_stats or {}).get("sent", 0))
        print(f"  hold_home_sent: {home_sent_count > 0}", flush=True)
        print(f"  home_command_count: {home_sent_count}", flush=True)
        print(f"  repeat_home_hz: {repeat_home_hz:.1f}", flush=True)
        if home_stats is not None and home_stats.get("error") is not None:
            print(f"  home_command_error: {home_stats['error']}", flush=True)
    if tilt_safety_abort is not None:
        print(f"  tilt_safety_abort_sample: {tilt_safety_abort[0]}", flush=True)
        print(f"  tilt_safety_abort_deg: {tilt_safety_abort[1]:.3f}", flush=True)
        print(f"  tilt_safety_unload_ok: {safety_unload_ok}", flush=True)
    print(f"  fresh_valid_samples: {valid_count}/{total}", flush=True)
    print(f"  nonfresh_or_invalid_samples: {stale_count}/{total}", flush=True)
    print(f"  timestamp_changes: {timestamp_changes}", flush=True)
    print(f"  quat_payload_changes: {quat_changes}", flush=True)
    print(f"  gyro_payload_changes: {gyro_changes}", flush=True)
    print(f"  max_gyro_norm_rad_s: {max_gyro_norm:.6f}", flush=True)
    if max_gyro_sample is not None:
        print(
            "  max_gyro_sample: "
            f"idx={max_gyro_sample['idx']} ts={max_gyro_sample['ts']} "
            f"valid={max_gyro_sample['valid']} fresh={max_gyro_sample['fresh']} "
            f"gyro={_fmt_vec(max_gyro_sample['gyro'], digits=5)} "
            f"diag={max_gyro_sample['diag']}",
            flush=True,
        )
    print(f"  max_quat_angle_from_first_rad: {max_quat_angle_from_first:.6f}", flush=True)
    if max_quat_angle_sample is not None:
        print(
            "  max_quat_angle_sample: "
            f"idx={max_quat_angle_sample['idx']} ts={max_quat_angle_sample['ts']} "
            f"valid={max_quat_angle_sample['valid']} fresh={max_quat_angle_sample['fresh']} "
            f"quat={_fmt_vec(max_quat_angle_sample['quat'], digits=5)} "
            f"diag={max_quat_angle_sample['diag']}",
            flush=True,
        )
    print(f"  max_read_s: {max_read_s:.6f}", flush=True)
    print(f"  slow_reads_over_0.2s: {slow_read_count}/{total}", flush=True)
    print(f"  very_slow_reads_over_1.0s: {very_slow_read_count}/{total}", flush=True)

    if home_stats is not None and home_stats.get("error") is not None:
        print(f"Result: Home command loop failed: {home_stats['error']}", flush=True)
        return 5
    if tilt_safety_abort is not None:
        print("Result: IMU tilt safety abort.", flush=True)
        return 6

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
