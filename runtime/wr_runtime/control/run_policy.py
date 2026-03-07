from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import select

import numpy as np


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
for _p in (str(_REPO_ROOT), str(_RUNTIME_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from policy_contract.numpy.action import postprocess_action
from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicyBundle, validate_spec
from policy_contract.numpy.frames import gravity_local_from_quat, normalize_quat_xyzw

from configs import load_config
from wr_runtime.hardware.actuators import (
    HiwonderBoardActuators,
    joint_target_rad_to_servo_pos_elec_units,
)
from wr_runtime.hardware.bno085 import BNO085IMU
from wr_runtime.hardware.foot_switches import FootSwitches
from wr_runtime.hardware.imu import Imu
from wr_runtime.hardware.robot_io import HardwareRobotIO
from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.utils.mjcf import load_mjcf_model_info
from wr_runtime.validation.startup_validator import validate_runtime_interface
_DEBUG_JOINT_GROUPS: list[tuple[str, list[str]]] = [
    ("shoulder_pitch", ["left_shoulder_pitch", "right_shoulder_pitch"]),
    ("shoulder_roll", ["left_shoulder_roll", "right_shoulder_roll"]),
    ("elbow_pitch", ["left_elbow_pitch", "right_elbow_pitch"]),
    ("waist_yaw", ["waist_yaw"]),
    ("hip_pitch", ["left_hip_pitch", "right_hip_pitch"]),
    ("hip_roll", ["left_hip_roll", "right_hip_roll"]),
    ("knee_pitch", ["left_knee_pitch", "right_knee_pitch"]),
    ("ankle_pitch", ["left_ankle_pitch", "right_ankle_pitch"]),
    ("wrist_pitch", ["left_wrist_pitch", "right_wrist_pitch"]),
]


def _print_joint_debug(
    *,
    spec,
    cfg,
    actuators: HiwonderBoardActuators,
    action_raw: np.ndarray,
    action_post: np.ndarray,
    ctrl_targets_rad: np.ndarray,
    joint_pos_rad: np.ndarray,
    joint_pos_norm: np.ndarray,
) -> None:
    actuator_names = list(spec.robot.actuator_names)
    name_to_idx = {name: index for index, name in enumerate(actuator_names)}

    target_elec = joint_target_rad_to_servo_pos_elec_units(
        np.asarray(ctrl_targets_rad, dtype=np.float32),
        actuators.offsets_unit,
        actuators.motor_signs,
        actuators.centers_rad,
        actuators.servo_model,
    )
    obs_elec = joint_target_rad_to_servo_pos_elec_units(
        np.asarray(joint_pos_rad, dtype=np.float32),
        actuators.offsets_unit,
        actuators.motor_signs,
        actuators.centers_rad,
        actuators.servo_model,
    )

    control_headers = [
        "Joint(servo_id)",
        "action_raw",
        "action_post",
        "target_deg",
        "motor unit",
        "obs_elect",
        "obs_deg",
        "obs_to_model",
    ]
    metadata_headers = [
        "Joint(servo_id)",
        "target_deg_range",
        "policy_action_sign",
        "offset_unit",
        "motor_center_deg",
    ]
    control_rows: list[list[str]] = []
    metadata_rows: list[list[str]] = []

    for group_name, group_candidates in _DEBUG_JOINT_GROUPS:
        present_names = [joint_name for joint_name in group_candidates if joint_name in name_to_idx]
        if not present_names:
            missing_label = f"{group_name}(<not present>)"
            control_rows.append([missing_label, "-", "-", "-", "-", "-", "-"])
            metadata_rows.append([missing_label, "-", "-", "-", "-"])
            continue

        for joint_name in present_names:
            idx = name_to_idx[joint_name]
            joint_spec = spec.robot.joints[joint_name]
            servo = cfg.servo_controller.servos.get(joint_name)
            servo_id = int(servo.id) if servo is not None else -1
            offset_unit = int(servo.offset_unit) if servo is not None else 0
            motor_center_deg = float(servo.motor_center_mujoco_deg) if servo is not None else 0.0

            raw_action = float(action_raw[idx])
            post_action = float(action_post[idx])
            target_rad = float(ctrl_targets_rad[idx])
            target_deg = float(np.rad2deg(target_rad))
            target_elec_unit = int(np.rint(float(target_elec[idx])))
            target_conceptual_unit = int(target_elec_unit - offset_unit)

            obs_rad = float(joint_pos_rad[idx])
            obs_deg = float(np.rad2deg(obs_rad))
            obs_elec_unit = int(np.rint(float(obs_elec[idx])))
            obs_to_model = float(joint_pos_norm[idx])

            range_min_deg = float(np.rad2deg(float(joint_spec.range_min_rad)))
            range_max_deg = float(np.rad2deg(float(joint_spec.range_max_rad)))
            policy_action_sign = float(joint_spec.policy_action_sign)

            joint_label = f"{joint_name}({servo_id})"
            control_rows.append(
                [
                    joint_label,
                    f"{raw_action:+.2f}",
                    f"{post_action:+.2f}",
                    f"{target_deg:+.2f}",
                    f"{target_conceptual_unit:+d}/{target_elec_unit:d}",
                    f"{obs_elec_unit:d}",
                    f"{obs_deg:+.2f}",
                    f"{obs_to_model:+.2f}",
                ]
            )
            metadata_rows.append(
                [
                    joint_label,
                    f"[{range_min_deg:+.2f},{range_max_deg:+.2f}]",
                    f"{policy_action_sign:+.2f}",
                    f"{offset_unit:+d}",
                    f"{motor_center_deg:+.2f}",
                ]
            )

    def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def _row_text(values: list[str]) -> str:
            padded = [f"{value:<{widths[index]}}" for index, value in enumerate(values)]
            return "[debug] | " + " | ".join(padded) + " |"

        sep = "[debug] +-" + "-+-".join("-" * width for width in widths) + "-+"
        print(f"\n[debug] ===== {title} =====")
        print(sep)
        print(_row_text(headers))
        print(sep)
        for row in rows:
            print(_row_text(row))
        print(sep)

    _print_table("per-joint control/obs", control_headers, control_rows)
    _print_table("per-joint metadata", metadata_headers, metadata_rows)


def _poll_stop_token(buf: str) -> tuple[str, bool]:
    """Non-blocking stdin poll for stop tokens.

    Returns (new_buf, should_stop).

    Supported stop tokens (entered then newline): q, quit, exit.
    """
    if not sys.stdin.isatty():
        return buf, False
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return buf, False
    if not r:
        return buf, False

    try:
        chunk = sys.stdin.read(1)
    except Exception:
        return buf, False
    if chunk == "":
        # stdin closed
        return buf, True
    if chunk in {"\n", "\r"}:
        token = buf.strip().lower()
        buf = ""
        if token in {"q", "quit", "exit"}:
            return buf, True
        return buf, False
    return buf + chunk, False


def _imu_sanity_check(
    imu: Imu,
    *,
    samples: int,
    gravity_z_tol: float,
    gyro_norm_tol: float,
    sleep_s: float,
) -> None:
    """Fail fast if IMU outputs look wrong before commanding motors.

    Checks:
      - quat norms are close to 1
      - gravity_local z is near -1 when upright
      - gyro norm is small when still
    """

    target = max(1, int(samples))
    quats = []
    gyro_norms: list[float] = []
    gravities = []
    invalid = 0

    # Some BNO08X stacks can return None/invalid for a short warm-up period even after
    # enable_feature(). Prefer waiting for enough valid samples over failing instantly.
    max_attempts = max(target * 10, 50)
    for _ in range(max_attempts):
        s = imu.read()
        if not getattr(s, "valid", True):
            invalid += 1
            time.sleep(max(0.0, float(sleep_s)))
            continue

        q = normalize_quat_xyzw(np.asarray(s.quat_xyzw, dtype=np.float32))
        quats.append(q)
        g = gravity_local_from_quat(q)
        gravities.append(g)
        gyro = np.asarray(s.gyro_rad_s, dtype=np.float32)
        gyro_norms.append(float(np.linalg.norm(gyro)))

        if len(quats) >= target:
            break
        time.sleep(max(0.0, float(sleep_s)))

    if len(quats) < target:
        extra = ""
        if hasattr(imu, "error_count") and hasattr(imu, "last_error"):
            try:
                extra = f" (error_count={getattr(imu, 'error_count')}, last_error={getattr(imu, 'last_error')})"
            except Exception:
                extra = ""
        raise RuntimeError(
            "IMU sanity check failed: insufficient valid samples. "
            f"valid={len(quats)}/{target}, invalid={invalid}, max_attempts={max_attempts}{extra}. "
            "Check I2C address, wiring/power integrity, and BNO08X bring-up."
        )

    quats_arr = np.stack(quats, axis=0)
    norms = np.linalg.norm(quats_arr, axis=1)
    norm_dev = float(np.max(np.abs(norms - 1.0)))

    g_local_mean = np.mean(np.stack(gravities, axis=0), axis=0)
    g_z_err = abs(float(g_local_mean[2]) + 1.0)

    gyro_mean_norm = float(np.mean(gyro_norms))

    if norm_dev > 0.05:
        raise RuntimeError(f"IMU sanity check failed: quat norm deviation {norm_dev:.3f} > 0.05")
    if g_z_err > gravity_z_tol:
        raise RuntimeError(
            f"IMU sanity check failed: gravity z error {g_z_err:.3f} > {gravity_z_tol:.3f} (sensor upright?)"
        )
    if gyro_mean_norm > gyro_norm_tol:
        raise RuntimeError(
            f"IMU sanity check failed: mean gyro norm {gyro_mean_norm:.3f} rad/s > {gyro_norm_tol:.3f} (sensor should be still)"
        )


def _init_imu_with_timeout(
    *,
    init_timeout_s: float,
    i2c_address: int,
    upside_down: bool,
    axis_map: list[str] | None,
    suppress_debug: bool,
    i2c_frequency_hz: int,
    init_retries: int,
    sampling_hz: int,
) -> BNO085IMU:
    """Create BNO085IMU with an optional wall-clock timeout.

    Some hardware/I2C failure modes can block inside the IMU constructor.
    Timeout gives a deterministic failure path instead of a silent hang.
    """
    if init_timeout_s <= 0.0:
        return BNO085IMU(
            i2c_address=i2c_address,
            upside_down=upside_down,
            axis_map=axis_map,
            suppress_debug=suppress_debug,
            i2c_frequency_hz=i2c_frequency_hz,
            init_retries=init_retries,
            sampling_hz=sampling_hz,
        )

    result: dict[str, BNO085IMU] = {}
    err: dict[str, Exception] = {}

    def _worker() -> None:
        try:
            result["imu"] = BNO085IMU(
                i2c_address=i2c_address,
                upside_down=upside_down,
                axis_map=axis_map,
                suppress_debug=suppress_debug,
                i2c_frequency_hz=i2c_frequency_hz,
                init_retries=init_retries,
                sampling_hz=sampling_hz,
            )
        except Exception as exc:  # noqa: BLE001
            err["exc"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(init_timeout_s)

    if thread.is_alive():
        raise TimeoutError(
            "IMU initialization timed out. "
            f"timeout={init_timeout_s:.1f}s. "
            "Check I2C wiring/address/power, or use --imu-init-timeout-s 0 to disable timeout."
        )
    if "exc" in err:
        raise err["exc"]
    return result["imu"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WildRobot ONNX policy on hardware")
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Bundle directory (contains wildrobot_config.json + policy_spec.json + policy.onnx)",
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help=(
            "Optional runtime JSON with *hardware calibration* (default: ~/.wildrobot/config.json). "
            "If used together with --bundle, the bundle provides policy/model/spec assets and --runtime-config "
            "provides servo/IMU/foot-switch settings."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Deprecated alias for --runtime-config. Prefer --runtime-config to avoid confusion with training configs."
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional text log filename. Mirrors all console output to this file.",
    )
    parser.add_argument("--log-path", type=str, default=None, help="Optional .npz path to save replay logs on exit")
    parser.add_argument(
        "--log-steps",
        type=int,
        default=None,
        help="Optional max steps to log before auto-stopping (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--skip-imu-check",
        action="store_true",
        help="Skip IMU sanity check before enabling control loop (not recommended)",
    )
    parser.add_argument(
        "--imu-check-samples",
        type=int,
        default=20,
        help="Number of IMU samples to average for sanity check",
    )
    parser.add_argument(
        "--imu-gravity-z-tol",
        type=float,
        default=0.25,
        help="Allowed error for gravity_local z vs -1 (abs error)",
    )
    parser.add_argument(
        "--imu-gyro-norm-tol",
        type=float,
        default=1.0,
        help="Allowed mean gyro norm (rad/s) during sanity check",
    )
    parser.add_argument(
        "--imu-check-sleep-s",
        type=float,
        default=0.02,
        help="Sleep between IMU samples during sanity check",
    )
    parser.add_argument(
        "--imu-init-timeout-s",
        type=float,
        default=10.0,
        help=(
            "IMU constructor timeout in seconds. "
            "Set 0 or negative to disable timeout (wait forever)."
        ),
    )
    parser.add_argument(
        "--debug",
        type=float,
        default=None,
        metavar="time-interval-in-second",
        help=(
            "Enable periodic per-joint debug prints. "
            "Value is print interval in seconds (e.g., --debug 1.0). "
            "First print happens immediately after loop start."
        ),
    )
    args = parser.parse_args()
    if args.debug is not None and float(args.debug) <= 0.0:
        parser.error("--debug must be > 0 (seconds)")

    if args.runtime_config is not None and args.config is not None:
        parser.error("Pass only one of --runtime-config or --config (deprecated).")

    console_log_fh = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log:
        log_file_path = Path(args.log).expanduser()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        console_log_fh = log_file_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(original_stdout, console_log_fh)
        sys.stderr = _TeeStream(original_stderr, console_log_fh)
        print(f"Console log tee enabled: {log_file_path}")

    bundle_dir: Path | None = Path(args.bundle) if args.bundle else None

    # Config selection:
    # - If user provides --runtime-config, treat it as the hardware-calibration source of truth.
    # - Else, if --bundle is provided, fall back to bundle-local wildrobot_config.json.
    # - Else, rely on default config search in configs.load_config().
    config_path: str | None
    if args.runtime_config is not None:
        config_path = args.runtime_config
    elif args.config is not None:
        print("Warning: --config is deprecated; use --runtime-config instead.")
        config_path = args.config
    elif bundle_dir is not None:
        cfg_path = bundle_dir / "wildrobot_config.json"
        if not cfg_path.exists():
            raise SystemExit(
                f"Bundle missing wildrobot_config.json: {cfg_path}. "
                "Either add it to the bundle or pass --runtime-config pointing to your hardware config JSON."
            )
        config_path = str(cfg_path)
        print(
            "Using bundle-local wildrobot_config.json for hardware calibration. "
            "If joints move incorrectly, pass --runtime-config ~/.wildrobot/config.json (or your calibrated file)."
        )
    else:
        config_path = None

    resolved_config_path = str(Path(config_path).expanduser().resolve()) if config_path else "<default search>"
    resolved_bundle_path = str(bundle_dir.expanduser().resolve()) if bundle_dir is not None else "<from config>"
    print(f"Resolved config: {resolved_config_path}")
    print(f"Resolved bundle: {resolved_bundle_path}")

    cfg = load_config(config_path)

    # Bundle selection:
    # - If --bundle is provided, always load policy assets from it.
    # - Else, infer bundle location from cfg.policy_onnx_path.
    if bundle_dir is None:
        bundle_dir = Path(cfg.policy_resolved_path).parent

    bundle_dir = bundle_dir.expanduser().resolve()

    bundle = PolicyBundle.load(bundle_dir)
    spec = bundle.spec
    validate_spec(spec)

    # MJCF selection:
    # - If --bundle is provided, prefer bundle-local wildrobot.xml for consistent actuator ordering.
    # - Else, fall back to cfg.mjcf_path.
    mjcf_path = (bundle_dir / "wildrobot.xml") if (bundle_dir / "wildrobot.xml").exists() else Path(cfg.mjcf_resolved_path)
    mjcf_path = mjcf_path.expanduser().resolve()
    print(f"Resolved MJCF: {mjcf_path}")
    mjcf_info = load_mjcf_model_info(mjcf_path)
    joint_names = mjcf_info.actuator_names

    policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=spec.model.input_name,
        output_name=spec.model.output_name,
    )
    print(f"ONNX policy input='{policy.info.input_name}' output='{policy.info.output_name}'")
    if policy.info.obs_dim is not None:
        print(f"ONNX policy obs_dim={policy.info.obs_dim}")
    if policy.info.action_dim is not None:
        print(f"ONNX policy action_dim={policy.info.action_dim}")

    # Fail-fast interface checks (before touching hardware)
    robot_cfg = validate_runtime_interface(
        cfg=cfg,
        mjcf_info=mjcf_info,
        spec=spec,
        onnx_obs_dim=policy.info.obs_dim,
        onnx_action_dim=policy.info.action_dim,
    )
    print(
        "Startup validation OK: "
        f"obs_dim={robot_cfg.obs_dim} action_dim={robot_cfg.action_dim} "
        f"actuators={robot_cfg.actuator_names}"
    )

    control_dt = 1.0 / cfg.control.hz
    print(f"Initializing IMU (timeout={float(args.imu_init_timeout_s):.1f}s)...")
    imu = _init_imu_with_timeout(
        init_timeout_s=float(args.imu_init_timeout_s),
        i2c_address=cfg.bno085.i2c_address,
        upside_down=cfg.bno085.upside_down,
        axis_map=cfg.bno085.axis_map,
        suppress_debug=cfg.bno085.suppress_debug,
        i2c_frequency_hz=cfg.bno085.i2c_frequency_hz,
        init_retries=cfg.bno085.init_retries,
        sampling_hz=int(cfg.control.hz),
    )
    print("IMU init OK.")

    print(
        "IMU config: "
        f"addr=0x{cfg.bno085.i2c_address:02X} upside_down={cfg.bno085.upside_down} "
        f"axis_map={cfg.bno085.axis_map if cfg.bno085.axis_map is not None else ['+X','+Y','+Z']} "
        f"freq={cfg.bno085.i2c_frequency_hz}Hz retries={cfg.bno085.init_retries}"
    )

    if args.skip_imu_check:
        print("Skipping IMU sanity check (requested).")
    else:
        print("Running IMU sanity check (no motors commanded)...")
        _imu_sanity_check(
            imu,
            samples=args.imu_check_samples,
            gravity_z_tol=args.imu_gravity_z_tol,
            gyro_norm_tol=args.imu_gyro_norm_tol,
            sleep_s=args.imu_check_sleep_s,
        )
        print("IMU sanity check passed.")

    move_time_ms = (
        cfg.servo_controller.default_move_time_ms
        if cfg.servo_controller.default_move_time_ms is not None
        else int(control_dt * 1000.0)
    )

    actuators = HiwonderBoardActuators(
        actuator_names=spec.robot.actuator_names,
        servo_ids=cfg.servo_controller.servo_ids,
        joint_offset_units=cfg.servo_controller.joint_offset_units,
        joint_motor_signs=cfg.servo_controller.joint_motor_signs,
        joint_motor_center_mujoco_deg=cfg.servo_controller.joint_motor_center_mujoco_deg,
        port=cfg.servo_controller.port,
        baudrate=cfg.servo_controller.baudrate,
        default_move_time_ms=move_time_ms,
    )

    foot = FootSwitches(cfg.foot_switches.get_all_pins())

    robot_io = HardwareRobotIO(
        actuator_names=spec.robot.actuator_names,
        control_dt=control_dt,
        actuators=actuators,
        imu=imu,
        foot_switches=foot,
    )

    state = PolicyState.init(spec)
    last_time = time.time()

    log_path = Path(args.log_path).expanduser() if args.log_path else None
    log_steps = int(args.log_steps) if args.log_steps is not None else None

    log_quat: list[np.ndarray] = []
    log_gyro: list[np.ndarray] = []
    log_joint_pos: list[np.ndarray] = []
    log_joint_vel: list[np.ndarray] = []
    log_foot: list[np.ndarray] = []
    log_vel_cmd: list[np.ndarray] = []
    log_timestamp_s: list[np.ndarray] = []
    log_dt_s: list[np.ndarray] = []

    print(f"Running control loop at {cfg.control.hz} Hz with {len(joint_names)} actuators")
    print("Type 'q' then Enter to stop (recommended over Ctrl+C for some SSH setups)")
    print("Ctrl+C to stop")
    if args.debug is not None:
        print(f"Debug print enabled: interval={float(args.debug):.2f}s")

    # Print the first debug table as soon as the control loop starts, then
    # continue at the requested interval.
    next_debug_time = time.monotonic() if args.debug is not None else None

    try:
        stop_buf = ""
        while True:
            loop_start = time.time()
            dt = loop_start - last_time
            last_time = loop_start

            stop_buf, should_stop = _poll_stop_token(stop_buf)
            if should_stop:
                break

            try:
                signals = robot_io.read()

                obs = build_observation(
                    spec=spec,
                    state=state,
                    signals=signals,
                    velocity_cmd=np.array([cfg.control.velocity_cmd], dtype=np.float32),
                )
                joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
                    spec=spec,
                    joint_pos_rad=signals.joint_pos_rad,
                )

                action_raw = policy.predict(obs)
                action_post, state = postprocess_action(spec=spec, state=state, action_raw=action_raw)
                ctrl_targets = NumpyCalibOps.action_to_ctrl(spec=spec, action=action_post)
                robot_io.write_ctrl(ctrl_targets)

                if next_debug_time is not None:
                    now_monotonic = time.monotonic()
                    if now_monotonic >= next_debug_time:
                        _print_joint_debug(
                            spec=spec,
                            cfg=cfg,
                            actuators=actuators,
                            action_raw=np.asarray(action_raw, dtype=np.float32),
                            action_post=np.asarray(action_post, dtype=np.float32),
                            ctrl_targets_rad=np.asarray(ctrl_targets, dtype=np.float32),
                            joint_pos_rad=np.asarray(signals.joint_pos_rad, dtype=np.float32),
                            joint_pos_norm=np.asarray(joint_pos_norm, dtype=np.float32),
                        )
                        next_debug_time = now_monotonic + float(args.debug)
            except Exception as exc:  # noqa: BLE001
                print(f"Runtime error in control loop: {exc}")
                actuators.disable()
                raise

            if log_path is not None:
                log_quat.append(np.asarray(signals.quat_xyzw, dtype=np.float32))
                log_gyro.append(np.asarray(signals.gyro_rad_s, dtype=np.float32))
                log_joint_pos.append(np.asarray(signals.joint_pos_rad, dtype=np.float32))
                log_joint_vel.append(np.asarray(signals.joint_vel_rad_s, dtype=np.float32))
                log_foot.append(np.asarray(signals.foot_switches, dtype=np.float32))
                log_vel_cmd.append(np.asarray([cfg.control.velocity_cmd], dtype=np.float32))
                log_timestamp_s.append(np.asarray([signals.timestamp_s], dtype=np.float64))
                log_dt_s.append(np.asarray([dt], dtype=np.float64))
                if log_steps is not None and len(log_quat) >= log_steps:
                    break

            period = 1.0 / cfg.control.hz
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if log_path is not None and log_quat:
            np.savez(
                log_path,
                quat_xyzw=np.stack(log_quat, axis=0),
                gyro_rad_s=np.stack(log_gyro, axis=0),
                joint_pos_rad=np.stack(log_joint_pos, axis=0),
                joint_vel_rad_s=np.stack(log_joint_vel, axis=0),
                foot_switches=np.stack(log_foot, axis=0),
                velocity_cmd=np.stack(log_vel_cmd, axis=0),
                timestamp_s=np.concatenate(log_timestamp_s, axis=0),
                dt_s=np.concatenate(log_dt_s, axis=0),
            )
            print(f"Saved replay log to {log_path}")
        robot_io.close()
        if console_log_fh is not None:
            try:
                console_log_fh.flush()
            finally:
                console_log_fh.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
