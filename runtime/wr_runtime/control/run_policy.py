"""``wildrobot-run-policy`` — deterministic control loop for the latest bundle.

Loads a deployment bundle containing standing and walking policy contracts and
uses the repository's canonical hardware configuration. Supports a hardware-free
``--dry-run`` mode for smoke tests and safe validation on a developer machine.

Examples
--------
Dry run (no hardware), 5 steps, straight walk::

    uv run --project runtime wildrobot-run-policy \
      --bundle /tmp/wr_runtime_smoke9_bundle_check \
      --dry-run --max-steps 5 --velocity-cmd 0.13,0.0,0.0

Hardware run (on the robot), forward walk for 500 control steps::

    uv run --project runtime wildrobot-run-policy \
      --bundle /path/to/bundle \
      --max-steps 500 --velocity-cmd 0.13,0.0,0.0

Stable-only run with the integrated standing stabilizer::

    uv run --project runtime wildrobot-run-policy \
      --bundle /path/to/bundle \
      --stable-only
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import faulthandler
import itertools
import json
import math
import re
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence, TextIO

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.spec import PolicyBundle, validate_spec

from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.control.mock_robot_io import MockRobotIO
from wr_runtime.control.policy_runner import RuntimePolicyRunner
from wr_runtime.control.runtime_policy_config import RuntimePolicyConfig
from wr_runtime.control.runtime_policy_config import StandingRuntimePolicyConfig
from wr_runtime.control.standing_policy_runner import StandingPolicyRunner
from wr_runtime.deployment_bundle import DeploymentBundle, is_deployment_bundle
from wr_runtime.logging.policy_telemetry import PolicyTelemetryRecorder


_LEG_LOG_JOINTS = (
    ("LHP", "left_hip_pitch"),
    ("LHR", "left_hip_roll"),
    ("LK", "left_knee_pitch"),
    ("LAP", "left_ankle_pitch"),
    ("LAR", "left_ankle_roll"),
    ("RHP", "right_hip_pitch"),
    ("RHR", "right_hip_roll"),
    ("RK", "right_knee_pitch"),
    ("RAP", "right_ankle_pitch"),
    ("RAR", "right_ankle_roll"),
)

_FOOT_SWITCH_LABELS = ("left_toe", "left_heel", "right_toe", "right_heel")
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"
_STARTUP_STABILITY_WINDOW_S = 0.4
_STARTUP_STABILITY_MIN_FOOTSWITCH_PRESSED_RATIO = 0.9
_STARTUP_STABILITY_MAX_TILT_DEG = 15.0
_STARTUP_STABILITY_MAX_GYRO_RAD_S = 0.35
_STARTUP_STABILITY_MAX_LEG_ERROR_DEG = 8.0
_STARTUP_POSE_BLEND_S = 2.0
_STARTUP_POSE_HOLD_S = 5.0
_DEFAULT_FALL_TILT_DEG = 45.0
_RUN_POLICY_LOG_DIR = Path(__file__).resolve().parents[3] / "_run_policy_logs"
_STANDING_LAYOUT_IDS = {
    "wr_obs_v1", "wr_obs_v9_standing", "wr_obs_v10_standing_recovery"
}
_WALKING_LAYOUT_IDS = {"wr_obs_v8_cmd3d", "wr_obs_v11_cmd3d_proprio"}


class _LogStream:
    def __init__(
        self,
        console_stream: TextIO,
        log_stream: TextIO,
        *,
        mirror_console: bool,
    ):
        self._console_stream = console_stream
        self._log_stream = log_stream
        self._mirror_console = bool(mirror_console)

    @property
    def encoding(self):
        return getattr(self._console_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._console_stream, "errors", "replace")

    def write(self, text: str) -> int:
        self._log_stream.write(text)
        self._log_stream.flush()
        if self._mirror_console:
            self._console_stream.write(text)
            self._console_stream.flush()
        return len(text)

    def flush(self) -> None:
        self._log_stream.flush()
        if self._mirror_console:
            self._console_stream.flush()

    def isatty(self) -> bool:
        return self._mirror_console and bool(self._console_stream.isatty())


def _policy_requires_footswitches(spec) -> bool:
    if str(spec.observation.layout_id) == "wr_obs_v10_standing_recovery":
        return True
    return any(field.name == "foot_switches" for field in spec.observation.layout)


def _validate_footswitch_configuration(
    *,
    enabled: bool,
    policy_specs: Sequence[tuple[str, object]],
) -> None:
    if enabled:
        return
    required_roles = [
        role for role, spec in policy_specs if _policy_requires_footswitches(spec)
    ]
    if required_roles:
        raise SystemExit(
            "Hardware foot switches are disabled, but these active policies "
            f"require foot-contact input: {required_roles}. Re-enable the switches "
            "before running them."
        )


class _TargetBlendRobotIO:
    """Blend initial walking writes from the final standing target."""

    def __init__(self, robot_io, *, initial_target: np.ndarray, blend_steps: int):
        self._robot_io = robot_io
        self.actuator_names = list(robot_io.actuator_names)
        self._initial_target = np.asarray(initial_target, dtype=np.float32).reshape(-1)
        if self._initial_target.size != len(self.actuator_names):
            raise ValueError(
                "initial blend target size does not match hardware actuator count"
            )
        self._blend_steps = max(0, int(blend_steps))
        self._write_step = 0
        self.last_commanded_q_rad = self._initial_target.copy()

    def read(self):
        return self._robot_io.read()

    def write_ctrl(self, target_q_rad: np.ndarray) -> None:
        target = np.asarray(target_q_rad, dtype=np.float32).reshape(-1)
        if self._write_step < self._blend_steps:
            scale = float(self._write_step + 1) / float(self._blend_steps)
            target = self._initial_target + np.float32(scale) * (
                target - self._initial_target
            )
        self._write_step += 1
        self.last_commanded_q_rad = target.astype(np.float32)
        self._robot_io.write_ctrl(self.last_commanded_q_rad)

    def __getattr__(self, name: str):
        return getattr(self._robot_io, name)


@contextlib.contextmanager
def _output_log_context(log_path: Optional[str], *, mirror_console: bool):
    if log_path is None:
        yield
        return

    path = Path(log_path).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as log_stream:
        stdout = _LogStream(sys.stdout, log_stream, mirror_console=mirror_console)
        stderr = _LogStream(sys.stderr, log_stream, mirror_console=mirror_console)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def _bundle_log_tokens(bundle_path: str, *, stable_only: bool) -> tuple[str, str]:
    stem = Path(bundle_path).expanduser().name.lower()
    role = "stand(?:ing)?" if stable_only else "walk(?:ing)?"
    match = re.search(rf"{role}[_-]?(v\d+)[_-](ckpt\d+)", stem)
    if match is None:
        match = re.search(r"(v\d+).*?(ckpt\d+)", stem)
    if match is not None:
        return match.group(1), match.group(2)

    safe_stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "bundle"
    return safe_stem, "ckpt-unknown"


def _default_run_policy_log_path(
    bundle_path: str,
    *,
    stable_only: bool,
    log_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> Path:
    version, checkpoint = _bundle_log_tokens(bundle_path, stable_only=stable_only)
    mode = "stable" if stable_only else "walk"
    timestamp = (now or dt.datetime.now().astimezone()).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    root = _RUN_POLICY_LOG_DIR if log_dir is None else Path(log_dir)
    return root / f"{version}_{checkpoint}_{mode}_{timestamp}.log"


def _resolve_telemetry_path(value: Optional[str], log_path: Path) -> Path | None:
    if value is None:
        return None
    path = log_path.with_suffix(".npz") if value == "" else Path(value).expanduser()
    if path.suffix != ".npz":
        raise SystemExit("--telemetry output path must end in .npz")
    return path


def _create_telemetry_recorder(
    args: argparse.Namespace,
    *,
    actuator_names: Sequence[str],
    ctrl_dt: float,
    bundle_path: Path,
    hardware_config_path: Path,
) -> PolicyTelemetryRecorder | None:
    telemetry_path = getattr(args, "telemetry_path", None)
    if telemetry_path is None:
        return None
    recorder = PolicyTelemetryRecorder(
        telemetry_path,
        actuator_names=actuator_names,
        ctrl_dt=ctrl_dt,
        bundle_path=bundle_path,
        hardware_config_path=hardware_config_path,
    )
    args.telemetry_recorder = recorder
    print(f"Policy telemetry enabled: {telemetry_path}", flush=True)
    return recorder


def _install_stack_dump_signal() -> None:
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        return
    try:
        faulthandler.register(sigusr1, file=sys.stderr, all_threads=True)
    except Exception:
        pass


def _parse_velocity_cmd(text: Optional[str], default: List[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float32).reshape(3)
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if len(parts) == 1:
        return np.array([float(parts[0]), 0.0, 0.0], dtype=np.float32)
    if len(parts) == 3:
        return np.array([float(p) for p in parts], dtype=np.float32)
    raise SystemExit(
        f"--velocity-cmd must be 'vx' or 'vx,vy,wz'; got {text!r}"
    )


def _load_optional_runtime_control_dt(bundle_path: Path, *, default: float = 0.02) -> float:
    path = bundle_path / "runtime_policy_config.json"
    if not path.exists():
        return float(default)
    try:
        data = json.loads(path.read_text())
    except Exception:
        return float(default)
    if isinstance(data, dict):
        if data.get("ctrl_dt") is not None:
            return float(data["ctrl_dt"])
        if data.get("control_hz") is not None and float(data["control_hz"]) > 0.0:
            return 1.0 / float(data["control_hz"])
    return float(default)


def _default_velocity_cmd_for_layout(spec, runtime_config: RuntimePolicyConfig | None) -> list[float]:
    if runtime_config is not None:
        return list(runtime_config.default_velocity_cmd)
    if spec.observation.layout_id in _STANDING_LAYOUT_IDS:
        return [0.0, 0.0, 0.0]
    raise SystemExit(
        "runtime_policy_config.json is required for this policy layout; "
        f"got layout={spec.observation.layout_id!r}"
    )


def _resolve_run_bundle_path(
    *, bundle_arg: Optional[str], stable_only: bool
) -> Path:
    if bundle_arg is None:
        raise SystemExit("--bundle is required.")
    return Path(bundle_arg)


def _resolve_hardware_config_path(hardware_config_arg: Optional[str]) -> Path:
    from configs import DEFAULT_HARDWARE_CONFIG_PATH

    if hardware_config_arg is not None:
        return Path(hardware_config_arg)
    return DEFAULT_HARDWARE_CONFIG_PATH


def _policy_loop_max_steps(
    *, stable_only: bool, dry_run: bool, max_steps: int
) -> int | None:
    """Run real stable-only control until the operator interrupts it."""
    if stable_only and not dry_run:
        return None
    return int(max_steps)


def _startup_home_hold_steps(
    *,
    stable_only: bool,
    dry_run: bool,
    command_norm: float,
    command_deadzone: float,
    duration_s: float,
    ctrl_dt: float,
) -> int:
    """Return startup preparation steps for standing or commanded walking."""
    if bool(dry_run) or float(duration_s) <= 0.0:
        return 0
    if not bool(stable_only) and float(command_norm) <= float(command_deadzone):
        return 0
    return max(1, int(round(float(duration_s) / max(float(ctrl_dt), 1e-9))))


def _native_17d_runtime_plan(
    spec, *, policy_role: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return the direct policy-to-hardware plan for a native 17D policy."""
    active_names = list(spec.robot.actuator_names)
    if int(spec.model.action_dim) != 17 or len(active_names) != 17:
        raise SystemExit(
            f"{policy_role} policy must use the native 17-actuator contract; "
            f"got action_dim={spec.model.action_dim}, actuators={len(active_names)}"
        )
    wrist_names = [name for name in active_names if "wrist" in name]
    if wrist_names:
        raise SystemExit(
            f"{policy_role} policy contains deprecated wrist actuators: {wrist_names}"
        )
    if spec.robot.home_ctrl_rad is None:
        raise SystemExit(f"{policy_role} policy_spec.robot.home_ctrl_rad is required")
    active_home = np.asarray(spec.robot.home_ctrl_rad, dtype=np.float32).reshape(-1)
    if active_home.size != len(active_names):
        raise SystemExit(
            f"home_ctrl_rad length {active_home.size} != active actuator count "
            f"{len(active_names)}"
        )

    return (
        active_names,
        active_home,
        np.asarray(
            [spec.robot.joints[name].range_min_rad for name in active_names],
            dtype=np.float32,
        ),
        np.asarray(
            [spec.robot.joints[name].range_max_rad for name in active_names],
            dtype=np.float32,
        ),
    )


def _walking_runtime_plan(spec):
    return _native_17d_runtime_plan(spec, policy_role="walking")


def _standing_runtime_plan(spec):
    return _native_17d_runtime_plan(spec, policy_role="standing")


def _build_hardware_robot_io(
    *,
    runtime_config_path: Path,
    actuator_names: List[str],
    control_dt: float,
    loaded_runtime_config=None,
):
    """Construct the real hardware RobotIO from the runtime config.

    Imported lazily (GPIO / serial / IMU bus backends are Linux-only), and wires the
    concrete hardware classes (HiwonderCachedActuators / BNO085IMU / FootSwitches)
    directly — there are no ``*.from_config`` factories.
    """
    from configs import WrRuntimeConfig
    from wr_runtime.hardware.actuators import HiwonderCachedActuators
    from wr_runtime.hardware.bno085 import BNO085IMU
    from wr_runtime.hardware.foot_switches import DisabledFootSwitches, FootSwitches
    from wr_runtime.hardware.hiwonder_ttl_bus import (
        RawServoBus,
        RawServoBusConfig,
        SerialTransport,
        SerialTransportConfig,
    )
    from wr_runtime.hardware.robot_io import HardwareRobotIO
    from wr_runtime.hardware.servo_io_worker import (
        MultiBoardServoIO,
        ServoIOWorker,
        ServoIOWorkerConfig,
    )

    cfg = loaded_runtime_config or WrRuntimeConfig.load(runtime_config_path)
    sc = cfg.servo_controller

    # Fail fast with an actionable message if the runtime config does not cover
    # every actuator in the policy spec (the actuator constructor would
    # otherwise raise a bare KeyError mid-init).  This catches stale configs
    # missing newer joints (e.g. left/right_ankle_roll on the native 17-actuator
    # spec) before any hardware is touched.
    missing = [n for n in actuator_names if n not in sc.servo_ids]
    if missing:
        raise SystemExit(
            "Runtime config is missing servo entries for "
            f"{missing} (required by the policy spec's actuator_names). "
            f"Add them under servo_controller.servos in {runtime_config_path}."
        )

    # HardwareRobotIO.write_ctrl calls set_targets_rad(move_time_ms=None), so a
    # default move time is required; fall back to one control period.
    default_move_time_ms = sc.default_move_time_ms
    if default_move_time_ms is None:
        default_move_time_ms = max(1, int(round(control_dt * 1000.0)))

    read_schedule = getattr(cfg, "servo_read_schedule", None)
    read_schedule_max_cache_age_s = getattr(read_schedule, "max_cache_age_s", {})

    controller_type = str(getattr(sc, "type", "hiwonder_ttl_bus")).lower()
    if controller_type in {"hiwonder", "hiwonder_board", "lsc"}:
        raise SystemExit(
            "servo_controller.type='hiwonder' uses the deprecated LSC "
            "controller-board protocol. Use servo_controller.type='hiwonder_ttl_bus' "
            "with the USB TTL debug-board /dev/serial/by-id port."
        )
    if controller_type not in {"hiwonder_ttl_bus", "hiwonder_ttl_debug_board"}:
        raise SystemExit(
            f"Unsupported servo_controller.type={sc.type!r}. "
            "Use 'hiwonder_ttl_bus' for the USB TTL debug board."
        )
    configured_boards = tuple(getattr(sc, "effective_boards", ()) or ())
    if not configured_boards:
        configured_boards = (
            SimpleNamespace(
                name="legacy_board",
                port=str(sc.port),
                servo_ids=tuple(int(sc.servo_ids[name]) for name in actuator_names),
            ),
        )
    board_by_servo_id: dict[int, str] = {}
    for board in configured_boards:
        for servo_id in board.servo_ids:
            sid = int(servo_id)
            if sid in board_by_servo_id:
                raise SystemExit(
                    f"Servo ID {sid} is assigned to multiple servo_controller.boards"
                )
            board_by_servo_id[sid] = str(board.name)
    missing_board_ids = sorted(
        int(sc.servo_ids[name])
        for name in actuator_names
        if int(sc.servo_ids[name]) not in board_by_servo_id
    )
    if missing_board_ids:
        raise SystemExit(
            "servo_controller.boards is missing policy servo IDs: "
            f"{missing_board_ids}. Rerun calibrate.py --calibrate-servo-board."
        )

    workers_by_board = {}
    port_by_board = {}
    for board in configured_boards:
        board_id_set = {int(servo_id) for servo_id in board.servo_ids}
        board_actuator_names = [
            name for name in actuator_names if int(sc.servo_ids[name]) in board_id_set
        ]
        if not board_actuator_names:
            continue
        read_groups, read_group_schedule = _build_ttl_servo_read_schedule(
            actuator_names=board_actuator_names,
            servo_ids=sc.servo_ids,
            max_cache_age_s=read_schedule_max_cache_age_s,
        )
        transport = SerialTransport(
            SerialTransportConfig(port=str(board.port), baudrate=int(sc.baudrate))
        )
        raw_bus = RawServoBus(transport, RawServoBusConfig())
        workers_by_board[str(board.name)] = ServoIOWorker(
            raw_bus,
            ServoIOWorkerConfig(
                servo_ids=tuple(
                    int(sc.servo_ids[name]) for name in board_actuator_names
                ),
                read_groups=tuple(read_groups),
                read_group_schedule=tuple(read_group_schedule),
            ),
            worker_name=str(board.name),
        )
        port_by_board[str(board.name)] = str(board.port)

    if len(workers_by_board) == 1:
        servo_io = next(iter(workers_by_board.values()))
    else:
        servo_io = MultiBoardServoIO(
            workers_by_board,
            servo_ids=tuple(int(sc.servo_ids[name]) for name in actuator_names),
        )
    if len(port_by_board) == 1:
        port_label = next(iter(port_by_board.values()))
    else:
        port_label = ",".join(
            f"{name}={port}" for name, port in port_by_board.items()
        )
    actuators = HiwonderCachedActuators(
        actuator_names=actuator_names,
        servo_ids=sc.servo_ids,
        default_move_time_ms=default_move_time_ms,
        joint_servo_offset_units=sc.joint_servo_offset_units,
        joint_motor_unit_directions=sc.joint_motor_unit_directions,
        joint_angle_at_servo_center_deg=sc.joint_angle_at_servo_center_deg,
        servo_io=servo_io,
        cache_age_limits_s=read_schedule_max_cache_age_s,
        port=port_label,
        baudrate=sc.baudrate,
    )
    imu = BNO085IMU(
        transport=cfg.bno085.transport,
        i2c_address=cfg.bno085.i2c_address,
        upside_down=cfg.bno085.upside_down,
        sampling_hz=(
            int(cfg.bno085.sampling_hz)
            if cfg.bno085.sampling_hz is not None
            else max(1, int(round(1.0 / control_dt)))
        ),
        axis_map=cfg.bno085.axis_map,
        suppress_debug=cfg.bno085.suppress_debug,
        i2c_frequency_hz=cfg.bno085.i2c_frequency_hz,
        spi_baudrate=cfg.bno085.spi_baudrate,
        spi_read_skip_bytes=cfg.bno085.spi_read_skip_bytes,
        spi_cs_pin=cfg.bno085.spi_cs_pin,
        spi_int_pin=cfg.bno085.spi_int_pin,
        spi_reset_pin=cfg.bno085.spi_reset_pin,
        spi_wake_pin=cfg.bno085.spi_wake_pin,
        init_retries=cfg.bno085.init_retries,
        enable_rotation_vector=cfg.bno085.enable_rotation_vector,
    )
    foot_switches = (
        FootSwitches(pins=cfg.foot_switches.get_all_pins())
        if bool(getattr(cfg.foot_switches, "enabled", True))
        else DisabledFootSwitches()
    )
    servo_io.start()
    return HardwareRobotIO(
        actuator_names=actuator_names,
        control_dt=control_dt,
        actuators=actuators,
        imu=imu,
        foot_switches=foot_switches,
    )


def _build_ttl_servo_read_schedule(
    *,
    actuator_names: Sequence[str],
    servo_ids: dict[str, int],
    max_cache_age_s: dict[str, float],
):
    from wr_runtime.hardware.servo_io_worker import ServoReadGroup

    names = [str(name) for name in actuator_names]
    label = _infer_servo_read_group_label(0, names)
    group = ServoReadGroup(
        name=label,
        servo_ids=tuple(int(servo_ids[name]) for name in names),
        max_cache_age_s=_group_cache_age_limit_s(
            names, max_cache_age_s=max_cache_age_s
        ),
    )
    return [group], [label]


def _infer_servo_read_group_label(group_idx: int, names: Sequence[str]) -> str:
    names_set = set(names)
    if names_set and all(name.startswith("left_") for name in names_set):
        if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
            return "left_leg"
    if names_set and all(name.startswith("right_") for name in names_set):
        if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
            return "right_leg"
    if any("shoulder" in name or "elbow" in name for name in names_set):
        return "torso_arms"
    return f"group_{group_idx}"


def _group_cache_age_limit_s(
    names: Sequence[str], *, max_cache_age_s: dict[str, float]
) -> float:
    if any(part in name for name in names for part in ("hip", "knee", "ankle")):
        key = "leg"
    elif any(part in name for name in names for part in ("shoulder", "elbow")):
        key = "arm"
    else:
        key = "default"
    defaults = {"leg": 0.12, "arm": 1.25, "default": 1.25}
    return float(max_cache_age_s.get(key, max_cache_age_s.get("default", defaults[key])))


def _actuator_indices(
    actuator_names: Optional[List[str]], joint_names: tuple[str, ...]
) -> List[int]:
    if not actuator_names:
        return []
    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    return [by_name[name] for name in joint_names if name in by_name]


def _format_leg_targets_deg(
    target_q_rad: np.ndarray, actuator_names: Optional[List[str]]
) -> str:
    return _format_leg_values_deg(target_q_rad, actuator_names)


def _format_leg_values_deg(
    values_rad: np.ndarray, actuator_names: Optional[List[str]]
) -> str:
    values = np.asarray(values_rad, dtype=np.float32).reshape(-1)
    if not actuator_names:
        return ""

    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    parts = []
    for label, name in _LEG_LOG_JOINTS:
        idx = by_name.get(name)
        if idx is not None and idx < values.size:
            parts.append(f"{label}={float(np.rad2deg(values[idx])):+.1f}")
    return " ".join(parts)


def _format_foot_switches(info: dict) -> str:
    if info.get("footswitch_available") is False:
        return "fs=disabled"
    signals = info.get("signals")
    if signals is None:
        return ""
    switches = np.asarray(signals.foot_switches, dtype=np.float32).reshape(-1)
    if switches.size != 4:
        return ""
    values = [int(round(float(v))) for v in switches]
    return f"fs=[LT={values[0]},LH={values[1]},RT={values[2]},RH={values[3]}]"


def _footswitch_values(info: dict) -> list[int] | None:
    if info.get("footswitch_available") is False:
        return None
    signals = info.get("signals")
    if signals is None:
        return None
    switches = np.asarray(signals.foot_switches, dtype=np.float32).reshape(-1)
    if switches.size != len(_FOOT_SWITCH_LABELS) or not np.all(np.isfinite(switches)):
        return None
    return [int(round(float(v))) for v in switches]


def _quat_wxyz_to_rpy_rad(quat_wxyz: np.ndarray) -> tuple[float, float, float] | None:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return None
    w, x, y, z = (quat / norm).tolist()

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_wxyz_tilt_rad(quat_wxyz: np.ndarray) -> float | None:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return None
    _, x, y, _ = (quat / norm).tolist()
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, body_z_world_z)))


def _info_tilt_deg(info: dict) -> float | None:
    signals = info.get("signals")
    if signals is None:
        return None
    tilt = _quat_wxyz_tilt_rad(getattr(signals, "quat_wxyz", []))
    if tilt is None:
        return None
    return float(math.degrees(tilt))


def _info_gyro_norm_rad_s(info: dict) -> float | None:
    signals = info.get("signals")
    if signals is None:
        return None
    gyro = np.asarray(getattr(signals, "gyro_rad_s", []), dtype=np.float32).reshape(-1)
    if gyro.size != 3 or not np.all(np.isfinite(gyro)):
        return None
    return float(np.linalg.norm(gyro))


def _abort_if_fallen(*, info: dict, step: int, max_tilt_deg: float) -> None:
    tilt_deg = _info_tilt_deg(info)
    if tilt_deg is None or tilt_deg <= float(max_tilt_deg):
        return

    signals = info.get("signals")
    rpy = (
        None
        if signals is None
        else _quat_wxyz_to_rpy_rad(getattr(signals, "quat_wxyz", []))
    )
    rpy_text = ""
    if rpy is not None:
        roll_deg, pitch_deg, yaw_deg = (math.degrees(value) for value in rpy)
        rpy_text = (
            f" roll={roll_deg:.1f}deg pitch={pitch_deg:.1f}deg "
            f"yaw={yaw_deg:.1f}deg"
        )
    raise SystemExit(
        f"Fall safety abort at policy step {int(step)}: tilt={tilt_deg:.1f}deg"
        f"{rpy_text} > {float(max_tilt_deg):.1f}deg; unloading servos."
    )


def _format_base_orientation(signals) -> str:
    rpy = _quat_wxyz_to_rpy_rad(getattr(signals, "quat_wxyz", []))
    tilt = _quat_wxyz_tilt_rad(getattr(signals, "quat_wxyz", []))
    if rpy is None or tilt is None:
        return ""
    rpy_deg = [float(math.degrees(v)) for v in rpy]
    return (
        "rpy_deg="
        f"{np.round(np.asarray(rpy_deg, dtype=np.float32), 1).tolist()} "
        f"tilt_deg={float(math.degrees(tilt)):.1f}"
    )


def _ramped_velocity_cmd(
    velocity_cmd: np.ndarray,
    *,
    step: int,
    ramp_steps: int,
) -> tuple[np.ndarray, float]:
    cmd = np.asarray(velocity_cmd, dtype=np.float32).reshape(3)
    scale = _ramp_scale(step=step, ramp_steps=ramp_steps)
    return (cmd * np.float32(scale)).astype(np.float32), scale


def _ramp_scale(*, step: int, ramp_steps: int) -> float:
    steps = int(ramp_steps)
    if steps <= 1:
        return 1.0
    return min(1.0, max(0.0, float(step + 1) / float(steps)))


def _format_policy_diagnostics(
    *,
    info: dict,
    actuator_names: Optional[List[str]],
    leg_indices: List[int],
    spec,
) -> str:
    raw = np.asarray(info.get("raw_action", []), dtype=np.float32).reshape(-1)
    applied = np.asarray(info.get("applied_action", []), dtype=np.float32).reshape(-1)
    if raw.size == 0:
        return ""

    parts = [f"|raw|max={float(np.max(np.abs(raw))):.3f}"]
    control_mode = info.get("control_mode")
    if control_mode:
        parts.append(f"mode={control_mode}")
    if leg_indices:
        parts.append(f"leg|raw|max={float(np.max(np.abs(raw[leg_indices]))):.3f}")
        parts.append(_format_lr_action_delta("raw_lr", raw, actuator_names))
        if applied.size == raw.size:
            parts.append(_format_lr_action_delta("applied_lr", applied, actuator_names))

    obs_debug = info.get("obs_debug")
    if isinstance(obs_debug, dict):
        cmd = np.asarray(obs_debug.get("velocity_cmd", []), dtype=np.float32).reshape(-1)
        phase = np.asarray(
            obs_debug.get("phase_sin_cos", []), dtype=np.float32
        ).reshape(-1)
        bin_idx = obs_debug.get("reference_bin_idx")
        if cmd.size:
            parts.append(f"obs_cmd={np.round(cmd, 3).tolist()}")
        if bin_idx is not None:
            parts.append(f"ref_bin={bin_idx}")
        if phase.size == 2:
            parts.append(f"phase={np.round(phase, 3).tolist()}")

    signals = info.get("signals")
    if signals is not None:
        if actuator_names:
            try:
                joint_vel_norm = np.asarray(
                    NumpyCalibOps.normalize_joint_vel(
                        spec=spec,
                        joint_vel_rad_s=signals.joint_vel_rad_s,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception:
                joint_vel_norm = np.asarray([], dtype=np.float32)
            if joint_vel_norm.size:
                parts.append(f"jvel_norm|max={float(np.max(np.abs(joint_vel_norm))):.3f}")
                if leg_indices:
                    parts.append(
                        f"leg_jvel_norm|max="
                        f"{float(np.max(np.abs(joint_vel_norm[leg_indices]))):.3f}"
                    )
        gyro = np.asarray(signals.gyro_rad_s, dtype=np.float32).reshape(-1)
        if gyro.size == 3:
            parts.append(f"gyro={np.round(gyro, 4).tolist()}")
        orientation = _format_base_orientation(signals)
        if orientation:
            parts.append(orientation)

    return "diag[" + " ".join(p for p in parts if p) + "]"


def _leg_error_max_deg(info: dict, leg_indices: List[int]) -> float | None:
    if not leg_indices:
        return None
    target_value = info.get("previous_commanded_q_rad")
    if target_value is None:
        target_value = info.get("target_q_rad", [])
    target = np.asarray(target_value, dtype=np.float32).reshape(-1)
    signals = info.get("signals")
    if target.size == 0 or signals is None:
        return None
    observed = np.asarray(signals.joint_pos_rad, dtype=np.float32).reshape(-1)
    if observed.size < target.size:
        return None
    err_rad = float(np.max(np.abs(target[leg_indices] - observed[leg_indices])))
    return float(np.rad2deg(err_rad))


def _startup_home_stability_errors(
    *,
    infos: List[dict],
    ctrl_dt: float,
    leg_indices: List[int],
    window_s: float,
    min_footswitch_pressed_ratio: float,
    max_tilt_deg: float,
    max_gyro_rad_s: float,
    max_leg_error_deg: float,
) -> tuple[List[str], List[str], List[str]]:
    if not infos:
        return ["no startup home samples were collected"], [], []

    window_steps = max(
        1,
        min(
            len(infos),
            int(round(float(window_s) / max(float(ctrl_dt), 1e-9))),
        ),
    )
    window = infos[-window_steps:]
    errors: List[str] = []
    warnings: List[str] = []
    summary: List[str] = [f"window_steps={window_steps}"]

    footswitch_disabled = any(
        info.get("footswitch_available") is False for info in window
    )
    foot_rows = [] if footswitch_disabled else [
        values
        for info in window
        if (values := _footswitch_values(info)) is not None
    ]
    if footswitch_disabled:
        summary.append("footswitches=disabled")
    elif not foot_rows:
        warnings.append(
            "footswitch samples unavailable during final startup home window"
        )
    else:
        foot_matrix = np.asarray(foot_rows, dtype=np.float32)
        pressed_ratio = np.mean(foot_matrix >= 0.5, axis=0)
        final_values = [int(v) for v in foot_matrix[-1].tolist()]
        open_final = [
            name
            for name, value in zip(_FOOT_SWITCH_LABELS, final_values)
            if int(value) == 0
        ]
        low_ratio = [
            f"{name}={float(ratio):.2f}"
            for name, ratio in zip(_FOOT_SWITCH_LABELS, pressed_ratio.tolist())
            if float(ratio) < float(min_footswitch_pressed_ratio)
        ]
        summary.append(
            "final_fs="
            f"[LT={final_values[0]},LH={final_values[1]},"
            f"RT={final_values[2]},RH={final_values[3]}]"
        )
        summary.append(
            "fs_pressed_ratio="
            + str(
                {
                    name: round(float(ratio), 2)
                    for name, ratio in zip(_FOOT_SWITCH_LABELS, pressed_ratio.tolist())
                }
            )
        )
        if open_final:
            warnings.append(f"final footswitches open: {open_final}")
        if low_ratio:
            warnings.append(
                "footswitch pressed ratio below "
                f"{float(min_footswitch_pressed_ratio):.2f}: {low_ratio}"
            )

    tilt_values = [
        value for info in window if (value := _info_tilt_deg(info)) is not None
    ]
    if not tilt_values:
        errors.append("IMU tilt unavailable during final startup home window")
    else:
        max_tilt = max(float(v) for v in tilt_values)
        summary.append(f"max_tilt_deg={max_tilt:.1f}")
        if max_tilt > float(max_tilt_deg):
            errors.append(
                f"max tilt {max_tilt:.1f}deg > {float(max_tilt_deg):.1f}deg"
            )

    gyro_values = [
        value for info in window if (value := _info_gyro_norm_rad_s(info)) is not None
    ]
    if not gyro_values:
        errors.append("IMU gyro unavailable during final startup home window")
    else:
        max_gyro = max(float(v) for v in gyro_values)
        summary.append(f"max_gyro_rad_s={max_gyro:.3f}")
        if max_gyro > float(max_gyro_rad_s):
            errors.append(
                f"max gyro {max_gyro:.3f}rad/s > {float(max_gyro_rad_s):.3f}rad/s"
            )

    leg_errors = [
        value
        for info in window
        if (value := _leg_error_max_deg(info, leg_indices)) is not None
    ]
    if leg_indices and not leg_errors:
        errors.append("leg pose error unavailable during final startup home window")
    elif leg_errors:
        max_leg_error = max(float(v) for v in leg_errors)
        summary.append(f"max_leg_err_deg={max_leg_error:.1f}")
        if max_leg_error > float(max_leg_error_deg):
            errors.append(
                "max leg home error "
                f"{max_leg_error:.1f}deg > {float(max_leg_error_deg):.1f}deg"
            )

    return errors, warnings, summary


def _format_lr_action_delta(
    label: str,
    action: np.ndarray,
    actuator_names: Optional[List[str]],
) -> str:
    if not actuator_names:
        return ""
    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    pairs = (
        ("HP", "left_hip_pitch", "right_hip_pitch"),
        ("HR", "left_hip_roll", "right_hip_roll"),
        ("K", "left_knee_pitch", "right_knee_pitch"),
        ("AP", "left_ankle_pitch", "right_ankle_pitch"),
        ("AR", "left_ankle_roll", "right_ankle_roll"),
    )
    parts = []
    for short, left_name, right_name in pairs:
        li = by_name.get(left_name)
        ri = by_name.get(right_name)
        if li is None or ri is None or li >= action.size or ri >= action.size:
            continue
        parts.append(f"{short}={float(action[li] - action[ri]):+.3f}")
    if not parts:
        return ""
    return f"{label}=[" + " ".join(parts) + "]"


def _format_rad_deg(value_rad: float) -> str:
    return f"{float(np.rad2deg(value_rad)):+.1f}"


def _format_ms(value_s: float | None) -> str:
    if value_s is None or not np.isfinite(float(value_s)):
        return "n/a"
    return f"{float(value_s) * 1000.0:.1f}"


def _format_hz_from_period(value_s: float | None) -> str:
    if value_s is None or not np.isfinite(float(value_s)) or float(value_s) <= 0.0:
        return "n/a"
    return f"{1.0 / float(value_s):.1f}"


def _timing_values(samples: Sequence[dict], key: str) -> List[float]:
    out = []
    for sample in samples:
        value = sample.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            out.append(value_f)
    return out


def _timing_avg(samples: Sequence[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.mean(values))


def _timing_max(samples: Sequence[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.max(values))


def _timing_sum(samples: Sequence[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.sum(values))


def _timing_percentile(
    samples: Sequence[dict], key: str, percentile: float
) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(percentile)))


def _format_step_timing(timing_s: dict) -> str:
    loop_period_s = timing_s.get("loop_period")
    return (
        f"timing[loop_hz={_format_hz_from_period(loop_period_s)} "
        f"work_ms={_format_ms(timing_s.get('work'))} "
        f"read_ms={_format_ms(timing_s.get('read'))} "
        f"policy_ms={_format_ms(timing_s.get('policy'))} "
        f"write_ms={_format_ms(timing_s.get('write'))} "
        f"servo_read_ms={_format_ms(timing_s.get('io_actuator_read'))} "
        f"servo_write_ms={_format_ms(timing_s.get('io_write_ctrl'))} "
        f"worker_queue_ms={_format_ms(timing_s.get('io_servo_latest_write_queue_latency_s'))} "
        f"worker_write_ms={_format_ms(timing_s.get('io_servo_latest_write_latency_s'))} "
        f"worker_read_ms={_format_ms(timing_s.get('io_servo_latest_read_latency_s'))} "
        f"servo_cache_age_ms={_format_ms(timing_s.get('io_servo_cache_age_max_s'))} "
        f"leg_cache_age_ms={_format_ms(timing_s.get('io_servo_cache_age_leg_max_s'))}]"
    )


def _format_servo_step_metrics(metrics: dict | None) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return ""
    group = metrics.get("servo_read_group")
    ids = metrics.get("servo_read_ids")
    age_s = metrics.get("servo_cache_age_max_s")
    stale = metrics.get("servo_cache_stale_joint_count")
    uninit = metrics.get("servo_cache_uninitialized_count")
    write_commands = metrics.get("servo_write_commands")
    write_skipped = metrics.get("servo_write_commands_skipped")
    write_deadband = metrics.get("servo_write_deadband_units")
    return (
        "servo_cache="
        f"[group={group} ids={ids} age_ms={_format_ms(age_s)} "
        f"stale={stale} uninit={uninit} "
        f"writes={write_commands} skipped={write_skipped} deadband={write_deadband}]"
    )


def _metric_delta(first: dict, last: dict, key: str) -> int | None:
    try:
        return int(last.get(key, 0)) - int(first.get(key, 0))
    except (TypeError, ValueError):
        return None


def _print_io_bottleneck_summary(timing_samples: Sequence[dict]) -> None:
    components = [
        ("imu", "io_imu_read"),
        ("servo_read", "io_actuator_read"),
        ("footswitch", "io_footswitch_read"),
        ("signal_build", "io_signal_build"),
        ("servo_write", "io_write_ctrl"),
        ("worker_queue", "io_servo_latest_write_queue_latency_s"),
        ("worker_write", "io_servo_latest_write_latency_s"),
        ("worker_read", "io_servo_latest_read_latency_s"),
        ("read_total", "io_read_total"),
    ]
    ranked: list[tuple[float, str, float | None, float | None]] = []
    for label, key in components:
        p95_s = _timing_percentile(timing_samples, key, 95.0)
        max_s = _timing_max(timing_samples, key)
        if p95_s is None and max_s is None:
            continue
        score = max_s if max_s is not None and np.isfinite(max_s) else p95_s
        if score is None or not np.isfinite(score):
            continue
        ranked.append((float(score), label, p95_s, max_s))
    if not ranked:
        return
    ranked.sort(reverse=True)
    fields = [
        f"{label}={_format_ms(p95_s)}/{_format_ms(max_s)}"
        for _, label, p95_s, max_s in ranked[:5]
    ]
    print(
        "  IO bottleneck p95/max ms: " + " ".join(fields),
        flush=True,
    )


def _print_timing_summary(
    *,
    timing_samples: Sequence[dict],
    ctrl_dt: float,
    realtime: bool,
    completed: bool = True,
    servo_metric_samples: Sequence[dict] | None = None,
) -> None:
    if not timing_samples:
        return
    target_hz = 1.0 / float(ctrl_dt) if float(ctrl_dt) > 0.0 else float("nan")
    loop_avg_s = _timing_avg(timing_samples, "loop_period")
    work_avg_s = _timing_avg(timing_samples, "work")
    deadline_misses = sum(
        1 for sample in timing_samples if float(sample.get("work", 0.0)) > float(ctrl_dt)
    )
    print(
        "Timing summary: "
        f"status={'completed' if completed else 'partial'} "
        f"steps={len(timing_samples)} target_hz={target_hz:.1f} realtime={realtime} "
        f"loop_hz_avg={_format_hz_from_period(loop_avg_s)} "
        f"work_hz_avg={_format_hz_from_period(work_avg_s)} "
        f"deadline_misses={deadline_misses}/{len(timing_samples)} "
        f"work_ms_avg={_format_ms(work_avg_s)} "
        f"work_ms_p95={_format_ms(_timing_percentile(timing_samples, 'work', 95.0))} "
        f"work_ms_max={_format_ms(_timing_max(timing_samples, 'work'))}",
        flush=True,
    )
    print(
        "  Step avg/p95/max ms: "
        f"read={_format_ms(_timing_avg(timing_samples, 'read'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'read', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'read'))} "
        f"obs={_format_ms(_timing_avg(timing_samples, 'obs'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'obs', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'obs'))} "
        f"policy={_format_ms(_timing_avg(timing_samples, 'policy'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'policy', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'policy'))} "
        f"compose={_format_ms(_timing_avg(timing_samples, 'compose'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'compose', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'compose'))} "
        f"write={_format_ms(_timing_avg(timing_samples, 'write'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'write', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'write'))}",
        flush=True,
    )
    print(
        "  IO avg/p95/max ms: "
        f"imu={_format_ms(_timing_avg(timing_samples, 'io_imu_read'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'io_imu_read', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'io_imu_read'))} "
        f"servo_read={_format_ms(_timing_avg(timing_samples, 'io_actuator_read'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'io_actuator_read', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'io_actuator_read'))} "
        f"footswitch={_format_ms(_timing_avg(timing_samples, 'io_footswitch_read'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'io_footswitch_read', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'io_footswitch_read'))} "
        f"servo_write={_format_ms(_timing_avg(timing_samples, 'io_write_ctrl'))}/"
        f"{_format_ms(_timing_percentile(timing_samples, 'io_write_ctrl', 95.0))}/"
        f"{_format_ms(_timing_max(timing_samples, 'io_write_ctrl'))}",
        flush=True,
    )
    _print_io_bottleneck_summary(timing_samples)
    servo_metric_samples = servo_metric_samples or []
    if servo_metric_samples or _timing_values(timing_samples, "io_servo_cache_age_max_s"):
        last_metrics = servo_metric_samples[-1] if servo_metric_samples else {}
        first_metrics = servo_metric_samples[0] if servo_metric_samples else {}
        print(
            "  Servo cache avg/p95/max ms: "
            f"all={_format_ms(_timing_avg(timing_samples, 'io_servo_cache_age_max_s'))}/"
            f"{_format_ms(_timing_percentile(timing_samples, 'io_servo_cache_age_max_s', 95.0))}/"
            f"{_format_ms(_timing_max(timing_samples, 'io_servo_cache_age_max_s'))} "
            f"leg={_format_ms(_timing_avg(timing_samples, 'io_servo_cache_age_leg_max_s'))}/"
            f"{_format_ms(_timing_percentile(timing_samples, 'io_servo_cache_age_leg_max_s', 95.0))}/"
            f"{_format_ms(_timing_max(timing_samples, 'io_servo_cache_age_leg_max_s'))} "
            f"arm={_format_ms(_timing_avg(timing_samples, 'io_servo_cache_age_arm_max_s'))}/"
            f"{_format_ms(_timing_percentile(timing_samples, 'io_servo_cache_age_arm_max_s', 95.0))}/"
            f"{_format_ms(_timing_max(timing_samples, 'io_servo_cache_age_arm_max_s'))}",
            flush=True,
        )
        print(
            "  Servo read/cache summary: "
            f"read_count={last_metrics.get('servo_read_count')} "
            f"read_fail_count={last_metrics.get('servo_read_fail_count')} "
            f"stale_joint_count_max={_timing_max(timing_samples, 'io_servo_cache_stale_joint_count')} "
            f"uninitialized_joint_count_max={_timing_max(timing_samples, 'io_servo_cache_uninitialized_count')} "
            f"last_group={last_metrics.get('servo_read_group')} "
            f"last_ids={last_metrics.get('servo_read_ids')} "
            f"write_commands={last_metrics.get('servo_write_commands')} "
            f"write_skipped={last_metrics.get('servo_write_commands_skipped')} "
            f"write_deadband_units={last_metrics.get('servo_write_deadband_units')} "
            f"write_failures={last_metrics.get('servo_write_failures')} "
            f"write_targets={last_metrics.get('servo_write_targets_submitted')} "
            f"write_replaced={last_metrics.get('servo_write_targets_replaced')}",
            flush=True,
        )
        if servo_metric_samples:
            print(
                "  Servo worker sampled delta: "
                f"reads={_metric_delta(first_metrics, last_metrics, 'servo_read_count')} "
                f"read_fail={_metric_delta(first_metrics, last_metrics, 'servo_read_fail_count')} "
                f"deadline_reads={_metric_delta(first_metrics, last_metrics, 'servo_cache_deadline_reads')} "
                f"forced_reads={_metric_delta(first_metrics, last_metrics, 'servo_forced_read_after_write')} "
                f"forced_missed={_metric_delta(first_metrics, last_metrics, 'servo_forced_read_after_write_missed')} "
                f"targets={_metric_delta(first_metrics, last_metrics, 'servo_write_targets_submitted')} "
                f"replaced={_metric_delta(first_metrics, last_metrics, 'servo_write_targets_replaced')} "
                f"write_cmd={_metric_delta(first_metrics, last_metrics, 'servo_write_commands')} "
                f"write_skipped={_metric_delta(first_metrics, last_metrics, 'servo_write_commands_skipped')} "
                f"write_fail={_metric_delta(first_metrics, last_metrics, 'servo_write_failures')} "
                f"queue_ms_avg/p95/max="
                f"{_format_ms(_timing_avg(timing_samples, 'io_servo_latest_write_queue_latency_s'))}/"
                f"{_format_ms(_timing_percentile(timing_samples, 'io_servo_latest_write_queue_latency_s', 95.0))}/"
                f"{_format_ms(_timing_max(timing_samples, 'io_servo_latest_write_queue_latency_s'))}",
                flush=True,
            )


def _run_hardware_preflight(
    *,
    robot_io,
    actuator_names: List[str],
    home_q_rad: np.ndarray,
    joint_min_rad: np.ndarray,
    joint_max_rad: np.ndarray,
    imu_startup_timeout_s: float,
    home_tolerance_deg: float,
    max_tilt_deg: float | None = None,
) -> None:
    """Print and validate hardware state before the policy writes commands."""
    errors: List[str] = []
    warnings: List[str] = []
    print("Hardware preflight:", flush=True)

    _preflight_servos(
        robot_io=robot_io,
        actuator_names=actuator_names,
        home_q_rad=home_q_rad,
        joint_min_rad=joint_min_rad,
        joint_max_rad=joint_max_rad,
        home_tolerance_deg=home_tolerance_deg,
        errors=errors,
        warnings=warnings,
    )
    _preflight_imu(
        robot_io=robot_io,
        imu_startup_timeout_s=imu_startup_timeout_s,
        max_tilt_deg=max_tilt_deg,
        errors=errors,
    )
    _preflight_footswitches(
        robot_io=robot_io,
        errors=errors,
        warnings=warnings,
    )

    for warning in warnings:
        print(f"  {_ANSI_YELLOW}WARNING: {warning}{_ANSI_RESET}", flush=True)
    if errors:
        print("Hardware preflight FAILED:", flush=True)
        for error in errors:
            print(f"  ERROR: {error}", flush=True)
        error_lines = "\n".join(f"  - {error}" for error in errors)
        raise SystemExit(
            "Hardware preflight failed; fix errors before running policy:\n"
            f"{error_lines}"
        )
    print("Hardware preflight OK.", flush=True)


def _preflight_servos(
    *,
    robot_io,
    actuator_names: List[str],
    home_q_rad: np.ndarray,
    joint_min_rad: np.ndarray,
    joint_max_rad: np.ndarray,
    home_tolerance_deg: float,
    errors: List[str],
    warnings: List[str],
) -> None:
    actuators = robot_io.actuators
    port = getattr(actuators, "port", "unknown")
    baudrate = getattr(actuators, "baudrate", "unknown")
    controller = getattr(actuators, "controller", None)
    voltage = None
    if controller is not None and hasattr(controller, "get_battery_voltage"):
        try:
            voltage = controller.get_battery_voltage()
        except Exception as exc:
            warnings.append(f"servo board voltage read failed: {exc!r}")

    voltage_text = "unknown" if voltage is None else f"{float(voltage):.2f}V"
    print(f"  Servo bus: port={port} baud={baudrate} voltage={voltage_text}", flush=True)

    wait_for_cache = getattr(actuators, "wait_for_initial_cache", None)
    if callable(wait_for_cache) and not wait_for_cache(timeout_s=3.0):
        last_error = getattr(actuators, "_last_error", None)
        suffix = f": {last_error!r}" if last_error is not None else ""
        errors.append(f"servo cache initialization failed{suffix}")

    try:
        positions = actuators.get_positions_rad()
    except Exception as exc:
        positions = None
        errors.append(f"servo position read raised {exc!r}")

    if positions is None:
        last_error = getattr(actuators, "_last_error", None)
        suffix = f": {last_error!r}" if last_error is not None else ""
        errors.append(f"servo position read failed{suffix}")
        return

    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    expected_n = len(actuator_names)
    if pos.size != expected_n:
        errors.append(f"servo position count {pos.size} != actuator count {expected_n}")
        return

    ids = list(getattr(actuators, "servo_ids_list", []))
    if len(ids) != expected_n:
        ids = [None] * expected_n

    finite = np.isfinite(pos)
    limit_tol = np.deg2rad(5.0)
    home_tol = np.deg2rad(max(0.0, float(home_tolerance_deg)))
    max_home_err = 0.0
    print("  Servos:", flush=True)
    for idx, name in enumerate(actuator_names):
        sid = ids[idx]
        sid_text = "?" if sid is None else str(int(sid))
        q = float(pos[idx])
        home = float(home_q_rad[idx])
        qmin = float(joint_min_rad[idx])
        qmax = float(joint_max_rad[idx])
        home_err = q - home
        max_home_err = max(max_home_err, abs(home_err))
        status = "OK"
        if not bool(finite[idx]):
            status = "ERROR"
            errors.append(f"{name} servo id={sid_text} readback is non-finite")
        elif q < qmin - limit_tol or q > qmax + limit_tol:
            status = "WARN"
            warnings.append(
                f"{name} servo id={sid_text} readback {_format_rad_deg(q)}deg "
                f"is outside policy range [{_format_rad_deg(qmin)}, {_format_rad_deg(qmax)}]deg"
            )
        elif abs(home_err) > home_tol:
            status = "WARN"
        print(
            "    "
            f"{name:<20} id={sid_text:>3} "
            f"pos={_format_rad_deg(q)}deg "
            f"home={_format_rad_deg(home)}deg "
            f"err={_format_rad_deg(home_err)}deg "
            f"range=[{_format_rad_deg(qmin)}, {_format_rad_deg(qmax)}]deg "
            f"{status}",
            flush=True,
        )

    if max_home_err > home_tol:
        warnings.append(
            f"max servo home error {float(np.rad2deg(max_home_err)):.1f}deg "
            f"> tolerance {float(home_tolerance_deg):.1f}deg"
        )


def _preflight_imu(
    *,
    robot_io,
    imu_startup_timeout_s: float,
    max_tilt_deg: float | None,
    errors: List[str],
) -> None:
    try:
        if hasattr(robot_io, "wait_for_valid_imu_sample"):
            robot_io.wait_for_valid_imu_sample(timeout_s=float(imu_startup_timeout_s))
        sample = getattr(robot_io, "_last_fresh_imu_sample", None)
        if sample is None:
            sample = robot_io.imu.read()
    except Exception as exc:
        errors.append(f"IMU valid sample unavailable: {exc}")
        print(f"  IMU: ERROR {exc}", flush=True)
        return

    valid = bool(getattr(sample, "valid", True))
    fresh = bool(getattr(sample, "fresh", True))
    quat = np.asarray(getattr(sample, "quat_wxyz", []), dtype=np.float32).reshape(-1)
    gyro = np.asarray(getattr(sample, "gyro_rad_s", []), dtype=np.float32).reshape(-1)
    quat_norm = float(np.linalg.norm(quat)) if quat.size == 4 else float("nan")
    tilt_rad = _quat_wxyz_tilt_rad(quat)
    tilt_deg = None if tilt_rad is None else float(math.degrees(tilt_rad))
    tilt_text = "n/a" if tilt_deg is None else f"{tilt_deg:.1f}deg"
    imu = robot_io.imu
    diag = getattr(imu, "diag", None)
    diag_text = f" diag={diag}" if diag else ""
    print(
        "  IMU: "
        f"valid={valid} "
        f"fresh={fresh} "
        f"quat={np.round(quat, 4).tolist()} "
        f"quat_norm={quat_norm:.3f} "
        f"tilt={tilt_text} "
        f"gyro={np.round(gyro, 4).tolist()} "
        f"errors={getattr(imu, 'error_count', 0)} "
        f"last_error={getattr(imu, 'last_error', None)}"
        f"{diag_text}",
        flush=True,
    )
    if not valid:
        errors.append("IMU sample is invalid")
    if not fresh:
        errors.append("IMU sample is not fresh")
    if (
        quat.size != 4
        or not np.all(np.isfinite(quat))
        or not (0.9 <= quat_norm <= 1.1)
    ):
        errors.append(
            f"IMU quaternion is invalid: quat={quat.tolist()} norm={quat_norm:.3f}"
        )
    elif (
        max_tilt_deg is not None
        and tilt_deg is not None
        and tilt_deg > float(max_tilt_deg)
    ):
        errors.append(
            f"initial body tilt {tilt_deg:.1f}deg > {float(max_tilt_deg):.1f}deg"
        )
    if gyro.size != 3 or not np.all(np.isfinite(gyro)):
        errors.append(f"IMU gyro is invalid: gyro={gyro.tolist()}")


def _preflight_footswitches(
    *,
    robot_io,
    errors: List[str],
    warnings: List[str],
) -> None:
    if not bool(getattr(robot_io.foot_switches, "available", True)):
        print("  Footswitches: disabled by hardware config", flush=True)
        return
    try:
        sample = robot_io.foot_switches.read()
    except Exception as exc:
        errors.append(f"footswitch read failed: {exc!r}")
        print(f"  Footswitches: ERROR {exc!r}", flush=True)
        return

    switches = np.asarray(sample.switches, dtype=np.float32).reshape(-1)
    if switches.size != len(_FOOT_SWITCH_LABELS) or not np.all(np.isfinite(switches)):
        errors.append(f"footswitch sample invalid: {switches.tolist()}")
        print(f"  Footswitches: ERROR values={switches.tolist()}", flush=True)
        return

    values = [int(round(float(v))) for v in switches]
    states = ", ".join(
        f"{name}={value}" for name, value in zip(_FOOT_SWITCH_LABELS, values)
    )
    print(f"  Footswitches: {states} (1=pressed, 0=open)", flush=True)
    open_names = [
        name for name, value in zip(_FOOT_SWITCH_LABELS, values) if value == 0
    ]
    if open_names:
        warnings.append(f"initial footswitches open at walk start: {open_names}")


def _run_startup_home_hold(
    *,
    runner: RuntimePolicyRunner,
    velocity_cmd: np.ndarray,
    steps: int,
    log_steps: int,
    ctrl_dt: float,
    realtime: bool,
    leg_indices: List[int],
    stability_check: bool,
    stability_window_s: float = _STARTUP_STABILITY_WINDOW_S,
    stability_min_footswitch_pressed_ratio: float = (
        _STARTUP_STABILITY_MIN_FOOTSWITCH_PRESSED_RATIO
    ),
    stability_max_tilt_deg: float = _STARTUP_STABILITY_MAX_TILT_DEG,
    stability_max_gyro_rad_s: float = _STARTUP_STABILITY_MAX_GYRO_RAD_S,
    stability_max_leg_error_deg: float = _STARTUP_STABILITY_MAX_LEG_ERROR_DEG,
    confirm_before_walk: bool = False,
    confirm_imu_timeout_s: float = 3.0,
    input_fn: Callable[[str], str] | None = None,
    fall_tilt_deg: float = _DEFAULT_FALL_TILT_DEG,
    telemetry: PolicyTelemetryRecorder | None = None,
    telemetry_phase: str = "startup_home",
) -> None:
    """Command home before policy walking, then reset policy episode state."""
    hold_steps = max(0, int(steps))
    if hold_steps <= 0:
        return

    print(
        "Startup home hold: "
        f"steps={hold_steps} duration_s={hold_steps * float(ctrl_dt):.2f} "
        "mode=startup_home_hold",
        flush=True,
    )
    last_info: dict | None = None
    hold_infos: List[dict] = []
    for step in range(hold_steps):
        loop_start_s = time.monotonic()
        info = runner.step(
            velocity_cmd,
            force_home_hold=True,
            home_hold_mode="startup_home_hold",
        )
        work_s = time.monotonic() - loop_start_s
        timing_s = dict(info.get("timing_s", {}))
        timing_s["work"] = work_s
        info["timing_s"] = timing_s
        last_info = info
        hold_infos.append(info)
        if telemetry is not None:
            telemetry.record(
                info,
                phase=telemetry_phase,
                loop_step=step,
                requested_velocity_cmd=velocity_cmd,
            )
        _abort_if_fallen(
            info=info,
            step=step,
            max_tilt_deg=float(fall_tilt_deg),
        )

        should_log = (
            step == 0
            or step == hold_steps - 1
            or (log_steps > 0 and step % log_steps == 0)
        )
        if should_log:
            leg_err_deg = _leg_error_max_deg(info, leg_indices)
            err_text = (
                "n/a" if leg_err_deg is None else f"{float(leg_err_deg):.1f}"
            )
            foot_summary = _format_foot_switches(info)
            foot_text = f" {foot_summary}" if foot_summary else ""
            orientation = _format_base_orientation(info.get("signals"))
            orientation_text = f" {orientation}" if orientation else ""
            print(
                f"[startup_home {step + 1:4d}/{hold_steps:4d}] "
                f"leg_err|max_deg={err_text}{foot_text}{orientation_text} "
                f"{_format_step_timing(timing_s)}",
                flush=True,
            )

        if realtime and not (confirm_before_walk and step == hold_steps - 1):
            elapsed = time.monotonic() - loop_start_s
            remaining = ctrl_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    if confirm_before_walk:
        print("Startup home hold loop complete; entering manual start gate.", flush=True)
    final_err = _leg_error_max_deg(last_info or {}, leg_indices)
    final_err_text = "n/a" if final_err is None else f"{float(final_err):.1f}"
    print(f"Startup home hold complete: leg_err|max_deg={final_err_text}.", flush=True)
    if confirm_before_walk:
        print(
            "Manual start gate: home hold is complete. Settle the robot now.",
            flush=True,
        )
        print(
            "Type 'y' then Enter to start policy walking; press Enter or 'n' to stop.",
            flush=True,
        )
        try:
            prompt = "Start policy walking now? [y/N]:"
            print(prompt, flush=True)
            if input_fn is None:
                answer = sys.stdin.readline()
            else:
                answer = input_fn(f"{prompt} ")
            answer = str(answer).strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            raise SystemExit(
                "Startup walk cancelled by user after startup home hold. "
                "Robot remains at the commanded home pose."
            )
        print(
            "Manual start confirmed; refreshing home stability window before command.",
            flush=True,
        )
        robot_io = getattr(runner, "_robot_io", None)
        wait_for_imu = getattr(robot_io, "wait_for_valid_imu_sample", None)
        if callable(wait_for_imu):
            print(
                "Re-priming IMU after manual pause "
                f"(timeout {float(confirm_imu_timeout_s):.1f}s)...",
                flush=True,
            )
            wait_for_imu(timeout_s=float(confirm_imu_timeout_s))
        refresh_steps = max(
            1,
            int(round(float(stability_window_s) / max(float(ctrl_dt), 1e-9))),
        )
        hold_infos = []
        last_info = None
        for _ in range(refresh_steps):
            loop_start_s = time.monotonic()
            info = runner.step(
                velocity_cmd,
                force_home_hold=True,
                home_hold_mode="startup_home_confirm",
            )
            work_s = time.monotonic() - loop_start_s
            timing_s = dict(info.get("timing_s", {}))
            timing_s["work"] = work_s
            info["timing_s"] = timing_s
            last_info = info
            hold_infos.append(info)
            if telemetry is not None:
                telemetry.record(
                    info,
                    phase=f"{telemetry_phase}_confirm",
                    loop_step=len(hold_infos) - 1,
                    requested_velocity_cmd=velocity_cmd,
                )
            _abort_if_fallen(
                info=info,
                step=len(hold_infos) - 1,
                max_tilt_deg=float(fall_tilt_deg),
            )
            if realtime:
                remaining = ctrl_dt - (time.monotonic() - loop_start_s)
                if remaining > 0:
                    time.sleep(remaining)
        final_err = _leg_error_max_deg(last_info or {}, leg_indices)
        final_err_text = "n/a" if final_err is None else f"{float(final_err):.1f}"
        print(f"Startup home confirm complete: leg_err|max_deg={final_err_text}.", flush=True)

    print("Checking startup home stability before command.", flush=True)
    if stability_check:
        errors, warnings, summary = _startup_home_stability_errors(
            infos=hold_infos,
            ctrl_dt=ctrl_dt,
            leg_indices=leg_indices,
            window_s=stability_window_s,
            min_footswitch_pressed_ratio=stability_min_footswitch_pressed_ratio,
            max_tilt_deg=stability_max_tilt_deg,
            max_gyro_rad_s=stability_max_gyro_rad_s,
            max_leg_error_deg=stability_max_leg_error_deg,
        )
        summary_text = " ".join(summary)
        for warning in warnings:
            print(f"  {_ANSI_YELLOW}WARNING: {warning}{_ANSI_RESET}", flush=True)
        if errors:
            print(f"Startup home stability FAILED: {summary_text}", flush=True)
            for error in errors:
                print(f"  ERROR: {error}", flush=True)
            error_lines = "\n".join(f"  - {error}" for error in errors)
            raise SystemExit(
                "Startup home stability failed; refusing to start policy walking. "
                "Robot remains at the commanded home pose.\n"
                f"{error_lines}"
            )
        print(f"Startup home stability OK: {summary_text}", flush=True)
    else:
        print("Startup home stability check disabled.", flush=True)

    print("resetting policy state before command.", flush=True)
    runner.reset()


def _run_standing_stabilization(
    *,
    runner: StandingPolicyRunner,
    steps: int,
    log_steps: int,
    ctrl_dt: float,
    realtime: bool,
    actuator_names: list[str],
    diagnostic_log_policy: bool,
    stability_check: bool,
    stability_max_tilt_deg: float,
    confirm_before_walk: bool,
    confirm_imu_timeout_s: float,
    fall_tilt_deg: float,
    telemetry: PolicyTelemetryRecorder | None = None,
) -> dict:
    """Run the standing policy, optionally confirm, then verify a fresh window."""
    velocity_cmd = np.zeros(3, dtype=np.float32)
    leg_indices = _actuator_indices(
        actuator_names, tuple(name for _, name in _LEG_LOG_JOINTS)
    )

    def _run_steps(count: int, *, label: str) -> list[dict]:
        infos: list[dict] = []
        for step in range(max(1, int(count))):
            loop_start_s = time.monotonic()
            info = runner.step(velocity_cmd)
            timing_s = dict(info.get("timing_s", {}))
            timing_s["work"] = time.monotonic() - loop_start_s
            info["timing_s"] = timing_s
            infos.append(info)
            if telemetry is not None:
                telemetry.record(
                    info,
                    phase=label,
                    loop_step=step,
                    requested_velocity_cmd=velocity_cmd,
                )
            _abort_if_fallen(
                info=info,
                step=step,
                max_tilt_deg=float(fall_tilt_deg),
            )
            if step == 0 or step == count - 1 or (
                log_steps > 0 and step % log_steps == 0
            ):
                leg_err = _leg_error_max_deg(info, leg_indices)
                leg_text = "n/a" if leg_err is None else f"{leg_err:.1f}"
                diagnostic_parts: list[str] = []
                if diagnostic_log_policy:
                    target_summary = _format_leg_targets_deg(
                        info["target_q_rad"], actuator_names
                    )
                    observed_summary = _format_leg_values_deg(
                        info["signals"].joint_pos_rad, actuator_names
                    )
                    if target_summary:
                        diagnostic_parts.append(f"leg_deg={target_summary}")
                    if observed_summary:
                        diagnostic_parts.append(f"obs_leg_deg={observed_summary}")
                    policy_diagnostic = _format_policy_diagnostics(
                        info=info,
                        actuator_names=actuator_names,
                        leg_indices=leg_indices,
                        spec=runner.spec,
                    )
                    if policy_diagnostic:
                        diagnostic_parts.append(policy_diagnostic)
                diagnostic = " ".join(diagnostic_parts)
                print(
                    f"[{label} {step + 1:4d}/{count:4d}] "
                    f"leg_err|max_deg={leg_text} {_format_foot_switches(info)}"
                    f"{' ' + diagnostic if diagnostic else ''}",
                    flush=True,
                )
            if stability_check:
                tilt_deg = _info_tilt_deg(info)
                gyro_norm = _info_gyro_norm_rad_s(info)
                safety_errors = []
                if tilt_deg is not None and tilt_deg > float(stability_max_tilt_deg):
                    safety_errors.append(
                        f"tilt {tilt_deg:.1f}deg > {float(stability_max_tilt_deg):.1f}deg"
                    )
                if (
                    gyro_norm is not None
                    and gyro_norm > _STARTUP_STABILITY_MAX_GYRO_RAD_S
                ):
                    safety_errors.append(
                        f"gyro {gyro_norm:.3f}rad/s > "
                        f"{_STARTUP_STABILITY_MAX_GYRO_RAD_S:.3f}rad/s"
                    )
                if safety_errors:
                    raise SystemExit(
                        f"Standing safety abort at {label} step {step + 1}: "
                        + "; ".join(safety_errors)
                        + ". Keep the robot supported while runtime unloads servos."
                    )
            if realtime:
                remaining = float(ctrl_dt) - (time.monotonic() - loop_start_s)
                if remaining > 0.0:
                    time.sleep(remaining)
        return infos

    print(
        "Startup standing stabilization: "
        f"steps={max(1, int(steps))} "
        f"duration_s={max(1, int(steps)) * float(ctrl_dt):.2f}",
        flush=True,
    )
    infos = _run_steps(max(1, int(steps)), label="startup_standing")

    if confirm_before_walk:
        print("Start policy walking now? [y/N]:", flush=True)
        answer = sys.stdin.readline().strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit(
                "Walking cancelled after standing stabilization; unloading servos."
            )
        robot_io = getattr(runner, "_robot_io", None)
        wait_for_imu = getattr(robot_io, "wait_for_valid_imu_sample", None)
        if callable(wait_for_imu):
            wait_for_imu(timeout_s=float(confirm_imu_timeout_s))
        refresh_steps = max(
            1,
            int(round(_STARTUP_STABILITY_WINDOW_S / max(float(ctrl_dt), 1e-9))),
        )
        infos = _run_steps(refresh_steps, label="standing_confirm")

    if stability_check:
        errors, warnings, summary = _startup_home_stability_errors(
            infos=infos,
            ctrl_dt=ctrl_dt,
            leg_indices=leg_indices,
            window_s=_STARTUP_STABILITY_WINDOW_S,
            min_footswitch_pressed_ratio=(
                _STARTUP_STABILITY_MIN_FOOTSWITCH_PRESSED_RATIO
            ),
            max_tilt_deg=float(stability_max_tilt_deg),
            max_gyro_rad_s=_STARTUP_STABILITY_MAX_GYRO_RAD_S,
            max_leg_error_deg=_STARTUP_STABILITY_MAX_LEG_ERROR_DEG,
        )
        summary_text = " ".join(summary)
        for warning in warnings:
            print(f"  {_ANSI_YELLOW}WARNING: {warning}{_ANSI_RESET}", flush=True)
        if errors:
            raise SystemExit(
                "Standing stability failed; refusing to switch to walking: "
                + "; ".join(errors)
            )
        print(f"Standing stability OK: {summary_text}", flush=True)

    return infos[-1]


def run_policy_loop(
    *,
    runner: RuntimePolicyRunner,
    max_steps: int | None,
    velocity_cmd: np.ndarray,
    log_steps: int,
    ctrl_dt: float,
    realtime: bool,
    actuator_names: Optional[List[str]] = None,
    diagnostic_log_policy: bool = False,
    startup_home_hold_steps: int = 0,
    startup_command_ramp_steps: int = 0,
    startup_action_ramp_steps: int = 0,
    startup_stability_check: bool = True,
    startup_stability_max_tilt_deg: float = _STARTUP_STABILITY_MAX_TILT_DEG,
    startup_confirm_before_walk: bool = False,
    startup_confirm_input_fn: Callable[[str], str] | None = None,
    startup_confirm_imu_timeout_s: float = 3.0,
    fall_tilt_deg: float = _DEFAULT_FALL_TILT_DEG,
    telemetry: PolicyTelemetryRecorder | None = None,
    telemetry_phase: str = "policy",
) -> List[dict]:
    """Run for ``max_steps`` iterations, or until interrupted when it is ``None``."""
    logs: List[dict] = []
    leg_indices = _actuator_indices(
        actuator_names, tuple(name for _, name in _LEG_LOG_JOINTS)
    )
    _run_startup_home_hold(
        runner=runner,
        velocity_cmd=velocity_cmd,
        steps=startup_home_hold_steps,
        log_steps=log_steps,
        ctrl_dt=ctrl_dt,
        realtime=realtime,
        leg_indices=leg_indices,
        stability_check=bool(startup_stability_check),
        stability_max_tilt_deg=float(startup_stability_max_tilt_deg),
        confirm_before_walk=bool(startup_confirm_before_walk),
        confirm_imu_timeout_s=float(startup_confirm_imu_timeout_s),
        input_fn=startup_confirm_input_fn,
        fall_tilt_deg=float(fall_tilt_deg),
        telemetry=telemetry,
    )
    history_size = max(1, int(round(60.0 / max(float(ctrl_dt), 1e-9))))
    timing_samples = deque(maxlen=history_size) if max_steps is None else []
    servo_metric_samples = deque(maxlen=history_size) if max_steps is None else []
    last_loop_start_s: float | None = None
    completed = False
    steps = itertools.count() if max_steps is None else range(int(max_steps))
    try:
        for step in steps:
            loop_start_s = time.monotonic()
            loop_period_s = (
                None if last_loop_start_s is None else loop_start_s - last_loop_start_s
            )
            last_loop_start_s = loop_start_s
            step_velocity_cmd, command_ramp_scale = _ramped_velocity_cmd(
                velocity_cmd,
                step=step,
                ramp_steps=startup_command_ramp_steps,
            )
            action_ramp_scale = _ramp_scale(
                step=step,
                ramp_steps=startup_action_ramp_steps,
            )
            info = runner.step(step_velocity_cmd, action_scale=action_ramp_scale)
            if int(startup_command_ramp_steps) > 0:
                info["requested_velocity_cmd"] = np.asarray(
                    velocity_cmd, dtype=np.float32
                ).reshape(3)
                info["commanded_velocity_cmd"] = step_velocity_cmd.copy()
                info["command_ramp_scale"] = float(command_ramp_scale)
            if int(startup_action_ramp_steps) > 0:
                info["action_ramp_scale"] = float(action_ramp_scale)
            work_s = time.monotonic() - loop_start_s
            timing_s = dict(info.get("timing_s", {}))
            timing_s["work"] = work_s
            if loop_period_s is not None:
                timing_s["loop_period"] = loop_period_s
            info["timing_s"] = timing_s
            timing_samples.append(timing_s)
            servo_metrics = info.get("servo_metrics")
            if isinstance(servo_metrics, dict) and servo_metrics:
                servo_metric_samples.append(servo_metrics)
            if telemetry is not None:
                telemetry.record(
                    info,
                    phase=telemetry_phase,
                    loop_step=int(step),
                    requested_velocity_cmd=velocity_cmd,
                )
            _abort_if_fallen(
                info=info,
                step=int(step),
                max_tilt_deg=float(fall_tilt_deg),
            )
            is_last_step = max_steps is not None and step == int(max_steps) - 1
            if log_steps > 0 and (step % log_steps == 0 or is_last_step):
                applied = info["applied_action"]
                target = info["target_q_rad"]
                leg_applied_max = (
                    float(np.max(np.abs(applied[leg_indices]))) if leg_indices else None
                )
                leg_summary = _format_leg_targets_deg(target, actuator_names)
                observed = np.asarray(info["signals"].joint_pos_rad, dtype=np.float32)
                observed_leg_summary = _format_leg_values_deg(observed, actuator_names)
                leg_err_max_deg = _leg_error_max_deg(info, leg_indices)
                foot_summary = _format_foot_switches(info)
                extra_parts = []
                if leg_applied_max is not None:
                    extra_parts.append(f"leg|applied|max={leg_applied_max:.3f}")
                if leg_summary:
                    extra_parts.append(f"leg_deg={leg_summary}")
                if observed_leg_summary:
                    extra_parts.append(f"obs_leg_deg={observed_leg_summary}")
                if leg_err_max_deg is not None:
                    extra_parts.append(f"leg_err|max_deg={leg_err_max_deg:.1f}")
                if foot_summary:
                    extra_parts.append(foot_summary)
                orientation = _format_base_orientation(info["signals"])
                if orientation:
                    extra_parts.append(orientation)
                if int(startup_command_ramp_steps) > 0:
                    extra_parts.append(
                        f"cmd_scale={float(info.get('command_ramp_scale', 1.0)):.3f}"
                    )
                if int(startup_action_ramp_steps) > 0:
                    extra_parts.append(
                        f"action_scale={float(info.get('action_ramp_scale', 1.0)):.3f}"
                    )
                if diagnostic_log_policy:
                    diag = _format_policy_diagnostics(
                        info=info,
                        actuator_names=actuator_names,
                        leg_indices=leg_indices,
                        spec=runner.spec,
                    )
                    if diag:
                        extra_parts.append(diag)
                servo_summary = _format_servo_step_metrics(servo_metrics)
                if servo_summary:
                    extra_parts.append(servo_summary)
                extra_parts.append(_format_step_timing(timing_s))
                extra = " " + " ".join(extra_parts) if extra_parts else ""
                print(
                    f"[step {step:5d}] idx={info['step_idx']:5d} "
                    f"|applied|max={float(np.max(np.abs(applied))):.3f} "
                    f"target[0:3]={np.round(target[:3], 4).tolist()}"
                    f"{extra}",
                    flush=True,
                )
                if max_steps is not None:
                    logs.append(info)
            if realtime:
                elapsed = time.monotonic() - loop_start_s
                remaining = ctrl_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        completed = True
    finally:
        _print_timing_summary(
            timing_samples=timing_samples,
            ctrl_dt=ctrl_dt,
            realtime=realtime,
            completed=completed,
            servo_metric_samples=servo_metric_samples,
        )
    return logs


def main(argv: Optional[List[str]] = None) -> int:
    _install_stack_dump_signal()
    parser = argparse.ArgumentParser(
        description="Run a WildRobot policy bundle (latest v8 home-residual contract)."
    )
    parser.add_argument(
        "--bundle", type=str, default=None,
        help=(
            "Deployment bundle containing standing/walking policies, or a legacy "
            "single-policy bundle."
        ),
    )
    parser.add_argument(
        "--stable-only",
        "--stable_only",
        action="store_true",
        help=(
            "Run only the deployment bundle's 17-action standing stabilizer "
            "until interrupted."
        ),
    )
    parser.add_argument(
        "--hardware-config",
        "--runtime-config",
        dest="hardware_config",
        type=str,
        default=None,
        help=(
            "Physical robot configuration (servo IDs/calibration, IMU, GPIO). "
            "Defaults to runtime/configs/hardware_config.json; "
            "--runtime-config is a legacy alias."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with mock IO (no hardware): exercises the full loop for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help=(
            "Number of walking or dry-run control steps. Real --stable-only "
            "runs until interrupted."
        ),
    )
    parser.add_argument("--log-steps", type=int, default=20, help="Log every N steps (0=off).")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log",
        type=str,
        default=None,
        help=(
            "Override the automatic _run_policy_logs path while still printing "
            "to the console."
        ),
    )
    log_group.add_argument(
        "--log-only",
        type=str,
        default=None,
        help="Write stdout/stderr to this file without printing to the console.",
    )
    parser.add_argument(
        "--telemetry",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Save every control sample to a compressed .npz file. With no PATH, "
            "write beside the text log using the same filename stem."
        ),
    )
    parser.add_argument(
        "--velocity-cmd", type=str, default=None,
        help="Command 'vx' or 'vx,vy,wz' (default: bundle default_velocity_cmd).",
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Do not sleep to maintain control_hz (default: realtime ON for hardware, OFF for --dry-run).",
    )
    parser.add_argument(
        "--imu-startup-timeout-s",
        type=float,
        default=3.0,
        help="Hardware only: wait this long for the first valid IMU sample before starting control.",
    )
    parser.add_argument(
        "--fall-tilt-deg",
        type=float,
        default=_DEFAULT_FALL_TILT_DEG,
        help=(
            "Stop policy control and unload servos when body tilt exceeds this "
            f"angle (default: {_DEFAULT_FALL_TILT_DEG:.1f} degrees)."
        ),
    )
    # Retain obsolete flags so existing launch commands continue to parse.
    footswitch_group = parser.add_mutually_exclusive_group()
    footswitch_group.add_argument(
        "--allow-unpressed-footswitch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    footswitch_group.add_argument(
        "--require-pressed-footswitch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--preflight-home-tolerance-deg",
        type=float,
        default=25.0,
        help="Warn when any servo starts this many degrees away from policy home.",
    )
    parser.add_argument(
        "--startup-home-hold-s",
        type=float,
        default=2.0,
        help=(
            "Hardware only: before stable-only control or a nonzero walking "
            "command, hold the bundled home pose for this many seconds, then "
            "reset policy state (default: 2.0; set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--startup-pose-blend-s",
        type=float,
        default=_STARTUP_POSE_BLEND_S,
        help=(
            "Hardware standing startup: linearly blend from the measured servo "
            "pose to bundled home before enforcing the stability gate "
            f"(default: {_STARTUP_POSE_BLEND_S:.1f}; set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--startup-pose-hold-s",
        type=float,
        default=_STARTUP_POSE_HOLD_S,
        help=(
            "Integrated hardware bundle only: hold home after the measured-pose "
            f"blend before starting policy inference (default: {_STARTUP_POSE_HOLD_S:.1f})."
        ),
    )
    parser.add_argument(
        "--confirm-before-walk",
        action="store_true",
        help=(
            "Hardware only: after startup home hold, prompt before starting "
            "policy walking. Answering yes refreshes the home stability window; "
            "answering no leaves the robot at home and exits."
        ),
    )
    parser.add_argument(
        "--disable-startup-stability-check",
        action="store_true",
        help=(
            "Do not fail after startup home hold when body stability checks fail. "
            "Use only for suspended diagnostics."
        ),
    )
    parser.add_argument(
        "--startup-stability-max-tilt-deg",
        type=float,
        default=_STARTUP_STABILITY_MAX_TILT_DEG,
        help=(
            "Maximum body tilt allowed during the final startup home stability "
            f"window before walking (default: {_STARTUP_STABILITY_MAX_TILT_DEG:.1f})."
        ),
    )
    parser.add_argument(
        "--startup-command-ramp-s",
        type=float,
        default=0.0,
        help=(
            "For nonzero velocity commands, linearly ramp the command from zero "
            "to the requested value over this many seconds after any startup "
            "home hold (default: 0.0, disabled)."
        ),
    )
    parser.add_argument(
        "--startup-action-ramp-s",
        type=float,
        default=0.0,
        help=(
            "For nonzero velocity commands, linearly ramp policy residual actions "
            "from zero to full scale over this many seconds after any startup "
            "home hold (default: 0.0, disabled)."
        ),
    )
    parser.add_argument(
        "--skip-hardware-preflight",
        action="store_true",
        help="Skip hardware preflight checks before the policy loop.",
    )
    parser.add_argument(
        "--diagnostic-log-policy",
        action="store_true",
        help=(
            "Append raw-action, left/right leg action deltas, selected reference "
            "bin, phase, command, gyro, base roll/pitch/yaw, tilt, and "
            "normalized joint-velocity summaries to each normal step log line."
        ),
    )
    parser.add_argument(
        "--zero-cmd-hold-home-deadzone",
        type=float,
        default=1e-6,
        help=(
            "Hold the bundled home pose instead of running the walking policy when "
            "all velocity command components are within this absolute value "
            "(default: 1e-6)."
        ),
    )
    parser.add_argument(
        "--disable-zero-cmd-hold-home",
        action="store_true",
        help=(
            "Run the walking policy even for a zero velocity command. Use this only "
            "for policy debugging; the default matches the safe stand behavior."
        ),
    )
    args = parser.parse_args(argv)

    if not math.isfinite(float(args.fall_tilt_deg)) or not (
        0.0 < float(args.fall_tilt_deg) <= 180.0
    ):
        parser.error("--fall-tilt-deg must be finite and in (0, 180]")

    if args.log is not None:
        log_path = Path(args.log).expanduser()
    elif args.log_only is not None:
        log_path = Path(args.log_only).expanduser()
    else:
        log_path = _default_run_policy_log_path(
            args.bundle or "bundle",
            stable_only=bool(args.stable_only),
        )
    args.telemetry_path = _resolve_telemetry_path(args.telemetry, log_path)
    mirror_console = args.log_only is None
    with _output_log_context(str(log_path), mirror_console=mirror_console):
        print(f"Policy log: {log_path}", flush=True)
        outcome = "error"
        error_text: str | None = None
        try:
            result = _run_policy_from_args(args)
            outcome = "completed" if result == 0 else "failed"
            return result
        except SystemExit as exc:
            code = exc.code
            if code is None:
                outcome = "completed"
                return 0
            if isinstance(code, int):
                outcome = "completed" if code == 0 else "aborted"
                error_text = None if code == 0 else f"SystemExit({code})"
                return int(code)
            outcome = "aborted"
            error_text = str(code)
            print(code, file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            outcome = "interrupted"
            error_text = "KeyboardInterrupt"
            print("Interrupted.", file=sys.stderr)
            return 130
        except BaseException as exc:
            outcome = "error"
            error_text = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return 1
        finally:
            recorder = getattr(args, "telemetry_recorder", None)
            if recorder is not None:
                try:
                    recorder.save(outcome=outcome, error=error_text)
                except Exception:
                    print("Failed to save policy telemetry:", file=sys.stderr)
                    traceback.print_exc()


def _load_runtime_onnx_policy(bundle: PolicyBundle) -> OnnxPolicy:
    policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=bundle.spec.model.input_name,
        output_name=bundle.spec.model.output_name,
        expected_obs_dim=int(bundle.spec.model.obs_dim),
        expected_action_dim=int(bundle.spec.model.action_dim),
    )
    if policy.info.obs_dim is not None and int(policy.info.obs_dim) != int(
        bundle.spec.model.obs_dim
    ):
        raise SystemExit(
            f"ONNX obs_dim {policy.info.obs_dim} != spec {bundle.spec.model.obs_dim}"
        )
    if policy.info.action_dim is not None and int(policy.info.action_dim) != int(
        bundle.spec.model.action_dim
    ):
        raise SystemExit(
            f"ONNX action_dim {policy.info.action_dim} != spec "
            f"{bundle.spec.model.action_dim}"
        )
    return policy


def _run_deployment_bundle_from_args(
    args: argparse.Namespace, bundle_path: Path
) -> int:
    deployment = DeploymentBundle.load(bundle_path)
    standing_bundle = deployment.policy_bundle("standing")
    walking_bundle = deployment.policy_bundle("walking")
    validate_spec(standing_bundle.spec)
    validate_spec(walking_bundle.spec)
    if standing_bundle.spec.observation.layout_id not in _STANDING_LAYOUT_IDS:
        raise SystemExit(
            "Deployment standing policy must use one of "
            f"{sorted(_STANDING_LAYOUT_IDS)}"
        )
    if walking_bundle.spec.observation.layout_id not in _WALKING_LAYOUT_IDS:
        raise SystemExit(
            "Deployment walking policy must use one of "
            f"{sorted(_WALKING_LAYOUT_IDS)}"
        )

    standing_cfg = StandingRuntimePolicyConfig.from_json(
        deployment.policy_dir("standing") / "runtime_policy_config.json"
    )
    walking_cfg = RuntimePolicyConfig.from_json(
        deployment.policy_dir("walking") / "runtime_policy_config.json"
    )
    if abs(float(standing_cfg.ctrl_dt) - float(walking_cfg.ctrl_dt)) > 1e-9:
        raise SystemExit(
            "Standing/walking control periods differ: "
            f"{standing_cfg.ctrl_dt} != {walking_cfg.ctrl_dt}"
        )
    ctrl_dt = float(walking_cfg.ctrl_dt)

    from configs import WrRuntimeConfig

    hardware_config_path = _resolve_hardware_config_path(args.hardware_config)
    hardware_cfg = WrRuntimeConfig.load(
        hardware_config_path,
        robot_config_path=deployment.robot_config_path,
    )
    if not args.dry_run:
        _validate_footswitch_configuration(
            enabled=bool(getattr(hardware_cfg.foot_switches, "enabled", True)),
            policy_specs=(
                [("standing", standing_bundle.spec)]
                if args.stable_only
                else [
                    ("standing", standing_bundle.spec),
                    ("walking", walking_bundle.spec),
                ]
            ),
        )
    walking_names = list(walking_bundle.spec.robot.actuator_names)
    standing_names = list(standing_bundle.spec.robot.actuator_names)
    if walking_names != standing_names:
        raise SystemExit(
            "Standing and walking policy actuator orders must match: "
            f"standing={standing_names}, walking={walking_names}"
        )
    hardware_names, hardware_home, hardware_min, hardware_max = (
        _walking_runtime_plan(walking_bundle.spec)
    )
    _standing_runtime_plan(standing_bundle.spec)
    telemetry = _create_telemetry_recorder(
        args,
        actuator_names=walking_names,
        ctrl_dt=ctrl_dt,
        bundle_path=bundle_path,
        hardware_config_path=hardware_config_path,
    )

    standing_policy = _load_runtime_onnx_policy(standing_bundle)
    walking_policy = _load_runtime_onnx_policy(walking_bundle)

    if args.dry_run:
        base_robot_io = MockRobotIO(
            actuator_names=hardware_names,
            control_dt=ctrl_dt,
            home_q_rad=hardware_home,
        )
        realtime = False
    else:
        base_robot_io = _build_hardware_robot_io(
            runtime_config_path=hardware_config_path,
            actuator_names=hardware_names,
            control_dt=ctrl_dt,
            loaded_runtime_config=hardware_cfg,
        )
        if not args.skip_hardware_preflight:
            try:
                _run_hardware_preflight(
                    robot_io=base_robot_io,
                    actuator_names=hardware_names,
                    home_q_rad=hardware_home,
                    joint_min_rad=hardware_min,
                    joint_max_rad=hardware_max,
                    imu_startup_timeout_s=float(args.imu_startup_timeout_s),
                    home_tolerance_deg=float(args.preflight_home_tolerance_deg),
                    max_tilt_deg=float(args.startup_stability_max_tilt_deg),
                )
            except BaseException:
                base_robot_io.close()
                raise
        elif hasattr(base_robot_io, "wait_for_valid_imu_sample"):
            base_robot_io.wait_for_valid_imu_sample(
                timeout_s=float(args.imu_startup_timeout_s)
            )
        realtime = not args.no_realtime

    velocity_cmd = _parse_velocity_cmd(
        args.velocity_cmd, list(walking_cfg.default_velocity_cmd)
    )
    manifest_transition = deployment.manifest.get("transition", {})
    standing_min_s = float(manifest_transition.get("standing_min_duration_s", 2.0))
    requested_standing_s = max(0.0, float(args.startup_home_hold_s))
    standing_steps = max(
        1,
        int(round(max(standing_min_s, requested_standing_s) / ctrl_dt)),
    )
    pose_blend_steps = (
        0
        if bool(args.dry_run)
        else max(
            0,
            int(round(max(0.0, float(args.startup_pose_blend_s)) / ctrl_dt)),
        )
    )
    pose_hold_steps = (
        0
        if pose_blend_steps <= 0
        else max(
            0,
            int(round(max(0.0, float(args.startup_pose_hold_s)) / ctrl_dt)),
        )
    )
    pose_prep_steps = pose_blend_steps + pose_hold_steps
    standing_robot_io = base_robot_io
    if pose_blend_steps > 0:
        try:
            initial_signals = base_robot_io.read()
        except BaseException:
            base_robot_io.close()
            raise
        initial_q = np.asarray(
            initial_signals.joint_pos_rad, dtype=np.float32
        ).reshape(-1)
        if initial_q.size != len(standing_names) or not np.all(np.isfinite(initial_q)):
            base_robot_io.close()
            raise SystemExit(
                "Cannot start standing pose blend: initial joint readback is invalid "
                f"(size={initial_q.size}, expected={len(standing_names)})."
            )
        standing_robot_io = _TargetBlendRobotIO(
            base_robot_io,
            initial_target=initial_q,
            blend_steps=pose_blend_steps,
        )
    standing_runner = StandingPolicyRunner(
        spec=standing_bundle.spec,
        policy=standing_policy,
        robot_io=standing_robot_io,
        runtime_config=standing_cfg,
    )
    manifest_blend_s = float(manifest_transition.get("walking_action_ramp_s", 0.5))
    blend_s = (
        float(args.startup_action_ramp_s)
        if float(args.startup_action_ramp_s) > 0.0
        else manifest_blend_s
    )
    blend_steps = max(0, int(round(blend_s / ctrl_dt)))

    print(
        f"Running deployment bundle {bundle_path} | hardware_config={hardware_config_path} "
        f"| control_hz={1.0 / ctrl_dt:.1f} | hardware_actuators={len(hardware_names)} "
        f"| standing=({standing_bundle.spec.model.obs_dim},"
        f"{standing_bundle.spec.model.action_dim}) "
        f"| walking=({walking_bundle.spec.model.obs_dim},"
        f"{walking_bundle.spec.model.action_dim}) "
        f"| stable_only={bool(args.stable_only)} "
        f"| fall_tilt_deg={float(args.fall_tilt_deg):.1f} "
        f"| startup_pose_blend_steps={pose_blend_steps} "
        f"| startup_pose_hold_steps={pose_hold_steps}",
        flush=True,
    )

    try:
        if pose_prep_steps > 0:
            _run_startup_home_hold(
                runner=standing_runner,
                velocity_cmd=np.zeros(3, dtype=np.float32),
                steps=pose_prep_steps,
                log_steps=args.log_steps,
                ctrl_dt=ctrl_dt,
                realtime=realtime,
                leg_indices=_actuator_indices(
                    standing_names, tuple(name for _, name in _LEG_LOG_JOINTS)
                ),
                stability_check=not bool(args.disable_startup_stability_check),
                stability_max_tilt_deg=float(args.startup_stability_max_tilt_deg),
                confirm_before_walk=False,
                confirm_imu_timeout_s=float(args.imu_startup_timeout_s),
                fall_tilt_deg=float(args.fall_tilt_deg),
                telemetry=telemetry,
                telemetry_phase="startup_pose",
            )
        if args.stable_only:
            if not args.dry_run:
                print(
                    "Stable-only control will run until interrupted (Ctrl+C).",
                    flush=True,
                )
                _run_standing_stabilization(
                    runner=standing_runner,
                    steps=standing_steps,
                    log_steps=args.log_steps,
                    ctrl_dt=ctrl_dt,
                    realtime=realtime,
                    actuator_names=standing_names,
                    diagnostic_log_policy=bool(args.diagnostic_log_policy),
                    stability_check=not bool(args.disable_startup_stability_check),
                    stability_max_tilt_deg=float(
                        args.startup_stability_max_tilt_deg
                    ),
                    confirm_before_walk=False,
                    confirm_imu_timeout_s=float(args.imu_startup_timeout_s),
                    fall_tilt_deg=float(args.fall_tilt_deg),
                    telemetry=telemetry,
                )
            run_policy_loop(
                runner=standing_runner,
                max_steps=_policy_loop_max_steps(
                    stable_only=True,
                    dry_run=bool(args.dry_run),
                    max_steps=int(args.max_steps),
                ),
                velocity_cmd=np.zeros(3, dtype=np.float32),
                log_steps=args.log_steps,
                ctrl_dt=ctrl_dt,
                realtime=realtime,
                actuator_names=standing_names,
                diagnostic_log_policy=bool(args.diagnostic_log_policy),
                fall_tilt_deg=float(args.fall_tilt_deg),
                telemetry=telemetry,
                telemetry_phase="standing",
            )
        else:
            last_standing = _run_standing_stabilization(
                runner=standing_runner,
                steps=standing_steps,
                log_steps=args.log_steps,
                ctrl_dt=ctrl_dt,
                realtime=realtime,
                actuator_names=standing_names,
                diagnostic_log_policy=bool(args.diagnostic_log_policy),
                stability_check=(
                    not bool(args.disable_startup_stability_check)
                    and not bool(args.dry_run)
                ),
                stability_max_tilt_deg=float(args.startup_stability_max_tilt_deg),
                confirm_before_walk=(
                    bool(args.confirm_before_walk) and not bool(args.dry_run)
                ),
                confirm_imu_timeout_s=float(args.imu_startup_timeout_s),
                fall_tilt_deg=float(args.fall_tilt_deg),
                telemetry=telemetry,
            )
            blended_robot_io = _TargetBlendRobotIO(
                base_robot_io,
                initial_target=np.asarray(
                    last_standing["target_q_rad"], dtype=np.float32
                ),
                blend_steps=blend_steps,
            )
            walking_runner = RuntimePolicyRunner(
                spec=walking_bundle.spec,
                runtime_config=walking_cfg,
                policy=walking_policy,
                robot_io=blended_robot_io,
                zero_cmd_hold_home_deadzone=(
                    None
                    if bool(args.disable_zero_cmd_hold_home)
                    else max(0.0, float(args.zero_cmd_hold_home_deadzone))
                ),
            )
            run_policy_loop(
                runner=walking_runner,
                max_steps=args.max_steps,
                velocity_cmd=velocity_cmd,
                log_steps=args.log_steps,
                ctrl_dt=ctrl_dt,
                realtime=realtime,
                actuator_names=walking_names,
                diagnostic_log_policy=bool(args.diagnostic_log_policy),
                startup_command_ramp_steps=max(
                    0,
                    int(round(float(args.startup_command_ramp_s) / ctrl_dt)),
                ),
                startup_action_ramp_steps=blend_steps,
                startup_stability_check=False,
                fall_tilt_deg=float(args.fall_tilt_deg),
                telemetry=telemetry,
                telemetry_phase="walking",
            )
    finally:
        base_robot_io.close()
    print("Run complete.", flush=True)
    return 0


def _run_policy_from_args(args: argparse.Namespace) -> int:
    stable_only = bool(args.stable_only)
    bundle_path = _resolve_run_bundle_path(
        bundle_arg=args.bundle,
        stable_only=stable_only,
    )
    if is_deployment_bundle(bundle_path):
        return _run_deployment_bundle_from_args(args, bundle_path)
    bundle = PolicyBundle.load(bundle_path)
    validate_spec(bundle.spec)
    from configs import WrRuntimeConfig

    hardware_config_path = _resolve_hardware_config_path(args.hardware_config)
    loaded_hardware_config = WrRuntimeConfig.load(hardware_config_path)

    layout_id = str(bundle.spec.observation.layout_id)
    if stable_only:
        if layout_id not in _STANDING_LAYOUT_IDS:
            raise SystemExit(
                "The integrated stable-only bundle must use a standing layout "
                f"from {sorted(_STANDING_LAYOUT_IDS)!r}; got {layout_id!r}."
            )
        if int(bundle.spec.model.action_dim) != 17:
            raise SystemExit(
                "The integrated stable-only bundle must have 17 policy actions; "
                f"got {bundle.spec.model.action_dim}."
            )
    runtime_config: RuntimePolicyConfig | None = None
    if layout_id in _WALKING_LAYOUT_IDS:
        runtime_cfg_path = bundle_path / "runtime_policy_config.json"
        runtime_config = RuntimePolicyConfig.from_json(runtime_cfg_path)
        ctrl_dt = float(runtime_config.ctrl_dt)
        actuator_names = list(bundle.spec.robot.actuator_names)
        (
            hardware_actuator_names,
            hardware_home,
            hardware_joint_min,
            hardware_joint_max,
        ) = _walking_runtime_plan(bundle.spec)
    elif layout_id in _STANDING_LAYOUT_IDS:
        ctrl_dt = _load_optional_runtime_control_dt(bundle_path, default=0.02)
        actuator_names = list(bundle.spec.robot.actuator_names)
        (
            hardware_actuator_names,
            hardware_home,
            hardware_joint_min,
            hardware_joint_max,
        ) = _standing_runtime_plan(bundle.spec)
    else:
        raise SystemExit(
            f"Unsupported runtime layout={layout_id!r}; supported layouts are "
            f"{sorted(_WALKING_LAYOUT_IDS)!r} and {sorted(_STANDING_LAYOUT_IDS)!r}."
        )

    if not args.dry_run:
        _validate_footswitch_configuration(
            enabled=bool(
                getattr(loaded_hardware_config.foot_switches, "enabled", True)
            ),
            policy_specs=[("standing" if stable_only else "walking", bundle.spec)],
        )

    telemetry = _create_telemetry_recorder(
        args,
        actuator_names=hardware_actuator_names,
        ctrl_dt=ctrl_dt,
        bundle_path=bundle_path,
        hardware_config_path=hardware_config_path,
    )

    if stable_only:
        if args.velocity_cmd is not None:
            print(
                "--stable-only ignores --velocity-cmd and uses [0.0, 0.0, 0.0].",
                flush=True,
            )
        velocity_cmd = np.zeros(3, dtype=np.float32)
    else:
        velocity_cmd = _parse_velocity_cmd(
            args.velocity_cmd,
            _default_velocity_cmd_for_layout(bundle.spec, runtime_config),
        )

    policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=bundle.spec.model.input_name,
        output_name=bundle.spec.model.output_name,
        expected_obs_dim=int(bundle.spec.model.obs_dim),
        expected_action_dim=int(bundle.spec.model.action_dim),
    )
    # Fail fast on ONNX/spec dim mismatch BEFORE the first control tick (mirrors
    # wildrobot-validate-bundle; the run command must not silently broadcast a
    # wrong-sized action into a full joint target).
    if policy.info.obs_dim is not None and int(policy.info.obs_dim) != int(
        bundle.spec.model.obs_dim
    ):
        raise SystemExit(
            f"ONNX obs_dim {policy.info.obs_dim} != spec {bundle.spec.model.obs_dim}"
        )
    if policy.info.action_dim is not None and int(policy.info.action_dim) != int(
        bundle.spec.model.action_dim
    ):
        raise SystemExit(
            f"ONNX action_dim {policy.info.action_dim} != spec "
            f"{bundle.spec.model.action_dim}"
        )

    if args.dry_run:
        robot_io = MockRobotIO(
            actuator_names=hardware_actuator_names,
            control_dt=ctrl_dt,
            home_q_rad=hardware_home,
        )
        realtime = False  # dry-run is a smoke test; never sleep
    else:
        if not args.skip_hardware_preflight and hardware_home is None:
            raise SystemExit(
                "policy_spec.robot.home_ctrl_rad is required for hardware preflight"
            )
        robot_io = _build_hardware_robot_io(
            runtime_config_path=hardware_config_path,
            actuator_names=hardware_actuator_names,
            control_dt=ctrl_dt,
            loaded_runtime_config=loaded_hardware_config,
        )
        if not args.skip_hardware_preflight:
            try:
                _run_hardware_preflight(
                    robot_io=robot_io,
                    actuator_names=hardware_actuator_names,
                    home_q_rad=hardware_home,
                    joint_min_rad=hardware_joint_min,
                    joint_max_rad=hardware_joint_max,
                    imu_startup_timeout_s=float(args.imu_startup_timeout_s),
                    home_tolerance_deg=float(args.preflight_home_tolerance_deg),
                    max_tilt_deg=float(args.startup_stability_max_tilt_deg),
                )
            except BaseException:
                try:
                    robot_io.close()
                finally:
                    raise
        elif hasattr(robot_io, "wait_for_valid_imu_sample"):
            print(
                "Skipping hardware preflight; waiting for first valid IMU sample "
                f"(timeout {float(args.imu_startup_timeout_s):.1f}s)...",
                flush=True,
            )
            robot_io.wait_for_valid_imu_sample(
                timeout_s=float(args.imu_startup_timeout_s)
            )
        realtime = not args.no_realtime

    startup_pose_blend_steps = 0
    if (
        stable_only
        and not bool(args.dry_run)
        and float(args.startup_home_hold_s) > 0.0
        and float(args.startup_pose_blend_s) > 0.0
    ):
        startup_home_steps = max(
            1,
            int(round(float(args.startup_home_hold_s) / max(float(ctrl_dt), 1e-9))),
        )
        startup_pose_blend_steps = min(
            startup_home_steps,
            max(
                1,
                int(
                    round(
                        float(args.startup_pose_blend_s)
                        / max(float(ctrl_dt), 1e-9)
                    )
                ),
            ),
        )
        try:
            initial_signals = robot_io.read()
        except BaseException:
            robot_io.close()
            raise
        initial_q = np.asarray(
            initial_signals.joint_pos_rad, dtype=np.float32
        ).reshape(-1)
        if initial_q.size != len(hardware_actuator_names) or not np.all(
            np.isfinite(initial_q)
        ):
            robot_io.close()
            raise SystemExit(
                "Cannot start standing pose blend: initial joint readback is invalid "
                f"(size={initial_q.size}, expected={len(hardware_actuator_names)})."
            )
        robot_io = _TargetBlendRobotIO(
            robot_io,
            initial_target=initial_q,
            blend_steps=startup_pose_blend_steps,
        )

    zero_cmd_hold_home_deadzone = (
        None
        if bool(args.disable_zero_cmd_hold_home) or layout_id in _STANDING_LAYOUT_IDS
        else max(0.0, float(args.zero_cmd_hold_home_deadzone))
    )
    if layout_id in _STANDING_LAYOUT_IDS:
        runner = StandingPolicyRunner(
            spec=bundle.spec,
            policy=policy,
            robot_io=robot_io,
            zero_cmd_hold_home_deadzone=zero_cmd_hold_home_deadzone,
        )
    else:
        assert runtime_config is not None
        runner = RuntimePolicyRunner(
            spec=bundle.spec,
            runtime_config=runtime_config,
            policy=policy,
            robot_io=robot_io,
            zero_cmd_hold_home_deadzone=zero_cmd_hold_home_deadzone,
        )
    cmd_norm = float(np.max(np.abs(velocity_cmd)))
    startup_deadzone = 0.0 if zero_cmd_hold_home_deadzone is None else float(
        zero_cmd_hold_home_deadzone
    )
    startup_home_hold_steps = _startup_home_hold_steps(
        stable_only=stable_only,
        dry_run=bool(args.dry_run),
        command_norm=cmd_norm,
        command_deadzone=startup_deadzone,
        duration_s=float(args.startup_home_hold_s),
        ctrl_dt=float(ctrl_dt),
    )
    startup_command_ramp_steps = 0
    if cmd_norm > startup_deadzone and float(args.startup_command_ramp_s) > 0.0:
        startup_command_ramp_steps = max(
            1,
            int(round(float(args.startup_command_ramp_s) / max(float(ctrl_dt), 1e-9))),
        )
    startup_action_ramp_steps = 0
    if cmd_norm > startup_deadzone and float(args.startup_action_ramp_s) > 0.0:
        startup_action_ramp_steps = max(
            1,
            int(round(float(args.startup_action_ramp_s) / max(float(ctrl_dt), 1e-9))),
        )
    startup_stability_max_tilt_deg = float(args.startup_stability_max_tilt_deg)

    control_hz = 1.0 / float(ctrl_dt) if float(ctrl_dt) > 0.0 else float("nan")
    runtime_mode = (
        f"residual_base={runtime_config.loc_ref_residual_base}"
        if runtime_config is not None
        else "stand_contract=wr_obs_v1"
    )
    print(
        f"Running bundle {bundle_path} | layout={bundle.spec.observation.layout_id} "
        f"| {runtime_mode} "
        f"| control_hz={control_hz:.1f} "
        f"| hardware_config={hardware_config_path} "
        f"| policy_actuators={len(actuator_names)} "
        f"| hardware_actuators={len(hardware_actuator_names)} "
        f"| cmd={velocity_cmd.tolist()} | dry_run={args.dry_run} "
        f"| stable_only={stable_only} "
        f"| zero_cmd_hold_home={zero_cmd_hold_home_deadzone is not None} "
        f"| startup_home_hold_steps={startup_home_hold_steps} "
        f"| startup_pose_blend_steps={startup_pose_blend_steps} "
        f"| confirm_before_walk={bool(args.confirm_before_walk)} "
        f"| startup_stability_check={not bool(args.disable_startup_stability_check)} "
        f"| startup_stability_max_tilt_deg={startup_stability_max_tilt_deg:.1f} "
        f"| fall_tilt_deg={float(args.fall_tilt_deg):.1f} "
        f"| startup_command_ramp_steps={startup_command_ramp_steps} "
        f"| startup_action_ramp_steps={startup_action_ramp_steps}",
        flush=True,
    )
    try:
        run_policy_loop(
            runner=runner,
            max_steps=_policy_loop_max_steps(
                stable_only=stable_only,
                dry_run=bool(args.dry_run),
                max_steps=int(args.max_steps),
            ),
            velocity_cmd=velocity_cmd,
            log_steps=args.log_steps,
            ctrl_dt=ctrl_dt,
            realtime=realtime,
            actuator_names=actuator_names,
            diagnostic_log_policy=bool(args.diagnostic_log_policy),
            startup_home_hold_steps=startup_home_hold_steps,
            startup_command_ramp_steps=startup_command_ramp_steps,
            startup_action_ramp_steps=startup_action_ramp_steps,
            startup_stability_check=not bool(args.disable_startup_stability_check),
            startup_stability_max_tilt_deg=startup_stability_max_tilt_deg,
            startup_confirm_before_walk=bool(args.confirm_before_walk),
            startup_confirm_imu_timeout_s=float(args.imu_startup_timeout_s),
            fall_tilt_deg=float(args.fall_tilt_deg),
            telemetry=telemetry,
            telemetry_phase="standing" if stable_only else "walking",
        )
    finally:
        try:
            robot_io.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"Warning: robot_io.close() failed: {exc}", flush=True)
    print("Run complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
