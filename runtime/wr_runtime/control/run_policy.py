"""``wildrobot-run-policy`` — deterministic control loop for the latest bundle.

Loads a policy bundle (``policy.onnx`` + ``policy_spec.json`` +
``runtime_policy_config.json``) and runs the v8 home-base-residual control loop
at ``control_hz``.  Supports a hardware-free ``--dry-run`` mode (mock IO) for
smoke tests and safe validation on a developer machine.

Examples
--------
Dry run (no hardware), 5 steps, straight walk::

    uv run --project runtime wildrobot-run-policy \
      --bundle /tmp/wr_runtime_smoke9_bundle_check \
      --dry-run --max-steps 5 --velocity-cmd 0.13,0.0,0.0

Hardware run (on the robot), forward walk for 500 control steps::

    uv run --project runtime wildrobot-run-policy \
      --bundle /path/to/bundle \
      --runtime-config ~/.wildrobot/config.json \
      --max-steps 500 --velocity-cmd 0.13,0.0,0.0
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, TextIO

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.spec import PolicyBundle, validate_spec

from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.control.mock_robot_io import MockRobotIO
from wr_runtime.control.policy_runner import RuntimePolicyRunner
from wr_runtime.control.runtime_policy_config import RuntimePolicyConfig


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


def _build_hardware_robot_io(
    *, runtime_config_path: Path, actuator_names: List[str], control_dt: float
):
    """Construct the real hardware RobotIO from the runtime config.

    Imported lazily (GPIO / serial / IMU bus backends are Linux-only), and wires the
    concrete hardware classes (HiwonderCachedActuators / BNO085IMU / FootSwitches)
    directly — there are no ``*.from_config`` factories.
    """
    from configs import WrRuntimeConfig
    from wr_runtime.hardware.actuators import HiwonderCachedActuators
    from wr_runtime.hardware.bno085 import BNO085IMU
    from wr_runtime.hardware.foot_switches import FootSwitches
    from wr_runtime.hardware.hiwonder_ttl_bus import (
        RawServoBus,
        RawServoBusConfig,
        SerialTransport,
        SerialTransportConfig,
    )
    from wr_runtime.hardware.robot_io import HardwareRobotIO
    from wr_runtime.hardware.servo_io_worker import (
        ServoIOWorker,
        ServoIOWorkerConfig,
    )

    cfg = WrRuntimeConfig.load(runtime_config_path)
    sc = cfg.servo_controller

    # Fail fast with an actionable message if the runtime config does not cover
    # every actuator in the policy spec (the actuator constructor would
    # otherwise raise a bare KeyError mid-init).  This catches stale configs
    # missing newer joints (e.g. left/right_ankle_roll on the v8 21-actuator
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
    read_schedule_mode = getattr(read_schedule, "mode", "full")
    read_schedule_groups = getattr(read_schedule, "groups", [])
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
    read_groups, read_group_schedule = _build_ttl_servo_read_schedule(
        actuator_names=actuator_names,
        servo_ids=sc.servo_ids,
        read_schedule_groups=read_schedule_groups,
        max_cache_age_s=read_schedule_max_cache_age_s,
    )
    transport = SerialTransport(
        SerialTransportConfig(port=sc.port, baudrate=int(sc.baudrate))
    )
    raw_bus = RawServoBus(transport, RawServoBusConfig())
    servo_io = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            servo_ids=tuple(int(sc.servo_ids[name]) for name in actuator_names),
            read_groups=tuple(read_groups),
            read_group_schedule=tuple(read_group_schedule),
        ),
    )
    servo_io.start()
    actuators = HiwonderCachedActuators(
        actuator_names=actuator_names,
        servo_ids=sc.servo_ids,
        default_move_time_ms=default_move_time_ms,
        joint_servo_offset_units=sc.joint_servo_offset_units,
        joint_motor_unit_directions=sc.joint_motor_unit_directions,
        joint_angle_at_zero_unit_deg=sc.joint_angle_at_zero_unit_deg,
        servo_io=servo_io,
        cache_age_limits_s=read_schedule_max_cache_age_s,
        port=sc.port,
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
    foot_switches = FootSwitches(pins=cfg.foot_switches.get_all_pins())
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
    read_schedule_groups: Sequence[Sequence[str]],
    max_cache_age_s: dict[str, float],
):
    from wr_runtime.hardware.servo_io_worker import ServoReadGroup

    groups = list(read_schedule_groups) or [list(actuator_names)]
    unique_groups: dict[str, ServoReadGroup] = {}
    schedule: list[str] = []
    for group_idx, names in enumerate(groups):
        label = _infer_servo_read_group_label(group_idx, names)
        ids = tuple(int(servo_ids[name]) for name in names)
        if label not in unique_groups:
            unique_groups[label] = ServoReadGroup(
                name=label,
                servo_ids=ids,
                max_cache_age_s=_group_cache_age_limit_s(
                    names, max_cache_age_s=max_cache_age_s
                ),
            )
        schedule.append(label)
    return list(unique_groups.values()), schedule


def _infer_servo_read_group_label(group_idx: int, names: Sequence[str]) -> str:
    names_set = set(names)
    if names_set and all(name.startswith("left_") for name in names_set):
        if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
            return "left_leg"
    if names_set and all(name.startswith("right_") for name in names_set):
        if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
            return "right_leg"
    if any("wrist" in name or "shoulder" in name or "elbow" in name for name in names_set):
        return "torso_arms"
    return f"group_{group_idx}"


def _group_cache_age_limit_s(
    names: Sequence[str], *, max_cache_age_s: dict[str, float]
) -> float:
    if any(part in name for name in names for part in ("hip", "knee", "ankle")):
        key = "leg"
    elif any("wrist" in name for name in names):
        key = "wrist"
    elif any(part in name for name in names for part in ("shoulder", "elbow")):
        key = "arm"
    else:
        key = "default"
    defaults = {"leg": 0.12, "arm": 1.25, "wrist": 1.25, "default": 1.25}
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
    signals = info.get("signals")
    if signals is None:
        return ""
    switches = np.asarray(signals.foot_switches, dtype=np.float32).reshape(-1)
    if switches.size != 4:
        return ""
    values = [int(round(float(v))) for v in switches]
    return f"fs=[LT={values[0]},LH={values[1]},RT={values[2]},RH={values[3]}]"


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
    if signals is not None and actuator_names:
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

    return "diag[" + " ".join(p for p in parts if p) + "]"


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


def _timing_values(samples: List[dict], key: str) -> List[float]:
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


def _timing_avg(samples: List[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.mean(values))


def _timing_max(samples: List[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.max(values))


def _timing_sum(samples: List[dict], key: str) -> float | None:
    values = _timing_values(samples, key)
    if not values:
        return None
    return float(np.sum(values))


def _timing_percentile(samples: List[dict], key: str, percentile: float) -> float | None:
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


def _print_io_bottleneck_summary(timing_samples: List[dict]) -> None:
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
    timing_samples: List[dict],
    ctrl_dt: float,
    realtime: bool,
    completed: bool = True,
    servo_metric_samples: List[dict] | None = None,
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
    require_all_footswitches: bool,
    home_tolerance_deg: float,
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
        errors=errors,
    )
    _preflight_footswitches(
        robot_io=robot_io,
        require_all_footswitches=require_all_footswitches,
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


def _preflight_imu(*, robot_io, imu_startup_timeout_s: float, errors: List[str]) -> None:
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
    quat = np.asarray(getattr(sample, "quat_xyzw", []), dtype=np.float32).reshape(-1)
    gyro = np.asarray(getattr(sample, "gyro_rad_s", []), dtype=np.float32).reshape(-1)
    quat_norm = float(np.linalg.norm(quat)) if quat.size == 4 else float("nan")
    imu = robot_io.imu
    diag = getattr(imu, "diag", None)
    diag_text = f" diag={diag}" if diag else ""
    print(
        "  IMU: "
        f"valid={valid} "
        f"fresh={fresh} "
        f"quat={np.round(quat, 4).tolist()} "
        f"quat_norm={quat_norm:.3f} "
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
    if gyro.size != 3 or not np.all(np.isfinite(gyro)):
        errors.append(f"IMU gyro is invalid: gyro={gyro.tolist()}")


def _preflight_footswitches(
    *,
    robot_io,
    require_all_footswitches: bool,
    errors: List[str],
    warnings: List[str],
) -> None:
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
        if require_all_footswitches:
            errors.append(
                "footswitches open at walk start: "
                f"{open_names}; use --allow-unpressed-footswitch for suspended tests"
            )
        else:
            warnings.append(
                "initial footswitches open at walk start: "
                f"{open_names}; continuing because unpressed footswitches are allowed"
            )


def run_policy_loop(
    *,
    runner: RuntimePolicyRunner,
    max_steps: int,
    velocity_cmd: np.ndarray,
    log_steps: int,
    ctrl_dt: float,
    realtime: bool,
    actuator_names: Optional[List[str]] = None,
    diagnostic_log_policy: bool = False,
) -> List[dict]:
    """Run the control loop for ``max_steps`` iterations; return per-log infos."""
    logs: List[dict] = []
    leg_indices = _actuator_indices(
        actuator_names, tuple(name for _, name in _LEG_LOG_JOINTS)
    )
    timing_samples: List[dict] = []
    servo_metric_samples: List[dict] = []
    last_loop_start_s: float | None = None
    completed = False
    try:
        for step in range(int(max_steps)):
            loop_start_s = time.monotonic()
            loop_period_s = (
                None if last_loop_start_s is None else loop_start_s - last_loop_start_s
            )
            last_loop_start_s = loop_start_s
            info = runner.step(velocity_cmd)
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
            if log_steps > 0 and (step % log_steps == 0 or step == max_steps - 1):
                applied = info["applied_action"]
                target = info["target_q_rad"]
                leg_applied_max = (
                    float(np.max(np.abs(applied[leg_indices]))) if leg_indices else None
                )
                leg_summary = _format_leg_targets_deg(target, actuator_names)
                observed = np.asarray(info["signals"].joint_pos_rad, dtype=np.float32)
                observed_leg_summary = _format_leg_values_deg(observed, actuator_names)
                leg_err_max = (
                    float(np.max(np.abs(target[leg_indices] - observed[leg_indices])))
                    if leg_indices and observed.size >= target.size
                    else None
                )
                foot_summary = _format_foot_switches(info)
                extra_parts = []
                if leg_applied_max is not None:
                    extra_parts.append(f"leg|applied|max={leg_applied_max:.3f}")
                if leg_summary:
                    extra_parts.append(f"leg_deg={leg_summary}")
                if observed_leg_summary:
                    extra_parts.append(f"obs_leg_deg={observed_leg_summary}")
                if leg_err_max is not None:
                    extra_parts.append(
                        f"leg_err|max_deg={float(np.rad2deg(leg_err_max)):.1f}"
                    )
                if foot_summary:
                    extra_parts.append(foot_summary)
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
    parser = argparse.ArgumentParser(
        description="Run a WildRobot policy bundle (latest v8 home-residual contract)."
    )
    parser.add_argument(
        "--bundle", type=str, required=True,
        help="Bundle directory (policy.onnx + policy_spec.json + runtime_policy_config.json)",
    )
    parser.add_argument(
        "--runtime-config", type=str, default=None,
        help="Hardware runtime config JSON (servo IDs/calibration). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with mock IO (no hardware): exercises the full loop for smoke tests.",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Number of control steps.")
    parser.add_argument("--log-steps", type=int, default=20, help="Log every N steps (0=off).")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log",
        type=str,
        default=None,
        help="Write stdout/stderr to this file while still printing to the console.",
    )
    log_group.add_argument(
        "--log-only",
        type=str,
        default=None,
        help="Write stdout/stderr to this file without printing to the console.",
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
    footswitch_group = parser.add_mutually_exclusive_group()
    footswitch_group.add_argument(
        "--allow-unpressed-footswitch",
        dest="allow_unpressed_footswitch",
        action="store_true",
        default=True,
        help=(
            "Hardware preflight only: do not fail if one or more footswitches are "
            "open (default)."
        ),
    )
    footswitch_group.add_argument(
        "--require-pressed-footswitch",
        dest="allow_unpressed_footswitch",
        action="store_false",
        help="Hardware preflight only: fail if one or more footswitches are open.",
    )
    parser.add_argument(
        "--preflight-home-tolerance-deg",
        type=float,
        default=25.0,
        help="Warn when any servo starts this many degrees away from policy home.",
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
            "bin, phase, command, gyro, and normalized joint-velocity summaries "
            "to each normal step log line."
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

    log_path = args.log if args.log is not None else args.log_only
    if log_path is not None:
        with _output_log_context(log_path, mirror_console=args.log is not None):
            try:
                return _run_policy_from_args(args)
            except SystemExit as exc:
                code = exc.code
                if code is None:
                    return 0
                if isinstance(code, int):
                    return int(code)
                print(code, file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                print("Interrupted.", file=sys.stderr)
                return 130
            except BaseException:
                traceback.print_exc()
                return 1

    return _run_policy_from_args(args)


def _run_policy_from_args(args: argparse.Namespace) -> int:
    bundle_path = Path(args.bundle)
    bundle = PolicyBundle.load(bundle_path)
    validate_spec(bundle.spec)
    runtime_cfg_path = bundle_path / "runtime_policy_config.json"
    runtime_config = RuntimePolicyConfig.from_json(runtime_cfg_path)

    velocity_cmd = _parse_velocity_cmd(
        args.velocity_cmd, runtime_config.default_velocity_cmd
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

    actuator_names = list(bundle.spec.robot.actuator_names)
    ctrl_dt = float(runtime_config.ctrl_dt)

    if args.dry_run:
        home = (
            np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32)
            if bundle.spec.robot.home_ctrl_rad is not None
            else None
        )
        robot_io = MockRobotIO(
            actuator_names=actuator_names, control_dt=ctrl_dt, home_q_rad=home
        )
        realtime = False  # dry-run is a smoke test; never sleep
    else:
        if args.runtime_config is None:
            raise SystemExit("--runtime-config is required unless --dry-run is set.")
        if not args.skip_hardware_preflight and bundle.spec.robot.home_ctrl_rad is None:
            raise SystemExit(
                "policy_spec.robot.home_ctrl_rad is required for hardware preflight"
            )
        robot_io = _build_hardware_robot_io(
            runtime_config_path=Path(args.runtime_config),
            actuator_names=actuator_names,
            control_dt=ctrl_dt,
        )
        if not args.skip_hardware_preflight:
            home = np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32)
            joint_min = np.asarray(
                [
                    float(bundle.spec.robot.joints[name].range_min_rad)
                    for name in actuator_names
                ],
                dtype=np.float32,
            )
            joint_max = np.asarray(
                [
                    float(bundle.spec.robot.joints[name].range_max_rad)
                    for name in actuator_names
                ],
                dtype=np.float32,
            )
            try:
                _run_hardware_preflight(
                    robot_io=robot_io,
                    actuator_names=actuator_names,
                    home_q_rad=home,
                    joint_min_rad=joint_min,
                    joint_max_rad=joint_max,
                    imu_startup_timeout_s=float(args.imu_startup_timeout_s),
                    require_all_footswitches=not bool(
                        args.allow_unpressed_footswitch
                    ),
                    home_tolerance_deg=float(args.preflight_home_tolerance_deg),
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
            robot_io.wait_for_valid_imu_sample(timeout_s=float(args.imu_startup_timeout_s))
        realtime = not args.no_realtime

    runner = RuntimePolicyRunner(
        spec=bundle.spec,
        runtime_config=runtime_config,
        policy=policy,
        robot_io=robot_io,
        zero_cmd_hold_home_deadzone=(
            None
            if bool(args.disable_zero_cmd_hold_home)
            else max(0.0, float(args.zero_cmd_hold_home_deadzone))
        ),
    )

    print(
        f"Running bundle {bundle_path} | layout={bundle.spec.observation.layout_id} "
        f"| residual_base={runtime_config.loc_ref_residual_base} "
        f"| control_hz={runtime_config.control_hz:.1f} "
        f"| cmd={velocity_cmd.tolist()} | dry_run={args.dry_run} "
        f"| zero_cmd_hold_home={not bool(args.disable_zero_cmd_hold_home)}",
        flush=True,
    )
    try:
        run_policy_loop(
            runner=runner,
            max_steps=args.max_steps,
            velocity_cmd=velocity_cmd,
            log_steps=args.log_steps,
            ctrl_dt=ctrl_dt,
            realtime=realtime,
            actuator_names=actuator_names,
            diagnostic_log_policy=bool(args.diagnostic_log_policy),
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
