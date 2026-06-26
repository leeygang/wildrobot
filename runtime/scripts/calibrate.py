#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import signal
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path when running as a script (e.g. `python3 runtime/scripts/calibrate.py ...`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if _RUNTIME_ROOT.exists() and str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from configs.config import ServoConfig, WrRuntimeConfig  # noqa: E402

# Calibration constants (use ServoConfig constants for conversion)
DEFAULT_MOVE_MS = 300
DEFAULT_STEP_UNITS = 5
DEFAULT_DELTA_RAD = 0.5
VERIFY_TOL_RAD = 0.05
PANIC_KEY = "q"
DEFAULT_PAUSE_S = 3.0
POLICY_ACTION_HOME_PAUSE_S = 3.0
RANGE_TEST_MS = 3000
DEFAULT_IMU_SAMPLES = 100
DEFAULT_IMU_BASELINE_S = 3.0
DEFAULT_IMU_MOTION_S = 3.0
DEFAULT_IMU_PREVIEW_S = 2.0
DEFAULT_IMU_PROGRESS_EVERY_S = 0.0
DEFAULT_PROMPT_REPRINT_S = 3.0
DEFAULT_IMU_GUIDANCE_PAUSE_S = 5.0
DEFAULT_IMU_MIN_ANGLE_RAD = 0.5
# Keep status prints sparse; some SSH/PTY stacks will drop/reorder lines under high churn.
DEFAULT_IMU_STATUS_HZ = 2.0
DEFAULT_IMU_MAX_ATTEMPTS = 3
DEFAULT_IMU_AXIS_ALIGN_COS = 0.9  # require inferred axis to align with expected sensor axis
DEFAULT_IMU_MAX_ATTEMPTS_SINGLE_AXIS = 5
CALIBRATION_IMU_SAMPLING_HZ = 200

POSITIVE_HINTS = {
    # These hints describe what POSITIVE MuJoCo radians look like physically.
    "left_hip_pitch": "left leg swings forward",
    "left_hip_roll": "left leg moves inward (toward body midline)",
    "left_knee_pitch": "left knee bends (foot moves backward)",
    "left_ankle_pitch": "left toes go up (dorsiflex)",
    "right_hip_pitch": "right leg swings backward",  # inverted range
    "right_hip_roll": "right leg moves outward (away from body midline)",  # inverted range
    "right_knee_pitch": "right knee bends (foot moves backward)",  # same as left
    "right_ankle_pitch": "right toes go up (dorsiflex)",  # same as left
    # Torso
    "waist_yaw": "torso turns left (counter-clockwise viewed from above)",
    # Left arm
    "left_shoulder_pitch": "left upper arm lifts forward",
    "left_shoulder_roll": "left upper arm lifts inward (toward torso)",
    "left_elbow_pitch": "left elbow bends",
    "left_wrist_yaw": "left forearm rotates outward",
    "left_wrist_pitch": "left hand tilts upward",
    # Right arm
    "right_shoulder_pitch": "right upper arm lifts backward",  # inverted range
    "right_shoulder_roll": "right upper arm lifts outward (away from torso)",  # inverted range
    "right_elbow_pitch": "right elbow bends",
    "right_wrist_yaw": "right forearm rotates outward",
    "right_wrist_pitch": "right hand tilts upward",
}


@dataclass
class JointState:
    offset: int
    motor_sign: int


@dataclass(frozen=True)
class JointAxisMetadata:
    local_axis: Optional[Tuple[float, float, float]]
    init_world_axis: Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class PolicyActionSetup:
    base_rad_by_joint: Dict[str, float]
    residual_scale_by_joint: Dict[str, float]
    residual_base: str
    residual_mode: str
    action_delay_steps: int
    action_filter_alpha: float
    source: str


@dataclass(frozen=True)
class PolicyActionEvaluation:
    action_raw: float
    action_applied: float
    residual_rad: float
    target_rad_unclipped: float
    target_rad: float
    clipped_by_joint_range: bool


def normalize_joint_state(offset: float | int, motor_sign: float | int) -> JointState:
    offset_int = int(round(float(offset)))
    motor_sign_f = float(motor_sign)
    motor_sign_int = int(round(motor_sign_f))
    if motor_sign_int not in (-1, 1):
        motor_sign_int = 1 if motor_sign_f >= 0 else -1
    return JointState(offset=offset_int, motor_sign=motor_sign_int)


def _parse_axis(value: object) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _axis_label(axis: Optional[Tuple[float, float, float]]) -> str:
    if axis is None:
        return "n/a"
    values = [float(v) for v in axis]
    idx = int(np.argmax(np.abs(values)))
    if abs(values[idx]) < 1e-9:
        return "n/a"
    sign = "+" if values[idx] >= 0.0 else "-"
    return f"{sign}{'XYZ'[idx]}"


def _format_axis(axis: Optional[Tuple[float, float, float]]) -> str:
    if axis is None:
        return "n/a"
    return "[" + ", ".join(f"{float(v):+.2f}" for v in axis) + "]"


def _axis_summary(axis: Optional[Tuple[float, float, float]]) -> str:
    return f"{_format_axis(axis)} ({_axis_label(axis)})"


def prompt_select_joints(joint_names: List[str]) -> List[str]:
    print("Available joints:")
    for idx, name in enumerate(joint_names, start=1):
        print(f"  #{idx}: {name}")
    raw = input("Select joints by number (e.g. #1,#3 or 1 3), or 'all': ").strip().lower()
    if not raw:
        raise ValueError("No joints selected")
    if raw == "all":
        return list(joint_names)

    tokens = [t for t in raw.replace(",", " ").split() if t]
    indices: List[int] = []
    for token in tokens:
        token = token.lstrip("#")
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid joint selector: '{token}'") from exc
        if idx < 1 or idx > len(joint_names):
            raise ValueError(f"Joint index out of range: {idx} (valid: 1..{len(joint_names)})")
        indices.append(idx)

    # Deduplicate while preserving order
    seen: set[int] = set()
    selected: List[str] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(joint_names[idx - 1])
    return selected


def prompt_select_joints_by_servo_id(
    joint_names: List[str],
    *,
    servo_cfgs: Dict[str, ServoConfig],
) -> List[str]:
    print("Available joints:")
    servo_id_to_joint: Dict[int, str] = {}
    for joint in joint_names:
        servo = servo_cfgs[joint]
        min_rad, max_rad = float(servo.rad_range[0]), float(servo.rad_range[1])
        min_deg = float(np.rad2deg(min_rad))
        max_deg = float(np.rad2deg(max_rad))
        servo_id_to_joint[int(servo.id)] = str(joint)
        print(f"  #{int(servo.id)}: {joint}: deg range: {min_deg:.1f}, {max_deg:.1f}")

    raw = input(
        "Select joints by servo id (e.g. #21,#23 or 21 23), 'all', or 'q' to quit: "
    ).strip().lower()
    if not raw:
        raise ValueError("No joints selected")
    if raw in {"q", "quit", "exit"}:
        return []
    if raw == "all":
        return list(joint_names)

    tokens = [t for t in raw.replace(",", " ").split() if t]
    selected: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.lstrip("#")
        if not token.isdigit():
            raise ValueError(f"Invalid servo id selector: '{token}'")
        sid = int(token)
        joint = servo_id_to_joint.get(sid)
        if joint is None:
            valid = ", ".join(str(servo_cfgs[j].id) for j in joint_names)
            raise ValueError(f"Unknown servo id: {sid}. Valid IDs: {valid}")
        if joint not in seen:
            seen.add(joint)
            selected.append(joint)
    return selected


_FOOTSWITCH_RAW_TARGETS_ORDER = (
    "left_toe",
    "left_heel",
    "right_toe",
    "right_heel",
)
_FOOTSWITCH_TARGETS_ORDER = (
    *_FOOTSWITCH_RAW_TARGETS_ORDER,
    "left_foot",
    "right_foot",
)
_FOOTSWITCH_ALL_RAW_TARGET = "all_footswitches"
_FOOTSWITCH_ALL_RAW_ALIASES = {
    "all_footswitch",
    "all_footswitches",
    "all_foot_switch",
    "all_foot_switches",
    "all_switches",
    "all4",
    "all_4",
}


def _is_all_raw_footswitch_selector(raw: str) -> bool:
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in _FOOTSWITCH_ALL_RAW_ALIASES


def _append_footswitch_target(selected: List[str], seen: set[str], name: str) -> None:
    if name not in seen:
        seen.add(name)
        selected.append(name)


def prompt_select_footswitch_targets() -> List[str]:
    print("Available footswitch targets:")
    for idx, name in enumerate(_FOOTSWITCH_TARGETS_ORDER, start=1):
        print(f"  #{idx}: {name}")
    all_raw_idx = len(_FOOTSWITCH_TARGETS_ORDER) + 1
    print(f"  #{all_raw_idx}: {_FOOTSWITCH_ALL_RAW_TARGET} ({', '.join(_FOOTSWITCH_RAW_TARGETS_ORDER)})")
    raw = (
        input(
            "Select targets by number or name (e.g. 1 3 or left_toe,right_foot), "
            "'all_footswitches' for the 4 raw switches, 'all', or 'q' to quit: "
        )
        .strip()
        .lower()
    )
    if not raw:
        raise ValueError("No footswitch targets selected")
    if raw in {"q", "quit", "exit"}:
        return []
    if raw == "all":
        return list(_FOOTSWITCH_TARGETS_ORDER)
    if _is_all_raw_footswitch_selector(raw):
        return list(_FOOTSWITCH_RAW_TARGETS_ORDER)

    tokens = [t for t in raw.replace(",", " ").split() if t]
    indices: List[int] = []
    names: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith("#"):
            token = token[1:]
        if token.isdigit():
            indices.append(int(token))
        else:
            names.append(token)

    selected: List[str] = []
    seen: set[str] = set()

    for idx in indices:
        if idx == all_raw_idx:
            for name in _FOOTSWITCH_RAW_TARGETS_ORDER:
                _append_footswitch_target(selected, seen, name)
            continue
        if idx < 1 or idx > all_raw_idx:
            raise ValueError(
                f"Target index out of range: {idx} (valid: 1..{all_raw_idx})"
            )
        name = _FOOTSWITCH_TARGETS_ORDER[idx - 1]
        _append_footswitch_target(selected, seen, name)

    allowed = {n for n in _FOOTSWITCH_TARGETS_ORDER}
    for name in names:
        name = name.replace("-", "_")
        if _is_all_raw_footswitch_selector(name):
            for raw_name in _FOOTSWITCH_RAW_TARGETS_ORDER:
                _append_footswitch_target(selected, seen, raw_name)
            continue
        if name not in allowed:
            allowed_names = sorted(allowed | {_FOOTSWITCH_ALL_RAW_TARGET})
            raise ValueError(f"Unknown footswitch target: {name!r} (allowed: {allowed_names})")
        _append_footswitch_target(selected, seen, name)

    if not selected:
        raise ValueError("No footswitch targets selected")
    return selected


def _format_footswitch_status_line(status: Dict[str, bool], selected: List[str]) -> str:
    return ", ".join(f"{k}={1 if status[k] else 0}" for k in selected)


def _footswitch_status_from_sample(switches: List[bool]) -> Dict[str, bool]:
    if len(switches) != 4:
        raise ValueError(f"Expected 4 foot switch values, got {len(switches)}")
    left_toe, left_heel, right_toe, right_heel = [bool(x) for x in switches]
    return {
        "left_toe": left_toe,
        "left_heel": left_heel,
        "right_toe": right_toe,
        "right_heel": right_heel,
        "left_foot": bool(left_toe or left_heel),
        "right_foot": bool(right_toe or right_heel),
    }


def _poll_footswitch_control_token(buf: str) -> tuple[str, str]:
    """Non-blocking stdin poll for quit tokens.

    Returns (new_buf, action) where action is one of: 'none', 'back', 'exit'.
    """
    if not sys.stdin.isatty():
        return buf, "none"
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return buf, "none"
    if not r:
        return buf, "none"

    try:
        chunk = sys.stdin.read(1)
    except Exception:
        return buf, "none"
    if chunk == "":
        return buf, "exit"
    if chunk in {"\n", "\r"}:
        token = buf.strip().lower()
        buf = ""
        if token in {"q"}:
            return buf, "back"
        if token in {"quit", "exit"}:
            return buf, "exit"
        return buf, "none"
    return buf + chunk, "none"


def calibrate_footswitches(*, config: WrRuntimeConfig) -> None:
    print("\n== Foot Switch Calibration/Test ==", flush=True)
    print(
        "This mode prints live footswitch status for the selected signals.\n"
        "Expected electrical behavior (per runtime docs):\n"
        "  - Not pressed: GPIO reads HIGH -> runtime reports False\n"
        "  - Pressed (short to GND): GPIO reads LOW -> runtime reports True\n",
        flush=True,
    )

    # Outer loop: select targets; inner loop: live readout.

    try:
        from runtime.wr_runtime.hardware.foot_switches import FootSwitches
    except Exception as exc:
        print(
            "Failed to import FootSwitches driver. If you're on a Raspberry Pi, install Blinka: 'pip install adafruit-blinka'.",
            flush=True,
        )
        raise

    pins = config.foot_switches.get_all_pins()
    try:
        foot = FootSwitches(pins)
    except Exception as exc:
        print("Failed to initialize foot switches. Check wiring and config 'foot_switches' pins.", flush=True)
        raise

    try:
        while True:
            selected = prompt_select_footswitch_targets()
            if not selected:
                return
            print(f"Selected targets: {', '.join(selected)}", flush=True)

            print("\nLive readout starting.", flush=True)
            print("- Press 'q' then Enter to go back to target selection.", flush=True)
            print("- Press 'exit' then Enter to exit this mode.", flush=True)
            print("- '1' means pressed/contact closed; '0' means open.", flush=True)

            buf = ""
            last: Optional[Dict[str, bool]] = None
            last_print_s = 0.0
            while True:
                buf, action = _poll_footswitch_control_token(buf)
                if action == "back":
                    print("(back to selection)", flush=True)
                    break
                if action == "exit":
                    return

                sample = foot.read()
                status = _footswitch_status_from_sample(sample.switches)
                now = time.monotonic()

                changed = False
                if last is None:
                    changed = True
                else:
                    for k in selected:
                        if bool(status[k]) != bool(last.get(k, False)):
                            changed = True
                            break

                heartbeat = (now - last_print_s) >= 1.0
                if changed or heartbeat:
                    line = _format_footswitch_status_line(status, selected)
                    print(line, flush=True)
                    last = {k: bool(status[k]) for k in selected}
                    last_print_s = now

                time.sleep(0.02)
    finally:
        try:
            foot.close()
        except Exception:
            pass


def read_position(controller, servo_id: int, *, retries: int = 5, retry_delay_s: float = 0.1) -> Optional[int]:
    for attempt in range(1, retries + 1):
        try:
            if hasattr(controller, "serial") and hasattr(controller.serial, "reset_input_buffer"):
                controller.serial.reset_input_buffer()
        except Exception:
            pass

        resp = controller.read_servo_positions([servo_id])
        if resp:
            for sid, pos in resp:
                if int(sid) == int(servo_id):
                    return int(pos)

        if attempt < retries:
            time.sleep(retry_delay_s)
    return None


def move_and_wait(
    controller,
    servo_id: int,
    target_units: int,
    move_ms: int,
) -> None:
    controller.move_servos([(servo_id, target_units)], move_ms)
    time.sleep(move_ms / 1000.0 + 0.05)


def announce_and_pause(message: str, pause_s: float) -> None:
    if message:
        print(message)
    if pause_s > 0:
        print(f"Pausing {pause_s:.1f}s...")
        time.sleep(pause_s)

def print_all_joint_positions(
    controller,
    *,
    joint_names: List[str],
    servo_cfgs: Dict[str, ServoConfig],
    states: Dict[str, JointState],
    include_rad: bool,
) -> None:
    servo_ids = [servo_cfgs[j].id for j in joint_names]
    resp = controller.read_servo_positions(servo_ids)
    if not resp:
        print("Position readback failed (no response).")
        return

    pos_by_id = {int(sid): int(pos) for sid, pos in resp}
    print("Current servo positions:")
    for joint in joint_names:
        servo = servo_cfgs[joint]
        units = pos_by_id.get(int(servo.id))
        if units is None:
            print(f"  {joint}: id={servo.id} units=?")
            continue
        if include_rad:
            st = states[joint]
            pos_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                units,
                motor_sign=st.motor_sign,
                offset=st.offset,
            )
            print(f"  {joint}: id={servo.id} units={units} ctrl_rad={pos_rad:+.4f}")
        else:
            print(f"  {joint}: id={servo.id} units={units}")


def read_all_home_ctrl_rad(
    controller,
    *,
    joint_names: List[str],
    servo_cfgs: Dict[str, ServoConfig],
    states: Dict[str, JointState],
) -> List[float]:
    servo_ids = [servo_cfgs[j].id for j in joint_names]
    resp = controller.read_servo_positions(servo_ids)
    if not resp:
        raise RuntimeError("Failed to read servo positions for home recording.")
    pos_by_id = {int(sid): int(pos) for sid, pos in resp}
    home_ctrl: List[float] = []
    missing: List[str] = []
    for joint in joint_names:
        servo = servo_cfgs[joint]
        units = pos_by_id.get(int(servo.id))
        if units is None:
            missing.append(joint)
            continue
        st = states[joint]
        home_ctrl.append(
            servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                units,
                motor_sign=st.motor_sign,
                offset=st.offset,
            )
        )
    if missing:
        raise RuntimeError(f"Missing readback for joints: {missing}")
    return home_ctrl


def yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    resp = input(prompt + suffix).strip().lower()
    if not resp:
        return default
    return resp.startswith("y")


@contextlib.contextmanager
def _alarm_timeout(seconds: float, *, message: str):
    """Best-effort timeout for operations that can hang (Linux only).

    Uses SIGALRM, so it only works reliably in the main thread on Unix.
    """

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


def load_bundle_spec(bundle_dir: Path) -> tuple[List[str], List[float]]:
    spec_path = bundle_dir / "policy_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"policy_spec.json not found in bundle {bundle_dir}")
    data = json.loads(spec_path.read_text())
    robot = data.get("robot", {})
    actuator_names = robot.get("actuator_names")
    home = robot.get("home_ctrl_rad")
    if not isinstance(actuator_names, list) or not actuator_names:
        raise ValueError("bundle policy_spec.json missing robot.actuator_names")
    if not isinstance(home, list):
        raise ValueError("bundle policy_spec.json missing robot.home_ctrl_rad")
    if len(home) != len(actuator_names):
        raise ValueError("bundle policy_spec.json actuator_names/home_ctrl_rad length mismatch")
    return [str(name) for name in actuator_names], [float(x) for x in home]


def resolve_robot_config_path(
    raw_config: dict,
    config_path: Path,
    *,
    bundle_dir: Optional[Path] = None,
) -> Path:
    config_dir = Path(config_path).parent.resolve()
    candidates: List[Path] = []

    json_robot_cfg = raw_config.get("robot_config_path")
    if isinstance(json_robot_cfg, str) and json_robot_cfg.strip():
        p = Path(json_robot_cfg).expanduser()
        candidates.append(p if p.is_absolute() else (config_dir / p).resolve())

    candidates.extend(
        [
            _REPO_ROOT / "assets" / "v2" / "mujoco_robot_config.json",
            Path("assets/v2/mujoco_robot_config.json"),
        ]
    )
    if bundle_dir is not None:
        candidates.append(Path(bundle_dir) / "mujoco_robot_config.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_robot_xml_path(
    *,
    robot_config_path: Path,
    config: WrRuntimeConfig,
    bundle_dir: Optional[Path] = None,
) -> Optional[Path]:
    candidates: List[Path] = []

    if robot_config_path.exists():
        try:
            robot_cfg = json.loads(robot_config_path.read_text())
            generated_from = robot_cfg.get("generated_from")
            if isinstance(generated_from, str) and generated_from.strip():
                candidates.append(Path(generated_from).expanduser())
        except Exception:
            pass

    try:
        candidates.append(config.mjcf_resolved_path)
    except Exception:
        pass

    if bundle_dir is not None:
        candidates.append(Path(bundle_dir) / "wildrobot.xml")
    candidates.append(_REPO_ROOT / "assets" / "v2" / "wildrobot.xml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_joint_axis_metadata(
    *,
    robot_config_path: Path,
    robot_xml_path: Optional[Path],
) -> Dict[str, JointAxisMetadata]:
    init_world_axes: Dict[str, Tuple[float, float, float]] = {}
    if robot_config_path.exists():
        data = json.loads(robot_config_path.read_text())
        for spec in data.get("actuated_joint_specs", []):
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            axis = _parse_axis(spec.get("init_world_axis"))
            if isinstance(name, str) and axis is not None:
                init_world_axes[name] = axis

    local_axes: Dict[str, Tuple[float, float, float]] = {}
    if robot_xml_path is not None and robot_xml_path.exists():
        root = ET.parse(robot_xml_path).getroot()
        for elem in root.findall(".//joint"):
            name = elem.attrib.get("name")
            if not name:
                continue
            axis_raw = elem.attrib.get("axis", "0 0 1")
            try:
                axis_values = [float(x) for x in axis_raw.strip().split()]
            except ValueError:
                continue
            axis = _parse_axis(axis_values)
            if axis is not None:
                local_axes[str(name)] = axis

    names = set(init_world_axes) | set(local_axes)
    return {
        name: JointAxisMetadata(
            local_axis=local_axes.get(name),
            init_world_axis=init_world_axes.get(name),
        )
        for name in names
    }


def load_policy_action_setup(
    *,
    args: argparse.Namespace,
    config: WrRuntimeConfig,
    joint_names: List[str],
) -> PolicyActionSetup:
    base_rad_by_joint = {joint: 0.0 for joint in joint_names}
    residual_scale_by_joint = {
        joint: float(config.control.action_scale_rad) for joint in joint_names
    }
    residual_base = "zero"
    residual_mode = "absolute"
    action_delay_steps = 0
    action_filter_alpha = 0.0
    source = "runtime config action_scale_rad fallback"

    bundle_arg = getattr(args, "bundle", None)
    bundle_dir = Path(bundle_arg) if bundle_arg else None
    if bundle_dir is None:
        try:
            candidate = config.policy_resolved_path.parent
            if (candidate / "policy_spec.json").exists():
                bundle_dir = candidate
        except Exception:
            bundle_dir = None

    if bundle_dir is not None:
        bundle_names, bundle_home = load_bundle_spec(bundle_dir)
        home_by_joint = dict(zip(bundle_names, bundle_home, strict=True))
        base_rad_by_joint = {
            joint: float(home_by_joint.get(joint, 0.0)) for joint in joint_names
        }
        residual_base = "home"
        source = "policy_spec home_ctrl_rad + runtime config action_scale_rad fallback"

        runtime_cfg_path = bundle_dir / "runtime_policy_config.json"
        if runtime_cfg_path.exists():
            runtime_cfg = json.loads(runtime_cfg_path.read_text())
            residual_base = str(runtime_cfg.get("loc_ref_residual_base", residual_base))
            residual_mode = str(runtime_cfg.get("loc_ref_residual_mode", residual_mode))
            action_delay_steps = int(runtime_cfg.get("action_delay_steps", action_delay_steps))
            action_filter_alpha = float(runtime_cfg.get("action_filter_alpha", action_filter_alpha))

            scalar = float(runtime_cfg.get("loc_ref_residual_scale", config.control.action_scale_rad))
            per_joint = runtime_cfg.get("loc_ref_residual_scale_per_joint", {}) or {}
            residual_scale_by_joint = {
                joint: float(per_joint.get(joint, scalar)) for joint in joint_names
            }

            per_actuator = runtime_cfg.get("residual_scale_per_actuator", []) or []
            if isinstance(per_actuator, list) and len(per_actuator) == len(bundle_names):
                scale_by_bundle_joint = {
                    name: float(scale)
                    for name, scale in zip(bundle_names, per_actuator, strict=True)
                }
                residual_scale_by_joint = {
                    joint: float(scale_by_bundle_joint.get(joint, scalar))
                    for joint in joint_names
                }
            source = str(runtime_cfg_path)

    return PolicyActionSetup(
        base_rad_by_joint=base_rad_by_joint,
        residual_scale_by_joint=residual_scale_by_joint,
        residual_base=residual_base,
        residual_mode=residual_mode,
        action_delay_steps=action_delay_steps,
        action_filter_alpha=action_filter_alpha,
        source=source,
    )


def compose_policy_action_target_rad(
    *,
    base_rad: float,
    action: float,
    residual_scale_rad: float,
    rad_range: Tuple[float, float],
) -> PolicyActionEvaluation:
    action_raw = float(action)
    action_applied = float(np.clip(action_raw, -1.0, 1.0))
    residual_rad = action_applied * float(residual_scale_rad)
    target_unclipped = float(base_rad) + residual_rad
    min_rad = float(rad_range[0])
    max_rad = float(rad_range[1])
    target_rad = float(np.clip(target_unclipped, min_rad, max_rad))
    return PolicyActionEvaluation(
        action_raw=action_raw,
        action_applied=action_applied,
        residual_rad=residual_rad,
        target_rad_unclipped=target_unclipped,
        target_rad=target_rad,
        clipped_by_joint_range=abs(target_rad - target_unclipped) > 1e-9,
    )


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    if args.bundle:
        bundle_cfg = Path(args.bundle) / "wildrobot_config.json"
        if bundle_cfg.exists():
            return bundle_cfg
    return _REPO_ROOT / "runtime" / "configs" / "runtime_config_v2.json"


def resolve_joint_names(
    *,
    args: argparse.Namespace,
    servo_cfgs: Dict[str, ServoConfig],
) -> List[str]:
    if not args.bundle:
        return list(servo_cfgs.keys())

    actuator_names, _ = load_bundle_spec(Path(args.bundle))
    missing = [name for name in actuator_names if name not in servo_cfgs]
    if missing:
        raise ValueError(
            "Runtime config is missing servo entries required by bundle actuator order: "
            f"{missing}"
        )
    return list(actuator_names)


def load_home_from_keyframes_xml(path: Path, joint_count: int) -> List[float]:
    tree = ET.parse(path)
    root = tree.getroot()
    key_elem = None
    for elem in root.findall(".//key"):
        name = elem.attrib.get("name")
        if name == "home":
            key_elem = elem
            break
    if key_elem is None:
        key_elem = root.find(".//key")
    if key_elem is None:
        raise ValueError("No <key> entries found in keyframes.xml")
    qpos_str = key_elem.attrib.get("qpos")
    if not qpos_str:
        raise ValueError("Keyframe missing qpos attribute")
    values = [float(x) for x in qpos_str.strip().split()]
    start = 7
    end = start + joint_count
    if len(values) < end:
        raise ValueError(
            f"Keyframe qpos shorter than expected (need at least {end} values for root+{joint_count} joints, got {len(values)})"
        )
    return values[start:end]


def load_home_from_scene(scene_xml: Path, joint_names: List[str]) -> List[float]:
    try:
        import mujoco
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MuJoCo is required for --scene-xml") from exc

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    key_id = 0
    if model.nkey > 0:
        try:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            key_id = 0
        if key_id < 0:
            key_id = 0
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    home_ctrl: List[float] = []
    for name in joint_names:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise ValueError(f"Actuator '{name}' not found in scene XML for home pose")
        trn_type = model.actuator_trntype[act_id]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"Actuator '{name}' does not target a joint (trntype={int(trn_type)})")
        joint_id = int(model.actuator_trnid[act_id][0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        home_ctrl.append(float(data.qpos[qpos_adr]))

    return home_ctrl


def resolve_home_ctrl(args: argparse.Namespace, joint_names: List[str]) -> List[float]:
    if args.bundle:
        bundle_names, bundle_home = load_bundle_spec(Path(args.bundle))
        home_by_name = dict(zip(bundle_names, bundle_home, strict=True))
        missing = [name for name in joint_names if name not in home_by_name]
        if missing:
            raise ValueError(
                "bundle policy_spec.json missing home_ctrl_rad entries for requested joints: "
                f"{missing}"
            )
        return [float(home_by_name[name]) for name in joint_names]
    if args.scene_xml:
        return load_home_from_scene(Path(args.scene_xml), joint_names)
    if args.keyframes_xml:
        return load_home_from_keyframes_xml(Path(args.keyframes_xml), len(joint_names))
    return [0.0] * len(joint_names)


def panic_and_exit(controller, servo_ids: Iterable[int]) -> None:
    try:
        controller.unload_servos(list(servo_ids))
    finally:
        controller.close()
    sys.exit("Panic unload requested; exiting.")

def wait_until_unload(controller, servo_ids: Iterable[int], *, prompt: str) -> None:
    """Keep servos loaded until the user requests unload."""
    print(prompt)
    while True:
        try:
            resp = input("> ").strip().lower()
        except KeyboardInterrupt:
            print("\nCtrl+C received.")
            panic_and_exit(controller, servo_ids)
        if resp == PANIC_KEY:
            panic_and_exit(controller, servo_ids)
        if not resp:
            continue
        print(f"Unknown input '{resp}'. Press '{PANIC_KEY}' then Enter to unload.")


def motor_sign_prompt(
    joint: str,
    hint: str,
    delta_rad: float,
    axis: Optional[JointAxisMetadata],
) -> str:
    local_axis = _axis_label(axis.local_axis) if axis is not None else "n/a"
    world_axis = _axis_label(axis.init_world_axis) if axis is not None else "n/a"
    return (
        f"I commanded raw servo units upward by about +{delta_rad:.3f} rad worth of units on {joint}.\n"
        "MuJoCo positive direction for this joint:\n"
        f"  local_axis: {local_axis}\n"
        f"  init_world_axis: {world_axis}\n"
        f"  right-hand rule: point your right thumb along init_world_axis ({world_axis}); curled fingers are positive MuJoCo rotation.\n"
        f"  expected positive motion: {hint}\n"
        "Did the raw +servo-unit move match that positive MuJoCo direction?\n"
        "  y = yes (servo_unit_direction / motor_unit_direction = +1)\n"
        "  n = no  (servo_unit_direction / motor_unit_direction = -1)\n"
        "  r = repeat with a different delta\n"
        f"  {PANIC_KEY} = panic unload"
    )


def calibrate_motor_sign(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    hint: str,
    axis: Optional[JointAxisMetadata],
    *,
    all_servo_ids: Iterable[int],
    move_ms: int,
    pause_s: float,
) -> int:
    print(f"\n-- Servo unit direction calibration for {joint} (servo {servo.id}) --")
    delta_rad_used = DEFAULT_DELTA_RAD

    while True:
        # Always start the motor_sign test from center to make the "first move" unambiguous.
        center_units = ServoConfig.UNITS_CENTER
        announce_and_pause(
            f"Step: move {joint} to center units ({center_units})",
            pause_s,
        )
        _move_servo_units_20deg_per_s(
            controller,
            servo,
            center_units,
            fallback_ms=max(int(move_ms), 1000),
            min_ms=1000,
        )
        # Direction calibration should be done in raw servo unit space:
        # start from mechanical center (500) and move +units. This avoids coupling
        # direction detection to any current joint_angle_at_zero_unit_deg or offset.
        delta_units = int(round(float(delta_rad_used) * float(servo.UNITS_PER_RAD)))
        if delta_units == 0:
            delta_units = 1
        plus_units = max(ServoConfig.UNITS_MIN, min(ServoConfig.UNITS_MAX, center_units + delta_units))
        announce_and_pause(
            f"Step: command +delta ({delta_rad_used:.3f} rad) -> units {plus_units} (ignoring existing servo_unit_direction)",
            pause_s,
        )
        _move_servo_units_20deg_per_s(
            controller,
            servo,
            plus_units,
            fallback_ms=max(int(move_ms), 1000),
            min_ms=1000,
        )
        resp = input(motor_sign_prompt(joint, hint, delta_rad_used, axis) + "\n> ").strip().lower()
        if resp == PANIC_KEY:
            panic_and_exit(controller, all_servo_ids)
        if resp.startswith("y"):
            print("servo_unit_direction / motor_unit_direction set to +1")
            announce_and_pause(
                f"Step: return {joint} to center units ({center_units})",
                pause_s,
            )
            _move_servo_units_20deg_per_s(
                controller,
                servo,
                center_units,
                fallback_ms=max(int(move_ms), 1000),
                min_ms=1000,
            )
            return 1
        if resp.startswith("n"):
            print("servo_unit_direction / motor_unit_direction set to -1")
            announce_and_pause(
                f"Step: return {joint} to center units ({center_units})",
                pause_s,
            )
            _move_servo_units_20deg_per_s(
                controller,
                servo,
                center_units,
                fallback_ms=max(int(move_ms), 1000),
                min_ms=1000,
            )
            return -1
        if resp.startswith("r"):
            try:
                new_delta = input("Enter new delta in radians (blank to keep): ").strip()
                if new_delta:
                    delta_rad_used = float(new_delta)
            except ValueError:
                print("Invalid delta, keeping previous value.")
            continue
        print("Input not recognized; please respond with y, n, r, or panic key.")


def calibrate_offset(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    all_servo_ids: Iterable[int],
    move_ms: int,
    step_units: int,
    pause_s: float,
    *,
    record_pos: bool,
    all_joint_names: List[str],
    all_servo_cfgs: Dict[str, ServoConfig],
    all_states: Dict[str, JointState],
) -> Tuple[Optional[int], bool, bool]:
    """Calibrate offset for a joint.

    Returns:
      (new_offset, save_requested, quit_joint_without_save)
    """
    print(f"\n-- Offset calibration for {joint} (servo {servo.id}) --")
    # Offset calibration should not depend on any existing offset value.
    target_units = ServoConfig.UNITS_CENTER
    announce_and_pause(
        f"Step: move {joint} to raw center units ({target_units})",
        pause_s,
    )
    _move_servo_units_20deg_per_s(
        controller,
        servo,
        target_units,
        fallback_ms=max(int(move_ms), 1000),
        min_ms=1000,
    )
    commands_msg = (
        "Jog the joint until it matches your neutral pose. Commands:"
        "\n  a/d = -/+ step; A/D = -/+ 5x step; c or empty = confirm;"
        "\n  s = save config now; q = quit joint (without save);"
        "\n  m = enter offset manually (units);"
    )
    if record_pos:
        commands_msg += "\n  p = print all joint positions;"
    commands_msg += f"\n  {PANIC_KEY} = panic unload"
    print(commands_msg)
    read_failures = 0

    def _capture_offset_from_current_pos() -> Optional[int]:
        nonlocal read_failures
        pos = read_position(controller, servo.id)
        if pos is None:
            read_failures += 1
            print("Failed to read position.")
            if read_failures >= 3:
                try:
                    vbat = controller.get_battery_voltage()
                    print(f"Debug: battery voltage readback={vbat}")
                except Exception:
                    pass
                print(
                    "If this persists, check servo ID/wiring or use 'm' to enter offset manually."
                )
            return None
        offset_units = int(pos) - int(ServoConfig.UNITS_CENTER)
        print(
            f"Captured offset {offset_units} (units), raw position {pos} "
            f"(offset = current_pos - {int(ServoConfig.UNITS_CENTER)})"
        )
        return offset_units

    while True:
        cmd = input("(a/d/A/D/c/s/m/p/q/enter) > ").strip()
        if not cmd or cmd.lower() == "c":
            offset_units = _capture_offset_from_current_pos()
            if offset_units is None:
                continue
            return offset_units, False, False
        if cmd.lower() == "m":
            raw = input(
                f"Enter offset in servo units (offset = current_pos - {int(ServoConfig.UNITS_CENTER)}): "
            ).strip()
            try:
                offset_units = int(raw)
            except ValueError:
                print("Invalid integer offset.")
                continue
            print(f"Using manual offset {offset_units} (units).")
            return offset_units, False, False
        if cmd.lower() == "q":
            print(f"Aborted offset calibration for {joint}; quitting joint without saving.")
            return None, False, True
        if cmd.lower() == "s":
            offset_units = _capture_offset_from_current_pos()
            if offset_units is None:
                continue
            print(f"Save requested from offset calibration for {joint}.")
            return offset_units, True, False
        if record_pos and cmd.lower() == "p":
            print_all_joint_positions(
                controller,
                joint_names=all_joint_names,
                servo_cfgs=all_servo_cfgs,
                states=all_states,
                include_rad=True,
            )
            continue
        if cmd == PANIC_KEY:
            panic_and_exit(controller, all_servo_ids)
        delta = 0
        if cmd in ("a", "A"):
            delta = -step_units * (5 if cmd == "A" else 1)
        elif cmd in ("d", "D"):
            delta = step_units * (5 if cmd == "D" else 1)
        else:
            print("Unknown command; use a/d/A/D to jog, c to confirm, s to save, q to quit.")
            continue
        target_units = max(ServoConfig.UNITS_MIN, min(ServoConfig.UNITS_MAX, target_units + delta))
        _move_servo_units_20deg_per_s(
            controller,
            servo,
            target_units,
            fallback_ms=max(int(move_ms), 1000),
            min_ms=1000,
        )
        pos = read_position(controller, servo.id)
        if pos is not None:
            pos_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                pos,
                motor_sign=state.motor_sign,
                offset=0,
            )
            print(f"Moved to {pos} units (~{pos_rad:.3f} rad, offset assumed 0).")


def verify_zero(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    move_ms: int,
    pause_s: float,
) -> None:
    center_target_rad = float(servo.center_rad)
    target_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        center_target_rad,
        motor_sign=state.motor_sign,
        offset=state.offset,
    )
    announce_and_pause(
        f"Step: verify {joint} at center_rad={center_target_rad:+.4f} -> units {target_units}",
        pause_s,
    )
    move_and_wait(controller, servo.id, target_units, move_ms)
    pos = read_position(controller, servo.id)
    if pos is None:
        print("Verification readback failed.")
        return
    pos_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
        pos,
        motor_sign=state.motor_sign,
        offset=state.offset,
    )
    center_err = float(pos_rad) - center_target_rad
    status = "OK" if abs(center_err) <= VERIFY_TOL_RAD else "drift"
    print(
        f"Verify center: {pos} units => {pos_rad:.4f} rad "
        f"(center_err={center_err:+.4f} rad, {status})"
    )


def _move_ms_for_speed_20deg_per_s(current_deg: float, target_deg: float) -> int:
    delta_deg = abs(float(target_deg) - float(current_deg))
    if delta_deg <= 1e-6:
        return 100
    return max(100, int(round((delta_deg / 20.0) * 1000.0)))


def _move_ms_from_units_for_speed_20deg_per_s(
    current_units: int,
    target_units: int,
    *,
    units_per_rad: float,
    min_ms: int = 1000,
) -> int:
    delta_units = abs(int(target_units) - int(current_units))
    if delta_units <= 0:
        return int(min_ms)
    delta_rad = float(delta_units) / float(units_per_rad)
    delta_deg = float(np.rad2deg(delta_rad))
    return max(int(min_ms), int(round((delta_deg / 20.0) * 1000.0)))


def _move_servo_units_20deg_per_s(
    controller,
    servo: ServoConfig,
    target_units: int,
    *,
    fallback_ms: int = 1000,
    min_ms: int = 1000,
) -> None:
    current_units = read_position(controller, int(servo.id))
    if current_units is not None:
        move_ms = _move_ms_from_units_for_speed_20deg_per_s(
            int(current_units),
            int(target_units),
            units_per_rad=float(servo.UNITS_PER_RAD),
            min_ms=int(min_ms),
        )
    else:
        move_ms = max(int(min_ms), int(fallback_ms))
        print("Current position read failed; using fallback move time.")
    move_and_wait(controller, int(servo.id), int(target_units), int(move_ms))


def _group_move_ms_20deg_per_s(
    controller,
    servo_cfg_by_id: Dict[int, ServoConfig],
    commands: List[Tuple[int, int]],
    *,
    fallback_ms: int = 1000,
    min_ms: int = 1000,
) -> int:
    max_move_ms = int(min_ms)
    read_failed = False
    for servo_id, target_units in commands:
        servo = servo_cfg_by_id.get(int(servo_id))
        if servo is None:
            max_move_ms = max(max_move_ms, int(max(min_ms, fallback_ms)))
            continue
        current_units = read_position(controller, int(servo_id))
        if current_units is None:
            read_failed = True
            joint_move_ms = int(max(min_ms, fallback_ms))
        else:
            joint_move_ms = _move_ms_from_units_for_speed_20deg_per_s(
                int(current_units),
                int(target_units),
                units_per_rad=float(servo.UNITS_PER_RAD),
                min_ms=int(min_ms),
            )
        max_move_ms = max(max_move_ms, int(joint_move_ms))
    if read_failed:
        print("Some current positions failed to read; using fallback move time for those joints.")
    return int(max_move_ms)


def print_joint_calibration_state(
    controller,
    *,
    joint: str,
    servo: ServoConfig,
    state: JointState,
    axis_metadata: Dict[str, JointAxisMetadata],
    policy_setup: PolicyActionSetup,
    hint: str,
) -> None:
    min_deg = float(np.rad2deg(float(servo.rad_range[0])))
    max_deg = float(np.rad2deg(float(servo.rad_range[1])))
    pos_units = read_position(controller, int(servo.id))
    axis = axis_metadata.get(joint)
    base_rad = float(policy_setup.base_rad_by_joint.get(joint, 0.0))
    scale_rad = float(policy_setup.residual_scale_by_joint.get(joint, 0.0))

    print(f"\n-- Joint state: {joint} (servo {int(servo.id)}) --")
    print(f"  ctrl_range_deg: [{min_deg:.3f}, {max_deg:.3f}]")
    print(f"  servo_offset_unit: {int(state.offset):+d}")
    print(f"  servo_unit_direction / motor_unit_direction: {int(state.motor_sign):+d}")
    print(f"  joint_angle_at_zero_unit_deg: {float(servo.joint_angle_at_zero_unit_deg):+.3f}")
    if axis is not None:
        print(f"  local_axis: {_axis_summary(axis.local_axis)}")
        print(f"  init_world_axis: {_axis_summary(axis.init_world_axis)}")
    else:
        print("  local_axis: n/a")
        print("  init_world_axis: n/a")
    print(f"  positive_mujoco_rad_hint: {hint}")
    print(
        "  policy_action_setup: "
        f"target=clip(base + clip(action,-1,1)*scale, joint_range), "
        f"base={base_rad:+.6f} rad ({float(np.rad2deg(base_rad)):+.3f} deg), "
        f"scale={scale_rad:.6f} rad"
    )
    print(
        "  policy_action_source: "
        f"{policy_setup.source} "
        f"(residual_base={policy_setup.residual_base}, mode={policy_setup.residual_mode}, "
        f"delay_steps={policy_setup.action_delay_steps}, filter_alpha={policy_setup.action_filter_alpha:.3f})"
    )

    if pos_units is None:
        print("  current_servo_elect_unit: <read failed>")
        return

    conceptual_units = int(pos_units) - int(state.offset)
    joint_target_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
        int(pos_units),
        motor_sign=int(state.motor_sign),
        offset=int(state.offset),
    )
    print(f"  current_servo_elect_unit: {int(pos_units)}")
    print(f"  current_servo_conceptual_unit: {int(conceptual_units)}")
    joint_target_deg = float(np.rad2deg(float(joint_target_rad)))
    print(f"  calculated_joint_target_rad: {float(joint_target_rad):+.6f} ({joint_target_deg:+.3f} deg)")


def evaluate_policy_action(
    controller,
    *,
    joint: str,
    servo: ServoConfig,
    state: JointState,
    policy_setup: PolicyActionSetup,
    move_ms_fallback: int,
) -> None:
    scale_rad = float(policy_setup.residual_scale_by_joint.get(joint, 0.0))
    home_rad = float(policy_setup.base_rad_by_joint.get(joint, 0.0))
    print(f"\n-- Policy action evaluator for {joint} (servo {servo.id}) --")
    print("  Current setup: target=clip(base + clip(action,-1,1)*residual_scale, joint_range)")
    print("  Enter the APPLIED action value. The runtime may delay/filter raw policy output before it becomes applied.")
    print(
        f"  Bundle/setup: home/base={home_rad:+.6f} rad ({float(np.rad2deg(home_rad)):+.3f} deg), "
        f"scale={scale_rad:.6f} rad, source={policy_setup.source}"
    )

    raw_action = input("Enter applied policy action [-1..1]: ").strip()
    try:
        action = float(raw_action)
    except ValueError:
        print("Invalid number.")
        return

    eval_result = compose_policy_action_target_rad(
        base_rad=home_rad,
        action=action,
        residual_scale_rad=scale_rad,
        rad_range=servo.rad_range,
    )
    home_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        home_rad,
        motor_sign=int(state.motor_sign),
        offset=int(state.offset),
    )
    target_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        eval_result.target_rad,
        motor_sign=int(state.motor_sign),
        offset=int(state.offset),
    )
    home_conceptual_units = int(home_units) - int(state.offset)
    conceptual_units = int(target_units) - int(state.offset)
    home_deg = float(np.rad2deg(home_rad))
    target_deg_unclipped = float(np.rad2deg(eval_result.target_rad_unclipped))
    target_deg = float(np.rad2deg(eval_result.target_rad))

    print(f"  action_raw: {eval_result.action_raw:+.6f}")
    print(f"  action_applied_clipped: {eval_result.action_applied:+.6f}")
    print(f"  residual_rad: {eval_result.residual_rad:+.6f} ({float(np.rad2deg(eval_result.residual_rad)):+.3f} deg)")
    print(f"  home_joint_rad: {home_rad:+.6f} ({home_deg:+.3f} deg)")
    print(f"  home_motor_elect_unit: {int(home_units)}")
    print(f"  home_motor_conceptual_unit: {int(home_conceptual_units)}")
    print(f"  target_joint_rad_unclipped: {eval_result.target_rad_unclipped:+.6f} ({target_deg_unclipped:+.3f} deg)")
    print(f"  target_joint_rad: {eval_result.target_rad:+.6f} ({target_deg:+.3f} deg)")
    print(f"  motor_elect_unit: {int(target_units)}")
    print(f"  motor_conceptual_unit: {int(conceptual_units)}")
    if eval_result.clipped_by_joint_range:
        min_deg = float(np.rad2deg(float(servo.rad_range[0])))
        max_deg = float(np.rad2deg(float(servo.rad_range[1])))
        print(f"  WARNING: target clipped to joint range [{min_deg:+.3f}, {max_deg:+.3f}] deg")

    if yes_no(
        f"Move to home first, wait {POLICY_ACTION_HOME_PAUSE_S:.1f}s, then move to this policy-action target?",
        default=True,
    ):
        current_units = read_position(controller, int(servo.id))
        if current_units is not None:
            current_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                int(current_units),
                motor_sign=int(state.motor_sign),
                offset=int(state.offset),
            )
            current_deg = float(np.rad2deg(float(current_rad)))
            home_move_ms = _move_ms_for_speed_20deg_per_s(current_deg, home_deg)
        else:
            home_move_ms = int(max(move_ms_fallback, 100))
            print("Current position read failed; using fallback move time.")
        print(f"Moving {joint} to policy home/base ({int(home_units)} units)...")
        move_and_wait(controller, int(servo.id), int(home_units), int(home_move_ms))
        home_pos = read_position(controller, int(servo.id))
        if home_pos is not None:
            print(f"Home readback elect unit={int(home_pos)} conceptual={int(home_pos) - int(state.offset)}")
        print(f"Waiting {POLICY_ACTION_HOME_PAUSE_S:.1f}s before action target...")
        time.sleep(POLICY_ACTION_HOME_PAUSE_S)

        action_move_ms = _move_ms_for_speed_20deg_per_s(home_deg, target_deg)
        print(f"Moving {joint} to policy-action target ({int(target_units)} units)...")
        move_and_wait(controller, int(servo.id), int(target_units), int(action_move_ms))
        pos = read_position(controller, int(servo.id))
        if pos is not None:
            print(f"Moved. Readback elect unit={int(pos)} conceptual={int(pos) - int(state.offset)}")


def range_test_joint(
    controller,
    servo: ServoConfig,
    joint: str,
    *,
    state: Optional[JointState] = None,
    speed_deg_per_s: Optional[float] = None,
) -> None:
    """Run a joint through its full range: center -> min -> max -> min -> center."""
    print(f"\n-- Range test for {joint} (servo {servo.id}) --")
    effective_state = state or normalize_joint_state(
        offset=servo.offset,
        motor_sign=servo.motor_sign,
    )

    # Calculate positions using current calibration and joint limits.
    # Note: servo.joint_target_rad_to_elect_unit() clamps to [0, 1000]. If your MuJoCo joint range requires
    # more than the servo can physically cover (240deg total), targets will clip.
    min_rad, max_rad = servo.rad_range

    def _target_rad_to_units(target_rad: float) -> int:
        return servo.joint_target_rad_to_elect_unit_for_calibrate(
            float(target_rad),
            motor_sign=int(effective_state.motor_sign),
            offset=int(effective_state.offset),
        )

    def _units_to_target_rad(units: int) -> float:
        return servo.servo_elect_units_to_joint_target_rad_for_calibrate(
            int(units),
            motor_sign=int(effective_state.motor_sign),
            offset=int(effective_state.offset),
        )

    center_units = _target_rad_to_units(0.0)

    def _raw_units(target_rad: float) -> float:
        delta = float(target_rad) - float(servo.center_rad)
        return (
            float(servo.UNITS_CENTER)
            + float(effective_state.offset)
            + float(effective_state.motor_sign) * (delta * float(servo.UNITS_PER_RAD))
        )

    min_units_raw = _raw_units(float(min_rad))
    max_units_raw = _raw_units(float(max_rad))
    min_units = _target_rad_to_units(float(min_rad))
    max_units = _target_rad_to_units(float(max_rad))

    min_deg = float(np.rad2deg(float(min_rad)))
    max_deg = float(np.rad2deg(float(max_rad)))

    # Show effective mapping and reachable ctrl range given servo unit limits.
    reachable_min_rad = float(_units_to_target_rad(int(servo.UNITS_MIN)))
    reachable_max_rad = float(_units_to_target_rad(int(servo.UNITS_MAX)))
    reachable_min_deg = float(np.rad2deg(reachable_min_rad))
    reachable_max_deg = float(np.rad2deg(reachable_max_rad))

    print(f"  Center: {center_units} units (0.0 rad)")
    print(f"  Deg range: {min_deg:.1f} to {max_deg:.1f}")
    print(
        "  Calibration: "
        f"joint_angle_at_zero_unit_deg={float(servo.joint_angle_at_zero_unit_deg):+.1f} "
        f"servo_unit_direction/motor_unit_direction={int(effective_state.motor_sign):+d} "
        f"servo_offset_unit={int(effective_state.offset)}"
    )
    print(
        "  Reachable (by servo limits): "
        f"units {int(servo.UNITS_MIN)}..{int(servo.UNITS_MAX)} => ctrl {reachable_min_deg:+.1f}..{reachable_max_deg:+.1f} deg"
    )
    print(f"  Min: {min_units} units ({min_rad:.3f} rad)")
    print(f"  Max: {max_units} units ({max_rad:.3f} rad)")
    if speed_deg_per_s is None:
        print(f"  Move duration: {RANGE_TEST_MS}ms per segment")
    else:
        print(f"  Move speed: {float(speed_deg_per_s):.3f} deg/s")

    clipped = []
    if min_units_raw < servo.UNITS_MIN - 1e-6 or min_units_raw > servo.UNITS_MAX + 1e-6:
        clipped.append(f"min_raw={min_units_raw:.1f}")
    if max_units_raw < servo.UNITS_MIN - 1e-6 or max_units_raw > servo.UNITS_MAX + 1e-6:
        clipped.append(f"max_raw={max_units_raw:.1f}")
    if clipped:
        print(
            "WARNING: requested joint range clips to servo limits (0..1000 units): "
            + ", ".join(clipped)
            + ". This usually means joint_angle_at_zero_unit_deg/motor_unit_direction/servo_offset_unit needs adjustment, or the MuJoCo range exceeds servo capability."
        )


    try:
        start_deg: Optional[float] = None
        current_pos = read_position(controller, int(servo.id))
        if current_pos is not None:
            start_deg = float(np.rad2deg(float(_units_to_target_rad(int(current_pos)))))

        def _move_and_report(label: str, target_units: int) -> None:
            nonlocal start_deg
            target_deg_cmd = float(np.rad2deg(float(_units_to_target_rad(int(target_units)))))
            if speed_deg_per_s is None or float(speed_deg_per_s) <= 0.0 or start_deg is None:
                move_ms = int(RANGE_TEST_MS)
            else:
                move_ms = _move_ms_for_speed_20deg_per_s(start_deg, target_deg_cmd)
            print(f"Moving to {label} ({target_units} units)...")
            controller.move_servos([(servo.id, int(target_units))], int(move_ms))
            time.sleep(int(move_ms) / 1000.0 + 0.1)
            pos = read_position(controller, servo.id)
            if pos is None:
                print("  Readback: failed")
                return
            pos_rad = _units_to_target_rad(int(pos))
            pos_deg = float(np.rad2deg(float(pos_rad)))
            print(f"  Readback: {int(pos)} units => {pos_rad:+.3f} rad ({pos_deg:+.1f} deg)")
            start_deg = pos_deg

        # Move to min
        _move_and_report("min", min_units)

        # Move to max
        _move_and_report("max", max_units)

        # Move back to min
        _move_and_report("min", min_units)

        # Move back to center
        _move_and_report("center", center_units)

        print(f"Range test for {joint} complete.")

    except KeyboardInterrupt:
        print("\nRange test interrupted. Moving to center...")
        controller.move_servos([(servo.id, center_units)], RANGE_TEST_MS)
        time.sleep(RANGE_TEST_MS / 1000.0 + 0.1)


def write_config(
    base_data: dict,
    output_path: Path,
    updates: Dict[str, JointState],
    *,
    home_ctrl_rad: Optional[List[float]] = None,
) -> None:
    # Canonical block
    servo_block = base_data.setdefault("servo_controller", {})
    servo_block.setdefault("type", "hiwonder")
    servos = servo_block.setdefault("servos", {})
    for joint, state in updates.items():
        if joint not in servos:
            servos[joint] = {}
        servos[joint]["servo_offset_unit"] = int(state.offset)
        servos[joint]["motor_unit_direction"] = int(state.motor_sign)

    if home_ctrl_rad is not None:
        servo_block["home_ctrl_rad"] = [float(x) for x in home_ctrl_rad]

    _write_json_with_retries(output_path, base_data)

def write_bno_config(
    base_data: dict,
    output_path: Path,
    *,
    i2c_address: Optional[int] = None,
    upside_down: Optional[bool] = None,
    axis_map: Optional[List[str]] = None,
    axis_map_measurements: Optional[dict] = None,
) -> None:
    bno = base_data.setdefault("bno085", {})
    if i2c_address is not None:
        bno["i2c_address"] = hex(int(i2c_address))
    if upside_down is not None:
        bno["upside_down"] = bool(upside_down)
    if axis_map is not None:
        normalized_axis_map = [str(x).strip().upper() for x in axis_map]
        if WrRuntimeConfig._axis_map_det(normalized_axis_map) < 0:
            raise ValueError(
                "bno085.axis_map must be right-handed with determinant +1 "
                f"(got det=-1 for {axis_map}); flip two axes, not one"
            )
        bno["axis_map"] = normalized_axis_map
    if axis_map_measurements is not None:
        bno["axis_map_measurements"] = axis_map_measurements
    _write_json_with_retries(output_path, base_data)


def _write_json_with_retries(path: Path, data: dict, *, attempts: int = 3, delay_s: float = 0.2) -> None:
    """Write JSON config with small retries (helps on flaky FS/SSH setups)."""
    last_exc: Exception | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            path.write_text(payload)
            # Best-effort durability.
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception:
                pass
            return
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(float(delay_s) * attempt)
    raise RuntimeError(f"Failed to write config to {path}") from last_exc


def _expected_sensor_axis_for_body(body_axis: str) -> tuple[str, np.ndarray]:
    """Return (axis_letter, unit_vector) under the 'no permutation' assumption."""
    if body_axis == "body_x":
        return "X", np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if body_axis == "body_y":
        return "Y", np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if body_axis == "body_z":
        return "Z", np.array([0.0, 0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unknown body axis: {body_axis}")


def _apply_upside_down_quat(quat_xyzw: List[float]) -> List[float]:
    # Must match runtime/wr_runtime/hardware/bno085.py behavior.
    x, y, z, w = [float(v) for v in quat_xyzw]
    return [x, -y, -z, w]


def _init_calibration_bno085(BNO085IMU, *, config: WrRuntimeConfig, upside_down: bool):
    """Create a ToddlerBot-style BNO08x reader for calibration.

    ToddlerBot reads BNO08x from a background thread at 200 Hz and enables gyro plus
    one quaternion report. Calibration needs the same fresh-stream behavior more than
    the runtime fallback report path.
    """
    kwargs = {
        "i2c_address": config.bno085.i2c_address,
        "upside_down": upside_down,
        "sampling_hz": CALIBRATION_IMU_SAMPLING_HZ,
        "polling_mode": False,
        "suppress_debug": config.bno085.suppress_debug,
        "i2c_frequency_hz": config.bno085.i2c_frequency_hz,
        "init_retries": config.bno085.init_retries,
        "enable_rotation_vector": False,
    }
    try:
        return BNO085IMU(**kwargs)
    except TypeError as exc:
        # Back-compat for target images with an older local BNO085IMU wrapper.
        if "unexpected" not in str(exc):
            raise
        kwargs.pop("enable_rotation_vector", None)
        return BNO085IMU(**kwargs)


def calibrate_imu_upside_down(
    *,
    config: WrRuntimeConfig,
    raw_config: dict,
    output_path: Path,
    samples: int,
) -> None:
    """Calibrate BNO08X `upside_down` using a simple gravity sanity check.

    This does not solve arbitrary mounting rotations; it only helps detect an inverted mount.
    """
    try:
        from policy_contract.numpy.frames import gravity_local_from_quat, normalize_quat_xyzw
        from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    except Exception as exc:
        raise RuntimeError(
            "IMU calibration requires runtime IMU deps on the target (Adafruit Blinka + BNO08X). "
            "Install them on the Pi and re-run."
        ) from exc

    print("\n-- IMU Upside-Down Calibration --", flush=True)
    print("Place the robot IMU/base upright and still (standing pose).", flush=True)
    print("We compare expected gravity_local ≈ [0, 0, -1].", flush=True)
    est_s = max(1, int(samples)) * 0.01
    print(f"Collecting ~{max(1, int(samples))} samples (~{est_s:.1f}s)...", flush=True)

    try:
        with _alarm_timeout(
            10,
            message=(
                "Timed out initializing BNO085 IMU. This usually means I2C is not responding or the process is blocked on the I2C bus lock. "
                "Check wiring/address, confirm /dev/i2c-1 exists, and run 'sudo i2cdetect -y 1' on the robot."
            ),
        ):
            imu = _init_calibration_bno085(
                BNO085IMU,
                config=config,
                upside_down=False,
            )
    except TimeoutError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        time.sleep(0.2)
        _log_line(
            "IMU init: "
            f"polling_mode={getattr(imu, 'polling_mode', None)} "
            f"sampling_hz={getattr(imu, 'sampling_hz', None)} "
            f"enable_rotation_vector={getattr(imu, 'enable_rotation_vector', None)} "
            f"suppress_debug={getattr(imu, 'suppress_debug', None)}"
        )
        # Ensure the background reader is producing fresh samples (not stuck on I2C).
        last_ts = float(getattr(imu.read(), "timestamp_s", 0.0))
        start_wait = time.monotonic()
        while True:
            s0 = imu.read()
            ts0 = float(getattr(s0, "timestamp_s", 0.0))
            if ts0 > 0.0 and ts0 != last_ts and bool(getattr(s0, "valid", True)):
                last_ts = ts0
                break
            if ts0 > 0.0 and ts0 != last_ts:
                last_ts = ts0
            if time.monotonic() - start_wait > 3.0:
                raise RuntimeError(
                    "IMU is not producing fresh valid samples. "
                    "Check I2C wiring/address, and verify the BNO08X is visible with 'sudo i2cdetect -y 1'."
                )
            time.sleep(0.05)

        g_normal = []
        g_flipped = []
        total = max(1, int(samples))
        stale_since = time.monotonic()
        collected = 0
        reads = 0
        while collected < total:
            s = imu.read()
            reads += 1
            ts = float(getattr(s, "timestamp_s", 0.0))
            valid = bool(getattr(s, "valid", True))
            fresh = ts != last_ts and ts > 0.0
            if fresh:
                last_ts = ts
            if fresh and valid:
                stale_since = time.monotonic()
            elif time.monotonic() - stale_since > 2.0:
                raise RuntimeError(
                    "IMU sample stream stalled during calibration (no fresh valid samples). "
                    "This often indicates an intermittent I2C bus lock or power issue."
                )
            if not (fresh and valid):
                time.sleep(0.01)
                continue

            q = normalize_quat_xyzw(np.asarray(s.quat_xyzw, dtype=np.float32))
            g0 = gravity_local_from_quat(q)
            g1 = gravity_local_from_quat(
                normalize_quat_xyzw(np.asarray(_apply_upside_down_quat(q.tolist()), dtype=np.float32))
            )
            g_normal.append(g0)
            g_flipped.append(g1)
            collected += 1
            if collected % 25 == 0:
                print(f"  ... {collected}/{total} fresh samples (reads={reads})", flush=True)
            time.sleep(0.01)

        print("Computing gravity means...", flush=True)
        g0_mean = np.mean(np.stack(g_normal, axis=0), axis=0)
        g1_mean = np.mean(np.stack(g_flipped, axis=0), axis=0)

        print("", flush=True)
        print(f"Current (upside_down=false) gravity_local mean: {g0_mean}", flush=True)
        print(f"Flipped  (upside_down=true)  gravity_local mean: {g1_mean}\n", flush=True)

        score0 = abs(float(g0_mean[2]) + 1.0)
        score1 = abs(float(g1_mean[2]) + 1.0)
        recommended = bool(score1 < score0)

        current = bool(getattr(config.bno085, "upside_down", False))
        print(
            f"Recommended upside_down: {recommended} (z-error false={score0:.3f}, true={score1:.3f})",
            flush=True,
        )
        print(f"Current     upside_down: {current}", flush=True)

        if recommended == current:
            print("Result: config already matches recommendation; not writing.", flush=True)
            print("Upside-down calibration complete.", flush=True)
            return

        resp = _prompt_choice_strict(
            f"Recommendation differs. Write bno085.upside_down={recommended} to config?"
        )
        if resp == "q":
            print("Quit requested; not writing config.")
            return
        if resp != "y":
            print("Not writing config.")
            print("Upside-down calibration complete.", flush=True)
            return

        write_bno_config(raw_config, output_path, upside_down=recommended)
        print(f"Wrote IMU config to {output_path}")
        print(f"Result: saved bno085.upside_down={recommended}")
        print("Upside-down calibration complete.", flush=True)
    finally:
        try:
            imu.close()
        except Exception:
            pass


def _prompt_choice(prompt: str, *, default_yes: bool = True) -> str:
    suffix = " [Y/n/q]: " if default_yes else " [y/N/q]: "
    while True:
        resp = input(prompt + suffix).strip().lower()

        # Treat empty as default.
        if resp == "" or resp in {"enter", "return"}:
            return "y" if default_yes else "n"

        if resp in {"q", "quit"}:
            return "q"
        if resp in {"y", "yes"}:
            return "y"
        if resp in {"n", "no"}:
            return "n"

        # Fall back to first-letter behavior for common typos (e.g., 'yep', 'nah').
        if resp.startswith("q"):
            return "q"
        if resp.startswith("y"):
            return "y"
        if resp.startswith("n"):
            return "n"

        print("Please answer with 'y', 'n', or 'q'.")


def _prompt_choice_strict(prompt: str, *, context: Optional[str] = None) -> str:
    """Prompt for y/n/q requiring an explicit answer (Enter is not accepted)."""
    while True:
        if context:
            print(context, flush=True)
        print(prompt, flush=True)
        print("Type y/n/q then press Enter:", flush=True)
        resp = input().strip().lower()
        if resp == "":
            print("Please answer with 'y', 'n', or 'q'.", flush=True)
            continue
        if resp in {"q", "quit"} or resp.startswith("q"):
            return "q"
        if resp in {"y", "yes"} or resp.startswith("y"):
            return "y"
        if resp in {"n", "no"} or resp.startswith("n"):
            return "n"
        print("Please answer with 'y', 'n', or 'q'.", flush=True)


def _prompt_accept_measurement(*, prompt: str, context: str) -> str:
    """Prompt for accepting a measurement with extra actions.

    Returns one of: y (accept), n (reject), q (quit this step), r (redo capture), d (reprint details).
    """
    # Use select-based token waiting to avoid the "hidden prompt -> blocked input()" failure mode
    # seen on some PTY/SSH stacks. The prompt will be reprinted periodically.
    token = _wait_for_token(
        prompt_lines=[
            context.rstrip(),
            prompt,
            "Type y/n/q (accept/reject/quit), r (redo), or d (details), then press Enter.",
        ],
        valid=["y", "n", "r", "d"],
        quit_tokens=["q", "quit"],
    )
    if token == "":
        return "d"
    return token[0].lower()

def _capture_mean_gyro(imu, *, duration_s: float, sample_dt_s: float = 0.01) -> np.ndarray:
    start = time.time()
    samples: List[np.ndarray] = []
    while time.time() - start < duration_s:
        s = imu.read()
        samples.append(np.asarray(s.gyro_rad_s, dtype=np.float32))
        time.sleep(sample_dt_s)
    if not samples:
        return np.zeros(3, dtype=np.float32)
    return np.mean(np.stack(samples, axis=0), axis=0)


def _capture_imu_series(
    imu,
    *,
    duration_s: float,
    last_ts: float,
    label: str = "imu",
    sample_dt_s: float = 0.01,
    stall_timeout_s: float = 2.0,
    progress_every_s: float = DEFAULT_IMU_PROGRESS_EVERY_S,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Capture fresh valid IMU samples for duration, ensuring timestamp_s advances."""
    start = time.monotonic()
    stale_since = time.monotonic()
    samples: List[np.ndarray] = []
    quat_samples: List[np.ndarray] = []
    last_progress = start
    start_ts = float(last_ts)
    timestamp_updates = 0
    duplicate_reads = 0
    valid_reads = 0
    invalid_reads = 0
    zero_gyro_reads = 0
    reads = 0

    while time.monotonic() - start < duration_s:
        t0 = time.monotonic()
        s = imu.read()
        t1 = time.monotonic()
        reads += 1
        read_dt = t1 - t0
        if read_dt > 0.2:
            raise RuntimeError(
                f"IMU read() is unexpectedly slow ({read_dt:.3f}s). "
                "This can indicate an I2C bus lock or a blocked background reader thread."
            )
        ts = float(getattr(s, "timestamp_s", 0.0))
        valid = bool(getattr(s, "valid", True))
        timestamp_changed = ts > 0.0 and ts != last_ts
        if timestamp_changed:
            last_ts = ts
            timestamp_updates += 1
        else:
            duplicate_reads += 1

        gyro_sample = np.asarray(s.gyro_rad_s, dtype=np.float32)
        if valid:
            valid_reads += 1
        else:
            invalid_reads += 1
        if gyro_sample.shape == (3,) and float(np.linalg.norm(gyro_sample)) < 1e-8:
            zero_gyro_reads += 1
        if timestamp_changed and valid:
            stale_since = time.monotonic()
            samples.append(gyro_sample)
            quat_samples.append(np.asarray(s.quat_xyzw, dtype=np.float32))
        elif time.monotonic() - stale_since > stall_timeout_s:
            raise RuntimeError(
                "IMU sample stream stalled during gyro capture (no fresh valid samples). "
                f"Check I2C stability/power. reads={reads} valid_reads={valid_reads} "
                f"invalid_reads={invalid_reads} duplicate_reads={duplicate_reads}; {_format_imu_debug(imu, s)}"
            )
        now = time.monotonic()
        if progress_every_s > 0 and now - last_progress >= progress_every_s:
            elapsed = now - start
            remaining = max(0.0, duration_s - elapsed)
            _log_line(
                f"{label}: captured {len(samples)} fresh samples "
                f"(t={elapsed:.1f}/{duration_s:.1f}s, remaining={remaining:.1f}s, reads={reads})"
            )
            last_progress = now
        time.sleep(sample_dt_s)

    if not samples:
        if progress_every_s > 0:
            print("  ... capture complete: 0 fresh samples", flush=True)
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            last_ts,
        )
    if progress_every_s > 0:
        elapsed = time.monotonic() - start
        _log_line(
            f"{label}: capture complete: {len(samples)} fresh samples (t={elapsed:.1f}/{duration_s:.1f}s) "
            f"reads={reads} valid_reads={valid_reads} invalid_reads={invalid_reads} "
            f"timestamp_updates={timestamp_updates} duplicate_reads={duplicate_reads} "
            f"zero_gyro_reads={zero_gyro_reads} ts={start_ts:.3f}->{last_ts:.3f}"
        )
    return (
        np.stack(samples, axis=0).astype(np.float32),
        np.stack(quat_samples, axis=0).astype(np.float32),
        last_ts,
    )


def _wait_for_imu_stream(imu, *, timeout_s: float = 2.0) -> float:
    """Wait until the IMU wrapper starts producing fresh samples.

    Returns the latest timestamp_s to use as a baseline for subsequent capture calls.
    """
    start = time.monotonic()
    last_sample = imu.read()
    last_ts = float(getattr(last_sample, "timestamp_s", 0.0) or 0.0)
    reads = 1
    first_read_s = time.monotonic() - start
    if first_read_s >= timeout_s and last_ts > 0.0 and bool(getattr(last_sample, "valid", True)):
        _log_line(
            "IMU first read was slow but valid "
            f"(read_s={first_read_s:.3f}, timestamp_s={last_ts:.3f}); continuing."
        )
        return last_ts
    while time.monotonic() - start < timeout_s:
        s = imu.read()
        reads += 1
        last_sample = s
        ts = float(getattr(s, "timestamp_s", 0.0) or 0.0)
        if ts > 0.0 and ts != last_ts and bool(getattr(s, "valid", True)):
            return ts
        if ts > 0.0 and ts != last_ts:
            last_ts = ts
        time.sleep(0.01)
    raise RuntimeError(
        "IMU stream did not produce fresh valid samples (timestamp_s not advancing with valid=True). "
        "Check I2C wiring/address and ensure the IMU is powered. "
        f"reads={reads}; {_format_imu_debug(imu, last_sample)}"
    )


def _format_imu_debug(imu, sample) -> str:
    parts = [
        f"polling_mode={getattr(imu, 'polling_mode', None)}",
        f"valid={getattr(sample, 'valid', None)}",
        f"timestamp_s={getattr(sample, 'timestamp_s', None)}",
    ]
    quat = np.asarray(getattr(sample, "quat_xyzw", []), dtype=np.float32).reshape(-1)
    if quat.size == 4:
        parts.append(f"quat_norm={float(np.linalg.norm(quat)):.3f}")
        parts.append(f"quat={np.round(quat, 4).tolist()}")
    gyro = np.asarray(getattr(sample, "gyro_rad_s", []), dtype=np.float32).reshape(-1)
    if gyro.size == 3:
        parts.append(f"gyro={np.round(gyro, 4).tolist()}")
    for attr in ("error_count", "last_error", "diag"):
        if hasattr(imu, attr):
            try:
                parts.append(f"{attr}={getattr(imu, attr)}")
            except Exception:
                pass
    last_traceback = getattr(imu, "last_traceback", None)
    if last_traceback:
        last_line = str(last_traceback).strip().splitlines()[-1]
        parts.append(f"last_traceback_tail={last_line}")
    return "; ".join(parts)


def _countdown(*, label: str, seconds: float) -> None:
    """Print a short countdown for interactive calibration steps."""
    steps = int(max(0, round(seconds)))
    if steps <= 0:
        return
    print(label, flush=True)
    for i in range(steps, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1.0)
    print("  GO", flush=True)


def _pause(*, label: str, seconds: float) -> None:
    """Pause for a fixed duration with minimal output.

    On some SSH/PTY stacks, per-second countdown prints can be dropped/reordered. For the
    critical "get still" periods, we keep output to two lines: start + done.
    """
    s = float(seconds)
    if s <= 0:
        return
    _log_line(f"{label} ({s:.0f}s)...")
    time.sleep(s)
    _log_line("OK")


def _log_line(msg: str) -> None:
    # Add a monotonic timestamp so that, even if a terminal buffers/delays output,
    # the operator can still reason about ordering.
    print(f"[t={time.monotonic():.3f}] {msg}", flush=True)


def _supports_color() -> bool:
    # Respect NO_COLOR (https://no-color.org/) and only colorize on TTYs.
    if not sys.stdout.isatty():
        return False
    if "NO_COLOR" in os.environ:
        return False
    term = os.environ.get("TERM", "")
    if term in {"", "dumb"}:
        return False
    return True


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _yellow(text: str) -> str:
    return _color(text, "33")


def _red(text: str) -> str:
    return _color(text, "31")


def _bell() -> None:
    # Terminal bell. Some SSH terminals will still play it even if lines are dropped.
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _status_loop(*, label: str, seconds: float, hz: float = DEFAULT_IMU_STATUS_HZ) -> None:
    """Print a repeating status line for `seconds` to survive occasional dropped lines."""
    s = float(seconds)
    if s <= 0:
        return
    period = 1.0 / max(1.0, float(hz))
    end = time.monotonic() + s
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            break
        _log_line(f"{label} (remaining={remaining:.1f}s)")
        time.sleep(min(period, max(0.0, remaining)))

def _wait_for_token(
    *,
    prompt_lines: List[str],
    heartbeat_lines: Optional[List[str]] = None,
    valid: List[str],
    quit_tokens: List[str],
    reprint_every_s: float = DEFAULT_PROMPT_REPRINT_S,
) -> str:
    """Wait for a typed token from stdin, reprinting prompt periodically.

    This avoids the common failure mode where a single printed prompt line is dropped
    by the terminal/PTY stack, leaving the script blocked in input() with no visible cue.
    """
    valid_set = {v.lower() for v in valid}
    quit_set = {v.lower() for v in quit_tokens}

    def _print_prompt() -> None:
        print("", flush=True)
        for line in prompt_lines:
            for sub in str(line).splitlines():
                print(sub, flush=True)

    def _print_heartbeat() -> None:
        # Keep this short to avoid drowning out interactive sessions.
        hint = "/".join(valid + quit_tokens)
        if heartbeat_lines:
            print("", flush=True)
            for line in heartbeat_lines:
                for sub in str(line).splitlines():
                    print(sub, flush=True)
        print(f"(waiting: {hint})", flush=True)

    # If not interactive, fall back to blocking input once.
    if not sys.stdin.isatty():
        _print_prompt()
        s = input().strip().lower()
        return s

    buf = ""
    last_print = 0.0
    printed_full = False
    while True:
        now = time.monotonic()
        if now - last_print >= max(0.1, float(reprint_every_s)):
            if not printed_full:
                _print_prompt()
                printed_full = True
            else:
                _print_heartbeat()
            last_print = now

        r, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not r:
            continue

        chunk = sys.stdin.read(1)
        if chunk == "":
            # EOF
            return "q"
        if chunk in {"\n", "\r"}:
            token = buf.strip().lower()
            buf = ""
            if token == "":
                continue
            if token in quit_set:
                return token
            if token in valid_set:
                return token
            print(f"Unrecognized input: {token!r} (valid: {valid + quit_tokens})", flush=True)
            continue
        buf += chunk


@dataclass(frozen=True)
class AxisMeasurement:
    body_axis: str
    angle_rad: float
    axis_sensor: np.ndarray  # (3,) float32 unit vector


def _infer_rotation_axis_from_gyro(
    *,
    baseline_gyro: np.ndarray,
    motion_gyro: np.ndarray,
    duration_s: float,
) -> Optional[tuple[float, np.ndarray]]:
    motion = np.asarray(motion_gyro, dtype=np.float32).reshape(-1, 3)
    if motion.shape[0] == 0:
        return None

    baseline = np.asarray(baseline_gyro, dtype=np.float32).reshape(-1, 3)
    bias = (
        np.mean(baseline, axis=0).astype(np.float32)
        if baseline.shape[0] > 0
        else np.zeros(3, dtype=np.float32)
    )
    corrected = motion - bias.reshape(1, 3)
    speed = np.linalg.norm(corrected, axis=1).astype(np.float32)
    finite = np.isfinite(speed)
    if not np.any(finite):
        return None

    speed_finite = speed[finite]
    threshold = max(0.05, 0.25 * float(np.percentile(speed_finite, 90.0)))
    active = finite & (speed >= threshold)
    if int(np.count_nonzero(active)) < 3:
        active = finite

    dt = float(duration_s) / max(1, int(np.count_nonzero(finite)))
    corrected_active = corrected[active]
    positive_impulse = np.sum(np.clip(corrected_active, 0.0, None), axis=0).astype(np.float32) * dt
    negative_impulse = np.sum(np.clip(corrected_active, None, 0.0), axis=0).astype(np.float32) * dt
    signed_impulses = np.concatenate([positive_impulse, -negative_impulse]).astype(np.float32)
    impulse_idx = int(np.argmax(signed_impulses))
    signed_angle_rad = float(signed_impulses[impulse_idx])
    if not np.isfinite(signed_angle_rad) or signed_angle_rad < 1e-6:
        return None

    axis_idx = impulse_idx % 3
    axis_sign = 1.0 if impulse_idx < 3 else -1.0
    same_direction = active & ((corrected[:, axis_idx] * axis_sign) > 0.0)
    if int(np.count_nonzero(same_direction)) < 3:
        same_direction = active

    rot_vec = np.sum(corrected[same_direction], axis=0).astype(np.float32) * dt
    angle_rad = float(np.linalg.norm(rot_vec))
    if not np.isfinite(angle_rad) or angle_rad < 1e-6:
        return None

    return angle_rad, (rot_vec / angle_rad).astype(np.float32)


def _format_vec3(v: np.ndarray, *, precision: int = 4) -> str:
    a = np.asarray(v, dtype=np.float32).reshape(3)
    return "[" + ", ".join(f"{float(x):+.{precision}f}" for x in a) + "]"


def _log_gyro_axis_debug(
    *,
    body_axis_name: str,
    baseline_gyro: np.ndarray,
    motion_gyro: np.ndarray,
    duration_s: float,
) -> None:
    baseline = np.asarray(baseline_gyro, dtype=np.float32).reshape(-1, 3)
    motion = np.asarray(motion_gyro, dtype=np.float32).reshape(-1, 3)
    if motion.shape[0] == 0:
        _log_line(f"{body_axis_name}: gyro debug: no motion gyro samples")
        return

    bias = (
        np.mean(baseline, axis=0).astype(np.float32)
        if baseline.shape[0] > 0
        else np.zeros(3, dtype=np.float32)
    )
    corrected = motion - bias.reshape(1, 3)
    speed = np.linalg.norm(corrected, axis=1).astype(np.float32)
    finite = np.isfinite(speed)
    finite_count = int(np.count_nonzero(finite))
    speed_finite = speed[finite] if finite_count > 0 else np.zeros(1, dtype=np.float32)
    threshold = max(0.05, 0.25 * float(np.percentile(speed_finite, 90.0)))
    active = finite & (speed >= threshold)
    if int(np.count_nonzero(active)) < 3:
        active = finite
    dt = float(duration_s) / max(1, finite_count)
    corrected_active = corrected[active] if int(np.count_nonzero(active)) > 0 else corrected[finite]
    positive_impulse = np.sum(np.clip(corrected_active, 0.0, None), axis=0).astype(np.float32) * dt
    negative_impulse = np.sum(np.clip(corrected_active, None, 0.0), axis=0).astype(np.float32) * dt
    signed_impulses = np.concatenate([positive_impulse, -negative_impulse]).astype(np.float32)
    labels = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]
    best_idx = int(np.argmax(signed_impulses))
    impulse_text = ", ".join(f"{label}={float(value):.4f}" for label, value in zip(labels, signed_impulses, strict=True))
    mid_idx = motion.shape[0] // 2

    _log_line(
        f"{body_axis_name}: gyro debug samples baseline={baseline.shape[0]} motion={motion.shape[0]} "
        f"finite_motion={finite_count}"
    )
    if baseline.shape[0] > 0:
        _log_line(
            f"{body_axis_name}: gyro debug baseline mean_rad_s={_format_vec3(bias)} "
            f"std_rad_s={_format_vec3(np.std(baseline, axis=0).astype(np.float32))}"
        )
    _log_line(
        f"{body_axis_name}: gyro debug motion mean_rad_s={_format_vec3(np.mean(motion, axis=0).astype(np.float32))} "
        f"max_abs_rad_s={_format_vec3(np.max(np.abs(motion), axis=0).astype(np.float32))}"
    )
    _log_line(
        f"{body_axis_name}: gyro debug corrected_speed_rad_s "
        f"p50={float(np.percentile(speed_finite, 50.0)):.4f} "
        f"p90={float(np.percentile(speed_finite, 90.0)):.4f} "
        f"p99={float(np.percentile(speed_finite, 99.0)):.4f} "
        f"max={float(np.max(speed_finite)):.4f} threshold={threshold:.4f} "
        f"active={int(np.count_nonzero(active))}/{motion.shape[0]}"
    )
    _log_line(f"{body_axis_name}: gyro debug signed_impulse_rad {impulse_text} best={labels[best_idx]}")
    _log_line(
        f"{body_axis_name}: gyro debug motion first/mid/last "
        f"{_format_vec3(motion[0])} / {_format_vec3(motion[mid_idx])} / {_format_vec3(motion[-1])}"
    )


def _load_axis_map_measurements(raw_config: dict) -> dict[str, np.ndarray]:
    bno_cfg = raw_config.get("bno085", {}) if isinstance(raw_config, dict) else {}
    measurements = bno_cfg.get("axis_map_measurements", {})
    if not isinstance(measurements, dict):
        return {}

    axes: dict[str, np.ndarray] = {}
    for key in ("body_x", "body_y", "body_z"):
        value = measurements.get(key)
        if not isinstance(value, list) or len(value) != 3:
            continue
        axis = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if np.isfinite(norm) and norm > 1e-6:
            axes[key] = (axis / norm).astype(np.float32)
    return axes


def _measure_axis_once(
    *,
    imu,
    body_axis_name: str,
    instruction: str,
    baseline_s: float = DEFAULT_IMU_BASELINE_S,
    motion_s: float = DEFAULT_IMU_MOTION_S,
) -> Optional[AxisMeasurement]:
    """Capture baseline + motion and compute dominant rotation axis in sensor frame.

    This function is intentionally non-interactive (no input prompts).
    """
    _log_line("-" * 60)
    _log_line(f"IMU axis calibration: {body_axis_name}")
    instruction_lines = [ln.strip() for ln in instruction.splitlines() if ln.strip()]
    action_hint = ""
    if instruction_lines and instruction_lines[0].lower().startswith("action:"):
        action_hint = instruction_lines[0].split(":", 1)[1].strip()
    for line in instruction.splitlines():
        _log_line(line)
    _pause(label=f"{body_axis_name}: get robot still; baseline starts in", seconds=DEFAULT_IMU_GUIDANCE_PAUSE_S)

    _log_line(f"{body_axis_name}: baseline capture (hold still) {baseline_s:.1f}s")
    last_ts = _wait_for_imu_stream(imu, timeout_s=2.0)
    baseline_gyro, baseline_quat, last_ts = _capture_imu_series(
        imu,
        duration_s=float(baseline_s),
        last_ts=last_ts,
        label=f"{body_axis_name}: BASELINE",
        sample_dt_s=0.01,
        progress_every_s=1.0,
    )
    _log_line(f"{body_axis_name}: baseline complete (samples={int(baseline_quat.shape[0])})")
    baseline_ref_quat = (
        baseline_quat[-1].astype(np.float32)
        if baseline_quat.shape[0] > 0
        else np.array([0, 0, 0, 1], dtype=np.float32)
    )

    if action_hint:
        _log_line(f"{body_axis_name}: prepare to rotate ({action_hint})")
    else:
        _log_line(f"{body_axis_name}: prepare to rotate (motion capture starts soon)")
    _status_loop(
        label=(
            f"{body_axis_name}: GET READY for {action_hint} (rotate window)"
            if action_hint
            else f"{body_axis_name}: GET READY for motion capture (rotate window)"
        ),
        seconds=3.0,
        hz=DEFAULT_IMU_STATUS_HZ,
    )
    if action_hint:
        _log_line(_yellow(f"{body_axis_name}: ROTATE NOW - {action_hint} ({motion_s:.1f}s)"))
    else:
        _log_line(_yellow(f"{body_axis_name}: ROTATE NOW (motion capture {motion_s:.1f}s)"))
    _bell()
    _bell()
    _log_line(_yellow(f"{body_axis_name}: motion capture started"))
    motion_gyro, motion_quat, last_ts = _capture_imu_series(
        imu,
        duration_s=float(motion_s),
        last_ts=last_ts,
        label=_yellow(f"{body_axis_name}: ROTATE NOW"),
        sample_dt_s=0.01,
        progress_every_s=1.0,
    )
    _log_line(_yellow(f"{body_axis_name}: ROTATE NOW window ended"))
    _log_line(_yellow(f"{body_axis_name}: motion capture finished"))
    _log_line(f"{body_axis_name}: motion complete (samples={int(motion_quat.shape[0])})")
    _log_line(f"{body_axis_name}: computing dominant rotation axis")
    if motion_gyro.shape[0] == 0:
        _log_line(f"{body_axis_name}: ERROR no IMU samples captured")
        return None
    compute_t0 = time.monotonic()

    try:
        with _alarm_timeout(
            2.0,
            message=(
                f"{body_axis_name}: axis inference timed out. "
                "This should be instant; if it hangs, something is wrong with the runtime environment."
            ),
        ):
            _log_line(f"{body_axis_name}: axis inference step: gyro dominant axis")
            _log_gyro_axis_debug(
                body_axis_name=body_axis_name,
                baseline_gyro=baseline_gyro,
                motion_gyro=motion_gyro,
                duration_s=float(motion_s),
            )
            gyro_result = _infer_rotation_axis_from_gyro(
                baseline_gyro=baseline_gyro,
                motion_gyro=motion_gyro,
                duration_s=float(motion_s),
            )
            if gyro_result is not None:
                gyro_angle_rad, gyro_axis = gyro_result
                _log_line(f"{body_axis_name}: gyro_dominant_angle_rad={gyro_angle_rad:.3f}")
                _log_line(f"{body_axis_name}: gyro_axis_sensor={gyro_axis.tolist()}")
            else:
                _log_line(
                    _yellow(
                        f"{body_axis_name}: gyro could not infer a dominant axis; trying quaternion fallback."
                    )
                )

            def quat_normalize_vec(q: np.ndarray) -> np.ndarray:
                q = q.astype(np.float32)
                n = float(np.linalg.norm(q))
                if not np.isfinite(n) or n < 1e-6:
                    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                return (q / n).astype(np.float32)

            def quat_normalize_batch(q: np.ndarray) -> np.ndarray:
                q = q.astype(np.float32)
                n = np.linalg.norm(q, axis=1, keepdims=True).astype(np.float32)
                safe = n > 1e-6
                out = np.zeros_like(q, dtype=np.float32)
                out[:, 3] = 1.0
                out[safe[:, 0]] = (q[safe[:, 0]] / n[safe[:, 0]]).astype(np.float32)
                return out

            def quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
                return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

            def quat_mul_xyzw_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
                bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
                return np.stack(
                    [
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz,
                    ],
                    axis=1,
                ).astype(np.float32)

            _log_line(f"{body_axis_name}: axis inference step: normalize")
            q0 = quat_normalize_vec(baseline_ref_quat)
            q0_inv = quat_conj_xyzw(q0)
            motion_q = quat_normalize_batch(motion_quat)

            _log_line(f"{body_axis_name}: axis inference step: relative quat")
            rel = quat_mul_xyzw_batch(motion_q, q0_inv.reshape(1, 4))
            rel = quat_normalize_batch(rel)

            _log_line(f"{body_axis_name}: axis inference step: dominant axis")
            w = np.clip(np.abs(rel[:, 3]), 0.0, 1.0).astype(np.float32)
            ang = (2.0 * np.arccos(w)).astype(np.float32)
            i_max = int(np.argmax(ang))
            _log_line(f"{body_axis_name}: quat_max_relative_angle_rad={float(ang[i_max]):.6f}")
            mid_idx = motion_q.shape[0] // 2
            _log_line(
                f"{body_axis_name}: quat debug baseline_ref={np.round(q0, 4).tolist()} "
                f"motion_first/mid/last={np.round(motion_q[0], 4).tolist()} / "
                f"{np.round(motion_q[mid_idx], 4).tolist()} / {np.round(motion_q[-1], 4).tolist()}"
            )
            q_rel = rel[i_max].astype(np.float32)
            v = q_rel[:3].astype(np.float32)
            v_norm = float(np.linalg.norm(v))
            quat_result: Optional[tuple[float, np.ndarray]] = None
            if not np.isfinite(v_norm) or v_norm < 1e-6:
                _log_line(
                    _yellow(
                        f"{body_axis_name}: quaternion relative rotation vector is near zero; "
                        "using gyro inference if available."
                    )
                )
            else:
                quat_axis = (v / v_norm).astype(np.float32)
                quat_angle_rad = float(ang[i_max])
                quat_result = (quat_angle_rad, quat_axis)
                _log_line(f"{body_axis_name}: quat_max_relative_angle_rad={quat_angle_rad:.3f}")
                _log_line(f"{body_axis_name}: quat_axis_sensor={quat_axis.tolist()}")

            if gyro_result is not None:
                angle_rad, axis = gyro_result
                source = "gyro"
            elif quat_result is not None:
                angle_rad, axis = quat_result
                source = "quaternion"
            else:
                _log_line(
                    _red(
                        f"{body_axis_name}: could not infer a dominant axis from gyro or quaternion. "
                        "Try a larger single-direction rotation."
                    )
                )
                return None
    except TimeoutError as exc:
        _log_line(_red(str(exc)))
        return None
    except BaseException as exc:
        _log_line(_red(f"{body_axis_name}: axis inference failed: {type(exc).__name__}: {exc}"))
        _log_line(_red(traceback.format_exc()))
        return None

    compute_dt = time.monotonic() - compute_t0
    _log_line(f"{body_axis_name}: axis inference time {compute_dt:.3f}s")
    _log_line(f"{body_axis_name}: selected_axis_source={source}")
    _log_line(f"{body_axis_name}: selected_rotation_angle_rad={angle_rad:.3f}")
    _log_line(f"{body_axis_name}: selected_rotation_axis_sensor={axis.tolist()}")
    return AxisMeasurement(body_axis=body_axis_name, angle_rad=angle_rad, axis_sensor=axis)


def _calibrate_axis_from_motion(
    *,
    imu,
    body_axis_name: str,
    instruction: str,
    baseline_s: float = DEFAULT_IMU_BASELINE_S,
    motion_s: float = DEFAULT_IMU_MOTION_S,
    preview_s: float = DEFAULT_IMU_PREVIEW_S,
) -> Optional[np.ndarray]:
    print("\n" + "-" * 60)
    print(f"IMU axis calibration for body axis {body_axis_name}")
    print(instruction)
    # Give the operator time to read and prepare without racing the script.
    _pause(label="Get robot still; baseline starts in", seconds=DEFAULT_IMU_GUIDANCE_PAUSE_S)

    # Avoid additional "press Enter" prompts here: they are easy to miss under some
    # PTY/launcher setups and lead to a "stuck" experience. If the user chose to run
    # this step, proceed automatically with a short countdown.
    while True:
        # Baseline: no countdown to reduce output interleaving; just instruct clearly.
        print("", flush=True)
        print(f"Baseline capture: hold still for {baseline_s:.1f}s...", flush=True)
        last_ts = _wait_for_imu_stream(imu, timeout_s=2.0)
        baseline_gyro, baseline_quat, last_ts = _capture_imu_series(
            imu, duration_s=float(baseline_s), last_ts=last_ts, sample_dt_s=0.01
        )
        baseline_ref_quat = (
            baseline_quat[-1].astype(np.float32)
            if baseline_quat.shape[0] > 0
            else np.array([0, 0, 0, 1], dtype=np.float32)
        )

        print("", flush=True)
        print("Motion capture is next.", flush=True)
        _countdown(label="Motion capture starts", seconds=3.0)
        print(f"Starting motion capture for {motion_s:.1f}s. Do not press keys during capture.", flush=True)
        print(f"MOVE NOW! Rotate a lot around the instructed axis for ~{motion_s:.1f}s.", flush=True)
        motion_gyro, motion_quat, last_ts = _capture_imu_series(
            imu, duration_s=float(motion_s), last_ts=last_ts, sample_dt_s=0.01
        )
        print("Motion capture complete.", flush=True)
        print("Computing dominant rotation axis...", flush=True)
        if motion_gyro.shape[0] == 0:
            print("No IMU samples captured.", flush=True)
            return None

        # Quaternion-based inference (robust even if gyro reports are missing/zero).
        # Find the motion sample with maximum rotation from baseline_ref_quat and use its axis.
        compute_t0 = time.monotonic()

        def quat_normalize_vec(q: np.ndarray) -> np.ndarray:
            q = q.astype(np.float32)
            n = float(np.linalg.norm(q))
            if not np.isfinite(n) or n < 1e-6:
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return (q / n).astype(np.float32)

        def quat_normalize_batch(q: np.ndarray) -> np.ndarray:
            q = q.astype(np.float32)
            n = np.linalg.norm(q, axis=1, keepdims=True).astype(np.float32)
            safe = n > 1e-6
            out = np.zeros_like(q, dtype=np.float32)
            out[:, 3] = 1.0
            out[safe[:, 0]] = (q[safe[:, 0]] / n[safe[:, 0]]).astype(np.float32)
            return out

        def quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
            return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

        def quat_mul_xyzw_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Quaternion multiply (xyzw) for batch a (N,4) and b (4,) or (N,4)."""
            ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
            return np.stack(
                [
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                    aw * bw - ax * bx - ay * by - az * bz,
                ],
                axis=1,
            ).astype(np.float32)

        q0 = quat_normalize_vec(baseline_ref_quat)
        q0_inv = quat_conj_xyzw(q0)
        motion_q = quat_normalize_batch(motion_quat)
        rel = quat_mul_xyzw_batch(motion_q, q0_inv.reshape(1, 4))
        rel = quat_normalize_batch(rel)

        w = np.clip(np.abs(rel[:, 3]), 0.0, 1.0).astype(np.float32)
        ang = (2.0 * np.arccos(w)).astype(np.float32)
        i_max = int(np.argmax(ang))
        q_rel = rel[i_max].astype(np.float32)
        v = q_rel[:3].astype(np.float32)
        v_norm = float(np.linalg.norm(v))
        if not np.isfinite(v_norm) or v_norm < 1e-6:
            print(
                "Could not infer a dominant axis (rotation vector near zero). Try a larger single-direction rotation.",
                flush=True,
            )
            return None

        axis = (v / v_norm).astype(np.float32)

        angle_rad = float(ang[i_max])
        axis_list = axis.tolist()
        compute_dt = time.monotonic() - compute_t0
        print(f"Axis inference time: {compute_dt:.2f}s", flush=True)
        result_summary = (
            f"Result for {body_axis_name}:\n"
            f"  max_relative_rotation_angle_rad: {angle_rad:.3f}\n"
            f"  relative_rotation_axis_sensor: {axis_list}\n"
        )
        action = _prompt_accept_measurement(
            prompt=f"Accept this measurement for {body_axis_name}?",
            context=result_summary,
        )
        if action == "y":
            return axis
        if action == "q":
            return None
        if action == "r":
            print("Redo requested; re-running this axis measurement.", flush=True)
            continue
        if action == "d":
            continue
        print("Rejected measurement; leaving this axis unset.", flush=True)
        return None


def calibrate_imu_axis_map(
    *,
    config: WrRuntimeConfig,
    raw_config: dict,
    output_path: Path,
    upside_down: bool,
) -> None:
    raise RuntimeError("Use calibrate_imu_axis_map_step() instead.")


def calibrate_imu_axis_map_step(
    *,
    config: WrRuntimeConfig,
    raw_config: dict,
    output_path: Path,
    upside_down: bool,
    step: str,
) -> None:
    try:
        from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    except Exception as exc:
        raise RuntimeError(
            "IMU axis calibration requires runtime IMU deps on the target (Adafruit Blinka + BNO08X)."
        ) from exc

    bno_cfg = raw_config.setdefault("bno085", {}) if isinstance(raw_config, dict) else {}
    measurements = bno_cfg.get("axis_map_measurements", {})
    if not isinstance(measurements, dict):
        measurements = {}

    measured_axes: dict[str, np.ndarray] = {}
    for k in ("body_x", "body_y", "body_z"):
        v = measurements.get(k)
        if isinstance(v, list) and len(v) == 3:
            measured_axes[k] = np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)

    def axis_vec_to_mapping(axis_vec: np.ndarray) -> str:
        axis_vec = np.asarray(axis_vec, dtype=np.float32).reshape(3)
        idx = int(np.argmax(np.abs(axis_vec)))
        sign = "+" if float(axis_vec[idx]) >= 0.0 else "-"
        letter = ["X", "Y", "Z"][idx]
        return f"{sign}{letter}"

    def solve_axis_map_from_measurements(*, axes_by_body: dict[str, np.ndarray]) -> Optional[list[str]]:
        required = ["body_x", "body_y", "body_z"]
        if any(k not in axes_by_body for k in required):
            return None

        a = np.stack([axes_by_body[k].astype(np.float32) for k in required], axis=0)
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-6, None)

        basis = np.eye(3, dtype=np.float32)
        axis_letters = ["X", "Y", "Z"]

        import itertools

        best_score = -1e9
        best_perm = None
        best_signs = None

        for perm in itertools.permutations([0, 1, 2], 3):
            for signs in itertools.product([-1.0, 1.0], repeat=3):
                rows = np.stack([signs[i] * basis[perm[i]] for i in range(3)], axis=0).astype(np.float32)
                if float(np.linalg.det(rows)) < 0.0:
                    continue
                score = float(np.sum(rows * a))
                if score > best_score:
                    best_score = score
                    best_perm = perm
                    best_signs = signs

        if best_perm is None or best_signs is None:
            return None

        axis_map: list[str] = []
        for i in range(3):
            sign = "+" if best_signs[i] > 0 else "-"
            axis_map.append(f"{sign}{axis_letters[best_perm[i]]}")

        avg = best_score / 3.0
        print(f"Solved axis_map avg_dot={avg:.3f}", flush=True)
        return axis_map

    if step == "clear":
        bno_cfg.pop("axis_map_measurements", None)
        write_bno_config(raw_config, output_path, axis_map_measurements=None)
        print(f"Cleared saved axis measurements in {output_path}", flush=True)
        return

    if step == "solve":
        axis_map = solve_axis_map_from_measurements(axes_by_body=measured_axes)
        if axis_map is None:
            have = sorted(measured_axes.keys())
            print(f"Not enough measurements to solve axis_map (need body_x/body_y/body_z). Have: {have}", flush=True)
            return
        resp = _prompt_choice_strict(f"Write bno085.axis_map={axis_map} to config?")
        if resp == "y":
            write_bno_config(raw_config, output_path, axis_map=axis_map)
            print(f"Wrote bno085.axis_map to {output_path}: {axis_map}", flush=True)
        return

    if step not in {"body_x", "body_y", "body_z"}:
        raise ValueError(f"Unknown axis_map calibration step: {step!r}")

    try:
        with _alarm_timeout(
            10,
            message=(
                "Timed out initializing BNO085 IMU for axis measurement. Check I2C wiring/address and ensure no other process is holding the I2C lock."
            ),
        ):
            imu = _init_calibration_bno085(
                BNO085IMU,
                config=config,
                upside_down=upside_down,
            )
    except TimeoutError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        time.sleep(0.2)
        _log_line(
            "IMU init: "
            f"polling_mode={getattr(imu, 'polling_mode', None)} "
            f"sampling_hz={getattr(imu, 'sampling_hz', None)} "
            f"enable_rotation_vector={getattr(imu, 'enable_rotation_vector', None)} "
            f"suppress_debug={getattr(imu, 'suppress_debug', None)}"
        )
        instruction_by_axis = {
            "body_z": (
                "Action: yaw LEFT (turn counter-clockwise when viewed from above), then return to still.\n"
                "Expected: positive rotation around +Z in the robot body frame."
            ),
            "body_y": (
                "Action: pitch UP (nose up), then return to still.\n"
                "Expected: positive rotation around +Y in the robot body frame."
            ),
            "body_x": (
                "Action: roll RIGHT (right side down), then return to still.\n"
                "Expected: positive rotation around +X in the robot body frame."
            ),
        }
        measured = _calibrate_axis_from_motion(
            imu=imu,
            body_axis_name=step,
            instruction=instruction_by_axis[step],
        )
        if measured is None:
            print("No measurement saved.", flush=True)
            return

        # Save raw axis measurement (preferred for global solve).
        measurements[step] = [float(measured[0]), float(measured[1]), float(measured[2])]
        bno_cfg["axis_map_measurements"] = measurements
        write_bno_config(raw_config, output_path, axis_map_measurements=measurements)

        proposed = axis_vec_to_mapping(measured)
        print(f"Saved {step} measurement to {output_path}", flush=True)
        print(f"Naive per-axis mapping would be: {step} -> {proposed}", flush=True)
    finally:
        try:
            imu.close()
        except Exception:
            pass


def calibrate_imu_axis_map_full(
    *,
    config: WrRuntimeConfig,
    raw_config: dict,
    output_path: Path,
    upside_down: bool,
) -> None:
    """Run body_z -> body_y -> body_x measurements, then solve+write axis_map.

    This flow is intentionally non-interactive (no input prompts) until the very end,
    to avoid PTY/prompt rendering issues on some SSH setups.
    """
    saved_axes_by_body = _load_axis_map_measurements(raw_config)
    imu = None

    if len(saved_axes_by_body) < 3:
        try:
            from runtime.wr_runtime.hardware.bno085 import BNO085IMU
        except Exception as exc:
            raise RuntimeError(
                "IMU axis calibration requires runtime IMU deps on the target (Adafruit Blinka + BNO08X)."
            ) from exc

        def _init_imu():
            return _init_calibration_bno085(
                BNO085IMU,
                config=config,
                upside_down=upside_down,
            )

        try:
            with _alarm_timeout(
                10,
                message=(
                    "Timed out initializing BNO085 IMU for axis calibration. Check I2C wiring/address and ensure no other process is holding the I2C lock."
                ),
            ):
                imu = _init_imu()
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

    try:
        if imu is not None:
            time.sleep(0.2)
            _log_line(
                "IMU init: "
                f"polling_mode={getattr(imu, 'polling_mode', None)} "
                f"sampling_hz={getattr(imu, 'sampling_hz', None)} "
                f"enable_rotation_vector={getattr(imu, 'enable_rotation_vector', None)} "
                f"suppress_debug={getattr(imu, 'suppress_debug', None)}"
            )
        instruction_by_axis = {
            "body_z": (
                "Action: yaw LEFT (turn counter-clockwise when viewed from above), then return to still.\n"
                "Expected: positive rotation around +Z in the robot body frame."
            ),
            "body_y": (
                "Action: pitch UP (nose up), then return to still.\n"
                "Expected: positive rotation around +Y in the robot body frame."
            ),
            "body_x": (
                "Action: roll RIGHT (right side down), then return to still.\n"
                "Expected: positive rotation around +X in the robot body frame."
            ),
        }

        bno_cfg = raw_config.setdefault("bno085", {}) if isinstance(raw_config, dict) else {}
        raw_measurements = bno_cfg.get("axis_map_measurements", {})
        measurements: dict[str, list[float]] = (
            dict(raw_measurements) if isinstance(raw_measurements, dict) else {}
        )
        axes_by_body = dict(saved_axes_by_body)
        angles_by_body: dict[str, float] = {}
        align_by_body: dict[str, float] = {}
        axis_map: list[str] = ["+X", "+Y", "+Z"]
        for step, axis in axes_by_body.items():
            axis_letter, expected_axis = _expected_sensor_axis_for_body(step)
            dot = float(np.dot(axis.astype(np.float32), expected_axis))
            align_by_body[step] = abs(dot)
            idx = {"body_x": 0, "body_y": 1, "body_z": 2}[step]
            sign = "+" if dot >= 0.0 else "-"
            axis_map[idx] = f"{sign}{axis_letter}"
            _log_line(
                f"{step}: reusing saved measurement axis_sensor={axis.tolist()} "
                f"alignment(|dot|)={align_by_body[step]:.3f}"
            )

        for step in ("body_z", "body_y", "body_x"):
            if step in axes_by_body:
                continue
            if imu is None:
                print(f"Missing saved measurement for {step}; rerun with IMU connected.", flush=True)
                return
            accepted: Optional[AxisMeasurement] = None
            for attempt in range(1, DEFAULT_IMU_MAX_ATTEMPTS + 1):
                _log_line(f"{step}: attempt {attempt}/{DEFAULT_IMU_MAX_ATTEMPTS}")
                m = _measure_axis_once(
                    imu=imu,
                    body_axis_name=step,
                    instruction=instruction_by_axis[step],
                    baseline_s=DEFAULT_IMU_BASELINE_S,
                    motion_s=DEFAULT_IMU_MOTION_S,
                )
                if m is None:
                    print(f"Failed to measure {step}. Aborting without writing config.", flush=True)
                    return

                if m.angle_rad >= DEFAULT_IMU_MIN_ANGLE_RAD:
                    axis_letter, expected_axis = _expected_sensor_axis_for_body(step)
                    dot = float(np.dot(m.axis_sensor.astype(np.float32), expected_axis))
                    align = abs(dot)
                    _log_line(f"{step}: axis alignment |dot|={align:.3f} (expected {axis_letter})")
                    if align < DEFAULT_IMU_AXIS_ALIGN_COS:
                        _log_line(
                            _red(
                                f"{step}: dominant axis not aligned with expected {axis_letter} "
                                f"(|dot|={align:.3f} < {DEFAULT_IMU_AXIS_ALIGN_COS:.3f}). "
                                "Redo with a cleaner single-axis rotation, or your IMU mount is rotated (needs full permutation solve)."
                            )
                        )
                        _pause(label=f"{step}: resetting before retry", seconds=2.0)
                        continue

                    sign = "+" if dot >= 0.0 else "-"
                    idx = {"body_x": 0, "body_y": 1, "body_z": 2}[step]
                    axis_map[idx] = f"{sign}{axis_letter}"
                    accepted = m
                    align_by_body[step] = align
                    _log_line(
                        f"{step}: angle OK (>= {DEFAULT_IMU_MIN_ANGLE_RAD:.3f}), "
                        f"alignment OK (>= {DEFAULT_IMU_AXIS_ALIGN_COS:.3f})"
                    )
                    break

                _log_line(
                    _red(
                        f"{step}: rotation too small (angle_rad={m.angle_rad:.3f} < {DEFAULT_IMU_MIN_ANGLE_RAD:.3f}). "
                        "Redo: rotate more during the motion window."
                    )
                )
                # Give the operator a moment to reset the robot to a stable still pose.
                _pause(label=f"{step}: resetting before retry", seconds=2.0)

            if accepted is None:
                print(
                    f"{step}: rotation too small after {DEFAULT_IMU_MAX_ATTEMPTS} attempts; aborting without writing config.",
                    flush=True,
                )
                return

            axes_by_body[step] = accepted.axis_sensor.astype(np.float32)
            angles_by_body[step] = float(accepted.angle_rad)
            measurements[step] = [
                float(accepted.axis_sensor[0]),
                float(accepted.axis_sensor[1]),
                float(accepted.axis_sensor[2]),
            ]
            bno_cfg["axis_map_measurements"] = measurements
            write_bno_config(raw_config, output_path, axis_map_measurements=measurements)
            _log_line(f"{step}: accepted angle_rad={angles_by_body[step]:.3f} axis_sensor={axes_by_body[step].tolist()}")
            _log_line(
                f"{step}: saved measurement to {output_path}; rerun option 2 to resume if interrupted."
            )
            if step != "body_x":
                _pause(label=f"{step}: reposition for next axis", seconds=3.0)
        avg_align = float(np.mean([align_by_body[k] for k in ("body_x", "body_y", "body_z")]))

        def angle_text(step: str) -> str:
            angle = angles_by_body.get(step)
            if angle is None:
                return "saved"
            return f"{angle:.3f}"

        summary = (
            "IMU axis_map calibration summary:\n"
            f"  body_z: angle_rad={angle_text('body_z')} axis_sensor={axes_by_body['body_z'].tolist()}\n"
            f"  body_y: angle_rad={angle_text('body_y')} axis_sensor={axes_by_body['body_y'].tolist()}\n"
            f"  body_x: angle_rad={angle_text('body_x')} axis_sensor={axes_by_body['body_x'].tolist()}\n"
            f"  assumption: no permutation (only sign flips)\n"
            f"  solved_axis_map: {axis_map}\n"
            f"  avg_alignment(|dot|): {avg_align:.3f}\n"
        )
        summary_lines = summary.rstrip().splitlines()
        for line in summary_lines:
            _log_line(line)
        token = _wait_for_token(
            prompt_lines=[
                f"Write bno085.axis_map={axis_map} to {output_path}?",
                "Type y then Enter to write, n then Enter to abort, or r then Enter to redo calibration.",
            ],
            heartbeat_lines=summary_lines + ["", f"Proposed axis_map: {axis_map}"],
            valid=["y", "n", "r"],
            quit_tokens=["q", "quit"],
            reprint_every_s=DEFAULT_PROMPT_REPRINT_S,
        )
        if token == "r":
            _log_line("Redo requested: restarting full calibration.")
            return calibrate_imu_axis_map_full(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
            )
        if token != "y":
            print("Not writing config.", flush=True)
            return

        bno_cfg = raw_config.setdefault("bno085", {}) if isinstance(raw_config, dict) else {}
        bno_cfg["axis_map_measurements"] = measurements
        write_bno_config(raw_config, output_path, axis_map=axis_map, axis_map_measurements=measurements)
        print(f"Wrote bno085.axis_map to {output_path}: {axis_map}", flush=True)
    finally:
        if imu is not None:
            try:
                imu.close()
            except Exception:
                pass


def calibrate_imu_axis_sign_only(
    *,
    config: WrRuntimeConfig,
    raw_config: dict,
    output_path: Path,
    upside_down: bool,
    body_axis: str,
) -> None:
    """Calibrate a single body axis sign under the 'no permutation' assumption.

    This checks that the inferred dominant rotation axis aligns with the expected IMU axis
    (abs(dot) >= DEFAULT_IMU_AXIS_ALIGN_COS) and only solves the sign (+/-).
    """
    try:
        from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    except Exception as exc:
        raise RuntimeError("BNO085 IMU deps not available on this machine.") from exc

    axis_letter, expected_axis = _expected_sensor_axis_for_body(body_axis)
    instruction_by_axis = {
        "body_z": (
            "Action: yaw LEFT (turn counter-clockwise when viewed from above), then return to still.\n"
            "Expected: positive rotation around +Z in the robot body frame."
        ),
        "body_y": (
            "Action: pitch UP (nose up), then return to still.\n"
            "Expected: positive rotation around +Y in the robot body frame."
        ),
        "body_x": (
            "Action: roll RIGHT (right side down), then return to still.\n"
            "Expected: positive rotation around +X in the robot body frame."
        ),
    }
    if body_axis not in instruction_by_axis:
        raise ValueError(f"Unknown body_axis: {body_axis}")

    def _init_imu():
        return _init_calibration_bno085(
            BNO085IMU,
            config=config,
            upside_down=upside_down,
        )

    with _alarm_timeout(
        10,
        message=(
            "Timed out initializing BNO085 IMU for axis calibration. Check I2C wiring/address and ensure no other process is holding the I2C lock."
        ),
    ):
        imu = _init_imu()

    try:
        time.sleep(0.2)
        _log_line(
            "IMU init: "
            f"polling_mode={getattr(imu, 'polling_mode', None)} "
            f"sampling_hz={getattr(imu, 'sampling_hz', None)} "
            f"enable_rotation_vector={getattr(imu, 'enable_rotation_vector', None)} "
            f"suppress_debug={getattr(imu, 'suppress_debug', None)}"
        )

        accepted: Optional[AxisMeasurement] = None
        align: float = 0.0
        sign: str = "+"
        for attempt in range(1, DEFAULT_IMU_MAX_ATTEMPTS_SINGLE_AXIS + 1):
            _log_line(f"{body_axis}: attempt {attempt}/{DEFAULT_IMU_MAX_ATTEMPTS_SINGLE_AXIS}")
            m = _measure_axis_once(
                imu=imu,
                body_axis_name=body_axis,
                instruction=instruction_by_axis[body_axis],
                baseline_s=DEFAULT_IMU_BASELINE_S,
                motion_s=DEFAULT_IMU_MOTION_S,
            )
            if m is None:
                _log_line(_red(f"{body_axis}: measurement failed; aborting."))
                return

            if m.angle_rad < DEFAULT_IMU_MIN_ANGLE_RAD:
                _log_line(
                    _red(
                        f"{body_axis}: rotation too small (angle_rad={m.angle_rad:.3f} < {DEFAULT_IMU_MIN_ANGLE_RAD:.3f}). "
                        "Rotate more during the motion window."
                    )
                )
                _pause(label=f"{body_axis}: resetting before retry", seconds=2.0)
                continue

            dot = float(np.dot(m.axis_sensor.astype(np.float32), expected_axis))
            align = abs(dot)
            _log_line(f"{body_axis}: axis alignment |dot|={align:.3f} (expected {axis_letter})")
            if align < DEFAULT_IMU_AXIS_ALIGN_COS:
                _log_line(
                    _red(
                        f"{body_axis}: dominant axis not aligned with expected {axis_letter} "
                        f"(|dot|={align:.3f} < {DEFAULT_IMU_AXIS_ALIGN_COS:.3f}). "
                        "This suggests your IMU mount is rotated (needs full axis_map solve) or the motion was not clean."
                    )
                )
                _pause(label=f"{body_axis}: resetting before retry", seconds=2.0)
                continue

            sign = "+" if dot >= 0.0 else "-"
            accepted = m
            break

        if accepted is None:
            _log_line(
                _red(
                    f"{body_axis}: failed after {DEFAULT_IMU_MAX_ATTEMPTS_SINGLE_AXIS} attempts; not writing config."
                )
            )
            return

        idx = {"body_x": 0, "body_y": 1, "body_z": 2}[body_axis]
        current_map = getattr(config.bno085, "axis_map", None)
        axis_map = list(current_map) if isinstance(current_map, list) and len(current_map) == 3 else ["+X", "+Y", "+Z"]
        axis_map[idx] = f"{sign}{axis_letter}"

        # Update measurements block for traceability.
        bno_cfg = raw_config.setdefault("bno085", {}) if isinstance(raw_config, dict) else {}
        measurements = bno_cfg.get("axis_map_measurements", {})
        if not isinstance(measurements, dict):
            measurements = {}
        measurements[body_axis] = [
            float(accepted.axis_sensor[0]),
            float(accepted.axis_sensor[1]),
            float(accepted.axis_sensor[2]),
        ]

        result_lines = [
            "IMU single-axis calibration result:",
            f"  axis: {body_axis}",
            f"  angle_rad: {accepted.angle_rad:.3f}",
            f"  axis_sensor: {accepted.axis_sensor.tolist()}",
            f"  expected_axis: {axis_letter}",
            f"  alignment(|dot|): {align:.3f}",
            f"  recommended mapping: {body_axis}={sign}{axis_letter}",
            f"  new axis_map: {axis_map}",
        ]
        for line in result_lines:
            _log_line(line)

        token = _wait_for_token(
            prompt_lines=[
                f"Write bno085.axis_map={axis_map} to {output_path}?",
                "Type y then Enter to write, n then Enter to abort, or r then Enter to redo this axis.",
            ],
            heartbeat_lines=result_lines + ["", f"Proposed axis_map: {axis_map}"],
            valid=["y", "n", "r"],
            quit_tokens=["q", "quit"],
            reprint_every_s=DEFAULT_PROMPT_REPRINT_S,
        )
        if token == "r":
            _log_line("Redo requested: restarting this axis calibration.")
            return calibrate_imu_axis_sign_only(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
                body_axis=body_axis,
            )
        if token != "y":
            _log_line("Not writing config.")
            return

        write_bno_config(raw_config, output_path, axis_map=axis_map, axis_map_measurements=measurements)
        _log_line(f"Wrote bno085.axis_map to {output_path}: {axis_map}")
    finally:
        try:
            imu.close()
        except Exception:
            pass


def main() -> None:
    # Under some launchers (`uv run`, systemd, SSH without a tty), Python may buffer stdout.
    # Make prompts/progress unmissable.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Avoid noisy Blinka warning interleaving with interactive prompts.
    warnings.filterwarnings(
        "ignore",
        message="I2C frequency is not settable in python, ignoring!",
        category=RuntimeWarning,
    )

    examples = """
Examples (copy/paste):
  # Dry-run: show planned moves only (no serial required)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --dry-run

  # Move robot to home pose only (no calibration), then wait until you press 'q' to unload
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --go-home --keyframes-xml assets/v2/keyframes.xml

  # Inspect current pose and optionally record it as home_ctrl_rad (press 'c' to save, 'q' to unload)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --record-pos

        # Interactive calibration mode (per-joint submenu: p/q/a/d/m/r/o/s/z/b/x)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --calibrate

  # Test range of motion for joints interactively
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --range

  # Calibrate IMU upside_down (simple inversion check using gravity vector)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --calibrate-imu

    # Footswitch calibration/test: select which switch signals to display, then press the switches to verify wiring
    uv run python runtime/scripts/calibrate.py --config runtime/configs/runtime_config_v2.json --calibrate-footswitch
""".strip()

    parser = argparse.ArgumentParser(
        description="Interactive servo calibration (servo_offset_unit + servo_unit_direction/motor_unit_direction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument("--h", action="help", help="Show help (alias)")
    parser.add_argument("--?", action="help", help="Show help (alias)")
    parser.add_argument(
        "--config",
        default=None,
        help="Input runtime config path (default: bundle's wildrobot_config.json if --bundle is set, else runtime/configs/runtime_config_v2.json)",
    )
    parser.add_argument("--output", help="Optional output path; default is in-place update with backup")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Interactive calibration mode: select a joint, then use submenu "
            "p(print state)/q(target deg)/a(policy action)/d(direction)/m(motor units)/r(range test)/o(offset)/s(save)."
        ),
    )
    parser.add_argument(
        "--calibrate-imu",
        action="store_true",
        help="Interactive IMU calibration: upside_down (mount inversion) + axis_map (frame remap)",
    )
    parser.add_argument(
        "--calibrate-footswitch",
        action="store_true",
        help="Interactive footswitch test: select signals and print live pressed/open status",
    )
    parser.add_argument(
        "--imu-samples",
        type=int,
        default=DEFAULT_IMU_SAMPLES,
        help="Number of IMU samples to average during calibration",
    )
    parser.add_argument(
        "--range",
        action="store_true",
        help="Range test mode: reset to center, then interactively test each joint's full range of motion",
    )
    parser.add_argument("--step-units", type=int, default=DEFAULT_STEP_UNITS, help="Jog step size in servo units")
    parser.add_argument("--move-ms", type=int, default=DEFAULT_MOVE_MS, help="Move duration in milliseconds")
    parser.add_argument("--bundle", help="Policy bundle directory containing policy_spec.json")
    parser.add_argument("--scene-xml", help="Scene XML path (requires MuJoCo installed)")
    parser.add_argument(
        "--keyframes-xml",
        help="keyframes.xml path for home pose (supports arbitrary actuator count; uses qpos[7:7+num_actuators])",
    )
    parser.add_argument("--go-home", action="store_true", help="Move to home pose before calibration")
    parser.add_argument(
        "--record-pos",
        action="store_true",
        help="Enable pose inspection: 'p' to print positions during calibration, 'c' to record current pose as home_ctrl_rad",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=DEFAULT_PAUSE_S,
        help="Seconds to pause before each commanded move (set 0 to disable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not connect to hardware; print planned moves and config writes only",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args)
    if not config_path.exists():
        sys.exit(f"Config path not found: {config_path}")

    raw_config = json.loads(config_path.read_text())
    config = WrRuntimeConfig.load(config_path)
    servo_cfgs = config.hiwonder_controller.servos
    joint_names = resolve_joint_names(args=args, servo_cfgs=servo_cfgs)

    if not (
        args.calibrate
        or args.calibrate_imu
        or args.calibrate_footswitch
        or args.range
        or args.go_home
        or args.record_pos
        or args.dry_run
    ):
        parser.error(
            "Must specify a mode: --calibrate, --calibrate-imu, --calibrate-footswitch, --range, --go-home, or --record-pos"
        )

    if args.calibrate_imu:
        output_path = Path(args.output) if args.output else config_path
        print("\n== IMU Calibration ==")
        print("Choose one step to run (keeps output minimal and avoids long interactive sessions).", flush=True)
        print(
            "\nOptions:\n"
            "  1) Calibrate upside_down (mount inversion)\n"
            "  2) Calibrate axis_map signs (resumable: body_z -> body_y -> body_x, then write)\n"
            "  3) Calibrate body_z sign only (yaw-left)\n"
            "  4) Calibrate body_y sign only (pitch-up)\n"
            "  5) Calibrate body_x sign only (roll-right)\n"
            "  6) Clear saved axis_map measurements\n"
            "  q) Quit\n",
            flush=True,
        )
        print("Enter choice:", flush=True)
        choice = input().strip().lower()
        if choice.startswith("q"):
            return

        if choice == "1":
            calibrate_imu_upside_down(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                samples=int(args.imu_samples),
            )
            return

        # Reload config (in case upside_down was updated earlier).
        raw_config = json.loads(output_path.read_text())
        config = WrRuntimeConfig.load(output_path)
        upside_down = bool(getattr(config.bno085, "upside_down", False))

        if choice == "6":
            bno_cfg = raw_config.setdefault("bno085", {}) if isinstance(raw_config, dict) else {}
            bno_cfg.pop("axis_map_measurements", None)
            _write_json_with_retries(output_path, raw_config)
            print(f"Cleared saved bno085.axis_map_measurements in {output_path}", flush=True)
            return

        if choice == "2":
            calibrate_imu_axis_map_full(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
            )
            return
        if choice == "3":
            calibrate_imu_axis_sign_only(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
                body_axis="body_z",
            )
            return
        if choice == "4":
            calibrate_imu_axis_sign_only(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
                body_axis="body_y",
            )
            return
        if choice == "5":
            calibrate_imu_axis_sign_only(
                config=config,
                raw_config=raw_config,
                output_path=output_path,
                upside_down=upside_down,
                body_axis="body_x",
            )
            return

        print("Invalid choice.", flush=True)
        return

    if args.calibrate_footswitch:
        try:
            calibrate_footswitches(config=config)
        except KeyboardInterrupt:
            print("\nInterrupted.", flush=True)
        return

    hints = dict(POSITIVE_HINTS)

    states: Dict[str, JointState] = {
        j: normalize_joint_state(offset=servo_cfgs[j].offset, motor_sign=servo_cfgs[j].motor_sign)
        for j in joint_names
    }
    servo_ids = [servo_cfgs[j].id for j in joint_names]
    bundle_dir = Path(args.bundle) if args.bundle else None
    robot_config_path = resolve_robot_config_path(
        raw_config,
        config_path,
        bundle_dir=bundle_dir,
    )
    robot_xml_path = resolve_robot_xml_path(
        robot_config_path=robot_config_path,
        config=config,
        bundle_dir=bundle_dir,
    )
    axis_metadata = load_joint_axis_metadata(
        robot_config_path=robot_config_path,
        robot_xml_path=robot_xml_path,
    )
    policy_setup = load_policy_action_setup(args=args, config=config, joint_names=joint_names)

    if args.dry_run:
        print("DRY RUN: not connecting to hardware.")
        if args.go_home:
            home_ctrl = resolve_home_ctrl(args, joint_names)
            print("Home ctrl (rad):")
            for joint, rad in zip(joint_names, home_ctrl, strict=True):
                servo = servo_cfgs[joint]
                state = states[joint]
                units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                    rad,
                    motor_sign=state.motor_sign,
                    offset=state.offset,
                )
                print(f"  {joint}: rad={rad:+.4f} -> servo_id={servo.id} units={units}")
        if args.calibrate:
            print("Calibration mode (dry run):")
            print(f"  Axis metadata: robot_config={robot_config_path}, robot_xml={robot_xml_path}")
            print(f"  Policy action setup: {policy_setup.source}")
            print("  Available joints:")
            for joint in joint_names:
                servo = servo_cfgs[joint]
                state = states[joint]
                hint = hints.get(joint, "positive motion")
                axis = axis_metadata.get(joint)
                local_axis = _axis_summary(axis.local_axis) if axis is not None else "n/a"
                init_world_axis = _axis_summary(axis.init_world_axis) if axis is not None else "n/a"
                base_rad = float(policy_setup.base_rad_by_joint.get(joint, 0.0))
                scale_rad = float(policy_setup.residual_scale_by_joint.get(joint, 0.0))
                print(
                    f"    #{servo.id}: {joint} (servo_id={servo.id}, servo_offset_unit={state.offset}, "
                    f"servo_unit_direction/motor_unit_direction={state.motor_sign}, "
                    f"joint_angle_at_zero_unit_deg={float(servo.joint_angle_at_zero_unit_deg):+.3g}, "
                    f"local_axis={local_axis}, init_world_axis={init_world_axis}, "
                    f"policy_base_rad={base_rad:+.4f}, policy_scale_rad={scale_rad:.4f}, hint='{hint}')"
                )
        if args.range:
            print("Range test mode (dry run):")
            print("  Available joints:")
            for joint in joint_names:
                servo = servo_cfgs[joint]
                state = states[joint]
                min_rad, max_rad = servo.rad_range
                min_deg = float(np.rad2deg(float(min_rad)))
                max_deg = float(np.rad2deg(float(max_rad)))
                center_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                    0.0,
                    motor_sign=state.motor_sign,
                    offset=state.offset,
                )
                min_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                    min_rad,
                    motor_sign=state.motor_sign,
                    offset=state.offset,
                )
                max_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                    max_rad,
                    motor_sign=state.motor_sign,
                    offset=state.offset,
                )
                print(
                    f"    #{int(servo.id)}: {joint}: deg range: {min_deg:.1f}, {max_deg:.1f} "
                    f"(center={center_units}, min={min_units} [{min_rad:.3f} rad], max={max_units} [{max_rad:.3f} rad])"
                )
            print(f"  Range test duration: {RANGE_TEST_MS}ms per segment (min->max->min->center)")
        return

    from wr_runtime.hardware.hiwonder_board_controller import HiwonderBoardController

    controller = HiwonderBoardController(config.hiwonder_controller)
    try:
        if args.go_home:
            home_ctrl = resolve_home_ctrl(args, joint_names)
            if len(home_ctrl) != len(joint_names):
                raise ValueError("home_ctrl_rad length mismatch with joints")
            if yes_no("Move all servos to home pose now?", default=True):
                cmds = []
                for joint, rad in zip(joint_names, home_ctrl):
                    servo = servo_cfgs[joint]
                    state = states[joint]
                    units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                        rad,
                        motor_sign=state.motor_sign,
                        offset=state.offset,
                    )
                    cmds.append((servo.id, units))
                announce_and_pause(
                    f"Step: move all joints to home pose (duration {max(args.move_ms, 800)} ms)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, max(args.move_ms, 800))
                time.sleep(max(args.move_ms, 800) / 1000.0 + 0.2)

        if args.range:
            # Range test mode: reset to center, then interactively test joints
            print("\n-- Range Test Mode --")
            if yes_no("Reset all servos to MuJoCo joint_pos_deg 0 (0.0 rad) before testing?", default=True):
                cmds = []
                for joint in joint_names:
                    servo = servo_cfgs[joint]
                    state = states[joint]
                    units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                        0.0,
                        motor_sign=state.motor_sign,
                        offset=state.offset,
                    )
                    cmds.append((servo.id, units))
                servo_cfg_by_id = {int(servo_cfgs[j].id): servo_cfgs[j] for j in joint_names}
                center_move_ms = _group_move_ms_20deg_per_s(
                    controller,
                    servo_cfg_by_id,
                    cmds,
                    fallback_ms=max(int(args.move_ms), 1000),
                    min_ms=1000,
                )
                announce_and_pause(
                    f"Step: move all joints to MuJoCo joint_pos_deg 0 (0.0 rad, duration {center_move_ms} ms)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, int(center_move_ms))
                time.sleep(int(center_move_ms) / 1000.0 + 0.1)

            while True:
                print("\n" + "-" * 40)
                try:
                    range_selected = prompt_select_joints_by_servo_id(joint_names, servo_cfgs=servo_cfgs)
                except ValueError as exc:
                    print(str(exc))
                    continue

                if not range_selected:
                    print("Exiting range test.")
                    break

                for joint in range_selected:
                    if joint not in servo_cfgs:
                        print(f"Joint {joint} not found; skipping.")
                        continue
                    servo = servo_cfgs[joint]
                    range_test_joint(controller, servo, joint, state=states[joint])

            # Note: unreachable due to infinite loop above; user exits via Ctrl+C or 'q'

        if args.record_pos and not args.calibrate:
            print(
                "\n-- Pose Inspection --\n"
                "Adjust the robot physically/joints as needed.\n"
                "Commands:\n"
                "  p = print all joint positions (units + ctrl_rad)\n"
                "  c = record current pose as home_ctrl_rad\n"
                f"  {PANIC_KEY} = unload and exit\n"
            )
            while True:
                cmd = input("(p/c/q) > ").strip().lower()
                if cmd == PANIC_KEY:
                    panic_and_exit(controller, servo_ids)
                if cmd == "p":
                    print_all_joint_positions(
                        controller,
                        joint_names=joint_names,
                        servo_cfgs=servo_cfgs,
                        states=states,
                        include_rad=True,
                    )
                    continue
                if cmd == "c":
                    home_ctrl = read_all_home_ctrl_rad(
                        controller,
                        joint_names=joint_names,
                        servo_cfgs=servo_cfgs,
                        states=states,
                    )
                    output_path = Path(args.output) if args.output else config_path
                    write_config(raw_config, output_path, {}, home_ctrl_rad=home_ctrl)
                    print(f"Wrote home_ctrl_rad to {output_path}")
                    continue
                print("Unknown input; use p, c, or q.")

        if args.go_home and not args.calibrate and not args.record_pos:
            wait_until_unload(
                controller,
                servo_ids,
                prompt=(
                    "Home pose set. Servos remain loaded.\n"
                    f"Press '{PANIC_KEY}' then Enter to unload and exit (Ctrl+C also unloads)."
                ),
            )

        if args.calibrate:
            # Unified calibration mode
            print("\n-- Calibration Mode --")
            print("Flow: center all joints → select joint → choose action → calibrate → repeat")
            print(
                "Per-joint submenu:\n"
                "  p=print state, q=target deg evaluator, a=policy action evaluator,\n"
                "  d=calibrate servo_unit_direction, r=single-joint range test (20 deg/s),\n"
                "  m=set motor electric unit, o=calibrate offset, s=save to config, b=back to joint list, x=panic unload"
            )

            # Step 0: Move all joints to MuJoCo joint position 0 deg (0.0 rad).
            if yes_no("Move all joints to MuJoCo joint_pos_deg 0 (0.0 rad) before calibrating?", default=True):
                cmds = []
                for joint in joint_names:
                    servo = servo_cfgs[joint]
                    state = states[joint]
                    units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                        0.0,
                        motor_sign=state.motor_sign,
                        offset=state.offset,
                    )
                    cmds.append((servo.id, units))
                servo_cfg_by_id = {int(servo_cfgs[j].id): servo_cfgs[j] for j in joint_names}
                center_move_ms = _group_move_ms_20deg_per_s(
                    controller,
                    servo_cfg_by_id,
                    cmds,
                    fallback_ms=max(int(args.move_ms), 1000),
                    min_ms=1000,
                )
                announce_and_pause(
                    f"Step: move all joints to MuJoCo joint_pos_deg 0 (duration {center_move_ms} ms, 20 deg/s, min 1s)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, int(center_move_ms))
                time.sleep(int(center_move_ms) / 1000.0 + 0.2)

            calibrated: Dict[str, JointState] = {}
            save_on_exit = False

            def _save_calibration_updates(updates_to_save: Dict[str, JointState]) -> None:
                output_path = Path(args.output) if args.output else config_path
                write_config(raw_config, output_path, updates_to_save)
                print(f"Saved calibration to {output_path}")
                print("Saved joint calibration values:")
                for saved_joint in sorted(updates_to_save.keys()):
                    saved_state = updates_to_save[saved_joint]
                    print(
                        f"  {saved_joint}: offset={int(saved_state.offset):+d}, "
                        f"servo_unit_direction/motor_unit_direction={int(saved_state.motor_sign):+d}"
                    )

            while True:
                # Step 1: List joints with current status
                print("\n" + "=" * 50)
                print("Available joints:")
                servo_id_to_joint: Dict[int, str] = {}
                for joint in joint_names:
                    state = states[joint]
                    servo = servo_cfgs[joint]
                    servo_id_to_joint[servo.id] = joint
                    print(
                        f"  #{servo.id}: {joint} (id={servo.id}, servo_unit_direction/motor_unit_direction={state.motor_sign:+d}, "
                        f"servo_offset_unit={state.offset:+d}, joint_angle_at_zero_unit_deg={float(servo.joint_angle_at_zero_unit_deg):+.3g})"
                    )
                print("\n  q = quit (discard changes)")
                print("  s = save and quit")

                # Step 2: User selects joint
                raw = input("\nSelect servo # (or 'q' to quit, 's' to save+quit): ").strip().lower()
                if raw == "q" or raw == PANIC_KEY:
                    save_on_exit = False
                    break
                if raw == "s":
                    save_on_exit = True
                    break

                try:
                    servo_id = int(raw.lstrip("#"))
                    joint = servo_id_to_joint.get(servo_id)
                    if joint is None:
                        valid_servo_ids = ", ".join(str(servo_cfgs[j].id) for j in joint_names)
                        print(f"Invalid servo ID. Enter one of: {valid_servo_ids}.")
                        continue
                except ValueError:
                    print("Invalid input. Enter a number, 'q', or 's'.")
                    continue

                servo = servo_cfgs[joint]
                state = states[joint]
                hint = hints.get(joint, "positive motion")

                print(f"\nSelected: {joint}")
                print(f"  Hint (positive MuJoCo rad): {hint}")

                while True:
                    current_state = states[joint]
                    print("\nJoint menu:")
                    print("  p = print state")
                    print("  q = evaluate direct target joint deg")
                    print("  a = evaluate policy action residual")
                    print("  d = calibrate servo_unit_direction")
                    print("  m = set motor electric unit")
                    print("  r = run range test for this joint")
                    print("  o = calibrate offset")
                    print("  s = save calibration to config file")
                    print("  z = move to joint_pos_deg 0 (MuJoCo 0 deg)")
                    print("  b = back to joint list")
                    print("  x = panic unload and exit")

                    action = input("Choose action (p/q/a/d/m/r/o/s/z/b/x): ").strip().lower()

                    if action == "x":
                        panic_and_exit(controller, servo_ids)

                    if action == "b" or not action:
                        break

                    if action == "p":
                        print_joint_calibration_state(
                            controller,
                            joint=joint,
                            servo=servo,
                            state=current_state,
                            axis_metadata=axis_metadata,
                            policy_setup=policy_setup,
                            hint=hint,
                        )
                        continue

                    if action == "q":
                        raw_deg = input("Enter target joint deg: ").strip()
                        try:
                            target_deg = float(raw_deg)
                        except ValueError:
                            print("Invalid number.")
                            continue

                        target_rad = float(np.deg2rad(target_deg))
                        target_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                            target_rad,
                            motor_sign=int(current_state.motor_sign),
                            offset=int(current_state.offset),
                        )
                        conceptual_units = int(target_units) - int(current_state.offset)
                        print(f"  target_joint_deg: {target_deg:+.4f}")
                        print(f"  target_joint_rad: {target_rad:+.6f}")
                        print(f"  motor_elect_unit: {int(target_units)}")
                        print(f"  motor_conceptual_unit: {int(conceptual_units)}")

                        if yes_no("Move motor to this target?", default=True):
                            current_units = read_position(controller, int(servo.id))
                            if current_units is not None:
                                current_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                                    int(current_units),
                                    motor_sign=int(current_state.motor_sign),
                                    offset=int(current_state.offset),
                                )
                                current_deg = float(np.rad2deg(float(current_rad)))
                                move_ms = _move_ms_for_speed_20deg_per_s(current_deg, target_deg)
                            else:
                                move_ms = int(max(args.move_ms, 100))
                                print("Current position read failed; using fallback move time.")
                            move_and_wait(controller, int(servo.id), int(target_units), int(move_ms))
                            pos = read_position(controller, int(servo.id))
                            if pos is not None:
                                print(f"Moved. Readback elect unit={int(pos)} conceptual={int(pos) - int(current_state.offset)}")
                        continue

                    if action == "a":
                        evaluate_policy_action(
                            controller,
                            joint=joint,
                            servo=servo,
                            state=current_state,
                            policy_setup=policy_setup,
                            move_ms_fallback=int(args.move_ms),
                        )
                        continue

                    if action == "d":
                        new_direction = calibrate_motor_sign(
                            controller,
                            servo,
                            joint,
                            states[joint],
                            hint,
                            axis_metadata.get(joint),
                            all_servo_ids=servo_ids,
                            move_ms=args.move_ms,
                            pause_s=float(args.pause_s),
                        )
                        states[joint].motor_sign = int(new_direction)
                        calibrated[joint] = states[joint]
                        print(
                            f"Updated in-memory servo_unit_direction / motor_unit_direction for {joint}: "
                            f"{int(new_direction):+d}. Use 's' to write it to config."
                        )
                        continue

                    if action == "m":
                        raw_units = input("Enter target motor electric unit [0..1000]: ").strip()
                        try:
                            target_units = int(raw_units)
                        except ValueError:
                            print("Invalid integer.")
                            continue
                        if target_units < int(ServoConfig.UNITS_MIN) or target_units > int(ServoConfig.UNITS_MAX):
                            print(
                                f"Out of range. Enter a value in [{int(ServoConfig.UNITS_MIN)}..{int(ServoConfig.UNITS_MAX)}]."
                            )
                            continue

                        conceptual_units = int(target_units) - int(current_state.offset)
                        target_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                            int(target_units),
                            motor_sign=int(current_state.motor_sign),
                            offset=int(current_state.offset),
                        )
                        target_deg = float(np.rad2deg(float(target_rad)))
                        print(f"  motor_elect_unit: {int(target_units)}")
                        print(f"  motor_conceptual_unit: {int(conceptual_units)}")
                        print(f"  target_joint_deg: {target_deg:+.4f}")
                        print(f"  target_joint_rad: {float(target_rad):+.6f}")

                        if yes_no("Move motor to this target?", default=True):
                            current_units = read_position(controller, int(servo.id))
                            if current_units is not None:
                                current_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                                    int(current_units),
                                    motor_sign=int(current_state.motor_sign),
                                    offset=int(current_state.offset),
                                )
                                current_deg = float(np.rad2deg(float(current_rad)))
                                move_ms = max(1000, _move_ms_for_speed_20deg_per_s(current_deg, target_deg))
                            else:
                                move_ms = 1000
                                print("Current position read failed; using fallback move time.")
                            move_and_wait(controller, int(servo.id), int(target_units), int(move_ms))
                            pos = read_position(controller, int(servo.id))
                            if pos is not None:
                                print(
                                    f"Moved. Readback elect unit={int(pos)} conceptual={int(pos) - int(current_state.offset)}"
                                )
                        continue

                    if action == "r":
                        range_test_joint(
                            controller,
                            servo,
                            joint,
                            state=current_state,
                            speed_deg_per_s=20.0,
                        )
                        continue

                    if action == "z":
                        target_joint_pos_deg = 0.0
                        target_rad = float(np.deg2rad(target_joint_pos_deg))
                        target_deg = float(np.rad2deg(target_rad))
                        target_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
                            target_rad,
                            motor_sign=int(current_state.motor_sign),
                            offset=int(current_state.offset),
                        )
                        conceptual_units = int(target_units) - int(current_state.offset)
                        print(f"  target_joint_pos_deg: {target_joint_pos_deg:+.4f}")
                        print(f"  target_joint_rad: {target_rad:+.6f}")
                        print(f"  target_joint_deg: {target_deg:+.4f}")
                        print(f"  motor_elect_unit: {int(target_units)}")
                        print(f"  motor_conceptual_unit: {int(conceptual_units)}")

                        current_units = read_position(controller, int(servo.id))
                        if current_units is not None:
                            current_rad = servo.servo_elect_units_to_joint_target_rad_for_calibrate(
                                int(current_units),
                                motor_sign=int(current_state.motor_sign),
                                offset=int(current_state.offset),
                            )
                            current_deg = float(np.rad2deg(float(current_rad)))
                            move_ms = max(1000, _move_ms_for_speed_20deg_per_s(current_deg, target_deg))
                        else:
                            move_ms = 1000
                            print("Current position read failed; using fallback move time.")

                        move_and_wait(controller, int(servo.id), int(target_units), int(move_ms))
                        pos = read_position(controller, int(servo.id))
                        if pos is not None:
                            print(f"Moved. Readback elect unit={int(pos)} conceptual={int(pos) - int(current_state.offset)}")
                        continue

                    if action == "s":
                        updates_to_save: Dict[str, JointState] = dict(calibrated)
                        updates_to_save[joint] = states[joint]
                        _save_calibration_updates(updates_to_save)
                        calibrated.clear()
                        save_on_exit = False
                        continue

                    if action == "o":
                        state_before_offset = JointState(
                            offset=int(states[joint].offset),
                            motor_sign=int(states[joint].motor_sign),
                        )
                        new_offset, save_requested, quit_joint_without_save = calibrate_offset(
                            controller,
                            servo,
                            joint,
                            states[joint],
                            all_servo_ids=servo_ids,
                            move_ms=args.move_ms,
                            step_units=args.step_units,
                            pause_s=float(args.pause_s),
                            record_pos=bool(args.record_pos),
                            all_joint_names=joint_names,
                            all_servo_cfgs=servo_cfgs,
                            all_states=states,
                        )
                        if quit_joint_without_save:
                            states[joint] = state_before_offset
                            calibrated.pop(joint, None)
                            break
                        if new_offset is not None:
                            states[joint].offset = new_offset
                            verify_zero(
                                controller,
                                servo,
                                joint,
                                states[joint],
                                move_ms=args.move_ms,
                                pause_s=float(args.pause_s),
                            )
                        calibrated[joint] = states[joint]
                        if save_requested:
                            updates_to_save: Dict[str, JointState] = dict(calibrated)
                            updates_to_save[joint] = states[joint]
                            _save_calibration_updates(updates_to_save)
                            calibrated.clear()
                            save_on_exit = False
                            continue

                    if action == "o":
                        continue

                    print("Unknown action. Use p/q/a/d/m/r/o/s/z/b/x.")

                # Loop back to step 1

            # Save calibrated joints only if explicitly requested.
            if not calibrated:
                print("No changes made.")
            elif save_on_exit:
                _save_calibration_updates(calibrated)
            else:
                print("Discarded calibration changes (not saved).")

    finally:
        try:
            try:
                controller.unload_servos(servo_ids)
            except Exception:
                pass
        finally:
            try:
                controller.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
