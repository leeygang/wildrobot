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
from typing import Dict, Iterable, List, Optional

import numpy as np

# Ensure repo root is on sys.path when running as a script (e.g. `python3 runtime/scripts/calibrate.py ...`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if _RUNTIME_ROOT.exists() and str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from runtime.configs.config import ServoConfig, WrRuntimeConfig  # noqa: E402

# Calibration constants (use ServoConfig constants for conversion)
DEFAULT_MOVE_MS = 300
DEFAULT_STEP_UNITS = 5
DEFAULT_DELTA_RAD = 0.5
VERIFY_TOL_RAD = 0.05
PANIC_KEY = "q"
DEFAULT_PAUSE_S = 3.0
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

POSITIVE_HINTS = {
    # These hints describe what POSITIVE MuJoCo radians look like physically.
    # For joints with inverted ranges (mirror_sign=-1), the positive direction
    # is OPPOSITE to the symmetric motion direction.
    #
    # Left leg (mirror_sign=1.0): positive rad = "natural" direction
    "left_hip_pitch": "left leg swings forward",
    "left_hip_roll": "left leg moves inward (toward body midline)",
    "left_knee_pitch": "left knee bends (foot moves backward)",
    "left_ankle_pitch": "left toes go up (dorsiflex)",
    #
    # Right leg: check mirror_sign to determine positive direction
    # - mirror_sign=-1.0 (hip_pitch, hip_roll): positive rad = OPPOSITE of left
    # - mirror_sign=1.0 (knee, ankle): positive rad = SAME as left
    "right_hip_pitch": "right leg swings backward",  # inverted range
    "right_hip_roll": "right leg moves outward (away from body midline)",  # inverted range
    "right_knee_pitch": "right knee bends (foot moves backward)",  # same as left
    "right_ankle_pitch": "right toes go up (dorsiflex)",  # same as left
}


@dataclass
class JointState:
    offset: int
    direction: int


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
            pos_rad = servo.units_to_rad_for_calibrate(units, direction=st.direction, offset=st.offset)
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
        home_ctrl.append(servo.units_to_rad_for_calibrate(units, direction=st.direction, offset=st.offset))
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


def load_home_from_bundle(bundle_dir: Path, joint_count: int) -> List[float]:
    spec_path = bundle_dir / "policy_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"policy_spec.json not found in bundle {bundle_dir}")
    data = json.loads(spec_path.read_text())
    robot = data.get("robot", {})
    home = robot.get("home_ctrl_rad")
    if home is None:
        raise ValueError("bundle policy_spec.json missing robot.home_ctrl_rad")
    if not isinstance(home, list) or len(home) != joint_count:
        raise ValueError("robot.home_ctrl_rad length mismatch")
    return [float(x) for x in home]


def load_home_from_keyframes_xml(path: Path, joint_count: int) -> List[float]:
    if joint_count != 8:
        raise ValueError("keyframes.xml path only supported when actuator count is 8")
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
        raise ValueError("Keyframe qpos shorter than expected")
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
        return load_home_from_bundle(Path(args.bundle), len(joint_names))
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


def direction_prompt(joint: str, hint: str, delta_rad: float) -> str:
    return (
        f"I will command +{delta_rad:.3f} rad on {joint}. Did the joint move toward: {hint}?\n"
        "  y = yes (direction = +1)\n"
        "  n = no  (direction = -1)\n"
        "  r = repeat with a different delta\n"
        f"  {PANIC_KEY} = panic unload"
    )


def calibrate_direction(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    hint: str,
    *,
    all_servo_ids: Iterable[int],
    move_ms: int,
    center_units: int,
    pause_s: float,
) -> int:
    print(f"\n-- Direction calibration for {joint} (servo {servo.id}) --")
    delta_rad_used = DEFAULT_DELTA_RAD

    while True:
        # Always start the direction test from center to make the "first move" unambiguous.
        announce_and_pause(
            f"Step: move {joint} to center units ({center_units})",
            pause_s,
        )
        move_and_wait(controller, servo.id, center_units, move_ms)
        plus_units = servo.rad_to_units_for_calibrate(
            delta_rad_used,
            direction=1,
            offset=state.offset,
        )
        announce_and_pause(
            f"Step: command +delta ({delta_rad_used:.3f} rad) -> units {plus_units} (ignoring existing direction)",
            pause_s,
        )
        move_and_wait(controller, servo.id, plus_units, move_ms)
        resp = input(direction_prompt(joint, hint, delta_rad_used) + "\n> ").strip().lower()
        if resp == PANIC_KEY:
            panic_and_exit(controller, all_servo_ids)
        if resp.startswith("y"):
            print("Direction set to +1")
            announce_and_pause(
                f"Step: return {joint} to center units ({center_units})",
                pause_s,
            )
            move_and_wait(controller, servo.id, center_units, move_ms)
            return 1
        if resp.startswith("n"):
            print("Direction set to -1")
            announce_and_pause(
                f"Step: return {joint} to center units ({center_units})",
                pause_s,
            )
            move_and_wait(controller, servo.id, center_units, move_ms)
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
) -> Optional[int]:
    """Calibrate offset for a joint. Returns new offset, or None to skip (keep original)."""
    print(f"\n-- Offset calibration for {joint} (servo {servo.id}) --")
    # Offset calibration should not depend on any existing offset value.
    target_units = ServoConfig.UNITS_CENTER
    announce_and_pause(
        f"Step: move {joint} to raw center units ({target_units})",
        pause_s,
    )
    move_and_wait(controller, servo.id, target_units, move_ms)
    commands_msg = (
        "Jog the joint until it matches your neutral pose. Commands:"
        "\n  a/d = -/+ step; A/D = -/+ 5x step; c or empty = confirm;"
        "\n  s = skip offset (save direction only); q = quit joint;"
        "\n  m = enter offset manually (units);"
    )
    if record_pos:
        commands_msg += "\n  p = print all joint positions;"
    commands_msg += f"\n  {PANIC_KEY} = panic unload"
    print(commands_msg)
    read_failures = 0
    while True:
        cmd = input("(a/d/A/D/c/s/m/p/q/enter) > ").strip()
        if not cmd or cmd.lower() == "c":
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
                continue
            offset_units = pos - ServoConfig.UNITS_CENTER
            print(f"Captured offset {offset_units} (units), raw position {pos}")
            return offset_units
        if cmd.lower() == "m":
            raw = input("Enter offset in servo units (offset = current_pos - 500): ").strip()
            try:
                offset_units = int(raw)
            except ValueError:
                print("Invalid integer offset.")
                continue
            print(f"Using manual offset {offset_units} (units).")
            return offset_units
        if cmd.lower() == "q":
            print(f"Aborted offset calibration for {joint}; keeping previous offset {state.offset}.")
            return state.offset
        if cmd.lower() == "s":
            print(f"Skipping offset calibration for {joint}; direction will be saved, offset unchanged.")
            return None
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
            print("Unknown command; use a/d/A/D to jog, c to confirm, s to skip.")
            continue
        target_units = max(ServoConfig.UNITS_MIN, min(ServoConfig.UNITS_MAX, target_units + delta))
        move_and_wait(controller, servo.id, target_units, move_ms)
        pos = read_position(controller, servo.id)
        if pos is not None:
            pos_rad = servo.units_to_rad_for_calibrate(pos, direction=state.direction, offset=0)
            print(f"Moved to {pos} units (~{pos_rad:.3f} rad, offset assumed 0).")


def verify_zero(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    move_ms: int,
    pause_s: float,
) -> None:
    target_units = servo.rad_to_units_for_calibrate(
        0.0,
        direction=state.direction,
        offset=state.offset,
    )
    announce_and_pause(
        f"Step: verify {joint} at target_rad=0.0 -> units {target_units}",
        pause_s,
    )
    move_and_wait(controller, servo.id, target_units, move_ms)
    pos = read_position(controller, servo.id)
    if pos is None:
        print("Verification readback failed.")
        return
    pos_rad = servo.units_to_rad_for_calibrate(
        pos,
        direction=state.direction,
        offset=state.offset,
    )
    status = "OK" if abs(pos_rad) <= VERIFY_TOL_RAD else "drift"
    print(f"Verify zero: {pos} units => {pos_rad:.4f} rad ({status})")


def range_test_joint(
    controller,
    servo: ServoConfig,
    joint: str,
) -> None:
    """Run a joint through its full range: center -> min -> max -> min -> center."""
    print(f"\n-- Range test for {joint} (servo {servo.id}) --")

    # Calculate positions using current calibration (direction + offset) and joint limits
    center_units = servo.rad_to_units(0.0)
    min_rad, max_rad = servo.rad_range
    min_units = servo.rad_to_units(min_rad)
    max_units = servo.rad_to_units(max_rad)

    print(f"  Center: {center_units} units (0.0 rad)")
    print(f"  Min: {min_units} units ({min_rad:.3f} rad)")
    print(f"  Max: {max_units} units ({max_rad:.3f} rad)")
    print(f"  Move duration: {RANGE_TEST_MS}ms per segment")


    try:
        # Move to min
        print(f"Moving to min ({min_units} units)...")
        controller.move_servos([(servo.id, min_units)], RANGE_TEST_MS)
        time.sleep(RANGE_TEST_MS / 1000.0 + 0.1)

        # Move to max
        print(f"Moving to max ({max_units} units)...")
        controller.move_servos([(servo.id, max_units)], RANGE_TEST_MS)
        time.sleep(RANGE_TEST_MS / 1000.0 + 0.1)

        # Move back to min
        print(f"Moving back to min ({min_units} units)...")
        controller.move_servos([(servo.id, min_units)], RANGE_TEST_MS)
        time.sleep(RANGE_TEST_MS / 1000.0 + 0.1)

        # Move back to center
        print(f"Moving back to center ({center_units} units)...")
        controller.move_servos([(servo.id, center_units)], RANGE_TEST_MS)
        time.sleep(RANGE_TEST_MS / 1000.0 + 0.1)

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
    hiw = base_data.setdefault("Hiwonder_controller", {})
    servos = hiw.setdefault("servos", {})
    for joint, state in updates.items():
        if joint not in servos:
            servos[joint] = {}
        servos[joint]["offset"] = int(state.offset)
        servos[joint]["direction"] = int(state.direction)
    if home_ctrl_rad is not None:
        hiw["home_ctrl_rad"] = [float(x) for x in home_ctrl_rad]
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
        bno["axis_map"] = [str(x) for x in axis_map]
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
            try:
                imu = BNO085IMU(
                    i2c_address=config.bno085.i2c_address,
                    upside_down=False,
                    sampling_hz=50,
                    polling_mode=True,
                )
            except TypeError:
                # Back-compat with older BNO085IMU implementations.
                imu = BNO085IMU(
                    i2c_address=config.bno085.i2c_address,
                    upside_down=False,
                    sampling_hz=50,
                )
    except TimeoutError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        time.sleep(0.2)
        # Ensure the background reader is producing fresh samples (not stuck on I2C).
        last_ts = float(getattr(imu.read(), "timestamp_s", 0.0))
        start_wait = time.monotonic()
        while True:
            s0 = imu.read()
            ts0 = float(getattr(s0, "timestamp_s", 0.0))
            if ts0 > 0.0 and ts0 != last_ts:
                last_ts = ts0
                break
            if time.monotonic() - start_wait > 3.0:
                raise RuntimeError(
                    "IMU is not producing samples (timestamp_s not updating). "
                    "Check I2C wiring/address, and verify the BNO08X is visible with 'sudo i2cdetect -y 1'."
                )
            time.sleep(0.05)

        g_normal = []
        g_flipped = []
        total = max(1, int(samples))
        stale_since = time.monotonic()
        for i in range(total):
            s = imu.read()
            ts = float(getattr(s, "timestamp_s", 0.0))
            if ts != last_ts and ts > 0.0:
                last_ts = ts
                stale_since = time.monotonic()
            elif time.monotonic() - stale_since > 2.0:
                raise RuntimeError(
                    "IMU sample stream stalled during calibration (timestamp_s stopped updating). "
                    "This often indicates an intermittent I2C bus lock or power issue."
                )

            q = normalize_quat_xyzw(np.asarray(s.quat_xyzw, dtype=np.float32))
            g0 = gravity_local_from_quat(q)
            g1 = gravity_local_from_quat(
                normalize_quat_xyzw(np.asarray(_apply_upside_down_quat(q.tolist()), dtype=np.float32))
            )
            g_normal.append(g0)
            g_flipped.append(g1)
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{total}", flush=True)
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
    """Capture IMU samples for duration, ensuring timestamp_s advances."""
    start = time.monotonic()
    stale_since = time.monotonic()
    samples: List[np.ndarray] = []
    quat_samples: List[np.ndarray] = []
    last_progress = start

    while time.monotonic() - start < duration_s:
        t0 = time.monotonic()
        s = imu.read()
        t1 = time.monotonic()
        read_dt = t1 - t0
        if read_dt > 0.2:
            raise RuntimeError(
                f"IMU read() is unexpectedly slow ({read_dt:.3f}s). "
                "This can indicate an I2C bus lock or a blocked background reader thread."
            )
        ts = float(getattr(s, "timestamp_s", 0.0))
        if ts > 0.0 and ts != last_ts:
            last_ts = ts
            stale_since = time.monotonic()
        elif time.monotonic() - stale_since > stall_timeout_s:
            raise RuntimeError(
                "IMU sample stream stalled during gyro capture (timestamp_s stopped updating). "
                "Check I2C stability/power."
            )

        samples.append(np.asarray(s.gyro_rad_s, dtype=np.float32))
        quat_samples.append(np.asarray(s.quat_xyzw, dtype=np.float32))
        now = time.monotonic()
        if progress_every_s > 0 and now - last_progress >= progress_every_s:
            elapsed = now - start
            remaining = max(0.0, duration_s - elapsed)
            _log_line(
                f"{label}: captured {len(samples)} samples (t={elapsed:.1f}/{duration_s:.1f}s, remaining={remaining:.1f}s)"
            )
            last_progress = now
        time.sleep(sample_dt_s)

    if not samples:
        if progress_every_s > 0:
            print("  ... capture complete: 0 samples", flush=True)
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            last_ts,
        )
    if progress_every_s > 0:
        elapsed = time.monotonic() - start
        _log_line(f"{label}: capture complete: {len(samples)} samples (t={elapsed:.1f}/{duration_s:.1f}s)")
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
    last_ts = float(getattr(imu.read(), "timestamp_s", 0.0))
    while time.monotonic() - start < timeout_s:
        s = imu.read()
        ts = float(getattr(s, "timestamp_s", 0.0))
        if ts > 0.0 and ts != last_ts:
            return ts
        time.sleep(0.01)
    raise RuntimeError(
        "IMU stream did not produce fresh samples (timestamp_s not advancing). "
        "Check I2C wiring/address and ensure the IMU is powered."
    )


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
    _, baseline_quat, last_ts = _capture_imu_series(
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
    _bell()
    _bell()
    if action_hint:
        _log_line(_yellow(f"{body_axis_name}: ROTATE NOW — {action_hint} ({motion_s:.1f}s)"))
    else:
        _log_line(_yellow(f"{body_axis_name}: ROTATE NOW (motion capture {motion_s:.1f}s)"))
    _log_line(_yellow(f"{body_axis_name}: motion capture started"))
    _, motion_quat, last_ts = _capture_imu_series(
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
    if motion_quat.shape[0] == 0:
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
            q_rel = rel[i_max].astype(np.float32)
            v = q_rel[:3].astype(np.float32)
            v_norm = float(np.linalg.norm(v))
            if not np.isfinite(v_norm) or v_norm < 1e-6:
                _log_line(
                    _red(
                        f"{body_axis_name}: could not infer a dominant axis (rotation vector near zero). "
                        "Try a larger single-direction rotation."
                    )
                )
                return None

            axis = (v / v_norm).astype(np.float32)
            angle_rad = float(ang[i_max])
    except TimeoutError as exc:
        _log_line(_red(str(exc)))
        return None
    except BaseException as exc:
        _log_line(_red(f"{body_axis_name}: axis inference failed: {type(exc).__name__}: {exc}"))
        _log_line(_red(traceback.format_exc()))
        return None

    compute_dt = time.monotonic() - compute_t0
    _log_line(f"{body_axis_name}: axis inference time {compute_dt:.3f}s")
    _log_line(f"{body_axis_name}: max_relative_rotation_angle_rad={angle_rad:.3f}")
    _log_line(f"{body_axis_name}: relative_rotation_axis_sensor={axis.tolist()}")
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
            imu = BNO085IMU(
                i2c_address=config.bno085.i2c_address,
                upside_down=upside_down,
                sampling_hz=50,
            )
    except TimeoutError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        time.sleep(0.2)
        _log_line(
            "IMU init: "
            f"polling_mode={getattr(imu, 'polling_mode', None)} "
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
    try:
        from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    except Exception as exc:
        raise RuntimeError(
            "IMU axis calibration requires runtime IMU deps on the target (Adafruit Blinka + BNO08X)."
        ) from exc

    def _init_imu():
        try:
            return BNO085IMU(
                i2c_address=config.bno085.i2c_address,
                upside_down=upside_down,
                sampling_hz=50,
                polling_mode=True,
            )
        except TypeError:
            return BNO085IMU(
                i2c_address=config.bno085.i2c_address,
                upside_down=upside_down,
                sampling_hz=50,
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
        time.sleep(0.2)
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

        measurements: dict[str, list[float]] = {}
        axes_by_body: dict[str, np.ndarray] = {}
        angles_by_body: dict[str, float] = {}
        align_by_body: dict[str, float] = {}
        axis_map: list[str] = ["+X", "+Y", "+Z"]

        for step in ("body_z", "body_y", "body_x"):
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
            _log_line(f"{step}: accepted angle_rad={angles_by_body[step]:.3f} axis_sensor={axes_by_body[step].tolist()}")
            if step != "body_x":
                _pause(label=f"{step}: reposition for next axis", seconds=3.0)
        avg_align = float(np.mean([align_by_body[k] for k in ("body_x", "body_y", "body_z")]))

        summary = (
            "IMU axis_map calibration summary:\n"
            f"  body_z: angle_rad={angles_by_body['body_z']:.3f} axis_sensor={axes_by_body['body_z'].tolist()}\n"
            f"  body_y: angle_rad={angles_by_body['body_y']:.3f} axis_sensor={axes_by_body['body_y'].tolist()}\n"
            f"  body_x: angle_rad={angles_by_body['body_x']:.3f} axis_sensor={axes_by_body['body_x'].tolist()}\n"
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
        try:
            return BNO085IMU(
                i2c_address=config.bno085.i2c_address,
                upside_down=upside_down,
                sampling_hz=50,
                polling_mode=True,
            )
        except TypeError:
            return BNO085IMU(
                i2c_address=config.bno085.i2c_address,
                upside_down=upside_down,
                sampling_hz=50,
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
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --dry-run

  # Move robot to home pose only (no calibration), then wait until you press 'q' to unload
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --go-home --keyframes-xml assets/keyframes.xml

  # Inspect current pose and optionally record it as home_ctrl_rad (press 'c' to save, 'q' to unload)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --record-pos

  # Interactive calibration mode (select joints and choose direction or offset calibration)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --calibrate

  # Test range of motion for joints interactively
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --range

  # Calibrate IMU upside_down (simple inversion check using gravity vector)
  uv run python runtime/scripts/calibrate.py --config runtime/configs/wr_runtime_config.json --calibrate-imu
""".strip()

    parser = argparse.ArgumentParser(
        description="Interactive servo calibration (offset + direction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument("--h", action="help", help="Show help (alias)")
    parser.add_argument("--?", action="help", help="Show help (alias)")
    parser.add_argument(
        "--config",
        default="runtime/configs/wr_runtime_config.json",
        help="Input runtime config path",
    )
    parser.add_argument("--output", help="Optional output path; default is in-place update with backup")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Interactive calibration mode: select joints and choose direction or offset calibration",
    )
    parser.add_argument(
        "--calibrate-imu",
        action="store_true",
        help="Interactive IMU calibration: upside_down (mount inversion) + axis_map (frame remap)",
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
    parser.add_argument("--keyframes-xml", help="keyframes.xml path for home pose (actuator count must be 8)")
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

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config path not found: {config_path}")

    raw_config = json.loads(config_path.read_text())
    config = WrRuntimeConfig.load(config_path)
    servo_cfgs = config.hiwonder_controller.servos
    joint_names = list(servo_cfgs.keys())

    if not (args.calibrate or args.calibrate_imu or args.range or args.go_home or args.record_pos or args.dry_run):
        parser.error("Must specify a mode: --calibrate, --range, --go-home, or --record-pos")

    if args.calibrate_imu:
        output_path = Path(args.output) if args.output else config_path
        print("\n== IMU Calibration ==")
        print("Choose one step to run (keeps output minimal and avoids long interactive sessions).", flush=True)
        print(
            "\nOptions:\n"
            "  1) Calibrate upside_down (mount inversion)\n"
            "  2) Calibrate axis_map signs (body_z -> body_y -> body_x, then write)\n"
            "  3) Calibrate body_z sign only (yaw-left)\n"
            "  4) Calibrate body_y sign only (pitch-up)\n"
            "  5) Calibrate body_x sign only (roll-right)\n"
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

    hints = dict(POSITIVE_HINTS)

    states: Dict[str, JointState] = {
        j: JointState(offset=servo_cfgs[j].offset, direction=servo_cfgs[j].direction)
        for j in joint_names
    }
    servo_ids = [servo_cfgs[j].id for j in joint_names]

    if args.dry_run:
        print("DRY RUN: not connecting to hardware.")
        if args.go_home:
            home_ctrl = resolve_home_ctrl(args, joint_names)
            print("Home ctrl (rad):")
            for joint, rad in zip(joint_names, home_ctrl, strict=True):
                servo = servo_cfgs[joint]
                state = states[joint]
                units = servo.rad_to_units_for_calibrate(
                    rad,
                    direction=state.direction,
                    offset=state.offset,
                )
                print(f"  {joint}: rad={rad:+.4f} -> servo_id={servo.id} units={units}")
        if args.calibrate:
            print("Calibration mode (dry run):")
            print("  Available joints:")
            for idx, joint in enumerate(joint_names, start=1):
                servo = servo_cfgs[joint]
                state = states[joint]
                hint = hints.get(joint, "positive motion")
                print(f"    #{idx}: {joint} (servo_id={servo.id}, offset={state.offset}, direction={state.direction}, hint='{hint}')")
        if args.range:
            print("Range test mode (dry run):")
            print("  Available joints:")
            for idx, joint in enumerate(joint_names, start=1):
                servo = servo_cfgs[joint]
                min_rad, max_rad = servo.rad_range
                center_units = servo.rad_to_units(0.0)
                min_units = servo.rad_to_units(min_rad)
                max_units = servo.rad_to_units(max_rad)
                print(f"    #{idx}: {joint} (servo_id={servo.id}, center={center_units}, min={min_units} [{min_rad:.3f} rad], max={max_units} [{max_rad:.3f} rad])")
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
                    units = servo.rad_to_units_for_calibrate(
                        rad,
                        direction=state.direction,
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
            if yes_no("Reset all servos to adjusted center (0.0 rad) before testing?", default=True):
                cmds = [(servo_cfgs[j].id, servo_cfgs[j].rad_to_units(0.0)) for j in joint_names]
                announce_and_pause(
                    "Step: move all joints to adjusted center (0.0 rad, duration 800 ms)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, 800)
                time.sleep(0.9)

            while True:
                print("\n" + "-" * 40)
                try:
                    range_selected = prompt_select_joints(joint_names)
                except ValueError as exc:
                    print(str(exc))
                    continue

                for joint in range_selected:
                    if joint not in servo_cfgs:
                        print(f"Joint {joint} not found; skipping.")
                        continue
                    servo = servo_cfgs[joint]
                    range_test_joint(controller, servo, joint)

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

            # Step 0: Move all joints to adjusted center (0.0 rad with current offsets)
            if yes_no("Move all joints to adjusted center (0.0 rad) before calibrating?", default=True):
                cmds = []
                for joint in joint_names:
                    servo = servo_cfgs[joint]
                    state = states[joint]
                    units = servo.rad_to_units_for_calibrate(
                        0.0,
                        direction=state.direction,
                        offset=state.offset,
                    )
                    cmds.append((servo.id, units))
                announce_and_pause(
                    f"Step: move all joints to adjusted center (duration {max(args.move_ms, 800)} ms)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, max(args.move_ms, 800))
                time.sleep(max(args.move_ms, 800) / 1000.0 + 0.2)

            calibrated: Dict[str, JointState] = {}

            while True:
                # Step 1: List joints with current status
                print("\n" + "=" * 50)
                print("Available joints:")
                for idx, joint in enumerate(joint_names, start=1):
                    state = states[joint]
                    servo = servo_cfgs[joint]
                    hint = hints.get(joint, "")
                    print(f"  #{idx}: {joint} (id={servo.id}, dir={state.direction:+d}, offset={state.offset:+d})")
                print(f"\n  q = quit and save")

                # Step 2: User selects joint
                raw = input("\nSelect joint # (or 'q' to quit): ").strip().lower()
                if raw == PANIC_KEY or raw == "q":
                    break

                try:
                    idx = int(raw.lstrip("#"))
                    if idx < 1 or idx > len(joint_names):
                        print(f"Invalid index. Enter 1-{len(joint_names)}.")
                        continue
                    joint = joint_names[idx - 1]
                except ValueError:
                    print("Invalid input. Enter a number or 'q'.")
                    continue

                servo = servo_cfgs[joint]
                state = states[joint]
                hint = hints.get(joint, "positive motion")

                # Step 3: Ask which action to take
                print(f"\nSelected: {joint}")
                print(f"  Current: direction={state.direction:+d}, offset={state.offset:+d}")
                print(f"  Hint (positive MuJoCo rad): {hint}")
                print("\nAction:")
                print("  d = calibrate direction")
                print("  o = calibrate offset")
                print("  b = both (direction first, then offset)")
                print("  s = skip (go back to list)")

                action = input("Choose action (d/o/b/s): ").strip().lower()

                if action == "s" or not action:
                    continue

                # Step 4: Perform calibration
                center_units = servo.rad_to_units_for_calibrate(0.0, direction=1, offset=state.offset)

                if action in ("d", "b"):
                    new_dir = calibrate_direction(
                        controller,
                        servo,
                        joint,
                        state,
                        hint,
                        all_servo_ids=servo_ids,
                        move_ms=args.move_ms,
                        center_units=center_units,
                        pause_s=float(args.pause_s),
                    )
                    states[joint].direction = new_dir
                    calibrated[joint] = states[joint]

                if action in ("o", "b"):
                    new_offset = calibrate_offset(
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
                    if new_offset is not None:
                        states[joint].offset = new_offset
                        verify_zero(controller, servo, joint, states[joint], move_ms=args.move_ms, pause_s=float(args.pause_s))
                    calibrated[joint] = states[joint]

                # Loop back to step 1

            # Save calibrated joints
            if calibrated:
                output_path = Path(args.output) if args.output else config_path
                write_config(raw_config, output_path, calibrated)
                print(f"Wrote updated calibration to {output_path}")
            else:
                print("No changes made.")

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
