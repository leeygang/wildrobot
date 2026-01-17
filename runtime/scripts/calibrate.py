#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base_data, indent=2))


def main() -> None:
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

    if not (args.calibrate or args.range or args.go_home or args.record_pos or args.dry_run):
        parser.error("Must specify a mode: --calibrate, --range, --go-home, or --record-pos")

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
