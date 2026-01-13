#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Ensure repo root is on sys.path when running as a script (e.g. `python3 runtime/scripts/calibrate.py ...`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.configs.config import ServoConfig, WrRuntimeConfig  # noqa: E402

RANGE_RAD = 4.1887902047863905
UNITS_MIN = 0
UNITS_MAX = 1000
UNITS_CENTER = 500
UNITS_PER_RAD = UNITS_MAX / RANGE_RAD
DEFAULT_MOVE_MS = 300
DEFAULT_STEP_UNITS = 5
DEFAULT_DELTA_RAD = 0.5
VERIFY_TOL_RAD = 0.05
PANIC_KEY = "q"
DEFAULT_PAUSE_S = 3.0

POSITIVE_HINTS = {
    "left_hip_pitch": "left leg swings forward",
    "right_hip_pitch": "right leg swings forward",
    "left_knee_pitch": "left knee bends (foot moves backward toward the body)",
    "right_knee_pitch": "right knee bends (foot moves backward toward the body)",
    "left_ankle_pitch": "left toes go up (dorsiflex)",
    "right_ankle_pitch": "right toes go up (dorsiflex)",
    "left_hip_roll": "left leg moves outward (to the left, away from body midline)",
    "right_hip_roll": "right leg moves outward (to the right, away from body midline)",
}


@dataclass
class JointState:
    offset: int
    direction: int


def clamp_units(value: float) -> int:
    return int(max(UNITS_MIN, min(UNITS_MAX, round(value))))


def parse_hint_override(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Hint override must be in the form joint=hint")
    joint, hint = value.split("=", 1)
    joint = joint.strip()
    if not joint:
        raise argparse.ArgumentTypeError("Joint name in hint override is empty")
    return joint, hint.strip()


def rad_to_units(
    servo: ServoConfig,
    target_rad: float,
    *,
    direction_override: Optional[int] = None,
    offset_override: Optional[int] = None,
) -> int:
    direction = direction_override if direction_override is not None else servo.direction
    offset = offset_override if offset_override is not None else servo.offset
    units = UNITS_CENTER + float(direction) * target_rad * UNITS_PER_RAD + offset
    return clamp_units(units)


def units_to_rad(
    servo: ServoConfig,
    units: int,
    *,
    direction_override: Optional[int] = None,
    offset_override: Optional[int] = None,
) -> float:
    direction = direction_override if direction_override is not None else servo.direction
    offset = offset_override if offset_override is not None else servo.offset
    return float(direction) * (units - UNITS_CENTER - offset) * (RANGE_RAD / UNITS_MAX)


def read_position(controller, servo_id: int) -> Optional[int]:
    resp = controller.read_servo_positions([servo_id])
    if not resp:
        return None
    return resp[0][1]


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
    delta_rad: float,
    delta_units: Optional[int],
    move_ms: int,
    center_units: int,
    pause_s: float,
) -> int:
    print(f"\n-- Direction calibration for {joint} (servo {servo.id}) --")
    delta_rad_used = (
        float(delta_units) / UNITS_PER_RAD if delta_units is not None else float(delta_rad)
    )
    if delta_rad_used == 0:
        delta_rad_used = DEFAULT_DELTA_RAD

    while True:
        # Always start the direction test from center to make the "first move" unambiguous.
        announce_and_pause(
            f"Step: move {joint} to center units ({center_units})",
            pause_s,
        )
        move_and_wait(controller, servo.id, center_units, move_ms)
        plus_units = rad_to_units(
            servo,
            delta_rad_used,
            direction_override=1,
            offset_override=state.offset,
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
) -> int:
    print(f"\n-- Offset calibration for {joint} (servo {servo.id}) --")
    target_units = rad_to_units(
        servo,
        0.0,
        direction_override=state.direction,
        offset_override=state.offset,
    )
    announce_and_pause(
        f"Step: move {joint} to target_rad=0.0 using calibrated direction/offset -> units {target_units}",
        pause_s,
    )
    move_and_wait(controller, servo.id, target_units, move_ms)
    print(
        "Jog the joint until it matches your neutral pose. Commands:"
        "\n  a/d = -/+ step; A/D = -/+ 5x step; c or empty = confirm;"
        f" {PANIC_KEY} = panic unload"
    )
    while True:
        cmd = input("(a/d/A/D/c/enter/panic) > ").strip()
        if not cmd or cmd.lower() == "c":
            pos = read_position(controller, servo.id)
            if pos is None:
                print("Failed to read position; retrying.")
                continue
            offset_units = pos - UNITS_CENTER
            print(f"Captured offset {offset_units} (units), raw position {pos}")
            return offset_units
        if cmd == PANIC_KEY:
            panic_and_exit(controller, all_servo_ids)
        delta = 0
        if cmd in ("a", "A"):
            delta = -step_units * (5 if cmd == "A" else 1)
        elif cmd in ("d", "D"):
            delta = step_units * (5 if cmd == "D" else 1)
        else:
            print("Unknown command; use a/d/A/D or confirm.")
            continue
        target_units = clamp_units(target_units + delta)
        move_and_wait(controller, servo.id, target_units, move_ms)
        pos = read_position(controller, servo.id)
        if pos is not None:
            pos_rad = units_to_rad(
                servo,
                pos,
                direction_override=state.direction,
                offset_override=state.offset,
            )
            print(f"Moved to {pos} units (~{pos_rad:.3f} rad).")


def verify_zero(
    controller,
    servo: ServoConfig,
    joint: str,
    state: JointState,
    move_ms: int,
    pause_s: float,
) -> None:
    target_units = rad_to_units(
        servo,
        0.0,
        direction_override=state.direction,
        offset_override=state.offset,
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
    pos_rad = units_to_rad(
        servo,
        pos,
        direction_override=state.direction,
        offset_override=state.offset,
    )
    status = "OK" if abs(pos_rad) <= VERIFY_TOL_RAD else "drift"
    print(f"Verify zero: {pos} units => {pos_rad:.4f} rad ({status})")


def write_config(
    base_data: dict,
    output_path: Path,
    updates: Dict[str, JointState],
) -> None:
    hiw = base_data.setdefault("Hiwonder_controller", {})
    servos = hiw.setdefault("servos", {})
    for joint, state in updates.items():
        if joint not in servos:
            servos[joint] = {}
        servos[joint]["offset"] = int(state.offset)
        servos[joint]["direction"] = int(state.direction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base_data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive servo calibration (offset + direction)")
    parser.add_argument(
        "--config",
        default="runtime/configs/wr_runtime_config.json",
        help="Input runtime config path",
    )
    parser.add_argument("--output", help="Optional output path; default is in-place update with backup")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Calibrate all joints in config order")
    group.add_argument("--joints", help="Comma-separated joint list to calibrate")
    parser.add_argument("--step-units", type=int, default=DEFAULT_STEP_UNITS, help="Jog step size in servo units")
    parser.add_argument("--move-ms", type=int, default=DEFAULT_MOVE_MS, help="Move duration in milliseconds")
    parser.add_argument("--delta-rad", type=float, default=DEFAULT_DELTA_RAD, help="Delta radians for direction test")
    parser.add_argument("--delta-units", type=int, help="Override delta in servo units for direction test")
    parser.add_argument("--bundle", help="Policy bundle directory containing policy_spec.json")
    parser.add_argument("--scene-xml", help="Scene XML path (requires MuJoCo installed)")
    parser.add_argument("--keyframes-xml", help="keyframes.xml path for home pose (actuator count must be 8)")
    parser.add_argument("--go-home", action="store_true", help="Move to home pose before calibration")
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
    parser.add_argument(
        "--hint",
        action="append",
        type=parse_hint_override,
        help="Override positive motion hint per joint (joint=hint)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config path not found: {config_path}")

    raw_config = json.loads(config_path.read_text())
    config = WrRuntimeConfig.load(config_path)
    servo_cfgs = config.hiwonder_controller.servos
    joint_names = list(servo_cfgs.keys())

    if args.joints:
        selected = [j.strip() for j in args.joints.split(",") if j.strip()]
    else:
        selected = joint_names if args.all else []
    for j in selected:
        if j not in servo_cfgs:
            sys.exit(f"Joint {j} not found in config; available: {joint_names}")

    hints = dict(POSITIVE_HINTS)
    if args.hint:
        for joint, hint in args.hint:
            hints[joint] = hint

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
                units = rad_to_units(
                    servo,
                    rad,
                    direction_override=state.direction,
                    offset_override=state.offset,
                )
                print(f"  {joint}: rad={rad:+.4f} -> servo_id={servo.id} units={units}")
        print("Planned calibration targets:")
        for joint in selected:
            servo = servo_cfgs[joint]
            state = states[joint]
            center_units = clamp_units(UNITS_CENTER + state.offset)
            hint = hints.get(joint, "positive motion")
            print(
                f"  {joint}: servo_id={servo.id} offset={state.offset} direction={state.direction} "
                f"center_units={center_units} hint='{hint}'"
            )
        output_path = Path(args.output) if args.output else config_path
        print(f"Would write updates to: {output_path}")
        return

    from runtime.hardware.hiwonder_board_controller import HiwonderBoardController

    controller = HiwonderBoardController(config.hiwonder_controller)
    try:
        if args.go_home:
            home_ctrl = resolve_home_ctrl(args, joint_names)
            if len(home_ctrl) != len(joint_names):
                raise ValueError("home_ctrl_rad length mismatch with joints")
            if yes_no("Move all servos to home pose now?", default=False):
                cmds = []
                for joint, rad in zip(joint_names, home_ctrl):
                    servo = servo_cfgs[joint]
                    state = states[joint]
                    units = rad_to_units(
                        servo,
                        rad,
                        direction_override=state.direction,
                        offset_override=state.offset,
                    )
                    cmds.append((servo.id, units))
                announce_and_pause(
                    f"Step: move all joints to home pose (duration {max(args.move_ms, 800)} ms)",
                    float(args.pause_s),
                )
                controller.move_servos(cmds, max(args.move_ms, 800))
                time.sleep(max(args.move_ms, 800) / 1000.0 + 0.2)

        for joint in selected:
            servo = servo_cfgs[joint]
            state = states[joint]
            center_units = clamp_units(UNITS_CENTER + state.offset)
            hint = hints.get(joint, "positive motion")
            new_dir = calibrate_direction(
                controller,
                servo,
                joint,
                state,
                hint,
                all_servo_ids=servo_ids,
                delta_rad=args.delta_rad,
                delta_units=args.delta_units,
                move_ms=args.move_ms,
                center_units=center_units,
                pause_s=float(args.pause_s),
            )
            states[joint].direction = new_dir
            new_offset = calibrate_offset(
                controller,
                servo,
                joint,
                states[joint],
                all_servo_ids=servo_ids,
                move_ms=args.move_ms,
                step_units=args.step_units,
                pause_s=float(args.pause_s),
            )
            states[joint].offset = new_offset
            verify_zero(controller, servo, joint, states[joint], move_ms=args.move_ms, pause_s=float(args.pause_s))

        output_path = Path(args.output) if args.output else config_path
        write_config(raw_config, output_path, {j: states[j] for j in selected})
        print(f"Wrote updated calibration to {output_path}")
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
