#!/usr/bin/env python3
"""Replay ToddlerBot's ZMP walking reference through TB's own MuJoCo controller.

This is a reference-only diagnostic: no PPO policy is used.  Each control step
loads the ZMP reference motor target for the requested command and sends it
through ToddlerBot's normal ``MuJoCoSim.step()`` path, which in turn uses the
ToddlerBot ``MotorController`` torque model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def _default_toddlerbot_root() -> Path:
    return Path(__file__).resolve().parents[2] / "toddlerbot"


def _load_toddlerbot(root: Path):
    root = root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
    from toddlerbot.sim.mujoco_sim import MuJoCoSim
    from toddlerbot.sim.robot import Robot

    return Robot, MuJoCoSim, WalkZMPReference


def _body_id(model, name: str) -> int:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found in ToddlerBot model: {name}")
    return body_id


def _body_rpy(data, body_id: int) -> tuple[float, float, float]:
    quat_wxyz = data.xquat[body_id]
    quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
    roll, pitch, yaw = R.from_quat(quat_xyzw).as_euler("xyz")
    return float(roll), float(pitch), float(yaw)


def _state_row(sim, torso_id: int, pelvis_id: int) -> dict[str, float]:
    roll, pitch, yaw = _body_rpy(sim.data, torso_id)
    return {
        "root_z": float(sim.data.qpos[2]),
        "torso_z": float(sim.data.xpos[torso_id, 2]),
        "pelvis_z": float(sim.data.xpos[pelvis_id, 2]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def _termination(row: dict[str, float], height_floor: float, tilt_limit: float) -> str | None:
    if row["torso_z"] < height_floor:
        return f"torso_z<{height_floor:.2f}"
    if abs(row["pitch"]) > tilt_limit:
        return f"|pitch|>{tilt_limit:.2f}"
    if abs(row["roll"]) > tilt_limit:
        return f"|roll|>{tilt_limit:.2f}"
    return None


def run(args: argparse.Namespace) -> int:
    Robot, MuJoCoSim, WalkZMPReference = _load_toddlerbot(args.toddlerbot_root)

    robot = Robot(args.robot)
    sim = MuJoCoSim(
        robot,
        n_frames=args.n_frames,
        dt=args.sim_dt,
        fixed_base=False,
        vis_type=args.vis,
        controller_type="torque",
    )
    motion_ref = WalkZMPReference(
        robot,
        sim.control_dt,
        args.cycle_time,
        fixed_base=False,
    )

    command = np.zeros(8, dtype=np.float32)
    command[5] = np.float32(args.vx)

    state_ref = motion_ref.get_default_state()
    state_ref = motion_ref.get_state_ref(0.0, command, state_ref, init_idx=0)
    sim.set_qpos(np.asarray(state_ref["qpos"], dtype=np.float32))
    sim.data.qvel[:] = 0.0
    sim.forward()
    sim.set_motor_target(np.asarray(state_ref["motor_pos"], dtype=np.float32))

    torso_id = _body_id(sim.model, "torso")
    pelvis_id = _body_id(sim.model, "pelvis_link")

    print("ToddlerBot ZMP prior replay through TB MuJoCoSim/MotorController")
    print(f"  toddlerbot_root: {args.toddlerbot_root.resolve()}")
    print(f"  robot:           {args.robot}")
    print(f"  vx:              {args.vx:.3f} m/s")
    print(f"  cycle_time:      {args.cycle_time:.3f} s")
    print(f"  control_dt:      {sim.control_dt:.3f} s ({args.n_frames} x {args.sim_dt:g})")
    print(f"  termination:     torso_z<{args.height_floor:.2f} or |pitch/roll|>{args.tilt_limit:.2f}")
    print()
    print(f"{'step':>5} {'t':>6} {'root_z':>8} {'torso_z':>8} {'pelvis_z':>8} {'pitch':>8} {'roll':>8} {'target_rmse':>11}")
    print("-" * 78)

    termination = None
    last_row: dict[str, float] | None = None
    steps_survived = 0

    for step in range(args.max_steps + 1):
        t_curr = step * sim.control_dt
        row = _state_row(sim, torso_id, pelvis_id)
        last_row = row

        motor_pos = sim.data.qpos[sim.q_start_idx + sim.motor_indices]
        motor_target = np.asarray(state_ref["motor_pos"], dtype=np.float32)
        target_rmse = float(np.sqrt(np.mean((motor_pos - motor_target) ** 2)))

        if step % args.trace_every == 0 or step <= args.trace_initial:
            print(
                f"{step:5d} {t_curr:6.2f} {row['root_z']:8.3f} {row['torso_z']:8.3f} "
                f"{row['pelvis_z']:8.3f} {row['pitch']:+8.3f} {row['roll']:+8.3f} "
                f"{target_rmse:11.4f}"
            )

        termination = _termination(row, args.height_floor, args.tilt_limit)
        if termination is not None:
            steps_survived = step
            break
        if step == args.max_steps:
            steps_survived = step
            break

        torso_yaw = 0.0 if args.no_torso_yaw_feedback else row["yaw"]
        state_ref = motion_ref.get_state_ref(
            t_curr,
            command,
            state_ref,
            init_idx=0,
            torso_yaw=torso_yaw,
        )
        sim.set_motor_target(np.asarray(state_ref["motor_pos"], dtype=np.float32))
        sim.step()

    print()
    if termination is None:
        print(f"RESULT: survived {steps_survived} ctrl steps ({steps_survived * sim.control_dt:.2f}s)")
    else:
        assert last_row is not None
        print(
            "RESULT: terminated at "
            f"step={steps_survived} t={steps_survived * sim.control_dt:.2f}s "
            f"reason={termination} pitch={last_row['pitch']:+.3f} "
            f"roll={last_row['roll']:+.3f} torso_z={last_row['torso_z']:.3f}"
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toddlerbot-root",
        type=Path,
        default=_default_toddlerbot_root(),
        help="Path to the ToddlerBot repository.",
    )
    parser.add_argument("--robot", default="toddlerbot_2xc")
    parser.add_argument("--vx", type=float, default=0.20)
    parser.add_argument("--cycle-time", type=float, default=0.72)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--height-floor", type=float, default=0.20)
    parser.add_argument("--tilt-limit", type=float, default=1.0)
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--trace-every", type=int, default=5)
    parser.add_argument("--trace-initial", type=int, default=5)
    parser.add_argument("--vis", default="")
    parser.add_argument(
        "--no-torso-yaw-feedback",
        action="store_true",
        help="Do not pass measured torso yaw back into WalkZMPReference.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
