#!/usr/bin/env python3
"""MuJoCo viewer for ZMP reference library (v0.20.0-B verification).

Replays reference trajectories from the offline library on the WildRobot
MuJoCo model.  The viewer sets joint position targets (ctrl) from the
library's q_ref and lets the PD servos track them, showing whether the
reference is physically trackable.

Usage:
  # Interactive viewer at cmd=0.15
  uv run mjpython training/eval/view_zmp_in_mujoco.py --vx 0.15

  # Headless recording
  uv run mjpython training/eval/view_zmp_in_mujoco.py --vx 0.15 \
      --headless --horizon 100 --print-every 5

  # Custom library path
  uv run mjpython training/eval/view_zmp_in_mujoco.py \
      --library-path /tmp/zmp_ref_library --vx 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from control.references.reference_library import ReferenceLibrary
from training.utils.ctrl_order import CtrlOrderMapper


def _load_model():
    """Load WildRobot MuJoCo model."""
    model_path = Path("assets/v2/scene_flat_terrain.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def _load_policy_spec():
    """Load policy spec and extract actuator names."""
    spec_path = Path("assets/v2/mujoco_robot_config.json")
    with open(spec_path) as f:
        spec = json.load(f)
    return [j["name"] for j in spec["actuated_joint_specs"]]


def _init_standing(model, data):
    """Initialize robot to standing keyframe."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def run_viewer(
    lib: ReferenceLibrary,
    vx: float,
    headless: bool = False,
    horizon: int = 200,
    print_every: int = 10,
    sim_dt_factor: int = 1,
) -> None:
    model, data = _load_model()
    actuator_names = _load_policy_spec()
    mapper = CtrlOrderMapper(model, actuator_names)

    traj = lib.lookup(vx)
    n_cycle = traj.n_steps
    print(f"Trajectory: vx={traj.command_vx:.3f}, steps={n_cycle}, "
          f"cycle_time={traj.cycle_time:.3f}s")
    print(f"Viewer: horizon={horizon}, dt_factor={sim_dt_factor}")

    _init_standing(model, data)

    # Physics dt and reference dt
    ref_dt = traj.dt
    physics_dt = model.opt.timestep
    physics_steps_per_ref = max(1, int(round(ref_dt / physics_dt)))

    # Startup ramp: gradually blend from standing to walking posture
    # over startup_steps to avoid a violent initial transition.
    startup_steps = 50
    standing_q = np.zeros(19, dtype=np.float32)  # standing: all joints near zero
    first_q = traj.q_ref[0]

    print(f"Startup ramp: {startup_steps} steps blending to walking posture")
    for i in range(startup_steps):
        alpha = (i + 1) / startup_steps
        blended = standing_q * (1 - alpha) + first_q * alpha
        mapper.set_all_ctrl(data, blended)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)

    root_z_after_ramp = data.qpos[2]
    print(f"After ramp: root_z={root_z_after_ramp:.4f}")

    step_log = []

    def step_fn(step_idx: int) -> dict:
        """Execute one reference step."""
        cycle_idx = step_idx % n_cycle

        # Get q_ref for this step (policy order)
        q_ref = traj.q_ref[cycle_idx]

        # Set ctrl via the mapper (handles policy → MuJoCo ordering)
        mapper.set_all_ctrl(data, q_ref)

        # Step physics
        for _ in range(physics_steps_per_ref * sim_dt_factor):
            mujoco.mj_step(model, data)

        # Collect diagnostics
        root_pos = data.qpos[:3].copy()
        root_quat = data.qpos[3:7].copy()
        # Approximate pitch from quaternion
        qw, qx, qy, qz = root_quat
        pitch = np.arctan2(2 * (qw * qy - qz * qx),
                           1 - 2 * (qx**2 + qy**2))
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                          1 - 2 * (qx**2 + qy**2))

        return {
            "step": step_idx,
            "cycle_idx": cycle_idx,
            "root_x": root_pos[0],
            "root_y": root_pos[1],
            "root_z": root_pos[2],
            "pitch": pitch,
            "roll": roll,
            "stance": int(traj.stance_foot_id[cycle_idx]),
            "contact_l": float(traj.contact_mask[cycle_idx, 0]),
            "contact_r": float(traj.contact_mask[cycle_idx, 1]),
        }

    if headless:
        print(f"\n{'step':>5} {'cyc':>4} {'root_z':>7} {'pitch':>7} "
              f"{'roll':>7} {'root_x':>7} {'stance':>6}")
        print("-" * 55)

        for i in range(horizon):
            info = step_fn(i)
            step_log.append(info)

            if i % print_every == 0 or i == horizon - 1:
                st = "L" if info["stance"] == 0 else "R"
                print(f"{i:5d} {info['cycle_idx']:4d} {info['root_z']:7.4f} "
                      f"{info['pitch']:+7.4f} {info['roll']:+7.4f} "
                      f"{info['root_x']:+7.4f} {st:>6}")

            # Check termination
            if abs(info["pitch"]) > 0.8 or info["root_z"] < 0.15:
                print(f"\n  TERMINATED at step {i}: "
                      f"pitch={info['pitch']:.3f} root_z={info['root_z']:.3f}")
                break

        # Summary
        if step_log:
            pitches = [s["pitch"] for s in step_log]
            rolls = [s["roll"] for s in step_log]
            heights = [s["root_z"] for s in step_log]
            x_final = step_log[-1]["root_x"]
            survived = len(step_log)
            print(f"\nSummary ({survived}/{horizon} steps):")
            print(f"  pitch: mean={np.mean(pitches):+.4f} "
                  f"p95={np.percentile(np.abs(pitches), 95):.4f}")
            print(f"  roll:  mean={np.mean(rolls):+.4f} "
                  f"p95={np.percentile(np.abs(rolls), 95):.4f}")
            print(f"  height: mean={np.mean(heights):.4f} "
                  f"min={np.min(heights):.4f}")
            print(f"  forward: x={x_final:+.4f}m in {survived} steps")
    else:
        # Interactive viewer
        step_idx = [0]

        def controller(model, data):
            info = step_fn(step_idx[0])
            step_idx[0] += 1
            if step_idx[0] % print_every == 0:
                st = "L" if info["stance"] == 0 else "R"
                print(f"step={info['step']:5d} z={info['root_z']:.3f} "
                      f"pitch={info['pitch']:+.3f} x={info['root_x']:+.3f} {st}")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0, 0, 0.4]

            while viewer.is_running():
                controller(model, data)
                viewer.sync()
                time.sleep(ref_dt * sim_dt_factor)


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo viewer for ZMP reference library")
    parser.add_argument("--library-path", type=str, default=None,
                        help="Path to reference library (default: generate fresh)")
    parser.add_argument("--vx", type=float, default=0.15,
                        help="Forward speed command")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless with text output")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Number of reference steps to run")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print diagnostics every N steps")
    parser.add_argument("--sim-dt-factor", type=int, default=1,
                        help="Physics substeps per reference step")
    args = parser.parse_args()

    if args.library_path:
        lib = ReferenceLibrary.load(args.library_path)
    else:
        print("No library path given, generating fresh...")
        from control.zmp.zmp_walk import ZMPWalkGenerator
        gen = ZMPWalkGenerator()
        lib = gen.build_library()

    print(lib.summary())
    print()
    run_viewer(
        lib, args.vx,
        headless=args.headless,
        horizon=args.horizon,
        print_every=args.print_every,
        sim_dt_factor=args.sim_dt_factor,
    )


if __name__ == "__main__":
    main()
