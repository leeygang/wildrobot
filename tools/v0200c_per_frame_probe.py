#!/usr/bin/env python3
"""v0.20.0-C round-9 per-frame divergence probe.

The free-floating viewer at vx=0.15 walks BACKWARD even though the
LQR-based prior demands forward motion (round-9b: design fully
aligned with ToddlerBot).  This script runs the deterministic
viewer-equivalent loop and logs, per ctrl step, the prior's intent
vs the physics state so we can localise where forward intent is
lost.

Scope note (round-10 reviewer): this probe assigns "stance side"
from ``traj.stance_foot_id[idx]`` (the schedule).  That is correct
ground truth in the first few frames before physics diverges from
the schedule, but once the body starts falling it stops being
reliable — the schedule says one foot is in stance while real
physics may have either foot off the floor.  Use this probe for
the early-transient diagnosis (first ~20 ctrl steps); for
post-divergence contact analysis, replace ``stance`` with the
real MuJoCo foot-floor contact (see
``training/eval/view_zmp_in_mujoco.py:_foot_floor_in_contact``).

Per step it logs:

  - prior pelvis_x / pelvis_y     (com_pos from traj)
  - actual pelvis x / y / z       (qpos[:3])
  - pelvis pitch                  (extracted from qpos[3:7])
  - prior stance foot world x     (left_foot_pos / right_foot_pos)
  - actual stance foot world x    (data.xpos[L_id or R_id])
  - prior LIPM lever (x_com - x_zmp)
  - actual lever                  (pelvis_x - stance_foot_x)
  - sign of lever diff            (catches the moment intent flips)

Outputs a CSV at /tmp/v0200c_probe.csv and prints first-divergence
summary so the offending frame can be inspected directly.

Usage::

    uv run python tools/v0200c_per_frame_probe.py --vx 0.15 --horizon 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.references.reference_library import ReferenceLibrary
from training.utils.ctrl_order import CtrlOrderMapper


def _quat_to_pitch(qw: float, qx: float, qy: float, qz: float) -> float:
    return float(np.arctan2(2 * (qw * qy - qz * qx),
                            1 - 2 * (qx ** 2 + qy ** 2)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--csv", type=str, default="/tmp/v0200c_probe.csv")
    p.add_argument("--scene", type=str,
                   default="assets/v2/scene_flat_terrain.xml")
    args = p.parse_args()

    # ---- Build trajectory + MuJoCo model ---------------------------------
    from control.zmp.zmp_walk import ZMPWalkGenerator
    gen = ZMPWalkGenerator()
    lib = gen.build_library()
    traj = lib.lookup(args.vx)
    print(f"trajectory: vx={args.vx} n_steps={traj.q_ref.shape[0]} "
          f"dt={traj.dt} cycle_time={traj.cycle_time}")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mapper = CtrlOrderMapper(
        model,
        [j["name"] for j in __import__("json").load(
            open("assets/v2/mujoco_robot_config.json"))["actuated_joint_specs"]],
    )

    # Reset to walk_start keyframe (matches viewer's _init_standing).
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "walk_start")
    if kid < 0:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)

    # Build the same act_to_qpos / policy_to_mj indexing the viewer uses.
    act_to_qpos = np.array([
        model.jnt_qposadr[model.actuator_trnid[k, 0]]
        for k in range(model.nu)
    ])
    policy_to_mj = np.array(mapper.policy_to_mj_order)

    # 50-step startup ramp matching view_zmp_in_mujoco.py:376-393.
    physics_steps_per_ref = max(1, int(round(traj.dt / model.opt.timestep)))
    mj_qpos_home = data.qpos[act_to_qpos]
    standing_q = mj_qpos_home[policy_to_mj].astype(np.float64)
    first_q = traj.q_ref[0]
    n_ramp = 50
    for r in range(n_ramp):
        a = (r + 1) / n_ramp
        blended = standing_q * (1 - a) + first_q * a
        mapper.set_all_ctrl(data, blended)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)
    print(f"after ramp: pelvis x={data.qpos[0]:+.4f} y={data.qpos[1]:+.4f} "
          f"z={data.qpos[2]:+.4f} pitch={_quat_to_pitch(*data.qpos[3:7]):+.4f}")

    # ---- Body / foot ids -------------------------------------------------
    L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if pelvis_id < 0:
        # Fall back: free-joint root body is body index 1 in standard MJCF.
        pelvis_id = 1

    n_steps = traj.q_ref.shape[0]

    # Anchor the trajectory's frame: the prior is in trajectory-local
    # coordinates (cycle 0 starts at x=0).  After the keyframe init +
    # ramp, the body sits near the world origin.  We log raw values so
    # the user can compare deltas directly.
    rows: list[dict] = []
    for step_idx in range(args.horizon):
        idx = min(step_idx, n_steps - 1)
        q_cmd = traj.q_ref[idx]
        mapper.set_all_ctrl(data, q_cmd)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)

        # Prior intent (trajectory frame ≈ world after init ramp).
        prior_com_x = float(traj.com_pos[idx, 0]) if traj.com_pos is not None else float("nan")
        prior_com_y = float(traj.com_pos[idx, 1]) if traj.com_pos is not None else float("nan")
        prior_pelv_x = float(traj.pelvis_pos[idx, 0]) if traj.pelvis_pos is not None else float("nan")
        prior_pelv_y = float(traj.pelvis_pos[idx, 1]) if traj.pelvis_pos is not None else float("nan")
        prior_L_x = float(traj.left_foot_pos[idx, 0]) if traj.left_foot_pos is not None else float("nan")
        prior_R_x = float(traj.right_foot_pos[idx, 0]) if traj.right_foot_pos is not None else float("nan")
        stance = int(traj.stance_foot_id[idx])
        prior_stance_x = prior_L_x if stance == 0 else prior_R_x

        # Physics state.
        actual_pelv_x = float(data.qpos[0])
        actual_pelv_y = float(data.qpos[1])
        actual_pelv_z = float(data.qpos[2])
        pitch = _quat_to_pitch(*data.qpos[3:7])
        actual_L_x = float(data.xpos[L_id, 0])
        actual_R_x = float(data.xpos[R_id, 0])
        actual_stance_x = actual_L_x if stance == 0 else actual_R_x

        # LIPM forward lever: positive ⇒ COM is ahead of stance foot
        # (prior demands forward acceleration); negative ⇒ behind ⇒
        # backward acceleration.
        prior_lever = prior_com_x - prior_stance_x
        actual_lever = actual_pelv_x - actual_stance_x

        rows.append({
            "step": step_idx,
            "phase": float(traj.phase[idx]) if traj.phase is not None else 0.0,
            "stance": "L" if stance == 0 else "R",
            "prior_com_x": prior_com_x,
            "actual_pelv_x": actual_pelv_x,
            "delta_x": actual_pelv_x - prior_com_x,
            "prior_com_y": prior_com_y,
            "actual_pelv_y": actual_pelv_y,
            "actual_pelv_z": actual_pelv_z,
            "pitch": pitch,
            "prior_stance_x": prior_stance_x,
            "actual_stance_x": actual_stance_x,
            "prior_lever": prior_lever,
            "actual_lever": actual_lever,
            "lever_sign_flip": 1 if (prior_lever > 0) != (actual_lever > 0) else 0,
        })

        # Soft termination on fall.
        if abs(pitch) > 0.8 or actual_pelv_z < 0.15:
            print(f"  fallen at step {step_idx}: pitch={pitch:.3f} z={actual_pelv_z:.3f}")
            break

    # ---- Write CSV + summary --------------------------------------------
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.csv}")

    # Print compact table.
    print(f"\n{'step':>4} {'ph':>5} {'st':>2} "
          f"{'pCOMx':>8} {'aPLVx':>8} {'dx':>8} "
          f"{'pSFx':>8} {'aSFx':>8} "
          f"{'pLever':>8} {'aLever':>8} {'pitch':>7} {'flip':>4}")
    for i, r in enumerate(rows):
        if i % 5 == 0 or i == len(rows) - 1 or r["lever_sign_flip"]:
            print(f"{r['step']:4d} {r['phase']:5.2f} {r['stance']:>2} "
                  f"{r['prior_com_x']:+8.4f} {r['actual_pelv_x']:+8.4f} "
                  f"{r['delta_x']:+8.4f} "
                  f"{r['prior_stance_x']:+8.4f} {r['actual_stance_x']:+8.4f} "
                  f"{r['prior_lever']:+8.4f} {r['actual_lever']:+8.4f} "
                  f"{r['pitch']:+7.3f} {r['lever_sign_flip']:4d}")

    # First lever-sign-flip diagnosis.
    flips = [r for r in rows if r["lever_sign_flip"]]
    if flips:
        print(f"\nFirst lever sign flip at step {flips[0]['step']}: "
              f"prior_lever={flips[0]['prior_lever']:+.4f} "
              f"actual_lever={flips[0]['actual_lever']:+.4f}")
    else:
        print("\nNo prior-vs-actual lever sign flips in the logged window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
