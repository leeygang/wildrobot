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


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return the 3x3 rotation matrix R for a unit wxyz quaternion such
    that ``world_v = R @ body_v``.  The inverse (world -> body) is R.T."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
             2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
             2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
             1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument(
        "--torso-alpha",
        type=float,
        default=200.0,
        help="Alpha for torso_pos_xy raw reward exp(-alpha * err^2)",
    )
    p.add_argument(
        "--foot-alpha",
        type=float,
        default=200.0,
        help=(
            "Alpha for v0.20.2 ref_foot_pos_body raw reward "
            "exp(-alpha * (||err_left||^2 + ||err_right||^2)).  "
            "TB pos default 200; calibrate downward if open-loop drift "
            "kills the iter-0 signal."
        ),
    )
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

        # ---- v0.20.2 ref_foot_pos_body raw reward + per-foot Euclidean
        # body-frame foot-vs-pelvis offset error.  Mirrors the env block:
        # err = inv(R_root) @ (foot_world - pelvis_world)
        #         - (prior_foot_world - prior_pelvis_world).
        actual_pelv_world = np.array(
            [actual_pelv_x, actual_pelv_y, actual_pelv_z], dtype=np.float64
        )
        actual_L_world = np.array(data.xpos[L_id], dtype=np.float64)
        actual_R_world = np.array(data.xpos[R_id], dtype=np.float64)
        R_world_from_body = _quat_to_rotmat(*data.qpos[3:7])
        # World->body inverse rotation = R.T for orthonormal R.
        actual_L_body_rel = R_world_from_body.T @ (actual_L_world - actual_pelv_world)
        actual_R_body_rel = R_world_from_body.T @ (actual_R_world - actual_pelv_world)
        prior_pelv_world = (
            np.array(traj.pelvis_pos[idx], dtype=np.float64)
            if traj.pelvis_pos is not None
            else np.zeros(3, dtype=np.float64)
        )
        prior_L_world = (
            np.array(traj.left_foot_pos[idx], dtype=np.float64)
            if traj.left_foot_pos is not None
            else np.zeros(3, dtype=np.float64)
        )
        prior_R_world = (
            np.array(traj.right_foot_pos[idx], dtype=np.float64)
            if traj.right_foot_pos is not None
            else np.zeros(3, dtype=np.float64)
        )
        prior_L_body_rel = prior_L_world - prior_pelv_world
        prior_R_body_rel = prior_R_world - prior_pelv_world
        foot_pos_body_err_l = float(
            np.linalg.norm(actual_L_body_rel - prior_L_body_rel)
        )
        foot_pos_body_err_r = float(
            np.linalg.norm(actual_R_body_rel - prior_R_body_rel)
        )
        foot_pos_body_raw = float(
            np.exp(
                -float(args.foot_alpha)
                * (foot_pos_body_err_l ** 2 + foot_pos_body_err_r ** 2)
            )
        )

        # LIPM forward lever: positive ⇒ COM is ahead of stance foot
        # (prior demands forward acceleration); negative ⇒ behind ⇒
        # backward acceleration.
        prior_lever = prior_com_x - prior_stance_x
        actual_lever = actual_pelv_x - actual_stance_x
        torso_dx = actual_pelv_x - prior_pelv_x
        torso_dy = actual_pelv_y - prior_pelv_y
        torso_pos_xy_err_m = float(np.hypot(torso_dx, torso_dy))
        torso_pos_xy_raw = float(
            np.exp(-float(args.torso_alpha) * torso_pos_xy_err_m * torso_pos_xy_err_m)
        )

        rows.append({
            "step": step_idx,
            "phase": float(traj.phase[idx]) if traj.phase is not None else 0.0,
            "stance": "L" if stance == 0 else "R",
            "prior_com_x": prior_com_x,
            "actual_pelv_x": actual_pelv_x,
            "delta_x": actual_pelv_x - prior_com_x,
            "prior_pelv_x": prior_pelv_x,
            "prior_pelv_y": prior_pelv_y,
            "prior_com_y": prior_com_y,
            "actual_pelv_y": actual_pelv_y,
            "actual_pelv_z": actual_pelv_z,
            "torso_pos_xy_err_m": torso_pos_xy_err_m,
            "torso_pos_xy_raw": torso_pos_xy_raw,
            "foot_pos_body_err_l_m": foot_pos_body_err_l,
            "foot_pos_body_err_r_m": foot_pos_body_err_r,
            "foot_pos_body_raw": foot_pos_body_raw,
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

    # Torso XY calibration summary for reward/torso_pos_xy.
    sample_steps = (10, 30, 50, 70)
    by_step = {int(r["step"]): r for r in rows}
    print(
        f"\nTorso XY probe (alpha={args.torso_alpha:.1f}): "
        "step -> err_m, raw_reward"
    )
    for s in sample_steps:
        r = by_step.get(s)
        if r is None:
            print(f"  step {s:>3}: n/a")
        else:
            print(
                f"  step {s:>3}: err={r['torso_pos_xy_err_m']:.4f} m, "
                f"raw={r['torso_pos_xy_raw']:.6f}"
            )

    first_50 = [r for r in rows if int(r["step"]) <= 50]
    if first_50:
        min_raw_50 = min(float(r["torso_pos_xy_raw"]) for r in first_50)
        print(f"  min raw reward over steps 0..50: {min_raw_50:.6f}")

    # v0.20.2: ref_foot_pos_body calibration summary.  Same alive-signal
    # bar as torso_pos_xy: min raw over steps 0..50 should be ≥ 0.05 so
    # PPO sees a usable gradient through the prior's accel ramp.
    print(
        f"\nFoot-pos body probe (alpha={args.foot_alpha:.1f}): "
        "step -> err_l_m, err_r_m, raw_reward"
    )
    for s in sample_steps:
        r = by_step.get(s)
        if r is None:
            print(f"  step {s:>3}: n/a")
        else:
            print(
                f"  step {s:>3}: "
                f"err_l={r['foot_pos_body_err_l_m']:.4f} m, "
                f"err_r={r['foot_pos_body_err_r_m']:.4f} m, "
                f"raw={r['foot_pos_body_raw']:.6f}"
            )
    if first_50:
        min_foot_raw_50 = min(float(r["foot_pos_body_raw"]) for r in first_50)
        print(f"  min raw reward over steps 0..50: {min_foot_raw_50:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
