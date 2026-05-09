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
    """Return the 3x3 rotation matrix R such that ``world_v = R @ body_v``.
    Inverse (world -> body) is R.T."""
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
    p.add_argument("--vx", type=float, default=0.20)  # Phase 9D operating point (TB-step/leg-matched at cycle_time=0.96 s)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument(
        "--torso-alpha",
        type=float,
        default=200.0,
        help="Alpha for torso_pos_xy raw reward exp(-alpha * err^2)",
    )
    p.add_argument(
        "--lin-vel-z-alpha",
        type=float,
        default=200.0,
        help=(
            "Alpha for lin_vel_z raw reward exp(-alpha * (vz - ref_vz)^2). "
            "TB lin_vel_tracking_sigma=200 default."
        ),
    )
    p.add_argument(
        "--ang-vel-xy-alpha",
        type=float,
        default=0.5,
        help=(
            "Alpha for ang_vel_xy raw reward exp(-alpha * (gyro_x^2 + gyro_y^2)). "
            "TB ang_vel_tracking_sigma=0.5 default."
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
    # Match the env's reward-side gyro path
    # (training/sim_adapter/mjx_signals.py reads the ``chest_imu_gyro``
    # sensor; falls back to data.qvel[3:6] only if absent).  Reading the
    # sensor here makes the probe's ang_vel_xy values apples-to-apples
    # with the env's actual reward signal under any IMU sensor noise.
    gyro_sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "chest_imu_gyro"
    )
    if gyro_sensor_id >= 0:
        gyro_sensor_adr = int(model.sensor_adr[gyro_sensor_id])
    else:
        gyro_sensor_adr = -1  # use qvel[3:6] fallback path
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if pelvis_id < 0:
        # Fall back: free-joint root body is body index 1 in standard MJCF.
        pelvis_id = 1

    n_steps = traj.q_ref.shape[0]
    ctrl_dt = float(traj.dt)

    # Anchor the trajectory's frame: the prior is in trajectory-local
    # coordinates (cycle 0 starts at x=0).  After the keyframe init +
    # ramp, the body sits near the world origin.  We log raw values so
    # the user can compare deltas directly.
    prev_prior_pelv_z = (
        float(traj.pelvis_pos[0, 2])
        if traj.pelvis_pos is not None
        else float("nan")
    )
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
        torso_dx = actual_pelv_x - prior_pelv_x
        torso_dy = actual_pelv_y - prior_pelv_y
        torso_pos_xy_err_m = float(np.hypot(torso_dx, torso_dy))
        torso_pos_xy_raw = float(
            np.exp(-float(args.torso_alpha) * torso_pos_xy_err_m * torso_pos_xy_err_m)
        )

        # ---- v0.20.2 smoke6 lin_vel_z + ang_vel_xy alive bands -----------
        # lin_vel_z (TB-aligned): rotate world velocity by inv(root_quat)
        # to true body-local; reference is finite-diff of prior pelvis_z.
        world_lin_vel = np.array(data.qvel[0:3], dtype=np.float64)
        R_world_from_body = _quat_to_rotmat(*data.qpos[3:7])
        body_lin_vel = R_world_from_body.T @ world_lin_vel
        body_lin_vel_z = float(body_lin_vel[2])
        prior_pelv_z_now = (
            float(traj.pelvis_pos[idx, 2])
            if traj.pelvis_pos is not None
            else float("nan")
        )
        ref_lin_vel_z = (prior_pelv_z_now - prev_prior_pelv_z) / ctrl_dt
        prev_prior_pelv_z = prior_pelv_z_now
        lin_vel_z_err = body_lin_vel_z - ref_lin_vel_z
        lin_vel_z_raw = float(
            np.exp(-float(args.lin_vel_z_alpha) * lin_vel_z_err * lin_vel_z_err)
        )

        # ang_vel_xy: read the same gyro signal the env reward sees
        # (chest_imu_gyro sensor → matches signals_raw.gyro_rad_s in
        # training/sim_adapter/mjx_signals.py).  Falls back to qvel[3:6]
        # if the sensor doesn't exist; under IMU sensor noise the sensor
        # path will differ from raw qvel by a tiny σ=0.0002 rad/s.
        if gyro_sensor_adr >= 0:
            body_ang_vel = np.array(
                data.sensordata[gyro_sensor_adr : gyro_sensor_adr + 3],
                dtype=np.float64,
            )
        else:
            body_ang_vel = np.array(data.qvel[3:6], dtype=np.float64)
        ang_vel_xy_sq_sum = float(
            body_ang_vel[0] * body_ang_vel[0]
            + body_ang_vel[1] * body_ang_vel[1]
        )
        ang_vel_xy_raw = float(
            np.exp(-float(args.ang_vel_xy_alpha) * ang_vel_xy_sq_sum)
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
            "lin_vel_z_err_m_s": lin_vel_z_err,
            "lin_vel_z_raw": lin_vel_z_raw,
            "ang_vel_xy_rad_s": float(np.sqrt(ang_vel_xy_sq_sum)),
            "ang_vel_xy_raw": ang_vel_xy_raw,
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

    # v0.20.2 smoke6: lin_vel_z calibration summary.
    print(
        f"\nlin_vel_z probe (alpha={args.lin_vel_z_alpha:.1f}): "
        "step -> err_m_s, raw_reward"
    )
    for s in sample_steps:
        r = by_step.get(s)
        if r is None:
            print(f"  step {s:>3}: n/a")
        else:
            print(
                f"  step {s:>3}: "
                f"err={r['lin_vel_z_err_m_s']:+.4f} m/s, "
                f"raw={r['lin_vel_z_raw']:.6f}"
            )
    if first_50:
        min_lvz_raw_50 = min(float(r["lin_vel_z_raw"]) for r in first_50)
        print(f"  min raw reward over steps 0..50: {min_lvz_raw_50:.6f}")

    # v0.20.2 smoke6: ang_vel_xy calibration summary.
    print(
        f"\nang_vel_xy probe (alpha={args.ang_vel_xy_alpha:.2f}): "
        "step -> ||gyro_xy||_rad_s, raw_reward"
    )
    for s in sample_steps:
        r = by_step.get(s)
        if r is None:
            print(f"  step {s:>3}: n/a")
        else:
            print(
                f"  step {s:>3}: "
                f"|gyro_xy|={r['ang_vel_xy_rad_s']:.4f} rad/s, "
                f"raw={r['ang_vel_xy_raw']:.6f}"
            )
    if first_50:
        min_avxy_raw_50 = min(float(r["ang_vel_xy_raw"]) for r in first_50)
        print(f"  min raw reward over steps 0..50: {min_avxy_raw_50:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
