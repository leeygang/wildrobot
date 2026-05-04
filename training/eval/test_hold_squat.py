#!/usr/bin/env python3
"""Minimal test: can the MuJoCo servo hold a squat posture?

Initializes the robot in the support posture (from IK), sets ctrl = qpos,
and runs pure MuJoCo physics with NO env/reference/adapter logic.
Just servo hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

project_root = Path(__file__).parent.parent.parent


def main() -> None:
    model_path = project_root / "assets" / "v2" / "scene_flat_terrain.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)

    # Load keyframe as starting point
    if mj_model.nkey > 0:
        mj_data.qpos[:] = mj_model.key_qpos[0]
        mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Find joint indices by name
    def jnt_qpos_addr(name: str) -> int:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(mj_model.jnt_qposadr[jid])

    def act_id(name: str) -> int:
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    # Compute support posture joint angles from simple 2-link IK
    from math import acos, atan2, pi, sqrt, sin, cos
    l1, l2 = 0.21, 0.21
    support_height = 0.39 + 0.02  # current support_pelvis_height_offset_m
    margin = 0.02  # current stance_extension_margin_m (constant now)
    target_z = -(support_height - margin)  # = -0.39
    target_x = 0.0
    r = sqrt(target_x**2 + target_z**2)
    r = min(r, l1 + l2 - 1e-4)
    cos_knee = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    cos_knee = max(-1, min(1, cos_knee))
    knee_internal = acos(cos_knee)
    knee_pitch = max(0, pi - knee_internal)
    cos_hip_aux = max(-1, min(1, (l1**2 + r**2 - l2**2) / (2 * l1 * r)))
    hip_aux = acos(cos_hip_aux)
    hip_line = atan2(target_x, -target_z)
    hip_pitch = hip_line - hip_aux
    ankle_pitch = -(hip_pitch + knee_pitch)

    print(f"IK for support posture (target_x={target_x:.3f}, target_z={target_z:.3f}):")
    print(f"  hip_pitch  = {hip_pitch:+.4f} rad ({hip_pitch*180/pi:+.1f} deg)")
    print(f"  knee_pitch = {knee_pitch:+.4f} rad ({knee_pitch*180/pi:+.1f} deg)")
    print(f"  ankle_pitch= {ankle_pitch:+.4f} rad ({ankle_pitch*180/pi:+.1f} deg)")
    print()

    # Set BOTH legs to support posture
    joints = {
        "left_hip_pitch": -hip_pitch,   # sign flip per IK adapter convention
        "right_hip_pitch": -hip_pitch,
        "left_knee_pitch": knee_pitch,
        "right_knee_pitch": knee_pitch,
        "left_ankle_pitch": ankle_pitch,
        "right_ankle_pitch": ankle_pitch,
    }
    for name, angle in joints.items():
        addr = jnt_qpos_addr(name)
        mj_data.qpos[addr] = angle

    # Forward to compute positions, then ground the robot
    mujoco.mj_forward(mj_model, mj_data)

    # Find lowest foot contact point and shift root z to ground it
    foot_geom_names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    min_z = float("inf")
    for gname in foot_geom_names:
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        pos = mj_data.geom_xpos[gid]
        size = mj_model.geom_size[gid]
        # For box geoms, bottom = center_z - half_height (size[2] for boxes oriented properly)
        # But geoms may be rotated, use a conservative estimate
        bottom = pos[2] - size[0]  # size[0] is the smallest half-extent for thin boxes
        min_z = min(min_z, bottom)
    mj_data.qpos[2] -= min_z  # shift root so lowest point is at z=0
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    root_h = mj_data.qpos[2]
    print(f"Grounded root height: {root_h:.4f}m")
    print()

    # Set ctrl = current joint positions (hold this pose)
    for i in range(mj_model.nu):
        aname = mj_model.actuator(i).name
        jname = mj_model.actuator(i).name  # same name for position actuators
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            addr = int(mj_model.jnt_qposadr[jid])
            mj_data.ctrl[i] = mj_data.qpos[addr]

    # Print initial ctrl for leg joints
    print("Initial ctrl (hold targets):")
    for name in ["left_hip_pitch", "right_hip_pitch",
                  "left_knee_pitch", "right_knee_pitch",
                  "left_ankle_pitch", "right_ankle_pitch"]:
        aid = act_id(name)
        print(f"  {name:25s} ctrl={mj_data.ctrl[aid]:+.4f}")
    print()

    # Run simulation — just hold, no reference, no env, no adapter
    ctrl_dt = 0.02
    sim_dt = float(mj_model.opt.timestep)
    substeps = max(1, int(round(ctrl_dt / sim_dt)))
    n_steps = 200  # 4 seconds at 50Hz

    print(f"Running {n_steps} steps ({n_steps * ctrl_dt:.1f}s), substeps={substeps}, sim_dt={sim_dt}")
    print()

    lk_aid = act_id("left_knee_pitch")
    rk_aid = act_id("right_knee_pitch")
    la_aid = act_id("left_ankle_pitch")
    ra_aid = act_id("right_ankle_pitch")
    lk_addr = jnt_qpos_addr("left_knee_pitch")
    rk_addr = jnt_qpos_addr("right_knee_pitch")
    la_addr = jnt_qpos_addr("left_ankle_pitch")
    ra_addr = jnt_qpos_addr("right_ankle_pitch")

    header = f"{'step':>4} {'time':>5} | {'Lk_ctrl':>8} {'Lk_q':>8} {'Lk_err':>8} {'Lk_F':>8} | {'Rk_ctrl':>8} {'Rk_q':>8} {'Rk_F':>8} | {'La_q':>8} {'Ra_q':>8} | {'root_h':>7} {'pitch':>7} {'p_rate':>7}"
    print(header)
    print("-" * len(header))

    for step in range(1, n_steps + 1):
        # ctrl stays FIXED — no updates, no reference, just hold
        for _ in range(substeps):
            mujoco.mj_step(mj_model, mj_data)

        if step % 10 == 0 or step <= 5 or step == n_steps:
            lk_q = mj_data.qpos[lk_addr]
            rk_q = mj_data.qpos[rk_addr]
            la_q = mj_data.qpos[la_addr]
            ra_q = mj_data.qpos[ra_addr]
            lk_f = mj_data.actuator_force[lk_aid]
            rk_f = mj_data.actuator_force[rk_aid]
            qw, qx, qy, qz = mj_data.qpos[3:7]
            pitch = float(np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2)))
            # Approximate pitch rate from angular velocity
            p_rate = float(mj_data.qvel[4]) if mj_data.qvel.shape[0] > 4 else 0.0
            print(
                f"{step:4d} {step*ctrl_dt:5.2f} | "
                f"{mj_data.ctrl[lk_aid]:+8.4f} {lk_q:+8.4f} {mj_data.ctrl[lk_aid]-lk_q:+8.4f} {lk_f:+8.3f} | "
                f"{mj_data.ctrl[rk_aid]:+8.4f} {rk_q:+8.4f} {rk_f:+8.3f} | "
                f"{la_q:+8.4f} {ra_q:+8.4f} | "
                f"{mj_data.qpos[2]:7.4f} {pitch:+7.4f} {p_rate:+7.3f}"
            )


if __name__ == "__main__":
    main()
