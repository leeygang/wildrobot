#!/usr/bin/env python3
"""Find a balanced support posture by sweeping foot-forward offset."""
from __future__ import annotations

import sys
from math import acos, atan2, pi, sqrt
from pathlib import Path

import mujoco
import numpy as np

project_root = Path(__file__).parent.parent.parent


def test_posture(target_x: float, target_z: float, n_steps: int = 200) -> dict:
    """Run pure MuJoCo hold test and return stability metrics."""
    l1, l2 = 0.21, 0.21
    r = sqrt(target_x**2 + target_z**2)
    r = min(r, l1 + l2 - 1e-4)
    cos_k = max(-1, min(1, (l1**2 + l2**2 - r**2) / (2*l1*l2)))
    knee = max(0, pi - acos(cos_k))
    cos_h = max(-1, min(1, (l1**2 + r**2 - l2**2) / (2*l1*r)))
    hip = atan2(target_x, -target_z) - acos(cos_h)
    ankle = -(hip + knee)

    mj_model = mujoco.MjModel.from_xml_path(str(project_root / "assets/v2/scene_flat_terrain.xml"))
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mj_data.qpos[:] = mj_model.key_qpos[0]
        mj_data.qvel[:] = 0.0

    def jnt_addr(name): return int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    def act_idx(name): return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    joints = {
        "left_hip_pitch": -hip, "right_hip_pitch": -hip,
        "left_knee_pitch": knee, "right_knee_pitch": knee,
        "left_ankle_pitch": ankle, "right_ankle_pitch": ankle,
    }
    for name, angle in joints.items():
        mj_data.qpos[jnt_addr(name)] = angle
    mujoco.mj_forward(mj_model, mj_data)

    # Ground the robot
    foot_geoms = ["left_heel", "left_toe", "right_heel", "right_toe"]
    min_z = min(
        mj_data.geom_xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][2]
        - mj_model.geom_size[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][0]
        for g in foot_geoms
    )
    mj_data.qpos[2] -= min_z
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Set ctrl = current joint angles
    for i in range(mj_model.nu):
        jname = mj_model.actuator(i).name
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            mj_data.ctrl[i] = mj_data.qpos[int(mj_model.jnt_qposadr[jid])]

    ctrl_dt = 0.02
    substeps = max(1, int(round(ctrl_dt / mj_model.opt.timestep)))
    init_h = float(mj_data.qpos[2])

    pitch_max = 0.0
    pitch_at_end = 0.0
    survived = True
    lk_act = act_idx("left_knee_pitch")
    lk_addr = jnt_addr("left_knee_pitch")

    for step in range(1, n_steps + 1):
        for _ in range(substeps):
            mujoco.mj_step(mj_model, mj_data)
        qw, qx, qy, qz = mj_data.qpos[3:7]
        pitch = float(np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2)))
        pitch_max = max(pitch_max, abs(pitch))
        pitch_at_end = pitch
        if abs(pitch) > 0.82 or mj_data.qpos[2] < 0.15:
            survived = False
            break

    lk_q_final = float(mj_data.qpos[lk_addr])
    return {
        "target_x": target_x, "target_z": target_z,
        "hip": hip, "knee": knee, "ankle": ankle,
        "init_h": init_h, "pitch_max": pitch_max, "pitch_end": pitch_at_end,
        "knee_final": lk_q_final, "knee_err": knee - lk_q_final,
        "survived": survived, "steps": step,
    }


def main():
    target_z = -0.39  # support_height(0.41) - margin(0.02)
    print(f"Sweeping target_x for target_z={target_z:.3f}, 200 steps (4s)")
    print()
    print(f"{'tx':>6} {'hip':>7} {'knee':>7} {'ankle':>7} | {'survived':>8} {'steps':>5} {'pitch_max':>9} {'pitch_end':>9} {'knee_end':>8} {'knee_err':>8}")
    print("-" * 95)

    for tx in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        r = test_posture(tx, target_z)
        print(
            f"{r['target_x']:6.3f} {r['hip']:+7.3f} {r['knee']:+7.3f} {r['ankle']:+7.3f} | "
            f"{'YES' if r['survived'] else 'NO':>8} {r['steps']:5d} {r['pitch_max']:+9.4f} {r['pitch_end']:+9.4f} {r['knee_final']:+8.4f} {r['knee_err']:+8.4f}"
        )


if __name__ == "__main__":
    main()
