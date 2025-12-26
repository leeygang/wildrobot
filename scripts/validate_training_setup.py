#!/usr/bin/env python3
"""Validate MuJoCo setup before training.

Run this script before starting RL training to catch configuration issues early.

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/validate_training_setup.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from playground_amp.envs.wildrobot_env import get_assets


def validate_setup(model_path: str = "assets/scene_flat_terrain.xml"):
    """Run all validation checks."""
    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    mj_data = mujoco.MjData(mj_model)

    errors = []
    warnings = []

    print("=" * 60)
    print("MUJOCO TRAINING SETUP VALIDATION")
    print("=" * 60)

    # 1. Coordinate System
    print("\n1. COORDINATE SYSTEM")
    gravity = mj_model.opt.gravity
    print(f"   Gravity: {gravity}")
    if not np.allclose(gravity, [0, 0, -9.81], atol=0.1):
        errors.append("Gravity should be [0, 0, -9.81] for Z-up world")
    else:
        print("   ✓ Z-up world (gravity along -Z)")

    # 2. Home Keyframe
    print("\n2. HOME KEYFRAME")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    home_pos = mj_data.qpos[0:3].copy()
    home_quat = mj_data.qpos[3:7].copy()
    print(f"   Position: {home_pos}")
    print(f"   Quaternion (wxyz): {home_quat}")

    # Check robot is roughly upright
    w, x, y, z = home_quat
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))

    if abs(pitch) > 5 or abs(roll) > 5:
        warnings.append(f"Home keyframe not upright: pitch={pitch:.1f}°, roll={roll:.1f}°")
    else:
        print(f"   ✓ Upright (pitch={pitch:.1f}°, roll={roll:.1f}°)")

    # 3. Joint Limits
    print("\n3. JOINT LIMITS & ACTUATORS")
    for i in range(mj_model.nu):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jnt_id = mj_model.actuator_trnid[i, 0]
        jnt_range = mj_model.jnt_range[jnt_id]
        ctrl_range = mj_model.actuator_ctrlrange[i]
        if not np.allclose(jnt_range, ctrl_range):
            errors.append(f"{act_name}: jnt_range != ctrl_range")
            print(f"   ✗ {act_name}: MISMATCH")
        else:
            print(f"   ✓ {act_name}: [{jnt_range[0]:+.3f}, {jnt_range[1]:+.3f}]")

    # 4. Actuator Gains
    print("\n4. ACTUATOR GAINS")
    for i in range(mj_model.nu):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        kp = mj_model.actuator_gainprm[i, 0]
        kv = mj_model.actuator_biasprm[i, 2]  # damping term
        print(f"   {act_name}: kp={kp:.1f}, kv={abs(kv):.1f}")
        if kp == 0:
            warnings.append(f"{act_name}: kp=0 (no position control)")

    # 5. Stability Test
    print("\n5. STATIC STABILITY TEST (2 seconds at home pose)")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mj_data.ctrl[:] = mj_data.qpos[7 : 7 + mj_model.nu]

    mj_model.opt.timestep = 0.002
    initial_height = mj_data.qpos[2]

    for _ in range(1000):  # 2 seconds
        mujoco.mj_step(mj_model, mj_data)

    final_height = mj_data.qpos[2]
    w, x, y, z = mj_data.qpos[3:7]
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    final_pitch = np.degrees(np.arcsin(sinp))
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    final_roll = np.degrees(np.arctan2(sinr, cosr))

    print(f"   Initial height: {initial_height:.4f}m")
    print(f"   Final height: {final_height:.4f}m")
    print(f"   Final pitch: {final_pitch:.1f}°, roll: {final_roll:.1f}°")

    if abs(final_pitch) > 15 or abs(final_roll) > 15:
        errors.append(f"Robot fell during stability test (pitch={final_pitch:.1f}°, roll={final_roll:.1f}°)")
        print("   ✗ UNSTABLE - Robot fell!")
    elif final_height < 0.3:
        errors.append(f"Robot collapsed during stability test (height={final_height:.4f}m)")
        print("   ✗ UNSTABLE - Robot collapsed!")
    else:
        print("   ✓ STABLE")

    # 6. Contact Detection
    print("\n6. GROUND CONTACT")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    foot_geoms = ["left_toe", "left_heel", "right_toe", "right_heel"]
    foot_geom_ids = []
    for name in foot_geoms:
        try:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            foot_geom_ids.append(gid)
        except:
            warnings.append(f"Foot geom '{name}' not found")

    contacts_found = 0
    total_force = 0
    wrench = np.zeros(6)
    for ci in range(mj_data.ncon):
        contact = mj_data.contact[ci]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 == 0 and g2 in foot_geom_ids) or (g2 == 0 and g1 in foot_geom_ids):
            contacts_found += 1
            mujoco.mj_contactForce(mj_model, mj_data, ci, wrench)
            if wrench[0] > 0:
                total_force += wrench[0]

    robot_mass = sum(mj_model.body_mass)
    expected_force = robot_mass * 9.81
    print(f"   Foot contacts: {contacts_found}")
    print(f"   Total contact force: {total_force:.1f}N (expected ~{expected_force:.1f}N)")

    if contacts_found == 0:
        errors.append("No foot-ground contacts detected!")
        print("   ✗ NO GROUND CONTACT")
    elif total_force < expected_force * 0.5:
        warnings.append(f"Low contact force ({total_force:.1f}N < {expected_force*0.5:.1f}N)")
        print("   ⚠ LOW CONTACT FORCE")
    else:
        print("   ✓ Good ground contact")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   • {e}")

    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   • {w}")

    if not errors and not warnings:
        print("\n✅ All checks passed! Ready for training.")
    elif not errors:
        print("\n⚠ Checks passed with warnings. Review before training.")
    else:
        print("\n❌ Validation failed. Fix errors before training.")

    return len(errors) == 0


if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
