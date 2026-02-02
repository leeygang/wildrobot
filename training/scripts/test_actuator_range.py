#!/usr/bin/env python3
"""
Quick test to verify actuator response and check if model can physically bend knees/hips.
This helps distinguish between model/actuator issues vs policy/reward issues.
"""

import mujoco
import numpy as np
import time

# Load model
model_path = "assets/v1/scene_flat_terrain.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Print actuator info
print("=" * 60)
print("Actuator Configuration")
print("=" * 60)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    # Get actuator properties
    ctrl_range = model.actuator_ctrlrange[i]
    force_range = model.actuator_forcerange[i]
    gainprm = model.actuator_gainprm[i]  # kp is first element
    biasprm = model.actuator_biasprm[i]  # kv related

    print(f"{i}: {name:20s} ctrl=[{ctrl_range[0]:6.2f}, {ctrl_range[1]:6.2f}] "
          f"force=[{force_range[0]:5.1f}, {force_range[1]:5.1f}] kp={gainprm[0]:.1f}")

print("\n" + "=" * 60)
print("Joint Ranges (from model)")
print("=" * 60)

# Find joint ranges for actuated joints
joint_names = ["left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle",
               "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle"]

for jname in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        jnt_range = model.jnt_range[jid]
        print(f"{jname:20s}: [{np.degrees(jnt_range[0]):6.1f}°, {np.degrees(jnt_range[1]):6.1f}°]")

# Reset to keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("\n" + "=" * 60)
print("Initial State (keyframe)")
print("=" * 60)
print(f"Height: {data.qpos[2]:.3f}m")
print(f"Ctrl: {data.ctrl}")

# Test 1: Try to bend knees
print("\n" + "=" * 60)
print("Test 1: Bend both knees to 0.8 rad (~46°)")
print("=" * 60)

mujoco.mj_resetDataKeyframe(model, data, 0)
# Set knee targets (indices 2 and 6 for left_knee and right_knee)
target_knee = 0.8  # radians
data.ctrl[2] = target_knee  # left_knee
data.ctrl[6] = target_knee  # right_knee

# Step simulation for 2 seconds
for _ in range(1000):  # 2s at 500Hz
    mujoco.mj_step(model, data)

# Check actual joint positions
left_knee_pos = data.qpos[7 + 2]  # After 7 qpos for freejoint
right_knee_pos = data.qpos[7 + 6]

print(f"Target knee: {target_knee:.2f} rad ({np.degrees(target_knee):.1f}°)")
print(f"Left knee actual: {left_knee_pos:.3f} rad ({np.degrees(left_knee_pos):.1f}°)")
print(f"Right knee actual: {right_knee_pos:.3f} rad ({np.degrees(right_knee_pos):.1f}°)")

# Check torques used
print(f"\nActuator forces (left_knee, right_knee): {data.actuator_force[2]:.2f}, {data.actuator_force[6]:.2f}")
print(f"Force limit: ±4.0 Nm")

# Test 2: Try full hip swing
print("\n" + "=" * 60)
print("Test 2: Hip pitch swing test")
print("=" * 60)

mujoco.mj_resetDataKeyframe(model, data, 0)
# Try swinging left hip forward (positive pitch based on range)
data.ctrl[0] = 0.8  # left_hip_pitch forward
data.ctrl[4] = -0.8  # right_hip_pitch back (opposite direction due to asymmetric range)

for _ in range(1000):
    mujoco.mj_step(model, data)

left_hip_pos = data.qpos[7 + 0]
right_hip_pos = data.qpos[7 + 4]

print(f"Left hip pitch: {left_hip_pos:.3f} rad ({np.degrees(left_hip_pos):.1f}°)")
print(f"Right hip pitch: {right_hip_pos:.3f} rad ({np.degrees(right_hip_pos):.1f}°)")
print(f"Hip forces: {data.actuator_force[0]:.2f}, {data.actuator_force[4]:.2f}")

# Test 3: Check max torque scenario - deep squat
print("\n" + "=" * 60)
print("Test 3: Deep squat (maximum knee bend)")
print("=" * 60)

mujoco.mj_resetDataKeyframe(model, data, 0)
# Max knee bend
data.ctrl[2] = 1.2  # left_knee max
data.ctrl[6] = 1.2  # right_knee max
# Slight hip bend to compensate
data.ctrl[0] = 0.3
data.ctrl[4] = 0.3

for _ in range(1000):
    mujoco.mj_step(model, data)

print(f"Height: {data.qpos[2]:.3f}m (initial was ~0.47m)")
print(f"Left knee: {data.qpos[7+2]:.3f} rad ({np.degrees(data.qpos[7+2]):.1f}°)")
print(f"Right knee: {data.qpos[7+6]:.3f} rad ({np.degrees(data.qpos[7+6]):.1f}°)")
print(f"Knee forces: {data.actuator_force[2]:.2f}, {data.actuator_force[6]:.2f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if abs(left_knee_pos - target_knee) < 0.1 and abs(right_knee_pos - target_knee) < 0.1:
    print("✅ Knees can bend to target - Model/actuators are OK")
    print("   → Issue is policy/reward, not model")
else:
    print("⚠️ Knees cannot reach target")
    print(f"   Gap: {abs(left_knee_pos - target_knee):.2f} rad")
    print("   → Check actuator force limits or kp gains")
