#!/usr/bin/env python3
"""Debug script to check model structure"""
import mujoco

model = mujoco.MjModel.from_xml_path('assets/wildrobot.xml')
print('Total qpos:', model.nq)
print('Total qvel (nv):', model.nv)
print('Number of actuators:', model.nu)
print('\nJoint names in order:')
for i in range(model.njnt):
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    type_name = ['free', 'ball', 'slide', 'hinge'][jnt_type]
    qpos_addr = model.jnt_qposadr[i]
    print(f'  {i}: {jnt_name} ({type_name}) - qpos addr: {qpos_addr}')

print('\nActuator names in order:')
for i in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    print(f'  {i}: {act_name} -> joint: {jnt_name}')
