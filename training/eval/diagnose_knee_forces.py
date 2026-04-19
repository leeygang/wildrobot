#!/usr/bin/env python3
"""Diagnose why knee servos don't track targets during startup.

Prints per-step actuator forces, contact forces, and joint state for
the stance-leg knee to identify what resists knee flexion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.envs.env_info import WR_INFO_KEY


def main() -> None:
    # v0.20.1: ppo_walking_v0193a.yaml was deleted; pass an explicit
    # config via this script's CLI (currently hardcoded -- update when
    # the smoke YAML or a v0.20.x diagnostic config lands).
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke.yaml")
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)

    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 60
    cfg.env.min_velocity = 0.10
    cfg.env.max_velocity = 0.10
    cfg.env.loc_ref_enabled = True
    cfg.env.controller_stack = "ppo"
    cfg.env.base_ctrl_enabled = False
    cfg.env.fsm_enabled = False
    cfg.env.push_enabled = False
    cfg.env.action_delay_steps = 0
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    zero_action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    step_fn = jax.jit(lambda s: env.step(s, zero_action))

    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)

    # Find actuator indices for left knee and right knee
    actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    left_knee_act_id = next(i for i, n in enumerate(actuator_names) if "left_knee_pitch" in n)
    right_knee_act_id = next(i for i, n in enumerate(actuator_names) if "right_knee_pitch" in n)
    left_ankle_act_id = next(i for i, n in enumerate(actuator_names) if "left_ankle_pitch" in n)
    right_ankle_act_id = next(i for i, n in enumerate(actuator_names) if "right_ankle_pitch" in n)

    # Find joint indices
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    left_knee_jnt_id = next(i for i, n in enumerate(joint_names) if "left_knee_pitch" in n)
    right_knee_jnt_id = next(i for i, n in enumerate(joint_names) if "right_knee_pitch" in n)
    left_ankle_jnt_id = next(i for i, n in enumerate(joint_names) if "left_ankle_pitch" in n)
    right_ankle_jnt_id = next(i for i, n in enumerate(joint_names) if "right_ankle_pitch" in n)

    # Get actuator kp and forcerange
    left_knee_kp = float(mj_model.actuator_gainprm[left_knee_act_id, 0])
    left_knee_frange = mj_model.actuator_forcerange[left_knee_act_id]
    print(f"Left knee actuator: kp={left_knee_kp:.1f}  forcerange=[{left_knee_frange[0]:.1f}, {left_knee_frange[1]:.1f}]")
    print(f"Right knee actuator: kp={float(mj_model.actuator_gainprm[right_knee_act_id, 0]):.1f}")
    print()

    header = (
        f"{'step':>4} | {'L_knee_ctrl':>10} {'L_knee_q':>9} {'L_knee_err':>10} "
        f"{'L_knee_F':>9} {'L_knee_Fsat':>10} | "
        f"{'R_knee_ctrl':>10} {'R_knee_q':>9} {'R_knee_err':>10} "
        f"{'R_knee_F':>9} {'R_knee_Fsat':>10} | "
        f"{'L_ank_ctrl':>10} {'L_ank_q':>9} {'L_ank_F':>9} | "
        f"{'root_h':>7} {'pitch':>7}"
    )
    print(header)
    print("-" * len(header))

    for step in range(1, 41):
        state = step_fn(state)

        # Sync to CPU MjData for force inspection
        qpos_np = np.asarray(state.data.qpos, dtype=np.float64)
        qvel_np = np.asarray(state.data.qvel, dtype=np.float64)
        ctrl_np = np.asarray(state.data.ctrl, dtype=np.float64)
        mj_data.qpos[:] = qpos_np
        mj_data.qvel[:] = qvel_np
        mj_data.ctrl[:] = ctrl_np
        mujoco.mj_forward(mj_model, mj_data)

        # Joint positions
        lk_q = float(mj_data.qpos[mj_model.jnt_qposadr[left_knee_jnt_id]])
        rk_q = float(mj_data.qpos[mj_model.jnt_qposadr[right_knee_jnt_id]])
        la_q = float(mj_data.qpos[mj_model.jnt_qposadr[left_ankle_jnt_id]])

        # Ctrl targets
        lk_ctrl = float(mj_data.ctrl[left_knee_act_id])
        rk_ctrl = float(mj_data.ctrl[right_knee_act_id])
        la_ctrl = float(mj_data.ctrl[left_ankle_act_id])

        # Actuator forces (what MuJoCo actually applies after clamping)
        lk_force = float(mj_data.actuator_force[left_knee_act_id])
        rk_force = float(mj_data.actuator_force[right_knee_act_id])
        la_force = float(mj_data.actuator_force[left_ankle_act_id])

        # Compute demanded force before clamping
        lk_vel = float(mj_data.qvel[mj_model.jnt_dofadr[left_knee_jnt_id]])
        rk_vel = float(mj_data.qvel[mj_model.jnt_dofadr[right_knee_jnt_id]])
        lk_kp = float(mj_model.actuator_gainprm[left_knee_act_id, 0])
        lk_kv = float(-mj_model.actuator_biasprm[left_knee_act_id, 2])
        lk_demanded = lk_kp * (lk_ctrl - lk_q) - lk_kv * lk_vel
        rk_kp = float(mj_model.actuator_gainprm[right_knee_act_id, 0])
        rk_kv = float(-mj_model.actuator_biasprm[right_knee_act_id, 2])
        rk_demanded = rk_kp * (rk_ctrl - rk_q) - rk_kv * rk_vel

        # Saturation flag
        lk_sat = "SAT" if abs(lk_demanded) > abs(lk_force) + 0.01 else ""
        rk_sat = "SAT" if abs(rk_demanded) > abs(rk_force) + 0.01 else ""

        # Root height and pitch
        root_h = float(qpos_np[2])
        # Rough pitch from quaternion
        qw, qx, qy, qz = qpos_np[3], qpos_np[4], qpos_np[5], qpos_np[6]
        pitch = float(np.arctan2(2 * (qw * qy - qz * qx), 1 - 2 * (qx**2 + qy**2)))

        print(
            f"{step:4d} | "
            f"{lk_ctrl:+10.4f} {lk_q:+9.4f} {lk_ctrl-lk_q:+10.4f} "
            f"{lk_force:+9.3f} {lk_sat:>10s} | "
            f"{rk_ctrl:+10.4f} {rk_q:+9.4f} {rk_ctrl-rk_q:+10.4f} "
            f"{rk_force:+9.3f} {rk_sat:>10s} | "
            f"{la_ctrl:+10.4f} {la_q:+9.4f} {la_force:+9.3f} | "
            f"{root_h:7.4f} {pitch:+7.4f}"
        )


if __name__ == "__main__":
    main()
