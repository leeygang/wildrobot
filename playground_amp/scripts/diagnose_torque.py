#!/usr/bin/env python3
"""
Diagnose per-actuator torque usage during policy execution.
This helps identify which actuators are saturating and why.
"""

import mujoco
import numpy as np
import pickle
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.training.ppo_core import create_networks, sample_actions
from playground_amp.configs.training_config import load_training_config, load_robot_config

# Configuration
DEFAULT_CHECKPOINT = "playground_amp/checkpoints/wildrobot_ppo_20251228_011310/checkpoint_350_45875200.pkl"
DEFAULT_CONFIG = "playground_amp/configs/ppo_walking.yaml"
MODEL_PATH = "assets/scene_flat_terrain.xml"
NUM_STEPS = 500
VELOCITY_CMD = 0.65

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # Load configs
    training_cfg = load_training_config(args.config)
    robot_cfg = load_robot_config(project_root / "assets" / "robot_config.yaml")

    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Actuator names for logging
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)

    # Create network matching training config
    obs_dim = 35
    action_dim = 8
    policy_hidden = tuple(training_cfg.networks.actor.hidden_sizes)
    value_hidden = tuple(training_cfg.networks.critic.hidden_sizes)

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=policy_hidden,
        value_hidden_dims=value_hidden,
    )

    # Extract policy params from checkpoint
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ())
    if processor_params is None:
        processor_params = ()

    # JIT compile
    @jax.jit
    def get_action(obs, rng):
        obs_batch = obs[None, ...]
        action, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs_batch, rng, deterministic=True
        )
        return action[0]

    # Warmup JIT
    print("Warming up JIT...")
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros(obs_dim)
    _ = get_action(dummy_obs, rng)
    print("JIT warmup done.")

    # Sensor addresses
    def get_sensor_id(name):
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name == name:
                return i
        return -1

    orientation_sensor_id = get_sensor_id(robot_cfg.orientation_sensor)
    linvel_sensor_id = get_sensor_id(robot_cfg.local_linvel_sensor)

    if orientation_sensor_id < 0:
        raise ValueError("Missing required orientation sensor for gravity computation")
    if linvel_sensor_id < 0:
        raise ValueError("Missing required local_linvel sensor for observation")

    orientation_adr = model.sensor_adr[orientation_sensor_id]
    linvel_adr = model.sensor_adr[linvel_sensor_id]

    # Action filter settings
    # action_filter_alpha > 0 means filtering is enabled
    action_filter_alpha = training_cfg.env.action_filter_alpha
    use_action_filter = action_filter_alpha > 0
    prev_action = np.zeros(action_dim, dtype=np.float32)

    # Physics substeps
    ctrl_dt = training_cfg.env.ctrl_dt
    sim_dt = model.opt.timestep
    n_substeps = int(ctrl_dt / sim_dt)

    def get_observation(data, velocity_cmd, prev_action):
        """Build observation matching training format."""
        quat = data.sensordata[orientation_adr:orientation_adr+4].copy()
        w, x, y, z = quat
        # R is the rotation matrix from quaternion (body-from-world)
        r = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )
        # R @ world_gravity transforms world gravity into body frame (matches CAL)
        gravity = (r @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
        angvel = data.qvel[3:6].copy()
        linvel = data.sensordata[linvel_adr:linvel_adr+3].copy()

        joint_pos = data.qpos[7:7+8].copy()
        joint_vel = data.qvel[6:6+8].copy()

        obs = np.concatenate([
            gravity, angvel, linvel,
            joint_pos, joint_vel,
            prev_action,
            [velocity_cmd], [0.0]
        ])
        return obs.astype(np.float32)

    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Storage for analysis
    torque_history = []
    action_history = []
    joint_pos_history = []
    saturation_count = np.zeros(8)

    print(f"\nRunning policy for {NUM_STEPS} steps...")
    print("=" * 80)

    for step in range(NUM_STEPS):
        # Get observation
        obs = get_observation(data, VELOCITY_CMD, prev_action)

        # Get action from policy
        rng, action_rng = jax.random.split(rng)
        action = np.array(get_action(jnp.array(obs), action_rng))

        # Apply action filter
        if use_action_filter:
            filtered_action = action_filter_alpha * prev_action + (1 - action_filter_alpha) * action
        else:
            filtered_action = action
        prev_action = filtered_action.copy()

        # Apply to simulation
        data.ctrl[:] = filtered_action

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        # Record torques and positions
        torques = data.actuator_force.copy()
        torque_history.append(torques)
        action_history.append(filtered_action.copy())
        joint_pos_history.append(data.qpos[7:7+8].copy())

        # Count saturation (>3.8 Nm, i.e. >95% of 4.0)
        saturation_count += (np.abs(torques) > 3.8).astype(float)

        # Print periodic updates
        if step % 100 == 0:
            height = data.qpos[2]
            vel = data.sensordata[linvel_adr]
            print(f"Step {step:4d}: height={height:.3f}m, vel={vel:.2f}m/s")
            print(f"  Torques: {' '.join([f'{t:5.2f}' for t in torques])}")
            print(f"  Actions: {' '.join([f'{a:5.2f}' for a in filtered_action])}")

        # Check termination
        if data.qpos[2] < 0.2:
            print(f"\n⚠️ Robot fell at step {step}")
            break

    # Analysis
    torque_array = np.array(torque_history)
    action_array = np.array(action_history)
    joint_pos_array = np.array(joint_pos_history)

    print("\n" + "=" * 80)
    print("TORQUE ANALYSIS")
    print("=" * 80)

    for i, name in enumerate(actuator_names):
        torques = torque_array[:, i]
        actions = action_array[:, i]
        positions = joint_pos_array[:, i]

        sat_pct = 100 * saturation_count[i] / len(torque_history)
        print(f"\n{name}:")
        print(f"  Torque: mean={np.mean(np.abs(torques)):5.2f}, max={np.max(np.abs(torques)):5.2f}, "
              f"sat={sat_pct:4.1f}%")
        print(f"  Action: mean={np.mean(actions):5.2f}, range=[{np.min(actions):5.2f}, {np.max(actions):5.2f}]")
        print(f"  Position: mean={np.degrees(np.mean(positions)):5.1f}°, "
              f"range=[{np.degrees(np.min(positions)):5.1f}°, {np.degrees(np.max(positions)):5.1f}°]")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall_sat = 100 * np.sum(saturation_count) / (len(torque_history) * 8)
    print(f"Overall saturation: {overall_sat:.1f}%")

    # Identify problematic actuators
    high_sat = [(name, 100*saturation_count[i]/len(torque_history))
                for i, name in enumerate(actuator_names)
                if saturation_count[i] > len(torque_history) * 0.5]

    if high_sat:
        print("\n⚠️ High saturation actuators (>50%):")
        for name, pct in sorted(high_sat, key=lambda x: -x[1]):
            print(f"  - {name}: {pct:.1f}%")
    else:
        print("\n✅ No actuators with >50% saturation")

    # Check action range usage
    print("\nAction range analysis:")
    for i, name in enumerate(actuator_names):
        action_range = np.max(action_array[:, i]) - np.min(action_array[:, i])
        print(f"  {name}: uses {action_range:.2f} rad of range")

if __name__ == "__main__":
    main()
