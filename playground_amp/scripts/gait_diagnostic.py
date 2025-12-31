#!/usr/bin/env python3
"""
Comprehensive gait diagnostic for analyzing trained walking policies.

Extracts:
- Per-joint torque time series (mean, max, RMS)
- Foot contact duty factor (stance ratio)
- Gait phase offset between legs
- Torque distribution analysis

Usage:
    uv run python playground_amp/scripts/gait_diagnostic.py --checkpoint <path>
"""

import argparse
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.training.ppo_core import create_networks, sample_actions
from playground_amp.configs.training_config import load_training_config, load_robot_config


DEFAULT_CHECKPOINT = "playground_amp/checkpoints/wildrobot_ppo_20251228_205536/checkpoint_520_68157440.pkl"
DEFAULT_CONFIG = "playground_amp/configs/ppo_walking.yaml"
MODEL_PATH = "assets/scene_flat_terrain.xml"
NUM_STEPS = 500  # 10 seconds at 50Hz
VELOCITY_CMD = 0.65
CONTACT_THRESHOLD = 5.0  # N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    args = parser.parse_args()

    # Load configs
    training_cfg = load_training_config(args.config)
    robot_cfg = load_robot_config(project_root / "assets" / "robot_config.yaml")

    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Actuator names
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    # Get foot geom IDs for contact detection
    left_toe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    left_heel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel")
    right_toe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")
    right_heel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    print(f"Foot geoms: left_toe={left_toe_id}, left_heel={left_heel_id}, right_toe={right_toe_id}, right_heel={right_heel_id}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)

    ckpt_iter = checkpoint.get("iteration", "?")
    ckpt_metrics = checkpoint.get("metrics", {})
    print(f"  Iteration: {ckpt_iter}")
    if ckpt_metrics:
        print(f"  Reward: {ckpt_metrics.get('episode_reward', 'N/A'):.2f}")
        print(f"  Velocity: {ckpt_metrics.get('forward_velocity', 'N/A'):.2f} m/s")

    # Create network
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
    print("JIT warmup done.\n")

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
    use_action_filter = training_cfg.env.use_action_filter
    action_filter_alpha = training_cfg.env.action_filter_alpha
    prev_action = np.zeros(action_dim, dtype=np.float32)

    # Physics substeps
    ctrl_dt = training_cfg.env.ctrl_dt
    sim_dt = model.opt.timestep
    n_substeps = int(ctrl_dt / sim_dt)

    def get_observation(data, velocity_cmd, prev_action):
        quat = data.sensordata[orientation_adr:orientation_adr+4].copy()
        w, x, y, z = quat
        r = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )
        gravity = (r.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
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

    def get_foot_contacts(data):
        """Get contact forces for each foot."""
        left_force = 0.0
        right_force = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check if floor is involved
            if geom1 != floor_id and geom2 != floor_id:
                continue

            other_geom = geom1 if geom2 == floor_id else geom2

            # Get contact force magnitude
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            force_mag = np.linalg.norm(force[:3])

            if other_geom in [left_toe_id, left_heel_id]:
                left_force += force_mag
            elif other_geom in [right_toe_id, right_heel_id]:
                right_force += force_mag

        return left_force, right_force

    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Storage
    torque_history = []
    action_history = []
    joint_pos_history = []
    joint_vel_history = []
    left_contact_history = []
    right_contact_history = []
    velocity_history = []
    height_history = []

    print(f"Running policy for {args.steps} steps ({args.steps * ctrl_dt:.1f}s)...")
    print("=" * 80)

    for step in range(args.steps):
        # Get observation
        obs = get_observation(data, VELOCITY_CMD, prev_action)

        # Get action
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

        # Record data
        torque_history.append(data.actuator_force.copy())
        action_history.append(filtered_action.copy())
        joint_pos_history.append(data.qpos[7:7+8].copy())
        joint_vel_history.append(data.qvel[6:6+8].copy())

        left_f, right_f = get_foot_contacts(data)
        left_contact_history.append(left_f)
        right_contact_history.append(right_f)

        velocity_history.append(data.sensordata[linvel_adr])
        height_history.append(data.qpos[2])

        # Check termination
        if data.qpos[2] < 0.2:
            print(f"\n⚠️ Robot fell at step {step}")
            break

    # Convert to arrays
    torques = np.array(torque_history)
    actions = np.array(action_history)
    joint_pos = np.array(joint_pos_history)
    joint_vel = np.array(joint_vel_history)
    left_contacts = np.array(left_contact_history)
    right_contacts = np.array(right_contact_history)
    velocities = np.array(velocity_history)
    heights = np.array(height_history)

    n_steps = len(torques)
    print(f"\nCompleted {n_steps} steps")

    # =========================================================================
    # TORQUE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TORQUE ANALYSIS (per joint)")
    print("=" * 80)
    print(f"{'Joint':<20} {'Mean':>8} {'RMS':>8} {'Max':>8} {'Sat%':>8} {'Range':>12}")
    print("-" * 80)

    torque_limit = 4.0
    for i, name in enumerate(actuator_names):
        t = torques[:, i]
        mean_t = np.mean(np.abs(t))
        rms_t = np.sqrt(np.mean(t**2))
        max_t = np.max(np.abs(t))
        sat_pct = 100 * np.sum(np.abs(t) > 3.8) / n_steps

        print(f"{name:<20} {mean_t:>8.2f} {rms_t:>8.2f} {max_t:>8.2f} {sat_pct:>7.1f}% [{np.min(t):>5.2f}, {np.max(t):>5.2f}]")

    # Joint group analysis
    print("\n" + "-" * 80)
    print("TORQUE BY JOINT GROUP")
    print("-" * 80)

    groups = {
        "Hip Pitch": [0, 4],  # left_hip_pitch, right_hip_pitch
        "Hip Roll": [1, 5],   # left_hip_roll, right_hip_roll
        "Knee": [2, 6],       # left_knee, right_knee
        "Ankle": [3, 7],      # left_ankle, right_ankle
    }

    for group_name, indices in groups.items():
        group_torques = torques[:, indices].flatten()
        mean_t = np.mean(np.abs(group_torques))
        rms_t = np.sqrt(np.mean(group_torques**2))
        max_t = np.max(np.abs(group_torques))
        sat_pct = 100 * np.sum(np.abs(group_torques) > 3.8) / len(group_torques)
        print(f"{group_name:<15} mean={mean_t:>5.2f}  RMS={rms_t:>5.2f}  max={max_t:>5.2f}  sat={sat_pct:>5.1f}%")

    # =========================================================================
    # FOOT CONTACT & GAIT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GAIT ANALYSIS")
    print("=" * 80)

    left_stance = left_contacts > CONTACT_THRESHOLD
    right_stance = right_contacts > CONTACT_THRESHOLD

    # Duty factor (stance ratio)
    left_duty = np.mean(left_stance)
    right_duty = np.mean(right_stance)
    double_support = np.mean(left_stance & right_stance)
    flight = np.mean(~left_stance & ~right_stance)
    alternating = np.mean(left_stance ^ right_stance)

    print(f"\nDuty Factor (stance ratio):")
    print(f"  Left foot:  {left_duty:.1%} stance")
    print(f"  Right foot: {right_duty:.1%} stance")
    print(f"  Double support: {double_support:.1%}")
    print(f"  Flight phase: {flight:.1%}")
    print(f"  Alternating (XOR): {alternating:.1%}  <- gait_periodicity reward target")

    # Phase offset analysis (find dominant gait frequency)
    print(f"\nGait Phase Analysis:")

    # Compute cross-correlation to find phase offset
    left_centered = left_stance.astype(float) - left_stance.mean()
    right_centered = right_stance.astype(float) - right_stance.mean()

    if np.std(left_centered) > 0 and np.std(right_centered) > 0:
        cross_corr = np.correlate(left_centered, right_centered, mode='full')
        cross_corr = cross_corr / (np.std(left_centered) * np.std(right_centered) * n_steps)
        lags = np.arange(-n_steps + 1, n_steps)

        # Find lag at minimum correlation (anti-phase = walking)
        min_lag_idx = np.argmin(cross_corr)
        min_lag = lags[min_lag_idx]
        phase_offset_deg = (min_lag / n_steps) * 360

        # Find lag at maximum correlation (in-phase = hopping)
        max_lag_idx = np.argmax(cross_corr)
        max_lag = lags[max_lag_idx]

        print(f"  Anti-phase lag: {min_lag} steps ({min_lag * ctrl_dt * 1000:.0f}ms)")
        print(f"  In-phase lag: {max_lag} steps")

        if abs(min_lag) > 10 and abs(min_lag) < n_steps // 2:
            gait_period = abs(min_lag) * 2  # Full gait cycle
            gait_freq = 1.0 / (gait_period * ctrl_dt)
            print(f"  Estimated gait frequency: {gait_freq:.2f} Hz (period={gait_period * ctrl_dt:.2f}s)")
    else:
        print("  Could not compute phase offset (constant contact pattern)")

    # =========================================================================
    # JOINT RANGE USAGE
    # =========================================================================
    print("\n" + "=" * 80)
    print("JOINT RANGE USAGE")
    print("=" * 80)

    joint_limits = {
        "left_hip_pitch": (-5, 90),
        "left_hip_roll": (-90, 10),
        "left_knee_pitch": (0, 80),
        "left_ankle_pitch": (-45, 45),
        "right_hip_pitch": (-90, 5),
        "right_hip_roll": (-10, 90),
        "right_knee_pitch": (0, 80),
        "right_ankle_pitch": (-45, 45),
    }

    print(f"{'Joint':<20} {'Mean':>8} {'Min':>8} {'Max':>8} {'Range':>8} {'Limit':>12} {'Used%':>8}")
    print("-" * 80)

    for i, name in enumerate(actuator_names):
        pos = np.degrees(joint_pos[:, i])
        mean_p = np.mean(pos)
        min_p = np.min(pos)
        max_p = np.max(pos)
        range_p = max_p - min_p

        # Get limits for this joint
        joint_name = name.replace("_pitch", "_pitch").replace("_roll", "_roll")
        limits = joint_limits.get(joint_name, (-90, 90))
        limit_range = limits[1] - limits[0]
        used_pct = 100 * range_p / limit_range if limit_range > 0 else 0

        print(f"{name:<20} {mean_p:>7.1f}° {min_p:>7.1f}° {max_p:>7.1f}° {range_p:>7.1f}° [{limits[0]:>4}°,{limits[1]:>3}°] {used_pct:>7.1f}%")

    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nVelocity: {np.mean(velocities):.2f} m/s (target: {VELOCITY_CMD})")
    print(f"Height: {np.mean(heights):.3f} m")
    print(f"Episode length: {n_steps} steps ({n_steps * ctrl_dt:.1f}s)")

    # Identify torque hotspots
    print("\nTorque hotspots (by RMS):")
    rms_by_joint = [(name, np.sqrt(np.mean(torques[:, i]**2))) for i, name in enumerate(actuator_names)]
    rms_by_joint.sort(key=lambda x: -x[1])
    for name, rms in rms_by_joint[:3]:
        print(f"  - {name}: RMS={rms:.2f} Nm")

    # Gait quality assessment
    print("\nGait quality:")
    if alternating > 0.7:
        print(f"  ✅ Good alternating gait ({alternating:.1%})")
    elif alternating > 0.4:
        print(f"  ⚠️ Partial alternating gait ({alternating:.1%})")
    else:
        print(f"  ❌ Poor gait pattern ({alternating:.1%} alternating)")

    if double_support > 0.5:
        print(f"  ⚠️ High double support ({double_support:.1%}) - shuffling gait")
    if flight > 0.1:
        print(f"  ⚠️ Has flight phase ({flight:.1%}) - running/hopping")


if __name__ == "__main__":
    main()
