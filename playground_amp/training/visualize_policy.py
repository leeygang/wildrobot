#!/usr/bin/env python3
"""Visualize trained WildRobot policy in MuJoCo viewer.

This script loads a trained checkpoint and runs the policy in real-time
using MuJoCo's native viewer.

Usage:
    # On macOS, use mjpython (required for viewer):
    mjpython playground_amp/training/visualize_policy.py

    # Or use headless mode to record video without display:
    uv run python playground_amp/training/visualize_policy.py --headless --record output.mp4

    # Visualize specific checkpoint
    mjpython playground_amp/training/visualize_policy.py --checkpoint path/to/checkpoint.pkl

    # Visualize with different config
    mjpython playground_amp/training/visualize_policy.py --config playground_amp/configs/ppo_walking.yaml

    # With velocity command (for walking)
    mjpython playground_amp/training/visualize_policy.py --velocity-cmd 0.5

Controls (viewer mode):
    - Space: Pause/Resume
    - Backspace: Reset environment
    - Escape: Exit
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path (training/ -> playground_amp/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.configs.training_config import (
    load_robot_config,
    load_training_config,
)
from playground_amp.configs.robot_config import get_robot_config
from playground_amp.envs.wildrobot_env import ObsLayout
from playground_amp.training.ppo_core import create_networks, sample_actions


# Default paths (relative to project_root)
DEFAULT_CONFIG_PATH = project_root / "playground_amp" / "configs" / "ppo_walking.yaml"
DEFAULT_CHECKPOINT_PATH = project_root / "playground_amp" / "checkpoints" / "final_ppo_policy.pkl"
DEFAULT_ROBOT_CONFIG_PATH = project_root / "assets" / "robot_config.yaml"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize trained WildRobot policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pkl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (no sampling noise)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (with sampling noise)",
    )
    parser.add_argument(
        "--velocity-cmd",
        type=float,
        default=None,
        help="Fixed velocity command (default: random from config range)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time)",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save video recording (e.g., output.mp4)",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=10.0,
        help="Duration of recording in seconds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (for recording or batch evaluation)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to run (headless mode). Default: 1 for headless, unlimited for viewer",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg: str | None) -> Path | None:
    """Resolve checkpoint path, trying multiple locations.

    Tries:
    1. Absolute path as-is
    2. Relative to current working directory
    3. Relative to project root
    4. Default checkpoint path

    Args:
        checkpoint_arg: Path from command line or None

    Returns:
        Resolved Path if found, None otherwise
    """
    if checkpoint_arg is None:
        # Use default
        if DEFAULT_CHECKPOINT_PATH.exists():
            return DEFAULT_CHECKPOINT_PATH
        return None

    path = Path(checkpoint_arg)

    # Try as-is (handles absolute paths and correct relative paths)
    if path.exists():
        return path.resolve()

    # Try relative to project root
    project_relative = project_root / path
    if project_relative.exists():
        return project_relative.resolve()

    # Try relative to project root with playground_amp prefix stripped/added
    if str(path).startswith("playground_amp/"):
        without_prefix = project_root / Path(str(path).replace("playground_amp/", "", 1))
        if without_prefix.exists():
            return without_prefix.resolve()
    else:
        with_prefix = project_root / "playground_amp" / path
        if with_prefix.exists():
            return with_prefix.resolve()

    return None


def list_available_checkpoints() -> list[Path]:
    """Find all available checkpoint files."""
    checkpoints = []

    # Search in common locations
    search_paths = [
        project_root / "playground_amp" / "checkpoints",
        project_root / "checkpoints",
        Path.cwd() / "playground_amp" / "checkpoints",
        Path.cwd() / "checkpoints",
    ]

    for search_path in search_paths:
        if search_path.exists():
            checkpoints.extend(search_path.glob("**/*.pkl"))

    # Deduplicate by resolving to absolute paths
    seen = set()
    unique = []
    for p in checkpoints:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    return sorted(unique, key=lambda p: p.stat().st_mtime, reverse=True)


def main():
    """Main visualization loop."""
    args = parse_args()

    # Determine deterministic mode
    deterministic = not args.stochastic

    # Load robot config
    if DEFAULT_ROBOT_CONFIG_PATH.exists():
        robot_cfg = load_robot_config(DEFAULT_ROBOT_CONFIG_PATH)
        print(f"Loaded robot config: {robot_cfg.robot_name}")
    else:
        print(f"Error: Robot config not found at {DEFAULT_ROBOT_CONFIG_PATH}")
        print("Run 'cd assets && python post_process.py' to generate it.")
        return 1

    # Load training config
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    print(f"Loading config from: {config_path}")
    training_cfg = load_training_config(config_path)

    # Resolve checkpoint path (tries multiple locations)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if checkpoint_path is None:
        print(f"Error: Checkpoint not found: {args.checkpoint or 'default'}")
        print("\nSearched locations:")
        if args.checkpoint:
            print(f"  - {args.checkpoint} (as provided)")
            print(f"  - {project_root / args.checkpoint}")
        else:
            print(f"  - {DEFAULT_CHECKPOINT_PATH}")

        available = list_available_checkpoints()
        if available:
            print(f"\nAvailable checkpoints ({len(available)} found):")
            for p in available[:10]:  # Show top 10 most recent
                try:
                    mtime = p.stat().st_mtime
                    from datetime import datetime
                    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                except:
                    mtime_str = "unknown"
                print(f"  - {p} ({mtime_str})")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
        else:
            print("\nNo checkpoints found in common locations.")
        return 1

    print(f"Loading checkpoint from: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint mode: {checkpoint.get('mode', 'unknown')}")

    # Load MuJoCo model for native simulation (FAST on CPU)
    model_path = project_root / training_cfg.env.model_path
    print(f"Loading model from: {model_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)

    # Get dimensions from model
    obs_dim = 35  # Matches WildRobotEnv observation space
    action_dim = mj_model.nu  # Number of actuators

    print(f"Native MuJoCo: obs_dim={obs_dim}, action_dim={action_dim}")

    # Create PPO networks with same architecture as training
    policy_hidden = tuple(training_cfg.networks.actor.hidden_sizes)
    value_hidden = tuple(training_cfg.networks.critic.hidden_sizes)

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=policy_hidden,
        value_hidden_dims=value_hidden,
    )

    # Extract policy params from checkpoint
    # Brax expects policy_params = {'params': {...}} format
    policy_params = checkpoint["policy_params"]

    processor_params = checkpoint.get("processor_params", ())
    if processor_params is None:
        processor_params = ()

    # JIT compile the action sampling function (this is fast, ~1ms)
    @jax.jit
    def get_action(obs, rng):
        """Get action from policy."""
        obs_batch = obs[None, ...]  # Add batch dimension
        action, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs_batch, rng, deterministic
        )
        return action[0]  # Remove batch dimension

    # Warmup JIT compilation for policy network
    print("Warming up policy network JIT compilation...")
    dummy_obs = jnp.zeros(obs_dim)
    rng = jax.random.PRNGKey(42)
    rng, warmup_rng = jax.random.split(rng)
    _ = get_action(dummy_obs, warmup_rng)
    print("JIT warmup complete.")

    # Reset MuJoCo to initial state (use keyframe if available)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    # Physics substeps per control step
    ctrl_dt = training_cfg.env.ctrl_dt
    sim_dt = mj_model.opt.timestep
    n_substeps = int(ctrl_dt / sim_dt)
    print(f"Control dt: {ctrl_dt}s, Sim dt: {sim_dt}s, Substeps: {n_substeps}")

    # Get sensor addresses for native MuJoCo using robot config
    robot_cfg = get_robot_config()

    def get_sensor_id(name):
        """Get sensor ID by name."""
        for i in range(mj_model.nsensor):
            sensor_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name == name:
                return i
        return -1

    gravity_sensor_name = robot_cfg.gravity_sensor
    angvel_sensor_name = robot_cfg.global_angvel_sensor
    local_linvel_sensor_name = robot_cfg.local_linvel_sensor
    gravity_sensor_id = get_sensor_id(gravity_sensor_name)
    angvel_sensor_id = get_sensor_id(angvel_sensor_name)
    local_linvel_sensor_id = get_sensor_id(local_linvel_sensor_name)

    print(f"Using sensors: gravity={gravity_sensor_name} (id={gravity_sensor_id}), angvel={angvel_sensor_name} (id={angvel_sensor_id})")
    print(f"              local_linvel={local_linvel_sensor_name} (id={local_linvel_sensor_id})")

    if gravity_sensor_id < 0 or angvel_sensor_id < 0:
        print(f"Warning: Could not find required sensors")
        print("Available sensors:")
        for i in range(mj_model.nsensor):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            print(f"  {i}: {name}")

    # Get sensor data addresses
    gravity_adr = mj_model.sensor_adr[gravity_sensor_id] if gravity_sensor_id >= 0 else 0
    angvel_adr = mj_model.sensor_adr[angvel_sensor_id] if angvel_sensor_id >= 0 else 0
    local_linvel_adr = mj_model.sensor_adr[local_linvel_sensor_id] if local_linvel_sensor_id >= 0 else 0

    def get_forward_velocity(mj_data):
        """Get forward velocity from local_linvel sensor (body-local frame)."""
        return mj_data.sensordata[local_linvel_adr]  # x component = forward

    # Get actuator joint addresses from robot config
    # The actuator_qpos_addr and actuator_qvel_addr are the indices into qpos/qvel
    actuator_names = robot_cfg.actuator_names
    actuator_joints = robot_cfg.actuator_joints

    # Get joint qpos/qvel addresses
    actuator_qpos_addrs = []
    actuator_qvel_addrs = []
    for joint_name in actuator_joints:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            actuator_qpos_addrs.append(mj_model.jnt_qposadr[joint_id])
            actuator_qvel_addrs.append(mj_model.jnt_dofadr[joint_id])
        else:
            print(f"Warning: Joint {joint_name} not found")
            actuator_qpos_addrs.append(0)
            actuator_qvel_addrs.append(0)

    actuator_qpos_addrs = np.array(actuator_qpos_addrs)
    actuator_qvel_addrs = np.array(actuator_qvel_addrs)

    # Track previous action
    prev_action = np.zeros(action_dim, dtype=np.float32)

    def get_observation(mj_data, velocity_cmd, prev_action_in):
        """Compute observation vector using ObsLayout from training code.

        This matches the exact format used by WildRobotEnv during training.
        """
        # Gravity from sensor (pelvis_upvector - z-axis in local frame)
        gravity = mj_data.sensordata[gravity_adr:gravity_adr+3].copy()

        # Angular velocity from sensor (world frame)
        angvel_world = mj_data.sensordata[angvel_adr:angvel_adr+3].copy()

        # Linear velocity from qvel (world frame)
        linvel_world = mj_data.qvel[0:3].copy()

        # Get heading (yaw) from quaternion for heading-local transform
        quat = mj_data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat

        # Extract yaw from quaternion (ZYX convention) - same as WildRobotEnv
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        norm = np.sqrt(siny_cosp * siny_cosp + cosy_cosp * cosy_cosp + 1e-8)
        sin_yaw = siny_cosp / norm
        cos_yaw = cosy_cosp / norm

        # Rotate to heading-local frame: R_z(-yaw) @ v_world
        # linvel
        vx, vy, vz = linvel_world
        linvel = np.array([
            cos_yaw * vx + sin_yaw * vy,
            -sin_yaw * vx + cos_yaw * vy,
            vz
        ])

        # angvel
        wx, wy, wz = angvel_world
        angvel = np.array([
            cos_yaw * wx + sin_yaw * wy,
            -sin_yaw * wx + cos_yaw * wy,
            wz
        ])

        # Joint positions using same addresses as training
        joint_pos = mj_data.qpos[actuator_qpos_addrs].copy()

        # Joint velocities using same addresses as training
        joint_vel = mj_data.qvel[actuator_qvel_addrs].copy()

        # Use ObsLayout.build_obs to ensure exact format matching
        # Note: ObsLayout.build_obs uses JAX arrays, so we build manually here
        # to avoid JAX overhead in the visualization loop
        obs = np.concatenate([
            gravity,            # 3 - gravity vector
            angvel,             # 3 - angular velocity (heading-local)
            linvel,             # 3 - linear velocity (heading-local)
            joint_pos,          # 8 - joint positions
            joint_vel,          # 8 - joint velocities
            prev_action_in,     # 8 - previous action
            [velocity_cmd],     # 1 - velocity command
            [0.0],              # 1 - padding
        ])

        return obs.astype(np.float32)

    # Action filtering configuration (match training)
    use_action_filter = training_cfg.env.use_action_filter
    action_filter_alpha = training_cfg.env.action_filter_alpha if use_action_filter else 0.0
    print(f"Action filter: enabled={use_action_filter}, alpha={action_filter_alpha}")

    def apply_action(mj_data, action, prev_action_for_filter):
        """Apply action to MuJoCo model with optional low-pass filtering.

        In training, actions from policy (range [-1, 1] via tanh) are passed
        through a low-pass filter before being set as ctrl.
        MuJoCo then clamps to joint limits automatically.

        Returns filtered action for use as prev_action next step.
        """
        action_np = np.array(action)

        # Apply action filtering (low-pass) if enabled
        if use_action_filter:
            filtered_action = action_filter_alpha * prev_action_for_filter + (1.0 - action_filter_alpha) * action_np
        else:
            filtered_action = action_np

        # Pass action directly - MuJoCo handles clamping to ctrlrange
        mj_data.ctrl[:] = filtered_action
        return filtered_action

    def step_physics(mj_model, mj_data, n_substeps):
        """Step physics simulation for one control step."""
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

    def reset_robot(mj_model, mj_data):
        """Reset robot to initial state."""
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        else:
            mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

    def check_termination(mj_data, min_height=0.15):
        """Check if episode should terminate (robot fell)."""
        height = mj_data.qpos[2]
        return height < min_height

    # Generate velocity command
    if args.velocity_cmd is not None:
        velocity_cmd = args.velocity_cmd
    else:
        velocity_cmd = (training_cfg.env.min_velocity + training_cfg.env.max_velocity) / 2
    print(f"Velocity command: {velocity_cmd:.2f} m/s")

    # Control timing
    ctrl_dt = training_cfg.env.ctrl_dt

    # Determine mode
    headless = args.headless
    is_macos = sys.platform == "darwin"

    # On macOS, we'll try to launch the viewer and catch the error if mjpython isn't being used
    # This is more reliable than trying to detect mjpython in advance

    print(f"\n{'=' * 60}")
    print("WildRobot Policy Visualization")
    print(f"{'=' * 60}")
    print(f"  Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"  Render: {'Headless' if headless else 'Interactive viewer'}")
    print(f"  Velocity range: [{training_cfg.env.min_velocity:.2f}, {training_cfg.env.max_velocity:.2f}] m/s")
    if args.velocity_cmd is not None:
        print(f"  Fixed velocity cmd: {args.velocity_cmd:.2f} m/s")
    print(f"  Control dt: {ctrl_dt}s ({1/ctrl_dt:.0f} Hz)")
    if not headless:
        print(f"  Playback speed: {args.speed}x")
    print(f"{'=' * 60}")
    if not headless:
        print("\nControls:")
        print("  Space     - Pause/Resume")
        print("  Backspace - Reset")
        print("  Escape    - Exit")
        print(f"{'=' * 60}\n")

    # Recording setup
    frames = []
    record_steps = int(args.record_duration / ctrl_dt) if args.record else 0
    if args.record:
        print(f"Recording {args.record_duration}s ({record_steps} steps) to {args.record}")

    # Determine number of episodes
    max_episodes = args.num_episodes
    if max_episodes is None and headless and not args.record:
        max_episodes = 1  # Default to 1 episode in headless mode without recording

    step_count = 0
    episode_count = 0

    if headless:
        # Headless mode - run without viewer
        # Try to create renderer with appropriate backend
        renderer = None
        if args.record:
            import os
            # Try different rendering backends for headless mode
            original_backend = os.environ.get("MUJOCO_GL", None)

            backends_to_try = ["osmesa", "egl", "glfw"]
            for backend in backends_to_try:
                try:
                    os.environ["MUJOCO_GL"] = backend
                    # Force reimport of mujoco to pick up new backend
                    import importlib
                    import mujoco as mj_reimport
                    importlib.reload(mj_reimport)
                    renderer = mj_reimport.Renderer(mj_model, 640, 480)
                    print(f"  Using {backend} backend for rendering")
                    break
                except Exception as e:
                    if backend == backends_to_try[-1]:
                        # Last backend failed, try without setting env var
                        if original_backend:
                            os.environ["MUJOCO_GL"] = original_backend
                        else:
                            os.environ.pop("MUJOCO_GL", None)
                        print(f"  Warning: Could not initialize renderer ({e})")
                        print(f"  Video recording disabled. Install osmesa or run on Linux for headless rendering.")
                    continue

        print("\nRunning in headless mode (native MuJoCo - fast)...")

        max_steps = record_steps if args.record else 10000  # Limit steps if not recording
        max_episode_steps = int(training_cfg.env.episode_length / ctrl_dt) if hasattr(training_cfg.env, 'episode_length') else 500

        while step_count < max_steps:
            # Get observation from native MuJoCo state
            obs = get_observation(mj_data, velocity_cmd, prev_action)
            obs_jax = jnp.array(obs)

            # Get action from policy
            rng, action_rng = jax.random.split(rng)
            action = get_action(obs_jax, action_rng)

            # Apply action with filtering and step physics (native MuJoCo - FAST)
            filtered_action = apply_action(mj_data, action, prev_action)
            step_physics(mj_model, mj_data, n_substeps)

            # Update previous action for next observation (use filtered action)
            prev_action = filtered_action

            step_count += 1

            # Recording
            if renderer and step_count <= record_steps:
                renderer.update_scene(mj_data)
                frame = renderer.render()
                frames.append(frame)

            # Check for episode end (robot fell or max steps)
            done = check_termination(mj_data) or (step_count >= max_episode_steps)
            if done:
                episode_count += 1
                forward_vel = get_forward_velocity(mj_data)  # body-local forward velocity
                height = mj_data.qpos[2]
                print(f"Episode {episode_count} ended at step {step_count}: vel={forward_vel:.2f}m/s, height={height:.2f}m")

                # Check if we've reached max episodes
                if max_episodes and episode_count >= max_episodes:
                    break

                # Reset environment
                reset_robot(mj_model, mj_data)
                prev_action = np.zeros(action_dim, dtype=np.float32)
                step_count = 0

        if renderer:
            renderer.close()

        # Save video
        if args.record and frames:
            import imageio
            print(f"Saving video to {args.record}...")
            imageio.mimsave(args.record, frames, fps=int(1/ctrl_dt))
            print(f"Done! Saved {len(frames)} frames.")

    else:
        # Interactive viewer mode (native MuJoCo - fast)
        # On macOS with mjpython, we need to use a different approach
        # The passive viewer runs in a separate thread, but we control the simulation
        max_episode_steps = int(training_cfg.env.episode_length / ctrl_dt) if hasattr(training_cfg.env, 'episode_length') else 500

        try:
            with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
                # Set viewer options
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -15
                viewer.cam.azimuth = 135  # View from behind-right
                viewer.cam.lookat[:] = [0, 0, 0.4]

                print(f"Viewer started (native MuJoCo - fast). is_running={viewer.is_running()}")

                # Main simulation loop
                while viewer.is_running():
                    step_start = time.time()

                    # Get observation from native MuJoCo state
                    obs = get_observation(mj_data, velocity_cmd, prev_action)
                    obs_jax = jnp.array(obs)

                    # Get action from policy
                    rng, action_rng = jax.random.split(rng)
                    action = get_action(obs_jax, action_rng)

                    # Apply action with filtering and step physics (native MuJoCo - FAST)
                    filtered_action = apply_action(mj_data, action, prev_action)
                    step_physics(mj_model, mj_data, n_substeps)

                    # Update previous action for next observation (use filtered action)
                    prev_action = filtered_action

                    step_count += 1

                    # Camera tracking: follow the robot
                    robot_pos = mj_data.qpos[0:3]  # [x, y, z] position
                    viewer.cam.lookat[:] = [robot_pos[0], robot_pos[1], 0.4]

                    # Debug: Print progress every 50 steps
                    if step_count % 50 == 0:
                        forward_vel = get_forward_velocity(mj_data)  # body-local forward velocity
                        height = mj_data.qpos[2]
                        action_sum = float(jnp.sum(jnp.abs(action)))
                        print(f"  Step {step_count}: vel={forward_vel:.2f}m/s, height={height:.3f}m, pos=({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")

                    # Recording (if enabled with viewer)
                    if args.record and step_count <= record_steps:
                        renderer = mujoco.Renderer(mj_model, 640, 480)
                        renderer.update_scene(mj_data)
                        frame = renderer.render()
                        frames.append(frame)
                        renderer.close()

                        if step_count == record_steps:
                            import imageio
                            print(f"Saving video to {args.record}...")
                            imageio.mimsave(args.record, frames, fps=int(1/ctrl_dt))
                            print("Done!")

                    # Check for episode end (robot fell or max steps)
                    done = check_termination(mj_data) or (step_count >= max_episode_steps)
                    if done:
                        episode_count += 1
                        forward_vel = get_forward_velocity(mj_data)  # body-local forward velocity
                        height = mj_data.qpos[2]
                        print(f"Episode {episode_count} ended at step {step_count}: vel={forward_vel:.2f}m/s, height={height:.2f}m")

                        # Check if we've reached max episodes
                        if max_episodes and episode_count >= max_episodes:
                            break

                        # Reset environment
                        reset_robot(mj_model, mj_data)
                        prev_action = np.zeros(action_dim, dtype=np.float32)
                        step_count = 0

                    # Sync viewer
                    viewer.sync()

                    # Timing for real-time playback
                    elapsed = time.time() - step_start
                    sleep_time = (ctrl_dt / args.speed) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except RuntimeError as e:
            if "mjpython" in str(e) and is_macos:
                print(f"\n⚠️  macOS requires mjpython for the viewer.")
                print("   Run with: uv run mjpython playground_amp/training/visualize_policy.py ...")
                print("   Or use --headless mode: uv run python ... --headless --num-episodes 1")
                return 1
            raise

    print(f"\nVisualization ended. {episode_count} episodes completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
