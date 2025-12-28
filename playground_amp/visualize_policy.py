#!/usr/bin/env python3
"""Visualize trained WildRobot policy in MuJoCo viewer.

This script loads a trained checkpoint and runs the policy in real-time
using MuJoCo's native viewer.

Usage:
    # Visualize latest PPO policy
    uv run python playground_amp/visualize_policy.py

    # Visualize specific checkpoint
    uv run python playground_amp/visualize_policy.py --checkpoint path/to/checkpoint.pkl

    # Visualize with different config
    uv run python playground_amp/visualize_policy.py --config playground_amp/configs/ppo_standing.yaml

    # With velocity command (for walking)
    uv run python playground_amp/visualize_policy.py --velocity-cmd 0.5

Controls:
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.configs.training_config import (
    load_robot_config,
    load_training_config,
)
from playground_amp.training.ppo_core import create_networks, sample_actions


# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "ppo_standing.yaml"
DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "final_ppo_policy.pkl"
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
    return parser.parse_args()


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

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_PATH
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for p in Path("playground_amp/checkpoints").glob("**/*.pkl"):
            print(f"  - {p}")
        return 1

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint mode: {checkpoint.get('mode', 'unknown')}")

    # Create environment (uses JAX/MJX internally but we'll extract data for viewer)
    from playground_amp.envs.wildrobot_env import WildRobotEnv

    print("Creating WildRobotEnv...")
    env = WildRobotEnv(config=training_cfg)

    # Get dimensions
    obs_dim = env.observation_size
    action_dim = env.action_size

    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")

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
    policy_params = checkpoint["policy_params"]
    processor_params = ()  # No observation normalization

    # JIT compile the action sampling function
    @jax.jit
    def get_action(obs, rng):
        """Get action from policy."""
        obs_batch = obs[None, ...]  # Add batch dimension
        action, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs_batch, rng, deterministic
        )
        return action[0]  # Remove batch dimension

    # Initialize environment state using JAX
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    env_state = env.reset(reset_rng)

    # Load MuJoCo model for native viewer
    model_path = project_root / training_cfg.env.model_path
    print(f"Loading model from: {model_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)

    # Copy initial state from env to viewer
    def sync_state_to_viewer(env_state, mj_data):
        """Sync JAX env state to MuJoCo viewer data."""
        # Get qpos/qvel from env_state.pipeline_state
        qpos = np.array(env_state.pipeline_state.qpos)
        qvel = np.array(env_state.pipeline_state.qvel)
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = qvel
        mujoco.mj_forward(mj_model, mj_data)

    sync_state_to_viewer(env_state, mj_data)

    # Control timing
    ctrl_dt = training_cfg.env.ctrl_dt

    print(f"\n{'=' * 60}")
    print("WildRobot Policy Visualization")
    print(f"{'=' * 60}")
    print(f"  Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"  Velocity range: [{training_cfg.env.min_velocity:.2f}, {training_cfg.env.max_velocity:.2f}] m/s")
    if args.velocity_cmd is not None:
        print(f"  Fixed velocity cmd: {args.velocity_cmd:.2f} m/s")
    print(f"  Control dt: {ctrl_dt}s ({1/ctrl_dt:.0f} Hz)")
    print(f"  Playback speed: {args.speed}x")
    print(f"{'=' * 60}")
    print("\nControls:")
    print("  Space     - Pause/Resume")
    print("  Backspace - Reset")
    print("  Escape    - Exit")
    print(f"{'=' * 60}\n")

    # Recording setup
    if args.record:
        import imageio
        frames = []
        record_steps = int(args.record_duration / ctrl_dt)
        print(f"Recording {args.record_duration}s ({record_steps} steps) to {args.record}")

    # Launch viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Set viewer options
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 0.4]

        step_count = 0
        episode_count = 0

        while viewer.is_running():
            step_start = time.time()

            # Get observation from current env state
            obs = env_state.obs

            # Get action from policy
            rng, action_rng = jax.random.split(rng)
            action = get_action(obs, action_rng)

            # Step environment using JAX
            env_state = env.step(env_state, action)

            # Sync state to viewer
            sync_state_to_viewer(env_state, mj_data)

            step_count += 1

            # Recording
            if args.record and step_count <= record_steps:
                # Render frame
                renderer = mujoco.Renderer(mj_model, 640, 480)
                renderer.update_scene(mj_data)
                frame = renderer.render()
                frames.append(frame)
                renderer.close()

                if step_count == record_steps:
                    print(f"Saving video to {args.record}...")
                    imageio.mimsave(args.record, frames, fps=int(1/ctrl_dt))
                    print("Done!")

            # Check for episode end and auto-reset
            done = float(env_state.done) > 0.5
            if done:
                episode_count += 1
                # Get episode info from metrics dict using registry
                from playground_amp.training.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
                metrics_vec = env_state.metrics[METRICS_VEC_KEY]
                forward_vel = float(metrics_vec[METRIC_INDEX["forward_velocity"]])
                height = float(env_state.pipeline_state.qpos[2])
                print(f"Episode {episode_count} ended at step {step_count}: vel={forward_vel:.2f}m/s, height={height:.2f}m")

                # Reset environment
                rng, reset_rng = jax.random.split(rng)
                env_state = env.reset(reset_rng)
                sync_state_to_viewer(env_state, mj_data)
                step_count = 0

            # Sync viewer
            viewer.sync()

            # Timing for real-time playback
            elapsed = time.time() - step_start
            sleep_time = (ctrl_dt / args.speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\nVisualization ended. {episode_count} episodes completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
