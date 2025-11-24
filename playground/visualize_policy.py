#!/usr/bin/env python3
"""
Visualize trained WildRobot policy and record video.

Usage:
    python visualize_policy.py --checkpoint checkpoints/latest.pkl --output videos/policy.mp4

Works in headless environments (SSH) using EGL GPU rendering!
"""

import argparse
import os
from pathlib import Path

# Enable headless rendering via EGL (GPU-accelerated, works over SSH)
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
import yaml
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params

from mujoco_playground._src import wrapper as mp_wrapper
from wildrobot.locomotion import WildRobotLocomotion


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict):
    """Create the WildRobot environment."""
    env = WildRobotLocomotion(
        task="wildrobot_flat",
        config=None,  # Will use defaults
    )

    # Wrap with training wrappers (same as training)
    env = mp_wrapper.wrap_for_brax_training(
        env,
        episode_length=config["training"]["episode_length"],
        action_repeat=config["ppo"]["action_repeat"],
    )

    return env


def load_policy(checkpoint_path: str, env):
    """Load trained policy from checkpoint."""
    import pickle

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    params = checkpoint_data["params"]

    # Create policy network (same architecture as training)
    network_factory = ppo_networks.make_ppo_networks
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x: x,  # No preprocessing
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return make_policy, params


def rollout_policy(
    env,
    make_policy,
    params: Params,
    n_steps: int = 1000,
    deterministic: bool = True,
    seed: int = 0,
):
    """
    Rollout the policy and collect states for rendering.

    Args:
        env: The environment
        make_policy: Policy inference function
        params: Policy parameters
        n_steps: Number of steps to run
        deterministic: Use deterministic policy
        seed: Random seed

    Returns:
        List of environment states
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Create policy
    policy = make_policy(params, deterministic=deterministic)

    # Reset environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    states = [state]

    print(f"Rolling out policy for {n_steps} steps...")
    for step in range(n_steps):
        if step % 100 == 0:
            print(f"  Step {step}/{n_steps}")

        # Get action from policy
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)

        # Step environment
        state = jit_step(state, action)
        states.append(state)

        # Check if done
        if state.done:
            print(f"Episode ended at step {step}")
            break

    return states


def render_video(
    env,
    states,
    output_path: str,
    height: int = 480,
    width: int = 640,
    camera: str = "track",
    fps: int = 50,
):
    """
    Render states to video.

    Args:
        env: The environment
        states: List of environment states
        output_path: Path to save video
        height: Video height
        width: Video width
        camera: Camera name
        fps: Frames per second
    """
    print(f"Rendering {len(states)} frames to video...")

    # Render frames
    frames = env.render(
        states,
        height=height,
        width=width,
        camera=camera,
    )

    # Convert to numpy array
    frames = np.array(frames)

    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving video to {output_path}...")
    media.write_video(str(output_path), frames, fps=fps)
    print(f"Video saved successfully!")

    return output_path


def visualize_policy(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    n_steps: int = 1000,
    deterministic: bool = True,
    seed: int = 0,
    height: int = 480,
    width: int = 640,
    camera: str = "track",
    fps: int = 50,
):
    """
    Main function to visualize trained policy.

    Args:
        checkpoint_path: Path to trained policy checkpoint
        config_path: Path to training config
        output_path: Path to save output video
        n_steps: Number of steps to run
        deterministic: Use deterministic policy
        seed: Random seed
        height: Video height
        width: Video width
        camera: Camera name
        fps: Frames per second
    """
    print("=" * 60)
    print("WildRobot Policy Visualization")
    print("=" * 60)

    # Load config
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    # Create environment
    print(f"Creating environment...")
    env = create_env(config)

    # Load policy
    print(f"\nLoading policy from: {checkpoint_path}")
    make_policy, params = load_policy(checkpoint_path, env)

    # Rollout policy
    print(f"\nRolling out policy...")
    states = rollout_policy(
        env,
        make_policy,
        params,
        n_steps=n_steps,
        deterministic=deterministic,
        seed=seed,
    )

    # Render video
    print(f"\nRendering video...")
    video_path = render_video(
        env,
        states,
        output_path,
        height=height,
        width=width,
        camera=camera,
        fps=fps,
    )

    print("\n" + "=" * 60)
    print(f"SUCCESS! Video saved to: {video_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained WildRobot policy and record video"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint (.pkl file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quick.yaml",
        help="Path to training config (default: quick.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videos/policy.mp4",
        help="Output video path (default: videos/policy.mp4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width (default: 640)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="track",
        help="Camera name (default: track)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Video FPS (default: 50)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )

    args = parser.parse_args()

    visualize_policy(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        n_steps=args.steps,
        deterministic=not args.stochastic,
        seed=args.seed,
        height=args.height,
        width=args.width,
        camera=args.camera,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
