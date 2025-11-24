#!/usr/bin/env python3
"""
Visualize trained WildRobot policy and record video.

Usage:
    # Single environment
    python visualize_policy.py --checkpoint final_policy.pkl --output single.mp4

    # Grid of 4 environments
    python visualize_policy.py --checkpoint final_policy.pkl --output grid.mp4 --grid --n_envs 4

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
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params

from wildrobot.locomotion import WildRobotLocomotion


def create_env():
    """Create the WildRobot environment (UNWRAPPED for visualization)."""
    env = WildRobotLocomotion(
        task="wildrobot_flat",
        config=None,  # Will use defaults
    )

    # DON'T wrap - we want a single unwrapped environment for visualization
    # The wrapped environment expects batched inputs (num_envs dimension)
    # For visualization, we only need 1 environment
    return env


def load_policy(checkpoint_path: str, env):
    """Load trained policy from checkpoint."""
    import pickle

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    params = checkpoint_data["params"]
    config = checkpoint_data["config"]

    # Extract network architecture from saved config
    policy_hidden_layers = config["network"]["policy_hidden_layers"]
    value_hidden_layers = config["network"]["value_hidden_layers"]

    print(f"  Policy network: {policy_hidden_layers}")
    print(f"  Value network: {value_hidden_layers}")

    # Create policy network with SAME architecture as training
    # Note: preprocess_observations_fn takes (obs, processor_params)
    network_factory = ppo_networks.make_ppo_networks
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda obs, params: obs,  # No preprocessing
        policy_hidden_layer_sizes=policy_hidden_layers,
        value_hidden_layer_sizes=value_hidden_layers,
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
        env: The unwrapped environment
        make_policy: Policy inference function
        params: Policy parameters
        n_steps: Number of steps to run
        deterministic: Use deterministic policy
        seed: Random seed

    Returns:
        List of environment states (for rendering)
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Create policy
    policy = make_policy(params, deterministic=deterministic)

    # Reset environment
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    # Collect full state objects (env.render expects states with .data attribute)
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
        done_value = float(state.done)
        if done_value > 0.5:
            print(f"Episode ended at step {step}")
            break

    print(f"Collected {len(states)} states for rendering")
    return states


def render_grid_video(
    env,
    make_policy,
    params: Params,
    output_path: str,
    n_envs: int = 4,
    n_steps: int = 1000,
    deterministic: bool = True,
    base_seed: int = 0,
    height: int = 240,
    width: int = 320,
    camera: str = "track",
    fps: int = 50,
):
    """
    Render multiple environment rollouts in a grid layout (like loco_mujoco).

    Args:
        env: The environment
        make_policy: Policy inference function
        params: Policy parameters
        output_path: Path to save output video
        n_envs: Number of environments to render (must be perfect square: 4, 9, 16, etc.)
        n_steps: Number of steps per rollout
        deterministic: Use deterministic policy
        base_seed: Base random seed (each env gets base_seed + env_id)
        height: Height per environment video
        width: Width per environment video
        camera: Camera name
        fps: Frames per second

    Returns:
        Path to saved grid video
    """
    # Validate n_envs is a perfect square
    grid_size = int(np.sqrt(n_envs))
    if grid_size * grid_size != n_envs:
        raise ValueError(
            f"n_envs must be a perfect square (4, 9, 16, etc.), got {n_envs}"
        )

    print("=" * 60)
    print(f"Rendering {n_envs} environments in {grid_size}x{grid_size} grid")
    print("=" * 60)

    # Collect trajectories from multiple environments
    all_trajectories = []
    for env_id in range(n_envs):
        print(f"\nRollout {env_id + 1}/{n_envs} (seed={base_seed + env_id}):")
        states = rollout_policy(
            env,
            make_policy,
            params,
            n_steps=n_steps,
            deterministic=deterministic,
            seed=base_seed + env_id,
        )
        all_trajectories.append(states)

    # Find minimum trajectory length (in case some ended early)
    min_length = min(len(traj) for traj in all_trajectories)
    print(f"\nMin trajectory length: {min_length} frames")

    # Render each trajectory
    print(f"\nRendering {n_envs} trajectories...")
    all_frames = []
    for env_id, traj in enumerate(all_trajectories):
        print(f"  Rendering environment {env_id + 1}/{n_envs}...")
        # Use only min_length frames
        frames = env.render(
            traj[:min_length],
            height=height,
            width=width,
            camera=camera,
        )
        all_frames.append(np.array(frames))

    # Create grid layout
    print(f"\nCreating {grid_size}x{grid_size} grid layout...")
    grid_frames = []
    for frame_idx in range(min_length):
        if frame_idx % 100 == 0:
            print(f"  Processing frame {frame_idx}/{min_length}...")

        # Extract frame from each environment
        env_frames = [all_frames[i][frame_idx] for i in range(n_envs)]

        # Arrange in grid
        rows = []
        for row_idx in range(grid_size):
            row_start = row_idx * grid_size
            row_end = row_start + grid_size
            row_frames = env_frames[row_start:row_end]
            row = np.hstack(row_frames)
            rows.append(row)

        grid = np.vstack(rows)
        grid_frames.append(grid)

    # Save grid video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving grid video to {output_path}...")
    media.write_video(str(output_path), grid_frames, fps=fps)

    print("\n" + "=" * 60)
    print(f"SUCCESS! Grid video saved to: {output_path}")
    print(f"Grid size: {grid_size}x{grid_size} ({n_envs} environments)")
    print(f"Resolution: {width * grid_size}x{height * grid_size}")
    print(f"Frames: {len(grid_frames)}")
    print("=" * 60)

    return output_path


def visualize_single_env(
    env,
    make_policy,
    params: Params,
    output_path: str,
    n_steps: int = 1000,
    deterministic: bool = True,
    seed: int = 0,
    height: int = 480,
    width: int = 640,
    camera: str = None,
    fps: int = 50,
):
    """
    Visualize trained policy (single environment).

    Args:
        env: The environment
        make_policy: Policy inference function
        params: Policy parameters
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
    print("WildRobot Policy Visualization (Single Environment)")
    print("=" * 60)

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
    print(f"\nRendering {len(states)} frames to video...")
    frames = env.render(
        states,
        height=height,
        width=width,
        camera=camera,
    )

    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving video to {output_path}...")
    media.write_video(str(output_path), np.array(frames), fps=fps)

    print("\n" + "=" * 60)
    print(f"SUCCESS! Video saved to: {output_path}")
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
        default=None,
        help="Camera name (default: None for free camera)",
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

    # Grid rendering options
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Render multiple environments in a grid (like loco_mujoco)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="Number of environments for grid rendering (must be perfect square: 4, 9, 16, etc.) (default: 4)",
    )

    args = parser.parse_args()

    # Create environment
    print(f"Creating environment...")
    env = create_env()

    # Load policy
    print(f"\nLoading policy from: {args.checkpoint}")
    make_policy, params = load_policy(args.checkpoint, env)

    # Render grid or single video
    if args.grid:
        # Grid rendering (multiple environments)
        render_grid_video(
            env=env,
            make_policy=make_policy,
            params=params,
            output_path=args.output,
            n_envs=args.n_envs,
            n_steps=args.steps,
            deterministic=not args.stochastic,
            base_seed=args.seed,
            height=args.height,
            width=args.width,
            camera=args.camera,
            fps=args.fps,
        )
    else:
        # Single environment rendering
        visualize_single_env(
            env=env,
            make_policy=make_policy,
            params=params,
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
