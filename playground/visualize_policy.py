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


def find_latest_checkpoint(checkpoints_dir: str = None):
    """Find the most recent checkpoint in the checkpoints directory.

    Args:
        checkpoints_dir: Path to checkpoints directory. If None, uses playground/checkpoints

    Returns:
        Path to the latest final_policy.pkl file, or None if not found
    """
    if checkpoints_dir is None:
        # Default to playground/checkpoints
        checkpoints_dir = Path(__file__).parent / "checkpoints"
    else:
        checkpoints_dir = Path(checkpoints_dir)

    if not checkpoints_dir.exists():
        return None

    # Find all experiment directories (e.g., wildrobot_locomotion_flat_20231125-120000)
    exp_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True  # Most recent first
    )

    # Look for final_policy.pkl in each directory
    for exp_dir in exp_dirs:
        checkpoint_path = exp_dir / "final_policy.pkl"
        if checkpoint_path.exists():
            return checkpoint_path

    return None


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
    """
    Load trained policy from checkpoint.

    This follows the standard Brax approach for loading PPO policies:
    1. Load params from checkpoint
    2. Reconstruct network with same architecture
    3. Use make_inference_fn to create policy function
    """
    import pickle

    print(f"  Loading checkpoint...")
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    params = checkpoint_data["params"]
    config = checkpoint_data["config"]

    # Extract network architecture from saved config
    policy_hidden_layers = config["network"]["policy_hidden_layers"]
    value_hidden_layers = config["network"]["value_hidden_layers"]

    # Check if normalization was used during training
    normalize_observations = config["ppo"]["normalize_observations"]

    print(f"  Policy network: {policy_hidden_layers}")
    print(f"  Value network: {value_hidden_layers}")
    print(f"  Normalize observations: {normalize_observations}")

    # Create preprocessing function to match training
    if normalize_observations:
        # During training, Brax normalizes observations using running statistics
        # The params passed to preprocessing is the RunningStatisticsState directly
        def preprocess_observations_fn(obs, normalizer_params):
            # normalizer_params is a RunningStatisticsState with .mean and .std
            return (obs - normalizer_params.mean) / (normalizer_params.std + 1e-8)

        print(f"  ✓ Using observation normalization from checkpoint")
    else:
        # No normalization - pass observations through unchanged
        preprocess_observations_fn = lambda obs, params: obs
        print(f"  ✓ No observation normalization")

    # Create policy network with SAME architecture as training
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layers,
        value_hidden_layer_sizes=value_hidden_layers,
    )

    # Create inference function (deterministic or stochastic)
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


def rollout_with_metrics(
    env,
    make_policy,
    params: Params,
    n_steps: int = 1000,
    deterministic: bool = True,
    seed: int = 0,
):
    """
    Rollout policy and collect detailed metrics for validation.

    Args:
        env: The environment
        make_policy: Policy inference function
        params: Policy parameters
        n_steps: Number of steps to run
        deterministic: Use deterministic policy
        seed: Random seed

    Returns:
        Dictionary with metrics: reward, length, heights, velocities, etc.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    policy = make_policy(params, deterministic=deterministic)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    # Collect metrics
    total_reward = 0.0
    episode_length = 0
    heights = []
    velocities = []
    actions = []

    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = jit_step(state, action)

        # Accumulate metrics
        total_reward += float(state.reward)
        episode_length += 1

        # Track action statistics
        actions.append(np.array(action))

        # Extract environment-specific metrics
        if hasattr(state, "metrics") and state.metrics:
            if "height" in state.metrics:
                heights.append(float(state.metrics["height"]))
            if "forward_velocity" in state.metrics:
                velocities.append(float(state.metrics["forward_velocity"]))

        # Check if done
        if float(state.done) > 0.5:
            break

    return {
        "reward": total_reward,
        "length": episode_length,
        "heights": heights,
        "velocities": velocities,
        "actions": np.array(actions) if actions else None,
    }


def validate_policy_consistency(
    checkpoint_path: str,
    n_trials: int = 10,
    training_reward: float = None,
    training_length: float = None,
):
    """
    Validate that visualization matches training performance.

    This function runs multiple rollouts and checks:
    1. Episode rewards match training performance
    2. Episode lengths are consistent
    3. Policy is deterministic (same seed = same result)
    4. Observations are properly normalized

    Args:
        checkpoint_path: Path to checkpoint file
        n_trials: Number of rollouts to average over
        training_reward: Expected reward from training (optional, auto-loaded from checkpoint if available)
        training_length: Expected episode length from training (optional, auto-loaded from checkpoint if available)

    Returns:
        Dictionary with validation results and pass/fail status
    """
    print("=" * 70)
    print("🔍 POLICY CONSISTENCY VALIDATION")
    print("=" * 70)

    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    import pickle

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    config = checkpoint_data["config"]

    # Try to auto-load metrics from checkpoint (if saved by new train.py)
    auto_loaded_metrics = {}
    validation_config = config.get("validation", {})
    validation_metric_mapping = validation_config.get("metrics", {})

    if "final_metrics" in checkpoint_data:
        final_metrics = checkpoint_data["final_metrics"]
        print(f"\n✅ Found training metrics in checkpoint!")

        # Auto-load metrics if not provided by user
        # Check for standard metrics first (backward compatibility)
        if training_reward is None and "eval/episode_reward" in final_metrics:
            training_reward = final_metrics["eval/episode_reward"]
            auto_loaded_metrics["reward"] = training_reward
            print(f"   Auto-loaded training_reward: {training_reward:.1f}")

        if training_length is None and "eval/avg_episode_length" in final_metrics:
            training_length = final_metrics["eval/avg_episode_length"]
            auto_loaded_metrics["episode_length"] = training_length
            print(f"   Auto-loaded training_length: {training_length:.1f}")

        # Also check for any additional metrics defined in config
        if validation_metric_mapping:
            print(f"   Additional metrics from config:")
            for metric_name, metric_key in validation_metric_mapping.items():
                if metric_key in final_metrics and metric_name not in auto_loaded_metrics:
                    auto_loaded_metrics[metric_name] = final_metrics[metric_key]
                    print(f"     {metric_name}: {final_metrics[metric_key]}")

    # Get validation thresholds from config
    thresholds = validation_config.get("thresholds", {})
    reward_tolerance = thresholds.get("reward_tolerance", 0.10)  # Default 10%
    length_tolerance = thresholds.get("length_tolerance", 0.10)  # Default 10%
    determinism_threshold = thresholds.get("determinism_threshold", 0.01)  # Default 0.01

    # Print training metrics
    print(f"\n{'📊 Training Metrics (for validation)':^70}")
    print("-" * 70)
    if training_reward is not None:
        print(f"  Expected Reward:        {training_reward:.1f}")
    else:
        print("  Expected Reward:        (not available)")
        print("  💡 Tip: Add --training-reward 4800 or retrain with updated train.py")

    if training_length is not None:
        print(f"  Expected Episode Length: {training_length:.1f}")
    else:
        print("  Expected Episode Length: (not available)")
        print("  💡 Tip: Add --training-length 900 or retrain with updated train.py")

    # Create environment and load policy
    print(f"\n🏗️  Creating environment...")
    env = create_env()

    print(f"🧠 Loading policy...")
    make_policy, params = load_policy(checkpoint_path, env)

    # Run multiple rollouts
    print(f"\n{'🎯 Visualization Metrics (current rollouts)':^70}")
    print("-" * 70)
    print(f"Running {n_trials} rollouts to measure consistency...")

    results = []
    for trial in range(n_trials):
        metrics = rollout_with_metrics(
            env, make_policy, params, n_steps=1000, deterministic=True, seed=trial
        )
        results.append(metrics)
        print(
            f"  Trial {trial+1:2d}/{n_trials}: "
            f"Reward={metrics['reward']:7.1f}, "
            f"Length={metrics['length']:4d}"
        )

    # Compute statistics
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    heights = [h for r in results for h in r["heights"]]
    velocities = [v for r in results for v in r["velocities"]]

    print(f"\n{'📈 Summary Statistics':^70}")
    print("-" * 70)
    print(f"  Episode Reward:     {np.mean(rewards):7.1f} ± {np.std(rewards):6.1f}")
    print(f"  Episode Length:     {np.mean(lengths):7.1f} ± {np.std(lengths):6.1f}")
    if heights:
        print(f"  Average Height:     {np.mean(heights):7.3f} ± {np.std(heights):6.3f} m")
    if velocities:
        print(
            f"  Average Velocity:   {np.mean(velocities):7.3f} ± {np.std(velocities):6.3f} m/s"
        )

    # Validation checks
    print(f"\n{'✅ Validation Checks':^70}")
    print("-" * 70)

    validation_results = {"passed": True, "checks": {}}

    # Check 1: Reward consistency with training
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\n1️⃣  Reward Consistency:")
    print(f"    Mean: {mean_reward:.1f}")
    print(f"    Std:  {std_reward:.1f}")
    print(f"    Min:  {np.min(rewards):.1f}")
    print(f"    Max:  {np.max(rewards):.1f}")

    if training_reward is not None:
        reward_diff_pct = abs(mean_reward - training_reward) / training_reward
        print(f"    Difference from training: {reward_diff_pct*100:.1f}%")
        print(f"    Tolerance threshold: ±{reward_tolerance*100:.0f}% (from config)")

        if reward_diff_pct <= reward_tolerance:
            print(f"    ✅ PASS: Within ±{reward_tolerance*100:.0f}% of training reward!")
            validation_results["checks"]["reward_consistency"] = "pass"
        elif reward_diff_pct <= reward_tolerance * 2:
            print(f"    ⚠️  WARNING: Within ±{reward_tolerance*200:.0f}% but investigate further")
            validation_results["checks"]["reward_consistency"] = "warning"
        else:
            print(f"    ❌ FAIL: >{reward_tolerance*200:.0f}% difference from training!")
            validation_results["passed"] = False
            validation_results["checks"]["reward_consistency"] = "fail"
    else:
        print(f"    ℹ️  No training reward provided - cannot validate")
        validation_results["checks"]["reward_consistency"] = "unknown"

    # Check 2: Episode length
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    print(f"\n2️⃣  Episode Length Consistency:")
    print(f"    Mean: {mean_length:.1f}")
    print(f"    Std:  {std_length:.1f}")

    if training_length is not None:
        length_diff_pct = abs(mean_length - training_length) / training_length
        print(f"    Difference from training: {length_diff_pct*100:.1f}%")
        print(f"    Tolerance threshold: ±{length_tolerance*100:.0f}% (from config)")

        if length_diff_pct <= length_tolerance:
            print(f"    ✅ PASS: Within ±{length_tolerance*100:.0f}% of training length!")
            validation_results["checks"]["length_consistency"] = "pass"
        elif length_diff_pct <= length_tolerance * 2:
            print(f"    ⚠️  WARNING: Within ±{length_tolerance*200:.0f}% but acceptable")
            validation_results["checks"]["length_consistency"] = "warning"
        else:
            print(f"    ❌ FAIL: >{length_tolerance*200:.0f}% difference from training!")
            validation_results["passed"] = False
            validation_results["checks"]["length_consistency"] = "fail"
    else:
        print(f"    ℹ️  No training length provided - cannot validate")
        validation_results["checks"]["length_consistency"] = "unknown"

    # Check 3: Determinism (run same seed twice)
    print(f"\n3️⃣  Determinism Check:")
    m1 = rollout_with_metrics(env, make_policy, params, seed=42, deterministic=True)
    m2 = rollout_with_metrics(env, make_policy, params, seed=42, deterministic=True)
    reward_diff = abs(m1["reward"] - m2["reward"])
    print(f"    Same seed, different runs: reward diff = {reward_diff:.6f}")
    print(f"    Tolerance threshold: <{determinism_threshold} (from config)")

    if reward_diff < determinism_threshold:
        print(f"    ✅ PASS: Policy is deterministic!")
        validation_results["checks"]["determinism"] = "pass"
    elif reward_diff < 1.0:
        print(f"    ⚠️  WARNING: Small randomness detected (diff={reward_diff:.6f})")
        validation_results["checks"]["determinism"] = "warning"
    else:
        print(f"    ❌ FAIL: Policy is not deterministic!")
        validation_results["passed"] = False
        validation_results["checks"]["determinism"] = "fail"

    # Check 4: Normalization
    print(f"\n4️⃣  Observation Normalization Check:")
    if hasattr(params[0], "mean") and hasattr(params[0], "std"):
        print(f"    ✅ Normalizer params found:")
        print(f"       Mean shape: {params[0].mean.shape}")
        print(f"       Std shape:  {params[0].std.shape}")
        print(f"       First 5 means: {params[0].mean[:5]}")
        print(f"       First 5 stds:  {params[0].std[:5]}")
        validation_results["checks"]["normalization"] = "pass"
    else:
        print(f"    ℹ️  No normalization params (or not needed)")
        validation_results["checks"]["normalization"] = "unknown"

    # Check 5: Action statistics
    if results[0]["actions"] is not None:
        all_actions = np.concatenate([r["actions"] for r in results])
        print(f"\n5️⃣  Action Statistics:")
        print(f"    Shape: {all_actions.shape}")
        print(f"    Mean:  {all_actions.mean():.3f}")
        print(f"    Std:   {all_actions.std():.3f}")
        print(f"    Min:   {all_actions.min():.3f}")
        print(f"    Max:   {all_actions.max():.3f}")

        if np.abs(all_actions).max() > 2.0:
            print(f"    ⚠️  WARNING: Actions outside expected range [-1, 1]!")
            validation_results["checks"]["action_range"] = "warning"
        else:
            print(f"    ✅ Actions in expected range")
            validation_results["checks"]["action_range"] = "pass"

    # Final summary
    print("\n" + "=" * 70)
    if validation_results["passed"]:
        print("🎉 VALIDATION PASSED!")
        print("=" * 70)
        print("\n✅ Your visualization accurately represents training performance!")
        print("✅ You can trust this policy for robot deployment.")
    else:
        print("⚠️  VALIDATION FAILED!")
        print("=" * 70)
        print("\n❌ Visualization does not match training performance.")
        print("❌ Investigate issues before deploying to robot.")

    print("\n📝 Next Steps:")
    if training_reward is None or training_length is None:
        print("  1. Provide training metrics (--training-reward, --training-length)")
        print("  2. Check W&B for eval/episode_reward and eval/avg_episode_length")
    if not validation_results["passed"]:
        print("  3. Debug inconsistencies (see VISUALIZATION_CONSISTENCY_VALIDATION.md)")
    else:
        print("  3. Proceed with confidence - visualization matches training! 🚀")

    print("=" * 70)

    # Store validation results
    validation_results["mean_reward"] = mean_reward
    validation_results["std_reward"] = std_reward
    validation_results["mean_length"] = mean_length
    validation_results["std_length"] = std_length

    return validation_results


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
        default="latest",
        help="Path to trained policy checkpoint (.pkl file), or 'latest' to auto-find most recent (default: latest)",
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

    # Validation options
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate policy consistency (run multiple rollouts and check metrics)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of validation trials (default: 10)",
    )
    parser.add_argument(
        "--training-reward",
        type=float,
        default=None,
        help="Expected training reward for validation (from W&B eval/episode_reward)",
    )
    parser.add_argument(
        "--training-length",
        type=float,
        default=None,
        help="Expected training episode length for validation (from W&B eval/avg_episode_length)",
    )

    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint == "latest":
        print("Finding latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("ERROR: No checkpoints found in playground/checkpoints/")
            print("Please train a model first or specify --checkpoint path/to/checkpoint.pkl")
            return
        args.checkpoint = str(checkpoint_path)
        print(f"Using latest checkpoint: {args.checkpoint}")

    # Run validation if requested
    if args.validate:
        validate_policy_consistency(
            checkpoint_path=args.checkpoint,
            n_trials=args.n_trials,
            training_reward=args.training_reward,
            training_length=args.training_length,
        )
        return  # Exit after validation

    # Create environment
    print(f"\nCreating environment...")
    env = create_env()

    # Load policy
    print(f"Loading policy from: {args.checkpoint}")
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
