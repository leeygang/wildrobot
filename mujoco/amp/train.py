#!/usr/bin/env python3
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train WildRobot AMP with PPO for human-like walking."""

# CRITICAL: Set rendering backend BEFORE importing mujoco
import os
import platform

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Detect platform and set appropriate rendering backend
system = platform.system()

if system == "Darwin":  # macOS
    # macOS uses GLFW (built-in with MuJoCo)
    os.environ["MUJOCO_GL"] = "glfw"
    print("macOS detected: Using GLFW renderer")
else:  # Linux
    # Local Linux session with display - use GPU rendering via EGL
    os.environ["MUJOCO_GL"] = "egl"
    print("Local Linux session detected: Using EGL renderer (GPU accelerated)")

import datetime
import functools
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import wandb
import yaml

from absl import app, flags, logging
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from etils import epath
from ml_collections import config_dict

# Add amp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from amp import walk_env
from common.video_renderer import VideoRenderer

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

        

def render_videos(
    env,
    params,
    cfg,
    output_dir,
    num_videos=3,
    max_steps=600,
    camera="track",
    seed=0,
    make_inference_fn=None,
):
    """Render evaluation videos using trained policy.

    Args:
        env: Environment instance
        params: Policy parameters (either from training or loaded checkpoint)
        cfg: Config dict
        output_dir: Directory to save videos
        num_videos: Number of videos to render
        max_steps: Maximum steps per video
        camera: Camera name for rendering
        seed: Random seed
        make_inference_fn: Optional inference function factory (if from training).
                          If None, will create from params directly.

    Returns:
        List of (video_path, stats) tuples
    """

    # Create inference function
    if make_inference_fn is not None:
        inference_fn = make_inference_fn(params)
    else:
        inference_fn = VideoRenderer.create_inference_from_checkpoint(
            params=params,
            cfg=cfg,
            env=env,
        )

    # Create renderer from config
    renderer = VideoRenderer.from_config(
        env=env,
        cfg=cfg,
        output_dir=output_dir,
        camera=camera,
        max_steps=max_steps,
    )

    print(f"\nRendering settings:")
    print(f"  Number of videos: {num_videos}")
    print(f"  Max steps: {max_steps}")
    print(f"  Resolution: {renderer.width}x{renderer.height}")
    print(f"  Camera: {renderer.camera if renderer.camera else 'free (default)'}")
    print(f"  FPS: {renderer.fps}")
    print(f"  Seed: {seed}\n")

    # Render videos
    rng = jax.random.PRNGKey(seed)
    results = renderer.render_multiple_videos(
        inference_fn=inference_fn,
        rng=rng,
        num_videos=num_videos,
        video_prefix="eval_video",
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"Rendered {len(results)} videos successfully!")
    for i, (video_path, stats) in enumerate(results, 1):
        print(f"  {i}. {video_path.name}")
        print(f"     Length: {stats['length']} steps | "
              f"Velocity: {stats['velocity']:.3f} m/s | "
              f"Height: {stats['height']:.3f} m")
    print(f"{'='*80}\n")

    return results


def generate_and_log_metrics(
    num_steps,
    metrics,
    elapsed,
    start_time,
    cfg,
    ppo_config,
    metrics_log_file,
):
    """Generate metrics entry and log to JSONL and W&B.

    Args:
        num_steps: Current training step
        metrics: Raw metrics dict from Brax
        elapsed: Elapsed time since training started
        start_time: Training start time
        cfg: Config dict
        ppo_config: PPO configuration
        metrics_log_file: File handle for JSONL logging

    Returns:
        dict: Extracted metrics for printing (eval_reward, eval_length, forward_vel,
              height, success, distance, sps)
    """
    # Extract metrics from raw metrics dict
    eval_reward = metrics.get("eval/episode_reward", 0.0)
    eval_length = metrics.get("eval/avg_episode_length", 0.0)
    forward_vel = metrics.get("eval/episode_forward_velocity", 0.0) / max(eval_length, 1.0)
    height = metrics.get("eval/episode_height", 0.0) / max(eval_length, 1.0)
    success = metrics.get("eval/episode_success", 0.0)
    distance = metrics.get("eval/episode_distance_walked", 0.0)
    sps = num_steps / elapsed if elapsed > 0 else 0

    # Save metrics to log file (always, regardless of W&B)
    # IMPROVED STRUCTURE: Separate rewards & penalties, remove duplicates
    metrics_entry = {
        "step": int(num_steps),
        "timestamp": time.time(),
        "summary": {},  # Will populate after separating rewards/penalties
        "rewards": {},
        "penalties": {},
        "contact": {},
        "other": {},
    }

    # First pass: Collect and separate reward components
    total_rewards = 0.0
    total_penalties = 0.0

    # Get list of tracked reward components from config (explicit, not filtered)
    tracked_components = set(cfg.get("tracked_reward_components", []))

    # Track weighted contributions separately
    metrics_entry["rewards_weighted"] = {}
    metrics_entry["penalties_weighted"] = {}
    total_rewards_weighted = 0.0
    total_penalties_weighted = 0.0

    for key, value in metrics.items():
        value_per_step = float(value / max(eval_length, 1.0))

        if key.startswith("eval/episode_reward/"):
            component_name = key.replace("eval/episode_reward/", "")

            # Only track components listed in config
            if component_name not in tracked_components:
                continue

            # Get weight for this component
            weight = float(cfg["reward_weights"].get(component_name, 0.0))
            weighted_value = value_per_step * weight

            # Separate positive (rewards) from negative (penalties)
            # UNWEIGHTED (for debugging raw component values)
            if value_per_step >= 0:
                # Positive component - it's a reward
                metrics_entry["rewards"][component_name] = value_per_step
                total_rewards += value_per_step
            else:
                # Negative component - it's a penalty (store as positive for clarity)
                metrics_entry["penalties"][component_name] = abs(value_per_step)
                total_penalties += abs(value_per_step)

            # WEIGHTED (actual contribution to training)
            if weighted_value >= 0:
                metrics_entry["rewards_weighted"][component_name] = weighted_value
                total_rewards_weighted += weighted_value
            else:
                metrics_entry["penalties_weighted"][component_name] = abs(weighted_value)
                total_penalties_weighted += abs(weighted_value)

        elif key.startswith("eval/episode_contact/"):
            metric_name = key.replace("eval/episode_contact/", "")
            metrics_entry["contact"][metric_name] = value_per_step
        elif key.startswith("eval/"):
            metric_name = key.replace("eval/episode_", "").replace("eval/", "")
            # Skip duplicates that will be in summary
            if metric_name not in ["reward", "success", "avg_episode_length",
                                  "forward_velocity", "height", "distance_walked", "sps"]:
                metrics_entry["other"][metric_name] = float(value_per_step) if "episode" in key else float(value)

    # Add totals to rewards and penalties buckets
    metrics_entry["rewards"]["total"] = total_rewards
    metrics_entry["penalties"]["total"] = total_penalties
    metrics_entry["rewards_weighted"]["total"] = total_rewards_weighted
    metrics_entry["penalties_weighted"]["total"] = total_penalties_weighted

    # Optional gating diagnostic components (treated as rewards with weight 0)
    # Extract per-step values if present. These are diagnostic only and do not contribute to reward.
    gate_active_rate = 0.0
    velocity_threshold_scale_val = 1.0
    if "eval/episode_reward/tracking_gate_active" in metrics:
        gate_active_rate = metrics["eval/episode_reward/tracking_gate_active"] / max(eval_length, 1.0)
    if "eval/episode_reward/velocity_threshold_scale" in metrics:
        velocity_threshold_scale_val = metrics["eval/episode_reward/velocity_threshold_scale"] / max(eval_length, 1.0)

    # Now populate summary with clean, deduplicated topline metrics
    reward_per_step = float(eval_reward / max(eval_length, 1.0))
    metrics_entry["summary"] = {
        "reward_per_step": reward_per_step,
        "reward_per_step_std": float(metrics.get("eval/episode_reward_std", 0.0) / max(eval_length, 1.0)),
        "total_rewards": total_rewards_weighted,  # FIXED: Use weighted values
        "total_penalties": total_penalties_weighted,  # FIXED: Use weighted values
        "success_rate": float(success),
        "forward_velocity": float(forward_vel),
        "episode_length": float(eval_length),
        "sps": float(sps),
        # Diagnostics (gating + scheduling)
        "tracking_gate_active_rate": float(gate_active_rate),
        "velocity_threshold_scale": float(velocity_threshold_scale_val),
    }

    # Add non-duplicate metrics to other bucket
    metrics_entry["other"]["velocity_command"] = float(metrics.get("eval/episode_velocity_command", 0.0) / max(eval_length, 1.0))
    metrics_entry["other"]["height"] = float(height)
    metrics_entry["other"]["distance_walked"] = float(distance)
    metrics_entry["other"]["walltime"] = time.time() - start_time
    metrics_entry["other"]["epoch_eval_time"] = float(metrics.get("eval/sps", 0.0))

    # Write to JSONL file
    metrics_log_file.write(json.dumps(metrics_entry) + "\n")
    metrics_log_file.flush()  # Ensure it's written immediately

    # Log to W&B with categorization
    if cfg["logging"]["use_wandb"]:
        # === TOPLINE METRICS (High-level RL training health indicators) ===
        topline_metrics = {
            "steps": num_steps,
            # Core RL metrics
            "topline/episode_reward": eval_reward,  # Total episode reward (weighted)
            "topline/reward_per_step": reward_per_step,  # Average reward per step (weighted)
            "topline/total_rewards": total_rewards_weighted if 'rewards_weighted' in metrics_entry else total_rewards,  # Sum of positive components
            "topline/total_penalties": total_penalties_weighted if 'penalties_weighted' in metrics_entry else total_penalties,  # Sum of negative components

            # Task performance metrics
            "topline/success_rate": success,
            "topline/forward_velocity": forward_vel,
            "topline/episode_length": eval_length,
            "topline/distance_walked": distance,

            # Robot state metrics
            "topline/height": height,

            # Training efficiency (moved to debug metrics)
            # Diagnostics (gating + scheduling)
            # NOTE: tracking-gate diagnostics are runtime/diagnostic fields and
            # should not be part of the high-level topline metrics. They are
            # instead logged under debug metrics below.
        }
        
        # Phase 1 contact metrics (if available)
        alternation_ratio = metrics.get("eval/episode_contact/alternation_ratio")
        if alternation_ratio is not None:
            topline_metrics["topline/alternation_ratio"] = alternation_ratio / max(eval_length, 1.0)
        
        avg_sliding = None
        left_sliding = metrics.get("eval/episode_contact/left_sliding_vel")
        right_sliding = metrics.get("eval/episode_contact/right_sliding_vel")
        if left_sliding is not None and right_sliding is not None:
            avg_sliding = (left_sliding + right_sliding) / (2.0 * max(eval_length, 1.0))
            topline_metrics["topline/avg_sliding_vel"] = avg_sliding

        # Static config gating parameters are diagnostic; move to debug logs.

        wandb.log(topline_metrics)

        # === DEBUG METRICS (Detailed unweighted component breakdown) ===
        debug_metrics = {"steps": num_steps}

        # All individual reward components (UNWEIGHTED - raw physical values)
        for key, value in metrics.items():
            if key.startswith("eval/episode_reward/"):
                component_name = key.replace("eval/episode_reward/", "")
                value_per_step = value / max(eval_length, 1.0)

                # Get weight to also log weighted value
                weight = float(cfg["reward_weights"].get(component_name, 0.0))
                weighted_value = value_per_step * weight

                # Log both unweighted (for physical interpretation) and weighted (for training impact)
                debug_metrics[f"debug/reward_unweighted/{component_name}"] = value_per_step
                debug_metrics[f"debug/reward_weighted/{component_name}"] = weighted_value

        # All contact metrics
        for key, value in metrics.items():
            if key.startswith("eval/episode_contact/"):
                component_name = key.replace("eval/episode_contact/", "")
                debug_metrics[f"debug/contact/{component_name}"] = value / max(eval_length, 1.0)

        # Other eval metrics
        for key, value in metrics.items():
            if key.startswith("eval/") and not key.startswith("eval/episode_reward/") and not key.startswith("eval/episode_contact/"):
                if key not in ["eval/episode_reward", "eval/avg_episode_length", "eval/episode_forward_velocity",
                               "eval/episode_height", "eval/episode_success", "eval/episode_distance_walked"]:
                    debug_name = key.replace("eval/episode_", "").replace("eval/", "")
                    debug_metrics[f"debug/other/{debug_name}"] = value / max(eval_length, 1.0) if "episode" in key else value

        # Add gating diagnostics, scheduling and runtime diagnostics to debug metrics
        debug_metrics["debug/other/tracking_gate_active_rate"] = float(gate_active_rate)
        debug_metrics["debug/other/velocity_threshold_scale"] = float(velocity_threshold_scale_val)
        # Log training steps-per-second as a debug/runtime metric (not topline)
        debug_metrics["debug/other/sps"] = float(sps)
        if cfg["reward_weights"].get("tracking_gate_velocity") is not None:
            debug_metrics["debug/config/tracking_gate_velocity"] = cfg["reward_weights"].get("tracking_gate_velocity")
        if cfg["reward_weights"].get("tracking_gate_scale") is not None:
            debug_metrics["debug/config/tracking_gate_scale"] = cfg["reward_weights"].get("tracking_gate_scale")

        wandb.log(debug_metrics)

    # Return extracted metrics for progress printing
    return {
        "eval_reward": eval_reward,
        "eval_length": eval_length,
        "forward_vel": forward_vel,
        "height": height,
        "success": success,
        "distance": distance,
        "sps": sps,
    }


class ProgressCallback:
    """Progress callback for tracking training metrics.

    This class wraps the progress tracking logic and provides a callable
    interface for the PPO training loop.
    """

    def __init__(self, times_list, start_time, cfg, ppo_config, metrics_log_file, quick_verify, diagnostic=False, 
                 ckpt_dir=None, save_best=False, best_reward_tracker=None):
        """Initialize progress callback.

        Args:
            times_list: List to track timing (will be mutated)
            start_time: Training start time
            cfg: Config dict
            ppo_config: PPO configuration
            metrics_log_file: File handle for JSONL logging
            quick_verify: Boolean flag for verbose output
            diagnostic: Boolean flag for diagnostic logging
            ckpt_dir: Directory for checkpoints (for best checkpoint saving)
            save_best: Whether to save best checkpoints
            best_reward_tracker: Shared dict to track best reward
        """
        self.times_list = times_list
        self.start_time = start_time
        self.cfg = cfg
        self.ppo_config = ppo_config
        self.metrics_log_file = metrics_log_file
        self.quick_verify = quick_verify
        self.diagnostic = diagnostic
        self.first_evals = 0  # Track first few evals for diagnostic
        self.ckpt_dir = ckpt_dir
        self.save_best = save_best
        self.best_reward_tracker = best_reward_tracker

    def __call__(self, num_steps, metrics):
        """Progress callback function called by PPO training loop.

        Args:
            num_steps: Current training step
            metrics: Metrics dict from training
        """
        self.times_list.append(time.monotonic())
        elapsed = self.times_list[-1] - self.times_list[1] if len(self.times_list) > 1 else 0

        # Generate and log metrics (JSONL + W&B) - returns extracted metrics
        extracted_metrics = generate_and_log_metrics(
            num_steps=num_steps,
            metrics=metrics,
            elapsed=elapsed,
            start_time=self.start_time,
            cfg=self.cfg,
            ppo_config=self.ppo_config,
            metrics_log_file=self.metrics_log_file,
        )

        # Unpack metrics for printing
        eval_reward = extracted_metrics["eval_reward"]
        eval_length = extracted_metrics["eval_length"]
        forward_vel = extracted_metrics["forward_vel"]
        height = extracted_metrics["height"]
        success = extracted_metrics["success"]
        distance = extracted_metrics["distance"]
        sps = extracted_metrics["sps"]

        # Progress bar
        progress_pct = (num_steps / self.ppo_config.num_timesteps) * 100
        eta_seconds = (self.ppo_config.num_timesteps - num_steps) / sps if sps > 0 else 0
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Step {num_steps:,}/{self.ppo_config.num_timesteps:,} ({progress_pct:.1f}%) | "
              f"Reward: {eval_reward:.2f} | Vel: {forward_vel:.3f} m/s | "
              f"Height: {height:.3f}m | Success: {success:.2%} | "
              f"SPS: {sps:.0f} | ETA: {eta}")

        # Best checkpoint tracking (store step, progress_fn doesn't have params)
        if self.save_best and self.best_reward_tracker is not None:
            reward_per_step = eval_reward / max(eval_length, 1.0)
            if reward_per_step > self.best_reward_tracker['best_reward']:
                self.best_reward_tracker['best_reward'] = reward_per_step
                self.best_reward_tracker['best_step'] = num_steps
                print(f"  🏆 New best reward: {reward_per_step:.3f}/step at {num_steps:,}")

        # DIAGNOSTIC LOGGING FOR FIRST FEW EVALS
        if self.diagnostic and self.first_evals < 5:
            self.first_evals += 1
            print(f"\n{'='*80}")
            print(f"DIAGNOSTIC - Eval #{self.first_evals} at step {num_steps:,}")
            print(f"{'='*80}")
            print(f"  Forward velocity:     {forward_vel:+.4f} m/s")
            print(f"  Episode length:       {eval_length:.1f} steps")
            print(f"  Height:               {height:.3f} m")
            print(f"  Distance walked:      {distance:.3f} m")
            print(f"  Velocity command:     {metrics.get('eval/episode_velocity_command', 0.0) / max(eval_length, 1.0):.3f} m/s")
            
            # Extract detailed reward breakdown
            print(f"\n  Reward breakdown:")
            for key, value in sorted(metrics.items()):
                if key.startswith("eval/episode_reward/"):
                    component = key.replace("eval/episode_reward/", "")
                    per_step = float(value / max(eval_length, 1.0))
                    if abs(per_step) > 0.01:  # Only show significant components
                        weight = float(self.cfg["reward_weights"].get(component, 0.0))
                        weighted = per_step * weight
                        print(f"    {component:30s}: {per_step:+8.3f} × {weight:6.2f} = {weighted:+8.3f}")
            print(f"{'='*80}\n")

        # VERBOSE LOGGING FOR QUICK_VERIFY MODE
        if self.quick_verify:
            print("\n" + "="*70)
            print("📊 DETAILED REWARD BREAKDOWN (per step)")
            print("="*70)

            # Collect all reward components
            reward_components = {}
            for key, value in metrics.items():
                if key.startswith("eval/episode_reward/"):
                    component_name = key.replace("eval/episode_reward/", "")
                    reward_per_step = value / max(eval_length, 1.0)
                    reward_components[component_name] = reward_per_step

            # Sort by absolute value (largest penalties/rewards first)
            sorted_components = sorted(reward_components.items(), key=lambda x: abs(x[1]), reverse=True)

            print("\n🔴 Top Penalties/Rewards:")
            for i, (name, value) in enumerate(sorted_components[:10], 1):
                sign = "+" if value >= 0 else ""
                print(f"  {i:2d}. {name:30s} {sign}{value:>8.3f}")

            # Calculate total from components
            total_from_components = sum(reward_components.values())
            print(f"\n{'='*70}")
            print(f"Total from components:  {total_from_components:>8.3f} per step")
            print(f"Reported reward:        {eval_reward / max(eval_length, 1.0):>8.3f} per step")
            print(f"Episode reward:         {eval_reward:>8.3f} (over {eval_length:.0f} steps)")
            print(f"{'='*70}")

            # Show contact metrics
            print("\n📍 CONTACT METRICS:")
            contact_metrics = {}
            for key, value in metrics.items():
                if key.startswith("eval/episode_contact/"):
                    metric_name = key.replace("eval/episode_contact/", "")
                    metric_per_step = value / max(eval_length, 1.0)
                    contact_metrics[metric_name] = metric_per_step

            for name, value in sorted(contact_metrics.items()):
                print(f"  {name:40s} {value:>8.3f}")

            print("="*70 + "\n")


# Flags
_CONFIG = flags.DEFINE_string(
    "config", "phase1_contact.yaml", "Config file name (in amp/)"
)
_QUICK_VERIFY = flags.DEFINE_boolean(
    "quick_verify", False, "Run quick verification mode (~20 seconds)"
)
_TERRAIN = flags.DEFINE_string(
    "terrain", None, "Terrain type: 'flat' or 'rough' (overrides config)"
)
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", None, "Number of timesteps (overrides config)"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", None, "Number of parallel environments (overrides config)"
)
_NUM_EVALS = flags.DEFINE_integer(
    "num_evals", None, "Number of evaluations (overrides config)"
)
_SEED = flags.DEFINE_integer("seed", None, "Random seed (overrides config)")
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint", None, "Path to checkpoint to load"
)
_SKIP_TRAIN_IF_CKPT = flags.DEFINE_boolean(
    "skip_training_if_checkpoint", False, "If --load_checkpoint is provided, skip training and run eval/render using the checkpoint"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", None, "Use Weights & Biases (overrides config)"
)
_RENDER_ONLY = flags.DEFINE_boolean(
    "render_only", False, "Skip training, only render videos from existing checkpoint"
)
_RENDER_CHECKPOINT = flags.DEFINE_string(
    "render_checkpoint", None, "Checkpoint directory for rendering (required if --render_only)"
)
_RENDER_NUM_VIDEOS = flags.DEFINE_integer(
    "render_num_videos", 3, "Number of videos to render"
)
_RENDER_CAMERA = flags.DEFINE_string(
    "render_camera", "track", "Camera name for rendering: track, side, front, top"
)
_DIAGNOSTIC = flags.DEFINE_boolean(
    "diagnostic", False, "Enable diagnostic logging for velocity/heading debugging"
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_quick_verify_overrides(cfg: dict) -> dict:
    """Apply quick_verify overrides to config."""
    qv = cfg.get("quick_verify", {})

    # Override training params
    cfg["training"]["num_timesteps"] = qv.get("num_timesteps", 10000)
    cfg["training"]["num_evals"] = qv.get("num_evals", 2)
    cfg["training"]["episode_length"] = qv.get("episode_length", 100)

    # Override PPO params
    cfg["ppo"]["num_envs"] = qv.get("num_envs", 128)
    cfg["ppo"]["num_eval_envs"] = qv.get("num_eval_envs", 16)

    # Override logging/rendering
    cfg["logging"]["use_wandb"] = qv.get("use_wandb", False)
    cfg["rendering"]["render_videos"] = qv.get("render_videos", False)

    print("\n" + "="*80)
    print("🔍 QUICK VERIFY MODE ENABLED")
    print("="*80)
    print(f"  Timesteps: {cfg['training']['num_timesteps']:,} (~20 seconds)")
    print(f"  Evals: {cfg['training']['num_evals']}")
    print(f"  Episode length: {cfg['training']['episode_length']}")
    print(f"  Num envs: {cfg['ppo']['num_envs']}")
    print(f"  W&B logging: {cfg['logging']['use_wandb']}")
    print(f"  Video rendering: {cfg['rendering']['render_videos']}")
    print("="*80 + "\n")

    return cfg


def create_env_config(cfg: dict) -> config_dict.ConfigDict:
    """Create environment config from main config dict.

    Extracts environment-specific parameters from the full config and creates
    a ConfigDict suitable for environment initialization.

    Args:
        cfg: Main config dict loaded from YAML

    Returns:
        ConfigDict for environment initialization

    Raises:
        ValueError: If required config sections or parameters are missing
    """
    # Validate required sections exist
    if "env" not in cfg:
        raise ValueError("Config missing required 'env' section")
    if "reward_weights" not in cfg:
        raise ValueError("Config missing required 'reward_weights' section")

    env_config = config_dict.ConfigDict()

    # Environment parameters (exclude 'terrain' which is used for task name)
    env_params = [
        "ctrl_dt", "sim_dt",
        "velocity_command_mode", "min_velocity", "max_velocity",
        "use_action_filter", "action_filter_alpha",
        "use_phase_signal", "phase_period", "num_phase_clocks",
        "min_height", "max_height",
    ]

    # Validate and copy each parameter
    for param in env_params:
        if param not in cfg["env"]:
            raise ValueError(f"Config missing required env parameter: '{param}'")
        env_config[param] = cfg["env"][param]

    # Add reward weights (from top-level config)
    env_config.reward_weights = cfg["reward_weights"]

    return env_config


def main(argv):
    del argv

    logging.set_verbosity(logging.INFO)

    # === RENDER-ONLY MODE ===
    if _RENDER_ONLY.value:
        if not _RENDER_CHECKPOINT.value:
            raise ValueError("--render_checkpoint is required when using --render_only")

        import pickle

        checkpoint_dir = Path(_RENDER_CHECKPOINT.value)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Load config from checkpoint
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        print(f"\n{'='*80}")
        print(f"WildRobot Video Rendering (Render-Only Mode)")
        print(f"{'='*80}")
        print(f"  Checkpoint: {checkpoint_dir}")
        print(f"  Config: {config_path.name}")
        print(f"{'='*80}\n")

        # Load checkpoint
        checkpoint_path = checkpoint_dir / "checkpoints" / "final_policy.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            params = pickle.load(f)
        print(f"✓ Checkpoint loaded\n")

        # Create environment config using helper function
        env_config = create_env_config(cfg)

        # Create environment
        terrain = cfg["env"]["terrain"]
        task_name = f"wildrobot_{terrain}"
        env = walk_env.WildRobotWalkEnv(task=task_name, config=env_config)

        print(f"Environment: {env.__class__.__name__}")
        print(f"  Task: {task_name}")
        print(f"  Observation size: {env.observation_size}")
        print(f"  Action size: {env.action_size}\n")

        # Output directory
        video_dir = checkpoint_dir / "rendered_videos"
        video_dir.mkdir(exist_ok=True)

        # Render videos using shared function
        render_videos(
            env=env,
            params=params,
            cfg=cfg,
            output_dir=video_dir,
            num_videos=_RENDER_NUM_VIDEOS.value,
            max_steps=600,
            camera=_RENDER_CAMERA.value,
            seed=cfg["training"]["seed"],
            make_inference_fn=None,  # Will be created from params in render_videos()
        )

        return

    # === NORMAL TRAINING MODE ===

    # Load config
    config_path = Path(__file__).parent / _CONFIG.value
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    # Apply quick verify if requested
    if _QUICK_VERIFY.value or cfg.get("quick_verify", {}).get("enabled", False):
        cfg = apply_quick_verify_overrides(cfg)

    # Apply CLI overrides
    if _TERRAIN.value:
        cfg["env"]["terrain"] = _TERRAIN.value
    if _NUM_TIMESTEPS.value:
        cfg["training"]["num_timesteps"] = _NUM_TIMESTEPS.value
    if _NUM_ENVS.value:
        cfg["ppo"]["num_envs"] = _NUM_ENVS.value
    if _NUM_EVALS.value:
        cfg["training"]["num_evals"] = _NUM_EVALS.value
    if _SEED.value:
        cfg["training"]["seed"] = _SEED.value
    if _USE_WANDB.value is not None:
        cfg["logging"]["use_wandb"] = _USE_WANDB.value

    # Extract config values
    terrain = cfg["env"]["terrain"]
    task_name = f"wildrobot_{terrain}"

    print(f"\n{'='*80}")
    print(f"WildRobot AMP Training")
    print(f"{'='*80}")
    print(f"  Config: {_CONFIG.value}")
    print(f"  Terrain: {terrain}")
    print(f"  Task: {task_name}")
    print(f"{'='*80}\n")

    # Create environment config using helper function
    # Optional warm-start easing parameters
    warm_cfg = cfg.get("training", {}).get("warm_start", {})
    lr_warmup_scale = float(warm_cfg.get("lr_warmup_scale", 1.0))
    gate_ease_scale = float(warm_cfg.get("gate_ease_scale", 1.0))

    # Preserve original gate value for possible two-phase training
    original_gate = cfg["reward_weights"].get("tracking_gate_scale", None)

    env_config = create_env_config(cfg)

    # Create environment
    env = walk_env.WildRobotWalkEnv(task=task_name, config=env_config)

    print(f"Environment: {env.__class__.__name__}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size: {env.action_size}")
    print(f"  Velocity range: {env_config.min_velocity}-{env_config.max_velocity} m/s")

    # Show active reward components
    print(f"\nActive Reward Components:")
    for name, weight in cfg["reward_weights"].items():
        if weight != 0.0 and name not in ["tracking_sigma"]:
            print(f"  {name}: {weight}")

    # PPO config
    ppo_config = config_dict.ConfigDict()
    ppo_config.num_timesteps = cfg["training"]["num_timesteps"]
    ppo_config.num_evals = cfg["training"]["num_evals"]
    ppo_config.reward_scaling = cfg["ppo"]["reward_scaling"]
    ppo_config.episode_length = cfg["training"]["episode_length"]
    ppo_config.normalize_observations = cfg["ppo"]["normalize_observations"]
    ppo_config.action_repeat = cfg["ppo"]["action_repeat"]
    ppo_config.unroll_length = cfg["ppo"]["unroll_length"]
    ppo_config.num_minibatches = cfg["ppo"]["num_minibatches"]
    ppo_config.num_updates_per_batch = cfg["ppo"]["num_updates_per_batch"]
    ppo_config.discounting = cfg["ppo"]["discounting"]
    ppo_config.learning_rate = cfg["ppo"]["learning_rate"]
    ppo_config.entropy_cost = cfg["ppo"]["entropy_cost"]
    ppo_config.num_envs = cfg["ppo"]["num_envs"]
    ppo_config.batch_size = cfg["ppo"]["batch_size"]
    ppo_config.max_grad_norm = cfg["ppo"]["max_grad_norm"]
    ppo_config.clipping_epsilon = cfg["ppo"]["clipping_epsilon"]

    # Network config
    network_config = config_dict.ConfigDict()
    network_config.policy_hidden_layer_sizes = cfg["network"]["policy_hidden_layers"]
    network_config.value_hidden_layer_sizes = cfg["network"]["value_hidden_layers"]

    print(f"\nTraining Configuration:")
    print(f"  Timesteps: {ppo_config.num_timesteps:,}")
    print(f"  Environments: {ppo_config.num_envs}")
    print(f"  Batch size: {ppo_config.batch_size}")
    print(f"  Learning rate: {ppo_config.learning_rate}")
    print(f"  Episode length: {ppo_config.episode_length}")
    print(f"  Policy network: {network_config.policy_hidden_layer_sizes}")
    print(f"  Value network: {network_config.value_hidden_layer_sizes}")

    # Setup experiment directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    config_name = Path(_CONFIG.value).stem  # e.g., "phase1_contact"

    if _QUICK_VERIFY.value:
        exp_name = f"quickverify_{config_name}_{timestamp}"
    else:
        exp_name = f"{config_name}_{terrain}_{timestamp}"

    # Consolidated training_logs directory structure
    amp_dir = Path(__file__).parent
    exp_dir = amp_dir / "training_logs" / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    # Create directories
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"  Directory: {exp_dir}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Logs: {log_dir}\n")

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Initialize W&B
    if cfg["logging"]["use_wandb"]:
        wandb_tags = cfg["logging"].get("wandb_tags", [])
        if _QUICK_VERIFY.value:
            wandb_tags.append("quick_verify")

        # Define metric groups for W&B
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=exp_name,
            config=cfg,
            tags=wandb_tags,
            dir=str(log_dir),
        )

        # Configure metric grouping and categories
        wandb.define_metric("steps")
        wandb.define_metric("topline/*", step_metric="steps")
        wandb.define_metric("debug/*", step_metric="steps")

        print(f"W&B initialized: {wandb.run.url}\n")

    # Network factory with explicit architecture
    def network_factory(observation_size, action_size, preprocess_observations_fn=None):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=network_config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=network_config.value_hidden_layer_sizes,
        )

    # Optional warm-start: load pickle checkpoint params
    initial_params = None
    checkpoint_path = _LOAD_CHECKPOINT.value
    if checkpoint_path:
        import pickle
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                initial_params = pickle.load(f)
            
            # Validate network architecture compatibility
            try:
                # Brax checkpoint structure: params is a tuple (policy_params, value_params, ...)
                if isinstance(initial_params, tuple):
                    policy_params = initial_params[0]  # First element is policy params
                else:
                    # Newer format might be a dict
                    policy_params = initial_params.get('policy', initial_params)
                
                # Navigate to first hidden layer kernel
                if 'params' in policy_params:
                    policy_shape = policy_params['params']['hidden_0']['kernel'].shape
                else:
                    policy_shape = policy_params['hidden_0']['kernel'].shape
                    
                expected_shape = (env.observation_size, network_config.policy_hidden_layer_sizes[0])
                if policy_shape != expected_shape:
                    print(f"\nERROR: Network architecture mismatch!")
                    print(f"  Checkpoint has: obs_size={policy_shape[0]}, first_hidden={policy_shape[1]}")
                    print(f"  Current config: obs_size={expected_shape[0]}, first_hidden={expected_shape[1]}")
                    print(f"  Config policy_hidden_layers: {network_config.policy_hidden_layer_sizes}")
                    print(f"\nHint: Ensure network.policy_hidden_layers in {_CONFIG.value} matches the checkpoint.")
                    print(f"      Phase 0 uses [256, 256, 128]. Update your YAML file to match.\n")
                    sys.exit(1)
                print(f"Warm-start enabled: loaded params from {checkpoint_path}")
                print(f"  Network architecture validated: {network_config.policy_hidden_layer_sizes}")
            except Exception as e:
                print(f"Warning: Could not validate checkpoint architecture: {e}")
                print(f"Warm-start enabled: loaded params from {checkpoint_path}")
        else:
            print(f"Warning: checkpoint not found: {checkpoint_path}")

    # Progress callback
    times = [time.monotonic()]
    start_time = time.time()  # Track absolute start time for walltime

    # Create metrics log file
    metrics_log_path = exp_dir / "metrics.jsonl"
    metrics_log_file = open(metrics_log_path, "w")
    print(f"Metrics log: {metrics_log_path}\n")

    # Checkpoint configuration
    checkpoint_config = cfg.get('checkpointing', {})
    save_interval = checkpoint_config.get('save_interval', 500_000)  # Default 500K
    save_best = checkpoint_config.get('save_best', True)
    
    checkpoint_tracker = {
        'best_reward': float('-inf'),
        'best_step': 0,
        'last_save_step': 0,
    }
    
    # Create progress callback using class (no nested functions!)
    progress = ProgressCallback(
        times_list=times,
        start_time=start_time,
        cfg=cfg,
        ppo_config=ppo_config,
        metrics_log_file=metrics_log_file,
        quick_verify=_QUICK_VERIFY.value,
        diagnostic=_DIAGNOSTIC.value,
        ckpt_dir=ckpt_dir,
        save_best=save_best,
        best_reward_tracker=checkpoint_tracker,
    )
    
    def policy_params_callback(step, make_policy, params):
        """Callback to save checkpoints during training.
        
        Brax calls this with (step, make_policy, params) - no metrics included.
        We only do periodic saves here, not best checkpoint (no metrics available).
        
        Args:
            step: Current training step
            make_policy: Policy function factory
            params: Network parameters (dict with normalizer_params, policy_params, etc.)
        """
        import pickle
        
        # Periodic checkpoint saving (configurable interval)
        if step - checkpoint_tracker['last_save_step'] >= save_interval:
            ckpt_path = ckpt_dir / f"checkpoint_step_{step}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(params, f)
            checkpoint_tracker['last_save_step'] = step
            print(f"  💾 Saved periodic checkpoint at step {step:,}: {ckpt_path.name}")

    # Quick verify + warm-start incompatibility check
    if _QUICK_VERIFY.value and initial_params is not None:
        print("\nERROR: Cannot use --quick_verify with --load_checkpoint (warm-start).")
        print("  Quick verify uses a lightweight config that may not match checkpoint architecture.")
        print("  For warm-start testing, run a short full training instead:")
        print(f"    uv run amp/train.py --config {_CONFIG.value} --load_checkpoint {checkpoint_path} --num_timesteps 50000\n")
        sys.exit(1)
    
    # Train or skip based on checkpoint usage
    make_inference_fn = None  # Ensure defined for both branches
    if initial_params is not None and _SKIP_TRAIN_IF_CKPT.value:
        print("Checkpoint provided and --skip_training_if_checkpoint set: skipping training.")
        params = initial_params

        # Run a short eval rollout via renderer to produce metrics
        try:
            eval_dir = exp_dir / "eval_skip_training"
            eval_dir.mkdir(exist_ok=True)

            # Use 1 video, short horizon for quick verification
            results = render_videos(
                env=env,
                params=params,
                cfg=cfg,
                output_dir=eval_dir,
                num_videos=1,
                camera="track",
                seed=cfg["training"]["seed"],
                make_inference_fn=make_inference_fn,
            )

            # Log topline metrics from the rollout
            if results:
                _, stats = results[0]
                summary_entry = {
                    "step": 0,
                    "timestamp": time.time(),
                    "summary": {
                        "reward_per_step": None,
                        "total_rewards": None,
                        "total_penalties": None,
                        "success_rate": 0.0,
                        "forward_velocity": float(stats.get("velocity", 0.0)),
                        "episode_length": float(stats.get("length", 0.0)),
                        "sps": 0.0,
                        "tracking_gate_active_rate": 0.0,
                        "velocity_threshold_scale": 1.0,
                    },
                    "other": {
                        "height": float(stats.get("height", 0.0)),
                        "distance_walked": float(stats.get("velocity", 0.0)) * float(stats.get("length", 0.0)) * env_config.ctrl_dt,
                        "walltime": 0.0,
                    },
                }
                metrics_log_file.write(json.dumps(summary_entry) + "\n")
                metrics_log_file.flush()
                print(f"Eval (skip training): vel={summary_entry['summary']['forward_velocity']:.3f} m/s | "
                      f"len={summary_entry['summary']['episode_length']:.0f} steps | "
                      f"height={summary_entry['other']['height']:.3f} m\n")
        except Exception as e:
            print(f"Warning: eval during skip-training failed: {e}")
    else:
        # Warm-start stabilization tweaks when training from checkpoint
        if initial_params is not None:
            original_lr = ppo_config.learning_rate
            original_entropy = ppo_config.entropy_cost
            scaled_lr = float(original_lr) * (lr_warmup_scale if lr_warmup_scale != 1.0 else 0.5)
            ppo_config.learning_rate = scaled_lr
            ppo_config.entropy_cost = float(original_entropy) * 1.5
            print(
                f"Warm-start stabilization: learning_rate {original_lr}→{ppo_config.learning_rate} (scale), "
                f"entropy_cost {original_entropy}→{ppo_config.entropy_cost}"
            )

        print("Starting training...\n")
        
        # Train with warm-start via restore_params (Brax 0.10.4+)
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=ppo_config.num_timesteps,
            num_evals=ppo_config.num_evals,
            reward_scaling=ppo_config.reward_scaling,
            episode_length=ppo_config.episode_length,
            normalize_observations=ppo_config.normalize_observations,
            action_repeat=ppo_config.action_repeat,
            unroll_length=ppo_config.unroll_length,
            num_minibatches=ppo_config.num_minibatches,
            num_updates_per_batch=ppo_config.num_updates_per_batch,
            discounting=ppo_config.discounting,
            learning_rate=ppo_config.learning_rate,
            entropy_cost=ppo_config.entropy_cost,
            num_envs=ppo_config.num_envs,
            batch_size=ppo_config.batch_size,
            seed=cfg["training"]["seed"],
            network_factory=network_factory,
            progress_fn=progress,
            policy_params_fn=policy_params_callback,  # Checkpoint saving callback
            max_grad_norm=ppo_config.max_grad_norm,
            clipping_epsilon=ppo_config.clipping_epsilon,
            restore_params=initial_params,  # None if not warm-starting
        )

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}\n")

    # Close metrics log file
    metrics_log_file.close()
    print(f"Metrics saved to: {metrics_log_path}\n")

    # Save final checkpoint
    if not _QUICK_VERIFY.value and not (_SKIP_TRAIN_IF_CKPT.value and initial_params is not None):
        import pickle
        final_ckpt_path = ckpt_dir / "final_policy.pkl"
        with open(final_ckpt_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Saved final checkpoint: {final_ckpt_path}\n")
        
        # Report best checkpoint info
        if save_best and checkpoint_tracker['best_step'] > 0:
            print(f"Best checkpoint was at step {checkpoint_tracker['best_step']:,} "
                  f"with reward {checkpoint_tracker['best_reward']:.3f}/step")
            print(f"Note: Best checkpoint saved during training (check checkpoint_step_*.pkl files)\n")

    # Render videos if enabled
    should_render_videos = cfg["rendering"]["render_videos"]
    if _QUICK_VERIFY.value:
        # In quick_verify mode, use the quick_verify.render_videos setting instead
        should_render_videos = cfg.get("quick_verify", {}).get("render_videos", False)

    if should_render_videos:
        # Render videos using shared function
        results = render_videos(
            env=env,
            params=params,
            cfg=cfg,
            output_dir=exp_dir,
            num_videos=3,
            max_steps=min(cfg["training"]["episode_length"], 600),
            camera="track",
            seed=cfg["training"]["seed"],
            make_inference_fn=make_inference_fn,  # From training
        )

        # Upload to W&B if enabled
        if cfg["logging"]["use_wandb"]:
            fps = int(1.0 / env_config.ctrl_dt)
            for i, (video_path, stats) in enumerate(results):
                wandb.log({f"eval_video_{i}": wandb.Video(str(video_path), fps=fps)})
            print(f"  ✓ Videos uploaded to W&B\n")

    if cfg["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
