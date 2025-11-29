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
    # Check if running in SSH session
    is_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    
    if is_ssh:
        # SSH session - prefer osmesa (software rendering) or EGL with xvfb-run
        import sys
        try:
            import ctypes
            osmesa_available = False
            for lib_name in ['libOSMesa.so', 'libOSMesa.so.8', 'libOSMesa.so.6']:
                try:
                    ctypes.CDLL(lib_name)
                    osmesa_available = True
                    print(f"SSH session detected. OSMesa found: {lib_name}")
                    break
                except OSError:
                    continue

            if osmesa_available:
                os.environ["MUJOCO_GL"] = "osmesa"
                print("Using osmesa renderer (software rendering)")
            else:
                os.environ["MUJOCO_GL"] = "egl"
                print("OSMesa not found, using egl renderer")
                print("Note: Run with 'xvfb-run -a python ...' for headless rendering")
        except Exception as e:
            print(f"Warning: Could not detect rendering backend: {e}")
            os.environ["MUJOCO_GL"] = "egl"
    else:
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
_SEED = flags.DEFINE_integer("seed", None, "Random seed (overrides config)")
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint", None, "Path to checkpoint to load"
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


def main(argv):
    del argv

    logging.set_verbosity(logging.INFO)

    # === RENDER-ONLY MODE ===
    if _RENDER_ONLY.value:
        if not _RENDER_CHECKPOINT.value:
            raise ValueError("--render_checkpoint is required when using --render_only")

        from pathlib import Path
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

        # Create environment config
        env_config = config_dict.ConfigDict()
        env_config.ctrl_dt = cfg["env"]["ctrl_dt"]
        env_config.sim_dt = cfg["env"]["sim_dt"]
        env_config.velocity_command_mode = cfg["env"]["velocity_command_mode"]
        env_config.min_velocity = cfg["env"]["min_velocity"]
        env_config.max_velocity = cfg["env"]["max_velocity"]
        env_config.use_action_filter = cfg["env"]["use_action_filter"]
        env_config.action_filter_alpha = cfg["env"]["action_filter_alpha"]
        env_config.use_phase_signal = cfg["env"]["use_phase_signal"]
        env_config.phase_period = cfg["env"]["phase_period"]
        env_config.num_phase_clocks = cfg["env"]["num_phase_clocks"]
        env_config.min_height = cfg["env"]["min_height"]
        env_config.max_height = cfg["env"]["max_height"]
        env_config.reward_weights = cfg["reward_weights"]

        # Create environment
        terrain = cfg["env"]["terrain"]
        task_name = f"wildrobot_{terrain}"
        env = walk_env.WildRobotWalkEnv(task=task_name, config=env_config)

        print(f"Environment: {env.__class__.__name__}")
        print(f"  Task: {task_name}")
        print(f"  Observation size: {env.observation_size}")
        print(f"  Action size: {env.action_size}\n")

        # Create network factory
        network_config = cfg["network"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=network_config["policy_hidden_layers"],
            value_hidden_layer_sizes=network_config["value_hidden_layers"],
        )

        # Create inference function from params
        print("Creating inference function...")
        
        # Build network to get the structure  
        ppo_networks_tuple = network_factory(env.observation_size, env.action_size)
        policy_network = ppo_networks_tuple.policy_network

        # Extract params - structure is (normalizer_params, policy_params, value_params)
        normalizer_params = params[0]  # RunningStatisticsState
        policy_params = params[1]  # Policy params dict

        # Create inference function
        # policy_network.apply signature: (processor_params, policy_params, obs)
        def inference_fn(obs, rng):
            """Run policy inference - deterministic (use mean)."""
            # Apply policy network with normalizer and policy params
            logits = policy_network.apply(normalizer_params, policy_params, obs)
            
            # For Gaussian policies, logits contains [mean, log_std]
            # Use just the mean for deterministic evaluation
            action_dim = env.action_size
            mean_action = logits[:action_dim]
            
            return mean_action, {}

        print("✓ Inference function created\n")

        # JIT compile environment functions only
        print("JIT compiling environment...")
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        print("✓ JIT compilation complete\n")

        # Rendering settings
        num_videos = _RENDER_NUM_VIDEOS.value
        max_steps = 600
        camera = _RENDER_CAMERA.value
        render_height = cfg["rendering"]["render_height"]
        render_width = cfg["rendering"]["render_width"]
        fps = int(1.0 / env_config.ctrl_dt)
        seed = cfg["training"]["seed"]

        # Check if requested camera exists, otherwise use default
        available_cameras = [env.mj_model.camera(i).name for i in range(env.mj_model.ncam)]
        if camera not in available_cameras:
            if available_cameras:
                camera = available_cameras[0]
                print(f"⚠️  Camera '{_RENDER_CAMERA.value}' not found. Using '{camera}' instead.")
                print(f"    Available cameras: {', '.join(available_cameras)}")
            else:
                camera = None  # Will use default free camera
                print(f"⚠️  No named cameras found in model. Using default free camera.")

        print(f"\nRendering settings:")
        print(f"  Number of videos: {num_videos}")
        print(f"  Max steps: {max_steps}")
        print(f"  Resolution: {render_width}x{render_height}")
        print(f"  Camera: {camera if camera else 'free (default)'}")
        print(f"  FPS: {fps}")
        print(f"  Seed: {seed}\n")

        # Output directory
        video_dir = checkpoint_dir / "rendered_videos"
        video_dir.mkdir(exist_ok=True)

        # Create VideoRenderer
        renderer = VideoRenderer(
            env=env,
            output_dir=video_dir,
            width=render_width,
            height=render_height,
            fps=fps,
            camera=camera,
            max_steps=max_steps,
        )

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
        print(f"{'='*80}")

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

    # Create environment config
    env_config = config_dict.ConfigDict()
    env_config.ctrl_dt = cfg["env"]["ctrl_dt"]
    env_config.sim_dt = cfg["env"]["sim_dt"]
    env_config.velocity_command_mode = cfg["env"]["velocity_command_mode"]
    env_config.min_velocity = cfg["env"]["min_velocity"]
    env_config.max_velocity = cfg["env"]["max_velocity"]
    env_config.use_action_filter = cfg["env"]["use_action_filter"]
    env_config.action_filter_alpha = cfg["env"]["action_filter_alpha"]
    env_config.use_phase_signal = cfg["env"]["use_phase_signal"]
    env_config.phase_period = cfg["env"]["phase_period"]
    env_config.num_phase_clocks = cfg["env"]["num_phase_clocks"]
    env_config.min_height = cfg["env"]["min_height"]
    env_config.max_height = cfg["env"]["max_height"]

    # Pass reward weights to environment
    env_config.reward_weights = cfg["reward_weights"]

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
    print(f"  Network: {network_config.policy_hidden_layer_sizes}")

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

    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=network_config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=network_config.value_hidden_layer_sizes,
    )

    # Progress callback
    times = [time.monotonic()]
    start_time = time.time()  # Track absolute start time for walltime

    # Create metrics log file
    metrics_log_path = exp_dir / "metrics.jsonl"
    metrics_log_file = open(metrics_log_path, "w")
    print(f"Metrics log: {metrics_log_path}\n")

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        elapsed = times[-1] - times[1] if len(times) > 1 else 0

        # Extract metrics
        eval_reward = metrics.get("eval/episode_reward", 0.0)
        eval_length = metrics.get("eval/avg_episode_length", 0.0)
        forward_vel = metrics.get("eval/episode_forward_velocity", 0.0) / max(eval_length, 1.0)
        height = metrics.get("eval/episode_height", 0.0) / max(eval_length, 1.0)
        success = metrics.get("eval/episode_success", 0.0)
        distance = metrics.get("eval/episode_distance_walked", 0.0)

        # Progress bar
        progress_pct = (num_steps / ppo_config.num_timesteps) * 100
        sps = num_steps / elapsed if elapsed > 0 else 0
        eta_seconds = (ppo_config.num_timesteps - num_steps) / sps if sps > 0 else 0
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Step {num_steps:,}/{ppo_config.num_timesteps:,} ({progress_pct:.1f}%) | "
              f"Reward: {eval_reward:.2f} | Vel: {forward_vel:.3f} m/s | "
              f"Height: {height:.3f}m | Success: {success:.2%} | "
              f"SPS: {sps:.0f} | ETA: {eta}")

        # VERBOSE LOGGING FOR QUICK_VERIFY MODE
        if _QUICK_VERIFY.value:
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

        for key, value in metrics.items():
            value_per_step = float(value / max(eval_length, 1.0))

            if key.startswith("eval/episode_reward/"):
                component_name = key.replace("eval/episode_reward/", "")

                # Only track components listed in config
                if component_name not in tracked_components:
                    continue

                # Separate positive (rewards) from negative (penalties)
                if value_per_step >= 0:
                    # Positive component - it's a reward
                    metrics_entry["rewards"][component_name] = value_per_step
                    total_rewards += value_per_step
                else:
                    # Negative component - it's a penalty (store as positive for clarity)
                    metrics_entry["penalties"][component_name] = abs(value_per_step)
                    total_penalties += abs(value_per_step)

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

        # Now populate summary with clean, deduplicated topline metrics
        reward_per_step = float(eval_reward / max(eval_length, 1.0))
        metrics_entry["summary"] = {
            "reward_per_step": reward_per_step,
            "reward_per_step_std": float(metrics.get("eval/episode_reward_std", 0.0) / max(eval_length, 1.0)),
            "total_rewards": total_rewards,
            "total_penalties": total_penalties,
            "success_rate": float(success),
            "forward_velocity": float(forward_vel),
            "episode_length": float(eval_length),
            "sps": float(sps),
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
            # === TOPLINE METRICS (Critical for training monitoring) ===
            reward_per_step = float(eval_reward / max(eval_length, 1.0))

            topline_metrics = {
                "steps": num_steps,
                "topline/episode_reward": eval_reward,  # Total episode reward
                "topline/reward_per_step": reward_per_step,  # Reward per step (NEW!)
                "topline/success_rate": success,
                "topline/forward_velocity": forward_vel,
                "topline/height": height,
                "topline/distance_walked": distance,
                "topline/sps": sps,
            }

            # Add key contact metrics if available
            contact_left_force = metrics.get("eval/episode_contact/left_foot_force", 0.0) / max(eval_length, 1.0)
            contact_right_force = metrics.get("eval/episode_contact/right_foot_force", 0.0) / max(eval_length, 1.0)
            avg_air_time = metrics.get("eval/episode_contact/avg_air_time", 0.0) / max(eval_length, 1.0)
            air_time_threshold_met = metrics.get("eval/episode_contact/both_meet_air_time_threshold", 0.0) / max(eval_length, 1.0)

            topline_metrics.update({
                "topline/contact_left_force": contact_left_force,
                "topline/contact_right_force": contact_right_force,
                "topline/avg_air_time": avg_air_time,
                "topline/air_time_threshold_met": air_time_threshold_met,
            })

            # Add key reward components
            reward_foot_contact = metrics.get("eval/episode_reward/foot_contact", 0.0) / max(eval_length, 1.0)
            reward_foot_sliding = metrics.get("eval/episode_reward/foot_sliding", 0.0) / max(eval_length, 1.0)
            reward_foot_air_time = metrics.get("eval/episode_reward/foot_air_time", 0.0) / max(eval_length, 1.0)
            reward_z_velocity = metrics.get("eval/episode_reward/z_velocity", 0.0) / max(eval_length, 1.0)
            reward_tracking_xy = metrics.get("eval/episode_reward/tracking_exp_xy", 0.0) / max(eval_length, 1.0)
            reward_tracking_lin_xy = metrics.get("eval/episode_reward/tracking_lin_xy", 0.0) / max(eval_length, 1.0)
            reward_joint_velocity = metrics.get("eval/episode_reward/joint_velocity", 0.0) / max(eval_length, 1.0)
            reward_joint_acceleration = metrics.get("eval/episode_reward/joint_acceleration", 0.0) / max(eval_length, 1.0)
            reward_mechanical_power = metrics.get("eval/episode_reward/mechanical_power", 0.0) / max(eval_length, 1.0)

            topline_metrics.update({
                "topline/reward_foot_contact": reward_foot_contact,
                "topline/reward_foot_sliding": reward_foot_sliding,
                "topline/reward_foot_air_time": reward_foot_air_time,
                "topline/reward_z_velocity": reward_z_velocity,
                "topline/reward_tracking_exp_xy": reward_tracking_xy,
                "topline/reward_tracking_lin_xy": reward_tracking_lin_xy,
                "topline/reward_joint_velocity": reward_joint_velocity,
                "topline/reward_joint_acceleration": reward_joint_acceleration,
                "topline/reward_mechanical_power": reward_mechanical_power,
            })

            wandb.log(topline_metrics)

            # === DEBUG METRICS (Detailed breakdown for debugging) ===
            # Log debug metrics every eval for comprehensive tracking
            debug_metrics = {}

            # All individual reward components (except already in topline)
            for key, value in metrics.items():
                if key.startswith("eval/episode_reward/"):
                    component_name = key.replace("eval/episode_reward/", "")
                    if component_name not in ["foot_contact", "foot_sliding", "foot_air_time", "z_velocity", "tracking_exp_xy"]:
                        debug_metrics[f"debug/reward/{component_name}"] = value / max(eval_length, 1.0)

            # All individual contact metrics (except already in topline)
            for key, value in metrics.items():
                if key.startswith("eval/episode_contact/"):
                    component_name = key.replace("eval/episode_contact/", "")
                    if component_name not in ["left_foot_force", "right_foot_force", "avg_air_time", "both_meet_air_time_threshold"]:
                        debug_metrics[f"debug/contact/{component_name}"] = value / max(eval_length, 1.0)

            # Other eval metrics
            for key, value in metrics.items():
                if key.startswith("eval/") and not key.startswith("eval/episode_reward/") and not key.startswith("eval/episode_contact/"):
                    if key not in ["eval/episode_reward", "eval/avg_episode_length", "eval/episode_forward_velocity",
                                   "eval/episode_height", "eval/episode_success", "eval/episode_distance_walked"]:
                        debug_name = key.replace("eval/episode_", "").replace("eval/", "")
                        debug_metrics[f"debug/{debug_name}"] = value / max(eval_length, 1.0) if "episode" in key else value

            if debug_metrics:
                wandb.log(debug_metrics)

    # Train
    print("Starting training...\n")
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
        max_grad_norm=ppo_config.max_grad_norm,  # Fixed: was max_gradient_norm
        clipping_epsilon=ppo_config.clipping_epsilon,
    )

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}\n")

    # Close metrics log file
    metrics_log_file.close()
    print(f"Metrics saved to: {metrics_log_path}\n")

    # Save final checkpoint
    if not _QUICK_VERIFY.value:
        final_ckpt_path = ckpt_dir / "final_policy.pkl"
        with open(final_ckpt_path, "wb") as f:
            import pickle
            pickle.dump(params, f)
        print(f"Saved final checkpoint: {final_ckpt_path}\n")

    # Render videos if enabled
    # In quick_verify mode, check the quick_verify.render_videos setting
    should_render_videos = cfg["rendering"]["render_videos"]
    if _QUICK_VERIFY.value:
        # In quick_verify mode, use the quick_verify.render_videos setting instead
        should_render_videos = cfg.get("quick_verify", {}).get("render_videos", False)

    if should_render_videos:
        print("Rendering evaluation videos...")
        print(f"  Output directory: {exp_dir}\n")

        # Create inference function
        inference_fn = make_inference_fn(params)

        # Create VideoRenderer
        renderer = VideoRenderer(
            env=env,
            output_dir=exp_dir,
            width=cfg["rendering"]["render_width"],
            height=cfg["rendering"]["render_height"],
            fps=int(1.0 / env_config.ctrl_dt),
            camera="track",
            max_steps=min(cfg["training"]["episode_length"], 600),
        )

        # Render videos
        rng = jax.random.PRNGKey(cfg["training"]["seed"])
        results = renderer.render_multiple_videos(
            inference_fn=inference_fn,
            rng=rng,
            num_videos=3,
            video_prefix="eval_video",
        )

        # Upload to W&B if enabled
        if cfg["logging"]["use_wandb"]:
            for i, (video_path, stats) in enumerate(results):
                wandb.log({f"eval_video_{i}": wandb.Video(str(video_path), fps=renderer.fps)})
            print(f"  ✓ Videos uploaded to W&B\n")

        print(f"Rendering complete!\n")

    if cfg["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
