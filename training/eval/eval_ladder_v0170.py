#!/usr/bin/env python3
"""Comprehensive evaluation ladder for trained policies (post-training evaluation).

This script provides detailed benchmarking across standardized difficulty levels
that would be too expensive to run during training (requires multiple JIT compilations
with different push configurations).

Training-time eval (training/core/training_loop.py) runs simplified push/clean passes
for efficiency. Use this script for comprehensive post-training benchmarking.

Evaluation suites (from standing_training.md):
- eval_clean: no pushes
- eval_easy: 5N x 10 steps
- eval_medium: 8N x 10 steps  
- eval_hard: 10N x 10 steps
- eval_hard_long: 9N x 15 steps

v0.17.1 targets:
- eval_easy > 95%
- eval_medium > 75%
- eval_hard > 60%

Usage:
  # Run all suites on GPU (RECOMMENDED - fast)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint <path> --config <yaml> --platform gpu
  
  # Run critical suites only (faster)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint <path> --config <yaml> --suite eval_medium,eval_hard
  
  # Smaller batch if GPU memory limited (still fast on GPU)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint <path> --config <yaml> --num-envs 64
  
  # CPU mode (DEBUG/EMERGENCY ONLY - extremely slow, 30-60min)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint <path> --config <yaml> --platform cpu --suite eval_hard --num-envs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

# Project root (set before heavy JAX imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EvalSpec(NamedTuple):
    """Specification for a single evaluation suite."""
    name: str
    force_n: float
    duration_steps: int
    push_enabled: bool


# Fixed evaluation ladder for v0.17.0
EVAL_LADDER_V0170: List[EvalSpec] = [
    EvalSpec(name="eval_clean", force_n=0.0, duration_steps=0, push_enabled=False),
    EvalSpec(name="eval_easy", force_n=5.0, duration_steps=10, push_enabled=True),
    EvalSpec(name="eval_medium", force_n=8.0, duration_steps=10, push_enabled=True),
    EvalSpec(name="eval_hard", force_n=10.0, duration_steps=10, push_enabled=True),
    EvalSpec(name="eval_hard_long", force_n=9.0, duration_steps=15, push_enabled=True),
]

# v0.17.1 target thresholds
V0171_TARGETS = {
    "eval_easy": 0.95,
    "eval_medium": 0.75,
    "eval_hard": 0.60,
}


def _set_jax_platform(platform: str) -> None:
    """Set JAX platform before importing JAX.
    
    Args:
        platform: 'cpu', 'gpu', or 'auto'
        
    Note:
        CPU mode is EXTREMELY SLOW (30-60 minutes for minimal evaluation) and should
        only be used for emergency debugging. For normal checkpoint validation, use GPU
        or reduce batch size with --num-envs 64 while staying on GPU.
        
        When platform='cpu', you may see a harmless JAX plugin error like:
        "Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda13.initialize()"
        This is expected and safe - JAX is discovering that CUDA is unavailable and falling back to CPU.
    """
    # Clear any stale backend overrides first. A bad inherited value like
    # JAX_PLATFORMS=rocm can poison the eval subprocess even when the CLI asks
    # for GPU or auto.
    for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
        os.environ.pop(key, None)
    # Undo CPU isolation if a previous invocation left it in the environment.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    if platform == "cpu":
        # Force CPU and hide GPUs completely
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Disable CUDA plugin discovery to avoid initialization errors
        os.environ["JAX_PLUGINS"] = ""
        # Legacy compatibility
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        print("⚠️  WARNING: CPU mode is EXTREMELY SLOW (30-60min for minimal eval)")
        print("    Recommended: Use GPU with smaller batch (--num-envs 64) instead")
        print("    JAX platform forced to CPU (GPU discovery disabled)")
        print("    Note: JAX plugin errors during initialization are expected and harmless")
    elif platform == "gpu":
        # This JAX build exposes the GPU backend as 'cuda', not 'gpu'.
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["JAX_PLATFORM_NAME"] = "cuda"
        print("JAX platform forced to CUDA GPU (recommended for fast evaluation)")
    else:
        print("JAX platform auto-detect enabled")


def _run_eval_suite(
    checkpoint_path: Path,
    config_path: Path,
    eval_spec: EvalSpec,
    num_envs: int,
    num_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run a single evaluation suite and return metrics.
    
    Imports JAX-heavy modules inside this function so platform can be set first.
    """
    # Import JAX modules here (after platform is set)
    import jax
    import jax.numpy as jnp
    
    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv
    from training.core.checkpoint import load_checkpoint
    from training.algos.ppo.ppo_core import create_networks
    from training.eval.eval_policy import _collect_eval_rollout, _compute_eval_metrics
    
    # Fresh config for each suite
    training_cfg = load_training_config(config_path)
    
    # Override push settings for this suite
    training_cfg.env.push_enabled = eval_spec.push_enabled
    if eval_spec.push_enabled:
        training_cfg.env.push_force_min = float(eval_spec.force_n)
        training_cfg.env.push_force_max = float(eval_spec.force_n)
        training_cfg.env.push_duration_steps = int(eval_spec.duration_steps)
    
    # Override eval sizing
    training_cfg.ppo.num_envs = int(num_envs)
    training_cfg.ppo.rollout_steps = int(num_steps)
    
    # Load robot config (needed by env) - follow the pattern from eval_policy.py and force_sweep.py
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    if not robot_cfg_path.exists():
        raise FileNotFoundError(
            f"Robot config not found: {robot_cfg_path} "
            "(check env.assets_root or env.robot_config_path in the eval config)"
        )
    load_robot_config(robot_cfg_path)
    
    # Freeze config and create environment
    training_cfg.freeze()
    env = WildRobotEnv(config=training_cfg)
    
    # Create batched initial state - follow force_sweep.py pattern
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    batched_reset = jax.vmap(env.reset)
    env_state = batched_reset(reset_rngs)
    
    # Extract dimensions from actual state
    obs_dim = int(env_state.obs.shape[-1])
    action_dim = int(env.action_size)
    
    # Create networks with correct API - follow force_sweep.py pattern
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
    )
    
    # Load checkpoint as dict, not object - follow force_sweep.py pattern
    checkpoint = load_checkpoint(str(checkpoint_path))
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ())
    
    # Run evaluation rollout with correct API
    traj, _ = _collect_eval_rollout(
        env=env,
        env_state=env_state,
        policy_params=policy_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=num_steps,
        deterministic=True,
    )
    
    # Compute metrics with correct API 
    metrics = _compute_eval_metrics(traj, num_steps)
    
    # Cleanup to free device memory
    del traj, env_state, env, ppo_network
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fixed evaluation ladder for v0.17.0 standing policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Run critical suites on GPU after training finishes
  uv run python training/eval/eval_ladder_v0170.py --checkpoint checkpoints/policy.pkl --config configs/ppo_standing_v0171.yaml --platform gpu --suite eval_medium,eval_hard --num-envs 64
  
  # Full evaluation on GPU (all suites, takes ~10-15min)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint checkpoints/policy.pkl --config configs/ppo_standing_v0171.yaml --platform gpu
  
  # If GPU memory limited (still fast on GPU)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint checkpoints/policy.pkl --config configs/ppo_standing_v0171.yaml --num-envs 32
  
  # CPU mode (EMERGENCY/DEBUG ONLY - extremely slow, ~30-60min)
  uv run python training/eval/eval_ladder_v0170.py --checkpoint checkpoints/policy.pkl --config configs/ppo_standing_v0171.yaml --platform cpu --suite eval_hard --num-envs 1
        """
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to checkpoint (.pkl)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config (.yaml)")
    parser.add_argument("--platform", type=str, default="auto", choices=["cpu", "gpu", "auto"],
                        help="JAX platform: 'gpu' (recommended), 'auto' (let JAX decide), 'cpu' (debug/emergency only, VERY slow)")
    parser.add_argument("--suite", type=str,
                        help="Comma-separated list of suites to run (e.g., 'eval_medium,eval_hard'). Default: all")
    parser.add_argument("--num-envs", type=int, default=128,
                        help="Number of parallel environments (default: 128)")
    parser.add_argument("--num-steps", type=int, default=500,
                        help="Episode length (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path,
                        help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Set JAX platform BEFORE any JAX imports
    _set_jax_platform(args.platform)
    
    # Validate inputs
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nTip: Check the path and ensure the checkpoint file exists.")
        return 1
        
    if not args.config.exists():
        print(f"Error: Config not found: {args.config}")
        print("\nTip: Check the path and ensure the config file exists.")
        return 1
    
    # Parse suite filter
    if args.suite:
        requested_suites = set(s.strip() for s in args.suite.split(","))
        suites_to_run = [s for s in EVAL_LADDER_V0170 if s.name in requested_suites]
        if not suites_to_run:
            print(f"Error: No valid suites found in: {args.suite}")
            print(f"Valid suites: {', '.join(s.name for s in EVAL_LADDER_V0170)}")
            return 1
    else:
        suites_to_run = EVAL_LADDER_V0170
    
    print(f"Running v0.17.0 evaluation ladder:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    print(f"  Platform: {args.platform}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print(f"  Suites: {', '.join(s.name for s in suites_to_run)}")
    print()
    
    results = {}
    failed_suites = []
    
    # Run each evaluation suite
    for eval_spec in suites_to_run:
        print(f"Running {eval_spec.name}...")
        if eval_spec.push_enabled:
            print(f"  Push: {eval_spec.force_n}N x {eval_spec.duration_steps} steps")
        else:
            print("  No pushes")
            
        try:
            metrics = _run_eval_suite(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                eval_spec=eval_spec,
                num_envs=args.num_envs,
                num_steps=args.num_steps,
                seed=args.seed,
            )
            
            results[eval_spec.name] = metrics
            
            # Print key results
            success_rate = metrics.get("success_rate", 0.0)
            episode_length = metrics.get("episode_length", 0.0)
            term_height_low = metrics.get("term_height_low_frac", 0.0)
            
            print(f"  Success: {success_rate:.1%}")
            print(f"  Ep Length: {episode_length:.1f}")
            print(f"  Height Fail: {term_height_low:.1%}")
            print()
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg}")
            results[eval_spec.name] = {"error": error_msg}
            failed_suites.append((eval_spec.name, error_msg))
            print()
    
    # Print summary table
    print("=" * 60)
    print("EVALUATION LADDER SUMMARY")
    print("=" * 60)
    print(f"{'Suite':<15} {'Success':<8} {'Ep Len':<8} {'Height Fail':<10}")
    print("-" * 60)
    
    for eval_spec in suites_to_run:
        metrics = results.get(eval_spec.name, {})
        if "error" in metrics:
            print(f"{eval_spec.name:<15} {'ERROR':<8}")
        else:
            success = metrics.get("success_rate", 0.0)
            ep_len = metrics.get("episode_length", 0.0)
            height_fail = metrics.get("term_height_low_frac", 0.0)
            print(f"{eval_spec.name:<15} {success:>6.1%} {ep_len:>6.1f} {height_fail:>8.1%}")
    
    # Check v0.17.1 targets
    print()
    print("=" * 60)
    print("V0.17.1 TARGET ASSESSMENT")
    print("=" * 60)
    
    targets_met = []
    targets_missed = []
    
    for suite_name, threshold in V0171_TARGETS.items():
        metrics = results.get(suite_name, {})
        if "error" not in metrics:
            success_rate = metrics.get("success_rate", 0.0)
            status = "✓" if success_rate >= threshold else "✗"
            targets_met.append(suite_name) if success_rate >= threshold else targets_missed.append(suite_name)
            print(f"{status} {suite_name}: {success_rate:.1%} (target: {threshold:.1%})")
        else:
            print(f"✗ {suite_name}: ERROR (target: {threshold:.1%})")
            targets_missed.append(suite_name)
    
    print()
    if not targets_missed:
        print("✓ ALL v0.17.1 targets met!")
    else:
        print(f"✗ Targets missed: {', '.join(targets_missed)}")
        if targets_met:
            print(f"✓ Targets met: {', '.join(targets_met)}")
    
    # Print error summary if any failures
    if failed_suites:
        print()
        print("=" * 60)
        print("ERROR SUMMARY")
        print("=" * 60)
        for suite_name, error_msg in failed_suites:
            print(f"{suite_name}: {error_msg}")
        
        print()
        print("TROUBLESHOOTING:")
        if "cuSolver" in " ".join(e for _, e in failed_suites) or "GPU" in " ".join(e for _, e in failed_suites):
            print("  GPU initialization or computation failed.")
            print()
            print("  RECOMMENDED SOLUTIONS (in order of preference):")
            print("  1. If training is running: Stop or pause the training job, then rerun eval")
            print("  2. Use another machine with idle GPU")
            print("  3. Reduce batch size while staying on GPU (still fast):")
            print(f"     uv run python training/eval/eval_ladder_v0170.py --checkpoint {args.checkpoint} --config {args.config} --platform gpu --num-envs 64")
            print()
            print("  LAST RESORT (extremely slow, 30-60min):")
            print("  4. CPU mode for emergency debugging only:")
            print(f"     uv run python training/eval/eval_ladder_v0170.py --checkpoint {args.checkpoint} --config {args.config} --platform cpu --suite eval_hard --num-envs 1")
        elif args.num_envs >= 128:
            print("  Large batch size may cause memory issues. Try reducing --num-envs:")
            print(f"    uv run python training/eval/eval_ladder_v0170.py --checkpoint {args.checkpoint} --config {args.config} --num-envs 64")
        print()
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return 0 if not failed_suites else 1


if __name__ == "__main__":
    sys.exit(main())
