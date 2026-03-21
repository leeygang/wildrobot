#!/usr/bin/env python3
"""Fixed evaluation ladder for v0.17.0 standing-reset branch.

This script runs a trained policy through the standardized evaluation suites
defined in standing_training.md:
- eval_clean: no pushes
- eval_easy: 5N x 10 steps
- eval_medium: 8N x 10 steps  
- eval_hard: 10N x 10 steps
- eval_hard_long: 9N x 15 steps

Usage:
  uv run python training/eval/eval_ladder_v0170.py --checkpoint <path> --config <yaml>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple

import jax
import jax.numpy as jnp

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.core.checkpoint import load_checkpoint
from training.algos.ppo.ppo_core import create_networks

# Reuse eval helpers
from training.eval.eval_policy import _collect_eval_rollout, _compute_eval_metrics


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


def _run_eval_suite(
    checkpoint_path: Path,
    config_path: Path,
    eval_spec: EvalSpec,
    num_envs: int,
    num_steps: int,
    seed: int,
) -> Dict[str, float]:
    """Run a single evaluation suite and return metrics."""
    
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
    parser = argparse.ArgumentParser(description="Fixed evaluation ladder for v0.17.0")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to checkpoint (.pkl)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config (.yaml)")
    parser.add_argument("--num-envs", type=int, default=128,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=500,
                        help="Episode length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=Path,
                        help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
        
    if not args.config.exists():
        print(f"Error: Config not found: {args.config}")
        return 1
    
    print(f"Running v0.17.0 evaluation ladder:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    print(f"  Envs: {args.num_envs}, Steps: {args.num_steps}")
    print()
    
    results = {}
    
    # Run each evaluation suite
    for eval_spec in EVAL_LADDER_V0170:
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
            print(f"  ERROR: {e}")
            results[eval_spec.name] = {"error": str(e)}
            print()
    
    # Print summary table
    print("=" * 60)
    print("EVALUATION LADDER SUMMARY")
    print("=" * 60)
    print(f"{'Suite':<15} {'Success':<8} {'Ep Len':<8} {'Height Fail':<10}")
    print("-" * 60)
    
    for eval_spec in EVAL_LADDER_V0170:
        metrics = results.get(eval_spec.name, {})
        if "error" in metrics:
            print(f"{eval_spec.name:<15} {'ERROR':<8}")
        else:
            success = metrics.get("success_rate", 0.0)
            ep_len = metrics.get("episode_length", 0.0)
            height_fail = metrics.get("term_height_low_frac", 0.0)
            print(f"{eval_spec.name:<15} {success:>6.1%} {ep_len:>6.1f} {height_fail:>8.1%}")
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())