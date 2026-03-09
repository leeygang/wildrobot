#!/usr/bin/env python3
"""Force-sweep evaluation: find the bracing ceiling for a trained policy.

Runs a single checkpoint at multiple push force levels (FSM and base controller
disabled by default) and reports success_rate, episode_length, and termination
fractions at each level.  The output identifies the **bracing ceiling** — the
force at which bracing/crouch recovery can no longer survive.

Usage:
  uv run python training/eval/force_sweep.py \
      --checkpoint <path.pkl> \
      --forces 9,12,15,18,22,25 \
      --num-envs 128 --num-steps 500

See training/docs/step_trait_base_controller_design.md § "Bracing Ceiling
Calibration" for context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp

# Project root ----------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.core.checkpoint import load_checkpoint
from training.algos.ppo.ppo_core import create_networks

# Reuse the eval helpers from eval_policy.py
from training.eval.eval_policy import _collect_eval_rollout, _compute_eval_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUCCESS_THRESHOLD = 0.60  # bracing ceiling = force where success < this


def _estimate_bracing_ceiling(
    results: List[Dict],
    threshold: float = _SUCCESS_THRESHOLD,
) -> float | None:
    """Linear-interpolate the force where success_rate crosses *threshold*.

    Returns None if all levels are above or all are below the threshold.
    """
    for i in range(1, len(results)):
        s_prev = results[i - 1]["success_rate"]
        s_curr = results[i]["success_rate"]
        if s_prev >= threshold and s_curr < threshold:
            f_prev = results[i - 1]["force_n"]
            f_curr = results[i]["force_n"]
            # Linear interpolation
            t = (s_prev - threshold) / (s_prev - s_curr) if s_prev != s_curr else 0.5
            return f_prev + t * (f_curr - f_prev)
    return None


def _run_single_force_level(
    checkpoint_path: Path,
    config_path: Path,
    force_n: float,
    push_duration: int,
    num_envs: int,
    num_steps: int,
    seed: int,
    keep_fsm: bool,
) -> Dict[str, float]:
    """Create env at a specific force level, eval, return metrics dict."""

    # Fresh config for each level (push force is traced as a JIT constant)
    training_cfg = load_training_config(config_path)

    # Override push settings: fixed force for this level
    training_cfg.env.push_enabled = True
    training_cfg.env.push_force_min = float(force_n)
    training_cfg.env.push_force_max = float(force_n)
    training_cfg.env.push_duration_steps = int(push_duration)

    # Disable controllers unless --keep-fsm
    if not keep_fsm:
        training_cfg.env.fsm_enabled = False
        training_cfg.env.base_ctrl_enabled = False

    # Override eval sizing
    training_cfg.ppo.num_envs = int(num_envs)
    training_cfg.ppo.rollout_steps = int(num_steps)

    training_cfg.freeze()

    # Load robot config (needed by env)
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)

    # Create env, networks
    env = WildRobotEnv(config=training_cfg)

    rng = jax.random.PRNGKey(seed)
    reset_rngs = jax.random.split(rng, num_envs)
    batched_reset = jax.vmap(env.reset)
    env_state = batched_reset(reset_rngs)

    obs_dim = int(env_state.obs.shape[-1])
    action_dim = int(env.action_size)

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
    )

    checkpoint = load_checkpoint(str(checkpoint_path))
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ())

    # Run eval rollout
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

    metrics = _compute_eval_metrics(traj, num_steps)

    # Explicit cleanup to free device memory before next level
    del traj, env_state, env, ppo_network
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force-sweep eval: find the bracing ceiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pkl"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_standing_push.yaml",
        help="Base training config YAML",
    )
    parser.add_argument(
        "--forces",
        type=str,
        default="9,12,15,18,22,25",
        help="Comma-separated force levels in Newtons",
    )
    parser.add_argument(
        "--push-duration",
        type=int,
        default=15,
        help="Push duration in control steps",
    )
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel envs")
    parser.add_argument(
        "--num-steps", type=int, default=500, help="Eval rollout horizon"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Write JSON results to file"
    )
    parser.add_argument(
        "--keep-fsm",
        action="store_true",
        help="Keep FSM/base-ctrl enabled (for comparing FSM vs pure policy)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    forces = [float(f.strip()) for f in args.forces.split(",")]
    if not forces:
        raise ValueError("--forces must contain at least one value")

    fsm_label = "enabled (--keep-fsm)" if args.keep_fsm else "disabled"

    print()
    print("Force Sweep Evaluation")
    print("=" * 72)
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"  Config:        {config_path}")
    print(f"  Forces (N):    {forces}")
    print(f"  Push duration: {args.push_duration} steps")
    print(f"  Num envs:      {args.num_envs}")
    print(f"  Eval steps:    {args.num_steps}")
    print(f"  FSM:           {fsm_label}")
    print("=" * 72)
    print()

    results: List[Dict] = []

    for i, force in enumerate(forces):
        label = f"[{i + 1}/{len(forces)}] {force:5.1f}N"
        print(f"{label}: evaluating ...", end="", flush=True)
        t0 = time.time()

        metrics = _run_single_force_level(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            force_n=force,
            push_duration=args.push_duration,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            seed=args.seed,
            keep_fsm=args.keep_fsm,
        )

        elapsed = time.time() - t0
        sr = metrics["success_rate"]
        el = metrics["episode_length"]
        print(f" success={sr:.1%}  ep_len={el:.0f}  ({elapsed:.0f}s)")

        results.append(
            {
                "force_n": force,
                "success_rate": float(metrics["success_rate"]),
                "episode_length": float(metrics["episode_length"]),
                "term_height_low_frac": float(metrics["term_height_low_frac"]),
                "term_pitch_frac": float(metrics["term_pitch_frac"]),
                "term_roll_frac": float(metrics["term_roll_frac"]),
            }
        )

    # --- Summary table -------------------------------------------------------
    print()
    print("=" * 72)
    print(
        f" {'Force(N)':>8s} | {'Success%':>9s} | {'Ep Len':>7s} | "
        f"{'H_Low%':>7s} | {'Pitch%':>7s} | {'Roll%':>7s}"
    )
    print("-" * 72)
    for r in results:
        print(
            f" {r['force_n']:8.1f} | {r['success_rate']:8.1%} | "
            f"{r['episode_length']:7.1f} | {r['term_height_low_frac']:6.1%} | "
            f"{r['term_pitch_frac']:6.1%} | {r['term_roll_frac']:6.1%}"
        )
    print("=" * 72)

    ceiling = _estimate_bracing_ceiling(results)
    if ceiling is not None:
        print(
            f"Bracing ceiling estimate: ~{ceiling:.0f}N "
            f"(success drops below {_SUCCESS_THRESHOLD:.0%})"
        )
    elif all(r["success_rate"] >= _SUCCESS_THRESHOLD for r in results):
        print(
            f"Bracing ceiling: above {forces[-1]:.0f}N "
            f"(success stayed above {_SUCCESS_THRESHOLD:.0%} at all levels)"
        )
    else:
        print(
            f"Bracing ceiling: below {forces[0]:.0f}N "
            f"(success already below {_SUCCESS_THRESHOLD:.0%} at lowest level)"
        )

    # --- JSON output ----------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        payload = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "push_duration_steps": args.push_duration,
            "fsm_enabled": args.keep_fsm,
            "base_ctrl_enabled": args.keep_fsm,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "results": results,
            "bracing_ceiling_n": ceiling,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
