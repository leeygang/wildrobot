#!/usr/bin/env python3
"""Evaluate a trained policy on WildRobot environment (MJX, headless).

This runs a deterministic (default) or stochastic rollout and reports
aggregated metrics similar to training logs.

Usage:
  uv run python training/eval/eval_policy.py --checkpoint <path> --config <yaml>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp

# Add project root to path (eval/ -> training/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.core.checkpoint import load_checkpoint
from training.core.metrics_registry import (
    METRIC_INDEX,
    METRICS_VEC_KEY,
    aggregate_metrics,
)
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.core.rollout import TrajectoryBatch
from training.envs.env_info import WR_INFO_KEY


def _format_metric(value: float, fmt: str = ".3f") -> str:
    return format(float(value), fmt)


def _collect_eval_rollout(
    env: WildRobotEnv,
    env_state,
    policy_params,
    processor_params,
    ppo_network,
    rng: jax.Array,
    num_steps: int,
    deterministic: bool,
) -> tuple[TrajectoryBatch, object]:
    batch_step = jax.vmap(env.step)

    def step_fn(carry, _):
        state, rng = carry
        rng, action_rng = jax.random.split(rng)
        obs = state.obs
        actions, raw_actions, log_prob = sample_actions(
            processor_params,
            policy_params,
            ppo_network,
            obs,
            action_rng,
            deterministic=deterministic,
        )
        next_state = batch_step(state, actions)

        step_data = {
            "obs": obs,
            "action": raw_actions,
            "log_prob": log_prob,
            "value": jnp.zeros((obs.shape[0],), dtype=jnp.float32),
            "task_reward": next_state.reward,
            "done": next_state.done,
            "truncation": next_state.info[WR_INFO_KEY].truncated,
            "next_obs": next_state.obs,
            "metrics_vec": next_state.metrics[METRICS_VEC_KEY],
            "step_count": next_state.info[WR_INFO_KEY].step_count,
        }

        return (next_state, rng), step_data

    (final_state, _), rollout = jax.lax.scan(
        step_fn, (env_state, rng), None, length=num_steps
    )

    traj = TrajectoryBatch(
        obs=rollout["obs"],
        actions=rollout["action"],
        log_probs=rollout["log_prob"],
        values=rollout["value"],
        task_rewards=rollout["task_reward"],
        dones=rollout["done"],
        truncations=rollout["truncation"],
        next_obs=rollout["next_obs"],
        bootstrap_value=jnp.zeros((rollout["obs"].shape[1],), dtype=jnp.float32),
        metrics_vec=rollout["metrics_vec"],
        step_counts=rollout["step_count"],
        foot_contacts=None,
        root_heights=None,
        prev_joint_positions=None,
    )

    return traj, final_state


def _compute_eval_metrics(traj: TrajectoryBatch, num_steps: int) -> Dict[str, float]:
    agg_metrics = aggregate_metrics(traj.metrics_vec, traj.dones)

    episode_reward = jnp.mean(jnp.sum(traj.task_rewards, axis=0))
    task_reward_mean = jnp.mean(traj.task_rewards)
    reward_per_step = episode_reward / float(num_steps)

    ep_step_idx = METRIC_INDEX["episode_step_count"]
    episode_step_counts = traj.metrics_vec[..., ep_step_idx]
    completed_episode_lengths = episode_step_counts * traj.dones
    total_completed_length = jnp.sum(completed_episode_lengths)
    total_done = jnp.sum(traj.dones)
    episode_length = jnp.where(total_done > 0, total_completed_length / total_done, 0.0)

    total_truncated = jnp.sum(traj.truncations)
    success_rate = jnp.where(total_done > 0, total_truncated / total_done, 0.0)

    term_idx_height_low = METRIC_INDEX["term/height_low"]
    term_idx_height_high = METRIC_INDEX["term/height_high"]
    term_idx_pitch = METRIC_INDEX["term/pitch"]
    term_idx_roll = METRIC_INDEX["term/roll"]
    term_idx_truncated = METRIC_INDEX["term/truncated"]

    total_term_height_low = jnp.sum(traj.metrics_vec[..., term_idx_height_low])
    total_term_height_high = jnp.sum(traj.metrics_vec[..., term_idx_height_high])
    total_term_pitch = jnp.sum(traj.metrics_vec[..., term_idx_pitch])
    total_term_roll = jnp.sum(traj.metrics_vec[..., term_idx_roll])
    total_term_truncated = jnp.sum(traj.metrics_vec[..., term_idx_truncated])

    term_height_low_frac = jnp.where(total_done > 0, total_term_height_low / total_done, 0.0)
    term_height_high_frac = jnp.where(total_done > 0, total_term_height_high / total_done, 0.0)
    term_pitch_frac = jnp.where(total_done > 0, total_term_pitch / total_done, 0.0)
    term_roll_frac = jnp.where(total_done > 0, total_term_roll / total_done, 0.0)
    term_truncated_frac = jnp.where(total_done > 0, total_term_truncated / total_done, 0.0)

    agg_metrics.update(
        {
            "episode_reward": episode_reward,
            "task_reward_mean": task_reward_mean,
            "reward_per_step": reward_per_step,
            "episode_length": episode_length,
            "success_rate": success_rate,
            "term_height_low_frac": term_height_low_frac,
            "term_height_high_frac": term_height_high_frac,
            "term_pitch_frac": term_pitch_frac,
            "term_roll_frac": term_roll_frac,
            "term_truncated_frac": term_truncated_frac,
            "total_done": total_done,
        }
    )

    return {k: float(v) for k, v in agg_metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WildRobot policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_standing.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel envs")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Rollout steps (defaults to config.ppo.rollout_steps)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling instead of deterministic actions",
    )
    parser.add_argument("--output", type=str, default=None, help="Write metrics JSON to file")
    parser.add_argument(
        "--compare-metrics",
        type=str,
        default=None,
        help="Path to wandb metrics.jsonl for comparison (uses last logged line).",
    )
    parser.add_argument(
        "--compare-iteration",
        type=int,
        default=None,
        help="Iteration index to match in metrics.jsonl (defaults to checkpoint iteration).",
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

    training_cfg = load_training_config(config_path)
    compare_rollout_steps = int(training_cfg.ppo.rollout_steps)

    # Load robot config (required for env init)
    robot_cfg_path = Path("assets/robot_config.yaml")
    load_robot_config(robot_cfg_path)

    training_cfg.ppo.num_envs = int(args.num_envs)
    if args.num_steps is not None:
        training_cfg.ppo.rollout_steps = int(args.num_steps)
    training_cfg.freeze()

    env = WildRobotEnv(config=training_cfg)

    # Build policy network and load checkpoint params
    rng = jax.random.PRNGKey(args.seed)
    reset_rngs = jax.random.split(rng, training_cfg.ppo.num_envs)
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

    deterministic = not args.stochastic
    traj, _ = _collect_eval_rollout(
        env=env,
        env_state=env_state,
        policy_params=policy_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=training_cfg.ppo.rollout_steps,
        deterministic=deterministic,
    )

    metrics = _compute_eval_metrics(traj, training_cfg.ppo.rollout_steps)

    print("\nEvaluation Results")
    print("=" * 60)
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config:     {config_path}")
    print(f"  num_envs:   {training_cfg.ppo.num_envs}")
    print(f"  num_steps:  {training_cfg.ppo.rollout_steps}")
    print(f"  mode:       {'deterministic' if deterministic else 'stochastic'}")
    print("=" * 60)
    print(
        "  reward_per_step="
        f"{_format_metric(metrics['reward_per_step'], '.6f')} | "
        f"reward_sum={_format_metric(metrics['episode_reward'], '.2f')} | "
        f"ep_len={_format_metric(metrics['episode_length'], '.0f')} | "
        f"success={_format_metric(metrics['success_rate'], '.2%')}"
    )
    print(
        "  term: h_low="
        f"{_format_metric(metrics['term_height_low_frac'], '.1%')} | "
        f"h_high={_format_metric(metrics['term_height_high_frac'], '.1%')} | "
        f"pitch={_format_metric(metrics['term_pitch_frac'], '.1%')} | "
        f"roll={_format_metric(metrics['term_roll_frac'], '.1%')}"
    )

    if args.compare_metrics:
        compare_path = Path(args.compare_metrics)
        if not compare_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {compare_path}")
        compare_iter = args.compare_iteration
        if compare_iter is None:
            compare_iter = _parse_checkpoint_iteration(checkpoint_path)
        _print_comparison(
            metrics,
            compare_path,
            compare_iter,
            compare_rollout_steps,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"\nWrote metrics to {output_path}")

    return 0


def _parse_checkpoint_iteration(checkpoint_path: Path) -> int | None:
    name = checkpoint_path.name
    if not name.startswith("checkpoint_"):
        return None
    parts = name.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _read_last_metrics_line(path: Path) -> Dict[str, float]:
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    if last is None:
        raise ValueError(f"No valid JSON lines in {path}")
    return {k: float(v) for k, v in last.items() if isinstance(v, (int, float))}


def _read_metrics_for_iteration(
    path: Path, iteration: int
) -> Dict[str, float] | None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("progress/iteration") == iteration:
                return {
                    k: float(v) for k, v in data.items() if isinstance(v, (int, float))
                }
    return None


def _print_comparison(
    eval_metrics: Dict[str, float],
    metrics_path: Path,
    iteration: int | None,
    rollout_steps: int,
) -> None:
    train_metrics = None
    if iteration is not None:
        train_metrics = _read_metrics_for_iteration(metrics_path, iteration)
    if train_metrics is None:
        train_metrics = _read_last_metrics_line(metrics_path)
        iteration = None
    mapping = {
        "reward_sum": "env/episode_reward",
        "episode_length": "env/episode_length",
        "success_rate": "env/success_rate",
        "term_height_low_frac": "env/term_height_low_frac",
        "term_height_high_frac": "env/term_height_high_frac",
        "term_pitch_frac": "env/term_pitch_frac",
        "term_roll_frac": "env/term_roll_frac",
    }

    header = "last line"
    if iteration is not None:
        header = f"iteration {iteration}"
    print(f"\nComparison vs training metrics ({header})")
    if iteration is None:
        print(f"Using last line from {metrics_path}")
    else:
        print(f"Using progress/iteration={iteration} from {metrics_path}")
    print("=" * 60)
    train_reward = train_metrics.get("env/episode_reward")
    if train_reward is not None and rollout_steps > 0:
        train_per_step = train_reward / float(rollout_steps)
        eval_per_step = eval_metrics.get("reward_per_step", float("nan"))
        delta = eval_per_step - train_per_step
        print(
            f"  {'reward_per_step':20s} eval={_format_metric(eval_per_step, '.6f')} | "
            f"train={_format_metric(train_per_step, '.6f')} | "
            f"delta={_format_metric(delta, '.6f')}"
        )

    for eval_key, train_key in mapping.items():
        if train_key not in train_metrics:
            continue
        if eval_key == "reward_sum":
            eval_val = eval_metrics.get("episode_reward")
        else:
            eval_val = eval_metrics.get(eval_key)
        if eval_val is None:
            continue
        train_val = train_metrics[train_key]
        delta = eval_val - train_val
        print(
            f"  {eval_key:20s} eval={_format_metric(eval_val, '.4f')} | "
            f"train={_format_metric(train_val, '.4f')} | "
            f"delta={_format_metric(delta, '.4f')}"
        )

    if eval_metrics.get("total_done", 0.0) == 0.0:
        print(
            "  note: no terminations in eval window; episode_length/success_rate "
            "may be uninformative. Increase --num-steps for full episodes."
        )


if __name__ == "__main__":
    raise SystemExit(main())
