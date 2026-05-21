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
from training.exports.export_onnx import get_checkpoint_dims
from training.core.metrics_registry import (
    METRIC_INDEX,
    METRICS_VEC_KEY,
    aggregate_metrics,
)
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.core.rollout import TrajectoryBatch
from training.envs.env_info import WR_INFO_KEY


def _disable_cmd_resample_for_eval(env_cfg) -> bool:
    """Match the training eval sentinel: non-negative eval cmd pins rollout cmd."""
    return float(env_cfg.eval_velocity_cmd) >= 0.0


def _network_activation_name(training_cfg) -> str:
    """Resolve the shared PPO activation and enforce actor/critic parity."""
    actor_activation = str(training_cfg.networks.actor.activation).lower()
    critic_activation = str(training_cfg.networks.critic.activation).lower()
    if actor_activation != critic_activation:
        raise ValueError(
            f"actor.activation ({actor_activation!r}) and "
            f"critic.activation ({critic_activation!r}) must match."
        )
    return actor_activation


def _format_metric(value: float, fmt: str = ".3f") -> str:
    return format(float(value), fmt)


def _extract_policy_log_std(policy_params, action_dim: int) -> Dict[str, float]:
    """Pull policy log_std bias from the last layer of the actor network.

    Brax MLPs lay the last Dense layer out with bias shape ``(2 *
    action_dim,)`` for a parametric Gaussian: first ``action_dim`` are
    the mean biases, second ``action_dim`` are the log_std biases.  A
    policy whose mean is near zero but log_std grows during training
    reduces to "rely on exploration noise" — surfacing this exposes the
    failure mode where eval-time deterministic actions go to ~0 even
    though training metrics looked alive.
    """
    try:
        params = policy_params["params"]
        last_layer_key = max(
            (k for k in params if k.startswith("hidden_")),
            key=lambda k: int(k.split("_")[1]),
        )
        bias = params[last_layer_key]["bias"]
        if bias.shape[0] != 2 * action_dim:
            return {}
        log_std = bias[action_dim:]
        return {
            "policy/log_std_mean": float(jnp.mean(log_std)),
            "policy/log_std_min": float(jnp.min(log_std)),
            "policy/log_std_max": float(jnp.max(log_std)),
            "policy/std_mean": float(jnp.mean(jnp.exp(log_std))),
        }
    except (KeyError, ValueError, IndexError, TypeError):
        return {}


def _collect_eval_rollout(
    env: WildRobotEnv,
    env_state,
    policy_params,
    processor_params,
    ppo_network,
    rng: jax.Array,
    num_steps: int,
    deterministic: bool,
    disable_cmd_resample: bool,
    disable_pushes: bool,
) -> tuple[TrajectoryBatch, object]:
    batch_step = jax.vmap(
        lambda state, action: env.step(
            state,
            action,
            disable_cmd_resample=disable_cmd_resample,
            disable_pushes=disable_pushes,
        )
    )

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
        critic_obs=rollout["obs"],
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
    )

    return traj, final_state


def _compute_eval_metrics(traj: TrajectoryBatch, num_steps: int) -> Dict[str, float]:
    agg_metrics = aggregate_metrics(traj.metrics_vec, traj.dones)

    # rollout_reward_sum: per-env sum-over-rollout-window, mean across
    # envs.  This is NOT the per-episode return; it equals it only when
    # the rollout horizon equals the episode horizon.  Matches the
    # legacy ``episode_reward`` value emitted by the training loop, so
    # the comparison print stays bit-identical against legacy wandb logs.
    rollout_reward_sum = jnp.mean(jnp.sum(traj.task_rewards, axis=0))
    task_reward_mean = jnp.mean(traj.task_rewards)
    reward_per_step = rollout_reward_sum / float(num_steps)

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

    per_step_height_low = traj.metrics_vec[..., term_idx_height_low]
    per_step_height_high = traj.metrics_vec[..., term_idx_height_high]
    per_step_pitch = traj.metrics_vec[..., term_idx_pitch]
    per_step_roll = traj.metrics_vec[..., term_idx_roll]
    per_step_truncated = traj.metrics_vec[..., term_idx_truncated]

    # height_low / height_high / truncated are real termination causes
    # under both relaxed and strict termination, so sum-per-step over a
    # done episode = "did this cause terminate".  Keep classic naming.
    term_height_low_frac = jnp.where(
        total_done > 0, jnp.sum(per_step_height_low) / total_done, 0.0
    )
    term_height_high_frac = jnp.where(
        total_done > 0, jnp.sum(per_step_height_high) / total_done, 0.0
    )
    term_truncated_frac = jnp.where(
        total_done > 0, jnp.sum(per_step_truncated) / total_done, 0.0
    )

    # term_pitch_frac / term_roll_frac: true termination-cause fraction
    # (∈ [0, 1]).  Mask the per-step pitch/roll occupancy flag by dones
    # so only the terminal step counts.  Mirrors training_loop.py post
    # the v0.20.1 rename — the eval value and ``env/term_pitch_frac``
    # in wandb now have identical semantics.  Zero under
    # use_relaxed_termination + pitch-only violations because terminal
    # steps only fire on height there.
    pitch_at_done = jnp.sum(per_step_pitch * traj.dones)
    roll_at_done = jnp.sum(per_step_roll * traj.dones)
    term_pitch_frac = jnp.where(total_done > 0, pitch_at_done / total_done, 0.0)
    term_roll_frac = jnp.where(total_done > 0, roll_at_done / total_done, 0.0)
    # soft_violation_*_frac: the legacy formula (per-step occupancy /
    # total_done).  Unbounded under relaxed termination.  Matches
    # training's ``env/soft_violation_pitch_frac`` / ``soft_violation_roll_frac``.
    soft_violation_pitch_frac = jnp.where(
        total_done > 0, jnp.sum(per_step_pitch) / total_done, 0.0
    )
    soft_violation_roll_frac = jnp.where(
        total_done > 0, jnp.sum(per_step_roll) / total_done, 0.0
    )

    # Action-magnitude diagnostics: ``debug/action_abs_mean`` near 0
    # with a wide policy std (see ``policy/log_std_mean`` printed at
    # startup) is the smoking gun for a policy whose mean head never
    # left init and whose training-time "achieved vx" was driven by
    # exploration noise.  ``traj.actions`` here are the raw pre-tanh
    # actions in policy space (Brax convention) — magnitudes ≥ ~0.2
    # indicate the policy mean has moved meaningfully off init.
    action_abs = jnp.abs(traj.actions)
    debug_action_abs_mean = jnp.mean(action_abs)
    debug_action_abs_max = jnp.max(action_abs)

    agg_metrics.update(
        {
            "rollout_reward_sum": rollout_reward_sum,
            "episode_reward": rollout_reward_sum,  # legacy alias
            "task_reward_mean": task_reward_mean,
            "reward_per_step": reward_per_step,
            "episode_length": episode_length,
            "success_rate": success_rate,
            "term_height_low_frac": term_height_low_frac,
            "term_height_high_frac": term_height_high_frac,
            "term_pitch_frac": term_pitch_frac,
            "term_roll_frac": term_roll_frac,
            "term_truncated_frac": term_truncated_frac,
            "soft_violation_pitch_frac": soft_violation_pitch_frac,
            "soft_violation_roll_frac": soft_violation_roll_frac,
            "debug/action_abs_mean": debug_action_abs_mean,
            "debug/action_abs_max": debug_action_abs_max,
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
        required=True,
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
    parser.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "Disable push perturbations during the eval rollout (passes "
            "disable_pushes=True to env.step).  Use this for clean A/B "
            "comparisons when the training config has push_enabled=true."
        ),
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

    # Load robot config (required for env init, variant-aware)
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    if not robot_cfg_path.exists():
        raise FileNotFoundError(
            f"Robot config not found: {robot_cfg_path} "
            "(check env.assets_root or env.robot_config_path in the eval config)"
        )
    load_robot_config(robot_cfg_path)

    training_cfg.ppo.num_envs = int(args.num_envs)
    if args.num_steps is not None:
        training_cfg.ppo.rollout_steps = int(args.num_steps)
    training_cfg.freeze()

    env = WildRobotEnv(config=training_cfg)

    # Build policy network and load checkpoint params
    rng = jax.random.PRNGKey(args.seed)
    reset_rngs = jax.random.split(rng, training_cfg.ppo.num_envs)
    batched_reset = jax.vmap(env.reset_for_eval)
    env_state = batched_reset(reset_rngs)

    obs_dim = int(env_state.obs.shape[-1])
    action_dim = int(env.action_size)
    activation = _network_activation_name(training_cfg)

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
        activation=activation,
    )

    checkpoint = load_checkpoint(str(checkpoint_path))
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ())
    checkpoint_obs_dim, checkpoint_action_dim = get_checkpoint_dims(checkpoint_path)
    if checkpoint_obs_dim != obs_dim or checkpoint_action_dim != action_dim:
        raise ValueError(
            "Checkpoint policy dimensions do not match the current eval env: "
            f"checkpoint obs/action=({checkpoint_obs_dim}, {checkpoint_action_dim}), "
            f"env obs/action=({obs_dim}, {action_dim}). Use a checkpoint trained "
            "against this robot/config contract, or evaluate with the matching "
            "historical code/assets."
        )

    deterministic = not args.stochastic
    disable_cmd_resample = _disable_cmd_resample_for_eval(training_cfg.env)
    disable_pushes = bool(args.no_push)
    traj, _ = _collect_eval_rollout(
        env=env,
        env_state=env_state,
        policy_params=policy_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=training_cfg.ppo.rollout_steps,
        deterministic=deterministic,
        disable_cmd_resample=disable_cmd_resample,
        disable_pushes=disable_pushes,
    )

    metrics = _compute_eval_metrics(traj, training_cfg.ppo.rollout_steps)
    metrics.update(_extract_policy_log_std(policy_params, action_dim))

    # Layout / action-mapping / residual-base affect what the policy
    # actually does at runtime; logging them at the top makes it obvious
    # when an eval was run against the wrong contract (e.g., v6 vs v7
    # obs, q_ref vs home residual base).
    layout_id = None
    mapping_id = None
    spec_json = checkpoint.get("policy_spec_json")
    if isinstance(spec_json, str):
        try:
            spec_dict = json.loads(spec_json)
            layout_id = spec_dict.get("observation", {}).get("layout_id")
            mapping_id = spec_dict.get("action", {}).get("mapping_id")
        except json.JSONDecodeError:
            pass

    # Stale-default-config heuristic.  If the user took the standing
    # default but the checkpoint declares a walking obs layout, the eval
    # will still execute (env init uses the loaded config), but the
    # checkpoint-dim check above only catches obs/action shape mismatch
    # — it cannot catch reward-recipe / loc_ref / DR drift between the
    # standing default and the walking config the policy actually
    # trained against.  Surface this as a warning so the user knows to
    # pass ``--config`` pointing at the run's saved training_config.yaml.
    walking_layouts = {
        "wr_obs_v4",
        "wr_obs_v6_offline_ref_history",
        "wr_obs_v7_phase_proprio",
    }
    cfg_version = getattr(training_cfg, "version_name", "") or ""
    cfg_looks_walking = (
        "walking" in cfg_version.lower()
        or "v0.20" in cfg_version.lower()
        or "loco" in cfg_version.lower()
    )
    if layout_id in walking_layouts and not cfg_looks_walking:
        print(
            "WARNING: checkpoint declares walking layout "
            f"{layout_id!r} but the loaded config "
            f"(version_name={cfg_version!r}) does not look walking-related.\n"
            "  Pass --config pointing at the training_config.yaml saved "
            "alongside this checkpoint so the env's reward recipe, "
            "loc_ref settings, and DR profile match what the policy "
            "trained against.",
            file=sys.stderr,
        )

    print("\nEvaluation Results")
    print("=" * 60)
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  config:     {config_path}")
    print(f"  num_envs:   {training_cfg.ppo.num_envs}")
    print(f"  num_steps:  {training_cfg.ppo.rollout_steps}")
    print(f"  mode:       {'deterministic' if deterministic else 'stochastic'}")
    if layout_id is not None:
        print(f"  layout:     {layout_id}")
    if mapping_id is not None:
        print(f"  mapping:    {mapping_id}")
    residual_base = getattr(training_cfg.env, "loc_ref_residual_base", None)
    if residual_base is not None:
        print(f"  residual_base: {residual_base}")
    if disable_cmd_resample:
        print(f"  eval_cmd:   pinned vx={training_cfg.env.eval_velocity_cmd:.3f}")
    cfg_push_enabled = bool(getattr(training_cfg.env, "push_enabled", False))
    if cfg_push_enabled:
        push_state = "disabled by --no-push" if disable_pushes else "ON (config)"
    else:
        push_state = "off (config)"
    print(f"  pushes:     {push_state}")
    print("=" * 60)
    print(
        "  reward_per_step="
        f"{_format_metric(metrics['reward_per_step'], '.6f')} | "
        f"reward_sum={_format_metric(metrics['rollout_reward_sum'], '.2f')} | "
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
    print(
        "  soft: pitch="
        f"{_format_metric(metrics['soft_violation_pitch_frac'], '.1%')} | "
        f"roll={_format_metric(metrics['soft_violation_roll_frac'], '.1%')}"
        "  (per-step occupancy / done; unbounded under relaxed termination)"
    )
    _print_policy_diagnostics(metrics)
    _print_walking_dashboard(metrics)

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


def _print_policy_diagnostics(metrics: Dict[str, float]) -> None:
    """Surface action magnitude + policy std so a policy whose mean
    head never left init is obvious without running a stochastic eval
    for comparison.  A deterministic ``debug/action_abs_mean`` of ~0.05
    paired with ``policy/log_std_mean`` near init (-0.69 ⇒ std=0.5) or
    higher means the trained policy is operating in the "exploration
    noise drives all motion" regime — training metrics that show
    forward_velocity > 0 are then driven by noise, not a learned skill.
    """
    aam = metrics.get("debug/action_abs_mean")
    aax = metrics.get("debug/action_abs_max")
    if aam is not None and aax is not None:
        print(
            f"  action: abs.mean={_format_metric(aam, '.4f')} "
            f"abs.max={_format_metric(aax, '.4f')}  "
            "(values < ~0.05 ⇒ policy mean near init)"
        )
    log_std_mean = metrics.get("policy/log_std_mean")
    std_mean = metrics.get("policy/std_mean")
    if log_std_mean is not None and std_mean is not None:
        print(
            f"  policy: log_std_mean={_format_metric(log_std_mean, '.3f')} "
            f"(std={_format_metric(std_mean, '.3f')})  "
            "(init log_std=-0.693 ⇒ std=0.5; growing log_std ⇒ more exploration)"
        )


def _print_walking_dashboard(metrics: Dict[str, float]) -> None:
    """G4 / G5 acceptance gates for v0.20.1 walking, computed inline
    from ``agg_metrics`` produced by ``aggregate_metrics``.  See
    ``training/docs/walking_training.md`` v0.20.1 § and the matching
    block in the analyzer (``skills/wildrobot-training-analyze``).
    """
    fwd_v = metrics.get("forward_velocity")
    cmd_v = metrics.get("velocity_command")
    cmd_err = metrics.get("tracking/cmd_vs_achieved_forward")
    cmd_ratio = metrics.get("tracking/forward_velocity_cmd_ratio")
    step_len = metrics.get("tracking/step_length_touchdown_event_m")
    res_kn_l = metrics.get("tracking/residual_knee_left_abs")
    res_kn_r = metrics.get("tracking/residual_knee_right_abs")
    res_hp_l = metrics.get("tracking/residual_hip_pitch_left_abs")
    res_hp_r = metrics.get("tracking/residual_hip_pitch_right_abs")
    contact = metrics.get("ref/contact_phase_match")
    bq_err = metrics.get("ref/body_quat_err_deg")

    # Only print the walking dashboard if the run actually exposed
    # walking-specific tracking metrics — keeps standing-eval output
    # uncluttered.
    if fwd_v is None and cmd_ratio is None and step_len is None:
        return

    def _gate(value: float | None, predicate, label: str) -> str:
        if value is None:
            return "n/a"
        return "PASS" if predicate(value) else f"FAIL ({label})"

    print("=" * 60)
    print("  v0.20.1 walking gates (G4 promotion + G5 anti-exploit):")
    if fwd_v is not None:
        print(
            f"    G4 forward_velocity ≥ 0.075       : "
            f"{_format_metric(fwd_v, '+.4f')} m/s  "
            f"{_gate(fwd_v, lambda v: v >= 0.075, '< 0.075')}"
        )
    if cmd_v is not None:
        print(f"    -- velocity_command (eval cmd)    : {_format_metric(cmd_v, '+.4f')} m/s")
    if cmd_err is not None:
        print(
            f"    G4 cmd_vs_achieved_forward ≤ 0.075: "
            f"{_format_metric(cmd_err, '.4f')} m/s  "
            f"{_gate(cmd_err, lambda v: v <= 0.075, '> 0.075')}"
        )
    if step_len is not None:
        print(
            f"    G4 step_length_touchdown ≥ 0.030  : "
            f"{_format_metric(step_len, '+.4f')} m  "
            f"{_gate(step_len, lambda v: v >= 0.030, '< 0.030')}"
        )
    if cmd_ratio is not None:
        print(
            f"    G5 forward_velocity_cmd_ratio ∈ [0.6, 1.5]: "
            f"{_format_metric(cmd_ratio, '+.3f')}  "
            f"{_gate(cmd_ratio, lambda v: 0.6 <= v <= 1.5, 'out of band')}"
        )
    for name, val in (
        ("residual_hip_pitch_left_abs", res_hp_l),
        ("residual_hip_pitch_right_abs", res_hp_r),
        ("residual_knee_left_abs", res_kn_l),
        ("residual_knee_right_abs", res_kn_r),
    ):
        if val is not None:
            status = _gate(val, lambda v: v <= 0.20, "> 0.20")
            print(f"    G5 {name:30s} ≤ 0.20: {_format_metric(val, '.3f')}  {status}")
    if contact is not None:
        print(
            f"    info ref/contact_phase_match > 0.5 : "
            f"{_format_metric(contact, '.3f')}  "
            f"{'PASS' if contact > 0.5 else 'low (gait drift)'}"
        )
    if bq_err is not None:
        print(
            f"    info ref/body_quat_err_deg < 10°   : "
            f"{_format_metric(bq_err, '.2f')}°  "
            f"{'PASS' if bq_err < 10.0 else 'over'}"
        )


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

    def _get_train_value(*keys: str) -> float | None:
        for k in keys:
            v = train_metrics.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    header = "last line"
    if iteration is not None:
        header = f"iteration {iteration}"
    print(f"\nComparison vs training metrics ({header})")
    if iteration is None:
        print(f"Using last line from {metrics_path}")
    else:
        print(f"Using progress/iteration={iteration} from {metrics_path}")
    print("=" * 60)
    # Training logs may report episode reward in different units depending on how
    # rollouts are aggregated. Prefer a direct "reward per step" metric if present.
    train_per_step = _get_train_value(
        "env/reward_per_step",
        "debug/task_reward_per_step",
    )
    if train_per_step is None:
        train_reward_sum = _get_train_value(
            "env/rollout_reward_sum", "env/episode_reward"
        )
        if train_reward_sum is not None and rollout_steps > 0:
            train_per_step = train_reward_sum / float(rollout_steps)

    eval_per_step = eval_metrics.get("reward_per_step")
    if eval_per_step is not None and train_per_step is not None:
        delta = float(eval_per_step) - float(train_per_step)
        print(
            f"  {'reward_per_step':20s} eval={_format_metric(eval_per_step, '.6f')} | "
            f"train={_format_metric(train_per_step, '.6f')} | "
            f"delta={_format_metric(delta, '.6f')}"
        )

    # Compare common scalars with flexible key fallback (wandb naming changed over time).
    # NOTE on term_pitch_frac / term_roll_frac: post the v0.20.1 rename,
    # eval and ``env/term_pitch_frac`` both mean "pitch fault AT done
    # step / total dones".  The per-step occupancy ratio (the legacy
    # formula) is now ``soft_violation_*_frac`` on both sides.
    comparisons: list[tuple[str, float | None, float | None]] = [
        (
            "reward_sum",
            eval_metrics.get("rollout_reward_sum"),
            _get_train_value("env/rollout_reward_sum", "env/episode_reward"),
        ),
        (
            "episode_length",
            eval_metrics.get("episode_length"),
            _get_train_value("env/episode_length"),
        ),
        (
            "success_rate",
            eval_metrics.get("success_rate"),
            _get_train_value("env/success_rate"),
        ),
        (
            "term_height_low_frac",
            eval_metrics.get("term_height_low_frac"),
            _get_train_value("env/term_height_low_frac", "term_height_low_frac"),
        ),
        (
            "term_height_high_frac",
            eval_metrics.get("term_height_high_frac"),
            _get_train_value("env/term_height_high_frac", "term_height_high_frac"),
        ),
        (
            "term_pitch_frac",
            eval_metrics.get("term_pitch_frac"),
            _get_train_value("env/term_pitch_frac", "term_pitch_frac"),
        ),
        (
            "term_roll_frac",
            eval_metrics.get("term_roll_frac"),
            _get_train_value("env/term_roll_frac", "term_roll_frac"),
        ),
        (
            "soft_violation_pitch_frac",
            eval_metrics.get("soft_violation_pitch_frac"),
            _get_train_value(
                "env/soft_violation_pitch_frac", "soft_violation_pitch_frac"
            ),
        ),
        (
            "soft_violation_roll_frac",
            eval_metrics.get("soft_violation_roll_frac"),
            _get_train_value(
                "env/soft_violation_roll_frac", "soft_violation_roll_frac"
            ),
        ),
        (
            "term_truncated_frac",
            eval_metrics.get("term_truncated_frac"),
            _get_train_value("env/term_truncated_frac", "term_truncated_frac"),
        ),
    ]

    for name, eval_val, train_val in comparisons:
        if eval_val is None or train_val is None:
            continue
        delta = float(eval_val) - float(train_val)
        print(
            f"  {name:20s} eval={_format_metric(eval_val, '.4f')} | "
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
