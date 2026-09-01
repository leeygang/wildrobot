#!/usr/bin/env python3
"""Diagnose left/right symmetry of a walking checkpoint.

The diagnostic separates three questions that are otherwise conflated by a
normal rollout:

1. Is the command-conditioned reference mirror-symmetric?
2. Is the actor equivariant under a sagittal reflection?
3. Do paired +vy/-vy closed-loop rollouts produce mirrored actions and torque
   occupancy?

The observation transform is intentionally limited to ``wr_obs_v8_cmd3d``.
That is the active 17-DoF walking contract and lets every transformed channel
be explicit rather than silently treating an unknown field as symmetric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from assets.robot_config import load_robot_config
from policy_contract.jax.symmetry import (
    joint_mirror_transform,
    mirror_actions,
    mirror_observations,
)
from policy_contract.spec import PolicySpec
from training.algos.ppo.ppo_core import create_networks
from training.configs.training_config import load_training_config
from training.core.checkpoint import load_checkpoint
from training.core.metrics_registry import TORQUE_ACTUATOR_NAMES
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.eval_policy import (
    _collect_eval_rollout,
    _compute_eval_metrics,
    _network_activation_name,
)
from training.exports.export_onnx import get_checkpoint_dims


def actor_mirror_metrics(
    *,
    observations: jax.Array,
    spec: PolicySpec,
    ppo_network: Any,
    processor_params: Any,
    policy_params: Any,
) -> dict[str, float]:
    """Compare pi(M(o)) against M(pi(o)) using deterministic actions."""
    logits = ppo_network.policy_network.apply(
        processor_params, policy_params, observations
    )
    actions = ppo_network.parametric_action_distribution.mode(logits)
    mirrored_obs = mirror_observations(observations, spec)
    mirrored_logits = ppo_network.policy_network.apply(
        processor_params, policy_params, mirrored_obs
    )
    actions_from_mirrored_obs = ppo_network.parametric_action_distribution.mode(
        mirrored_logits
    )
    expected_mirrored_actions = mirror_actions(actions, spec)
    error = actions_from_mirrored_obs - expected_mirrored_actions
    abs_error = jnp.abs(error)
    mean_abs_action = jnp.mean(jnp.abs(expected_mirrored_actions))

    metrics = {
        "action_mae": float(jnp.mean(abs_error)),
        "action_rmse": float(jnp.sqrt(jnp.mean(jnp.square(error)))),
        "action_max_abs_error": float(jnp.max(abs_error)),
        "action_mean_abs": float(mean_abs_action),
        "action_relative_mae": float(
            jnp.mean(abs_error) / jnp.maximum(mean_abs_action, jnp.float32(1e-6))
        ),
    }
    per_joint_mae = jnp.mean(abs_error, axis=tuple(range(abs_error.ndim - 1)))
    for name, value in zip(spec.robot.actuator_names, np.asarray(per_joint_mae)):
        metrics[f"joint_mae/{name}"] = float(value)
    return metrics


def _reference_symmetry_metrics(
    env: WildRobotEnv,
    positive_cmd: tuple[float, float, float],
    negative_cmd: tuple[float, float, float],
) -> dict[str, float | int]:
    """Compare mirrored q_ref trajectories after phase alignment."""
    keys = np.asarray(env._offline_cmd_keys, dtype=np.float32)
    arrays = env._offline_jax_arrays
    if keys.ndim != 2 or keys.shape[1] != 3 or np.asarray(arrays["q_ref"]).ndim != 3:
        raise ValueError("Reference symmetry requires a command-conditioned 3D library")

    pos_idx = int(np.argmin(np.linalg.norm(keys - np.asarray(positive_cmd), axis=1)))
    neg_idx = int(np.argmin(np.linalg.norm(keys - np.asarray(negative_cmd), axis=1)))
    if not np.allclose(keys[pos_idx], positive_cmd, atol=1e-6):
        raise ValueError(
            "Positive command bin not found: "
            f"requested={positive_cmd}, nearest={keys[pos_idx]}"
        )
    if not np.allclose(keys[neg_idx], negative_cmd, atol=1e-6):
        raise ValueError(
            "Negative command bin not found: "
            f"requested={negative_cmd}, nearest={keys[neg_idx]}"
        )

    phase = np.stack(
        [np.asarray(arrays["phase_sin"]), np.asarray(arrays["phase_cos"])],
        axis=-1,
    )
    pos_phase = phase[pos_idx]
    neg_phase = phase[neg_idx]
    phase_errors = np.asarray(
        [np.mean(np.square(np.roll(neg_phase, -shift, axis=0) + pos_phase))
         for shift in range(pos_phase.shape[0])],
        dtype=np.float64,
    )
    phase_shift = int(np.argmin(phase_errors))

    q_ref = np.asarray(arrays["q_ref"], dtype=np.float32)
    homes = np.asarray(env._home_q_rad, dtype=np.float32)
    pos_residual = q_ref[pos_idx] - homes
    neg_residual = np.roll(q_ref[neg_idx] - homes, -phase_shift, axis=0)
    expected_neg = np.asarray(mirror_actions(jnp.asarray(pos_residual), env._policy_spec))
    error = neg_residual - expected_neg
    abs_error = np.abs(error)
    metrics: dict[str, float | int] = {
        "positive_bin_index": pos_idx,
        "negative_bin_index": neg_idx,
        "phase_shift_steps": phase_shift,
        "phase_alignment_rmse": float(np.sqrt(phase_errors[phase_shift])),
        "q_ref_residual_mae_rad": float(np.mean(abs_error)),
        "q_ref_residual_max_abs_error_rad": float(np.max(abs_error)),
    }
    per_joint = np.mean(abs_error, axis=0)
    for name, value in zip(env._policy_spec.robot.actuator_names, per_joint):
        metrics[f"joint_mae_rad/{name}"] = float(value)
    return metrics


def _action_summary(
    *,
    observations: jax.Array,
    spec: PolicySpec,
    ppo_network: Any,
    processor_params: Any,
    policy_params: Any,
) -> tuple[dict[str, float], jax.Array]:
    logits = ppo_network.policy_network.apply(
        processor_params, policy_params, observations
    )
    actions = ppo_network.parametric_action_distribution.mode(logits)
    mean = jnp.mean(actions, axis=tuple(range(actions.ndim - 1)))
    abs_mean = jnp.mean(jnp.abs(actions), axis=tuple(range(actions.ndim - 1)))
    summary: dict[str, float] = {}
    for i, name in enumerate(spec.robot.actuator_names):
        summary[f"mean/{name}"] = float(mean[i])
        summary[f"abs_mean/{name}"] = float(abs_mean[i])
    return summary, mean


def _select_rollout_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keep = {
        "forward_velocity",
        "episode_length",
        "success_rate",
        "total_done",
        "walking_fall_env_count",
        "walking_fall_env_frac",
        "walking_stable_body_tilt_deg_mean",
        "walking_stable_body_tilt_deg_p95",
        "walking_stable_body_tilt_deg_max",
        "walking_survivor_final_body_tilt_deg_max",
        "walking_stable_max_actuator_torque_sat_frac",
    }
    selected = {key: value for key, value in metrics.items() if key in keep}
    aliases = {
        "tracking/lateral_velocity_abs": "lateral_velocity_abs",
        "tracking/lateral_velocity_signed_m_s": "lateral_velocity_signed_m_s",
        "tracking/ang_vel_z_signed_rad_s": "ang_vel_z_signed_rad_s",
        "tracking/world_y_drift_abs_m": "world_y_drift_abs_m",
        "tracking/world_y_drift_signed_m": "world_y_drift_signed_m",
        "tracking/yaw_drift_abs_rad": "yaw_drift_abs_rad",
        "tracking/yaw_drift_signed_rad": "yaw_drift_signed_rad",
    }
    for source, destination in aliases.items():
        if source in metrics:
            selected[destination] = metrics[source]
    for name in TORQUE_ACTUATOR_NAMES:
        key = f"walking_stable_torque/{name}/sat_frac"
        if key in metrics:
            selected[key] = metrics[key]
    return selected


def _build_eval(
    *,
    checkpoint_path: Path,
    config_path: Path,
    cmd: tuple[float, float, float],
    num_envs: int,
    num_steps: int,
    seed: int,
) -> tuple[Any, ...]:
    training_cfg = load_training_config(config_path)
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    load_robot_config(robot_cfg_path)
    training_cfg.ppo.num_envs = int(num_envs)
    training_cfg.ppo.rollout_steps = int(num_steps)
    training_cfg.env.eval_velocity_cmd = tuple(float(v) for v in cmd)
    training_cfg.freeze()

    env = WildRobotEnv(config=training_cfg)
    rng = jax.random.PRNGKey(seed)
    reset_rngs = jax.random.split(rng, num_envs)
    env_state = jax.vmap(env.reset_for_eval)(reset_rngs)
    obs_dim = int(env_state.obs.shape[-1])
    action_dim = int(env.action_size)
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
        activation=_network_activation_name(training_cfg),
    )
    checkpoint = load_checkpoint(str(checkpoint_path))
    checkpoint_dims = get_checkpoint_dims(checkpoint_path)
    if checkpoint_dims != (obs_dim, action_dim):
        raise ValueError(
            "Checkpoint policy dimensions do not match the eval config: "
            f"checkpoint={checkpoint_dims}, env={(obs_dim, action_dim)}"
        )
    return (
        training_cfg,
        env,
        env_state,
        ppo_network,
        checkpoint["policy_params"],
        checkpoint.get("processor_params", ()),
        rng,
    )


def _run_one_direction(
    *,
    checkpoint_path: Path,
    config_path: Path,
    cmd: tuple[float, float, float],
    num_envs: int,
    num_steps: int,
    seed: int,
) -> tuple[dict[str, Any], WildRobotEnv]:
    (
        training_cfg,
        env,
        env_state,
        ppo_network,
        policy_params,
        processor_params,
        rng,
    ) = _build_eval(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        cmd=cmd,
        num_envs=num_envs,
        num_steps=num_steps,
        seed=seed,
    )
    traj, _ = _collect_eval_rollout(
        env=env,
        env_state=env_state,
        policy_params=policy_params,
        processor_params=processor_params,
        ppo_network=ppo_network,
        rng=rng,
        num_steps=num_steps,
        deterministic=True,
        disable_cmd_resample=True,
        disable_pushes=True,
    )
    metrics = _compute_eval_metrics(
        traj,
        num_steps,
        ctrl_dt=float(training_cfg.env.ctrl_dt),
        include_walking_orientation=True,
    )
    action_summary, mean_action = _action_summary(
        observations=traj.obs,
        spec=env._policy_spec,
        ppo_network=ppo_network,
        processor_params=processor_params,
        policy_params=policy_params,
    )
    actor_symmetry = actor_mirror_metrics(
        observations=traj.obs,
        spec=env._policy_spec,
        ppo_network=ppo_network,
        processor_params=processor_params,
        policy_params=policy_params,
    )
    result = {
        "command": list(cmd),
        "rollout": _select_rollout_metrics(metrics),
        "actions": action_summary,
        "actor_mirror": actor_symmetry,
        "_mean_action": mean_action,
    }
    return result, env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reference, actor, and closed-loop left/right symmetry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--vx", type=float, default=0.13)
    parser.add_argument("--vy", type=float, default=0.065)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vy <= 0.0:
        raise ValueError("--vy must be positive; the script evaluates both signs")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    positive_cmd = (float(args.vx), float(args.vy), float(args.wz))
    negative_cmd = (float(args.vx), -float(args.vy), -float(args.wz))
    positive, positive_env = _run_one_direction(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        cmd=positive_cmd,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    negative, _ = _run_one_direction(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        cmd=negative_cmd,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    spec = positive_env._policy_spec
    positive_mean = positive.pop("_mean_action")
    negative_mean = negative.pop("_mean_action")
    expected_negative_mean = mirror_actions(positive_mean, spec)
    mean_action_error = np.asarray(negative_mean - expected_negative_mean)

    pos_rollout = positive["rollout"]
    neg_rollout = negative["rollout"]
    paired: dict[str, Any] = {
        "mean_action_mirror_mae": float(np.mean(np.abs(mean_action_error))),
        "mean_action_mirror_max_abs_error": float(np.max(np.abs(mean_action_error))),
        "achieved_vy_sum_m_s": float(
            pos_rollout["lateral_velocity_signed_m_s"]
            + neg_rollout["lateral_velocity_signed_m_s"]
        ),
        "world_y_drift_sum_m": float(
            pos_rollout["world_y_drift_signed_m"]
            + neg_rollout["world_y_drift_signed_m"]
        ),
    }
    for i, name in enumerate(spec.robot.actuator_names):
        paired[f"mean_action_mirror_error/{name}"] = float(mean_action_error[i])

    torque_pair_errors = []
    source, _ = joint_mirror_transform(spec.robot.actuator_names)
    names = list(spec.robot.actuator_names)
    for dst_idx, src_idx in enumerate(np.asarray(source)):
        dst_name = names[dst_idx]
        src_name = names[int(src_idx)]
        pos_key = f"walking_stable_torque/{src_name}/sat_frac"
        neg_key = f"walking_stable_torque/{dst_name}/sat_frac"
        if pos_key not in pos_rollout or neg_key not in neg_rollout:
            continue
        error = float(neg_rollout[neg_key] - pos_rollout[pos_key])
        paired[f"stable_torque_sat_mirror_error/{dst_name}"] = error
        torque_pair_errors.append(abs(error))
    if torque_pair_errors:
        paired["stable_torque_sat_mirror_mae"] = float(np.mean(torque_pair_errors))
        paired["stable_torque_sat_mirror_max_abs_error"] = float(
            np.max(torque_pair_errors)
        )

    probe = jnp.arange(spec.model.obs_dim, dtype=jnp.float32)
    obs_involution = mirror_observations(mirror_observations(probe, spec), spec)
    action_probe = jnp.linspace(-1.0, 1.0, spec.model.action_dim)
    action_involution = mirror_actions(mirror_actions(action_probe, spec), spec)

    report = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "transform_validation": {
            "observation_involution_max_abs_error": float(
                jnp.max(jnp.abs(obs_involution - probe))
            ),
            "action_involution_max_abs_error": float(
                jnp.max(jnp.abs(action_involution - action_probe))
            ),
        },
        "reference_mirror": _reference_symmetry_metrics(
            positive_env, positive_cmd, negative_cmd
        ),
        "positive_vy": positive,
        "negative_vy": negative,
        "paired_closed_loop": paired,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
        print(f"Wrote symmetry diagnostic to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
