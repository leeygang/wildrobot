#!/usr/bin/env python3
"""Measure hip/ankle roll load sharing by measured support phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from assets.robot_config import load_robot_config
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.configs.training_config import load_training_config
from training.core.checkpoint import load_checkpoint
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.eval_policy import _network_activation_name
from training.eval.standing_orientation import (
    WALKING_PRE_FALL_WINDOW_S,
    WALKING_STABLE_START_S,
    _walking_rollout_masks,
)
from training.exports.export_onnx import get_checkpoint_dims


ROLL_JOINT_NAMES = (
    "left_hip_roll",
    "left_ankle_roll",
    "right_hip_roll",
    "right_ankle_roll",
)


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _summarize_joint(
    *,
    mask: np.ndarray,
    joint_index: int,
    torque_nm: np.ndarray,
    torque_ratio: np.ndarray,
    policy_action: np.ndarray,
    applied_action: np.ndarray,
    target_error_rad: np.ndarray,
) -> dict[str, float]:
    torque = torque_nm[..., joint_index][mask]
    ratio = torque_ratio[..., joint_index][mask]
    policy = policy_action[..., joint_index][mask]
    applied = applied_action[..., joint_index][mask]
    target_error = target_error_rad[..., joint_index][mask]
    abs_torque = np.abs(torque)
    abs_policy = np.abs(policy)
    abs_applied = np.abs(applied)
    abs_target_error = np.abs(target_error)
    return {
        "torque_signed_mean_nm": float(np.mean(torque)) if torque.size else 0.0,
        "torque_abs_mean_nm": float(np.mean(abs_torque)) if torque.size else 0.0,
        "torque_abs_p95_nm": _percentile(abs_torque, 95.0),
        "torque_ratio_mean": float(np.mean(ratio)) if ratio.size else 0.0,
        "torque_ratio_p95": _percentile(ratio, 95.0),
        "torque_saturation_frac": (float(np.mean(ratio > 0.95)) if ratio.size else 0.0),
        "policy_action_signed_mean": (float(np.mean(policy)) if policy.size else 0.0),
        "policy_action_abs_mean": (float(np.mean(abs_policy)) if policy.size else 0.0),
        "applied_action_signed_mean": (
            float(np.mean(applied)) if applied.size else 0.0
        ),
        "applied_action_abs_mean": (
            float(np.mean(abs_applied)) if applied.size else 0.0
        ),
        "applied_action_abs_p95": _percentile(abs_applied, 95.0),
        "target_error_signed_mean_rad": (
            float(np.mean(target_error)) if target_error.size else 0.0
        ),
        "target_error_abs_mean_rad": (
            float(np.mean(abs_target_error)) if target_error.size else 0.0
        ),
        "target_error_abs_p95_rad": _percentile(abs_target_error, 95.0),
    }


def _summarize_support_leverage(
    *,
    mask: np.ndarray,
    lateral_lever_m: np.ndarray,
    robot_weight_n: float,
) -> dict[str, float]:
    lever = lateral_lever_m[mask]
    abs_lever = np.abs(lever)
    moment = abs_lever * robot_weight_n
    return {
        "lateral_lever_signed_mean_m": (float(np.mean(lever)) if lever.size else 0.0),
        "lateral_lever_abs_mean_m": (float(np.mean(abs_lever)) if lever.size else 0.0),
        "lateral_lever_abs_p95_m": _percentile(abs_lever, 95.0),
        "quasi_static_gravity_moment_abs_mean_nm": (
            float(np.mean(moment)) if moment.size else 0.0
        ),
        "quasi_static_gravity_moment_abs_p95_nm": _percentile(moment, 95.0),
    }


def summarize_roll_load_sharing(
    *,
    joint_names: Sequence[str],
    torque_nm: np.ndarray,
    torque_ratio: np.ndarray,
    policy_action: np.ndarray,
    applied_action: np.ndarray,
    target_error_rad: np.ndarray,
    com_to_left_foot_lateral_m: np.ndarray,
    com_to_right_foot_lateral_m: np.ndarray,
    left_loaded: np.ndarray,
    right_loaded: np.ndarray,
    dones: np.ndarray,
    truncations: np.ndarray,
    ctrl_dt: float,
    robot_weight_n: float,
    stable_start_s: float = WALKING_STABLE_START_S,
    pre_fall_window_s: float = WALKING_PRE_FALL_WINDOW_S,
) -> dict[str, Any]:
    """Summarize roll-joint behavior by measured foot-support phase."""
    state_arrays = (
        torque_nm,
        torque_ratio,
        policy_action,
        applied_action,
        target_error_rad,
    )
    if any(array.ndim != 3 for array in state_arrays):
        raise ValueError("actuator arrays must have shape (T, N, A)")
    if any(array.shape != torque_nm.shape for array in state_arrays[1:]):
        raise ValueError("actuator arrays must have matching shapes")
    if torque_nm.shape[-1] != len(joint_names):
        raise ValueError("joint_names must match the actuator dimension")
    step_shape = torque_nm.shape[:2]
    step_arrays = (
        com_to_left_foot_lateral_m,
        com_to_right_foot_lateral_m,
        left_loaded,
        right_loaded,
        dones,
        truncations,
    )
    if any(array.shape != step_shape for array in step_arrays):
        raise ValueError("support/geometry/done arrays must have shape (T, N)")
    if ctrl_dt <= 0.0:
        raise ValueError("ctrl_dt must be positive")
    if robot_weight_n <= 0.0:
        raise ValueError("robot_weight_n must be positive")

    (
        first_episode,
        _terminal_event,
        _failure_event,
        failed_env,
        survivor_env,
        stable_mask,
        pre_fall_mask,
        _first_failure_index,
    ) = _walking_rollout_masks(
        jnp.asarray(dones),
        jnp.asarray(truncations),
        ctrl_dt=ctrl_dt,
        stable_start_s=stable_start_s,
        pre_fall_window_s=pre_fall_window_s,
    )
    valid_state = np.asarray(dones) <= 0.5
    windows = {
        "stable_survivors": np.asarray(stable_mask) & valid_state,
        "pre_fall": np.asarray(pre_fall_mask) & valid_state,
    }
    support_masks = {
        "left_only": (left_loaded > 0.5) & (right_loaded <= 0.5),
        "right_only": (left_loaded <= 0.5) & (right_loaded > 0.5),
        "double_support": (left_loaded > 0.5) & (right_loaded > 0.5),
        "flight": (left_loaded <= 0.5) & (right_loaded <= 0.5),
    }
    loaded_foot_leverage = {
        "left_only": com_to_left_foot_lateral_m,
        "right_only": com_to_right_foot_lateral_m,
    }
    joint_indices = {name: joint_names.index(name) for name in ROLL_JOINT_NAMES}

    output: dict[str, Any] = {
        "first_episode_sample_count": int(np.sum(np.asarray(first_episode))),
        "survivor_env_count": int(np.sum(np.asarray(survivor_env))),
        "failed_env_count": int(np.sum(np.asarray(failed_env))),
        "windows": {},
    }
    for window_name, window_mask in windows.items():
        window_output: dict[str, Any] = {
            "sample_count": int(np.sum(window_mask)),
            "support_phases": {},
        }
        for phase_name, support_mask in support_masks.items():
            phase_mask = window_mask & support_mask
            phase_output = {
                "sample_count": int(np.sum(phase_mask)),
                "sample_fraction": (
                    float(np.sum(phase_mask) / np.sum(window_mask))
                    if np.any(window_mask)
                    else 0.0
                ),
                "joints": {
                    joint_name: _summarize_joint(
                        mask=phase_mask,
                        joint_index=joint_index,
                        torque_nm=torque_nm,
                        torque_ratio=torque_ratio,
                        policy_action=policy_action,
                        applied_action=applied_action,
                        target_error_rad=target_error_rad,
                    )
                    for joint_name, joint_index in joint_indices.items()
                },
            }
            leverage = loaded_foot_leverage.get(phase_name)
            if leverage is not None:
                phase_output["com_to_loaded_foot"] = _summarize_support_leverage(
                    mask=phase_mask,
                    lateral_lever_m=leverage,
                    robot_weight_n=robot_weight_n,
                )
            window_output["support_phases"][phase_name] = phase_output
        output["windows"][window_name] = window_output
    return output


def _collect_rollout(
    *,
    env: WildRobotEnv,
    env_state: Any,
    policy_params: Any,
    processor_params: Any,
    ppo_network: Any,
    rng: jax.Array,
    num_steps: int,
) -> Mapping[str, jax.Array]:
    batch_step = jax.vmap(
        lambda state, action: env.step(
            state,
            action,
            disable_cmd_resample=True,
            disable_pushes=True,
        )
    )
    policy_ctrl_ids = env._ctrl_mapper.policy_to_mj_order_jax
    qpos_ids = env._actuator_qpos_addrs
    actuator_ids = env._cal._actuator_ids
    force_limits = env._cal._force_limits
    root_body_id = int(
        mujoco.mj_name2id(
            env._mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(env._robot_config.floating_base_body),
        )
    )
    if root_body_id < 0:
        raise ValueError(
            "Floating-base body not found: " f"{env._robot_config.floating_base_body}"
        )

    def step_fn(carry, _):
        state, scan_rng = carry
        scan_rng, action_rng = jax.random.split(scan_rng)
        policy_action, _, _ = sample_actions(
            processor_params,
            policy_params,
            ppo_network,
            state.obs,
            action_rng,
            deterministic=True,
        )
        next_state = batch_step(state, policy_action)
        metrics = next_state.metrics[METRICS_VEC_KEY]
        torque_nm = next_state.data.actuator_force[:, actuator_ids]
        ctrl_policy = next_state.data.ctrl[:, policy_ctrl_ids]
        q_actual = next_state.data.qpos[:, qpos_ids]
        root_quat = next_state.data.qpos[:, 3:7]
        qw, qx, qy, qz = [root_quat[:, index] for index in range(4)]
        base_lateral = jnp.stack(
            (
                2.0 * (qx * qy - qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz + qw * qx),
            ),
            axis=-1,
        )
        whole_body_com = next_state.data.subtree_com[:, root_body_id]
        left_foot_position = next_state.data.xpos[:, env._left_foot_body_id]
        right_foot_position = next_state.data.xpos[:, env._right_foot_body_id]
        output = {
            "torque_nm": torque_nm,
            "torque_ratio": jnp.abs(torque_nm) / (force_limits + 1e-6),
            "policy_action": policy_action,
            "applied_action": next_state.info[WR_INFO_KEY].prev_action,
            "target_error_rad": ctrl_policy - q_actual,
            "com_to_left_foot_lateral_m": jnp.sum(
                (whole_body_com - left_foot_position) * base_lateral,
                axis=-1,
            ),
            "com_to_right_foot_lateral_m": jnp.sum(
                (whole_body_com - right_foot_position) * base_lateral,
                axis=-1,
            ),
            "left_loaded": metrics[:, METRIC_INDEX["support/left_loaded"]],
            "right_loaded": metrics[:, METRIC_INDEX["support/right_loaded"]],
            "done": next_state.done,
            "truncation": metrics[:, METRIC_INDEX["term/truncated"]],
        }
        return (next_state, scan_rng), output

    (_, _), rollout = jax.lax.scan(
        step_fn,
        (env_state, rng),
        None,
        length=num_steps,
    )
    return rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure roll-joint load sharing by measured support phase",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--vx", type=float, default=0.13)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    training_cfg = load_training_config(args.config)
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    load_robot_config(robot_cfg_path)
    training_cfg.ppo.num_envs = int(args.num_envs)
    training_cfg.ppo.rollout_steps = int(args.num_steps)
    training_cfg.env.eval_velocity_cmd = (float(args.vx), 0.0, 0.0)
    training_cfg.freeze()

    env = WildRobotEnv(config=training_cfg)
    rng = jax.random.PRNGKey(args.seed)
    env_state = jax.vmap(env.reset_for_eval)(jax.random.split(rng, args.num_envs))
    obs_dim = int(env_state.obs.shape[-1])
    action_dim = int(env.action_size)
    checkpoint_dims = get_checkpoint_dims(args.checkpoint)
    if checkpoint_dims != (obs_dim, action_dim):
        raise ValueError(
            "Checkpoint policy dimensions do not match the eval config: "
            f"checkpoint={checkpoint_dims}, env={(obs_dim, action_dim)}"
        )
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
        activation=_network_activation_name(training_cfg),
    )
    checkpoint = load_checkpoint(str(args.checkpoint))
    rollout = jax.device_get(
        _collect_rollout(
            env=env,
            env_state=env_state,
            policy_params=checkpoint["policy_params"],
            processor_params=checkpoint.get("processor_params", ()) or (),
            ppo_network=ppo_network,
            rng=rng,
            num_steps=args.num_steps,
        )
    )
    joint_names = tuple(env._policy_spec.robot.actuator_names)
    summary = summarize_roll_load_sharing(
        joint_names=joint_names,
        torque_nm=np.asarray(rollout["torque_nm"]),
        torque_ratio=np.asarray(rollout["torque_ratio"]),
        policy_action=np.asarray(rollout["policy_action"]),
        applied_action=np.asarray(rollout["applied_action"]),
        target_error_rad=np.asarray(rollout["target_error_rad"]),
        com_to_left_foot_lateral_m=np.asarray(rollout["com_to_left_foot_lateral_m"]),
        com_to_right_foot_lateral_m=np.asarray(rollout["com_to_right_foot_lateral_m"]),
        left_loaded=np.asarray(rollout["left_loaded"]),
        right_loaded=np.asarray(rollout["right_loaded"]),
        dones=np.asarray(rollout["done"]),
        truncations=np.asarray(rollout["truncation"]),
        ctrl_dt=float(training_cfg.env.ctrl_dt),
        robot_weight_n=float(
            env._mj_model.body_subtreemass[
                mujoco.mj_name2id(
                    env._mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    str(env._robot_config.floating_base_body),
                )
            ]
            * np.linalg.norm(env._mj_model.opt.gravity)
        ),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "command": [float(args.vx), 0.0, 0.0],
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "summary": summary,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
