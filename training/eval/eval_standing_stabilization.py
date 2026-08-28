#!/usr/bin/env python3
"""Evaluate bounded standing stabilization against paired home hold.

The evaluator samples initial tilt, angular velocity, and foot stagger inside
the requested deployment envelope. Every sampled state is cloned across the
deterministic policy and a zero-residual home-hold controller. Clean and
configured-push suites therefore use common initial conditions, randomized
dynamics, and push schedules.

Pass/fail is computed from the original episode only. The evaluator bypasses
the training time limit so a 60-second rollout is continuous, while physical
height and tilt terminations remain active. A pass requires the robot to remain
inside the configured tilt, gyro, joint-to-home, and bilateral support bounds
for the final continuous settle window.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.configs.training_config import load_training_config
from training.core.checkpoint import load_checkpoint
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.eval_policy import _network_activation_name
from training.cal.types import CoordinateFrame
from training.eval.eval_standing_recovery_grid import (
    _apply_recovery_condition,
    _pitch_from_quat_wxyz,
    _roll_from_quat_wxyz,
)
from training.exports.export_onnx import get_checkpoint_dims


_CONTROLLERS = ("policy", "home")
_KNOWN_SUITES = ("clean", "push")


def _continuous_eval_step(
    env: WildRobotEnv,
    state,
    action,
    *,
    disable_pushes: bool,
):
    """Step without the training timeout, retaining physical termination."""
    return env.step(
        state,
        action,
        disable_cmd_resample=True,
        disable_pushes=disable_pushes,
        disable_time_limit=True,
    )


def _parse_suites(value: str) -> tuple[str, ...]:
    suites = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not suites:
        raise argparse.ArgumentTypeError("Expected at least one evaluation suite")
    unknown = sorted(set(suites) - set(_KNOWN_SUITES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown suites {unknown}; expected a comma-separated subset of {_KNOWN_SUITES}"
        )
    return tuple(dict.fromkeys(suites))


def _parse_range(value: str) -> tuple[float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated low,high range: {value!r}"
        ) from exc
    if len(parts) != 2 or not all(math.isfinite(part) for part in parts):
        raise argparse.ArgumentTypeError(
            f"Expected exactly two finite range values: {value!r}"
        )
    if parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("Range low must be <= high")
    return float(parts[0]), float(parts[1])


def _sample_initial_conditions(
    *,
    seed: int,
    num_envs: int,
    max_tilt_deg: float,
    max_gyro_rad_s: float,
    foot_stagger_range_m: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Sample uniformly over tilt and gyro disks plus a stagger interval."""
    rng = np.random.default_rng(int(seed))
    tilt_radius = np.deg2rad(float(max_tilt_deg)) * np.sqrt(rng.random(num_envs))
    tilt_angle = rng.uniform(0.0, 2.0 * np.pi, num_envs)
    gyro_radius = float(max_gyro_rad_s) * np.sqrt(rng.random(num_envs))
    gyro_angle = rng.uniform(0.0, 2.0 * np.pi, num_envs)
    return {
        "roll_rad": tilt_radius * np.cos(tilt_angle),
        "pitch_rad": tilt_radius * np.sin(tilt_angle),
        "roll_rate_rad_s": gyro_radius * np.cos(gyro_angle),
        "pitch_rate_rad_s": gyro_radius * np.sin(gyro_angle),
        "foot_stagger_m": rng.uniform(
            foot_stagger_range_m[0], foot_stagger_range_m[1], num_envs
        ),
    }


def _tilt_from_quat_wxyz(quat: jax.Array) -> jax.Array:
    quat = quat / jnp.maximum(
        jnp.linalg.norm(quat, axis=-1, keepdims=True), jnp.float32(1e-8)
    )
    body_z_world_z = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
    return jnp.arccos(jnp.clip(body_z_world_z, -1.0, 1.0)).astype(jnp.float32)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _rate_summary(values: list[bool]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=bool)
    successes = int(np.sum(array))
    low, high = _wilson_interval(successes, int(array.size))
    return {
        "rate": float(np.mean(array)) if array.size else 0.0,
        "count": successes,
        "total": int(array.size),
        "wilson95_low": low,
        "wilson95_high": high,
    }


def _summarize_rollout(
    rollout: dict[str, np.ndarray],
    *,
    ctrl_dt: float,
    settle_window_s: float,
    settle_tilt_deg: float,
    settle_gyro_rad_s: float,
    settle_joint_max_deg: float,
    settle_joint_rms_deg: float,
    initial_conditions: dict[str, np.ndarray],
) -> dict[str, Any]:
    active = np.asarray(rollout["active"], dtype=bool)
    failed = np.asarray(rollout["failed"], dtype=bool)
    tilt_deg = np.rad2deg(np.asarray(rollout["tilt_rad"], dtype=np.float64))
    gyro_norm = np.asarray(rollout["gyro_norm_rad_s"], dtype=np.float64)
    joint_max_deg = np.rad2deg(
        np.asarray(rollout["joint_home_max_rad"], dtype=np.float64)
    )
    joint_rms_deg = np.rad2deg(
        np.asarray(rollout["joint_home_rms_rad"], dtype=np.float64)
    )
    both_loaded = np.asarray(rollout["both_loaded"], dtype=np.float64)
    within = np.asarray(rollout["within_envelope"], dtype=bool)
    _, n_envs = within.shape
    window_steps = max(1, int(round(float(settle_window_s) / float(ctrl_dt))))
    per_env: list[dict[str, Any]] = []

    for env_idx in range(n_envs):
        valid_indices = np.flatnonzero(active[:, env_idx])
        last_idx = int(valid_indices[-1]) if valid_indices.size else 0
        valid_count = last_idx + 1
        fell = bool(np.any(failed[:valid_count, env_idx]))
        final_start = max(0, valid_count - window_steps)
        final_window_complete = valid_count >= window_steps
        final_slice = slice(final_start, valid_count)
        final_tilt = float(np.max(tilt_deg[final_slice, env_idx]))
        final_gyro = float(np.max(gyro_norm[final_slice, env_idx]))
        final_joint_max = float(np.max(joint_max_deg[final_slice, env_idx]))
        final_joint_rms = float(np.max(joint_rms_deg[final_slice, env_idx]))
        final_both_loaded = bool(
            np.all(both_loaded[final_slice, env_idx] > 0.5)
        )
        criteria = {
            "tilt": final_tilt <= float(settle_tilt_deg),
            "gyro": final_gyro <= float(settle_gyro_rad_s),
            "joint_home_max": final_joint_max <= float(settle_joint_max_deg),
            "joint_home_rms": final_joint_rms <= float(settle_joint_rms_deg),
            "both_loaded": final_both_loaded,
        }
        final_stable = bool(
            not fell and final_window_complete and all(criteria.values())
        )
        fail_reasons: list[str] = []
        if fell:
            fail_reasons.append("fall")
        if not final_window_complete:
            fail_reasons.append("insufficient_window")
        fail_reasons.extend(name for name, passed in criteria.items() if not passed)

        final_run_start = valid_count
        while final_run_start > 0 and within[final_run_start - 1, env_idx]:
            final_run_start -= 1
        stable_tail_steps = valid_count - final_run_start
        settle_time_s = (
            float(final_run_start * ctrl_dt)
            if final_stable and stable_tail_steps >= window_steps
            else None
        )
        valid_slice = slice(0, valid_count)
        per_env.append(
            {
                "env_index": env_idx,
                "passed": final_stable,
                "fell": fell,
                "fail_reasons": fail_reasons,
                "settle_time_s": settle_time_s,
                "stable_tail_s": float(stable_tail_steps * ctrl_dt),
                "peak_tilt_deg": float(np.max(tilt_deg[valid_slice, env_idx])),
                "peak_gyro_rad_s": float(np.max(gyro_norm[valid_slice, env_idx])),
                "peak_joint_home_max_deg": float(
                    np.max(joint_max_deg[valid_slice, env_idx])
                ),
                "final_tilt_deg_max": final_tilt,
                "final_gyro_rad_s_max": final_gyro,
                "final_joint_home_max_deg": final_joint_max,
                "final_joint_home_rms_deg": final_joint_rms,
                "final_both_loaded": final_both_loaded,
                "both_loaded_fraction": float(
                    np.mean(both_loaded[valid_slice, env_idx])
                ),
                "initial_roll_deg": float(
                    np.rad2deg(initial_conditions["roll_rad"][env_idx])
                ),
                "initial_pitch_deg": float(
                    np.rad2deg(initial_conditions["pitch_rad"][env_idx])
                ),
                "initial_roll_rate_rad_s": float(
                    initial_conditions["roll_rate_rad_s"][env_idx]
                ),
                "initial_pitch_rate_rad_s": float(
                    initial_conditions["pitch_rate_rad_s"][env_idx]
                ),
                "foot_stagger_m": float(
                    initial_conditions["foot_stagger_m"][env_idx]
                ),
            }
        )

    passed = [bool(item["passed"]) for item in per_env]
    fell = [bool(item["fell"]) for item in per_env]
    settle_times = [
        float(item["settle_time_s"])
        for item in per_env
        if item["settle_time_s"] is not None
    ]
    failure_reason_counts = {
        reason: sum(reason in item["fail_reasons"] for item in per_env)
        for reason in (
            "fall",
            "insufficient_window",
            "tilt",
            "gyro",
            "joint_home_max",
            "joint_home_rms",
            "both_loaded",
        )
    }
    return {
        "pass": _rate_summary(passed),
        "fall": _rate_summary(fell),
        "settle_time_s_mean": (
            float(np.mean(settle_times)) if settle_times else None
        ),
        "settle_time_s_p95": (
            float(np.percentile(settle_times, 95.0)) if settle_times else None
        ),
        "peak_tilt_deg_max": max(item["peak_tilt_deg"] for item in per_env),
        "final_tilt_deg_max": max(item["final_tilt_deg_max"] for item in per_env),
        "final_gyro_rad_s_max": max(
            item["final_gyro_rad_s_max"] for item in per_env
        ),
        "final_joint_home_max_deg": max(
            item["final_joint_home_max_deg"] for item in per_env
        ),
        "final_joint_home_rms_deg_max": max(
            item["final_joint_home_rms_deg"] for item in per_env
        ),
        "both_loaded_fraction_mean": float(
            np.mean([item["both_loaded_fraction"] for item in per_env])
        ),
        "failure_reason_counts": failure_reason_counts,
        "per_env": per_env,
    }


def _paired_comparison(policy: dict[str, Any], home: dict[str, Any]) -> dict[str, Any]:
    policy_envs = policy["per_env"]
    home_envs = home["per_env"]
    policy_only = []
    home_only = []
    lower_peak_tilt = []
    for policy_env, home_env in zip(policy_envs, home_envs, strict=True):
        policy_pass = bool(policy_env["passed"])
        home_pass = bool(home_env["passed"])
        policy_only.append(policy_pass and not home_pass)
        home_only.append(home_pass and not policy_pass)
        lower_peak_tilt.append(
            float(policy_env["peak_tilt_deg"]) < float(home_env["peak_tilt_deg"])
        )
    return {
        "policy_pass_home_fail": _rate_summary(policy_only),
        "home_pass_policy_fail": _rate_summary(home_only),
        "policy_lower_peak_tilt": _rate_summary(lower_peak_tilt),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded standing stabilization against home hold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-s", type=float, default=60.0)
    parser.add_argument("--suites", type=_parse_suites, default=("clean", "push"))
    parser.add_argument("--initial-max-tilt-deg", type=float, default=4.0)
    parser.add_argument("--initial-max-gyro-rad-s", type=float, default=0.35)
    parser.add_argument(
        "--foot-stagger-range-m", type=_parse_range, default=(-0.04, 0.04)
    )
    parser.add_argument("--settle-tilt-deg", type=float, default=3.0)
    parser.add_argument("--settle-gyro-rad-s", type=float, default=0.10)
    parser.add_argument("--settle-joint-max-deg", type=float, default=8.0)
    parser.add_argument("--settle-joint-rms-deg", type=float, default=4.0)
    parser.add_argument("--settle-window-s", type=float, default=0.5)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "num_envs": args.num_envs,
        "rollout_s": args.rollout_s,
        "settle_tilt_deg": args.settle_tilt_deg,
        "settle_gyro_rad_s": args.settle_gyro_rad_s,
        "settle_joint_max_deg": args.settle_joint_max_deg,
        "settle_joint_rms_deg": args.settle_joint_rms_deg,
        "settle_window_s": args.settle_window_s,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0.0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.initial_max_tilt_deg < 0.0 or args.initial_max_gyro_rad_s < 0.0:
        raise ValueError("Initial tilt and gyro limits must be non-negative")
    if args.settle_window_s > args.rollout_s:
        raise ValueError("--settle-window-s cannot exceed --rollout-s")
    if not args.checkpoint.is_file() or not args.config.is_file():
        raise FileNotFoundError("Checkpoint and config must both be files")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    if args.platform:
        jax.config.update("jax_platform_name", args.platform)

    training_cfg = load_training_config(args.config)
    if "push" in args.suites and not bool(training_cfg.env.push_enabled):
        raise ValueError("The push suite requires env.push_enabled=true in the config")
    robot_cfg_path = Path(training_cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)
    training_cfg.freeze()
    env = WildRobotEnv(config=training_cfg)
    ctrl_dt = float(training_cfg.env.ctrl_dt)
    num_steps = max(1, int(round(args.rollout_s / ctrl_dt)))

    checkpoint = load_checkpoint(str(args.checkpoint))
    obs_dim, action_dim = get_checkpoint_dims(args.checkpoint)
    if obs_dim != int(env._policy_spec.model.obs_dim):
        raise ValueError(
            f"Checkpoint obs_dim={obs_dim} does not match env obs_dim="
            f"{env._policy_spec.model.obs_dim}"
        )
    if action_dim != env.action_size:
        raise ValueError(
            f"Checkpoint action_dim={action_dim} does not match env action_dim="
            f"{env.action_size}"
        )
    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(training_cfg.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(training_cfg.networks.critic.hidden_sizes),
        activation=_network_activation_name(training_cfg),
    )
    policy_params = checkpoint["policy_params"]
    processor_params = checkpoint.get("processor_params", ()) or ()

    sampled = _sample_initial_conditions(
        seed=args.seed,
        num_envs=args.num_envs,
        max_tilt_deg=args.initial_max_tilt_deg,
        max_gyro_rad_s=args.initial_max_gyro_rad_s,
        foot_stagger_range_m=args.foot_stagger_range_m,
    )
    sampled_jax = {
        key: jnp.asarray(value, dtype=jnp.float32) for key, value in sampled.items()
    }
    base_rng = jax.random.PRNGKey(args.seed)
    reset_rngs = jax.random.split(base_rng, args.num_envs)

    @jax.jit
    def reset_batch(rngs, conditions):
        def reset_one(rng, roll, pitch, roll_rate, pitch_rate, stagger):
            base = env.reset_for_eval(
                rng,
                cmd_override=jnp.zeros(3, dtype=jnp.float32),
                perturb_pose=False,
            )
            return _apply_recovery_condition(
                env,
                base,
                roll_rad=roll,
                pitch_rad=pitch,
                roll_rate_rad_s=roll_rate,
                pitch_rate_rad_s=pitch_rate,
                foot_stagger_m=stagger,
            )

        return jax.vmap(reset_one)(
            rngs,
            conditions["roll_rad"],
            conditions["pitch_rad"],
            conditions["roll_rate_rad_s"],
            conditions["pitch_rate_rad_s"],
            conditions["foot_stagger_m"],
        )

    initial_state = reset_batch(reset_rngs, sampled_jax)
    initial_quat = initial_state.data.qpos[:, 3:7]
    actual_roll = np.asarray(_roll_from_quat_wxyz(initial_quat))
    actual_pitch = np.asarray(_pitch_from_quat_wxyz(initial_quat))
    actual_rates = np.asarray(initial_state.data.qvel[:, 3:6])
    actual_initial_tilt_deg = np.rad2deg(np.asarray(_tilt_from_quat_wxyz(initial_quat)))
    actual_initial_gyro = np.linalg.norm(actual_rates, axis=1)
    left_foot, right_foot = jax.vmap(
        lambda data: env._cal.get_foot_positions(
            data,
            normalize=False,
            frame=CoordinateFrame.WORLD,
        )
    )(initial_state.data)
    actual_stagger = np.asarray(left_foot[:, 0] - right_foot[:, 0])
    initial_conditions = {
        "roll_rad": actual_roll,
        "pitch_rad": actual_pitch,
        "roll_rate_rad_s": actual_rates[:, 0],
        "pitch_rate_rad_s": actual_rates[:, 1],
        "foot_stagger_m": actual_stagger,
    }
    @partial(
        jax.jit,
        static_argnames=("controller", "disable_pushes", "steps"),
    )
    def run_controller(
        start_state,
        rng,
        *,
        controller: str,
        disable_pushes: bool,
        steps: int,
    ):
        batch_step = jax.vmap(
            lambda state, action: _continuous_eval_step(
                env,
                state,
                action,
                disable_pushes=disable_pushes,
            )
        )

        def scan_step(carry, _):
            state, scan_rng, alive = carry
            scan_rng, action_rng = jax.random.split(scan_rng)
            if controller == "policy":
                action, _, _ = sample_actions(
                    processor_params,
                    policy_params,
                    ppo_network,
                    state.obs,
                    action_rng,
                    deterministic=True,
                )
            else:
                action = jnp.zeros(
                    (state.obs.shape[0], action_dim), dtype=jnp.float32
                )
            next_state = batch_step(state, action)
            metrics = next_state.metrics[METRICS_VEC_KEY]
            truncated = metrics[:, METRIC_INDEX["term/truncated"]] > 0.5
            done = next_state.done > 0.5
            quat = next_state.data.qpos[:, 3:7]
            tilt_rad = _tilt_from_quat_wxyz(quat)
            gyro_norm = jnp.linalg.norm(next_state.data.qvel[:, 3:6], axis=1)
            joint_q = next_state.data.qpos[:, env._actuator_qpos_addrs]
            joint_err = joint_q - env._home_q_rad
            joint_max = jnp.max(jnp.abs(joint_err), axis=1)
            joint_rms = jnp.sqrt(jnp.mean(joint_err * joint_err, axis=1))
            both_loaded = metrics[:, METRIC_INDEX["support/both_loaded"]]
            within = (
                (tilt_rad <= jnp.deg2rad(args.settle_tilt_deg))
                & (gyro_norm <= jnp.float32(args.settle_gyro_rad_s))
                & (joint_max <= jnp.deg2rad(args.settle_joint_max_deg))
                & (joint_rms <= jnp.deg2rad(args.settle_joint_rms_deg))
                & (both_loaded > 0.5)
            )
            output = {
                "active": alive,
                "failed": done & ~truncated,
                "tilt_rad": tilt_rad,
                "gyro_norm_rad_s": gyro_norm,
                "joint_home_max_rad": joint_max,
                "joint_home_rms_rad": joint_rms,
                "both_loaded": both_loaded,
                "within_envelope": within,
            }
            alive = alive & ~done
            return (next_state, scan_rng, alive), output

        initial_alive = jnp.ones(start_state.obs.shape[0], dtype=bool)
        (_, _, _), rollout = jax.lax.scan(
            scan_step,
            (start_state, rng, initial_alive),
            None,
            length=steps,
        )
        return rollout

    results: dict[str, Any] = {}
    for suite_index, suite in enumerate(args.suites):
        controller_results: dict[str, Any] = {}
        for controller_index, controller in enumerate(_CONTROLLERS):
            rollout_rng = jax.random.fold_in(
                base_rng, suite_index * len(_CONTROLLERS) + controller_index
            )
            rollout = jax.device_get(
                run_controller(
                    initial_state,
                    rollout_rng,
                    controller=controller,
                    disable_pushes=(suite == "clean"),
                    steps=num_steps,
                )
            )
            controller_results[controller] = _summarize_rollout(
                rollout,
                ctrl_dt=ctrl_dt,
                settle_window_s=args.settle_window_s,
                settle_tilt_deg=args.settle_tilt_deg,
                settle_gyro_rad_s=args.settle_gyro_rad_s,
                settle_joint_max_deg=args.settle_joint_max_deg,
                settle_joint_rms_deg=args.settle_joint_rms_deg,
                initial_conditions=initial_conditions,
            )
        results[suite] = {
            "controllers": controller_results,
            "policy_vs_home": _paired_comparison(
                controller_results["policy"], controller_results["home"]
            ),
        }
        policy = controller_results["policy"]
        home = controller_results["home"]
        print(
            f"{suite}: policy pass={policy['pass']['count']}/{args.num_envs} "
            f"fall={policy['fall']['count']}/{args.num_envs}; "
            f"home pass={home['pass']['count']}/{args.num_envs} "
            f"fall={home['fall']['count']}/{args.num_envs}"
        )

    wr = initial_state.info[WR_INFO_KEY]
    push_force_xy = np.asarray(wr.push_schedule.force_xy)
    persistent_pitch_error = np.asarray(
        wr.domain_rand_persistent_torso_pitch_error_rad
    )
    output = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "ctrl_dt_s": ctrl_dt,
        "rollout_s": args.rollout_s,
        "training_time_limit_disabled": True,
        "suites": list(args.suites),
        "initial_envelope": {
            "max_tilt_deg": args.initial_max_tilt_deg,
            "max_gyro_rad_s": args.initial_max_gyro_rad_s,
            "foot_stagger_range_m": list(args.foot_stagger_range_m),
            "actual_tilt_deg_max": float(np.max(actual_initial_tilt_deg)),
            "actual_gyro_rad_s_max": float(np.max(actual_initial_gyro)),
            "actual_foot_stagger_m_min": float(np.min(actual_stagger)),
            "actual_foot_stagger_m_max": float(np.max(actual_stagger)),
        },
        "acceptance": {
            "tilt_deg_max": args.settle_tilt_deg,
            "gyro_rad_s_max": args.settle_gyro_rad_s,
            "joint_home_max_deg": args.settle_joint_max_deg,
            "joint_home_rms_deg": args.settle_joint_rms_deg,
            "continuous_window_s": args.settle_window_s,
            "both_feet_loaded": True,
            "first_episode_only": True,
        },
        "configured_push": {
            "enabled": bool(training_cfg.env.push_enabled),
            "force_min_n": float(training_cfg.env.push_force_min),
            "force_max_n": float(training_cfg.env.push_force_max),
            "duration_steps": int(training_cfg.env.push_duration_steps),
            "sample_force_n_min": float(np.min(np.linalg.norm(push_force_xy, axis=1))),
            "sample_force_n_max": float(np.max(np.linalg.norm(push_force_xy, axis=1))),
        },
        "persistent_torso_pitch_calibration": {
            "configured_range_rad": list(
                training_cfg.env.domain_rand_persistent_torso_pitch_error_range
            ),
            "configured_range_deg": list(
                np.rad2deg(
                    training_cfg.env.domain_rand_persistent_torso_pitch_error_range
                )
            ),
            "sample_rad_min": float(np.min(persistent_pitch_error)),
            "sample_rad_max": float(np.max(persistent_pitch_error)),
            "sample_abs_rad_max": float(np.max(np.abs(persistent_pitch_error))),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
