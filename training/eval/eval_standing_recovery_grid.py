#!/usr/bin/env python3
"""Evaluate standing recovery over pitch, pitch-rate, and foot-stagger grids.

Each initial state is cloned across three controllers:

* ``policy``: closed-loop deterministic checkpoint actions.
* ``home``: zero residual, which holds the configured home pose.
* ``frozen``: the checkpoint's first action repeated open-loop.

The paired runs separate a policy response from passive/home-pose dynamics and
report both the first response-window acceleration and the full recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from control.kinematics.leg_ik import (
    LegIkConfig,
    solve_leg_sagittal_ik_jax,
)
from policy_contract.jax import frames as jax_frames
from training.algos.ppo.ppo_core import create_networks, sample_actions
from training.configs.training_config import load_training_config
from training.core.checkpoint import load_checkpoint
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.cal.types import CoordinateFrame
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.eval_policy import _network_activation_name
from training.exports.export_onnx import get_checkpoint_dims


_CONTROLLERS = ("policy", "home", "frozen")
_LEG_IK_CONFIG = LegIkConfig()


@dataclass(frozen=True)
class RecoveryCondition:
    pitch_deg: float
    pitch_rate_rad_s: float
    foot_stagger_m: float


def _parse_csv_floats(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value")
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Grid values must be finite")
    return values


def _conditions(
    pitch_deg: Iterable[float],
    pitch_rate_rad_s: Iterable[float],
    foot_stagger_m: Iterable[float],
) -> list[RecoveryCondition]:
    return [
        RecoveryCondition(float(pitch), float(rate), float(stagger))
        for pitch in pitch_deg
        for rate in pitch_rate_rad_s
        for stagger in foot_stagger_m
    ]


def _pitch_from_quat_wxyz(quat: jax.Array) -> jax.Array:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sin_pitch = jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return jnp.arcsin(sin_pitch).astype(jnp.float32)


def _roll_from_quat_wxyz(quat: jax.Array) -> jax.Array:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return jnp.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    ).astype(jnp.float32)


def _stagger_leg_pitch_values(
    leg_q: jax.Array, foot_stagger_m: jax.Array
) -> jax.Array:
    """Return leg-pitch qpos with ``left_x - right_x == foot_stagger_m``.

    ``leg_q`` order is left hip/knee/ankle then right hip/knee/ankle. WR's
    left hip axis is mirrored, so it is negated before and after sagittal IK.
    Both feet retain their mean height and mean pitch.
    """
    leg_q = jnp.asarray(leg_q, dtype=jnp.float32).reshape(6)
    left_hip = -leg_q[0]
    right_hip = leg_q[3]
    left_x, left_z = _forward_leg_sagittal_jax(left_hip, leg_q[1])
    right_x, right_z = _forward_leg_sagittal_jax(right_hip, leg_q[4])
    center_x = 0.5 * (left_x + right_x)
    common_z = 0.5 * (left_z + right_z)
    common_foot_pitch = 0.5 * (
        left_hip + leg_q[1] + leg_q[2]
        + right_hip + leg_q[4] + leg_q[5]
    )

    # The sagittal IK convention's +x maps to -world-x in the assembled WR
    # model. Negate here so the CLI convention remains physical/world based.
    half_stagger = -0.5 * jnp.asarray(foot_stagger_m, dtype=jnp.float32)
    left = solve_leg_sagittal_ik_jax(
        target_x_m=center_x + half_stagger,
        target_z_m=common_z,
        config=_LEG_IK_CONFIG,
    )
    right = solve_leg_sagittal_ik_jax(
        target_x_m=center_x - half_stagger,
        target_z_m=common_z,
        config=_LEG_IK_CONFIG,
    )
    left_ankle = left[2] + common_foot_pitch
    right_ankle = right[2] + common_foot_pitch
    return jnp.stack(
        [-left[0], left[1], left_ankle, right[0], right[1], right_ankle]
    ).astype(jnp.float32)


def _forward_leg_sagittal_jax(
    hip_pitch_rad: jax.Array, knee_pitch_rad: jax.Array
) -> tuple[jax.Array, jax.Array]:
    l1 = jnp.float32(_LEG_IK_CONFIG.upper_leg_length_m)
    l2 = jnp.float32(_LEG_IK_CONFIG.lower_leg_length_m)
    x = l1 * jnp.sin(hip_pitch_rad) + l2 * jnp.sin(
        hip_pitch_rad + knee_pitch_rad
    )
    z = -l1 * jnp.cos(hip_pitch_rad) - l2 * jnp.cos(
        hip_pitch_rad + knee_pitch_rad
    )
    return x.astype(jnp.float32), z.astype(jnp.float32)


def _apply_recovery_condition(
    env: WildRobotEnv,
    base_state,
    *,
    pitch_rad: jax.Array,
    pitch_rate_rad_s: jax.Array,
    foot_stagger_m: jax.Array,
    roll_rad: jax.Array | float = 0.0,
    roll_rate_rad_s: jax.Array | float = 0.0,
):
    """Rebuild one reset state with a prescribed standing condition."""
    wr = base_state.info[WR_INFO_KEY]
    qpos = base_state.data.qpos

    # Match the training reset's structured pitch perturbation, but prescribe
    # the final world pitch. The home keyframe itself is not exactly zero pitch,
    # so applying ``pitch_rad`` as a delta would initialize the wrong state.
    rng_pitch, rng_hip, rng_knee, rng_imu, rng_joint_feedback = jax.random.split(
        base_state.rng, 5
    )
    del rng_pitch  # Pitch is prescribed rather than sampled for this evaluator.
    base_pitch = _pitch_from_quat_wxyz(qpos[3:7])
    base_roll = _roll_from_quat_wxyz(qpos[3:7])
    pitch_delta = jnp.asarray(pitch_rad, dtype=jnp.float32) - base_pitch
    roll_delta = jnp.asarray(roll_rad, dtype=jnp.float32) - base_roll
    pitch_abs = jnp.abs(pitch_delta)
    hip_delta = jax.random.uniform(rng_hip, (), minval=0.0, maxval=pitch_abs)
    knee_delta = jax.random.uniform(
        rng_knee, (), minval=0.0, maxval=jnp.maximum(pitch_abs - hip_delta, 0.0)
    )
    ankle_delta = jnp.maximum(pitch_abs - hip_delta - knee_delta, 0.0)
    leg_delta = jnp.stack(
        [hip_delta, knee_delta, ankle_delta, hip_delta, knee_delta, ankle_delta]
    )
    leg_delta = leg_delta * env._leg_pitch_joint_signs * jnp.sign(pitch_delta)
    pitched_leg_q = jnp.clip(
        qpos[env._leg_pitch_qpos_addrs] + leg_delta,
        env._leg_pitch_joint_mins,
        env._leg_pitch_joint_maxs,
    )
    qpos = qpos.at[env._leg_pitch_qpos_addrs].set(pitched_leg_q)
    delta_quat = env._euler_xyz_to_quat_wxyz(
        roll_delta, pitch_delta, jnp.float32(0.0)
    )
    root_quat = env._quat_mul_wxyz(delta_quat, qpos[3:7])
    qpos = qpos.at[3:7].set(jax_frames.normalize_quat_wxyz(root_quat))

    # Solve the stagger after applying the torso perturbation. Doing this in
    # the opposite order lets the randomized hip/knee/ankle pitch partition
    # overwrite the requested foot relationship.
    pitched_qpos = qpos

    def candidate_for_ik_stagger(ik_stagger):
        leg_q = _stagger_leg_pitch_values(
            pitched_qpos[env._leg_pitch_qpos_addrs], ik_stagger
        )
        leg_q = jnp.clip(
            leg_q,
            env._leg_pitch_joint_mins,
            env._leg_pitch_joint_maxs,
        )
        candidate_qpos = pitched_qpos.at[env._leg_pitch_qpos_addrs].set(leg_q)
        probe_data = mjx.make_data(env._mjx_model).replace(
            qpos=candidate_qpos,
            qvel=jnp.zeros(env.mj_model.nv, dtype=jnp.float32),
            ctrl=base_state.data.ctrl,
        )
        probe_data = mjx.forward(env._mjx_model, probe_data)
        left_foot, right_foot = env._cal.get_foot_positions(
            probe_data,
            normalize=False,
            frame=CoordinateFrame.WORLD,
        )
        return candidate_qpos, left_foot[0] - right_foot[0]

    requested_stagger = jnp.asarray(foot_stagger_m, dtype=jnp.float32)

    def refine_stagger(_, ik_stagger):
        _, actual_stagger = candidate_for_ik_stagger(ik_stagger)
        return ik_stagger + requested_stagger - actual_stagger

    ik_stagger = jax.lax.fori_loop(0, 3, refine_stagger, requested_stagger)
    qpos, _ = candidate_for_ik_stagger(ik_stagger)

    qvel = jnp.zeros(env.mj_model.nv, dtype=jnp.float32)
    qvel = qvel.at[3].set(jnp.asarray(roll_rate_rad_s, dtype=jnp.float32))
    qvel = qvel.at[4].set(jnp.asarray(pitch_rate_rad_s, dtype=jnp.float32))
    dr_params = {
        "friction_scale": wr.domain_rand_friction_scale,
        "mass_scales": wr.domain_rand_mass_scales,
        "kp_scales": wr.domain_rand_kp_scales,
        "frictionloss_scales": wr.domain_rand_frictionloss_scales,
        "joint_offsets": wr.domain_rand_joint_offsets,
        "backlash": wr.domain_rand_backlash,
    }
    return env._make_initial_state(
        rng=base_state.rng,
        qpos=qpos,
        velocity_cmd=jnp.zeros(3, dtype=jnp.float32),
        push_schedule=wr.push_schedule,
        dr_params=dr_params,
        imu_init_rng=rng_imu,
        cmd_rng=wr.cmd_rng,
        joint_feedback_rng=rng_joint_feedback,
        rsi_qvel=qvel,
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _rate_summary(values) -> dict[str, float]:
    values = np.asarray([bool(value) for value in values if value is not None], dtype=bool)
    successes = int(np.sum(values))
    low, high = _wilson_interval(successes, int(values.size))
    return {
        "rate": float(np.mean(values)) if values.size else 0.0,
        "count": successes,
        "total": int(values.size),
        "wilson95_low": low,
        "wilson95_high": high,
    }


def _summarize_rollout(
    rollout: dict[str, np.ndarray],
    *,
    initial_pitch_rad: np.ndarray,
    initial_pitch_rate_rad_s: np.ndarray,
    ctrl_dt: float,
    response_window_s: float,
) -> dict[str, Any]:
    active = np.asarray(rollout["active"], dtype=bool)
    done = np.asarray(rollout["done"], dtype=bool)
    pitch = np.asarray(rollout["pitch"], dtype=np.float64)
    rate = np.asarray(rollout["pitch_rate"], dtype=np.float64)
    both_loaded = np.asarray(rollout["both_loaded"], dtype=np.float64)
    torque_max = np.asarray(rollout["torque_abs_max"], dtype=np.float64)
    action_abs_max = np.asarray(rollout["action_abs_max"], dtype=np.float64)
    zeros = np.zeros_like(pitch)
    recovery_phase = np.asarray(rollout.get("recovery_phase", zeros), dtype=np.float64)
    recovery_steps = np.asarray(
        rollout.get("recovery_step_count", zeros), dtype=np.float64
    )
    recovery_touchdown = np.asarray(
        rollout.get("recovery_touchdown", zeros), dtype=np.float64
    )
    squat_recovered = np.asarray(
        rollout.get("squat_recovered", zeros), dtype=np.float64
    )
    unnecessary_liftoff = np.asarray(
        rollout.get("unnecessary_liftoff", zeros), dtype=np.float64
    )
    n_steps, n_seeds = pitch.shape
    response_step = min(
        n_steps - 1, max(0, int(round(response_window_s / ctrl_dt)) - 1)
    )
    final_window_steps = min(n_steps, max(1, int(round(0.5 / ctrl_dt))))
    seed_results: list[dict[str, Any]] = []

    for seed_idx in range(n_seeds):
        valid_indices = np.flatnonzero(active[:, seed_idx])
        last_idx = int(valid_indices[-1]) if valid_indices.size else 0
        response_idx = min(response_step, last_idx)
        response_time = (response_idx + 1) * ctrl_dt
        theta0 = float(initial_pitch_rad[seed_idx])
        omega0 = float(initial_pitch_rate_rad_s[seed_idx])
        omega_response = float(rate[response_idx, seed_idx])
        avg_accel = (omega_response - omega0) / max(response_time, 1e-9)
        phase_error = theta0 + response_window_s * omega0
        direction = float(np.sign(phase_error))
        corrective = None if direction == 0.0 else bool(direction * avg_accel < 0.0)
        valid_pitch = pitch[: last_idx + 1, seed_idx]
        valid_rate = rate[: last_idx + 1, seed_idx]
        valid_both = both_loaded[: last_idx + 1, seed_idx]
        fell = bool(np.any(done[:, seed_idx]))
        final_start = max(0, last_idx + 1 - final_window_steps)
        recovered = bool(
            not fell
            and np.all(np.abs(pitch[final_start : last_idx + 1, seed_idx]) < np.deg2rad(3.0))
            and np.all(np.abs(rate[final_start : last_idx + 1, seed_idx]) < 0.1)
        )
        stepped = bool(np.max(recovery_steps[: last_idx + 1, seed_idx]) > 0.5)
        touched_down = bool(
            np.sum(recovery_touchdown[: last_idx + 1, seed_idx]) > 0.5
        )
        returned_to_squat = bool(
            np.sum(squat_recovered[: last_idx + 1, seed_idx]) > 0.5
        )
        seed_results.append(
            {
                "initial_pitch_rad": theta0,
                "initial_pitch_rate_rad_s": omega0,
                "phase_error_direction": direction,
                "response_pitch_rate_rad_s": omega_response,
                "response_avg_pitch_accel_rad_s2": avg_accel,
                "corrective_response": corrective,
                "recovered": recovered,
                "stepped": stepped,
                "touched_down": touched_down,
                "returned_to_squat": returned_to_squat,
                "step_recovery_succeeded": bool(
                    stepped and touched_down and returned_to_squat and recovered
                ),
                "unnecessary_liftoff": bool(
                    np.sum(unnecessary_liftoff[: last_idx + 1, seed_idx]) > 0.5
                ),
                "final_recovery_phase": int(recovery_phase[last_idx, seed_idx]),
                "fell": fell,
                "hardware_tilt_limit_exceeded": bool(
                    np.any(np.abs(valid_pitch) > np.deg2rad(10.0))
                ),
                "contact_lost": bool(np.any(valid_both < 0.5)),
                "peak_abs_pitch_deg": float(np.rad2deg(np.max(np.abs(valid_pitch)))),
                "peak_abs_pitch_rate_rad_s": float(np.max(np.abs(valid_rate))),
                "both_loaded_fraction": float(np.mean(valid_both)),
                "peak_torque_fraction": float(
                    np.max(torque_max[: last_idx + 1, seed_idx])
                ),
                "peak_action_abs": float(
                    np.max(action_abs_max[: last_idx + 1, seed_idx])
                ),
            }
        )

    def values(key: str, dtype=float) -> np.ndarray:
        return np.asarray([item[key] for item in seed_results], dtype=dtype)

    return {
        "response_avg_pitch_accel_rad_s2_mean": float(
            np.mean(values("response_avg_pitch_accel_rad_s2"))
        ),
        "response_pitch_rate_rad_s_mean": float(
            np.mean(values("response_pitch_rate_rad_s"))
        ),
        "corrective_response": _rate_summary(
            [item["corrective_response"] for item in seed_results]
        ),
        "recovery": _rate_summary(values("recovered", bool)),
        "stepped": _rate_summary(values("stepped", bool)),
        "touchdown": _rate_summary(values("touched_down", bool)),
        "returned_to_squat": _rate_summary(values("returned_to_squat", bool)),
        "step_recovery_success": _rate_summary(
            values("step_recovery_succeeded", bool)
        ),
        "unnecessary_liftoff": _rate_summary(
            values("unnecessary_liftoff", bool)
        ),
        "fall": _rate_summary(values("fell", bool)),
        "hardware_tilt_limit_exceeded": _rate_summary(
            values("hardware_tilt_limit_exceeded", bool)
        ),
        "contact_lost": _rate_summary(values("contact_lost", bool)),
        "peak_abs_pitch_deg_mean": float(np.mean(values("peak_abs_pitch_deg"))),
        "peak_abs_pitch_deg_max": float(np.max(values("peak_abs_pitch_deg"))),
        "peak_abs_pitch_rate_rad_s_mean": float(
            np.mean(values("peak_abs_pitch_rate_rad_s"))
        ),
        "both_loaded_fraction_mean": float(np.mean(values("both_loaded_fraction"))),
        "peak_torque_fraction_mean": float(np.mean(values("peak_torque_fraction"))),
        "peak_action_abs_mean": float(np.mean(values("peak_action_abs"))),
        "per_seed": seed_results,
    }


def _paired_comparison(policy: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    policy_seeds = policy["per_seed"]
    baseline_seeds = baseline["per_seed"]
    better = []
    for policy_seed, baseline_seed in zip(policy_seeds, baseline_seeds, strict=True):
        direction = float(policy_seed["phase_error_direction"])
        policy_accel = float(policy_seed["response_avg_pitch_accel_rad_s2"])
        baseline_accel = float(baseline_seed["response_avg_pitch_accel_rad_s2"])
        better.append(
            None
            if direction == 0.0
            else direction * (policy_accel - baseline_accel) < 0.0
        )
    return {"stronger_corrective_response": _rate_summary(better)}


def _foot_support_length_m(model: mujoco.MjModel) -> float | None:
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    bounds: list[tuple[float, float]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id or not int(model.geom_contype[geom_id]):
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        half_x = float(np.sum(np.abs(rotation[0]) * model.geom_size[geom_id]))
        center_x = float(data.geom_xpos[geom_id, 0])
        bounds.append((center_x - half_x, center_x + half_x))
    if not bounds:
        return None
    return max(high for _, high in bounds) - min(low for low, _ in bounds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standing recovery over pitch/rate/foot-stagger states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pitch-deg",
        type=_parse_csv_floats,
        default=_parse_csv_floats("-10,-7.5,-5,-2.5,0,2.5,5,7.5,10"),
    )
    parser.add_argument(
        "--pitch-rate-rad-s",
        type=_parse_csv_floats,
        default=_parse_csv_floats("-0.6,-0.3,0,0.3,0.6"),
    )
    parser.add_argument(
        "--foot-stagger-m",
        type=_parse_csv_floats,
        default=_parse_csv_floats("-0.04,0,0.04"),
        help="Positive means the left foot is ahead of the right foot",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--response-window-ms", type=float, default=300.0)
    parser.add_argument("--rollout-s", type=float, default=5.0)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.platform:
        jax.config.update("jax_platform_name", args.platform)
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be > 0")
    if args.response_window_ms <= 0.0 or args.rollout_s <= 0.0:
        raise ValueError("Response window and rollout duration must be > 0")
    response_window_s = args.response_window_ms / 1000.0
    if response_window_s > args.rollout_s:
        raise ValueError("Response window cannot exceed rollout duration")
    if not args.checkpoint.is_file() or not args.config.is_file():
        raise FileNotFoundError("Checkpoint and config must both be files")

    training_cfg = load_training_config(args.config)
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
    env_obs_dim = int(env._policy_spec.model.obs_dim)
    if obs_dim != env_obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim={obs_dim} does not match env obs_dim={env_obs_dim}"
        )
    if action_dim != env.action_size:
        raise ValueError(
            f"Checkpoint action_dim={action_dim} does not match env action_size={env.action_size}"
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
    batch_step = jax.vmap(
        lambda state, action: env.step(
            state,
            action,
            disable_cmd_resample=True,
            disable_pushes=True,
        )
    )

    @jax.jit
    def reset_batch(rngs, pitch_rad, pitch_rate_rad_s, foot_stagger_m):
        def reset_one(rng):
            base = env.reset_for_eval(
                rng,
                cmd_override=jnp.zeros(3, dtype=jnp.float32),
                perturb_pose=False,
            )
            return _apply_recovery_condition(
                env,
                base,
                pitch_rad=pitch_rad,
                pitch_rate_rad_s=pitch_rate_rad_s,
                foot_stagger_m=foot_stagger_m,
            )

        return jax.vmap(reset_one)(rngs)

    @partial(jax.jit, static_argnames=("controller", "steps"))
    def run_controller(initial_state, rng, *, controller: str, steps: int):
        rng, first_action_rng = jax.random.split(rng)
        first_action, _, _ = sample_actions(
            processor_params,
            policy_params,
            ppo_network,
            initial_state.obs,
            first_action_rng,
            deterministic=True,
        )
        frozen_action = first_action

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
            elif controller == "home":
                action = jnp.zeros_like(frozen_action)
            else:
                action = frozen_action
            next_state = batch_step(state, action)
            metrics = next_state.metrics[METRICS_VEC_KEY]
            output = {
                "active": alive,
                "done": next_state.done > 0.5,
                "pitch": metrics[:, METRIC_INDEX["term/pitch_val"]],
                "pitch_rate": metrics[:, METRIC_INDEX["debug/pitch_rate"]],
                "both_loaded": metrics[:, METRIC_INDEX["support/both_loaded"]],
                "torque_abs_max": metrics[:, METRIC_INDEX["debug/torque_abs_max"]],
                "action_abs_max": jnp.max(jnp.abs(action), axis=-1),
                "recovery_phase": metrics[:, METRIC_INDEX["recovery/phase"]],
                "recovery_step_count": metrics[
                    :, METRIC_INDEX["recovery/step_count"]
                ],
                "recovery_touchdown": metrics[
                    :, METRIC_INDEX["recovery/touchdown_event"]
                ],
                "squat_recovered": metrics[
                    :, METRIC_INDEX["recovery/squat_recovered_event"]
                ],
                "unnecessary_liftoff": metrics[
                    :, METRIC_INDEX["recovery/unnecessary_liftoff_event"]
                ],
            }
            alive = alive & ~(next_state.done > 0.5)
            return (next_state, scan_rng, alive), output

        initial_alive = jnp.ones(initial_state.obs.shape[0], dtype=bool)
        (_, _, _), rollout = jax.lax.scan(
            scan_step,
            (initial_state, rng, initial_alive),
            None,
            length=steps,
        )
        return rollout

    conditions = _conditions(args.pitch_deg, args.pitch_rate_rad_s, args.foot_stagger_m)
    base_rng = jax.random.PRNGKey(args.seed)
    # Reuse the same per-seed dynamics samples for every grid condition so
    # differences across pitch/rate/stagger values are paired, not sampling noise.
    seed_rngs = jax.random.split(base_rng, args.num_seeds)
    results: list[dict[str, Any]] = []
    print(
        f"Recovery grid: {len(conditions)} conditions x {args.num_seeds} seeds "
        f"x {len(_CONTROLLERS)} controllers, {num_steps} steps; "
        f"policy obs/action={obs_dim}/{action_dim}, model actuators={env.mj_model.nu}"
    )

    for condition_index, condition in enumerate(conditions):
        initial_state = reset_batch(
            seed_rngs,
            jnp.float32(np.deg2rad(condition.pitch_deg)),
            jnp.float32(condition.pitch_rate_rad_s),
            jnp.float32(condition.foot_stagger_m),
        )
        initial_pitch = np.asarray(_pitch_from_quat_wxyz(initial_state.data.qpos[:, 3:7]))
        initial_rate = np.asarray(initial_state.data.qvel[:, 4])
        initial_left_foot, initial_right_foot = jax.vmap(
            lambda data: env._cal.get_foot_positions(
                data,
                normalize=False,
                frame=CoordinateFrame.WORLD,
            )
        )(initial_state.data)
        actual_stagger = np.asarray(initial_left_foot[:, 0] - initial_right_foot[:, 0])
        controller_results: dict[str, Any] = {}
        for controller_index, controller in enumerate(_CONTROLLERS):
            controller_rng = jax.random.fold_in(
                base_rng, condition_index * len(_CONTROLLERS) + controller_index
            )
            rollout = jax.device_get(
                run_controller(
                    initial_state,
                    controller_rng,
                    controller=controller,
                    steps=num_steps,
                )
            )
            controller_results[controller] = _summarize_rollout(
                rollout,
                initial_pitch_rad=initial_pitch,
                initial_pitch_rate_rad_s=initial_rate,
                ctrl_dt=ctrl_dt,
                response_window_s=response_window_s,
            )
        comparison = {
            "policy_vs_home": _paired_comparison(
                controller_results["policy"], controller_results["home"]
            ),
            "policy_vs_frozen": _paired_comparison(
                controller_results["policy"], controller_results["frozen"]
            ),
        }
        result = {
            "pitch_deg": condition.pitch_deg,
            "pitch_rate_rad_s": condition.pitch_rate_rad_s,
            "foot_stagger_m": condition.foot_stagger_m,
            "initial_state": {
                "actual_pitch_deg_mean": float(np.rad2deg(np.mean(initial_pitch))),
                "actual_pitch_deg_min": float(np.rad2deg(np.min(initial_pitch))),
                "actual_pitch_deg_max": float(np.rad2deg(np.max(initial_pitch))),
                "actual_pitch_rate_rad_s_mean": float(np.mean(initial_rate)),
                "actual_foot_stagger_m_mean": float(np.mean(actual_stagger)),
                "actual_foot_stagger_m_min": float(np.min(actual_stagger)),
                "actual_foot_stagger_m_max": float(np.max(actual_stagger)),
            },
            "controllers": controller_results,
            "comparisons": comparison,
        }
        results.append(result)
        policy = controller_results["policy"]
        corrective_summary = policy["corrective_response"]
        better_summary = comparison["policy_vs_home"]["stronger_corrective_response"]
        corrective_text = (
            "n/a" if corrective_summary["total"] == 0 else f"{corrective_summary['rate']:.0%}"
        )
        better_text = (
            "n/a" if better_summary["total"] == 0 else f"{better_summary['rate']:.0%}"
        )
        print(
            f"  pitch={condition.pitch_deg:+5.1f}deg "
            f"rate={condition.pitch_rate_rad_s:+.2f}rad/s "
            f"stagger={condition.foot_stagger_m:+.3f}m | "
            f"corrective={corrective_text} "
            f"better_home={better_text} recovery={policy['recovery']['rate']:.0%} "
            f"fall={policy['fall']['rate']:.0%}"
        )

    foot_length = _foot_support_length_m(env.mj_model)
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "ctrl_dt_s": ctrl_dt,
        "response_window_s": response_window_s,
        "rollout_s": args.rollout_s,
        "foot_support_length_m": foot_length,
        "stagger_sign_convention": "positive means left foot ahead of right foot",
        "controllers": list(_CONTROLLERS),
        "conditions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
