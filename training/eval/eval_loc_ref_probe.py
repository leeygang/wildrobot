#!/usr/bin/env python3
"""Nominal-only locomotion-reference probe for v0.19.3b diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config, TrainingConfig
from training.core.metrics_registry import METRICS_VEC_KEY, METRIC_INDEX
from training.envs.wildrobot_env import WildRobotEnv


def _dominant_term(metrics_vec: jnp.ndarray) -> str:
    term_names = (
        ("term/height_low", METRIC_INDEX["term/height_low"]),
        ("term/pitch", METRIC_INDEX["term/pitch"]),
        ("term/roll", METRIC_INDEX["term/roll"]),
    )
    totals = [(name, float(jnp.sum(metrics_vec[..., idx]))) for name, idx in term_names]
    return max(totals, key=lambda x: x[1])[0]


def run_nominal_probe(
    *,
    config_path: str,
    forward_cmd: float,
    horizon: int,
    seed: int = 0,
    residual_scale_override: float | None = 0.0,
    num_envs: int = 1,
) -> Dict[str, Any]:
    cfg: TrainingConfig = load_training_config(config_path)
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)

    cfg.ppo.num_envs = int(num_envs)
    cfg.ppo.rollout_steps = int(horizon)
    cfg.env.min_velocity = float(forward_cmd)
    cfg.env.max_velocity = float(forward_cmd)
    cfg.env.loc_ref_enabled = True
    if residual_scale_override is not None:
        cfg.env.loc_ref_residual_scale = float(residual_scale_override)
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(seed)
    reset_rngs = jax.random.split(rng, int(num_envs))
    env_state = jax.vmap(env.reset)(reset_rngs)
    zero_action = jnp.zeros((int(num_envs), env.action_size), dtype=jnp.float32)

    def _step(carry, _):
        s = carry
        ns = jax.vmap(lambda ss, aa: env.step(ss, aa))(s, zero_action)
        return ns, ns

    _, rollout_states = jax.lax.scan(_step, env_state, None, length=int(horizon))
    metrics_vec = rollout_states.metrics[METRICS_VEC_KEY]  # (T, N, M)

    done_any_t = jnp.any(rollout_states.done > 0.5, axis=1)
    done_idx = jnp.where(done_any_t, size=1, fill_value=int(horizon) - 1)[0][0]
    done_step = int(done_idx) + 1

    summary = {
        "config": str(config_path),
        "forward_cmd": float(forward_cmd),
        "horizon": int(horizon),
        "num_envs": int(num_envs),
        "done_step": done_step,
        "dominant_termination": _dominant_term(metrics_vec),
        "forward_velocity_mean": float(jnp.mean(metrics_vec[..., METRIC_INDEX["forward_velocity"]])),
        "forward_velocity_last": float(metrics_vec[done_idx, 0, METRIC_INDEX["forward_velocity"]]),
        "tracking_nominal_q_abs_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["tracking/nominal_q_abs_mean"]])
        ),
        "tracking_loc_ref_left_reachable": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_left_reachable"]])
        ),
        "tracking_loc_ref_right_reachable": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_right_reachable"]])
        ),
        "tracking_loc_ref_phase_progress_last": float(
            metrics_vec[done_idx, 0, METRIC_INDEX["tracking/loc_ref_phase_progress"]]
        ),
        "tracking_loc_ref_phase_progress_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_phase_progress"]])
        ),
        "tracking_loc_ref_phase_progress_std": float(
            jnp.std(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_phase_progress"]])
        ),
        "tracking_loc_ref_stance_foot_last": float(
            metrics_vec[done_idx, 0, METRIC_INDEX["tracking/loc_ref_stance_foot"]]
        ),
        "debug_loc_ref_speed_scale_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_speed_scale"]])
        ),
        "debug_loc_ref_phase_scale_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_phase_scale"]])
        ),
        "debug_loc_ref_overspeed_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_overspeed"]])
        ),
        "debug_loc_ref_nominal_vs_applied_q_l1_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_nominal_vs_applied_q_l1"]])
        ),
        "debug_loc_ref_applied_q_abs_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_applied_q_abs_mean"]])
        ),
        "debug_loc_ref_nominal_q_abs_mean_debug": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_nominal_q_abs_mean"]])
        ),
        "debug_loc_ref_swing_x_target_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_swing_x_target"]])
        ),
        "debug_loc_ref_swing_x_actual_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_swing_x_actual"]])
        ),
        "debug_loc_ref_swing_x_error_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_swing_x_error"]])
        ),
        "debug_loc_ref_pelvis_pitch_target_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_pelvis_pitch_target"]])
        ),
        "debug_loc_ref_root_pitch_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_root_pitch"]])
        ),
        "debug_loc_ref_root_pitch_rate_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_root_pitch_rate"]])
        ),
        "debug_loc_ref_swing_x_scale_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_swing_x_scale"]])
        ),
        "debug_loc_ref_pelvis_pitch_scale_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_pelvis_pitch_scale"]])
        ),
        "debug_loc_ref_support_gate_active_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_support_gate_active"]])
        ),
        "debug_m3_swing_pos_error": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/m3_swing_pos_error"]])
        ),
        "debug_m3_swing_vel_error": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/m3_swing_vel_error"]])
        ),
        "debug_m3_foothold_error": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/m3_foothold_error"]])
        ),
        "reward_m3_swing_foot_tracking": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["reward/m3_swing_foot_tracking"]])
        ),
        "reward_m3_foothold_consistency": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["reward/m3_foothold_consistency"]])
        ),
        "term_pitch_frac": float(jnp.mean(metrics_vec[..., METRIC_INDEX["term/pitch"]])),
        "term_height_low_frac": float(jnp.mean(metrics_vec[..., METRIC_INDEX["term/height_low"]])),
        "term_roll_frac": float(jnp.mean(metrics_vec[..., METRIC_INDEX["term/roll"]])),
    }
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nominal-only loc-ref probe")
    p.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking_v0193a.yaml",
    )
    p.add_argument("--forward-cmd", type=float, default=0.10)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=0.0)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_nominal_probe(
        config_path=args.config,
        forward_cmd=args.forward_cmd,
        horizon=args.horizon,
        seed=args.seed,
        residual_scale_override=args.residual_scale,
        num_envs=args.num_envs,
    )
    print("Nominal-only loc-ref probe summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
