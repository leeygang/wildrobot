#!/usr/bin/env python3
"""Nominal-only locomotion-reference probe for v0.19.3b diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

import jax
import jax.numpy as jnp

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config, TrainingConfig
from training.core.metrics_registry import METRICS_VEC_KEY, METRIC_INDEX, METRIC_NAMES
from training.envs.wildrobot_env import WildRobotEnv


ABLATION_CHOICES = (
    "freeze-swing-x",
    "zero-pelvis-pitch",
    "no-dcm",
    "slow-phase",
    "tight-stance",
)

TRACE_FIELDS: tuple[tuple[str, str], ...] = (
    ("phase_progress", "tracking/loc_ref_phase_progress"),
    ("stance_foot", "tracking/loc_ref_stance_foot"),
    ("hybrid_mode_id", "debug/loc_ref_hybrid_mode_id"),
    ("progression_permission", "debug/loc_ref_progression_permission"),
    ("forward_velocity", "forward_velocity"),
    ("root_pitch", "debug/loc_ref_root_pitch"),
    ("root_pitch_rate", "debug/loc_ref_root_pitch_rate"),
    ("root_roll", "debug/roll"),
    ("root_height", "debug/root_height"),
    ("root_height_rate", "debug/root_height_rate"),
    ("lateral_velocity", "debug/lateral_vel"),
    ("step_length_m", "debug/step_length_m"),
    ("nominal_swing_x_target", "debug/loc_ref_swing_x_target"),
    ("foothold_x_raw", "debug/loc_ref_foothold_x_raw"),
    ("nominal_step_length", "debug/loc_ref_nominal_step_length"),
    ("actual_swing_x", "debug/loc_ref_swing_x_actual"),
    ("swing_x_error", "debug/loc_ref_swing_x_error"),
    ("nominal_pelvis_pitch_target", "debug/loc_ref_pelvis_pitch_target"),
    ("support_gate_active", "debug/loc_ref_support_gate_active"),
    ("support_health", "debug/loc_ref_support_health"),
    ("support_instability", "debug/loc_ref_support_instability"),
    ("swing_x_scale", "debug/loc_ref_swing_x_scale"),
    ("pelvis_pitch_scale", "debug/loc_ref_pelvis_pitch_scale"),
    ("nominal_vs_applied_q_gap", "debug/loc_ref_nominal_vs_applied_q_l1"),
)

COMPARE_KEYS: tuple[str, ...] = (
    "done_step",
    "dominant_termination",
    "forward_velocity_mean",
    "forward_velocity_last",
    "reward_m3_swing_foot_tracking",
    "reward_m3_foothold_consistency",
    "debug_loc_ref_nominal_vs_applied_q_l1_mean",
    "debug_loc_ref_support_gate_active_mean",
)


def _dominant_term(metrics_vec: jnp.ndarray, *, done_reached: bool) -> str:
    term_names = (
        ("term/height_low", METRIC_INDEX["term/height_low"]),
        ("term/pitch", METRIC_INDEX["term/pitch"]),
        ("term/roll", METRIC_INDEX["term/roll"]),
    )
    totals = [(name, float(jnp.sum(metrics_vec[..., idx]))) for name, idx in term_names]
    max_total = max(total for _, total in totals)
    if (not done_reached and max_total <= 0.0) or max_total <= 0.0:
        return "none"
    return max(totals, key=lambda x: x[1])[0]


def apply_nominal_ablation(
    cfg: TrainingConfig,
    *,
    ablation: str | None,
    slow_phase_multiplier: float = 2.0,
) -> None:
    """Apply one explicit channel-isolation ablation to probe config."""
    if ablation is None:
        return
    if ablation not in ABLATION_CHOICES:
        raise ValueError(f"Unknown ablation: {ablation}")

    env = cfg.env
    if ablation == "freeze-swing-x":
        env.loc_ref_max_swing_x_delta_m = 0.0
        env.loc_ref_swing_target_blend = 0.0
    elif ablation == "zero-pelvis-pitch":
        env.loc_ref_pelvis_pitch_gain = 0.0
        env.loc_ref_max_pelvis_pitch_rad = 0.0
    elif ablation == "no-dcm":
        env.loc_ref_dcm_placement_gain = 0.0
    elif ablation == "slow-phase":
        env.loc_ref_step_time_s = float(env.loc_ref_step_time_s) * float(
            max(1.0, slow_phase_multiplier)
        )
    elif ablation == "tight-stance":
        env.loc_ref_stance_extension_margin_m = 0.0
        env.loc_ref_stance_height_blend = 0.0
        env.loc_ref_max_swing_z_delta_m = min(float(env.loc_ref_max_swing_z_delta_m), 0.005)


def _build_probe_trace(
    metrics_vec: jnp.ndarray,
    *,
    done_idx: int,
    max_steps: int | None = None,
) -> Dict[str, Any]:
    """Build compact single-env per-step trace from rollout metrics."""
    steps = int(done_idx) + 1
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    trace_slice = metrics_vec[:steps, 0, :]
    trace: Dict[str, Any] = {"step": list(range(1, steps + 1))}
    for out_key, metric_key in TRACE_FIELDS:
        trace[out_key] = (
            jnp.asarray(trace_slice[:, METRIC_INDEX[metric_key]], dtype=jnp.float32)
            .tolist()
        )
    return trace


def _save_probe_trace(trace: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        arrays: Dict[str, np.ndarray] = {}
        for key, values in trace.items():
            if key == "step":
                arrays[key] = np.asarray(values, dtype=np.int32)
            else:
                arrays[key] = np.asarray(values, dtype=np.float32)
        np.savez(path, **arrays)
        return
    path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")


def compare_probe_summaries(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Compare key probe metrics for nominal gate decisions."""
    result: Dict[str, Dict[str, Any]] = {
        "baseline": {},
        "candidate": {},
        "delta": {},
    }
    for key in COMPARE_KEYS:
        b = baseline.get(key)
        c = candidate.get(key)
        result["baseline"][key] = b
        result["candidate"][key] = c
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            result["delta"][key] = float(c) - float(b)
        else:
            result["delta"][key] = None
    return result


def _print_probe_comparison(comparison: Dict[str, Dict[str, Any]]) -> None:
    print("Nominal probe comparison")
    for key in COMPARE_KEYS:
        b = comparison["baseline"].get(key)
        c = comparison["candidate"].get(key)
        d = comparison["delta"].get(key)
        if isinstance(d, float):
            print(f"- {key}: baseline={b} candidate={c} delta={d:+.6f}")
        else:
            print(f"- {key}: baseline={b} candidate={c}")


def run_nominal_probe(
    *,
    config_path: str,
    forward_cmd: float,
    horizon: int,
    seed: int = 0,
    residual_scale_override: float | None = 0.0,
    num_envs: int = 1,
    ablation: str | None = None,
    slow_phase_multiplier: float = 2.0,
    trace_output: str | None = None,
    trace_max_steps: int | None = None,
    stop_on_done: bool = True,
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
    apply_nominal_ablation(
        cfg,
        ablation=ablation,
        slow_phase_multiplier=slow_phase_multiplier,
    )
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(seed)
    reset_rngs = jax.random.split(rng, int(num_envs))
    env_state = jax.vmap(env.reset)(reset_rngs)
    zero_action = jnp.zeros((int(num_envs), env.action_size), dtype=jnp.float32)

    step_once = jax.jit(
        lambda s, a: jax.vmap(lambda ss, aa: env.step(ss, aa))(s, a)
    )
    state = env_state
    metrics_steps: list[jnp.ndarray] = []
    done_steps: list[jnp.ndarray] = []
    for _ in range(int(horizon)):
        state = step_once(state, zero_action)
        metrics_steps.append(state.metrics[METRICS_VEC_KEY])
        done_steps.append(state.done)
        if stop_on_done and bool(np.any(np.asarray(state.done > 0.5))):
            break

    metrics_vec_raw = jnp.stack(metrics_steps, axis=0)  # (T_eff, N, M)
    metrics_vec = jnp.nan_to_num(
        metrics_vec_raw, nan=0.0, posinf=0.0, neginf=0.0
    )
    dones = jnp.stack(done_steps, axis=0)  # (T_eff, N)

    done_any_t = jnp.any(dones > 0.5, axis=1)
    done_reached = bool(np.any(np.asarray(done_any_t)))
    done_idx = int(np.argmax(np.asarray(done_any_t))) if done_reached else int(metrics_vec.shape[0] - 1)
    done_step = done_idx + 1

    if trace_output is not None and int(num_envs) != 1:
        raise ValueError("Trace output currently supports only --num-envs 1")

    summary = {
        "config": str(config_path),
        "forward_cmd": float(forward_cmd),
        "horizon": int(horizon),
        "num_envs": int(num_envs),
        "ablation": str(ablation) if ablation is not None else "none",
        "steps_ran": int(metrics_vec.shape[0]),
        "done_reached": done_reached,
        "done_step": done_step,
        "dominant_termination": _dominant_term(metrics_vec, done_reached=done_reached),
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
        "tracking_loc_ref_mode_id_last": float(
            metrics_vec[done_idx, 0, METRIC_INDEX["tracking/loc_ref_mode_id"]]
        ),
        "tracking_loc_ref_mode_id_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_mode_id"]])
        ),
        "tracking_loc_ref_progression_permission_mean": float(
            jnp.mean(
                metrics_vec[..., METRIC_INDEX["tracking/loc_ref_progression_permission"]]
            )
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
        "debug_loc_ref_support_health_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_support_health"]])
        ),
        "debug_loc_ref_support_instability_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_support_instability"]])
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
        "debug_loc_ref_foothold_x_raw_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_foothold_x_raw"]])
        ),
        "debug_loc_ref_nominal_step_length_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_nominal_step_length"]])
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
        "debug_loc_ref_hybrid_mode_id_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_hybrid_mode_id"]])
        ),
        "debug_loc_ref_progression_permission_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_progression_permission"]])
        ),
        "debug_loc_ref_swing_x_scale_active_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_swing_x_scale_active"]])
        ),
        "debug_loc_ref_phase_scale_active_mean": float(
            jnp.mean(metrics_vec[..., METRIC_INDEX["debug/loc_ref_phase_scale_active"]])
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
        "debug_phase_progress_min": float(
            jnp.min(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_phase_progress"]])
        ),
        "debug_phase_progress_max": float(
            jnp.max(metrics_vec[..., METRIC_INDEX["tracking/loc_ref_phase_progress"]])
        ),
    }
    nonfinite_mask = ~jnp.isfinite(metrics_vec_raw)
    nonfinite_any_t = jnp.any(nonfinite_mask, axis=(1, 2))
    first_nonfinite_idx = jnp.where(nonfinite_any_t, size=1, fill_value=-1)[0][0]
    first_nonfinite_idx_i = int(first_nonfinite_idx)
    summary["debug_nonfinite_step"] = (
        first_nonfinite_idx_i + 1 if first_nonfinite_idx_i >= 0 else -1
    )
    summary["debug_nonfinite_count"] = int(jnp.sum(nonfinite_mask))
    if first_nonfinite_idx_i >= 0:
        first_mask = np.asarray(nonfinite_mask[first_nonfinite_idx_i, 0, :])
        summary["debug_nonfinite_metric_names_first_step"] = [
            METRIC_NAMES[i] for i, bad in enumerate(first_mask) if bool(bad)
        ]
    else:
        summary["debug_nonfinite_metric_names_first_step"] = []

    if trace_output is not None:
        trace = _build_probe_trace(
            metrics_vec,
            done_idx=int(done_idx),
            max_steps=trace_max_steps,
        )
        _save_probe_trace(trace, trace_output)
        summary["trace_output"] = str(trace_output)
        summary["trace_steps"] = int(len(trace["step"]))
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nominal-only loc-ref probe")
    # v0.20.1: ppo_walking_v0193a.yaml was deleted; smoke YAML lands
    # with task #49.
    p.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking_v0201_smoke.yaml",
    )
    p.add_argument("--forward-cmd", type=float, default=0.10)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual-scale", type=float, default=0.0)
    p.add_argument("--ablation", type=str, choices=ABLATION_CHOICES, default=None)
    p.add_argument("--slow-phase-multiplier", type=float, default=2.0)
    p.add_argument("--trace-output", type=str, default=None)
    p.add_argument("--trace-max-steps", type=int, default=None)
    p.add_argument(
        "--full-horizon",
        action="store_true",
        help="Run full horizon even after done (default stops at first done).",
    )
    p.add_argument("--compare-base", type=str, default=None)
    p.add_argument("--compare-new", type=str, default=None)
    p.add_argument("--compare-output-json", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from training.configs.cli_helpers import fail_if_config_missing
    # ``--compare-*`` paths bypass the config load, so only guard the
    # config when we'll actually consume it.
    if not (args.compare_base or args.compare_new):
        fail_if_config_missing(args.config)
    if (args.compare_base is None) ^ (args.compare_new is None):
        raise SystemExit("--compare-base and --compare-new must be provided together")
    if args.compare_base is not None and args.compare_new is not None:
        baseline = json.loads(Path(args.compare_base).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.compare_new).read_text(encoding="utf-8"))
        comparison = compare_probe_summaries(baseline, candidate)
        _print_probe_comparison(comparison)
        if args.compare_output_json:
            Path(args.compare_output_json).write_text(
                json.dumps(comparison, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return 0
    summary = run_nominal_probe(
        config_path=args.config,
        forward_cmd=args.forward_cmd,
        horizon=args.horizon,
        seed=args.seed,
        residual_scale_override=args.residual_scale,
        num_envs=args.num_envs,
        ablation=args.ablation,
        slow_phase_multiplier=args.slow_phase_multiplier,
        trace_output=args.trace_output,
        trace_max_steps=args.trace_max_steps,
        stop_on_done=not args.full_horizon,
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
