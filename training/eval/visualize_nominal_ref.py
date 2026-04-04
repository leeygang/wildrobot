#!/usr/bin/env python3
"""Nominal-only M2.5 reference viewer/debugger (q_target = q_ref)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.configs.training_config import TrainingConfig, load_training_config
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize nominal-only locomotion reference (q_target=q_ref)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking_v0193a.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--forward-cmd",
        type=float,
        default=0.10,
        help="Fixed forward command velocity (m/s)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=256,
        help="Max rollout steps",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive viewer",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional video output path (mp4/gif)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print diagnostics every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env reset",
    )
    parser.add_argument(
        "--stop-on-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop rollout as soon as done is reached",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _dominant_termination_from_metrics(metrics_vec: jnp.ndarray) -> str:
    term_order = (
        ("term/height_low", METRIC_INDEX["term/height_low"]),
        ("term/pitch", METRIC_INDEX["term/pitch"]),
        ("term/roll", METRIC_INDEX["term/roll"]),
    )
    vals = [(name, float(metrics_vec[idx])) for name, idx in term_order]
    best_name, best_val = max(vals, key=lambda x: x[1])
    if best_val <= 0.0:
        return "none"
    return best_name


def _configure_nominal_only(cfg: TrainingConfig, *, forward_cmd: float, horizon: int) -> None:
    """Force config into strict nominal-validation mode for M2.5 debugging."""
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = int(horizon)
    cfg.env.min_velocity = float(forward_cmd)
    cfg.env.max_velocity = float(forward_cmd)
    cfg.env.loc_ref_enabled = True
    cfg.env.controller_stack = "ppo"
    cfg.env.base_ctrl_enabled = False
    cfg.env.fsm_enabled = False
    cfg.env.push_enabled = False
    cfg.env.action_delay_steps = 0


def _sync_viewer_data(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, state_data) -> None:
    mj_data.qpos[:] = np.asarray(state_data.qpos, dtype=np.float64)
    mj_data.qvel[:] = np.asarray(state_data.qvel, dtype=np.float64)
    if mj_model.nu > 0:
        mj_data.ctrl[:] = np.asarray(state_data.ctrl, dtype=np.float64)
    mujoco.mj_forward(mj_model, mj_data)


def _format_step_line(step: int, state, metrics_vec: jnp.ndarray) -> str:
    wr = state.info[WR_INFO_KEY]
    return (
        f"step={step:04d} "
        f"cmd={float(wr.velocity_cmd):.3f} "
        f"v_fwd={float(metrics_vec[METRIC_INDEX['forward_velocity']]):+.3f} "
        f"pitch={float(metrics_vec[METRIC_INDEX['debug/loc_ref_root_pitch']]):+.3f} "
        f"pitch_rate={float(metrics_vec[METRIC_INDEX['debug/loc_ref_root_pitch_rate']]):+.3f} "
        f"mode={int(round(float(metrics_vec[METRIC_INDEX['debug/loc_ref_hybrid_mode_id']])))} "
        f"support={float(metrics_vec[METRIC_INDEX['debug/loc_ref_support_health']]):.3f} "
        f"perm={float(metrics_vec[METRIC_INDEX['debug/loc_ref_progression_permission']]):.3f} "
        f"swing_x_scale_active={float(metrics_vec[METRIC_INDEX['debug/loc_ref_swing_x_scale_active']]):.3f} "
        f"q_gap={float(metrics_vec[METRIC_INDEX['debug/loc_ref_nominal_vs_applied_q_l1']]):.5f}"
    )


def _save_video(record_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import mediapy as media

        media.write_video(record_path, frames, fps=fps)
        return
    except ModuleNotFoundError:
        pass
    import imageio

    imageio.mimsave(record_path, frames, fps=fps)


def run_nominal_viewer(args: argparse.Namespace) -> int:
    cfg = load_training_config(args.config)
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)
    _configure_nominal_only(cfg, forward_cmd=args.forward_cmd, horizon=args.horizon)
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(int(args.seed))
    state = env.reset(rng)
    zero_action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    step_once = jax.jit(lambda s: env.step(s, zero_action))

    mj_model = env._mj_model  # noqa: SLF001
    mj_data = mujoco.MjData(mj_model)
    _sync_viewer_data(mj_model, mj_data, state.data)

    print(
        f"[nominal-viewer] config={args.config} forward_cmd={args.forward_cmd:.3f} "
        f"horizon={args.horizon} mode={'headless' if args.headless else 'viewer'}"
    )
    print(
        "[nominal-viewer] residual policy action fixed to zeros -> q_target = q_ref "
        "(pushes disabled, action_delay_steps=0)"
    )

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(mj_model, 960, 720) if args.record else None
    term_totals = {"term/height_low": 0.0, "term/pitch": 0.0, "term/roll": 0.0}
    done_reached = False
    done_step = -1
    done_dom = "none"
    ctrl_dt = float(cfg.env.ctrl_dt)
    print_every = max(1, int(args.print_every))

    def run_step(step_idx: int) -> bool:
        nonlocal state, done_reached, done_step, done_dom
        state = step_once(state)
        _sync_viewer_data(mj_model, mj_data, state.data)
        metrics_vec = state.metrics[METRICS_VEC_KEY]
        for key in term_totals:
            term_totals[key] += float(metrics_vec[METRIC_INDEX[key]])
        done = bool(np.asarray(state.done > 0.5))
        if (step_idx % print_every == 0) or done or (step_idx == 1):
            print(_format_step_line(step_idx, state, metrics_vec))
        if renderer is not None:
            renderer.update_scene(mj_data)
            frames.append(renderer.render())
        if done:
            done_reached = True
            done_step = step_idx
            done_dom = _dominant_termination_from_metrics(metrics_vec)
            print(f"[nominal-viewer] done at step={done_step} dominant_termination={done_dom}")
            if args.stop_on_done:
                return False
        return True

    if args.headless:
        for step_idx in range(1, int(args.horizon) + 1):
            if not run_step(step_idx):
                break
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 135
            viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
            for step_idx in range(1, int(args.horizon) + 1):
                if not viewer.is_running():
                    break
                t0 = time.time()
                if not run_step(step_idx):
                    viewer.sync()
                    break
                viewer.sync()
                dt_sleep = (ctrl_dt / 1.0) - (time.time() - t0)
                if dt_sleep > 0.0:
                    time.sleep(dt_sleep)

    if renderer is not None:
        renderer.close()
        _save_video(args.record, frames, fps=max(1, int(round(1.0 / ctrl_dt))))
        print(f"[nominal-viewer] wrote video: {args.record} ({len(frames)} frames)")

    if done_reached:
        print(f"[nominal-viewer] summary: done_step={done_step} dominant_termination={done_dom}")
    else:
        dom = max(term_totals.items(), key=lambda x: x[1])[0] if max(term_totals.values()) > 0.0 else "none"
        print(
            f"[nominal-viewer] summary: done_step=none dominant_termination={dom} "
            f"(horizon={args.horizon})"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_nominal_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
