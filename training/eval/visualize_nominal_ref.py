#!/usr/bin/env python3
"""Nominal-only M2.5 reference viewer/debugger (q_target = q_ref)."""

from __future__ import annotations

import argparse
import atexit
import io
import shlex
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.cal.cal import CoordinateFrame
from training.configs.training_config import TrainingConfig, load_training_config
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv


class _TeeTextIO(io.TextIOBase):
    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


def _enable_temp_logging() -> Path:
    log_path = Path(tempfile.gettempdir()) / (
        f"wildrobot_nominal_ref_{int(time.time())}.log"
    )
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeTextIO(orig_stdout, log_file)
    sys.stderr = _TeeTextIO(orig_stderr, log_file)

    def _cleanup() -> None:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.flush()
        log_file.close()

    atexit.register(_cleanup)
    print(f"[nominal-viewer] log file: {log_path}")
    print(f"[nominal-viewer] command: {shlex.join(sys.argv)}")
    return log_path


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
    parser.add_argument(
        "--force-support-only",
        action="store_true",
        help="Force walking_ref_v2 to remain in SUPPORT_STABILIZE for posture diagnosis",
    )
    parser.add_argument(
        "--init-from-nominal-qref",
        action="store_true",
        help="Replace reset joints with nominal_q_ref and ground stance foot before visualization",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write the invoked command and viewer console output to a temp log file",
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


def _replace_state_with_nominal_qref(env: WildRobotEnv, state):
    wr = state.info[WR_INFO_KEY]
    nominal_q_ref = jnp.asarray(wr.nominal_q_ref, dtype=jnp.float32)
    qpos = state.data.qpos.at[env._actuator_qpos_addrs].set(nominal_q_ref)  # noqa: SLF001
    ctrl = nominal_q_ref
    data = state.data.replace(qpos=qpos, ctrl=ctrl)
    data = mjx.forward(env._mjx_model, data)  # noqa: SLF001
    return state.replace(data=data)


def _compute_stance_ground_shift(*, stance_foot: int, left_foot_z: float, right_foot_z: float) -> float:
    stance_foot_z = left_foot_z if int(stance_foot) == 0 else right_foot_z
    return -stance_foot_z


def _replace_state_with_nominal_qref_grounded(env: WildRobotEnv, state):
    state = _replace_state_with_nominal_qref(env, state)
    wr = state.info[WR_INFO_KEY]
    stance_foot = int(jnp.asarray(wr.loc_ref_stance_foot, dtype=jnp.int32))
    left_foot_pos, right_foot_pos = env._cal.get_foot_positions(  # noqa: SLF001
        state.data, normalize=False
    )
    left_foot_z = float(left_foot_pos[2])
    right_foot_z = float(right_foot_pos[2])
    root_spec = env._cal.root_spec  # noqa: SLF001
    if root_spec is None:
        raise ValueError("ControlAbstractionLayer root_spec is required for grounded nominal init")
    root_z_addr = int(root_spec.qpos_addr + 2)
    root_dz = _compute_stance_ground_shift(
        stance_foot=stance_foot, left_foot_z=left_foot_z, right_foot_z=right_foot_z
    )
    qpos = state.data.qpos.at[root_z_addr].add(jnp.asarray(root_dz, dtype=state.data.qpos.dtype))
    data = state.data.replace(qpos=qpos)
    data = mjx.forward(env._mjx_model, data)  # noqa: SLF001
    grounded_state = state.replace(data=data)

    grounded_left_foot_pos, grounded_right_foot_pos = env._cal.get_foot_positions(  # noqa: SLF001
        grounded_state.data, normalize=False
    )
    grounded_left_foot_z = float(grounded_left_foot_pos[2])
    grounded_right_foot_z = float(grounded_right_foot_pos[2])
    grounded_stance_foot_z = (
        grounded_left_foot_z if stance_foot == 0 else grounded_right_foot_z
    )
    grounded_root_height = float(env._cal.get_root_height(grounded_state.data))  # noqa: SLF001
    geometry = {
        "stance_foot": float(stance_foot),
        "root_height": grounded_root_height,
        "root_dz_applied": root_dz,
        "stance_foot_z_before": left_foot_z if stance_foot == 0 else right_foot_z,
        "stance_foot_z": grounded_stance_foot_z,
        "left_foot_z": grounded_left_foot_z,
        "right_foot_z": grounded_right_foot_z,
    }
    return grounded_state, geometry


def _format_grounded_nominal_init_line(geometry: dict[str, float]) -> str:
    stance = int(round(float(geometry["stance_foot"])))
    return (
        "[nominal-viewer] nominal_qref init grounded: "
        f"stance={stance} "
        f"root_h={float(geometry['root_height']):.3f} "
        f"root_dz={float(geometry['root_dz_applied']):+.4f} "
        f"stance_foot_z_before={float(geometry['stance_foot_z_before']):+.4f} "
        f"stance_foot_z={float(geometry['stance_foot_z']):+.4f} "
        f"left_foot_z={float(geometry['left_foot_z']):+.4f} "
        f"right_foot_z={float(geometry['right_foot_z']):+.4f}"
    )


def _extract_reset_lateral_semantics(env: WildRobotEnv, state) -> dict[str, float]:
    wr = state.info[WR_INFO_KEY]
    stance = int(jnp.asarray(wr.loc_ref_stance_foot, dtype=jnp.int32))
    left_foot_h, right_foot_h = env._cal.get_foot_positions(  # noqa: SLF001
        state.data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
    )
    left_y = float(left_foot_h[1])
    right_y = float(right_foot_h[1])
    next_foothold = jnp.asarray(wr.loc_ref_next_foothold, dtype=jnp.float32)
    swing_target = jnp.asarray(wr.loc_ref_swing_pos, dtype=jnp.float32)
    base_support_y = float(next_foothold[1])
    swing_y_target = float(swing_target[1])
    swing_y_actual = right_y - left_y if stance == 0 else left_y - right_y
    return {
        "stance": float(stance),
        "left_y": left_y,
        "right_y": right_y,
        "width_actual": abs(right_y - left_y),
        "width_reference": abs(base_support_y),
        "width_commanded": abs(swing_y_target),
        "base_support_y": base_support_y,
        "lateral_release_y": swing_y_target - base_support_y,
        "swing_y_target": swing_y_target,
        "swing_y_actual": swing_y_actual,
        "swing_y_error": swing_y_actual - swing_y_target,
    }


def _extract_lateral_semantics(metrics_vec: jnp.ndarray) -> dict[str, float]:
    return {
        "stance": float(metrics_vec[METRIC_INDEX["tracking/loc_ref_stance_foot"]]),
        "left_y": float(metrics_vec[METRIC_INDEX["debug/loc_ref_left_foot_y_actual"]]),
        "right_y": float(metrics_vec[METRIC_INDEX["debug/loc_ref_right_foot_y_actual"]]),
        "width_actual": float(metrics_vec[METRIC_INDEX["debug/loc_ref_support_width_y_actual"]]),
        "width_reference": float(
            metrics_vec[METRIC_INDEX["debug/loc_ref_support_width_y_nominal"]]
        ),
        "width_commanded": float(
            metrics_vec[METRIC_INDEX["debug/loc_ref_support_width_y_commanded"]]
        ),
        "base_support_y": float(metrics_vec[METRIC_INDEX["debug/loc_ref_base_support_y"]]),
        "lateral_release_y": float(metrics_vec[METRIC_INDEX["debug/loc_ref_lateral_release_y"]]),
        "swing_y_target": float(metrics_vec[METRIC_INDEX["debug/loc_ref_swing_y_target"]]),
        "swing_y_actual": float(metrics_vec[METRIC_INDEX["debug/loc_ref_swing_y_actual"]]),
        "swing_y_error": float(metrics_vec[METRIC_INDEX["debug/loc_ref_swing_y_error"]]),
    }


def _format_reset_lateral_line_from_semantics(sem: dict[str, float]) -> str:
    stance = int(round(sem["stance"]))
    return (
        "[nominal-viewer] reset lateral geometry: "
        f"stance={stance} "
        f"left_y={sem['left_y']:+.3f} "
        f"right_y={sem['right_y']:+.3f} "
        f"width_act={sem['width_actual']:.3f} "
        f"width_ref={sem['width_reference']:.3f} "
        f"width_cmd={sem['width_commanded']:.3f} "
        f"base_y={sem['base_support_y']:+.3f} "
        f"release_y={sem['lateral_release_y']:+.3f} "
        f"swing_y_tgt={sem['swing_y_target']:+.3f} "
        f"swing_y_act={sem['swing_y_actual']:+.3f}"
    )


def _format_reset_lateral_line(metrics_vec: jnp.ndarray) -> str:
    return _format_reset_lateral_line_from_semantics(_extract_lateral_semantics(metrics_vec))


def _format_reset_lateral_line_from_state(env: WildRobotEnv, state) -> str:
    return _format_reset_lateral_line_from_semantics(_extract_reset_lateral_semantics(env, state))


def _format_step_line(step: int, state, metrics_vec: jnp.ndarray) -> str:
    wr = state.info[WR_INFO_KEY]
    sem = _extract_lateral_semantics(metrics_vec)
    return (
        f"step={step:04d} "
        f"cmd={float(wr.velocity_cmd):.3f} "
        f"v_fwd={float(metrics_vec[METRIC_INDEX['forward_velocity']]):+.3f} "
        f"stance={int(round(sem['stance']))} "
        f"left_y={sem['left_y']:+.3f} "
        f"right_y={sem['right_y']:+.3f} "
        f"width_act={sem['width_actual']:.3f} "
        f"width_ref={sem['width_reference']:.3f} "
        f"width_cmd={sem['width_commanded']:.3f} "
        f"base_y={sem['base_support_y']:+.3f} "
        f"release_y={sem['lateral_release_y']:+.3f} "
        f"pitch={float(metrics_vec[METRIC_INDEX['debug/loc_ref_root_pitch']]):+.3f} "
        f"pitch_rate={float(metrics_vec[METRIC_INDEX['debug/loc_ref_root_pitch_rate']]):+.3f} "
        f"mode={int(round(float(metrics_vec[METRIC_INDEX['debug/loc_ref_hybrid_mode_id']])))} "
        f"support={float(metrics_vec[METRIC_INDEX['debug/loc_ref_support_health']]):.3f} "
        f"perm={float(metrics_vec[METRIC_INDEX['debug/loc_ref_progression_permission']]):.3f} "
        f"swing_x_scale_active={float(metrics_vec[METRIC_INDEX['debug/loc_ref_swing_x_scale_active']]):.3f} "
        f"swing_y_tgt={sem['swing_y_target']:+.3f} "
        f"swing_y_act={sem['swing_y_actual']:+.3f} "
        f"swing_y_err={sem['swing_y_error']:+.3f} "
        f"pelvis_roll_tgt={float(metrics_vec[METRIC_INDEX['debug/loc_ref_pelvis_roll_target']]):+.3f} "
        f"lhr_tgt={float(metrics_vec[METRIC_INDEX['debug/loc_ref_hip_roll_left_target']]):+.3f} "
        f"rhr_tgt={float(metrics_vec[METRIC_INDEX['debug/loc_ref_hip_roll_right_target']]):+.3f} "
        f"q_gap={float(metrics_vec[METRIC_INDEX['debug/loc_ref_nominal_vs_applied_q_l1']]):.5f}"
    )


def _extract_support_posture(
    env: WildRobotEnv,
    state,
    metrics_vec: jnp.ndarray | None = None,
    *,
    stance_foot: int | None = None,
) -> dict[str, float | str]:
    wr = state.info[WR_INFO_KEY]
    nominal_q_ref = jnp.asarray(wr.nominal_q_ref, dtype=jnp.float32)
    actual_q = jnp.asarray(state.data.qpos, dtype=jnp.float32)[env._actuator_qpos_addrs]  # noqa: SLF001
    if stance_foot is None:
        if metrics_vec is not None:
            stance = int(round(float(metrics_vec[METRIC_INDEX["tracking/loc_ref_stance_foot"]])))
        else:
            stance = int(jnp.asarray(wr.loc_ref_stance_foot, dtype=jnp.int32))
    else:
        stance = int(stance_foot)
    root_pose = env._cal.get_root_pose(state.data)  # noqa: SLF001
    root_roll, _, _ = root_pose.euler_angles()
    root_height = root_pose.height

    if stance == 0:
        leg = "L"
        idx_hip_roll = env._idx_left_hip_roll  # noqa: SLF001
        idx_hip_pitch = env._idx_left_hip_pitch  # noqa: SLF001
        idx_knee_pitch = env._idx_left_knee_pitch  # noqa: SLF001
        idx_ankle_pitch = env._idx_left_ankle_pitch  # noqa: SLF001
    else:
        leg = "R"
        idx_hip_roll = env._idx_right_hip_roll  # noqa: SLF001
        idx_hip_pitch = env._idx_right_hip_pitch  # noqa: SLF001
        idx_knee_pitch = env._idx_right_knee_pitch  # noqa: SLF001
        idx_ankle_pitch = env._idx_right_ankle_pitch  # noqa: SLF001

    def joint_triplet(idx: int) -> tuple[float, float, float]:
        if idx < 0:
            return 0.0, 0.0, 0.0
        target = float(nominal_q_ref[idx])
        actual = float(actual_q[idx])
        return target, actual, actual - target

    hr_tgt, hr_act, hr_err = joint_triplet(idx_hip_roll)
    hp_tgt, hp_act, hp_err = joint_triplet(idx_hip_pitch)
    kp_tgt, kp_act, kp_err = joint_triplet(idx_knee_pitch)
    ap_tgt, ap_act, ap_err = joint_triplet(idx_ankle_pitch)
    return {
        "stance_leg": leg,
        "root_roll": float(root_roll),
        "root_height": float(root_height),
        "hip_roll_tgt": hr_tgt,
        "hip_roll_act": hr_act,
        "hip_roll_err": hr_err,
        "hip_pitch_tgt": hp_tgt,
        "hip_pitch_act": hp_act,
        "hip_pitch_err": hp_err,
        "knee_pitch_tgt": kp_tgt,
        "knee_pitch_act": kp_act,
        "knee_pitch_err": kp_err,
        "ankle_pitch_tgt": ap_tgt,
        "ankle_pitch_act": ap_act,
        "ankle_pitch_err": ap_err,
    }


def _format_support_posture_line(
    env: WildRobotEnv,
    state,
    metrics_vec: jnp.ndarray | None = None,
    *,
    stance_foot: int | None = None,
) -> str:
    pose = _extract_support_posture(env, state, metrics_vec, stance_foot=stance_foot)
    return (
        "[nominal-viewer] support-posture "
        f"leg={pose['stance_leg']} "
        f"root_roll={pose['root_roll']:+.3f} "
        f"root_h={pose['root_height']:.3f} "
        f"hr_tgt={pose['hip_roll_tgt']:+.3f} hr_act={pose['hip_roll_act']:+.3f} hr_err={pose['hip_roll_err']:+.3f} "
        f"hp_tgt={pose['hip_pitch_tgt']:+.3f} hp_act={pose['hip_pitch_act']:+.3f} hp_err={pose['hip_pitch_err']:+.3f} "
        f"kp_tgt={pose['knee_pitch_tgt']:+.3f} kp_act={pose['knee_pitch_act']:+.3f} kp_err={pose['knee_pitch_err']:+.3f} "
        f"ap_tgt={pose['ankle_pitch_tgt']:+.3f} ap_act={pose['ankle_pitch_act']:+.3f} ap_err={pose['ankle_pitch_err']:+.3f}"
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
    if args.force_support_only:
        env._walking_ref_v2_cfg = replace(  # noqa: SLF001
            env._walking_ref_v2_cfg, debug_force_support_only=True  # noqa: SLF001
        )
    rng = jax.random.PRNGKey(int(args.seed))
    state = env.reset(rng)
    grounded_init_geometry: dict[str, float] | None = None
    if args.init_from_nominal_qref:
        state, grounded_init_geometry = _replace_state_with_nominal_qref_grounded(env, state)
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
    if args.force_support_only:
        print("[nominal-viewer] forcing walking_ref_v2 into SUPPORT_STABILIZE for posture-only diagnosis")
    if args.init_from_nominal_qref:
        print(
            "[nominal-viewer] reset actuated joints replaced with current nominal_q_ref posture "
            "and stance-foot grounding applied"
        )
        if grounded_init_geometry is not None:
            print(_format_grounded_nominal_init_line(grounded_init_geometry))
    print(_format_reset_lateral_line_from_state(env, state))
    reset_stance = int(jnp.asarray(state.info[WR_INFO_KEY].loc_ref_stance_foot, dtype=jnp.int32))
    print(_format_support_posture_line(env, state, stance_foot=reset_stance))

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
            print(_format_support_posture_line(env, state, metrics_vec))
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
    if args.log:
        _enable_temp_logging()
    return run_nominal_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
