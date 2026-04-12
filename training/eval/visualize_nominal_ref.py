#!/usr/bin/env python3
"""Nominal-only M2.5 reference viewer/debugger (q_target = q_ref)."""

from __future__ import annotations

import argparse
import atexit
import io
import os
import platform
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


def _has_display() -> bool:
    """Check whether a graphical display is available."""
    if platform.system() == "Darwin":
        return True
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    return bool(display or wayland)


def _configure_gl_backend() -> None:
    """Set MUJOCO_GL for headless Linux when not already configured.

    On macOS the native CGL backend is used automatically.
    On Linux without a display, prefer EGL (GPU-accelerated) and fall back
    to OSMesa (software) if EGL is unavailable.
    """
    if platform.system() != "Linux":
        return
    if os.environ.get("MUJOCO_GL"):
        return  # user already chose
    if _has_display():
        return  # GLFW will work
    # Headless Linux: try EGL first, then OSMesa.
    try:
        os.environ["MUJOCO_GL"] = "egl"
        # Quick probe — import mujoco.egl is not a thing, but creating a
        # Renderer will fail fast if EGL is missing.  We defer the real check
        # to _try_create_renderer().
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"


_configure_gl_backend()

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import load_robot_config
from training.cal.cal import CoordinateFrame
from training.configs.training_config import TrainingConfig, load_training_config
from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from control.references.walking_ref_v2 import (
    StartupRouteReason,
    StartupRouteStage,
    WalkingRefV2Mode,
)


_ROUTE_STAGE_LABELS = {
    int(StartupRouteStage.A): "A",
    int(StartupRouteStage.W1): "W1",
    int(StartupRouteStage.W2): "W2",
    int(StartupRouteStage.W3): "W3",
    int(StartupRouteStage.B): "B",
}

_ROUTE_REASON_LABELS = {
    int(StartupRouteReason.NONE): "NONE",
    int(StartupRouteReason.PELVIS_NOT_REALIZED): "PELVIS_NOT_REALIZED",
    int(StartupRouteReason.JOINT_TRACKING_LAG): "JOINT_TRACKING_LAG",
    int(StartupRouteReason.ROOT_PITCH_LIMIT): "ROOT_PITCH_LIMIT",
    int(StartupRouteReason.ROOT_PITCH_RATE_LIMIT): "ROOT_PITCH_RATE_LIMIT",
    int(StartupRouteReason.SUPPORT_HEALTH_LOW): "SUPPORT_HEALTH_LOW",
    int(StartupRouteReason.TIMEOUT_FALLBACK): "TIMEOUT_FALLBACK",
    int(StartupRouteReason.ROUTE_COMPLETE): "ROUTE_COMPLETE",
}


def _route_stage_label(stage_id: int) -> str:
    return _ROUTE_STAGE_LABELS.get(int(stage_id), f"STAGE_{int(stage_id)}")


def _route_reason_label(reason_id: int) -> str:
    return _ROUTE_REASON_LABELS.get(int(reason_id), f"REASON_{int(reason_id)}")


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


def _print_log_only(message: str) -> None:
    stdout = sys.stdout
    if isinstance(stdout, _TeeTextIO):
        stdout._secondary.write(message + "\n")  # noqa: SLF001
        stdout._secondary.flush()  # noqa: SLF001
        return
    print(message)


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
    class _HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        """Keep defaults and preserve multi-line examples."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize nominal-only locomotion reference (q_target=q_ref).\n\n"
            "Runs the environment with zero residual policy action so command-path\n"
            "diagnostics isolate nominal reference delivery and posture realization."
        ),
        formatter_class=_HelpFormatter,
        add_help=False,
        epilog=(
            "Examples:\n"
            "  uv run python training/eval/visualize_nominal_ref.py --headless\n"
            "  uv run python training/eval/visualize_nominal_ref.py --forward-cmd 0.10 --horizon 64 --log --disable-action-filter\n"
            "  uv run python training/eval/visualize_nominal_ref.py --force-support-only --init-from-nominal-qref --disable-action-filter\n"
            "  uv run python training/eval/visualize_nominal_ref.py --startup-target-rate-deg-s 10 --disable-action-filter"
        ),
    )
    parser.add_argument(
        "-h",
        "--help",
        "--?",
        action="help",
        default=argparse.SUPPRESS,
        help="Show command options, explanations, and examples",
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
        nargs="?",
        const="",
        default=None,
        help="Record video (mp4). Optionally specify output path; if omitted, a temp file is used.",
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
        "--startup-settle-steps",
        type=int,
        default=0,
        help=(
            "Smooth A->B transition: gradually interpolate from keyframe (A) to "
            "support posture (B) over N steps via pure MuJoCo, then continue with "
            "normal control.  Only used with --force-support-only.  0 = disabled."
        ),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write the invoked command and viewer console output to a temp log file",
    )
    parser.add_argument(
        "--disable-action-filter",
        action="store_true",
        help="Debug-only: force env.action_filter_alpha=0.0 for this nominal viewer run",
    )
    parser.add_argument(
        "--startup-target-rate-deg-s",
        type=float,
        default=None,
        help=(
            "Debug-only: override loc_ref_v2_startup_target_rate_design_rad_s "
            "(deg/s) for startup/support-entry shaping sweeps"
        ),
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


def _configure_nominal_only(
    cfg: TrainingConfig,
    *,
    forward_cmd: float,
    horizon: int,
    disable_action_filter: bool = False,
    startup_target_rate_deg_s: float | None = None,
) -> None:
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
    if disable_action_filter:
        cfg.env.action_filter_alpha = 0.0
    if startup_target_rate_deg_s is not None:
        cfg.env.loc_ref_v2_startup_target_rate_design_rad_s = float(
            np.deg2rad(float(startup_target_rate_deg_s))
        )


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


def _geom_bottom_z(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, geom_id: int) -> float:
    geom_center = np.asarray(mj_data.geom_xpos[geom_id], dtype=np.float64)
    geom_xmat = np.asarray(mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    geom_size = np.asarray(mj_model.geom_size[geom_id], dtype=np.float64)
    geom_type = int(mj_model.geom_type[geom_id])

    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        z_extent = np.abs(geom_xmat[2, 0]) * geom_size[0]
        z_extent += np.abs(geom_xmat[2, 1]) * geom_size[1]
        z_extent += np.abs(geom_xmat[2, 2]) * geom_size[2]
    elif geom_type in (
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        z_extent = float(geom_size[0])
    else:
        # Fallback for unsupported geom types. For the current foot contact geoms
        # we expect boxes, but keep the helper safe.
        z_extent = 0.0

    return float(geom_center[2] - z_extent)


def _compute_support_bottom_zs(env: WildRobotEnv, state) -> tuple[float, float]:
    mj_model = env._mj_model  # noqa: SLF001
    mj_data = mujoco.MjData(mj_model)
    _sync_viewer_data(mj_model, mj_data, state.data)

    left_foot, right_foot = env._cal._foot_specs  # noqa: SLF001
    left_z = min(
        _geom_bottom_z(mj_model, mj_data, int(left_foot.toe_geom_id)),
        _geom_bottom_z(mj_model, mj_data, int(left_foot.heel_geom_id)),
    )
    right_z = min(
        _geom_bottom_z(mj_model, mj_data, int(right_foot.toe_geom_id)),
        _geom_bottom_z(mj_model, mj_data, int(right_foot.heel_geom_id)),
    )
    return float(left_z), float(right_z)


def _replace_state_with_nominal_qref_grounded(env: WildRobotEnv, state):
    state = _replace_state_with_nominal_qref(env, state)
    wr = state.info[WR_INFO_KEY]
    stance_foot = int(jnp.asarray(wr.loc_ref_stance_foot, dtype=jnp.int32))
    left_support_z, right_support_z = _compute_support_bottom_zs(env, state)
    root_spec = env._cal.root_spec  # noqa: SLF001
    if root_spec is None:
        raise ValueError("ControlAbstractionLayer root_spec is required for grounded nominal init")
    root_z_addr = int(root_spec.qpos_addr + 2)
    root_dz = _compute_stance_ground_shift(
        stance_foot=stance_foot, left_foot_z=left_support_z, right_foot_z=right_support_z
    )
    qpos = state.data.qpos.at[root_z_addr].add(jnp.asarray(root_dz, dtype=state.data.qpos.dtype))
    data = state.data.replace(qpos=qpos)
    data = mjx.forward(env._mjx_model, data)  # noqa: SLF001
    grounded_state = state.replace(data=data)

    grounded_left_support_z, grounded_right_support_z = _compute_support_bottom_zs(
        env, grounded_state
    )
    grounded_stance_foot_z = (
        grounded_left_support_z if stance_foot == 0 else grounded_right_support_z
    )
    grounded_root_height = float(env._cal.get_root_height(grounded_state.data))  # noqa: SLF001
    geometry = {
        "stance_foot": float(stance_foot),
        "root_height": grounded_root_height,
        "root_dz_applied": root_dz,
        "stance_support_z_before": left_support_z if stance_foot == 0 else right_support_z,
        "stance_foot_z": grounded_stance_foot_z,
        "left_support_z": grounded_left_support_z,
        "right_support_z": grounded_right_support_z,
    }
    return grounded_state, geometry


def _format_grounded_nominal_init_line(geometry: dict[str, float]) -> str:
    stance = int(round(float(geometry["stance_foot"])))
    return (
        "[nominal-viewer] nominal_qref init grounded: "
        f"stance={stance} "
        f"root_h={float(geometry['root_height']):.3f} "
        f"root_dz={float(geometry['root_dz_applied']):+.4f} "
        f"stance_support_z_before={float(geometry['stance_support_z_before']):+.4f} "
        f"stance_support_z={float(geometry['stance_foot_z']):+.4f} "
        f"left_support_z={float(geometry['left_support_z']):+.4f} "
        f"right_support_z={float(geometry['right_support_z']):+.4f}"
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
    route_stage = _route_stage_label(
        int(
            jnp.asarray(
                getattr(wr, "loc_ref_startup_route_stage_id", int(StartupRouteStage.B)),
                dtype=jnp.int32,
            )
        )
    )
    route_reason = _route_reason_label(
        int(
            jnp.asarray(
                getattr(wr, "loc_ref_startup_route_transition_reason", int(StartupRouteReason.NONE)),
                dtype=jnp.int32,
            )
        )
    )
    route_prog = float(
        jnp.asarray(getattr(wr, "loc_ref_startup_route_progress", 1.0), dtype=jnp.float32)
    )
    route_cap = float(
        jnp.asarray(getattr(wr, "loc_ref_startup_route_ceiling", 1.0), dtype=jnp.float32)
    )
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
        f"route={route_stage} "
        f"route_prog={route_prog:.3f}/{route_cap:.3f} "
        f"route_reason={route_reason} "
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
    ctrl = jnp.asarray(state.data.ctrl, dtype=jnp.float32)
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
    pelvis_height_ref = float(
        jnp.asarray(getattr(wr, "loc_ref_pelvis_height", root_height), dtype=jnp.float32)
    )

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

    def joint_quintuplet(idx: int) -> tuple[float, float, float, float, float]:
        """Return (q_ref, ctrl, actual, ref_err, ctrl_err) for a joint."""
        if idx < 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        q_ref = float(nominal_q_ref[idx])
        q_ctrl = float(ctrl[idx])
        q_actual = float(actual_q[idx])
        return q_ref, q_ctrl, q_actual, q_actual - q_ref, q_actual - q_ctrl

    hr_ref, hr_ctrl, hr_act, hr_ref_err, hr_ctrl_err = joint_quintuplet(idx_hip_roll)
    hp_ref, hp_ctrl, hp_act, hp_ref_err, hp_ctrl_err = joint_quintuplet(idx_hip_pitch)
    kp_ref, kp_ctrl, kp_act, kp_ref_err, kp_ctrl_err = joint_quintuplet(idx_knee_pitch)
    ap_ref, ap_ctrl, ap_act, ap_ref_err, ap_ctrl_err = joint_quintuplet(idx_ankle_pitch)
    return {
        "stance_leg": leg,
        "pelvis_height_ref": pelvis_height_ref,
        "root_roll": float(root_roll),
        "root_height": float(root_height),
        "hip_roll_ref": hr_ref,
        "hip_roll_ctrl": hr_ctrl,
        "hip_roll_act": hr_act,
        "hip_roll_ref_err": hr_ref_err,
        "hip_roll_ctrl_err": hr_ctrl_err,
        "hip_pitch_ref": hp_ref,
        "hip_pitch_ctrl": hp_ctrl,
        "hip_pitch_act": hp_act,
        "hip_pitch_ref_err": hp_ref_err,
        "hip_pitch_ctrl_err": hp_ctrl_err,
        "knee_pitch_ref": kp_ref,
        "knee_pitch_ctrl": kp_ctrl,
        "knee_pitch_act": kp_act,
        "knee_pitch_ref_err": kp_ref_err,
        "knee_pitch_ctrl_err": kp_ctrl_err,
        "ankle_pitch_ref": ap_ref,
        "ankle_pitch_ctrl": ap_ctrl,
        "ankle_pitch_act": ap_act,
        "ankle_pitch_ref_err": ap_ref_err,
        "ankle_pitch_ctrl_err": ap_ctrl_err,
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
        f"hr_ref={pose['hip_roll_ref']:+.3f} hr_ctrl={pose['hip_roll_ctrl']:+.3f} hr_act={pose['hip_roll_act']:+.3f} "
        f"hp_ref={pose['hip_pitch_ref']:+.3f} hp_ctrl={pose['hip_pitch_ctrl']:+.3f} hp_act={pose['hip_pitch_act']:+.3f} "
        f"kp_ref={pose['knee_pitch_ref']:+.3f} kp_ctrl={pose['knee_pitch_ctrl']:+.3f} kp_act={pose['knee_pitch_act']:+.3f} "
        f"ap_ref={pose['ankle_pitch_ref']:+.3f} ap_ctrl={pose['ankle_pitch_ctrl']:+.3f} ap_act={pose['ankle_pitch_act']:+.3f}"
    )


def _try_create_renderer(
    mj_model: mujoco.MjModel, width: int = 960, height: int = 720
) -> mujoco.Renderer | None:
    """Create an offscreen renderer, returning *None* on GL failure."""
    try:
        return mujoco.Renderer(mj_model, height, width)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[nominal-viewer] WARNING: could not create offscreen renderer "
            f"({exc!r}); --record will be skipped.  "
            f"On headless Linux try: MUJOCO_GL=egl or MUJOCO_GL=osmesa"
        )
        return None


def _try_launch_viewer(mj_model, mj_data):
    """Try to open the interactive MuJoCo viewer, return None on failure."""
    try:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(mj_model, mj_data)
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0.0, 0.0, 0.4]
        return viewer
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            print(
                f"[nominal-viewer] ERROR: interactive viewer requires mjpython on macOS.\n"
                f"  Run with:  mjpython {' '.join(sys.argv)}\n"
                f"  Or add --headless to skip the viewer."
            )
        else:
            print(
                f"[nominal-viewer] WARNING: could not launch interactive viewer "
                f"({exc!r}); falling back to headless mode."
            )
        return None
    except Exception as exc:  # noqa: BLE001
        print(
            f"[nominal-viewer] WARNING: could not launch interactive viewer "
            f"({exc!r}); falling back to headless mode."
        )
        return None


def _save_video(record_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import mediapy as media

        media.write_video(record_path, frames, fps=fps)
        return
    except ModuleNotFoundError:
        pass
    try:
        import imageio

        imageio.mimsave(record_path, frames, fps=fps)
        return
    except ModuleNotFoundError:
        pass
    print(
        f"[nominal-viewer] WARNING: neither mediapy nor imageio installed; "
        f"cannot write video to {record_path}"
    )


def _check_tracking_threshold(q_act: float, q_tgt: float, abs_floor: float = 0.05, frac_tol: float = 0.10) -> bool:
    """Check if actual joint tracks target within threshold.
    
    Rule: abs(q_act - q_tgt) <= max(abs_floor, frac_tol * abs(q_tgt))
    """
    error = abs(q_act - q_tgt)
    threshold = max(abs_floor, frac_tol * abs(q_tgt))
    return error <= threshold


def _is_stable_window(
    mode_history: list[int],
    pitch_history: list[float],
    pitch_rate_history: list[float],
    min_dwell: int = 5,
    pitch_thresh: float = 0.30,  # ~17 degrees
    pitch_rate_thresh: float = 1.5,  # rad/s
) -> bool:
    """Check if recent history indicates a stable window for evaluation.
    
    Stable if:
    - Same mode maintained for at least min_dwell steps
    - abs(pitch) < pitch_thresh for all recent steps
    - abs(pitch_rate) < pitch_rate_thresh for all recent steps
    """
    if len(mode_history) < min_dwell:
        return False
    recent_modes = mode_history[-min_dwell:]
    if len(set(recent_modes)) > 1:
        return False
    recent_pitch = pitch_history[-min_dwell:]
    recent_pitch_rate = pitch_rate_history[-min_dwell:]
    return (
        all(abs(p) < pitch_thresh for p in recent_pitch)
        and all(abs(pr) < pitch_rate_thresh for pr in recent_pitch_rate)
    )


class _StageTracker:
    """Tracks per-stage posture diagnostics."""
    
    def __init__(self):
        self.stages = {}  # mode_id -> stage data
        self.current_mode = None
        self.current_mode_entry_step = None
        self.mode_history = []
        self.pitch_history = []
        self.pitch_rate_history = []
        
    def update(self, step: int, mode_id: int, pitch: float, pitch_rate: float, posture: dict):
        """Update stage tracking with current step data."""
        # Track mode transitions
        if mode_id != self.current_mode:
            if self.current_mode is not None:
                # Exiting previous stage
                stage = self.stages[self.current_mode]
                stage["exit_step"] = step - 1
            self.current_mode = mode_id
            self.current_mode_entry_step = step
            if mode_id not in self.stages:
                self.stages[mode_id] = {
                    "entry_step": step,
                    "exit_step": None,
                    "samples": [],
                    "stable_samples": [],
                }
        
        # Update history for stable window detection
        self.mode_history.append(mode_id)
        self.pitch_history.append(pitch)
        self.pitch_rate_history.append(pitch_rate)
        if len(self.mode_history) > 10:
            self.mode_history.pop(0)
            self.pitch_history.pop(0)
            self.pitch_rate_history.pop(0)
        
        # Record sample
        sample = {
            "step": step,
            "hr_tgt": posture["hip_roll_ref"],
            "hr_act": posture["hip_roll_act"],
            "hp_tgt": posture["hip_pitch_ref"],
            "hp_act": posture["hip_pitch_act"],
            "kp_tgt": posture["knee_pitch_ref"],
            "kp_act": posture["knee_pitch_act"],
            "ap_tgt": posture["ankle_pitch_ref"],
            "ap_act": posture["ankle_pitch_act"],
        }
        self.stages[mode_id]["samples"].append(sample)
        
        # Check if this is a stable window sample
        if _is_stable_window(self.mode_history, self.pitch_history, self.pitch_rate_history):
            self.stages[mode_id]["stable_samples"].append(sample)
    
    def finalize(self, final_step: int):
        """Finalize tracking at end of run."""
        if self.current_mode is not None and self.stages[self.current_mode]["exit_step"] is None:
            self.stages[self.current_mode]["exit_step"] = final_step
    
    def get_summary(self) -> str:
        """Generate stage diagnostics summary."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("STAGE DIAGNOSTICS SUMMARY")
        lines.append("=" * 80)
        lines.append("Threshold rule: abs(q_act - q_tgt) <= max(0.05 rad, 0.10 * abs(q_tgt))")
        lines.append("Stable window: 5+ steps same mode, abs(pitch)<0.30, abs(pitch_rate)<1.5")
        lines.append("")
        
        mode_names = {
            int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP): "STARTUP_SUPPORT_RAMP",
            int(WalkingRefV2Mode.SUPPORT_STABILIZE): "SUPPORT_STABILIZE",
            int(WalkingRefV2Mode.SWING_RELEASE): "SWING_RELEASE",
            int(WalkingRefV2Mode.TOUCHDOWN_CAPTURE): "TOUCHDOWN_CAPTURE",
            int(WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE): "POST_TOUCHDOWN_SETTLE",
        }
        
        for mode_id in sorted(self.stages.keys()):
            stage = self.stages[mode_id]
            mode_name = mode_names.get(mode_id, f"MODE_{mode_id}")
            
            entry = stage["entry_step"]
            exit_step = stage["exit_step"] if stage["exit_step"] is not None else "ongoing"
            has_stable = len(stage["stable_samples"]) > 0
            
            lines.append(f"Stage: {mode_name} (mode_id={mode_id})")
            lines.append(f"  Entry/Exit: step {entry} -> {exit_step}")
            lines.append(f"  Total samples: {len(stage['samples'])}")
            lines.append(f"  Stable window found: {'yes' if has_stable else 'no'} ({len(stage['stable_samples'])} samples)")
            
            if has_stable:
                samples = stage["stable_samples"]
                joints = ["hr", "hp", "kp", "ap"]
                joint_names = {
                    "hr": "hip_roll",
                    "hp": "hip_pitch",
                    "kp": "knee_pitch",
                    "ap": "ankle_pitch",
                }
                
                for joint in joints:
                    tgts = [s[f"{joint}_tgt"] for s in samples]
                    acts = [s[f"{joint}_act"] for s in samples]
                    errs = [abs(a - t) for a, t in zip(acts, tgts)]
                    
                    tgt_mean = np.mean(tgts)
                    act_mean = np.mean(acts)
                    err_mean = np.mean(errs)
                    err_max = np.max(errs)
                    
                    # Check threshold
                    passes = [
                        _check_tracking_threshold(a, t)
                        for a, t in zip(acts, tgts)
                    ]
                    pass_rate = sum(passes) / len(passes) if passes else 0.0
                    
                    status = "PASS" if pass_rate >= 0.8 else "FAIL"
                    lines.append(
                        f"  {joint_names[joint]:12s}: tgt_mean={tgt_mean:+.3f} act_mean={act_mean:+.3f} "
                        f"err_mean={err_mean:.3f} err_max={err_max:.3f} pass_rate={pass_rate:.0%} [{status}]"
                    )
            else:
                lines.append("  (No stable window - cannot evaluate tracking)")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


class _CommandPathTracker:
    """Track command-path signals across startup and early/late support phases."""

    _JOINTS = ("hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch")

    def __init__(self, *, ctrl_dt: float, support_entry_window_s: float):
        self.ctrl_dt = float(max(ctrl_dt, 1e-6))
        self.support_entry_window_s = float(max(support_entry_window_s, 0.0))
        self.samples: list[dict[str, float | int | str]] = []
        self.prev_ctrl: dict[str, float] | None = None
        self.prev_ref: dict[str, float] | None = None

    def _phase_label(self, mode_id: int, mode_time_s: float) -> str:
        startup_mode = int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP)
        support_mode = int(WalkingRefV2Mode.SUPPORT_STABILIZE)
        if mode_id == startup_mode:
            return "STARTUP_SUPPORT_RAMP"
        if mode_id == support_mode:
            if mode_time_s < self.support_entry_window_s:
                return "EARLY_SUPPORT_STABILIZE"
            return "LATE_SUPPORT_STABILIZE"
        return "OTHER"

    def update(self, step: int, mode_id: int, mode_time_s: float, posture: dict) -> None:
        """Update tracking with current step data."""
        phase_label = self._phase_label(mode_id, mode_time_s)
        for joint in self._JOINTS:
            q_ref = float(posture[f"{joint}_ref"])
            q_ctrl = float(posture[f"{joint}_ctrl"])
            q_act = float(posture[f"{joint}_act"])

            if self.prev_ctrl is None:
                ctrl_delta = 0.0
            else:
                ctrl_delta = q_ctrl - float(self.prev_ctrl.get(joint, q_ctrl))
            if self.prev_ref is None:
                ref_delta = 0.0
            else:
                ref_delta = q_ref - float(self.prev_ref.get(joint, q_ref))

            self.samples.append(
                {
                    "step": int(step),
                    "mode": int(mode_id),
                    "phase": phase_label,
                    "joint": joint,
                    "q_ref": q_ref,
                    "q_ctrl": q_ctrl,
                    "q_act": q_act,
                    "ref_to_ctrl_gap": q_ctrl - q_ref,
                    "ctrl_to_act_gap": q_act - q_ctrl,
                    "ctrl_delta": ctrl_delta,
                    "ref_delta": ref_delta,
                }
            )

        self.prev_ctrl = {joint: float(posture[f"{joint}_ctrl"]) for joint in self._JOINTS}
        self.prev_ref = {joint: float(posture[f"{joint}_ref"]) for joint in self._JOINTS}

    def get_summary(self) -> str:
        """Generate command-path summary for startup + early/late support."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("COMMAND PATH DIAGNOSTICS SUMMARY")
        lines.append("=" * 80)
        lines.append("Tracing: q_ref (target) -> ctrl -> q_actual")
        lines.append(
            f"Phases: startup, early support (<{self.support_entry_window_s:.3f}s), "
            "late support (>= window)"
        )
        lines.append("Divergence threshold: |gap| > 0.100 rad")
        lines.append("")

        phase_order = (
            "STARTUP_SUPPORT_RAMP",
            "EARLY_SUPPORT_STABILIZE",
            "LATE_SUPPORT_STABILIZE",
        )
        for phase_name in phase_order:
            phase_samples = [s for s in self.samples if s["phase"] == phase_name]
            if not phase_samples:
                lines.append(f"Phase: {phase_name}")
                lines.append("  (No samples)")
                lines.append("")
                continue

            lines.append(f"Phase: {phase_name} ({len(phase_samples) // len(self._JOINTS)} steps)")
            for joint_name in self._JOINTS:
                joint_samples = [s for s in phase_samples if s["joint"] == joint_name]
                if not joint_samples:
                    continue
                q_ref = [float(s["q_ref"]) for s in joint_samples]
                q_ctrl = [float(s["q_ctrl"]) for s in joint_samples]
                q_act = [float(s["q_act"]) for s in joint_samples]
                ref_to_ctrl = [abs(float(s["ref_to_ctrl_gap"])) for s in joint_samples]
                ctrl_to_act = [abs(float(s["ctrl_to_act_gap"])) for s in joint_samples]
                ref_deltas = [abs(float(s["ref_delta"])) for s in joint_samples]
                ctrl_deltas = [abs(float(s["ctrl_delta"])) for s in joint_samples]

                first_ref_ctrl_div = next((int(s["step"]) for s in joint_samples if abs(float(s["ref_to_ctrl_gap"])) > 0.1), None)
                first_ctrl_act_div = next((int(s["step"]) for s in joint_samples if abs(float(s["ctrl_to_act_gap"])) > 0.1), None)

                lines.append(f"  {joint_name:11s}:")
                lines.append(
                    f"    target mean/max_abs={float(np.mean(q_ref)):+.3f}/{float(np.max(np.abs(q_ref))):.3f}  "
                    f"ctrl mean/max_abs={float(np.mean(q_ctrl)):+.3f}/{float(np.max(np.abs(q_ctrl))):.3f}  "
                    f"actual mean/max_abs={float(np.mean(q_act)):+.3f}/{float(np.max(np.abs(q_act))):.3f}"
                )
                lines.append(
                    f"    target->ctrl err mean/max={float(np.mean(ref_to_ctrl)):.3f}/{float(np.max(ref_to_ctrl)):.3f}  "
                    f"ctrl->actual err mean/max={float(np.mean(ctrl_to_act)):.3f}/{float(np.max(ctrl_to_act)):.3f}"
                )
                lines.append(
                    f"    target step Δ mean/max={float(np.mean(ref_deltas)):.3f}/{float(np.max(ref_deltas)):.3f} rad/step  "
                    f"max_rate={float(np.max(ref_deltas) / self.ctrl_dt):.3f} rad/s  "
                    f"max_Δctrl/step={float(np.max(ctrl_deltas)):.3f}"
                )
                lines.append(
                    "    first_divergence "
                    f"target->ctrl={first_ref_ctrl_div if first_ref_ctrl_div is not None else 'none'} "
                    f"ctrl->actual={first_ctrl_act_div if first_ctrl_act_div is not None else 'none'}"
                )
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class _StartupRouteTracker:
    """Tracks staged startup route A/W1/W2/W3/B transitions and posture realization."""

    _STAGE_ORDER = ("A", "W1", "W2", "W3", "B")
    _CHANNELS = (
        ("pelvis_height", "pelvis_height_ref", "root_height"),
        ("hip_roll", "hip_roll_ref", "hip_roll_act"),
        ("hip_pitch", "hip_pitch_ref", "hip_pitch_act"),
        ("knee_pitch", "knee_pitch_ref", "knee_pitch_act"),
        ("ankle_pitch", "ankle_pitch_ref", "ankle_pitch_act"),
    )

    def __init__(self) -> None:
        self.current_stage: str | None = None
        self.current_stage_id: int | None = None
        self.current_entry_step: int | None = None
        self.stage_records: dict[str, dict[str, object]] = {}
        self.transitions: list[dict[str, object]] = []

    def _ensure_stage(self, stage: str, entry_step: int) -> None:
        if stage not in self.stage_records:
            self.stage_records[stage] = {
                "entry_step": int(entry_step),
                "exit_step": None,
                "samples": [],
            }

    def update(
        self,
        *,
        step: int,
        stage_id: int,
        reason_id: int,
        mode_id: int,
        posture: dict[str, float | str],
    ) -> str | None:
        stage = _route_stage_label(stage_id)
        reason = _route_reason_label(reason_id)
        transition_line = None
        if stage != self.current_stage:
            prev_stage = self.current_stage
            prev_stage_id = self.current_stage_id
            if prev_stage is not None and prev_stage in self.stage_records:
                self.stage_records[prev_stage]["exit_step"] = int(step - 1)
                prev_entry = int(self.stage_records[prev_stage]["entry_step"])
                dwell = int(max((step - 1) - prev_entry + 1, 0))
                if reason == "NONE":
                    if prev_stage_id is not None and stage_id > prev_stage_id:
                        reason = "ADVANCE"
                    elif prev_stage_id is not None and stage_id < prev_stage_id:
                        reason = "REGRESS"
                self.transitions.append(
                    {
                        "from": prev_stage,
                        "to": stage,
                        "step": int(step),
                        "reason": reason,
                        "dwell": dwell,
                        "mode_id": int(mode_id),
                    }
                )
                transition_line = (
                    "[nominal-viewer] startup-route transition "
                    f"{prev_stage}->{stage} at step={step} "
                    f"reason={reason} dwell={dwell}"
                )
            self.current_stage = stage
            self.current_stage_id = int(stage_id)
            self.current_entry_step = int(step)
            self._ensure_stage(stage, step)

        self._ensure_stage(stage, step)
        sample = {"step": int(step), "mode_id": int(mode_id)}
        for _, ref_key, act_key in self._CHANNELS:
            sample[ref_key] = float(posture[ref_key])
            sample[act_key] = float(posture[act_key])
        self.stage_records[stage]["samples"].append(sample)
        return transition_line

    def finalize(self, final_step: int) -> None:
        if self.current_stage is not None and self.current_stage in self.stage_records:
            if self.stage_records[self.current_stage]["exit_step"] is None:
                self.stage_records[self.current_stage]["exit_step"] = int(final_step)

    def get_summary(self) -> str:
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("STARTUP ROUTE STAGE SUMMARY (A -> W1 -> W2 -> W3 -> B)")
        lines.append("=" * 80)
        if self.transitions:
            lines.append("Transitions:")
            for t in self.transitions:
                lines.append(
                    f"  step={int(t['step']):04d} {t['from']}->{t['to']} "
                    f"reason={t['reason']} dwell={int(t['dwell'])} mode={int(t['mode_id'])}"
                )
        else:
            lines.append("Transitions: none")
        lines.append("")
        for stage in self._STAGE_ORDER:
            rec = self.stage_records.get(stage)
            if rec is None:
                continue
            entry = int(rec["entry_step"])
            exit_step = int(rec["exit_step"]) if rec["exit_step"] is not None else entry
            dwell = max(exit_step - entry + 1, 0)
            samples = rec["samples"]
            lines.append(f"Stage {stage}: entry={entry} exit={exit_step} dwell={dwell} samples={len(samples)}")
            if not samples:
                lines.append("  (No samples)")
                lines.append("")
                continue
            for channel_name, ref_key, act_key in self._CHANNELS:
                refs = np.asarray([float(s[ref_key]) for s in samples], dtype=np.float32)
                acts = np.asarray([float(s[act_key]) for s in samples], dtype=np.float32)
                errs = np.abs(acts - refs)
                lines.append(
                    f"  {channel_name:12s}: "
                    f"target_mean={float(np.mean(refs)):+.3f} actual_mean={float(np.mean(acts)):+.3f} "
                    f"err_mean/max={float(np.mean(errs)):.3f}/{float(np.max(errs)):.3f}"
                )
            lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


def run_nominal_viewer(args: argparse.Namespace) -> int:
    cfg = load_training_config(args.config)
    original_action_filter_alpha = float(cfg.env.action_filter_alpha)
    original_startup_target_rate_rad_s = float(cfg.env.loc_ref_v2_startup_target_rate_design_rad_s)
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)
    _configure_nominal_only(
        cfg,
        forward_cmd=args.forward_cmd,
        horizon=args.horizon,
        disable_action_filter=args.disable_action_filter,
        startup_target_rate_deg_s=args.startup_target_rate_deg_s,
    )
    action_filter_alpha = float(cfg.env.action_filter_alpha)
    startup_target_rate_rad_s = float(cfg.env.loc_ref_v2_startup_target_rate_design_rad_s)
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
    if args.disable_action_filter:
        print(
            "[nominal-viewer] action filter override active: "
            f"alpha {original_action_filter_alpha:.3f} -> {action_filter_alpha:.3f}"
        )
    else:
        print(f"[nominal-viewer] action filter from config: alpha={action_filter_alpha:.3f}")
    if args.startup_target_rate_deg_s is not None:
        print(
            "[nominal-viewer] startup/support-entry target-rate override active: "
            f"{np.degrees(original_startup_target_rate_rad_s):.1f} -> "
            f"{np.degrees(startup_target_rate_rad_s):.1f} deg/s"
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
    renderer = _try_create_renderer(mj_model) if args.record else None
    if args.record and renderer is None:
        print("[nominal-viewer] recording disabled due to renderer failure")
    term_totals = {"term/height_low": 0.0, "term/pitch": 0.0, "term/roll": 0.0}
    done_reached = False
    done_step = -1
    done_dom = "none"
    ctrl_dt = float(cfg.env.ctrl_dt)
    print_every = max(1, int(args.print_every))
    stage_tracker = _StageTracker()
    startup_route_tracker = _StartupRouteTracker()
    cmd_path_tracker = _CommandPathTracker(
        ctrl_dt=ctrl_dt,
        support_entry_window_s=float(env._walking_ref_v2_cfg.support_entry_shaping_window_s),  # noqa: SLF001
    )

    # Pure-hold mode: when --force-support-only --init-from-nominal-qref,
    # bypass env.step() entirely and run pure MuJoCo with frozen ctrl.
    # This isolates servo capability from env/reference/adapter logic.
    # Also uses pure MuJoCo init (not MJX) to avoid self-collision
    # explosions that occur when MJX initializes deep squat postures.
    use_pure_hold = args.force_support_only and (args.init_from_nominal_qref or args.startup_settle_steps > 0)
    if use_pure_hold:
        _wr = state.info[WR_INFO_KEY]
        _nominal_q_ref = np.asarray(jnp.asarray(_wr.nominal_q_ref, dtype=jnp.float32), dtype=np.float64)
        _actuator_qpos_addrs = env._actuator_qpos_addrs  # noqa: SLF001
        sim_substeps = max(1, int(round(ctrl_dt / float(mj_model.opt.timestep))))
        _jnt_addr = lambda n: int(mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)])

        # Compute B posture: keyframe with leg joints from nominal_q_ref
        _qpos_B = np.array(mj_model.key_qpos[0], dtype=np.float64) if mj_model.nkey > 0 else mj_data.qpos.copy()
        _leg_joint_names = [
            "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch",
            "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch",
        ]
        for _jname in _leg_joint_names:
            _jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
            _addr = int(mj_model.jnt_qposadr[_jid])
            for _ei, _ea in enumerate(_actuator_qpos_addrs):
                if int(_ea) == _addr:
                    _qpos_B[_addr] = float(_nominal_q_ref[_ei])
                    break

        settle_steps = args.startup_settle_steps
        if settle_steps > 0 and not args.init_from_nominal_qref:
            # Smooth A→B transition will happen inside run_step (visible in viewer)
            print(f"[nominal-viewer] SMOOTH A→B TRANSITION: {settle_steps} steps, then pure hold")
            _qpos_A = np.array(mj_model.key_qpos[0], dtype=np.float64) if mj_model.nkey > 0 else mj_data.qpos.copy()

            # Start from A (standing keyframe)
            mj_data.qpos[:] = _qpos_A
            mj_data.qvel[:] = 0.0
            for _i in range(mj_model.nu):
                _jname = mj_model.actuator(_i).name
                _jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
                if _jid >= 0:
                    mj_data.ctrl[_i] = mj_data.qpos[int(mj_model.jnt_qposadr[_jid])]
            mujoco.mj_forward(mj_model, mj_data)

            # hold_ctrl will be set after transition completes
            hold_ctrl = None

        else:
            # Instant init: set joints directly to B (original pure-hold)
            print("[nominal-viewer] PURE HOLD MODE: bypassing env.step(), running raw MuJoCo with frozen ctrl")
            if mj_model.nkey > 0:
                mj_data.qpos[:] = mj_model.key_qpos[0]
            mj_data.qvel[:] = 0.0
            for _jname in _leg_joint_names:
                _jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
                _addr = int(mj_model.jnt_qposadr[_jid])
                mj_data.qpos[_addr] = _qpos_B[_addr]
            mujoco.mj_forward(mj_model, mj_data)

            # Ground
            _foot_geoms = ["left_heel", "left_toe", "right_heel", "right_toe"]
            _min_z = min(
                mj_data.geom_xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][2]
                - mj_model.geom_size[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][0]
                for g in _foot_geoms
            )
            mj_data.qpos[2] -= _min_z
            mj_data.qvel[:] = 0.0

        # Set ctrl = current joint positions (hold the final posture)
        for _i in range(mj_model.nu):
            _jname = mj_model.actuator(_i).name
            _jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
            if _jid >= 0:
                mj_data.ctrl[_i] = mj_data.qpos[int(mj_model.jnt_qposadr[_jid])]
        mujoco.mj_forward(mj_model, mj_data)

        # For instant-init, hold_ctrl is ready now.
        # For settle transition, hold_ctrl stays None until transition completes inside run_step.
        if hold_ctrl is not None or settle_steps <= 0:
            hold_ctrl = mj_data.ctrl.copy()

        _root_h = float(mj_data.qpos[2])
        _lk_q = float(mj_data.qpos[_jnt_addr("left_knee_pitch")])
        _rk_q = float(mj_data.qpos[_jnt_addr("right_knee_pitch")])
        print(
            f"[nominal-viewer] hold init: "
            f"root_h={_root_h:.3f} Lk={_lk_q:.3f} Rk={_rk_q:.3f}"
        )

    def run_step(step_idx: int) -> bool:
        nonlocal state, done_reached, done_step, done_dom, hold_ctrl

        if use_pure_hold:
            # During A→B transition: interpolate qpos and ctrl each step
            if hold_ctrl is None and settle_steps > 0 and step_idx <= settle_steps:
                alpha = step_idx / settle_steps
                for _jname in _leg_joint_names:
                    _jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
                    _addr = int(mj_model.jnt_qposadr[_jid])
                    _q_interp = (1.0 - alpha) * _qpos_A[_addr] + alpha * _qpos_B[_addr]
                    mj_data.qpos[_addr] = _q_interp
                    _aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, _jname)
                    if _aid >= 0:
                        mj_data.ctrl[_aid] = _q_interp
                if step_idx <= 3 or step_idx % 20 == 0:
                    _dbg_lk = float(mj_data.qpos[_jnt_addr("left_knee_pitch")])
                    _dbg_rk = float(mj_data.qpos[_jnt_addr("right_knee_pitch")])
                    _dbg_lk_aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_pitch")
                    _dbg_ctrl = float(mj_data.ctrl[_dbg_lk_aid]) if _dbg_lk_aid >= 0 else -999
                    print(f"  [A→B] step={step_idx} alpha={alpha:.2f} qpos_Lk={_dbg_lk:.4f} qpos_Rk={_dbg_rk:.4f} ctrl[{_dbg_lk_aid}]={_dbg_ctrl:.4f}")
                mj_data.qvel[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
                # Re-ground: adjust root z so feet stay on the ground
                _foot_geoms = ["left_heel", "left_toe", "right_heel", "right_toe"]
                _min_z = min(
                    mj_data.geom_xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][2]
                    - mj_model.geom_size[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)][0]
                    for g in _foot_geoms
                )
                mj_data.qpos[2] -= _min_z
                mujoco.mj_forward(mj_model, mj_data)
                for _ in range(sim_substeps):
                    mujoco.mj_step(mj_model, mj_data)
                # Finalize hold_ctrl when transition completes
                if step_idx == settle_steps:
                    for _i in range(mj_model.nu):
                        _jname_i = mj_model.actuator(_i).name
                        _jid_i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, _jname_i)
                        if _jid_i >= 0:
                            mj_data.ctrl[_i] = mj_data.qpos[int(mj_model.jnt_qposadr[_jid_i])]
                    hold_ctrl = mj_data.ctrl.copy()
                    print(
                        f"[nominal-viewer] A→B complete at step {step_idx}: "
                        f"root_h={mj_data.qpos[2]:.3f} "
                        f"Lk={mj_data.qpos[_jnt_addr('left_knee_pitch')]:.3f} "
                        f"Rk={mj_data.qpos[_jnt_addr('right_knee_pitch')]:.3f}"
                    )
            else:
                # Pure hold: frozen ctrl
                for _ in range(sim_substeps):
                    mujoco.mj_step(mj_model, mj_data)
            # Extract pitch for termination check
            qw, qx, qy, qz = mj_data.qpos[3:7]
            pitch = float(np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2)))
            root_h = float(mj_data.qpos[2])
            done = abs(pitch) > 0.82 or root_h < 0.15
            if (step_idx % print_every == 0) or done or (step_idx == 1):
                # Print diagnostics using MuJoCo joint names (not env indices)
                _lk_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_pitch")
                _rk_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_pitch")
                _la_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch")
                _lk_aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_pitch")
                _rk_aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_pitch")
                lk_q = float(mj_data.qpos[mj_model.jnt_qposadr[_lk_jid]])
                rk_q = float(mj_data.qpos[mj_model.jnt_qposadr[_rk_jid]])
                la_q = float(mj_data.qpos[mj_model.jnt_qposadr[_la_jid]])
                lk_f = float(mj_data.actuator_force[_lk_aid]) if _lk_aid >= 0 else 0.0
                rk_f = float(mj_data.actuator_force[_rk_aid]) if _rk_aid >= 0 else 0.0
                _lk_ctrl = float(mj_data.ctrl[_lk_aid]) if _lk_aid >= 0 else 0.0
                _rk_ctrl = float(mj_data.ctrl[_rk_aid]) if _rk_aid >= 0 else 0.0
                _hold_lk = float(hold_ctrl[_lk_aid]) if hold_ctrl is not None and _lk_aid >= 0 else _lk_ctrl
                _hold_rk = float(hold_ctrl[_rk_aid]) if hold_ctrl is not None and _rk_aid >= 0 else _rk_ctrl
                line = (
                    f"step={step_idx:04d} "
                    f"root_h={root_h:.4f} pitch={pitch:+.4f} "
                    f"Lk_ctrl={_lk_ctrl:+.4f} Lk_q={lk_q:+.4f} Lk_F={lk_f:+.3f}  "
                    f"Rk_ctrl={_rk_ctrl:+.4f} Rk_q={rk_q:+.4f} Rk_F={rk_f:+.3f}  "
                    f"La_q={la_q:+.4f}"
                )
                if args.log:
                    _print_log_only(line)
                else:
                    print(line)
            if renderer is not None:
                renderer.update_scene(mj_data)
                frames.append(renderer.render())
            if done:
                done_reached = True
                done_step = step_idx
                done_dom = "term/pitch" if abs(pitch) > 0.82 else "term/height_low"
                print(f"[nominal-viewer] done at step={done_step} dominant_termination={done_dom}")
                if args.stop_on_done:
                    return False
            return True

        # Normal env-based stepping (non-pure-hold mode)
        state = step_once(state)
        _sync_viewer_data(mj_model, mj_data, state.data)
        metrics_vec = state.metrics[METRICS_VEC_KEY]
        
        # Update stage tracker
        wr_info = state.info[WR_INFO_KEY]
        mode_id = int(jnp.asarray(wr_info.loc_ref_mode_id, dtype=jnp.int32))
        mode_time_s = float(jnp.asarray(wr_info.loc_ref_mode_time, dtype=jnp.float32))
        route_stage_id = int(
            jnp.asarray(
                getattr(wr_info, "loc_ref_startup_route_stage_id", int(StartupRouteStage.B)),
                dtype=jnp.int32,
            )
        )
        route_reason_id = int(
            jnp.asarray(
                getattr(wr_info, "loc_ref_startup_route_transition_reason", int(StartupRouteReason.NONE)),
                dtype=jnp.int32,
            )
        )
        pitch = float(metrics_vec[METRIC_INDEX["debug/pitch"]])
        pitch_rate = float(metrics_vec[METRIC_INDEX["debug/pitch_rate"]])
        posture = _extract_support_posture(env, state, metrics_vec)
        stage_tracker.update(step_idx, mode_id, pitch, pitch_rate, posture)
        transition_line = startup_route_tracker.update(
            step=step_idx,
            stage_id=route_stage_id,
            reason_id=route_reason_id,
            mode_id=mode_id,
            posture=posture,
        )
        if transition_line is not None:
            if args.log:
                _print_log_only(transition_line)
            else:
                print(transition_line)
        cmd_path_tracker.update(step_idx, mode_id, mode_time_s, posture)
        
        for key in term_totals:
            term_totals[key] += float(metrics_vec[METRIC_INDEX[key]])
        done = bool(np.asarray(state.done > 0.5))
        if (step_idx % print_every == 0) or done or (step_idx == 1):
            step_line = _format_step_line(step_idx, state, metrics_vec)
            posture_line = _format_support_posture_line(env, state, metrics_vec)
            if args.log:
                _print_log_only(step_line)
                _print_log_only(posture_line)
            else:
                print(step_line)
                print(posture_line)
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

    use_headless = args.headless
    if not use_headless and not _has_display():
        print(
            "[nominal-viewer] no display detected; switching to --headless. "
            "Set DISPLAY or WAYLAND_DISPLAY to use the interactive viewer."
        )
        use_headless = True

    if use_headless:
        for step_idx in range(1, int(args.horizon) + 1):
            if not run_step(step_idx):
                break
    else:
        viewer = _try_launch_viewer(mj_model, mj_data)
        if viewer is None:
            # Fallback: run headless
            for step_idx in range(1, int(args.horizon) + 1):
                if not run_step(step_idx):
                    break
        else:
            with viewer:
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

    # Finalize stage tracking
    final_step = done_step if done_reached else args.horizon
    stage_tracker.finalize(final_step)
    startup_route_tracker.finalize(final_step)
    
    if done_reached:
        print(f"[nominal-viewer] summary: done_step={done_step} dominant_termination={done_dom}")
    else:
        dom = max(term_totals.items(), key=lambda x: x[1])[0] if max(term_totals.values()) > 0.0 else "none"
        print(
            f"[nominal-viewer] summary: done_step=none dominant_termination={dom} "
            f"(horizon={args.horizon})"
        )
    
    # Print stage diagnostics
    stage_summary = stage_tracker.get_summary()
    if args.log:
        _print_log_only(stage_summary)
    else:
        print(stage_summary)
    
    # Print command-path diagnostics
    cmd_path_summary = cmd_path_tracker.get_summary()
    if args.log:
        _print_log_only(cmd_path_summary)
    else:
        print(cmd_path_summary)
    startup_route_summary = startup_route_tracker.get_summary()
    if args.log:
        _print_log_only(startup_route_summary)
    else:
        print(startup_route_summary)
    
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.log:
        _enable_temp_logging()
    # Auto-generate record path if --record is used without a filename.
    if args.record is not None and args.record == "":
        import tempfile
        args.record = os.path.join(
            tempfile.gettempdir(),
            f"wildrobot_nominal_{int(time.time())}.mp4",
        )
    if args.record is not None:
        print(f"[nominal-viewer] record file: {args.record}")
    return run_nominal_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
