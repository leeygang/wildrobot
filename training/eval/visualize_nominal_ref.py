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
from control.references.walking_ref_v2 import WalkingRefV2Mode


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
    parser.add_argument(
        "--disable-action-filter",
        action="store_true",
        help="Debug-only: force env.action_filter_alpha=0.0 for this nominal viewer run",
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
    """Track command path signals (q_ref → ctrl → q_actual) for startup/support diagnosis."""
    
    def __init__(self):
        self.samples = []
        self.prev_ctrl = None
    
    def update(self, step: int, mode_id: int, posture: dict):
        """Update tracking with current step data."""
        # Extract command-path signals for each joint
        for joint in ["hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch"]:
            q_ref = posture[f"{joint}_ref"]
            q_ctrl = posture[f"{joint}_ctrl"]
            q_act = posture[f"{joint}_act"]
            ref_err = posture[f"{joint}_ref_err"]
            ctrl_err = posture[f"{joint}_ctrl_err"]
            
            # Compute per-step ctrl delta
            if self.prev_ctrl is None or joint not in self.prev_ctrl:
                ctrl_delta = 0.0
            else:
                ctrl_delta = q_ctrl - self.prev_ctrl[joint]
            
            self.samples.append({
                "step": step,
                "mode": mode_id,
                "joint": joint,
                "q_ref": q_ref,
                "q_ctrl": q_ctrl,
                "q_act": q_act,
                "ref_to_ctrl_gap": q_ctrl - q_ref,
                "ctrl_to_act_gap": q_act - q_ctrl,
                "ctrl_delta": ctrl_delta,
            })
        
        # Store current ctrl for next delta computation
        self.prev_ctrl = {
            joint: posture[f"{joint}_ctrl"]
            for joint in ["hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch"]
        }
    
    def get_summary(self, startup_mode: int = 0, support_mode: int = 1) -> str:
        """Generate command-path summary for startup/support modes."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("COMMAND PATH DIAGNOSTICS SUMMARY")
        lines.append("=" * 80)
        lines.append("Tracing: q_ref (nominal) → ctrl (final target) → q_actual")
        lines.append("")
        
        # Filter to startup and support modes
        startup_samples = [s for s in self.samples if s["mode"] == startup_mode]
        support_samples = [s for s in self.samples if s["mode"] == support_mode]
        
        for mode_name, samples in [("STARTUP_SUPPORT_RAMP", startup_samples), ("SUPPORT_STABILIZE", support_samples)]:
            if not samples:
                lines.append(f"Stage: {mode_name}")
                lines.append("  (No samples)")
                lines.append("")
                continue
            
            lines.append(f"Stage: {mode_name} ({len(samples) // 4} steps)")
            
            for joint_name in ["hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch"]:
                joint_samples = [s for s in samples if s["joint"] == joint_name]
                if not joint_samples:
                    continue
                
                ref_to_ctrl_gaps = [abs(s["ref_to_ctrl_gap"]) for s in joint_samples]
                ctrl_to_act_gaps = [abs(s["ctrl_to_act_gap"]) for s in joint_samples]
                ctrl_deltas = [abs(s["ctrl_delta"]) for s in joint_samples]
                
                mean_ref_to_ctrl = sum(ref_to_ctrl_gaps) / len(ref_to_ctrl_gaps)
                max_ref_to_ctrl = max(ref_to_ctrl_gaps)
                mean_ctrl_to_act = sum(ctrl_to_act_gaps) / len(ctrl_to_act_gaps)
                max_ctrl_to_act = max(ctrl_to_act_gaps)
                max_ctrl_delta = max(ctrl_deltas)
                
                # Find first major divergence (>0.1 rad)
                first_ref_ctrl_div = next((s["step"] for s in joint_samples if abs(s["ref_to_ctrl_gap"]) > 0.1), None)
                first_ctrl_act_div = next((s["step"] for s in joint_samples if abs(s["ctrl_to_act_gap"]) > 0.1), None)
                
                lines.append(f"  {joint_name:12s}:")
                lines.append(f"    ref→ctrl: mean={mean_ref_to_ctrl:.3f} max={max_ref_to_ctrl:.3f} " +
                           (f"[diverge@step{first_ref_ctrl_div}]" if first_ref_ctrl_div else "[OK]"))
                lines.append(f"    ctrl→act: mean={mean_ctrl_to_act:.3f} max={max_ctrl_to_act:.3f} " +
                           (f"[diverge@step{first_ctrl_act_div}]" if first_ctrl_act_div else "[OK]"))
                lines.append(f"    max_Δctrl/step: {max_ctrl_delta:.3f} rad/step")
            
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


def run_nominal_viewer(args: argparse.Namespace) -> int:
    cfg = load_training_config(args.config)
    original_action_filter_alpha = float(cfg.env.action_filter_alpha)
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = project_root / robot_cfg_path
    load_robot_config(robot_cfg_path)
    _configure_nominal_only(
        cfg,
        forward_cmd=args.forward_cmd,
        horizon=args.horizon,
        disable_action_filter=args.disable_action_filter,
    )
    action_filter_alpha = float(cfg.env.action_filter_alpha)
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
    cmd_path_tracker = _CommandPathTracker()

    def run_step(step_idx: int) -> bool:
        nonlocal state, done_reached, done_step, done_dom
        state = step_once(state)
        _sync_viewer_data(mj_model, mj_data, state.data)
        metrics_vec = state.metrics[METRICS_VEC_KEY]
        
        # Update stage tracker
        wr_info = state.info[WR_INFO_KEY]
        mode_id = int(jnp.asarray(wr_info.loc_ref_mode_id, dtype=jnp.int32))
        pitch = float(metrics_vec[METRIC_INDEX["debug/pitch"]])
        pitch_rate = float(metrics_vec[METRIC_INDEX["debug/pitch_rate"]])
        posture = _extract_support_posture(env, state, metrics_vec)
        stage_tracker.update(step_idx, mode_id, pitch, pitch_rate, posture)
        cmd_path_tracker.update(step_idx, mode_id, posture)
        
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
    
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.log:
        _enable_temp_logging()
    return run_nominal_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
