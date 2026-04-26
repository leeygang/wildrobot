#!/usr/bin/env python3
"""Compare WildRobot and ToddlerBot reference quality on shared gates.

This script keeps the original shared P0 geometry gate and adds:

- FK-realized nominal gait-shape metrics at ``vx=0.15`` by default
- P2 smoothness metrics on the reference asset itself
- P1 free-base zero-residual replay metrics across the walking bins

Usage::

    uv run ./tools/reference_geometry_parity.py
    uv run ./tools/reference_geometry_parity.py --nominal-only
    uv run ./tools/reference_geometry_parity.py --json
    uv run ./tools/reference_geometry_parity.py --strict
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.zmp.zmp_walk import ZMPWalkConfig, ZMPWalkGenerator
from training.utils.ctrl_order import CtrlOrderMapper


_STANCE_TOL_M = 0.003
_SWING_MIN_M = -0.002
_PARITY_MARGIN_M = 0.002
_VX_BINS = (0.10, 0.15, 0.20, 0.25)
_VX_BINS_INSCOPE = (0.10, 0.15)
_P1_VX_BINS = (0.10, 0.15, 0.20)
_PROBE_FRAMES = (0, 5, 10, 16, 22, 28, 32, 48, 64)
_TB_VARIANTS = ("toddlerbot_2xc", "toddlerbot_2xm")
# Shared leg-joint ordering used by every parity layer that compares
# WR vs TB on the 8 leg DoFs (P1A foot trajectories, P2 q_ref smoothness,
# P1 closed-loop RMSE).  The order MUST stay aligned with WR's
# ``traj.q_ref[:, :8]`` slice produced by ``ZMPWalkGenerator`` (indices
# [0..7] = hip_pitch_L, hip_pitch_R, hip_roll_L, hip_roll_R, knee_L,
# knee_R, ankle_pitch_L, ankle_pitch_R).  Reordering one tuple without
# the other will silently swap channels at parity time.
_WR_SHARED_LEG_NAMES = (
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_knee_pitch",
    "right_knee_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
)
_TB_SHARED_LEG_NAMES = (
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_knee",
    "right_knee",
    "left_ankle_pitch",
    "right_ankle_pitch",
)
# P1 closed-loop replay uses one shared physical-failure floor so WR and
# TB are judged against the same termination definition.  These values
# are the WR env's working thresholds — TB's PPO walk env terminates only
# on height (per the A.1 audit in walking_training.md), so applying
# pitch/roll caps to TB is a stricter floor than its training env uses.
# The intent is "same definition of physical failure", not "TB's training
# env defaults".  Documented as the same caveat in
# training/docs/reference_parity_scorecard.md § P1.
_SHARED_PITCH_FAIL_RAD = 0.8
_SHARED_ROLL_FAIL_RAD = 0.6
_WR_ROOT_Z_FAIL_M = 0.15
_TB_TORSO_Z_FAIL_M = 0.15
_SAT_MARGIN_RAD = 0.01

# Characteristic robot dimensions for size-normalised parity (Option A).
# The absolute parity gates above stay in metric units for sanity; the
# normalised companion gates below use these dims to put WR and TB on
# the same Froude-equivalent footing despite WR being ~1.6-1.8x larger.
#
# Sources:
#   WR  leg_length_m = ZMPWalkConfig.upper_leg_m + lower_leg_m  (0.193 + 0.180)
#   WR  com_height_m = ZMPWalkConfig.com_height_m at default cfg (~0.462 m)
#   TB-2xc / TB-2xm leg_length_m = robot.yml hip_to_ankle_pitch_z
#   TB-2xc / TB-2xm com_height_m = robot.yml com_z
# Both TB variants share the same kinematics today; if a future TB build
# diverges, split the entry below.
_GRAVITY_MPS2 = 9.81


def _wr_dims() -> dict[str, float]:
    """WildRobot characteristic dimensions, computed from the live config
    so any change to ``ZMPWalkConfig`` (e.g. leg refit) propagates here
    without manual sync.
    """
    cfg = ZMPWalkConfig()
    return {
        "leg_length_m": float(cfg.upper_leg_m + cfg.lower_leg_m),
        "com_height_m": float(cfg.com_height_m),
    }


_TB_DIMS: dict[str, dict[str, float]] = {
    "toddlerbot_2xc": {"leg_length_m": 0.2115, "com_height_m": 0.2859},
    "toddlerbot_2xm": {"leg_length_m": 0.2115, "com_height_m": 0.2859},
}


def _robot_dims(name: str) -> dict[str, float]:
    if name == "wildrobot":
        return _wr_dims()
    if name in _TB_DIMS:
        return _TB_DIMS[name]
    return {"leg_length_m": float("nan"), "com_height_m": float("nan")}


def _safe_div(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(numer) / float(denom)


@dataclass
class GateSummary:
    name: str
    total_failures: int
    in_scope_failures: int
    worst_stance_z: float
    worst_swing_z: float
    failures: list[str]


@dataclass
class FKGaitMetrics:
    name: str
    vx: float
    duration_s: float
    touchdowns_left: int
    touchdowns_right: int
    touchdown_balance_ratio: float
    touchdown_rate_hz: float
    step_length_mean_m: float
    step_length_min_m: float
    step_length_max_m: float
    touchdown_speed_proxy_mps: float
    swing_clearance_mean_m: float
    swing_clearance_min_m: float
    double_support_frac: float
    foot_z_step_max_m: float


@dataclass
class SmoothnessMetrics:
    name: str
    vx: float
    shared_leg_q_step_max_rad: float
    pelvis_z_step_max_m: float
    swing_foot_z_step_max_m: float
    contact_flips_per_cycle: float


@dataclass
class NormalizedGaitMetrics:
    """Size-normalised companion to ``FKGaitMetrics`` / ``ClosedLoopMetrics``.

    Lets WR (leg ~0.37 m, COM ~0.46 m) be compared fairly with TB
    (leg ~0.21 m, COM ~0.29 m).  Computed post-hoc from the absolute
    metrics + the robot's characteristic dimensions; absolute fields are
    untouched so existing tooling keeps working.

    All fields are dimensionless except ``leg_length_m`` / ``com_height_m``
    which are recorded for reproducibility.
    """

    name: str
    vx: float
    leg_length_m: float
    com_height_m: float
    forward_speed_froude: float = float("nan")  # vx^2 / (g * h), characterises operating point
    # FK / nominal-asset normalisations (NaN if computed from a closed-loop row only)
    step_length_per_leg: float = float("nan")
    swing_clearance_per_com_height: float = float("nan")
    cadence_froude_norm: float = float("nan")  # td_rate * sqrt(h / g), dimensionless cadence
    swing_foot_z_step_per_clearance: float = float("nan")
    # Closed-loop normalisations (NaN for FK-derived rows)
    achieved_forward_per_cmd: float = float("nan")
    terminal_drift_rate_per_leg_per_s: float = float("nan")
    td_step_length_per_leg: float = float("nan")


@dataclass
class ClosedLoopMetrics:
    name: str
    vx: float
    horizon_steps: int
    survived_steps: int
    shared_leg_rmse_rad: float
    achieved_forward_mps: float
    ref_contact_match_frac: float
    touchdown_step_length_mean_m: float
    touchdown_step_length_min_m: float
    touchdown_rate_hz: float
    joint_limit_step_frac: float
    termination_mode: str
    terminal_root_x_m: float


def _repo_roots() -> tuple[Path, Path]:
    wildrobot_root = Path(__file__).resolve().parents[1]
    toddlerbot_root = wildrobot_root.parent / "toddlerbot"
    return wildrobot_root, toddlerbot_root


def _fmt_float(value: float, width: int = 10, precision: int = 4) -> str:
    if not np.isfinite(value):
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}"


def _pick_python_with_modules(
    candidates: list[Path | str], modules: list[str], cwd: Path
) -> str:
    check = (
        "import importlib\n"
        f"mods = {modules!r}\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "raise SystemExit(0 if not missing else 1)\n"
    )
    for candidate in candidates:
        candidate_str = str(candidate)
        if "/" in candidate_str and not Path(candidate_str).exists():
            continue
        result = subprocess.run(
            [candidate_str, "-c", check],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate_str
    raise RuntimeError(
        "Could not find a Python interpreter with required modules: "
        + ", ".join(modules)
    )


def _quat_to_pitch_roll(quat_wxyz: np.ndarray) -> tuple[float, float]:
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    pitch = np.arctan2(2.0 * (qw * qy - qz * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return float(pitch), float(roll)


def _safe_stat(arr: np.ndarray, fn) -> float:
    if arr.size == 0:
        return float("nan")
    return float(fn(arr))


def _touchdown_indices(contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_touchdowns = np.where((contact[:-1, 0] < 0.5) & (contact[1:, 0] > 0.5))[0] + 1
    right_touchdowns = np.where((contact[:-1, 1] < 0.5) & (contact[1:, 1] > 0.5))[0] + 1
    return left_touchdowns, right_touchdowns


def _combined_step_lengths(
    left_touchdowns: np.ndarray,
    right_touchdowns: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
) -> np.ndarray:
    step_lengths: list[float] = []
    for idx in left_touchdowns:
        step_lengths.append(float(left_foot_pos[idx, 0] - right_foot_pos[idx, 0]))
    for idx in right_touchdowns:
        step_lengths.append(float(right_foot_pos[idx, 0] - left_foot_pos[idx, 0]))
    return np.asarray(step_lengths, dtype=np.float64)


def _swing_clearance(foot_pos: np.ndarray, contact_col: np.ndarray) -> np.ndarray:
    starts = np.where((contact_col[:-1] > 0.5) & (contact_col[1:] < 0.5))[0] + 1
    ends = np.where((contact_col[:-1] < 0.5) & (contact_col[1:] > 0.5))[0] + 1
    if np.any(contact_col > 0.5):
        baseline = float(np.median(foot_pos[contact_col > 0.5, 2]))
    else:
        baseline = float(np.median(foot_pos[:, 2]))
    peaks: list[float] = []
    for start in starts:
        touchdown_candidates = ends[ends > start]
        if len(touchdown_candidates) == 0:
            continue
        end = int(touchdown_candidates[0])
        peaks.append(float(np.max(foot_pos[start : end + 1, 2]) - baseline))
    return np.asarray(peaks, dtype=np.float64)


def _estimate_cycle_time_s(contact: np.ndarray, dt: float) -> float:
    left_touchdowns, right_touchdowns = _touchdown_indices(contact)
    diffs: list[np.ndarray] = []
    if len(left_touchdowns) >= 2:
        diffs.append(np.diff(left_touchdowns).astype(np.float64) * dt)
    if len(right_touchdowns) >= 2:
        diffs.append(np.diff(right_touchdowns).astype(np.float64) * dt)
    if diffs:
        return float(np.median(np.concatenate(diffs, axis=0)))
    return float(len(contact) * dt)


def _compute_fk_gait_metrics(
    name: str,
    vx: float,
    contact: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    dt: float,
) -> FKGaitMetrics:
    left_touchdowns, right_touchdowns = _touchdown_indices(contact)
    step_arr = _combined_step_lengths(
        left_touchdowns, right_touchdowns, left_foot_pos, right_foot_pos
    )
    left_clearance = _swing_clearance(left_foot_pos, contact[:, 0])
    right_clearance = _swing_clearance(right_foot_pos, contact[:, 1])
    clearance_arr = np.concatenate([left_clearance, right_clearance], axis=0)
    duration_s = float(len(contact) * dt)
    total_touchdowns = len(left_touchdowns) + len(right_touchdowns)
    max_touchdowns = max(len(left_touchdowns), len(right_touchdowns), 1)
    min_touchdowns = min(len(left_touchdowns), len(right_touchdowns))
    foot_z_step_max_m = max(
        float(np.max(np.abs(np.diff(left_foot_pos[:, 2])))),
        float(np.max(np.abs(np.diff(right_foot_pos[:, 2])))),
    )
    step_length_mean_m = _safe_stat(step_arr, np.mean)
    touchdown_speed_proxy_mps = (
        float(step_length_mean_m * total_touchdowns / duration_s / 2.0)
        if np.isfinite(step_length_mean_m) and duration_s > 1e-8
        else float("nan")
    )
    return FKGaitMetrics(
        name=name,
        vx=vx,
        duration_s=duration_s,
        touchdowns_left=len(left_touchdowns),
        touchdowns_right=len(right_touchdowns),
        touchdown_balance_ratio=float(min_touchdowns / max_touchdowns),
        touchdown_rate_hz=float(total_touchdowns / duration_s),
        step_length_mean_m=step_length_mean_m,
        step_length_min_m=_safe_stat(step_arr, np.min),
        step_length_max_m=_safe_stat(step_arr, np.max),
        touchdown_speed_proxy_mps=touchdown_speed_proxy_mps,
        swing_clearance_mean_m=_safe_stat(clearance_arr, np.mean),
        swing_clearance_min_m=_safe_stat(clearance_arr, np.min),
        double_support_frac=float(np.mean(np.sum(contact, axis=1) == 2)),
        foot_z_step_max_m=foot_z_step_max_m,
    )


def _compute_swing_foot_z_step_max(
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    contact: np.ndarray,
) -> float:
    max_step = 0.0
    for foot_pos, contact_col in (
        (left_foot_pos, contact[:, 0]),
        (right_foot_pos, contact[:, 1]),
    ):
        if len(contact_col) < 2:
            continue
        swing_edge_mask = np.logical_or(contact_col[:-1] < 0.5, contact_col[1:] < 0.5)
        if np.any(swing_edge_mask):
            max_step = max(
                max_step,
                float(np.max(np.abs(np.diff(foot_pos[:, 2]))[swing_edge_mask])),
            )
    return max_step


def _compute_smoothness_metrics(
    name: str,
    vx: float,
    shared_leg_q_ref: np.ndarray,
    pelvis_z: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    contact: np.ndarray,
    dt: float,
) -> SmoothnessMetrics:
    # ``pelvis_z_step_max_m`` is a SENTINEL gate, not a variation metric.
    # Both ZMP-style priors structurally produce a constant commanded
    # pelvis (COM) z — WR pins ``traj.pelvis_pos[:, 2] = com_height_m``;
    # TB's mujoco_replay anchors the torso in fixed_base.  So both sides
    # legitimately report ~0 here.  A nonzero value would indicate a
    # planner discontinuity bug, which is what the gate is supposed to
    # catch.  Do NOT read a 0/0 PASS as "WR is smoother" — see the P2
    # note in training/docs/reference_parity_scorecard.md.
    q_step_max = float(np.max(np.abs(np.diff(shared_leg_q_ref, axis=0))))
    pelvis_z_step_max = float(np.max(np.abs(np.diff(pelvis_z))))
    swing_foot_z_step_max = _compute_swing_foot_z_step_max(
        left_foot_pos, right_foot_pos, contact
    )
    bit_flips = int(np.sum(contact[1:] != contact[:-1]))
    cycle_time_s = _estimate_cycle_time_s(contact, dt)
    duration_s = float(len(contact) * dt)
    n_cycles = max(duration_s / max(cycle_time_s, 1e-8), 1.0)
    return SmoothnessMetrics(
        name=name,
        vx=vx,
        shared_leg_q_step_max_rad=q_step_max,
        pelvis_z_step_max_m=pelvis_z_step_max,
        swing_foot_z_step_max_m=swing_foot_z_step_max,
        contact_flips_per_cycle=float(bit_flips / n_cycles),
    )


def _compute_normalized_from_fk(
    name: str,
    vx: float,
    gait: FKGaitMetrics,
    smooth: SmoothnessMetrics,
) -> NormalizedGaitMetrics:
    """Build the size-normalised companion to the FK / nominal-asset row."""
    dims = _robot_dims(name)
    L = dims["leg_length_m"]
    h = dims["com_height_m"]
    cadence_norm = (
        float(gait.touchdown_rate_hz) * math.sqrt(h / _GRAVITY_MPS2)
        if np.isfinite(gait.touchdown_rate_hz) and h > 0
        else float("nan")
    )
    forward_froude = (
        (vx * vx) / (_GRAVITY_MPS2 * h)
        if np.isfinite(vx) and h > 0
        else float("nan")
    )
    return NormalizedGaitMetrics(
        name=name,
        vx=vx,
        leg_length_m=L,
        com_height_m=h,
        forward_speed_froude=forward_froude,
        step_length_per_leg=_safe_div(gait.step_length_mean_m, L),
        swing_clearance_per_com_height=_safe_div(gait.swing_clearance_mean_m, h),
        cadence_froude_norm=cadence_norm,
        swing_foot_z_step_per_clearance=_safe_div(
            smooth.swing_foot_z_step_max_m, gait.swing_clearance_mean_m
        ),
    )


def _compute_normalized_from_closed_loop(
    metrics: ClosedLoopMetrics, ctrl_dt: float = 0.02
) -> NormalizedGaitMetrics:
    """Build the size-normalised companion to a P1 closed-loop row.

    Drift rate is the per-second pelvis x divided by leg length, so
    "WR drifted backward by 1.8 leg-lengths per second" reads cleanly
    against the same number on TB.
    """
    dims = _robot_dims(metrics.name)
    L = dims["leg_length_m"]
    h = dims["com_height_m"]
    duration_s = max(int(metrics.survived_steps) * ctrl_dt, 1e-8)
    drift_rate_mps = metrics.terminal_root_x_m / duration_s
    forward_froude = (
        (metrics.vx * metrics.vx) / (_GRAVITY_MPS2 * h)
        if np.isfinite(metrics.vx) and h > 0
        else float("nan")
    )
    cadence_norm = (
        float(metrics.touchdown_rate_hz) * math.sqrt(h / _GRAVITY_MPS2)
        if np.isfinite(metrics.touchdown_rate_hz) and h > 0
        else float("nan")
    )
    return NormalizedGaitMetrics(
        name=metrics.name,
        vx=metrics.vx,
        leg_length_m=L,
        com_height_m=h,
        forward_speed_froude=forward_froude,
        cadence_froude_norm=cadence_norm,
        achieved_forward_per_cmd=_safe_div(metrics.achieved_forward_mps, metrics.vx),
        terminal_drift_rate_per_leg_per_s=_safe_div(drift_rate_mps, L),
        td_step_length_per_leg=_safe_div(metrics.touchdown_step_length_mean_m, L),
    )


def _same_foot_touchdown_steps(xs: list[float]) -> np.ndarray:
    if len(xs) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.asarray(xs, dtype=np.float64)) / 2.0


def _combined_touchdown_step_stats(
    left_touchdown_x: list[float], right_touchdown_x: list[float]
) -> tuple[float, float]:
    combined = np.concatenate(
        [
            _same_foot_touchdown_steps(left_touchdown_x),
            _same_foot_touchdown_steps(right_touchdown_x),
        ],
        axis=0,
    )
    return _safe_stat(combined, np.mean), _safe_stat(combined, np.min)


def _box_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    pos = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom_id]
    return float(
        pos[2]
        - abs(rot[2, 0]) * sx
        - abs(rot[2, 1]) * sy
        - abs(rot[2, 2]) * sz
    )


def _load_wr_assets():
    wildrobot_root, _ = _repo_roots()
    model = mujoco.MjModel.from_xml_path(
        str(wildrobot_root / "assets" / "v2" / "scene_flat_terrain.xml")
    )
    data = mujoco.MjData(model)
    with open(wildrobot_root / "assets" / "v2" / "mujoco_robot_config.json") as f:
        spec = json.load(f)
    actuator_names = [j["name"] for j in spec["actuated_joint_specs"]]
    mapper = CtrlOrderMapper(model, actuator_names)
    act_to_qpos = np.array(
        [model.jnt_qposadr[model.actuator_trnid[k, 0]] for k in range(model.nu)]
    )
    geom_ids = {
        "left_heel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel"),
        "left_toe": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe"),
        "right_heel": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel"),
        "right_toe": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe"),
        "floor": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
    }
    body_ids = {
        "left_foot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"),
        "right_foot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"),
    }
    shared_joint_limits = np.array(
        [
            model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in _WR_SHARED_LEG_NAMES
        ],
        dtype=np.float64,
    )
    return model, data, mapper, act_to_qpos, geom_ids, body_ids, shared_joint_limits


def _init_wr_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for key_name in ("walk_start", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            break
    mujoco.mj_forward(model, data)


def _wr_set_reference_pose(
    data: mujoco.MjData,
    traj,
    idx: int,
    mapper: CtrlOrderMapper,
    act_to_qpos: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    mj_q = np.zeros(len(mapper.policy_to_mj_order), dtype=np.float64)
    mj_q[mapper.policy_to_mj_order] = traj.q_ref[idx]
    for actuator_idx in range(len(act_to_qpos)):
        data.qpos[act_to_qpos[actuator_idx]] = mj_q[actuator_idx]
    data.qpos[0:3] = traj.pelvis_pos[idx, :3]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]


def _wr_foot_center_pos(data: mujoco.MjData, geom_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    left = 0.5 * (
        np.asarray(data.geom_xpos[geom_ids["left_heel"]], dtype=np.float64)
        + np.asarray(data.geom_xpos[geom_ids["left_toe"]], dtype=np.float64)
    )
    right = 0.5 * (
        np.asarray(data.geom_xpos[geom_ids["right_heel"]], dtype=np.float64)
        + np.asarray(data.geom_xpos[geom_ids["right_toe"]], dtype=np.float64)
    )
    return left, right


def _wr_collision_geoms(model: mujoco.MjModel, body_id: int) -> set[int]:
    return {
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id and int(model.geom_contype[g]) != 0
    }


def _foot_floor_in_contact(
    data: mujoco.MjData, floor_geom_id: int, geom_ids: set[int]
) -> bool:
    for con_idx in range(data.ncon):
        con = data.contact[con_idx]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        if (g1 == floor_geom_id and g2 in geom_ids) or (g2 == floor_geom_id and g1 in geom_ids):
            return True
    return False


def _summarize_wildrobot() -> GateSummary:
    model, data, mapper, act_to_qpos, geom_ids, _body_ids, _limits = _load_wr_assets()
    left_geoms = [geom_ids["left_heel"], geom_ids["left_toe"]]
    right_geoms = [geom_ids["right_heel"], geom_ids["right_toe"]]
    failures: list[str] = []
    worst_stance_z = -1e9
    worst_swing_z = 1e9
    gen = ZMPWalkGenerator()
    for vx in _VX_BINS:
        traj = gen.generate(vx)
        for frame_idx in _PROBE_FRAMES:
            if frame_idx >= traj.n_steps:
                continue
            _wr_set_reference_pose(data, traj, frame_idx, mapper, act_to_qpos)
            mujoco.mj_forward(model, data)
            left_z = min(_box_min_z(model, data, geom_id) for geom_id in left_geoms)
            right_z = min(_box_min_z(model, data, geom_id) for geom_id in right_geoms)
            for side, z, contact in (
                ("L", left_z, float(traj.contact_mask[frame_idx, 0])),
                ("R", right_z, float(traj.contact_mask[frame_idx, 1])),
            ):
                if contact > 0.5:
                    worst_stance_z = max(worst_stance_z, z)
                    if z > _STANCE_TOL_M:
                        failures.append(
                            f"WR vx={vx:.2f} frame={frame_idx} {side} stance z={z:+.4f} > {_STANCE_TOL_M}"
                        )
                else:
                    worst_swing_z = min(worst_swing_z, z)
                    if z < _SWING_MIN_M:
                        failures.append(
                            f"WR vx={vx:.2f} frame={frame_idx} {side} swing z={z:+.4f} < {_SWING_MIN_M}"
                        )
    in_scope_failures = sum(
        any(f"vx={vx:.2f}" in failure for vx in _VX_BINS_INSCOPE)
        for failure in failures
    )
    return GateSummary(
        name="wildrobot",
        total_failures=len(failures),
        in_scope_failures=in_scope_failures,
        worst_stance_z=worst_stance_z,
        worst_swing_z=worst_swing_z,
        failures=failures,
    )


def _summarize_toddlerbot(toddlerbot_root: Path, robot_name: str) -> GateSummary:
    tb_python = _pick_python_with_modules(
        [
            toddlerbot_root / ".venv" / "bin" / "python",
            "python3",
            "/usr/bin/python3",
        ],
        ["joblib", "lz4", "mujoco", "numpy"],
        cwd=toddlerbot_root,
    )
    helper = f"""
import json
from pathlib import Path
import joblib
import mujoco
import numpy as np

STANCE_TOL = {_STANCE_TOL_M!r}
SWING_MIN = {_SWING_MIN_M!r}
VX_BINS = {_VX_BINS!r}
VX_BINS_INSCOPE = {_VX_BINS_INSCOPE!r}
PROBE_FRAMES = {_PROBE_FRAMES!r}
toddlerbot_root = Path({str(toddlerbot_root)!r})
robot_name = {robot_name!r}

def box_min_z(model, data, geom_id):
    pos = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom_id]
    return float(
        pos[2]
        - abs(rot[2, 0]) * sx
        - abs(rot[2, 1]) * sy
        - abs(rot[2, 2]) * sz
    )

scene_path = toddlerbot_root / "toddlerbot" / "descriptions" / robot_name / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
q_home = np.array(model.keyframe("home").qpos, dtype=np.float64)
suffix = "_2xc" if "2xc" in robot_name else "_2xm"
lookup_path = toddlerbot_root / "motion" / f"walk_zmp{{suffix}}.lz4"
lookup_keys, motion_ref_list = joblib.load(lookup_path)
lookup_keys = np.array(lookup_keys, dtype=np.float32)
left_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link_collision")
right_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link_collision")
failures = []
worst_stance_z = -1e9
worst_swing_z = 1e9
for vx in VX_BINS:
    target = np.array([vx, 0.0, 0.0], dtype=np.float32)
    idx = int(np.argmin(np.linalg.norm(lookup_keys - target, axis=1)))
    traj = motion_ref_list[idx]
    for frame_idx in PROBE_FRAMES:
        if frame_idx >= len(traj["qpos"]):
            continue
        data.qpos[:] = q_home
        data.qpos[7:] = np.asarray(traj["qpos"][frame_idx], dtype=np.float64)
        mujoco.mj_forward(model, data)
        left_z = box_min_z(model, data, left_geom)
        right_z = box_min_z(model, data, right_geom)
        for side, z, contact in (
            ("L", left_z, float(traj["contact"][frame_idx][0])),
            ("R", right_z, float(traj["contact"][frame_idx][1])),
        ):
            if contact > 0.5:
                worst_stance_z = max(worst_stance_z, z)
                if z > STANCE_TOL:
                    failures.append(
                        f"{{robot_name}} vx={{vx:.2f}} frame={{frame_idx}} {{side}} stance z={{z:+.4f}} > {{STANCE_TOL}}"
                    )
            else:
                worst_swing_z = min(worst_swing_z, z)
                if z < SWING_MIN:
                    failures.append(
                        f"{{robot_name}} vx={{vx:.2f}} frame={{frame_idx}} {{side}} swing z={{z:+.4f}} < {{SWING_MIN}}"
                    )

in_scope_failures = sum(
    any(f"vx={{vx:.2f}}" in failure for vx in VX_BINS_INSCOPE)
    for failure in failures
)
print(json.dumps({{
    "name": robot_name,
    "total_failures": len(failures),
    "in_scope_failures": in_scope_failures,
    "worst_stance_z": worst_stance_z,
    "worst_swing_z": worst_swing_z,
    "failures": failures,
}}))
"""
    result = subprocess.run(
        [tb_python, "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=toddlerbot_root,
    )
    return GateSummary(**json.loads(result.stdout))


def _wr_fk_and_smoothness(vx: float) -> tuple[FKGaitMetrics, SmoothnessMetrics]:
    traj = ZMPWalkGenerator().generate(vx)
    n_steps = traj.n_steps

    # Phase 3: prefer the asset's stored realized site_pos when present.
    # Falls back to per-frame re-FK only when the trajectory predates
    # Phase 3 (e.g. a pre-Phase-3 cached asset on disk).  The
    # ``site_names`` tuple gives stable indexing without baking integer
    # offsets into this tool.
    if (
        traj.site_pos is not None
        and traj.site_names is not None
        and "left_foot_center" in traj.site_names
        and "right_foot_center" in traj.site_names
    ):
        left_idx = traj.site_names.index("left_foot_center")
        right_idx = traj.site_names.index("right_foot_center")
        left_foot = np.asarray(traj.site_pos[:, left_idx, :], dtype=np.float64)
        right_foot = np.asarray(traj.site_pos[:, right_idx, :], dtype=np.float64)
    else:
        model, data, mapper, act_to_qpos, geom_ids, _body_ids, _limits = _load_wr_assets()
        left_foot = np.zeros((n_steps, 3), dtype=np.float64)
        right_foot = np.zeros((n_steps, 3), dtype=np.float64)
        for idx in range(n_steps):
            _wr_set_reference_pose(data, traj, idx, mapper, act_to_qpos)
            mujoco.mj_forward(model, data)
            left_foot[idx], right_foot[idx] = _wr_foot_center_pos(data, geom_ids)
    gait = _compute_fk_gait_metrics(
        "wildrobot",
        vx,
        np.asarray(traj.contact_mask, dtype=np.float32),
        left_foot,
        right_foot,
        float(traj.dt),
    )
    smooth = _compute_smoothness_metrics(
        "wildrobot",
        vx,
        np.asarray(traj.q_ref[:, :8], dtype=np.float32),
        np.asarray(traj.pelvis_pos[:, 2], dtype=np.float32),
        left_foot,
        right_foot,
        np.asarray(traj.contact_mask, dtype=np.float32),
        float(traj.dt),
    )
    return gait, smooth


def _tb_nominal_bundle(
    toddlerbot_root: Path, robot_name: str, vx: float
) -> tuple[FKGaitMetrics, SmoothnessMetrics]:
    tb_python = _pick_python_with_modules(
        [
            toddlerbot_root / ".venv" / "bin" / "python",
            "python3",
            "/usr/bin/python3",
        ],
        ["joblib", "lz4", "mujoco", "numpy"],
        cwd=toddlerbot_root,
    )
    helper = f"""
import json
from pathlib import Path
import joblib
import mujoco
import numpy as np

toddlerbot_root = Path({str(toddlerbot_root)!r})
robot_name = {robot_name!r}
vx = {vx!r}
shared_leg_names = {list(_TB_SHARED_LEG_NAMES)!r}
suffix = "_2xc" if "2xc" in robot_name else "_2xm"
lookup_keys, motion_ref_list = joblib.load(toddlerbot_root / "motion" / f"walk_zmp{{suffix}}.lz4")
lookup_keys = np.asarray(lookup_keys, dtype=np.float32)
idx = int(np.argmin(np.linalg.norm(lookup_keys - np.array([vx, 0.0, 0.0], np.float32), axis=1)))
traj = motion_ref_list[idx]
contact = np.asarray(traj["contact"], dtype=np.float32)
site_pos = np.asarray(traj["site_pos"], dtype=np.float32)
left_foot_pos = site_pos[:, 1, :]
right_foot_pos = site_pos[:, 3, :]
time = np.asarray(traj["time"], dtype=np.float32)
dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.02
fixed_model = mujoco.MjModel.from_xml_path(str(toddlerbot_root / "toddlerbot" / "descriptions" / robot_name / "scene_fixed.xml"))
shared_leg_q = np.stack([
    np.asarray(traj["qpos"], dtype=np.float32)[:, int(fixed_model.jnt_qposadr[mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)])]
    for name in shared_leg_names
], axis=1)

def touchdown_indices(contact):
    left = np.where((contact[:-1, 0] < 0.5) & (contact[1:, 0] > 0.5))[0] + 1
    right = np.where((contact[:-1, 1] < 0.5) & (contact[1:, 1] > 0.5))[0] + 1
    return left, right

def safe_stat(arr, fn):
    if arr.size == 0:
        return float("nan")
    return float(fn(arr))

def swing_clearance(foot_pos, contact_col):
    starts = np.where((contact_col[:-1] > 0.5) & (contact_col[1:] < 0.5))[0] + 1
    ends = np.where((contact_col[:-1] < 0.5) & (contact_col[1:] > 0.5))[0] + 1
    baseline = float(np.median(foot_pos[contact_col > 0.5, 2])) if np.any(contact_col > 0.5) else float(np.median(foot_pos[:, 2]))
    peaks = []
    for start in starts:
        td = ends[ends > start]
        if len(td) == 0:
            continue
        peaks.append(float(np.max(foot_pos[start:int(td[0])+1, 2]) - baseline))
    return np.asarray(peaks, dtype=np.float64)

left_td, right_td = touchdown_indices(contact)
step_lengths = []
for td_idx in left_td:
    step_lengths.append(float(left_foot_pos[td_idx, 0] - right_foot_pos[td_idx, 0]))
for td_idx in right_td:
    step_lengths.append(float(right_foot_pos[td_idx, 0] - left_foot_pos[td_idx, 0]))
step_arr = np.asarray(step_lengths, dtype=np.float64)
clearance_arr = np.concatenate(
    [swing_clearance(left_foot_pos, contact[:, 0]), swing_clearance(right_foot_pos, contact[:, 1])],
    axis=0,
)
duration_s = float(len(contact) * dt)
total_touchdowns = len(left_td) + len(right_td)
max_touchdowns = max(len(left_td), len(right_td), 1)
min_touchdowns = min(len(left_td), len(right_td))
foot_z_step_max_m = max(
    float(np.max(np.abs(np.diff(left_foot_pos[:, 2])))),
    float(np.max(np.abs(np.diff(right_foot_pos[:, 2])))),
)
step_mean = safe_stat(step_arr, np.mean)
if len(left_td) >= 2:
    left_cycle_s = np.diff(left_td).astype(np.float64) * dt
else:
    left_cycle_s = np.asarray([], dtype=np.float64)
if len(right_td) >= 2:
    right_cycle_s = np.diff(right_td).astype(np.float64) * dt
else:
    right_cycle_s = np.asarray([], dtype=np.float64)
cycle_arr = np.concatenate([left_cycle_s, right_cycle_s], axis=0)
cycle_time_s = safe_stat(cycle_arr, np.median)
if not np.isfinite(cycle_time_s) or cycle_time_s <= 1e-8:
    cycle_time_s = duration_s
bit_flips = int(np.sum(contact[1:] != contact[:-1]))
n_cycles = max(duration_s / cycle_time_s, 1.0)

swing_step = 0.0
for foot_pos, contact_col in ((left_foot_pos, contact[:, 0]), (right_foot_pos, contact[:, 1])):
    if len(contact_col) < 2:
        continue
    mask = np.logical_or(contact_col[:-1] < 0.5, contact_col[1:] < 0.5)
    if np.any(mask):
        swing_step = max(swing_step, float(np.max(np.abs(np.diff(foot_pos[:, 2]))[mask])))

payload = {{
    "gait": {{
        "name": robot_name,
        "vx": vx,
        "duration_s": duration_s,
        "touchdowns_left": len(left_td),
        "touchdowns_right": len(right_td),
        "touchdown_balance_ratio": float(min_touchdowns / max_touchdowns),
        "touchdown_rate_hz": float(total_touchdowns / duration_s),
        "step_length_mean_m": step_mean,
        "step_length_min_m": safe_stat(step_arr, np.min),
        "step_length_max_m": safe_stat(step_arr, np.max),
        "touchdown_speed_proxy_mps": float(step_mean * total_touchdowns / duration_s / 2.0) if np.isfinite(step_mean) else float("nan"),
        "swing_clearance_mean_m": safe_stat(clearance_arr, np.mean),
        "swing_clearance_min_m": safe_stat(clearance_arr, np.min),
        "double_support_frac": float(np.mean(np.sum(contact, axis=1) == 2)),
        "foot_z_step_max_m": foot_z_step_max_m,
    }},
    "smooth": {{
        "name": robot_name,
        "vx": vx,
        "shared_leg_q_step_max_rad": float(np.max(np.abs(np.diff(shared_leg_q, axis=0)))),
        "pelvis_z_step_max_m": float(np.max(np.abs(np.diff(np.asarray(traj["body_pos"], dtype=np.float32)[:, 1, 2])))),
        "swing_foot_z_step_max_m": swing_step,
        "contact_flips_per_cycle": float(bit_flips / n_cycles),
    }},
}}
print(json.dumps(payload))
"""
    result = subprocess.run(
        [tb_python, "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=toddlerbot_root,
    )
    payload = json.loads(result.stdout)
    return FKGaitMetrics(**payload["gait"]), SmoothnessMetrics(**payload["smooth"])


def _closed_loop_wildrobot(vx: float, horizon: int) -> ClosedLoopMetrics:
    model, data, mapper, act_to_qpos, geom_ids, body_ids, shared_joint_limits = _load_wr_assets()
    traj = ZMPWalkGenerator().generate(vx)
    horizon = min(int(horizon), int(traj.n_steps))
    _init_wr_standing(model, data)
    physics_steps_per_ref = max(1, int(round(float(traj.dt) / float(model.opt.timestep))))
    standing_q = data.qpos[act_to_qpos][mapper.policy_to_mj_order].astype(np.float64)
    first_q = np.asarray(traj.q_ref[0], dtype=np.float64)
    for ramp_idx in range(50):
        alpha = (ramp_idx + 1) / 50.0
        blended = standing_q * (1.0 - alpha) + first_q * alpha
        mapper.set_all_ctrl(data, blended)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)
    start_x = float(data.qpos[0])
    left_contact_geoms = _wr_collision_geoms(model, body_ids["left_foot"])
    right_contact_geoms = _wr_collision_geoms(model, body_ids["right_foot"])
    prev_left_in = _foot_floor_in_contact(data, geom_ids["floor"], left_contact_geoms)
    prev_right_in = _foot_floor_in_contact(data, geom_ids["floor"], right_contact_geoms)
    left_touchdown_x: list[float] = []
    right_touchdown_x: list[float] = []
    q_err: list[np.ndarray] = []
    any_sat: list[bool] = []
    contact_matches = 0
    survived_steps = 0
    termination_mode = "ok"
    for step_idx in range(horizon):
        idx = min(step_idx, traj.n_steps - 1)
        q_ref = np.asarray(traj.q_ref[idx], dtype=np.float64)
        mapper.set_all_ctrl(data, q_ref)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)
        actual_policy_q = data.qpos[act_to_qpos][mapper.policy_to_mj_order][:8].astype(np.float64)
        q_err.append(actual_policy_q - q_ref[:8])
        any_sat.append(
            bool(
                np.any(
                    (actual_policy_q <= shared_joint_limits[:, 0] + _SAT_MARGIN_RAD)
                    | (actual_policy_q >= shared_joint_limits[:, 1] - _SAT_MARGIN_RAD)
                )
            )
        )
        left_in = _foot_floor_in_contact(data, geom_ids["floor"], left_contact_geoms)
        right_in = _foot_floor_in_contact(data, geom_ids["floor"], right_contact_geoms)
        ref_contact = np.asarray(traj.contact_mask[idx], dtype=np.float64) > 0.5
        contact_matches += int(left_in == bool(ref_contact[0]) and right_in == bool(ref_contact[1]))
        left_center, right_center = _wr_foot_center_pos(data, geom_ids)
        if left_in and not prev_left_in:
            left_touchdown_x.append(float(left_center[0]))
        if right_in and not prev_right_in:
            right_touchdown_x.append(float(right_center[0]))
        prev_left_in = left_in
        prev_right_in = right_in
        pitch, roll = _quat_to_pitch_roll(np.asarray(data.qpos[3:7], dtype=np.float64))
        survived_steps = step_idx + 1
        if abs(pitch) > _SHARED_PITCH_FAIL_RAD:
            termination_mode = "pitch"
            break
        if abs(roll) > _SHARED_ROLL_FAIL_RAD:
            termination_mode = "roll"
            break
        if float(data.qpos[2]) < _WR_ROOT_Z_FAIL_M:
            termination_mode = "height"
            break
    err_arr = np.asarray(q_err, dtype=np.float64)
    duration_s = max(survived_steps * float(traj.dt), 1e-8)
    step_mean, step_min = _combined_touchdown_step_stats(left_touchdown_x, right_touchdown_x)
    return ClosedLoopMetrics(
        name="wildrobot",
        vx=vx,
        horizon_steps=horizon,
        survived_steps=survived_steps,
        shared_leg_rmse_rad=float(np.sqrt(np.mean(err_arr ** 2))) if err_arr.size else float("nan"),
        achieved_forward_mps=float((float(data.qpos[0]) - start_x) / duration_s),
        ref_contact_match_frac=float(contact_matches / max(survived_steps, 1)),
        touchdown_step_length_mean_m=step_mean,
        touchdown_step_length_min_m=step_min,
        touchdown_rate_hz=float((len(left_touchdown_x) + len(right_touchdown_x)) / duration_s),
        joint_limit_step_frac=float(np.mean(np.asarray(any_sat, dtype=np.float64))) if any_sat else float("nan"),
        termination_mode=termination_mode,
        terminal_root_x_m=float(data.qpos[0]),
    )


def _closed_loop_toddlerbot(
    toddlerbot_root: Path, robot_name: str, vx_bins: list[float], horizon: int
) -> list[ClosedLoopMetrics]:
    """Run TB free-base zero-residual replay across ``vx_bins`` in one
    subprocess.  Returns one ``ClosedLoopMetrics`` per bin in input order.

    Loading TB + the lz4 lookup is the dominant per-call cost; folding
    the per-vx loop into the helper lets us pay it once per variant
    instead of once per (variant, vx).
    """
    tb_python = _pick_python_with_modules(
        [
            toddlerbot_root / ".venv" / "bin" / "python",
            "python3",
            "/usr/bin/python3",
        ],
        ["joblib", "lz4", "mujoco", "numpy", "scipy", "yaml"],
        cwd=toddlerbot_root,
    )
    helper = f"""
import json
from pathlib import Path
import joblib
import mujoco
import numpy as np

toddlerbot_root = Path({str(toddlerbot_root)!r})
robot_name = {robot_name!r}
vx_bins = {list(vx_bins)!r}
horizon = {horizon!r}
shared_leg_names = {list(_TB_SHARED_LEG_NAMES)!r}
sat_margin = {_SAT_MARGIN_RAD!r}
pitch_fail = {_SHARED_PITCH_FAIL_RAD!r}
roll_fail = {_SHARED_ROLL_FAIL_RAD!r}
torso_z_fail = {_TB_TORSO_Z_FAIL_M!r}

import sys
sys.path.insert(0, str(toddlerbot_root))
from toddlerbot.sim.robot import Robot
from toddlerbot.sim.mujoco_sim import MuJoCoSim

def quat_to_pitch_roll(quat_wxyz):
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    pitch = np.arctan2(2.0 * (qw * qy - qz * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return float(pitch), float(roll)

def foot_floor_in_contact(data, floor_geom_id, geom_ids):
    for con_idx in range(data.ncon):
        con = data.contact[con_idx]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        if (g1 == floor_geom_id and g2 in geom_ids) or (g2 == floor_geom_id and g1 in geom_ids):
            return True
    return False

def same_foot_steps(xs):
    if len(xs) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.asarray(xs, dtype=np.float64)) / 2.0

suffix = "_2xc" if "2xc" in robot_name else "_2xm"
lookup_keys, motion_ref_list = joblib.load(toddlerbot_root / "motion" / f"walk_zmp{{suffix}}.lz4")
lookup_keys = np.asarray(lookup_keys, dtype=np.float32)
fixed_model = mujoco.MjModel.from_xml_path(str(toddlerbot_root / "toddlerbot" / "descriptions" / robot_name / "scene_fixed.xml"))
free_scene = toddlerbot_root / "toddlerbot" / "descriptions" / robot_name / "scene.xml"
robot = Robot(robot_name)

shared_joint_limits = np.array([
    fixed_model.jnt_range[mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in shared_leg_names
], dtype=np.float64)
shared_qpos_adrs = [
    int(fixed_model.jnt_qposadr[mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    for name in shared_leg_names
]
joint_qpos_adrs = {{
    name: int(fixed_model.jnt_qposadr[mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    for name in robot.joint_ordering
}}
home_motor = np.asarray(list(robot.default_motor_angles.values()), dtype=np.float32)

results = []
for vx in vx_bins:
    idx = int(np.argmin(np.linalg.norm(lookup_keys - np.array([vx, 0.0, 0.0], np.float32), axis=1)))
    traj = motion_ref_list[idx]
    contact_ref = np.asarray(traj["contact"], dtype=np.float32)
    time_arr = np.asarray(traj["time"], dtype=np.float32)
    traj_dt = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 0.02

    # Each vx bin needs a fresh free-base sim — accumulated state from
    # the previous bin's rollout would corrupt this one's start_x and
    # contact baseline.
    sim = MuJoCoSim(
        robot,
        fixed_base=False,
        vis_type="",
        controller_type="torque",
        xml_path=str(free_scene),
    )
    # FIX #3 (review feedback): the trajectory is replayed at one
    # ``q_ref`` slice per ``sim.step()``.  If the sim's control_dt
    # diverges from the lookup's dt, the q_ref index advances at one
    # rate while physics integrates at another — silently mis-pairing
    # reference and actuals.  Fail loud here instead of producing a
    # wrong RMSE / forward-velocity / contact-match number.
    assert abs(float(sim.control_dt) - traj_dt) < 1e-6, (
        f"TB sim.control_dt={{float(sim.control_dt):.6f}} disagrees with "
        f"lookup dt={{traj_dt:.6f}} for {{robot_name}} vx={{vx:.2f}}"
    )

    if hasattr(sim, "home_qpos"):
        sim.data.qpos[:] = sim.home_qpos
        mujoco.mj_forward(sim.model, sim.data)

    def target_motor_order(frame_idx, _traj=traj):
        joint_angles = {{
            name: float(np.asarray(_traj["qpos"], dtype=np.float32)[frame_idx, adr])
            for name, adr in joint_qpos_adrs.items()
        }}
        motor_angles = robot.joint_to_motor_angles(joint_angles)
        return np.asarray([motor_angles[name] for name in robot.motor_ordering], dtype=np.float32)

    first_target = target_motor_order(0)
    for ramp_idx in range(50):
        alpha = (ramp_idx + 1) / 50.0
        sim.set_motor_target(home_motor * (1.0 - alpha) + first_target * alpha)
        sim.step()

    start_x = float(sim.data.body("torso").xpos[0])
    floor_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_contact_geoms = {{
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link_collision"),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_pitch_link_collision"),
    }}
    right_contact_geoms = {{
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link_collision"),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_pitch_link_collision"),
    }}
    left_contact_geoms = {{g for g in left_contact_geoms if g >= 0}}
    right_contact_geoms = {{g for g in right_contact_geoms if g >= 0}}
    prev_left_in = foot_floor_in_contact(sim.data, floor_geom_id, left_contact_geoms)
    prev_right_in = foot_floor_in_contact(sim.data, floor_geom_id, right_contact_geoms)

    q_err = []
    any_sat = []
    contact_matches = 0
    left_touchdown_x = []
    right_touchdown_x = []
    survived_steps = 0
    termination_mode = "ok"
    horizon_eff = min(int(horizon), len(traj["qpos"]))
    for step_idx in range(horizon_eff):
        sim.set_motor_target(target_motor_order(step_idx))
        sim.step()
        q_ref_shared = np.asarray(traj["qpos"], dtype=np.float32)[step_idx, shared_qpos_adrs]
        q_actual_shared = np.asarray([sim.data.joint(name).qpos.item() for name in shared_leg_names], dtype=np.float64)
        q_err.append(q_actual_shared - q_ref_shared.astype(np.float64))
        any_sat.append(
            bool(
                np.any(
                    (q_actual_shared <= shared_joint_limits[:, 0] + sat_margin)
                    | (q_actual_shared >= shared_joint_limits[:, 1] - sat_margin)
                )
            )
        )
        left_in = foot_floor_in_contact(sim.data, floor_geom_id, left_contact_geoms)
        right_in = foot_floor_in_contact(sim.data, floor_geom_id, right_contact_geoms)
        ref_contact = contact_ref[step_idx] > 0.5
        contact_matches += int(left_in == bool(ref_contact[0]) and right_in == bool(ref_contact[1]))
        left_site_x = float(sim.data.site("left_foot_center").xpos[0])
        right_site_x = float(sim.data.site("right_foot_center").xpos[0])
        if left_in and not prev_left_in:
            left_touchdown_x.append(left_site_x)
        if right_in and not prev_right_in:
            right_touchdown_x.append(right_site_x)
        prev_left_in = left_in
        prev_right_in = right_in
        pitch, roll = quat_to_pitch_roll(np.asarray(sim.data.body("torso").xquat, dtype=np.float64))
        torso_z = float(sim.data.body("torso").xpos[2])
        survived_steps = step_idx + 1
        if abs(pitch) > pitch_fail:
            termination_mode = "pitch"
            break
        if abs(roll) > roll_fail:
            termination_mode = "roll"
            break
        if torso_z < torso_z_fail:
            termination_mode = "height"
            break

    err_arr = np.asarray(q_err, dtype=np.float64)
    duration_s = max(survived_steps * sim.control_dt, 1e-8)
    combined_steps = np.concatenate([same_foot_steps(left_touchdown_x), same_foot_steps(right_touchdown_x)], axis=0)
    results.append({{
        "name": robot_name,
        "vx": vx,
        "horizon_steps": horizon_eff,
        "survived_steps": survived_steps,
        "shared_leg_rmse_rad": float(np.sqrt(np.mean(err_arr ** 2))) if err_arr.size else float("nan"),
        "achieved_forward_mps": float((float(sim.data.body("torso").xpos[0]) - start_x) / duration_s),
        "ref_contact_match_frac": float(contact_matches / max(survived_steps, 1)),
        "touchdown_step_length_mean_m": float(np.mean(combined_steps)) if combined_steps.size else float("nan"),
        "touchdown_step_length_min_m": float(np.min(combined_steps)) if combined_steps.size else float("nan"),
        "touchdown_rate_hz": float((len(left_touchdown_x) + len(right_touchdown_x)) / duration_s),
        "joint_limit_step_frac": float(np.mean(np.asarray(any_sat, dtype=np.float64))) if any_sat else float("nan"),
        "termination_mode": termination_mode,
        "terminal_root_x_m": float(sim.data.body("torso").xpos[0]),
    }})
    sim.close()

print(json.dumps(results))
"""
    result = subprocess.run(
        [tb_python, "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=toddlerbot_root,
    )
    return [ClosedLoopMetrics(**payload) for payload in json.loads(result.stdout)]


def _geometry_parity_row(summary_wr: GateSummary, summary_tb: GateSummary) -> tuple[bool, dict[str, str]]:
    in_scope_ok = summary_wr.in_scope_failures == 0 and summary_tb.in_scope_failures == 0
    full_ok = summary_wr.total_failures <= summary_tb.total_failures
    stance_ok = summary_wr.worst_stance_z <= summary_tb.worst_stance_z + _PARITY_MARGIN_M
    swing_ok = summary_wr.worst_swing_z >= summary_tb.worst_swing_z - _PARITY_MARGIN_M
    parity_ok = in_scope_ok and full_ok and stance_ok and swing_ok
    return parity_ok, {
        "baseline": summary_tb.name,
        "in_scope": "PASS" if in_scope_ok else "FAIL",
        "full_gate": "PASS" if full_ok else "FAIL",
        "stance_gate": "PASS" if stance_ok else "FAIL",
        "swing_gate": "PASS" if swing_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }


def _fk_parity_row(metrics_wr: FKGaitMetrics, metrics_tb: FKGaitMetrics) -> tuple[bool, dict[str, str]]:
    step_len_ok = metrics_wr.step_length_mean_m >= 0.90 * metrics_tb.step_length_mean_m
    cadence_ok = metrics_wr.touchdown_rate_hz <= 1.20 * metrics_tb.touchdown_rate_hz
    speed_ok = abs(metrics_wr.touchdown_speed_proxy_mps - metrics_tb.touchdown_speed_proxy_mps) <= 0.02
    clearance_ok = metrics_wr.swing_clearance_mean_m >= 0.85 * metrics_tb.swing_clearance_mean_m
    ds_ok = abs(metrics_wr.double_support_frac - metrics_tb.double_support_frac) <= 0.05
    parity_ok = step_len_ok and cadence_ok and speed_ok and clearance_ok and ds_ok
    return parity_ok, {
        "baseline": metrics_tb.name,
        "step_len_gate": "PASS" if step_len_ok else "FAIL",
        "cadence_gate": "PASS" if cadence_ok else "FAIL",
        "speed_gate": "PASS" if speed_ok else "FAIL",
        "clearance_gate": "PASS" if clearance_ok else "FAIL",
        "ds_gate": "PASS" if ds_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }


def _smoothness_parity_row(
    smooth_wr: SmoothnessMetrics, smooth_tb: SmoothnessMetrics
) -> tuple[bool, dict[str, str]]:
    q_ok = smooth_wr.shared_leg_q_step_max_rad <= 1.10 * smooth_tb.shared_leg_q_step_max_rad
    pelvis_ok = smooth_wr.pelvis_z_step_max_m <= 1.10 * smooth_tb.pelvis_z_step_max_m
    swing_ok = smooth_wr.swing_foot_z_step_max_m <= 1.10 * smooth_tb.swing_foot_z_step_max_m
    flips_ok = smooth_wr.contact_flips_per_cycle <= 1.10 * smooth_tb.contact_flips_per_cycle
    parity_ok = q_ok and pelvis_ok and swing_ok and flips_ok
    return parity_ok, {
        "baseline": smooth_tb.name,
        "q_gate": "PASS" if q_ok else "FAIL",
        "pelvis_gate": "PASS" if pelvis_ok else "FAIL",
        "swing_gate": "PASS" if swing_ok else "FAIL",
        "flip_gate": "PASS" if flips_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }


def _trackability_gate_detail(
    metrics_wr: ClosedLoopMetrics, metrics_tb: ClosedLoopMetrics
) -> tuple[bool, dict[str, str]]:
    rmse_ok = metrics_wr.shared_leg_rmse_rad <= 1.10 * metrics_tb.shared_leg_rmse_rad
    forward_ok = metrics_wr.achieved_forward_mps >= metrics_tb.achieved_forward_mps - 0.02
    episode_ok = metrics_wr.survived_steps >= 0.95 * metrics_tb.survived_steps
    contact_ok = metrics_wr.ref_contact_match_frac >= max(0.90, metrics_tb.ref_contact_match_frac - 0.05)
    step_ok = (
        np.isfinite(metrics_wr.touchdown_step_length_mean_m)
        and np.isfinite(metrics_tb.touchdown_step_length_mean_m)
        and metrics_wr.touchdown_step_length_mean_m >= 0.90 * metrics_tb.touchdown_step_length_mean_m
    )
    limit_ok = metrics_wr.joint_limit_step_frac <= metrics_tb.joint_limit_step_frac + 0.05
    overall_ok = rmse_ok and forward_ok and episode_ok and contact_ok and step_ok and limit_ok
    return overall_ok, {
        "rmse": "PASS" if rmse_ok else "FAIL",
        "forward": "PASS" if forward_ok else "FAIL",
        "episode": "PASS" if episode_ok else "FAIL",
        "contact": "PASS" if contact_ok else "FAIL",
        "step": "PASS" if step_ok else "FAIL",
        "limit": "PASS" if limit_ok else "FAIL",
        "verdict": "PASS" if overall_ok else "FAIL",
    }


_TRACKABILITY_GATE_KEYS = ("rmse", "forward", "episode", "contact", "step", "limit")


def _aggregate_trackability_parity(
    wr_rows: list[ClosedLoopMetrics], tb_rows: list[ClosedLoopMetrics]
) -> tuple[bool, dict[str, str]]:
    tb_by_vx = {row.vx: row for row in tb_rows}
    fields: dict[str, list[str]] = {key: [] for key in _TRACKABILITY_GATE_KEYS}
    overall = True
    for wr_row in wr_rows:
        tb_row = tb_by_vx[wr_row.vx]
        row_ok, detail = _trackability_gate_detail(wr_row, tb_row)
        overall = overall and row_ok
        for key in fields:
            fields[key].append(f"{detail[key]}@{wr_row.vx:.2f}")
    summary = {key: ", ".join(values) for key, values in fields.items()}
    summary["verdict"] = "PASS" if overall else "FAIL"
    return overall, summary


def _trackability_per_bin_matrix(
    wr_rows: list[ClosedLoopMetrics], tb_rows: list[ClosedLoopMetrics]
) -> dict[str, dict[float, str]]:
    """Return ``{metric: {vx: PASS/FAIL}}`` for one TB variant.

    Cleanup #5 (review feedback): the per-line ``rmse: PASS@0.10, FAIL@0.15``
    string is hard to scan when there are 6 metrics × N bins.  This shape
    is what ``_print_trackability_matrix`` renders as a small table so the
    load-bearing failing bin is obvious at a glance.
    """
    tb_by_vx = {row.vx: row for row in tb_rows}
    matrix: dict[str, dict[float, str]] = {key: {} for key in _TRACKABILITY_GATE_KEYS}
    for wr_row in wr_rows:
        tb_row = tb_by_vx[wr_row.vx]
        _row_ok, detail = _trackability_gate_detail(wr_row, tb_row)
        for key in _TRACKABILITY_GATE_KEYS:
            matrix[key][wr_row.vx] = detail[key]
    return matrix


# ---------------------------------------------------------------------------
# Size-normalised parity (Option A)
# ---------------------------------------------------------------------------
# Companion gates that compare WR vs TB after dividing out leg length /
# COM height.  See training/docs/reference_parity_scorecard.md
# § "Size-normalised parity (Option A)" for the rationale.

_NORM_FK_GATE_KEYS = (
    "step_per_leg",
    "clearance_per_h",
    "cadence_norm",
    "swing_step_per_clr",
)

_NORM_TRACK_GATE_KEYS = (
    "fwd_per_cmd",
    "drift_per_leg",
    "td_step_per_leg",
)


def _normalized_fk_parity_row(
    norm_wr: NormalizedGaitMetrics, norm_tb: NormalizedGaitMetrics
) -> tuple[bool, dict[str, str]]:
    """Apply the same gate shapes as ``_fk_parity_row`` / smoothness, but
    on the size-normalised numbers.  Thresholds mirror the absolute ones."""
    step_ok = norm_wr.step_length_per_leg >= 0.85 * norm_tb.step_length_per_leg
    clearance_ok = (
        norm_wr.swing_clearance_per_com_height
        >= 0.85 * norm_tb.swing_clearance_per_com_height
    )
    cadence_ok = norm_wr.cadence_froude_norm <= 1.20 * norm_tb.cadence_froude_norm
    swing_step_ok = (
        norm_wr.swing_foot_z_step_per_clearance
        <= 1.10 * norm_tb.swing_foot_z_step_per_clearance
    )
    parity_ok = step_ok and clearance_ok and cadence_ok and swing_step_ok
    return parity_ok, {
        "baseline": norm_tb.name,
        "step_per_leg": "PASS" if step_ok else "FAIL",
        "clearance_per_h": "PASS" if clearance_ok else "FAIL",
        "cadence_norm": "PASS" if cadence_ok else "FAIL",
        "swing_step_per_clr": "PASS" if swing_step_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }


def _normalized_trackability_gate_detail(
    norm_wr: NormalizedGaitMetrics, norm_tb: NormalizedGaitMetrics
) -> tuple[bool, dict[str, str]]:
    """Per-bin normalised P1 verdict.

    - ``fwd_per_cmd``: WR within 0.10 of TB on (achieved / cmd) ratio.
    - ``drift_per_leg``: WR's |drift_rate / leg_length| <= TB's + 0.5 leg/s.
    - ``td_step_per_leg``: WR step length per leg >= 0.85 x TB.
    """
    fwd_ok = (
        np.isfinite(norm_wr.achieved_forward_per_cmd)
        and np.isfinite(norm_tb.achieved_forward_per_cmd)
        and norm_wr.achieved_forward_per_cmd >= norm_tb.achieved_forward_per_cmd - 0.10
    )
    drift_ok = (
        np.isfinite(norm_wr.terminal_drift_rate_per_leg_per_s)
        and np.isfinite(norm_tb.terminal_drift_rate_per_leg_per_s)
        and abs(norm_wr.terminal_drift_rate_per_leg_per_s)
        <= abs(norm_tb.terminal_drift_rate_per_leg_per_s) + 0.5
    )
    step_ok = (
        np.isfinite(norm_wr.td_step_length_per_leg)
        and np.isfinite(norm_tb.td_step_length_per_leg)
        and norm_wr.td_step_length_per_leg >= 0.85 * norm_tb.td_step_length_per_leg
    )
    overall_ok = fwd_ok and drift_ok and step_ok
    return overall_ok, {
        "fwd_per_cmd": "PASS" if fwd_ok else "FAIL",
        "drift_per_leg": "PASS" if drift_ok else "FAIL",
        "td_step_per_leg": "PASS" if step_ok else "FAIL",
        "verdict": "PASS" if overall_ok else "FAIL",
    }


def _aggregate_normalized_trackability(
    norm_wr_rows: list[NormalizedGaitMetrics],
    norm_tb_rows: list[NormalizedGaitMetrics],
) -> tuple[bool, dict[str, str]]:
    tb_by_vx = {row.vx: row for row in norm_tb_rows}
    fields: dict[str, list[str]] = {key: [] for key in _NORM_TRACK_GATE_KEYS}
    overall = True
    for wr_row in norm_wr_rows:
        tb_row = tb_by_vx[wr_row.vx]
        row_ok, detail = _normalized_trackability_gate_detail(wr_row, tb_row)
        overall = overall and row_ok
        for key in fields:
            fields[key].append(f"{detail[key]}@{wr_row.vx:.2f}")
    summary = {key: ", ".join(values) for key, values in fields.items()}
    summary["verdict"] = "PASS" if overall else "FAIL"
    return overall, summary


def _normalized_trackability_per_bin_matrix(
    norm_wr_rows: list[NormalizedGaitMetrics],
    norm_tb_rows: list[NormalizedGaitMetrics],
) -> dict[str, dict[float, str]]:
    tb_by_vx = {row.vx: row for row in norm_tb_rows}
    matrix: dict[str, dict[float, str]] = {key: {} for key in _NORM_TRACK_GATE_KEYS}
    for wr_row in norm_wr_rows:
        tb_row = tb_by_vx[wr_row.vx]
        _row_ok, detail = _normalized_trackability_gate_detail(wr_row, tb_row)
        for key in _NORM_TRACK_GATE_KEYS:
            matrix[key][wr_row.vx] = detail[key]
    return matrix


def _print_geometry_table(
    summary_wr: GateSummary,
    summary_tbs: list[GateSummary],
    parity_rows: list[dict[str, str]],
) -> None:
    parity_by_name = {row["baseline"]: row for row in parity_rows}
    print("Geometry comparison table")
    print(
        f"{'robot':<16} "
        f"{'failures@vx<=0.15':>18} "
        f"{'failures@all_vx':>17} "
        f"{'stance_max_z':>14} "
        f"{'swing_min_z':>14}"
    )
    print(
        f"{'':<16} "
        f"{'(gate)':>18} "
        f"{'(diag)':>17} "
        f"{'(>+0.003 fail)':>14} "
        f"{'(<-0.002 fail)':>14}"
    )
    print("-" * 84)
    for summary in [summary_wr, *summary_tbs]:
        print(
            f"{summary.name:<16} "
            f"{summary.in_scope_failures:>18d} "
            f"{summary.total_failures:>17d} "
            f"{summary.worst_stance_z:>+14.4f} "
            f"{summary.worst_swing_z:>+14.4f}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_name[robot_name]
        print(
            f"WR vs {robot_name}: {row['verdict']} "
            f"(full_vx={row['full_gate']}, stance={row['stance_gate']}, swing={row['swing_gate']})"
        )


def _print_fk_table(
    gait_wr: FKGaitMetrics,
    gait_tbs: list[FKGaitMetrics],
    parity_rows: list[dict[str, str]],
    vx: float,
) -> None:
    parity_by_name = {row["baseline"]: row for row in parity_rows}
    print()
    print(f"FK-realized gait comparison @ vx={vx:.2f}")
    print(
        f"{'robot':<16} "
        f"{'step_len_mean':>14} "
        f"{'touchdown_rate':>16} "
        f"{'touchdown_speed':>17} "
        f"{'clr_mean':>10} "
        f"{'clr_min':>10} "
        f"{'ds_frac':>10} "
        f"{'foot_z_step':>12}"
    )
    print(
        f"{'':<16} "
        f"{'(m)':>14} "
        f"{'(Hz)':>16} "
        f"{'(m/s)':>17} "
        f"{'(m)':>10} "
        f"{'(m)':>10} "
        f"{'':>10} "
        f"{'(m/frame)':>12}"
    )
    print("-" * 117)
    for metrics in [gait_wr, *gait_tbs]:
        print(
            f"{metrics.name:<16} "
            f"{_fmt_float(metrics.step_length_mean_m, 14)} "
            f"{_fmt_float(metrics.touchdown_rate_hz, 16)} "
            f"{_fmt_float(metrics.touchdown_speed_proxy_mps, 17)} "
            f"{_fmt_float(metrics.swing_clearance_mean_m, 10)} "
            f"{_fmt_float(metrics.swing_clearance_min_m, 10)} "
            f"{_fmt_float(metrics.double_support_frac, 10)} "
            f"{_fmt_float(metrics.foot_z_step_max_m, 12)}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_name[robot_name]
        print(
            f"WR FK parity vs {robot_name}: {row['verdict']} "
            f"(step_len={row['step_len_gate']}, cadence={row['cadence_gate']}, "
            f"speed={row['speed_gate']}, clearance={row['clearance_gate']}, ds_frac={row['ds_gate']})"
        )


def _print_smoothness_table(
    smooth_wr: SmoothnessMetrics,
    smooth_tbs: list[SmoothnessMetrics],
    parity_rows: list[dict[str, str]],
    vx: float,
) -> None:
    parity_by_name = {row["baseline"]: row for row in parity_rows}
    print()
    print(f"Smoothness comparison @ vx={vx:.2f}")
    print(
        f"{'robot':<16} "
        f"{'leg_q_step_max':>15} "
        f"{'pelvis_z_step':>15} "
        f"{'swing_z_step':>15} "
        f"{'contact_flips/cycle':>20}"
    )
    print(
        f"{'':<16} "
        f"{'(rad/frame)':>15} "
        f"{'(m/frame)':>15} "
        f"{'(m/frame)':>15} "
        f"{'':>20}"
    )
    print("-" * 86)
    for metrics in [smooth_wr, *smooth_tbs]:
        print(
            f"{metrics.name:<16} "
            f"{_fmt_float(metrics.shared_leg_q_step_max_rad, 15)} "
            f"{_fmt_float(metrics.pelvis_z_step_max_m, 15)} "
            f"{_fmt_float(metrics.swing_foot_z_step_max_m, 15)} "
            f"{_fmt_float(metrics.contact_flips_per_cycle, 20)}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_name[robot_name]
        print(
            f"WR smoothness parity vs {robot_name}: {row['verdict']} "
            f"(q_step={row['q_gate']}, pelvis={row['pelvis_gate']}, "
            f"swing={row['swing_gate']}, flips={row['flip_gate']})"
        )


def _print_trackability_table(
    wr_rows: list[ClosedLoopMetrics],
    tb_rows_by_robot: dict[str, list[ClosedLoopMetrics]],
    parity_by_robot: dict[str, dict[str, str]],
) -> None:
    print()
    print("Closed-loop zero-residual comparison")
    print(
        f"{'robot':<16} "
        f"{'vx':>6} "
        f"{'leg_rmse':>10} "
        f"{'fwd_mps':>10} "
        f"{'survived':>10} "
        f"{'contact_match':>14} "
        f"{'td_step_mean':>13} "
        f"{'sat_step_frac':>13} "
        f"{'termination':>12}"
    )
    print(
        f"{'':<16} "
        f"{'':>6} "
        f"{'(rad)':>10} "
        f"{'':>10} "
        f"{'(steps)':>10} "
        f"{'(0..1)':>14} "
        f"{'(m)':>13} "
        f"{'(0..1)':>13} "
        f"{'':>12}"
    )
    print("-" * 116)
    rows = list(wr_rows)
    for robot_name in _TB_VARIANTS:
        rows.extend(tb_rows_by_robot[robot_name])
    rows.sort(key=lambda row: (row.name, row.vx))
    for row in rows:
        print(
            f"{row.name:<16} "
            f"{row.vx:>6.2f} "
            f"{_fmt_float(row.shared_leg_rmse_rad, 10)} "
            f"{_fmt_float(row.achieved_forward_mps, 10)} "
            f"{row.survived_steps:>10d} "
            f"{_fmt_float(row.ref_contact_match_frac, 14)} "
            f"{_fmt_float(row.touchdown_step_length_mean_m, 13)} "
            f"{_fmt_float(row.joint_limit_step_frac, 13)} "
            f"{row.termination_mode:>12}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_robot[robot_name]
        print(
            f"WR P1 parity vs {robot_name}: {row['verdict']} "
            f"(rmse={row['rmse']}, forward={row['forward']}, episode={row['episode']}, "
            f"contact={row['contact']}, step_len={row['step']}, sat={row['limit']})"
        )

    # Per-bin matrix: which (metric, vx) pair is the load-bearing fail
    # against each TB variant.  Easier to scan than the comma-separated
    # detail strings above when there are 6 metrics × N bins.
    if wr_rows:
        vx_bins_seen = sorted({row.vx for row in wr_rows})
        for robot_name in _TB_VARIANTS:
            matrix = _trackability_per_bin_matrix(
                wr_rows, tb_rows_by_robot[robot_name]
            )
            print()
            print(f"WR vs {robot_name} per-bin matrix")
            header = f"{'metric':<10}" + "".join(
                f" {('vx=' + f'{vx:.2f}'):>10}" for vx in vx_bins_seen
            )
            print(header)
            print("-" * len(header))
            for key in _TRACKABILITY_GATE_KEYS:
                cells = "".join(
                    f" {matrix[key].get(vx, 'n/a'):>10}" for vx in vx_bins_seen
                )
                print(f"{key:<10}{cells}")


def _print_normalized_fk_table(
    norm_wr: NormalizedGaitMetrics,
    norm_tbs: list[NormalizedGaitMetrics],
    parity_rows: list[dict[str, str]],
    vx: float,
) -> None:
    parity_by_name = {row["baseline"]: row for row in parity_rows}
    print()
    print(f"Size-normalised gait comparison @ vx={vx:.2f}")
    print(
        f"{'robot':<16} "
        f"{'leg(m)':>8} "
        f"{'h(m)':>8} "
        f"{'step/leg':>10} "
        f"{'clr/h':>10} "
        f"{'cad·√(h/g)':>12} "
        f"{'swing_step/clr':>14} "
        f"{'Froude(vx²/gh)':>16}"
    )
    print("-" * 99)
    for metrics in [norm_wr, *norm_tbs]:
        print(
            f"{metrics.name:<16} "
            f"{_fmt_float(metrics.leg_length_m, 8, 3)} "
            f"{_fmt_float(metrics.com_height_m, 8, 3)} "
            f"{_fmt_float(metrics.step_length_per_leg, 10)} "
            f"{_fmt_float(metrics.swing_clearance_per_com_height, 10)} "
            f"{_fmt_float(metrics.cadence_froude_norm, 12)} "
            f"{_fmt_float(metrics.swing_foot_z_step_per_clearance, 14)} "
            f"{_fmt_float(metrics.forward_speed_froude, 16)}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_name[robot_name]
        print(
            f"WR norm parity vs {robot_name}: {row['verdict']} "
            f"(step/leg={row['step_per_leg']}, clr/h={row['clearance_per_h']}, "
            f"cadence_norm={row['cadence_norm']}, swing_step/clr={row['swing_step_per_clr']})"
        )


def _print_normalized_trackability_table(
    norm_wr_rows: list[NormalizedGaitMetrics],
    norm_tb_rows_by_robot: dict[str, list[NormalizedGaitMetrics]],
    parity_by_robot: dict[str, dict[str, str]],
) -> None:
    print()
    print("Size-normalised closed-loop comparison")
    print(
        f"{'robot':<16} "
        f"{'vx':>6} "
        f"{'fwd/cmd':>10} "
        f"{'drift(leg/s)':>14} "
        f"{'td_step/leg':>12} "
        f"{'cad·√(h/g)':>12} "
        f"{'Froude':>10}"
    )
    print("-" * 84)
    rows = list(norm_wr_rows)
    for robot_name in _TB_VARIANTS:
        rows.extend(norm_tb_rows_by_robot[robot_name])
    rows.sort(key=lambda row: (row.name, row.vx))
    for row in rows:
        print(
            f"{row.name:<16} "
            f"{row.vx:>6.2f} "
            f"{_fmt_float(row.achieved_forward_per_cmd, 10)} "
            f"{_fmt_float(row.terminal_drift_rate_per_leg_per_s, 14)} "
            f"{_fmt_float(row.td_step_length_per_leg, 12)} "
            f"{_fmt_float(row.cadence_froude_norm, 12)} "
            f"{_fmt_float(row.forward_speed_froude, 10)}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_robot[robot_name]
        print(
            f"WR norm-P1 parity vs {robot_name}: {row['verdict']} "
            f"(fwd/cmd={row['fwd_per_cmd']}, drift/leg={row['drift_per_leg']}, "
            f"td_step/leg={row['td_step_per_leg']})"
        )

    if norm_wr_rows:
        vx_bins_seen = sorted({row.vx for row in norm_wr_rows})
        for robot_name in _TB_VARIANTS:
            matrix = _normalized_trackability_per_bin_matrix(
                norm_wr_rows, norm_tb_rows_by_robot[robot_name]
            )
            print()
            print(f"WR vs {robot_name} normalised P1 per-bin matrix")
            header = f"{'metric':<18}" + "".join(
                f" {('vx=' + f'{vx:.2f}'):>10}" for vx in vx_bins_seen
            )
            print(header)
            print("-" * len(header))
            for key in _NORM_TRACK_GATE_KEYS:
                cells = "".join(
                    f" {matrix[key].get(vx, 'n/a'):>10}" for vx in vx_bins_seen
                )
                print(f"{key:<18}{cells}")


def _build_payload(
    summary_wr: GateSummary | None,
    summary_tbs: list[GateSummary],
    geometry_rows: list[dict[str, str]],
    gait_wr: FKGaitMetrics,
    gait_tbs: list[FKGaitMetrics],
    fk_rows: list[dict[str, str]],
    smooth_wr: SmoothnessMetrics,
    smooth_tbs: list[SmoothnessMetrics],
    smooth_rows: list[dict[str, str]],
    p1_wr: list[ClosedLoopMetrics],
    p1_tbs: dict[str, list[ClosedLoopMetrics]],
    p1_rows: dict[str, dict[str, str]],
    norm_gait_wr: NormalizedGaitMetrics,
    norm_gait_tbs: list[NormalizedGaitMetrics],
    norm_fk_rows: list[dict[str, str]],
    norm_p1_wr: list[NormalizedGaitMetrics],
    norm_p1_tbs: dict[str, list[NormalizedGaitMetrics]],
    norm_p1_rows: dict[str, dict[str, str]],
    args: argparse.Namespace,
    overall_pass: bool,
    norm_overall_pass: bool,
) -> dict:
    return {
        "config": {
            "vx_nominal": args.vx,
            "geometry_vx_bins": list(_VX_BINS),
            "geometry_vx_bins_in_scope": list(_VX_BINS_INSCOPE),
            "trackability_vx_bins": list(_P1_VX_BINS if not args.nominal_only else [args.vx]),
            "probe_frames": list(_PROBE_FRAMES),
            "stance_fail_if_gt_m": _STANCE_TOL_M,
            "swing_fail_if_lt_m": _SWING_MIN_M,
            "parity_margin_m": _PARITY_MARGIN_M,
            "gravity_mps2": _GRAVITY_MPS2,
            "robot_dims": {
                "wildrobot": _wr_dims(),
                **{name: _TB_DIMS[name] for name in _TB_VARIANTS},
            },
        },
        "geometry_summary": [] if summary_wr is None else [asdict(summary_wr), *[asdict(v) for v in summary_tbs]],
        "geometry_parity": geometry_rows,
        "fk_gait_metrics": [asdict(gait_wr), *[asdict(v) for v in gait_tbs]],
        "fk_gait_parity": fk_rows,
        "smoothness_metrics": [asdict(smooth_wr), *[asdict(v) for v in smooth_tbs]],
        "smoothness_parity": smooth_rows,
        "trackability_metrics": [asdict(v) for v in p1_wr]
        + [asdict(row) for robot_rows in p1_tbs.values() for row in robot_rows],
        "trackability_parity": p1_rows,
        "normalized_fk_metrics": [asdict(norm_gait_wr), *[asdict(v) for v in norm_gait_tbs]],
        "normalized_fk_parity": norm_fk_rows,
        "normalized_trackability_metrics": [asdict(v) for v in norm_p1_wr]
        + [asdict(row) for robot_rows in norm_p1_tbs.values() for row in robot_rows],
        "normalized_trackability_parity": norm_p1_rows,
        "mode": {
            "nominal_only": args.nominal_only,
            "strict": args.strict,
        },
        "overall_pass": overall_pass,
        "normalized_overall_pass": norm_overall_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare WildRobot and ToddlerBot on geometry, FK, smoothness, and zero-residual replay."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if WR fails any included parity layer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of formatted tables.",
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Skip P0 full-matrix geometry and P1 multi-bin sweep; use only --vx.",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.15,
        help="Nominal velocity used for FK and smoothness tables.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Closed-loop replay horizon in control steps.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tools/parity_report.json",
        help=(
            "Path to write the JSON payload to (set to empty string to "
            "disable). Stdout output is unaffected — use --json to also "
            "stream it to stdout."
        ),
    )
    args = parser.parse_args()

    wildrobot_root, toddlerbot_root = _repo_roots()
    if not toddlerbot_root.exists():
        raise FileNotFoundError(
            f"ToddlerBot repo not found at expected sibling path: {toddlerbot_root}"
        )

    summary_wr: GateSummary | None = None
    summary_tbs: list[GateSummary] = []
    geometry_rows: list[dict[str, str]] = []
    geometry_ok = True
    if not args.nominal_only:
        summary_wr = _summarize_wildrobot()
        summary_tbs = [
            _summarize_toddlerbot(toddlerbot_root, robot_name)
            for robot_name in _TB_VARIANTS
        ]
        geometry_pairs = [_geometry_parity_row(summary_wr, row) for row in summary_tbs]
        geometry_rows = [row for _ok, row in geometry_pairs]
        geometry_ok = all(ok for ok, _row in geometry_pairs)

    gait_wr, smooth_wr = _wr_fk_and_smoothness(args.vx)
    gait_tbs: list[FKGaitMetrics] = []
    smooth_tbs: list[SmoothnessMetrics] = []
    for robot_name in _TB_VARIANTS:
        gait_tb, smooth_tb = _tb_nominal_bundle(toddlerbot_root, robot_name, args.vx)
        gait_tbs.append(gait_tb)
        smooth_tbs.append(smooth_tb)
    fk_pairs = [_fk_parity_row(gait_wr, gait_tb) for gait_tb in gait_tbs]
    fk_rows = [row for _ok, row in fk_pairs]
    fk_ok = all(ok for ok, _row in fk_pairs)
    smooth_pairs = [
        _smoothness_parity_row(smooth_wr, smooth_tb) for smooth_tb in smooth_tbs
    ]
    smooth_rows = [row for _ok, row in smooth_pairs]
    smooth_ok = all(ok for ok, _row in smooth_pairs)

    p1_vx_bins = [args.vx] if args.nominal_only else list(_P1_VX_BINS)
    p1_wr = [_closed_loop_wildrobot(vx, args.horizon) for vx in p1_vx_bins]
    p1_tbs: dict[str, list[ClosedLoopMetrics]] = {}
    p1_rows: dict[str, dict[str, str]] = {}
    p1_ok = True
    for robot_name in _TB_VARIANTS:
        rows = _closed_loop_toddlerbot(
            toddlerbot_root, robot_name, p1_vx_bins, args.horizon
        )
        p1_tbs[robot_name] = rows
        ok, parity_row = _aggregate_trackability_parity(p1_wr, rows)
        p1_rows[robot_name] = parity_row
        p1_ok = p1_ok and ok

    # Size-normalised companion gates (Option A).  Computed post-hoc from
    # the absolute metrics and each robot's characteristic dimensions so
    # WR (~1.6-1.8x larger) and TB are read on the same Froude-equivalent
    # footing.  See training/docs/reference_parity_scorecard.md.
    norm_gait_wr = _compute_normalized_from_fk("wildrobot", args.vx, gait_wr, smooth_wr)
    norm_gait_tbs = [
        _compute_normalized_from_fk(metrics.name, args.vx, metrics, smooth)
        for metrics, smooth in zip(gait_tbs, smooth_tbs)
    ]
    norm_fk_pairs = [
        _normalized_fk_parity_row(norm_gait_wr, norm_gait_tb)
        for norm_gait_tb in norm_gait_tbs
    ]
    norm_fk_rows = [row for _ok, row in norm_fk_pairs]
    norm_fk_ok = all(ok for ok, _row in norm_fk_pairs)

    norm_p1_wr = [_compute_normalized_from_closed_loop(row) for row in p1_wr]
    norm_p1_tbs: dict[str, list[NormalizedGaitMetrics]] = {}
    norm_p1_rows: dict[str, dict[str, str]] = {}
    norm_p1_ok = True
    for robot_name in _TB_VARIANTS:
        norm_rows = [
            _compute_normalized_from_closed_loop(row) for row in p1_tbs[robot_name]
        ]
        norm_p1_tbs[robot_name] = norm_rows
        ok, parity_row = _aggregate_normalized_trackability(norm_p1_wr, norm_rows)
        norm_p1_rows[robot_name] = parity_row
        norm_p1_ok = norm_p1_ok and ok

    overall_ok = geometry_ok and fk_ok and smooth_ok and p1_ok
    norm_overall_ok = norm_fk_ok and norm_p1_ok

    payload = _build_payload(
        summary_wr=summary_wr,
        summary_tbs=summary_tbs,
        geometry_rows=geometry_rows,
        gait_wr=gait_wr,
        gait_tbs=gait_tbs,
        fk_rows=fk_rows,
        smooth_wr=smooth_wr,
        smooth_tbs=smooth_tbs,
        smooth_rows=smooth_rows,
        p1_wr=p1_wr,
        p1_tbs=p1_tbs,
        p1_rows=p1_rows,
        norm_gait_wr=norm_gait_wr,
        norm_gait_tbs=norm_gait_tbs,
        norm_fk_rows=norm_fk_rows,
        norm_p1_wr=norm_p1_wr,
        norm_p1_tbs=norm_p1_tbs,
        norm_p1_rows=norm_p1_rows,
        args=args,
        overall_pass=overall_ok,
        norm_overall_pass=norm_overall_ok,
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parents[1] / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not args.nominal_only and summary_wr is not None:
            _print_geometry_table(summary_wr, summary_tbs, geometry_rows)
        _print_fk_table(gait_wr, gait_tbs, fk_rows, args.vx)
        _print_smoothness_table(smooth_wr, smooth_tbs, smooth_rows, args.vx)
        _print_trackability_table(p1_wr, p1_tbs, p1_rows)
        _print_normalized_fk_table(norm_gait_wr, norm_gait_tbs, norm_fk_rows, args.vx)
        _print_normalized_trackability_table(norm_p1_wr, norm_p1_tbs, norm_p1_rows)
        print()
        print(
            f"Absolute parity verdict:        {'PASS' if overall_ok else 'FAIL'}"
        )
        print(
            f"Size-normalised parity verdict: {'PASS' if norm_overall_ok else 'FAIL'}"
        )
        if args.out:
            print(f"JSON payload saved to: {out_path}")

    # Strict mode requires both verdicts: a pass on absolute alone could
    # mask a real proportional gap (and vice versa).
    if args.strict and not (overall_ok and norm_overall_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
