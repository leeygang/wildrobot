#!/usr/bin/env python3
"""Compare WildRobot and ToddlerBot on the shared geometry gate.

This script mirrors the shared P0 gate from:

- ``tests/test_v0200c_geometry.py`` in WildRobot
- ``tests/test_walk_reference_geometry.py`` in ToddlerBot

It prints one compact summary table per robot and a parity verdict for
WildRobot against each ToddlerBot variant.

Usage::

    python3 tools/reference_geometry_parity.py
    python3 tools/reference_geometry_parity.py --strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


_STANCE_TOL_M = 0.003
_SWING_MIN_M = -0.002
_PARITY_MARGIN_M = 0.002
_VX_BINS = (0.10, 0.15, 0.20, 0.25)
_VX_BINS_INSCOPE = (0.10, 0.15)
_PROBE_FRAMES = (0, 5, 10, 16, 22, 28, 32, 48, 64)
_TB_VARIANTS = ("toddlerbot_2xc", "toddlerbot_2xm")


@dataclass
class GateSummary:
    name: str
    total_failures: int
    in_scope_failures: int
    worst_stance_z: float
    worst_swing_z: float
    failures: list[str]


@dataclass
class GaitMetrics:
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
    step_length_cv: float
    stride_speed_proxy_mps: float
    swing_clearance_mean_m: float
    swing_clearance_min_m: float
    double_support_frac: float
    foot_z_step_max_m: float


def _box_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Return the lowest world-z of a box geom."""
    pos = data.geom_xpos[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom_id]
    return float(
        pos[2]
        - abs(rot[2, 0]) * sx
        - abs(rot[2, 1]) * sy
        - abs(rot[2, 2]) * sz
    )


def _repo_roots() -> tuple[Path, Path]:
    """Return `(wildrobot_root, toddlerbot_root)`."""
    wildrobot_root = Path(__file__).resolve().parents[1]
    toddlerbot_root = wildrobot_root.parent / "toddlerbot"
    return wildrobot_root, toddlerbot_root


def _pick_python_with_modules(
    candidates: list[Path | str], modules: list[str], cwd: Path
) -> str:
    """Pick the first Python executable that can import all requested modules."""
    check = (
        "import importlib\n"
        f"mods = {modules!r}\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "raise SystemExit(0 if not missing else 1)\n"
    )
    for candidate in candidates:
        candidate_str = str(candidate)
        if not Path(candidate_str).exists() and "/" in candidate_str:
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


def _compute_gait_metrics(
    name: str,
    vx: float,
    contact: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    dt: float,
) -> GaitMetrics:
    """Compute task-space gait metrics from contact + foot trajectories."""
    left_touchdowns = np.where((contact[:-1, 0] < 0.5) & (contact[1:, 0] > 0.5))[0] + 1
    right_touchdowns = np.where((contact[:-1, 1] < 0.5) & (contact[1:, 1] > 0.5))[0] + 1

    step_lengths: list[float] = []
    for idx in left_touchdowns:
        step_lengths.append(float(left_foot_pos[idx, 0] - right_foot_pos[idx, 0]))
    for idx in right_touchdowns:
        step_lengths.append(float(right_foot_pos[idx, 0] - left_foot_pos[idx, 0]))
    step_arr = np.asarray(step_lengths, dtype=np.float64)

    def _swing_clearance(foot_pos: np.ndarray, contact_col: np.ndarray) -> np.ndarray:
        starts = np.where((contact_col[:-1] > 0.5) & (contact_col[1:] < 0.5))[0] + 1
        ends = np.where((contact_col[:-1] < 0.5) & (contact_col[1:] > 0.5))[0] + 1
        baseline = float(np.median(foot_pos[contact_col > 0.5, 2]))
        peaks: list[float] = []
        for start in starts:
            touchdown_candidates = ends[ends > start]
            if len(touchdown_candidates) == 0:
                continue
            end = int(touchdown_candidates[0])
            peaks.append(float(np.max(foot_pos[start : end + 1, 2]) - baseline))
        return np.asarray(peaks, dtype=np.float64)

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

    step_length_mean_m = float(np.mean(step_arr))
    step_length_std_m = float(np.std(step_arr))
    return GaitMetrics(
        name=name,
        vx=vx,
        duration_s=duration_s,
        touchdowns_left=len(left_touchdowns),
        touchdowns_right=len(right_touchdowns),
        touchdown_balance_ratio=float(min_touchdowns / max_touchdowns),
        touchdown_rate_hz=float(total_touchdowns / duration_s),
        step_length_mean_m=step_length_mean_m,
        step_length_min_m=float(np.min(step_arr)),
        step_length_max_m=float(np.max(step_arr)),
        step_length_cv=float(step_length_std_m / max(step_length_mean_m, 1e-8)),
        stride_speed_proxy_mps=float(step_length_mean_m * total_touchdowns / duration_s / 2.0),
        swing_clearance_mean_m=float(np.mean(clearance_arr)),
        swing_clearance_min_m=float(np.min(clearance_arr)),
        double_support_frac=float(np.mean(np.sum(contact, axis=1) == 2)),
        foot_z_step_max_m=foot_z_step_max_m,
    )


def _summarize_wildrobot(wildrobot_root: Path) -> GateSummary:
    """Run the WR geometry gate and return summary metrics."""
    venv_python = wildrobot_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"WildRobot venv python not found: {venv_python}")

    helper = f"""
import json
import mujoco
import numpy as np
import sys
from pathlib import Path

wildrobot_root = Path({str(wildrobot_root)!r})
sys.path.insert(0, str(wildrobot_root))

from control.zmp.zmp_walk import ZMPWalkGenerator
from training.utils.ctrl_order import CtrlOrderMapper

STANCE_TOL = {_STANCE_TOL_M!r}
SWING_MIN = {_SWING_MIN_M!r}
VX_BINS = {_VX_BINS!r}
VX_BINS_INSCOPE = {_VX_BINS_INSCOPE!r}
PROBE_FRAMES = {_PROBE_FRAMES!r}

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

model = mujoco.MjModel.from_xml_path(str(wildrobot_root / "assets" / "v2" / "scene_flat_terrain.xml"))
data = mujoco.MjData(model)
with open(wildrobot_root / "assets" / "v2" / "mujoco_robot_config.json") as f:
    spec = json.load(f)
mapper = CtrlOrderMapper(model, [j["name"] for j in spec["actuated_joint_specs"]])
act_to_qpos = np.array([model.jnt_qposadr[model.actuator_trnid[k, 0]] for k in range(model.nu)])
left_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("left_heel", "left_toe")]
right_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("right_heel", "right_toe")]

failures = []
worst_stance_z = -1e9
worst_swing_z = 1e9
gen = ZMPWalkGenerator()

for vx in VX_BINS:
    traj = gen.generate(vx)
    for frame_idx in PROBE_FRAMES:
        if frame_idx >= traj.n_steps:
            continue
        data.qpos[:] = 0.0
        mj_q = np.zeros(len(mapper.policy_to_mj_order), dtype=np.float64)
        mj_q[mapper.policy_to_mj_order] = traj.q_ref[frame_idx]
        for actuator_idx in range(len(act_to_qpos)):
            data.qpos[act_to_qpos[actuator_idx]] = mj_q[actuator_idx]
        data.qpos[0:3] = traj.pelvis_pos[frame_idx, :3]
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        left_z = min(box_min_z(model, data, geom_id) for geom_id in left_geoms)
        right_z = min(box_min_z(model, data, geom_id) for geom_id in right_geoms)
        for side, z, contact in (
            ("L", left_z, float(traj.contact_mask[frame_idx, 0])),
            ("R", right_z, float(traj.contact_mask[frame_idx, 1])),
        ):
            role = "stance" if contact > 0.5 else "swing"
            if role == "stance":
                worst_stance_z = max(worst_stance_z, z)
                if z > STANCE_TOL:
                    failures.append(
                        f"WR vx={{vx:.2f}} frame={{frame_idx}} {{side}} stance z={{z:+.4f}} > {{STANCE_TOL}}"
                    )
            else:
                worst_swing_z = min(worst_swing_z, z)
                if z < SWING_MIN:
                    failures.append(
                        f"WR vx={{vx:.2f}} frame={{frame_idx}} {{side}} swing z={{z:+.4f}} < {{SWING_MIN}}"
                    )

in_scope_failures = sum(
    any(f"vx={{vx:.2f}}" in failure for vx in VX_BINS_INSCOPE)
    for failure in failures
)
print(json.dumps({{
    "name": "wildrobot",
    "total_failures": len(failures),
    "in_scope_failures": in_scope_failures,
    "worst_stance_z": worst_stance_z,
    "worst_swing_z": worst_swing_z,
    "failures": failures,
}}))
"""
    result = subprocess.run(
        [str(venv_python), "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=wildrobot_root,
    )
    payload = json.loads(result.stdout)
    return GateSummary(**payload)


def _summarize_toddlerbot(toddlerbot_root: Path, robot_name: str) -> GateSummary:
    """Run the TB geometry gate for one robot variant."""
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
            role = "stance" if contact > 0.5 else "swing"
            if role == "stance":
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


def _gait_metrics_wildrobot(wildrobot_root: Path, vx: float) -> GaitMetrics:
    """Compute WR nominal gait metrics in the WR venv."""
    venv_python = wildrobot_root / ".venv" / "bin" / "python"
    helper = f"""
import json
import numpy as np
import sys
from pathlib import Path

wildrobot_root = Path({str(wildrobot_root)!r})
sys.path.insert(0, str(wildrobot_root))
from control.zmp.zmp_walk import ZMPWalkGenerator

vx = {vx!r}
traj = ZMPWalkGenerator().generate(vx)
contact = np.asarray(traj.contact_mask, dtype=np.float32)
left_foot_pos = np.asarray(traj.left_foot_pos, dtype=np.float32)
right_foot_pos = np.asarray(traj.right_foot_pos, dtype=np.float32)
dt = float(traj.dt)

def compute(contact, left_foot_pos, right_foot_pos, dt):
    left_touchdowns = np.where((contact[:-1, 0] < 0.5) & (contact[1:, 0] > 0.5))[0] + 1
    right_touchdowns = np.where((contact[:-1, 1] < 0.5) & (contact[1:, 1] > 0.5))[0] + 1
    step_lengths = []
    for idx in left_touchdowns:
        step_lengths.append(float(left_foot_pos[idx, 0] - right_foot_pos[idx, 0]))
    for idx in right_touchdowns:
        step_lengths.append(float(right_foot_pos[idx, 0] - left_foot_pos[idx, 0]))
    step_arr = np.asarray(step_lengths, dtype=np.float64)

    def swing_clearance(foot_pos, contact_col):
        starts = np.where((contact_col[:-1] > 0.5) & (contact_col[1:] < 0.5))[0] + 1
        ends = np.where((contact_col[:-1] < 0.5) & (contact_col[1:] > 0.5))[0] + 1
        baseline = float(np.median(foot_pos[contact_col > 0.5, 2]))
        peaks = []
        for start in starts:
            touchdown_candidates = ends[ends > start]
            if len(touchdown_candidates) == 0:
                continue
            end = int(touchdown_candidates[0])
            peaks.append(float(np.max(foot_pos[start:end+1, 2]) - baseline))
        return np.asarray(peaks, dtype=np.float64)

    clearance_arr = np.concatenate(
        [swing_clearance(left_foot_pos, contact[:, 0]), swing_clearance(right_foot_pos, contact[:, 1])],
        axis=0,
    )
    duration_s = float(len(contact) * dt)
    total_touchdowns = len(left_touchdowns) + len(right_touchdowns)
    max_touchdowns = max(len(left_touchdowns), len(right_touchdowns), 1)
    min_touchdowns = min(len(left_touchdowns), len(right_touchdowns))
    step_length_mean_m = float(np.mean(step_arr))
    step_length_std_m = float(np.std(step_arr))
    foot_z_step_max_m = max(
        float(np.max(np.abs(np.diff(left_foot_pos[:, 2])))),
        float(np.max(np.abs(np.diff(right_foot_pos[:, 2])))),
    )
    return {{
        "name": "wildrobot",
        "vx": vx,
        "duration_s": duration_s,
        "touchdowns_left": len(left_touchdowns),
        "touchdowns_right": len(right_touchdowns),
        "touchdown_balance_ratio": float(min_touchdowns / max_touchdowns),
        "touchdown_rate_hz": float(total_touchdowns / duration_s),
        "step_length_mean_m": step_length_mean_m,
        "step_length_min_m": float(np.min(step_arr)),
        "step_length_max_m": float(np.max(step_arr)),
        "step_length_cv": float(step_length_std_m / max(step_length_mean_m, 1e-8)),
        "stride_speed_proxy_mps": float(step_length_mean_m * total_touchdowns / duration_s / 2.0),
        "swing_clearance_mean_m": float(np.mean(clearance_arr)),
        "swing_clearance_min_m": float(np.min(clearance_arr)),
        "double_support_frac": float(np.mean(np.sum(contact, axis=1) == 2)),
        "foot_z_step_max_m": foot_z_step_max_m,
    }}

print(json.dumps(compute(contact, left_foot_pos, right_foot_pos, dt)))
"""
    result = subprocess.run(
        [str(venv_python), "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=wildrobot_root,
    )
    return GaitMetrics(**json.loads(result.stdout))


def _gait_metrics_toddlerbot(
    toddlerbot_root: Path, robot_name: str, vx: float
) -> GaitMetrics:
    """Compute TB nominal gait metrics from the lookup table."""
    tb_python = _pick_python_with_modules(
        [
            toddlerbot_root / ".venv" / "bin" / "python",
            "python3",
            "/usr/bin/python3",
        ],
        ["joblib", "lz4", "numpy"],
        cwd=toddlerbot_root,
    )
    helper = f"""
import json
from pathlib import Path
import joblib
import numpy as np

toddlerbot_root = Path({str(toddlerbot_root)!r})
robot_name = {robot_name!r}
vx = {vx!r}
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

left_touchdowns = np.where((contact[:-1, 0] < 0.5) & (contact[1:, 0] > 0.5))[0] + 1
right_touchdowns = np.where((contact[:-1, 1] < 0.5) & (contact[1:, 1] > 0.5))[0] + 1
step_lengths = []
for idx_td in left_touchdowns:
    step_lengths.append(float(left_foot_pos[idx_td, 0] - right_foot_pos[idx_td, 0]))
for idx_td in right_touchdowns:
    step_lengths.append(float(right_foot_pos[idx_td, 0] - left_foot_pos[idx_td, 0]))
step_arr = np.asarray(step_lengths, dtype=np.float64)

def swing_clearance(foot_pos, contact_col):
    starts = np.where((contact_col[:-1] > 0.5) & (contact_col[1:] < 0.5))[0] + 1
    ends = np.where((contact_col[:-1] < 0.5) & (contact_col[1:] > 0.5))[0] + 1
    baseline = float(np.median(foot_pos[contact_col > 0.5, 2]))
    peaks = []
    for start in starts:
        touchdown_candidates = ends[ends > start]
        if len(touchdown_candidates) == 0:
            continue
        end = int(touchdown_candidates[0])
        peaks.append(float(np.max(foot_pos[start:end+1, 2]) - baseline))
    return np.asarray(peaks, dtype=np.float64)

clearance_arr = np.concatenate(
    [swing_clearance(left_foot_pos, contact[:, 0]), swing_clearance(right_foot_pos, contact[:, 1])],
    axis=0,
)
duration_s = float(len(contact) * dt)
total_touchdowns = len(left_touchdowns) + len(right_touchdowns)
max_touchdowns = max(len(left_touchdowns), len(right_touchdowns), 1)
min_touchdowns = min(len(left_touchdowns), len(right_touchdowns))
step_length_mean_m = float(np.mean(step_arr))
step_length_std_m = float(np.std(step_arr))
foot_z_step_max_m = max(
    float(np.max(np.abs(np.diff(left_foot_pos[:, 2])))),
    float(np.max(np.abs(np.diff(right_foot_pos[:, 2])))),
)
print(json.dumps({{
    "name": robot_name,
    "vx": vx,
    "duration_s": duration_s,
    "touchdowns_left": len(left_touchdowns),
    "touchdowns_right": len(right_touchdowns),
    "touchdown_balance_ratio": float(min_touchdowns / max_touchdowns),
    "touchdown_rate_hz": float(total_touchdowns / duration_s),
    "step_length_mean_m": step_length_mean_m,
    "step_length_min_m": float(np.min(step_arr)),
    "step_length_max_m": float(np.max(step_arr)),
    "step_length_cv": float(step_length_std_m / max(step_length_mean_m, 1e-8)),
    "stride_speed_proxy_mps": float(step_length_mean_m * total_touchdowns / duration_s / 2.0),
    "swing_clearance_mean_m": float(np.mean(clearance_arr)),
    "swing_clearance_min_m": float(np.min(clearance_arr)),
    "double_support_frac": float(np.mean(np.sum(contact, axis=1) == 2)),
    "foot_z_step_max_m": foot_z_step_max_m,
}}))
"""
    result = subprocess.run(
        [tb_python, "-c", helper],
        check=True,
        capture_output=True,
        text=True,
        cwd=toddlerbot_root,
    )
    return GaitMetrics(**json.loads(result.stdout))


def _print_summary_table(summaries: list[GateSummary]) -> None:
    """Print a compact per-robot summary table."""
    print("Shared geometry gate summary")
    print(
        f"{'robot':<16} "
        f"{'failures':>16} "
        f"{'failures':>15} "
        f"{'stance_max_z':>30} "
        f"{'swing_min_z':>30}"
    )
    print(
        f"{'':<16} "
        f"{'vx<=0.15':>16} "
        f"{'all_vx':>15} "
        f"{'(fail if > +0.003)':>30} "
        f"{'(fail if < -0.002)':>30}"
    )
    print("-" * 114)
    for summary in summaries:
        print(
            f"{summary.name:<16} "
            f"{summary.in_scope_failures:>16d} "
            f"{summary.total_failures:>15d} "
            f"{summary.worst_stance_z:>+30.4f} "
            f"{summary.worst_swing_z:>+30.4f}"
        )


def _parity_row(summary_wr: GateSummary, summary_tb: GateSummary) -> tuple[bool, dict[str, str]]:
    """Return one P0 parity row for WR against one TB baseline."""
    stance_limit = summary_tb.worst_stance_z + _PARITY_MARGIN_M
    swing_limit = summary_tb.worst_swing_z - _PARITY_MARGIN_M
    stance_ok = summary_wr.worst_stance_z <= stance_limit
    swing_ok = summary_wr.worst_swing_z >= swing_limit
    in_scope_ok = summary_wr.in_scope_failures == 0 and summary_tb.in_scope_failures == 0
    full_ok = summary_wr.total_failures <= summary_tb.total_failures
    parity_ok = in_scope_ok and stance_ok and swing_ok and full_ok

    row = {
        "baseline": summary_tb.name,
        "in_scope": "PASS" if in_scope_ok else "FAIL",
        "wr_vs_tb_full": f"{summary_wr.total_failures} vs {summary_tb.total_failures}",
        "full_gate": "PASS" if full_ok else "FAIL",
        "wr_stance": f"{summary_wr.worst_stance_z:+.4f}",
        "tb_stance_cap": f"{stance_limit:+.4f}",
        "stance_gate": "PASS" if stance_ok else "FAIL",
        "wr_swing": f"{summary_wr.worst_swing_z:+.4f}",
        "tb_swing_floor": f"{swing_limit:+.4f}",
        "swing_gate": "PASS" if swing_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }
    return parity_ok, row


def _print_geometry_robot_table(
    summary_wr: GateSummary,
    summary_tbs: list[GateSummary],
    rows: list[dict[str, str]],
) -> None:
    """Print geometry metrics with robot as the first column."""
    parity_by_name = {row["baseline"]: row for row in rows}

    print()
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
    print(
        f"{summary_wr.name:<16} "
        f"{summary_wr.in_scope_failures:>18d} "
        f"{summary_wr.total_failures:>17d} "
        f"{summary_wr.worst_stance_z:>+14.4f} "
        f"{summary_wr.worst_swing_z:>+14.4f}"
    )
    for summary in summary_tbs:
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


def _gait_parity_row(metrics_wr: GaitMetrics, metrics_tb: GaitMetrics) -> tuple[bool, dict[str, str]]:
    """Return one nominal-gait parity row for WR against one TB baseline."""
    step_len_ok = metrics_wr.step_length_mean_m >= 0.90 * metrics_tb.step_length_mean_m
    cadence_ok = metrics_wr.touchdown_rate_hz <= 1.10 * metrics_tb.touchdown_rate_hz
    proxy_ok = abs(metrics_wr.stride_speed_proxy_mps - metrics_tb.stride_speed_proxy_mps) <= 0.01
    clearance_ok = metrics_wr.swing_clearance_mean_m >= 0.85 * metrics_tb.swing_clearance_mean_m
    ds_ok = abs(metrics_wr.double_support_frac - metrics_tb.double_support_frac) <= 0.05
    parity_ok = step_len_ok and cadence_ok and proxy_ok and clearance_ok and ds_ok

    row = {
        "baseline": metrics_tb.name,
        "wr_tb_step_len": f"{metrics_wr.step_length_mean_m:.4f}/{metrics_tb.step_length_mean_m:.4f}",
        "step_len_gate": "PASS" if step_len_ok else "FAIL",
        "wr_tb_touchdown_rate": f"{metrics_wr.touchdown_rate_hz:.3f}/{metrics_tb.touchdown_rate_hz:.3f}",
        "cadence_gate": "PASS" if cadence_ok else "FAIL",
        "wr_tb_stride_proxy": f"{metrics_wr.stride_speed_proxy_mps:.4f}/{metrics_tb.stride_speed_proxy_mps:.4f}",
        "proxy_gate": "PASS" if proxy_ok else "FAIL",
        "wr_tb_clearance": f"{metrics_wr.swing_clearance_mean_m:.4f}/{metrics_tb.swing_clearance_mean_m:.4f}",
        "clearance_gate": "PASS" if clearance_ok else "FAIL",
        "wr_tb_ds": f"{metrics_wr.double_support_frac:.4f}/{metrics_tb.double_support_frac:.4f}",
        "ds_gate": "PASS" if ds_ok else "FAIL",
        "verdict": "PASS" if parity_ok else "FAIL",
    }
    return parity_ok, row


def _print_gait_parity_table(rows: list[dict[str, str]], vx: float) -> None:
    """Print stride/contact parity table at the nominal comparison vx."""
    print()
    print(f"Nominal stride/contact parity @ vx={vx:.2f}")
    print(
        f"{'baseline':<16} "
        f"{'WR/TB step_len':>16} "
        f"{'step>=0.90xTB':>15} "
        f"{'WR/TB td_rate':>16} "
        f"{'rate<=1.10xTB':>16} "
        f"{'WR/TB stride_proxy':>18} "
        f"{'|diff|<=0.01':>14} "
        f"{'WR/TB clr_mean':>16} "
        f"{'clr>=0.85xTB':>14} "
        f"{'WR/TB ds_frac':>14} "
        f"{'|diff|<=0.05':>14} "
        f"{'verdict':>8}"
    )
    print("-" * 195)
    for row in rows:
        print(
            f"{row['baseline']:<16} "
            f"{row['wr_tb_step_len']:>16} "
            f"{row['step_len_gate']:>15} "
            f"{row['wr_tb_touchdown_rate']:>16} "
            f"{row['cadence_gate']:>16} "
            f"{row['wr_tb_stride_proxy']:>18} "
            f"{row['proxy_gate']:>14} "
            f"{row['wr_tb_clearance']:>16} "
            f"{row['clearance_gate']:>14} "
            f"{row['wr_tb_ds']:>14} "
            f"{row['ds_gate']:>14} "
            f"{row['verdict']:>8}"
        )


def _print_nominal_robot_table(
    gait_wr: GaitMetrics,
    gait_tbs: list[GaitMetrics],
    rows: list[dict[str, str]],
    vx: float,
) -> None:
    """Print nominal gait metrics with robot as the first column."""
    tb_by_name = {metrics.name: metrics for metrics in gait_tbs}
    parity_by_name = {row["baseline"]: row for row in rows}

    print()
    print(f"Nominal comparison table @ vx={vx:.2f}")
    print(
        f"{'robot':<16} "
        f"{'step_len_mean':>14} "
        f"{'touchdown_rate':>16} "
        f"{'stride_proxy':>14} "
        f"{'clr_mean':>12} "
        f"{'clr_min':>12} "
        f"{'ds_frac':>10} "
        f"{'td_balance':>12} "
        f"{'foot_z_step_max':>16}"
    )
    print(
        f"{'':<16} "
        f"{'(m)':>14} "
        f"{'(Hz)':>16} "
        f"{'(m/s)':>14} "
        f"{'(m)':>12} "
        f"{'(m)':>12} "
        f"{'':>10} "
        f"{'(0..1)':>12} "
        f"{'(m/frame)':>16}"
    )
    print("-" * 126)
    print(
        f"{gait_wr.name:<16} "
        f"{gait_wr.step_length_mean_m:>14.4f} "
        f"{gait_wr.touchdown_rate_hz:>16.4f} "
        f"{gait_wr.stride_speed_proxy_mps:>14.4f} "
        f"{gait_wr.swing_clearance_mean_m:>12.4f} "
        f"{gait_wr.swing_clearance_min_m:>12.4f} "
        f"{gait_wr.double_support_frac:>10.4f} "
        f"{gait_wr.touchdown_balance_ratio:>12.4f} "
        f"{gait_wr.foot_z_step_max_m:>16.4f}"
    )
    for robot_name in _TB_VARIANTS:
        metrics = tb_by_name[robot_name]
        print(
            f"{metrics.name:<16} "
            f"{metrics.step_length_mean_m:>14.4f} "
            f"{metrics.touchdown_rate_hz:>16.4f} "
            f"{metrics.stride_speed_proxy_mps:>14.4f} "
            f"{metrics.swing_clearance_mean_m:>12.4f} "
            f"{metrics.swing_clearance_min_m:>12.4f} "
            f"{metrics.double_support_frac:>10.4f} "
            f"{metrics.touchdown_balance_ratio:>12.4f} "
            f"{metrics.foot_z_step_max_m:>16.4f}"
        )
    print()
    for robot_name in _TB_VARIANTS:
        row = parity_by_name[robot_name]
        print(
            f"WR nominal parity vs {robot_name}: {row['verdict']} "
            f"(step_len={row['step_len_gate']}, cadence={row['cadence_gate']}, "
            f"proxy={row['proxy_gate']}, clearance={row['clearance_gate']}, ds_frac={row['ds_gate']})"
        )


def _build_report_payload(
    summary_wr: GateSummary,
    summary_tbs: list[GateSummary],
    geometry_parity_pairs: list[tuple[bool, dict[str, str]]],
    gait_wr: GaitMetrics,
    gait_tbs: list[GaitMetrics],
    gait_parity_pairs: list[tuple[bool, dict[str, str]]],
    vx: float,
) -> dict:
    """Build a machine-readable report payload."""
    return {
        "config": {
            "vx_nominal": vx,
            "geometry": {
                "stance_fail_if_gt_m": _STANCE_TOL_M,
                "swing_fail_if_lt_m": _SWING_MIN_M,
                "parity_margin_m": _PARITY_MARGIN_M,
                "vx_bins": list(_VX_BINS),
                "vx_bins_in_scope": list(_VX_BINS_INSCOPE),
                "probe_frames": list(_PROBE_FRAMES),
            },
            "gait_parity": {
                "step_length_min_ratio_vs_tb": 0.90,
                "touchdown_rate_max_ratio_vs_tb": 1.10,
                "stride_speed_proxy_abs_diff_max_mps": 0.01,
                "swing_clearance_mean_min_ratio_vs_tb": 0.85,
                "double_support_frac_abs_diff_max": 0.05,
            },
        },
        "geometry_summary": [asdict(summary_wr), *[asdict(summary) for summary in summary_tbs]],
        "geometry_parity": {
            "rows": [row for _ok, row in geometry_parity_pairs],
            "overall_pass": all(ok for ok, _row in geometry_parity_pairs),
        },
        "nominal_gait_metrics": [asdict(gait_wr), *[asdict(metrics) for metrics in gait_tbs]],
        "nominal_gait_parity": {
            "rows": [row for _ok, row in gait_parity_pairs],
            "overall_pass": all(ok for ok, _row in gait_parity_pairs),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare WildRobot and ToddlerBot on the shared geometry gate."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if WR fails P0 parity against any TB variant.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of formatted tables.",
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Skip the full geometry matrix and report only nominal gait metrics/parity.",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.15,
        help="Nominal velocity used for stride/contact parity metrics.",
    )
    args = parser.parse_args()

    wildrobot_root, toddlerbot_root = _repo_roots()
    if not toddlerbot_root.exists():
        raise FileNotFoundError(
            f"ToddlerBot repo not found at expected sibling path: {toddlerbot_root}"
        )

    summary_wr: GateSummary | None = None
    summary_tbs: list[GateSummary] = []
    parity_pairs: list[tuple[bool, dict[str, str]]] = []
    parity_results: list[bool] = []

    if not args.nominal_only:
        summary_wr = _summarize_wildrobot(wildrobot_root)
        summary_tbs = [
            _summarize_toddlerbot(toddlerbot_root, robot_name)
            for robot_name in _TB_VARIANTS
        ]
        parity_pairs = [_parity_row(summary_wr, summary_tb) for summary_tb in summary_tbs]
        parity_results = [ok for ok, _row in parity_pairs]

    gait_wr = _gait_metrics_wildrobot(wildrobot_root, args.vx)
    gait_tbs = [
        _gait_metrics_toddlerbot(toddlerbot_root, robot_name, args.vx)
        for robot_name in _TB_VARIANTS
    ]
    gait_parity_pairs = [_gait_parity_row(gait_wr, gait_tb) for gait_tb in gait_tbs]
    gait_parity_results = [ok for ok, _row in gait_parity_pairs]

    geometry_ok = all(parity_results) if not args.nominal_only else True
    overall_ok = geometry_ok and all(gait_parity_results)

    if args.json:
        payload = _build_report_payload(
            summary_wr=summary_wr or GateSummary("wildrobot", 0, 0, 0.0, 0.0, []),
            summary_tbs=summary_tbs,
            geometry_parity_pairs=parity_pairs,
            gait_wr=gait_wr,
            gait_tbs=gait_tbs,
            gait_parity_pairs=gait_parity_pairs,
            vx=args.vx,
        )
        payload["mode"] = {
            "nominal_only": args.nominal_only,
            "strict": args.strict,
        }
        payload["overall_pass"] = overall_ok
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not args.nominal_only:
            _print_summary_table([summary_wr, *summary_tbs])
            parity_rows = [row for _ok, row in parity_pairs]
            _print_geometry_robot_table(summary_wr, summary_tbs, parity_rows)
        gait_parity_rows = [row for _ok, row in gait_parity_pairs]
        _print_nominal_robot_table(gait_wr, gait_tbs, gait_parity_rows, args.vx)
        print()
        print(f"Overall reference parity verdict: {'PASS' if overall_ok else 'FAIL'}")

    if args.strict and not overall_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
