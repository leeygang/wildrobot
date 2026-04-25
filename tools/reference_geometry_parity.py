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
from dataclasses import dataclass
from pathlib import Path

import joblib
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
    scene_path = toddlerbot_root / "toddlerbot" / "descriptions" / robot_name / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    q_home = np.array(model.keyframe("home").qpos, dtype=np.float64)

    suffix = "_2xc" if "2xc" in robot_name else "_2xm"
    lookup_path = toddlerbot_root / "motion" / f"walk_zmp{suffix}.lz4"
    lookup_keys, motion_ref_list = joblib.load(lookup_path)
    lookup_keys = np.array(lookup_keys, dtype=np.float32)

    left_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link_collision"
    )
    right_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link_collision"
    )

    failures: list[str] = []
    worst_stance_z = -1e9
    worst_swing_z = 1e9

    for vx in _VX_BINS:
        target = np.array([vx, 0.0, 0.0], dtype=np.float32)
        idx = int(np.argmin(np.linalg.norm(lookup_keys - target, axis=1)))
        traj = motion_ref_list[idx]

        for frame_idx in _PROBE_FRAMES:
            if frame_idx >= len(traj["qpos"]):
                continue

            data.qpos[:] = q_home
            data.qpos[7:] = np.asarray(traj["qpos"][frame_idx], dtype=np.float64)
            mujoco.mj_forward(model, data)

            left_z = _box_min_z(model, data, left_geom)
            right_z = _box_min_z(model, data, right_geom)

            for side, z, contact in (
                ("L", left_z, float(traj["contact"][frame_idx][0])),
                ("R", right_z, float(traj["contact"][frame_idx][1])),
            ):
                role = "stance" if contact > 0.5 else "swing"
                if role == "stance":
                    worst_stance_z = max(worst_stance_z, z)
                    if z > _STANCE_TOL_M:
                        failures.append(
                            f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                            f"stance z={z:+.4f} > {_STANCE_TOL_M}"
                        )
                else:
                    worst_swing_z = min(worst_swing_z, z)
                    if z < _SWING_MIN_M:
                        failures.append(
                            f"{robot_name} vx={vx:.2f} frame={frame_idx} {side} "
                            f"swing z={z:+.4f} < {_SWING_MIN_M}"
                        )

    in_scope_failures = sum(
        any(f"vx={vx:.2f}" in failure for vx in _VX_BINS_INSCOPE)
        for failure in failures
    )
    return GateSummary(
        name=robot_name,
        total_failures=len(failures),
        in_scope_failures=in_scope_failures,
        worst_stance_z=worst_stance_z,
        worst_swing_z=worst_swing_z,
        failures=failures,
    )


def _print_summary_table(summaries: list[GateSummary]) -> None:
    """Print a compact per-robot summary table."""
    print("Shared geometry gate summary")
    print(
        f"{'robot':<16} {'in_scope':>8} {'full':>6} "
        f"{'worst_stance_z':>16} {'worst_swing_z':>15}"
    )
    print("-" * 68)
    for summary in summaries:
        print(
            f"{summary.name:<16} {summary.in_scope_failures:>8d} {summary.total_failures:>6d} "
            f"{summary.worst_stance_z:>+16.4f} {summary.worst_swing_z:>+15.4f}"
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


def _print_parity_table(rows: list[dict[str, str]]) -> None:
    """Print compact WR-vs-TB parity table."""
    print()
    print("P0 parity table")
    print(
        f"{'baseline':<16} {'in_scope':>8} {'WR/TB full':>12} {'full':>6} "
        f"{'WR stance':>10} {'TB+margin':>10} {'stance':>8} "
        f"{'WR swing':>10} {'TB-margin':>10} {'swing':>8} {'verdict':>8}"
    )
    print("-" * 122)
    for row in rows:
        print(
            f"{row['baseline']:<16} {row['in_scope']:>8} {row['wr_vs_tb_full']:>12} {row['full_gate']:>6} "
            f"{row['wr_stance']:>10} {row['tb_stance_cap']:>10} {row['stance_gate']:>8} "
            f"{row['wr_swing']:>10} {row['tb_swing_floor']:>10} {row['swing_gate']:>8} {row['verdict']:>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare WildRobot and ToddlerBot on the shared geometry gate."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if WR fails P0 parity against any TB variant.",
    )
    args = parser.parse_args()

    wildrobot_root, toddlerbot_root = _repo_roots()
    if not toddlerbot_root.exists():
        raise FileNotFoundError(
            f"ToddlerBot repo not found at expected sibling path: {toddlerbot_root}"
        )

    summary_wr = _summarize_wildrobot(wildrobot_root)
    summary_tbs = [
        _summarize_toddlerbot(toddlerbot_root, robot_name)
        for robot_name in _TB_VARIANTS
    ]
    _print_summary_table([summary_wr, *summary_tbs])

    parity_pairs = [_parity_row(summary_wr, summary_tb) for summary_tb in summary_tbs]
    parity_results = [ok for ok, _row in parity_pairs]
    parity_rows = [row for _ok, row in parity_pairs]
    _print_parity_table(parity_rows)
    overall_ok = all(parity_results)

    print()
    print(f"Overall P0 parity verdict: {'PASS' if overall_ok else 'FAIL'}")

    if args.strict and not overall_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
