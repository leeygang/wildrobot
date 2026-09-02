#!/usr/bin/env python3
"""Evaluate symmetric walking stance offsets with MuJoCo forward kinematics.

The candidate offset pattern narrows the stance while counter-rotating each
ankle so the feet retain their home-pose orientation::

    left_hip_roll   += offset
    right_hip_roll  -= offset
    left_ankle_roll -= offset
    right_ankle_roll += offset

This is a geometry check, not a dynamics or policy evaluation.  It is intended
to reject mechanically invalid candidates before spending time on training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from assets.robot_config import RobotConfig
from training.configs.training_config import load_training_config


OFFSET_SIGNS = {
    "left_hip_roll": 1.0,
    "right_hip_roll": -1.0,
    "left_ankle_roll": -1.0,
    "right_ankle_roll": 1.0,
}


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(object_id)


def _home_qpos(model: mujoco.MjModel) -> np.ndarray:
    key_id = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return np.asarray(model.key_qpos[key_id], dtype=np.float64).copy()


def _rotation_delta_deg(reference: np.ndarray, candidate: np.ndarray) -> float:
    relative = reference.T @ candidate
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _geom_interval_along_axis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: Sequence[int],
    axis_world: np.ndarray,
) -> tuple[float, float]:
    lower = np.inf
    upper = -np.inf
    for geom_id in geom_ids:
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(
                "Foot-clearance calculation currently requires box collision "
                f"geometries; geom={model.geom(geom_id).name}"
            )
        rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        half_extent = float(
            np.sum(np.abs(axis_world @ rotation) * model.geom_size[geom_id])
        )
        center = float(np.dot(data.geom_xpos[geom_id], axis_world))
        lower = min(lower, center - half_extent)
        upper = max(upper, center + half_extent)
    return float(lower), float(upper)


def _count_robot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body_1 = int(model.geom_bodyid[int(contact.geom1)])
        body_2 = int(model.geom_bodyid[int(contact.geom2)])
        if body_1 != 0 and body_2 != 0:
            count += 1
    return count


def analyze_stance_candidate(
    *,
    model: mujoco.MjModel,
    robot_config: RobotConfig,
    home_qpos: np.ndarray,
    home_foot_rotations: dict[str, np.ndarray],
    offset_rad: float,
    close_feet_threshold_m: float,
    max_support_torque_ratio: float,
    max_foot_orientation_delta_deg: float,
    max_sole_height_delta_m: float,
) -> dict[str, Any]:
    """Return deterministic forward-kinematics metrics for one offset."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    for joint_name, sign in OFFSET_SIGNS.items():
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] += sign * float(offset_rad)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    root_body_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        str(robot_config.floating_base_body),
    )
    left_foot_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_config.left_foot_body
    )
    right_foot_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_config.right_foot_body
    )
    base_rotation = np.asarray(data.xmat[root_body_id]).reshape(3, 3)
    lateral_axis = base_rotation[:, 1]
    vertical_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

    left_foot_position = np.asarray(data.xpos[left_foot_id])
    right_foot_position = np.asarray(data.xpos[right_foot_id])
    foot_separation_m = abs(
        float(np.dot(left_foot_position - right_foot_position, lateral_axis))
    )

    left_geom_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in robot_config.feet_left_geoms
    ]
    right_geom_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in robot_config.feet_right_geoms
    ]
    left_interval = _geom_interval_along_axis(model, data, left_geom_ids, lateral_axis)
    right_interval = _geom_interval_along_axis(
        model, data, right_geom_ids, lateral_axis
    )
    if np.mean(left_interval) >= np.mean(right_interval):
        inner_foot_clearance_m = left_interval[0] - right_interval[1]
    else:
        inner_foot_clearance_m = right_interval[0] - left_interval[1]

    left_z_interval = _geom_interval_along_axis(
        model, data, left_geom_ids, vertical_axis
    )
    right_z_interval = _geom_interval_along_axis(
        model, data, right_geom_ids, vertical_axis
    )
    sole_height_delta_m = abs(left_z_interval[0] - right_z_interval[0])

    foot_orientation_delta_deg = {}
    for side, body_id in (("left", left_foot_id), ("right", right_foot_id)):
        candidate_rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
        foot_orientation_delta_deg[side] = _rotation_delta_deg(
            home_foot_rotations[side], candidate_rotation
        )

    whole_body_com = np.asarray(data.subtree_com[root_body_id])
    robot_mass_kg = float(model.body_subtreemass[root_body_id])
    robot_weight_n = robot_mass_kg * float(np.linalg.norm(model.opt.gravity))
    support = {}
    for side, foot_position in (
        ("left", left_foot_position),
        ("right", right_foot_position),
    ):
        lever_signed_m = float(np.dot(whole_body_com - foot_position, lateral_axis))
        moment_nm = robot_weight_n * abs(lever_signed_m)
        actuator_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hip_roll")
        torque_limit_nm = float(np.max(np.abs(model.actuator_forcerange[actuator_id])))
        support[side] = {
            "com_to_foot_lateral_signed_m": lever_signed_m,
            "com_to_foot_lateral_abs_m": abs(lever_signed_m),
            "quasi_static_gravity_moment_nm": moment_nm,
            "hip_roll_torque_limit_nm": torque_limit_nm,
            "quasi_static_support_ratio": moment_nm / torque_limit_nm,
        }

    joint_limit_margin_rad = {}
    joints_within_limits = True
    for joint_name in OFFSET_SIGNS:
        joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        position = float(data.qpos[qpos_address])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        margin = min(position - lower, upper - position)
        joint_limit_margin_rad[joint_name] = margin
        joints_within_limits = joints_within_limits and margin >= 0.0

    self_contact_count = _count_robot_contacts(model, data)
    gates = {
        "left_support_ratio": (
            support["left"]["quasi_static_support_ratio"] <= max_support_torque_ratio
        ),
        "right_support_ratio": (
            support["right"]["quasi_static_support_ratio"] <= max_support_torque_ratio
        ),
        "foot_separation": foot_separation_m >= close_feet_threshold_m,
        "inner_foot_clearance": inner_foot_clearance_m >= 0.0,
        "foot_orientation": max(foot_orientation_delta_deg.values())
        <= max_foot_orientation_delta_deg,
        "sole_height_symmetry": sole_height_delta_m <= max_sole_height_delta_m,
        "joint_limits": joints_within_limits,
        "self_collision": self_contact_count == 0,
    }
    return {
        "offset_rad": float(offset_rad),
        "offset_deg": float(np.rad2deg(offset_rad)),
        "foot_center_separation_m": foot_separation_m,
        "close_feet_threshold_m": float(close_feet_threshold_m),
        "close_feet_margin_m": foot_separation_m - close_feet_threshold_m,
        "inner_foot_clearance_m": inner_foot_clearance_m,
        "sole_height_delta_m": sole_height_delta_m,
        "foot_orientation_delta_deg": foot_orientation_delta_deg,
        "whole_body_com_world_m": whole_body_com.tolist(),
        "robot_mass_kg": robot_mass_kg,
        "robot_weight_n": robot_weight_n,
        "support": support,
        "joint_limit_margin_rad": joint_limit_margin_rad,
        "self_contact_count": self_contact_count,
        "gates": gates,
        "passed": all(gates.values()),
    }


def load_stance_inputs(
    config_path: Path,
) -> tuple[mujoco.MjModel, RobotConfig, np.ndarray, dict[str, np.ndarray], float]:
    training_config = load_training_config(config_path)
    scene_path = _resolve_project_path(training_config.env.scene_xml_path)
    robot_config_path = _resolve_project_path(training_config.env.robot_config_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    robot_config = RobotConfig.from_file(robot_config_path)
    home_qpos = _home_qpos(model)
    home_data = mujoco.MjData(model)
    home_data.qpos[:] = home_qpos
    mujoco.mj_forward(model, home_data)
    home_foot_rotations = {
        "left": np.asarray(
            home_data.xmat[
                _named_id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    robot_config.left_foot_body,
                )
            ]
        )
        .reshape(3, 3)
        .copy(),
        "right": np.asarray(
            home_data.xmat[
                _named_id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    robot_config.right_foot_body,
                )
            ]
        )
        .reshape(3, 3)
        .copy(),
    }
    return (
        model,
        robot_config,
        home_qpos,
        home_foot_rotations,
        float(training_config.env.close_feet_threshold),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify symmetric walking stance offsets with MuJoCo FK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--offset-rad",
        type=float,
        nargs="+",
        default=[0.0, 0.03, 0.035, 0.04],
        help="Symmetric hip-roll offset magnitudes to evaluate",
    )
    parser.add_argument("--max-support-torque-ratio", type=float, default=0.8)
    parser.add_argument("--max-foot-orientation-delta-deg", type=float, default=1.0)
    parser.add_argument("--max-sole-height-delta-m", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if any(offset < 0.0 for offset in args.offset_rad):
        raise ValueError("--offset-rad values must be non-negative")

    (
        model,
        robot_config,
        home_qpos,
        home_foot_rotations,
        close_feet_threshold_m,
    ) = load_stance_inputs(args.config)
    candidates = [
        analyze_stance_candidate(
            model=model,
            robot_config=robot_config,
            home_qpos=home_qpos,
            home_foot_rotations=home_foot_rotations,
            offset_rad=offset,
            close_feet_threshold_m=close_feet_threshold_m,
            max_support_torque_ratio=float(args.max_support_torque_ratio),
            max_foot_orientation_delta_deg=float(args.max_foot_orientation_delta_deg),
            max_sole_height_delta_m=float(args.max_sole_height_delta_m),
        )
        for offset in args.offset_rad
    ]
    result = {
        "config": str(args.config.resolve()),
        "offset_pattern": {
            joint_name: sign for joint_name, sign in OFFSET_SIGNS.items()
        },
        "thresholds": {
            "max_support_torque_ratio": float(args.max_support_torque_ratio),
            "max_foot_orientation_delta_deg": float(
                args.max_foot_orientation_delta_deg
            ),
            "max_sole_height_delta_m": float(args.max_sole_height_delta_m),
            "close_feet_threshold_m": close_feet_threshold_m,
        },
        "candidates": candidates,
    }

    print(
        "offset  feet_sep  close_margin  inner_clear  max_support  "
        "foot_delta  result"
    )
    for candidate in candidates:
        max_support_ratio = max(
            side["quasi_static_support_ratio"] for side in candidate["support"].values()
        )
        max_foot_delta = max(candidate["foot_orientation_delta_deg"].values())
        print(
            f"{candidate['offset_rad']:6.3f}  "
            f"{candidate['foot_center_separation_m']:8.4f}  "
            f"{candidate['close_feet_margin_m']:+12.4f}  "
            f"{candidate['inner_foot_clearance_m']:11.4f}  "
            f"{max_support_ratio:11.3f}  "
            f"{max_foot_delta:10.3f}  "
            f"{'PASS' if candidate['passed'] else 'FAIL'}"
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0 if any(candidate["passed"] for candidate in candidates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
