#!/usr/bin/env python3
"""Numerically verify a reference-motion retarget against the MuJoCo robot."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_robot_config
from training.envs.wildrobot_env import get_assets
from training.reference_motion.loader import load_reference_motion
from training.reference_motion.retarget import (
    LEG_ACTUATOR_NAMES,
    _align_forward_axis,
    _convert_source_root_quat_to_wxyz,
    _extract_nominal_cycle,
    _resample_gmr_motion,
    _source_dof_to_leg_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/v2/mujoco_robot_config.json",
        help="Path to WildRobot robot config JSON.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="assets/v2/scene_flat_terrain.xml",
        help="MuJoCo scene XML used for FK/contact verification.",
    )
    parser.add_argument(
        "--reference-npz",
        type=str,
        default="training/reference_motion/wildrobot_walk_forward_v002.npz",
        help="Exported reference-motion clip (.npz).",
    )
    parser.add_argument(
        "--reference-metadata",
        type=str,
        default="training/reference_motion/wildrobot_walk_forward_v002.json",
        help="Exported reference-motion metadata (.json).",
    )
    parser.add_argument(
        "--source-motion",
        type=str,
        default="training/data/gmr/walking_slow01.pkl",
        help="Source GMR motion used for retarget export.",
    )
    parser.add_argument(
        "--source-root-quat-format",
        type=str,
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion convention of source root_rot in source motion file.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="training/reference_motion/verification/wildrobot_walk_forward_v002_report.json",
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def _load_model(model_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        model_path.read_text(),
        assets=get_assets(model_path.parent),
    )


def _joint_qpos_addr(model: mujoco.MjModel, actuator_names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in actuator_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        out[name] = int(model.jnt_qposadr[joint_id])
    return out


def _set_reference_state(
    data: mujoco.MjData,
    *,
    qpos_addr: dict[str, int],
    actuator_names: list[str],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    for idx, name in enumerate(actuator_names):
        data.qpos[qpos_addr[name]] = float(joint_pos[idx])


def _velocity_from_position(values: np.ndarray, dt: float) -> np.ndarray:
    grad = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return grad
    grad[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    grad[0] = (values[1] - values[0]) / dt
    grad[-1] = (values[-1] - values[-2]) / dt
    return grad


def _load_source_motion(
    source_path: Path,
    *,
    source_root_quat_format: str,
    target_dt: float,
) -> dict[str, Any]:
    with source_path.open("rb") as f:
        source_motion = pickle.load(f)
    root_pos = np.asarray(source_motion["root_pos"], dtype=np.float32)
    root_quat = _convert_source_root_quat_to_wxyz(
        np.asarray(source_motion["root_rot"], dtype=np.float32),
        source_root_quat_format=source_root_quat_format,
    )
    dof_pos = np.asarray(source_motion["dof_pos"], dtype=np.float32)
    root_pos, root_quat, dof_pos = _resample_gmr_motion(
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=dof_pos,
        source_fps=float(source_motion["fps"]),
        target_dt=target_dt,
    )
    root_pos, root_quat, dof_pos = _extract_nominal_cycle(
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=dof_pos,
    )
    root_pos, root_quat = _align_forward_axis(root_pos, root_quat)
    return {
        "metadata": source_motion,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "dof_pos": dof_pos,
        "leg_seed": _source_dof_to_leg_order(dof_pos),
    }


def _mean_abs_deg(values_rad: np.ndarray) -> float:
    return float(np.degrees(np.mean(np.abs(values_rad))))


def _compute_touchdown_step_lengths(root_x: np.ndarray, contact: np.ndarray) -> list[float]:
    contact_i = contact.astype(np.int32)
    touchdown_idx = np.where((contact_i[1:] == 1) & (contact_i[:-1] == 0))[0] + 1
    if touchdown_idx.size < 2:
        return []
    return [float(root_x[b] - root_x[a]) for a, b in zip(touchdown_idx[:-1], touchdown_idx[1:])]


def _phase_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def verify_retarget(args: argparse.Namespace) -> dict[str, Any]:
    robot = load_robot_config(args.robot_config)
    clip = load_reference_motion(
        args.reference_npz,
        args.reference_metadata,
        actuator_names=robot.actuator_names,
    )
    source = _load_source_motion(
        Path(args.source_motion),
        source_root_quat_format=args.source_root_quat_format,
        target_dt=float(clip.dt),
    )

    model_path = Path(args.model_path)
    model = _load_model(model_path)
    data = mujoco.MjData(model)
    qpos_addr = _joint_qpos_addr(model, list(robot.actuator_names))

    left_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"))
    right_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"))
    geom_ids = {
        "left_toe": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")),
        "left_heel": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel")),
        "right_toe": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")),
        "right_heel": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel")),
    }

    frame_count = int(clip.frame_count)
    left_fk_err = np.zeros(frame_count, dtype=np.float32)
    right_fk_err = np.zeros(frame_count, dtype=np.float32)
    toe_heel_z = {name: np.zeros(frame_count, dtype=np.float32) for name in geom_ids}

    for i in range(frame_count):
        _set_reference_state(
            data,
            qpos_addr=qpos_addr,
            actuator_names=list(robot.actuator_names),
            root_pos=clip.root_pos[i],
            root_quat=clip.root_quat[i],
            joint_pos=clip.joint_pos[i],
        )
        mujoco.mj_forward(model, data)
        left_fk_err[i] = float(np.linalg.norm(data.xpos[left_body_id] - clip.left_foot_pos[i]))
        right_fk_err[i] = float(np.linalg.norm(data.xpos[right_body_id] - clip.right_foot_pos[i]))
        for name, geom_id in geom_ids.items():
            toe_heel_z[name][i] = float(data.geom_xpos[geom_id][2])

    left_min_geom_z = np.minimum(toe_heel_z["left_toe"], toe_heel_z["left_heel"])
    right_min_geom_z = np.minimum(toe_heel_z["right_toe"], toe_heel_z["right_heel"])
    left_xy_speed = np.linalg.norm(_velocity_from_position(clip.left_foot_pos[:, :2], clip.dt), axis=1)
    right_xy_speed = np.linalg.norm(_velocity_from_position(clip.right_foot_pos[:, :2], clip.dt), axis=1)
    left_contact = clip.left_foot_contact > 0.5
    right_contact = clip.right_foot_contact > 0.5
    left_proxy = (left_min_geom_z <= 0.005) & (left_xy_speed <= 0.90)
    right_proxy = (right_min_geom_z <= 0.005) & (right_xy_speed <= 0.90)

    cycle_delta = (clip.root_pos[-1] + clip.root_lin_vel[-1] * float(clip.dt)) - clip.root_pos[0]
    seam_root = (clip.root_pos[-1] - cycle_delta) - clip.root_pos[0]
    seam_left = (clip.left_foot_pos[-1] - cycle_delta) - clip.left_foot_pos[0]
    seam_right = (clip.right_foot_pos[-1] - cycle_delta) - clip.right_foot_pos[0]
    seam_joint = clip.joint_pos[-1] - clip.joint_pos[0]
    seam_quat = 1.0 - abs(float(np.dot(clip.root_quat[0], clip.root_quat[-1])))

    root_quat_xyzw = np.stack(
        [clip.root_quat[:, 1], clip.root_quat[:, 2], clip.root_quat[:, 3], clip.root_quat[:, 0]],
        axis=-1,
    )
    root_euler = Rotation.from_quat(root_quat_xyzw).as_euler("xyz", degrees=False)

    leg_indices = [list(robot.actuator_names).index(name) for name in LEG_ACTUATOR_NAMES]
    export_leg = clip.joint_pos[:, leg_indices]
    source_leg = source["leg_seed"]
    left_step_lengths = _compute_touchdown_step_lengths(clip.root_pos[:, 0], left_contact)
    right_step_lengths = _compute_touchdown_step_lengths(clip.root_pos[:, 0], right_contact)

    report: dict[str, Any] = {
        "clip": {
            "name": clip.name,
            "frame_count": frame_count,
            "dt": float(clip.dt),
            "metadata_nominal_forward_velocity_mps": float(
                clip.metadata.get("nominal_forward_velocity_mps", 0.0)
            ),
            "measured_mean_root_x_velocity_mps": float(np.mean(clip.root_lin_vel[:, 0])),
            "root_x_displacement_m": float(clip.root_pos[-1, 0] - clip.root_pos[0, 0]),
            "mean_root_height_m": float(np.mean(clip.root_pos[:, 2])),
        },
        "source": {
            "source_motion_path": str(Path(args.source_motion)),
            "source_robot": str(source["metadata"].get("robot", "unknown")),
            "source_file": str(source["metadata"].get("source_file", "unknown")),
            "source_fps": float(source["metadata"].get("fps", 0.0)),
            "resampled_frame_count": int(source_leg.shape[0]),
        },
        "fk_consistency": {
            "left_mean_error_m": float(np.mean(left_fk_err)),
            "left_max_error_m": float(np.max(left_fk_err)),
            "right_mean_error_m": float(np.mean(right_fk_err)),
            "right_max_error_m": float(np.max(right_fk_err)),
        },
        "ground_contact_geometry": {
            "left_min_geom_z": _phase_summary(left_min_geom_z),
            "right_min_geom_z": _phase_summary(right_min_geom_z),
            "left_contact_mean_geom_z_m": float(np.mean(left_min_geom_z[left_contact])),
            "left_swing_mean_geom_z_m": float(np.mean(left_min_geom_z[~left_contact])),
            "right_contact_mean_geom_z_m": float(np.mean(right_min_geom_z[right_contact])),
            "right_swing_mean_geom_z_m": float(np.mean(right_min_geom_z[~right_contact])),
            "left_contact_proxy_agreement": float(np.mean(left_proxy == left_contact)),
            "right_contact_proxy_agreement": float(np.mean(right_proxy == right_contact)),
        },
        "gait_structure": {
            "left_contact_rate": float(np.mean(left_contact)),
            "right_contact_rate": float(np.mean(right_contact)),
            "single_support_rate": float(np.mean(np.logical_xor(left_contact, right_contact))),
            "double_support_rate": float(np.mean(np.logical_and(left_contact, right_contact))),
            "flight_rate": float(np.mean(np.logical_not(np.logical_or(left_contact, right_contact)))),
            "left_transitions": int(np.sum(np.abs(np.diff(left_contact.astype(np.int32))) > 0)),
            "right_transitions": int(np.sum(np.abs(np.diff(right_contact.astype(np.int32))) > 0)),
            "left_touchdown_step_lengths_m": left_step_lengths,
            "right_touchdown_step_lengths_m": right_step_lengths,
        },
        "posture": {
            "mean_abs_roll_deg": _mean_abs_deg(root_euler[:, 0]),
            "mean_abs_pitch_deg": _mean_abs_deg(root_euler[:, 1]),
            "max_abs_roll_deg": float(np.degrees(np.max(np.abs(root_euler[:, 0])))),
            "max_abs_pitch_deg": float(np.degrees(np.max(np.abs(root_euler[:, 1])))),
        },
        "loop_seam": {
            "root_seam_norm_m": float(np.linalg.norm(seam_root)),
            "root_seam_yz_norm_m": float(np.linalg.norm(seam_root[1:])),
            "left_foot_seam_norm_m": float(np.linalg.norm(seam_left)),
            "right_foot_seam_norm_m": float(np.linalg.norm(seam_right)),
            "joint_seam_mean_abs_rad": float(np.mean(np.abs(seam_joint))),
            "joint_seam_max_abs_rad": float(np.max(np.abs(seam_joint))),
            "quat_seam_error": float(seam_quat),
        },
        "source_export_comparison": {
            "leg_joint_mae_rad": float(np.mean(np.abs(source_leg - export_leg))),
            "leg_joint_max_abs_rad": float(np.max(np.abs(source_leg - export_leg))),
            "source_root_x_displacement_m": float(source["root_pos"][-1, 0] - source["root_pos"][0, 0]),
            "source_mean_root_x_velocity_mps": float(np.mean(np.diff(source["root_pos"][:, 0]) / clip.dt)),
            "export_mean_root_x_velocity_mps": float(np.mean(clip.root_lin_vel[:, 0])),
        },
        "checks": {},
        "notes": [],
    }

    checks = report["checks"]
    checks["fk_consistency"] = bool(
        report["fk_consistency"]["left_max_error_m"] < 0.03
        and report["fk_consistency"]["right_max_error_m"] < 0.03
        and report["fk_consistency"]["left_mean_error_m"] < 0.005
        and report["fk_consistency"]["right_mean_error_m"] < 0.005
    )
    checks["forward_progress"] = bool(
        report["clip"]["measured_mean_root_x_velocity_mps"] > 0.10
        and report["clip"]["root_x_displacement_m"] > 0.20
    )
    checks["alternating_contacts"] = bool(
        report["gait_structure"]["left_transitions"] >= 2
        and report["gait_structure"]["right_transitions"] >= 2
        and report["gait_structure"]["single_support_rate"] > 0.40
        and report["gait_structure"]["flight_rate"] < 0.05
    )
    checks["loop_seam"] = bool(
        report["loop_seam"]["root_seam_norm_m"] < 0.03
        and report["loop_seam"]["left_foot_seam_norm_m"] < 0.03
        and report["loop_seam"]["right_foot_seam_norm_m"] < 0.03
        and report["loop_seam"]["joint_seam_max_abs_rad"] < 0.20
        and report["loop_seam"]["quat_seam_error"] < 0.01
    )
    checks["ground_penetration"] = bool(
        float(np.percentile(left_min_geom_z, 5)) > -0.01
        and float(np.percentile(right_min_geom_z, 5)) > -0.01
        and report["ground_contact_geometry"]["left_contact_mean_geom_z_m"] > -0.01
        and report["ground_contact_geometry"]["right_contact_mean_geom_z_m"] > -0.01
    )
    checks["contact_label_plausibility"] = bool(
        report["ground_contact_geometry"]["left_contact_proxy_agreement"] > 0.70
        and report["ground_contact_geometry"]["right_contact_proxy_agreement"] > 0.70
        and report["ground_contact_geometry"]["left_contact_mean_geom_z_m"]
        < report["ground_contact_geometry"]["left_swing_mean_geom_z_m"]
        and report["ground_contact_geometry"]["right_contact_mean_geom_z_m"]
        < report["ground_contact_geometry"]["right_swing_mean_geom_z_m"]
    )

    if report["source"]["source_robot"].lower() == "wildrobot":
        report["notes"].append(
            "Source motion is already a WildRobot GMR clip; this verifies export/adaptation quality, not a full human-to-robot retarget pipeline."
        )
    if not checks["ground_penetration"]:
        report["notes"].append(
            "Reference clip places support geoms below ground by more than the allowed margin; root height and/or foot target geometry should be corrected before training."
        )
    if report["source_export_comparison"]["leg_joint_mae_rad"] < 0.05:
        report["notes"].append(
            "Exported lower-body motion remains close to the source WildRobot clip; v1 is primarily a feasibility/adaptation pass, not a strong motion redesign."
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        report["verdict"] = "fail"
        report["failed_checks"] = failed
    else:
        report["verdict"] = "pass"
        report["failed_checks"] = []
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("Retarget verification summary")
    print(f"  clip: {report['clip']['name']} ({report['clip']['frame_count']} frames @ {report['clip']['dt']:.3f}s)")
    print(f"  verdict: {report['verdict']}")
    if report["failed_checks"]:
        print(f"  failed_checks: {', '.join(report['failed_checks'])}")
    print(
        "  root progress: "
        f"{report['clip']['measured_mean_root_x_velocity_mps']:.3f} m/s, "
        f"{report['clip']['root_x_displacement_m']:.3f} m"
    )
    print(
        "  FK error: "
        f"L mean/max {report['fk_consistency']['left_mean_error_m']:.4f}/{report['fk_consistency']['left_max_error_m']:.4f} m, "
        f"R mean/max {report['fk_consistency']['right_mean_error_m']:.4f}/{report['fk_consistency']['right_max_error_m']:.4f} m"
    )
    print(
        "  contacts: "
        f"single {report['gait_structure']['single_support_rate']:.3f}, "
        f"double {report['gait_structure']['double_support_rate']:.3f}, "
        f"flight {report['gait_structure']['flight_rate']:.3f}"
    )
    print(
        "  support geom z: "
        f"L contact {report['ground_contact_geometry']['left_contact_mean_geom_z_m']:.4f} m, "
        f"R contact {report['ground_contact_geometry']['right_contact_mean_geom_z_m']:.4f} m"
    )
    print(
        "  seam: "
        f"root {report['loop_seam']['root_seam_norm_m']:.4f} m, "
        f"L foot {report['loop_seam']['left_foot_seam_norm_m']:.4f} m, "
        f"R foot {report['loop_seam']['right_foot_seam_norm_m']:.4f} m"
    )
    print(
        "  source/export leg delta: "
        f"mae {report['source_export_comparison']['leg_joint_mae_rad']:.4f} rad, "
        f"max {report['source_export_comparison']['leg_joint_max_abs_rad']:.4f} rad"
    )
    if report["notes"]:
        print("  notes:")
        for note in report["notes"]:
            print(f"    - {note}")


def main() -> None:
    args = parse_args()
    report = verify_retarget(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
