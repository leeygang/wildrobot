"""Verification helpers for GMR and exported reference-motion clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from training.configs.training_config import load_robot_config
from training.envs.wildrobot_env import get_assets
from training.reference_motion.retarget import LEG_ACTUATOR_NAMES, _source_dof_to_leg_order


def expand_gmr_dof_to_actuator_layout(
    dof_pos: np.ndarray,
    *,
    actuator_names: Sequence[str],
) -> np.ndarray:
    """Map an 8-DOF lower-body GMR motion into the full WildRobot actuator layout."""
    arr = np.asarray(dof_pos, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected dof_pos shape (T, D), got {arr.shape}")
    actuator_names = list(actuator_names)
    actuator_dim = len(actuator_names)
    if arr.shape[1] == actuator_dim:
        return arr.astype(np.float32)
    if arr.shape[1] != len(LEG_ACTUATOR_NAMES):
        raise ValueError(
            f"Unsupported dof_pos dimension {arr.shape[1]} for actuator_dim={actuator_dim}"
        )

    leg_order = _source_dof_to_leg_order(arr)
    actuator_name_to_idx = {name: i for i, name in enumerate(actuator_names)}
    out = np.zeros((arr.shape[0], actuator_dim), dtype=np.float32)
    for leg_idx, name in enumerate(LEG_ACTUATOR_NAMES):
        out[:, actuator_name_to_idx[name]] = leg_order[:, leg_idx]
    return out


def load_mujoco_model_with_assets(model_path: str | Path) -> mujoco.MjModel:
    model_path = Path(model_path)
    return mujoco.MjModel.from_xml_string(
        model_path.read_text(),
        assets=get_assets(model_path.parent),
    )


def map_motion_to_robot_layout(
    motion: dict[str, Any],
    *,
    model_path: str | Path,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Expand source motion to full robot actuator order and clamp to joint limits."""
    robot_config_path = Path(model_path).parent / "mujoco_robot_config.json"
    robot_cfg = load_robot_config(robot_config_path)
    actuator_names = list(robot_cfg.actuator_names)
    dof_pos = expand_gmr_dof_to_actuator_layout(motion["dof_pos"], actuator_names=actuator_names)

    joint_limits = np.asarray(
        [[joint["range"][0], joint["range"][1]] for joint in robot_cfg.actuated_joints],
        dtype=np.float32,
    )
    clipped = np.clip(dof_pos, joint_limits[:, 0], joint_limits[:, 1]).astype(np.float32)
    return clipped, actuator_names, joint_limits[:, 0], joint_limits[:, 1]


def _joint_qpos_indices(model: mujoco.MjModel, actuator_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in actuator_names
        ],
        dtype=np.int32,
    )


def compute_gmr_kinematic_verification(
    motion: dict[str, Any],
    *,
    model_path: str | Path,
) -> dict[str, Any]:
    """Compute kinematic grounding/posture metrics for a GMR motion on the MuJoCo model."""
    model = load_mujoco_model_with_assets(model_path)
    data = mujoco.MjData(model)
    dof_pos, actuator_names, _, _ = map_motion_to_robot_layout(motion, model_path=model_path)
    qpos_idx = _joint_qpos_indices(model, actuator_names)

    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    root_quat = np.zeros_like(root_rot_xyzw)
    root_quat[:, 0] = root_rot_xyzw[:, 3]
    root_quat[:, 1] = root_rot_xyzw[:, 0]
    root_quat[:, 2] = root_rot_xyzw[:, 1]
    root_quat[:, 3] = root_rot_xyzw[:, 2]

    geom_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in ("left_toe", "left_heel", "right_toe", "right_heel")
    }
    body_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in ("left_foot", "right_foot")
    }

    frame_count = int(dof_pos.shape[0])
    geom_z = {name: np.zeros(frame_count, dtype=np.float32) for name in geom_ids}
    foot_body_y = {name: np.zeros(frame_count, dtype=np.float32) for name in body_ids}
    foot_body_x = {name: np.zeros(frame_count, dtype=np.float32) for name in body_ids}

    for i in range(frame_count):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        data.qpos[qpos_idx] = dof_pos[i]
        mujoco.mj_forward(model, data)
        for name, geom_id in geom_ids.items():
            geom_z[name][i] = float(data.geom_xpos[geom_id][2])
        for name, body_id in body_ids.items():
            foot_body_x[name][i] = float(data.xpos[body_id][0])
            foot_body_y[name][i] = float(data.xpos[body_id][1])

    lowest_support_z = np.minimum.reduce(
        [geom_z["left_toe"], geom_z["left_heel"], geom_z["right_toe"], geom_z["right_heel"]]
    )
    stance_width = foot_body_y["left_foot"] - foot_body_y["right_foot"]
    step_length = foot_body_x["left_foot"] - foot_body_x["right_foot"]

    euler = Rotation.from_quat(root_rot_xyzw).as_euler("xyz", degrees=True)
    left_knee = np.degrees(dof_pos[:, actuator_names.index("left_knee_pitch")])
    right_knee = np.degrees(dof_pos[:, actuator_names.index("right_knee_pitch")])

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "frame_count": frame_count,
        "lowest_support_z_m": _summary(lowest_support_z),
        "left_toe_z_m": _summary(geom_z["left_toe"]),
        "left_heel_z_m": _summary(geom_z["left_heel"]),
        "right_toe_z_m": _summary(geom_z["right_toe"]),
        "right_heel_z_m": _summary(geom_z["right_heel"]),
        "stance_width_m": _summary(stance_width),
        "left_minus_right_foot_x_m": _summary(step_length),
        "mean_abs_pitch_deg": float(np.mean(np.abs(euler[:, 1]))),
        "max_abs_pitch_deg": float(np.max(np.abs(euler[:, 1]))),
        "mean_abs_roll_deg": float(np.mean(np.abs(euler[:, 0]))),
        "max_abs_roll_deg": float(np.max(np.abs(euler[:, 0]))),
        "left_knee_deg": _summary(left_knee),
        "right_knee_deg": _summary(right_knee),
    }
