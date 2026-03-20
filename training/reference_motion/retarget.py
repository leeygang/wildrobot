"""WildRobot-specific reference-motion retarget/export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from training.reference_motion.loader import ReferenceMotionClip
from training.reference_motion.phase import wrap_phase

LEG_ACTUATOR_NAMES = (
    "left_hip_pitch",
    "right_hip_pitch",
    "left_hip_roll",
    "right_hip_roll",
    "left_knee_pitch",
    "right_knee_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
)

SOURCE_DOF_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
)


@dataclass(frozen=True)
class JointLimits:
    """Per-actuator joint limits in WildRobot actuator order."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _KinematicsContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_qpos_addr: dict[str, int]
    left_foot_body_id: int
    right_foot_body_id: int


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """First-order finite difference with wrap-safe edge handling."""
    if values.shape[0] < 2:
        return np.zeros_like(values)
    grad = np.zeros_like(values, dtype=np.float32)
    grad[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    grad[0] = (values[1] - values[0]) / dt
    grad[-1] = (values[-1] - values[-2]) / dt
    return grad.astype(np.float32)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.clip(norms, 1e-8, None)).astype(np.float32)


def _continuous_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(quat).copy()
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    return np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=-1)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    return np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=-1)


def _root_ang_vel_from_quat(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    q = _continuous_quat_wxyz(quat_wxyz)
    r = Rotation.from_quat(_quat_wxyz_to_xyzw(q))
    rel = (r[1:] * r[:-1].inv()).as_rotvec() / float(dt)
    ang_vel = np.zeros((q.shape[0], 3), dtype=np.float32)
    if rel.shape[0] == 0:
        return ang_vel
    ang_vel[0] = rel[0].astype(np.float32)
    ang_vel[-1] = rel[-1].astype(np.float32)
    if rel.shape[0] > 1:
        ang_vel[1:-1] = (0.5 * (rel[:-1] + rel[1:])).astype(np.float32)
    return ang_vel


def _convert_source_root_quat_to_wxyz(
    source_root_quat: np.ndarray,
    *,
    source_root_quat_format: str,
) -> np.ndarray:
    fmt = str(source_root_quat_format).strip().lower()
    quat = np.asarray(source_root_quat, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"source root quaternion must have shape (T, 4), got {quat.shape}")
    if fmt == "wxyz":
        return _continuous_quat_wxyz(quat)
    if fmt == "xyzw":
        return _continuous_quat_wxyz(_quat_xyzw_to_wxyz(quat))
    raise ValueError(f"Unsupported source_root_quat_format={source_root_quat_format!r}, expected 'xyzw' or 'wxyz'")


def _interp_array(values: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return np.interp(t_dst, t_src, arr).astype(np.float32)
    out = np.empty((t_dst.shape[0], arr.shape[1]), dtype=np.float32)
    for idx in range(arr.shape[1]):
        out[:, idx] = np.interp(t_dst, t_src, arr[:, idx]).astype(np.float32)
    return out


def _resample_gmr_motion(
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: float,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = int(root_pos.shape[0])
    duration = float((frame_count - 1) / source_fps)
    target_fps = 1.0 / float(target_dt)
    out_frames = max(8, int(round(duration * target_fps)) + 1)
    t_src = np.linspace(0.0, duration, frame_count, dtype=np.float64)
    t_dst = np.linspace(0.0, duration, out_frames, dtype=np.float64)

    resampled_root_pos = _interp_array(root_pos, t_src, t_dst)
    resampled_dof = _interp_array(dof_pos, t_src, t_dst)

    src_rot = Rotation.from_quat(_quat_wxyz_to_xyzw(_continuous_quat_wxyz(root_quat)))
    slerp = Slerp(t_src, src_rot)
    dst_rot = slerp(t_dst).as_quat()
    resampled_root_quat = _quat_xyzw_to_wxyz(dst_rot)
    return resampled_root_pos, _continuous_quat_wxyz(resampled_root_quat), resampled_dof


def _extract_nominal_cycle(
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    min_cycle_frames: int = 45,
    max_cycle_frames: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(dof_pos.shape[0])
    if n <= min_cycle_frames + 2:
        return root_pos, root_quat, dof_pos

    signal = np.asarray(dof_pos[:, 2], dtype=np.float32) - float(np.mean(dof_pos[:, 2]))
    lag_min = min(min_cycle_frames, n // 3)
    lag_max = min(max_cycle_frames, n - 2)
    if lag_max <= lag_min:
        return root_pos, root_quat, dof_pos

    best_lag = lag_min
    best_corr = -np.inf
    for lag in range(lag_min, lag_max + 1):
        corr = float(np.dot(signal[:-lag], signal[lag:]))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    best_start = 0
    best_score = np.inf
    for start in range(0, n - best_lag):
        end = start + best_lag
        seam_joint = float(np.linalg.norm(dof_pos[end] - dof_pos[start]))
        drift_x = float(root_pos[end, 0] - root_pos[start, 0])
        seam_root = float(np.linalg.norm((root_pos[end] - root_pos[start]) - np.array([drift_x, 0.0, 0.0])))
        seam_quat = 1.0 - abs(float(np.dot(root_quat[start], root_quat[end])))
        score = seam_joint + 2.0 * seam_root + 0.3 * seam_quat
        if score < best_score:
            best_score = score
            best_start = start

    window = slice(best_start, best_start + best_lag)
    return root_pos[window], root_quat[window], dof_pos[window]


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    x = xy[:, 0]
    y = xy[:, 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1).astype(np.float32)


def _align_forward_axis(root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aligned_pos = np.asarray(root_pos, dtype=np.float32).copy()
    aligned_quat = np.asarray(root_quat, dtype=np.float32).copy()
    aligned_pos -= aligned_pos[0]

    disp_xy = aligned_pos[-1, :2] - aligned_pos[0, :2]
    yaw = float(np.arctan2(disp_xy[1], disp_xy[0])) if float(np.linalg.norm(disp_xy)) > 1e-6 else 0.0
    rot_q = _quat_from_yaw(-yaw)
    aligned_pos[:, :2] = _rotate_xy(aligned_pos[:, :2], -yaw)
    aligned_pos[:, 1] -= float(np.mean(aligned_pos[:, 1]))

    for i in range(aligned_quat.shape[0]):
        aligned_quat[i] = _quat_multiply_wxyz(rot_q, aligned_quat[i])
    aligned_quat = _continuous_quat_wxyz(aligned_quat)

    if float(np.mean(np.diff(aligned_pos[:, 0]))) < 0.0:
        extra_q = _quat_from_yaw(np.pi)
        aligned_pos[:, :2] = _rotate_xy(aligned_pos[:, :2], np.pi)
        for i in range(aligned_quat.shape[0]):
            aligned_quat[i] = _quat_multiply_wxyz(extra_q, aligned_quat[i])
        aligned_quat = _continuous_quat_wxyz(aligned_quat)
    return aligned_pos, aligned_quat


def _create_kinematics_context(
    *,
    model_path: str | Path,
    leg_joint_names: Sequence[str],
) -> _KinematicsContext:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_qpos_addr: dict[str, int] = {}
    for name in leg_joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        joint_qpos_addr[name] = int(model.jnt_qposadr[jnt_id])

    left_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    if left_foot_body_id < 0 or right_foot_body_id < 0:
        raise ValueError("Expected left_foot and right_foot bodies in MuJoCo model")

    return _KinematicsContext(
        model=model,
        data=data,
        joint_qpos_addr=joint_qpos_addr,
        left_foot_body_id=int(left_foot_body_id),
        right_foot_body_id=int(right_foot_body_id),
    )


def _fk_feet(
    *,
    ctx: _KinematicsContext,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    leg_joint_values: np.ndarray,
    leg_joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    data = ctx.data
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    for idx, name in enumerate(leg_joint_names):
        data.qpos[ctx.joint_qpos_addr[name]] = float(leg_joint_values[idx])
    mujoco.mj_forward(ctx.model, data)
    left = np.asarray(data.xpos[ctx.left_foot_body_id], dtype=np.float32).copy()
    right = np.asarray(data.xpos[ctx.right_foot_body_id], dtype=np.float32).copy()
    return left, right


def _source_dof_to_leg_order(source_dof: np.ndarray) -> np.ndarray:
    # Source order: [LHP, LHR, LKP, LAP, RHP, RHR, RKP, RAP]
    # Leg order:    [LHP, RHP, LHR, RHR, LKP, RKP, LAP, RAP]
    out = np.zeros((source_dof.shape[0], len(LEG_ACTUATOR_NAMES)), dtype=np.float32)
    out[:, 0] = source_dof[:, 0]
    out[:, 1] = source_dof[:, 4]
    out[:, 2] = source_dof[:, 1]
    out[:, 3] = source_dof[:, 5]
    out[:, 4] = source_dof[:, 2]
    out[:, 5] = source_dof[:, 6]
    out[:, 6] = source_dof[:, 3]
    out[:, 7] = source_dof[:, 7]
    return out.astype(np.float32)


def _estimate_contacts(
    *,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_h = left_foot_pos[:, 2]
    right_h = right_foot_pos[:, 2]
    left_vz = _finite_difference(left_h.reshape(-1, 1), dt).reshape(-1)
    right_vz = _finite_difference(right_h.reshape(-1, 1), dt).reshape(-1)
    left_speed = np.linalg.norm(_finite_difference(left_foot_pos[:, :2], dt), axis=1)
    right_speed = np.linalg.norm(_finite_difference(right_foot_pos[:, :2], dt), axis=1)

    left_th = float(np.percentile(left_h, 35.0) + 0.008)
    right_th = float(np.percentile(right_h, 35.0) + 0.008)

    left_contact = (left_h <= left_th) & (np.abs(left_vz) <= 0.35) & (left_speed <= 0.90)
    right_contact = (right_h <= right_th) & (np.abs(right_vz) <= 0.35) & (right_speed <= 0.90)

    for i in range(left_contact.shape[0]):
        if not left_contact[i] and not right_contact[i]:
            if left_h[i] <= right_h[i]:
                left_contact[i] = True
            else:
                right_contact[i] = True
        if left_contact[i] and right_contact[i]:
            if left_h[i] + 0.018 < right_h[i]:
                right_contact[i] = False
            elif right_h[i] + 0.018 < left_h[i]:
                left_contact[i] = False
    return left_contact.astype(np.float32), right_contact.astype(np.float32)


def retarget_wildrobot_clip(
    clip: ReferenceMotionClip,
    *,
    joint_limits: JointLimits,
    pelvis_height_range: tuple[float, float] = (0.38, 0.52),
    stance_width_range: tuple[float, float] = (0.14, 0.26),
    retarget_version: str = "wildrobot_v0.16.0",
) -> ReferenceMotionClip:
    """Apply minimal WildRobot feasibility constraints to a clip."""
    joint_pos = np.clip(clip.joint_pos, joint_limits.lower, joint_limits.upper).astype(np.float32)
    joint_vel = _finite_difference(joint_pos, clip.dt)

    root_pos = np.asarray(clip.root_pos, dtype=np.float32).copy()
    root_pos[:, 2] = np.clip(root_pos[:, 2], pelvis_height_range[0], pelvis_height_range[1])
    root_lin_vel = _finite_difference(root_pos, clip.dt)

    left_foot_pos = np.asarray(clip.left_foot_pos, dtype=np.float32).copy()
    right_foot_pos = np.asarray(clip.right_foot_pos, dtype=np.float32).copy()
    center_y = 0.5 * (left_foot_pos[:, 1] + right_foot_pos[:, 1])
    width = np.clip(left_foot_pos[:, 1] - right_foot_pos[:, 1], stance_width_range[0], stance_width_range[1])
    left_foot_pos[:, 1] = center_y + 0.5 * width
    right_foot_pos[:, 1] = center_y - 0.5 * width

    metadata = dict(clip.metadata)
    metadata["retarget_version"] = retarget_version
    metadata["pelvis_height_m"] = float(np.mean(root_pos[:, 2]))
    metadata["stance_width_m"] = float(np.mean(left_foot_pos[:, 1] - right_foot_pos[:, 1]))

    return ReferenceMotionClip(
        name=clip.name,
        dt=clip.dt,
        phase=wrap_phase(clip.phase),
        root_pos=root_pos,
        root_quat=np.asarray(clip.root_quat, dtype=np.float32),
        root_lin_vel=root_lin_vel,
        root_ang_vel=np.asarray(clip.root_ang_vel, dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
        left_foot_contact=np.asarray(clip.left_foot_contact, dtype=np.float32),
        right_foot_contact=np.asarray(clip.right_foot_contact, dtype=np.float32),
        metadata=metadata,
    )


def generate_real_retarget_nominal_walk_clip(
    *,
    actuator_names: Sequence[str],
    joint_limits: JointLimits,
    source_path: str | Path = "training/data/gmr/walking_slow01.pkl",
    model_path: str | Path = "assets/v2/scene_flat_terrain.xml",
    source_root_quat_format: str = "xyzw",
    dt: float = 0.02,
    name: str = "wildrobot_walk_forward_v002",
    pelvis_height_range: tuple[float, float] = (0.39, 0.52),
    stance_width_range: tuple[float, float] = (0.14, 0.24),
    min_knee_flexion_rad: float = 0.12,
) -> ReferenceMotionClip:
    """Generate `real retarget v1` from a local GMR source walk clip."""
    if len(actuator_names) != int(joint_limits.lower.shape[0]):
        raise ValueError("actuator_names and joint_limits size mismatch")
    source_path = Path(source_path)
    model_path = Path(model_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with source_path.open("rb") as f:
        source_motion = pickle.load(f)
    if not isinstance(source_motion, dict):
        raise ValueError("source motion must be a dict")
    for key in ("root_pos", "root_rot", "dof_pos", "fps"):
        if key not in source_motion:
            raise ValueError(f"source motion missing required key: {key}")

    raw_root_pos = np.asarray(source_motion["root_pos"], dtype=np.float32)
    raw_root_quat_wxyz = _convert_source_root_quat_to_wxyz(
        np.asarray(source_motion["root_rot"], dtype=np.float32),
        source_root_quat_format=source_root_quat_format,
    )
    raw_dof_pos = np.asarray(source_motion["dof_pos"], dtype=np.float32)
    if raw_dof_pos.shape[1] != len(SOURCE_DOF_NAMES):
        raise ValueError(
            f"Expected source dof_pos second dim={len(SOURCE_DOF_NAMES)}, got {raw_dof_pos.shape[1]}"
        )
    source_fps = float(source_motion["fps"])
    if source_fps <= 1e-6:
        raise ValueError(f"Invalid source fps: {source_fps}")

    root_pos, root_quat, source_dof = _resample_gmr_motion(
        root_pos=raw_root_pos,
        root_quat=raw_root_quat_wxyz,
        dof_pos=raw_dof_pos,
        source_fps=source_fps,
        target_dt=dt,
    )
    root_pos, root_quat, source_dof = _extract_nominal_cycle(
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=source_dof,
    )
    root_pos, root_quat = _align_forward_axis(root_pos, root_quat)
    root_z = root_pos[:, 2] - float(np.mean(root_pos[:, 2])) + 0.44
    root_pos[:, 2] = np.clip(root_z, pelvis_height_range[0], pelvis_height_range[1])
    root_quat = _continuous_quat_wxyz(root_quat)

    leg_seed = _source_dof_to_leg_order(source_dof)
    actuator_index = {name: i for i, name in enumerate(actuator_names)}
    leg_indices = [actuator_index[name] for name in LEG_ACTUATOR_NAMES]
    leg_lower = joint_limits.lower[leg_indices].astype(np.float32)
    leg_upper = joint_limits.upper[leg_indices].astype(np.float32)
    leg_seed = np.clip(leg_seed, leg_lower, leg_upper)

    ctx = _create_kinematics_context(model_path=model_path, leg_joint_names=LEG_ACTUATOR_NAMES)
    target_left = np.zeros((leg_seed.shape[0], 3), dtype=np.float32)
    target_right = np.zeros((leg_seed.shape[0], 3), dtype=np.float32)
    for i in range(leg_seed.shape[0]):
        left_i, right_i = _fk_feet(
            ctx=ctx,
            root_pos=root_pos[i],
            root_quat=root_quat[i],
            leg_joint_values=leg_seed[i],
            leg_joint_names=LEG_ACTUATOR_NAMES,
        )
        target_left[i] = left_i
        target_right[i] = right_i
    center_y = 0.5 * (target_left[:, 1] + target_right[:, 1])
    width = np.clip(target_left[:, 1] - target_right[:, 1], stance_width_range[0], stance_width_range[1])
    target_left[:, 1] = center_y + 0.5 * width
    target_right[:, 1] = center_y - 0.5 * width

    solved_leg = np.zeros_like(leg_seed, dtype=np.float32)
    prev = leg_seed[0].copy()
    for i in range(leg_seed.shape[0]):
        source_i = leg_seed[i]
        left_target_i = target_left[i]
        right_target_i = target_right[i]

        def residual(x: np.ndarray) -> np.ndarray:
            left_fk, right_fk = _fk_feet(
                ctx=ctx,
                root_pos=root_pos[i],
                root_quat=root_quat[i],
                leg_joint_values=x.astype(np.float32),
                leg_joint_names=LEG_ACTUATOR_NAMES,
            )
            knee = x[[4, 5]]
            foot_res = 14.0 * np.concatenate([left_fk - left_target_i, right_fk - right_target_i], axis=0)
            prior_res = 0.55 * (x - source_i)
            smooth_res = 0.35 * (x - prev)
            knee_res = 4.0 * np.clip(float(min_knee_flexion_rad) - knee, 0.0, None)
            width_res = np.asarray(
                [
                    3.0
                    * (
                        np.clip(left_fk[1] - right_fk[1], stance_width_range[0], stance_width_range[1])
                        - (left_target_i[1] - right_target_i[1])
                    )
                ],
                dtype=np.float32,
            )
            return np.concatenate([foot_res, prior_res, smooth_res, knee_res, width_res], axis=0).astype(
                np.float64
            )

        solve = least_squares(
            residual,
            x0=source_i.astype(np.float64),
            bounds=(leg_lower.astype(np.float64), leg_upper.astype(np.float64)),
            max_nfev=50,
            method="trf",
        )
        solved = solve.x.astype(np.float32)
        solved_leg[i] = solved
        prev = solved

    frame_count = int(solved_leg.shape[0])
    joint_pos = np.zeros((frame_count, len(actuator_names)), dtype=np.float32)
    for j, idx in enumerate(leg_indices):
        joint_pos[:, idx] = solved_leg[:, j]
    if "waist_yaw" in actuator_index:
        root_yaw = Rotation.from_quat(_quat_wxyz_to_xyzw(root_quat)).as_euler("xyz", degrees=False)[:, 2]
        joint_pos[:, actuator_index["waist_yaw"]] = np.clip(0.25 * root_yaw, -0.15, 0.15).astype(np.float32)

    joint_pos = np.clip(joint_pos, joint_limits.lower, joint_limits.upper).astype(np.float32)
    joint_vel = _finite_difference(joint_pos, dt)

    left_foot_pos = np.zeros((frame_count, 3), dtype=np.float32)
    right_foot_pos = np.zeros((frame_count, 3), dtype=np.float32)
    for i in range(frame_count):
        leg_state = np.asarray([joint_pos[i, idx] for idx in leg_indices], dtype=np.float32)
        left_i, right_i = _fk_feet(
            ctx=ctx,
            root_pos=root_pos[i],
            root_quat=root_quat[i],
            leg_joint_values=leg_state,
            leg_joint_names=LEG_ACTUATOR_NAMES,
        )
        left_foot_pos[i] = left_i
        right_foot_pos[i] = right_i

    left_contact, right_contact = _estimate_contacts(
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
        dt=dt,
    )

    phase = (np.arange(frame_count, dtype=np.float32) / float(frame_count)).astype(np.float32)
    root_lin_vel = _finite_difference(root_pos, dt)
    root_ang_vel = _root_ang_vel_from_quat(root_quat, dt)
    nominal_speed = float(np.mean(root_lin_vel[:, 0]))

    metadata = {
        "name": name,
        "source_name": str(source_path.name),
        "source_format": "gmr_pkl",
        "source_root_quat_format": str(source_root_quat_format),
        "frame_count": frame_count,
        "dt": float(dt),
        "actuator_names": list(actuator_names),
        "loop_mode": "wrap",
        "phase_mode": "normalized_cycle",
        "retarget_version": "wildrobot_real_retarget_v1",
        "nominal_forward_velocity_mps": nominal_speed,
        "notes": (
            "real retarget v1 from local GMR source (walking_slow01). "
            "8-DOF lower-body source mapped to 19-actuator WildRobot via per-frame least_squares "
            "foot-target feasibility solve with joint bounds, knee-floor regularization, and frame-to-frame "
            "smoothness. This v1 pass adapts an already WildRobot-retargeted source clip rather than performing "
            "a full de novo retarget."
        ),
    }

    clip = ReferenceMotionClip(
        name=name,
        dt=float(dt),
        phase=wrap_phase(phase),
        root_pos=root_pos.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        left_foot_pos=left_foot_pos.astype(np.float32),
        right_foot_pos=right_foot_pos.astype(np.float32),
        left_foot_contact=left_contact.astype(np.float32),
        right_foot_contact=right_contact.astype(np.float32),
        metadata=metadata,
    )
    return retarget_wildrobot_clip(
        clip,
        joint_limits=joint_limits,
        pelvis_height_range=pelvis_height_range,
        stance_width_range=stance_width_range,
        retarget_version="wildrobot_real_retarget_v1",
    )


def generate_bootstrap_nominal_walk_clip(
    *,
    actuator_names: Sequence[str],
    joint_limits: JointLimits,
    frame_count: int = 72,
    dt: float = 0.02,
    nominal_forward_velocity_mps: float = 0.10,
    name: str = "wildrobot_walk_forward_v001",
) -> ReferenceMotionClip:
    """Generate a deterministic robot-native nominal walk clip (bootstrap placeholder)."""
    if frame_count < 4:
        raise ValueError(f"frame_count must be >= 4, got {frame_count}")
    if len(actuator_names) != int(joint_limits.lower.shape[0]):
        raise ValueError("actuator_names and joint_limits size mismatch")

    t = np.arange(frame_count, dtype=np.float32)
    phase = (t / float(frame_count)).astype(np.float32)
    theta = 2.0 * np.pi * phase
    x = nominal_forward_velocity_mps * dt * t

    root_pos = np.stack(
        [
            x,
            np.zeros_like(x),
            0.45 + 0.006 * np.sin(2.0 * theta),
        ],
        axis=-1,
    ).astype(np.float32)
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frame_count, 1))
    root_lin_vel = np.stack(
        [
            np.full_like(x, nominal_forward_velocity_mps),
            np.zeros_like(x),
            _finite_difference(root_pos[:, 2:3], dt).reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)
    root_ang_vel = np.zeros((frame_count, 3), dtype=np.float32)

    a = len(actuator_names)
    joint_pos = np.zeros((frame_count, a), dtype=np.float32)
    index = {name: i for i, name in enumerate(actuator_names)}
    swing = np.sin(theta)
    swing_opposite = np.sin(theta + np.pi)

    def _set_if_exists(name: str, value: np.ndarray) -> None:
        if name in index:
            joint_pos[:, index[name]] = value.astype(np.float32)

    _set_if_exists("left_hip_pitch", 0.22 + 0.20 * swing)
    _set_if_exists("right_hip_pitch", -0.22 + 0.20 * swing_opposite)
    _set_if_exists("left_knee_pitch", 0.32 + 0.28 * np.clip(swing, 0.0, None))
    _set_if_exists("right_knee_pitch", 0.32 + 0.28 * np.clip(swing_opposite, 0.0, None))
    _set_if_exists("left_ankle_pitch", -0.08 - 0.10 * np.clip(swing, 0.0, None))
    _set_if_exists("right_ankle_pitch", -0.08 - 0.10 * np.clip(swing_opposite, 0.0, None))
    _set_if_exists("left_hip_roll", -0.02 + 0.04 * np.sin(theta + np.pi / 2.0))
    _set_if_exists("right_hip_roll", 0.02 + 0.04 * np.sin(theta - np.pi / 2.0))
    _set_if_exists("waist_yaw", 0.03 * np.sin(theta))

    joint_pos = np.clip(joint_pos, joint_limits.lower, joint_limits.upper).astype(np.float32)
    joint_vel = _finite_difference(joint_pos, dt)

    left_foot_pos = np.stack(
        [
            x + 0.055 * np.sin(theta),
            np.full_like(x, 0.095),
            0.028 * np.clip(np.sin(theta), 0.0, None),
        ],
        axis=-1,
    ).astype(np.float32)
    right_foot_pos = np.stack(
        [
            x + 0.055 * np.sin(theta + np.pi),
            np.full_like(x, -0.095),
            0.028 * np.clip(np.sin(theta + np.pi), 0.0, None),
        ],
        axis=-1,
    ).astype(np.float32)
    left_contact = (left_foot_pos[:, 2] < 0.006).astype(np.float32)
    right_contact = (right_foot_pos[:, 2] < 0.006).astype(np.float32)

    metadata = {
        "name": name,
        "source_name": "bootstrap_nominal_walk",
        "source_format": "wildrobot_procedural_v0",
        "frame_count": int(frame_count),
        "dt": float(dt),
        "actuator_names": list(actuator_names),
        "loop_mode": "wrap",
        "phase_mode": "normalized_cycle",
        "retarget_version": "wildrobot_v0.16.0",
        "notes": "Deterministic bootstrap placeholder clip generated locally (no external mocap).",
        "nominal_forward_velocity_mps": float(nominal_forward_velocity_mps),
    }

    clip = ReferenceMotionClip(
        name=name,
        dt=float(dt),
        phase=phase.astype(np.float32),
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
        left_foot_contact=left_contact,
        right_foot_contact=right_contact,
        metadata=metadata,
    )
    return retarget_wildrobot_clip(clip, joint_limits=joint_limits)
