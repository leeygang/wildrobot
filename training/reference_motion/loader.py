"""Typed reference-motion clip loader for teacher bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


REQUIRED_NPZ_KEYS = (
    "dt",
    "phase",
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "left_foot_pos",
    "right_foot_pos",
    "left_foot_contact",
    "right_foot_contact",
)

REQUIRED_METADATA_KEYS = (
    "name",
    "source_name",
    "source_format",
    "frame_count",
    "dt",
    "actuator_names",
    "loop_mode",
    "phase_mode",
)


@dataclass(frozen=True)
class ReferenceMotionClip:
    """In-memory typed reference-motion clip."""

    name: str
    dt: float
    phase: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    left_foot_pos: np.ndarray
    right_foot_pos: np.ndarray
    left_foot_contact: np.ndarray
    right_foot_contact: np.ndarray
    metadata: Dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.phase.shape[0])

    @property
    def actuator_dim(self) -> int:
        return int(self.joint_pos.shape[1])


def _as_float_scalar(value: np.ndarray | float, key: str) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {arr.shape}")
    return float(arr.reshape(()))


def _require_keys(container: Dict[str, Any], required: Sequence[str], context: str) -> None:
    missing = [k for k in required if k not in container]
    if missing:
        raise ValueError(f"Missing required {context} keys: {missing}")


def _validate_shapes(arrays: Dict[str, np.ndarray], actuator_count: int) -> None:
    t = arrays["phase"].shape[0]
    expected = {
        "phase": (t,),
        "root_pos": (t, 3),
        "root_quat": (t, 4),
        "root_lin_vel": (t, 3),
        "root_ang_vel": (t, 3),
        "joint_pos": (t, actuator_count),
        "joint_vel": (t, actuator_count),
        "left_foot_pos": (t, 3),
        "right_foot_pos": (t, 3),
        "left_foot_contact": (t,),
        "right_foot_contact": (t,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} shape mismatch: expected {shape}, got {arrays[key].shape}")


def load_reference_motion(
    npz_path: str | Path,
    metadata_path: str | Path,
    *,
    actuator_names: Sequence[str] | None = None,
) -> ReferenceMotionClip:
    """Load and validate a reference-motion clip from `.npz` + `.json`."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Reference npz not found: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {metadata_path}")

    with np.load(npz_path, allow_pickle=False) as raw:
        npz_data = {k: raw[k] for k in raw.files}
    _require_keys(npz_data, REQUIRED_NPZ_KEYS, ".npz")

    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON must be an object")
    _require_keys(metadata, REQUIRED_METADATA_KEYS, "metadata")

    file_actuator_names = list(metadata["actuator_names"])
    if actuator_names is not None and list(actuator_names) != file_actuator_names:
        raise ValueError(
            "Actuator name mismatch between runtime robot and reference metadata: "
            f"runtime={list(actuator_names)}, metadata={file_actuator_names}"
        )

    dt = _as_float_scalar(npz_data["dt"], "dt")
    phase = np.asarray(npz_data["phase"], dtype=np.float32).reshape(-1)
    joint_pos = np.asarray(npz_data["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(npz_data["joint_vel"], dtype=np.float32)
    actuator_count = int(joint_pos.shape[1])

    arrays = {
        "phase": phase,
        "root_pos": np.asarray(npz_data["root_pos"], dtype=np.float32),
        "root_quat": np.asarray(npz_data["root_quat"], dtype=np.float32),
        "root_lin_vel": np.asarray(npz_data["root_lin_vel"], dtype=np.float32),
        "root_ang_vel": np.asarray(npz_data["root_ang_vel"], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "left_foot_pos": np.asarray(npz_data["left_foot_pos"], dtype=np.float32),
        "right_foot_pos": np.asarray(npz_data["right_foot_pos"], dtype=np.float32),
        "left_foot_contact": np.asarray(npz_data["left_foot_contact"], dtype=np.float32).reshape(-1),
        "right_foot_contact": np.asarray(npz_data["right_foot_contact"], dtype=np.float32).reshape(-1),
    }
    _validate_shapes(arrays, actuator_count=actuator_count)

    frame_count = int(phase.shape[0])
    if int(metadata["frame_count"]) != frame_count:
        raise ValueError(
            f"metadata frame_count mismatch: expected {frame_count}, got {metadata['frame_count']}"
        )
    if abs(float(metadata["dt"]) - dt) > 1e-8:
        raise ValueError(f"metadata dt mismatch: npz={dt}, metadata={metadata['dt']}")
    if len(file_actuator_names) != actuator_count:
        raise ValueError(
            f"actuator count mismatch: metadata={len(file_actuator_names)}, joint_pos={actuator_count}"
        )
    if arrays["root_quat"].shape[1] != 4:
        raise ValueError("root_quat must have shape (T, 4)")

    if np.any(phase < -1e-6) or np.any(phase >= 1.0 + 1e-6):
        raise ValueError("phase must be in [0, 1)")

    return ReferenceMotionClip(
        name=str(metadata["name"]),
        dt=dt,
        phase=arrays["phase"],
        root_pos=arrays["root_pos"],
        root_quat=arrays["root_quat"],
        root_lin_vel=arrays["root_lin_vel"],
        root_ang_vel=arrays["root_ang_vel"],
        joint_pos=arrays["joint_pos"],
        joint_vel=arrays["joint_vel"],
        left_foot_pos=arrays["left_foot_pos"],
        right_foot_pos=arrays["right_foot_pos"],
        left_foot_contact=arrays["left_foot_contact"],
        right_foot_contact=arrays["right_foot_contact"],
        metadata=metadata,
    )


def save_reference_motion_clip(
    clip: ReferenceMotionClip,
    npz_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Save a typed clip as `.npz` + `.json`."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        dt=np.asarray(clip.dt, dtype=np.float32),
        phase=np.asarray(clip.phase, dtype=np.float32),
        root_pos=np.asarray(clip.root_pos, dtype=np.float32),
        root_quat=np.asarray(clip.root_quat, dtype=np.float32),
        root_lin_vel=np.asarray(clip.root_lin_vel, dtype=np.float32),
        root_ang_vel=np.asarray(clip.root_ang_vel, dtype=np.float32),
        joint_pos=np.asarray(clip.joint_pos, dtype=np.float32),
        joint_vel=np.asarray(clip.joint_vel, dtype=np.float32),
        left_foot_pos=np.asarray(clip.left_foot_pos, dtype=np.float32),
        right_foot_pos=np.asarray(clip.right_foot_pos, dtype=np.float32),
        left_foot_contact=np.asarray(clip.left_foot_contact, dtype=np.float32),
        right_foot_contact=np.asarray(clip.right_foot_contact, dtype=np.float32),
    )
    metadata_path.write_text(json.dumps(clip.metadata, indent=2))
