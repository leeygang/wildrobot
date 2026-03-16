"""Teacher-rollout dataset contract utilities for v0.16.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


REQUIRED_SHARD_KEYS: tuple[str, ...] = (
    "obs",
    "actions",
    "phase",
    "velocity_cmd",
    "forward_velocity",
    "done",
    "left_foot_contact",
    "right_foot_contact",
)

RECOMMENDED_SHARD_KEYS: tuple[str, ...] = (
    "teacher_action_mean",
    "teacher_action_std",
    "teacher_value",
    "root_pitch",
    "root_pitch_rate",
)

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "teacher_checkpoint",
    "teacher_config",
    "observation_layout",
    "action_dim",
    "num_shards",
    "num_samples",
    "phase_dim",
    "velocity_cmd_range",
)


@dataclass(frozen=True)
class RolloutDatasetMetadata:
    """Metadata contract for teacher rollout shards."""

    teacher_checkpoint: str
    teacher_config: str
    observation_layout: str
    action_dim: int
    num_shards: int
    num_samples: int
    phase_dim: int
    velocity_cmd_range: tuple[float, float]
    extras: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_config": self.teacher_config,
            "observation_layout": self.observation_layout,
            "action_dim": int(self.action_dim),
            "num_shards": int(self.num_shards),
            "num_samples": int(self.num_samples),
            "phase_dim": int(self.phase_dim),
            "velocity_cmd_range": [
                float(self.velocity_cmd_range[0]),
                float(self.velocity_cmd_range[1]),
            ],
        }
        payload.update(self.extras)
        return payload


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"Missing required {context} keys: {missing}")


def validate_rollout_arrays(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Validate rollout arrays against the v0.16.1 shard contract."""
    _require_keys(arrays, REQUIRED_SHARD_KEYS, "rollout shard")
    normalized: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if key in {"obs", "actions"}:
            if arr.ndim != 2:
                raise ValueError(f"{key} must be rank-2, got shape {arr.shape}")
            normalized[key] = arr.astype(np.float32)
            continue
        if key == "phase":
            if arr.ndim not in (1, 2):
                raise ValueError(f"phase must be rank-1 or rank-2, got shape {arr.shape}")
            normalized[key] = arr.astype(np.float32)
            continue
        if key in {"teacher_action_mean", "teacher_action_std"}:
            if arr.ndim != 2:
                raise ValueError(f"{key} must be rank-2, got shape {arr.shape}")
            normalized[key] = arr.astype(np.float32)
            continue
        if arr.ndim != 1:
            raise ValueError(f"{key} must be rank-1, got shape {arr.shape}")
        normalized[key] = arr.astype(np.float32)

    n = int(normalized["obs"].shape[0])
    action_dim = int(normalized["actions"].shape[1])
    if normalized["actions"].shape[0] != n:
        raise ValueError(
            f"actions length mismatch: obs={n}, actions={normalized['actions'].shape[0]}"
        )
    for key in REQUIRED_SHARD_KEYS:
        if normalized[key].shape[0] != n:
            raise ValueError(
                f"{key} length mismatch: expected {n}, got {normalized[key].shape[0]}"
            )
    for key in RECOMMENDED_SHARD_KEYS:
        if key in normalized and normalized[key].shape[0] != n:
            raise ValueError(
                f"{key} length mismatch: expected {n}, got {normalized[key].shape[0]}"
            )
    for key in ("teacher_action_mean", "teacher_action_std"):
        if key in normalized and normalized[key].shape[1] != action_dim:
            raise ValueError(
                f"{key} action dim mismatch: expected {action_dim}, got {normalized[key].shape[1]}"
            )
    return normalized


def write_rollout_shard(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one rollout shard `.npz` after contract validation."""
    normalized = validate_rollout_arrays(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **normalized)


def load_rollout_shard(path: str | Path) -> Dict[str, np.ndarray]:
    """Load and validate one rollout shard `.npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return validate_rollout_arrays(arrays)


def write_metadata(path: str | Path, metadata: RolloutDatasetMetadata) -> None:
    """Write dataset metadata JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_json_dict(), indent=2))


def load_metadata(path: str | Path) -> RolloutDatasetMetadata:
    """Load dataset metadata JSON and validate required keys."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata JSON must be an object")
    _require_keys(payload, REQUIRED_METADATA_KEYS, "dataset metadata")
    velocity_range_raw = payload["velocity_cmd_range"]
    if not isinstance(velocity_range_raw, (list, tuple)) or len(velocity_range_raw) != 2:
        raise ValueError("velocity_cmd_range must be [min, max]")
    extras = {
        k: v for k, v in payload.items() if k not in set(REQUIRED_METADATA_KEYS)
    }
    return RolloutDatasetMetadata(
        teacher_checkpoint=str(payload["teacher_checkpoint"]),
        teacher_config=str(payload["teacher_config"]),
        observation_layout=str(payload["observation_layout"]),
        action_dim=int(payload["action_dim"]),
        num_shards=int(payload["num_shards"]),
        num_samples=int(payload["num_samples"]),
        phase_dim=int(payload["phase_dim"]),
        velocity_cmd_range=(float(velocity_range_raw[0]), float(velocity_range_raw[1])),
        extras=extras,
    )


def list_shards(dataset_dir: str | Path) -> list[Path]:
    """List shard files sorted by shard index."""
    dataset_dir = Path(dataset_dir)
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    return shards
