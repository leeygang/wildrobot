#!/usr/bin/env python3
"""Shard teacher rollouts into the v0.16.1 student dataset contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.imitation.dataset import (
    RolloutDatasetMetadata,
    RECOMMENDED_SHARD_KEYS,
    REQUIRED_SHARD_KEYS,
    validate_rollout_arrays,
    write_metadata,
    write_rollout_shard,
)


def shard_teacher_rollouts(
    arrays: Mapping[str, np.ndarray],
    *,
    output_dir: str | Path,
    shard_size: int,
    teacher_checkpoint: str,
    teacher_config: str,
    observation_layout: str,
    metadata_extras: dict | None = None,
) -> RolloutDatasetMetadata:
    """Write sharded rollout dataset and return metadata."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}")

    validated = validate_rollout_arrays(arrays)
    n = int(validated["obs"].shape[0])
    action_dim = int(validated["actions"].shape[1])
    phase_dim = int(validated["phase"].shape[1] if validated["phase"].ndim == 2 else 1)
    velocity_cmd = validated["velocity_cmd"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_count = 0
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        shard = {k: v[start:end] for k, v in validated.items()}
        shard_path = out_dir / f"shard_{shard_count:04d}.npz"
        write_rollout_shard(shard_path, shard)
        shard_count += 1

    extras = dict(metadata_extras or {})
    if "recommended_fields_present" not in extras:
        extras["recommended_fields_present"] = sorted(
            [k for k in RECOMMENDED_SHARD_KEYS if k in validated]
        )

    metadata = RolloutDatasetMetadata(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        observation_layout=observation_layout,
        action_dim=action_dim,
        num_shards=shard_count,
        num_samples=n,
        phase_dim=phase_dim,
        velocity_cmd_range=(float(np.min(velocity_cmd)), float(np.max(velocity_cmd))),
        extras=extras,
    )
    write_metadata(out_dir / "metadata.json", metadata)
    return metadata


def _load_source_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as src:
        arrays = {k: src[k] for k in src.files}
    missing = [k for k in REQUIRED_SHARD_KEYS if k not in arrays]
    if missing:
        raise ValueError(
            f"Source npz missing required rollout arrays: {missing}. "
            f"Expected keys include {REQUIRED_SHARD_KEYS}."
        )
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--teacher-config", type=str, required=True)
    parser.add_argument("--observation-layout", type=str, default="wr_obs_v3")
    parser.add_argument("--shard-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = _load_source_npz(args.source_npz)
    metadata = shard_teacher_rollouts(
        arrays,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_config=args.teacher_config,
        observation_layout=args.observation_layout,
    )
    print("Teacher rollout shards written:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  num_samples: {metadata.num_samples}")
    print(f"  num_shards: {metadata.num_shards}")
    print(f"  action_dim: {metadata.action_dim}")
    print(f"  phase_dim: {metadata.phase_dim}")


if __name__ == "__main__":
    main()
