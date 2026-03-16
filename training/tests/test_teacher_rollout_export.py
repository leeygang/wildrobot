# training/tests/test_teacher_rollout_export.py
"""v0.16.1 teacher rollout dataset shard contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.imitation.collect_teacher_rollouts import shard_teacher_rollouts
from training.imitation.dataset import (
    REQUIRED_SHARD_KEYS,
    list_shards,
    load_metadata,
    load_rollout_shard,
)


def _make_rollout_arrays(n: int = 17, obs_dim: int = 12, action_dim: int = 5) -> dict[str, np.ndarray]:
    t = np.arange(n, dtype=np.float32)
    obs = np.stack([np.sin(0.03 * t + i) for i in range(obs_dim)], axis=-1).astype(np.float32)
    actions = np.stack([np.tanh(0.05 * t + i) for i in range(action_dim)], axis=-1).astype(np.float32)
    phase = np.stack([np.sin(2.0 * np.pi * t / n), np.cos(2.0 * np.pi * t / n)], axis=-1).astype(np.float32)
    return {
        "obs": obs,
        "actions": actions,
        "phase": phase,
        "velocity_cmd": np.full((n,), 0.1, dtype=np.float32),
        "forward_velocity": np.linspace(0.02, 0.14, n, dtype=np.float32),
        "done": (t % 11 == 0).astype(np.float32),
        "left_foot_contact": (t % 2 == 0).astype(np.float32),
        "right_foot_contact": (t % 2 == 1).astype(np.float32),
        "teacher_action_mean": actions.copy(),
        "teacher_action_std": np.full((n, action_dim), 0.15, dtype=np.float32),
        "teacher_value": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "root_pitch": np.zeros((n,), dtype=np.float32),
        "root_pitch_rate": np.zeros((n,), dtype=np.float32),
    }


@pytest.mark.unit
def test_rollout_shard_export_and_metadata(tmp_path: Path):
    arrays = _make_rollout_arrays()
    meta = shard_teacher_rollouts(
        arrays,
        output_dir=tmp_path / "dataset",
        shard_size=6,
        teacher_checkpoint="teacher_ckpt.pkl",
        teacher_config="training/configs/ppo_walking_teacher.yaml",
        observation_layout="wr_obs_v3",
    )
    assert meta.num_samples == arrays["obs"].shape[0]
    assert meta.num_shards == 3
    assert meta.action_dim == arrays["actions"].shape[1]
    assert meta.phase_dim == arrays["phase"].shape[1]
    assert meta.velocity_cmd_range[0] <= meta.velocity_cmd_range[1]

    shards = list_shards(tmp_path / "dataset")
    assert len(shards) == 3
    first = load_rollout_shard(shards[0])
    for key in REQUIRED_SHARD_KEYS:
        assert key in first
    assert first["obs"].shape[1] == arrays["obs"].shape[1]

    loaded_meta = load_metadata(tmp_path / "dataset" / "metadata.json")
    assert loaded_meta.teacher_checkpoint == "teacher_ckpt.pkl"
    assert loaded_meta.observation_layout == "wr_obs_v3"
    assert loaded_meta.num_shards == 3

