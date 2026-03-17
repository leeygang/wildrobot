# training/tests/test_teacher_rollout_export.py
"""v0.16.1 teacher rollout dataset shard contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from training.imitation.collect_teacher_rollouts import (
    collect_teacher_rollouts_from_checkpoint,
    shard_teacher_rollouts,
)
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


@pytest.mark.unit
def test_teacher_checkpoint_rollout_collection_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import training.imitation.collect_teacher_rollouts as mod

    action_dim = 2
    teacher_obs_dim = 25  # wr_obs_teacher with action_dim=2

    class FakeWrInfo(NamedTuple):
        velocity_cmd: jnp.ndarray

    class FakeState(NamedTuple):
        obs: jnp.ndarray
        done: jnp.ndarray
        metrics: dict
        info: dict

    class FakeEnv:
        def __init__(self, config):
            self._config = config

        def reset(self, rng):
            obs = jnp.zeros((teacher_obs_dim,), dtype=jnp.float32)
            # teacher_phase slice at [17:19] => [sin, cos] = [0, 1]
            obs = obs.at[18].set(1.0)
            return FakeState(
                obs=obs,
                done=jnp.asarray(0.0, dtype=jnp.float32),
                metrics={
                    "debug/left_force": jnp.asarray(6.0, dtype=jnp.float32),
                    "debug/right_force": jnp.asarray(1.0, dtype=jnp.float32),
                    "debug/forward_vel": jnp.asarray(0.08, dtype=jnp.float32),
                    "debug/pitch": jnp.asarray(0.01, dtype=jnp.float32),
                    "debug/pitch_rate": jnp.asarray(0.02, dtype=jnp.float32),
                },
                info={"wr": FakeWrInfo(velocity_cmd=jnp.asarray(0.1, dtype=jnp.float32))},
            )

        def step(self, state, action):
            return state

    def _fake_build_policy_spec(*, layout_id: str, **kwargs):
        if layout_id == "wr_obs_teacher":
            layout = [
                SimpleNamespace(name="gravity_local", size=3),
                SimpleNamespace(name="angvel_heading_local", size=3),
                SimpleNamespace(name="joint_pos_normalized", size=action_dim),
                SimpleNamespace(name="joint_vel_normalized", size=action_dim),
                SimpleNamespace(name="foot_switches", size=4),
                SimpleNamespace(name="prev_action", size=action_dim),
                SimpleNamespace(name="velocity_cmd", size=1),
                SimpleNamespace(name="teacher_phase", size=2),
                SimpleNamespace(name="teacher_joint_pos_target", size=action_dim),
                SimpleNamespace(name="teacher_root_lin_vel_target", size=2),
                SimpleNamespace(name="teacher_root_height_target", size=1),
                SimpleNamespace(name="padding", size=1),
            ]
            obs_dim = teacher_obs_dim
        elif layout_id == "wr_obs_v3":
            layout = [
                SimpleNamespace(name="gravity_local", size=3),
                SimpleNamespace(name="angvel_heading_local", size=3),
                SimpleNamespace(name="joint_pos_normalized", size=action_dim),
                SimpleNamespace(name="joint_vel_normalized", size=action_dim),
                SimpleNamespace(name="foot_switches", size=4),
                SimpleNamespace(name="prev_action", size=action_dim),
                SimpleNamespace(name="velocity_cmd", size=1),
                SimpleNamespace(name="gait_clock", size=4),
                SimpleNamespace(name="padding", size=1),
            ]
            obs_dim = 22
        else:
            raise ValueError(layout_id)
        return SimpleNamespace(
            model=SimpleNamespace(obs_dim=obs_dim, action_dim=action_dim),
            observation=SimpleNamespace(layout=layout),
        )

    fake_cfg = SimpleNamespace(
        env=SimpleNamespace(
            teacher_enabled=True,
            actor_obs_layout_id="wr_obs_teacher",
            action_filter_alpha=0.3,
            robot_config_path="assets/v2/mujoco_robot_config.json",
            contact_threshold_force=5.0,
        ),
        networks=SimpleNamespace(
            actor=SimpleNamespace(hidden_sizes=(8, 8)),
            critic=SimpleNamespace(hidden_sizes=(8, 8)),
        ),
        freeze=lambda: None,
    )

    monkeypatch.setattr(mod, "load_training_config", lambda _: fake_cfg)
    monkeypatch.setattr(mod, "load_robot_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "get_robot_config",
        lambda: SimpleNamespace(robot_name="wildrobot", actuated_joints=[]),
    )
    monkeypatch.setattr(mod, "build_policy_spec", _fake_build_policy_spec)
    monkeypatch.setattr(mod, "policy_spec_hash", lambda _spec: "hash")
    monkeypatch.setattr(
        mod,
        "load_checkpoint",
        lambda _path: {"policy_params": {"p": 1}, "processor_params": (), "config": {"policy_spec_hash": "hash"}},
    )
    monkeypatch.setattr(mod, "WildRobotEnv", FakeEnv)
    monkeypatch.setattr(mod, "create_networks", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        mod,
        "sample_actions",
        lambda **kwargs: (
            jnp.zeros((kwargs["obs"].shape[0], action_dim), dtype=jnp.float32),
            jnp.zeros((kwargs["obs"].shape[0], action_dim), dtype=jnp.float32),
            jnp.zeros((kwargs["obs"].shape[0],), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        mod,
        "compute_values",
        lambda **kwargs: jnp.zeros((kwargs["obs"].shape[0],), dtype=jnp.float32),
    )

    out_dir = tmp_path / "teacher_rollouts"
    metadata = collect_teacher_rollouts_from_checkpoint(
        teacher_checkpoint_path=tmp_path / "teacher.pkl",
        teacher_config_path=tmp_path / "teacher.yaml",
        output_dir=out_dir,
        shard_size=8,
        observation_layout="wr_obs_v3",
        num_envs=2,
        num_steps=4,
        seed=42,
        deterministic=True,
    )
    assert metadata.num_samples == 8
    assert metadata.observation_layout == "wr_obs_v3"
    shards = list_shards(out_dir)
    assert len(shards) == 1
    shard = load_rollout_shard(shards[0])
    assert shard["obs"].shape == (8, 22)
    assert shard["actions"].shape == (8, action_dim)
    # wr_obs_v3 gait_clock must be [sin(phase), cos(phase), sin(phase+pi), cos(phase+pi)].
    gait_clock = shard["obs"][0, 17:21]
    assert np.allclose(gait_clock, np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float32), atol=1e-6)
