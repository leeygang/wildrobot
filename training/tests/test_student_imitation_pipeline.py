# training/tests/test_student_imitation_pipeline.py
"""v0.16.1 student BC warm-start and PPO handoff tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from policy_contract.spec import policy_spec_hash
from policy_contract.spec_builder import build_policy_spec
from training.algos.ppo.ppo_core import create_networks, init_network_params
from training.configs.training_config import (
    get_robot_config,
    load_robot_config,
    load_training_config,
)
from training.core.training_loop import _apply_pretrained_checkpoint
from training.imitation.collect_teacher_rollouts import shard_teacher_rollouts
from training.imitation.pretrain_student import pretrain_student_behavior_cloning


def _make_dataset_for_spec(obs_dim: int, action_dim: int, n: int = 192) -> dict[str, np.ndarray]:
    t = np.arange(n, dtype=np.float32)
    obs = np.stack([np.sin(0.015 * t + i * 0.01) for i in range(obs_dim)], axis=-1).astype(np.float32)
    actions = np.tanh(obs[:, :action_dim]).astype(np.float32)
    return {
        "obs": obs,
        "actions": actions,
        "phase": np.stack([np.sin(2.0 * np.pi * t / n), np.cos(2.0 * np.pi * t / n)], axis=-1).astype(np.float32),
        "velocity_cmd": np.full((n,), 0.1, dtype=np.float32),
        "forward_velocity": np.full((n,), 0.08, dtype=np.float32),
        "done": (t % 47 == 0).astype(np.float32),
        "left_foot_contact": (t % 2 == 0).astype(np.float32),
        "right_foot_contact": (t % 2 == 1).astype(np.float32),
        "teacher_action_mean": actions.copy(),
        "teacher_action_std": np.full((n, action_dim), 0.12, dtype=np.float32),
        "teacher_value": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "root_pitch": np.zeros((n,), dtype=np.float32),
        "root_pitch_rate": np.zeros((n,), dtype=np.float32),
    }


@pytest.mark.unit
def test_student_bc_pretrain_and_warm_start(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[2]
    student_cfg_path = project_root / "training" / "configs" / "ppo_walking_student.yaml"
    cfg = load_training_config(student_cfg_path)
    load_robot_config(cfg.env.robot_config_path)
    robot_cfg = get_robot_config()
    spec = build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
        action_filter_alpha=float(cfg.env.action_filter_alpha),
        layout_id=str(cfg.env.actor_obs_layout_id),
    )

    arrays = _make_dataset_for_spec(spec.model.obs_dim, spec.model.action_dim, n=160)
    dataset_dir = tmp_path / "dataset"
    shard_teacher_rollouts(
        arrays,
        output_dir=dataset_dir,
        shard_size=64,
        teacher_checkpoint="teacher_ckpt.pkl",
        teacher_config="training/configs/ppo_walking_teacher.yaml",
        observation_layout="wr_obs_v3",
    )

    pretrained_path = tmp_path / "student_pretrained.pkl"
    checkpoint = pretrain_student_behavior_cloning(
        training_config_path=student_cfg_path,
        dataset_dir=dataset_dir,
        output_checkpoint_path=pretrained_path,
        steps=8,
        batch_size=48,
        learning_rate=5e-4,
        seed=7,
    )
    assert pretrained_path.exists()
    assert checkpoint["checkpoint_type"] == "student_pretrain_v0.16.1"
    assert np.isfinite(float(checkpoint["metrics"]["imitation_loss"]))

    with pretrained_path.open("rb") as f:
        loaded = pickle.load(f)
    assert loaded["policy_spec_hash"] == policy_spec_hash(spec)

    ppo_network = create_networks(
        obs_dim=spec.model.obs_dim,
        action_dim=spec.model.action_dim,
        policy_hidden_dims=cfg.networks.actor.hidden_sizes,
        value_hidden_dims=cfg.networks.critic.hidden_sizes,
    )
    proc_init, policy_init, value_init = init_network_params(
        ppo_network,
        spec.model.obs_dim,
        spec.model.action_dim,
        seed=123,
    )
    proc_loaded, policy_loaded, value_loaded = _apply_pretrained_checkpoint(
        loaded,
        current_policy_spec_hash=policy_spec_hash(spec),
        processor_params=proc_init,
        policy_params=policy_init,
        value_params=value_init,
    )

    init_norm = sum(
        float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(policy_init)
    )
    loaded_norm = sum(
        float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(policy_loaded)
    )
    assert abs(loaded_norm - init_norm) > 1e-4
    assert jax.tree_util.tree_structure(value_loaded) == jax.tree_util.tree_structure(value_init)
    assert proc_loaded == proc_init
