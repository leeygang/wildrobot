"""Regression coverage for the 17d6 actor-symmetry fine-tune."""

from __future__ import annotations

import copy
import pickle
import types

import numpy as np
import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import PPOConfig
from training.core.checkpoint import save_checkpoint_from_cpu


_SOURCE_CONFIG = "training/configs/ppo_walking_v0210_17d4_startup_mix.yaml"
_FINETUNE_CONFIG = "training/configs/ppo_walking_v0210_17d6_symmetry_finetune.yaml"


def test_mirror_loss_is_disabled_by_default() -> None:
    assert PPOConfig().mirror_loss_coef == pytest.approx(0.0)


def test_17d6_enables_only_the_actor_symmetry_optimization_change() -> None:
    source = load_training_config(_SOURCE_CONFIG)
    finetune = load_training_config(_FINETUNE_CONFIG)

    assert source.ppo.mirror_loss_coef == pytest.approx(0.0)
    assert finetune.ppo.mirror_loss_coef == pytest.approx(0.1)
    assert finetune.ppo.iterations == 60
    assert finetune.ppo.log_interval == 1
    assert finetune.ppo.eval.post_training_top_k == 6
    assert finetune.reward_weights.saturation == pytest.approx(0.0)
    assert finetune.env.loc_ref_rsi_probability == pytest.approx(0.75)

    source_raw = copy.deepcopy(source.raw_config)
    finetune_raw = copy.deepcopy(finetune.raw_config)
    for raw in (source_raw, finetune_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["ppo"].pop("iterations")
        raw["ppo"].pop("log_interval")
        raw["ppo"].pop("mirror_loss_coef", None)
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["checkpoints"].pop("dir")
        raw["wandb"].pop("tags")
    assert finetune_raw == source_raw


def test_checkpoint_persists_mirror_config_and_metrics(tmp_path) -> None:
    cfg = load_training_config(_FINETUNE_CONFIG)
    state = types.SimpleNamespace(
        policy_params={},
        value_params={},
        processor_params={},
        policy_opt_state={},
        value_opt_state={},
        rng=np.array([0, 1], dtype=np.uint32),
    )
    metrics = types.SimpleNamespace(
        episode_reward=1.0,
        task_reward_mean=0.1,
        episode_length=100.0,
        policy_loss=0.01,
        value_loss=0.02,
        env_metrics={
            "forward_velocity": 0.1,
            "height": 0.46,
            "ppo/mirror_loss": 0.09,
            "ppo/mirror_action_rmse": 0.3,
            "ppo/mirror_loss_weighted": 0.009,
        },
    )

    checkpoint_path = save_checkpoint_from_cpu(
        state_cpu=state,
        config=cfg,
        iteration=1,
        total_steps=20,
        checkpoint_dir=str(tmp_path),
        metrics=metrics,
    )
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)

    assert checkpoint["config"]["mirror_loss_coef"] == pytest.approx(0.1)
    assert checkpoint["metrics"]["ppo/mirror_loss"] == pytest.approx(0.09)
    assert checkpoint["metrics"]["ppo/mirror_action_rmse"] == pytest.approx(0.3)
    assert checkpoint["metrics"]["ppo/mirror_loss_weighted"] == pytest.approx(
        0.009
    )
