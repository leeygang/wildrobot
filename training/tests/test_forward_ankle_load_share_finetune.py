"""Regression coverage for the 17d7 forward-only ankle-load-sharing run."""

from __future__ import annotations

import copy

import pytest

from training.configs.training_config import load_training_config


_SOURCE_CONFIG = "training/configs/ppo_walking_v0210_17d4_startup_mix.yaml"
_FINETUNE_CONFIG = (
    "training/configs/ppo_walking_v0210_17d7_forward_ankle_load_share.yaml"
)


def test_17d7_is_forward_only_and_keeps_strict_safety() -> None:
    cfg = load_training_config(_FINETUNE_CONFIG)

    assert cfg.env.cmd_sampler_3d_branched is False
    assert cfg.env.min_velocity_y == pytest.approx(0.0)
    assert cfg.env.max_velocity_y == pytest.approx(0.0)
    assert cfg.env.max_yaw_rate == pytest.approx(0.0)
    assert list(cfg.env.eval_velocity_cmd_probes) == []
    assert cfg.ppo.eval.post_training_strict_lateral_drift is False
    assert cfg.ppo.eval.post_training_strict_walking_safety is True


def test_17d7_changes_only_scope_schedule_and_ankle_roll_pose_cost() -> None:
    source = load_training_config(_SOURCE_CONFIG)
    finetune = load_training_config(_FINETUNE_CONFIG)

    source_pose = source.env.penalty_pose_weights_per_joint
    finetune_pose = finetune.env.penalty_pose_weights_per_joint
    assert source_pose["left_ankle_roll"] == pytest.approx(5.0)
    assert source_pose["right_ankle_roll"] == pytest.approx(5.0)
    assert finetune_pose["left_ankle_roll"] == pytest.approx(1.0)
    assert finetune_pose["right_ankle_roll"] == pytest.approx(1.0)
    assert finetune_pose["left_hip_roll"] == pytest.approx(1.0)
    assert finetune_pose["right_hip_roll"] == pytest.approx(1.0)

    assert finetune.ppo.mirror_loss_coef == pytest.approx(0.0)
    assert finetune.reward_weights.saturation == pytest.approx(0.0)
    assert finetune.reward_weights.torque == source.reward_weights.torque
    assert finetune.reward_weights.cmd_lateral_velocity_track == pytest.approx(1.0)
    assert finetune.env.loc_ref_rsi_probability == pytest.approx(0.75)
    assert finetune.ppo.iterations == 40
    assert finetune.checkpoints.interval == 5
    assert finetune.ppo.eval.post_training_top_k == 8

    source_raw = copy.deepcopy(source.raw_config)
    finetune_raw = copy.deepcopy(finetune.raw_config)
    for raw in (source_raw, finetune_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"]["penalty_pose_weights_per_joint"].pop("left_ankle_roll")
        raw["env"]["penalty_pose_weights_per_joint"].pop("right_ankle_roll")
        raw["env"].pop("min_velocity_y")
        raw["env"].pop("max_velocity_y")
        raw["env"].pop("eval_velocity_cmd_probes")
        raw["env"].pop("cmd_sampler_3d_branched")
        raw["env"].pop("cmd_sampler_walk_vy_exact_signed_bins")
        raw["ppo"].pop("iterations")
        raw["ppo"].pop("log_interval")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["ppo"]["eval"].pop("post_training_strict_lateral_drift")
        raw["checkpoints"].pop("dir")
        raw["checkpoints"].pop("interval")
        raw["wandb"].pop("tags")
    assert finetune_raw == source_raw
