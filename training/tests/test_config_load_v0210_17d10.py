from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_17d10_roll_ik_contract.yaml")


def test_17d10_is_longer_correction_only_finetune() -> None:
    cfg = load_training_config(CONFIG)

    assert cfg.version == "0.21.0-17d10"
    assert cfg.ppo.iterations == 100
    assert cfg.checkpoints.interval == 10
    assert cfg.ppo.eval.post_training_top_k == 10
    assert cfg.reward_weights.saturation == pytest.approx(0.0)


def test_17d10_keeps_proven_action_base_and_reset_mix() -> None:
    cfg = load_training_config(CONFIG)

    assert cfg.env.loc_ref_residual_base == "home"
    assert cfg.env.loc_ref_walking_joint_offsets_rad == {}
    assert cfg.env.loc_ref_rsi_enabled is True
    assert cfg.env.loc_ref_rsi_probability == pytest.approx(0.75)
    assert cfg.env.close_feet_threshold == pytest.approx(0.146)
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.13, 0.0, 0.0))
