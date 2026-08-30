"""Pin the short native-17D bidirectional-lateral diagnostic."""

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_CFG = Path("training/configs/ppo_walking_v0210_17d3_lateral_weight.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(str(_CFG))


def test_17d3_is_a_short_checkpoint_120_diagnostic(cfg) -> None:
    assert cfg.version == "0.21.0-17d3"
    assert cfg.ppo.iterations == 100
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-5)
    assert cfg.checkpoints.interval == 10


def test_17d3_changes_only_lateral_weight_from_17d2(cfg) -> None:
    source = load_training_config(
        "training/configs/ppo_walking_v0210_17d2_deployment_band.yaml"
    )
    assert cfg.env.min_velocity == source.env.min_velocity
    assert cfg.env.max_velocity == source.env.max_velocity
    assert cfg.env.min_velocity_y == source.env.min_velocity_y
    assert cfg.env.max_velocity_y == source.env.max_velocity_y
    assert cfg.env.cmd_sampler_walk_vy_exact_signed_bins is True
    assert cfg.env.eval_velocity_cmd_probes == source.env.eval_velocity_cmd_probes
    assert cfg.reward_weights.cmd_forward_velocity_alpha_y == pytest.approx(120.0)
    assert cfg.reward_weights.cmd_lateral_velocity_track == pytest.approx(2.0)
    assert source.reward_weights.cmd_lateral_velocity_track == pytest.approx(1.0)


def test_17d3_enables_strict_walking_safety(cfg) -> None:
    assert cfg.ppo.eval.post_training_top_k == 10
    assert cfg.ppo.eval.post_training_num_envs == 64
    assert cfg.ppo.eval.post_training_num_steps == 1000
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True
    assert cfg.ppo.eval.post_training_strict_walking_safety is True
    assert cfg.reward_weights.torque == pytest.approx(-0.001)


def test_17d3_quick_verify_uses_a_small_post_training_screen() -> None:
    cfg = load_training_config(str(_CFG))
    cfg.apply_overrides(cfg.raw_config["quick_verify"])
    assert cfg.ppo.iterations == 3
    assert cfg.ppo.eval.num_envs == 4
    assert cfg.ppo.eval.num_steps == 16
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_top_k == 1
    assert cfg.ppo.eval.post_training_num_envs == 4
    assert cfg.ppo.eval.post_training_num_steps == 16
