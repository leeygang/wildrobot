"""Pin the v0.21.0-17d4 mixed home/RSI startup fine-tune."""

from pathlib import Path

import jax
import numpy as np
import pytest

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv


_CFG = Path("training/configs/ppo_walking_v0210_17d4_startup_mix.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(str(_CFG))


@pytest.fixture(scope="module")
def env(cfg):
    return WildRobotEnv(config=cfg)


def test_17d4_mixes_home_and_rsi_starts(cfg) -> None:
    assert cfg.version == "0.21.0-17d4"
    assert cfg.env.loc_ref_rsi_enabled is True
    assert cfg.env.loc_ref_rsi_probability == pytest.approx(0.75)
    assert cfg.env.loc_ref_reset_base == "home"
    assert cfg.env.loc_ref_residual_base == "home"


def test_17d4_reset_samples_both_static_and_moving_starts(env) -> None:
    root_vx = []
    for seed in range(32):
        state = env.reset(jax.random.PRNGKey(seed))
        root_vx.append(float(np.asarray(state.data.qvel)[0]))
    assert any(abs(vx) < 1e-6 for vx in root_vx)
    assert any(vx > 0.05 for vx in root_vx)


def test_17d4_keeps_parent_reward_and_short_finetune_contract(cfg) -> None:
    parent = load_training_config(
        "training/configs/ppo_walking_v0210_17d2_deployment_band.yaml"
    )
    assert cfg.reward_weights.cmd_lateral_velocity_track == pytest.approx(1.0)
    assert (
        cfg.reward_weights.cmd_lateral_velocity_track
        == parent.reward_weights.cmd_lateral_velocity_track
    )
    assert cfg.reward_weights.torque == parent.reward_weights.torque
    assert cfg.ppo.iterations == 100
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-5)
    assert cfg.checkpoints.interval == 10


def test_17d4_uses_fall_aware_post_training_gate(cfg) -> None:
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_top_k == 10
    assert cfg.ppo.eval.post_training_num_envs == 64
    assert cfg.ppo.eval.post_training_num_steps == 1000
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True
    assert cfg.ppo.eval.post_training_strict_walking_safety is True


def test_rsi_probability_rejects_out_of_range_value(tmp_path) -> None:
    invalid = tmp_path / "invalid_rsi_probability.yaml"
    invalid.write_text(
        _CFG.read_text(encoding="utf-8").replace(
            "loc_ref_rsi_probability: 0.75",
            "loc_ref_rsi_probability: 1.01",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loc_ref_rsi_probability"):
        load_training_config(str(invalid))


def test_17d4_quick_verify_uses_small_post_training_screen() -> None:
    cfg = load_training_config(str(_CFG))
    cfg.apply_overrides(cfg.raw_config["quick_verify"])
    assert cfg.ppo.iterations == 3
    assert cfg.ppo.eval.post_training_top_k == 1
    assert cfg.ppo.eval.post_training_num_envs == 4
    assert cfg.ppo.eval.post_training_num_steps == 16
