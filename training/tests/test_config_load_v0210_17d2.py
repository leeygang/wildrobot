"""Pin the native-17D deployment-band fine-tune contract."""

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config


_CFG = Path("training/configs/ppo_walking_v0210_17d2_deployment_band.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(str(_CFG))


def test_17d2_targets_the_deployment_command_band(cfg) -> None:
    assert cfg.version == "0.21.0-17d2"
    assert cfg.env.min_velocity == pytest.approx(0.065)
    assert cfg.env.max_velocity == pytest.approx(0.13)
    assert cfg.env.loc_ref_offline_command_vx == pytest.approx(0.13)
    assert cfg.env.loc_ref_command_grid_interval == pytest.approx(0.065)


def test_17d2_adds_balanced_axis_split_lateral_training(cfg) -> None:
    assert cfg.env.cmd_sampler_3d_branched is True
    assert cfg.env.cmd_sampler_walk_vx_positive_only is True
    assert cfg.env.cmd_sampler_walk_vy_exact_signed_bins is True
    assert cfg.env.min_velocity_y == pytest.approx(-0.065)
    assert cfg.env.max_velocity_y == pytest.approx(0.065)
    assert cfg.env.loc_ref_offline_command_vy_grid == pytest.approx(
        (-0.065, 0.0, 0.065)
    )
    assert cfg.env.loc_ref_offline_command_yaw_rate_grid == (0.0,)
    assert cfg.reward_weights.cmd_velocity_track_dim == 1
    assert cfg.reward_weights.cmd_forward_velocity_alpha_y == pytest.approx(120.0)
    assert cfg.reward_weights.cmd_lateral_velocity_track == pytest.approx(1.0)
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(1.5)


def test_17d2_preserves_latency_model_and_strengthens_eval(cfg) -> None:
    assert cfg.env.joint_feedback_sample_hold_enabled is True
    assert cfg.env.joint_feedback_leg_period_steps_range == (4, 7)
    assert cfg.env.joint_feedback_upper_period_steps_range == (12, 24)
    assert cfg.reward_weights.torque == pytest.approx(-0.001)
    assert cfg.ppo.eval.post_training_num_envs == 64
    assert cfg.ppo.eval.post_training_num_steps == 1000
    assert cfg.ppo.eval.post_training_strict_lateral_drift is True
    assert cfg.env.eval_velocity_cmd == pytest.approx((0.13, 0.0, 0.0))
    assert cfg.env.eval_velocity_cmd_probes == (
        (0.13, 0.065, 0.0),
        (0.13, -0.065, 0.0),
    )
