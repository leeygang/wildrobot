from __future__ import annotations

import pytest


_CONFIG = "training/configs/ppo_standing_stabilizer_v0227.yaml"


def _load_config_and_spec():
    from assets.robot_config import get_robot_config, load_robot_config
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    cfg = load_training_config(_CONFIG)
    load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=get_robot_config()
    )
    return cfg, spec


def test_v0227_is_planner_free_outcome_driven_stabilization() -> None:
    cfg, spec = _load_config_and_spec()

    assert cfg.version == "0.22.7"
    assert spec.observation.layout_id == "wr_obs_v9_standing"
    assert spec.model.obs_dim == 59
    assert spec.model.action_dim == 17
    assert cfg.env.standing_recovery_enabled is False
    assert cfg.env.reset_torso_pitch_rate_range == [-0.6, 0.6]
    assert cfg.env.reset_foot_stagger_range_m == [-0.04, 0.04]
    assert cfg.env.penalty_pose_deadzone_rad == pytest.approx(0.0698132)
    assert cfg.env.penalty_pose_weight_default == 1.0
    assert cfg.reward_weights.ref_body_quat_alpha == 50.0
    assert cfg.reward_weights.ang_vel_xy == 2.0
    assert cfg.reward_weights.feet_phase == 0.25
    assert cfg.reward_weights.recovery_swing_track == 0.0
    assert cfg.reward_weights.recovery_touchdown == 0.0
    assert cfg.reward_weights.recovery_squat == 0.0
    assert cfg.reward_weights.unnecessary_step == 0.0


def test_v0227_runtime_metadata_disables_recovery_planner() -> None:
    from training.exports.runtime_metadata import build_runtime_policy_config

    cfg, spec = _load_config_and_spec()
    metadata = build_runtime_policy_config(env=vars(cfg.env), spec=spec)

    assert metadata["policy_role"] == "standing"
    assert "recovery" not in metadata


def test_negative_pose_deadzone_is_rejected() -> None:
    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(_CONFIG)
    cfg.env.penalty_pose_deadzone_rad = -0.01
    load_robot_config(cfg.env.robot_config_path)

    with pytest.raises(ValueError, match="penalty_pose_deadzone_rad"):
        WildRobotEnv(config=cfg)
