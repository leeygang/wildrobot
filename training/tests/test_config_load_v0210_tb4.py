from __future__ import annotations

from pathlib import Path

import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb4_hip_roll_margin_resume.yaml")
TB3_CONFIG = Path("training/configs/ppo_walking_v0210_tb3_forward_shared_stance.yaml")


def test_tb4_changes_only_the_targeted_margin_experiment() -> None:
    cfg = load_training_config(CONFIG)
    tb3 = load_training_config(TB3_CONFIG)

    assert cfg.version == "0.21.0-tb4"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 2_498_560
    assert cfg.ppo.eval.interval == 20
    assert cfg.checkpoints.interval == 20
    assert cfg.ppo.eval.post_training_top_k == 6

    assert cfg.env.torque_saturation_soft_limit_ratio == pytest.approx(0.90)
    assert cfg.env.torque_saturation_weight_default == pytest.approx(0.0)
    assert cfg.env.torque_saturation_weights_per_joint == {
        "left_hip_roll": pytest.approx(1.0),
        "right_hip_roll": pytest.approx(1.0),
    }
    assert cfg.reward_weights.torque == pytest.approx(0.0)
    assert cfg.reward_weights.saturation == pytest.approx(-0.01)

    assert cfg.env.home_joint_offsets_rad == tb3.env.home_joint_offsets_rad
    assert cfg.env.actor_obs_layout_id == tb3.env.actor_obs_layout_id
    assert cfg.env.policy_excluded_actuator_names == (
        tb3.env.policy_excluded_actuator_names
    )
    assert cfg.ppo.optimizer_profile == tb3.ppo.optimizer_profile
    assert cfg.ppo.mirror_loss_coef == pytest.approx(0.0)
    assert cfg.ppo.source_policy_kl_coef == pytest.approx(0.0)
    assert cfg.networks.actor == tb3.networks.actor
    assert cfg.networks.critic == tb3.networks.critic


def test_tb4_policy_contract_matches_tb3_for_full_state_resume() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    tb3 = load_training_config(TB3_CONFIG)
    tb4 = load_training_config(CONFIG)

    tb3_spec = build_policy_spec_from_training_config(
        training_cfg=tb3,
        robot_cfg=robot_cfg,
    )
    tb4_spec = build_policy_spec_from_training_config(
        training_cfg=tb4,
        robot_cfg=robot_cfg,
    )

    assert tb4_spec.to_json_dict() == tb3_spec.to_json_dict()
