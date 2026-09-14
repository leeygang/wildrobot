from __future__ import annotations

import copy
from pathlib import Path

import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb10_whole_leg_headroom.yaml")
BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb9_sysid_actuator_adaptation.yaml"
)

LEG_ACTUATORS = {
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
}


def test_tb10_changes_only_whole_leg_headroom_and_bookkeeping() -> None:
    cfg = load_training_config(CONFIG)
    base = load_training_config(BASE_CONFIG)

    assert cfg.version == "0.21.0-tb10"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 409_600
    assert cfg.env.torque_saturation_soft_limit_ratio == pytest.approx(0.8)
    assert cfg.env.torque_saturation_weight_default == pytest.approx(0.0)
    assert cfg.env.torque_saturation_weights_per_joint == {
        name: 1.0 for name in LEG_ACTUATORS
    }
    assert cfg.reward_weights.saturation == pytest.approx(-0.01)
    assert cfg.reward_weights.torque == pytest.approx(0.0)
    assert len(cfg.env.eval_velocity_cmd_probes) == 1
    assert cfg.env.eval_velocity_cmd_probes[0] == pytest.approx(
        (0.066667, 0.0, 0.0)
    )

    base_raw = copy.deepcopy(base.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (base_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"].pop("torque_saturation_soft_limit_ratio")
        raw["env"].pop("torque_saturation_weights_per_joint")
        raw["env"].pop("eval_velocity_cmd_probes")
        raw["checkpoints"].pop("dir")
        raw["wandb"].pop("tags")
    assert cfg_raw == base_raw


def test_tb10_preserves_tb9_policy_and_toddlerbot_learner_contract() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    base = load_training_config(BASE_CONFIG)
    cfg = load_training_config(CONFIG)

    base_spec = build_policy_spec_from_training_config(
        training_cfg=base,
        robot_cfg=robot_cfg,
    )
    cfg_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg_spec.to_json_dict() == base_spec.to_json_dict()
    assert cfg.ppo.epochs == 4
    assert cfg.ppo.optimizer_profile == "toddlerbot_rsl_rl_2_3_3"
    assert cfg.ppo.eval.num_envs == 64
    assert cfg.ppo.eval.num_steps == 1000
    assert cfg.ppo.eval.post_training_strict_walking_safety is True
