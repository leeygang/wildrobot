from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb9_sysid_actuator_adaptation.yaml")
BASE_CONFIG = Path("training/configs/ppo_walking_v0210_tb8_rated_actuator_headroom.yaml")


def test_tb9_changes_only_sysid_adaptation_bookkeeping_and_knee_margin() -> None:
    cfg = load_training_config(CONFIG)
    base = load_training_config(BASE_CONFIG)

    assert cfg.version == "0.21.0-tb9"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 409_600
    assert cfg.ppo.eval.interval == 5
    assert cfg.ppo.eval.num_envs == 64
    assert cfg.ppo.eval.num_steps == 1000
    assert cfg.ppo.eval.post_training_strict_walking_safety is True
    assert cfg.checkpoints.interval == 5
    assert cfg.env.torque_saturation_weights_per_joint == {
        "left_hip_roll": 1.0,
        "left_knee_pitch": 1.0,
        "right_hip_roll": 1.0,
        "right_knee_pitch": 1.0,
    }

    base_raw = copy.deepcopy(base.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (base_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"].pop("torque_saturation_weights_per_joint")
        raw["ppo"].pop("iterations")
        raw["checkpoints"].pop("dir")
        raw["wandb"].pop("tags")
    assert cfg_raw == base_raw


def test_tb9_preserves_tb8_actor_contract_and_toddlerbot_randomization() -> None:
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
    assert cfg.env.domain_rand_kp_scale_range == pytest.approx([0.9, 1.1])
    assert cfg.env.domain_rand_damping_scale_range == pytest.approx([0.8, 1.2])
    assert cfg.env.domain_rand_armature_scale_range == pytest.approx([0.8, 1.2])
    assert cfg.env.domain_rand_frictionloss_scale_range == pytest.approx([0.8, 1.2])


def test_tb9_compiles_the_validated_htd45h_nominal() -> None:
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(CONFIG)
    env = WildRobotEnv(config=cfg)
    model = env._mj_model

    for actuator_name in env._policy_spec.robot.actuator_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name
        )
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        dof_id = int(model.jnt_dofadr[joint_id])
        assert model.dof_damping[dof_id] == pytest.approx(1.10618)
        assert model.dof_armature[dof_id] == pytest.approx(0.024992)
        assert model.dof_frictionloss[dof_id] == pytest.approx(0.324094)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(31.902)

    np.testing.assert_allclose(
        model.actuator_forcerange,
        np.tile([-4.4129925, 4.4129925], (model.nu, 1)),
    )
