from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.configs.zmp_reference import zmp_walk_config_from_env
from training.exports.runtime_metadata import build_runtime_policy_config
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path("training/configs/ppo_walking_v0210_tb8_rated_actuator_headroom.yaml")
CHAMPION_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb4_hip_roll_margin_resume.yaml"
)


def test_tb8_changes_only_actuator_limit_and_run_bookkeeping() -> None:
    cfg = load_training_config(CONFIG)
    champion = load_training_config(CHAMPION_CONFIG)

    assert cfg.version == "0.21.0-tb8"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 204_800
    assert cfg.ppo.eval.interval == 5
    assert cfg.checkpoints.interval == 5
    assert cfg.ppo.eval.post_training_top_k == 2
    assert cfg.env.actuator_force_limit_nm == pytest.approx(4.4129925)

    zmp_cfg = zmp_walk_config_from_env(cfg.env)
    assert zmp_cfg.cycle_time_s == pytest.approx(0.96)
    assert zmp_cfg.double_support_s == pytest.approx(0.16)
    assert zmp_cfg.single_support_s == pytest.approx(0.32)
    assert zmp_cfg.double_support_s + zmp_cfg.single_support_s == pytest.approx(0.48)

    champion_raw = copy.deepcopy(champion.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (champion_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"].pop("actuator_force_limit_nm", None)
        raw["ppo"].pop("iterations")
        raw["ppo"]["eval"].pop("interval")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["checkpoints"].pop("dir")
        raw["checkpoints"].pop("interval")
        raw["wandb"].pop("tags")
    assert cfg_raw == champion_raw


def test_tb8_preserves_champion_actor_contract() -> None:
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    champion = load_training_config(CHAMPION_CONFIG)
    cfg = load_training_config(CONFIG)

    champion_spec = build_policy_spec_from_training_config(
        training_cfg=champion,
        robot_cfg=robot_cfg,
    )
    cfg_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    assert cfg_spec.to_json_dict() == champion_spec.to_json_dict()


def test_tb8_applies_hardware_rated_actuator_limit() -> None:
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(CONFIG)
    env = WildRobotEnv(config=cfg)

    np.testing.assert_allclose(
        env._mj_model.actuator_forcerange,
        np.tile([-4.4129925, 4.4129925], (env._mj_model.nu, 1)),
    )


def test_tb8_runtime_export_keeps_the_full_forward_command_grid() -> None:
    cfg = load_training_config(CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )

    metadata = build_runtime_policy_config(
        env=cfg.raw_config["env"],
        spec=spec,
    )

    np.testing.assert_allclose(
        np.asarray(metadata["reference"]["cmd_keys"], dtype=np.float32),
        np.asarray(
            [
                [cfg.env.min_velocity, 0.0, 0.0],
                [cfg.env.max_velocity, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=5e-5,
    )
