from __future__ import annotations

import copy
from pathlib import Path

import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.configs.zmp_reference import zmp_walk_config_from_env
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb7_double_support_timing.yaml"
)
CHAMPION_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb4_hip_roll_margin_resume.yaml"
)


def test_tb7_changes_only_reference_timing_and_run_bookkeeping() -> None:
    cfg = load_training_config(CONFIG)
    champion = load_training_config(CHAMPION_CONFIG)

    assert cfg.version == "0.21.0-tb7"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 204_800
    assert cfg.ppo.eval.interval == 5
    assert cfg.checkpoints.interval == 5
    assert cfg.ppo.eval.post_training_top_k == 2
    assert cfg.env.loc_ref_single_double_ratio == pytest.approx(5.0 / 3.0)

    zmp_cfg = zmp_walk_config_from_env(cfg.env)
    assert zmp_cfg.cycle_time_s == pytest.approx(0.96)
    assert zmp_cfg.double_support_s == pytest.approx(0.18)
    assert zmp_cfg.single_support_s == pytest.approx(0.30)
    assert zmp_cfg.double_support_s + zmp_cfg.single_support_s == pytest.approx(0.48)

    champion_raw = copy.deepcopy(champion.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (champion_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"].pop("loc_ref_single_double_ratio", None)
        raw["ppo"].pop("iterations")
        raw["ppo"]["eval"].pop("interval")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["checkpoints"].pop("dir")
        raw["checkpoints"].pop("interval")
        raw["wandb"].pop("tags")
    assert cfg_raw == champion_raw


def test_tb7_preserves_champion_actor_contract() -> None:
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
