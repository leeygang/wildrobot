from __future__ import annotations

import copy
from pathlib import Path

import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.configs.zmp_reference import zmp_walk_config_from_env
from training.eval.audit_reference_feasibility import (
    build_reference_feasibility_report,
)
from training.policy_spec_utils import build_policy_spec_from_training_config


CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb11_reference_geometry_com_lever.yaml"
)
BASE_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb9_sysid_actuator_adaptation.yaml"
)
TB10_CONFIG = Path(
    "training/configs/ppo_walking_v0210_tb10_whole_leg_headroom.yaml"
)


def test_tb11_changes_only_reference_geometry_com_reward_and_evaluation() -> None:
    cfg = load_training_config(CONFIG)
    base = load_training_config(BASE_CONFIG)

    assert cfg.version == "0.21.0-tb11"
    assert cfg.ppo.num_envs * cfg.ppo.rollout_steps * cfg.ppo.iterations == 409_600
    assert cfg.env.loc_ref_default_stance_width_m == pytest.approx(0.0782)
    assert cfg.env.loc_ref_hip_lateral_offset_m == pytest.approx(0.0892)
    assert cfg.reward_weights.single_support_com_lateral == pytest.approx(-0.05)
    assert cfg.env.eval_velocity_cmd_probes == ((0.066667, 0.0, 0.0),)
    assert cfg.ppo.eval.post_training_top_k == 4

    base_raw = copy.deepcopy(base.raw_config)
    cfg_raw = copy.deepcopy(cfg.raw_config)
    for raw in (base_raw, cfg_raw):
        raw.pop("version")
        raw.pop("version_name")
        raw["env"].pop("loc_ref_default_stance_width_m")
        raw["env"].pop("loc_ref_hip_lateral_offset_m", None)
        raw["env"].pop("eval_velocity_cmd_probes")
        raw["ppo"]["eval"].pop("post_training_top_k")
        raw["reward_weights"].pop("single_support_com_lateral", None)
        raw["checkpoints"].pop("dir")
        raw["wandb"].pop("tags")
    assert cfg_raw == base_raw


def test_tb11_preserves_tb9_actor_contract() -> None:
    cfg = load_training_config(CONFIG)
    base = load_training_config(BASE_CONFIG)
    robot_cfg = load_robot_config(cfg.env.robot_config_path)

    cfg_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
    )
    base_spec = build_policy_spec_from_training_config(
        training_cfg=base,
        robot_cfg=robot_cfg,
    )

    assert cfg_spec.to_json_dict() == base_spec.to_json_dict()


def test_tb11_reference_geometry_matches_mjcf() -> None:
    cfg = load_training_config(CONFIG)
    base = load_training_config(BASE_CONFIG)
    zmp_cfg = zmp_walk_config_from_env(cfg.env)
    base_zmp_cfg = zmp_walk_config_from_env(base.env)

    assert zmp_cfg.hip_lateral_offset_m == pytest.approx(0.0892)
    assert zmp_cfg.default_stance_width_m == pytest.approx(0.0782)
    assert base_zmp_cfg.hip_lateral_offset_m == pytest.approx(0.0536)

    report = build_reference_feasibility_report(
        config_path=CONFIG.resolve(),
        commands=[(0.133333, 0.0, 0.0)],
    )
    command = report["commands"][0]
    assert report["passed"] is True
    assert report["morphology"]["hip_lateral_offset_gap_m"] < 0.001
    assert command["planned_foot_separation_m"]["mean"] > 0.146
    assert command["realized_foot_separation_m"]["mean"] > 0.146
    for side in ("left", "right"):
        assert command["support"][side]["realization_gap_abs_m"]["p95"] < 0.002
    assert report["reference_coupling"]["joint_reference_is_critic_only"] is True


def test_reference_audit_detects_tb10_geometry_mismatch() -> None:
    report = build_reference_feasibility_report(
        config_path=TB10_CONFIG.resolve(),
        commands=[(0.133333, 0.0, 0.0)],
    )
    command = report["commands"][0]

    assert report["passed"] is False
    assert report["morphology"]["hip_lateral_offset_gap_m"] > 0.03
    assert command["planned_foot_separation_m"]["mean"] < 0.146
    for side in ("left", "right"):
        assert command["support"][side]["realization_gap_abs_m"]["p95"] > 0.03
