from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jp

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import WildRobotEnv
from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
from policy_contract.calib import JaxCalibOps


def _make_env() -> WildRobotEnv:
    cfg = load_training_config("training/configs/ppo_walking.yaml")
    load_robot_config(Path(cfg.env.assets_root) / "mujoco_robot_config.json")
    cfg.freeze()
    return WildRobotEnv(config=cfg)


def test_v0193_config_parses_loc_ref_fields() -> None:
    cfg = load_training_config("training/configs/ppo_walking.yaml")
    assert cfg.version == "0.19.3"
    assert cfg.env.actor_obs_layout_id == "wr_obs_v4"
    assert cfg.env.loc_ref_enabled is True
    assert cfg.env.loc_ref_residual_scale > 0.0
    assert cfg.reward_weights.m3_pelvis_orientation_tracking >= 0.0


def test_wr_obs_v4_contains_reference_fields() -> None:
    env = _make_env()
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    assert wr.loc_ref_next_foothold.shape == (2,)
    assert wr.loc_ref_swing_pos.shape == (3,)
    assert wr.loc_ref_swing_vel.shape == (3,)
    assert wr.loc_ref_history.shape == (4,)
    assert wr.nominal_q_ref.shape == (env.action_size,)
    assert state.obs.shape[-1] == env.observation_size


def test_residual_action_composes_around_nominal_q_ref() -> None:
    env = _make_env()
    state = env.reset(jax.random.PRNGKey(1))
    wr = state.info[WR_INFO_KEY]
    zero_policy = jp.zeros((env.action_size,), dtype=jp.float32)
    raw_action, residual_delta_q = env._compose_loc_ref_residual_action(  # noqa: SLF001
        policy_action=zero_policy,
        nominal_q_ref=wr.nominal_q_ref,
    )
    ctrl = JaxCalibOps.action_to_ctrl(spec=env._policy_spec, action=raw_action)  # noqa: SLF001
    assert jp.max(jp.abs(residual_delta_q)) == 0.0
    assert jp.max(jp.abs(ctrl - wr.nominal_q_ref)) < 1e-4


def test_step_emits_m3_metrics_and_preserves_standing_regression() -> None:
    env = _make_env()
    state = env.reset(jax.random.PRNGKey(2))
    state = env.step(state, jp.zeros((env.action_size,), dtype=jp.float32))
    metrics = unpack_metrics(state.metrics[METRICS_VEC_KEY])
    assert "reward/m3_pelvis_orientation_tracking" in metrics
    assert "tracking/cmd_vs_achieved_forward" in metrics
    assert "tracking/residual_q_abs_mean" in metrics
    assert "tracking/loc_ref_left_reachable" in metrics
    assert "tracking/loc_ref_right_reachable" in metrics
    assert "tracking/loc_ref_phase_progress" in metrics
    assert "debug/m3_swing_pos_error" in metrics
    assert "debug/m3_swing_vel_error" in metrics
    assert "debug/m3_foothold_error" in metrics
    assert "term/height_low" in metrics
    assert jp.isfinite(metrics["reward/m3_pelvis_orientation_tracking"])
    assert jp.isfinite(metrics["tracking/residual_q_abs_mean"])
    assert metrics["tracking/loc_ref_phase_progress"] >= 0.0
    assert metrics["tracking/loc_ref_phase_progress"] <= 1.0


def test_v0193a_config_parses_nominal_path_knobs() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    assert cfg.env.loc_ref_step_time_s > 0.0
    assert cfg.env.loc_ref_max_step_length_m >= cfg.env.loc_ref_min_step_length_m
    assert 0.0 <= cfg.env.loc_ref_dcm_placement_gain <= 1.0
