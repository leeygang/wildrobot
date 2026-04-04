from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jp

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.env_info import WR_INFO_KEY
from training.envs.wildrobot_env import (
    WildRobotEnv,
    _loc_ref_support_health,
    _loc_ref_support_scales,
)
from control.references.walking_ref_v2 import compute_support_health_v2_jax, WalkingRefV2Config
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
    assert "debug/loc_ref_nominal_vs_applied_q_l1" in metrics
    assert "debug/loc_ref_swing_x_target" in metrics
    assert "debug/loc_ref_swing_x_actual" in metrics
    assert "debug/loc_ref_swing_x_error" in metrics
    assert "debug/loc_ref_pelvis_pitch_target" in metrics
    assert "debug/loc_ref_root_pitch" in metrics
    assert "debug/loc_ref_root_pitch_rate" in metrics
    assert "debug/loc_ref_swing_x_scale" in metrics
    assert "debug/loc_ref_pelvis_pitch_scale" in metrics
    assert "debug/loc_ref_support_gate_active" in metrics
    assert "debug/loc_ref_swing_x_scale_active" in metrics
    assert "debug/loc_ref_phase_scale_active" in metrics
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
    assert cfg.env.loc_ref_overspeed_deadband >= 0.0
    assert cfg.env.loc_ref_overspeed_brake_gain >= 0.0
    assert cfg.env.loc_ref_overspeed_phase_slowdown_gain >= 0.0
    assert 0.0 < cfg.env.loc_ref_overspeed_phase_min_scale <= 1.0
    assert cfg.env.loc_ref_pitch_brake_start_rad >= 0.0
    assert cfg.env.loc_ref_pitch_brake_gain >= 0.0
    assert cfg.env.loc_ref_swing_x_brake_pitch_start_rad >= 0.0
    assert cfg.env.loc_ref_swing_x_brake_overspeed_deadband >= 0.0
    assert cfg.env.loc_ref_swing_x_brake_gain >= 0.0
    assert 0.0 <= cfg.env.loc_ref_swing_x_min_scale <= 1.0
    assert cfg.env.loc_ref_pelvis_pitch_brake_gain >= 0.0
    assert 0.0 <= cfg.env.loc_ref_pelvis_pitch_min_scale <= 1.0
    assert cfg.env.loc_ref_support_pitch_rate_start_rad_s >= 0.0
    assert cfg.env.loc_ref_support_health_gain >= 0.0
    assert 0.0 <= cfg.env.loc_ref_support_release_phase_start <= 1.0
    assert 0.0 <= cfg.env.loc_ref_support_foothold_min_scale <= 1.0
    assert 0.0 <= cfg.env.loc_ref_support_swing_progress_min_scale <= 1.0
    assert 0.0 < cfg.env.loc_ref_support_phase_min_scale <= 1.0
    assert cfg.env.loc_ref_max_lateral_release_m >= 0.0
    assert cfg.env.loc_ref_max_lateral_release_m <= cfg.env.loc_ref_max_lateral_step_m
    assert abs(cfg.env.loc_ref_max_lateral_release_m - 0.015) <= 1e-9
    # Ablation-overridable channels should parse cleanly.
    assert cfg.env.loc_ref_swing_target_blend >= 0.0
    assert cfg.env.loc_ref_step_time_s > 0.0


def test_loc_ref_support_health_drops_under_instability() -> None:
    health, instability = _loc_ref_support_health(
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.80, dtype=jp.float32),
        pitch_rad=jp.asarray(0.18, dtype=jp.float32),
        pitch_rate_rad_s=jp.asarray(1.8, dtype=jp.float32),
        overspeed_deadband=jp.asarray(0.01, dtype=jp.float32),
        pitch_start_rad=jp.asarray(0.06, dtype=jp.float32),
        pitch_rate_start_rad_s=jp.asarray(0.60, dtype=jp.float32),
        health_gain=jp.asarray(10.0, dtype=jp.float32),
    )
    assert instability > 0.0
    assert health < 0.5


def test_loc_ref_support_health_is_one_when_stable() -> None:
    health, instability = _loc_ref_support_health(
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.09, dtype=jp.float32),
        pitch_rad=jp.asarray(0.01, dtype=jp.float32),
        pitch_rate_rad_s=jp.asarray(0.10, dtype=jp.float32),
        overspeed_deadband=jp.asarray(0.01, dtype=jp.float32),
        pitch_start_rad=jp.asarray(0.06, dtype=jp.float32),
        pitch_rate_start_rad_s=jp.asarray(0.60, dtype=jp.float32),
        health_gain=jp.asarray(10.0, dtype=jp.float32),
    )
    assert instability == 0.0
    assert jp.abs(health - 1.0) < 1e-6


def test_loc_ref_support_scales_brake_instability() -> None:
    swing_scale, pelvis_scale, overspeed, gate = _loc_ref_support_scales(
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.80, dtype=jp.float32),
        pitch_rad=jp.asarray(0.25, dtype=jp.float32),
        swing_x_brake_pitch_start_rad=jp.asarray(0.06, dtype=jp.float32),
        swing_x_brake_overspeed_deadband=jp.asarray(0.01, dtype=jp.float32),
        swing_x_brake_gain=jp.asarray(12.0, dtype=jp.float32),
        swing_x_min_scale=jp.asarray(0.0, dtype=jp.float32),
        pelvis_pitch_brake_gain=jp.asarray(14.0, dtype=jp.float32),
        pelvis_pitch_min_scale=jp.asarray(0.0, dtype=jp.float32),
    )
    assert overspeed > 0.0
    assert gate > 0.0
    assert swing_scale < 0.5
    assert pelvis_scale < 0.5


def test_loc_ref_support_scales_idle_when_stable() -> None:
    swing_scale, pelvis_scale, overspeed, gate = _loc_ref_support_scales(
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.09, dtype=jp.float32),
        pitch_rad=jp.asarray(0.01, dtype=jp.float32),
        swing_x_brake_pitch_start_rad=jp.asarray(0.06, dtype=jp.float32),
        swing_x_brake_overspeed_deadband=jp.asarray(0.01, dtype=jp.float32),
        swing_x_brake_gain=jp.asarray(12.0, dtype=jp.float32),
        swing_x_min_scale=jp.asarray(0.0, dtype=jp.float32),
        pelvis_pitch_brake_gain=jp.asarray(14.0, dtype=jp.float32),
        pelvis_pitch_min_scale=jp.asarray(0.0, dtype=jp.float32),
    )
    assert overspeed == 0.0
    assert gate == 0.0
    assert jp.abs(swing_scale - 1.0) < 1e-6
    assert jp.abs(pelvis_scale - 1.0) < 1e-6


def test_v2_support_health_matches_reference_definition() -> None:
    cfg = WalkingRefV2Config(
        support_overspeed_deadband=0.01,
        support_pitch_start_rad=0.06,
        support_pitch_rate_start_rad_s=0.60,
        support_health_gain=10.0,
    )
    health_env, instability_env = _loc_ref_support_health(
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.10, dtype=jp.float32),
        pitch_rad=jp.asarray(0.01, dtype=jp.float32),
        pitch_rate_rad_s=jp.asarray(0.10, dtype=jp.float32),
        overspeed_deadband=jp.asarray(cfg.support_overspeed_deadband, dtype=jp.float32),
        pitch_start_rad=jp.asarray(cfg.support_pitch_start_rad, dtype=jp.float32),
        pitch_rate_start_rad_s=jp.asarray(cfg.support_pitch_rate_start_rad_s, dtype=jp.float32),
        health_gain=jp.asarray(cfg.support_health_gain, dtype=jp.float32),
        left_foot_loaded=jp.asarray(True),
        right_foot_loaded=jp.asarray(True),
    )
    health_ref, instability_ref = compute_support_health_v2_jax(
        config=cfg,
        forward_speed_mps=jp.asarray(0.10, dtype=jp.float32),
        root_pitch_rad=jp.asarray(0.01, dtype=jp.float32),
        root_pitch_rate_rad_s=jp.asarray(0.10, dtype=jp.float32),
        com_velocity_stance_frame=jp.asarray([0.10, 0.0], dtype=jp.float32),
        left_foot_loaded=jp.asarray(True),
        right_foot_loaded=jp.asarray(True),
    )
    assert jp.abs(instability_ref - instability_env) < 1e-6
    assert jp.abs(health_ref - health_env) < 1e-6
