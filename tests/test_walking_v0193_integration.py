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


def test_v0193a_support_specific_margin_wiring() -> None:
    """Test that support-specific stance extension margin is wired and used correctly.
    
    Note on geometry: Larger stance_margin → less negative nominal_stance_z 
    → shorter effective leg length → MORE knee flexion (more crouched support).
    """
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    assert hasattr(cfg.env, "loc_ref_v2_support_stance_extension_margin_m")
    assert cfg.env.loc_ref_v2_support_stance_extension_margin_m == 0.055
    
    load_robot_config(Path(cfg.env.assets_root) / "mujoco_robot_config.json")
    cfg.freeze()
    env = WildRobotEnv(config=cfg)
    
    # Verify env stores both margins
    assert env._loc_ref_stance_extension_margin_m == jp.asarray(0.045, dtype=jp.float32)
    assert env._loc_ref_v2_support_stance_extension_margin_m == jp.asarray(0.055, dtype=jp.float32)
    
    # Test that adapter uses support margin for support modes
    # Mode 0 = STARTUP_SUPPORT_RAMP, Mode 1 = SUPPORT_STABILIZE
    support_q_refs = []
    for mode_id in [0, 1]:
        result = env._compute_nominal_q_ref_from_loc_ref(
            stance_foot_id=jp.asarray(0, dtype=jp.int32),
            swing_pos=jp.zeros(3, dtype=jp.float32),
            swing_vel=jp.zeros(3, dtype=jp.float32),
            pelvis_height=jp.asarray(0.44, dtype=jp.float32),
            pelvis_roll=jp.asarray(0.0, dtype=jp.float32),
            pelvis_pitch=jp.asarray(0.0, dtype=jp.float32),
            left_foot_pos_h=jp.asarray([0.0, 0.08, 0.0], dtype=jp.float32),
            right_foot_pos_h=jp.asarray([0.0, -0.08, 0.0], dtype=jp.float32),
            root_pitch=jp.asarray(0.0, dtype=jp.float32),
            root_pitch_rate=jp.asarray(0.0, dtype=jp.float32),
            forward_vel_h=jp.asarray(0.0, dtype=jp.float32),
            velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
            support_health=jp.asarray(1.0, dtype=jp.float32),
            mode_id=jp.asarray(mode_id, dtype=jp.int32),
        )
        # With support_health=1.0 and support margin=0.055:
        # stance_margin = 0.055 * (0.2 + 0.8*1.0) = 0.055
        # nominal_stance_z = -(0.44 - 0.055) = -0.385
        # Shorter effective leg (vs -0.395 with 0.045 margin) → MORE knee flexion
        assert result[0].shape == (env.action_size,)
        support_q_refs.append(result[0])
    
    # Test that adapter uses general margin for other modes (e.g., mode 2)
    result_other = env._compute_nominal_q_ref_from_loc_ref(
        stance_foot_id=jp.asarray(0, dtype=jp.int32),
        swing_pos=jp.zeros(3, dtype=jp.float32),
        swing_vel=jp.zeros(3, dtype=jp.float32),
        pelvis_height=jp.asarray(0.44, dtype=jp.float32),
        pelvis_roll=jp.asarray(0.0, dtype=jp.float32),
        pelvis_pitch=jp.asarray(0.0, dtype=jp.float32),
        left_foot_pos_h=jp.asarray([0.0, 0.08, 0.0], dtype=jp.float32),
        right_foot_pos_h=jp.asarray([0.0, -0.08, 0.0], dtype=jp.float32),
        root_pitch=jp.asarray(0.0, dtype=jp.float32),
        root_pitch_rate=jp.asarray(0.0, dtype=jp.float32),
        forward_vel_h=jp.asarray(0.0, dtype=jp.float32),
        velocity_cmd=jp.asarray(0.10, dtype=jp.float32),
        support_health=jp.asarray(1.0, dtype=jp.float32),
        mode_id=jp.asarray(2, dtype=jp.int32),
    )
    # With general margin=0.045:
    # stance_margin = 0.045 * (0.2 + 0.8*1.0) = 0.045
    # nominal_stance_z = -(0.44 - 0.045) = -0.395 (longer leg, less knee flexion)
    assert result_other[0].shape == (env.action_size,)
    nonsupport_q_ref = result_other[0]
    
    # CRITICAL: Verify geometric effect - support mode should produce MORE knee flexion
    # Find stance knee joint (left_knee_pitch when stance_foot_id=0)
    left_knee_idx = env._actuator_name_to_index.get("left_knee_pitch")
    if left_knee_idx is not None and left_knee_idx >= 0:
        support_knee = support_q_refs[0][left_knee_idx]  # mode 0
        nonsupport_knee = nonsupport_q_ref[left_knee_idx]  # mode 2
        # Support mode (margin=0.055) should have MORE positive knee angle than non-support (margin=0.045)
        # because shorter effective leg requires more flexion
        assert float(support_knee) > float(nonsupport_knee), (
            f"Support margin (0.055) should produce MORE knee flexion than general margin (0.045): "
            f"support_knee={float(support_knee):.4f} should be > nonsupport_knee={float(nonsupport_knee):.4f}"
        )
        # Verify the difference is material (at least 0.05 rad = ~3 degrees)
        assert float(support_knee) - float(nonsupport_knee) > 0.05, (
            f"Knee flexion difference too small: {float(support_knee) - float(nonsupport_knee):.4f} rad"
        )

