"""Centralized configuration management for WildRobot training.

This module provides utilities to load and manage training configuration
from training YAML files.

Design Principle: Single freezable config classes defined in training_runtime_config.py.
Call config.freeze() after CLI overrides to make configs JIT-compatible.

Usage:
    from training.configs.training_config import load_training_config
    from assets.robot_config import load_robot_config

    # Load robot config (path passed from train.py)
    robot_config = load_robot_config("assets/v2/mujoco_robot_config.json")

    # Load training config
    config = load_training_config("configs/ppo_walking_v0201_smoke.yaml")

    # Access config properties (new unified access pattern)
    print(config.networks.actor.hidden_sizes)  # (256, 256, 128)
    print(config.ppo.learning_rate)            # 3e-4

    # Modify config before training
    config.ppo.learning_rate = 1e-4

    # Freeze for JIT (call once before training)
    config.freeze()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Re-export robot config utilities
from assets.robot_config import (  # noqa: F401
    clear_robot_config_cache,
    get_obs_indices,
    get_robot_config,
    load_robot_config,
    RobotConfig,
)

# Import all config classes from runtime config
from training.configs.training_runtime_config import (
    ActorNetworkConfig,
    CheckpointConfig,
    CriticNetworkConfig,
    EnvConfig,
    Freezable,
    FrozenInstanceError,
    NetworksConfig,
    PPOConfig,
    PPOEvalConfig,
    PPORollbackConfig,
    RewardCompositionConfig,
    RewardWeightsConfig,
    TrainingConfig,
    VideoConfig,
    WandbConfig,
)
from training.configs.asset_paths import resolve_env_asset_paths


__all__ = [
    # Main config
    "TrainingConfig",
    "load_training_config",
    "get_training_config",
    "clear_config_cache",
    # Sub-configs
    "EnvConfig",
    "PPOConfig",
    "PPOEvalConfig",
    "PPORollbackConfig",
    "NetworksConfig",
    "ActorNetworkConfig",
    "CriticNetworkConfig",
    "RewardWeightsConfig",
    "RewardCompositionConfig",
    "CheckpointConfig",
    "WandbConfig",
    "VideoConfig",
    # Utils
    "Freezable",
    "FrozenInstanceError",
    # Robot config re-exports
    "RobotConfig",
    "load_robot_config",
    "get_robot_config",
]


def _parse_list_to_tuple(value: Any, default: Tuple) -> Tuple[int, ...]:
    """Convert list from YAML to tuple for hidden_sizes."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default


def _parse_env_config(config: Dict[str, Any]) -> EnvConfig:
    """Parse environment configuration from YAML dict."""
    env = config.get("env", {})
    resolved = resolve_env_asset_paths(env)
    controller_stack = str(env.get("controller_stack", "ppo"))
    allowed_controller_stacks = {"ppo", "mpc_standing", "ppo_teacher_standing"}
    if controller_stack not in allowed_controller_stacks:
        raise ValueError(
            f"Invalid env.controller_stack='{controller_stack}'. "
            f"Allowed values: {sorted(allowed_controller_stacks)}"
        )

    return EnvConfig(
        assets_root=resolved.assets_root,
        scene_xml_path=resolved.scene_xml_path,
        robot_config_path=resolved.robot_config_path,
        mjcf_path=resolved.mjcf_path,
        model_path=resolved.model_path,
        realism_profile_path=env.get("realism_profile_path"),
        sim_dt=env.get("sim_dt", 0.002),
        ctrl_dt=env.get("ctrl_dt", 0.02),
        max_episode_steps=env.get("max_episode_steps", 500),
        target_height=env.get("target_height", 0.45),
        height_target_two_sided=bool(env.get("height_target_two_sided", False)),
        min_height=env.get("min_height", 0.20),
        collapse_height_buffer=env.get("collapse_height_buffer", 0.02),
        collapse_height_sigma=env.get("collapse_height_sigma", 0.03),
        collapse_vz_gate_band=env.get("collapse_vz_gate_band", 0.05),
        max_height=env.get("max_height", 0.70),
        max_pitch=env.get("max_pitch", 0.8),
        max_roll=env.get("max_roll", 0.8),
        use_relaxed_termination=bool(env.get("use_relaxed_termination", False)),
        min_velocity=env.get("min_velocity", 0.0),
        max_velocity=env.get("max_velocity", 1.0),
        contact_threshold_force=env.get("contact_threshold_force", 5.0),
        contact_scale=env.get("contact_scale", 10.0),
        foot_switch_threshold=env.get("foot_switch_threshold", 2.0),
        # action_filter_alpha=0 disables filtering
        action_filter_alpha=env.get("action_filter_alpha", 0.7),
        actor_obs_layout_id=str(env.get("actor_obs_layout_id", "wr_obs_v1")),
        loc_ref_enabled=bool(env.get("loc_ref_enabled", False)),
        loc_ref_version=str(env.get("loc_ref_version", "v3_offline_library")),
        loc_ref_residual_scale=float(env.get("loc_ref_residual_scale", 0.18)),
        # v0.20.1 smoke residual bounds / §4.1 —
        # residual mode + per-joint override.
        loc_ref_residual_mode=str(env.get("loc_ref_residual_mode", "absolute")),
        loc_ref_residual_scale_per_joint=dict(
            env.get("loc_ref_residual_scale_per_joint", {})
        ),
        # v0.20.1 smoke8 — residual base selector.  See dataclass docstring.
        loc_ref_residual_base=str(env.get("loc_ref_residual_base", "q_ref")),
        # v0.20.1 smoke8b — penalty_pose anchor selector.  See dataclass docstring.
        loc_ref_penalty_pose_anchor=str(
            env.get("loc_ref_penalty_pose_anchor", "q_ref")
        ),
        # v0.20.1 v3_offline_library — offline ReferenceLibrary source.
        loc_ref_offline_library_path=(
            None if env.get("loc_ref_offline_library_path") in (None, "")
            else str(env.get("loc_ref_offline_library_path"))
        ),
        loc_ref_offline_command_vx=float(
            # Phase 9D operating point.  Matches the dataclass default
            # in training_runtime_config.py and the smoke YAML.
            env.get("loc_ref_offline_command_vx", 0.20)
        ),
        loc_ref_step_time_s=float(env.get("loc_ref_step_time_s", 0.36)),
        loc_ref_walking_pelvis_height_m=float(env.get("loc_ref_walking_pelvis_height_m",
            env.get("loc_ref_nominal_com_height_m", 0.40))),
        loc_ref_nominal_lateral_foot_offset_m=float(
            env.get("loc_ref_nominal_lateral_foot_offset_m", 0.09)
        ),
        loc_ref_min_step_length_m=float(env.get("loc_ref_min_step_length_m", 0.02)),
        loc_ref_max_step_length_m=float(env.get("loc_ref_max_step_length_m", 0.14)),
        loc_ref_max_lateral_step_m=float(env.get("loc_ref_max_lateral_step_m", 0.14)),
        loc_ref_swing_height_m=float(env.get("loc_ref_swing_height_m", 0.04)),
        loc_ref_pelvis_roll_bias_rad=float(env.get("loc_ref_pelvis_roll_bias_rad", 0.03)),
        loc_ref_pelvis_pitch_gain=float(env.get("loc_ref_pelvis_pitch_gain", 0.08)),
        loc_ref_max_pelvis_pitch_rad=float(env.get("loc_ref_max_pelvis_pitch_rad", 0.08)),
        loc_ref_dcm_placement_gain=float(env.get("loc_ref_dcm_placement_gain", 1.0)),
        loc_ref_swing_target_blend=float(env.get("loc_ref_swing_target_blend", 0.65)),
        loc_ref_stance_height_blend=float(env.get("loc_ref_stance_height_blend", 0.25)),
        loc_ref_support_margin_m=float(env.get("loc_ref_support_margin_m", 0.020)),
        loc_ref_walking_crouch_extra_m=float(env.get("loc_ref_walking_crouch_extra_m", 0.045)),
        loc_ref_max_swing_x_delta_m=float(env.get("loc_ref_max_swing_x_delta_m", 0.04)),
        loc_ref_max_swing_z_delta_m=float(env.get("loc_ref_max_swing_z_delta_m", 0.03)),
        loc_ref_swing_y_to_hip_roll=float(env.get("loc_ref_swing_y_to_hip_roll", 0.30)),
        loc_ref_overspeed_deadband=float(env.get("loc_ref_overspeed_deadband", 0.05)),
        loc_ref_overspeed_brake_gain=float(env.get("loc_ref_overspeed_brake_gain", 1.5)),
        loc_ref_overspeed_phase_slowdown_gain=float(
            env.get("loc_ref_overspeed_phase_slowdown_gain", 2.5)
        ),
        loc_ref_overspeed_phase_min_scale=float(
            env.get("loc_ref_overspeed_phase_min_scale", 0.2)
        ),
        loc_ref_pitch_brake_start_rad=float(env.get("loc_ref_pitch_brake_start_rad", 0.12)),
        loc_ref_pitch_brake_gain=float(env.get("loc_ref_pitch_brake_gain", 1.0)),
        loc_ref_swing_x_brake_pitch_start_rad=float(
            env.get("loc_ref_swing_x_brake_pitch_start_rad", 0.08)
        ),
        loc_ref_swing_x_brake_overspeed_deadband=float(
            env.get("loc_ref_swing_x_brake_overspeed_deadband", 0.03)
        ),
        loc_ref_swing_x_brake_gain=float(env.get("loc_ref_swing_x_brake_gain", 5.0)),
        loc_ref_swing_x_min_scale=float(env.get("loc_ref_swing_x_min_scale", 0.05)),
        loc_ref_pelvis_pitch_brake_gain=float(
            env.get("loc_ref_pelvis_pitch_brake_gain", 6.0)
        ),
        loc_ref_pelvis_pitch_min_scale=float(
            env.get("loc_ref_pelvis_pitch_min_scale", 0.0)
        ),
        loc_ref_support_pitch_rate_start_rad_s=float(
            env.get("loc_ref_support_pitch_rate_start_rad_s", 0.8)
        ),
        loc_ref_support_health_gain=float(
            env.get("loc_ref_support_health_gain", 4.0)
        ),
        loc_ref_support_release_phase_start=float(
            env.get("loc_ref_support_release_phase_start", 0.35)
        ),
        loc_ref_support_foothold_min_scale=float(
            env.get("loc_ref_support_foothold_min_scale", 0.35)
        ),
        loc_ref_support_swing_progress_min_scale=float(
            env.get("loc_ref_support_swing_progress_min_scale", 0.0)
        ),
        loc_ref_support_phase_min_scale=float(
            env.get("loc_ref_support_phase_min_scale", 0.15)
        ),
        loc_ref_max_lateral_release_m=float(
            env.get("loc_ref_max_lateral_release_m", 0.02)
        ),
        loc_ref_v2_support_open_threshold=float(
            env.get("loc_ref_v2_support_open_threshold", 0.60)
        ),
        loc_ref_v2_support_release_threshold=float(
            env.get("loc_ref_v2_support_release_threshold", 0.30)
        ),
        loc_ref_v2_touchdown_phase_min=float(
            env.get("loc_ref_v2_touchdown_phase_min", 0.55)
        ),
        loc_ref_v2_capture_hold_s=float(
            env.get("loc_ref_v2_capture_hold_s", 0.04)
        ),
        loc_ref_v2_settle_hold_s=float(
            env.get("loc_ref_v2_settle_hold_s", 0.04)
        ),
        loc_ref_v2_support_stabilize_max_foothold_scale=float(
            env.get("loc_ref_v2_support_stabilize_max_foothold_scale", 0.35)
        ),
        loc_ref_v2_post_settle_swing_scale=float(
            env.get("loc_ref_v2_post_settle_swing_scale", 0.15)
        ),
        loc_ref_v2_startup_ramp_s=float(
            env.get("loc_ref_v2_startup_ramp_s", 1.20)
        ),
        loc_ref_v2_startup_pelvis_height_offset_m=float(
            env.get("loc_ref_v2_startup_pelvis_height_offset_m", 0.07)
        ),
        loc_ref_v2_startup_support_open_health=float(
            env.get("loc_ref_v2_startup_support_open_health", 0.25)
        ),
        loc_ref_v2_startup_handoff_pitch_max_rad=float(
            env.get("loc_ref_v2_startup_handoff_pitch_max_rad", 0.30)
        ),
        loc_ref_v2_startup_handoff_pitch_rate_max_rad_s=float(
            env.get("loc_ref_v2_startup_handoff_pitch_rate_max_rad_s", 1.50)
        ),
        loc_ref_v2_startup_readiness_knee_err_good_rad=float(
            env.get("loc_ref_v2_startup_readiness_knee_err_good_rad", 0.10)
        ),
        loc_ref_v2_startup_readiness_knee_err_bad_rad=float(
            env.get("loc_ref_v2_startup_readiness_knee_err_bad_rad", 0.45)
        ),
        loc_ref_v2_startup_readiness_ankle_err_good_rad=float(
            env.get("loc_ref_v2_startup_readiness_ankle_err_good_rad", 0.08)
        ),
        loc_ref_v2_startup_readiness_ankle_err_bad_rad=float(
            env.get("loc_ref_v2_startup_readiness_ankle_err_bad_rad", 0.35)
        ),
        loc_ref_v2_startup_readiness_pitch_good_rad=float(
            env.get("loc_ref_v2_startup_readiness_pitch_good_rad", 0.05)
        ),
        loc_ref_v2_startup_readiness_pitch_bad_rad=float(
            env.get("loc_ref_v2_startup_readiness_pitch_bad_rad", 0.25)
        ),
        loc_ref_v2_startup_readiness_pitch_rate_good_rad_s=float(
            env.get("loc_ref_v2_startup_readiness_pitch_rate_good_rad_s", 0.60)
        ),
        loc_ref_v2_startup_readiness_pitch_rate_bad_rad_s=float(
            env.get("loc_ref_v2_startup_readiness_pitch_rate_bad_rad_s", 2.00)
        ),
        loc_ref_v2_startup_readiness_min_health=float(
            env.get("loc_ref_v2_startup_readiness_min_health", 0.20)
        ),
        loc_ref_v2_startup_progress_min_scale=float(
            env.get("loc_ref_v2_startup_progress_min_scale", 0.20)
        ),
        loc_ref_v2_startup_realization_lead_alpha=float(
            env.get("loc_ref_v2_startup_realization_lead_alpha", 0.25)
        ),
        loc_ref_v2_startup_handoff_min_readiness=float(
            env.get("loc_ref_v2_startup_handoff_min_readiness", 0.10)
        ),
        loc_ref_v2_startup_handoff_min_pelvis_realization=float(
            env.get("loc_ref_v2_startup_handoff_min_pelvis_realization", 0.60)
        ),
        loc_ref_v2_startup_handoff_min_alpha=float(
            env.get("loc_ref_v2_startup_handoff_min_alpha", 0.85)
        ),
        loc_ref_v2_startup_handoff_timeout_s=float(
            env.get("loc_ref_v2_startup_handoff_timeout_s", 1.20)
        ),
        loc_ref_v2_startup_target_rate_design_rad_s=float(
            env.get("loc_ref_v2_startup_target_rate_design_rad_s", 0.5235987756)
        ),
        loc_ref_v2_startup_target_rate_hard_cap_rad_s=float(
            env.get("loc_ref_v2_startup_target_rate_hard_cap_rad_s", 1.7453292520)
        ),
        loc_ref_v2_startup_route_w1_alpha=float(
            env.get("loc_ref_v2_startup_route_w1_alpha", 0.25)
        ),
        loc_ref_v2_startup_route_w2_alpha=float(
            env.get("loc_ref_v2_startup_route_w2_alpha", 0.50)
        ),
        loc_ref_v2_startup_route_w3_alpha=float(
            env.get("loc_ref_v2_startup_route_w3_alpha", 0.75)
        ),
        loc_ref_v2_startup_route_w1_scale=float(
            env.get("loc_ref_v2_startup_route_w1_scale", 0.20)
        ),
        loc_ref_v2_startup_route_w2_scale=float(
            env.get("loc_ref_v2_startup_route_w2_scale", 0.45)
        ),
        loc_ref_v2_startup_route_w3_scale=float(
            env.get("loc_ref_v2_startup_route_w3_scale", 0.75)
        ),
        loc_ref_v2_startup_route_w1_support_y_scale=float(
            env.get("loc_ref_v2_startup_route_w1_support_y_scale", 0.25)
        ),
        loc_ref_v2_startup_route_w2_support_y_scale=float(
            env.get("loc_ref_v2_startup_route_w2_support_y_scale", 0.60)
        ),
        loc_ref_v2_startup_route_w3_support_y_scale=float(
            env.get("loc_ref_v2_startup_route_w3_support_y_scale", 0.85)
        ),
        loc_ref_v2_startup_route_w1_pelvis_roll_scale=float(
            env.get("loc_ref_v2_startup_route_w1_pelvis_roll_scale", 0.45)
        ),
        loc_ref_v2_startup_route_w2_pelvis_roll_scale=float(
            env.get("loc_ref_v2_startup_route_w2_pelvis_roll_scale", 0.75)
        ),
        loc_ref_v2_startup_route_w3_pelvis_roll_scale=float(
            env.get("loc_ref_v2_startup_route_w3_pelvis_roll_scale", 0.90)
        ),
        loc_ref_v2_startup_route_w1_pelvis_pitch_scale=float(
            env.get("loc_ref_v2_startup_route_w1_pelvis_pitch_scale", 0.20)
        ),
        loc_ref_v2_startup_route_w2_pelvis_pitch_scale=float(
            env.get("loc_ref_v2_startup_route_w2_pelvis_pitch_scale", 0.50)
        ),
        loc_ref_v2_startup_route_w3_pelvis_pitch_scale=float(
            env.get("loc_ref_v2_startup_route_w3_pelvis_pitch_scale", 0.80)
        ),
        loc_ref_v2_startup_route_w1_pelvis_height_scale=float(
            env.get("loc_ref_v2_startup_route_w1_pelvis_height_scale", 0.15)
        ),
        loc_ref_v2_startup_route_w2_pelvis_height_scale=float(
            env.get("loc_ref_v2_startup_route_w2_pelvis_height_scale", 0.45)
        ),
        loc_ref_v2_startup_route_w3_pelvis_height_scale=float(
            env.get("loc_ref_v2_startup_route_w3_pelvis_height_scale", 0.80)
        ),
        loc_ref_v2_startup_route_w2_min_pelvis_realization=float(
            env.get("loc_ref_v2_startup_route_w2_min_pelvis_realization", 0.30)
        ),
        loc_ref_v2_startup_route_w3_min_pelvis_realization=float(
            env.get("loc_ref_v2_startup_route_w3_min_pelvis_realization", 0.55)
        ),
        loc_ref_v2_startup_route_w2_pitch_relax=float(
            env.get("loc_ref_v2_startup_route_w2_pitch_relax", 1.25)
        ),
        loc_ref_v2_startup_route_w2_pitch_rate_relax=float(
            env.get("loc_ref_v2_startup_route_w2_pitch_rate_relax", 1.25)
        ),
        loc_ref_v2_support_entry_shaping_window_s=float(
            env.get("loc_ref_v2_support_entry_shaping_window_s", 0.12)
        ),
        loc_ref_v2_support_pelvis_height_offset_m=float(
            env.get("loc_ref_v2_support_pelvis_height_offset_m", 0.00)
        ),
        start_from_support_posture=bool(env.get("start_from_support_posture", False)),
        com_trajectory_enabled=bool(env.get("com_trajectory_enabled", False)),
        com_trajectory_mode=str(env.get("com_trajectory_mode", "linear")),
        com_trajectory_max_behind_m=float(env.get("com_trajectory_max_behind_m", 0.01)),
        com_trajectory_max_ahead_m=float(env.get("com_trajectory_max_ahead_m", 0.03)),
        ankle_pushoff_enabled=bool(env.get("ankle_pushoff_enabled", False)),
        ankle_pushoff_phase_start=float(env.get("ankle_pushoff_phase_start", 0.70)),
        ankle_pushoff_max_rad=float(env.get("ankle_pushoff_max_rad", 0.15)),
        action_mapping_id=str(env.get("action_mapping_id", "pos_target_rad_v1")),
        clock_stride_period_steps=int(env.get("clock_stride_period_steps", 36)),
        clock_phase_gate_width=float(env.get("clock_phase_gate_width", 0.20)),
        controller_stack=controller_stack,
        mpc_residual_scale=float(env.get("mpc_residual_scale", 0.25)),
        mpc_pitch_kp=float(env.get("mpc_pitch_kp", 0.30)),
        mpc_pitch_kd=float(env.get("mpc_pitch_kd", 0.06)),
        mpc_roll_kp=float(env.get("mpc_roll_kp", 0.30)),
        mpc_roll_kd=float(env.get("mpc_roll_kd", 0.06)),
        mpc_height_kp=float(env.get("mpc_height_kp", 0.15)),
        mpc_action_clip=float(env.get("mpc_action_clip", 0.35)),
        mpc_step_trigger_threshold=float(env.get("mpc_step_trigger_threshold", 0.45)),
        teacher_enabled=bool(env.get("teacher_enabled", False)),
        teacher_hard_threshold=float(env.get("teacher_hard_threshold", 0.60)),
        teacher_target_x_min=float(env.get("teacher_target_x_min", -0.10)),
        teacher_target_x_max=float(env.get("teacher_target_x_max", 0.10)),
        teacher_target_y_left_min=float(env.get("teacher_target_y_left_min", 0.02)),
        teacher_target_y_left_max=float(env.get("teacher_target_y_left_max", 0.06)),
        teacher_target_y_right_min=float(env.get("teacher_target_y_right_min", -0.06)),
        teacher_target_y_right_max=float(env.get("teacher_target_y_right_max", -0.02)),
        whole_body_teacher_enabled=bool(env.get("whole_body_teacher_enabled", False)),
        whole_body_teacher_height_target_min=float(
            env.get("whole_body_teacher_height_target_min", 0.39)
        ),
        whole_body_teacher_height_target_max=float(
            env.get("whole_body_teacher_height_target_max", 0.42)
        ),
        whole_body_teacher_height_hard_gate=bool(
            env.get("whole_body_teacher_height_hard_gate", True)
        ),
        whole_body_teacher_com_vel_target=float(
            env.get("whole_body_teacher_com_vel_target", 0.12)
        ),
        whole_body_teacher_com_vel_active_speed_min=float(
            env.get("whole_body_teacher_com_vel_active_speed_min", 0.10)
        ),

        # M2: base controller + residual gating (optional)
        base_ctrl_enabled=bool(env.get("base_ctrl_enabled", False)),
        base_ctrl_pitch_kp=float(env.get("base_ctrl_pitch_kp", 0.25)),
        base_ctrl_pitch_kd=float(env.get("base_ctrl_pitch_kd", 0.05)),
        base_ctrl_roll_kp=float(env.get("base_ctrl_roll_kp", 0.25)),
        base_ctrl_roll_kd=float(env.get("base_ctrl_roll_kd", 0.05)),
        base_ctrl_hip_pitch_gain=float(env.get("base_ctrl_hip_pitch_gain", 1.0)),
        base_ctrl_ankle_pitch_gain=float(env.get("base_ctrl_ankle_pitch_gain", 0.7)),
        base_ctrl_hip_roll_gain=float(env.get("base_ctrl_hip_roll_gain", 1.0)),
        base_ctrl_action_clip=float(env.get("base_ctrl_action_clip", 0.35)),
        residual_scale_min=float(env.get("residual_scale_min", 0.30)),
        residual_scale_max=float(env.get("residual_scale_max", 1.00)),
        residual_gate_power=float(env.get("residual_gate_power", 1.00)),

        # FSM: foot-placement step state-machine (optional)
        fsm_enabled=bool(env.get("fsm_enabled", False)),
        fsm_trigger_threshold=float(env.get("fsm_trigger_threshold", 0.45)),
        fsm_recover_threshold=float(env.get("fsm_recover_threshold", 0.20)),
        fsm_trigger_hold_ticks=int(env.get("fsm_trigger_hold_ticks", 2)),
        fsm_touch_hold_ticks=int(env.get("fsm_touch_hold_ticks", 1)),
        fsm_swing_timeout_ticks=int(env.get("fsm_swing_timeout_ticks", 12)),
        fsm_x_nominal_m=float(env.get("fsm_x_nominal_m", 0.0)),
        fsm_y_nominal_m=float(env.get("fsm_y_nominal_m", 0.115)),
        fsm_k_lat_vel=float(env.get("fsm_k_lat_vel", 0.15)),
        fsm_k_roll=float(env.get("fsm_k_roll", 0.10)),
        fsm_k_pitch=float(env.get("fsm_k_pitch", 0.05)),
        fsm_k_fwd_vel=float(env.get("fsm_k_fwd_vel", 0.05)),
        fsm_x_step_min_m=float(env.get("fsm_x_step_min_m", -0.08)),
        fsm_x_step_max_m=float(env.get("fsm_x_step_max_m", 0.12)),
        fsm_y_step_inner_m=float(env.get("fsm_y_step_inner_m", 0.08)),
        fsm_y_step_outer_m=float(env.get("fsm_y_step_outer_m", 0.20)),
        fsm_step_max_delta_m=float(env.get("fsm_step_max_delta_m", 0.12)),
        fsm_swing_height_m=float(env.get("fsm_swing_height_m", 0.04)),
        fsm_swing_height_need_step_mult=float(env.get("fsm_swing_height_need_step_mult", 0.5)),
        fsm_swing_duration_ticks=int(env.get("fsm_swing_duration_ticks", 10)),
        fsm_swing_x_to_hip_pitch=float(env.get("fsm_swing_x_to_hip_pitch", 0.30)),
        fsm_swing_y_to_hip_roll=float(env.get("fsm_swing_y_to_hip_roll", 0.30)),
        fsm_swing_z_to_knee=float(env.get("fsm_swing_z_to_knee", 0.40)),
        fsm_swing_z_to_ankle=float(env.get("fsm_swing_z_to_ankle", 0.20)),
        fsm_resid_scale_swing=float(env.get("fsm_resid_scale_swing", 0.70)),
        fsm_resid_scale_stance=float(env.get("fsm_resid_scale_stance", 0.85)),
        fsm_resid_scale_recover=float(env.get("fsm_resid_scale_recover", 0.80)),
        fsm_arm_enabled=bool(env.get("fsm_arm_enabled", True)),
        fsm_arm_need_step_threshold=float(env.get("fsm_arm_need_step_threshold", 0.35)),
        fsm_arm_k_roll=float(env.get("fsm_arm_k_roll", 0.10)),
        fsm_arm_k_roll_rate=float(env.get("fsm_arm_k_roll_rate", 0.05)),
        fsm_arm_k_pitch_rate=float(env.get("fsm_arm_k_pitch_rate", 0.03)),
        fsm_arm_max_delta_rad=float(env.get("fsm_arm_max_delta_rad", 0.25)),

        push_enabled=env.get("push_enabled", False),
        push_start_step_min=env.get("push_start_step_min", 20),
        push_start_step_max=env.get("push_start_step_max", 200),
        push_duration_steps=env.get("push_duration_steps", 10),
        push_force_min=env.get("push_force_min", 0.0),
        push_force_max=env.get("push_force_max", 0.0),
        push_body=env.get("push_body", "waist"),
        push_bodies=list(env.get("push_bodies", [])),
        imu_gyro_noise_std=env.get("imu_gyro_noise_std", 0.0),
        imu_quat_noise_deg=env.get("imu_quat_noise_deg", 0.0),
        imu_latency_steps=env.get("imu_latency_steps", 0),
        imu_max_latency_steps=env.get("imu_max_latency_steps", 4),
        # v0.17.3b: domain randomization (disabled by default)
        domain_randomization_enabled=bool(env.get("domain_randomization_enabled", False)),
        domain_rand_friction_range=env.get("domain_rand_friction_range", [0.5, 1.0]),
        domain_rand_mass_scale_range=env.get("domain_rand_mass_scale_range", [0.9, 1.1]),
        domain_rand_kp_scale_range=env.get("domain_rand_kp_scale_range", [0.9, 1.1]),
        domain_rand_frictionloss_scale_range=env.get("domain_rand_frictionloss_scale_range", [0.9, 1.1]),
        domain_rand_joint_offset_rad=float(env.get("domain_rand_joint_offset_rad", 0.03)),
        action_delay_steps=int(env.get("action_delay_steps", 0)),
        # ToddlerBot-aligned additions (walking_training.md Appendix A.1/A.5).
        cmd_resample_steps=int(env.get("cmd_resample_steps", 0)),
        cmd_zero_chance=float(env.get("cmd_zero_chance", 0.0)),
        cmd_turn_chance=float(env.get("cmd_turn_chance", 0.0)),
        cmd_deadzone=float(env.get("cmd_deadzone", 0.0)),
        eval_velocity_cmd=float(env.get("eval_velocity_cmd", -1.0)),
        torso_pitch_soft_min_rad=float(env.get("torso_pitch_soft_min_rad", -0.2)),
        torso_pitch_soft_max_rad=float(env.get("torso_pitch_soft_max_rad", 0.2)),
        torso_roll_soft_min_rad=float(env.get("torso_roll_soft_min_rad", -0.1)),
        torso_roll_soft_max_rad=float(env.get("torso_roll_soft_max_rad", 0.1)),
        min_feet_y_dist=float(env.get("min_feet_y_dist", 0.07)),
        max_feet_y_dist=float(env.get("max_feet_y_dist", 0.13)),
        # v0.20.1 TB-active alignment Phase 2/3 (walking_training.md Appendix B).
        penalty_pose_weights_per_joint=dict(
            env.get("penalty_pose_weights_per_joint", {}) or {}
        ),
        penalty_pose_weight_default=float(
            env.get("penalty_pose_weight_default", 0.0)
        ),
        domain_rand_backlash_range=env.get(
            "domain_rand_backlash_range", [0.0, 0.0]
        ),
        domain_rand_backlash_activation=float(
            env.get("domain_rand_backlash_activation", 0.1)
        ),
    )


def _parse_ppo_config(config: Dict[str, Any]) -> PPOConfig:
    """Parse PPO configuration from YAML dict."""
    ppo = config.get("ppo", {})
    eval_cfg = ppo.get("eval", {})
    rollback_cfg = ppo.get("rollback", {})
    return PPOConfig(
        num_envs=ppo.get("num_envs", 1024),
        rollout_steps=ppo.get("rollout_steps", 128),
        iterations=ppo.get("iterations", 1000),
        learning_rate=ppo.get("learning_rate", 3e-4),
        gamma=ppo.get("gamma", 0.99),
        gae_lambda=ppo.get("gae_lambda", 0.95),
        clip_epsilon=ppo.get("clip_epsilon", 0.2),
        entropy_coef=ppo.get("entropy_coef", 0.01),
        value_loss_coef=ppo.get("value_loss_coef", 0.5),
        epochs=ppo.get("epochs", 4),
        num_minibatches=ppo.get("num_minibatches", 32),
        max_grad_norm=ppo.get("max_grad_norm", 0.5),
        log_interval=ppo.get("log_interval", 10),
        target_kl=ppo.get("target_kl", 0.0),
        kl_early_stop_multiplier=ppo.get("kl_early_stop_multiplier", 1.5),
        kl_lr_backoff_multiplier=ppo.get("kl_lr_backoff_multiplier", 2.0),
        kl_lr_backoff_factor=ppo.get("kl_lr_backoff_factor", 0.5),
        lr_schedule_end_factor=ppo.get("lr_schedule_end_factor", 1.0),
        entropy_schedule_end_factor=ppo.get("entropy_schedule_end_factor", 1.0),
        critic_privileged_enabled=bool(ppo.get("critic_privileged_enabled", False)),
        eval=PPOEvalConfig(
            enabled=eval_cfg.get("enabled", False),
            interval=eval_cfg.get("interval", 0),
            num_envs=eval_cfg.get("num_envs", 0),
            num_steps=eval_cfg.get("num_steps", 0),
            deterministic=eval_cfg.get("deterministic", True),
            seed_offset=eval_cfg.get("seed_offset", 10_000),
        ),
        rollback=PPORollbackConfig(
            enabled=rollback_cfg.get("enabled", False),
            patience=rollback_cfg.get("patience", 2),
            success_rate_drop_threshold=rollback_cfg.get(
                "success_rate_drop_threshold", 0.05
            ),
            lr_factor=rollback_cfg.get("lr_factor", 0.5),
        ),
    )


def _parse_networks_config(config: Dict[str, Any]) -> NetworksConfig:
    """Parse networks configuration from YAML dict."""
    networks = config.get("networks", {})
    actor = networks.get("actor", {})
    critic = networks.get("critic", {})

    return NetworksConfig(
        actor=ActorNetworkConfig(
            hidden_sizes=_parse_list_to_tuple(
                actor.get("hidden_sizes"), (256, 256, 128)
            ),
            activation=actor.get("activation", "elu"),
            log_std_init=actor.get("log_std_init", -1.0),
            min_log_std=actor.get("min_log_std", -5.0),
            max_log_std=actor.get("max_log_std", 2.0),
        ),
        critic=CriticNetworkConfig(
            hidden_sizes=_parse_list_to_tuple(
                critic.get("hidden_sizes"), (256, 256, 128)
            ),
            activation=critic.get("activation", "elu"),
        ),
    )


def _parse_reward_weights_config(config: Dict[str, Any]) -> RewardWeightsConfig:
    """Parse reward weights from YAML dict."""
    # Support both 'rewards:' (old) and 'reward_weights:' (new) format
    rewards = config.get("reward_weights", config.get("rewards", {}))
    return RewardWeightsConfig(
        alive=rewards.get("alive", 0.0),
        tracking_lin_vel=rewards.get("tracking_lin_vel", 2.0),
        lateral_velocity=rewards.get("lateral_velocity", -0.5),
        base_height=rewards.get("base_height", 0.5),
        orientation=rewards.get("orientation", -0.5),
        angular_velocity=rewards.get("angular_velocity", -0.05),
        pitch_rate=rewards.get("pitch_rate", 0.0),
        backward_lean=rewards.get("backward_lean", 0.0),
        negative_velocity=rewards.get("negative_velocity", 0.0),
        height_target=rewards.get("height_target", 0.0),
        height_target_sigma=rewards.get("height_target_sigma", 0.05),
        disturbed_height_target_scale=rewards.get("disturbed_height_target_scale", 1.0),
        disturbed_posture_scale=rewards.get("disturbed_posture_scale", 1.0),
        disturbed_orientation=rewards.get("disturbed_orientation", -0.5),
        height_floor=rewards.get("height_floor", 0.0),
        height_floor_threshold=rewards.get("height_floor_threshold", 0.20),
        height_floor_sigma=rewards.get("height_floor_sigma", 0.03),
        com_velocity_damping=rewards.get("com_velocity_damping", 0.0),
        com_velocity_damping_scale=rewards.get("com_velocity_damping_scale", 1.0),
        collapse_height=rewards.get("collapse_height", -0.2),
        collapse_vz=rewards.get("collapse_vz", -0.2),
        torque=rewards.get("torque", -0.001),
        saturation=rewards.get("saturation", -0.1),
        action_rate=rewards.get("action_rate", -0.01),
        joint_velocity=rewards.get("joint_velocity", -0.001),
        slip=rewards.get("slip", -0.5),
        clearance=rewards.get("clearance", 0.1),
        gait_periodicity=rewards.get("gait_periodicity", 0.0),
        hip_swing=rewards.get("hip_swing", 0.0),
        knee_swing=rewards.get("knee_swing", 0.0),
        hip_swing_min=rewards.get("hip_swing_min", 0.0),
        knee_swing_min=rewards.get("knee_swing_min", 0.0),
        flight_phase_penalty=rewards.get("flight_phase_penalty", 0.0),
        stance_width_penalty=rewards.get("stance_width_penalty", 0.0),
        stance_width_target=rewards.get("stance_width_target", 0.10),
        stance_width_sigma=rewards.get("stance_width_sigma", 0.05),
        forward_velocity_scale=rewards.get("forward_velocity_scale", 4.0),
        velocity_step_gate=rewards.get("velocity_step_gate", 0.0),
        velocity_standing_penalty=rewards.get("velocity_standing_penalty", 0.0),
        velocity_standing_threshold=rewards.get("velocity_standing_threshold", 0.2),
        velocity_cmd_min=rewards.get("velocity_cmd_min", 0.2),
        posture=rewards.get("posture", 0.0),
        posture_sigma=rewards.get("posture_sigma", 0.35),
        posture_gate_pitch=rewards.get("posture_gate_pitch", 0.35),
        posture_gate_roll=rewards.get("posture_gate_roll", 0.35),
        step_event=rewards.get("step_event", 0.0),
        foot_place=rewards.get("foot_place", 0.0),
        foot_place_sigma=rewards.get("foot_place_sigma", 0.12),
        foot_place_k_lat_vel=rewards.get("foot_place_k_lat_vel", 0.15),
        foot_place_k_roll=rewards.get("foot_place_k_roll", 0.10),
        foot_place_k_cmd_vel=rewards.get("foot_place_k_cmd_vel", 0.0),
        foot_place_k_pitch=rewards.get("foot_place_k_pitch", 0.05),
        foot_place_k_fwd_vel=rewards.get("foot_place_k_fwd_vel", 0.05),
        step_length=rewards.get("step_length", 0.0),
        step_length_target_base=rewards.get("step_length_target_base", 0.03),
        step_length_target_scale=rewards.get("step_length_target_scale", 0.25),
        step_length_sigma=rewards.get("step_length_sigma", 0.04),
        step_progress=rewards.get("step_progress", 0.0),
        step_progress_target_scale=rewards.get("step_progress_target_scale", 1.0),
        step_progress_sigma=rewards.get("step_progress_sigma", 0.05),
        arrest_pitch_rate=rewards.get("arrest_pitch_rate", 0.0),
        arrest_capture_error=rewards.get("arrest_capture_error", 0.0),
        post_touchdown_survival=rewards.get("post_touchdown_survival", 0.0),
        arrest_pitch_rate_scale=rewards.get("arrest_pitch_rate_scale", 3.0),
        arrest_capture_error_scale=rewards.get("arrest_capture_error_scale", 0.15),
        m3_pelvis_orientation_tracking=rewards.get("m3_pelvis_orientation_tracking", 0.0),
        m3_pelvis_height_tracking=rewards.get("m3_pelvis_height_tracking", 0.0),
        m3_swing_foot_tracking=rewards.get("m3_swing_foot_tracking", 0.0),
        m3_foothold_consistency=rewards.get("m3_foothold_consistency", 0.0),
        m3_residual_magnitude=rewards.get("m3_residual_magnitude", 0.0),
        m3_excessive_impact=rewards.get("m3_excessive_impact", 0.0),
        m3_pelvis_orientation_sigma=rewards.get("m3_pelvis_orientation_sigma", 0.20),
        m3_pelvis_height_sigma=rewards.get("m3_pelvis_height_sigma", 0.04),
        m3_swing_pos_sigma=rewards.get("m3_swing_pos_sigma", 0.08),
        m3_swing_vel_sigma=rewards.get("m3_swing_vel_sigma", 0.80),
        m3_foothold_sigma=rewards.get("m3_foothold_sigma", 0.10),
        m3_impact_force_threshold=rewards.get("m3_impact_force_threshold", 40.0),
        m3_impact_force_sigma=rewards.get("m3_impact_force_sigma", 20.0),
        dense_progress=rewards.get("dense_progress", 0.0),
        dense_progress_upright_pitch=rewards.get("dense_progress_upright_pitch", 0.25),
        dense_progress_upright_pitch_rate=rewards.get("dense_progress_upright_pitch_rate", 0.90),
        dense_progress_upright_sharpness=rewards.get("dense_progress_upright_sharpness", 10.0),
        forward_upright_gate_strength=rewards.get("forward_upright_gate_strength", 1.0),
        cycle_progress=rewards.get("cycle_progress", 0.0),
        cycle_progress_target_scale=rewards.get("cycle_progress_target_scale", 1.0),
        cycle_progress_sigma=rewards.get("cycle_progress_sigma", 0.08),
        propulsion_gate_step_length_weight=rewards.get("propulsion_gate_step_length_weight", 0.5),
        propulsion_gate_step_progress_weight=rewards.get("propulsion_gate_step_progress_weight", 0.5),
        step_need_pitch=rewards.get("step_need_pitch", 0.35),
        step_need_roll=rewards.get("step_need_roll", 0.35),
        step_need_lat_vel=rewards.get("step_need_lat_vel", 0.30),
        step_need_pitch_rate=rewards.get("step_need_pitch_rate", 1.00),
        teacher_target_step_xy=rewards.get("teacher_target_step_xy", 0.45),
        teacher_target_step_xy_sigma=rewards.get("teacher_target_step_xy_sigma", 0.06),
        teacher_step_required=rewards.get("teacher_step_required", 0.0),
        teacher_swing_foot=rewards.get("teacher_swing_foot", 0.0),
        teacher_recovery_height=rewards.get("teacher_recovery_height", 0.0),
        teacher_recovery_height_sigma=rewards.get("teacher_recovery_height_sigma", 0.03),
        teacher_com_velocity_reduction=rewards.get("teacher_com_velocity_reduction", 0.0),
        teacher_com_velocity_reduction_sigma=rewards.get(
            "teacher_com_velocity_reduction_sigma", 0.08
        ),
        teacher_knee_flex_min=rewards.get("teacher_knee_flex_min", 0.0),
        teacher_knee_flex_target=rewards.get("teacher_knee_flex_target", 0.30),
        teacher_knee_flex_sigma=rewards.get("teacher_knee_flex_sigma", 0.10),
        # v0.20.1 imitation-dominant residual reward family.
        ref_q_track=rewards.get("ref_q_track", 0.0),
        ref_body_quat_track=rewards.get("ref_body_quat_track", 0.0),
        torso_pos_xy=rewards.get("torso_pos_xy", 0.0),
        ref_contact_match=rewards.get("ref_contact_match", 0.0),
        lin_vel_z=rewards.get("lin_vel_z", 0.0),
        ang_vel_xy=rewards.get("ang_vel_xy", 0.0),
        cmd_forward_velocity_track=rewards.get("cmd_forward_velocity_track", 0.0),
        ref_q_track_alpha=rewards.get("ref_q_track_alpha", 1.0),
        ref_body_quat_alpha=rewards.get("ref_body_quat_alpha", 20.0),
        torso_pos_xy_alpha=rewards.get("torso_pos_xy_alpha", 200.0),
        ref_contact_match_sigma=rewards.get("ref_contact_match_sigma", 0.5),
        lin_vel_z_alpha=rewards.get("lin_vel_z_alpha", 200.0),
        ang_vel_xy_alpha=rewards.get("ang_vel_xy_alpha", 0.5),
        cmd_forward_velocity_alpha=rewards.get("cmd_forward_velocity_alpha", 4.0),
        # ToddlerBot-aligned shaping additions (walking_training.md Appendix A.3).
        feet_air_time=rewards.get("feet_air_time", 0.0),
        feet_clearance=rewards.get("feet_clearance", 0.0),
        feet_distance=rewards.get("feet_distance", 0.0),
        torso_pitch_soft=rewards.get("torso_pitch_soft", 0.0),
        torso_roll_soft=rewards.get("torso_roll_soft", 0.0),
        # v0.20.1 TB-active alignment Phase 2 (walking_training.md Appendix B).
        ref_feet_z_track=rewards.get("ref_feet_z_track", 0.0),
        ref_feet_z_track_alpha=rewards.get("ref_feet_z_track_alpha", 1428.6),
        penalty_pose=rewards.get("penalty_pose", 0.0),
        penalty_feet_ori=rewards.get("penalty_feet_ori", 0.0),
    )


def _parse_reward_composition_config(config: Dict[str, Any]) -> RewardCompositionConfig:
    """Parse reward composition from YAML dict."""
    reward = config.get("reward", {})

    # Parse clip values (can be null or [min, max])
    task_clip = reward.get("task_reward_clip")

    return RewardCompositionConfig(
        task_weight=reward.get("task_weight", 1.0),
        task_reward_clip=tuple(task_clip) if task_clip else None,
    )


def _parse_checkpoint_config(config: Dict[str, Any]) -> CheckpointConfig:
    """Parse checkpoint configuration from YAML dict."""
    ckpt = config.get("checkpoints", {})
    return CheckpointConfig(
        dir=ckpt.get("dir", "training/checkpoints"),
        interval=ckpt.get("interval", 50),
    )


def _parse_wandb_config(config: Dict[str, Any]) -> WandbConfig:
    """Parse W&B configuration from YAML dict."""
    wandb = config.get("wandb", {})
    return WandbConfig(
        enabled=wandb.get("enabled", True),
        project=wandb.get("project", "wildrobot"),
        mode=wandb.get("mode", "online"),
        tags=wandb.get("tags", []),
        entity=wandb.get("entity"),
        name=wandb.get("name"),
        log_frequency=wandb.get("log_frequency", 10),
        log_dir=wandb.get("log_dir", "training/wandb"),
    )


def _parse_video_config(config: Dict[str, Any]) -> VideoConfig:
    """Parse video configuration from YAML dict."""
    video = config.get("video", {})
    return VideoConfig(
        enabled=video.get("enabled", False),
        num_videos=video.get("num_videos", 1),
        episode_length=video.get("episode_length", 500),
        render_every=video.get("render_every", 2),
        width=video.get("width", 640),
        height=video.get("height", 480),
        fps=video.get("fps", 25),
        upload_to_wandb=video.get("upload_to_wandb", True),
        output_subdir=video.get("output_subdir", "videos"),
    )


def load_training_config_from_dict(config: Dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from a parsed YAML dict.

    Args:
        config: Parsed YAML configuration dictionary

    Returns:
        TrainingConfig instance (mutable, call freeze() when ready)
    """
    return TrainingConfig(
        # Version
        version=config.get("version", "0.11.1"),
        version_name=config.get("version_name", "PPO"),
        # Global seed
        seed=config.get("seed", 42),
        # Composed configs
        env=_parse_env_config(config),
        ppo=_parse_ppo_config(config),
        networks=_parse_networks_config(config),
        reward_weights=_parse_reward_weights_config(config),
        reward=_parse_reward_composition_config(config),
        checkpoints=_parse_checkpoint_config(config),
        wandb=_parse_wandb_config(config),
        video=_parse_video_config(config),
        # Raw config for additional access
        raw_config=config,
    )


# Cached training config
_training_config: Optional[TrainingConfig] = None


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load training configuration from YAML file.

    The returned config is mutable. After making any modifications,
    call config.freeze() to make it JIT-compatible.

    Args:
        config_path: Path to training config YAML (required)

    Returns:
        TrainingConfig instance (mutable)

    Example:
        config = load_training_config("configs/ppo_walking_v0201_smoke.yaml")
        config.ppo.learning_rate = 1e-4  # Override via CLI
        config.freeze()  # Freeze before training
    """
    global _training_config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _training_config = load_training_config_from_dict(config)
    _training_config.config_path = str(Path(config_path))
    return _training_config


def get_training_config() -> TrainingConfig:
    """Get cached training configuration.

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _training_config is None:
        raise RuntimeError(
            "Training config not loaded. Call load_training_config(path) first."
        )
    return _training_config


def clear_config_cache() -> None:
    """Clear cached configurations (both robot and training)."""
    global _training_config
    _training_config = None
    clear_robot_config_cache()
