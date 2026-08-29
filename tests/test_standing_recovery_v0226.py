from __future__ import annotations

import numpy as np


def test_v0226_contract_and_training_recipe() -> None:
    from assets.robot_config import get_robot_config, load_robot_config
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    cfg = load_training_config("training/configs/ppo_standing_recovery_v0226.yaml")
    load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=get_robot_config()
    )

    assert cfg.version == "0.22.6"
    assert spec.observation.layout_id == "wr_obs_v10_standing_recovery"
    assert spec.model.obs_dim == 66
    assert spec.model.action_dim == 17
    fields = {field.name: field.size for field in spec.observation.layout}
    assert "foot_switches" not in fields
    assert fields["recovery_foothold_xy"] == 2
    assert cfg.env.reset_torso_pitch_rate_range == [-0.6, 0.6]
    assert cfg.env.reset_foot_stagger_range_m == [-0.04, 0.04]
    assert cfg.reward_weights.feet_phase == 1.0
    assert cfg.reward_weights.recovery_squat == 4.0
    assert cfg.reward_weights.unnecessary_step < 0.0


def test_v0226_runtime_metadata_contains_recovery_planner() -> None:
    from assets.robot_config import get_robot_config, load_robot_config
    from training.configs.training_config import load_training_config
    from training.exports.runtime_metadata import build_runtime_policy_config
    from training.policy_spec_utils import build_policy_spec_from_training_config

    cfg = load_training_config("training/configs/ppo_standing_recovery_v0226.yaml")
    load_robot_config(cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=get_robot_config()
    )
    metadata = build_runtime_policy_config(env=vars(cfg.env), spec=spec)

    assert metadata["policy_role"] == "standing"
    assert metadata["recovery"]["enabled"] is True
    assert metadata["recovery"]["max_step_m"] == 0.10
    assert metadata["recovery"]["swing_duration_steps"] == 20


def test_v10_observation_numpy_jax_parity() -> None:
    import jax.numpy as jnp

    from policy_contract.jax.obs import build_observation_from_components as build_jax
    from policy_contract.numpy.obs import build_observation_from_components as build_numpy
    from policy_contract.spec_builder import build_policy_spec

    specs = [
        {"name": name, "range": [-1.0, 1.0], "max_velocity": 10.0}
        for name in ("left_hip_pitch", "right_hip_pitch")
    ]
    spec = build_policy_spec(
        robot_name="test",
        actuated_joint_specs=specs,
        action_filter_alpha=0.0,
        layout_id="wr_obs_v10_standing_recovery",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0, 0.0],
    )
    values = dict(
        spec=spec,
        gravity_local=np.array([0.1, -0.2, -0.97], dtype=np.float32),
        angvel_heading_local=np.array([0.2, -0.3, 0.0], dtype=np.float32),
        joint_pos_normalized=np.array([0.1, -0.1], dtype=np.float32),
        joint_vel_normalized=np.array([0.2, -0.2], dtype=np.float32),
        foot_switches=np.ones(4, dtype=np.float32),
        prev_action=np.array([0.3, -0.3], dtype=np.float32),
        velocity_cmd=np.zeros(1, dtype=np.float32),
        standing_recovery_command=np.array(
            [1.0, 1.0, 0.0, 0.5, 0.866, 0.7, -0.2], dtype=np.float32
        ),
    )
    obs_np = build_numpy(**values)
    obs_jax = np.asarray(
        build_jax(**{key: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                     for key, value in values.items()})
    )
    assert obs_np.shape == (21,)
    np.testing.assert_allclose(obs_np, obs_jax, atol=1e-6)
    np.testing.assert_allclose(obs_np[-8:-1], values["standing_recovery_command"])


def test_recovery_planner_steps_then_returns_to_squat_hold() -> None:
    from runtime.wr_runtime.control.standing_recovery_planner import (
        HOLD,
        SETTLE,
        SWING,
        StandingRecoveryPlannerConfig,
        StandingRecoveryPlannerState,
        advance_recovery_planner,
        encode_recovery_command,
    )

    cfg = StandingRecoveryPlannerConfig(
        enabled=True,
        swing_duration_steps=4,
        settle_min_steps=3,
        settle_max_steps=8,
    )
    state = advance_recovery_planner(
        StandingRecoveryPlannerState(),
        cfg,
        roll_rad=0.0,
        pitch_rad=0.12,
        roll_rate_rad_s=0.0,
        pitch_rate_rad_s=0.3,
        left_foot_x_m=-0.02,
        right_foot_x_m=0.02,
        left_loaded=True,
        right_loaded=True,
    )
    assert state.phase == SWING
    assert state.swing_foot == 0
    assert state.target_xy_m[0] > 0.0
    assert encode_recovery_command(state, cfg)[0] == 1.0

    for _ in range(cfg.swing_duration_steps):
        state = advance_recovery_planner(
            state,
            cfg,
            roll_rad=0.02,
            pitch_rad=0.04,
            roll_rate_rad_s=0.02,
            pitch_rate_rad_s=0.05,
            left_foot_x_m=0.05,
            right_foot_x_m=0.02,
            left_loaded=True,
            right_loaded=True,
        )
    assert state.phase == SETTLE

    for _ in range(cfg.settle_min_steps):
        state = advance_recovery_planner(
            state,
            cfg,
            roll_rad=0.01,
            pitch_rad=0.01,
            roll_rate_rad_s=0.01,
            pitch_rate_rad_s=0.01,
            left_foot_x_m=0.05,
            right_foot_x_m=0.02,
            left_loaded=True,
            right_loaded=True,
        )
    assert state.phase == HOLD
    np.testing.assert_allclose(
        encode_recovery_command(state, cfg),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_runtime_foot_estimate_uses_world_forward_sign() -> None:
    from runtime.wr_runtime.control.standing_policy_runner import (
        _estimate_sagittal_foot_x,
    )

    home = {
        "left_hip_pitch": 0.221289,
        "left_knee_pitch": 0.488909,
        "right_hip_pitch": -0.221849,
        "right_knee_pitch": 0.490533,
    }
    home_left_x, home_right_x = _estimate_sagittal_foot_x(home)
    advanced_left_x, _ = _estimate_sagittal_foot_x(
        {**home, "left_hip_pitch": home["left_hip_pitch"] + 0.1}
    )
    _, retracted_right_x = _estimate_sagittal_foot_x(
        {**home, "right_hip_pitch": home["right_hip_pitch"] + 0.1}
    )

    assert abs(home_left_x - home_right_x) < 0.001
    assert advanced_left_x > home_left_x
    assert retracted_right_x < home_right_x


def test_recovery_promotion_gate_requires_touchdown_and_squat_return() -> None:
    from training.core.post_training_eval import deterministic_standing_eval_gate

    metrics = {
        "mean_episode_length": 1000.0,
        "left_loaded": 0.98,
        "right_loaded": 0.98,
        "both_loaded": 0.96,
        "load_imbalance": 0.10,
        "body_tilt_deg": 3.0,
        "body_tilt_deg_peak": 10.0,
        "body_tilt_deg_final_max": 4.0,
        "torque_sat_frac": 0.01,
        "action_sat_frac": 0.01,
        "recovery_step_rate": 0.75,
        "recovery_touchdown_given_step_rate": 0.95,
        "recovery_squat_return_given_step_rate": 0.85,
        "recovery_unnecessary_liftoff_rate": 0.0,
    }
    passed = deterministic_standing_eval_gate(
        metrics, eval_num_steps=1000, require_recovery=True
    )
    assert passed.passed

    failed = deterministic_standing_eval_gate(
        {**metrics, "recovery_squat_return_given_step_rate": 0.5},
        eval_num_steps=1000,
        require_recovery=True,
    )
    assert not failed.passed
    assert failed.gates["recovery_squat_return_given_step"] is False
