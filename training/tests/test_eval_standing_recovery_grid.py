from __future__ import annotations

import numpy as np

from control.kinematics.leg_ik import LegIkConfig, forward_leg_sagittal
from training.eval.eval_standing_recovery_grid import (
    _apply_recovery_condition,
    _conditions,
    _parse_csv_floats,
    _stagger_leg_pitch_values,
    _summarize_rollout,
    _wilson_interval,
)


def test_parse_csv_and_condition_grid() -> None:
    assert _parse_csv_floats("-1, 0, 2.5") == [-1.0, 0.0, 2.5]
    grid = _conditions([-5.0, 5.0], [-0.3, 0.3], [-0.02, 0.0, 0.02])
    assert len(grid) == 12
    assert grid[0].pitch_deg == -5.0
    assert grid[-1].foot_stagger_m == 0.02


def test_stagger_leg_pitch_values_sets_requested_fore_aft_difference() -> None:
    home = np.asarray([0.221289, 0.488909, -0.248439, -0.221849, 0.490533, -0.24789])
    result = np.asarray(_stagger_leg_pitch_values(home, np.float32(0.03)))
    cfg = LegIkConfig()
    left_x, left_z = forward_leg_sagittal(
        hip_pitch_rad=-float(result[0]),
        knee_pitch_rad=float(result[1]),
        config=cfg,
    )
    right_x, right_z = forward_leg_sagittal(
        hip_pitch_rad=float(result[3]),
        knee_pitch_rad=float(result[4]),
        config=cfg,
    )
    # The helper negates the analytic convention so +stagger maps to +world-x
    # in the assembled robot model.
    assert np.isclose(left_x - right_x, -0.03, atol=1e-5)
    assert np.isclose(left_z, right_z, atol=1e-5)
    left_foot_pitch = -result[0] + result[1] + result[2]
    right_foot_pitch = result[3] + result[4] + result[5]
    assert np.isclose(left_foot_pitch, right_foot_pitch, atol=1e-6)


def test_summarize_rollout_marks_corrective_forward_response() -> None:
    rollout = {
        "active": np.ones((3, 2), dtype=bool),
        "done": np.zeros((3, 2), dtype=bool),
        "pitch": np.deg2rad(np.asarray([[5.0, 5.0], [4.0, 6.0], [2.0, 7.0]])),
        "pitch_rate": np.asarray([[0.2, 0.4], [0.0, 0.5], [-0.1, 0.6]]),
        "both_loaded": np.ones((3, 2)),
        "torque_abs_max": np.full((3, 2), 0.4),
        "action_abs_max": np.full((3, 2), 0.2),
    }
    summary = _summarize_rollout(
        rollout,
        initial_pitch_rad=np.deg2rad(np.asarray([5.0, 5.0])),
        initial_pitch_rate_rad_s=np.asarray([0.3, 0.3]),
        ctrl_dt=0.1,
        response_window_s=0.3,
    )
    assert summary["corrective_response"]["count"] == 1
    assert summary["fall"]["count"] == 0
    assert len(summary["per_seed"]) == 2


def test_wilson_interval_contains_observed_fraction() -> None:
    low, high = _wilson_interval(8, 10)
    assert low < 0.8 < high


def test_apply_recovery_condition_supports_native_joint_feedback_reset() -> None:
    import jax

    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config("training/configs/ppo_standing_stabilizer_v0227.yaml")
    load_robot_config(cfg.env.robot_config_path)
    env = WildRobotEnv(config=cfg)
    base = env.reset_for_eval(
        jax.random.PRNGKey(7),
        cmd_override=np.zeros(3, dtype=np.float32),
        perturb_pose=False,
    )

    state = _apply_recovery_condition(
        env,
        base,
        pitch_rad=np.float32(0.04),
        pitch_rate_rad_s=np.float32(-0.1),
        foot_stagger_m=np.float32(0.02),
    )

    assert state.obs.shape == (59,)
    assert np.isfinite(np.asarray(state.obs)).all()
