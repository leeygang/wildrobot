from __future__ import annotations

import numpy as np

from training.eval.eval_standing_stabilization import (
    _continuous_eval_step,
    _paired_comparison,
    _parse_controllers,
    _parse_range,
    _parse_suites,
    _sample_initial_conditions,
    _summarize_rollout,
)
from training.eval.standing_orientation import (
    summarize_orientation_rollout,
    summarize_walking_orientation_rollout,
)


def test_continuous_eval_step_disables_only_eval_time_limit() -> None:
    class FakeEnv:
        def step(self, state, action, **kwargs):
            return state, action, kwargs

    result = _continuous_eval_step(
        FakeEnv(), "state", "action", disable_pushes=True
    )
    assert result == (
        "state",
        "action",
        {
            "disable_cmd_resample": True,
            "disable_pushes": True,
            "disable_time_limit": True,
        },
    )


def test_sample_initial_conditions_respects_magnitude_bounds() -> None:
    sampled = _sample_initial_conditions(
        seed=42,
        num_envs=128,
        max_tilt_deg=4.0,
        max_gyro_rad_s=0.35,
        foot_stagger_range_m=(-0.03, 0.04),
    )
    tilt = np.hypot(sampled["roll_rad"], sampled["pitch_rad"])
    gyro = np.hypot(
        sampled["roll_rate_rad_s"], sampled["pitch_rate_rad_s"]
    )
    assert np.max(np.rad2deg(tilt)) <= 4.0
    assert np.max(gyro) <= 0.35
    assert np.min(sampled["foot_stagger_m"]) >= -0.03
    assert np.max(sampled["foot_stagger_m"]) <= 0.04
    repeated = _sample_initial_conditions(
        seed=42,
        num_envs=128,
        max_tilt_deg=4.0,
        max_gyro_rad_s=0.35,
        foot_stagger_range_m=(-0.03, 0.04),
    )
    np.testing.assert_array_equal(sampled["pitch_rad"], repeated["pitch_rad"])


def test_stabilization_summary_uses_final_continuous_window_and_first_episode() -> None:
    shape = (6, 3)
    rollout = {
        "active": np.asarray(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ],
            dtype=bool,
        ),
        "failed": np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        ),
        "tilt_rad": np.deg2rad(
            np.asarray(
                [
                    [4.0, 4.0, 4.0],
                    [3.5, 3.5, 5.0],
                    [2.0, 2.0, 20.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [2.0, 2.0, 0.0],
                ]
            )
        ),
        "yaw_rad": np.deg2rad(np.zeros(shape)),
        "gyro_norm_rad_s": np.full(shape, 0.05),
        "joint_home_max_rad": np.deg2rad(np.full(shape, 6.0)),
        "joint_home_rms_rad": np.deg2rad(np.full(shape, 3.0)),
        "both_loaded": np.ones(shape),
        "within_envelope": np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=bool,
        ),
    }
    initial = {
        "roll_rad": np.zeros(3),
        "pitch_rad": np.zeros(3),
        "roll_rate_rad_s": np.zeros(3),
        "pitch_rate_rad_s": np.zeros(3),
        "foot_stagger_m": np.zeros(3),
        "persistent_pitch_error_rad": np.deg2rad(
            np.asarray([-5.0, 0.0, 7.5])
        ),
    }
    summary = _summarize_rollout(
        rollout,
        ctrl_dt=0.1,
        settle_window_s=0.3,
        settle_tilt_deg=3.0,
        settle_gyro_rad_s=0.1,
        settle_joint_max_deg=8.0,
        settle_joint_rms_deg=4.0,
        initial_conditions=initial,
    )
    assert summary["pass"]["count"] == 1
    assert summary["fall"]["count"] == 1
    assert summary["per_env"][0]["settle_time_s"] == 0.2
    assert summary["per_env"][1]["passed"] is False
    assert summary["per_env"][1]["fail_reasons"] == ["tilt"]
    assert summary["per_env"][2]["fell"] is True
    assert summary["per_env"][2]["peak_tilt_deg"] == 20.0
    assert np.isclose(
        summary["per_env"][0]["persistent_pitch_calibration_error_deg"], -5.0
    )


def test_stabilization_summary_reports_yaw_separately_from_tilt() -> None:
    shape = (2, 1)
    rollout = {
        "active": np.ones(shape, dtype=bool),
        "failed": np.zeros(shape, dtype=bool),
        "tilt_rad": np.zeros(shape),
        "yaw_rad": np.deg2rad(np.asarray([[30.0], [45.0]])),
        "gyro_norm_rad_s": np.zeros(shape),
        "joint_home_max_rad": np.zeros(shape),
        "joint_home_rms_rad": np.zeros(shape),
        "both_loaded": np.ones(shape),
        "within_envelope": np.ones(shape, dtype=bool),
    }
    initial = {
        "roll_rad": np.zeros(1),
        "pitch_rad": np.zeros(1),
        "roll_rate_rad_s": np.zeros(1),
        "pitch_rate_rad_s": np.zeros(1),
        "foot_stagger_m": np.zeros(1),
        "persistent_pitch_error_rad": np.zeros(1),
    }
    summary = _summarize_rollout(
        rollout,
        ctrl_dt=0.1,
        settle_window_s=0.1,
        settle_tilt_deg=3.0,
        settle_gyro_rad_s=0.1,
        settle_joint_max_deg=8.0,
        settle_joint_rms_deg=4.0,
        initial_conditions=initial,
    )
    assert summary["pass"]["count"] == 1
    assert summary["peak_tilt_deg_max"] == 0.0
    assert summary["peak_yaw_error_deg_max"] == 45.0


def test_orientation_rollout_excludes_yaw_and_uses_final_window() -> None:
    half = np.deg2rad(np.asarray([0.0, 10.0, 20.0, 30.0]) / 2.0)
    yaw_quat = np.stack(
        [np.cos(half), np.zeros(4), np.zeros(4), np.sin(half)], axis=-1
    )[:, None, :]
    summary = summarize_orientation_rollout(
        yaw_quat,
        final_window_steps=2,
    )
    assert np.isclose(float(summary["body_tilt_deg_peak"]), 0.0)
    assert np.isclose(float(summary["body_tilt_deg_final_max"]), 0.0)
    assert np.isclose(float(summary["yaw_error_deg_peak"]), 30.0)
    assert np.isclose(float(summary["yaw_error_deg_final_max"]), 30.0)

    roll_half = np.deg2rad(np.asarray([0.0, 20.0, 4.0, 3.0]) / 2.0)
    roll_quat = np.stack(
        [np.cos(roll_half), np.sin(roll_half), np.zeros(4), np.zeros(4)],
        axis=-1,
    )[:, None, :]
    summary = summarize_orientation_rollout(
        roll_quat,
        final_window_steps=2,
    )
    assert np.isclose(float(summary["body_tilt_deg_peak"]), 20.0, atol=1e-3)
    assert np.isclose(
        float(summary["body_tilt_deg_final_max"]), 4.0, atol=1e-3
    )


def test_walking_orientation_separates_survivors_and_fall_tail() -> None:
    pitch_deg = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [2.0, 10.0, 4.0],
            [3.0, 30.0, 6.0],
            [4.0, 80.0, 8.0],
            [5.0, 80.0, 10.0],
        ],
        dtype=np.float32,
    )
    dones = np.zeros_like(pitch_deg)
    truncations = np.zeros_like(pitch_deg)
    dones[5, 0] = 1.0
    truncations[5, 0] = 1.0
    dones[3, 1] = 1.0

    summary = summarize_walking_orientation_rollout(
        np.zeros_like(pitch_deg),
        np.deg2rad(pitch_deg),
        dones,
        truncations,
        ctrl_dt=0.1,
        stable_start_s=0.2,
        pre_fall_window_s=0.2,
    )

    assert float(summary["walking_survivor_env_count"]) == 2.0
    assert float(summary["walking_fall_env_count"]) == 1.0
    assert np.isclose(float(summary["walking_fall_env_frac"]), 1.0 / 3.0)
    assert np.isclose(
        float(summary["walking_stable_body_tilt_deg_mean"]), 5.25, atol=1e-3
    )
    assert np.isclose(
        float(summary["walking_stable_body_tilt_deg_p95"]), 10.0, atol=1e-3
    )
    assert np.isclose(
        float(summary["walking_stable_body_tilt_deg_max"]), 10.0, atol=1e-3
    )
    assert np.isclose(
        float(summary["walking_survivor_final_body_tilt_deg_max"]),
        10.0,
        atol=1e-3,
    )
    assert np.isclose(
        float(summary["walking_pre_fall_body_tilt_deg_max"]), 10.0, atol=1e-3
    )
    assert np.isclose(
        float(summary["walking_fall_terminal_body_tilt_deg_max"]),
        30.0,
        atol=1e-3,
    )
    assert np.isclose(float(summary["walking_time_to_fall_s_min"]), 0.4)
    assert np.isclose(float(summary["body_tilt_deg_peak"]), 80.0, atol=1e-3)


def test_paired_comparison_counts_policy_and_home_wins() -> None:
    policy = {
        "per_env": [
            {"passed": True, "peak_tilt_deg": 4.0},
            {"passed": False, "peak_tilt_deg": 6.0},
        ]
    }
    home = {
        "per_env": [
            {"passed": False, "peak_tilt_deg": 5.0},
            {"passed": True, "peak_tilt_deg": 5.0},
        ]
    }
    comparison = _paired_comparison(policy, home)
    assert comparison["policy_pass_home_fail"]["count"] == 1
    assert comparison["home_pass_policy_fail"]["count"] == 1
    assert comparison["policy_lower_peak_tilt"]["count"] == 1


def test_cli_parsers() -> None:
    assert _parse_suites("push,clean,push") == ("push", "clean")
    assert _parse_controllers("policy,home,policy") == ("policy", "home")
    assert _parse_range("-0.04, 0.03") == (-0.04, 0.03)
