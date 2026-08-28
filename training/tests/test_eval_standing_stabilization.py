from __future__ import annotations

import numpy as np

from training.eval.eval_standing_stabilization import (
    _continuous_eval_step,
    _paired_comparison,
    _parse_range,
    _parse_suites,
    _sample_initial_conditions,
    _summarize_rollout,
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
    assert _parse_range("-0.04, 0.03") == (-0.04, 0.03)
