from __future__ import annotations

import pytest

from wildrobot.agents.evaluate_walking_candidate import (
    _aggregate_results,
    _gate_metrics,
    _source_failure_indices,
    _teacher_recoverability_aggregate,
)


def test_gate_metrics_maps_eval_policy_names() -> None:
    mapped = _gate_metrics(
        {
            "episode_length": 1000.0,
            "forward_velocity": 0.1,
            "tracking/cmd_vs_achieved_forward": 0.03,
            "tracking/step_length_touchdown_event_m": 0.08,
        }
    )

    assert mapped["mean_episode_length"] == 1000.0
    assert mapped["cmd_vs_achieved_forward"] == 0.03
    assert mapped["step_length_touchdown_event_m"] == 0.08


def test_aggregate_results_requires_every_seed_to_pass() -> None:
    results = [
        {
            "passed": True,
            "fail_reasons": [],
            "eval_metrics": {
                "walking_fall_env_count": 0,
                "walking_stable_body_tilt_deg_max": 8.0,
                "walking_survivor_final_body_tilt_deg_max": 4.0,
                "walking_stable_max_actuator_torque_sat_frac": 0.04,
            },
        },
        {
            "passed": False,
            "fail_reasons": ["walking_fall_env_frac"],
            "eval_metrics": {
                "walking_fall_env_count": 1,
                "walking_stable_body_tilt_deg_max": 9.0,
                "walking_survivor_final_body_tilt_deg_max": 5.0,
                "walking_stable_max_actuator_torque_sat_frac": 0.03,
            },
        },
    ]

    aggregate = _aggregate_results(results, num_envs=64)

    assert aggregate["passed"] is False
    assert aggregate["total_envs"] == 128
    assert aggregate["total_falls"] == 1
    assert aggregate["fail_reasons"] == ["walking_fall_env_frac"]
    assert aggregate["worst_stable_tilt_deg"] == 9.0


def test_zero_failure_confidence_bound_uses_all_environments() -> None:
    result = {
        "passed": True,
        "fail_reasons": [],
        "eval_metrics": {"walking_fall_env_count": 0},
    }

    aggregate = _aggregate_results([result] * 4, num_envs=64)

    assert aggregate["passed"] is True
    assert aggregate["total_envs"] == 256
    assert aggregate["zero_failure_probability_upper_95"] == pytest.approx(
        0.011633, abs=1e-6
    )


def test_teacher_recoverability_requires_zero_teacher_falls(tmp_path) -> None:
    source_report = tmp_path / "source.json"
    source_report.write_text(
        '{"seed_results":[{"seed":31000,"eval_metrics":{"failure_cases":'
        '[{"env_index":7},{"env_index":41}]}}]}'
    )
    source = _source_failure_indices(source_report)
    aggregate = {
        "passed": False,
        "total_falls": 1,
        "fail_reasons": ["walking_fall_env_frac"],
    }
    seed_results = [
        {
            "seed": 31000,
            "eval_metrics": {"failure_cases": [{"env_index": 12}]},
        }
    ]

    result = _teacher_recoverability_aggregate(
        aggregate, seed_results, source, [31000]
    )

    assert result["teacher_recoverability_passed"] is False
    assert result["source_failure_env_count"] == 2
    assert result["unrecovered_source_failure_env_indices"] == {31000: []}


def test_teacher_recoverability_passes_only_when_source_failures_exist() -> None:
    result = _teacher_recoverability_aggregate(
        {"passed": False, "total_falls": 0, "fail_reasons": []},
        [{"seed": 31000, "eval_metrics": {"failure_cases": []}}],
        {31000: {7, 41}},
        [31000],
    )

    assert result["teacher_recoverability_passed"] is True
    assert result["passed"] is True
