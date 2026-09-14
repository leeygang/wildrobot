from pathlib import Path

import pytest

from training.eval.sweep_actuator_capacity import (
    _minimum_limit,
    _positive_finite_values,
    _summarize_case,
)


def _passing_metrics(*, saturation: float = 0.04) -> dict[str, float]:
    return {
        "forward_velocity": 0.10,
        "episode_length": 100.0,
        "tracking/cmd_vs_achieved_forward": 0.0,
        "tracking/step_length_touchdown_event_m": 0.05,
        "walking_stable_sample_count": 100.0,
        "walking_fall_env_count": 0.0,
        "walking_fall_env_frac": 0.0,
        "walking_stable_body_tilt_deg_mean": 3.0,
        "walking_stable_body_tilt_deg_max": 8.0,
        "walking_survivor_final_body_tilt_deg_max": 5.0,
        "walking_stable_max_actuator_torque_sat_frac": saturation,
        "walking_stable_torque/left_hip_roll/sat_frac": saturation,
        "walking_stable_torque/right_hip_roll/sat_frac": 0.02,
        "walking_stable_torque/left_hip_roll/rms_nm": 2.0,
        "walking_stable_torque/right_hip_roll/rms_nm": 1.5,
        "walking_stable_torque/left_knee_pitch/rms_nm": 1.0,
        "walking_stable_torque/right_knee_pitch/rms_nm": 1.0,
    }


def test_summarize_case_uses_shared_safety_and_tracking_gates() -> None:
    case = _summarize_case(
        _passing_metrics(),
        torque_limit_nm=5.2,
        velocity_cmd_m_s=0.10,
        num_steps=100,
        metrics_path=Path("metrics.json"),
    )

    assert case["safety_passed"] is True
    assert case["full_gate_passed"] is True
    assert case["worst_stable_saturation_joint"] == "left_hip_roll"
    hip_roll = case["stable_bilateral_torque_rms"]["hip_roll"]
    assert hip_roll["sum_nm"] == pytest.approx(3.5)
    assert hip_roll["relative_imbalance"] == pytest.approx(0.25)


def test_minimum_limit_requires_every_command_to_pass() -> None:
    cases = [
        {"torque_limit_nm": 4.4, "velocity_cmd_m_s": 0.07, "safety_passed": True},
        {"torque_limit_nm": 4.4, "velocity_cmd_m_s": 0.13, "safety_passed": False},
        {"torque_limit_nm": 5.2, "velocity_cmd_m_s": 0.07, "safety_passed": True},
        {"torque_limit_nm": 5.2, "velocity_cmd_m_s": 0.13, "safety_passed": True},
    ]

    assert _minimum_limit(cases, (0.07, 0.13), "safety_passed") == pytest.approx(5.2)


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")]])
def test_positive_finite_values_rejects_invalid_inputs(values) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        _positive_finite_values(values, "test-values")
