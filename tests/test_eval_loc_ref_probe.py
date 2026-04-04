from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from training.eval.eval_loc_ref_probe import (
    compare_probe_summaries,
    run_nominal_probe,
)


def test_nominal_probe_runs_and_emits_required_keys() -> None:
    summary = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=32,
        seed=0,
        residual_scale_override=0.0,
        num_envs=1,
    )
    required = [
        "done_step",
        "done_reached",
        "steps_ran",
        "dominant_termination",
        "forward_velocity_mean",
        "tracking_nominal_q_abs_mean",
        "tracking_loc_ref_left_reachable",
        "tracking_loc_ref_right_reachable",
        "debug_m3_swing_pos_error",
        "debug_m3_swing_vel_error",
        "debug_m3_foothold_error",
        "reward_m3_swing_foot_tracking",
        "reward_m3_foothold_consistency",
        "tracking_loc_ref_phase_progress_last",
        "tracking_loc_ref_stance_foot_last",
        "tracking_loc_ref_mode_id_last",
        "tracking_loc_ref_mode_id_mean",
        "tracking_loc_ref_progression_permission_mean",
        "debug_loc_ref_speed_scale_mean",
        "debug_loc_ref_phase_scale_mean",
        "debug_loc_ref_overspeed_mean",
        "debug_loc_ref_support_health_mean",
        "debug_loc_ref_support_instability_mean",
        "debug_loc_ref_nominal_vs_applied_q_l1_mean",
        "debug_loc_ref_swing_x_target_mean",
        "debug_loc_ref_swing_x_actual_mean",
        "debug_loc_ref_swing_x_error_mean",
        "debug_loc_ref_pelvis_pitch_target_mean",
        "debug_loc_ref_root_pitch_mean",
        "debug_loc_ref_root_pitch_rate_mean",
        "debug_loc_ref_swing_x_scale_mean",
        "debug_loc_ref_pelvis_pitch_scale_mean",
        "debug_loc_ref_support_gate_active_mean",
        "debug_loc_ref_hybrid_mode_id_mean",
        "debug_loc_ref_progression_permission_mean",
        "debug_nonfinite_step",
        "debug_nonfinite_count",
        "debug_nonfinite_metric_names_first_step",
        "debug_phase_progress_min",
        "debug_phase_progress_max",
    ]
    for key in required:
        assert key in summary
    assert summary["done_step"] >= 1
    assert summary["steps_ran"] >= summary["done_step"]
    assert isinstance(summary["done_reached"], bool)
    assert summary["debug_nonfinite_step"] >= -1
    assert summary["debug_nonfinite_count"] >= 0


def test_probe_phase_progresses() -> None:
    s = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=24,
        seed=1,
        residual_scale_override=0.0,
        num_envs=1,
    )
    assert 0.0 <= s["tracking_loc_ref_phase_progress_last"] <= 1.0
    assert 0.0 <= s["tracking_loc_ref_phase_progress_mean"] <= 1.0
    assert s["tracking_loc_ref_phase_progress_std"] > 0.0
    assert 0.0 <= s["debug_loc_ref_speed_scale_mean"] <= 1.0
    assert 0.0 <= s["debug_loc_ref_phase_scale_mean"] <= 1.0
    assert 0.0 <= s["debug_loc_ref_support_health_mean"] <= 1.0
    assert s["debug_loc_ref_support_instability_mean"] >= 0.0
    assert 0.0 <= s["debug_loc_ref_swing_x_scale_mean"] <= 1.0
    assert 0.0 <= s["debug_loc_ref_pelvis_pitch_scale_mean"] <= 1.0
    assert 0.0 <= s["debug_loc_ref_support_gate_active_mean"] <= 1.0
    assert s["tracking_loc_ref_mode_id_mean"] >= 0.0
    assert 0.0 <= s["tracking_loc_ref_progression_permission_mean"] <= 1.0
    assert 0.0 <= s["debug_loc_ref_progression_permission_mean"] <= 1.0
    assert 0.0 <= s["debug_phase_progress_min"] <= 1.0
    assert 0.0 <= s["debug_phase_progress_max"] <= 1.0


def test_probe_trace_output_json_contains_required_fields(tmp_path: Path) -> None:
    trace_path = tmp_path / "probe_trace.json"
    summary = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=16,
        seed=2,
        residual_scale_override=0.0,
        num_envs=1,
        trace_output=str(trace_path),
    )
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    required_trace_keys = [
        "step",
        "phase_progress",
        "stance_foot",
        "hybrid_mode_id",
        "progression_permission",
        "forward_velocity",
        "root_pitch",
        "root_pitch_rate",
        "nominal_swing_x_target",
        "actual_swing_x",
        "swing_x_error",
        "nominal_pelvis_pitch_target",
        "support_gate_active",
        "support_health",
        "support_instability",
        "swing_x_scale",
        "pelvis_pitch_scale",
        "nominal_vs_applied_q_gap",
    ]
    for key in required_trace_keys:
        assert key in trace
        assert len(trace[key]) == len(trace["step"])
    assert summary["trace_steps"] == len(trace["step"])


def test_probe_trace_output_npz_contains_required_fields(tmp_path: Path) -> None:
    trace_path = tmp_path / "probe_trace.npz"
    run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=16,
        seed=3,
        residual_scale_override=0.0,
        num_envs=1,
        trace_output=str(trace_path),
    )
    data = np.load(trace_path)
    assert "step" in data.files
    assert "nominal_swing_x_target" in data.files
    assert "nominal_vs_applied_q_gap" in data.files
    assert data["step"].shape[0] == data["nominal_swing_x_target"].shape[0]


def test_probe_ablation_applies_and_reports_name() -> None:
    summary = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=16,
        seed=4,
        residual_scale_override=0.0,
        num_envs=1,
        ablation="zero-pelvis-pitch",
    )
    assert summary["ablation"] == "zero-pelvis-pitch"
    assert abs(summary["debug_loc_ref_pelvis_pitch_target_mean"]) < 1e-6


def test_slow_phase_ablation_short_horizon_completes() -> None:
    summary = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.08,
        horizon=16,
        seed=6,
        residual_scale_override=0.0,
        num_envs=1,
        ablation="slow-phase",
    )
    assert summary["ablation"] == "slow-phase"
    assert summary["steps_ran"] <= 16
    assert summary["done_step"] >= 1
    assert summary["debug_nonfinite_step"] >= -1
    assert summary["debug_nonfinite_count"] >= 0
    assert isinstance(summary["debug_nonfinite_metric_names_first_step"], list)


def test_trace_requires_single_env() -> None:
    try:
        run_nominal_probe(
            config_path="training/configs/ppo_walking_v0193a.yaml",
            forward_cmd=0.08,
            horizon=8,
            seed=5,
            residual_scale_override=0.0,
            num_envs=2,
            trace_output="/tmp/should_not_write.json",
        )
    except ValueError as exc:
        assert "num-envs 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError when trace_output with num_envs != 1")


def test_compare_probe_summaries_reports_expected_keys() -> None:
    baseline = {
        "done_step": 30,
        "dominant_termination": "term/pitch",
        "forward_velocity_mean": 0.20,
        "forward_velocity_last": 0.80,
        "reward_m3_swing_foot_tracking": 1.0e-7,
        "reward_m3_foothold_consistency": 0.02,
        "debug_loc_ref_nominal_vs_applied_q_l1_mean": 0.01,
        "debug_loc_ref_support_gate_active_mean": 0.9,
    }
    candidate = dict(baseline)
    candidate["done_step"] = 40
    candidate["forward_velocity_last"] = 0.60
    cmp_out = compare_probe_summaries(baseline, candidate)
    assert "baseline" in cmp_out and "candidate" in cmp_out and "delta" in cmp_out
    assert cmp_out["delta"]["done_step"] == 10.0
    assert abs(cmp_out["delta"]["forward_velocity_last"] + 0.2) < 1e-9


def test_dominant_termination_none_when_no_done_and_no_terms() -> None:
    summary = run_nominal_probe(
        config_path="training/configs/ppo_walking_v0193a.yaml",
        forward_cmd=0.0,
        horizon=1,
        seed=7,
        residual_scale_override=0.0,
        num_envs=1,
        stop_on_done=False,
    )
    if not summary["done_reached"]:
        term_sum = (
            summary["term_pitch_frac"]
            + summary["term_height_low_frac"]
            + summary["term_roll_frac"]
        )
        if abs(term_sum) < 1e-9:
            assert summary["dominant_termination"] == "none"
