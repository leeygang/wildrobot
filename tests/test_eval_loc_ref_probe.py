from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from training.eval.eval_loc_ref_probe import run_nominal_probe


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
    ]
    for key in required:
        assert key in summary
    assert summary["done_step"] >= 1


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
