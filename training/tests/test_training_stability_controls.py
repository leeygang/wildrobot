from __future__ import annotations

import inspect
from pathlib import Path

from training.configs.training_config import (
    load_training_config,
    load_training_config_from_dict,
)
from training.core.post_training_eval import (
    CheckpointMetricCandidate,
    deterministic_eval_gate,
    rank_checkpoint_candidates,
)
from training.core.training_loop import (
    _effective_ppo_epochs,
    _format_hhmmss,
    _is_eval_better,
    _linear_schedule_factor,
    _should_fire_callback,
    _should_trigger_rollback,
)


def test_linear_schedule_factor_bounds():
    assert _linear_schedule_factor(0.0, 0.5) == 1.0
    assert _linear_schedule_factor(1.0, 0.5) == 0.5
    assert _linear_schedule_factor(2.0, 0.5) == 0.5


def test_effective_ppo_epochs_uses_kl_guardrail():
    assert _effective_ppo_epochs(4, 0.01, 0.02, 1.5) == 4
    assert _effective_ppo_epochs(4, 0.05, 0.02, 1.5) == 1
    assert _effective_ppo_epochs(4, 0.05, 0.0, 1.5) == 4


def test_eval_comparison_tie_breaks_on_episode_length():
    assert _is_eval_better(0.8, 300.0, 0.7, 400.0)
    assert _is_eval_better(0.8, 301.0, 0.8, 300.0)
    assert not _is_eval_better(0.8, 299.0, 0.8, 300.0)


def test_should_trigger_rollback_uses_success_drop_threshold():
    assert _should_trigger_rollback(0.70, 0.80, 0.05)
    assert not _should_trigger_rollback(0.76, 0.80, 0.05)


def test_format_hhmmss_formats_elapsed_time():
    assert _format_hhmmss(0) == "00:00:00"
    assert _format_hhmmss(59) == "00:00:59"
    assert _format_hhmmss(60) == "00:01:00"
    assert _format_hhmmss(3661) == "01:01:01"
    assert _format_hhmmss(-2) == "00:00:00"


def test_should_fire_callback_covers_log_and_checkpoint_boundaries() -> None:
    assert _should_fire_callback(iteration=1, log_interval=50, checkpoint_interval=200) is True
    assert _should_fire_callback(iteration=10, log_interval=10, checkpoint_interval=100) is True
    assert _should_fire_callback(iteration=20, log_interval=7, checkpoint_interval=10) is True
    assert _should_fire_callback(iteration=9, log_interval=10, checkpoint_interval=20) is False


def _candidate(
    *,
    path: str,
    iteration: int,
    reward: float,
    forward_velocity: float | None = None,
    cmd_err: float | None = None,
    step_len: float | None = None,
    ep_len: float | None = None,
    cmd_ratio: float | None = None,
) -> CheckpointMetricCandidate:
    metrics: dict[str, float] = {"episode_reward": reward}
    if forward_velocity is not None:
        metrics["forward_velocity"] = forward_velocity
    if cmd_err is not None:
        metrics["tracking/cmd_vs_achieved_forward"] = cmd_err
    if step_len is not None:
        metrics["tracking/step_length_touchdown_event_m"] = step_len
    if ep_len is not None:
        metrics["episode_length"] = ep_len
    if cmd_ratio is not None:
        metrics["tracking/forward_velocity_cmd_ratio"] = cmd_ratio
    return CheckpointMetricCandidate(
        checkpoint_path=path,
        iteration=iteration,
        total_steps=iteration * 1000,
        metrics=metrics,
    )


def test_rank_checkpoint_candidates_prefers_walking_over_high_reward_basin() -> None:
    candidates = [
        _candidate(
            path="checkpoint_10.pkl",
            iteration=10,
            reward=380.0,  # high reward standing/basin exploit
            forward_velocity=0.01,
            cmd_err=0.12,
            step_len=0.0002,
            ep_len=500.0,
            cmd_ratio=0.08,
        ),
        _candidate(
            path="checkpoint_20.pkl",
            iteration=20,
            reward=220.0,
            forward_velocity=0.11,
            cmd_err=0.05,
            step_len=0.034,
            ep_len=500.0,
            cmd_ratio=0.95,
        ),
        _candidate(
            path="checkpoint_30.pkl",
            iteration=30,
            reward=210.0,
            forward_velocity=0.10,
            cmd_err=0.04,
            step_len=0.033,
            ep_len=498.0,
            cmd_ratio=0.90,
        ),
        _candidate(
            path="checkpoint_40.pkl",
            iteration=40,
            reward=205.0,
            forward_velocity=0.09,
            cmd_err=0.06,
            step_len=0.031,
            ep_len=490.0,
            cmd_ratio=0.85,
        ),
    ]

    ranked, used_filter_fallback = rank_checkpoint_candidates(candidates, top_k=3)
    assert used_filter_fallback is False
    ranked_paths = [c.checkpoint_path for c in ranked]
    assert ranked_paths == ["checkpoint_20.pkl", "checkpoint_30.pkl", "checkpoint_40.pkl"]


def test_rank_checkpoint_candidates_uses_reward_only_as_tiebreaker() -> None:
    candidates = [
        _candidate(
            path="checkpoint_a.pkl",
            iteration=100,
            reward=180.0,
            forward_velocity=0.10,
            cmd_err=0.05,
            step_len=0.032,
            ep_len=500.0,
            cmd_ratio=0.90,
        ),
        _candidate(
            path="checkpoint_b.pkl",
            iteration=110,
            reward=280.0,
            forward_velocity=0.10,
            cmd_err=0.05,
            step_len=0.032,
            ep_len=500.0,
            cmd_ratio=0.90,
        ),
    ]
    ranked, used_filter_fallback = rank_checkpoint_candidates(candidates, top_k=2)
    assert used_filter_fallback is False
    assert ranked[0].checkpoint_path == "checkpoint_b.pkl"
    assert ranked[1].checkpoint_path == "checkpoint_a.pkl"


def test_rank_checkpoint_candidates_missing_metrics_falls_back_predictably() -> None:
    candidates = [
        _candidate(path="checkpoint_sparse_a.pkl", iteration=10, reward=120.0),
        _candidate(path="checkpoint_sparse_b.pkl", iteration=20, reward=260.0),
    ]
    ranked, used_filter_fallback = rank_checkpoint_candidates(candidates, top_k=2)
    assert used_filter_fallback is False
    assert all(candidate.used_reward_fallback for candidate in ranked)
    assert ranked[0].checkpoint_path == "checkpoint_sparse_b.pkl"


def test_deterministic_eval_gate_pass_and_fail_cases() -> None:
    decision_pass = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert decision_pass.passed is True
    assert decision_pass.step_metric_available is True
    assert decision_pass.ratio_gate_applied is True

    decision_vel_fail = deterministic_eval_gate(
        {
            "forward_velocity": 0.04,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert decision_vel_fail.passed is False
    assert decision_vel_fail.gates["forward_velocity"] is False

    decision_cmd_fail = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.12,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert decision_cmd_fail.passed is False
    assert decision_cmd_fail.gates["cmd_vs_achieved_forward"] is False

    decision_step_fail = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.010,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert decision_step_fail.passed is False
    assert decision_step_fail.gates["step_length_touchdown_event_m"] is False

    decision_ratio_fail = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=0.20,
    )
    assert decision_ratio_fail.passed is False
    assert decision_ratio_fail.gates["forward_velocity_cmd_ratio"] is False

    decision_ratio_skipped = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=-1.0,
    )
    assert decision_ratio_skipped.passed is True
    assert decision_ratio_skipped.ratio_gate_applied is False


def test_post_training_eval_defaults_are_backward_compatible() -> None:
    cfg = load_training_config_from_dict({"ppo": {"eval": {"enabled": False}}})
    assert cfg.ppo.eval.enabled is False
    assert cfg.ppo.eval.post_training_enabled is False
    assert cfg.ppo.eval.post_training_top_k == 3
    assert cfg.ppo.eval.post_training_num_envs == 8
    assert cfg.ppo.eval.post_training_num_steps == 500
    assert cfg.ppo.eval.post_training_checkpoint_label == "eval_promoted"


def test_post_training_eval_can_enable_without_periodic_eval() -> None:
    cfg = load_training_config_from_dict(
        {
            "ppo": {
                "eval": {
                    "enabled": False,
                    "post_training_enabled": True,
                    "post_training_top_k": 3,
                    "post_training_num_envs": 8,
                    "post_training_num_steps": 500,
                    "post_training_checkpoint_label": "eval_promoted",
                }
            }
        }
    )
    assert cfg.ppo.eval.enabled is False
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_top_k == 3
    assert cfg.ppo.eval.post_training_num_envs == 8
    assert cfg.ppo.eval.post_training_num_steps == 500


def test_smoke12b_enables_post_training_without_periodic_eval() -> None:
    smoke12b = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "configs"
        / "ppo_walking_v0201_smoke12b.yaml"
    )
    cfg = load_training_config(str(smoke12b))
    assert cfg.ppo.eval.enabled is False
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_top_k == 3
    assert cfg.ppo.eval.post_training_num_envs == 8
    assert cfg.ppo.eval.post_training_num_steps == 500
    assert cfg.ppo.eval.post_training_checkpoint_label == "eval_promoted"


def test_parse_new_ppo_stability_fields():
    cfg = load_training_config_from_dict(
        {
            "ppo": {
                "target_kl": 0.02,
                "kl_early_stop_multiplier": 1.7,
                "kl_lr_backoff_multiplier": 2.1,
                "kl_lr_backoff_factor": 0.4,
                "lr_schedule_end_factor": 0.3,
                "entropy_schedule_end_factor": 0.2,
                "eval": {
                    "enabled": True,
                    "interval": 5,
                    "num_envs": 32,
                    "num_steps": 400,
                    "deterministic": True,
                    "seed_offset": 123,
                    "post_training_enabled": True,
                    "post_training_top_k": 5,
                    "post_training_num_envs": 12,
                    "post_training_num_steps": 250,
                    "post_training_checkpoint_label": "candidate_promoted",
                },
                "rollback": {
                    "enabled": True,
                    "patience": 3,
                    "success_rate_drop_threshold": 0.07,
                    "lr_factor": 0.6,
                },
            }
        }
    )

    assert cfg.ppo.target_kl == 0.02
    assert cfg.ppo.kl_early_stop_multiplier == 1.7
    assert cfg.ppo.kl_lr_backoff_multiplier == 2.1
    assert cfg.ppo.kl_lr_backoff_factor == 0.4
    assert cfg.ppo.lr_schedule_end_factor == 0.3
    assert cfg.ppo.entropy_schedule_end_factor == 0.2
    assert cfg.ppo.eval.enabled
    assert cfg.ppo.eval.interval == 5
    assert cfg.ppo.eval.num_envs == 32
    assert cfg.ppo.eval.num_steps == 400
    assert cfg.ppo.eval.seed_offset == 123
    assert cfg.ppo.eval.post_training_enabled is True
    assert cfg.ppo.eval.post_training_top_k == 5
    assert cfg.ppo.eval.post_training_num_envs == 12
    assert cfg.ppo.eval.post_training_num_steps == 250
    assert cfg.ppo.eval.post_training_checkpoint_label == "candidate_promoted"
    assert cfg.ppo.rollback.enabled
    assert cfg.ppo.rollback.patience == 3
    assert cfg.ppo.rollback.success_rate_drop_threshold == 0.07
    assert cfg.ppo.rollback.lr_factor == 0.6


def test_training_loop_has_no_online_promotion_eval_path() -> None:
    from training.core import training_loop

    train_source = inspect.getsource(training_loop.train)
    assert "promotion_eval" not in train_source
    assert "run_promotion_eval" not in train_source


# =============================================================================
# Smoke15 — eval-authority enforcement + deploy-facing soft signals
# =============================================================================


def test_deterministic_eval_gate_exposes_soft_signals_for_deploy_metrics() -> None:
    """Smoke15 — deterministic eval gate must expose
    ``soft_signals`` for the lateral / yaw / world-y deploy metrics.
    Healthy values clear the cap; pathological values trip the soft
    flag (report-only — passed remains True if hard gates pass)."""
    healthy = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
            "lateral_velocity_abs": 0.02,
            "yaw_drift_signed_rad": 0.05,
            "world_y_drift_signed_m": 0.02,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert healthy.passed is True
    assert healthy.soft_signals == {
        "lateral_velocity_abs": True,
        "yaw_drift_signed_rad": True,
        "world_y_drift_signed_m": True,
    }

    # Smoke14 tail values: lateral 0.20 m/s — well above the 0.10
    # m/s cap; yaw 0.6 rad — above 0.4 cap; world_y 0.5 m — above
    # 0.30 m cap.  Soft signals should ALL flip to False but the
    # hard ``passed`` flag still reflects only G4 gates.
    pathological = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
            "lateral_velocity_abs": 0.20,
            "yaw_drift_signed_rad": 0.60,
            "world_y_drift_signed_m": 0.50,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert pathological.passed is True, (
        "soft signals must NOT block promotion; only hard G4 gates do"
    )
    assert pathological.soft_signals["lateral_velocity_abs"] is False
    assert pathological.soft_signals["yaw_drift_signed_rad"] is False
    assert pathological.soft_signals["world_y_drift_signed_m"] is False


def test_deterministic_eval_gate_soft_signals_default_ok_when_missing() -> None:
    """Old-log back-compat: when the eval payload doesn't carry the
    smoke15 deploy fields, the corresponding soft_signals must
    default to True (signal not flagged, not unavailable)."""
    decision = deterministic_eval_gate(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        },
        eval_velocity_cmd=0.1333333333,
    )
    assert decision.passed is True
    assert decision.soft_signals["lateral_velocity_abs"] is True
    assert decision.soft_signals["yaw_drift_signed_rad"] is True
    assert decision.soft_signals["world_y_drift_signed_m"] is True


def test_walking_config_without_eval_raises_at_train_loop_startup() -> None:
    """Smoke15 walking-eval-authority guard: a walking config (one
    with non-zero cmd_forward_velocity_track) MUST have either
    mid-training or post-training eval enabled.  Otherwise the
    training loop refuses to start so a standing local minimum
    (smoke14 pattern) cannot misreport via env/episode_length.

    Validates the structural check by inspecting the source.  A full
    end-to-end test would need to actually spin up the training
    loop, which is too expensive for unit tests; the inline source
    inspection guards against the check being silently removed."""
    from training.core import training_loop

    src = inspect.getsource(training_loop.train)
    assert "is_walking_config" in src, (
        "training_loop.train must define is_walking_config to gate eval enforcement"
    )
    assert "Walking config detected" in src, (
        "training_loop.train must raise a clear error when walking config has no eval"
    )
    # Spot-check the exact message references smoke14 so the
    # rationale is preserved with the guard.
    assert "smoke14" in src.lower() or "Smoke14" in src


def test_deterministic_eval_gate_soft_caps_are_documented() -> None:
    """The three soft caps must be defined as module-level constants
    so they're easy to find / tighten later.  Pin the values used at
    introduction so a future tightening is a visible diff."""
    from training.core import post_training_eval as pte

    assert pte.LATERAL_VELOCITY_SOFT_CAP_MPS == 0.10
    assert pte.YAW_DRIFT_SOFT_CAP_RAD == 0.40
    assert pte.WORLD_Y_DRIFT_SOFT_CAP_M == 0.30
