from training.configs.training_config import load_training_config_from_dict
from training.core.training_loop import (
    _effective_ppo_epochs,
    _format_hhmmss,
    _is_eval_better,
    _linear_schedule_factor,
    _promotion_eval_pass,
    _promotion_window_trigger,
    _should_fire_callback,
    _should_trigger_rollback,
    _train_promotion_candidate_row_pass,
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


def _candidate_row(
    *,
    forward_velocity: float = 0.10,
    cmd_err: float = 0.05,
    step_length: float = 0.035,
    cmd_ratio: float = 1.0,
) -> dict:
    return {
        "forward_velocity": forward_velocity,
        "tracking/cmd_vs_achieved_forward": cmd_err,
        "tracking/step_length_touchdown_event_m": step_length,
        "tracking/forward_velocity_cmd_ratio": cmd_ratio,
    }


def test_promotion_candidate_trigger_3_of_5_is_true() -> None:
    rows = [
        _train_promotion_candidate_row_pass(_candidate_row(), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(), 490.0),
        _train_promotion_candidate_row_pass(_candidate_row(), 480.0),
        _train_promotion_candidate_row_pass(_candidate_row(forward_velocity=0.03), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(cmd_err=0.20), 500.0),
    ]
    triggered, hits = _promotion_window_trigger(rows, window=5, min_hits=3)
    assert hits == 3
    assert triggered is True


def test_promotion_candidate_trigger_2_of_5_is_false() -> None:
    rows = [
        _train_promotion_candidate_row_pass(_candidate_row(), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(forward_velocity=0.02), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(cmd_err=0.20), 500.0),
        _train_promotion_candidate_row_pass(_candidate_row(step_length=0.010), 500.0),
    ]
    triggered, hits = _promotion_window_trigger(rows, window=5, min_hits=3)
    assert hits == 2
    assert triggered is False


def test_promotion_candidate_ratio_outside_gate_is_false() -> None:
    assert (
        _train_promotion_candidate_row_pass(
            _candidate_row(cmd_ratio=1.6),
            500.0,
        )
        is False
    )
    assert (
        _train_promotion_candidate_row_pass(
            _candidate_row(cmd_ratio=0.5),
            500.0,
        )
        is False
    )


def test_promotion_candidate_missing_metric_is_false() -> None:
    row = _candidate_row()
    row.pop("tracking/step_length_touchdown_event_m")
    assert _train_promotion_candidate_row_pass(row, 500.0) is False


def test_promotion_eval_pass_and_fail_cases() -> None:
    is_pass, step_available = _promotion_eval_pass(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        }
    )
    assert step_available is True
    assert is_pass is True

    is_pass_vel, _ = _promotion_eval_pass(
        {
            "forward_velocity": 0.04,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "step_length_touchdown_event_m": 0.031,
        }
    )
    assert is_pass_vel is False

    is_pass_ep_len, _ = _promotion_eval_pass(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 420.0,
            "step_length_touchdown_event_m": 0.031,
        }
    )
    assert is_pass_ep_len is False

    is_pass_ratio_fail, _ = _promotion_eval_pass(
        {
            "forward_velocity": 0.09,
            "cmd_vs_achieved_forward": 0.04,
            "mean_episode_length": 500.0,
            "eval_velocity_cmd": 0.20,
            "step_length_touchdown_event_m": 0.031,
        }
    )
    assert is_pass_ratio_fail is False


def test_should_fire_callback_covers_log_and_checkpoint_boundaries() -> None:
    assert _should_fire_callback(iteration=1, log_interval=50, checkpoint_interval=200) is True
    assert _should_fire_callback(iteration=10, log_interval=10, checkpoint_interval=100) is True
    assert _should_fire_callback(iteration=20, log_interval=7, checkpoint_interval=10) is True
    assert _should_fire_callback(iteration=9, log_interval=10, checkpoint_interval=20) is False


def test_promotion_eval_defaults_are_backward_compatible() -> None:
    cfg = load_training_config_from_dict({"ppo": {"eval": {"enabled": False}}})
    assert cfg.ppo.eval.enabled is False
    assert cfg.ppo.eval.promotion_enabled is False
    assert cfg.ppo.eval.promotion_window == 5
    assert cfg.ppo.eval.promotion_min_hits == 3
    assert cfg.ppo.eval.promotion_num_envs == 8
    assert cfg.ppo.eval.promotion_num_steps == 500
    assert cfg.ppo.eval.promotion_checkpoint_label == "eval_promoted"


def test_promotion_eval_can_enable_without_periodic_eval() -> None:
    cfg = load_training_config_from_dict(
        {
            "ppo": {
                "eval": {
                    "enabled": False,
                    "promotion_enabled": True,
                    "promotion_window": 5,
                    "promotion_min_hits": 3,
                    "promotion_num_envs": 8,
                    "promotion_num_steps": 500,
                    "promotion_checkpoint_label": "eval_promoted",
                }
            }
        }
    )
    assert cfg.ppo.eval.enabled is False
    assert cfg.ppo.eval.promotion_enabled is True
    assert cfg.ppo.eval.promotion_num_envs == 8
    assert cfg.ppo.eval.promotion_num_steps == 500


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
                    "promotion_enabled": True,
                    "promotion_window": 7,
                    "promotion_min_hits": 4,
                    "promotion_num_envs": 12,
                    "promotion_num_steps": 250,
                    "promotion_checkpoint_label": "candidate_promoted",
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
    assert cfg.ppo.eval.promotion_enabled is True
    assert cfg.ppo.eval.promotion_window == 7
    assert cfg.ppo.eval.promotion_min_hits == 4
    assert cfg.ppo.eval.promotion_num_envs == 12
    assert cfg.ppo.eval.promotion_num_steps == 250
    assert cfg.ppo.eval.promotion_checkpoint_label == "candidate_promoted"
    assert cfg.ppo.rollback.enabled
    assert cfg.ppo.rollback.patience == 3
    assert cfg.ppo.rollback.success_rate_drop_threshold == 0.07
    assert cfg.ppo.rollback.lr_factor == 0.6
