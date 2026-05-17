from training.configs.training_config import load_training_config_from_dict
from training.core.training_loop import (
    _effective_ppo_epochs,
    _format_hhmmss,
    _is_eval_better,
    _linear_schedule_factor,
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
    assert cfg.ppo.rollback.enabled
    assert cfg.ppo.rollback.patience == 3
    assert cfg.ppo.rollback.success_rate_drop_threshold == 0.07
    assert cfg.ppo.rollback.lr_factor == 0.6
