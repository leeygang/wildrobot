from types import SimpleNamespace

from training.eval.eval_policy import _disable_cmd_resample_for_eval


def _env_cfg(eval_velocity_cmd: float):
    return SimpleNamespace(eval_velocity_cmd=eval_velocity_cmd)


def test_eval_policy_pinned_eval_cmd_disables_resample() -> None:
    assert _disable_cmd_resample_for_eval(_env_cfg(0.0)) is True
    assert _disable_cmd_resample_for_eval(_env_cfg(0.15)) is True


def test_eval_policy_sentinel_eval_cmd_keeps_resample_enabled() -> None:
    assert _disable_cmd_resample_for_eval(_env_cfg(-0.001)) is False
    assert _disable_cmd_resample_for_eval(_env_cfg(-1.0)) is False
