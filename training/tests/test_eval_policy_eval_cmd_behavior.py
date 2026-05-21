import sys
from types import SimpleNamespace

import pytest

from training.eval.eval_policy import (
    _disable_cmd_resample_for_eval,
    _network_activation_name,
    parse_args,
)


def _env_cfg(eval_velocity_cmd: float):
    return SimpleNamespace(eval_velocity_cmd=eval_velocity_cmd)


def test_eval_policy_pinned_eval_cmd_disables_resample() -> None:
    assert _disable_cmd_resample_for_eval(_env_cfg(0.0)) is True
    assert _disable_cmd_resample_for_eval(_env_cfg(0.15)) is True


def test_eval_policy_sentinel_eval_cmd_keeps_resample_enabled() -> None:
    assert _disable_cmd_resample_for_eval(_env_cfg(-0.001)) is False
    assert _disable_cmd_resample_for_eval(_env_cfg(-1.0)) is False


def _net_cfg(actor_activation: str, critic_activation: str):
    return SimpleNamespace(
        networks=SimpleNamespace(
            actor=SimpleNamespace(activation=actor_activation),
            critic=SimpleNamespace(activation=critic_activation),
        )
    )


def test_eval_policy_uses_training_activation_contract() -> None:
    assert _network_activation_name(_net_cfg("ELU", "elu")) == "elu"
    with pytest.raises(ValueError, match="must match"):
        _network_activation_name(_net_cfg("elu", "silu"))


def test_eval_policy_cli_requires_config(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_policy.py", "--checkpoint", "/tmp/fake.pkl"],
    )
    with pytest.raises(SystemExit):
        parse_args()
