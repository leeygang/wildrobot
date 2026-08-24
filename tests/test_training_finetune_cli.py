import sys

import pytest

from training.train import parse_args


def test_finetune_policy_is_an_alias_for_actor_initialization(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["training/train.py", "--finetune-policy", "actor.pkl"],
    )

    args = parse_args()

    assert args.init_policy == "actor.pkl"


def test_finetune_policy_and_resume_are_mutually_exclusive(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training/train.py",
            "--resume",
            "full-state.pkl",
            "--finetune-policy",
            "actor.pkl",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
