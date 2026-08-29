import json
import sys

import pytest

from training.train import _resolve_initial_policy_checkpoint_path, parse_args


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


def test_init_policy_directory_uses_rank_one_diagnostic_candidate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint_1_131072.pkl"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_40_5242880.pkl").write_bytes(b"checkpoint")
    (tmp_path / "post_training_eval_summary.json").write_text(
        json.dumps(
            {
                "selected_checkpoint_path": None,
                "top_k_candidates": [
                    {
                        "rank": 2,
                        "checkpoint_path": "remote/run/checkpoint_40_5242880.pkl",
                    },
                    {
                        "rank": 1,
                        "checkpoint_path": "remote/run/checkpoint_1_131072.pkl",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_initial_policy_checkpoint_path(str(tmp_path))

    assert resolved == str(checkpoint)


def test_init_policy_directory_rejects_ambiguous_checkpoints(tmp_path) -> None:
    (tmp_path / "checkpoint_1_1.pkl").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_2_2.pkl").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="Pass a checkpoint .pkl file explicitly"):
        _resolve_initial_policy_checkpoint_path(str(tmp_path))
