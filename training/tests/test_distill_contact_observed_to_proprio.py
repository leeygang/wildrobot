from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from training.scripts.distill_contact_observed_to_proprio import (
    _gate_failures,
    _resolve_teacher_checkpoint,
)


def _metrics(
    *,
    rmse: float,
    projected_rmse: float = 0.05,
    teacher_terminated: bool,
    student_terminated: bool,
):
    return {
        "validation_action_error": {"rmse": rmse},
        "projected_initial_validation_action_error": {
            "rmse": projected_rmse
        },
        "teacher_rollouts": {
            "trial": {
                "first_termination_step": 10 if teacher_terminated else None
            }
        },
        "distilled_student_rollouts": {
            "trial": {
                "first_termination_step": 10 if student_terminated else None
            }
        },
    }


def test_distillation_gate_passes_only_low_error_surviving_policy() -> None:
    assert not _gate_failures(
        _metrics(rmse=0.02, teacher_terminated=False, student_terminated=False),
        max_validation_rmse=0.08,
        require_no_terminations=True,
    )


def test_distillation_gate_reports_action_error_and_termination() -> None:
    failures = _gate_failures(
        _metrics(
            rmse=0.10,
            projected_rmse=0.11,
            teacher_terminated=False,
            student_terminated=True,
        ),
        max_validation_rmse=0.08,
        require_no_terminations=True,
    )

    assert len(failures) == 2
    assert "validation action RMSE" in failures[0]
    assert "distilled_student_rollouts terminated" in failures[1]


def test_distillation_gate_rejects_action_regression_from_projection() -> None:
    failures = _gate_failures(
        _metrics(
            rmse=0.06,
            projected_rmse=0.05,
            teacher_terminated=False,
            student_terminated=False,
        ),
        max_validation_rmse=0.08,
        require_no_terminations=True,
    )

    assert failures == [
        "distillation regressed validation action RMSE 0.050000 -> 0.060000"
    ]


def test_teacher_checkpoint_resolves_from_gpu_job_artifacts(tmp_path: Path) -> None:
    content = b"exact teacher checkpoint"
    expected_sha256 = hashlib.sha256(content).hexdigest()
    candidate = tmp_path / "jobs/job/artifacts/checkpoint_7_143360.pkl"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(content)

    resolved = _resolve_teacher_checkpoint(
        tmp_path / "missing/checkpoint_7_143360.pkl",
        expected_sha256=expected_sha256,
        search_roots=[(tmp_path / "jobs", "checkpoint_7_143360.pkl")],
    )

    assert resolved == candidate


def test_teacher_checkpoint_resolution_rejects_wrong_checkpoint(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "checkpoint_7_143360.pkl"
    candidate.write_bytes(b"wrong checkpoint")

    with pytest.raises(FileNotFoundError, match="exact 17d18 checkpoint-7"):
        _resolve_teacher_checkpoint(
            candidate,
            expected_sha256="0" * 64,
            search_roots=[],
        )
