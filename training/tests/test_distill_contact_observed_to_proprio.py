from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY, NUM_METRICS
from training.scripts.distill_contact_observed_to_proprio import (
    _gate_failures,
    _resolve_teacher_checkpoint,
)
from training.scripts.distill_walking_21d_to_17d import (
    _rollout_step_metrics,
    _summarize_rollout,
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


def test_rollout_summary_does_not_count_horizon_truncation_as_fall() -> None:
    summary = _summarize_rollout(
        done=[0.0, 0.0, 1.0],
        truncated=[0.0, 0.0, 1.0],
        root_xyz=[[0.0, 0.0, 0.45]] * 3,
        steps=3,
    )

    assert summary["termination_count"] == 0
    assert summary["first_termination_step"] is None
    assert summary["truncation_count"] == 1


def test_rollout_step_reads_truncation_from_terminal_metrics() -> None:
    metrics_vec = jnp.zeros(NUM_METRICS).at[
        METRIC_INDEX["term/truncated"]
    ].set(1.0)
    state = SimpleNamespace(
        done=jnp.float32(1.0),
        metrics={METRICS_VEC_KEY: metrics_vec},
        data=SimpleNamespace(qpos=jnp.asarray([1.0, 2.0, 3.0, 4.0])),
    )

    done, truncated, root_xyz = _rollout_step_metrics(state)

    assert float(done) == 1.0
    assert float(truncated) == 1.0
    assert root_xyz.tolist() == [1.0, 2.0, 3.0]
