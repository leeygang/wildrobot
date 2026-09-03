from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from wildrobot.agents import autonomous_training_loop as auto
from wildrobot.agents import remote_training_loop as remote


def _state(tmp_path: Path) -> dict:
    return {
        "branch": "main",
        "cycle": 1,
        "max_cycles": 4,
        "max_training_failures": 2,
        "training_failures": 0,
        "host": "gpu",
        "user": "robot",
        "port": None,
        "remote_repo": "/srv/wildrobot",
        "remote_jobs_root": "/srv/wildrobot-training-jobs",
        "active_job_id": "job-1",
        "active_git_sha": "a" * 40,
        "codex_timeout_minutes": 5,
        "codex_path": "/usr/local/bin/codex",
        "codex_model": None,
        "standing_checkpoint": None,
        "standing_config": None,
        "test_root": str(tmp_path),
    }


def test_codex_prompt_contains_exact_run_context(tmp_path: Path) -> None:
    state = _state(tmp_path)
    manifest = {
        "job_id": "job-1",
        "status": "completed",
        "git_sha": "a" * 40,
        "local_wandb_run_dir": "training/wandb/offline-run-id",
        "local_checkpoint_run_dir": "training/checkpoints/run-id",
        "checkpoint_run_dir": "/srv/jobs/run-id",
    }

    prompt = auto._codex_prompt(state, manifest)

    assert "one dominant failure mode" in prompt
    assert "a" * 40 in prompt
    assert "offline-run-id" in prompt
    assert "Do not modify files under `wildrobot/agents/`" in prompt


def test_next_checkpoint_must_stay_in_approved_gpu_roots(tmp_path: Path) -> None:
    state = _state(tmp_path)
    auto._validate_checkpoint_path(
        state,
        "/srv/wildrobot-training-jobs/job-1/artifacts/checkpoints/run/checkpoint.pkl",
    )
    auto._validate_checkpoint_path(state, "training/checkpoints/source/checkpoint.pkl")

    with pytest.raises(remote.TrainingLoopError, match="outside approved"):
        auto._validate_checkpoint_path(state, "/tmp/untrusted/checkpoint.pkl")


def test_completed_job_must_match_mac_and_state_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)

    with pytest.raises(remote.TrainingLoopError, match="same Git commit"):
        auto._require_training_commit(state, {"git_sha": "a" * 40})


def test_adopted_job_allows_only_control_plane_commits_after_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state["initial_run_adopted"] = True
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(
        auto,
        "_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(
        auto,
        "_git_output",
        lambda *args: "wildrobot/agents/remote_training_loop.py",
    )

    assert auto._require_training_commit(state, {"git_sha": "a" * 40}) == "b" * 40

    monkeypatch.setattr(
        auto,
        "_git_output",
        lambda *args: "training/envs/wildrobot_env.py",
    )
    with pytest.raises(remote.TrainingLoopError, match="training-relevant"):
        auto._require_training_commit(state, {"git_sha": "a" * 40})


def test_next_checkpoint_must_come_from_deterministic_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    checkpoint_dir = tmp_path / "training/checkpoints/run-id"
    checkpoint_dir.mkdir(parents=True)
    allowed = (
        "/srv/wildrobot-training-jobs/job-1/artifacts/checkpoints/run/"
        "checkpoint_10.pkl"
    )
    (checkpoint_dir / "post_training_eval_summary.json").write_text(
        json.dumps(
            {
                "selected_checkpoint_path": None,
                "top_k_candidates": [{"checkpoint_path": allowed}],
            }
        )
    )
    manifest = {
        "job_id": "job-1",
        "local_checkpoint_run_dir": "training/checkpoints/run-id",
    }
    decision = {
        "decision": "continue",
        "summary": "train longer from the measured top candidate",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": allowed,
        "verification": ["focused tests passed"],
    }
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)

    auto._validate_codex_result(state, decision, "a" * 40, manifest)
    decision["checkpoint"] = (
        "/srv/wildrobot-training-jobs/job-1/artifacts/checkpoints/run/" "invented.pkl"
    )
    with pytest.raises(remote.TrainingLoopError, match="not recorded"):
        auto._validate_codex_result(state, decision, "a" * 40, manifest)


def test_codex_exec_uses_structured_workspace_write_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    manifest = {"job_id": "job-1", "status": "completed", "git_sha": "a" * 40}
    local_root = tmp_path / "remote_jobs"
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto, "_validate_codex_result", lambda *_args: None)

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "decision": "stop",
                    "summary": "no safe next step",
                    "config": "",
                    "start_mode": "none",
                    "checkpoint": "",
                    "verification": [],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(auto.subprocess, "run", fake_run)

    decision = auto._invoke_codex(state, manifest)

    assert decision["decision"] == "stop"
    assert "--approve-for-me" in commands[0]
    assert "--sandbox" not in commands[0]
    log = (local_root / "job-1" / "codex_exec.log").read_text()
    assert "done" in log


def test_analyzer_failure_stops_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "remote_jobs"
    (local_root / "job-1").mkdir(parents=True)
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(
        auto.remote,
        "_sync_job",
        lambda *_args, **_kwargs: {
            "job_id": "job-1",
            "status": "completed",
            "local_wandb_run_dir": "training/wandb/run-id",
        },
    )
    monkeypatch.setattr(
        auto.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout="bad metrics", stderr="trace"
        ),
    )

    with pytest.raises(remote.TrainingLoopError, match="analyzer exited"):
        auto._run_analyzer(auto._context(_state(tmp_path)))
    assert "bad metrics" in (local_root / "job-1/analysis.txt").read_text()


def test_terminal_ready_job_exports_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    saved: list[dict] = []
    monkeypatch.setattr(auto, "_require_training_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda _context: {
            "job_id": "job-1",
            "status": "completed",
            "simulation_candidate_ready": True,
        },
    )
    monkeypatch.setattr(
        auto,
        "_export_ready_bundle",
        lambda *_args: {
            "bundle_path": str(tmp_path / "bundle"),
            "readiness": "walking_bundle_ready",
        },
    )
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._process_terminal_job(state, {"job_id": "job-1", "git_sha": "a" * 40})

    assert saved[-1]["status"] == "ready"
    assert saved[-1]["readiness"] == "walking_bundle_ready"


def test_terminal_failed_gate_can_commit_push_and_enqueue_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(auto, "_require_training_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda _context: {
            "job_id": "job-1",
            "status": "completed",
            "simulation_candidate_ready": False,
        },
    )
    monkeypatch.setattr(
        auto,
        "_invoke_codex",
        lambda *_args: {
            "decision": "continue",
            "summary": "target the measured failure",
            "config": "training/configs/next.yaml",
            "start_mode": "init_policy",
            "checkpoint": "/srv/wildrobot-training-jobs/job-1/checkpoint.pkl",
            "verification": ["focused tests passed"],
        },
    )
    monkeypatch.setattr(auto, "_push", lambda _branch: events.append("push"))
    monkeypatch.setattr(
        auto,
        "_enqueue",
        lambda *_args, **_kwargs: events.append("enqueue") or "job-2",
    )

    auto._process_terminal_job(state, {"job_id": "job-1", "git_sha": "a" * 40})

    assert events == ["push", "enqueue"]


def test_mac_service_installer_writes_launch_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", tmp_path / "logs")

    auto._install_mac_service(argparse.Namespace(interval=120))

    plist = (
        tmp_path / "Library/LaunchAgents/com.wildrobot.autonomous-training.plist"
    ).read_text()
    assert "autonomous_training_loop.py" in plist
    assert "<integer>120</integer>" in plist
    assert "<string>step</string>" in plist


def test_start_can_adopt_an_already_completed_gpu_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_sha = "a" * 40
    saved: list[dict] = []
    monkeypatch.setattr(auto, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: git_sha)
    monkeypatch.setattr(
        auto,
        "_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)
    monkeypatch.setattr(auto.shutil, "which", lambda _name: "/usr/local/bin/codex")
    monkeypatch.setattr(
        auto.remote,
        "_adopt_remote",
        lambda context, **_kwargs: {
            "job_id": context.job_id,
            "status": "completed",
            "git_sha": git_sha,
            "wandb_run_name": "offline-run-20260902_120000-runid123",
        },
    )
    monkeypatch.setattr(
        auto, "_push", lambda _branch: pytest.fail("adoption must not push")
    )
    monkeypatch.setattr(
        auto, "_enqueue", lambda *_args, **_kwargs: pytest.fail("must not enqueue")
    )
    monkeypatch.setattr(auto, "_save_state", lambda state: saved.append(dict(state)))
    args = argparse.Namespace(
        config="training/configs/walking.yaml",
        init_policy=None,
        resume=None,
        adopt_completed="offline-run-20260902_120000-runid123",
        training_git_sha=git_sha,
        branch="main",
        host="gpu",
        user="robot",
        port=None,
        remote_repo="/srv/wildrobot",
        max_cycles=4,
        max_training_failures=2,
        codex_path=None,
        codex_model=None,
        codex_timeout_minutes=5,
        standing_checkpoint=None,
        standing_config=None,
        new_run=False,
    )

    auto._start(args)

    assert saved[-1]["status"] == "active"
    assert saved[-1]["cycle"] == 1
    assert saved[-1]["active_job_id"].startswith("auto-01-walking-")
    assert saved[-1]["active_git_sha"] == git_sha
    assert saved[-1]["initial_run_adopted"] is True


def test_start_parser_uses_latest_completed_run_when_name_is_omitted() -> None:
    args = auto._parse_args(
        [
            "start",
            "--config",
            "training/configs/walking.yaml",
            "--adopt-completed",
        ]
    )

    assert args.adopt_completed == "latest"


def test_retry_reactivates_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(
        status="stopped_error",
        processed_job_id="job-1",
        stop_reason="codex failed",
    )
    saved: list[dict] = []
    monkeypatch.setattr(auto, "_load_state", lambda: dict(state))
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._retry(argparse.Namespace())

    assert saved[-1]["status"] == "active"
    assert saved[-1]["processed_job_id"] is None
    assert "stop_reason" not in saved[-1]
