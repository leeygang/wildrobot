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
        "status": "active",
        "stage": "training",
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


def test_codex_cannot_stop_on_an_ordinary_failed_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    decision = {
        "decision": "stop",
        "summary": "no obvious reward change",
        "config": "",
        "start_mode": "none",
        "checkpoint": "",
        "verification": [],
    }

    with pytest.raises(remote.TrainingLoopError, match="next bounded experiment"):
        auto._validate_codex_result(
            state,
            decision,
            "a" * 40,
            {"job_id": "job-1", "status": "completed"},
        )


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

    def fake_run(command, *, log_path, **_kwargs):
        commands.append(list(command))
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "decision": "continue",
                    "summary": "run the next bounded experiment",
                    "config": "training/configs/walking.yaml",
                    "start_mode": "init_policy",
                    "checkpoint": "/srv/wildrobot-training-jobs/job-1/checkpoint.pkl",
                    "verification": [],
                }
            )
        )
        log_path.write_text("done\n")
        return 0

    monkeypatch.setattr(auto.remote, "_run_streamed", fake_run)

    decision = auto._invoke_codex(state, manifest)

    assert decision["decision"] == "continue"
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

    def fail_analyzer(_command, *, log_path: Path, **_kwargs) -> int:
        log_path.write_text("bad metrics\ntrace\n")
        return 1

    monkeypatch.setattr(auto.remote, "_run_streamed", fail_analyzer)

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


def test_export_stage_reuses_an_already_valid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    local_root = tmp_path / "remote_jobs"
    bundle = local_root / "job-1/deployment_bundle"
    bundle.mkdir(parents=True)
    commands: list[list[str]] = []
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(
        auto.remote,
        "_sync_job",
        lambda *_args, **_kwargs: {
            "local_selected_checkpoint": "training/checkpoints/run/checkpoint.pkl",
            "local_checkpoint_run_dir": "training/checkpoints/run",
        },
    )

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "valid", "")

    monkeypatch.setattr(auto.subprocess, "run", fake_run)

    result = auto._export_ready_bundle(state, auto._context(state))

    assert result["bundle_path"] == str(bundle)
    assert result["readiness"] == "walking_bundle_ready"
    assert len(commands) == 1
    assert "runtime/wr_runtime/validation/validate_bundle.py" in commands[0]


def test_terminal_failed_gate_can_commit_push_and_enqueue_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)
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
        lambda *_args, **_kwargs: {
            "decision": "continue",
            "summary": "target the measured failure",
            "config": "training/configs/next.yaml",
            "start_mode": "init_policy",
            "checkpoint": "/srv/wildrobot-training-jobs/job-1/checkpoint.pkl",
            "verification": ["focused tests passed"],
        },
    )
    monkeypatch.setattr(auto, "_validate_codex_result", lambda *_args: None)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(auto, "_push", lambda _branch: events.append("push"))
    monkeypatch.setattr(
        auto,
        "_enqueue",
        lambda *_args, **_kwargs: events.append("enqueue") or "job-2",
    )

    auto._process_terminal_job(state, {"job_id": "job-1", "git_sha": "a" * 40})

    assert events == ["push", "enqueue"]


def test_enqueue_resets_status_for_the_new_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(last_remote_status="completed", last_poll_at="old")
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)
    monkeypatch.setattr(auto.remote, "_config_checkpoint_series", lambda _path: "x")
    monkeypatch.setattr(auto.remote, "_enqueue_remote", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)

    auto._enqueue(
        state,
        cycle=2,
        config="training/configs/next.yaml",
        start_mode="init_policy",
        checkpoint="training/checkpoints/source/checkpoint.pkl",
    )

    assert state["last_remote_status"] == "queued"
    assert state["last_poll_at"] is None


def test_enqueue_recovers_an_already_published_matching_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    pending = {
        "cycle": 2,
        "job_id": "job-2",
        "git_sha": "b" * 40,
        "config": "training/configs/next.yaml",
        "checkpoint_series": "walking/next",
        "start_mode": "init_policy",
        "checkpoint": "training/checkpoints/source/checkpoint.pkl",
    }
    state.update(stage="enqueue", pending_job=pending)
    saved: list[dict] = []
    monkeypatch.setattr(
        auto.remote,
        "_enqueue_remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            remote.TrainingLoopError("Job already exists")
        ),
    )
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: {
            "job_id": "job-2",
            "git_sha": "b" * 40,
            "source_config": "training/configs/next.yaml",
            "checkpoint_series": "walking/next",
            "start_mode": "init_policy",
            "start_checkpoint_request": (
                "training/checkpoints/source/checkpoint.pkl"
            ),
            "status": "running",
        },
    )
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    assert auto._resume_enqueue(state) == "job-2"

    assert saved[-1]["stage"] == "training"
    assert saved[-1]["active_job_id"] == "job-2"
    assert saved[-1]["last_remote_status"] == "running"
    assert "pending_job" not in saved[-1]


def test_fix_stage_recovers_completed_codex_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(stage="fix", fix_base_sha="a" * 40)
    decision = {
        "decision": "continue",
        "summary": "reuse completed fix",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": "training/checkpoints/source/checkpoint.pkl",
        "verification": ["tests passed"],
    }
    local_root = tmp_path / "remote_jobs"
    job_dir = local_root / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "codex_decision.json").write_text(json.dumps(decision))
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(auto, "_validate_codex_result", lambda *_args: None)
    monkeypatch.setattr(
        auto,
        "_invoke_codex",
        lambda *_args, **_kwargs: pytest.fail("must reuse persisted decision"),
    )

    recovered = auto._resume_codex_decision(state, {"job_id": "job-1"})

    assert recovered == decision


def test_push_stage_reuses_persisted_decision_without_reanalysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = {
        "decision": "continue",
        "summary": "persisted fix",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": "training/checkpoints/source/checkpoint.pkl",
        "verification": ["tests passed"],
    }
    state = {
        **_state(tmp_path),
        "stage": "push",
        "terminal_job_id": "job-1",
        "fix_base_sha": "a" * 40,
        "last_decision": decision,
    }
    local_root = tmp_path / "remote_jobs"
    job_dir = local_root / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / remote.MANIFEST_NAME).write_text(
        json.dumps({"job_id": "job-1", "status": "completed"})
    )
    events: list[str] = []
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(auto, "_validate_codex_result", lambda *_args: None)
    monkeypatch.setattr(auto, "_push", lambda _branch: events.append("push"))
    monkeypatch.setattr(
        auto,
        "_enqueue",
        lambda *_args, **_kwargs: events.append("enqueue") or "job-2",
    )
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda *_args: pytest.fail("must not rerun analysis"),
    )
    monkeypatch.setattr(
        auto,
        "_invoke_codex",
        lambda *_args, **_kwargs: pytest.fail("must not rerun Codex"),
    )

    auto._process_terminal_job(state)

    assert events == ["push", "enqueue"]


def test_analysis_stage_remains_resumable_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        **_state(tmp_path),
        "stage": "analysis",
        "terminal_job_id": "job-1",
    }
    local_root = tmp_path / "remote_jobs"
    job_dir = local_root / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / remote.MANIFEST_NAME).write_text(
        json.dumps({"job_id": "job-1", "status": "completed"})
    )
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_require_training_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda *_args: (_ for _ in ()).throw(
            remote.TrainingLoopError("analysis interrupted")
        ),
    )

    with pytest.raises(remote.TrainingLoopError, match="analysis interrupted"):
        auto._process_terminal_job(state)

    assert state["stage"] == "analysis"


def test_step_resumes_persisted_pipeline_stage_without_polling_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {**_state(tmp_path), "stage": "push"}
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: pytest.fail("must resume local pipeline stage"),
    )
    monkeypatch.setattr(
        auto,
        "_process_terminal_job",
        lambda *_args: calls.append("resume"),
    )

    assert auto._step_once() is state
    assert calls == ["resume"]


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
            "source_config": "training/configs/walking.yaml",
            "adopted_from_existing_run": True,
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


def test_adoption_recovers_an_already_published_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        **_state(tmp_path),
        "stage": "adopt",
        "active_config": "training/configs/walking.yaml",
        "active_git_sha": "a" * 40,
        "adoption_run_name": "offline-run-20260902-runid123",
    }
    saved: list[dict] = []
    monkeypatch.setattr(
        auto.remote,
        "_adopt_remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            remote.TrainingLoopError("Job already exists")
        ),
    )
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: {
            "job_id": "job-1",
            "status": "completed",
            "git_sha": "a" * 40,
            "source_config": "training/configs/walking.yaml",
            "adopted_from_existing_run": True,
            "wandb_run_name": "offline-run-20260902-runid123",
        },
    )
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    manifest = auto._resume_adoption(state)

    assert manifest["status"] == "completed"
    assert saved[-1]["stage"] == "training"
    assert "adoption_run_name" not in saved[-1]


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


def test_run_parser_defaults_to_ten_second_polling() -> None:
    args = auto._parse_args(["run"])

    assert args.func is auto._run
    assert args.poll_seconds == 10.0


def test_step_once_prints_remote_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {**_state(tmp_path), "status": "active"}
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: {"job_id": "job-1", "status": "running"},
    )

    result = auto._step_once(print_progress=True)

    assert result is state
    assert "cycle=1/4 job=job-1 remote_status=running" in capsys.readouterr().out


def test_foreground_poll_retries_remote_connectivity_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {**_state(tmp_path), "status": "active"}
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    saved: list[dict] = []
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    def fail_fetch(_context):
        raise remote.TrainingLoopError("SSH unavailable")

    monkeypatch.setattr(auto.remote, "_fetch_manifest", fail_fetch)

    result = auto._step_once(
        print_progress=True,
        tolerate_poll_errors=True,
    )

    assert result is state
    assert saved[-1]["status"] == "active"
    assert saved[-1]["last_remote_status"] == "unreachable"
    assert "remote_status=unreachable error=SSH unavailable" in capsys.readouterr().out


def test_run_polls_until_the_loop_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = [
        {**_state(tmp_path), "status": "active"},
        {
            **_state(tmp_path),
            "status": "stopped",
            "stop_reason": "maximum cycle count reached",
        },
    ]
    sleeps: list[float] = []
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(
        auto,
        "_step_once",
        lambda **_kwargs: states.pop(0),
    )
    monkeypatch.setattr(auto.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert auto._run(argparse.Namespace(poll_seconds=2.5)) == 0

    assert sleeps == [2.5]
    output = capsys.readouterr().out
    assert "running in the foreground" in output
    assert "maximum cycle count reached" in output


def test_retry_reactivates_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(
        status="stopped_error",
        stage="fix",
        processed_job_id="job-1",
        stop_reason="codex failed",
        fix_base_sha="a" * 40,
        fix_attempts=1,
    )
    saved: list[dict] = []
    monkeypatch.setattr(auto, "_load_state", lambda: dict(state))
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._retry(argparse.Namespace())

    assert saved[-1]["status"] == "active"
    assert saved[-1]["stage"] == "fix"
    assert saved[-1]["processed_job_id"] == "job-1"
    assert saved[-1]["fix_base_sha"] == "a" * 40
    assert saved[-1]["fix_attempts"] == 1
    assert "stop_reason" not in saved[-1]


@pytest.mark.parametrize(
    "stage",
    ["adopt", "training", "analysis", "fix", "push", "enqueue", "export"],
)
def test_retry_preserves_every_durable_pipeline_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    state = _state(tmp_path)
    state.update(
        status="stopped_error",
        stage=stage,
        pending_job={"job_id": "job-2"},
        last_decision={"decision": "continue"},
        stop_reason="interrupted",
    )
    saved: list[dict] = []
    monkeypatch.setattr(auto, "_load_state", lambda: dict(state))
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._retry(argparse.Namespace())

    assert saved[-1]["status"] == "active"
    assert saved[-1]["stage"] == stage
    assert saved[-1]["pending_job"] == {"job_id": "job-2"}
    assert saved[-1]["last_decision"] == {"decision": "continue"}


def test_retry_reactivates_a_previous_codex_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.pop("stage")
    state.update(
        status="stopped",
        processed_job_id="job-1",
        stop_reason="legacy Codex stop decision",
        last_decision={"decision": "stop"},
    )
    saved: list[dict] = []
    monkeypatch.setattr(auto, "_load_state", lambda: dict(state))
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._retry(argparse.Namespace())

    assert saved[-1]["status"] == "active"
    assert saved[-1]["processed_job_id"] is None
    assert saved[-1]["last_decision"] is None
