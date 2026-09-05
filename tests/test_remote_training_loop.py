from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from training.exports.export_policy_bundle import _load_training_job_provenance
from wildrobot.agents import remote_training_loop
from wildrobot.agents.remote_training_loop import (
    CHECKPOINT_MANIFEST_NAME,
    MANIFEST_NAME,
    TrainingLoopError,
    _adopt_completed_run,
    _build_remote_submit_script,
    _checkpoint_artifact_relative,
    _checkpoint_series,
    _collect_training_results,
    _collect_walking_evaluation_results,
    _copy_manifest_to_checkpoint,
    _find_completed_run,
    _initial_manifest,
    _write_effective_config,
    _write_json_atomic,
)


def test_effective_config_changes_only_wandb_artifact_location(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    destination = tmp_path / "effective.yaml"
    source_payload = {
        "version": "0.21.0",
        "checkpoints": {"dir": "training/checkpoints/walking"},
        "wandb": {"enabled": True, "mode": "offline", "project": "wildrobot"},
        "ppo": {"iterations": 10},
    }
    source.write_text(yaml.safe_dump(source_payload, sort_keys=False))

    _write_effective_config(
        source,
        destination,
        wandb_log_dir=tmp_path / "artifacts" / "wandb",
    )

    effective = yaml.safe_load(destination.read_text())
    assert effective["wandb"]["log_dir"] == str(tmp_path / "artifacts" / "wandb")
    del effective["wandb"]["log_dir"]
    assert effective == source_payload


def test_remote_submit_enqueues_exact_commit_for_gpu_service() -> None:
    script = _build_remote_submit_script(
        remote_repo="/srv/wildrobot",
        jobs_root="/srv/wildrobot-training-jobs",
        job_id="walking-deadbeef-20260902",
        git_sha="deadbeef" * 5,
        config="training/configs/walking.yaml",
        checkpoint_dir="walking-series",
        init_policy="training/checkpoints/source/checkpoint.pkl",
        resume=None,
    )

    assert "git -C /srv/wildrobot fetch origin" in script
    assert "mkdir -p /srv/wildrobot-training-jobs/walking-deadbeef-20260902" in script
    assert ("deadbeef" * 5) in script
    assert "job_manifest.json" in script
    assert ".job_manifest.json.tmp" in script
    assert "mv " in script
    assert "systemd-run" not in script
    assert "scp_to_remote" not in script

    manifest = _initial_manifest(
        context=remote_training_loop.RemoteContext(
            "walking-deadbeef-20260902", "gpu", "user", None, "/srv/wildrobot"
        ),
        git_sha="deadbeef" * 5,
        config="training/configs/walking.yaml",
        checkpoint_series="walking-series",
        init_policy="training/checkpoints/source/checkpoint.pkl",
        resume=None,
    )
    assert manifest["start_mode"] == "init_policy"
    assert manifest["start_checkpoint_request"].endswith("checkpoint.pkl")


def test_gpu_dispatcher_runs_one_valid_queued_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo = tmp_path / "wildrobot"
    python_path = remote_repo / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python")
    context = remote_training_loop.RemoteContext(
        "walking-job", "gpu", "robot", None, str(remote_repo)
    )
    manifest = _initial_manifest(
        context=context,
        git_sha="a" * 40,
        config="training/configs/walking.yaml",
        checkpoint_series="walking-series",
        init_policy=None,
        resume=None,
    )
    manifest_path = Path(context.job_root) / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    git_commands: list[list[str]] = []

    def fake_git(command, **_kwargs):
        git_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_worker(command, **_kwargs):
        completed = json.loads(manifest_path.read_text())
        completed["status"] = "completed"
        _write_json_atomic(manifest_path, completed)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(remote_training_loop, "_run", fake_git)
    monkeypatch.setattr(remote_training_loop.subprocess, "run", fake_worker)

    assert remote_training_loop._dispatch_queued_job(remote_repo) is True
    assert json.loads(manifest_path.read_text())["status"] == "completed"
    assert ["git", "fetch", "origin"] in git_commands
    assert any(command[:3] == ["git", "worktree", "add"] for command in git_commands)


def test_gpu_dispatcher_reuses_matching_recovered_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo = tmp_path / "wildrobot"
    worktree = tmp_path / "job/src"
    worktree.mkdir(parents=True)
    git_sha = "a" * 40
    commands: list[list[str]] = []

    def fake_git_output(*args, **_kwargs):
        return git_sha if args[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(remote_training_loop, "_git_output", fake_git_output)
    monkeypatch.setattr(
        remote_training_loop,
        "_run",
        lambda command, **_kwargs: commands.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    remote_training_loop._ensure_job_worktree(remote_repo, worktree, git_sha)

    assert not any(command[:3] == ["git", "worktree", "add"] for command in commands)


def test_gpu_dispatcher_rejects_manifest_paths_outside_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo = tmp_path / "wildrobot"
    remote_repo.mkdir()
    context = remote_training_loop.RemoteContext(
        "walking-job", "gpu", "robot", None, str(remote_repo)
    )
    manifest = _initial_manifest(
        context=context,
        git_sha="a" * 40,
        config="training/configs/walking.yaml",
        checkpoint_series="walking-series",
        init_policy=None,
        resume=None,
    )
    manifest["worktree"] = str(tmp_path / "outside")
    manifest_path = Path(context.job_root) / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    monkeypatch.setattr(
        remote_training_loop,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("invalid job reached Git"),
    )

    assert remote_training_loop._dispatch_queued_job(remote_repo) is True
    failed = json.loads(manifest_path.read_text())
    assert failed["status"] == "failed"
    assert "invalid worktree path" in failed["error"]


def test_gpu_service_installer_writes_user_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo = tmp_path / "wildrobot"
    python_path = remote_repo / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n")
    python_path.chmod(0o755)
    worker = remote_repo / "wildrobot/agents/remote_training_loop.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    remote_training_loop._install_gpu_service(
        argparse.Namespace(remote_repo=str(remote_repo))
    )

    unit = (
        tmp_path
        / "home/.config/systemd/user/wildrobot-training-gpu.service"
    ).read_text()
    assert f"ExecStart={python_path}" in unit
    assert "gpu-serve" in unit
    assert "Restart=on-failure" in unit


def test_checkpoint_series_strips_canonical_prefix() -> None:
    assert (
        _checkpoint_series("training/checkpoints/walking/17d11")
        == "walking/17d11"
    )


def test_checkpoint_artifact_path_rejects_traversal() -> None:
    with pytest.raises(TrainingLoopError, match="Unsafe checkpoint artifact path"):
        _checkpoint_artifact_relative("checkpoints/../../outside.pkl")


def test_result_collection_uses_authoritative_selected_checkpoint(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    artifact_root = job_root / "artifacts"
    checkpoint_series = artifact_root / "checkpoints" / "walking"
    checkpoint_run = checkpoint_series / "walking_v0210_20260902-runid123"
    checkpoint_run.mkdir(parents=True)
    selected = checkpoint_run / "checkpoint_10_204800.pkl"
    selected.write_bytes(b"policy")
    (checkpoint_run / "post_training_eval_summary.json").write_text(
        json.dumps({"selected_checkpoint_path": str(selected)})
    )
    wandb_run = artifact_root / "wandb" / "offline-run-20260902_120000-runid123"
    wandb_run.mkdir(parents=True)
    manifest = {
        "job_id": "walking-test",
        "artifact_root": str(artifact_root),
        "checkpoint_series_dir": str(checkpoint_series),
        "worktree": str(job_root / "src"),
    }

    _collect_training_results(manifest)

    assert manifest["simulation_candidate_ready"] is True
    assert manifest["result_complete"] is True
    assert manifest["wandb_run_id"] == "runid123"
    assert manifest["checkpoint_run_relpath"] == (
        "checkpoints/walking/walking_v0210_20260902-runid123"
    )
    assert manifest["selected_checkpoint_relpath"].endswith(
        "checkpoint_10_204800.pkl"
    )

    manifest_path = job_root / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    _copy_manifest_to_checkpoint(manifest_path, manifest)
    copied = json.loads((checkpoint_run / CHECKPOINT_MANIFEST_NAME).read_text())
    assert copied["job_id"] == "walking-test"


def test_result_collection_records_bootstrap_gate_status(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    report = artifact_root / "bootstrap/contact_free_distilled.metrics.json"
    checkpoint = artifact_root / "bootstrap/contact_free_distilled.pkl"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"gates": {"passed": True}}))
    checkpoint.write_bytes(b"policy")
    manifest = {
        "artifact_root": str(artifact_root),
        "checkpoint_series_dir": str(artifact_root / "checkpoints/walking"),
        "worktree": str(tmp_path / "src"),
        "bootstrap_mode": "contact_observed_to_proprio",
        "bootstrap_report": str(report),
        "bootstrap_checkpoint": str(checkpoint),
    }

    _collect_training_results(manifest)

    assert manifest["bootstrap_status"] == "passed"
    assert manifest["bootstrap_gates_passed"] is True
    assert manifest["bootstrap_checkpoint_exists"] is True


def test_bundle_provenance_reads_training_job_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_10_204800.pkl"
    checkpoint.write_bytes(b"policy")
    (tmp_path / CHECKPOINT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "job_id": "walking-17d11",
                "git_sha": "a" * 40,
                "git_dirty": False,
                "wandb_run_id": "runid123",
                "source_config_sha256": "b" * 64,
                "effective_config_sha256": "c" * 64,
                "start_checkpoint_sha256": "d" * 64,
            }
        )
    )

    provenance = _load_training_job_provenance(checkpoint)

    assert provenance == {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "training_job_id": "walking-17d11",
        "training_run_id": "runid123",
        "source_config_sha256": "b" * 64,
        "effective_config_sha256": "c" * 64,
        "initial_checkpoint_sha256": "d" * 64,
    }


def test_gpu_worker_records_completed_run_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_root = tmp_path / "job"
    artifact_root = job_root / "artifacts"
    checkpoint_series = artifact_root / "checkpoints" / "walking"
    worktree = job_root / "src"
    worktree.mkdir(parents=True)
    manifest = {
        "job_id": "walking-test",
        "status": "queued",
        "artifact_root": str(artifact_root),
        "checkpoint_series_dir": str(checkpoint_series),
        "worktree": str(worktree),
        "training_command": ["fake-training"],
    }
    _write_json_atomic(job_root / MANIFEST_NAME, manifest)
    monkeypatch.setattr(remote_training_loop, "_gpu_name", lambda: "Test GPU")
    monkeypatch.setattr(
        remote_training_loop, "_prepare_worker", lambda _manifest: ["fake-training"]
    )

    def fake_run(*_args, log_path: Path, **_kwargs):
        checkpoint_run = checkpoint_series / "walking_v0210_20260902-runid123"
        checkpoint_run.mkdir(parents=True)
        (checkpoint_run / "post_training_eval_summary.json").write_text(
            json.dumps({"selected_checkpoint_path": None})
        )
        (artifact_root / "wandb" / "offline-run-20260902_120000-runid123").mkdir(
            parents=True
        )
        with log_path.open("a") as log:
            log.write("training output\n")
        return 0

    monkeypatch.setattr(remote_training_loop, "_run_streamed", fake_run)

    result = remote_training_loop._gpu_worker(
        argparse.Namespace(job_root=str(job_root))
    )

    completed = json.loads((job_root / MANIFEST_NAME).read_text())
    assert result == 0
    assert completed["status"] == "completed"
    assert completed["simulation_candidate_ready"] is False
    assert completed["selected_checkpoint_path"] is None
    assert completed["training_attempt"] == 1
    checkpoint_manifest = (
        checkpoint_series
        / "walking_v0210_20260902-runid123"
        / CHECKPOINT_MANIFEST_NAME
    )
    assert checkpoint_manifest.is_file()
    assert "training output" in (job_root / "train.log").read_text()
    assert "attempt=1" in (job_root / "train.log").read_text()


def test_gpu_restart_requeues_interrupted_training_from_declared_start(
    tmp_path: Path,
) -> None:
    remote_repo = tmp_path / "wildrobot"
    jobs_root = tmp_path / "wildrobot-training-jobs"
    job_root = jobs_root / "walking-job"
    artifact_root = job_root / "artifacts"
    manifest = {
        "job_id": "walking-job",
        "status": "running",
        "remote_repo": str(remote_repo),
        "job_root": str(job_root),
        "worktree": str(job_root / "src"),
        "artifact_root": str(artifact_root),
        "checkpoint_series_dir": str(artifact_root / "checkpoints/walking"),
        "start_mode": "init_policy",
        "start_checkpoint_request": "training/checkpoints/source.pkl",
        "worker_pid": 123,
        "started_at": "old",
    }
    manifest_path = job_root / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)

    remote_training_loop._recover_interrupted_jobs(remote_repo)

    recovered = json.loads(manifest_path.read_text())
    assert recovered["status"] == "queued"
    assert recovered["start_mode"] == "init_policy"
    assert recovered["start_checkpoint_request"].endswith("source.pkl")
    assert recovered["recovery_action"] == "requeued_same_training_job"
    assert recovered["recovery_events"][0]["previous_status"] == "running"
    assert recovered["recovery_events"][0]["action"] == (
        "requeued_same_training_job"
    )
    assert "worker_pid" not in recovered


def test_gpu_restart_leaves_a_live_locked_worker_unchanged(tmp_path: Path) -> None:
    remote_repo = tmp_path / "wildrobot"
    jobs_root = tmp_path / "wildrobot-training-jobs"
    job_root = jobs_root / "walking-job"
    manifest_path = job_root / MANIFEST_NAME
    manifest = {
        "job_id": "walking-job",
        "status": "running",
        "remote_repo": str(remote_repo),
        "job_root": str(job_root),
        "worktree": str(job_root / "src"),
        "artifact_root": str(job_root / "artifacts"),
        "checkpoint_series_dir": str(job_root / "artifacts/checkpoints/walking"),
    }
    _write_json_atomic(manifest_path, manifest)
    lock_path = jobs_root / ".gpu-training.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

        remote_training_loop._recover_interrupted_jobs(remote_repo)

    assert json.loads(manifest_path.read_text()) == manifest


def test_gpu_restart_accepts_a_complete_result_written_before_crash(
    tmp_path: Path,
) -> None:
    remote_repo = tmp_path / "wildrobot"
    jobs_root = tmp_path / "wildrobot-training-jobs"
    job_root = jobs_root / "walking-job"
    artifact_root = job_root / "artifacts"
    checkpoint_series = artifact_root / "checkpoints/walking"
    checkpoint_run = checkpoint_series / "walking-runid123"
    checkpoint_run.mkdir(parents=True)
    checkpoint = checkpoint_run / "checkpoint_10.pkl"
    checkpoint.write_bytes(b"policy")
    (checkpoint_run / "post_training_eval_summary.json").write_text(
        json.dumps({"selected_checkpoint_path": str(checkpoint)})
    )
    wandb_run = artifact_root / "wandb/offline-run-20260902-runid123"
    wandb_run.mkdir(parents=True)
    manifest = {
        "job_id": "walking-job",
        "status": "running",
        "remote_repo": str(remote_repo),
        "job_root": str(job_root),
        "worktree": str(job_root / "src"),
        "artifact_root": str(artifact_root),
        "checkpoint_series_dir": str(checkpoint_series),
    }
    manifest_path = job_root / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)

    remote_training_loop._recover_interrupted_jobs(remote_repo)

    recovered = json.loads(manifest_path.read_text())
    assert recovered["status"] == "completed"
    assert recovered["result_complete"] is True
    assert recovered["recovery_action"] == "accepted_complete_result"
    assert (checkpoint_run / CHECKPOINT_MANIFEST_NAME).is_file()


def test_streamed_command_prints_and_logs_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "worker.log"

    returncode = remote_training_loop._run_streamed(
        [sys.executable, "-c", "print('live output')"],
        cwd=tmp_path,
        log_path=log_path,
        timeout_s=5,
    )

    assert returncode == 0
    assert "live output" in capsys.readouterr().out
    assert log_path.read_text() == "live output\n"


def test_prepare_worker_freezes_config_and_checkpoint_hash(tmp_path: Path) -> None:
    remote_repo = tmp_path / "wildrobot"
    worktree = tmp_path / "job" / "src"
    worktree.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=worktree, check=True
    )
    config = worktree / "training" / "configs" / "walking.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "checkpoints": {"dir": "training/checkpoints/walking"},
                "wandb": {"enabled": True, "mode": "offline"},
            },
            sort_keys=False,
        )
    )
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=worktree, check=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    python_path = remote_repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python")
    checkpoint = remote_repo / "training" / "checkpoints" / "source.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"source policy")
    job_root = worktree.parent
    artifact_root = job_root / "artifacts"
    manifest = {
        "git_sha": git_sha,
        "remote_repo": str(remote_repo),
        "worktree": str(worktree),
        "job_root": str(job_root),
        "artifact_root": str(artifact_root),
        "source_config": "training/configs/walking.yaml",
        "checkpoint_series_dir": str(artifact_root / "checkpoints" / "walking"),
        "start_mode": "init_policy",
        "start_checkpoint_request": "training/checkpoints/source.pkl",
    }

    command = remote_training_loop._prepare_worker(manifest)

    assert command[0] == str(python_path)
    assert command[-2:] == ["--init-policy", str(checkpoint)]
    assert manifest["start_checkpoint_sha256"] == remote_training_loop._sha256(
        checkpoint
    )
    effective = yaml.safe_load(Path(manifest["effective_config"]).read_text())
    assert effective["wandb"]["log_dir"] == str(artifact_root / "wandb")


def test_prepare_worker_runs_configured_distillation_before_training(
    tmp_path: Path,
) -> None:
    remote_repo = tmp_path / "wildrobot"
    worktree = tmp_path / "job" / "src"
    worktree.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=worktree, check=True
    )
    config = worktree / "training/configs/walking.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "bootstrap": {"mode": "contact_observed_to_proprio"},
                "checkpoints": {"dir": "training/checkpoints/walking"},
                "wandb": {"enabled": True, "mode": "offline"},
            },
            sort_keys=False,
        )
    )
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=worktree, check=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_path = remote_repo / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python")
    job_root = worktree.parent
    artifact_root = job_root / "artifacts"
    manifest = {
        "git_sha": git_sha,
        "remote_repo": str(remote_repo),
        "worktree": str(worktree),
        "job_root": str(job_root),
        "artifact_root": str(artifact_root),
        "source_config": "training/configs/walking.yaml",
        "checkpoint_series_dir": str(artifact_root / "checkpoints/walking"),
        "start_mode": None,
        "start_checkpoint_request": None,
    }

    command = remote_training_loop._prepare_worker(manifest)

    assert command[1] == "training/scripts/train_with_contact_distillation.py"
    assert manifest["bootstrap_mode"] == "contact_observed_to_proprio"
    assert manifest["bootstrap_status"] == "pending"
    assert manifest["bootstrap_report"].endswith(
        "bootstrap/contact_free_distilled.metrics.json"
    )


def test_prepare_worker_builds_walking_candidate_evaluation(tmp_path: Path) -> None:
    remote_repo = tmp_path / "wildrobot"
    worktree = tmp_path / "job/src"
    worktree.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=worktree, check=True
    )
    config = worktree / "training/configs/walking.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "checkpoints": {"dir": "training/checkpoints/walking"},
                "wandb": {"enabled": True, "mode": "offline"},
            },
            sort_keys=False,
        )
    )
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=worktree, check=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_path = remote_repo / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("python")
    checkpoint = tmp_path / "source-job/artifacts/checkpoints/run/checkpoint.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"policy")
    artifact_root = worktree.parent / "artifacts"
    manifest = {
        "job_kind": remote_training_loop.EVALUATION_JOB_KIND,
        "git_sha": git_sha,
        "remote_repo": str(remote_repo),
        "worktree": str(worktree),
        "job_root": str(worktree.parent),
        "artifact_root": str(artifact_root),
        "source_config": "training/configs/walking.yaml",
        "checkpoint_series_dir": str(artifact_root / "checkpoints/walking"),
        "start_mode": "init_policy",
        "start_checkpoint_request": str(checkpoint),
        "evaluation_purpose": "confirmation",
        "evaluation_seeds": [31000, 41000, 51000, 61000],
        "evaluation_num_envs": 64,
        "evaluation_num_steps": 1000,
    }

    command = remote_training_loop._prepare_worker(manifest)

    assert command[1] == "wildrobot/agents/evaluate_walking_candidate.py"
    assert command[command.index("--purpose") + 1] == "confirmation"
    assert command[command.index("--seeds") + 1] == "31000,41000,51000,61000"
    assert manifest["evaluation_status"] == "pending"


def test_collect_walking_evaluation_results_requires_confirmation_pass(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation_summary.json"
    report.write_text(json.dumps({"aggregate": {"passed": False}}))
    manifest = {
        "evaluation_purpose": "confirmation",
        "evaluation_report": str(report),
        "selected_checkpoint_path": "/srv/source/checkpoint.pkl",
    }

    _collect_walking_evaluation_results(manifest)

    assert manifest["result_complete"] is True
    assert manifest["evaluation_gates_passed"] is False
    assert manifest["confirmation_passed"] is False
    assert manifest["simulation_candidate_ready"] is False


def _completed_manual_run(
    repo: Path,
    *,
    run_name: str,
    checkpoint_series: str = "walking",
    complete: bool = True,
) -> tuple[Path, Path]:
    run_id = run_name.rsplit("-", 1)[-1]
    wandb_run = repo / "training/wandb" / run_name
    files = wandb_run / "files"
    files.mkdir(parents=True)
    (files / "metrics.jsonl").write_text("{}\n")
    (files / "config.json").write_text(
        json.dumps({"config": {"version": "0.21.0-test"}})
    )
    checkpoint_run = (
        repo
        / "training/checkpoints"
        / checkpoint_series
        / f"walking_v0210_20260902-{run_id}"
    )
    checkpoint_run.mkdir(parents=True)
    (checkpoint_run / "training_config.yaml").write_text(
        yaml.safe_dump({"version": "0.21.0-test"})
    )
    if complete:
        checkpoint = checkpoint_run / "checkpoint_10_204800.pkl"
        checkpoint.write_bytes(b"policy")
        (checkpoint_run / "post_training_eval_summary.json").write_text(
            json.dumps({"selected_checkpoint_path": str(checkpoint)})
        )
    return wandb_run, checkpoint_run


def test_find_completed_run_skips_newer_incomplete_run(tmp_path: Path) -> None:
    repo = tmp_path / "wildrobot"
    completed, checkpoint_run = _completed_manual_run(
        repo, run_name="offline-run-20260902_120000-complete1"
    )
    incomplete, _ = _completed_manual_run(
        repo,
        run_name="offline-run-20260902_130000-running2",
        complete=False,
    )
    completed.touch()
    incomplete.touch()

    found_wandb, found_checkpoint = _find_completed_run(repo, run_name=None)

    assert found_wandb == completed
    assert found_checkpoint == checkpoint_run


def test_find_completed_run_rejects_non_run_path(tmp_path: Path) -> None:
    repo = tmp_path / "wildrobot"
    with pytest.raises(TrainingLoopError, match="Invalid W&B run name"):
        _find_completed_run(repo, run_name="../offline-run-escape")


def test_adopt_completed_run_publishes_syncable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "wildrobot"
    config = repo / "training/configs/walking.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "version": "0.21.0-test",
                "checkpoints": {"dir": "training/checkpoints/walking"},
            }
        )
    )
    wandb_run, checkpoint_run = _completed_manual_run(
        repo, run_name="offline-run-20260902_120000-runid123"
    )
    git_sha = "a" * 40
    monkeypatch.setattr(
        remote_training_loop,
        "_git_output",
        lambda *args, **kwargs: git_sha if args[:2] == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        remote_training_loop,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )

    manifest = _adopt_completed_run(
        remote_repo=repo,
        job_id="adopt-runid123",
        source_config="training/configs/walking.yaml",
        expected_git_sha=git_sha,
        run_name=wandb_run.name,
    )

    assert manifest["status"] == "completed"
    assert manifest["adopted_from_existing_run"] is True
    assert manifest["git_sha"] == git_sha
    assert manifest["wandb_run_name"] == wandb_run.name
    assert manifest["checkpoint_run_dir"] == str(checkpoint_run)
    assert manifest["simulation_candidate_ready"] is True
    assert Path(manifest["selected_checkpoint_path"]).is_file()
    assert Path(manifest["job_root"], MANIFEST_NAME).is_file()
    assert (checkpoint_run / CHECKPOINT_MANIFEST_NAME).is_file()
