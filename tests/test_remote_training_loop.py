from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from training.exports.export_policy_bundle import _load_training_job_provenance
from wildrobot.agents import remote_training_loop
from wildrobot.agents.remote_training_loop import (
    CHECKPOINT_MANIFEST_NAME,
    MANIFEST_NAME,
    TrainingLoopError,
    _build_remote_submit_script,
    _checkpoint_artifact_relative,
    _checkpoint_series,
    _collect_training_results,
    _copy_manifest_to_checkpoint,
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

    def fake_run(*_args, **_kwargs):
        checkpoint_run = checkpoint_series / "walking_v0210_20260902-runid123"
        checkpoint_run.mkdir(parents=True)
        (checkpoint_run / "post_training_eval_summary.json").write_text(
            json.dumps({"selected_checkpoint_path": None})
        )
        (artifact_root / "wandb" / "offline-run-20260902_120000-runid123").mkdir(
            parents=True
        )
        return remote_training_loop.subprocess.CompletedProcess(
            args=["fake-training"], returncode=0
        )

    monkeypatch.setattr(remote_training_loop.subprocess, "run", fake_run)

    result = remote_training_loop._gpu_worker(
        argparse.Namespace(job_root=str(job_root))
    )

    completed = json.loads((job_root / MANIFEST_NAME).read_text())
    assert result == 0
    assert completed["status"] == "completed"
    assert completed["simulation_candidate_ready"] is False
    assert completed["selected_checkpoint_path"] is None
    checkpoint_manifest = (
        checkpoint_series
        / "walking_v0210_20260902-runid123"
        / CHECKPOINT_MANIFEST_NAME
    )
    assert checkpoint_manifest.is_file()


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
