#!/usr/bin/env python3
"""Supervised Mac-to-GPU training loop for WildRobot."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_REL_PATH = "wildrobot/agents/remote_training_loop.py"
DEFAULT_GPU_HOST = "linux-pc.local"
DEFAULT_GPU_USER = "leeygang"
DEFAULT_REMOTE_REPO = "/home/leeygang/projects/wildrobot"
LOCAL_JOB_ROOT = REPO_ROOT / "training" / "remote_jobs"
MANIFEST_NAME = "job_manifest.json"
CHECKPOINT_MANIFEST_NAME = "training_job_manifest.json"
ADOPTION_CONTROL_PLANE_PATHS = (
    ".gitignore",
    "tests/test_autonomous_training_loop.py",
    "tests/test_remote_training_loop.py",
    "training/exports/export_policy_bundle.py",
    "wildrobot/agents/",
)


class TrainingLoopError(RuntimeError):
    """A user-actionable orchestration failure."""


@dataclass(frozen=True)
class RemoteContext:
    job_id: str
    host: str
    user: str
    port: int | None
    remote_repo: str

    @property
    def jobs_root(self) -> str:
        repo = PurePosixPath(self.remote_repo)
        return str(repo.parent / f"{repo.name}-training-jobs")

    @property
    def job_root(self) -> str:
        return f"{self.jobs_root}/{self.job_id}"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TrainingLoopError(f"Missing job manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TrainingLoopError(f"Invalid job manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingLoopError(f"Job manifest must be a JSON object: {path}")
    return payload


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def _run_streamed(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
    append: bool = False,
) -> int:
    """Run a child process while teeing merged output to console and a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None

        def copy_output() -> None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            if timeout_s is None:
                returncode = process.wait()
            else:
                returncode = process.wait(timeout=timeout_s)
        except BaseException:
            process.kill()
            process.wait()
            reader.join(timeout=5)
            raise
        reader.join()
        return int(returncode)


def _git_output(*args: str, cwd: Path = REPO_ROOT) -> str:
    return _run(["git", *args], cwd=cwd).stdout.strip()


def _safe_job_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    if not safe or safe != value or len(safe) > 100:
        raise TrainingLoopError(
            "Job IDs may contain only letters, digits, '_' and '-' (max 100)."
        )
    return safe


def _new_job_id(config_path: str, git_sha: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return _safe_job_id(f"{Path(config_path).stem}-{git_sha[:8]}-{timestamp}")


def _checkpoint_series(checkpoint_dir: str) -> str:
    path = PurePosixPath(checkpoint_dir)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingLoopError("Checkpoint directory must be repository-relative.")
    parts = path.parts
    if parts[:2] == ("training", "checkpoints"):
        parts = parts[2:]
    if not parts:
        raise TrainingLoopError("Checkpoint directory must name a training series.")
    return PurePosixPath(*parts).as_posix()


def _checkpoint_artifact_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise TrainingLoopError(f"Unsafe checkpoint artifact path: {value}")
    if path.parts[0] != "checkpoints":
        raise TrainingLoopError(f"Unexpected checkpoint artifact path: {value}")
    return path


def _repo_config(path_text: str) -> str:
    path = Path(path_text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise TrainingLoopError(f"Config must be inside the repository: {path_text}") from exc
    if not resolved.is_file():
        raise TrainingLoopError(f"Config not found: {relative}")
    if _run(
        ["git", "ls-files", "--error-unmatch", relative.as_posix()],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise TrainingLoopError(f"Training config must be tracked by Git: {relative}")
    return relative.as_posix()


def _config_checkpoint_series(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text())
    checkpoints = config.get("checkpoints") if isinstance(config, dict) else None
    if not isinstance(checkpoints, dict) or not checkpoints.get("dir"):
        raise TrainingLoopError(f"Training config has no checkpoints.dir: {config_path}")
    return _checkpoint_series(str(checkpoints["dir"]))


def _ssh_command(context: RemoteContext, remote_command: str) -> list[str]:
    command = ["ssh"]
    if context.port is not None:
        command.extend(["-p", str(context.port)])
    command.extend([context.target, remote_command])
    return command


def _rsync_command(context: RemoteContext) -> list[str]:
    command = ["rsync", "-az"]
    if context.port is not None:
        command.extend(["-e", f"ssh -p {context.port}"])
    return command


def _shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _initial_manifest(
    *,
    context: RemoteContext,
    git_sha: str,
    config: str,
    checkpoint_series: str,
    init_policy: str | None,
    resume: str | None,
) -> dict[str, Any]:
    job_root = context.job_root
    artifact_root = f"{job_root}/artifacts"
    return {
        "schema_version": 1,
        "job_id": context.job_id,
        "status": "queued",
        "created_at": _utc_now(),
        "git_sha": git_sha,
        "git_dirty": False,
        "remote_repo": context.remote_repo,
        "worktree": f"{job_root}/src",
        "job_root": job_root,
        "artifact_root": artifact_root,
        "source_config": config,
        "checkpoint_series": checkpoint_series,
        "checkpoint_series_dir": f"{artifact_root}/checkpoints/{checkpoint_series}",
        "start_mode": "init_policy" if init_policy else "resume" if resume else None,
        "start_checkpoint_request": init_policy or resume,
        "systemd_unit": f"wildrobot-train-{context.job_id}"[:240],
        "simulation_candidate_ready": False,
    }


def _build_remote_submit_script(
    *,
    remote_repo: str,
    jobs_root: str,
    job_id: str,
    git_sha: str,
    config: str,
    checkpoint_dir: str,
    init_policy: str | None,
    resume: str | None,
) -> str:
    context = RemoteContext(job_id, "unused", "unused", None, remote_repo)
    if context.jobs_root != jobs_root:
        raise TrainingLoopError(
            f"Jobs root must be derived from the remote repository: {context.jobs_root}"
        )
    manifest = _initial_manifest(
        context=context,
        git_sha=git_sha,
        config=config,
        checkpoint_series=checkpoint_dir,
        init_policy=init_policy,
        resume=resume,
    )
    return _build_remote_enqueue_script(manifest)


def _build_remote_enqueue_script(manifest: dict[str, Any]) -> str:
    job_root = str(manifest["job_root"])
    remote_repo = str(manifest["remote_repo"])
    manifest_path = f"{job_root}/{MANIFEST_NAME}"
    temporary_path = f"{job_root}/.{MANIFEST_NAME}.tmp"
    encoded = base64.b64encode(
        (json.dumps(manifest, sort_keys=True) + "\n").encode()
    ).decode()
    return "\n".join(
        [
            "set -eu",
            f"test ! -e {shlex.quote(job_root)} || "
            f"{{ echo 'Job already exists: {manifest['job_id']}' >&2; exit 2; }}",
            f"git -C {shlex.quote(remote_repo)} fetch origin",
            f"git -C {shlex.quote(remote_repo)} cat-file -e "
            f"{shlex.quote(str(manifest['git_sha']) + '^{commit}')}",
            f"mkdir -p {shlex.quote(job_root)}",
            f"printf %s {shlex.quote(encoded)} | base64 -d > "
            f"{shlex.quote(temporary_path)}",
            f"mv {shlex.quote(temporary_path)} {shlex.quote(manifest_path)}",
        ]
    )


def _active_job_path() -> Path:
    return LOCAL_JOB_ROOT / "active_job.json"


def _save_active_context(context: RemoteContext) -> None:
    _write_json_atomic(_active_job_path(), asdict(context))


def _context_from_args(args: argparse.Namespace) -> RemoteContext:
    saved: dict[str, Any] = {}
    try:
        saved = _read_json(_active_job_path())
    except TrainingLoopError:
        pass
    job_id = args.job_id or saved.get("job_id")
    if not job_id:
        raise TrainingLoopError("Pass --job-id or submit a job first.")
    port = args.port if args.port is not None else saved.get("port")
    return RemoteContext(
        job_id=_safe_job_id(str(job_id)),
        host=str(
            args.host
            or saved.get("host")
            or os.environ.get("WILDROBOT_GPU_HOST", DEFAULT_GPU_HOST)
        ),
        user=str(
            args.user
            or saved.get("user")
            or os.environ.get("WILDROBOT_GPU_USER", DEFAULT_GPU_USER)
        ),
        port=int(port) if port is not None else None,
        remote_repo=str(
            args.remote_repo
            or saved.get("remote_repo")
            or os.environ.get("WILDROBOT_GPU_REPO", DEFAULT_REMOTE_REPO)
        ),
    )


def _fetch_manifest(context: RemoteContext) -> dict[str, Any]:
    path = f"{context.job_root}/{MANIFEST_NAME}"
    result = _run(_ssh_command(context, f"cat {shlex.quote(path)}"), check=False)
    if result.returncode:
        raise TrainingLoopError(
            f"Could not read remote job {context.job_id}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TrainingLoopError(f"Remote manifest is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise TrainingLoopError(f"Remote manifest must be a JSON object: {path}")
    return payload


def _enqueue_remote(
    context: RemoteContext,
    *,
    git_sha: str,
    config: str,
    checkpoint_series: str,
    init_policy: str | None,
    resume: str | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = _initial_manifest(
        context=context,
        git_sha=git_sha,
        config=config,
        checkpoint_series=checkpoint_series,
        init_policy=init_policy,
        resume=resume,
    )
    script = _build_remote_enqueue_script(manifest)
    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(script)
        return manifest
    result = _run(_ssh_command(context, script), check=False)
    if result.returncode:
        raise TrainingLoopError(
            f"GPU enqueue failed: {(result.stderr or result.stdout).strip()}"
        )
    _save_active_context(context)
    return manifest


def _adopt_remote(
    context: RemoteContext,
    *,
    git_sha: str,
    config: str,
    run_name: str | None,
) -> dict[str, Any]:
    command = [
        f"{context.remote_repo}/.venv/bin/python",
        f"{context.remote_repo}/{AGENT_REL_PATH}",
        "_adopt-completed-worker",
        "--remote-repo",
        context.remote_repo,
        "--job-id",
        context.job_id,
        "--config",
        config,
        "--training-git-sha",
        git_sha,
    ]
    if run_name is not None:
        command.extend(["--run-name", run_name])
    result = _run(_ssh_command(context, _shell_join(command)), check=False)
    if result.returncode:
        raise TrainingLoopError(
            "GPU completed-run adoption failed: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TrainingLoopError("GPU adoption returned an invalid manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise TrainingLoopError("GPU adoption did not return a completed job manifest.")
    _save_active_context(context)
    return manifest


def _remote_exists(context: RemoteContext, path: str, *, directory: bool) -> bool:
    flag = "-d" if directory else "-f"
    return (
        _run(
            _ssh_command(context, f"test {flag} {shlex.quote(path)}"),
            check=False,
        ).returncode
        == 0
    )


def _rsync_file(context: RemoteContext, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    command = _rsync_command(context)
    command.extend([f"{context.target}:{remote}", str(local)])
    if _run(command, capture_output=False, check=False).returncode:
        raise TrainingLoopError(f"rsync failed for {remote}")


def _rsync_tree(
    context: RemoteContext,
    remote: str,
    local: Path,
    includes: Sequence[str],
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    command = _rsync_command(context)
    command.append("--prune-empty-dirs")
    command.extend(f"--include={pattern}" for pattern in includes)
    command.append("--exclude=*")
    command.extend([f"{context.target}:{remote.rstrip('/')}/", f"{local}/"])
    if _run(command, capture_output=False, check=False).returncode:
        raise TrainingLoopError(f"rsync failed for {remote}")


def _resolve_start_checkpoint(manifest: dict[str, Any]) -> Path | None:
    request = manifest.get("start_checkpoint_request")
    if not request:
        return None
    path = Path(str(request))
    if not path.is_absolute():
        path = Path(manifest["remote_repo"]) / path
    path = path.resolve()
    if not path.is_file() or path.suffix != ".pkl":
        raise TrainingLoopError(
            "Remote --init-policy/--resume must resolve to one checkpoint .pkl "
            f"file: {path}"
        )
    return path


def _write_effective_config(
    source_path: Path, destination: Path, *, wandb_log_dir: Path
) -> None:
    config = yaml.safe_load(source_path.read_text())
    if not isinstance(config, dict):
        raise TrainingLoopError(f"Training config is not a mapping: {source_path}")
    wandb = config.setdefault("wandb", {})
    if not isinstance(wandb, dict):
        raise TrainingLoopError(f"wandb config is not a mapping: {source_path}")
    wandb["log_dir"] = str(wandb_log_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def _gpu_name() -> str | None:
    try:
        result = _run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _latest_dir(root: Path, prefixes: tuple[str, ...] | None = None) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if prefixes:
        candidates = [
            path for path in candidates if any(path.name.startswith(p) for p in prefixes)
        ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _is_adoption_control_plane_path(path: str) -> bool:
    return any(
        path == allowed or (allowed.endswith("/") and path.startswith(allowed))
        for allowed in ADOPTION_CONTROL_PLANE_PATHS
    )


def _wandb_run_id(run_name: str) -> str:
    if Path(run_name).name != run_name:
        raise TrainingLoopError(f"Invalid W&B run name: {run_name}")
    match = re.fullmatch(r"(?:offline-)?run-.+-([a-zA-Z0-9]+)", run_name)
    if match is None:
        raise TrainingLoopError(f"Invalid W&B run name: {run_name}")
    return match.group(1)


def _find_completed_run(
    remote_repo: Path, *, run_name: str | None
) -> tuple[Path, Path]:
    wandb_root = remote_repo / "training/wandb"
    checkpoint_root = remote_repo / "training/checkpoints"
    if run_name is not None:
        _wandb_run_id(run_name)
        candidates = [wandb_root / run_name]
    else:
        candidates = sorted(
            (
                path
                for path in wandb_root.iterdir()
                if path.is_dir()
                and path.name.startswith(("offline-run-", "run-"))
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if wandb_root.is_dir() else []

    for wandb_run in candidates:
        if not wandb_run.is_dir():
            continue
        run_id = _wandb_run_id(wandb_run.name)
        if not (wandb_run / "files/metrics.jsonl").is_file():
            continue
        checkpoint_runs = sorted(
            (
                path
                for path in checkpoint_root.rglob(f"*-{run_id}")
                if path.is_dir()
                and (path / "post_training_eval_summary.json").is_file()
                and (path / "training_config.yaml").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if checkpoint_root.is_dir() else []
        if checkpoint_runs:
            return wandb_run.resolve(), checkpoint_runs[0].resolve()

    requested = run_name or "the latest W&B run"
    raise TrainingLoopError(
        f"Could not find a completed training result for {requested}; required "
        "metrics.jsonl, training_config.yaml, and "
        "post_training_eval_summary.json were not all present."
    )


def _adopt_completed_run(
    *,
    remote_repo: Path,
    job_id: str,
    source_config: str,
    expected_git_sha: str,
    run_name: str | None,
) -> dict[str, Any]:
    remote_repo = remote_repo.resolve()
    expected_git_sha = expected_git_sha.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_git_sha):
        raise TrainingLoopError("--training-git-sha must be a full 40-character SHA.")
    if _git_output("status", "--porcelain", "--untracked-files=normal", cwd=remote_repo):
        raise TrainingLoopError("GPU repository must be clean before adoption.")
    if _run(
        ["git", "cat-file", "-e", f"{expected_git_sha}^{{commit}}"],
        cwd=remote_repo,
        check=False,
    ).returncode:
        raise TrainingLoopError(
            f"Training commit is not available on the GPU: {expected_git_sha}"
        )
    current_git_sha = _git_output("rev-parse", "HEAD", cwd=remote_repo)
    if current_git_sha != expected_git_sha:
        if _run(
            ["git", "merge-base", "--is-ancestor", expected_git_sha, current_git_sha],
            cwd=remote_repo,
            check=False,
        ).returncode:
            raise TrainingLoopError(
                "The declared training commit is not an ancestor of GPU HEAD."
            )
        changed = _git_output(
            "diff", "--name-only", f"{expected_git_sha}..{current_git_sha}", cwd=remote_repo
        ).splitlines()
        disallowed = [path for path in changed if not _is_adoption_control_plane_path(path)]
        if disallowed:
            raise TrainingLoopError(
                "GPU HEAD changed training-relevant files after the completed run: "
                + ", ".join(disallowed)
            )

    config_rel = PurePosixPath(source_config)
    if config_rel.is_absolute() or ".." in config_rel.parts:
        raise TrainingLoopError(f"Unsafe source config: {source_config}")
    config_path = (remote_repo / Path(*config_rel.parts)).resolve()
    if not config_path.is_file():
        raise TrainingLoopError(f"Training config not found on GPU: {source_config}")
    if _run(
        ["git", "ls-files", "--error-unmatch", config_rel.as_posix()],
        cwd=remote_repo,
        check=False,
    ).returncode:
        raise TrainingLoopError(f"Training config is not tracked: {source_config}")

    wandb_run, checkpoint_run = _find_completed_run(
        remote_repo, run_name=run_name
    )
    source_payload = yaml.safe_load(config_path.read_text())
    frozen_config = checkpoint_run / "training_config.yaml"
    frozen_payload = yaml.safe_load(frozen_config.read_text())
    source_version = source_payload.get("version") if isinstance(source_payload, dict) else None
    frozen_version = frozen_payload.get("version") if isinstance(frozen_payload, dict) else None
    if not source_version or source_version != frozen_version:
        raise TrainingLoopError(
            "The supplied source config does not match the completed run's frozen "
            f"version ({source_version!r} != {frozen_version!r})."
        )

    summary_path = checkpoint_run / "post_training_eval_summary.json"
    summary = _read_json(summary_path)
    selected_path: Path | None = None
    if summary.get("selected_checkpoint_path"):
        selected_path = Path(str(summary["selected_checkpoint_path"]))
        if not selected_path.is_absolute():
            selected_path = remote_repo / selected_path
        selected_path = selected_path.resolve()
        try:
            selected_path.relative_to(checkpoint_run)
        except ValueError as exc:
            raise TrainingLoopError(
                "The deterministic summary selected a checkpoint outside its run directory."
            ) from exc
        if not selected_path.is_file() or selected_path.suffix != ".pkl":
            raise TrainingLoopError(
                f"The deterministic selected checkpoint is missing: {selected_path}"
            )

    context = RemoteContext(
        job_id=_safe_job_id(job_id),
        host="localhost",
        user="local",
        port=None,
        remote_repo=str(remote_repo),
    )
    job_root = Path(context.job_root)
    if job_root.exists():
        raise TrainingLoopError(f"Job already exists: {context.job_id}")
    artifact_root = remote_repo / "training"
    checkpoint_root = artifact_root / "checkpoints"
    checkpoint_series_dir = checkpoint_run.parent
    source_snapshot = job_root / "source_training_config.yaml"
    effective_snapshot = job_root / "effective_training_config.yaml"
    job_root.mkdir(parents=True)
    shutil.copy2(config_path, source_snapshot)
    shutil.copy2(frozen_config, effective_snapshot)
    run_id = _wandb_run_id(wandb_run.name)
    manifest = {
        "schema_version": 1,
        "job_id": context.job_id,
        "status": "completed",
        "created_at": _utc_now(),
        "finished_at": datetime.fromtimestamp(
            summary_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "adopted_at": _utc_now(),
        "adopted_from_existing_run": True,
        "provenance_mode": "declared_training_commit",
        "provenance_note": (
            "The run predates the queue manifest. Its Git SHA was supplied at "
            "adoption and checked against the current control-plane-only diff."
        ),
        "git_sha": expected_git_sha,
        "adoption_host_git_sha": current_git_sha,
        "git_dirty": False,
        "remote_repo": str(remote_repo),
        "worktree": str(remote_repo),
        "job_root": str(job_root),
        "artifact_root": str(artifact_root),
        "source_config": config_rel.as_posix(),
        "source_config_sha256": _sha256(source_snapshot),
        "effective_config": str(effective_snapshot),
        "effective_config_sha256": _sha256(effective_snapshot),
        "checkpoint_series": checkpoint_series_dir.relative_to(
            checkpoint_root
        ).as_posix(),
        "checkpoint_series_dir": str(checkpoint_series_dir),
        "checkpoint_run_dir": str(checkpoint_run),
        "checkpoint_run_relpath": checkpoint_run.relative_to(
            artifact_root
        ).as_posix(),
        "wandb_run_dir": str(wandb_run),
        "wandb_run_name": wandb_run.name,
        "wandb_run_id": run_id,
        "post_training_eval_summary": str(summary_path),
        "selected_checkpoint_path": str(selected_path) if selected_path else None,
        "selected_checkpoint_relpath": _artifact_relative(
            selected_path, artifact_root
        ),
        "simulation_candidate_ready": selected_path is not None,
        "result_complete": True,
        "exit_code": 0,
        "start_mode": None,
        "start_checkpoint_request": None,
        "start_checkpoint": None,
        "start_checkpoint_sha256": None,
    }
    manifest_path = job_root / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    _copy_manifest_to_checkpoint(manifest_path, manifest)
    return manifest


def _artifact_relative(path: Path | None, artifact_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError:
        return None


def _collect_training_results(manifest: dict[str, Any]) -> None:
    artifact_root = Path(manifest["artifact_root"])
    checkpoint_run = _latest_dir(Path(manifest["checkpoint_series_dir"]))
    wandb_run = _latest_dir(
        artifact_root / "wandb", prefixes=("offline-run-", "run-")
    )
    manifest.update(
        {
            "checkpoint_run_dir": str(checkpoint_run) if checkpoint_run else None,
            "checkpoint_run_relpath": _artifact_relative(checkpoint_run, artifact_root),
            "wandb_run_dir": str(wandb_run) if wandb_run else None,
            "wandb_run_name": wandb_run.name if wandb_run else None,
            "wandb_run_id": wandb_run.name.rsplit("-", 1)[-1]
            if wandb_run
            else None,
        }
    )

    summary_path = (
        checkpoint_run / "post_training_eval_summary.json"
        if checkpoint_run
        else None
    )
    selected_path: Path | None = None
    if summary_path and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        selected = summary.get("selected_checkpoint_path")
        if selected:
            selected_path = Path(str(selected))
            if not selected_path.is_absolute():
                selected_path = Path(manifest["worktree"]) / selected_path
            selected_path = selected_path.resolve()
    manifest.update(
        {
            "post_training_eval_summary": str(summary_path)
            if summary_path and summary_path.is_file()
            else None,
            "selected_checkpoint_path": str(selected_path)
            if selected_path
            else None,
            "selected_checkpoint_relpath": _artifact_relative(
                selected_path, artifact_root
            ),
            "simulation_candidate_ready": bool(
                selected_path and selected_path.is_file()
            ),
            "result_complete": bool(summary_path and summary_path.is_file()),
        }
    )


def _copy_manifest_to_checkpoint(path: Path, manifest: dict[str, Any]) -> None:
    checkpoint_run = manifest.get("checkpoint_run_dir")
    if checkpoint_run:
        shutil.copy2(path, Path(checkpoint_run) / CHECKPOINT_MANIFEST_NAME)


def _prepare_worker(manifest: dict[str, Any]) -> list[str]:
    worktree = Path(manifest["worktree"]).resolve()
    expected_sha = str(manifest["git_sha"])
    if _git_output("rev-parse", "HEAD", cwd=worktree) != expected_sha:
        raise TrainingLoopError("GPU worktree does not match the submitted Git SHA.")
    if _git_output("status", "--porcelain", cwd=worktree):
        raise TrainingLoopError(f"GPU worktree is dirty: {worktree}")

    config_rel = PurePosixPath(str(manifest["source_config"]))
    if config_rel.is_absolute() or ".." in config_rel.parts:
        raise TrainingLoopError(f"Unsafe source config: {config_rel}")
    source_config = (worktree / Path(*config_rel.parts)).resolve()
    try:
        source_config.relative_to(worktree)
    except ValueError as exc:
        raise TrainingLoopError(f"Source config escapes the worktree: {config_rel}") from exc
    if not source_config.is_file():
        raise TrainingLoopError(f"Config is absent from commit {expected_sha}: {config_rel}")

    job_root = Path(manifest["job_root"])
    artifact_root = Path(manifest["artifact_root"])
    checkpoint_dir = Path(manifest["checkpoint_series_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = job_root / "source_training_config.yaml"
    effective_config = job_root / "effective_training_config.yaml"
    shutil.copy2(source_config, source_snapshot)
    _write_effective_config(
        source_config, effective_config, wandb_log_dir=artifact_root / "wandb"
    )

    checkpoint = _resolve_start_checkpoint(manifest)
    python_path = Path(manifest["remote_repo"]) / ".venv" / "bin" / "python"
    if not python_path.is_file():
        raise TrainingLoopError(f"GPU virtual environment missing: {python_path}")
    command = [
        str(python_path),
        "training/train.py",
        "--config",
        str(effective_config),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if checkpoint is not None:
        command.extend(
            [
                "--init-policy"
                if manifest.get("start_mode") == "init_policy"
                else "--resume",
                str(checkpoint),
            ]
        )
    manifest.update(
        {
            "source_config_sha256": _sha256(source_snapshot),
            "effective_config": str(effective_config),
            "effective_config_sha256": _sha256(effective_config),
            "start_checkpoint": str(checkpoint) if checkpoint else None,
            "start_checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
            "training_command": command,
        }
    )
    return command


def _gpu_worker(args: argparse.Namespace) -> int:
    job_root = Path(args.job_root).resolve()
    manifest_path = job_root / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    lock_path = job_root.parent / ".gpu-training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            manifest.update(
                status="failed",
                finished_at=_utc_now(),
                error="Another WildRobot GPU training job is active.",
            )
            _write_json_atomic(manifest_path, manifest)
            return 2

        try:
            command = _prepare_worker(manifest)
            training_attempt = int(manifest.get("training_attempt", 0)) + 1
            manifest.update(
                status="running",
                started_at=_utc_now(),
                hostname=socket.gethostname(),
                gpu_name=_gpu_name(),
                python_version=sys.version,
                worker_pid=os.getpid(),
                training_attempt=training_attempt,
            )
            manifest.pop("recovery_action", None)
            _write_json_atomic(manifest_path, manifest)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", PYTHONPATH=manifest["worktree"])
            train_log_path = job_root / "train.log"
            with train_log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{_utc_now()}] attempt={training_attempt} "
                    f"{_shell_join(command)}\n"
                )
                log.flush()
            print(f"Starting GPU training job {manifest['job_id']}...", flush=True)
            returncode = _run_streamed(
                command,
                cwd=Path(manifest["worktree"]),
                log_path=train_log_path,
                env=env,
                append=True,
            )
            manifest["exit_code"] = returncode
            _collect_training_results(manifest)
            manifest["status"] = "completed" if returncode == 0 else "failed"
            if returncode:
                manifest["error"] = f"training/train.py exited with {returncode}"
            print(
                f"GPU training job {manifest['job_id']} finished with status "
                f"{manifest['status']}.",
                flush=True,
            )
        except Exception as exc:
            manifest.update(
                status="failed",
                exit_code=1,
                error=f"{type(exc).__name__}: {exc}",
            )
        manifest["finished_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        _copy_manifest_to_checkpoint(manifest_path, manifest)
        return 0 if manifest["status"] == "completed" else 1


def _fail_manifest(path: Path, error: str) -> None:
    manifest = _read_json(path)
    manifest.update(status="failed", finished_at=_utc_now(), error=error)
    _write_json_atomic(path, manifest)


def _validate_queued_manifest(
    manifest: dict[str, Any], manifest_path: Path, remote_repo: Path
) -> None:
    job_root = manifest_path.parent.resolve()
    expected = {
        "remote_repo": remote_repo.resolve(),
        "job_root": job_root,
        "worktree": job_root / "src",
        "artifact_root": job_root / "artifacts",
    }
    for key, expected_path in expected.items():
        if Path(str(manifest.get(key, ""))).resolve() != expected_path:
            raise TrainingLoopError(f"Queued job has an invalid {key} path.")

    series = _checkpoint_series(str(manifest.get("checkpoint_series", "")))
    checkpoint_dir = Path(str(manifest.get("checkpoint_series_dir", ""))).resolve()
    if checkpoint_dir != (job_root / "artifacts/checkpoints" / series).resolve():
        raise TrainingLoopError("Queued job has an invalid checkpoint series path.")


def _ensure_job_worktree(
    remote_repo: Path, worktree: Path, git_sha: str
) -> None:
    if not worktree.exists():
        _run(
            ["git", "worktree", "add", "--detach", str(worktree), git_sha],
            cwd=remote_repo,
        )
        return
    if not worktree.is_dir():
        raise TrainingLoopError(f"GPU worktree path is not a directory: {worktree}")
    if _git_output("rev-parse", "HEAD", cwd=worktree) != git_sha:
        raise TrainingLoopError(
            f"Existing GPU worktree does not match submitted commit: {worktree}"
        )
    if _git_output("status", "--porcelain", cwd=worktree):
        raise TrainingLoopError(f"Existing GPU worktree is dirty: {worktree}")
    print(f"Reusing recovered GPU worktree {worktree}.", flush=True)


def _dispatch_queued_job(remote_repo: Path) -> bool:
    jobs_root = remote_repo.parent / f"{remote_repo.name}-training-jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    for manifest_path in sorted(jobs_root.glob(f"*/{MANIFEST_NAME}")):
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "queued":
            continue
        job_root = manifest_path.parent
        worktree = Path(str(manifest["worktree"]))
        try:
            _validate_queued_manifest(manifest, manifest_path, remote_repo)
            manifest.update(status="dispatching", dispatched_at=_utc_now())
            _write_json_atomic(manifest_path, manifest)
            _run(["git", "fetch", "origin"], cwd=remote_repo)
            git_sha = str(manifest["git_sha"])
            if _run(
                ["git", "cat-file", "-e", f"{git_sha}^{{commit}}"],
                cwd=remote_repo,
                check=False,
            ).returncode:
                raise TrainingLoopError(f"Commit is not available on GPU: {git_sha}")
            _ensure_job_worktree(remote_repo, worktree, git_sha)
            python_path = remote_repo / ".venv" / "bin" / "python"
            if not python_path.is_file():
                raise TrainingLoopError(
                    f"GPU virtual environment missing: {python_path}"
                )
            command = [
                str(python_path),
                str(worktree / AGENT_REL_PATH),
                "_gpu-worker",
                "--job-root",
                str(job_root),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree)
            result = subprocess.run(command, cwd=worktree, env=env, check=False)
            if result.returncode and _read_json(manifest_path).get("status") in {
                "queued",
                "dispatching",
                "running",
            }:
                _fail_manifest(
                    manifest_path, f"GPU worker exited with {result.returncode}"
                )
        except Exception as exc:
            _fail_manifest(manifest_path, f"{type(exc).__name__}: {exc}")
        return True
    return False


def _recover_interrupted_jobs(remote_repo: Path) -> None:
    jobs_root = remote_repo.parent / f"{remote_repo.name}-training-jobs"
    if not jobs_root.is_dir():
        return
    interrupted = []
    for manifest_path in sorted(jobs_root.glob(f"*/{MANIFEST_NAME}")):
        manifest = _read_json(manifest_path)
        if manifest.get("status") in {"dispatching", "running"}:
            interrupted.append((manifest_path, manifest))
    if not interrupted:
        return

    lock_path = jobs_root / ".gpu-training.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        for manifest_path, manifest in interrupted:
            previous_status = str(manifest["status"])
            _collect_training_results(manifest)
            if manifest.get("result_complete"):
                recovery_action = "accepted_complete_result"
                manifest.setdefault("recovery_events", []).append(
                    {
                        "recovered_at": _utc_now(),
                        "previous_status": previous_status,
                        "action": recovery_action,
                    }
                )
                manifest.update(
                    status="completed",
                    finished_at=_utc_now(),
                    recovery_action=recovery_action,
                )
                _write_json_atomic(manifest_path, manifest)
                _copy_manifest_to_checkpoint(manifest_path, manifest)
                print(
                    f"Recovered completed GPU job {manifest['job_id']} from artifacts.",
                    flush=True,
                )
                continue

            for key in (
                "worker_pid",
                "started_at",
                "finished_at",
                "exit_code",
                "error",
                "checkpoint_run_dir",
                "checkpoint_run_relpath",
                "wandb_run_dir",
                "wandb_run_name",
                "wandb_run_id",
                "post_training_eval_summary",
                "selected_checkpoint_path",
                "selected_checkpoint_relpath",
            ):
                manifest.pop(key, None)
            recovery_action = "requeued_same_training_job"
            manifest.setdefault("recovery_events", []).append(
                {
                    "recovered_at": _utc_now(),
                    "previous_status": previous_status,
                    "action": recovery_action,
                }
            )
            manifest.update(
                status="queued",
                simulation_candidate_ready=False,
                result_complete=False,
                recovery_action=recovery_action,
            )
            _write_json_atomic(manifest_path, manifest)
            print(
                f"Requeued interrupted GPU job {manifest['job_id']} from its "
                "declared starting checkpoint.",
                flush=True,
            )


def _gpu_serve(args: argparse.Namespace) -> int:
    remote_repo = Path(args.remote_repo).resolve()
    if args.poll_seconds <= 0:
        raise TrainingLoopError("--poll-seconds must be positive.")
    while True:
        _recover_interrupted_jobs(remote_repo)
        dispatched = _dispatch_queued_job(remote_repo)
        if args.once or (args.exit_when_idle and not dispatched):
            return 0
        if not dispatched:
            time.sleep(args.poll_seconds)


def _install_gpu_service(args: argparse.Namespace) -> int:
    remote_repo = Path(args.remote_repo).resolve()
    python_path = remote_repo / ".venv" / "bin" / "python"
    script_path = remote_repo / AGENT_REL_PATH
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise TrainingLoopError(f"GPU Python is missing or not executable: {python_path}")
    if not script_path.is_file():
        raise TrainingLoopError(f"GPU worker script is missing: {script_path}")
    unit_path = Path.home() / ".config/systemd/user/wildrobot-training-gpu.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=WildRobot GPU training queue worker",
                "After=network-online.target",
                "Wants=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={remote_repo}",
                f"Environment=PYTHONPATH={remote_repo}",
                f"ExecStart={python_path} {script_path} gpu-serve "
                f"--remote-repo {remote_repo}",
                "Restart=on-failure",
                "RestartSec=10",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
    )
    print(f"Wrote {unit_path}")
    print("Start manually with:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now wildrobot-training-gpu.service")
    return 0


def _submit(args: argparse.Namespace) -> int:
    if _git_output("status", "--porcelain", "--untracked-files=normal"):
        raise TrainingLoopError("Commit or remove local changes before submission.")
    config = _repo_config(args.config)
    git_sha = _git_output("rev-parse", "HEAD")
    context = RemoteContext(
        job_id=_safe_job_id(args.job_id)
        if args.job_id
        else _new_job_id(config, git_sha),
        host=args.host or os.environ.get("WILDROBOT_GPU_HOST", DEFAULT_GPU_HOST),
        user=args.user or os.environ.get("WILDROBOT_GPU_USER", DEFAULT_GPU_USER),
        port=args.port,
        remote_repo=args.remote_repo
        or os.environ.get("WILDROBOT_GPU_REPO", DEFAULT_REMOTE_REPO),
    )
    script = _build_remote_submit_script(
        remote_repo=context.remote_repo,
        jobs_root=context.jobs_root,
        job_id=context.job_id,
        git_sha=git_sha,
        config=config,
        checkpoint_dir=_config_checkpoint_series(REPO_ROOT / config),
        init_policy=args.init_policy,
        resume=args.resume,
    )
    if args.dry_run:
        print(json.dumps({**asdict(context), "git_sha": git_sha}, indent=2))
        print(script)
        return 0
    result = _run(_ssh_command(context, script), check=False)
    if result.returncode:
        raise TrainingLoopError(
            f"GPU submission failed: {(result.stderr or result.stdout).strip()}"
        )
    _save_active_context(context)
    print(result.stdout.strip())
    print(f"Queued {context.job_id} at commit {git_sha[:12]}")
    return 0


def _adopt(args: argparse.Namespace) -> int:
    if _git_output("status", "--porcelain", "--untracked-files=normal"):
        raise TrainingLoopError("Commit or remove local changes before adoption.")
    config = _repo_config(args.config)
    git_sha = args.training_git_sha or _git_output("rev-parse", "HEAD")
    context = RemoteContext(
        job_id=_safe_job_id(args.job_id)
        if args.job_id
        else _new_job_id(config, git_sha),
        host=args.host or os.environ.get("WILDROBOT_GPU_HOST", DEFAULT_GPU_HOST),
        user=args.user or os.environ.get("WILDROBOT_GPU_USER", DEFAULT_GPU_USER),
        port=args.port,
        remote_repo=args.remote_repo
        or os.environ.get("WILDROBOT_GPU_REPO", DEFAULT_REMOTE_REPO),
    )
    manifest = _adopt_remote(
        context,
        git_sha=git_sha,
        config=config,
        run_name=args.run_name,
    )
    print(
        f"Adopted completed run {manifest['wandb_run_name']} as job "
        f"{context.job_id} at training commit {git_sha[:12]}"
    )
    return 0


def _adopt_completed_worker(args: argparse.Namespace) -> int:
    manifest = _adopt_completed_run(
        remote_repo=Path(args.remote_repo),
        job_id=args.job_id,
        source_config=args.config,
        expected_git_sha=args.training_git_sha,
        run_name=args.run_name,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    manifest = _fetch_manifest(_context_from_args(args))
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    for label, key in (
        ("Job", "job_id"),
        ("Status", "status"),
        ("Commit", "git_sha"),
        ("Config", "source_config"),
        ("Started", "started_at"),
        ("Finished", "finished_at"),
        ("Training attempt", "training_attempt"),
        ("Recovery action", "recovery_action"),
        ("W&B run", "wandb_run_name"),
        ("Selected checkpoint", "selected_checkpoint_path"),
        ("Error", "error"),
    ):
        if manifest.get(key) is not None:
            print(f"{label}: {manifest[key]}")
    print(f"Simulation candidate ready: {bool(manifest.get('simulation_candidate_ready'))}")
    return 0


def _sync_job(
    context: RemoteContext, *, selected_checkpoint: bool
) -> dict[str, Any]:
    manifest = _fetch_manifest(context)
    local_job = LOCAL_JOB_ROOT / context.job_id
    _write_json_atomic(local_job / MANIFEST_NAME, manifest)
    for name in (
        "train.log",
        "source_training_config.yaml",
        "effective_training_config.yaml",
    ):
        remote = f"{context.job_root}/{name}"
        if _remote_exists(context, remote, directory=False):
            _rsync_file(context, remote, local_job / name)

    if manifest.get("wandb_run_dir") and manifest.get("wandb_run_name"):
        run_name = _safe_job_id(str(manifest["wandb_run_name"]))
        destination = REPO_ROOT / "training" / "wandb" / run_name
        _rsync_tree(
            context,
            str(manifest["wandb_run_dir"]),
            destination,
            (
                "*/",
                "config.json",
                "metrics.jsonl",
                "wandb-summary.json",
                "wandb-metadata.json",
            ),
        )
        manifest["local_wandb_run_dir"] = str(destination.relative_to(REPO_ROOT))

    if manifest.get("checkpoint_run_dir") and manifest.get("checkpoint_run_relpath"):
        relative = _checkpoint_artifact_relative(
            str(manifest["checkpoint_run_relpath"])
        )
        destination = REPO_ROOT / "training" / "checkpoints" / Path(*relative.parts[1:])
        _rsync_tree(
            context,
            str(manifest["checkpoint_run_dir"]),
            destination,
            ("*/", "*.json", "*.yaml"),
        )
        manifest["local_checkpoint_run_dir"] = str(destination.relative_to(REPO_ROOT))

    if selected_checkpoint:
        if not manifest.get("simulation_candidate_ready"):
            raise TrainingLoopError(
                "The deterministic selector did not promote a checkpoint."
            )
        relative = _checkpoint_artifact_relative(
            str(manifest["selected_checkpoint_relpath"])
        )
        destination = REPO_ROOT / "training" / "checkpoints" / Path(*relative.parts[1:])
        _rsync_file(context, str(manifest["selected_checkpoint_path"]), destination)
        manifest["local_selected_checkpoint"] = str(destination.relative_to(REPO_ROOT))

    _write_json_atomic(local_job / MANIFEST_NAME, manifest)
    return manifest


def _sync(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    manifest = _sync_job(context, selected_checkpoint=args.selected_checkpoint)
    print(f"Synced summary artifacts for {context.job_id}.")
    for key in (
        "local_wandb_run_dir",
        "local_checkpoint_run_dir",
        "local_selected_checkpoint",
    ):
        if manifest.get(key):
            print(manifest[key])
    return 0


def _analyze(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    manifest = _sync_job(context, selected_checkpoint=False)
    run_dir = manifest.get("local_wandb_run_dir")
    if not run_dir:
        raise TrainingLoopError("The job has no synchronized W&B run yet.")
    command = [
        shutil.which("uv") or "uv",
        "run",
        "python",
        "skills/wildrobot-training-analyze/scripts/analyze_offline_run.py",
        "--run-dir",
        str(REPO_ROOT / run_dir),
        "--wandb-root",
        str(REPO_ROOT / "training" / "wandb"),
        "--checkpoints-root",
        str(REPO_ROOT / "training" / "checkpoints"),
    ]
    result = _run(command, cwd=REPO_ROOT, check=False)
    report = result.stdout + ("\n" + result.stderr if result.stderr else "")
    report_path = LOCAL_JOB_ROOT / context.job_id / "analysis.txt"
    report_path.write_text(report)
    print(report, end="" if report.endswith("\n") else "\n")
    print(f"Analysis saved to {report_path.relative_to(REPO_ROOT)}")
    if result.returncode:
        raise TrainingLoopError(f"Training analyzer exited with {result.returncode}")
    return 0


def _add_remote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id")
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--port", type=int)
    parser.add_argument("--remote-repo")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised WildRobot Mac-to-GPU training loop"
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar=(
            "{submit,adopt-completed,status,sync,analyze,gpu-serve,"
            "install-gpu-service}"
        ),
    )
    submit = commands.add_parser("submit", help="submit an exact Git commit")
    submit.add_argument("--config", required=True)
    start = submit.add_mutually_exclusive_group()
    start.add_argument("--init-policy")
    start.add_argument("--resume")
    submit.add_argument("--job-id")
    submit.add_argument("--host")
    submit.add_argument("--user")
    submit.add_argument("--port", type=int)
    submit.add_argument("--remote-repo")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=_submit)

    adopt = commands.add_parser(
        "adopt-completed",
        help="adopt an already-completed GPU run into the job queue",
    )
    adopt.add_argument("--config", required=True)
    adopt.add_argument("--run-name")
    adopt.add_argument("--training-git-sha")
    adopt.add_argument("--job-id")
    adopt.add_argument("--host")
    adopt.add_argument("--user")
    adopt.add_argument("--port", type=int)
    adopt.add_argument("--remote-repo")
    adopt.set_defaults(func=_adopt)

    status = commands.add_parser("status", help="show the remote job manifest")
    _add_remote_args(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_status)

    sync = commands.add_parser("sync", help="sync analysis-sized artifacts")
    _add_remote_args(sync)
    sync.add_argument("--selected-checkpoint", action="store_true")
    sync.set_defaults(func=_sync)

    analyze = commands.add_parser("analyze", help="sync and run the analyzer")
    _add_remote_args(analyze)
    analyze.set_defaults(func=_analyze)

    gpu_serve = commands.add_parser(
        "gpu-serve", help="run the Ubuntu queued-job worker"
    )
    gpu_serve.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    gpu_serve.add_argument("--poll-seconds", type=float, default=10.0)
    gpu_serve.add_argument("--once", action="store_true")
    gpu_serve.add_argument("--exit-when-idle", action="store_true")
    gpu_serve.set_defaults(func=_gpu_serve)

    install_gpu = commands.add_parser(
        "install-gpu-service", help="write the Ubuntu user-systemd unit"
    )
    install_gpu.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    install_gpu.set_defaults(func=_install_gpu_service)

    worker = commands.add_parser("_gpu-worker", add_help=False)
    worker.add_argument("--job-root", required=True)
    worker.set_defaults(func=_gpu_worker)

    adopt_worker = commands.add_parser("_adopt-completed-worker", add_help=False)
    adopt_worker.add_argument("--remote-repo", required=True)
    adopt_worker.add_argument("--job-id", required=True)
    adopt_worker.add_argument("--config", required=True)
    adopt_worker.add_argument("--training-git-sha", required=True)
    adopt_worker.add_argument("--run-name")
    adopt_worker.set_defaults(func=_adopt_completed_worker)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        return int(args.func(args))
    except (FileNotFoundError, TrainingLoopError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
