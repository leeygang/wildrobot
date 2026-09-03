#!/usr/bin/env python3
"""Bounded autonomous WildRobot train-analyze-change loop for macOS."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wildrobot.agents import remote_training_loop as remote


REPO_ROOT = _REPO_ROOT
STATE_PATH = remote.LOCAL_JOB_ROOT / "autonomous_state.json"
LOCK_PATH = remote.LOCAL_JOB_ROOT / ".autonomous.lock"
DECISION_SCHEMA = Path(__file__).with_name("autonomous_decision.schema.json")
CODEX_INSTRUCTIONS = Path(__file__).with_name("AUTONOMOUS_CODEX_PROMPT.md")
AUTOMATION_PREFIX = "wildrobot/agents/"
FORBIDDEN_AUTONOMOUS_CHANGES = (AUTOMATION_PREFIX, "training/CHANGELOG.md")
PIPELINE_STAGES = {
    "adopt",
    "training",
    "analysis",
    "fix",
    "push",
    "enqueue",
    "export",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _git_output(*args: str) -> str:
    return _git(*args).stdout.strip()


def _require_clean_branch(branch: str) -> str:
    if _git_output("branch", "--show-current") != branch:
        raise remote.TrainingLoopError(f"Automation requires branch {branch!r}.")
    if _git_output("status", "--porcelain", "--untracked-files=normal"):
        raise remote.TrainingLoopError("Automation requires a clean Git worktree.")
    return _git_output("rev-parse", "HEAD")


def _require_adoption_compatible(training_sha: str, current_sha: str) -> None:
    if training_sha == current_sha:
        return
    if _git(
        "merge-base", "--is-ancestor", training_sha, current_sha, check=False
    ).returncode:
        raise remote.TrainingLoopError(
            "The adopted training commit is not an ancestor of Mac HEAD."
        )
    changed = _git_output(
        "diff", "--name-only", f"{training_sha}..{current_sha}"
    ).splitlines()
    disallowed = [
        path for path in changed if not remote._is_adoption_control_plane_path(path)
    ]
    if disallowed:
        raise remote.TrainingLoopError(
            "Mac HEAD changed training-relevant files after the adopted run: "
            + ", ".join(disallowed)
        )


def _load_state() -> dict[str, Any]:
    return remote._read_json(STATE_PATH)


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    remote._write_json_atomic(STATE_PATH, state)


def _context(state: dict[str, Any], job_id: str | None = None) -> remote.RemoteContext:
    return remote.RemoteContext(
        job_id=job_id or str(state["active_job_id"]),
        host=str(state["host"]),
        user=str(state["user"]),
        port=int(state["port"]) if state.get("port") is not None else None,
        remote_repo=str(state["remote_repo"]),
    )


def _job_id(cycle: int, config: str, git_sha: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return remote._safe_job_id(
        f"auto-{cycle:02d}-{Path(config).stem}-{git_sha[:8]}-{timestamp}"
    )


def _push(branch: str) -> None:
    result = _git("push", "origin", f"HEAD:{branch}", check=False)
    if result.returncode:
        raise remote.TrainingLoopError(
            f"git push failed: {(result.stderr or result.stdout).strip()}"
        )


def _stage(state: dict[str, Any]) -> str:
    stage = str(state.get("stage", "training"))
    if stage not in PIPELINE_STAGES:
        raise remote.TrainingLoopError(f"Unknown autonomous stage: {stage}")
    return stage


def _require_training_commit(state: dict[str, Any], manifest: dict[str, Any]) -> str:
    head = _require_clean_branch(str(state["branch"]))
    expected = str(manifest.get("git_sha", ""))
    if not expected or state.get("active_git_sha") != expected:
        raise remote.TrainingLoopError(
            "Mac HEAD, autonomous state, and the completed GPU job must use the "
            "same Git commit before analysis or export."
        )
    if head != expected:
        if not state.get("initial_run_adopted"):
            raise remote.TrainingLoopError(
                "Mac HEAD, autonomous state, and the completed GPU job must use the "
                "same Git commit before analysis or export."
            )
        _require_adoption_compatible(expected, head)
    return head


def _prepare_pending_job(
    state: dict[str, Any],
    *,
    cycle: int,
    config: str,
    start_mode: str,
    checkpoint: str,
) -> dict[str, Any]:
    git_sha = _require_clean_branch(str(state["branch"]))
    config = remote._repo_config(config)
    return {
        "cycle": cycle,
        "job_id": _job_id(cycle, config, git_sha),
        "git_sha": git_sha,
        "config": config,
        "checkpoint_series": remote._config_checkpoint_series(REPO_ROOT / config),
        "start_mode": start_mode,
        "checkpoint": checkpoint,
    }


def _validate_existing_submission(
    existing: dict[str, Any], pending: dict[str, Any]
) -> None:
    expected = {
        "job_id": pending["job_id"],
        "git_sha": pending["git_sha"],
        "source_config": pending["config"],
        "checkpoint_series": pending["checkpoint_series"],
        "start_mode": pending["start_mode"] if pending["checkpoint"] else None,
        "start_checkpoint_request": pending["checkpoint"] or None,
    }
    mismatched = [key for key, value in expected.items() if existing.get(key) != value]
    if mismatched:
        raise remote.TrainingLoopError(
            "Existing GPU job does not match the persisted enqueue intent: "
            + ", ".join(mismatched)
        )


def _resume_enqueue(state: dict[str, Any]) -> str:
    pending = state.get("pending_job")
    if not isinstance(pending, dict):
        raise remote.TrainingLoopError("enqueue stage is missing pending_job state.")
    context = _context(state, str(pending["job_id"]))
    try:
        manifest = remote._enqueue_remote(
            context,
            git_sha=str(pending["git_sha"]),
            config=str(pending["config"]),
            checkpoint_series=str(pending["checkpoint_series"]),
            init_policy=(
                str(pending["checkpoint"])
                if pending["start_mode"] == "init_policy"
                else None
            ),
            resume=(
                str(pending["checkpoint"])
                if pending["start_mode"] == "resume"
                else None
            ),
        )
    except remote.TrainingLoopError as submit_error:
        try:
            manifest = remote._fetch_manifest(context)
        except remote.TrainingLoopError:
            raise submit_error
        _validate_existing_submission(manifest, pending)
        print(f"Recovered existing GPU job {pending['job_id']}.", flush=True)

    state.update(
        cycle=int(pending["cycle"]),
        active_job_id=str(pending["job_id"]),
        active_git_sha=str(pending["git_sha"]),
        active_config=str(pending["config"]),
        stage="training",
        last_remote_status=str(manifest.get("status", "queued")),
        last_poll_at=None,
    )
    for key in (
        "pending_job",
        "terminal_job_id",
        "fix_base_sha",
        "fix_commit_sha",
        "fix_attempts",
    ):
        state.pop(key, None)
    _save_state(state)
    return str(state["active_job_id"])


def _enqueue(
    state: dict[str, Any],
    *,
    cycle: int,
    config: str,
    start_mode: str,
    checkpoint: str,
) -> str:
    state.update(
        stage="enqueue",
        pending_job=_prepare_pending_job(
            state,
            cycle=cycle,
            config=config,
            start_mode=start_mode,
            checkpoint=checkpoint,
        ),
    )
    _save_state(state)
    return _resume_enqueue(state)


def _resume_adoption(state: dict[str, Any]) -> dict[str, Any]:
    context = _context(state)
    try:
        manifest = remote._adopt_remote(
            context,
            git_sha=str(state["active_git_sha"]),
            config=str(state["active_config"]),
            run_name=state.get("adoption_run_name"),
        )
    except remote.TrainingLoopError as adoption_error:
        try:
            manifest = remote._fetch_manifest(context)
        except remote.TrainingLoopError:
            raise adoption_error
        print(f"Recovered adopted GPU job {context.job_id}.", flush=True)

    expected = {
        "job_id": state["active_job_id"],
        "git_sha": state["active_git_sha"],
        "source_config": state["active_config"],
        "adopted_from_existing_run": True,
    }
    if state.get("adoption_run_name"):
        expected["wandb_run_name"] = state["adoption_run_name"]
    mismatched = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatched:
        raise remote.TrainingLoopError(
            "Adopted GPU job does not match persisted intent: "
            + ", ".join(mismatched)
        )
    state.update(
        stage="training",
        last_remote_status=str(manifest.get("status", "completed")),
        last_poll_at=None,
    )
    state.pop("adoption_run_name", None)
    _save_state(state)
    return manifest


def _run_analyzer(context: remote.RemoteContext) -> dict[str, Any]:
    print(f"Syncing completed GPU job {context.job_id}...", flush=True)
    manifest = remote._sync_job(context, selected_checkpoint=False)
    run_dir = manifest.get("local_wandb_run_dir")
    report_path = remote.LOCAL_JOB_ROOT / context.job_id / "analysis.txt"
    if not run_dir:
        message = (
            "Analyzer not run because the job produced no synchronized W&B run.\n"
            f"Job status: {manifest.get('status')}\n"
            f"Training error: {manifest.get('error')}\n"
        )
        report_path.write_text(message)
        print(message, end="", flush=True)
        return manifest
    command = [
        shutil.which("uv") or "uv",
        "run",
        "python",
        "skills/wildrobot-training-analyze/scripts/analyze_offline_run.py",
        "--run-dir",
        str(REPO_ROOT / str(run_dir)),
        "--wandb-root",
        str(REPO_ROOT / "training/wandb"),
        "--checkpoints-root",
        str(REPO_ROOT / "training/checkpoints"),
    ]
    print(f"Running deterministic analyzer for {run_dir}...", flush=True)
    returncode = remote._run_streamed(
        command,
        cwd=REPO_ROOT,
        log_path=report_path,
    )
    if returncode:
        raise remote.TrainingLoopError(
            f"Training analyzer exited with {returncode}; see {report_path}"
        )
    manifest["local_analysis"] = str(report_path.relative_to(REPO_ROOT))
    remote._write_json_atomic(
        remote.LOCAL_JOB_ROOT / context.job_id / remote.MANIFEST_NAME, manifest
    )
    return manifest


def _allowed_next_checkpoints(manifest: dict[str, Any]) -> list[str]:
    candidates = {
        str(value)
        for key in (
            "start_checkpoint",
            "start_checkpoint_request",
            "selected_checkpoint_path",
        )
        if (value := manifest.get(key))
    }
    local_checkpoint_dir = manifest.get("local_checkpoint_run_dir")
    if local_checkpoint_dir:
        summary_path = (
            REPO_ROOT / str(local_checkpoint_dir) / "post_training_eval_summary.json"
        )
        if summary_path.is_file():
            summary = remote._read_json(summary_path)
            if summary.get("selected_checkpoint_path"):
                candidates.add(str(summary["selected_checkpoint_path"]))
            for candidate in summary.get("top_k_candidates", []):
                if isinstance(candidate, dict) and candidate.get("checkpoint_path"):
                    candidates.add(str(candidate["checkpoint_path"]))
    return sorted(candidates)


def _codex_prompt(state: dict[str, Any], manifest: dict[str, Any]) -> str:
    job_dir = remote.LOCAL_JOB_ROOT / str(manifest["job_id"])
    instructions = CODEX_INSTRUCTIONS.read_text()
    context = {
        "loop_cycle": state["cycle"],
        "max_cycles": state["max_cycles"],
        "job_status": manifest.get("status"),
        "training_git_sha": manifest.get("git_sha"),
        "provenance_mode": manifest.get("provenance_mode"),
        "provenance_note": manifest.get("provenance_note"),
        "remote_job_manifest": str(job_dir / remote.MANIFEST_NAME),
        "analysis_report": str(job_dir / "analysis.txt"),
        "local_wandb_run_dir": manifest.get("local_wandb_run_dir"),
        "local_checkpoint_run_dir": manifest.get("local_checkpoint_run_dir"),
        "remote_checkpoint_run_dir": manifest.get("checkpoint_run_dir"),
        "training_error": manifest.get("error"),
        "allowed_next_checkpoints": _allowed_next_checkpoints(manifest),
    }
    return f"{instructions}\n\nIteration context:\n{json.dumps(context, indent=2)}"


def _invoke_codex(
    state: dict[str, Any],
    manifest: dict[str, Any],
    *,
    before_sha: str | None = None,
) -> dict[str, Any]:
    before_sha = before_sha or _require_training_commit(state, manifest)
    codex_path = str(state.get("codex_path") or shutil.which("codex") or "")
    if not codex_path:
        raise remote.TrainingLoopError("codex executable not found on the Mac.")
    job_dir = remote.LOCAL_JOB_ROOT / str(manifest["job_id"])
    decision_path = job_dir / "codex_decision.json"
    final_message_path = job_dir / "codex_final.txt"
    log_path = job_dir / "codex_exec.log"
    decision_path.unlink(missing_ok=True)
    final_message_path.unlink(missing_ok=True)
    command = [
        codex_path,
        "exec",
        "--approve-for-me",
        "--ephemeral",
        "--output-schema",
        str(DECISION_SCHEMA),
        "--output-last-message",
        str(decision_path),
        "--cd",
        str(REPO_ROOT),
    ]
    if state.get("codex_model"):
        command.extend(["--model", str(state["codex_model"])])
    prompt = _codex_prompt(state, manifest)
    if int(state.get("fix_attempts", 0)) > 1:
        prompt += (
            "\n\nRecovery context:\n"
            f"A previous apply-fix attempt started from Git commit {before_sha}. "
            "Inspect and preserve any existing working-tree changes or commit "
            "after that base, finish the same bounded experiment, and leave "
            "exactly one clean commit after the base. Do not reset or discard "
            "the interrupted work."
        )
    command.append(prompt)
    print(f"Invoking Codex for job {manifest['job_id']}...", flush=True)
    returncode = remote._run_streamed(
        command,
        cwd=REPO_ROOT,
        log_path=log_path,
        timeout_s=int(state["codex_timeout_minutes"]) * 60,
        append=log_path.is_file(),
    )
    if returncode:
        raise remote.TrainingLoopError(
            f"codex exec failed with exit code {returncode}; see {log_path}"
        )
    if decision_path.is_file():
        final_message_path.write_text(decision_path.read_text())
    decision = remote._read_json(decision_path)
    _validate_codex_result(state, decision, before_sha, manifest)
    return decision


def _resume_codex_decision(
    state: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    before_sha = str(state.get("fix_base_sha", ""))
    if not before_sha:
        raise remote.TrainingLoopError("fix stage is missing fix_base_sha state.")
    decision_path = (
        remote.LOCAL_JOB_ROOT / str(manifest["job_id"]) / "codex_decision.json"
    )
    if decision_path.is_file():
        try:
            decision = remote._read_json(decision_path)
            _validate_codex_result(state, decision, before_sha, manifest)
        except remote.TrainingLoopError:
            pass
        else:
            print(
                f"Recovered completed Codex decision for job {manifest['job_id']}.",
                flush=True,
            )
            return decision

    state["fix_attempts"] = int(state.get("fix_attempts", 0)) + 1
    _save_state(state)
    return _invoke_codex(state, manifest, before_sha=before_sha)


def _validate_checkpoint_path(state: dict[str, Any], checkpoint: str) -> None:
    path = PurePosixPath(checkpoint)
    if path.suffix != ".pkl" or ".." in path.parts:
        raise remote.TrainingLoopError(f"Unsafe next checkpoint path: {checkpoint}")
    if not path.is_absolute():
        if path.parts[:2] != ("training", "checkpoints"):
            raise remote.TrainingLoopError(
                "Relative next checkpoints must be under training/checkpoints/."
            )
        return
    allowed = (
        PurePosixPath(str(state["remote_repo"])) / "training/checkpoints",
        PurePosixPath(str(state["remote_jobs_root"])),
    )
    if not any(path.is_relative_to(root) for root in allowed):
        raise remote.TrainingLoopError(
            f"Next checkpoint is outside approved GPU artifact roots: {checkpoint}"
        )


def _validate_codex_result(
    state: dict[str, Any],
    decision: dict[str, Any],
    before_sha: str,
    manifest: dict[str, Any],
) -> None:
    branch = str(state["branch"])
    after_sha = _require_clean_branch(branch)
    changed_files: list[str] = []
    if after_sha != before_sha:
        if _git(
            "merge-base", "--is-ancestor", before_sha, after_sha, check=False
        ).returncode:
            raise remote.TrainingLoopError(
                "Codex rewrote history instead of adding a commit."
            )
        commit_count = int(
            _git_output("rev-list", "--count", f"{before_sha}..{after_sha}")
        )
        if commit_count != 1:
            raise remote.TrainingLoopError(
                f"Codex must create at most one commit per cycle; created {commit_count}."
            )
        changed_files = _git_output(
            "diff", "--name-only", before_sha, after_sha
        ).splitlines()
        if any(
            path.startswith(prefix)
            for path in changed_files
            for prefix in FORBIDDEN_AUTONOMOUS_CHANGES
        ):
            raise remote.TrainingLoopError(
                "Codex modified a file reserved for manual review."
            )

    if not str(decision.get("summary", "")).strip():
        raise remote.TrainingLoopError("Codex returned an empty decision summary.")
    if decision.get("decision") != "continue":
        raise remote.TrainingLoopError(
            "Codex must provide the next bounded experiment; got decision "
            f"{decision.get('decision')!r}."
        )
    remote._repo_config(str(decision.get("config", "")))
    start_mode = str(decision.get("start_mode"))
    checkpoint = str(decision.get("checkpoint", ""))
    if start_mode not in {"init_policy", "resume"}:
        raise remote.TrainingLoopError(f"Invalid next start mode: {start_mode}")
    _validate_checkpoint_path(state, checkpoint)
    if checkpoint not in _allowed_next_checkpoints(manifest):
        raise remote.TrainingLoopError(
            "Next checkpoint is not recorded in the job manifest or "
            "deterministic evaluation summary."
        )


def _export_ready_bundle(
    state: dict[str, Any], context: remote.RemoteContext
) -> dict[str, Any]:
    manifest = remote._sync_job(context, selected_checkpoint=True)
    selected = REPO_ROOT / str(manifest["local_selected_checkpoint"])
    checkpoint_dir = REPO_ROOT / str(manifest["local_checkpoint_run_dir"])
    config = checkpoint_dir / "training_config.yaml"
    bundle = remote.LOCAL_JOB_ROOT / context.job_id / "deployment_bundle"
    staging_bundle = bundle.with_name(f".{bundle.name}.staging")
    validation_command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "runtime/wr_runtime/validation/validate_bundle.py",
        "--bundle",
        str(bundle),
    ]
    if bundle.is_dir():
        validate = subprocess.run(
            validation_command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if validate.returncode == 0:
            return {
                "bundle_path": str(bundle),
                "readiness": (
                    "deployment_bundle_ready"
                    if state.get("standing_checkpoint")
                    else "walking_bundle_ready"
                ),
            }
        raise remote.TrainingLoopError(
            "Existing deployment bundle is incomplete or invalid."
        )
    if staging_bundle.exists():
        shutil.rmtree(staging_bundle)
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "training/exports/export_policy_bundle_cli.py",
        "--walk-checkpoint",
        str(selected),
        "--training-config",
        str(config),
        "--bundle-path",
        str(staging_bundle),
    ]
    if state.get("standing_checkpoint"):
        command.extend(
            [
                "--standing-checkpoint",
                str(state["standing_checkpoint"]),
                "--standing-training-config",
                str(state["standing_config"]),
            ]
        )
    export = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    (bundle.parent / "bundle_export.log").write_text(
        export.stdout + "\n" + export.stderr
    )
    if export.returncode:
        raise remote.TrainingLoopError("Deployment bundle export failed.")
    validation_command[-1] = str(staging_bundle)
    validate = subprocess.run(
        validation_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (bundle.parent / "bundle_validation.log").write_text(
        validate.stdout + "\n" + validate.stderr
    )
    if validate.returncode:
        raise remote.TrainingLoopError("Deployment bundle validation failed.")
    staging_bundle.rename(bundle)
    return {
        "bundle_path": str(bundle),
        "readiness": (
            "deployment_bundle_ready"
            if state.get("standing_checkpoint")
            else "walking_bundle_ready"
        ),
    }


def _local_terminal_manifest(state: dict[str, Any]) -> dict[str, Any]:
    job_id = str(state.get("terminal_job_id") or state["active_job_id"])
    path = remote.LOCAL_JOB_ROOT / job_id / remote.MANIFEST_NAME
    if path.is_file():
        return remote._read_json(path)
    return remote._fetch_manifest(_context(state, job_id))


def _process_terminal_job(
    state: dict[str, Any], manifest: dict[str, Any] | None = None
) -> None:
    if _stage(state) == "training":
        if manifest is None:
            raise remote.TrainingLoopError(
                "training stage requires a terminal GPU manifest."
            )
        _require_training_commit(state, manifest)
        state.update(
            stage="analysis",
            terminal_job_id=str(manifest["job_id"]),
        )
        _save_state(state)

    while state.get("status") == "active":
        stage = _stage(state)
        if stage == "analysis":
            current = manifest or _local_terminal_manifest(state)
            _require_training_commit(state, current)
            current = _run_analyzer(
                _context(state, str(state["terminal_job_id"]))
            )
            fix_base_sha = _require_training_commit(state, current)
            state["processed_job_id"] = str(state["terminal_job_id"])
            if current.get("status") == "completed" and current.get(
                "simulation_candidate_ready"
            ):
                state["stage"] = "export"
                _save_state(state)
                manifest = current
                continue

            if current.get("status") == "failed":
                state["training_failures"] = int(
                    state.get("training_failures", 0)
                ) + 1
                if state["training_failures"] >= int(
                    state["max_training_failures"]
                ):
                    state.update(
                        status="stopped",
                        stop_reason="training failure limit",
                    )
                    _save_state(state)
                    return
            if int(state["cycle"]) >= int(state["max_cycles"]):
                state.update(
                    status="stopped",
                    stop_reason="maximum cycle count reached",
                )
                _save_state(state)
                return

            state.update(
                stage="fix",
                fix_base_sha=fix_base_sha,
                fix_attempts=0,
            )
            _save_state(state)
            manifest = current
            continue

        if stage == "export":
            context = _context(state, str(state["terminal_job_id"]))
            state.update(status="ready", **_export_ready_bundle(state, context))
            _save_state(state)
            return

        if stage == "enqueue":
            next_job = _resume_enqueue(state)
            print(f"Recovered enqueue of autonomous job: {next_job}")
            return

        current = manifest or _local_terminal_manifest(state)
        if stage == "fix":
            decision = _resume_codex_decision(state, current)
            state.update(
                stage="push",
                last_decision=decision,
                fix_commit_sha=_require_clean_branch(str(state["branch"])),
            )
            _save_state(state)
            manifest = current
            continue

        if stage == "push":
            decision = state.get("last_decision")
            if not isinstance(decision, dict):
                raise remote.TrainingLoopError(
                    "push stage is missing the validated Codex decision."
                )
            current_sha = _require_clean_branch(str(state["branch"]))
            if state.get("fix_commit_sha") and state["fix_commit_sha"] != current_sha:
                raise remote.TrainingLoopError(
                    "Git HEAD changed after the Codex decision was persisted."
                )
            _validate_codex_result(
                state,
                decision,
                str(state.get("fix_base_sha", "")),
                current,
            )
            _push(str(state["branch"]))
            next_job = _enqueue(
                state,
                cycle=int(state["cycle"]) + 1,
                config=str(decision["config"]),
                start_mode=str(decision["start_mode"]),
                checkpoint=str(decision["checkpoint"]),
            )
            print(f"Enqueued next autonomous job: {next_job}")
            return

        raise remote.TrainingLoopError(
            f"Stage {stage!r} cannot process a terminal GPU job."
        )


def _step_once(
    *,
    print_progress: bool = False,
    tolerate_poll_errors: bool = False,
) -> dict[str, Any] | None:
    if not STATE_PATH.is_file():
        return None
    state = _load_state()
    if state.get("status") != "active":
        return state
    try:
        stage = _stage(state)
        state.setdefault("stage", stage)
        if stage != "training":
            if print_progress:
                print(
                    f"[{_utc_now()}] cycle={state.get('cycle')}/"
                    f"{state.get('max_cycles')} job="
                    f"{state.get('terminal_job_id') or state.get('active_job_id')} "
                    f"pipeline_stage={stage}",
                    flush=True,
                )
            if stage == "adopt":
                _resume_adoption(state)
                return state
            _process_terminal_job(state)
            return state
        try:
            manifest = remote._fetch_manifest(_context(state))
        except remote.TrainingLoopError as exc:
            if not tolerate_poll_errors:
                raise
            state.update(
                last_remote_status="unreachable",
                last_poll_at=_utc_now(),
                last_poll_error=str(exc),
            )
            _save_state(state)
            if print_progress:
                print(
                    f"[{state['last_poll_at']}] cycle={state['cycle']}/"
                    f"{state['max_cycles']} job={state['active_job_id']} "
                    f"remote_status=unreachable error={exc}",
                    flush=True,
                )
            return state
        state["last_remote_status"] = manifest.get("status")
        state["last_poll_at"] = _utc_now()
        state.pop("last_poll_error", None)
        _save_state(state)
        if print_progress:
            print(
                f"[{state['last_poll_at']}] cycle={state['cycle']}/"
                f"{state['max_cycles']} job={state['active_job_id']} "
                f"remote_status={manifest.get('status', 'unknown')}",
                flush=True,
            )
        if manifest.get("status") in {"queued", "dispatching", "running"}:
            return state
        if state.get("processed_job_id") == manifest.get("job_id"):
            return state
        _process_terminal_job(state, manifest)
    except Exception as exc:
        state.update(
            status="stopped_error",
            stop_reason=f"{type(exc).__name__}: {exc}",
        )
        _save_state(state)
        raise
    return state


def _step(_args: argparse.Namespace) -> int:
    if not STATE_PATH.is_file():
        return 0
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        _step_once()
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.poll_seconds <= 0:
        raise remote.TrainingLoopError("--poll-seconds must be greater than zero.")
    if not STATE_PATH.is_file():
        raise remote.TrainingLoopError(
            "No autonomous state exists; run start before run."
        )
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise remote.TrainingLoopError(
                "Another autonomous supervisor is already running."
            ) from exc
        print(
            "Autonomous loop running in the foreground; press Ctrl-C to detach.",
            flush=True,
        )
        try:
            while True:
                state = _step_once(
                    print_progress=True,
                    tolerate_poll_errors=True,
                )
                if state is None:
                    raise remote.TrainingLoopError("Autonomous state disappeared.")
                if state.get("status") != "active":
                    reason = state.get("stop_reason")
                    suffix = f": {reason}" if reason else ""
                    print(
                        f"Autonomous loop finished with status "
                        f"{state.get('status')}{suffix}",
                        flush=True,
                    )
                    return 0
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print(
                "\nForeground monitor detached; autonomous state remains active.",
                flush=True,
            )
            return 130


def _start(args: argparse.Namespace) -> int:
    if STATE_PATH.is_file():
        previous = _load_state()
        if previous.get("status") == "active" or not args.new_run:
            raise remote.TrainingLoopError(
                "An autonomous state already exists; pass --new-run only after it stops."
            )
    if bool(args.standing_checkpoint) != bool(args.standing_config):
        raise remote.TrainingLoopError(
            "Provide both --standing-checkpoint and --standing-config, or neither."
        )
    if args.max_cycles < 1:
        raise remote.TrainingLoopError("--max-cycles must be at least 1.")
    if args.max_training_failures < 0:
        raise remote.TrainingLoopError("--max-training-failures cannot be negative.")
    if args.codex_timeout_minutes < 1:
        raise remote.TrainingLoopError("--codex-timeout-minutes must be at least 1.")
    if not PurePosixPath(args.remote_repo).is_absolute():
        raise remote.TrainingLoopError("--remote-repo must be an absolute Ubuntu path.")
    if args.training_git_sha and not args.adopt_completed:
        raise remote.TrainingLoopError(
            "--training-git-sha is only valid with --adopt-completed."
        )
    config = remote._repo_config(args.config)
    codex_path = (
        shutil.which(args.codex_path) if args.codex_path else shutil.which("codex")
    )
    if not codex_path:
        raise remote.TrainingLoopError("codex executable not found on the Mac.")
    standing_checkpoint: str | None = None
    standing_config: str | None = None
    if args.standing_checkpoint and args.standing_config:
        standing_checkpoint_path = Path(args.standing_checkpoint).resolve()
        standing_config_path = Path(args.standing_config).resolve()
        if (
            not standing_checkpoint_path.is_file()
            or standing_checkpoint_path.suffix != ".pkl"
        ):
            raise remote.TrainingLoopError(
                f"Standing checkpoint is not a .pkl file: {standing_checkpoint_path}"
            )
        if not standing_config_path.is_file() or standing_config_path.suffix not in {
            ".yaml",
            ".yml",
        }:
            raise remote.TrainingLoopError(
                f"Standing config is not a YAML file: {standing_config_path}"
            )
        standing_checkpoint = str(standing_checkpoint_path)
        standing_config = str(standing_config_path)
    git_sha = _require_clean_branch(args.branch)
    state: dict[str, Any] = {
        "schema_version": 2,
        "status": "active",
        "stage": "training",
        "created_at": _utc_now(),
        "branch": args.branch,
        "host": args.host,
        "user": args.user,
        "port": args.port,
        "remote_repo": args.remote_repo,
        "remote_jobs_root": remote.RemoteContext(
            "placeholder", args.host, args.user, args.port, args.remote_repo
        ).jobs_root,
        "cycle": 0,
        "max_cycles": args.max_cycles,
        "training_failures": 0,
        "max_training_failures": args.max_training_failures,
        "codex_path": codex_path,
        "codex_model": args.codex_model,
        "codex_timeout_minutes": args.codex_timeout_minutes,
        "standing_checkpoint": standing_checkpoint,
        "standing_config": standing_config,
        "initial_git_sha": git_sha,
    }
    if args.adopt_completed:
        training_git_sha = args.training_git_sha or git_sha
        if _git(
            "cat-file", "-e", f"{training_git_sha}^{{commit}}", check=False
        ).returncode:
            raise remote.TrainingLoopError(
                f"Training commit is not available on the Mac: {training_git_sha}"
            )
        _require_adoption_compatible(training_git_sha, git_sha)
        job_id = _job_id(1, config, training_git_sha)
        state.update(
            stage="adopt",
            cycle=1,
            active_job_id=job_id,
            active_git_sha=training_git_sha,
            active_config=config,
            initial_run_adopted=True,
            adoption_run_name=(
                None if args.adopt_completed == "latest" else args.adopt_completed
            ),
            last_decision=None,
        )
        _save_state(state)
        manifest = _resume_adoption(state)
        if manifest.get("git_sha") != training_git_sha:
            raise remote.TrainingLoopError(
                "Adopted manifest does not match the requested training commit."
            )
        print(
            f"Autonomous loop adopted completed run {manifest['wandb_run_name']} "
            f"as job {job_id}"
        )
        return 0
    mode = "init_policy" if args.init_policy else "resume" if args.resume else "none"
    checkpoint = args.init_policy or args.resume or ""
    if checkpoint:
        _validate_checkpoint_path(state, checkpoint)
    _push(args.branch)
    job_id = _enqueue(
        state,
        cycle=1,
        config=config,
        start_mode=mode,
        checkpoint=checkpoint,
    )
    print(f"Autonomous loop started with job {job_id}")
    return 0


def _mac_supervisor_running() -> bool:
    if not LOCK_PATH.is_file():
        return False
    with LOCK_PATH.open("r") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _stage_machine(state: dict[str, Any]) -> str:
    if state.get("status") != "active":
        return "none"
    return "GPU" if _stage(state) == "training" else "Mac"


def _git_commit_summary(git_sha: str | None) -> str | None:
    if not git_sha:
        return None
    result = _git("show", "-s", "--format=%h %s", git_sha, check=False)
    return result.stdout.strip() if result.returncode == 0 else str(git_sha)[:12]


def _job_cycle(job_id: str) -> int:
    match = re.match(r"auto-(\d+)-", job_id)
    return int(match.group(1)) if match else 0


def _training_result_summary(manifest: dict[str, Any]) -> str:
    status = str(manifest.get("status", "unknown"))
    if status not in {"completed", "failed"}:
        attempt = manifest.get("training_attempt")
        suffix = f", attempt {attempt}" if attempt is not None else ""
        return f"{status}{suffix}"
    if status == "failed":
        return f"failed: {manifest.get('error') or 'unknown error'}"

    checkpoint_dir = manifest.get("local_checkpoint_run_dir")
    summary_path = (
        REPO_ROOT / str(checkpoint_dir) / "post_training_eval_summary.json"
        if checkpoint_dir
        else None
    )
    if summary_path is None or not summary_path.is_file():
        readiness = "ready" if manifest.get("simulation_candidate_ready") else "not ready"
        return f"completed, {readiness}; deterministic summary not synchronized"

    summary = remote._read_json(summary_path)
    selected = summary.get("selected_checkpoint_path")
    candidates = [
        item for item in summary.get("top_k_candidates", []) if isinstance(item, dict)
    ]
    candidate = next(
        (
            item
            for item in candidates
            if selected and item.get("checkpoint_path") == selected
        ),
        candidates[0] if candidates else None,
    )
    readiness = "READY" if selected else "not ready"
    if candidate is None:
        return f"completed, {readiness}; no evaluated checkpoint metrics"
    metrics = candidate.get("eval_metrics", {})
    falls = metrics.get("walking_fall_env_count")
    survivors = metrics.get("walking_survivor_env_count")
    total = (
        int(float(falls) + float(survivors))
        if falls is not None and survivors is not None
        else None
    )
    parts = [
        f"completed, {readiness}",
        f"checkpoint={Path(str(candidate.get('checkpoint_path', 'unknown'))).name}",
    ]
    if falls is not None:
        falls_text = f"{int(float(falls))}/{total}" if total is not None else str(falls)
        parts.append(f"falls={falls_text}")
    for key, label, scale, suffix in (
        ("walking_stable_body_tilt_deg_mean", "tilt_mean", 1.0, "deg"),
        ("walking_stable_body_tilt_deg_max", "tilt_max", 1.0, "deg"),
        ("walking_stable_max_actuator_torque_sat_frac", "sat", 100.0, "%"),
        ("forward_velocity", "forward", 1.0, "m/s"),
    ):
        value = metrics.get(key)
        if value is not None:
            parts.append(f"{label}={float(value) * scale:.3f}{suffix}")
    return "; ".join(parts)


def _recent_cycle_summaries(
    state: dict[str, Any],
    current_manifest: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    if remote.LOCAL_JOB_ROOT.is_dir():
        for path in remote.LOCAL_JOB_ROOT.glob(f"*/{remote.MANIFEST_NAME}"):
            try:
                manifest = remote._read_json(path)
            except remote.TrainingLoopError:
                continue
            job_id = str(manifest.get("job_id", ""))
            if job_id.startswith("auto-"):
                manifests[job_id] = manifest
    if current_manifest is not None:
        manifests[str(current_manifest["job_id"])] = current_manifest
    elif state.get("active_job_id"):
        manifests.setdefault(
            str(state["active_job_id"]),
            {
                "job_id": state["active_job_id"],
                "status": state.get("last_remote_status", "unknown"),
                "git_sha": state.get("active_git_sha"),
                "source_config": state.get("active_config"),
            },
        )

    records = []
    for job_id, manifest in manifests.items():
        job_dir = remote.LOCAL_JOB_ROOT / job_id
        decision_path = job_dir / "codex_decision.json"
        decision = None
        if decision_path.is_file():
            try:
                decision = remote._read_json(decision_path)
            except remote.TrainingLoopError:
                pass
        records.append(
            {
                "cycle": _job_cycle(job_id),
                "job_id": job_id,
                "status": manifest.get("status", "unknown"),
                "config": manifest.get("source_config"),
                "wandb_run": manifest.get("wandb_run_name"),
                "input_patch": _git_commit_summary(manifest.get("git_sha")),
                "training_result": _training_result_summary(manifest),
                "next_patch": decision.get("summary") if decision else None,
                "next_config": decision.get("config") if decision else None,
            }
        )
    records.sort(key=lambda item: (int(item["cycle"]), str(item["job_id"])), reverse=True)
    return records[:limit]


def _status(args: argparse.Namespace) -> int:
    if args.last < 1:
        raise remote.TrainingLoopError("--last must be at least 1.")
    state = _load_state()
    current_manifest = None
    remote_error = None
    if state.get("active_job_id"):
        try:
            current_manifest = remote._fetch_manifest(_context(state))
        except remote.TrainingLoopError as exc:
            remote_error = str(exc)
    history = _recent_cycle_summaries(state, current_manifest, args.last)
    dashboard = {
        "status": state.get("status"),
        "stage": _stage(state),
        "stage_machine": _stage_machine(state),
        "mac_supervisor_running": _mac_supervisor_running(),
        "cycle": state.get("cycle"),
        "max_cycles": state.get("max_cycles"),
        "active_job_id": state.get("active_job_id"),
        "active_config": state.get("active_config"),
        "gpu_job_status": (
            current_manifest.get("status")
            if current_manifest is not None
            else state.get("last_remote_status", "unknown")
        ),
        "remote_error": remote_error,
        "recent_cycles": history,
    }
    if args.json:
        print(json.dumps(dashboard, indent=2, sort_keys=True))
        return 0

    print(f"Loop status: {dashboard['status']}")
    print(f"Current stage: {dashboard['stage']}")
    print(f"Stage machine: {dashboard['stage_machine']}")
    print(
        "Mac supervisor: "
        + ("running" if dashboard["mac_supervisor_running"] else "not running")
    )
    print(f"Cycle: {dashboard['cycle']}/{dashboard['max_cycles']}")
    print(f"Active job: {dashboard['active_job_id']}")
    print(f"GPU job status: {dashboard['gpu_job_status']}")
    print(f"Config: {dashboard['active_config']}")
    if remote_error:
        print(f"Remote status error: {remote_error}")
    print(f"\nRecent loop cycles (last {args.last}):")
    if not history:
        print("  none")
    for record in history:
        print(
            f"  Cycle {record['cycle']}: {record['status']} | {record['job_id']}"
        )
        print(f"    Input patch: {record['input_patch'] or 'unknown'}")
        print(f"    Config: {record['config'] or 'unknown'}")
        if record["wandb_run"]:
            print(f"    W&B: {record['wandb_run']}")
        print(f"    Result: {record['training_result']}")
        if record["next_patch"]:
            print(f"    Next patch: {record['next_patch']}")
        if record["next_config"]:
            print(f"    Next config: {record['next_config']}")
    return 0


def _stop(args: argparse.Namespace) -> int:
    state = _load_state()
    state.update(status="stopped", stop_reason=args.reason)
    _save_state(state)
    print("Autonomous loop stopped. The active GPU job, if any, was not killed.")
    return 0


def _retry(_args: argparse.Namespace) -> int:
    state = _load_state()
    if state.get("status") not in {"stopped", "stopped_error"}:
        raise remote.TrainingLoopError(
            "retry is only valid after a stopped loop."
        )
    legacy_state = "stage" not in state
    if state.get("status") == "stopped" and not legacy_state:
        raise remote.TrainingLoopError(
            "retry cannot override a configured stop limit; start a new run instead."
        )
    state["status"] = "active"
    if legacy_state:
        state["stage"] = "training"
        state["processed_job_id"] = None
        state["last_decision"] = None
    state.pop("stop_reason", None)
    _save_state(state)
    print(f"Autonomous loop reactivated for job {state['active_job_id']}.")
    return 0


def _install_mac_service(args: argparse.Namespace) -> int:
    python_path = REPO_ROOT / ".venv/bin/python"
    script_path = Path(__file__).resolve()
    if args.interval < 1:
        raise remote.TrainingLoopError("--interval must be at least 1 second.")
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise remote.TrainingLoopError(
            f"Mac Python is missing or not executable: {python_path}"
        )
    log_dir = remote.LOCAL_JOB_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path = (
        Path.home() / "Library/LaunchAgents/com.wildrobot.autonomous-training.plist"
    )
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.wildrobot.autonomous-training</string>
  <key>ProgramArguments</key>
  <array>
    <string>{python_path}</string>
    <string>{script_path}</string>
    <string>step</string>
  </array>
  <key>WorkingDirectory</key><string>{REPO_ROOT}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartInterval</key><integer>{args.interval}</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>{log_dir / 'mac-service.log'}</string>
  <key>StandardErrorPath</key><string>{log_dir / 'mac-service-error.log'}</string>
</dict>
</plist>
"""
    )
    print(f"Wrote {plist_path}")
    print("Start manually with:")
    print(f"  launchctl bootstrap gui/$(id -u) {plist_path}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous WildRobot training loop")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser(
        "start", help="enqueue a new job or adopt an already-completed GPU run"
    )
    start.add_argument("--config", required=True)
    checkpoint = start.add_mutually_exclusive_group()
    checkpoint.add_argument("--init-policy")
    checkpoint.add_argument("--resume")
    checkpoint.add_argument(
        "--adopt-completed",
        nargs="?",
        const="latest",
        metavar="RUN_NAME",
        help="start from the latest or named completed GPU W&B run",
    )
    start.add_argument(
        "--training-git-sha",
        help="Git commit used by a manually launched completed run",
    )
    start.add_argument("--branch", default="main")
    start.add_argument("--host", default=remote.DEFAULT_GPU_HOST)
    start.add_argument("--user", default=remote.DEFAULT_GPU_USER)
    start.add_argument("--port", type=int)
    start.add_argument("--remote-repo", default=remote.DEFAULT_REMOTE_REPO)
    start.add_argument("--max-cycles", type=int, default=8)
    start.add_argument("--max-training-failures", type=int, default=2)
    start.add_argument("--codex-path")
    start.add_argument("--codex-model")
    start.add_argument("--codex-timeout-minutes", type=int, default=60)
    start.add_argument("--standing-checkpoint")
    start.add_argument("--standing-config")
    start.add_argument("--new-run", action="store_true")
    start.set_defaults(func=_start)

    step = commands.add_parser("step", help="process one supervisor iteration")
    step.set_defaults(func=_step)
    run = commands.add_parser(
        "run", help="continuously supervise the active loop in the foreground"
    )
    run.add_argument("--poll-seconds", type=float, default=10.0)
    run.set_defaults(func=_run)
    status = commands.add_parser("status", help="show autonomous loop state")
    status.add_argument(
        "--last",
        type=int,
        default=5,
        help="number of recent patch/training cycles to show (default: 5)",
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_status)
    retry = commands.add_parser(
        "retry", help="retry the active job after an orchestration error"
    )
    retry.set_defaults(func=_retry)
    stop = commands.add_parser("stop", help="stop after the current GPU job")
    stop.add_argument("--reason", default="stopped manually")
    stop.set_defaults(func=_stop)
    install = commands.add_parser(
        "install-mac-service", help="write the macOS LaunchAgent"
    )
    install.add_argument("--interval", type=int, default=300)
    install.set_defaults(func=_install_mac_service)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        return int(args.func(args))
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        remote.TrainingLoopError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
