from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from wildrobot.agents import autonomous_training_loop as auto
from wildrobot.agents import remote_training_loop as remote
from wildrobot.agents import training_loop_web as web


def _state(tmp_path: Path) -> dict:
    return {
        "branch": "main",
        "status": "active",
        "stage": "training",
        "cycle": 1,
        "max_cycles": 4,
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
        "campaign_objective": dict(auto.CAMPAIGN_OBJECTIVE),
        "champion": None,
        "experiment_history": [],
        "test_root": str(tmp_path),
    }


def _experiment_fields(family: str = "reference_tracking") -> dict:
    return {
        "failure_mode": "measured deterministic gate failure",
        "hypothesis": "one bounded intervention improves the campaign champion",
        "intervention_family": family,
        "expected_outcome": "reduce the targeted hard-gate metric",
        "falsification_condition": "the champion objective does not improve",
    }


def test_codex_prompt_contains_exact_run_context(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["required_actor_obs_layout_id"] = "wr_obs_v11_cmd3d_proprio"
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
    assert '"required_actor_obs_layout_id": "wr_obs_v11_cmd3d_proprio"' in prompt
    assert '"campaign_objective"' in prompt
    assert '"required_intervention_families"' in prompt


def test_codex_cannot_change_frozen_actor_observation_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training/configs/contact_observed.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("env:\n  actor_obs_layout_id: wr_obs_v8_cmd3d\n")
    checkpoint = "training/checkpoints/source/checkpoint.pkl"
    state = {
        **_state(tmp_path),
        "required_actor_obs_layout_id": "wr_obs_v11_cmd3d_proprio",
    }
    decision = {
        **_experiment_fields(),
        "decision": "continue",
        "summary": "restore actor contacts",
        "config": "training/configs/contact_observed.yaml",
        "start_mode": "init_policy",
        "checkpoint": checkpoint,
        "verification": ["focused tests passed"],
    }
    manifest = {
        "job_id": "job-1",
        "selected_checkpoint_path": checkpoint,
    }
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)

    with pytest.raises(remote.TrainingLoopError, match="frozen actor observation"):
        auto._validate_codex_result(state, decision, "a" * 40, manifest)


def test_codex_can_retry_config_managed_bootstrap_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training/configs/bootstrap.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "bootstrap:\n"
        "  mode: contact_observed_to_proprio\n"
        "env:\n"
        "  actor_obs_layout_id: wr_obs_v11_cmd3d_proprio\n"
    )
    state = {
        **_state(tmp_path),
        "required_actor_obs_layout_id": "wr_obs_v11_cmd3d_proprio",
    }
    decision = {
        **_experiment_fields("infrastructure"),
        "decision": "continue",
        "summary": "retry corrected distillation gates",
        "config": "training/configs/bootstrap.yaml",
        "start_mode": "none",
        "checkpoint": "",
        "verification": ["focused tests passed"],
    }
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)

    auto._validate_codex_result(
        state,
        decision,
        "a" * 40,
        {"job_id": "job-1", "status": "failed"},
    )


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

    with pytest.raises(remote.TrainingLoopError, match="not an ancestor"):
        auto._require_training_commit(state, {"git_sha": "a" * 40})


def test_completed_job_allows_post_start_evaluator_and_control_plane_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(
        auto,
        "_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(
        auto,
        "_git_output",
        lambda *args: "\n".join(
            [
                "wildrobot/agents/autonomous_training_loop.py",
                "training/eval/eval_policy.py",
                "tests/test_autonomous_training_loop.py",
            ]
        ),
    )

    assert auto._require_training_commit(state, {"git_sha": "a" * 40}) == "b" * 40


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
        **_experiment_fields(),
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


def test_campaign_champion_uses_stability_first_lexicographic_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "training/checkpoints/run-id"
    checkpoint_dir.mkdir(parents=True)
    safer = "/srv/jobs/job-1/artifacts/checkpoints/run/safer.pkl"
    faster = "/srv/jobs/job-1/artifacts/checkpoints/run/faster.pkl"
    (checkpoint_dir / "post_training_eval_summary.json").write_text(
        json.dumps(
            {
                "top_k_candidates": [
                    {
                        "rank": 1,
                        "checkpoint_path": faster,
                        "passed": False,
                        "gates": {
                            "forward_velocity": True,
                            "walking_fall_env_frac": False,
                            "walking_stable_max_actuator_torque_sat_frac": True,
                        },
                        "fail_reasons": ["walking_fall_env_frac"],
                        "eval_metrics": {
                            "walking_fall_env_count": 2,
                            "walking_stable_body_tilt_deg_max": 8.0,
                            "walking_stable_body_tilt_deg_mean": 3.0,
                            "walking_stable_max_actuator_torque_sat_frac": 0.03,
                            "cmd_vs_achieved_forward": 0.02,
                            "forward_velocity": 0.12,
                        },
                    },
                    {
                        "rank": 2,
                        "checkpoint_path": safer,
                        "passed": False,
                        "gates": {
                            "forward_velocity": True,
                            "walking_fall_env_frac": False,
                            "walking_stable_max_actuator_torque_sat_frac": False,
                        },
                        "fail_reasons": [
                            "walking_fall_env_frac",
                            "walking_stable_max_actuator_torque_sat_frac",
                        ],
                        "eval_metrics": {
                            "walking_fall_env_count": 1,
                            "walking_stable_body_tilt_deg_max": 9.0,
                            "walking_stable_body_tilt_deg_mean": 4.0,
                            "walking_stable_max_actuator_torque_sat_frac": 0.12,
                            "cmd_vs_achieved_forward": 0.04,
                            "forward_velocity": 0.09,
                        },
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)

    candidate = auto._best_screen_candidate(
        {"local_checkpoint_run_dir": "training/checkpoints/run-id"}
    )

    assert candidate is not None
    assert candidate["checkpoint_path"] == safer
    assert candidate["objective"][0] == 1.0


def test_fall_without_prefall_saturation_routes_to_failure_state_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto,
        "_failure_evidence",
        lambda _manifest: {
            "falls": 1,
            "pre_fall_max_actuator_torque_sat_frac": 0.0,
            "fail_reasons": ["walking_fall_env_frac"],
        },
    )

    assert auto._required_intervention_families({"status": "completed"}) == [
        "failure_state_replay",
        "recovery_curriculum",
    ]


def test_stable_tilt_gate_failure_routes_to_recovery_curriculum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto,
        "_failure_evidence",
        lambda _manifest: {
            "falls": 0,
            "pre_fall_max_actuator_torque_sat_frac": 0.0,
            "fail_reasons": ["walking_stable_body_tilt_deg_max"],
        },
    )

    assert auto._required_intervention_families({"status": "completed"}) == [
        "recovery_curriculum"
    ]


def test_codex_cannot_repeat_twice_failed_intervention_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training/configs/next.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("env:\n  actor_obs_layout_id: wr_obs_v11_cmd3d_proprio\n")
    checkpoint = "training/checkpoints/source/checkpoint.pkl"
    state = {
        **_state(tmp_path),
        "required_actor_obs_layout_id": "wr_obs_v11_cmd3d_proprio",
        "experiment_history": [
            {
                "intervention_family": "failure_state_replay",
                "result": "no_champion_improvement",
            },
            {
                "intervention_family": "failure_state_replay",
                "result": "no_champion_improvement",
            },
        ],
    }
    decision = {
        **_experiment_fields("failure_state_replay"),
        "decision": "continue",
        "summary": "repeat failed replay",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": checkpoint,
        "verification": ["tests passed"],
    }
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)
    monkeypatch.setattr(
        auto,
        "_required_intervention_families",
        lambda _manifest: ["failure_state_replay", "recovery_curriculum"],
    )

    with pytest.raises(remote.TrainingLoopError, match="already failed"):
        auto._validate_codex_result(
            state,
            decision,
            "a" * 40,
            {"job_id": "job-1", "selected_checkpoint_path": checkpoint},
        )


def test_next_experiment_must_branch_from_campaign_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training/configs/next.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("env:\n  actor_obs_layout_id: wr_obs_v11_cmd3d_proprio\n")
    champion = "training/checkpoints/champion/checkpoint.pkl"
    child = "training/checkpoints/child/checkpoint.pkl"
    state = {
        **_state(tmp_path),
        "required_actor_obs_layout_id": "wr_obs_v11_cmd3d_proprio",
        "champion": {"checkpoint_path": champion, "metrics": {}},
    }
    decision = {
        **_experiment_fields("failure_state_replay"),
        "decision": "continue",
        "summary": "continue from the latest child",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": child,
        "verification": ["tests passed"],
    }
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "a" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)
    monkeypatch.setattr(
        auto,
        "_required_intervention_families",
        lambda _manifest: ["failure_state_replay", "recovery_curriculum"],
    )

    with pytest.raises(remote.TrainingLoopError, match="frozen campaign champion"):
        auto._validate_codex_result(
            state,
            decision,
            "a" * 40,
            {
                "job_id": "job-1",
                "selected_checkpoint_path": child,
                "start_checkpoint": champion,
            },
        )


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


def test_confirmed_candidate_exports_and_stops(
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
            "job_kind": remote.EVALUATION_JOB_KIND,
            "evaluation_purpose": "confirmation",
            "confirmation_passed": True,
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


def test_screen_candidate_enqueues_confirmation_before_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    candidate = {
        "checkpoint_path": "/srv/jobs/job-1/artifacts/checkpoints/run/checkpoint.pkl",
        "eligible": True,
        "passed_screen": True,
        "metrics": {"walking_fall_env_count": 0},
        "objective": [0, 8, 4, 0.04, 0.03, -0.1],
    }
    events: list[str] = []
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)
    monkeypatch.setattr(auto, "_require_training_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda _context: {
            "job_id": "job-1",
            "job_kind": remote.TRAINING_JOB_KIND,
            "status": "completed",
            "simulation_candidate_ready": True,
        },
    )
    monkeypatch.setattr(auto, "_record_campaign_result", lambda *_args: True)
    monkeypatch.setattr(auto, "_best_screen_candidate", lambda _manifest: candidate)
    monkeypatch.setattr(auto, "_push", lambda _branch: events.append("push"))
    monkeypatch.setattr(
        auto,
        "_enqueue_candidate_evaluation",
        lambda *_args, **kwargs: events.append(kwargs["purpose"]) or "eval-job",
    )
    monkeypatch.setattr(
        auto,
        "_export_ready_bundle",
        lambda *_args: pytest.fail("screening pass must not export"),
    )

    auto._process_terminal_job(state, {"job_id": "job-1", "git_sha": "a" * 40})

    assert events == ["push", "confirmation"]


def test_falling_screen_candidate_enqueues_failure_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    candidate = {
        "checkpoint_path": "/srv/jobs/job-1/artifacts/checkpoints/run/checkpoint.pkl",
        "eligible": True,
        "passed_screen": False,
        "metrics": {"walking_fall_env_count": 1},
        "objective": [1, 9, 4, 0.04, 0.03, -0.1],
    }
    events: list[str] = []
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)
    monkeypatch.setattr(auto, "_require_training_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        auto,
        "_run_analyzer",
        lambda _context: {
            "job_id": "job-1",
            "job_kind": remote.TRAINING_JOB_KIND,
            "status": "completed",
            "simulation_candidate_ready": False,
        },
    )
    monkeypatch.setattr(auto, "_record_campaign_result", lambda *_args: True)
    monkeypatch.setattr(auto, "_best_screen_candidate", lambda _manifest: candidate)
    monkeypatch.setattr(auto, "_push", lambda _branch: events.append("push"))
    monkeypatch.setattr(
        auto,
        "_enqueue_candidate_evaluation",
        lambda *_args, **kwargs: events.append(kwargs["purpose"]) or "eval-job",
    )

    auto._process_terminal_job(state, {"job_id": "job-1", "git_sha": "a" * 40})

    assert events == ["push", "failure_diagnostic"]


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


def test_failed_gpu_job_still_runs_analysis_fix_and_enqueue(
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
            "status": "failed",
            "simulation_candidate_ready": False,
        },
    )
    monkeypatch.setattr(
        auto,
        "_invoke_codex",
        lambda *_args, **_kwargs: {
            "decision": "continue",
            "summary": "fix failed bootstrap",
            "config": "training/configs/next.yaml",
            "start_mode": "none",
            "checkpoint": "",
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

    auto._process_terminal_job(
        state,
        {"job_id": "job-1", "git_sha": "a" * 40, "status": "failed"},
    )

    assert events == ["push", "enqueue"]
    assert state["status"] == "active"


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


def test_enqueue_records_falsifiable_experiment(tmp_path: Path, monkeypatch) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(auto.remote, "_repo_config", lambda path: path)
    monkeypatch.setattr(auto.remote, "_config_checkpoint_series", lambda _path: "x")
    monkeypatch.setattr(auto.remote, "_enqueue_remote", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)
    decision = {
        **_experiment_fields("failure_state_replay"),
        "decision": "continue",
        "summary": "label student failure states",
        "config": "training/configs/next.yaml",
        "start_mode": "init_policy",
        "checkpoint": "training/checkpoints/source/checkpoint.pkl",
        "verification": ["tests passed"],
    }

    auto._enqueue(
        state,
        cycle=2,
        config=decision["config"],
        start_mode=decision["start_mode"],
        checkpoint=decision["checkpoint"],
        decision=decision,
    )

    assert state["experiment_history"][0]["intervention_family"] == (
        "failure_state_replay"
    )
    assert state["experiment_history"][0]["falsification_condition"]


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


def test_candidate_confirmation_enqueues_four_independent_seed_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(
        cycle=3,
        confirmation_seeds=[31_000, 41_000, 51_000, 61_000],
        confirmation_num_envs=64,
        confirmation_num_steps=1000,
    )
    checkpoint = "/srv/jobs/job-3/artifacts/checkpoints/walking/run/checkpoint.pkl"
    manifest = {
        "job_id": "job-3",
        "artifact_root": "/srv/jobs/job-3/artifacts",
        "source_config": "training/configs/walking.yaml",
        "checkpoint_series": "walking",
        "checkpoint_run_dir": "/srv/jobs/job-3/artifacts/checkpoints/walking/run",
        "checkpoint_run_relpath": "checkpoints/walking/run",
    }
    submitted: list[dict] = []
    monkeypatch.setattr(auto, "_require_clean_branch", lambda _branch: "b" * 40)
    monkeypatch.setattr(
        auto.remote,
        "_enqueue_walking_evaluation_remote",
        lambda _context, **kwargs: submitted.append(kwargs)
        or {"status": "queued"},
    )
    monkeypatch.setattr(auto, "_save_state", lambda _state: None)

    auto._enqueue_candidate_evaluation(
        state,
        manifest,
        {"checkpoint_path": checkpoint},
        purpose="confirmation",
    )

    assert submitted[0]["seeds"] == [31_000, 41_000, 51_000, 61_000]
    assert submitted[0]["num_envs"] == 64
    assert submitted[0]["num_steps"] == 1000
    assert state["active_job_kind"] == remote.EVALUATION_JOB_KIND


def test_fix_stage_recovers_completed_codex_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.update(stage="fix", fix_base_sha="a" * 40)
    decision = {
        **_experiment_fields(),
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
    monkeypatch.setattr(
        auto, "_actor_obs_layout_id", lambda _path: "wr_obs_v11_cmd3d_proprio"
    )
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
    assert saved[-1]["initial_config"] == "training/configs/walking.yaml"
    assert (
        saved[-1]["required_actor_obs_layout_id"]
        == "wr_obs_v11_cmd3d_proprio"
    )


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


def test_status_parser_defaults_to_five_and_accepts_ten() -> None:
    assert auto._parse_args(["status"]).last == 5
    assert auto._parse_args(["status", "--last", "10"]).last == 10


def test_web_parser_defaults_to_local_port_8080() -> None:
    args = auto._parse_args(["web"])

    assert args.func is auto._web
    assert args.host == "127.0.0.1"
    assert args.port == 8080


def test_web_start_builds_existing_cli_command() -> None:
    args = web._build_start_args(
        {
            "config": "training/configs/contact_free.yaml",
            "start_mode": "adopt_completed",
            "source": "offline-run-id",
            "training_git_sha": "a" * 40,
            "max_cycles": 20,
            "gpu_host": "gpu.local",
            "gpu_user": "robot",
            "remote_repo": "/srv/wildrobot",
            "standing_checkpoint": "runtime/bundles/standing/checkpoint.pkl",
            "standing_config": "runtime/bundles/standing/training_config.yaml",
            "new_run": True,
        }
    )

    assert args[:3] == [
        "start",
        "--config",
        "training/configs/contact_free.yaml",
    ]
    assert args[args.index("--adopt-completed") + 1] == "offline-run-id"
    assert args[args.index("--training-git-sha") + 1] == "a" * 40
    assert args[args.index("--max-cycles") + 1] == "20"
    assert "--max-training-failures" not in args
    assert args[-1] == "--new-run"


def test_web_status_combines_loop_and_machine_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_config": "training/configs/contact_free.yaml",
                "branch": "main",
                "host": "gpu.local",
                "user": "robot",
                "remote_repo": "/srv/wildrobot",
                "max_cycles": 20,
            }
        )
    )
    loop = {
        "status": "active",
        "stage": "training",
        "stage_machine": "GPU",
        "mac_supervisor_running": True,
        "gpu_job_status": "running",
        "recent_cycles": [],
    }
    monkeypatch.setattr(web, "STATE_PATH", state_path)
    monkeypatch.setattr(
        web,
        "_run_cli",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(loop), ""
        ),
    )
    monkeypatch.setattr(
        web,
        "_gpu_service_status",
        lambda _state: {
            "host": "gpu.local",
            "target": "robot@gpu.local",
            "reachable": True,
            "service": "active",
            "error": None,
        },
    )
    monkeypatch.setattr(web.socket, "gethostname", lambda: "mac.local")
    monkeypatch.setattr(web, "SUPERVISOR_LOG_PATH", tmp_path / "missing.log")

    status = web.TrainingLoopWebController().status(last=5)

    assert status["loop"]["stage_machine"] == "GPU"
    assert status["mac"] == {
        "host": "mac.local",
        "online": True,
        "supervisor_running": True,
    }
    assert status["gpu"]["service"] == "active"
    assert status["gpu"]["job_status"] == "running"


def test_web_run_launches_detached_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    log_path = tmp_path / "supervisor.log"
    commands: list[tuple[list[str], dict]] = []

    class Process:
        pid = 1234

    def fake_popen(command, **kwargs):
        commands.append((list(command), kwargs))
        return Process()

    controller = web.TrainingLoopWebController()
    monkeypatch.setattr(web, "STATE_PATH", state_path)
    monkeypatch.setattr(web, "SUPERVISOR_LOG_PATH", log_path)
    monkeypatch.setattr(web, "_mac_supervisor_running", lambda: False)
    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)

    result = controller.run()

    assert result["ok"] is True
    assert "PID 1234" in result["message"]
    assert commands[0][0][-1] == "run"
    assert commands[0][1]["start_new_session"] is True


def test_web_requires_request_token_for_actions() -> None:
    class Controller:
        def status(self, *, last):
            return {"last": last}

        def run(self):
            return {"ok": True, "message": "started"}

        def stop(self):
            return {"ok": True, "message": "stopped"}

        def start(self, _payload):
            return {"ok": True, "message": "created"}

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), web._make_handler(Controller(), "secret")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{url}/api/status?last=7") as response:
            assert json.load(response) == {"last": 7}

        unauthorized = Request(f"{url}/api/run", data=b"{}", method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(unauthorized)
        assert exc_info.value.code == HTTPStatus.FORBIDDEN

        authorized = Request(
            f"{url}/api/run",
            data=b"{}",
            method="POST",
            headers={"X-WildRobot-Token": "secret"},
        )
        with urlopen(authorized) as response:
            assert json.load(response)["message"] == "started"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_status_shows_machine_stage_and_recent_cycle_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {
        **_state(tmp_path),
        "cycle": 2,
        "active_job_id": "auto-02-next",
        "active_git_sha": "b" * 40,
        "active_config": "training/configs/next.yaml",
    }
    local_root = tmp_path / "remote_jobs"
    job_dir = local_root / "auto-01-first"
    job_dir.mkdir(parents=True)
    checkpoint_dir = tmp_path / "training/checkpoints/run-1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "post_training_eval_summary.json").write_text(
        json.dumps(
            {
                "selected_checkpoint_path": None,
                "top_k_candidates": [
                    {
                        "checkpoint_path": "training/checkpoints/run-1/checkpoint.pkl",
                        "eval_metrics": {
                            "walking_fall_env_count": 2,
                            "walking_survivor_env_count": 62,
                            "walking_stable_body_tilt_deg_mean": 3.2,
                            "walking_stable_body_tilt_deg_max": 8.4,
                            "walking_stable_max_actuator_torque_sat_frac": 0.12,
                            "forward_velocity": 0.11,
                        },
                    }
                ],
            }
        )
    )
    (job_dir / remote.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "job_id": "auto-01-first",
                "status": "completed",
                "git_sha": "a" * 40,
                "source_config": "training/configs/first.yaml",
                "wandb_run_name": "offline-run-first",
                "local_checkpoint_run_dir": "training/checkpoints/run-1",
            }
        )
    )
    (job_dir / "codex_decision.json").write_text(
        json.dumps(
            {
                "summary": "reduce the measured fall mode",
                "config": "training/configs/next.yaml",
            }
        )
    )
    monkeypatch.setattr(auto, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_mac_supervisor_running", lambda: False)
    monkeypatch.setattr(
        auto,
        "_git_commit_summary",
        lambda sha: f"{str(sha)[:7]} test patch",
    )
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: {
            "job_id": "auto-02-next",
            "status": "running",
            "git_sha": "b" * 40,
            "source_config": "training/configs/next.yaml",
            "training_attempt": 1,
        },
    )

    auto._status(argparse.Namespace(last=5, json=False))

    output = capsys.readouterr().out
    assert "Current stage: training" in output
    assert "Stage machine: GPU" in output
    assert "Mac supervisor: not running" in output
    assert "GPU job status: running" in output
    assert "Cycle 2: running" in output
    assert "Cycle 1: completed" in output
    assert "falls=2/64" in output
    assert "sat=12.000%" in output
    assert "Next patch: reduce the measured fall mode" in output


def test_status_last_limits_recent_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = {**_state(tmp_path), "stage": "analysis"}
    local_root = tmp_path / "remote_jobs"
    for cycle in range(1, 4):
        job_dir = local_root / f"auto-{cycle:02d}-job"
        job_dir.mkdir(parents=True)
        (job_dir / remote.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "job_id": job_dir.name,
                    "status": "completed",
                    "git_sha": str(cycle) * 40,
                }
            )
        )
    monkeypatch.setattr(auto.remote, "LOCAL_JOB_ROOT", local_root)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_mac_supervisor_running", lambda: True)
    monkeypatch.setattr(auto, "_git_commit_summary", lambda sha: str(sha)[:7])
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: (_ for _ in ()).throw(
            remote.TrainingLoopError("offline")
        ),
    )

    auto._status(argparse.Namespace(last=2, json=False))

    output = capsys.readouterr().out
    assert "Stage machine: Mac" in output
    assert "Mac supervisor: running" in output
    assert "Cycle 3: completed" in output
    assert "Cycle 2: completed" in output
    assert "Cycle 1: completed" not in output


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
    monkeypatch.setattr(auto, "_load_state", lambda: states[0])
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


def test_run_reactivates_stopped_error_before_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        auto,
        "_step_once",
        lambda **_kwargs: {**state, "status": "ready"},
    )

    assert auto._run(argparse.Namespace(poll_seconds=2.5)) == 0

    assert saved[0]["status"] == "active"
    assert saved[0]["stage"] == "fix"
    assert saved[0]["processed_job_id"] == "job-1"
    assert saved[0]["last_error"] == "codex failed"
    assert saved[0]["automatic_retry_count"] == 1
    assert "Automatically reactivated stage fix" in capsys.readouterr().out


def test_run_reactivates_paused_stage_before_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)
    state.update(
        status="paused",
        stage="analysis",
        processed_job_id="job-1",
        pause_reason="manual investigation",
    )
    saved: list[dict] = []
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(auto, "PAUSE_PATH", tmp_path / "pause.json")
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(
        auto,
        "_step_once",
        lambda **_kwargs: {**state, "status": "ready"},
    )

    assert auto._run(argparse.Namespace(poll_seconds=2.5)) == 0

    assert saved[0]["status"] == "active"
    assert saved[0]["stage"] == "analysis"
    assert saved[0]["processed_job_id"] == "job-1"
    assert saved[0]["last_pause_reason"] == "manual investigation"
    assert "Resuming paused stage analysis" in capsys.readouterr().out


def test_stop_pauses_immediately_when_supervisor_is_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)
    state.update(status="active", stage="fix", fix_attempts=1)
    state_path = tmp_path / "state.json"
    auto.remote._write_json_atomic(state_path, state)
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(auto, "PAUSE_PATH", tmp_path / "pause.json")

    assert auto._stop(argparse.Namespace(reason="manual investigation")) == 0

    paused = auto.remote._read_json(state_path)
    assert paused["status"] == "paused"
    assert paused["stage"] == "fix"
    assert paused["fix_attempts"] == 1
    assert paused["pause_reason"] == "manual investigation"
    assert not auto.PAUSE_PATH.exists()
    assert "paused at stage fix" in capsys.readouterr().out


def test_stop_requests_pause_when_supervisor_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)
    state.update(status="active", stage="analysis")
    state_path = tmp_path / "state.json"
    pause_path = tmp_path / "pause.json"
    auto.remote._write_json_atomic(state_path, state)
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(auto, "PAUSE_PATH", pause_path)

    def busy_lock(*_args) -> None:
        raise BlockingIOError

    monkeypatch.setattr(auto.fcntl, "flock", busy_lock)

    assert auto._stop(argparse.Namespace(reason="take over")) == 0

    assert auto.remote._read_json(state_path)["status"] == "active"
    assert auto.remote._read_json(pause_path)["reason"] == "take over"
    assert "Pause requested" in capsys.readouterr().out


def test_step_honors_pending_pause_before_pipeline_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(tmp_path)
    state.update(status="active", stage="analysis")
    state_path = tmp_path / "state.json"
    pause_path = tmp_path / "pause.json"
    auto.remote._write_json_atomic(state_path, state)
    auto.remote._write_json_atomic(
        pause_path,
        {"requested_at": "2026-09-03T00:00:00+00:00", "reason": "take over"},
    )
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "PAUSE_PATH", pause_path)

    result = auto._step_once()

    assert result is not None
    assert result["status"] == "paused"
    assert result["stage"] == "analysis"
    assert result["pause_reason"] == "take over"
    assert not pause_path.exists()


def test_status_reports_pending_pause_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)
    state_path = tmp_path / "state.json"
    pause_path = tmp_path / "pause.json"
    state_path.write_text("{}")
    pause_path.write_text("{}")
    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "PAUSE_PATH", pause_path)
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_mac_supervisor_running", lambda: True)
    monkeypatch.setattr(
        auto.remote,
        "_fetch_manifest",
        lambda _context: {"job_id": "job-1", "status": "completed"},
    )
    monkeypatch.setattr(auto, "_recent_cycle_summaries", lambda *_args: [])

    assert auto._status(argparse.Namespace(last=5, json=False)) == 0

    assert "Pause requested: yes (waiting for a safe stage boundary)" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "stage",
    ["adopt", "training", "analysis", "fix", "push", "enqueue", "export"],
)
def test_error_reactivation_preserves_every_durable_pipeline_stage(
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
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))

    auto._reactivate_error(state)

    assert saved[-1]["status"] == "active"
    assert saved[-1]["stage"] == stage
    assert saved[-1]["pending_job"] == {"job_id": "job-2"}
    assert saved[-1]["last_decision"] == {"decision": "continue"}


def test_run_retries_a_new_stage_error_without_exiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    saved: list[dict] = []
    sleeps: list[float] = []
    attempts = 0

    def step_once(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            state.update(status="stopped_error", stop_reason="analysis failed")
            raise remote.TrainingLoopError("analysis failed")
        return {**state, "status": "ready"}

    monkeypatch.setattr(auto, "STATE_PATH", state_path)
    monkeypatch.setattr(auto, "LOCK_PATH", tmp_path / "loop.lock")
    monkeypatch.setattr(auto, "_load_state", lambda: state)
    monkeypatch.setattr(auto, "_save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(auto, "_step_once", step_once)
    monkeypatch.setattr(auto.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert auto._run(argparse.Namespace(poll_seconds=2.5)) == 0

    assert attempts == 2
    assert sleeps == [2.5]
    assert saved[-1]["status"] == "active"
    assert saved[-1]["last_error"] == "analysis failed"
    assert "Retrying stage training in 2.5s" in capsys.readouterr().out


def test_retry_command_is_removed() -> None:
    with pytest.raises(SystemExit):
        auto._parse_args(["retry"])


def test_training_failure_limit_option_is_removed() -> None:
    with pytest.raises(SystemExit):
        auto._parse_args(
            [
                "start",
                "--config",
                "training/configs/walking.yaml",
                "--max-training-failures",
                "2",
            ]
        )


def test_start_defaults_to_twenty_cycles() -> None:
    args = auto._parse_args(["start", "--config", "training/configs/walking.yaml"])

    assert args.max_cycles == 20
