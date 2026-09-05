#!/usr/bin/env python3
"""Distill a contact-observed walking teacher into the contact-free actor.

The teacher runs with the legacy v8 actor observation, including simulated foot
contacts.  Teacher observations are projected to the deployable v11
proprioceptive contract before supervised training, so contact remains
available to simulation rewards and the privileged critic but never reaches
the student actor.  A checkpoint is written only when held-out action error
and deterministic closed-loop rollout gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_contract.spec import PROPRIO_HISTORY_FRAMES
from training.algos.ppo.ppo_core import sample_actions
from training.core.metrics_registry import METRIC_INDEX, NUM_METRICS
from training.exports.export_onnx import get_checkpoint_dims
from training.policy_migration.contact_free import (
    SOURCE_LAYOUT_ID,
    TARGET_LAYOUT_ID,
    contact_free_observation_dim,
    project_v8_observation,
    project_v8_policy_params,
    retained_v8_observation_indices,
    v8_observation_dim,
)
from training.scripts.distill_walking_21d_to_17d import (
    _action_error_metrics,
    _collect_teacher_rollouts,
    _load_env,
    _network,
    _parse_commands,
    _rollout_metrics,
    _train_student,
)


DEFAULT_TEACHER_CHECKPOINT = PROJECT_ROOT / (
    "training/checkpoints/ppo_walking_v0210_17d18_early_torque_margin/"
    "effective_training_config_v0210-17d18_20260903_043843-7m0ekgap/"
    "checkpoint_7_143360.pkl"
)
DEFAULT_TEACHER_CONFIG = PROJECT_ROOT / (
    "training/configs/ppo_walking_v0210_17d18_early_torque_margin.yaml"
)
DEFAULT_STUDENT_CONFIG = PROJECT_ROOT / (
    "training/configs/ppo_walking_v0210_17d19_contact_free_proprio.yaml"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "training/checkpoints/contact_free_distillation/"
    "17d18_ckpt7_v8_to_v11_distilled.pkl"
)
EXPECTED_TEACHER_SHA256 = (
    "a1322717e0e5e6cf73debbcb7bcdd8f7a1ac6bed113c19139c8d7812fa133511"
)
FAILURE_REPLAY_CONTACT_METRICS = (
    "debug/left_toe_switch",
    "debug/left_heel_switch",
    "debug/right_toe_switch",
    "debug/right_heel_switch",
)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_teacher_checkpoint(
    requested: Path,
    *,
    expected_sha256: str = EXPECTED_TEACHER_SHA256,
    search_roots: list[tuple[Path, str]] | None = None,
) -> Path:
    requested = requested.expanduser().resolve()
    candidates = [requested]
    if search_roots is None:
        main_repo = Path(os.environ.get("WILDROBOT_MAIN_REPO", PROJECT_ROOT)).resolve()
        jobs_root = main_repo.parent / f"{main_repo.name}-training-jobs"
        search_roots = [
            (main_repo / "training/checkpoints", requested.name),
            (jobs_root, requested.name),
            (main_repo / "runtime/bundles", "checkpoint.pkl"),
        ]
    for root, pattern in search_roots:
        if root.is_dir():
            candidates.extend(sorted(root.rglob(pattern)))

    checked: list[Path] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        checked.append(candidate)
        if _checkpoint_sha256(candidate) == expected_sha256:
            if candidate != requested:
                print(
                    "Resolved missing teacher checkpoint from "
                    f"{candidate}",
                    flush=True,
                )
            return candidate

    locations = "\n  ".join(str(path) for path in checked) or "none"
    raise FileNotFoundError(
        "Could not find the exact 17d18 checkpoint-7 teacher. "
        f"Expected SHA-256 {expected_sha256}. Checked files:\n  {locations}\n"
        "Copy checkpoint_7_143360.pkl to the GPU or pass its path with "
        "--teacher-checkpoint."
    )


def _validate_contracts(teacher_cfg, teacher_spec, student_cfg, student_spec) -> None:
    teacher_layout = str(teacher_cfg.env.actor_obs_layout_id)
    student_layout = str(student_cfg.env.actor_obs_layout_id)
    if teacher_layout != SOURCE_LAYOUT_ID:
        raise ValueError(
            f"teacher actor layout must be {SOURCE_LAYOUT_ID}, got {teacher_layout}"
        )
    if student_layout != TARGET_LAYOUT_ID:
        raise ValueError(
            f"student actor layout must be {TARGET_LAYOUT_ID}, got {student_layout}"
        )
    action_dim = int(teacher_spec.model.action_dim)
    if action_dim != 17 or int(student_spec.model.action_dim) != action_dim:
        raise ValueError(
            "contact distillation requires matching native 17D action contracts"
        )
    if list(teacher_spec.robot.actuator_names) != list(
        student_spec.robot.actuator_names
    ):
        raise ValueError("teacher and student actuator ordering must match exactly")
    expected_teacher_obs = v8_observation_dim(action_dim)
    expected_student_obs = contact_free_observation_dim(action_dim)
    actual_dims = (
        int(teacher_spec.model.obs_dim),
        int(student_spec.model.obs_dim),
    )
    if actual_dims != (expected_teacher_obs, expected_student_obs):
        raise ValueError(
            "unexpected teacher/student observation dimensions: "
            f"{actual_dims} != {(expected_teacher_obs, expected_student_obs)}"
        )


def _collect_projected_dataset(
    *,
    env,
    network,
    processor_params,
    policy_params,
    commands: list[np.ndarray],
    rollout_steps: int,
    repeats: int,
    seed: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    teacher_obs, teacher_actions = _collect_teacher_rollouts(
        env=env,
        network=network,
        processor_params=processor_params,
        policy_params=policy_params,
        commands=commands,
        rollout_steps=rollout_steps,
        zero_action_indices=np.asarray([], dtype=np.int32),
        seed=seed,
        repeats=repeats,
        perturb_pose=True,
    )
    return (
        project_v8_observation(teacher_obs, action_dim=action_dim),
        teacher_actions,
    )


def _load_failure_replay_dataset(
    trace_path: Path,
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild contact-observed teacher inputs for saved student states.

    Evaluation traces store the actor observation before each action and the
    post-step metrics after that action.  Consequently, observation ``t`` uses
    the contact signals from metric row ``t - 1`` and its 15-frame history
    uses rows ``t - 16:t - 1``.  The first 16 rows of each retained trace are
    skipped because their preceding contact history is outside the trace.
    """
    with np.load(trace_path) as trace:
        required = {"observations", "metrics_vec", "valid_lengths"}
        missing = sorted(required.difference(trace.files))
        if missing:
            raise ValueError(f"failure trace {trace_path} is missing arrays: {missing}")
        observations = np.asarray(trace["observations"], dtype=np.float32)
        metrics_vec = np.asarray(trace["metrics_vec"], dtype=np.float32)
        valid_lengths = np.asarray(trace["valid_lengths"], dtype=np.int32)

    expected_student_dim = contact_free_observation_dim(action_dim)
    expected_teacher_dim = v8_observation_dim(action_dim)
    if observations.ndim != 3 or observations.shape[-1] != expected_student_dim:
        raise ValueError(
            "failure trace observations must have shape "
            f"(N, T, {expected_student_dim}), got {observations.shape}"
        )
    if metrics_vec.ndim != 3 or metrics_vec.shape[:2] != observations.shape[:2]:
        raise ValueError(
            "failure trace metrics must match observation trace axes, got "
            f"{metrics_vec.shape} vs {observations.shape}"
        )
    if metrics_vec.shape[-1] != NUM_METRICS:
        raise ValueError(
            f"failure trace metric width {metrics_vec.shape[-1]} != {NUM_METRICS}"
        )
    if valid_lengths.shape != (observations.shape[0],):
        raise ValueError(
            "failure trace valid_lengths must have one entry per trace, got "
            f"{valid_lengths.shape}"
        )

    retained = retained_v8_observation_indices(action_dim)
    contact_slots = np.setdiff1d(
        np.arange(expected_teacher_dim, dtype=np.int32),
        retained,
        assume_unique=True,
    )
    expected_contact_slots = 4 * (PROPRIO_HISTORY_FRAMES + 1)
    if contact_slots.size != expected_contact_slots:
        raise AssertionError(
            f"expected {expected_contact_slots} contact slots, got "
            f"{contact_slots.size}"
        )
    contact_metric_indices = np.asarray(
        [METRIC_INDEX[name] for name in FAILURE_REPLAY_CONTACT_METRICS],
        dtype=np.int32,
    )

    student_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    first_reconstructable_step = PROPRIO_HISTORY_FRAMES + 1
    for trace_index, valid_length_raw in enumerate(valid_lengths):
        valid_length = int(valid_length_raw)
        if valid_length > observations.shape[1] or valid_length < 0:
            raise ValueError(
                f"invalid trace length {valid_length} at index {trace_index}"
            )
        if valid_length <= first_reconstructable_step:
            continue
        student_trace = observations[trace_index, -valid_length:]
        contacts = metrics_vec[trace_index, -valid_length:][:, contact_metric_indices]
        teacher_trace = []
        for step_index in range(first_reconstructable_step, valid_length):
            teacher_obs = np.zeros(expected_teacher_dim, dtype=np.float32)
            teacher_obs[retained] = student_trace[step_index]
            history_start = step_index - PROPRIO_HISTORY_FRAMES - 1
            contact_values = np.concatenate(
                (
                    contacts[step_index - 1],
                    contacts[history_start : step_index - 1].reshape(-1),
                )
            )
            teacher_obs[contact_slots] = contact_values
            teacher_trace.append(teacher_obs)
        student_batches.append(student_trace[first_reconstructable_step:])
        teacher_batches.append(np.stack(teacher_trace))

    if not student_batches:
        raise ValueError(
            "failure trace has no observations with a complete contact history"
        )
    student_obs = np.concatenate(student_batches).astype(np.float32)
    teacher_obs = np.concatenate(teacher_batches).astype(np.float32)
    projected = project_v8_observation(teacher_obs, action_dim=action_dim)
    if not np.array_equal(projected, student_obs):
        raise AssertionError("reconstructed teacher observations changed actor inputs")
    return student_obs, teacher_obs


def _deterministic_actions(
    *,
    network,
    processor_params,
    policy_params,
    observations: np.ndarray,
) -> np.ndarray:
    actions, _, _ = sample_actions(
        processor_params,
        policy_params,
        network,
        jnp.asarray(observations),
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    return np.asarray(actions, dtype=np.float32)


def _rollout_suite(
    *,
    env,
    network,
    params,
    commands: list[np.ndarray],
    steps: int,
    repeats: int,
    seed: int,
    processor_params=(),
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for repeat in range(repeats):
        for command_index, command in enumerate(commands):
            command_text = ",".join(f"{float(value):g}" for value in command)
            key = f"command={command_text}/repeat={repeat}"
            results[key] = _rollout_metrics(
                env=env,
                network=network,
                params=params,
                processor_params=processor_params,
                command=command,
                steps=steps,
                seed=seed + 100_003 * repeat + 3_001 * command_index,
            )
    return results


def _gate_failures(
    metrics: dict, *, max_validation_rmse: float, require_no_terminations: bool
) -> list[str]:
    failures: list[str] = []
    validation_rmse = float(metrics["validation_action_error"]["rmse"])
    initial_metrics = metrics.get("initial_validation_action_error")
    if initial_metrics is None:
        initial_metrics = metrics["projected_initial_validation_action_error"]
    initial_rmse = float(initial_metrics["rmse"])
    if validation_rmse > max_validation_rmse:
        failures.append(
            f"validation action RMSE {validation_rmse:.6f} exceeds "
            f"{max_validation_rmse:.6f}"
        )
    initial_source = metrics.get("dataset", {}).get(
        "initial_source", "projected_teacher"
    )
    if initial_source == "projected_teacher" and validation_rmse > initial_rmse:
        failures.append(
            f"distillation regressed validation action RMSE "
            f"{initial_rmse:.6f} -> {validation_rmse:.6f}"
        )
    replay = metrics.get("failure_replay")
    if isinstance(replay, dict):
        replay_initial_rmse = float(replay["initial_action_error"]["rmse"])
        replay_final_rmse = float(replay["distilled_action_error"]["rmse"])
        if replay_final_rmse > max_validation_rmse:
            failures.append(
                f"failure-replay action RMSE {replay_final_rmse:.6f} exceeds "
                f"{max_validation_rmse:.6f}"
            )
        if replay_final_rmse >= replay_initial_rmse:
            failures.append(
                "failure replay did not reduce teacher-action RMSE "
                f"{replay_initial_rmse:.6f} -> {replay_final_rmse:.6f}"
            )
    if require_no_terminations:
        for suite_name in ("teacher_rollouts", "distilled_student_rollouts"):
            terminated = [
                key
                for key, result in metrics[suite_name].items()
                if result["first_termination_step"] is not None
            ]
            if terminated:
                failures.append(
                    f"{suite_name} terminated in {len(terminated)}/"
                    f"{len(metrics[suite_name])} validation rollouts"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT
    )
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--student-checkpoint-sha256", type=str)
    parser.add_argument("--failure-trace", type=Path)
    parser.add_argument("--failure-trace-sha256", type=str)
    parser.add_argument("--failure-replay-repeats", type=int, default=1)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--student-config", type=Path, default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--commands",
        type=_parse_commands,
        default=_parse_commands("0.065,0,0;0.13,0,0"),
    )
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--rollout-repeats", type=int, default=16)
    parser.add_argument("--validation-steps", type=int, default=1000)
    parser.add_argument("--validation-repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-validation-rmse", type=float, default=0.08)
    parser.add_argument(
        "--require-no-terminations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(
            args.rollout_steps,
            args.rollout_repeats,
            args.validation_steps,
            args.validation_repeats,
            args.epochs,
            args.batch_size,
            args.failure_replay_repeats,
        )
        < 1
    ):
        parser.error(
            "rollout, validation, epoch, repeat, and batch values must be positive"
        )
    if args.learning_rate <= 0.0 or args.max_validation_rmse <= 0.0:
        parser.error("learning rate and validation RMSE threshold must be positive")
    if (args.student_checkpoint is None) != (args.failure_trace is None):
        parser.error(
            "--student-checkpoint and --failure-trace must be provided together"
        )

    args.teacher_checkpoint = _resolve_teacher_checkpoint(args.teacher_checkpoint)
    if args.student_checkpoint is not None:
        args.student_checkpoint = args.student_checkpoint.expanduser().resolve()
        if not args.student_checkpoint.is_file():
            parser.error(
                f"student checkpoint does not exist: {args.student_checkpoint}"
            )
        args.failure_trace = args.failure_trace.expanduser().resolve()
        if not args.failure_trace.is_file():
            parser.error(f"failure trace does not exist: {args.failure_trace}")
        expected_digests = {
            "student checkpoint": (
                args.student_checkpoint,
                args.student_checkpoint_sha256,
            ),
            "failure trace": (args.failure_trace, args.failure_trace_sha256),
        }
        for label, (path, expected_digest) in expected_digests.items():
            if expected_digest and _checkpoint_sha256(path) != expected_digest:
                parser.error(f"{label} SHA-256 does not match the configured digest")
    report_path = args.report or args.output.with_suffix(".metrics.json")
    resolved_paths = {
        "teacher": args.teacher_checkpoint.resolve(),
        "output": args.output.resolve(),
        "report": report_path.resolve(),
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        parser.error(
            "teacher checkpoint, output checkpoint, and report must be distinct paths"
        )
    args.output.unlink(missing_ok=True)

    teacher_cfg, teacher_spec, teacher_env = _load_env(
        args.teacher_config, disable_feedback_delay=False
    )
    student_cfg, student_spec, student_env = _load_env(
        args.student_config, disable_feedback_delay=False
    )
    _validate_contracts(teacher_cfg, teacher_spec, student_cfg, student_spec)

    action_dim = int(student_spec.model.action_dim)
    expected_checkpoint_dims = (
        int(teacher_spec.model.obs_dim),
        int(teacher_spec.model.action_dim),
    )
    checkpoint_dims = get_checkpoint_dims(args.teacher_checkpoint)
    if checkpoint_dims != expected_checkpoint_dims:
        raise ValueError(
            f"teacher checkpoint dims {checkpoint_dims} != {expected_checkpoint_dims}"
        )
    teacher_checkpoint = pickle.loads(args.teacher_checkpoint.read_bytes())
    teacher_network = _network(
        teacher_cfg,
        obs_dim=teacher_spec.model.obs_dim,
        action_dim=teacher_spec.model.action_dim,
    )
    student_network = _network(
        student_cfg,
        obs_dim=student_spec.model.obs_dim,
        action_dim=student_spec.model.action_dim,
    )
    teacher_processor = teacher_checkpoint.get("processor_params", ())

    initial_policy_params = project_v8_policy_params(
        teacher_checkpoint["policy_params"], action_dim=action_dim
    )
    initial_source = "projected_teacher"
    if args.student_checkpoint is not None:
        student_dims = get_checkpoint_dims(args.student_checkpoint)
        expected_student_dims = (
            int(student_spec.model.obs_dim),
            int(student_spec.model.action_dim),
        )
        if student_dims != expected_student_dims:
            raise ValueError(
                f"student checkpoint dims {student_dims} != {expected_student_dims}"
            )
        student_checkpoint = pickle.loads(args.student_checkpoint.read_bytes())
        student_processor = student_checkpoint.get("processor_params", ())
        if student_processor not in ((), None):
            raise ValueError(
                "failure-state replay does not support normalized student inputs"
            )
        initial_policy_params = student_checkpoint["policy_params"]
        initial_source = "student_checkpoint"

    if args.student_checkpoint is None:
        print(
            "Collecting teacher trajectories: "
            f"commands={len(args.commands)} repeats={args.rollout_repeats} "
            f"steps={args.rollout_steps}",
            flush=True,
        )
        train_obs, train_actions = _collect_projected_dataset(
            env=teacher_env,
            network=teacher_network,
            processor_params=teacher_processor,
            policy_params=teacher_checkpoint["policy_params"],
            commands=args.commands,
            rollout_steps=args.rollout_steps,
            repeats=args.rollout_repeats,
            seed=args.seed,
            action_dim=action_dim,
        )
        validation_obs, validation_actions = _collect_projected_dataset(
            env=teacher_env,
            network=teacher_network,
            processor_params=teacher_processor,
            policy_params=teacher_checkpoint["policy_params"],
            commands=args.commands,
            rollout_steps=args.validation_steps,
            repeats=args.validation_repeats,
            seed=args.seed + 1_000_003,
            action_dim=action_dim,
        )
    else:
        print(
            "Collecting student anchor trajectories: "
            f"commands={len(args.commands)} repeats={args.rollout_repeats} "
            f"steps={args.rollout_steps}",
            flush=True,
        )
        anchor_kwargs = {
            "env": student_env,
            "network": student_network,
            "processor_params": (),
            "policy_params": initial_policy_params,
            "commands": args.commands,
            "zero_action_indices": np.asarray([], dtype=np.int32),
            "perturb_pose": True,
        }
        train_obs, train_actions = _collect_teacher_rollouts(
            **anchor_kwargs,
            rollout_steps=args.rollout_steps,
            repeats=args.rollout_repeats,
            seed=args.seed,
        )
        validation_obs, validation_actions = _collect_teacher_rollouts(
            **anchor_kwargs,
            rollout_steps=args.validation_steps,
            repeats=args.validation_repeats,
            seed=args.seed + 1_000_003,
        )

    replay_student_obs = None
    replay_teacher_actions = None
    replay_trace_sha256 = None
    if args.failure_trace is not None:
        replay_student_obs, replay_teacher_obs = _load_failure_replay_dataset(
            args.failure_trace,
            action_dim=action_dim,
        )
        replay_teacher_actions = _deterministic_actions(
            network=teacher_network,
            processor_params=teacher_processor,
            policy_params=teacher_checkpoint["policy_params"],
            observations=replay_teacher_obs,
        )
        replay_trace_sha256 = _checkpoint_sha256(args.failure_trace)
        train_obs = np.concatenate(
            [train_obs, np.tile(replay_student_obs, (args.failure_replay_repeats, 1))]
        )
        train_actions = np.concatenate(
            [
                train_actions,
                np.tile(replay_teacher_actions, (args.failure_replay_repeats, 1)),
            ]
        )

    params = _train_student(
        network=student_network,
        observations=train_obs,
        target_actions=train_actions,
        obs_dim=student_spec.model.obs_dim,
        action_dim=action_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        initial_policy_params=initial_policy_params,
    )

    rollout_seed = args.seed + 2_000_003
    initial_validation_error = _action_error_metrics(
        student_network,
        initial_policy_params,
        validation_obs,
        validation_actions,
        action_dim,
    )

    metrics = {
        "dataset": {
            "train_samples": int(train_obs.shape[0]),
            "validation_samples": int(validation_obs.shape[0]),
            "commands": [command.tolist() for command in args.commands],
            "initial_source": initial_source,
        },
        "initial_validation_action_error": initial_validation_error,
        "projected_initial_validation_action_error": initial_validation_error,
        "train_action_error": _action_error_metrics(
            student_network, params, train_obs, train_actions, action_dim
        ),
        "validation_action_error": _action_error_metrics(
            student_network,
            params,
            validation_obs,
            validation_actions,
            action_dim,
        ),
        "teacher_rollouts": _rollout_suite(
            env=teacher_env,
            network=teacher_network,
            params=teacher_checkpoint["policy_params"],
            processor_params=teacher_processor,
            commands=args.commands,
            steps=args.validation_steps,
            repeats=args.validation_repeats,
            seed=rollout_seed,
        ),
        "projected_initial_student_rollouts": _rollout_suite(
            env=student_env,
            network=student_network,
            params=initial_policy_params,
            commands=args.commands,
            steps=args.validation_steps,
            repeats=args.validation_repeats,
            seed=rollout_seed,
        ),
        "distilled_student_rollouts": _rollout_suite(
            env=student_env,
            network=student_network,
            params=params,
            commands=args.commands,
            steps=args.validation_steps,
            repeats=args.validation_repeats,
            seed=rollout_seed,
        ),
    }
    if replay_student_obs is not None and replay_teacher_actions is not None:
        metrics["failure_replay"] = {
            "trace_path": str(args.failure_trace),
            "trace_sha256": replay_trace_sha256,
            "unique_samples": int(replay_student_obs.shape[0]),
            "training_repeats": int(args.failure_replay_repeats),
            "training_samples": int(
                replay_student_obs.shape[0] * args.failure_replay_repeats
            ),
            "initial_action_error": _action_error_metrics(
                student_network,
                initial_policy_params,
                replay_student_obs,
                replay_teacher_actions,
                action_dim,
            ),
            "distilled_action_error": _action_error_metrics(
                student_network,
                params,
                replay_student_obs,
                replay_teacher_actions,
                action_dim,
            ),
        }

    failures = _gate_failures(
        metrics,
        max_validation_rmse=args.max_validation_rmse,
        require_no_terminations=args.require_no_terminations,
    )
    metrics["gates"] = {
        "passed": not failures,
        "failures": failures,
        "max_validation_rmse": args.max_validation_rmse,
        "require_no_terminations": args.require_no_terminations,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if failures:
        print(
            "Distillation gates failed; checkpoint was not written: "
            + "; ".join(failures),
            file=sys.stderr,
        )
        return 2

    output_checkpoint = {
        "iteration": 0,
        "total_steps": 0,
        "policy_params": jax.tree.map(np.asarray, params),
        "processor_params": (),
        "policy_spec_json": student_spec.to_json_dict(),
        "config": {
            "networks": {
                "actor": {"activation": student_cfg.networks.actor.activation}
            },
            "distillation": {
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "teacher_checkpoint_sha256": _checkpoint_sha256(
                    args.teacher_checkpoint
                ),
                "student_checkpoint_sha256": (
                    None
                    if args.student_checkpoint is None
                    else _checkpoint_sha256(args.student_checkpoint)
                ),
                "teacher_config": str(args.teacher_config),
                "student_config": str(args.student_config),
                "report": str(report_path),
                "student_checkpoint": (
                    None
                    if args.student_checkpoint is None
                    else str(args.student_checkpoint)
                ),
                "failure_trace": (
                    None if args.failure_trace is None else str(args.failure_trace)
                ),
                "failure_trace_sha256": replay_trace_sha256,
            },
        },
        "distillation_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(output_checkpoint))
    print(f"saved contact-free distilled checkpoint: {args.output}")
    print(f"saved distillation report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
