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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.exports.export_onnx import get_checkpoint_dims
from training.policy_migration.contact_free import (
    SOURCE_LAYOUT_ID,
    TARGET_LAYOUT_ID,
    contact_free_observation_dim,
    project_v8_observation,
    project_v8_policy_params,
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
    projected_rmse = float(
        metrics["projected_initial_validation_action_error"]["rmse"]
    )
    if validation_rmse > max_validation_rmse:
        failures.append(
            f"validation action RMSE {validation_rmse:.6f} exceeds "
            f"{max_validation_rmse:.6f}"
        )
    if validation_rmse > projected_rmse:
        failures.append(
            f"distillation regressed validation action RMSE "
            f"{projected_rmse:.6f} -> {validation_rmse:.6f}"
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
    if min(
        args.rollout_steps,
        args.rollout_repeats,
        args.validation_steps,
        args.validation_repeats,
        args.epochs,
        args.batch_size,
    ) < 1:
        parser.error(
            "rollout, validation, epoch, repeat, and batch values must be positive"
        )
    if args.learning_rate <= 0.0 or args.max_validation_rmse <= 0.0:
        parser.error("learning rate and validation RMSE threshold must be positive")

    args.teacher_checkpoint = _resolve_teacher_checkpoint(args.teacher_checkpoint)
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

    projected_initial_params = project_v8_policy_params(
        teacher_checkpoint["policy_params"], action_dim=action_dim
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
        initial_policy_params=projected_initial_params,
    )

    rollout_seed = args.seed + 2_000_003
    metrics = {
        "dataset": {
            "train_samples": int(train_obs.shape[0]),
            "validation_samples": int(validation_obs.shape[0]),
            "commands": [command.tolist() for command in args.commands],
        },
        "projected_initial_validation_action_error": _action_error_metrics(
            student_network,
            projected_initial_params,
            validation_obs,
            validation_actions,
            action_dim,
        ),
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
            params=projected_initial_params,
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
                "teacher_config": str(args.teacher_config),
                "student_config": str(args.student_config),
                "report": str(report_path),
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
