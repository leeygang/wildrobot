#!/usr/bin/env python3
"""Collect teacher rollouts and shard them into the v0.16.1 dataset contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_contract.spec import policy_spec_hash
from policy_contract.spec_builder import build_policy_spec
from training.algos.ppo.ppo_core import (
    compute_values,
    create_networks,
    sample_actions,
)
from training.configs.training_config import (
    get_robot_config,
    load_robot_config,
    load_training_config,
)
from training.core.checkpoint import load_checkpoint
from training.envs.wildrobot_env import WildRobotEnv
from training.imitation.dataset import (
    RolloutDatasetMetadata,
    RECOMMENDED_SHARD_KEYS,
    REQUIRED_SHARD_KEYS,
    validate_rollout_arrays,
    write_metadata,
    write_rollout_shard,
)


def shard_teacher_rollouts(
    arrays: Mapping[str, np.ndarray],
    *,
    output_dir: str | Path,
    shard_size: int,
    teacher_checkpoint: str,
    teacher_config: str,
    observation_layout: str,
    metadata_extras: dict | None = None,
) -> RolloutDatasetMetadata:
    """Write sharded rollout dataset and return metadata."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}")

    validated = validate_rollout_arrays(arrays)
    n = int(validated["obs"].shape[0])
    action_dim = int(validated["actions"].shape[1])
    phase_dim = int(validated["phase"].shape[1] if validated["phase"].ndim == 2 else 1)
    velocity_cmd = validated["velocity_cmd"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_count = 0
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        shard = {k: v[start:end] for k, v in validated.items()}
        shard_path = out_dir / f"shard_{shard_count:04d}.npz"
        write_rollout_shard(shard_path, shard)
        shard_count += 1

    extras = dict(metadata_extras or {})
    if "recommended_fields_present" not in extras:
        extras["recommended_fields_present"] = sorted(
            [k for k in RECOMMENDED_SHARD_KEYS if k in validated]
        )

    metadata = RolloutDatasetMetadata(
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        observation_layout=observation_layout,
        action_dim=action_dim,
        num_shards=shard_count,
        num_samples=n,
        phase_dim=phase_dim,
        velocity_cmd_range=(float(np.min(velocity_cmd)), float(np.max(velocity_cmd))),
        extras=extras,
    )
    write_metadata(out_dir / "metadata.json", metadata)
    return metadata


def _load_source_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as src:
        arrays = {k: src[k] for k in src.files}
    missing = [k for k in REQUIRED_SHARD_KEYS if k not in arrays]
    if missing:
        raise ValueError(
            f"Source npz missing required rollout arrays: {missing}. "
            f"Expected keys include {REQUIRED_SHARD_KEYS}."
        )
    return arrays


def _build_student_obs_projector(
    *,
    teacher_layout: str,
    student_layout: str,
    action_filter_alpha: float,
) -> tuple[int, int, Callable[[jnp.ndarray], jnp.ndarray], slice]:
    robot_cfg = get_robot_config()
    teacher_spec = build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
        action_filter_alpha=action_filter_alpha,
        layout_id=teacher_layout,
    )
    student_spec = build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
        action_filter_alpha=action_filter_alpha,
        layout_id=student_layout,
    )
    teacher_obs_dim = int(teacher_spec.model.obs_dim)
    student_obs_dim = int(student_spec.model.obs_dim)

    def _field_slice(layout, field_name: str) -> slice | None:
        offset = 0
        for field in layout:
            size = int(field.size)
            if field.name == field_name:
                return slice(offset, offset + size)
            offset += size
        return None

    # Identity path when layouts match.
    if teacher_layout == student_layout:
        phase_slice = _field_slice(teacher_spec.observation.layout, "teacher_phase")
        if phase_slice is None:
            gait_slice = _field_slice(teacher_spec.observation.layout, "gait_clock")
            if gait_slice is not None:
                phase_slice = slice(gait_slice.start, gait_slice.start + 2)
        if phase_slice is None:
            raise ValueError(
                f"Cannot infer phase slice from layout '{teacher_layout}' for rollout dataset."
            )
        return (
            teacher_obs_dim,
            student_obs_dim,
            lambda obs: obs.astype(jnp.float32),
            phase_slice,
        )

    # v0.16.1 intended path: wr_obs_teacher -> wr_obs_v3.
    if teacher_layout == "wr_obs_teacher" and student_layout == "wr_obs_v3":
        action_dim = int(teacher_spec.model.action_dim)
        base_dim = 3 + 3 + action_dim + action_dim + 4 + action_dim + 1
        phase_slice = slice(base_dim, base_dim + 2)

        def _project_teacher_to_v3(obs_teacher: jnp.ndarray) -> jnp.ndarray:
            obs_teacher = jnp.asarray(obs_teacher, dtype=jnp.float32)
            base = obs_teacher[..., :base_dim]
            teacher_phase = obs_teacher[..., phase_slice]
            gait_clock = jnp.stack(
                [
                    teacher_phase[..., 0],
                    teacher_phase[..., 1],
                    -teacher_phase[..., 0],
                    -teacher_phase[..., 1],
                ],
                axis=-1,
            )
            padding = jnp.zeros(obs_teacher.shape[:-1] + (1,), dtype=jnp.float32)
            projected = jnp.concatenate([base, gait_clock, padding], axis=-1)
            if int(projected.shape[-1]) != student_obs_dim:
                raise ValueError(
                    f"Projected obs_dim mismatch: got {projected.shape[-1]}, expected {student_obs_dim}"
                )
            return projected

        return teacher_obs_dim, student_obs_dim, _project_teacher_to_v3, phase_slice

    raise ValueError(
        f"Unsupported observation projection: teacher_layout={teacher_layout}, "
        f"student_layout={student_layout}"
    )


def collect_teacher_rollouts_from_checkpoint(
    *,
    teacher_checkpoint_path: str | Path,
    teacher_config_path: str | Path,
    output_dir: str | Path,
    shard_size: int,
    observation_layout: str,
    num_envs: int,
    num_steps: int,
    seed: int,
    deterministic: bool,
) -> RolloutDatasetMetadata:
    """Run teacher policy in env and export rollout shards for student pretraining."""
    if num_envs <= 0 or num_steps <= 0:
        raise ValueError(f"num_envs and num_steps must be > 0, got {num_envs}, {num_steps}")

    cfg = load_training_config(teacher_config_path)
    cfg.env.teacher_enabled = True
    cfg.freeze()
    robot_config_path = Path(cfg.env.robot_config_path)
    if not robot_config_path.is_absolute():
        robot_config_path = PROJECT_ROOT / robot_config_path
    load_robot_config(robot_config_path)
    teacher_obs_dim, student_obs_dim, project_student_obs, teacher_phase_slice = _build_student_obs_projector(
        teacher_layout=str(cfg.env.actor_obs_layout_id),
        student_layout=str(observation_layout),
        action_filter_alpha=float(cfg.env.action_filter_alpha),
    )

    teacher_ckpt = load_checkpoint(str(teacher_checkpoint_path))
    if "policy_params" not in teacher_ckpt:
        raise ValueError("Teacher checkpoint missing required key: policy_params")

    robot_cfg = get_robot_config()
    teacher_spec = build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
        action_filter_alpha=float(cfg.env.action_filter_alpha),
        layout_id=str(cfg.env.actor_obs_layout_id),
    )
    ckpt_hash = teacher_ckpt.get("config", {}).get("policy_spec_hash")
    current_hash = policy_spec_hash(teacher_spec)
    if ckpt_hash and ckpt_hash != current_hash:
        raise ValueError(
            "Teacher checkpoint policy contract mismatch: "
            f"ckpt={ckpt_hash}, current={current_hash}"
        )

    env = WildRobotEnv(config=cfg)
    ppo_network = create_networks(
        obs_dim=teacher_obs_dim,
        action_dim=teacher_spec.model.action_dim,
        policy_hidden_dims=cfg.networks.actor.hidden_sizes,
        value_hidden_dims=cfg.networks.critic.hidden_sizes,
    )
    policy_params = teacher_ckpt["policy_params"]
    value_params = teacher_ckpt.get("value_params")
    processor_params = teacher_ckpt.get("processor_params", ())

    reset_rng = jax.random.PRNGKey(seed)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    state = jax.vmap(env.reset)(reset_rngs)

    records: dict[str, list[np.ndarray]] = {
        "obs": [],
        "actions": [],
        "phase": [],
        "velocity_cmd": [],
        "forward_velocity": [],
        "done": [],
        "left_foot_contact": [],
        "right_foot_contact": [],
        "teacher_action_mean": [],
        "teacher_action_std": [],
        "teacher_value": [],
        "root_pitch": [],
        "root_pitch_rate": [],
    }

    rollout_rng = reset_rng
    contact_threshold = float(cfg.env.contact_threshold_force)
    for _ in range(num_steps):
        rollout_rng, act_rng = jax.random.split(rollout_rng)
        obs_teacher = state.obs
        obs_student = np.asarray(project_student_obs(obs_teacher), dtype=np.float32)
        phase_vec = np.asarray(obs_teacher[..., teacher_phase_slice], dtype=np.float32).reshape(num_envs, 2)
        actions, _, _ = sample_actions(
            processor_params=processor_params,
            policy_params=policy_params,
            ppo_network=ppo_network,
            obs=obs_teacher,
            rng=act_rng,
            deterministic=deterministic,
        )
        if deterministic:
            action_mean = actions
        else:
            action_mean, _, _ = sample_actions(
                processor_params=processor_params,
                policy_params=policy_params,
                ppo_network=ppo_network,
                obs=obs_teacher,
                rng=act_rng,
                deterministic=True,
            )
        if value_params is not None:
            teacher_value = compute_values(
                processor_params=processor_params,
                value_params=value_params,
                ppo_network=ppo_network,
                obs=obs_teacher,
            )
        else:
            teacher_value = jnp.zeros((num_envs,), dtype=jnp.float32)
        next_state = jax.vmap(lambda s, a: env.step(s, a))(state, actions)
        left_force = np.asarray(next_state.metrics["debug/left_force"], dtype=np.float32).reshape(-1)
        right_force = np.asarray(next_state.metrics["debug/right_force"], dtype=np.float32).reshape(-1)
        velocity_cmd = np.asarray(state.info["wr"].velocity_cmd, dtype=np.float32).reshape(-1)
        forward_velocity = np.asarray(next_state.metrics["debug/forward_vel"], dtype=np.float32).reshape(-1)
        done = np.asarray(next_state.done, dtype=np.float32).reshape(-1)
        root_pitch = np.asarray(next_state.metrics["debug/pitch"], dtype=np.float32).reshape(-1)
        root_pitch_rate = np.asarray(next_state.metrics["debug/pitch_rate"], dtype=np.float32).reshape(-1)

        records["obs"].append(obs_student.reshape(num_envs, -1))
        records["actions"].append(np.asarray(actions, dtype=np.float32).reshape(num_envs, -1))
        records["phase"].append(phase_vec)
        records["velocity_cmd"].append(velocity_cmd)
        records["forward_velocity"].append(forward_velocity)
        records["done"].append(done)
        records["left_foot_contact"].append((left_force > contact_threshold).astype(np.float32))
        records["right_foot_contact"].append((right_force > contact_threshold).astype(np.float32))
        records["teacher_action_mean"].append(np.asarray(action_mean, dtype=np.float32).reshape(num_envs, -1))
        records["teacher_action_std"].append(
            np.zeros((num_envs, teacher_spec.model.action_dim), dtype=np.float32)
        )
        records["teacher_value"].append(np.asarray(teacher_value, dtype=np.float32).reshape(-1))
        records["root_pitch"].append(root_pitch)
        records["root_pitch_rate"].append(root_pitch_rate)

        state = next_state

    arrays = {
        key: np.concatenate(values, axis=0).astype(np.float32) for key, values in records.items()
    }
    metadata = shard_teacher_rollouts(
        arrays,
        output_dir=output_dir,
        shard_size=shard_size,
        teacher_checkpoint=str(teacher_checkpoint_path),
        teacher_config=str(teacher_config_path),
        observation_layout=observation_layout,
        metadata_extras={
            "collector": "teacher_checkpoint_rollout",
            "teacher_obs_layout": str(cfg.env.actor_obs_layout_id),
            "num_envs": int(num_envs),
            "num_steps": int(num_steps),
            "deterministic": bool(deterministic),
        },
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--teacher-config", type=str, default=None)
    parser.add_argument("--observation-layout", type=str, default="wr_obs_v3")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic teacher actions (default: deterministic mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_npz:
        if not args.teacher_checkpoint or not args.teacher_config:
            raise ValueError("--teacher-checkpoint and --teacher-config are required with --source-npz")
        arrays = _load_source_npz(args.source_npz)
        metadata = shard_teacher_rollouts(
            arrays,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_config=args.teacher_config,
            observation_layout=args.observation_layout,
            metadata_extras={"collector": "source_npz"},
        )
    else:
        if not args.teacher_checkpoint or not args.teacher_config:
            raise ValueError(
                "Provide either --source-npz OR both --teacher-checkpoint and --teacher-config."
            )
        metadata = collect_teacher_rollouts_from_checkpoint(
            teacher_checkpoint_path=args.teacher_checkpoint,
            teacher_config_path=args.teacher_config,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            observation_layout=args.observation_layout,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            seed=args.seed,
            deterministic=not args.stochastic,
        )

    print("Teacher rollout shards written:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  num_samples: {metadata.num_samples}")
    print(f"  num_shards: {metadata.num_shards}")
    print(f"  action_dim: {metadata.action_dim}")
    print(f"  phase_dim: {metadata.phase_dim}")


if __name__ == "__main__":
    main()
