#!/usr/bin/env python3
"""Distill a 21-actuator walking actor into the native 17-actuator contract.

The teacher still runs in the 21-actuator MuJoCo model, but wrist actions are
zeroed before physics so the four wrist targets remain at model home.  Student
inputs and labels remove those same actuator channels.  The saved checkpoint
contains actor parameters only for later PPO initialization via --init-policy.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.robot_config import clear_robot_config_cache, load_robot_config
from training.algos.ppo.ppo_core import (
    create_networks,
    init_network_params,
    sample_actions,
)
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.exports.export_onnx import get_checkpoint_dims
from training.policy_migration.wrist_17d import (
    WRIST_ACTUATOR_NAMES,
    initialize_projected_policy_params,
    project_action,
    project_v8_observation,
)
from training.policy_spec_utils import build_policy_spec_from_training_config


DEFAULT_TEACHER_CHECKPOINT = (
    PROJECT_ROOT
    / "runtime/bundles/walking_v0210_smoke6_ckpt1650/checkpoint.pkl"
)
DEFAULT_TEACHER_CONFIG = (
    PROJECT_ROOT / "training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml"
)
DEFAULT_TEACHER_ROBOT_CONFIG = (
    PROJECT_ROOT
    / "runtime/bundles/walking_v0210_smoke6_ckpt1650/mujoco_robot_config.json"
)
DEFAULT_TEACHER_ROBOT_XML = (
    PROJECT_ROOT / "runtime/bundles/walking_v0210_smoke6_ckpt1650/wildrobot.xml"
)
DEFAULT_TEACHER_POLICY_SPEC = (
    PROJECT_ROOT / "runtime/bundles/walking_v0210_smoke6_ckpt1650/policy_spec.json"
)
DEFAULT_STUDENT_CONFIG = (
    PROJECT_ROOT
    / "training/configs/ppo_walking_v0210_smoke6_17d_latency_finetune.yaml"
)


def _parse_commands(text: str) -> list[np.ndarray]:
    commands: list[np.ndarray] = []
    for item in str(text).split(";"):
        values = [float(value.strip()) for value in item.split(",")]
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                f"each command must be vx,vy,wz; got {item!r}"
            )
        commands.append(np.asarray(values, dtype=np.float32))
    if not commands:
        raise argparse.ArgumentTypeError("at least one command is required")
    return commands


def _network(config, *, obs_dim: int, action_dim: int):
    actor_activation = str(config.networks.actor.activation).lower()
    critic_activation = str(config.networks.critic.activation).lower()
    if actor_activation != critic_activation:
        raise ValueError("actor and critic activations must match")
    return create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=tuple(config.networks.actor.hidden_sizes),
        value_hidden_dims=tuple(config.networks.critic.hidden_sizes),
        activation=actor_activation,
    )


def _load_env(
    config_path: Path,
    *,
    disable_feedback_delay: bool,
    allow_legacy_metric_actuators: bool = False,
    robot_config_path: Path | None = None,
    scene_xml_path: Path | None = None,
):
    config = load_training_config(config_path)
    if robot_config_path is not None:
        config.env.robot_config_path = str(robot_config_path)
    if scene_xml_path is not None:
        config.env.scene_xml_path = str(scene_xml_path)
        config.env.model_path = str(scene_xml_path)
    config.env.domain_randomization_enabled = False
    config.env.push_enabled = False
    config.env.imu_gyro_noise_std = 0.0
    config.env.imu_quat_noise_deg = 0.0
    config.env.imu_latency_steps = 0
    if disable_feedback_delay:
        config.env.joint_feedback_sample_hold_enabled = False
    # load_robot_config is a process-global singleton. Teacher and student use
    # different actuator contracts in this script, so each environment must
    # replace the previous cache before its spec and CAL are constructed.
    clear_robot_config_cache()
    robot_config = load_robot_config(config.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=config, robot_cfg=robot_config
    )
    return config, spec, WildRobotEnv(
        config,
        allow_legacy_metric_actuators=allow_legacy_metric_actuators,
    )


def _build_legacy_teacher_scene(
    *, robot_xml_path: Path, policy_spec_path: Path
) -> Path:
    """Build a temporary collision-only 21D scene for the archived teacher.

    Historical bundles intentionally do not duplicate visual mesh assets. CAD
    refreshes may delete or rename those files, so teacher rollouts remove
    non-colliding mesh geoms while retaining explicit body inertias and all
    primitive collision geoms.
    """
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"archived teacher MJCF has no compiler: {robot_xml_path}")
    compiler.attrib.pop("meshdir", None)

    for body in root.findall(".//body"):
        mesh_geoms = [
            geom for geom in body.findall("geom") if geom.get("type") == "mesh"
        ]
        if mesh_geoms and body.find("inertial") is None:
            raise ValueError(
                "cannot strip archived visual meshes from a body without an "
                f"explicit inertial: {body.get('name')}"
            )
        for geom in mesh_geoms:
            is_visual = geom.get("class") == "visual"
            is_non_colliding = (
                geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            )
            if not (is_visual or is_non_colliding):
                raise ValueError(
                    "archived teacher contains a physical mesh geom that cannot "
                    f"be stripped safely: {ET.tostring(geom, encoding='unicode')}"
                )
            body.remove(geom)

    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"archived teacher MJCF has no worldbody: {robot_xml_path}")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "pos": "0 0 0",
            "friction": "1 0.005 0.0001",
        },
    )

    spec = json.loads(policy_spec_path.read_text())
    names = [str(name) for name in spec["robot"]["actuator_names"]]
    home = [float(value) for value in spec["robot"]["home_ctrl_rad"]]
    home_by_name = dict(zip(names, home, strict=True))
    joint_names = [
        str(joint.get("name")) for joint in worldbody.findall(".//joint")
    ]
    missing = [name for name in joint_names if name not in home_by_name]
    if missing:
        raise ValueError(f"archived teacher home vector is missing joints: {missing}")

    keyframes_root = ET.parse(PROJECT_ROOT / "assets/v2/keyframes.xml").getroot()
    active_home = keyframes_root.find(".//key[@name='home']")
    if active_home is None or not active_home.get("qpos"):
        raise ValueError("active keyframes.xml has no home qpos")
    root_qpos = [float(value) for value in active_home.get("qpos", "").split()[:7]]
    qpos = root_qpos + [home_by_name[name] for name in joint_names]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": " ".join(f"{value:.9g}" for value in qpos)},
    )

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix="_wildrobot_21d_teacher.xml", delete=False
    ) as handle:
        output_path = Path(handle.name)
        tree.write(handle, encoding="utf-8", xml_declaration=True)
    return output_path


def _collect_teacher_rollouts(
    *,
    env: WildRobotEnv,
    network,
    processor_params,
    policy_params,
    commands: list[np.ndarray],
    rollout_steps: int,
    zero_action_indices: np.ndarray,
    seed: int,
    repeats: int = 1,
    perturb_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    zero_indices = jnp.asarray(zero_action_indices, dtype=jnp.int32)

    @jax.jit
    def rollout(initial_state, rng):
        def body(carry, _):
            state, action_rng = carry
            action_rng, sample_rng = jax.random.split(action_rng)
            action, _, _ = sample_actions(
                processor_params,
                policy_params,
                network,
                state.obs[None, :],
                sample_rng,
                deterministic=True,
            )
            action = action[0]
            physics_action = action.at[zero_indices].set(jnp.float32(0.0))
            next_state = env.step(
                state,
                physics_action,
                disable_pushes=True,
                disable_cmd_resample=True,
            )
            return (next_state, action_rng), (state.obs, action)

        return jax.lax.scan(
            body, (initial_state, rng), xs=None, length=rollout_steps
        )[1]

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for repeat in range(repeats):
        repeat_seed = seed + 100_003 * repeat
        for command_index, command in enumerate(commands):
            reset_rng = jax.random.PRNGKey(repeat_seed + 1009 * command_index)
            if perturb_pose:
                state = env.reset(
                    reset_rng,
                    perturb_pose=True,
                    velocity_cmd_override=jnp.asarray(command),
                )
            else:
                state = env.reset_for_eval(
                    reset_rng, cmd_override=jnp.asarray(command)
                )
            obs, action = rollout(
                state,
                jax.random.PRNGKey(repeat_seed + 2003 * command_index + 1),
            )
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
    return np.concatenate(observations), np.concatenate(actions)


def _train_student(
    *,
    network,
    observations: np.ndarray,
    target_actions: np.ndarray,
    obs_dim: int,
    action_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    initial_policy_params=None,
):
    _, policy_params, _ = init_network_params(
        network, obs_dim=obs_dim, action_dim=action_dim, seed=seed
    )
    if initial_policy_params is not None:
        policy_params = jax.tree.map(jnp.asarray, initial_policy_params)
    optimizer = optax.adam(float(learning_rate))
    optimizer_state = optimizer.init(policy_params)

    @jax.jit
    def update(params, opt_state, obs, targets):
        def loss_fn(candidate):
            logits = network.policy_network.apply((), candidate, obs)
            predicted = jnp.tanh(logits[..., :action_dim])
            return jnp.mean(jnp.square(predicted - targets))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    sample_count = observations.shape[0]
    for epoch in range(epochs):
        indices = rng.permutation(sample_count)
        losses: list[float] = []
        for start in range(0, sample_count, batch_size):
            batch = indices[start : start + batch_size]
            policy_params, optimizer_state, loss = update(
                policy_params,
                optimizer_state,
                jnp.asarray(observations[batch]),
                jnp.asarray(target_actions[batch]),
            )
            losses.append(float(loss))
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f"distill epoch={epoch + 1}/{epochs} mse={np.mean(losses):.8f}")
    return policy_params


def _action_error_metrics(network, params, obs, target, action_dim: int) -> dict:
    @jax.jit
    def predict(batch):
        logits = network.policy_network.apply((), params, batch)
        return jnp.tanh(logits[..., :action_dim])

    predicted = np.asarray(predict(jnp.asarray(obs)), dtype=np.float32)
    error = predicted - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs": float(np.max(np.abs(error))),
    }


def _rollout_metrics(
    *,
    env,
    network,
    params,
    command: np.ndarray,
    steps: int,
    seed: int,
    processor_params=(),
    zero_action_indices: np.ndarray | None = None,
) -> dict:
    zero_indices = jnp.asarray(
        [] if zero_action_indices is None else zero_action_indices,
        dtype=jnp.int32,
    )

    @jax.jit
    def rollout(initial_state, rng):
        def body(carry, _):
            state, action_rng = carry
            action_rng, sample_rng = jax.random.split(action_rng)
            action, _, _ = sample_actions(
                processor_params,
                params,
                network,
                state.obs[None, :],
                sample_rng,
                deterministic=True,
            )
            physics_action = action[0].at[zero_indices].set(jnp.float32(0.0))
            next_state = env.step(
                state,
                physics_action,
                disable_pushes=True,
                disable_cmd_resample=True,
            )
            return (next_state, action_rng), (
                next_state.done,
                next_state.data.qpos[:3],
            )

        return jax.lax.scan(body, (initial_state, rng), xs=None, length=steps)[1]

    state = env.reset_for_eval(
        jax.random.PRNGKey(seed), cmd_override=jnp.asarray(command)
    )
    done, root_xyz = rollout(state, jax.random.PRNGKey(seed + 1))
    done_np = np.asarray(done) > 0.5
    first_done = np.flatnonzero(done_np)
    return {
        "steps": int(steps),
        "termination_count": int(np.count_nonzero(done_np)),
        "first_termination_step": (
            None if first_done.size == 0 else int(first_done[0] + 1)
        ),
        "final_root_xyz_m": np.asarray(root_xyz[-1]).astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT
    )
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument(
        "--teacher-robot-config",
        type=Path,
        default=DEFAULT_TEACHER_ROBOT_CONFIG,
        help="Legacy 21D robot-config snapshot paired with the teacher checkpoint.",
    )
    parser.add_argument(
        "--teacher-robot-xml", type=Path, default=DEFAULT_TEACHER_ROBOT_XML
    )
    parser.add_argument(
        "--teacher-policy-spec", type=Path, default=DEFAULT_TEACHER_POLICY_SPEC
    )
    parser.add_argument("--student-config", type=Path, default=DEFAULT_STUDENT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--commands",
        type=_parse_commands,
        default=_parse_commands("0.065,0,0;0.13,0,0"),
    )
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.rollout_steps < 2 or args.validation_steps < 1 or args.epochs < 1:
        parser.error("rollout/validation steps and epochs must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")

    teacher_scene = _build_legacy_teacher_scene(
        robot_xml_path=args.teacher_robot_xml,
        policy_spec_path=args.teacher_policy_spec,
    )
    try:
        teacher_cfg, teacher_spec, teacher_env = _load_env(
            args.teacher_config,
            disable_feedback_delay=True,
            allow_legacy_metric_actuators=True,
            robot_config_path=args.teacher_robot_config,
            scene_xml_path=teacher_scene,
        )
    finally:
        teacher_scene.unlink(missing_ok=True)
    student_cfg, student_spec, student_env = _load_env(
        args.student_config, disable_feedback_delay=True
    )
    if teacher_spec.model.action_dim != 21 or student_spec.model.action_dim != 17:
        raise ValueError(
            "expected 21D teacher and 17D student; got "
            f"{teacher_spec.model.action_dim}D and {student_spec.model.action_dim}D"
        )
    checkpoint_dims = get_checkpoint_dims(args.teacher_checkpoint)
    expected_teacher_dims = (
        teacher_spec.model.obs_dim,
        teacher_spec.model.action_dim,
    )
    if checkpoint_dims != expected_teacher_dims:
        raise ValueError(
            f"teacher checkpoint dims {checkpoint_dims} != {expected_teacher_dims}"
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
    teacher_names = list(teacher_spec.robot.actuator_names)
    student_names = list(student_spec.robot.actuator_names)
    zero_indices = np.asarray(
        [teacher_names.index(name) for name in WRIST_ACTUATOR_NAMES],
        dtype=np.int32,
    )
    teacher_obs, teacher_actions = _collect_teacher_rollouts(
        env=teacher_env,
        network=teacher_network,
        processor_params=teacher_checkpoint.get("processor_params", ()),
        policy_params=teacher_checkpoint["policy_params"],
        commands=args.commands,
        rollout_steps=args.rollout_steps,
        zero_action_indices=zero_indices,
        seed=args.seed,
    )
    observations = project_v8_observation(
        teacher_obs,
        full_actuator_names=teacher_names,
        active_actuator_names=student_names,
    )
    target_actions = project_action(
        teacher_actions,
        full_actuator_names=teacher_names,
        active_actuator_names=student_names,
    )
    _, student_template_params, _ = init_network_params(
        student_network,
        obs_dim=student_spec.model.obs_dim,
        action_dim=student_spec.model.action_dim,
        seed=args.seed,
    )
    projected_initial_params = initialize_projected_policy_params(
        teacher_checkpoint["policy_params"],
        student_template_params,
        full_actuator_names=teacher_names,
        active_actuator_names=student_names,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(observations.shape[0])
    validation_count = max(1, int(round(len(indices) * args.validation_fraction)))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    params = _train_student(
        network=student_network,
        observations=observations[train_indices],
        target_actions=target_actions[train_indices],
        obs_dim=student_spec.model.obs_dim,
        action_dim=student_spec.model.action_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        initial_policy_params=projected_initial_params,
    )
    metrics = {
        "train_action_error": _action_error_metrics(
            student_network,
            params,
            observations[train_indices],
            target_actions[train_indices],
            student_spec.model.action_dim,
        ),
        "validation_action_error": _action_error_metrics(
            student_network,
            params,
            observations[validation_indices],
            target_actions[validation_indices],
            student_spec.model.action_dim,
        ),
        "native_17d_rollouts": {
            ",".join(f"{float(value):g}" for value in command): _rollout_metrics(
                env=student_env,
                network=student_network,
                params=params,
                command=command,
                steps=args.validation_steps,
                seed=args.seed + 3001 * index,
            )
            for index, command in enumerate(args.commands)
        },
        "teacher_21d_fixed_wrist_rollouts": {
            ",".join(f"{float(value):g}" for value in command): _rollout_metrics(
                env=teacher_env,
                network=teacher_network,
                params=teacher_checkpoint["policy_params"],
                processor_params=teacher_checkpoint.get("processor_params", ()),
                zero_action_indices=zero_indices,
                command=command,
                steps=args.validation_steps,
                seed=args.seed + 3001 * index,
            )
            for index, command in enumerate(args.commands)
        },
    }
    output_checkpoint = {
        "iteration": 0,
        "total_steps": 0,
        "policy_params": params,
        "processor_params": (),
        "policy_spec_json": student_spec.to_json_dict(),
        "config": {
            "networks": {
                "actor": {"activation": student_cfg.networks.actor.activation}
            },
            "migration": {
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "teacher_config": str(args.teacher_config),
                "student_config": str(args.student_config),
                "commands": [command.tolist() for command in args.commands],
                "rollout_steps_per_command": args.rollout_steps,
            },
        },
        "distillation_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pickle.dumps(output_checkpoint))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved native 17D actor checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
