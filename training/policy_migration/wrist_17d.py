"""Deterministic 21D-to-17D projections for the wrist-free walk actor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from training.envs.env_info import PROPRIO_HISTORY_FRAMES


WRIST_ACTUATOR_NAMES = (
    "left_wrist_yaw",
    "left_wrist_pitch",
    "right_wrist_yaw",
    "right_wrist_pitch",
)


def v8_observation_dim(action_dim: int) -> int:
    """Return the wr_obs_v8_cmd3d dimension for an actuator count."""
    return 3 + 3 + action_dim + action_dim + 4 + action_dim + 1 + 2 + (
        PROPRIO_HISTORY_FRAMES * (3 + 4 + 3 * action_dim)
    ) + 2 + 1


def retained_actuator_indices(
    full_actuator_names: Sequence[str], active_actuator_names: Sequence[str]
) -> np.ndarray:
    full_names = [str(name) for name in full_actuator_names]
    active_names = [str(name) for name in active_actuator_names]
    if len(set(full_names)) != len(full_names):
        raise ValueError("full_actuator_names contains duplicates")
    missing = [name for name in active_names if name not in set(full_names)]
    if missing:
        raise ValueError(f"active actuators absent from full order: {missing}")
    index = {name: idx for idx, name in enumerate(full_names)}
    return np.asarray([index[name] for name in active_names], dtype=np.int32)


def project_v8_observation(
    observation: np.ndarray,
    *,
    full_actuator_names: Sequence[str],
    active_actuator_names: Sequence[str],
) -> np.ndarray:
    """Remove excluded actuator channels from current and historical v8 obs."""
    obs = np.asarray(observation, dtype=np.float32)
    full_dim = len(full_actuator_names)
    active_dim = len(active_actuator_names)
    expected_input = v8_observation_dim(full_dim)
    if obs.ndim < 1 or obs.shape[-1] != expected_input:
        raise ValueError(
            f"expected observation final dimension {expected_input}, got {obs.shape}"
        )
    keep = retained_actuator_indices(full_actuator_names, active_actuator_names)

    cursor = 0

    def take(size: int) -> np.ndarray:
        nonlocal cursor
        value = obs[..., cursor : cursor + size]
        cursor += size
        return value

    gravity_and_gyro = take(6)
    joint_pos = take(full_dim)[..., keep]
    joint_vel = take(full_dim)[..., keep]
    foot_switches = take(4)
    prev_action = take(full_dim)[..., keep]
    velocity_and_phase = take(3)
    history = take(PROPRIO_HISTORY_FRAMES * (7 + 3 * full_dim)).reshape(
        *obs.shape[:-1], PROPRIO_HISTORY_FRAMES, 7 + 3 * full_dim
    )
    history_fixed = history[..., :7]
    history_pos = history[..., 7 : 7 + full_dim][..., keep]
    history_vel = history[..., 7 + full_dim : 7 + 2 * full_dim][..., keep]
    history_action = history[..., 7 + 2 * full_dim :][..., keep]
    projected_history = np.concatenate(
        [history_fixed, history_pos, history_vel, history_action], axis=-1
    ).reshape(*obs.shape[:-1], -1)
    tail = take(3)
    if cursor != expected_input:
        raise AssertionError(f"v8 projection consumed {cursor} of {expected_input}")

    projected = np.concatenate(
        [
            gravity_and_gyro,
            joint_pos,
            joint_vel,
            foot_switches,
            prev_action,
            velocity_and_phase,
            projected_history,
            tail,
        ],
        axis=-1,
    ).astype(np.float32)
    expected_output = v8_observation_dim(active_dim)
    if projected.shape[-1] != expected_output:
        raise AssertionError(
            f"projected observation dimension {projected.shape[-1]} != {expected_output}"
        )
    return projected


def project_action(
    action: np.ndarray,
    *,
    full_actuator_names: Sequence[str],
    active_actuator_names: Sequence[str],
) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.ndim < 1 or action_array.shape[-1] != len(full_actuator_names):
        raise ValueError(
            "action final dimension must match full_actuator_names; "
            f"got action={action_array.shape} names={len(full_actuator_names)}"
        )
    keep = retained_actuator_indices(full_actuator_names, active_actuator_names)
    return action_array[..., keep].astype(np.float32)


def initialize_projected_policy_params(
    teacher_policy_params: dict[str, Any],
    student_template_params: dict[str, Any],
    *,
    full_actuator_names: Sequence[str],
    active_actuator_names: Sequence[str],
) -> dict[str, Any]:
    """Copy a 21D actor into a 17D actor by selecting input/output channels."""
    teacher_layers = teacher_policy_params.get("params")
    student_layers = student_template_params.get("params")
    if not isinstance(teacher_layers, dict) or not isinstance(student_layers, dict):
        raise ValueError("teacher and student policy params must contain a params dict")
    layer_names = sorted(
        (name for name in teacher_layers if str(name).startswith("hidden_")),
        key=lambda name: int(str(name).split("_")[-1]),
    )
    if layer_names != sorted(
        (name for name in student_layers if str(name).startswith("hidden_")),
        key=lambda name: int(str(name).split("_")[-1]),
    ):
        raise ValueError("teacher and student actor layer names differ")
    if not layer_names:
        raise ValueError("actor contains no hidden_N layers")

    full_dim = len(full_actuator_names)
    keep_actions = retained_actuator_indices(
        full_actuator_names, active_actuator_names
    )
    observation_markers = np.arange(
        v8_observation_dim(full_dim), dtype=np.float32
    )
    keep_observations = project_v8_observation(
        observation_markers,
        full_actuator_names=full_actuator_names,
        active_actuator_names=active_actuator_names,
    ).astype(np.int32)
    keep_logits = np.concatenate(
        [keep_actions, full_dim + keep_actions]
    ).astype(np.int32)

    projected: dict[str, dict[str, np.ndarray]] = {}
    final_layer = layer_names[-1]
    for layer_name in layer_names:
        teacher_kernel = np.asarray(
            teacher_layers[layer_name]["kernel"], dtype=np.float32
        )
        teacher_bias = np.asarray(
            teacher_layers[layer_name]["bias"], dtype=np.float32
        )
        kernel = teacher_kernel
        bias = teacher_bias
        if layer_name == layer_names[0]:
            kernel = kernel[keep_observations, :]
        if layer_name == final_layer:
            kernel = kernel[:, keep_logits]
            bias = bias[keep_logits]

        expected_kernel_shape = np.shape(student_layers[layer_name]["kernel"])
        expected_bias_shape = np.shape(student_layers[layer_name]["bias"])
        if kernel.shape != expected_kernel_shape or bias.shape != expected_bias_shape:
            raise ValueError(
                f"projected {layer_name} shapes kernel={kernel.shape}, bias={bias.shape} "
                f"do not match student kernel={expected_kernel_shape}, "
                f"bias={expected_bias_shape}"
            )
        projected[layer_name] = {
            "kernel": kernel.copy(),
            "bias": bias.copy(),
        }
    return {"params": projected}
