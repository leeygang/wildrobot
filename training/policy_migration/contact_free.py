"""Migrate walking actors between contact-observed v8 and contact-free v11."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from policy_contract.spec import PROPRIO_HISTORY_FRAMES


SOURCE_LAYOUT_ID = "wr_obs_v8_cmd3d"
TARGET_LAYOUT_ID = "wr_obs_v11_cmd3d_proprio"


def v8_observation_dim(action_dim: int) -> int:
    bundle_size = 3 + 4 + 3 * int(action_dim)
    return 3 + 3 + 3 * int(action_dim) + 4 + 1 + 2 + (
        PROPRIO_HISTORY_FRAMES * bundle_size
    ) + 2 + 1


def contact_free_observation_dim(action_dim: int) -> int:
    bundle_size = 3 + 3 * int(action_dim)
    return 3 + 3 + 3 * int(action_dim) + 1 + 2 + (
        PROPRIO_HISTORY_FRAMES * bundle_size
    ) + 2 + 1


def retained_v8_observation_indices(action_dim: int) -> np.ndarray:
    """Return v8 indices retained by the contact-free v11 layout."""
    n = int(action_dim)
    keep: list[int] = []
    cursor = 0

    def retain(size: int) -> None:
        nonlocal cursor
        keep.extend(range(cursor, cursor + size))
        cursor += size

    retain(6 + 2 * n)  # gravity, angular velocity, joint position/velocity
    cursor += 4  # current foot switches
    retain(n + 1 + 2)  # previous action, vx command, phase
    for _ in range(PROPRIO_HISTORY_FRAMES):
        retain(3)  # historical angular velocity
        cursor += 4  # historical foot switches
        retain(3 * n)  # historical joint position/velocity/action
    retain(2 + 1)  # lateral/yaw command and padding

    expected = v8_observation_dim(n)
    if cursor != expected:
        raise AssertionError(f"v8 index projection consumed {cursor} of {expected}")
    indices = np.asarray(keep, dtype=np.int32)
    target = contact_free_observation_dim(n)
    if indices.size != target:
        raise AssertionError(f"contact-free projection has {indices.size} != {target}")
    return indices


def project_v8_observation(
    observation: np.ndarray, *, action_dim: int
) -> np.ndarray:
    """Remove current and historical foot-switch channels from a v8 obs."""
    values = np.asarray(observation, dtype=np.float32)
    expected = v8_observation_dim(action_dim)
    if values.ndim < 1 or values.shape[-1] != expected:
        raise ValueError(
            f"expected v8 observation final dimension {expected}, got {values.shape}"
        )
    return values[..., retained_v8_observation_indices(action_dim)].astype(
        np.float32
    )


def project_v8_policy_params(
    policy_params: dict[str, Any], *, action_dim: int
) -> dict[str, Any]:
    """Drop v8 contact-input rows from the actor's first-layer kernel."""
    layers = policy_params.get("params")
    if not isinstance(layers, dict):
        raise ValueError("policy_params must contain a params dictionary")
    hidden_names = sorted(
        (name for name in layers if str(name).startswith("hidden_")),
        key=lambda name: int(str(name).split("_")[-1]),
    )
    if not hidden_names:
        raise ValueError("actor contains no hidden_N layers")

    first_name = hidden_names[0]
    kernel = np.asarray(layers[first_name]["kernel"])
    expected = v8_observation_dim(action_dim)
    if kernel.ndim != 2 or kernel.shape[0] != expected:
        raise ValueError(
            f"source actor first-layer shape {kernel.shape} does not match "
            f"{expected}D {SOURCE_LAYOUT_ID}"
        )

    projected = copy.deepcopy(policy_params)
    projected["params"][first_name]["kernel"] = kernel[
        retained_v8_observation_indices(action_dim), :
    ].copy()
    return projected


def expand_contact_free_policy_params(
    policy_params: dict[str, Any], *, action_dim: int
) -> dict[str, Any]:
    """Restore v8 contact inputs with zero-initialized first-layer weights."""
    layers = policy_params.get("params")
    if not isinstance(layers, dict):
        raise ValueError("policy_params must contain a params dictionary")
    hidden_names = sorted(
        (name for name in layers if str(name).startswith("hidden_")),
        key=lambda name: int(str(name).split("_")[-1]),
    )
    if not hidden_names:
        raise ValueError("actor contains no hidden_N layers")

    first_name = hidden_names[0]
    kernel = np.asarray(layers[first_name]["kernel"])
    expected = contact_free_observation_dim(action_dim)
    if kernel.ndim != 2 or kernel.shape[0] != expected:
        raise ValueError(
            f"source actor first-layer shape {kernel.shape} does not match "
            f"{expected}D {TARGET_LAYOUT_ID}"
        )

    expanded_kernel = np.zeros(
        (v8_observation_dim(action_dim), kernel.shape[1]), dtype=kernel.dtype
    )
    expanded_kernel[retained_v8_observation_indices(action_dim), :] = kernel
    expanded = copy.deepcopy(policy_params)
    expanded["params"][first_name]["kernel"] = expanded_kernel
    return expanded
