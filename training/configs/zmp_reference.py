"""Shared construction helpers for training-side ZMP references."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from control.zmp.zmp_walk import ZMPWalkConfig


ROLL_ACTUATOR_NAMES = (
    "left_hip_roll",
    "left_ankle_roll",
    "right_hip_roll",
    "right_ankle_roll",
)


def _env_get(env: Any, name: str, default: Any) -> Any:
    if isinstance(env, Mapping):
        value = env.get(name, default)
    else:
        value = getattr(env, name, default)
    return default if value is None else value


def zmp_walk_config_from_env(
    env: Any, *, offline_library_path: str | None = None
) -> ZMPWalkConfig:
    """Return the ZMP config selected by a training/evaluation environment."""
    width_raw = _env_get(env, "loc_ref_default_stance_width_m", None)
    use_reference_roll_base = bool(
        _env_get(env, "loc_ref_walking_base_from_ref_init_roll", False)
    )
    if use_reference_roll_base:
        if width_raw is None:
            raise ValueError(
                "env.loc_ref_walking_base_from_ref_init_roll requires "
                "loc_ref_default_stance_width_m"
            )
        if str(_env_get(env, "loc_ref_residual_base", "q_ref")) != "home":
            raise ValueError(
                "env.loc_ref_walking_base_from_ref_init_roll requires "
                "loc_ref_residual_base='home'"
            )
        if _env_get(env, "loc_ref_walking_joint_offsets_rad", {}):
            raise ValueError(
                "env.loc_ref_walking_base_from_ref_init_roll cannot be "
                "combined with loc_ref_walking_joint_offsets_rad"
            )
    if width_raw is None:
        return ZMPWalkConfig()
    if offline_library_path:
        raise ValueError(
            "env.loc_ref_default_stance_width_m cannot be combined with "
            "loc_ref_offline_library_path because an existing library has "
            "already frozen its stance geometry"
        )
    width = float(width_raw)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(
            "env.loc_ref_default_stance_width_m must be finite and positive; "
            f"got {width_raw!r}"
        )
    return ZMPWalkConfig(default_stance_width_m=width)


def reference_roll_base_offsets(
    *,
    actuator_names: tuple[str, ...] | list[str],
    home_q_rad: np.ndarray,
    ref_init_q_rad: np.ndarray,
    enabled: bool,
    explicit_offsets: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Derive a static walking-base offset from frame-zero roll IK only.

    ToddlerBot anchors its residual policy to reference frame zero.  WR keeps
    its proven standing/home pitch configuration, so this option adopts only
    the four stance-defining hip/ankle-roll channels.  The time-varying q_ref
    remains generator-native and is not offset a second time.
    """
    names = tuple(actuator_names)
    home = np.asarray(home_q_rad, dtype=np.float32).reshape(-1)
    ref_init = np.asarray(ref_init_q_rad, dtype=np.float32).reshape(-1)
    if home.size != len(names) or ref_init.size != len(names):
        raise ValueError("home/ref-init vectors must match actuator_names")
    if not enabled:
        return np.zeros_like(home)
    if explicit_offsets:
        raise ValueError(
            "env.loc_ref_walking_base_from_ref_init_roll cannot be combined "
            "with loc_ref_walking_joint_offsets_rad"
        )
    missing = sorted(set(ROLL_ACTUATOR_NAMES) - set(names))
    if missing:
        raise ValueError(
            "frame-zero roll base requires these actuators: " + ", ".join(missing)
        )
    offsets = np.zeros_like(home)
    for name in ROLL_ACTUATOR_NAMES:
        idx = names.index(name)
        offsets[idx] = ref_init[idx] - home[idx]
    if not np.all(np.isfinite(offsets)):
        raise ValueError("frame-zero roll base offsets must be finite")
    return offsets
