from __future__ import annotations

import numpy as np

from policy_contract.spec import PolicySpec


def build_observation_from_components(
    *,
    spec: PolicySpec,
    gravity_local: np.ndarray,
    angvel_heading_local: np.ndarray,
    linvel_heading_local: np.ndarray,
    joint_pos_normalized: np.ndarray,
    joint_vel_normalized: np.ndarray,
    foot_switches: np.ndarray,
    prev_action: np.ndarray,
    velocity_cmd: np.ndarray,
) -> np.ndarray:
    if spec.observation.layout_id != "wr_obs_v1":
        raise ValueError(f"Unsupported layout_id: {spec.observation.layout_id}")

    if spec.observation.linvel_mode == "zero":
        linvel_heading_local = np.zeros_like(linvel_heading_local, dtype=np.float32)
    elif spec.observation.linvel_mode in ("sim", "dropout"):
        pass
    else:
        raise ValueError(f"Unsupported linvel_mode: {spec.observation.linvel_mode}")

    obs = np.concatenate(
        [
            np.asarray(gravity_local, dtype=np.float32).reshape(3),
            np.asarray(angvel_heading_local, dtype=np.float32).reshape(3),
            np.asarray(linvel_heading_local, dtype=np.float32).reshape(3),
            np.asarray(joint_pos_normalized, dtype=np.float32).reshape(-1),
            np.asarray(joint_vel_normalized, dtype=np.float32).reshape(-1),
            np.asarray(foot_switches, dtype=np.float32).reshape(4),
            np.asarray(prev_action, dtype=np.float32).reshape(-1),
            np.asarray(velocity_cmd, dtype=np.float32).reshape(1),
            np.zeros((1,), dtype=np.float32),
        ]
    )
    return obs.astype(np.float32)
