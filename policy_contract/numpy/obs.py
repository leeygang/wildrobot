from __future__ import annotations

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.frames import angvel_heading_local, gravity_local_from_quat
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
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


def build_observation(
    *,
    spec: PolicySpec,
    state: PolicyState,
    signals: Signals,
    velocity_cmd: np.ndarray,
) -> np.ndarray:
    gravity = gravity_local_from_quat(signals.quat_xyzw)
    angvel = angvel_heading_local(signals.gyro_rad_s, signals.quat_xyzw)
    linvel = np.zeros((3,), dtype=np.float32)

    joint_pos_norm = NumpyCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=signals.joint_pos_rad)
    joint_vel_norm = NumpyCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=signals.joint_vel_rad_s)

    return build_observation_from_components(
        spec=spec,
        gravity_local=gravity,
        angvel_heading_local=angvel,
        linvel_heading_local=linvel,
        joint_pos_normalized=joint_pos_norm,
        joint_vel_normalized=joint_vel_norm,
        foot_switches=signals.foot_switches,
        prev_action=state.prev_action,
        velocity_cmd=velocity_cmd,
    )
