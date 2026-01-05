from __future__ import annotations

import json

import numpy as np
import pytest

from policy_contract.spec import PolicySpec, validate_spec


def _valid_spec_dict() -> dict:
    return {
        "contract_name": "wildrobot_policy",
        "contract_version": "1.0.0",
        "spec_version": 1,
        "model": {
            "format": "onnx",
            "input_name": "observation",
            "output_name": "action",
            "dtype": "float32",
            "obs_dim": 36,
            "action_dim": 8,
        },
        "robot": {
            "robot_name": "WildRobotDev",
            "actuator_names": [
                "left_hip_pitch",
                "left_hip_roll",
                "left_knee_pitch",
                "left_ankle_pitch",
                "right_hip_pitch",
                "right_hip_roll",
                "right_knee_pitch",
                "right_ankle_pitch",
            ],
            "joints": {
                "left_hip_pitch": {
                    "range_min_rad": -0.1,
                    "range_max_rad": 1.5,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 10.0,
                },
                "left_hip_roll": {
                    "range_min_rad": -1.5,
                    "range_max_rad": 0.2,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 9.0,
                },
                "left_knee_pitch": {
                    "range_min_rad": 0.0,
                    "range_max_rad": 1.3,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 8.0,
                },
                "left_ankle_pitch": {
                    "range_min_rad": -0.7,
                    "range_max_rad": 0.8,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 7.0,
                },
                "right_hip_pitch": {
                    "range_min_rad": -1.5,
                    "range_max_rad": 0.1,
                    "mirror_sign": -1.0,
                    "max_velocity_rad_s": 10.0,
                },
                "right_hip_roll": {
                    "range_min_rad": -0.2,
                    "range_max_rad": 1.5,
                    "mirror_sign": -1.0,
                    "max_velocity_rad_s": 9.0,
                },
                "right_knee_pitch": {
                    "range_min_rad": 0.0,
                    "range_max_rad": 1.3,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 8.0,
                },
                "right_ankle_pitch": {
                    "range_min_rad": -0.7,
                    "range_max_rad": 0.8,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 7.0,
                },
            },
        },
        "observation": {
            "dtype": "float32",
            "layout_id": "wr_obs_v1",
            "layout": [
                {"name": "gravity_local", "size": 3, "frame": "local", "units": "unit_vector"},
                {"name": "angvel_heading_local", "size": 3, "frame": "heading_local", "units": "rad_s"},
                {"name": "joint_pos_normalized", "size": 8, "units": "normalized_-1_1"},
                {"name": "joint_vel_normalized", "size": 8, "units": "normalized_-1_1"},
                {"name": "foot_switches", "size": 4, "units": "bool_as_float"},
                {"name": "prev_action", "size": 8, "units": "normalized_-1_1"},
                {"name": "velocity_cmd", "size": 1, "units": "m_s"},
                {"name": "padding", "size": 1, "units": "unused"},
            ],
        },
        "action": {
            "dtype": "float32",
            "bounds": {"min": -1.0, "max": 1.0},
            "postprocess_id": "lowpass_v1",
            "postprocess_params": {"alpha": 0.7},
            "mapping_id": "pos_target_rad_v1",
        },
    }


@pytest.fixture()
def spec() -> PolicySpec:
    spec = PolicySpec.from_json(json.dumps(_valid_spec_dict()))
    validate_spec(spec)
    return spec


def test_action_to_ctrl_parity(spec: PolicySpec) -> None:
    jax = pytest.importorskip("jax.numpy")
    from policy_contract.calib import JaxCalibOps, NumpyCalibOps

    rng = np.random.RandomState(0)
    action = rng.uniform(-1.2, 1.2, size=(spec.model.action_dim,)).astype(np.float32)

    out_np = NumpyCalibOps.action_to_ctrl(spec=spec, action=action)
    out_jax = JaxCalibOps.action_to_ctrl(spec=spec, action=jax.asarray(action))

    np.testing.assert_allclose(out_np, np.asarray(out_jax), rtol=1e-6, atol=1e-6)


def test_ctrl_to_action_parity(spec: PolicySpec) -> None:
    jax = pytest.importorskip("jax.numpy")
    from policy_contract.calib import JaxCalibOps, NumpyCalibOps

    rng = np.random.RandomState(4)
    ctrl = rng.uniform(-0.5, 0.5, size=(spec.model.action_dim,)).astype(np.float32)

    out_np = NumpyCalibOps.ctrl_to_policy_action(spec=spec, ctrl_rad=ctrl)
    out_jax = JaxCalibOps.ctrl_to_policy_action(spec=spec, ctrl_rad=jax.asarray(ctrl))

    np.testing.assert_allclose(out_np, np.asarray(out_jax), rtol=1e-6, atol=1e-6)


def test_normalization_parity(spec: PolicySpec) -> None:
    jax = pytest.importorskip("jax.numpy")
    from policy_contract.calib import JaxCalibOps, NumpyCalibOps

    rng = np.random.RandomState(1)
    joint_pos = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    joint_vel = rng.uniform(-5.0, 5.0, size=(spec.model.action_dim,)).astype(np.float32)

    out_pos_np = NumpyCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=joint_pos)
    out_pos_jax = JaxCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=jax.asarray(joint_pos))

    out_vel_np = NumpyCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=joint_vel)
    out_vel_jax = JaxCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=jax.asarray(joint_vel))

    np.testing.assert_allclose(out_pos_np, np.asarray(out_pos_jax), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_vel_np, np.asarray(out_vel_jax), rtol=1e-6, atol=1e-6)


def test_build_observation_parity(spec: PolicySpec) -> None:
    jax = pytest.importorskip("jax.numpy")
    from policy_contract.jax.obs import build_observation_from_components as build_obs_jax
    from policy_contract.numpy.obs import build_observation_from_components as build_obs_np

    rng = np.random.RandomState(2)
    gravity = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
    angvel = rng.uniform(-2.0, 2.0, size=(3,)).astype(np.float32)
    joint_pos = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    joint_vel = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    foot = rng.uniform(0.0, 1.0, size=(4,)).astype(np.float32)
    prev_action = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    velocity_cmd = np.array([0.25], dtype=np.float32)

    obs_np = build_obs_np(
        spec=spec,
        gravity_local=gravity,
        angvel_heading_local=angvel,
        joint_pos_normalized=joint_pos,
        joint_vel_normalized=joint_vel,
        foot_switches=foot,
        prev_action=prev_action,
        velocity_cmd=velocity_cmd,
    )

    obs_jax = build_obs_jax(
        spec=spec,
        gravity_local=jax.asarray(gravity),
        angvel_heading_local=jax.asarray(angvel),
        joint_pos_normalized=jax.asarray(joint_pos),
        joint_vel_normalized=jax.asarray(joint_vel),
        foot_switches=jax.asarray(foot),
        prev_action=jax.asarray(prev_action),
        velocity_cmd=jax.asarray(velocity_cmd),
    )

    np.testing.assert_allclose(obs_np, np.asarray(obs_jax), rtol=1e-6, atol=1e-6)


def test_build_observation_signals_parity(spec: PolicySpec) -> None:
    jax = pytest.importorskip("jax.numpy")
    from policy_contract.jax.obs import build_observation as build_obs_jax
    from policy_contract.jax.signals import Signals as SignalsJax
    from policy_contract.jax.state import PolicyState as PolicyStateJax
    from policy_contract.numpy.obs import build_observation as build_obs_np
    from policy_contract.numpy.signals import Signals as SignalsNp
    from policy_contract.numpy.state import PolicyState as PolicyStateNp

    rng = np.random.RandomState(3)
    quat = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
    gyro = rng.uniform(-2.0, 2.0, size=(3,)).astype(np.float32)
    joint_pos = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    joint_vel = rng.uniform(-2.0, 2.0, size=(spec.model.action_dim,)).astype(np.float32)
    foot = rng.uniform(0.0, 1.0, size=(4,)).astype(np.float32)
    prev_action = rng.uniform(-1.0, 1.0, size=(spec.model.action_dim,)).astype(np.float32)
    velocity_cmd = np.array([0.5], dtype=np.float32)

    signals_np = SignalsNp(
        quat_xyzw=quat,
        gyro_rad_s=gyro,
        joint_pos_rad=joint_pos,
        joint_vel_rad_s=joint_vel,
        foot_switches=foot,
    )
    state_np = PolicyStateNp(prev_action=prev_action)

    signals_jax = SignalsJax(
        quat_xyzw=jax.asarray(quat),
        gyro_rad_s=jax.asarray(gyro),
        joint_pos_rad=jax.asarray(joint_pos),
        joint_vel_rad_s=jax.asarray(joint_vel),
        foot_switches=jax.asarray(foot),
    )
    state_jax = PolicyStateJax(prev_action=jax.asarray(prev_action))

    obs_np = build_obs_np(spec=spec, state=state_np, signals=signals_np, velocity_cmd=velocity_cmd)
    obs_jax = build_obs_jax(
        spec=spec,
        state=state_jax,
        signals=signals_jax,
        velocity_cmd=jax.asarray(velocity_cmd),
    )

    np.testing.assert_allclose(obs_np, np.asarray(obs_jax), rtol=1e-6, atol=1e-6)
