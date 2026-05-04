from __future__ import annotations

import json

import numpy as np
import pytest

from policy_contract.spec import PolicySpec, validate_spec


def _valid_spec_dict() -> dict:
    return {
        "contract_name": "wildrobot_policy",
        "contract_version": "1.0.0",
        "spec_version": 2,
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
                    "policy_action_sign": 1.0,
                    "max_velocity_rad_s": 10.0,
                },
                "left_hip_roll": {
                    "range_min_rad": -1.5,
                    "range_max_rad": 0.2,
                    "policy_action_sign": 1.0,
                    "max_velocity_rad_s": 9.0,
                },
                "left_knee_pitch": {
                    "range_min_rad": 0.0,
                    "range_max_rad": 1.3,
                    "policy_action_sign": 1.0,
                    "max_velocity_rad_s": 8.0,
                },
                "left_ankle_pitch": {
                    "range_min_rad": -0.7,
                    "range_max_rad": 0.8,
                    "policy_action_sign": 1.0,
                    "max_velocity_rad_s": 7.0,
                },
                "right_hip_pitch": {
                    "range_min_rad": -1.5,
                    "range_max_rad": 0.1,
                    "policy_action_sign": -1.0,
                    "max_velocity_rad_s": 10.0,
                },
                "right_hip_roll": {
                    "range_min_rad": -0.2,
                    "range_max_rad": 1.5,
                    "policy_action_sign": -1.0,
                    "max_velocity_rad_s": 9.0,
                },
                "right_knee_pitch": {
                    "range_min_rad": 0.0,
                    "range_max_rad": 1.3,
                    "policy_action_sign": 1.0,
                    "max_velocity_rad_s": 8.0,
                },
                "right_ankle_pitch": {
                    "range_min_rad": -0.7,
                    "range_max_rad": 0.8,
                    "policy_action_sign": 1.0,
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


# v0.20.1 wr_obs_v6_offline_ref_history parity ---------------------------------

def _v6_spec_dict() -> dict:
    spec = _valid_spec_dict()
    spec["observation"]["layout_id"] = "wr_obs_v6_offline_ref_history"
    return spec


@pytest.fixture()
def spec_v6() -> PolicySpec:
    spec = PolicySpec.from_json(json.dumps(_v6_spec_dict()))
    # The layout list is the v1 default but layout_id is v6;
    # validate_spec is layout-aware and would reject the mismatch.
    # The parity test only exercises builder dispatch on layout_id,
    # not full schema validation.
    return spec


def test_build_observation_parity_v6(spec_v6: PolicySpec) -> None:
    """JAX/NumPy parity for the v6 layout including proprio_history."""
    from policy_contract.spec import PROPRIO_HISTORY_FRAMES

    jax = pytest.importorskip("jax.numpy")
    from policy_contract.jax.obs import build_observation_from_components as build_obs_jax
    from policy_contract.numpy.obs import build_observation_from_components as build_obs_np

    rng = np.random.RandomState(6)
    n_act = spec_v6.model.action_dim
    bundle = 3 + 4 + 3 * n_act
    proprio_history = rng.uniform(
        -1.0, 1.0, size=(PROPRIO_HISTORY_FRAMES, bundle)
    ).astype(np.float32)

    common_kwargs = dict(
        spec=spec_v6,
        gravity_local=rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
        angvel_heading_local=rng.uniform(-2.0, 2.0, size=(3,)).astype(np.float32),
        joint_pos_normalized=rng.uniform(-1.0, 1.0, size=(n_act,)).astype(np.float32),
        joint_vel_normalized=rng.uniform(-1.0, 1.0, size=(n_act,)).astype(np.float32),
        foot_switches=rng.uniform(0.0, 1.0, size=(4,)).astype(np.float32),
        prev_action=rng.uniform(-1.0, 1.0, size=(n_act,)).astype(np.float32),
        velocity_cmd=np.array([0.15], dtype=np.float32),
        loc_ref_phase_sin_cos=rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32),
        loc_ref_stance_foot=np.array([1.0], dtype=np.float32),
        loc_ref_next_foothold=rng.uniform(-0.2, 0.2, size=(2,)).astype(np.float32),
        loc_ref_swing_pos=rng.uniform(-0.2, 0.2, size=(3,)).astype(np.float32),
        loc_ref_swing_vel=rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32),
        loc_ref_pelvis_targets=rng.uniform(0.3, 0.5, size=(3,)).astype(np.float32),
        loc_ref_history=rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32),
        loc_ref_q_ref=rng.uniform(-1.0, 1.0, size=(n_act,)).astype(np.float32),
        loc_ref_pelvis_pos=rng.uniform(-0.1, 0.1, size=(3,)).astype(np.float32),
        loc_ref_pelvis_vel=rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32),
        loc_ref_left_foot_pos=rng.uniform(-0.2, 0.2, size=(3,)).astype(np.float32),
        loc_ref_right_foot_pos=rng.uniform(-0.2, 0.2, size=(3,)).astype(np.float32),
        loc_ref_left_foot_vel=rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32),
        loc_ref_right_foot_vel=rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32),
        loc_ref_contact_mask=np.array([1.0, 0.0], dtype=np.float32),
        proprio_history=proprio_history,
    )

    obs_np = build_obs_np(**common_kwargs)
    jax_kwargs = {
        k: (jax.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in common_kwargs.items()
    }
    obs_jax = build_obs_jax(**jax_kwargs)

    np.testing.assert_allclose(obs_np, np.asarray(obs_jax), rtol=1e-6, atol=1e-6)


def test_v6_obs_dim_matches_design(spec_v6: PolicySpec) -> None:
    """v6 obs_dim = v4 obs_dim + (n_act + 20) ref-window + history.

    v6 is the active locomotion contract (v5 was deprecated along
    with the high-confidence prep).  Anchoring against v4 keeps the
    test independent of the deprecated layout while still covering
    the full v6 channel inventory.
    """
    from policy_contract.numpy.obs import build_observation_from_components as build_obs_np
    from policy_contract.spec import PROPRIO_HISTORY_FRAMES

    n_act = spec_v6.model.action_dim
    bundle = 3 + 4 + 3 * n_act
    history_size = PROPRIO_HISTORY_FRAMES * bundle
    # Reference-window block (q_ref + pelvis pos/vel + per-foot
    # pos/vel + contact_mask): n_act + 20 floats.  Same set v5 added
    # over v4; v6 inherits and appends history on top.
    ref_window_size = n_act + 20

    base_kwargs = dict(
        gravity_local=np.zeros(3, np.float32),
        angvel_heading_local=np.zeros(3, np.float32),
        joint_pos_normalized=np.zeros(n_act, np.float32),
        joint_vel_normalized=np.zeros(n_act, np.float32),
        foot_switches=np.zeros(4, np.float32),
        prev_action=np.zeros(n_act, np.float32),
        velocity_cmd=np.zeros(1, np.float32),
    )

    spec_v4_dict = _valid_spec_dict()
    spec_v4_dict["observation"]["layout_id"] = "wr_obs_v4"
    spec_v4 = PolicySpec.from_json(json.dumps(spec_v4_dict))
    obs_v4 = build_obs_np(spec=spec_v4, **base_kwargs)
    obs_v6 = build_obs_np(
        spec=spec_v6,
        proprio_history=np.zeros((PROPRIO_HISTORY_FRAMES, bundle), np.float32),
        **base_kwargs,
    )

    expected_delta = ref_window_size + history_size
    assert obs_v6.shape[0] - obs_v4.shape[0] == expected_delta, (
        f"v6 should add (ref_window={ref_window_size}) + (history="
        f"{history_size}) = {expected_delta} channels over v4; "
        f"got delta = {obs_v6.shape[0] - obs_v4.shape[0]}"
    )


def test_v6_requires_proprio_history(spec_v6: PolicySpec) -> None:
    """``proprio_history=None`` must raise for the v6 layout — the env
    is required to wire a per-step rolling buffer."""
    from policy_contract.numpy.obs import build_observation_from_components as build_obs_np

    n_act = spec_v6.model.action_dim
    with pytest.raises(ValueError, match="requires proprio_history"):
        build_obs_np(
            spec=spec_v6,
            gravity_local=np.zeros(3, np.float32),
            angvel_heading_local=np.zeros(3, np.float32),
            joint_pos_normalized=np.zeros(n_act, np.float32),
            joint_vel_normalized=np.zeros(n_act, np.float32),
            foot_switches=np.zeros(4, np.float32),
            prev_action=np.zeros(n_act, np.float32),
            velocity_cmd=np.zeros(1, np.float32),
            proprio_history=None,
        )
