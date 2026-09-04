from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from policy_contract.jax.obs import build_observation_from_components as build_jax_obs
from policy_contract.jax.symmetry import mirror_observations
from policy_contract.numpy.obs import (
    build_observation_from_components as build_numpy_obs,
)
from policy_contract.spec_builder import build_policy_spec
from training.policy_migration.contact_free import (
    contact_free_observation_dim,
    expand_contact_free_policy_params,
    project_v8_observation,
    project_v8_policy_params,
    retained_v8_observation_indices,
    v8_observation_dim,
)


ACTION_DIM = 17


def _spec(layout_id: str):
    names = [f"joint_{index}" for index in range(ACTION_DIM)]
    return build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {
                "name": name,
                "range": [-1.0, 1.0],
                "policy_action_sign": 1.0,
                "max_velocity_rad_s": 10.0,
            }
            for name in names
        ],
        action_filter_alpha=0.0,
        layout_id=layout_id,
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * ACTION_DIM,
    )


def _obs_kwargs(*, foot_switches: np.ndarray) -> dict:
    bundle_size = 3 + 3 * ACTION_DIM
    return {
        "gravity_local": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "angvel_heading_local": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "joint_pos_normalized": np.linspace(-0.5, 0.5, ACTION_DIM, dtype=np.float32),
        "joint_vel_normalized": np.linspace(0.5, -0.5, ACTION_DIM, dtype=np.float32),
        "foot_switches": foot_switches,
        "prev_action": np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32),
        "velocity_cmd": np.array([0.13, 0.0, 0.0], dtype=np.float32),
        "velocity_cmd_lateral_yaw": np.zeros(2, dtype=np.float32),
        "loc_ref_phase_sin_cos": np.array([0.0, 1.0], dtype=np.float32),
        "proprio_history": np.arange(15 * bundle_size, dtype=np.float32),
    }


def test_contact_free_layout_has_no_foot_channels_and_expected_dimension() -> None:
    spec = _spec("wr_obs_v11_cmd3d_proprio")

    assert spec.model.obs_dim == 873
    assert spec.model.obs_dim == contact_free_observation_dim(ACTION_DIM)
    assert "foot_switches" not in {field.name for field in spec.observation.layout}


def test_contact_free_numpy_and_jax_observations_ignore_switch_values() -> None:
    spec = _spec("wr_obs_v11_cmd3d_proprio")
    zeros = _obs_kwargs(foot_switches=np.zeros(4, dtype=np.float32))
    ones = _obs_kwargs(foot_switches=np.ones(4, dtype=np.float32))

    numpy_zero = build_numpy_obs(spec=spec, **zeros)
    numpy_one = build_numpy_obs(spec=spec, **ones)
    jax_one = np.asarray(
        build_jax_obs(
            spec=spec,
            **{key: jnp.asarray(value) for key, value in ones.items()},
        )
    )

    np.testing.assert_array_equal(numpy_zero, numpy_one)
    np.testing.assert_array_equal(numpy_one, jax_one)


def test_v8_projection_removes_current_and_historical_contacts() -> None:
    source = np.arange(v8_observation_dim(ACTION_DIM), dtype=np.float32)
    projected = project_v8_observation(source, action_dim=ACTION_DIM)
    keep = retained_v8_observation_indices(ACTION_DIM)

    assert source.shape == (937,)
    assert projected.shape == (873,)
    np.testing.assert_array_equal(projected, source[keep])
    assert len(set(range(source.size)) - set(keep.tolist())) == 64


def test_projected_actor_matches_v8_actor_when_contacts_are_zero() -> None:
    rng = np.random.default_rng(7)
    source_kernel = rng.normal(size=(937, 8)).astype(np.float32)
    source_bias = rng.normal(size=(8,)).astype(np.float32)
    policy_params = {
        "params": {
            "hidden_0": {"kernel": source_kernel, "bias": source_bias},
            "hidden_1": {
                "kernel": rng.normal(size=(8, 4)).astype(np.float32),
                "bias": rng.normal(size=(4,)).astype(np.float32),
            },
        }
    }
    source_obs = rng.normal(size=(937,)).astype(np.float32)
    keep = retained_v8_observation_indices(ACTION_DIM)
    removed = np.ones(source_obs.size, dtype=bool)
    removed[keep] = False
    source_obs[removed] = 0.0

    projected = project_v8_policy_params(policy_params, action_dim=ACTION_DIM)
    target_obs = project_v8_observation(source_obs, action_dim=ACTION_DIM)

    source_hidden = source_obs @ source_kernel + source_bias
    target_hidden = (
        target_obs @ projected["params"]["hidden_0"]["kernel"]
        + projected["params"]["hidden_0"]["bias"]
    )
    np.testing.assert_allclose(target_hidden, source_hidden, atol=1e-5)


def test_contact_free_symmetry_is_an_involution() -> None:
    names = [
        "waist_yaw",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_elbow_pitch",
        "left_hip_pitch",
        "left_hip_roll",
        "left_knee_pitch",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_elbow_pitch",
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee_pitch",
        "right_ankle_pitch",
        "right_ankle_roll",
    ]
    spec = build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {
                "name": name,
                "range": [-1.0, 1.0],
                "policy_action_sign": 1.0,
                "max_velocity_rad_s": 10.0,
            }
            for name in names
        ],
        action_filter_alpha=0.0,
        layout_id="wr_obs_v11_cmd3d_proprio",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * ACTION_DIM,
    )
    obs = jnp.arange(spec.model.obs_dim, dtype=jnp.float32)

    np.testing.assert_array_equal(
        mirror_observations(mirror_observations(obs, spec), spec), obs
    )


def test_expanded_actor_preserves_contact_free_actor_and_zeros_contact_weights(
) -> None:
    rng = np.random.default_rng(11)
    source_kernel = rng.normal(size=(873, 8)).astype(np.float32)
    source_bias = rng.normal(size=(8,)).astype(np.float32)
    policy_params = {
        "params": {
            "hidden_0": {"kernel": source_kernel, "bias": source_bias},
            "hidden_1": {
                "kernel": rng.normal(size=(8, 4)).astype(np.float32),
                "bias": rng.normal(size=(4,)).astype(np.float32),
            },
        }
    }
    source_obs = rng.normal(size=(873,)).astype(np.float32)
    target_obs = rng.normal(size=(937,)).astype(np.float32)
    keep = retained_v8_observation_indices(ACTION_DIM)
    target_obs[keep] = source_obs

    expanded = expand_contact_free_policy_params(
        policy_params, action_dim=ACTION_DIM
    )
    expanded_kernel = expanded["params"]["hidden_0"]["kernel"]
    removed = np.ones(target_obs.size, dtype=bool)
    removed[keep] = False

    source_hidden = source_obs @ source_kernel + source_bias
    target_hidden = target_obs @ expanded_kernel + source_bias
    np.testing.assert_allclose(target_hidden, source_hidden, atol=1e-5)
    np.testing.assert_array_equal(expanded_kernel[removed], 0.0)
    np.testing.assert_array_equal(
        project_v8_policy_params(expanded, action_dim=ACTION_DIM)["params"][
            "hidden_0"
        ]["kernel"],
        source_kernel,
    )
