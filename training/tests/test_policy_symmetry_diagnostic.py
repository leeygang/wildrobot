from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from policy_contract.spec_builder import build_policy_spec
from policy_contract.jax.symmetry import (
    joint_mirror_transform,
    mirror_actions,
    mirror_observations,
)


_ACTUATORS = [
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


def _spec():
    return build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {
                "name": name,
                "range": [-1.0, 1.0],
                "policy_action_sign": 1.0,
                "max_velocity_rad_s": 10.0,
            }
            for name in _ACTUATORS
        ],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v8_cmd3d",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.0] * len(_ACTUATORS),
    )


def _slices(spec):
    result = {}
    offset = 0
    for field in spec.observation.layout:
        result[field.name] = slice(offset, offset + field.size)
        offset += field.size
    return result


def test_joint_mirror_transform_matches_wr_axes_and_is_involution() -> None:
    spec = _spec()
    source, sign = joint_mirror_transform(spec.robot.actuator_names)
    source = np.asarray(source)
    sign = np.asarray(sign)
    index = {name: i for i, name in enumerate(_ACTUATORS)}

    assert source[index["left_hip_roll"]] == index["right_hip_roll"]
    assert sign[index["left_hip_roll"]] == -1.0
    assert source[index["left_knee_pitch"]] == index["right_knee_pitch"]
    assert sign[index["left_knee_pitch"]] == 1.0
    assert source[index["waist_yaw"]] == index["waist_yaw"]
    assert sign[index["waist_yaw"]] == -1.0

    action = jnp.linspace(-1.0, 1.0, len(_ACTUATORS))
    np.testing.assert_allclose(
        mirror_actions(mirror_actions(action, spec), spec), action, atol=0.0
    )


def test_v8_observation_mirror_transforms_every_directional_channel() -> None:
    spec = _spec()
    fields = _slices(spec)
    obs = jnp.arange(spec.model.obs_dim, dtype=jnp.float32)
    mirrored = np.asarray(mirror_observations(obs, spec))
    original = np.asarray(obs)

    np.testing.assert_array_equal(
        mirrored[fields["gravity_local"]],
        original[fields["gravity_local"]] * np.array([1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        mirrored[fields["angvel_heading_local"]],
        original[fields["angvel_heading_local"]] * np.array([-1.0, 1.0, -1.0]),
    )
    np.testing.assert_array_equal(
        mirrored[fields["foot_switches"]],
        original[fields["foot_switches"]][[2, 3, 0, 1]],
    )
    np.testing.assert_array_equal(
        mirrored[fields["loc_ref_phase_sin_cos"]],
        -original[fields["loc_ref_phase_sin_cos"]],
    )
    np.testing.assert_array_equal(
        mirrored[fields["velocity_cmd_lateral_yaw"]],
        -original[fields["velocity_cmd_lateral_yaw"]],
    )

    np.testing.assert_array_equal(
        mirror_observations(mirror_observations(obs, spec), spec), obs
    )


def test_v8_history_frames_use_same_joint_and_sensor_transform() -> None:
    spec = _spec()
    fields = _slices(spec)
    obs = np.zeros(spec.model.obs_dim, dtype=np.float32)
    action_dim = spec.model.action_dim
    bundle_size = 3 + 4 + 3 * action_dim
    history = np.arange(
        15 * bundle_size, dtype=np.float32
    ).reshape(15, bundle_size)
    obs[fields["proprio_history"]] = history.reshape(-1)

    mirrored = np.asarray(mirror_observations(jnp.asarray(obs), spec))
    mirrored_history = mirrored[fields["proprio_history"]].reshape(
        15, bundle_size
    )
    np.testing.assert_array_equal(
        mirrored_history[:, :3], history[:, :3] * np.array([-1.0, 1.0, -1.0])
    )
    np.testing.assert_array_equal(
        mirrored_history[:, 3:7], history[:, 3:7][:, [2, 3, 0, 1]]
    )
    for block_index in range(3):
        start = 7 + block_index * action_dim
        expected = np.asarray(
            mirror_actions(jnp.asarray(history[:, start : start + action_dim]), spec)
        )
        np.testing.assert_array_equal(
            mirrored_history[:, start : start + action_dim], expected
        )
