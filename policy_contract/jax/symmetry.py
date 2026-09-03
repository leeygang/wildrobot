"""Sagittal reflection for the active WildRobot walking contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from policy_contract.spec import PROPRIO_HISTORY_FRAMES, PolicySpec


# Joint-coordinate signs after swapping left/right.  These follow from the
# actuator axes in assets/v2/mujoco_robot_config.json under the axial-vector
# reflection rule a' = -diag(1, -1, 1) a.
_PAIRED_JOINT_SIGNS = {
    "shoulder_pitch": -1.0,
    "shoulder_roll": -1.0,
    "elbow_pitch": 1.0,
    "hip_pitch": -1.0,
    "hip_roll": -1.0,
    "knee_pitch": 1.0,
    "ankle_pitch": 1.0,
    "ankle_roll": -1.0,
}


def _field_slices(spec: PolicySpec) -> dict[str, slice]:
    fields: dict[str, slice] = {}
    offset = 0
    for field in spec.observation.layout:
        fields[field.name] = slice(offset, offset + int(field.size))
        offset += int(field.size)
    if offset != int(spec.model.obs_dim):
        raise ValueError(
            f"Observation layout sums to {offset}, expected {spec.model.obs_dim}"
        )
    return fields


def joint_mirror_transform(
    actuator_names: list[str] | tuple[str, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return source indices and signs for sagittal joint reflection."""
    names = list(actuator_names)
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("Actuator names must be unique")

    sources: list[int] = []
    signs: list[float] = []
    for name in names:
        if name == "waist_yaw":
            sources.append(index[name])
            signs.append(-1.0)
            continue
        if name.startswith("left_"):
            suffix = name[len("left_") :]
            partner = f"right_{suffix}"
        elif name.startswith("right_"):
            suffix = name[len("right_") :]
            partner = f"left_{suffix}"
        else:
            raise ValueError(f"Unsupported unpaired actuator: {name!r}")
        if suffix not in _PAIRED_JOINT_SIGNS:
            raise ValueError(f"No mirror sign defined for actuator: {name!r}")
        if partner not in index:
            raise ValueError(f"Missing mirror partner {partner!r} for {name!r}")
        sources.append(index[partner])
        signs.append(_PAIRED_JOINT_SIGNS[suffix])

    return (
        jnp.asarray(sources, dtype=jnp.int32),
        jnp.asarray(signs, dtype=jnp.float32),
    )


def mirror_actions(actions: jax.Array, spec: PolicySpec) -> jax.Array:
    """Mirror normalized residual actions in PolicySpec actuator order."""
    source, sign = joint_mirror_transform(spec.robot.actuator_names)
    values = jnp.asarray(actions, dtype=jnp.float32)
    return jnp.take(values, source, axis=-1) * sign


def _mirror_foot_switches(values: jax.Array) -> jax.Array:
    # [left_toe, left_heel, right_toe, right_heel]
    return jnp.take(values, jnp.asarray([2, 3, 0, 1]), axis=-1)


def mirror_observations(obs: jax.Array, spec: PolicySpec) -> jax.Array:
    """Mirror supported walking observations across the sagittal plane."""
    contact_free = spec.observation.layout_id == "wr_obs_v11_cmd3d_proprio"
    if spec.observation.layout_id not in {
        "wr_obs_v8_cmd3d",
        "wr_obs_v11_cmd3d_proprio",
    }:
        raise ValueError(
            "Walking symmetry supports wr_obs_v8_cmd3d and "
            "wr_obs_v11_cmd3d_proprio only; got "
            f"{spec.observation.layout_id!r}"
        )

    fields = _field_slices(spec)
    expected = {
        "gravity_local",
        "angvel_heading_local",
        "joint_pos_normalized",
        "joint_vel_normalized",
        "prev_action",
        "velocity_cmd",
        "loc_ref_phase_sin_cos",
        "proprio_history",
        "velocity_cmd_lateral_yaw",
        "padding",
    }
    if not contact_free:
        expected.add("foot_switches")
    if set(fields) != expected:
        raise ValueError(
            "Unexpected wr_obs_v8_cmd3d fields: "
            f"missing={sorted(expected - set(fields))}, "
            f"extra={sorted(set(fields) - expected)}"
        )

    values = jnp.asarray(obs, dtype=jnp.float32)
    if values.shape[-1] != int(spec.model.obs_dim):
        raise ValueError(
            f"Observation dim mismatch: got {values.shape[-1]}, "
            f"expected {spec.model.obs_dim}"
        )
    mirrored = values

    mirrored = mirrored.at[..., fields["gravity_local"]].set(
        values[..., fields["gravity_local"]]
        * jnp.asarray([1.0, -1.0, 1.0], dtype=jnp.float32)
    )
    mirrored = mirrored.at[..., fields["angvel_heading_local"]].set(
        values[..., fields["angvel_heading_local"]]
        * jnp.asarray([-1.0, 1.0, -1.0], dtype=jnp.float32)
    )
    for field_name in (
        "joint_pos_normalized",
        "joint_vel_normalized",
        "prev_action",
    ):
        mirrored = mirrored.at[..., fields[field_name]].set(
            mirror_actions(values[..., fields[field_name]], spec)
        )
    if not contact_free:
        mirrored = mirrored.at[..., fields["foot_switches"]].set(
            _mirror_foot_switches(values[..., fields["foot_switches"]])
        )
    # Swapping support legs advances an alternating gait by pi.
    mirrored = mirrored.at[..., fields["loc_ref_phase_sin_cos"]].set(
        -values[..., fields["loc_ref_phase_sin_cos"]]
    )
    # vx is invariant. vy and yaw rate reverse sign.
    mirrored = mirrored.at[..., fields["velocity_cmd_lateral_yaw"]].set(
        -values[..., fields["velocity_cmd_lateral_yaw"]]
    )

    action_dim = int(spec.model.action_dim)
    contact_size = 0 if contact_free else 4
    bundle_size = 3 + contact_size + 3 * action_dim
    history = values[..., fields["proprio_history"]].reshape(
        values.shape[:-1] + (PROPRIO_HISTORY_FRAMES, bundle_size)
    )
    mirrored_history = history
    mirrored_history = mirrored_history.at[..., 0:3].set(
        history[..., 0:3]
        * jnp.asarray([-1.0, 1.0, -1.0], dtype=jnp.float32)
    )
    offset = 3
    if not contact_free:
        mirrored_history = mirrored_history.at[..., 3:7].set(
            _mirror_foot_switches(history[..., 3:7])
        )
        offset = 7
    for _ in range(3):
        mirrored_history = mirrored_history.at[
            ..., offset : offset + action_dim
        ].set(
            mirror_actions(history[..., offset : offset + action_dim], spec)
        )
        offset += action_dim
    mirrored = mirrored.at[..., fields["proprio_history"]].set(
        mirrored_history.reshape(values.shape[:-1] + (-1,))
    )
    return mirrored
