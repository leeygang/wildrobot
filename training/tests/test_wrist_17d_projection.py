from __future__ import annotations

import numpy as np
import pytest
import mujoco

from training.policy_migration.wrist_17d import (
    initialize_projected_policy_params,
    project_action,
    project_v8_observation,
    v8_observation_dim,
)
from training.scripts.distill_walking_21d_to_17d import (
    DEFAULT_TEACHER_POLICY_SPEC,
    DEFAULT_TEACHER_ROBOT_XML,
    _build_legacy_teacher_scene,
)


FULL_NAMES = tuple(f"joint_{idx}" for idx in range(21))
ACTIVE_NAMES = tuple(name for idx, name in enumerate(FULL_NAMES) if idx not in {4, 5, 14, 15})
KEEP = np.asarray([idx for idx in range(21) if idx not in {4, 5, 14, 15}])


def test_v8_projection_removes_wrist_channels_from_every_actuator_block() -> None:
    obs = np.arange(v8_observation_dim(21), dtype=np.float32)
    projected = project_v8_observation(
        obs,
        full_actuator_names=FULL_NAMES,
        active_actuator_names=ACTIVE_NAMES,
    )
    assert projected.shape == (937,)

    src = 6
    dst = 6
    np.testing.assert_array_equal(projected[dst : dst + 17], obs[src : src + 21][KEEP])
    src += 21
    dst += 17
    np.testing.assert_array_equal(projected[dst : dst + 17], obs[src : src + 21][KEEP])

    history_src = 6 + 21 + 21 + 4 + 21 + 3
    history_dst = 6 + 17 + 17 + 4 + 17 + 3
    full_frame = 7 + 3 * 21
    active_frame = 7 + 3 * 17
    for frame in (0, 7, 14):
        source_frame = obs[
            history_src + frame * full_frame : history_src + (frame + 1) * full_frame
        ]
        projected_frame = projected[
            history_dst + frame * active_frame : history_dst + (frame + 1) * active_frame
        ]
        expected = np.concatenate(
            [
                source_frame[:7],
                source_frame[7 : 7 + 21][KEEP],
                source_frame[7 + 21 : 7 + 42][KEEP],
                source_frame[7 + 42 :][KEEP],
            ]
        )
        np.testing.assert_array_equal(projected_frame, expected)


def test_v8_projection_supports_batches_and_projects_actions() -> None:
    obs = np.zeros((3, v8_observation_dim(21)), dtype=np.float32)
    assert project_v8_observation(
        obs,
        full_actuator_names=FULL_NAMES,
        active_actuator_names=ACTIVE_NAMES,
    ).shape == (3, 937)
    actions = np.arange(42, dtype=np.float32).reshape(2, 21)
    np.testing.assert_array_equal(
        project_action(
            actions,
            full_actuator_names=FULL_NAMES,
            active_actuator_names=ACTIVE_NAMES,
        ),
        actions[:, KEEP],
    )


def test_v8_projection_rejects_wrong_contract() -> None:
    with pytest.raises(ValueError, match="final dimension"):
        project_v8_observation(
            np.zeros((1128,), dtype=np.float32),
            full_actuator_names=FULL_NAMES,
            active_actuator_names=ACTIVE_NAMES,
        )


def test_projected_policy_initialization_selects_input_and_output_channels() -> None:
    teacher = {
        "params": {
            "hidden_0": {
                "kernel": np.arange(v8_observation_dim(21) * 4, dtype=np.float32).reshape(
                    v8_observation_dim(21), 4
                ),
                "bias": np.arange(4, dtype=np.float32),
            },
            "hidden_1": {
                "kernel": np.arange(12, dtype=np.float32).reshape(4, 3),
                "bias": np.arange(3, dtype=np.float32),
            },
            "hidden_2": {
                "kernel": np.arange(3 * 42, dtype=np.float32).reshape(3, 42),
                "bias": np.arange(42, dtype=np.float32),
            },
        }
    }
    student = {
        "params": {
            "hidden_0": {
                "kernel": np.zeros((v8_observation_dim(17), 4), dtype=np.float32),
                "bias": np.zeros((4,), dtype=np.float32),
            },
            "hidden_1": {
                "kernel": np.zeros((4, 3), dtype=np.float32),
                "bias": np.zeros((3,), dtype=np.float32),
            },
            "hidden_2": {
                "kernel": np.zeros((3, 34), dtype=np.float32),
                "bias": np.zeros((34,), dtype=np.float32),
            },
        }
    }
    projected = initialize_projected_policy_params(
        teacher,
        student,
        full_actuator_names=FULL_NAMES,
        active_actuator_names=ACTIVE_NAMES,
    )
    projected_layers = projected["params"]
    marker_projection = project_v8_observation(
        np.arange(v8_observation_dim(21), dtype=np.float32),
        full_actuator_names=FULL_NAMES,
        active_actuator_names=ACTIVE_NAMES,
    ).astype(np.int32)
    np.testing.assert_array_equal(
        projected_layers["hidden_0"]["kernel"],
        teacher["params"]["hidden_0"]["kernel"][marker_projection],
    )
    np.testing.assert_array_equal(
        projected_layers["hidden_1"]["kernel"],
        teacher["params"]["hidden_1"]["kernel"],
    )
    logit_keep = np.concatenate([KEEP, 21 + KEEP])
    np.testing.assert_array_equal(
        projected_layers["hidden_2"]["bias"],
        teacher["params"]["hidden_2"]["bias"][logit_keep],
    )


def test_archived_teacher_scene_remains_21d_after_active_model_removal() -> None:
    scene_path = _build_legacy_teacher_scene(
        robot_xml_path=DEFAULT_TEACHER_ROBOT_XML,
        policy_spec_path=DEFAULT_TEACHER_POLICY_SPEC,
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)

    assert (model.nq, model.nv, model.nu) == (28, 27, 21)
    assert model.key_qpos.shape == (1, 28)
