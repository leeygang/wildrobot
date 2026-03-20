# training/tests/test_reference_motion.py
"""Reference-motion format and retargeting contract tests for v0.16.x."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from training.reference_motion.loader import (
    ReferenceMotionClip,
    load_reference_motion,
    save_reference_motion_clip,
)
from training.reference_motion.phase import phase_distance, phase_to_index, phase_to_sin_cos, wrap_phase
from training.reference_motion.retarget import (
    JointLimits,
    generate_bootstrap_nominal_walk_clip,
    retarget_wildrobot_clip,
)


def _make_dummy_clip(actuator_names: list[str], frame_count: int = 8) -> ReferenceMotionClip:
    t = np.arange(frame_count, dtype=np.float32)
    phase = (t / float(frame_count)).astype(np.float32)
    a = len(actuator_names)
    metadata = {
        "name": "dummy",
        "source_name": "unit_test",
        "source_format": "unit",
        "frame_count": frame_count,
        "dt": 0.02,
        "actuator_names": actuator_names,
        "loop_mode": "wrap",
        "phase_mode": "normalized_cycle",
    }
    return ReferenceMotionClip(
        name="dummy",
        dt=0.02,
        phase=phase,
        root_pos=np.stack([0.01 * t, np.zeros_like(t), np.full_like(t, 0.45)], axis=-1),
        root_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frame_count, 1)),
        root_lin_vel=np.zeros((frame_count, 3), dtype=np.float32),
        root_ang_vel=np.zeros((frame_count, 3), dtype=np.float32),
        joint_pos=np.zeros((frame_count, a), dtype=np.float32),
        joint_vel=np.zeros((frame_count, a), dtype=np.float32),
        left_foot_pos=np.zeros((frame_count, 3), dtype=np.float32),
        right_foot_pos=np.zeros((frame_count, 3), dtype=np.float32),
        left_foot_contact=np.ones((frame_count,), dtype=np.float32),
        right_foot_contact=np.ones((frame_count,), dtype=np.float32),
        metadata=metadata,
    )


@pytest.mark.unit
def test_loader_validates_shapes_and_metadata(tmp_path: Path, robot_config):
    actuator_names = list(robot_config.actuator_names)
    clip = _make_dummy_clip(actuator_names)
    npz_path = tmp_path / "clip.npz"
    meta_path = tmp_path / "clip.json"
    save_reference_motion_clip(clip, npz_path=npz_path, metadata_path=meta_path)

    loaded = load_reference_motion(npz_path, meta_path, actuator_names=actuator_names)
    assert loaded.frame_count == clip.frame_count
    assert loaded.actuator_dim == len(actuator_names)
    assert loaded.metadata["loop_mode"] == "wrap"
    assert loaded.metadata["phase_mode"] == "normalized_cycle"


@pytest.mark.unit
def test_loader_rejects_metadata_mismatch(tmp_path: Path, robot_config):
    actuator_names = list(robot_config.actuator_names)
    clip = _make_dummy_clip(actuator_names)
    npz_path = tmp_path / "clip.npz"
    meta_path = tmp_path / "clip.json"
    save_reference_motion_clip(clip, npz_path=npz_path, metadata_path=meta_path)

    bad_meta = json.loads(meta_path.read_text())
    bad_meta["frame_count"] = bad_meta["frame_count"] + 1
    meta_path.write_text(json.dumps(bad_meta))

    with pytest.raises(ValueError, match="frame_count mismatch"):
        load_reference_motion(npz_path, meta_path, actuator_names=actuator_names)


@pytest.mark.unit
def test_phase_utils_wrap_and_index():
    wrapped = wrap_phase(np.array([-0.1, 0.0, 0.2, 1.0, 1.4], dtype=np.float32))
    assert np.all(wrapped >= 0.0) and np.all(wrapped < 1.0)
    idx = phase_to_index(np.array([0.0, 0.24, 0.99], dtype=np.float32), frame_count=10)
    assert np.array_equal(idx, np.array([0, 2, 9], dtype=np.int32))
    d = phase_distance(np.array([0.98], dtype=np.float32), np.array([0.02], dtype=np.float32))
    assert float(d[0]) < 0.1
    sc = phase_to_sin_cos(np.array([0.25], dtype=np.float32))
    assert np.allclose(sc[0], np.array([1.0, 0.0], dtype=np.float32), atol=1e-5)


@pytest.mark.unit
def test_loader_enforces_actuator_alignment(tmp_path: Path, robot_config):
    actuator_names = list(robot_config.actuator_names)
    clip = _make_dummy_clip(actuator_names)
    npz_path = tmp_path / "clip.npz"
    meta_path = tmp_path / "clip.json"
    save_reference_motion_clip(clip, npz_path=npz_path, metadata_path=meta_path)

    shuffled = actuator_names.copy()
    shuffled.reverse()
    with pytest.raises(ValueError, match="Actuator name mismatch"):
        load_reference_motion(npz_path, meta_path, actuator_names=shuffled)


@pytest.mark.unit
def test_retarget_and_bootstrap_clip_sanity(robot_config):
    actuator_names = list(robot_config.actuator_names)
    lower = np.asarray([joint["range"][0] for joint in robot_config.actuated_joints], dtype=np.float32)
    upper = np.asarray([joint["range"][1] for joint in robot_config.actuated_joints], dtype=np.float32)
    limits = JointLimits(lower=lower, upper=upper)

    clip = generate_bootstrap_nominal_walk_clip(
        actuator_names=actuator_names,
        joint_limits=limits,
        frame_count=24,
        dt=0.02,
        nominal_forward_velocity_mps=0.10,
        name="bootstrap_test",
    )
    retargeted = retarget_wildrobot_clip(clip, joint_limits=limits)

    assert retargeted.joint_pos.shape == (24, len(actuator_names))
    assert np.all(retargeted.joint_pos <= upper + 1e-6)
    assert np.all(retargeted.joint_pos >= lower - 1e-6)
    stance = retargeted.left_foot_pos[:, 1] - retargeted.right_foot_pos[:, 1]
    assert np.all(stance >= 0.14 - 1e-6)
    assert np.all(stance <= 0.26 + 1e-6)
    assert np.mean(retargeted.root_pos[:, 2]) > 0.38


@pytest.mark.unit
def test_real_retarget_v002_contract_and_sanity(robot_config, project_root: Path):
    actuator_names = list(robot_config.actuator_names)
    npz_path = project_root / "training" / "reference_motion" / "wildrobot_walk_forward_v002.npz"
    meta_path = project_root / "training" / "reference_motion" / "wildrobot_walk_forward_v002.json"
    assert npz_path.exists(), f"Missing retargeted clip: {npz_path}"
    assert meta_path.exists(), f"Missing retargeted metadata: {meta_path}"

    clip = load_reference_motion(npz_path, meta_path, actuator_names=actuator_names)
    metadata = clip.metadata
    for key in (
        "name",
        "source_name",
        "source_format",
        "frame_count",
        "dt",
        "actuator_names",
        "loop_mode",
        "phase_mode",
        "retarget_version",
        "nominal_forward_velocity_mps",
        "notes",
    ):
        assert key in metadata, f"Missing metadata field: {key}"

    lower = np.asarray([joint["range"][0] for joint in robot_config.actuated_joints], dtype=np.float32)
    upper = np.asarray([joint["range"][1] for joint in robot_config.actuated_joints], dtype=np.float32)
    assert np.all(clip.joint_pos >= lower - 1e-6)
    assert np.all(clip.joint_pos <= upper + 1e-6)
    assert np.isfinite(clip.joint_vel).all()
    assert np.isfinite(clip.root_lin_vel).all()
    assert np.isfinite(clip.root_ang_vel).all()

    # Velocity smoothness sanity: large spikes would destabilize teacher tracking.
    assert float(np.percentile(np.abs(clip.joint_vel), 99)) < 8.0
    assert float(np.percentile(np.abs(np.diff(clip.joint_vel, axis=0)), 99)) < 8.0
    assert float(np.mean(clip.root_lin_vel[:, 0])) > 0.05
    quat_norm = np.linalg.norm(clip.root_quat, axis=1)
    assert np.allclose(quat_norm, 1.0, atol=1e-3)

    # Guard against quaternion-convention regressions from source motion ingestion.
    root_quat_xyzw = np.stack(
        [clip.root_quat[:, 1], clip.root_quat[:, 2], clip.root_quat[:, 3], clip.root_quat[:, 0]],
        axis=-1,
    )
    root_euler = Rotation.from_quat(root_quat_xyzw).as_euler("xyz", degrees=False)
    assert float(np.mean(np.abs(root_euler[:, 0]))) < 1.5

    left_contact = clip.left_foot_contact > 0.5
    right_contact = clip.right_foot_contact > 0.5
    left_transitions = int(np.sum(np.abs(np.diff(left_contact.astype(np.int32))) > 0))
    right_transitions = int(np.sum(np.abs(np.diff(right_contact.astype(np.int32))) > 0))
    alternating_rate = float(np.mean(np.logical_xor(left_contact, right_contact)))
    assert left_transitions >= 2
    assert right_transitions >= 2
    assert alternating_rate > 0.2
