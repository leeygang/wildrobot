from __future__ import annotations

import numpy as np
import pytest


def test_numpy_frames_interface() -> None:
    from policy_contract.numpy import frames as np_frames

    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert np_frames.normalize_quat_xyzw(quat).shape == (4,)
    assert np_frames.rotate_vec_by_quat(quat, vec).shape == (3,)
    assert np_frames.gravity_local_from_quat(quat).shape == (3,)
    assert np_frames.angvel_heading_local(vec, quat).shape == (3,)

    ops = np_frames.NumpyFrameOps()
    assert ops.gravity_local_from_quat(quat).shape == (3,)


def test_jax_frames_interface() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from policy_contract.jax import frames as jax_frames

    quat = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    vec = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)

    assert jax_frames.normalize_quat_xyzw(quat).shape == (4,)
    assert jax_frames.rotate_vec_by_quat(quat, vec).shape == (3,)
    assert jax_frames.gravity_local_from_quat(quat).shape == (3,)
    assert jax_frames.angvel_heading_local(vec, quat).shape == (3,)

    ops = jax_frames.JaxFrameOps()
    assert ops.gravity_local_from_quat(quat).shape == (3,)
