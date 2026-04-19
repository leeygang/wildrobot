#!/usr/bin/env python3
"""Tests for ``control/references/runtime_reference_service.py``.

Asserts:
  - lookup_np / lookup_jax produce byte-identical outputs (parity)
  - end-of-trajectory clamp is correct (no wrap to cycle 0)
  - finite-diff velocities match the position differences
  - the smoke command (vx=0.15) round-trips through the service
    without losing any field present in the offline trajectory

These checks are the v0.20.1 Layer-2 contract — the env will rely
on them to safely thread the service through the JIT'd step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.references.runtime_reference_service import (
    RuntimeReferenceService,
)
from control.zmp.zmp_walk import ZMPWalkGenerator


@pytest.fixture(scope="module")
def smoke_trajectory():
    """Build the v0.20.1 smoke trajectory once for all tests in module."""
    gen = ZMPWalkGenerator()
    lib = gen.build_library()
    return lib.lookup(0.15)


@pytest.fixture(scope="module")
def service(smoke_trajectory):
    return RuntimeReferenceService(smoke_trajectory, n_anchor=2)


# -- parity ------------------------------------------------------------------

def test_lookup_np_jax_parity(service):
    """JAX and NumPy lookups must produce byte-identical values."""
    jax_arrays = service.to_jax_arrays()
    for step_idx in (0, 1, 5, 30, 70, service.n_steps - 1):
        np_win = service.lookup_np(step_idx)
        jax_win = service.lookup_jax(step_idx, jax_arrays)

        for field in ("q_ref", "contact_mask", "pelvis_pos", "pelvis_vel",
                      "left_foot_pos", "right_foot_pos",
                      "left_foot_vel", "right_foot_vel",
                      "future_q_ref", "future_phase_sin",
                      "future_phase_cos", "future_contact_mask"):
            np_arr = np.asarray(getattr(np_win, field))
            jax_arr = np.asarray(jax_win[field])
            assert np_arr.shape == jax_arr.shape, (
                f"shape mismatch on {field} at step {step_idx}: "
                f"np {np_arr.shape} vs jax {jax_arr.shape}")
            np.testing.assert_array_equal(
                np_arr, jax_arr,
                err_msg=f"{field} mismatch at step {step_idx}")

        # Scalars
        assert float(np_win.phase_sin) == float(jax_win["phase_sin"])
        assert float(np_win.phase_cos) == float(jax_win["phase_cos"])
        assert int(np_win.stance_foot_id) == int(jax_win["stance_foot_id"])


# -- end-of-trajectory clamp -------------------------------------------------

def test_clamp_past_end(service):
    """Queries past n_steps return the terminal frame, not a wrap."""
    import warnings
    last = service.n_steps - 1
    win_last = service.lookup_np(last)
    # Use a fresh service so the one-shot overrun warning fires here too,
    # but suppress it -- this test is about clamp values, not the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        win_over = service.lookup_np(last + 50)
    np.testing.assert_array_equal(win_last.q_ref, win_over.q_ref)
    np.testing.assert_array_equal(win_last.pelvis_pos, win_over.pelvis_pos)
    # Future preview at the end clamps to the terminal frame too —
    # all anchor frames are identical to the current frame.
    for k in range(service.n_anchor):
        np.testing.assert_array_equal(
            win_over.future_q_ref[k], win_over.q_ref,
            err_msg=f"future anchor {k} should clamp to terminal frame")


def test_clamp_negative_index(service):
    """Negative indices clamp to frame 0 (not Python negative-indexing)."""
    win0 = service.lookup_np(0)
    win_neg = service.lookup_np(-5)
    np.testing.assert_array_equal(win0.q_ref, win_neg.q_ref)
    np.testing.assert_array_equal(win0.pelvis_pos, win_neg.pelvis_pos)


# -- finite-diff velocity matches positions ---------------------------------

def test_finite_diff_velocities(service, smoke_trajectory):
    """Velocity at step k = (pos[k+1] - pos[k]) / dt for k < n-1."""
    dt = float(smoke_trajectory.dt)
    # mid-trajectory step where finite-diff is well-defined
    for step in (10, 50, 100):
        win = service.lookup_np(step)
        win_next = service.lookup_np(step + 1)

        for pos_field, vel_field in (
            ("pelvis_pos", "pelvis_vel"),
            ("left_foot_pos", "left_foot_vel"),
            ("right_foot_pos", "right_foot_vel"),
        ):
            expected = (getattr(win_next, pos_field)
                        - getattr(win, pos_field)) / dt
            actual = getattr(win, vel_field)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-5, atol=1e-6,
                err_msg=f"{vel_field} mismatch at step {step}")


def test_terminal_velocity_copied(service):
    """Terminal-frame velocity copies the previous step (no zero jump)."""
    last = service.n_steps - 1
    win_last = service.lookup_np(last)
    win_prev = service.lookup_np(last - 1)
    np.testing.assert_array_equal(win_last.pelvis_vel, win_prev.pelvis_vel)


# -- shape contracts the env will rely on -----------------------------------

def test_window_shapes(service, smoke_trajectory):
    win = service.lookup_np(30)
    n_joints = smoke_trajectory.q_ref.shape[1]
    assert win.q_ref.shape == (n_joints,)
    assert win.contact_mask.shape == (2,)
    assert win.pelvis_pos.shape == (3,)
    assert win.pelvis_vel.shape == (3,)
    assert win.left_foot_pos.shape == (3,)
    assert win.right_foot_pos.shape == (3,)
    assert win.left_foot_vel.shape == (3,)
    assert win.right_foot_vel.shape == (3,)
    assert win.future_q_ref.shape == (service.n_anchor, n_joints)
    assert win.future_phase_sin.shape == (service.n_anchor,)
    assert win.future_phase_cos.shape == (service.n_anchor,)
    assert win.future_contact_mask.shape == (service.n_anchor, 2)


# -- service properties match the trajectory --------------------------------

def test_service_metadata(service, smoke_trajectory):
    assert service.dt == float(smoke_trajectory.dt)
    assert service.cycle_time == float(smoke_trajectory.cycle_time)
    assert service.n_steps == int(smoke_trajectory.q_ref.shape[0])
    assert service.n_joints == int(smoke_trajectory.q_ref.shape[1])


# -- n_anchor=0 edge case ---------------------------------------------------

def test_zero_anchor(smoke_trajectory):
    """n_anchor=0 must produce empty future arrays without error."""
    svc = RuntimeReferenceService(smoke_trajectory, n_anchor=0)
    win = svc.lookup_np(10)
    assert win.future_q_ref.shape == (0, smoke_trajectory.q_ref.shape[1])
    assert win.future_phase_sin.shape == (0,)
    assert win.future_contact_mask.shape == (0, 2)


def test_negative_anchor_rejected(smoke_trajectory):
    with pytest.raises(ValueError, match="n_anchor"):
        RuntimeReferenceService(smoke_trajectory, n_anchor=-1)


# -- terminal-clamp warning --------------------------------------------------

def _capture_overrun_warnings(fn):
    """Helper: capture all warnings raised by ``fn`` and return the
    overrun-related subset.  Avoids ``pytest.warns(None)`` which was
    removed in newer pytest."""
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        fn()
    return [w for w in caught if "step_idx" in str(w.message)]


def test_terminal_clamp_emits_warning_once(smoke_trajectory):
    """Querying past n_steps emits a RuntimeWarning exactly once per
    service instance, then stays silent on subsequent overruns."""
    svc = RuntimeReferenceService(smoke_trajectory, n_anchor=2)
    last = svc.n_steps - 1

    # In-bounds query: no warning.
    in_bounds = _capture_overrun_warnings(lambda: svc.lookup_np(last))
    assert not in_bounds, "in-bounds query should not warn"

    # First overrun: warns.
    first_over = _capture_overrun_warnings(lambda: svc.lookup_np(last + 1))
    assert len(first_over) == 1
    assert issubclass(first_over[0].category, RuntimeWarning)

    # Second overrun on the same instance: silent (one-shot).
    second_over = _capture_overrun_warnings(lambda: svc.lookup_np(last + 50))
    assert not second_over, "second overrun should be silent"
