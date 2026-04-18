#!/usr/bin/env python3
"""v0.20.0-A contract freeze verification.

Checks all exit criteria from reference_design.md:
  1. One documented prior schema exists and imports
  2. One viewer and one trace logger can consume that schema
  3. Schema covers all documented fields with rate limits and bounds
  4. Save/load roundtrip preserves data
  5. Active runtime FSM path is marked deprecated
  6. Preview window produces a valid obs vector

Usage:
  uv run python tests/test_v0200a_contract.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASSED = 0
FAILED = 0


def check(name: str, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAILED += 1


# ---- 1. Schema import -----------------------------------------------------

def test_schema_import():
    from control.references import (  # noqa: F401
        ReferenceLibrary,
        ReferenceLibraryMeta,
        ReferencePreviewWindow,
        ReferenceTrajectory,
    )


# ---- 2. Viewer consumes schema --------------------------------------------

def test_viewer_create_dummy():
    """The viewer can create and reload a dummy library."""
    tmpdir = tempfile.mkdtemp(prefix="v0200a_")
    try:
        from control.references.reference_library import (
            ReferenceLibrary,
            ReferenceLibraryMeta,
            ReferenceTrajectory,
        )

        n, nj, dt, ct = 25, 8, 0.02, 0.50
        phase = np.linspace(0, 1, n, endpoint=False).astype(np.float32)
        trajs = []
        for vx in [0.0, 0.10, 0.15]:
            trajs.append(ReferenceTrajectory(
                command_vx=vx, dt=dt, cycle_time=ct,
                q_ref=np.zeros((n, nj), dtype=np.float32),
                phase=phase,
                stance_foot_id=np.zeros(n, dtype=np.float32),
                contact_mask=np.ones((n, 2), dtype=np.float32),
                pelvis_pos=np.tile([0, 0, 0.42], (n, 1)).astype(np.float32),
                pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
                com_pos=np.tile([0, 0, 0.42], (n, 1)).astype(np.float32),
                left_foot_pos=np.tile([0, 0.08, 0], (n, 1)).astype(np.float32),
                left_foot_rpy=np.zeros((n, 3), dtype=np.float32),
                right_foot_pos=np.tile([0, -0.08, 0], (n, 1)).astype(np.float32),
                right_foot_rpy=np.zeros((n, 3), dtype=np.float32),
                generator_version="test_v0200a",
            ))
        lib = ReferenceLibrary(trajs, ReferenceLibraryMeta(generator="test"))
        lib.save(tmpdir)

        lib2 = ReferenceLibrary.load(tmpdir)
        assert len(lib2) == 3, f"Expected 3 entries, got {len(lib2)}"

        summary = lib2.summary()
        assert "3 entries" in summary
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_viewer_validate():
    """Validation returns empty for valid entries, non-empty for bad ones."""
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    # Valid entry
    n, nj = 10, 4
    good = ReferenceTrajectory(
        command_vx=0.1, dt=0.02, cycle_time=0.20,
        q_ref=np.zeros((n, nj), dtype=np.float32),
        phase=np.linspace(0, 1, n, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(n, dtype=np.float32),
        contact_mask=np.ones((n, 2), dtype=np.float32),
    )
    assert good.validate() == [], f"Good entry has issues: {good.validate()}"

    # Bad entry: missing required fields
    bad = ReferenceTrajectory(command_vx=0.1, dt=0.02, cycle_time=0.20)
    issues = bad.validate()
    assert len(issues) > 0, "Bad entry should have validation issues"

    # Library-level validation
    lib = ReferenceLibrary([good])
    assert lib.validate() == {}

    lib.add(bad)
    lib_issues = lib.validate()
    assert len(lib_issues) > 0


# ---- 3. Schema field coverage ---------------------------------------------

def test_trajectory_fields():
    from control.references.reference_library import ReferenceTrajectory

    fields = set(ReferenceTrajectory.__dataclass_fields__)
    required = {
        "command_vx", "command_vy", "command_yaw_rate",
        "dt", "cycle_time", "phase",
        "q_ref", "dq_ref",
        "pelvis_pos", "pelvis_rpy", "com_pos",
        "left_foot_pos", "left_foot_rpy",
        "right_foot_pos", "right_foot_rpy",
        "stance_foot_id", "contact_mask",
        "generator_version", "is_valid", "validation_notes",
    }
    missing = required - fields
    assert not missing, f"Missing trajectory fields: {missing}"


def test_preview_fields():
    from control.references.reference_library import ReferencePreviewWindow

    fields = set(ReferencePreviewWindow.__dataclass_fields__)
    required = {
        "q_ref", "phase_sin", "phase_cos", "stance_foot_id",
        "pelvis_height", "pelvis_roll", "pelvis_pitch",
        "swing_foot_pos", "swing_foot_vel", "next_foothold",
        "contact_mask",
        "future_q_ref", "future_phase_sin", "future_phase_cos",
        "future_contact_mask",
    }
    missing = required - fields
    assert not missing, f"Missing preview fields: {missing}"


def test_meta_fields():
    from control.references.reference_library import ReferenceLibraryMeta

    fields = set(ReferenceLibraryMeta.__dataclass_fields__)
    required = {
        "generator", "generator_version", "robot",
        "dt", "cycle_time", "n_joints",
        "command_range_vx", "command_range_vy", "command_range_yaw",
        "command_interval", "created",
    }
    missing = required - fields
    assert not missing, f"Missing meta fields: {missing}"


# ---- 4. Save/load roundtrip -----------------------------------------------

def test_roundtrip():
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceLibraryMeta,
        ReferenceTrajectory,
    )

    tmpdir = tempfile.mkdtemp(prefix="v0200a_rt_")
    try:
        n, nj = 25, 8
        q_orig = np.random.randn(n, nj).astype(np.float32)
        phase_orig = np.linspace(0, 1, n, endpoint=False).astype(np.float32)
        pelvis_orig = np.random.randn(n, 3).astype(np.float32)

        traj = ReferenceTrajectory(
            command_vx=0.15, command_vy=0.02, command_yaw_rate=-0.1,
            dt=0.02, cycle_time=0.50,
            q_ref=q_orig,
            phase=phase_orig,
            stance_foot_id=np.zeros(n, dtype=np.float32),
            contact_mask=np.ones((n, 2), dtype=np.float32),
            pelvis_pos=pelvis_orig,
            pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
            generator_version="roundtrip_test",
            validation_notes="test note",
        )
        meta = ReferenceLibraryMeta(
            generator="roundtrip", generator_version="1.2.3",
            command_range_vx=(-0.1, 0.3),
        )
        lib = ReferenceLibrary([traj], meta)
        lib.save(tmpdir)

        lib2 = ReferenceLibrary.load(tmpdir)
        assert len(lib2) == 1
        t2 = lib2.lookup(0.15, 0.02, -0.1)

        # Scalars
        assert t2.command_vx == 0.15
        assert t2.command_vy == 0.02
        assert t2.command_yaw_rate == -0.1
        assert t2.dt == 0.02
        assert t2.cycle_time == 0.50
        assert t2.generator_version == "roundtrip_test"
        assert t2.validation_notes == "test note"

        # Arrays
        assert np.allclose(t2.q_ref, q_orig), "q_ref mismatch"
        assert np.allclose(t2.phase, phase_orig), "phase mismatch"
        assert np.allclose(t2.pelvis_pos, pelvis_orig), "pelvis_pos mismatch"

        # Meta
        assert lib2.meta.generator == "roundtrip"
        assert lib2.meta.generator_version == "1.2.3"
        assert lib2.meta.command_range_vx == (-0.1, 0.3)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- 5. Deprecation marker ------------------------------------------------

def test_deprecation():
    import control.references.walking_ref_v2 as v2

    doc = v2.__doc__ or ""
    assert "deprecated" in doc.lower(), (
        "walking_ref_v2 module docstring must contain 'deprecated'"
    )


# ---- 6. Preview obs vector ------------------------------------------------

def test_preview_obs():
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    n, nj = 25, 8
    n_preview = 5
    phase = np.linspace(0, 1, n, endpoint=False).astype(np.float32)
    stance = np.zeros(n, dtype=np.float32)
    stance[n // 2:] = 1.0
    contact = np.ones((n, 2), dtype=np.float32)
    contact[3:9, 1] = 0.0  # right foot swing

    traj = ReferenceTrajectory(
        command_vx=0.15, dt=0.02, cycle_time=0.50,
        q_ref=np.random.randn(n, nj).astype(np.float32),
        phase=phase,
        stance_foot_id=stance,
        contact_mask=contact,
        pelvis_pos=np.tile([0, 0, 0.42], (n, 1)).astype(np.float32),
        pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
        left_foot_pos=np.tile([0, 0.08, 0], (n, 1)).astype(np.float32),
        right_foot_pos=np.tile([0, -0.08, 0], (n, 1)).astype(np.float32),
    )
    lib = ReferenceLibrary([traj])

    pw = lib.get_preview(vx=0.15, step_index=0, n_preview=n_preview)
    obs = pw.to_obs_array()

    # base: n_joints + 16, future: n_preview*(n_joints+4)
    expected_len = (nj + 16) + n_preview * (nj + 4)
    assert obs.shape == (expected_len,), (
        f"Expected obs shape ({expected_len},), got {obs.shape}"
    )
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert np.all(np.isfinite(obs)), "Obs vector contains non-finite values"

    # q_ref should be present in the obs vector (first nj elements)
    assert np.allclose(obs[:nj], pw.q_ref), "q_ref not at start of obs vector"

    # Phase sin/cos should be valid trig values
    assert -1.0 <= pw.phase_sin <= 1.0
    assert -1.0 <= pw.phase_cos <= 1.0

    # Stance foot should be 0 or 1
    assert pw.stance_foot_id in (0, 1)

    # q_ref shape
    assert pw.q_ref.shape == (nj,)
    assert pw.future_q_ref.shape == (n_preview, nj)


def test_preview_warns_on_overrun():
    """get_preview should emit a one-shot RuntimeWarning past n_steps.

    R2 horizon guard: linear-replay contract means callers must keep
    episode_horizon <= n_steps.  If they don't, the preview silently
    freezes — the warning makes the misconfiguration visible.
    """
    import warnings
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    n, nj = 8, 4
    traj = ReferenceTrajectory(
        command_vx=0.10, dt=0.02, cycle_time=0.16,
        q_ref=np.zeros((n, nj), dtype=np.float32),
        phase=np.linspace(0, 1, n, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(n, dtype=np.float32),
        contact_mask=np.ones((n, 2), dtype=np.float32),
        pelvis_pos=np.zeros((n, 3), dtype=np.float32),
        pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
        left_foot_pos=np.zeros((n, 3), dtype=np.float32),
        right_foot_pos=np.zeros((n, 3), dtype=np.float32),
    )
    lib = ReferenceLibrary([traj])

    # In-bounds query: no warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lib.get_preview(vx=0.10, step_index=n - 1, n_preview=2)
        assert not any("get_preview" in str(w.message) for w in caught), (
            "Should not warn when step_index is in-bounds"
        )

    # First overrun: warning fires
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lib.get_preview(vx=0.10, step_index=n + 5, n_preview=2)
        assert any("get_preview" in str(w.message) for w in caught), (
            f"Should warn on overrun, got: {[str(w.message) for w in caught]}"
        )

    # Second overrun on same key: silent (one-shot)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lib.get_preview(vx=0.10, step_index=n + 10, n_preview=2)
        assert not any("get_preview" in str(w.message) for w in caught), (
            "Overrun warning should be one-shot per command_key"
        )


def test_preview_clamps_at_end():
    """Preview at/past end of trajectory clamps to the last frame.

    v0.20.0-C contract: trajectories may span multiple gait cycles and
    are NOT periodic — the first cycle of a from-rest plan does not
    join the last steady-state cycle.  Wrapping with modulo would
    feed start-of-plan states again after end-of-plan and reintroduce
    the discontinuity the multi-cycle generator was designed to avoid.
    """
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    n, nj = 10, 4
    q = np.arange(n * nj, dtype=np.float32).reshape(n, nj)
    traj = ReferenceTrajectory(
        command_vx=0.10, dt=0.02, cycle_time=0.20,
        q_ref=q,
        phase=np.linspace(0, 1, n, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(n, dtype=np.float32),
        contact_mask=np.ones((n, 2), dtype=np.float32),
        pelvis_pos=np.zeros((n, 3), dtype=np.float32),
        pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
        left_foot_pos=np.zeros((n, 3), dtype=np.float32),
        right_foot_pos=np.zeros((n, 3), dtype=np.float32),
    )
    lib = ReferenceLibrary([traj])

    # Preview at the last frame: current and all future slots clamp to q[n-1]
    pw = lib.get_preview(vx=0.10, step_index=n - 1, n_preview=3)
    assert np.allclose(pw.q_ref, q[n - 1]), "Current q_ref wrong at last step"
    for k in range(3):
        assert np.allclose(pw.future_q_ref[k], q[n - 1]), (
            f"Future[{k}] should clamp to last frame, not wrap to start"
        )

    # Query past the end: still clamps
    pw_past = lib.get_preview(vx=0.10, step_index=n + 5, n_preview=2)
    assert np.allclose(pw_past.q_ref, q[n - 1])
    assert np.allclose(pw_past.future_q_ref[0], q[n - 1])

    # Mid-trajectory: future indices advance normally (no wrap activated yet)
    pw_mid = lib.get_preview(vx=0.10, step_index=2, n_preview=3)
    assert np.allclose(pw_mid.q_ref, q[2])
    assert np.allclose(pw_mid.future_q_ref[0], q[3])
    assert np.allclose(pw_mid.future_q_ref[1], q[4])
    assert np.allclose(pw_mid.future_q_ref[2], q[5])


def test_nearest_neighbor_lookup():
    """Lookup finds the closest command bin."""
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    n, nj = 10, 4
    base_kwargs = dict(
        dt=0.02, cycle_time=0.20,
        q_ref=np.zeros((n, nj), dtype=np.float32),
        phase=np.linspace(0, 1, n, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(n, dtype=np.float32),
        contact_mask=np.ones((n, 2), dtype=np.float32),
    )
    lib = ReferenceLibrary([
        ReferenceTrajectory(command_vx=0.0, **base_kwargs),
        ReferenceTrajectory(command_vx=0.10, **base_kwargs),
        ReferenceTrajectory(command_vx=0.20, **base_kwargs),
    ])

    assert lib.lookup(0.0).command_vx == 0.0
    assert lib.lookup(0.08).command_vx == 0.10  # closer to 0.10
    assert lib.lookup(0.18).command_vx == 0.20  # closer to 0.20
    assert lib.lookup(0.14).command_vx == 0.10  # closer to 0.10
    assert lib.lookup(0.16).command_vx == 0.20  # closer to 0.20


def test_foothold_finds_touchdown_transition():
    """next_foothold should find the 0->1 contact transition, not first contact=1."""
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceTrajectory,
    )

    n, nj = 20, 4
    # Contact pattern: right foot has double-support, then swing, then touchdown
    # Steps 0-5: double support (both feet contact=1)
    # Steps 6-14: right foot swing (contact=0)
    # Steps 15-19: right foot touchdown (contact=1 again)
    contact = np.ones((n, 2), dtype=np.float32)
    contact[6:15, 1] = 0.0  # right foot swings

    right_foot = np.zeros((n, 3), dtype=np.float32)
    right_foot[:, 1] = -0.08
    # Right foot moves forward during swing, lands at x=0.05
    right_foot[6:15, 0] = np.linspace(0, 0.05, 9)
    right_foot[15:, 0] = 0.05  # landed position

    traj = ReferenceTrajectory(
        command_vx=0.10, dt=0.02, cycle_time=0.40,
        q_ref=np.zeros((n, nj), dtype=np.float32),
        phase=np.linspace(0, 1, n, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(n, dtype=np.float32),
        contact_mask=contact,
        left_foot_pos=np.tile([0, 0.08, 0], (n, 1)).astype(np.float32),
        right_foot_pos=right_foot,
        pelvis_pos=np.zeros((n, 3), dtype=np.float32),
        pelvis_rpy=np.zeros((n, 3), dtype=np.float32),
    )
    lib = ReferenceLibrary([traj])

    # Query during double-support (step 2): swing_id=1 (right)
    # Should find touchdown at step 15 (0->1 transition), NOT step 2 (already 1)
    pw = lib.get_preview(vx=0.10, step_index=2, n_preview=3)
    assert pw.next_foothold[0] == np.float32(0.05), (
        f"Foothold x should be 0.05 (touchdown position), got {pw.next_foothold[0]}"
    )


def test_validate_malformed_qref():
    """validate() handles 1-D q_ref gracefully instead of crashing."""
    from control.references.reference_library import ReferenceTrajectory

    # 1-D q_ref (wrong shape)
    bad = ReferenceTrajectory(
        command_vx=0.1, dt=0.02, cycle_time=0.20,
        q_ref=np.zeros(10, dtype=np.float32),  # should be 2-D
        phase=np.linspace(0, 1, 10, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(10, dtype=np.float32),
        contact_mask=np.ones((10, 2), dtype=np.float32),
    )
    issues = bad.validate()
    assert len(issues) > 0, "1-D q_ref should produce validation issues"
    assert any("2-D" in issue or "q_ref" in issue for issue in issues), (
        f"Issues should mention q_ref shape problem: {issues}"
    )


def test_validate_non_array():
    """validate() handles non-ndarray fields gracefully."""
    from control.references.reference_library import ReferenceTrajectory

    bad = ReferenceTrajectory(
        command_vx=0.1, dt=0.02, cycle_time=0.20,
        q_ref=np.zeros((10, 4), dtype=np.float32),
        phase=np.linspace(0, 1, 10, endpoint=False).astype(np.float32),
        stance_foot_id=np.zeros(10, dtype=np.float32),
        contact_mask=np.ones((10, 2), dtype=np.float32),
        pelvis_pos=[[0, 0, 0.42]] * 10,  # list, not ndarray
    )
    issues = bad.validate()
    assert any("ndarray" in issue for issue in issues), (
        f"Non-ndarray should produce validation issue: {issues}"
    )


# ---- Runner ----------------------------------------------------------------

def main() -> None:
    print("v0.20.0-A contract freeze verification")
    print("=" * 50)

    print("\n1. Schema import")
    check("import all schema classes", test_schema_import)

    print("\n2. Viewer consumes schema")
    check("create dummy library and reload", test_viewer_create_dummy)
    check("validation logic", test_viewer_validate)

    print("\n3. Schema field coverage")
    check("ReferenceTrajectory fields", test_trajectory_fields)
    check("ReferencePreviewWindow fields", test_preview_fields)
    check("ReferenceLibraryMeta fields", test_meta_fields)

    print("\n4. Save/load roundtrip")
    check("roundtrip preserves all data", test_roundtrip)

    print("\n5. Deprecation")
    check("walking_ref_v2 marked deprecated", test_deprecation)

    print("\n6. Preview obs vector")
    check("obs vector shape, dtype, and q_ref presence", test_preview_obs)
    check("clamp at trajectory end (no wrap)", test_preview_clamps_at_end)
    check("one-shot overrun warning past n_steps", test_preview_warns_on_overrun)
    check("nearest-neighbor command lookup", test_nearest_neighbor_lookup)

    print("\n7. Edge cases from review")
    check("foothold finds 0->1 touchdown transition", test_foothold_finds_touchdown_transition)
    check("validate handles 1-D q_ref gracefully", test_validate_malformed_qref)
    check("validate handles non-ndarray gracefully", test_validate_non_array)

    print("\n" + "=" * 50)
    total = PASSED + FAILED
    print(f"Results: {PASSED}/{total} passed, {FAILED}/{total} failed")
    if FAILED > 0:
        print("v0.20.0-A: NOT VERIFIED")
        sys.exit(1)
    else:
        print("v0.20.0-A: VERIFIED")


if __name__ == "__main__":
    main()
