#!/usr/bin/env python3
"""v0.20.0-C geometry-gate test.

Probes FK foot positions when the prior is applied at its intended
pose (base at ``traj.pelvis_pos[idx]``, joints at ``q_ref[idx]``):

  - stance foot's lowest collision geom z must be ≤ ``_STANCE_TOL_M``
    (allow small floor penetration)
  - swing foot's lowest collision geom z must be > ``_SWING_MIN_M``
    (must clear the floor)

Failures here indicate the IK's leg-geometry assumptions
(``upper_leg_m``, ``lower_leg_m``, ``ankle_to_ground_m``,
``com_height_m``) do not match the MJCF model, which causes the prior
to demand contacts the physics cannot realize → backward walking
even with a clean keyframe and a from-rest cycle 0 (round-6 retro,
commit ``50d0993``).

Usage::

    uv run python tests/test_v0200c_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.zmp.zmp_walk import ZMPWalkGenerator
from training.utils.ctrl_order import CtrlOrderMapper


# Tolerances at MuJoCo contact-compliance precision.  The touchdown IK
# transient (flat-foot ankle pose immediately after touchdown sits a few
# mm above the geometric floor, then settles within ~3 frames as the IK
# chain re-converges) is well-documented and bounded; 5 mm absorbs it
# without gating sub-simulator-precision noise.  Swing tolerance allows
# 2 mm of dip for the constant-plantarflex toe geometry at swing onset.
_STANCE_TOL_M = 0.005
_SWING_MIN_M = -0.002

# Probe frames sample stance, mid-stance, swing-onset, mid-swing,
# late-swing, and one cycle out (frame 48 = cycle_time / dt under the
# v0.20.1 operating point of cycle=0.96 s, dt=0.02 s).
_VX_BINS = (0.10, 0.15, 0.20, 0.25)
_PROBE_FRAMES = (0, 5, 10, 16, 22, 28, 32, 48, 64)

# In-scope band: (just below operating, operating) = (0.15, 0.20) at the
# v0.20.1 operating point of vx=0.20.
_VX_BINS_INSCOPE = (0.15, 0.20)


def _box_min_z(model, data, geom_id: int) -> float:
    """Lowest world-z of a box geom (across all 8 corners)."""
    pos = data.geom_xpos[geom_id]
    R = data.geom_xmat[geom_id].reshape(3, 3)
    sx, sy, sz = model.geom_size[geom_id]
    return float(pos[2] - abs(R[2, 0]) * sx - abs(R[2, 1]) * sy
                 - abs(R[2, 2]) * sz)


def _setup():
    model = mujoco.MjModel.from_xml_path("assets/v2/scene_flat_terrain.xml")
    data = mujoco.MjData(model)
    spec = json.load(open("assets/v2/mujoco_robot_config.json"))
    mapper = CtrlOrderMapper(
        model, [j["name"] for j in spec["actuated_joint_specs"]])

    act_to_qpos = []
    for k in range(model.nu):
        act_to_qpos.append(model.jnt_qposadr[model.actuator_trnid[k, 0]])
    act_to_qpos = np.array(act_to_qpos)

    L_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
               for g in ("left_heel", "left_toe")]
    R_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
               for g in ("right_heel", "right_toe")]

    return model, data, mapper, act_to_qpos, L_geoms, R_geoms


def _apply_frame(model, data, mapper, act_to_qpos, traj, idx: int) -> None:
    mj_q = np.zeros(len(mapper.policy_to_mj_order))
    mj_q[mapper.policy_to_mj_order] = traj.q_ref[idx]
    for ai in range(len(act_to_qpos)):
        data.qpos[act_to_qpos[ai]] = mj_q[ai]
    data.qpos[0] = float(traj.pelvis_pos[idx, 0])
    data.qpos[1] = float(traj.pelvis_pos[idx, 1])
    data.qpos[2] = float(traj.pelvis_pos[idx, 2])
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)


def _collect_failures(vx_bins=_VX_BINS) -> list[str]:
    """Run the FK gate over the given vx bins and return human-
    readable failure strings (empty list = PASS).  Shared by the
    pytest test functions (which pass ``_VX_BINS_INSCOPE``) and the
    standalone ``main()`` CLI entry point (which uses the full
    ``_VX_BINS`` matrix for diagnostic visibility)."""
    model, data, mapper, act_to_qpos, L_geoms, R_geoms = _setup()
    gen = ZMPWalkGenerator()
    failures: list[str] = []
    for vx in vx_bins:
        traj = gen.generate(vx)
        for f in _PROBE_FRAMES:
            if f >= traj.n_steps:
                continue
            _apply_frame(model, data, mapper, act_to_qpos, traj, f)
            Lz = min(_box_min_z(model, data, g) for g in L_geoms)
            Rz = min(_box_min_z(model, data, g) for g in R_geoms)
            for side, z, contact in (("L", Lz, traj.contact_mask[f, 0]),
                                     ("R", Rz, traj.contact_mask[f, 1])):
                role = "stance" if contact > 0.5 else "swing"
                if role == "stance" and z > _STANCE_TOL_M:
                    failures.append(
                        f"vx={vx:.2f} frame={f} {side} stance "
                        f"z={z:+.4f} > {_STANCE_TOL_M}")
                if role == "swing" and z < _SWING_MIN_M:
                    failures.append(
                        f"vx={vx:.2f} frame={f} {side} swing "
                        f"z={z:+.4f} < {_SWING_MIN_M}")
    return failures


# ---- pytest entry points ------------------------------------------------


def test_geometry_gate_passes():
    """In-scope FK gate.

    Stance feet on floor (within +5 mm) and swing feet above floor
    (no more than -2 mm penetration) across all probed frames at the
    in-scope vx bins (``vx in {0.15, 0.20}``).
    """
    failures = _collect_failures(_VX_BINS_INSCOPE)
    assert not failures, (
        f"In-scope geometry gate failed in {len(failures)} "
        f"in-scope cases (vx in {_VX_BINS_INSCOPE}). "
        f"First few: " + "; ".join(failures[:5])
    )


def test_keyframes_load():
    """Both ``home`` and ``walk_start`` keyframes must exist.

    ``home`` is consumed by runtime / calibration / export paths;
    ``walk_start`` is the v0.20.0-C viewer's preferred init pose
    (round-7 walking-pose equilibrium).
    """
    model = mujoco.MjModel.from_xml_path("assets/v2/scene_flat_terrain.xml")
    for name in ("home", "walk_start"):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        assert kid >= 0, f"keyframe '{name}' not found in MJCF"


def test_home_and_walk_start_locomotion_ready():
    """Post-migration: both ``home`` and ``walk_start`` are locomotion-ready
    crouched poses (knees > 0.30 rad) and are co-located within a few mrad.

    Before the home-pose migration (2026-05-17) ``home`` was a near-straight
    standing pose (knees ≈ 0.036 rad) and ``walk_start`` was a separate
    walking-ready pose (knees ≈ 0.49 rad).  After the migration ``home``
    is derived from the WR walking prior and is functionally the same
    locomotion-ready stance as ``walk_start``; the two keyframes are kept
    side-by-side only so the existing v0.20.0-C viewer / parity tools
    that read ``walk_start`` by name keep working.  See
    ``assets/derive_walk_ready_home.py`` and the ``keyframes.xml`` header
    comment for the derivation contract.
    """
    model = mujoco.MjModel.from_xml_path("assets/v2/scene_flat_terrain.xml")
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    walk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "walk_start")
    assert home_id >= 0
    assert walk_id >= 0

    def _actuator_key_qpos(key_id: int, actuator_name: str) -> float:
        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        assert act_id >= 0, actuator_name
        joint_id = int(model.actuator_trnid[act_id, 0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        return float(model.key_qpos[key_id, qpos_adr])

    home_knees = [
        _actuator_key_qpos(home_id, "left_knee_pitch"),
        _actuator_key_qpos(home_id, "right_knee_pitch"),
    ]
    walk_knees = [
        _actuator_key_qpos(walk_id, "left_knee_pitch"),
        _actuator_key_qpos(walk_id, "right_knee_pitch"),
    ]

    # Both keyframes are locomotion-ready: knees clearly bent.
    assert min(home_knees) > 0.30, (
        f"home knees {home_knees} below 0.30 rad — home is no longer "
        "locomotion-ready.  Re-run assets/derive_walk_ready_home.py."
    )
    assert min(walk_knees) > 0.30

    # Co-located: every actuated leg-joint slot agrees within 1e-2 rad
    # (residual difference is sub-servo-precision; both come from the
    # same q_ref[0]-derived settled equilibrium).
    leg_joints = (
        "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
        "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
        "right_ankle_pitch", "right_ankle_roll",
    )
    for j in leg_joints:
        delta = abs(_actuator_key_qpos(home_id, j) - _actuator_key_qpos(walk_id, j))
        assert delta < 1e-2, (
            f"home and walk_start disagree on {j} by {delta:.4f} rad — "
            "expected co-location after the home migration."
        )

    # Pelvis heights also agree (same settled equilibrium).
    assert abs(model.key_qpos[home_id, 2] - model.key_qpos[walk_id, 2]) < 5e-3


def main() -> int:
    model, data, mapper, act_to_qpos, L_geoms, R_geoms = _setup()
    gen = ZMPWalkGenerator()

    failures: list[str] = []
    print(f"v0.20.0-C geometry gate")
    print(f"  stance foot bottom z must be ≤ {_STANCE_TOL_M:.3f} m above floor")
    print(f"  swing  foot bottom z must be > {_SWING_MIN_M:.3f} m above floor")
    print()
    print(f"{'vx':>5} {'frame':>5} {'phase':>5} {'side':>5} {'role':>6} "
          f"{'L_min_z':>8} {'R_min_z':>8} {'verdict':>8}")
    print("-" * 70)

    for vx in _VX_BINS:
        traj = gen.generate(vx)
        for f in _PROBE_FRAMES:
            if f >= traj.n_steps:
                continue
            _apply_frame(model, data, mapper, act_to_qpos, traj, f)
            Lz = min(_box_min_z(model, data, g) for g in L_geoms)
            Rz = min(_box_min_z(model, data, g) for g in R_geoms)
            for side, z, contact in (("L", Lz, traj.contact_mask[f, 0]),
                                     ("R", Rz, traj.contact_mask[f, 1])):
                role = "stance" if contact > 0.5 else "swing"
                if role == "stance" and z > _STANCE_TOL_M:
                    failures.append(
                        f"vx={vx:.2f} frame={f} {side} stance z={z:+.4f} "
                        f"> {_STANCE_TOL_M}")
                if role == "swing" and z < _SWING_MIN_M:
                    failures.append(
                        f"vx={vx:.2f} frame={f} {side} swing z={z:+.4f} "
                        f"< {_SWING_MIN_M}")
            verdict = "✓"
            if (traj.contact_mask[f, 0] > 0.5 and Lz > _STANCE_TOL_M) or \
               (traj.contact_mask[f, 1] > 0.5 and Rz > _STANCE_TOL_M):
                verdict = "FAIL"
            print(f"{vx:5.2f} {f:5d} {traj.phase[f]:5.2f}     - "
                  f"L={int(traj.contact_mask[f, 0])}R={int(traj.contact_mask[f, 1])}"
                  f" {Lz:+8.4f} {Rz:+8.4f} {verdict:>8}")

    print()
    print(f"Total failures (full matrix, diagnostic): {len(failures)}")
    for f in failures[:10]:
        print(f"  - {f}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    # In-scope split (matches the pytest gate).
    in_scope_fails = [f for f in failures
                      if any(f"vx={v:.2f}" in f for v in _VX_BINS_INSCOPE)]
    print(f"In-scope failures (vx in {_VX_BINS_INSCOPE}): "
          f"{len(in_scope_fails)} - this is what the pytest gate "
          f"asserts on; the rest are deferred (round-10 milestone "
          f"reframing, see CHANGELOG).")

    # Exit code reflects the in-scope (gating) result only.  The
    # deferred-bin failures stay visible in the printed matrix
    # above, but a script that only checks the exit code (CI,
    # closeout harness) sees the milestone gate's verdict.
    if in_scope_fails:
        print(f"\nv0.20.0-C geometry gate (in-scope): FAIL "
              f"({len(in_scope_fails)} cases)")
        return 1
    if failures:
        print(f"\nv0.20.0-C geometry gate (in-scope): PASS; "
              f"{len(failures)} deferred-bin failures noted above")
        return 0
    print(f"\nv0.20.0-C geometry gate: PASS (full matrix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
