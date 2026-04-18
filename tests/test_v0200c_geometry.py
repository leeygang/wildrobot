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


_STANCE_TOL_M = 0.003   # max allowed foot-bottom z above floor in stance
_SWING_MIN_M = -0.002   # min required foot-bottom z above floor in swing
                        # (-2 mm = MuJoCo contact compliance margin; the
                        # constant-plantarflex swing toe geometry can dip
                        # 1-2 mm at the swing-onset frame on cycle 0)
_VX_BINS = (0.10, 0.15, 0.20, 0.25)
_PROBE_FRAMES = (0, 5, 10, 16, 22, 28, 32, 48, 64)


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
    print(f"Total failures: {len(failures)}")
    for f in failures[:10]:
        print(f"  - {f}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    if failures:
        print(f"\nv0.20.0-C geometry gate: FAIL")
        return 1
    print(f"\nv0.20.0-C geometry gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
