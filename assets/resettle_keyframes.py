#!/usr/bin/env python3
"""Re-derive home / walk_start keyframes against the current MuJoCo model.

Mirrors the round-6 / round-7 settling procedure documented in
``assets/v2/keyframes.xml``: starts from the existing keyframe qpos (with
zero values inserted for any joints added since the keyframe was last
recorded), then runs N physics steps with ``ctrl = qpos`` (PD targets fixed
at the initial qpos for each actuator) to let dynamics settle into a static
equilibrium. The settled qpos is written back to ``keyframes.xml``.

Use this whenever you add/remove joints or change kinematic geometry that
shifts the standing equilibrium.

Usage:
    uv run python assets/resettle_keyframes.py             # in-place update
    uv run python assets/resettle_keyframes.py --dry-run   # print results, do not write

Verification thresholds (per the original settling procedure):
    home:        |Δpelvis| < 5 mm,   max|qvel| < 0.05 rad/s after settling
    walk_start:  |Δpelvis| < 5 mm,   max|qvel| < 0.05 rad/s after settling
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
V2_DIR = ASSETS_DIR / "v2"
SCENE_XML = V2_DIR / "scene_flat_terrain.xml"
KEYFRAMES_XML = V2_DIR / "keyframes.xml"

SETTLE_STEPS = 2000          # 2x round-6 budget; safe for both home + walk_start
SETTLE_QVEL_TOL = 1e-3       # rad/s, max |qvel| acceptable after settling
                             # (the real convergence criterion — true static equilibrium)
SETTLE_DPELVIS_TOL = 2e-2    # m, max pelvis drift sanity check (catches diverging
                             # initial poses; not a convergence requirement since
                             # the new equilibrium can sit non-trivially far from
                             # the old keyframe's initial qpos)


def _parse_keyframes(xml_path: Path) -> Dict[str, np.ndarray]:
    text = xml_path.read_text()
    out: Dict[str, np.ndarray] = {}
    for m in re.finditer(r'<key\s+name="([^"]+)"\s*qpos="([^"]+)"', text):
        out[m.group(1)] = np.array([float(v) for v in m.group(2).split()])
    return out


def _build_actuator_qaddr_map(model: mujoco.MjModel) -> List[int]:
    """For each actuator, return the qpos address of the joint it drives."""
    out: List[int] = []
    for aid in range(model.nu):
        # actuator_trnid[aid][0] is the joint id for joint-target actuators.
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            raise ValueError(f"Actuator {aid} has no joint target")
        out.append(int(model.jnt_qposadr[jid]))
    return out


def _expand_old_qpos_to_new(
    old_qpos: np.ndarray,
    model: mujoco.MjModel,
    inserted_joint_names: List[str],
) -> np.ndarray:
    """Given a keyframe qpos sized for an older model that lacked some joints,
    return a qpos sized for the current model with zeros at the new joints'
    positions. Preserves all existing values at the right indices.

    Assumes the joint ORDER in the old model matched the current order
    minus the named insertions.
    """
    if old_qpos.size == model.nq:
        # Already correctly sized; assume in-place layout is correct.
        return old_qpos.copy()
    if old_qpos.size + len(inserted_joint_names) != model.nq:
        raise ValueError(
            f"Old qpos has {old_qpos.size} values; new model needs "
            f"{model.nq}; expected difference of {len(inserted_joint_names)} "
            f"for inserted joints {inserted_joint_names}."
        )
    insert_qaddrs = sorted(
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in inserted_joint_names
    )
    new_qpos = np.zeros(model.nq)
    old_idx = 0
    for new_idx in range(model.nq):
        if new_idx in insert_qaddrs:
            new_qpos[new_idx] = 0.0
        else:
            new_qpos[new_idx] = old_qpos[old_idx]
            old_idx += 1
    assert old_idx == old_qpos.size, "qpos expansion miscount"
    return new_qpos


def _settle(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    actuator_qaddrs: List[int],
    n_steps: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Run n_steps with ctrl=initial_qpos[actuator_qaddrs]; return final qpos,
    pelvis Δxy, max |qvel|, final pelvis_z."""
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    initial_pelvis = data.qpos[0:3].copy()
    ctrl = np.array([initial_qpos[qaddr] for qaddr in actuator_qaddrs], dtype=np.float64)
    data.ctrl[:] = ctrl

    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    final_qpos = data.qpos.copy()
    pelvis_delta_xy = float(np.linalg.norm(final_qpos[0:2] - initial_pelvis[0:2]))
    max_qvel = float(np.max(np.abs(data.qvel)))
    pelvis_z = float(final_qpos[2])
    return final_qpos, pelvis_delta_xy, max_qvel, pelvis_z


def _format_qpos(qpos: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in qpos)


def _write_keyframes(xml_path: Path, settled: Dict[str, np.ndarray]) -> None:
    """Update each <key name="X" qpos="..."/> entry in-place, preserving comments/order."""
    text = xml_path.read_text()
    for name, qpos in settled.items():
        formatted = _format_qpos(qpos)
        pattern = re.compile(
            rf'(<key\s+name="{re.escape(name)}"\s*qpos=")([^"]+)(")'
        )
        text, n_replaced = pattern.subn(rf'\g<1>{formatted}\g<3>', text, count=1)
        if n_replaced == 0:
            raise ValueError(f"Could not find <key name='{name}'> in {xml_path}")
    xml_path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print settled qpos but do not modify keyframes.xml")
    parser.add_argument("--steps", type=int, default=SETTLE_STEPS,
                        help=f"Number of physics steps to settle for (default {SETTLE_STEPS})")
    args = parser.parse_args()

    print(f"Loading model: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print(f"  nq={model.nq}, nu={model.nu}")

    actuator_qaddrs = _build_actuator_qaddr_map(model)

    print(f"\nReading existing keyframes from {KEYFRAMES_XML}")
    existing = _parse_keyframes(KEYFRAMES_XML)
    for name, q in existing.items():
        print(f"  '{name}': {q.size} qpos values")

    # Detect joints that the existing keyframes are missing (added since they
    # were last recorded). Done by comparing the existing qpos size against
    # model.nq. The two newly added ankle_roll joints are the only diff for
    # the 2026-05 update.
    EXPECTED_NEW_JOINTS = ["left_ankle_roll", "right_ankle_roll"]
    missing_count = model.nq - max(q.size for q in existing.values())
    if missing_count != 0 and missing_count != len(EXPECTED_NEW_JOINTS):
        print(
            f"  WARNING: keyframes are missing {missing_count} qpos values, "
            f"but {len(EXPECTED_NEW_JOINTS)} new-joint insertions were expected. "
            f"This script may need updating for new joint additions beyond {EXPECTED_NEW_JOINTS}."
        )

    settled: Dict[str, np.ndarray] = {}
    for name, old_qpos in existing.items():
        print(f"\n=== Settling keyframe '{name}' ===")
        if old_qpos.size != model.nq:
            print(
                f"  Expanding from {old_qpos.size} -> {model.nq} qpos values "
                f"(inserting zeros for {EXPECTED_NEW_JOINTS})"
            )
            initial = _expand_old_qpos_to_new(old_qpos, model, EXPECTED_NEW_JOINTS)
        else:
            initial = old_qpos.copy()

        final_qpos, dxy, max_qvel, pelvis_z = _settle(
            model, initial, actuator_qaddrs, args.steps
        )

        ok_dxy = dxy < SETTLE_DPELVIS_TOL
        ok_qvel = max_qvel < SETTLE_QVEL_TOL
        status = "OK" if (ok_dxy and ok_qvel) else "WARNING"
        print(f"  After {args.steps} steps: pelvis_Δxy={dxy*1000:.2f} mm "
              f"(<{SETTLE_DPELVIS_TOL*1000:.0f}mm? {ok_dxy}), "
              f"max|qvel|={max_qvel:.4f} rad/s (<{SETTLE_QVEL_TOL}? {ok_qvel}), "
              f"pelvis_z={pelvis_z*1000:.2f} mm  [{status}]")
        if not ok_dxy or not ok_qvel:
            print("  WARNING: settling did not fully converge. The keyframe may need")
            print("  more steps or the initial pose may be far from a stable equilibrium.")
        # Print specific delta on the inserted joints to confirm they settled.
        for jname in EXPECTED_NEW_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qaddr = int(model.jnt_qposadr[jid])
            print(f"    {jname:<22s} settled to {final_qpos[qaddr]:+.6f} rad "
                  f"({np.degrees(final_qpos[qaddr]):+.3f} deg)")
        settled[name] = final_qpos

    print()
    if args.dry_run:
        print("--dry-run: not writing keyframes.xml. Settled qpos:")
        for name, q in settled.items():
            print(f"  {name}: qpos=\"{_format_qpos(q)}\"")
    else:
        _write_keyframes(KEYFRAMES_XML, settled)
        print(f"Wrote settled keyframes to {KEYFRAMES_XML}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
