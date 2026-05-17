#!/usr/bin/env python3
"""Derive a locomotion-ready, phase-neutral ``home`` keyframe from the WR walking prior.

Background
----------
ToddlerBot uses a single ``home`` keyframe as the fixed nominal action
anchor for walking (``default_action`` in ``toddlerbot/locomotion/mjx_env.py``
is the first walking-reference frame, which equals the ``home`` keyframe
qpos because TB's home keyframe IS the locomotion-ready crouched pose).

WildRobot historically used ``home`` for a near-straight standing pose
(knees ≈ 0.036 rad) and ``ref_init_q_rad`` for the offline walking
reference's frame-0 joint pose (knees ≈ 0.464 rad).  smoke9c works
around the gap by anchoring residuals/reset to ``ref_init``, leaving
``home`` semantically confused.

This script aligns WR with TB by deriving a phase-neutral, mirror-symmetric
joint pose from the active WR walking prior and installing it as the
canonical ``home`` keyframe.  The result is byte-equivalent in role to TB's
``home``: a constant locomotion-ready stance pose that PPO can residual on
top of.

Derivation
----------
1.  Build the offline walking reference library at the smoke9c operating
    point (``vx = 0.1333333333 m/s``).  The library is produced by
    ``control.zmp.zmp_walk.ZMPWalkGenerator``; a single ctrl cycle is
    48 frames (cycle_time = 0.96 s, dt = 0.02 s), 1104 frames total.

2.  Select all double-support frames whose physical joint configuration
    is mirror-symmetric within ``DSP_SYMMETRY_TOL_RAD``
    (``|left_hip_pitch + right_hip_pitch| < tol`` etc.).

    **Important honesty note**: at the current ZMPWalkGenerator output
    the only mirror-symmetric DSP window in the entire 1104-frame
    trajectory is the FROM-REST initial DSP at frames [0..7].  Every
    later DSP is either the mid-cycle COM-shift (frames 24..31, 72..79,
    ...) or the cross-cycle landing-pad transition (frames 48..55,
    96..103, ...); both are structurally asymmetric on hip_pitch /
    ankle_pitch by ≈0.176 rad (the gait is in steady-state forward
    locomotion, so the legs are mid-stride during DSP).  In practice
    this script averages the 8 from-rest initialization frames, NOT
    a cycle-invariant set of windows.  This is acceptable for the
    migration goal — frames [0..7] are the planner's intended initial
    standing pose — but it means the derived ``home`` is tied to the
    trajectory's phase origin.  If the prior's phase origin changes
    (e.g. the planner no longer starts from a static DSP), the
    derivation would need to be reworked.

3.  Average ``q_ref`` across the selected frames, then enforce exact
    L/R sign-mirror symmetry for paired leg joints (drops sub-mrad
    numerical drift from the IK solver).

4.  Compose a full keyframe ``qpos``:
        - freejoint slots seeded from the existing ``walk_start`` qpos
          (which is already a settled locomotion-ready equilibrium and
          therefore a much better starting point than zero or the old
          standing ``home``),
        - actuated leg slots overwritten with the derived joint pose,
        - arms / waist / ankle_roll left at zero (the walking prior
          plans only the sagittal leg chain; non-leg slots are zero in
          ``q_ref``).

5.  Run the same physics settling procedure used by
    ``assets/resettle_keyframes.py`` so the recorded ``key_qpos`` is a
    true static equilibrium.

6.  Verify the derived pose before promotion:
        - HARD gate: all values finite, within MJCF joint ranges, with
          exact L/R mirror symmetry on the paired leg joints, and the
          settle loop converges to static equilibrium (|pelvis Δxy| <
          2 cm, max|qvel| < 1e-3 rad/s after 2000 steps).
        - INFORMATIONAL: per-joint envelope report showing
          ``min_t q_ref(t)`` and ``max_t q_ref(t)`` against
          ``new_home ± residual_scale_per_joint``.  We do NOT gate on
          this — under TB/smoke9c semantics the residual is a small
          correction (typ. 0.25 rad/joint) and is not designed to
          *replace* q_ref(t); the gait emerges from reward shaping +
          residual.  The report is the basis for any future decision
          to grow the residual authority.

7.  Optionally write the settled qpos into the ``home`` <key> entry
    of ``assets/v2/keyframes.xml`` (``--write``; default is dry-run).

Usage
-----
    uv run python assets/derive_walk_ready_home.py            # dry-run (default)
    uv run python assets/derive_walk_ready_home.py --write    # update keyframes.xml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from control.zmp.zmp_walk import ZMPWalkGenerator  # noqa: E402


V2_DIR = REPO_ROOT / "assets" / "v2"
SCENE_XML = V2_DIR / "scene_flat_terrain.xml"
KEYFRAMES_XML = V2_DIR / "keyframes.xml"

# Operating point that v0.20.1 smoke9c trains and evaluates against
# (training/configs/ppo_walking_v0201_smoke9c.yaml :: loc_ref_offline_command_vx).
DERIVATION_VX = 0.1333333333

# Symmetry tolerance used to select phase-neutral DSP frames.  Loose
# enough to accept the early-DSP plateau (drifts only ~1e-3 across the
# 8-frame window) but tight enough to reject the mid-cycle DSP (which
# is ~0.18 rad asymmetric at the peak of the COM shift).
DSP_SYMMETRY_TOL_RAD = 5e-3

# Per-joint residual authority — must match smoke9c's
# loc_ref_residual_scale_per_joint for the leg joints.
RESIDUAL_SCALE_PER_LEG_JOINT_RAD = 0.25

# Leg joints whose q_ref(t) envelope must be covered by new_home ± residual.
LEG_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
    "right_ankle_pitch", "right_ankle_roll",
)

# Mirror-pair joints, used for exact L/R symmetry enforcement.
MIRROR_PAIRS_SAME_SIGN = (
    ("left_knee_pitch", "right_knee_pitch"),
    ("left_ankle_pitch", "right_ankle_pitch"),
)
MIRROR_PAIRS_OPP_SIGN = (
    ("left_hip_pitch", "right_hip_pitch"),
    ("left_hip_roll", "right_hip_roll"),
    ("left_ankle_roll", "right_ankle_roll"),
)

# Settling thresholds (re-use the budget proven for resettle_keyframes.py).
SETTLE_STEPS = 2000
SETTLE_QVEL_TOL = 1e-3
SETTLE_DPELVIS_TOL = 2e-2


# ----------------------------------------------------------------------------
# Walking-prior frame selection + averaging
# ----------------------------------------------------------------------------


def _load_walking_q_ref(vx: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(q_ref [n_steps, n_joints], contact_mask [n_steps, 2], joint_order)``.

    ``joint_order`` is the PolicySpec actuator order used by ZMPWalkGenerator
    to populate ``q_ref`` columns; it matches the MJCF actuator order one
    line below.  Returned as a Python list of names so callers can index by
    name without re-reading the layout JSON.
    """
    generator = ZMPWalkGenerator()
    library = generator.build_library_for_vx_values([vx])
    traj = library.lookup(vx)
    layout = generator._load_actuator_layout()
    joint_order = [str(name) for name in layout["actuator_names"]]
    if traj.q_ref.shape[1] != len(joint_order):
        raise RuntimeError(
            f"q_ref columns ({traj.q_ref.shape[1]}) != actuator count "
            f"({len(joint_order)}); layout drift between zmp_walk and the "
            f"MJCF will silently mis-align the keyframe."
        )
    return np.asarray(traj.q_ref, dtype=np.float64), \
        np.asarray(traj.contact_mask, dtype=np.float64), joint_order


def _select_phase_neutral_dsp_frames(
    q_ref: np.ndarray,
    contact_mask: np.ndarray,
    joint_order: List[str],
    *,
    sym_tol: float,
) -> np.ndarray:
    """Boolean mask selecting DSP frames whose physical pose is mirror-symmetric.

    Symmetry test runs on the leg pitch chain:
      - hip_pitch is opposite-sign in raw joint coords (axes mirrored in MJCF),
      - knee_pitch + ankle_pitch share sign across L/R.
    A frame is accepted iff every paired difference is within ``sym_tol`` rad.

    Caveat: at the current ZMPWalkGenerator output the only window of the
    trajectory that survives this test is the from-rest initial DSP
    (frames [0..7]).  See the module docstring for the full discussion.
    """
    name_to_col = {name: i for i, name in enumerate(joint_order)}

    def col(name: str) -> int:
        if name not in name_to_col:
            raise KeyError(f"joint '{name}' missing from walking-prior layout")
        return name_to_col[name]

    is_dsp = (contact_mask[:, 0] > 0.5) & (contact_mask[:, 1] > 0.5)

    sym_ok = np.ones_like(is_dsp, dtype=bool)
    for left, right in MIRROR_PAIRS_OPP_SIGN:
        d = q_ref[:, col(left)] + q_ref[:, col(right)]
        sym_ok &= np.abs(d) < sym_tol
    for left, right in MIRROR_PAIRS_SAME_SIGN:
        d = q_ref[:, col(left)] - q_ref[:, col(right)]
        sym_ok &= np.abs(d) < sym_tol

    return is_dsp & sym_ok


def _derive_joint_pose(
    q_ref: np.ndarray,
    selected: np.ndarray,
    joint_order: List[str],
) -> np.ndarray:
    """Average q_ref across the selected frames; enforce strict L/R symmetry."""
    if not selected.any():
        raise RuntimeError(
            "No phase-neutral DSP frames found in the walking prior.  Check "
            "the symmetry tolerance or the operating-point cycle structure."
        )
    pose = q_ref[selected].mean(axis=0).astype(np.float64)

    name_to_col = {name: i for i, name in enumerate(joint_order)}

    def avg_then_mirror(left: str, right: str, opposite: bool) -> None:
        i, j = name_to_col[left], name_to_col[right]
        # Use the symmetric magnitude derived from both legs to drop the
        # residual sub-mrad IK noise the per-frame average leaves behind.
        if opposite:
            mag = 0.5 * (pose[i] - pose[j])
            pose[i] = +mag
            pose[j] = -mag
        else:
            mag = 0.5 * (pose[i] + pose[j])
            pose[i] = mag
            pose[j] = mag

    for left, right in MIRROR_PAIRS_OPP_SIGN:
        avg_then_mirror(left, right, opposite=True)
    for left, right in MIRROR_PAIRS_SAME_SIGN:
        avg_then_mirror(left, right, opposite=False)

    return pose


# ----------------------------------------------------------------------------
# Envelope coverage verification
# ----------------------------------------------------------------------------


def _envelope_report(
    q_ref: np.ndarray,
    home_pose: np.ndarray,
    joint_order: List[str],
    residual_scale: float,
) -> List[str]:
    """Per-joint envelope report (informational only — not a gate).

    Reports each leg joint's q_ref(t) range against ``home ± residual_scale``.
    ``deficit`` is the worst-side miss (max of below/above out-of-band gaps);
    0 means q_ref fits inside ``home ± residual_scale``, positive values
    show by how much the residual would need to grow to cover the gait.
    """
    name_to_col = {name: i for i, name in enumerate(joint_order)}
    lines: List[str] = []
    for joint in LEG_JOINTS:
        col = name_to_col[joint]
        lo, hi = float(q_ref[:, col].min()), float(q_ref[:, col].max())
        center = float(home_pose[col])
        below_deficit = max(0.0, (center - residual_scale) - lo)
        above_deficit = max(0.0, hi - (center + residual_scale))
        worst_deficit = max(below_deficit, above_deficit)
        tag = "covered " if worst_deficit <= 1e-6 else "uncov.  "
        lines.append(
            f"  [{tag}] {joint:<20s} home={center:+.4f}  "
            f"q_ref ∈ [{lo:+.4f}, {hi:+.4f}]  "
            f"below_deficit={below_deficit:+.4f}  "
            f"above_deficit={above_deficit:+.4f}  worst={worst_deficit:+.4f}"
        )
    return lines


def _validate_static_pose(
    model: mujoco.MjModel,
    home_pose: np.ndarray,
    joint_order: List[str],
) -> Tuple[bool, List[str]]:
    """HARD gate: finite, within MJCF joint ranges, exact L/R mirror symmetry."""
    name_to_col = {name: i for i, name in enumerate(joint_order)}
    issues: List[str] = []

    for joint in LEG_JOINTS:
        col = name_to_col[joint]
        v = float(home_pose[col])
        if not np.isfinite(v):
            issues.append(f"{joint}: non-finite ({v})")
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        if v < lo - 1e-6 or v > hi + 1e-6:
            issues.append(
                f"{joint}={v:+.4f} outside MJCF range [{lo:+.4f}, {hi:+.4f}]"
            )

    sym_tol = 1e-8
    for left, right in MIRROR_PAIRS_OPP_SIGN:
        d = float(home_pose[name_to_col[left]] + home_pose[name_to_col[right]])
        if abs(d) > sym_tol:
            issues.append(
                f"opposite-sign mirror violated: {left}+{right}={d:+.2e}"
            )
    for left, right in MIRROR_PAIRS_SAME_SIGN:
        d = float(home_pose[name_to_col[left]] - home_pose[name_to_col[right]])
        if abs(d) > sym_tol:
            issues.append(
                f"same-sign mirror violated: {left}-{right}={d:+.2e}"
            )

    return (not issues), issues


# ----------------------------------------------------------------------------
# qpos assembly + settling
# ----------------------------------------------------------------------------


def _build_actuator_qaddr_map(model: mujoco.MjModel) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        jid = int(model.actuator_trnid[aid, 0])
        out[aname] = int(model.jnt_qposadr[jid])
    return out


def _read_existing_keyframe(name: str) -> np.ndarray:
    text = KEYFRAMES_XML.read_text()
    m = re.search(rf'<key\s+name="{re.escape(name)}"\s+qpos="([^"]+)"', text)
    if not m:
        raise RuntimeError(f"keyframe '{name}' not found in {KEYFRAMES_XML}")
    return np.array([float(v) for v in m.group(1).split()])


def _compose_initial_qpos(
    model: mujoco.MjModel,
    derived_joint_pose: np.ndarray,
    joint_order: List[str],
    seed_keyframe: str,
) -> np.ndarray:
    """Build a full-length qpos seeded from ``seed_keyframe`` with the leg slots
    overwritten by the derived joint pose.

    Non-leg actuator slots (waist_yaw, arms) are intentionally NOT overwritten
    by the derived pose (the walking prior plans only the leg chain, leaving
    other slots at zero) — keeping ``seed_keyframe``'s settled non-leg values
    avoids re-deriving them and keeps the seed close to equilibrium so the
    settling loop converges quickly.
    """
    if seed_keyframe == "walk_start":
        seed = _read_existing_keyframe("walk_start")
    elif seed_keyframe == "home":
        seed = _read_existing_keyframe("home")
    else:
        raise ValueError(f"unsupported seed keyframe '{seed_keyframe}'")
    if seed.size != model.nq:
        raise RuntimeError(
            f"seed keyframe '{seed_keyframe}' has {seed.size} qpos values; "
            f"model expects {model.nq}.  Run resettle_keyframes.py first to "
            f"expand the seed keyframe for any newly added joints."
        )

    qaddr_map = _build_actuator_qaddr_map(model)
    initial = seed.astype(np.float64).copy()
    name_to_col = {n: i for i, n in enumerate(joint_order)}
    for joint in LEG_JOINTS:
        initial[qaddr_map[joint]] = float(derived_joint_pose[name_to_col[joint]])
    return initial


def _settle(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    qaddr_map: Dict[str, int],
    n_steps: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Same settling loop as assets/resettle_keyframes.py — run with
    ctrl pinned to the initial joint values; return final qpos, pelvis Δxy,
    max |qvel|, final pelvis_z.

    The settled qpos is clipped to MJCF joint ranges before being returned
    so the written keyframe is byte-equal to what the env's per-joint
    range clip produces (a few µrad of settling drift can otherwise push
    near-zero slots like ``left_elbow_pitch`` a hair outside their MJCF
    range, leaving the runtime ``load_home_from_scene`` path one cycle
    out of sync with the env's ``_home_q_rad``).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    initial_pelvis = data.qpos[0:3].copy()
    ctrl = np.empty(model.nu, dtype=np.float64)
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        ctrl[aid] = float(initial_qpos[qaddr_map[aname]])
    data.ctrl[:] = ctrl

    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    final_qpos = data.qpos.copy()
    pelvis_dxy = float(np.linalg.norm(final_qpos[0:2] - initial_pelvis[0:2]))
    max_qvel = float(np.max(np.abs(data.qvel)))
    pelvis_z = float(final_qpos[2])

    # Clip actuated-joint qpos slots to MJCF joint ranges (the freejoint
    # qpos[0:7] has no limited range and is left untouched).
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        qaddr = qaddr_map[aname]
        jid = int(model.actuator_trnid[aid, 0])
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        final_qpos[qaddr] = float(min(max(final_qpos[qaddr], lo), hi))
    return final_qpos, pelvis_dxy, max_qvel, pelvis_z


# ----------------------------------------------------------------------------
# keyframes.xml writer
# ----------------------------------------------------------------------------


def _format_qpos(qpos: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in qpos)


def _write_home_keyframe(qpos: np.ndarray) -> None:
    text = KEYFRAMES_XML.read_text()
    formatted = _format_qpos(qpos)
    pattern = re.compile(r'(<key\s+name="home"\s+qpos=")([^"]+)(")')
    text, n_replaced = pattern.subn(rf'\g<1>{formatted}\g<3>', text, count=1)
    if n_replaced != 1:
        raise RuntimeError(
            f"Could not locate <key name='home' qpos='...'/> in {KEYFRAMES_XML}"
        )
    KEYFRAMES_XML.write_text(text)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--write", action="store_true",
        help="Update assets/v2/keyframes.xml in place (default: dry-run print)",
    )
    parser.add_argument(
        "--vx", type=float, default=DERIVATION_VX,
        help=f"Operating velocity for the walking prior (default {DERIVATION_VX})",
    )
    parser.add_argument(
        "--symmetry-tol", type=float, default=DSP_SYMMETRY_TOL_RAD,
        help="DSP mirror-symmetry tolerance in radians "
        f"(default {DSP_SYMMETRY_TOL_RAD})",
    )
    parser.add_argument(
        "--settle-steps", type=int, default=SETTLE_STEPS,
        help=f"Physics settling steps (default {SETTLE_STEPS})",
    )
    args = parser.parse_args()

    print(f"Building walking prior at vx={args.vx:+.4f} m/s ...")
    q_ref, contact_mask, joint_order = _load_walking_q_ref(args.vx)
    print(f"  q_ref shape={q_ref.shape}  joint_order has {len(joint_order)} entries")

    selected = _select_phase_neutral_dsp_frames(
        q_ref, contact_mask, joint_order, sym_tol=args.symmetry_tol,
    )
    n_selected = int(selected.sum())
    if n_selected == 0:
        print("ERROR: no phase-neutral DSP frames selected.")
        return 2
    print(
        f"  Selected {n_selected} / {len(selected)} frames "
        f"(DSP ∩ mirror-symmetric within {args.symmetry_tol} rad)"
    )

    derived = _derive_joint_pose(q_ref, selected, joint_order)

    print("\nDerived leg joint pose:")
    name_to_col = {n: i for i, n in enumerate(joint_order)}
    for joint in LEG_JOINTS:
        print(f"  {joint:<20s} = {derived[name_to_col[joint]]:+.6f} rad")

    print(f"\nLoading MJCF: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print(f"  nq={model.nq}  nu={model.nu}")

    print("\nStatic-pose gate (finite + within MJCF limits + L/R mirror):")
    static_ok, static_issues = _validate_static_pose(model, derived, joint_order)
    if static_ok:
        print("  OK")
    else:
        for issue in static_issues:
            print(f"  FAIL: {issue}")
        return 3

    print(
        "\nEnvelope coverage report  "
        f"(informational; residual ±{RESIDUAL_SCALE_PER_LEG_JOINT_RAD} rad):"
    )
    for line in _envelope_report(
        q_ref, derived, joint_order, RESIDUAL_SCALE_PER_LEG_JOINT_RAD,
    ):
        print(line)
    print(
        "  NOTE: under TB/smoke9c semantics the residual is a small corrective "
        "term;\n        non-zero deficits are expected and do NOT block "
        "promotion."
    )

    qaddr_map = _build_actuator_qaddr_map(model)
    initial = _compose_initial_qpos(
        model, derived, joint_order, seed_keyframe="walk_start",
    )

    print(f"\nSettling for {args.settle_steps} physics steps ...")
    settled, dxy, max_qvel, pelvis_z = _settle(
        model, initial, qaddr_map, args.settle_steps,
    )
    ok_dxy = dxy < SETTLE_DPELVIS_TOL
    ok_qvel = max_qvel < SETTLE_QVEL_TOL
    status = "OK" if (ok_dxy and ok_qvel) else "WARNING"
    print(
        f"  pelvis_Δxy={dxy * 1000:.2f} mm "
        f"(<{SETTLE_DPELVIS_TOL * 1000:.0f}mm? {ok_dxy})  "
        f"max|qvel|={max_qvel:.4f} rad/s (<{SETTLE_QVEL_TOL}? {ok_qvel})  "
        f"pelvis_z={pelvis_z * 1000:.2f} mm  [{status}]"
    )

    print("\nSettled leg joint values:")
    for joint in LEG_JOINTS:
        print(
            f"  {joint:<20s} init={initial[qaddr_map[joint]]:+.6f}  "
            f"settled={settled[qaddr_map[joint]]:+.6f}  "
            f"drift={settled[qaddr_map[joint]] - initial[qaddr_map[joint]]:+.6f}"
        )

    print(f"\nFull settled qpos ({settled.size} values):")
    print(f"  {_format_qpos(settled)}")

    if not (ok_dxy and ok_qvel):
        print("\nWARNING: settling did not fully converge — keyframe may need")
        print("more steps or the derived pose may be far from equilibrium.")
        if args.write:
            print("Refusing to write a non-converged keyframe.")
            return 4
        return 0

    if args.write:
        _write_home_keyframe(settled)
        print(f"\nWrote new 'home' keyframe to {KEYFRAMES_XML}")
    else:
        print("\n--dry-run (default): no files modified.  Pass --write to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
