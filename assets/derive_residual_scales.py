#!/usr/bin/env python3
"""Derive per-joint residual action scales from the WR walking prior + home.

Background
----------
With ``loc_ref_residual_base=home`` (smoke10+), the residual action contract is

    q_target[j] = home[j] + clip(action[j], -1, 1) * residual_scale_per_joint[j]

For a given prior and home pose, the residual must reach far enough that the
policy can express the gait amplitudes the prior demands (or, with imitation
turned off, the amplitudes that an emergent `feet_phase`-satisfying gait
needs).  smoke9c through smoke11 used a flat 0.25 rad/joint for every leg
joint, copied from ToddlerBot's `mjx_config.py:103`.  Direct measurement
against WR's v0.1333 m/s offline library shows that 0.25 is structurally too
tight on the WR prior:

  joint               prior amplitude   over 0.25 cap
  left_knee_pitch     0.89 rad          3.6x
  right_knee_pitch    0.89 rad          3.6x
  left_hip_pitch      0.45 rad          1.8x
  right_hip_pitch     0.52 rad          2.1x
  left_ankle_pitch    0.10 rad          within
  right_ankle_pitch   0.10 rad          within

WR is a longer-leg robot than TB (upper+lower leg 0.37 m vs ~0.20 m), so the
same kinematic ask (5 cm swing) requires more angular change at the knee/hip.
TB walks at 0.25 because TB's emergent gait has smaller amplitudes; WR
inherited the cap without re-deriving.

Derivation
----------
For each leg joint:

    amplitude[j] = max_t |q_ref(t)[j] - home[j]|          # from prior + keyframe
    scale[j]     = max(floor, amplitude[j] * coverage)    # bounded below

``coverage`` is a Python float that captures the tradeoff between
"residual can express the prior" (coverage=1.0) and "residual stays a
bounded correction so G5 anti-exploit still holds" (coverage=0.5-0.75).

For non-leg joints (waist_yaw, arms), output a single scalar floor
(``--non-leg-floor``, default 0.20) — those joints aren't in the
prior chain and the scalar matches the smoke9c default.

Two output modes:
  - default (stdout): print the YAML block ready to paste.
  - ``--write CONFIG_YAML``: in-place regex replace of the
    ``loc_ref_residual_scale_per_joint:`` block in the given config.
    Same single-block edit pattern as
    ``assets/derive_walk_ready_home.py::_write_home_keyframe``.

Usage
-----
    # Inspect at the smoke9c operating point with 75% prior coverage:
    uv run python assets/derive_residual_scales.py
    uv run python assets/derive_residual_scales.py --coverage 0.75

    # Generate scales for full prior coverage with 10% headroom:
    uv run python assets/derive_residual_scales.py --coverage 1.1

    # Regenerate and write into smoke12 in place:
    uv run python assets/derive_residual_scales.py \\
        --coverage 0.75 \\
        --write training/configs/ppo_walking_v0201_smoke12.yaml

Re-run this script whenever the prior is regenerated (new ZMP config, new
operating-point vx, new home keyframe).
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

# Smoke9c operating point — kept as a module constant so the smoke yaml,
# the home derivation, and the residual derivation all reference one
# place.  Override with ``--vx``.
DEFAULT_DERIVATION_VX = 0.1333333333

# Leg pitch + roll chain that the prior actually drives.  Non-leg joints
# (waist_yaw, shoulders, elbows, wrists) get the scalar fallback.
LEG_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_knee_pitch",
    "right_ankle_pitch", "right_ankle_roll",
)

# Minimum scale for any joint.  Joints whose prior amplitude * coverage
# is smaller than this floor keep the floor.  0.25 matches the smoke9c
# baseline so non-leg joints don't shrink relative to history.
DEFAULT_FLOOR = 0.25

# Default coverage of the prior's per-joint amplitude.  0.75 covers the
# main swing while keeping residual bounded enough to preserve the G5
# anti-exploit gate (max residual stays below the prior's peak demand).
# Pass ``--coverage 1.0`` (or higher with margin) to fully reach the
# prior; pass ``--coverage 0.5`` for a tighter, more TB-like cap.
DEFAULT_COVERAGE = 0.75


# ----------------------------------------------------------------------------
# Reading prior + home
# ----------------------------------------------------------------------------


def _load_prior_q_ref(vx: float) -> Tuple[np.ndarray, List[str]]:
    """Build the offline library at ``vx`` and return (q_ref, joint_order).

    Joint order is the PolicySpec actuator order used by ZMPWalkGenerator
    when populating ``q_ref`` columns; matches the MJCF actuator order.
    """
    generator = ZMPWalkGenerator()
    library = generator.build_library_for_vx_values([vx])
    traj = library.lookup(vx)
    layout = generator._load_actuator_layout()
    joint_order = [str(name) for name in layout["actuator_names"]]
    if traj.q_ref.shape[1] != len(joint_order):
        raise RuntimeError(
            f"q_ref columns ({traj.q_ref.shape[1]}) != actuator count "
            f"({len(joint_order)}); planner / MJCF actuator order drift."
        )
    return np.asarray(traj.q_ref, dtype=np.float64), joint_order


def _load_home_pose(joint_order: List[str], keyframe_name: str) -> np.ndarray:
    """Return the per-actuator home pose (rad), aligned to ``joint_order``."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if kid < 0:
        raise RuntimeError(
            f"keyframe '{keyframe_name}' not found in {SCENE_XML.name}; "
            "regenerate via assets/derive_walk_ready_home.py."
        )
    key_qpos = np.asarray(model.key_qpos[kid], dtype=np.float64)

    out = np.empty(len(joint_order), dtype=np.float64)
    for i, name in enumerate(joint_order):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(
                f"actuator '{name}' missing from MJCF (planner / model drift)."
            )
        jid = int(model.actuator_trnid[aid, 0])
        qaddr = int(model.jnt_qposadr[jid])
        out[i] = float(key_qpos[qaddr])
    return out


# ----------------------------------------------------------------------------
# Derivation
# ----------------------------------------------------------------------------


def _derive_per_joint_scales(
    q_ref: np.ndarray,
    home: np.ndarray,
    joint_order: List[str],
    *,
    coverage: float,
    floor: float,
) -> Tuple[Dict[str, float], List[str]]:
    """Compute residual scales for the leg joints + an analysis report.

    Returns
    -------
    scales : {leg_joint_name: scale_rad}
    report : per-joint markdown-ish lines suitable for printing
    """
    name_to_col = {n: i for i, n in enumerate(joint_order)}
    scales: Dict[str, float] = {}
    lines: List[str] = []
    lines.append(
        f"  {'joint':<22} {'home':>8} {'q_min':>8} {'q_max':>8} "
        f"{'|Δ|_max':>8} {'*cov':>7} {'->scale':>9}"
    )
    for joint in LEG_JOINTS:
        col = name_to_col[joint]
        h = float(home[col])
        qmin = float(q_ref[:, col].min())
        qmax = float(q_ref[:, col].max())
        amp = max(abs(qmin - h), abs(qmax - h))
        proposed = amp * coverage
        scale = max(floor, proposed)
        scales[joint] = scale
        floor_tag = " (floor)" if scale == floor and proposed < floor else ""
        lines.append(
            f"  {joint:<22} {h:+.4f} {qmin:+.4f} {qmax:+.4f} "
            f"{amp:8.4f} {proposed:7.4f}  {scale:7.4f}{floor_tag}"
        )
    return scales, lines


# ----------------------------------------------------------------------------
# YAML emitters
# ----------------------------------------------------------------------------


def _format_yaml_block(
    scales: Dict[str, float],
    *,
    non_leg_floor: float,
    coverage: float,
    vx: float,
    indent: str = "  ",
) -> str:
    """Render the loc_ref_residual_scale / per_joint block as YAML text."""
    lines = []
    lines.append(f"{indent}# Generated by assets/derive_residual_scales.py")
    lines.append(
        f"{indent}# vx={vx:.10g}  coverage={coverage:.3g}  "
        f"floor={non_leg_floor:.3g}"
    )
    lines.append(
        f"{indent}# Re-run that script after regenerating the prior / home."
    )
    lines.append(f"{indent}loc_ref_residual_mode: absolute")
    lines.append(f"{indent}loc_ref_residual_scale: {non_leg_floor:g}")
    lines.append(f"{indent}loc_ref_residual_scale_per_joint:")
    for joint in LEG_JOINTS:
        scale = scales[joint]
        # Round to 4 decimals — keeps reads tidy without losing precision
        # the env actually cares about.
        lines.append(f"{indent}  {joint}: {round(scale, 4)}")
    return "\n".join(lines) + "\n"


_PER_JOINT_BLOCK_RE = re.compile(
    r"(\n[ \t]*loc_ref_residual_mode:[^\n]*\n"
    r"[ \t]*loc_ref_residual_scale:[^\n]*\n"
    r"[ \t]*loc_ref_residual_scale_per_joint:\n"
    r"(?:[ \t]+[A-Za-z_]+:[ \t]*-?\d+(?:\.\d+)?[ \t]*\n)+)"
)


def _write_yaml_block(config_path: Path, new_block: str) -> None:
    """Replace the existing residual-scale block in-place.

    The block boundary is recognised by the three consecutive lines
    ``loc_ref_residual_mode`` / ``loc_ref_residual_scale`` /
    ``loc_ref_residual_scale_per_joint`` followed by one or more
    ``name: value`` lines.  Comments / blank lines OUTSIDE the block
    are preserved (regex replace, not yaml round-trip).
    """
    text = config_path.read_text()
    m = _PER_JOINT_BLOCK_RE.search(text)
    if not m:
        raise RuntimeError(
            f"Could not locate the loc_ref_residual_scale block in "
            f"{config_path}.  Expected a "
            "`loc_ref_residual_mode -> loc_ref_residual_scale -> "
            "loc_ref_residual_scale_per_joint:` triple followed by per-joint "
            "lines.  Adjust the regex if the file layout changed."
        )
    # Strip the leading newline from new_block since the regex captures one,
    # and ensure the block has the same leading newline format as before.
    replacement = "\n" + new_block.lstrip("\n")
    config_path.write_text(text[: m.start()] + replacement + text[m.end():])


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h", "--h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--vx", type=float, default=DEFAULT_DERIVATION_VX,
        help="Operating velocity for the offline prior (m/s).",
    )
    parser.add_argument(
        "--coverage", type=float, default=DEFAULT_COVERAGE,
        help=(
            "Per-joint scale = max(floor, prior_amplitude * coverage).  "
            "0.75 = 75%% of prior amplitude (keeps residual bounded); "
            "1.0 = full prior cover; 1.1 = 10%% headroom; "
            "0.5 = TB-like tight cap."
        ),
    )
    parser.add_argument(
        "--floor", type=float, default=DEFAULT_FLOOR,
        help="Lower bound on every per-joint scale (rad).",
    )
    parser.add_argument(
        "--non-leg-floor", type=float, default=0.20,
        help=(
            "Scalar fallback ``loc_ref_residual_scale`` for non-leg joints.  "
            "0.20 matches the smoke9c default."
        ),
    )
    parser.add_argument(
        "--keyframe", default="home",
        help="MJCF keyframe to read the home pose from.",
    )
    parser.add_argument(
        "--write", type=str, default=None,
        help=(
            "If set, regex-replace the residual-scale block in this YAML "
            "config in place.  Default: print to stdout only."
        ),
    )
    args = parser.parse_args()

    if args.coverage <= 0:
        print("ERROR: --coverage must be > 0.", file=sys.stderr)
        return 2
    if args.floor < 0:
        print("ERROR: --floor must be >= 0.", file=sys.stderr)
        return 2

    print(
        f"[derive_residual_scales] vx={args.vx:.10g}  "
        f"coverage={args.coverage:.3g}  floor={args.floor:.3g}  "
        f"keyframe={args.keyframe!r}"
    )

    print(f"[derive_residual_scales] building prior at vx={args.vx:.10g} ...")
    q_ref, joint_order = _load_prior_q_ref(args.vx)
    home = _load_home_pose(joint_order, args.keyframe)

    scales, report_lines = _derive_per_joint_scales(
        q_ref, home, joint_order,
        coverage=args.coverage,
        floor=args.floor,
    )
    print()
    print("[derive_residual_scales] per-joint analysis:")
    for line in report_lines:
        print(line)

    yaml_block = _format_yaml_block(
        scales,
        non_leg_floor=args.non_leg_floor,
        coverage=args.coverage,
        vx=args.vx,
    )
    print()
    print("[derive_residual_scales] generated YAML block:")
    print(yaml_block.rstrip("\n"))

    if args.write:
        config_path = Path(args.write)
        if not config_path.exists():
            print(f"ERROR: --write target not found: {config_path}",
                  file=sys.stderr)
            return 3
        _write_yaml_block(config_path, yaml_block)
        print(f"\n[derive_residual_scales] wrote block into {config_path}")
    else:
        print(
            "\n[derive_residual_scales] (--write CONFIG_YAML to splice this "
            "into a config in place)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
