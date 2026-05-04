#!/usr/bin/env python3
"""Compare collision-primitive sizing options A1 (OBB-derived) vs A2 (legacy hardcoded).

Generates two model variants under a temp directory by running the full post_process
pipeline twice: once unmodified (A1), once with sizes overridden to the historical
hand-tuned values (A2). Then:

  Method 2 (render): produces a collision-shape PNG snapshot for each variant so you
    can visually compare coverage.
  Method 3 (contact audit): for a battery of named + synthetic stress poses, compiles
    each model, runs mj_forward, and lists every active contact pair. Diffs A1 vs A2.

Both methods are intentionally read-only against the production assets — nothing under
``assets/v2/wildrobot.xml`` is touched. The script reads from
``assets/v2/onshape_export/`` and writes only to ``--out-dir`` (default
``/tmp/wr_compare``).

Usage:
    uv run python assets/compare_collision_options.py
    uv run python assets/compare_collision_options.py --out-dir /tmp/foo --no-render
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
V2_DIR = ASSETS_DIR / "v2"
EXPORT_DIR = V2_DIR / "onshape_export"

# Historical hand-tuned sizes copied verbatim from commit 7461912 (pre-OBB).
# Keys are the source mesh names that the post_process replace_* functions target.
LEGACY_SIZES: Dict[str, Dict[str, str]] = {
    "upper_leg":               {"type": "capsule", "size": "0.020000 0.060188"},
    "forearm":                 {"type": "box",     "size": "0.021754 0.042512 0.064691"},
    "htd_45h_collision":       {"type": "box",     "size": "0.010095 0.026704 0.030001"},
    "upper_leg_servo_connect": {"type": "box",     "size": "0.022780 0.033799 0.030266"},
    "toe_btm":                 {"type": "box",     "size": "0.002040 0.037100 0.045000"},
    "heel_btm":                {"type": "box",     "size": "0.002062 0.021100 0.045000"},
}

# Map collision primitives in the post_process output back to their source mesh
# by inspecting the parent body. Each entry: {parent_body_name: source_mesh_name}.
# This is needed because post_process pops the `mesh` attribute when it converts
# meshes to primitives, so we lose direct identification afterwards.
PARENT_BODY_TO_MESH: Dict[str, List[str]] = {
    "upper_leg":   ["upper_leg", "upper_leg_servo_connect"],
    "upper_leg_2": ["upper_leg", "upper_leg_servo_connect"],
    "left_hip":    ["htd_45h_collision"],
    "right_hip":   ["htd_45h_collision"],
    # Forearm bodies are named "fore_arm" / "fore_arm_2" in the raw export.
    "fore_arm":    ["forearm"],
    "fore_arm_2":  ["forearm"],
    # Foot toe/heel collision primitives end up under left_foot / right_foot
    # after add_collision_names renames the foot bodies.
    "left_foot":   ["toe_btm", "heel_btm"],
    "right_foot":  ["toe_btm", "heel_btm"],
}


def _import_post_process():
    """Import assets/post_process.py as a module without side effects."""
    spec = importlib.util.spec_from_file_location(
        "post_process", ASSETS_DIR / "post_process.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_export(stage_dir: Path) -> Path:
    """Copy the raw Onshape export into stage_dir; return the new variant directory."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    shutil.copy(EXPORT_DIR / "wildrobot.xml", stage_dir / "wildrobot.xml")
    shutil.copytree(EXPORT_DIR / "assets", stage_dir / "assets")
    # Pull in the auxiliary files post_process expects next to wildrobot.xml.
    for aux in ("sensors.xml", "joints_properties.xml", "actuator_order.txt",
                "scene_flat_terrain.xml", "keyframes.xml"):
        src = V2_DIR / aux
        if src.exists():
            shutil.copy(src, stage_dir / aux)
    return stage_dir


def _run_post_process(xml_path: Path, post_process) -> None:
    """Run the production post_process pipeline on xml_path (in-place)."""
    xml_str = str(xml_path)
    post_process.inject_additional_xml(xml_str)
    post_process.add_collision_names(xml_str)
    post_process.validate_foot_geoms(xml_str)
    post_process.ensure_root_body_pose(xml_str)
    post_process.add_option(xml_str)
    post_process.merge_default_blocks(xml_str)
    post_process.fix_collision_default_geom(xml_str)
    post_process.replace_upper_leg_collision_with_capsule(xml_str)
    post_process.replace_forearm_collision_with_box(xml_str)
    post_process.replace_htd_45h_collision_with_box(xml_str)
    post_process.replace_upper_leg_servo_connect_collision_with_box(xml_str)
    post_process.replace_foot_bottom_collision_with_box(xml_str)


def _override_sizes_to_legacy(xml_path: Path) -> int:
    """Walk wildrobot.xml and override A1 OBB sizes with LEGACY_SIZES values.

    Identifies primitives by walking the body tree and matching parent body name
    + primitive type against PARENT_BODY_TO_MESH expectations. Returns the count
    of geoms whose size was overridden.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    overridden = 0

    for body in root.findall(".//body[@name]"):
        body_name = body.get("name")
        candidate_meshes = PARENT_BODY_TO_MESH.get(body_name, [])
        if not candidate_meshes:
            continue
        for geom in body.findall("geom"):
            if geom.get("class") != "collision":
                continue
            geom_type = geom.get("type")
            if geom_type not in ("box", "capsule"):
                continue
            # Match by primitive type + body convention:
            # - upper_leg parent has both a capsule (upper_leg mesh) and a box
            #   (upper_leg_servo_connect mesh). Distinguish by type.
            # - foot parent has two boxes (toe_btm and heel_btm). Distinguish
            #   by primitive size: heel_btm is shorter on its y axis than toe_btm.
            mesh_for_geom: Optional[str] = None
            if body_name in ("upper_leg", "upper_leg_2"):
                if geom_type == "capsule":
                    mesh_for_geom = "upper_leg"
                else:  # box
                    mesh_for_geom = "upper_leg_servo_connect"
            elif body_name in ("left_foot", "right_foot"):
                # toe vs heel by checking pos along sagittal (x axis): toe is
                # forward (+x), heel is backward (-x).
                pos_vals = [float(x) for x in geom.get("pos", "0 0 0").split()]
                mesh_for_geom = "toe_btm" if pos_vals[0] > 0 else "heel_btm"
            elif body_name in ("left_hip", "right_hip"):
                mesh_for_geom = "htd_45h_collision"
            elif body_name in ("fore_arm", "fore_arm_2"):
                mesh_for_geom = "forearm"
            if mesh_for_geom is None:
                continue
            legacy = LEGACY_SIZES.get(mesh_for_geom)
            if legacy is None:
                continue
            if legacy["type"] != geom_type:
                continue
            geom.set("size", legacy["size"])
            overridden += 1

    if overridden > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_path)
    return overridden


@dataclass
class Variant:
    name: str
    xml_path: Path
    scene_path: Path


def _make_variant(name: str, out_root: Path, post_process, *, legacy_sizes: bool) -> Variant:
    var_dir = _stage_export(out_root / name)
    _run_post_process(var_dir / "wildrobot.xml", post_process)
    if legacy_sizes:
        n = _override_sizes_to_legacy(var_dir / "wildrobot.xml")
        print(f"  [{name}] overrode {n} primitive sizes to legacy hand-tuned values")
    # generate_robot_config too, so the model is complete (validate_model needs scene).
    post_process.generate_robot_config(
        str(var_dir / "wildrobot.xml"),
        str(var_dir / "mujoco_robot_config.json"),
    )
    return Variant(name=name, xml_path=var_dir / "wildrobot.xml",
                   scene_path=var_dir / "scene_flat_terrain.xml")


def _collect_collision_primitive_summary(xml_path: Path) -> List[Dict[str, str]]:
    tree = ET.parse(xml_path)
    out = []
    for body in tree.getroot().findall(".//body[@name]"):
        for geom in body.findall("geom"):
            if geom.get("class") != "collision":
                continue
            if geom.get("type") not in ("box", "capsule"):
                continue
            out.append({
                "body": body.get("name"),
                "type": geom.get("type"),
                "size": geom.get("size"),
                "pos": geom.get("pos"),
                "quat": geom.get("quat"),
            })
    return out


def _print_size_diff(a1_summary, a2_summary) -> None:
    print("\nSize diff (A1 OBB-derived vs A2 legacy):")
    print(f"  {'body':<20s} {'type':<8s} {'A1 size':<35s} {'A2 size':<35s}")
    seen = set()
    for a1, a2 in zip(a1_summary, a2_summary):
        key = (a1["body"], a1["type"])
        if key in seen:
            continue
        seen.add(key)
        same = "✓ same" if a1["size"] == a2["size"] else "↺ diff"
        print(f"  {a1['body']:<20s} {a1['type']:<8s} {a1['size']:<35s} {a2['size']:<35s} {same}")


# =====================================================================
# Method 2: render collision shapes
# =====================================================================

def _render_collision(variant: Variant, out_png: Path, height: int = 480, width: int = 720) -> None:
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
    data = mujoco.MjData(model)
    # Initialize from "home" keyframe if present.
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    scene_option = mujoco.MjvOption()
    # Show collision shapes (group 3) and HIDE visual (group 2) so the picture is collision-only.
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.6
    cam.elevation = -20.0
    cam.azimuth = 120.0
    cam.lookat[:] = data.qpos[0:3]
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    img = renderer.render()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(out_png)
    except ImportError:
        # Fallback: write raw NPY.
        np.save(out_png.with_suffix(".npy"), img)
        print(f"  PIL not available; wrote raw array to {out_png.with_suffix('.npy')}")


# =====================================================================
# Method 3: contact-pair audit
# =====================================================================

@dataclass
class PoseSpec:
    name: str
    keyframe: Optional[str] = None  # use named MJCF keyframe
    qpos_overrides: Optional[Dict[str, float]] = None  # joint_name -> radians

# Synthetic stress poses (joint values in radians).
STRESS_POSES: List[PoseSpec] = [
    PoseSpec("home", keyframe="home"),
    PoseSpec("walk_start", keyframe="walk_start"),
    PoseSpec("deep_squat", keyframe="home", qpos_overrides={
        "left_hip_pitch":  0.9, "right_hip_pitch":  -0.9,
        "left_knee_pitch": 1.6, "right_knee_pitch": 1.6,
        "left_ankle_pitch": -0.6, "right_ankle_pitch": -0.6,
    }),
    PoseSpec("knees_in", keyframe="home", qpos_overrides={
        "left_hip_roll":  0.15, "right_hip_roll": -0.15,
        "left_knee_pitch": 0.6, "right_knee_pitch": 0.6,
    }),
    PoseSpec("arms_in_to_body", keyframe="home", qpos_overrides={
        "left_shoulder_roll":  -1.2, "right_shoulder_roll": 1.2,
        "left_elbow_pitch":     1.2, "right_elbow_pitch":  -1.2,
    }),
]


def _set_pose(model, data, pose: PoseSpec) -> None:
    import mujoco
    if pose.keyframe is not None:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose.keyframe)
        if kid < 0:
            mujoco.mj_resetData(model, data)
        else:
            mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)
    if pose.qpos_overrides:
        for jname, val in pose.qpos_overrides.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qaddr = int(model.jnt_qposadr[jid])
            data.qpos[qaddr] = float(val)
    mujoco.mj_forward(model, data)


def _list_contacts(model, data) -> List[Tuple[str, str, float]]:
    import mujoco
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
        # Use penetration depth (more useful than contact frame norm).
        depth = float(c.dist)
        pair = tuple(sorted([n1, n2]))
        out.append((pair[0], pair[1], depth))
    return out


def _audit_variant(variant: Variant) -> Dict[str, List[Tuple[str, str, float]]]:
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
    data = mujoco.MjData(model)
    out = {}
    for pose in STRESS_POSES:
        _set_pose(model, data, pose)
        out[pose.name] = sorted(_list_contacts(model, data))
    return out


def _diff_audits(a1, a2) -> None:
    print("\nContact-pair audit (per pose):")
    for pose_name in a1.keys():
        c1 = set((g1, g2) for g1, g2, _ in a1[pose_name])
        c2 = set((g1, g2) for g1, g2, _ in a2[pose_name])
        only_a1 = c1 - c2
        only_a2 = c2 - c1
        common = c1 & c2
        print(f"\n  pose: {pose_name}")
        print(f"    common contact pairs: {len(common)}")
        if only_a2:
            print(f"    pairs only in A2 (legacy): {len(only_a2)}")
            for p in sorted(only_a2):
                print(f"      + {p[0]} <-> {p[1]}")
        if only_a1:
            print(f"    pairs only in A1 (OBB):    {len(only_a1)}")
            for p in sorted(only_a1):
                print(f"      + {p[0]} <-> {p[1]}")
        if not only_a1 and not only_a2:
            print("    (no difference)")


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/wr_compare"),
                        help="Working directory for staged variants and outputs")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip Method 2 (rendering)")
    parser.add_argument("--no-audit", action="store_true",
                        help="Skip Method 3 (contact audit)")
    args = parser.parse_args()

    sys.path.insert(0, str(ASSETS_DIR))
    post_process = _import_post_process()

    print(f"Staging A1 (OBB-derived sizes)  -> {args.out_dir/'a1'}")
    a1 = _make_variant("a1", args.out_dir, post_process, legacy_sizes=False)
    print(f"\nStaging A2 (legacy hand-tuned sizes) -> {args.out_dir/'a2'}")
    a2 = _make_variant("a2", args.out_dir, post_process, legacy_sizes=True)

    a1_summary = _collect_collision_primitive_summary(a1.xml_path)
    a2_summary = _collect_collision_primitive_summary(a2.xml_path)
    _print_size_diff(a1_summary, a2_summary)

    if not args.no_render:
        try:
            print("\nMethod 2: rendering collision shapes...")
            _render_collision(a1, args.out_dir / "a1_collision.png")
            _render_collision(a2, args.out_dir / "a2_collision.png")
            print(f"  wrote {args.out_dir/'a1_collision.png'}")
            print(f"  wrote {args.out_dir/'a2_collision.png'}")
        except Exception as exc:
            print(f"  Render failed: {exc}")
            print("  (re-run with --no-render to skip, or install Pillow if missing)")

    if not args.no_audit:
        print("\nMethod 3: running contact-pair audit...")
        try:
            a1_audit = _audit_variant(a1)
            a2_audit = _audit_variant(a2)
            _diff_audits(a1_audit, a2_audit)
        except Exception as exc:
            print(f"  Audit failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
