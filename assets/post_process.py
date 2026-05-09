import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh


def inject_additional_xml(
    xml_file: str,
    *,
    additional_files: Optional[List[str]] = None,
) -> None:
    """Splice top-level elements from auxiliary XML files into the main MJCF.

    Replaces the onshape-to-robot ``additional_xml`` config option that the
    upstream pipeline used to consume. Each auxiliary file is parsed and its
    root element is appended as a top-level child of ``<mujoco>``. Downstream
    steps (notably :func:`merge_default_blocks`) collapse any duplicate
    top-level blocks like ``<default>``.

    Files are looked up next to ``xml_file``. Missing files are skipped with
    a warning so the function stays safe to re-run.
    """
    if additional_files is None:
        additional_files = ["sensors.xml", "joints_properties.xml"]

    xml_path = Path(xml_file)
    base_dir = xml_path.parent

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for aux in additional_files:
        aux_path = base_dir / aux
        if not aux_path.exists():
            print(f"[inject] missing additional XML, skipping: {aux_path}")
            continue
        aux_root = ET.parse(aux_path).getroot()
        root.append(aux_root)
        print(f"[inject] injected <{aux_root.tag}> from {aux}")

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def add_option(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)  # Insert at the beginning

    eulerdamp_flag = option.find("flag[@eulerdamp='disable']")
    if eulerdamp_flag is None:
        flag = ET.Element("flag")
        flag.set("eulerdamp", "disable")
        option.append(flag)

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def add_collision_names(xml_file):
    """Add clear semantic names to foot collision geoms.

    v0.5.0: Changed from left_foot_btm_back/front to left_heel/toe
    for clarity.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # v0.11.x: Some newer exports use generic foot body names ("foot", "foot_2").
    # Normalize to stable names used by training + mujoco_robot_config.json.
    body_names = {b.get("name") for b in root.findall(".//body") if b.get("name")}
    if "left_foot" not in body_names and "foot" in body_names:
        for body in root.findall(".//body[@name='foot']"):
            body.set("name", "left_foot")
    if "right_foot" not in body_names and "foot_2" in body_names:
        for body in root.findall(".//body[@name='foot_2']"):
            body.set("name", "right_foot")

    # Map mesh prefixes to semantic collision geom names per foot side.
    # Supports both:
    # - direct meshes: toe_btm / heel_btm
    # - convex decomposition meshes: toe_btm_00000, heel_btm_00001, ...
    mesh_prefix_to_name = {
        "toe_btm": {"left": "left_toe", "right": "right_toe"},
        "heel_btm": {"left": "left_heel", "right": "right_heel"},
    }

    for side in ("left", "right"):
        body_name = f"{side}_foot"
        foot_body = root.find(f".//body[@name='{body_name}']")
        if foot_body is None:
            continue

        collision_geoms = [
            g for g in foot_body.findall("geom") if g.get("class") == "collision"
        ]

        for mesh_prefix, names in mesh_prefix_to_name.items():
            desired = names[side]
            # Pick one deterministic collision geom for each semantic toe/heel marker.
            # Keep a single named geom so downstream lookups remain stable.
            matches = [
                g
                for g in collision_geoms
                if (g.get("mesh") or "").startswith(mesh_prefix)
            ]
            if not matches:
                continue
            matches.sort(key=lambda g: g.get("mesh") or "")
            matches[0].set("name", desired)

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def rename_legs_and_arms_to_semantic_names(xml_file: str) -> None:
    """Rename inner-leg and arm bodies from raw Onshape names to semantic
    `left_*` / `right_*` names, matching the convention already used by hip
    and foot bodies.

    Raw Onshape exports ship with two inconsistent conventions in the same
    model: inner-leg bodies (`upper_leg`, `lower_leg`, `servo_bracket_top`)
    are plain on the LEFT side and `_2`-suffixed on the RIGHT, while arm
    bodies (`shoulder`, `upper_arm`, `fore_arm`, `palm`, `finger`) are plain
    on the RIGHT side and `_2`-suffixed on the LEFT (the inversion is a
    CAD-assembly mating quirk). Joint names in the raw export are already
    canonical `left_*` / `right_*`, so we use them as the side oracle.

    See assets/v2/cad_arm_chain_followups.md (Issue 3) for context. This
    function is the workaround until the Onshape source is renamed.
    """
    target_stems = (
        "upper_leg",
        "lower_leg",
        "servo_bracket_top",
        "shoulder",
        "upper_arm",
        "fore_arm",
        "palm",
        "finger",
    )

    tree = ET.parse(xml_file)
    root = tree.getroot()
    body_names = {b.get("name") for b in root.findall(".//body") if b.get("name")}

    renames: List[Tuple[str, str]] = []
    for stem in target_stems:
        for raw in (stem, f"{stem}_2"):
            if raw not in body_names:
                continue
            body = root.find(f".//body[@name='{raw}']")
            if body is None:
                continue
            joint = body.find("joint")
            jname = joint.get("name") if joint is not None else None
            if jname is None:
                continue
            if jname.startswith("left_"):
                target = f"left_{stem}"
            elif jname.startswith("right_"):
                target = f"right_{stem}"
            else:
                continue
            if raw == target:
                continue  # already canonical (idempotent re-run)
            if target in body_names:
                # Defensive: refuse silent collisions. This would only fire
                # if the raw export ever contained both `stem` and `left_stem`
                # (or `right_stem`) for the same side at the same time.
                raise ValueError(
                    f"Body rename collision: cannot rename '{raw}' -> "
                    f"'{target}' because '{target}' already exists."
                )
            body.set("name", target)
            body_names.add(target)
            body_names.discard(raw)
            renames.append((raw, target))

    if renames:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        joined = ", ".join(f"{a}->{b}" for a, b in renames)
        print(f"Renamed inner-leg/arm bodies to semantic names: {joined}")


def validate_foot_geoms(xml_file: str) -> None:
    """Validate left/right toe/heel collision geoms exist and are symmetric.

    This catches common export issues where mirrored parts end up with large
    baked-in mesh offsets (e.g., one side having a huge mesh_pos), which can
    look visually correct but produce asymmetric contacts and drift.

    Validation is best-effort:
    - Always performs XML-level presence checks.
    - If `mujoco` is importable, compiles the model and validates poses/friction
      and rejects pathological mesh reference offsets (mesh_pos).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    body_names = {b.get("name") for b in root.findall(".//body") if b.get("name")}
    missing_bodies = [b for b in ("left_foot", "right_foot") if b not in body_names]
    if missing_bodies:
        raise ValueError(f"Missing required foot bodies in XML: {missing_bodies}")

    geom_names = {g.get("name") for g in root.findall(".//geom") if g.get("name")}
    required_geoms = {"left_toe", "left_heel", "right_toe", "right_heel"}
    missing_geoms = sorted(required_geoms - geom_names)
    if missing_geoms:
        raise ValueError(
            f"Missing required named collision geoms in XML: {missing_geoms}. "
            "Did add_collision_names() run?"
        )

    try:
        import mujoco
        import numpy as np
    except Exception as exc:
        print(
            f"Warning: skipping compiled foot geom validation (mujoco import failed: {exc})."
        )
        print("  Tip: run via repo env: `uv run python assets/post_process.py`")
        return

    model = mujoco.MjModel.from_xml_path(str(Path(xml_file).resolve()))

    def _id(obj_type, name: str) -> int:
        return mujoco.mj_name2id(model, obj_type, name)

    left_body = _id(mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_body = _id(mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    if left_body < 0 or right_body < 0:
        raise ValueError("Compiled model missing left_foot/right_foot bodies.")

    # Forward-kinematics evaluation at the home keyframe (or default qpos if
    # no keyframe). World-frame comparison is the only frame-agnostic way to
    # check L/R foot symmetry, since the foot bodies' local frames are
    # mirror-related and their `geom_pos` / `geom_quat` values are also
    # mirror-related (not directly equal).
    data = mujoco.MjData(model)
    home_kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_kid)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    def _geom_info(name: str):
        gid = _id(mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Compiled model missing geom '{name}'.")
        return {
            "gid": gid,
            "body": int(model.geom_bodyid[gid]),
            "world_pos": data.geom_xpos[gid].copy(),
            "world_mat": data.geom_xmat[gid].reshape(3, 3).copy(),
            "friction": model.geom_friction[gid].copy(),
            "size": model.geom_size[gid].copy(),
            "type": int(model.geom_type[gid]),
            "dataid": int(model.geom_dataid[gid]),
        }

    lt = _geom_info("left_toe")
    lh = _geom_info("left_heel")
    rt = _geom_info("right_toe")
    rh = _geom_info("right_heel")

    if lt["body"] != left_body or lh["body"] != left_body:
        raise ValueError(
            f"Left toe/heel geoms not under left_foot body (got bodies {lt['body']}, {lh['body']})."
        )
    if rt["body"] != right_body or rh["body"] != right_body:
        raise ValueError(
            f"Right toe/heel geoms not under right_foot body (got bodies {rt['body']}, {rh['body']})."
        )

    # Frame-agnostic checks (size and friction are mirror-invariant).
    size_tol = 1e-4
    friction_tol = 1e-4
    for key, a, b in (
        ("toe geom_size", lt["size"], rt["size"]),
        ("heel geom_size", lh["size"], rh["size"]),
        ("toe geom_friction", lt["friction"], rt["friction"]),
        ("heel geom_friction", lh["friction"], rh["friction"]),
    ):
        if not np.allclose(a, b, atol=size_tol if "size" in key else friction_tol, rtol=0.0):
            raise ValueError(f"Foot symmetry check failed for {key}: {a} vs {b}")

    # World-frame mirror check: for sagittal-plane symmetry (mirror across the
    # body's XZ plane, i.e. y → -y), the right-side world AABB should be the
    # y-negated mirror of the left side. Comparing world AABB (rather than the
    # raw rotation matrix) is convention-agnostic — different CAD assemblies
    # can give equivalent boxes with locally-relabeled axes (e.g. local +y vs
    # -y), which produces a different rotation matrix but an identical
    # physical footprint. Tolerance accommodates sub-mm CAD mirror imperfections
    # (the 2026-04 upper_leg fix closed an 0.8 mm gap; residual sub-mm
    # asymmetries remain at the foot ~0.1 mm scale).
    world_pos_tol = 5e-4   # 0.5 mm — catches mm-scale CAD bugs, allows µm noise

    def _world_aabb_half_extents(info):
        # For a box with half-sizes s and world rotation R, world AABB
        # half-extent along world axis i = sum_j |R[i,j]| * s[j].
        return np.abs(info["world_mat"]) @ info["size"]

    for key, left, right in (
        ("toe", lt, rt),
        ("heel", lh, rh),
    ):
        # Mirror the left's world center across the XZ-plane (y -> -y), compare to right.
        mirrored_left_pos = left["world_pos"] * np.array([1.0, -1.0, 1.0])
        pos_err = mirrored_left_pos - right["world_pos"]
        if np.max(np.abs(pos_err)) > world_pos_tol:
            raise ValueError(
                f"Foot {key} world-pos mirror check failed "
                f"(max |Δ| = {np.max(np.abs(pos_err))*1000:.2f} mm > {world_pos_tol*1000:.1f} mm): "
                f"mirror(left)={mirrored_left_pos} vs right={right['world_pos']}"
            )
        left_extents = _world_aabb_half_extents(left)
        right_extents = _world_aabb_half_extents(right)
        ext_err = left_extents - right_extents
        if np.max(np.abs(ext_err)) > world_pos_tol:
            raise ValueError(
                f"Foot {key} world AABB extents mismatch "
                f"(max |Δ| = {np.max(np.abs(ext_err))*1000:.2f} mm > {world_pos_tol*1000:.1f} mm): "
                f"left={left_extents} vs right={right_extents}"
            )

    # Mesh reference offsets: large values indicate bad part-local origins.
    mesh_pos = getattr(model, "mesh_pos", None)
    if mesh_pos is not None:
        def _mesh_pos_for_geom(info):
            if info["type"] != int(mujoco.mjtGeom.mjGEOM_MESH):
                return None
            mesh_id = info["dataid"]
            if mesh_id < 0:
                return None
            return mesh_pos[mesh_id].copy()

        # With OBB-derived collision primitives (replace_*_with_box/capsule),
        # the box pose is computed from the source mesh's geometric centroid,
        # not from the mesh anchor. This check therefore becomes a soft CAD-
        # hygiene heuristic: it only catches truly pathological origins (e.g.,
        # parts anchored a meter away from their geometry). Onshape exports for
        # the new ankle_roll foot place the toe_btm anchor ~16 cm from its
        # centroid, which is well within sane CAD design and not a bug.
        mesh_origin_max = 0.30  # meters; sane upper bound for any single part
        for label, mp in (
            ("left_toe mesh_pos", _mesh_pos_for_geom(lt)),
            ("left_heel mesh_pos", _mesh_pos_for_geom(lh)),
            ("right_toe mesh_pos", _mesh_pos_for_geom(rt)),
            ("right_heel mesh_pos", _mesh_pos_for_geom(rh)),
        ):
            if mp is None:
                continue
            if float(np.max(np.abs(mp))) > mesh_origin_max:
                raise ValueError(f"{label} too large (bad mesh origin?): {mp}")

    print("Foot geom validation OK")




def ensure_root_body_pose(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find the worldbody element
    worldbody = root.find("worldbody")
    if worldbody is None:
        print("No worldbody found")
        return

    # Find the waist body
    waist = worldbody.find("body[@name='waist']")
    if waist is None:
        print("No waist body found")
        return

    # Spawn pose for keyframe-less loads. Sits ~9 mm above the home keyframe's
    # settled equilibrium (pelvis_z = 0.4714 m as of the 2026-05 ankle_roll
    # model) so the robot starts above the floor without a visible drop.
    # Re-tune if a future kinematic change shifts the home equilibrium by more
    # than ~5 mm.
    waist.set("pos", "0 0 0.48")

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def merge_default_blocks(xml_file):
    """Merge multiple top-level <default> blocks into a single <default> block.

    onshape-to-robot sometimes generates multiple top-level <default> blocks
    (one from the base config and one from additional XML includes).
    MuJoCo allows this, but it's cleaner to have a single top-level default.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find all top-level <default> elements
    defaults = root.findall("default")

    if len(defaults) <= 1:
        return  # Nothing to merge

    # Keep the first default block and merge others into it
    main_default = defaults[0]

    for other_default in defaults[1:]:
        # Move all children from other default blocks into the main one
        for child in list(other_default):
            main_default.append(child)
        # Remove the now-empty default block
        root.remove(other_default)

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)
    print(f"Merged {len(defaults)} default blocks into one")


def fix_collision_default_geom(xml_file):
    """Fix collision default class to include type='sphere' for missing geometry types.

    The collision default class has `<geom group="3"/>` without a type,
    which causes Isaac Sim to default to sphere and generate warnings.
    We explicitly set type='sphere' to silence the warning.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the collision default class
    for default in root.findall(".//default[@class='collision']"):
        for geom in default.findall("geom"):
            if geom.get("type") is None:
                # Set explicit type to sphere (MuJoCo's default for collision)
                geom.set("type", "sphere")
                print("Fixed collision default geom: added type='sphere'")

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


# =============================================================================
# Collision-primitive replacement helpers
#
# The replace_*_with_{box,capsule} functions below convert mesh-based collision
# geoms into primitive (box/capsule) collision geoms. Pose, orientation, and
# size are all derived on-the-fly from the source STL using trimesh's oriented
# bounding box (OBB). No values are hardcoded — any future Onshape edit that
# changes a mesh's local origin, orientation, or shape automatically flows
# through.
#
# Convention notes:
# - MJCF quaternions are (w, x, y, z).
# - MJCF box `size` is half-extents (x, y, z) in the geom's local frame.
# - MJCF capsule `size` is (radius, half_length); the cylindrical axis is the
#   geom's local +z. Total enclosed length = 2 * (radius + half_length).
# =============================================================================


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MJCF (w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MJCF (w, x, y, z) quaternion via Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _resolve_mesh_stl_path(root: ET.Element, mesh_name: str, base_dir: Path) -> Path:
    """Look up `<mesh name=... file=...>` in `<asset>` and return absolute STL path."""
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF has no <asset> block; cannot resolve mesh paths")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "assets") if compiler is not None else "assets"
    for mesh_elem in asset.findall("mesh"):
        # Mesh `name` defaults to the file stem if omitted.
        file_attr = mesh_elem.get("file")
        if file_attr is None:
            continue
        elem_name = mesh_elem.get("name", Path(file_attr).stem)
        if elem_name == mesh_name:
            return (base_dir / meshdir / file_attr).resolve()
    raise ValueError(f"<mesh name='{mesh_name}'> not found in <asset>")


def _compute_obb(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load STL and compute oriented bounding box in mesh-local frame.

    Returns:
        center: (3,) OBB center in mesh-local coords.
        rotation: (3, 3) rotation that maps OBB-local axes to mesh-local axes.
        half_extents: (3,) box half-dimensions along OBB-local axes (x, y, z).
    """
    mesh = trimesh.load_mesh(str(stl_path))
    obb = mesh.bounding_box_oriented
    transform = np.asarray(obb.primitive.transform, dtype=float)  # 4x4
    center = transform[:3, 3]
    rotation = transform[:3, :3]
    if np.linalg.det(rotation) < 0:
        # Flip one axis to enforce a right-handed basis (preserves OBB shape).
        rotation = rotation.copy()
        rotation[:, 0] *= -1.0
    half_extents = np.asarray(obb.primitive.extents, dtype=float) / 2.0
    return center, rotation, half_extents


def _compose_in_body_frame(
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    local_pos: np.ndarray,
    local_rot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose mesh-anchor pose with mesh-local primitive pose -> body-local pose."""
    R_source = _quat_to_matrix(source_quat)
    body_pos = source_pos + R_source @ local_pos
    body_rot = R_source @ local_rot
    body_quat = _matrix_to_quat(body_rot)
    return body_pos, body_quat


def _read_geom_pose(geom: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    """Parse geom's pos/quat attributes (defaults: origin, identity)."""
    pos = np.array([float(x) for x in geom.get("pos", "0 0 0").split()])
    quat = np.array([float(x) for x in geom.get("quat", "1 0 0 0").split()])
    return pos, quat


def _format_vec(values: np.ndarray, precision: int = 9) -> str:
    return " ".join(f"{float(v):.{precision}f}" for v in values)


def _replace_collision_with_box(xml_file: str, mesh_name_filter) -> int:
    """Replace `<geom mesh=X class='collision'>` with OBB-derived box primitives.

    Args:
        mesh_name_filter: callable(mesh_name: str) -> bool selecting which meshes to replace.

    Returns: count of geoms replaced.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    base_dir = Path(xml_file).parent
    obb_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    replaced = 0
    last_size_per_mesh: Dict[str, np.ndarray] = {}
    for geom in root.findall(".//geom[@class='collision']"):
        mesh_name = geom.get("mesh") or ""
        if not mesh_name_filter(mesh_name):
            continue
        if mesh_name not in obb_cache:
            stl_path = _resolve_mesh_stl_path(root, mesh_name, base_dir)
            obb_cache[mesh_name] = _compute_obb(stl_path)
        center, rotation, half_extents = obb_cache[mesh_name]

        source_pos, source_quat = _read_geom_pose(geom)
        body_pos, body_quat = _compose_in_body_frame(source_pos, source_quat, center, rotation)

        geom.set("type", "box")
        geom.set("size", _format_vec(half_extents, precision=6))
        geom.set("pos", _format_vec(body_pos))
        geom.set("quat", _format_vec(body_quat))
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1
        last_size_per_mesh[mesh_name] = half_extents

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        for mesh_name, size in last_size_per_mesh.items():
            print(
                f"  - {mesh_name}: box half-extents = {_format_vec(size, precision=6)}"
            )
    return replaced


def replace_upper_leg_collision_with_capsule(xml_file: str) -> None:
    """Replace upper_leg mesh collision geoms with OBB-derived capsule geoms.

    The capsule's central axis is aligned with the mesh's longest OBB extent.
    Radius is the larger of the two perpendicular OBB half-extents; half_length
    is the long-axis half-extent minus the radius (capsule end caps add 2*radius
    to the total enclosed length).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    base_dir = Path(xml_file).parent
    obb_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    replaced = 0
    for geom in root.findall(".//geom[@class='collision']"):
        if geom.get("mesh") != "upper_leg":
            continue
        mesh_name = geom.get("mesh") or ""
        if mesh_name not in obb_cache:
            stl_path = _resolve_mesh_stl_path(root, mesh_name, base_dir)
            obb_cache[mesh_name] = _compute_obb(stl_path)
        center, rotation, half_extents = obb_cache[mesh_name]

        long_idx = int(np.argmax(half_extents))
        perp_indices = [i for i in range(3) if i != long_idx]
        radius = float(max(half_extents[perp_indices[0]], half_extents[perp_indices[1]]))
        half_length = float(max(half_extents[long_idx] - radius, 1e-6))

        # Build a permutation that maps capsule-local axes [perp1, perp2, long]
        # to OBB-local axes [perp_indices[0], perp_indices[1], long_idx], so the
        # capsule's local +z aligns with the OBB's longest extent.
        permute = np.zeros((3, 3))
        for new_axis, old_axis in enumerate(perp_indices + [long_idx]):
            permute[old_axis, new_axis] = 1.0
        if np.linalg.det(permute) < 0:
            permute[:, 0] *= -1.0

        capsule_rot_in_mesh = rotation @ permute
        source_pos, source_quat = _read_geom_pose(geom)
        body_pos, body_quat = _compose_in_body_frame(
            source_pos, source_quat, center, capsule_rot_in_mesh
        )

        geom.set("type", "capsule")
        geom.set("size", f"{radius:.6f} {half_length:.6f}")
        geom.set("pos", _format_vec(body_pos))
        geom.set("quat", _format_vec(body_quat))
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            f"Replaced upper_leg collision meshes with capsules: {replaced} geoms"
        )


def replace_forearm_collision_with_box(xml_file: str) -> None:
    """Replace forearm mesh collision geoms with OBB-derived box primitives."""
    replaced = _replace_collision_with_box(
        xml_file, mesh_name_filter=lambda name: name == "forearm"
    )
    if replaced > 0:
        print(f"Replaced forearm collision meshes with boxes: {replaced} geoms")


def replace_htd_45h_collision_with_box(xml_file: str) -> None:
    """Replace HTD-45H collision mesh geoms with OBB-derived box primitives."""
    replaced = _replace_collision_with_box(
        xml_file, mesh_name_filter=lambda name: name == "htd_45h_collision"
    )
    if replaced > 0:
        print(f"Replaced HTD-45H collision meshes with boxes: {replaced} geoms")


def replace_upper_leg_servo_connect_collision_with_box(xml_file: str) -> None:
    """Replace upper_leg_servo_connect collision mesh geoms with OBB-derived boxes."""
    replaced = _replace_collision_with_box(
        xml_file, mesh_name_filter=lambda name: name == "upper_leg_servo_connect"
    )
    if replaced > 0:
        print(
            f"Replaced upper_leg_servo_connect collision meshes with boxes: {replaced} geoms"
        )


def replace_foot_bottom_collision_with_box(xml_file: str) -> None:
    """Replace toe_btm*/heel_btm* collision mesh geoms with OBB-derived boxes."""
    replaced = _replace_collision_with_box(
        xml_file,
        mesh_name_filter=lambda name: name.startswith("toe_btm") or name.startswith("heel_btm"),
    )
    if replaced > 0:
        print(f"Replaced foot bottom collision meshes with boxes: {replaced} geoms")


def add_common_includes(xml_file):
    """Insert common include files as the first children of the root <mujoco>.

    This ensures the following elements appear at the top of the document in order:
      <include file="../common/scene.xml"/>
      <include file="../common/mimic_sites.xml"/>

    The function is idempotent: it will not add duplicates if they already exist.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Collect existing include file attributes to avoid duplicates
    existing_files = [inc.get("file") for inc in root.findall("include")]

    desired = [
        "../common/scene.xml",
        "../common/mimic_sites.xml",
    ]

    # Build include elements for the ones missing, preserving the required order
    to_insert = []
    for f in desired:
        if f not in existing_files:
            inc = ET.Element("include")
            inc.set("file", f)
            to_insert.append(inc)

    if not to_insert:
        return  # Nothing to do

    # Insert as the first children of <mujoco>, but after any XML declaration; index 0
    # If there is a comment or processing instruction first, we still insert at position 0
    for idx, elem in enumerate(to_insert):
        root.insert(idx, elem)

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def _read_actuator_order(actuator_order_path: Path) -> List[str]:
    """Parse actuator_order.txt; one joint name per line, blanks/comments skipped."""
    lines = actuator_order_path.read_text(encoding="utf-8").splitlines()
    order: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        order.append(stripped)
    if not order:
        raise ValueError(f"actuator_order.txt is empty: {actuator_order_path}")
    return order


def generate_actuated_joints_config(
    root: ET.Element,
    joints_details: List[Dict[str, Any]],
    *,
    actuator_order_path: Path,
    existing_joint_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Generate actuated_joints configuration for CAL (Control Abstraction Layer).

    This function produces the configuration needed by CAL to:
    1. Map between policy actions and physical joint commands
    2. Normalize joint ranges for policy training

    Policy Action Sign Convention:
    - policy_action_sign = +1.0: Joint range maps directly to action range
    - policy_action_sign = -1.0: Joint range is inverted relative to paired limb motion

    Args:
        root: XML root element
        joints_details: List of joint details from generate_robot_config
        actuator_order_path: Path to actuator_order.txt (single source of truth
            for joint order; same file consumed by reorder_actuators.py).
        existing_joint_overrides: Optional per-joint overrides loaded from existing
            mujoco_robot_config.json, keyed by joint name.

    Returns:
        List of actuated joint configs with policy_action_sign, ordered to match
        actuator_order_path.
    """
    canonical_order = _read_actuator_order(actuator_order_path)

    # Build name-to-joint map for joints with ranges (actuated joints)
    actuated_joint_map = {
        j["name"]: j for j in joints_details if j.get("name") and j.get("range")
    }

    # Cross-check: every name in actuator_order.txt must be a discovered joint
    # and vice versa. Mirrors reorder_actuators.py's strictness so the JSON
    # output stays in lockstep with the MJCF actuator order.
    missing = [n for n in canonical_order if n not in actuated_joint_map]
    extra = [n for n in actuated_joint_map.keys() if n not in canonical_order]
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing in MJCF: " + ", ".join(missing))
        if extra:
            parts.append("missing in actuator_order.txt: " + ", ".join(extra))
        raise ValueError(
            "Actuator list mismatch between MJCF and "
            f"{actuator_order_path}: {'; '.join(parts)}"
        )

    # Default policy-action sign configuration.
    # Joints listed here have local +z axes that map to the SAME world axis as
    # their left counterpart (verified by composing parent body quats), so a
    # mirror-symmetric motion needs opposite q on the right side.
    # right_ankle_roll added 2026-05-03 by analogy with right_hip_roll: both
    # share world axis ≈ (-1, 0, 0) at home pose.
    explicit_policy_action_signs: Dict[str, float] = {
        "right_hip_pitch": -1.0,
        "right_hip_roll": -1.0,
        "right_ankle_roll": -1.0,
        "right_shoulder_pitch": -1.0,
        "right_shoulder_roll": -1.0,
    }

    actuated_joints: List[Dict[str, Any]] = []
    for joint_name in canonical_order:
        joint_info = actuated_joint_map[joint_name]
        range_min, range_max = joint_info.get("range", [-1.57, 1.57])

        existing_joint = (existing_joint_overrides or {}).get(joint_name, {})
        policy_action_sign = float(
            existing_joint.get(
                "policy_action_sign",
                explicit_policy_action_signs.get(joint_name, 1.0),
            )
        )
        if abs(abs(policy_action_sign) - 1.0) > 1e-6:
            raise ValueError(
                f"Invalid policy_action_sign for '{joint_name}': {policy_action_sign} (must be +/-1.0)"
            )

        actuated_joints.append({
            "name": joint_name,
            "range": [
                int(round(math.degrees(range_min))),
                int(round(math.degrees(range_max))),
            ],
            "policy_action_sign": policy_action_sign,
            "max_velocity": float(existing_joint.get("max_velocity", 10.0)),
        })

    return actuated_joints


def _write_robot_config_file(config: Dict[str, Any], output_file: str) -> None:
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".json":
        out_path.write_text(json.dumps(config, indent=2))
        return

    import yaml

    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_robot_config(
    xml_file: str, output_file: str = "mujoco_robot_config.json"
) -> Dict[str, Any]:
    """Generate robot configuration from MuJoCo XML.

    This function extracts all robot-specific information from the XML and
    saves it to a config file. This config is then used by all Python
    training code to avoid hardcoding robot specifications.

    Extracts:
        - Joint names and their order
        - Actuator names and their order
        - Joint limits (if specified)
        - Sensor names and types
        - Floating base information
        - Body hierarchy

    Args:
        xml_file: Path to the MuJoCo XML file
        output_file: Output config file path

    Returns:
        Dictionary containing the robot configuration
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    config: Dict[str, Any] = {
        "robot_name": root.get("model", "unknown"),
        "generated_from": xml_file,
        "joint_range_unit": "deg",
    }

    # =========================================================================
    # Extract Floating Base Info (scan for freejoint)
    # =========================================================================
    floating_base: Optional[Dict[str, Any]] = None

    # Check for freejoint elements
    for freejoint in root.findall(".//freejoint"):
        if floating_base is None:
            floating_base = {
                "name": freejoint.get("name"),
                "type": "free",
            }

    # Also check joint elements with type="free"
    for joint in root.findall(".//joint"):
        joint_type = joint.get("type", "hinge")
        if joint_type == "free":
            floating_base = {
                "name": joint.get("name"),
                "type": "free",
            }

    # Find the body containing the floating base
    if floating_base:
        root_body = None
        for body in root.findall(".//body"):
            for child in body:
                if child.tag in ("freejoint", "joint"):
                    if child.get("name") == floating_base["name"]:
                        root_body = body.get("name")
                        break
                    if child.get("type") == "free":
                        root_body = body.get("name")
                        break

        floating_base["root_body"] = root_body
        floating_base["qpos_dim"] = 7  # 3 pos + 4 quat
        floating_base["qvel_dim"] = 6  # 3 linear + 3 angular

    config["root_spec"] = {
        # Body/joint info (from floating_base)
        "body": floating_base["root_body"] if floating_base else "waist",
        "joint": floating_base["name"] if floating_base else None,
        "qpos_dim": 7,
        "qvel_dim": 6,
        # Sensors (from root_imu)
        "sensors": {
            "orientation": "chest_imu_quat",  # framequat for orientation
            "gyro": "chest_imu_gyro",  # gyroscope for angular velocity
            "accel": "chest_imu_accel",  # accelerometer
            "local_linvel": "pelvis_local_linvel",  # velocimeter for local linear velocity
        },
    }
    joints: List[Dict[str, Any]] = []
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type", "hinge")

        # Skip freejoint (handled above)
        if joint_type == "free":
            continue

        joint_info = {
            "name": joint_name,
            "type": joint_type,
            "class": joint.get("class"),
        }

        # Extract range if available
        range_str = joint.get("range")
        if range_str:
            range_vals = [float(x) for x in range_str.split()]
            joint_info["range"] = range_vals

        joints.append(joint_info)

    # =========================================================================
    # Generate Actuated Joints Config for CAL (v0.11.0)
    # Single source of truth for actuator names, joints, ranges, etc.
    # =========================================================================
    existing_joint_overrides: Dict[str, Dict[str, float]] = {}
    out_path = Path(output_file)
    if out_path.exists() and out_path.suffix.lower() == ".json":
        try:
            existing_config = json.loads(out_path.read_text())
            for entry in existing_config.get("actuated_joint_specs", []):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                existing_joint_overrides[str(name)] = {
                    "policy_action_sign": float(entry.get("policy_action_sign", 1.0)),
                    "max_velocity": float(entry.get("max_velocity", 10.0)),
                }
        except Exception as exc:
            print(f"Warning: failed to read existing {output_file} overrides: {exc}")

    actuator_order_path = Path(xml_file).parent / "actuator_order.txt"
    if not actuator_order_path.exists():
        raise FileNotFoundError(
            f"actuator_order.txt missing next to MJCF: {actuator_order_path}"
        )
    actuated_joints = generate_actuated_joints_config(
        root,
        joints,
        actuator_order_path=actuator_order_path,
        existing_joint_overrides=existing_joint_overrides,
    )
    config["actuated_joint_specs"] = actuated_joints

    num_actuators = len(actuated_joints)
    num_joints = len(joints)

    # =========================================================================
    # Extract Feet Configuration (for contact detection)
    # =========================================================================
    # Feet sites (for position tracking)
    feet_sites = []
    for site in root.findall(".//site[@name]"):
        site_name = site.get("name", "")
        if "foot" in site_name.lower() and "mimic" in site_name.lower():
            feet_sites.append(site_name)

    # Feet geoms (for contact detection)
    # v0.5.0: Search for new semantic names (left_toe, left_heel, etc.)
    # These are added by add_collision_names() function
    left_toe = None
    left_heel = None
    right_toe = None
    right_heel = None

    for geom in root.findall(".//geom[@name]"):
        geom_name = geom.get("name", "")
        if geom_name == "left_toe":
            left_toe = geom_name
        elif geom_name == "left_heel":
            left_heel = geom_name
        elif geom_name == "right_toe":
            right_toe = geom_name
        elif geom_name == "right_heel":
            right_heel = geom_name

    # Feet body names (for slip/clearance computation)
    # v0.10.1: Extract foot body names from XML structure
    left_foot_body = None
    right_foot_body = None

    for body in root.findall(".//body[@name]"):
        body_name = body.get("name", "")
        if body_name == "left_foot":
            left_foot_body = body_name
        elif body_name == "right_foot":
            right_foot_body = body_name

    # Build feet config with separate left/right specs
    config["foot_specs"] = {
        "left": {
            "body": left_foot_body or "left_foot",
            "toe": left_toe or "left_toe",
            "heel": left_heel or "left_heel",
        },
        "right": {
            "body": right_foot_body or "right_foot",
            "toe": right_toe or "right_toe",
            "heel": right_heel or "right_heel",
        },
        # Contact detection parameters
        "contact_scale": 10.0,  # Scaling factor for contact force normalization
    }

    # =========================================================================
    # Observation/Feature Structure (source of truth)
    # Dimensions and indices are computed at load time in robot_config.py
    # =========================================================================
    # Observation breakdown (what policy sees - proprioception + command)
    # Order: external references → internal state → contacts → history → command
    obs_breakdown = {
        "gravity_vector": 3,         # From IMU - body orientation reference
        "base_angular_velocity": 3,  # From IMU/gyro
        "base_linear_velocity": 3,   # From IMU/velocimeter
        "joint_positions": num_actuators,
        "joint_velocities": num_actuators,
        "foot_switches": 4,          # Derived from toe/heel contact forces
        "previous_action": num_actuators,
        "velocity_command": 1,
        "padding": 1,
    }

    config["observation_breakdown"] = obs_breakdown

    # =========================================================================
    # Save to JSON/YAML by extension
    # =========================================================================
    _write_robot_config_file(config, output_file)

    print(f"Generated robot config: {output_file}")
    print(f"  Actuated joints: {num_actuators}")
    print(f"  Total joints (incl. passive): {num_joints}")
    print(f"  Observation dim: {sum(obs_breakdown.values())}")

    return config


def validate_and_update_keyframes(
    scene_xml: Path,
    keyframes_xml: Path,
    keyframe_names: Sequence[str] = ("home", "walk_start"),
    settle_steps: int = 1000,
    drift_check_steps: int = 500,
    pos_drift_tol_m: float = 0.005,
    pitch_change_tol_deg: float = 2.0,
    settled_vlin_tol: float = 1e-4,
    settled_vang_tol: float = 1e-3,
    safety_min_pelvis_z_m: float = 0.40,
    safety_max_pitch_deg: float = 15.0,
) -> bool:
    """Settle each named keyframe under PD ctrl and rewrite drifted ones.

    Procedure per keyframe:
      1. Reset model to the keyframe's qpos.
      2. Set ``data.ctrl[k] = data.qpos[act_to_qpos[k]]`` so each
         position actuator holds its keyframe-specified target.
      3. Step ``settle_steps`` sim_dt steps (default 1000 = 2 s at
         sim_dt=0.002, matching the original round-6/7 settle horizon
         documented in keyframes.xml).
      4. Run a ``drift_check_steps``-step (default 1 s) drift check;
         record residual motion in pelvis xyz and base angular
         velocity.
      5. Compare settled state to the original keyframe.  Update only
         if BOTH:
           - drift exceeds the tolerance band (>`pos_drift_tol_m` OR
             pitch change > `pitch_change_tol_deg`), AND
           - the settled state is genuinely settled (residual drift
             AND max|vlin| AND max|vang| all under their thresholds).
      6. SAFETY GUARDS — refuse to write a new keyframe if either:
           - settled pelvis_z < `safety_min_pelvis_z_m` (body fell)
           - settled |pitch| > `safety_max_pitch_deg` (body tipped)
         This catches the failure mode where the WildRobot model is
         statically unstable under position-PD alone and the "settle"
         actually captures a slow-developing fall (e.g. >5 s settle
         drifts past the standing-pose attractor and into a
         fallen-over equilibrium).
      7. After processing all keyframes, rewrite ``keyframes_xml`` in
         place with new qpos for every flagged keyframe.

    Why this matters: small CAD geometric changes (re-export, mate
    adjustments, mass tweaks) shift the standing equilibrium by
    mm-to-cm, but the keyframes encoded in ``keyframes.xml`` still
    reference the old equilibrium.  Closed-loop replays starting from
    those stale keyframes immediately try to compensate for the
    offset, which manifests downstream (e.g. as extra hip_roll firing
    in Phase 10 diagnostics).

    Why the safety guards: the WildRobot v2 model is statically
    UNSTABLE under position-PD ctrl alone — without an active balance
    controller the body slowly tips forward and falls within ~1 s of
    physics time once the small-pitch attractor is left behind.  A
    naive settle (5+ s) captures the fallen state as the new
    "equilibrium", and a second post-process round would chase the
    drift further into the failed regime.  The safety guards (settle
    horizon matched to round-6/7 + sanity-check the captured pose)
    prevent the auto-update from ever writing a fallen-over keyframe.

    Returns True if any keyframe was updated.
    """
    import mujoco  # noqa: WPS433  -- imported lazily, mirrors validate_model.py

    if not scene_xml.exists():
        print(f"  [keyframe-update] scene XML missing: {scene_xml}; skipping")
        return False
    if not keyframes_xml.exists():
        print(f"  [keyframe-update] keyframes XML missing: {keyframes_xml}; skipping")
        return False

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    act_to_qpos = np.array(
        [model.jnt_qposadr[model.actuator_trnid[k, 0]] for k in range(model.nu)]
    )

    updates: Dict[str, np.ndarray] = {}
    for kn in keyframe_names:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, kn)
        if key_id < 0:
            print(f"  [keyframe-update] {kn!r} not found in scene; skipping")
            continue
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        q0 = data.qpos.copy()
        for k in range(model.nu):
            data.ctrl[k] = data.qpos[act_to_qpos[k]]
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        pos_after_settle = data.qpos[:3].copy()
        for _ in range(drift_check_steps):
            mujoco.mj_step(model, data)
        residual_drift = float(np.linalg.norm(data.qpos[:3] - pos_after_settle))
        max_vlin = float(np.max(np.abs(data.qvel[:3])))
        max_vang = float(np.max(np.abs(data.qvel[3:6])))

        pelvis_drift = float(np.linalg.norm(data.qpos[:3] - q0[:3]))
        quat = data.qpos[3:7]
        quat0 = q0[3:7]
        angle = 2 * float(np.arccos(min(1.0, abs(float(quat[0])))))
        angle0 = 2 * float(np.arccos(min(1.0, abs(float(quat0[0])))))
        pitch_change_deg = abs(np.degrees(angle - angle0))
        new_pitch_deg = abs(np.degrees(angle))
        new_pelvis_z = float(data.qpos[2])

        # Safety guards: the WR model is statically unstable under
        # position-PD ctrl alone; refuse to write a new keyframe if
        # the captured state has fallen below a sane standing band.
        safety_failed = (
            new_pelvis_z < safety_min_pelvis_z_m
            or new_pitch_deg > safety_max_pitch_deg
        )

        # Settled check: residual drift, lin vel, and ang vel all small.
        is_settled = (
            residual_drift <= pos_drift_tol_m
            and max_vlin <= settled_vlin_tol
            and max_vang <= settled_vang_tol
        )

        # Drift check: meaningful change from the keyframe-on-disk?
        drifted = (
            pelvis_drift > pos_drift_tol_m
            or pitch_change_deg > pitch_change_tol_deg
        )

        if safety_failed:
            status = (
                f"REFUSE (safety guard: pelvis_z={new_pelvis_z:.3f}m, "
                f"pitch={new_pitch_deg:.1f}°; the model fell during settle "
                f"— keyframe NOT updated)"
            )
        elif not drifted:
            status = "OK (within tolerance)"
        elif not is_settled:
            status = (
                "SKIP (drifted but not settled: residual motion still "
                f"{residual_drift * 1000:.2f}mm / vlin {max_vlin:.5f} / "
                f"vang {max_vang:.5f} — not a stable equilibrium)"
            )
        else:
            status = "UPDATE"
            updates[kn] = data.qpos.copy()

        print(
            f"  [keyframe-update] {kn!r}: pelvis_drift_from_keyframe="
            f"{pelvis_drift * 1000:.2f}mm  pitch_change="
            f"{pitch_change_deg:+.2f}°  new_pelvis_z={new_pelvis_z:.3f}m  "
            f"new_pitch={new_pitch_deg:.1f}°  residual_drift_"
            f"{drift_check_steps}_steps={residual_drift * 1000:.3f}mm  "
            f"max|vlin|={max_vlin:.5f}m/s  max|vang|={max_vang:.5f}rad/s  "
            f"-> {status}"
        )

    if not updates:
        print("  [keyframe-update] no keyframes updated")
        return False

    text = keyframes_xml.read_text()
    n_written = 0
    for kn, new_qpos in updates.items():
        new_qpos_str = " ".join(f"{v:.6g}" for v in new_qpos)
        # Match `<key name="{kn}" qpos="..."` allowing whitespace / newlines
        # between the name attribute and qpos attribute.
        pattern = re.compile(
            rf'(<key\s+name="{re.escape(kn)}"\s+qpos=")[^"]*(")',
            re.DOTALL,
        )
        new_text, n_subs = pattern.subn(
            lambda m, s=new_qpos_str: m.group(1) + s + m.group(2),
            text,
        )
        if n_subs == 0:
            print(
                f"  [keyframe-update] WARNING: failed to find qpos= block for "
                f"{kn!r} in {keyframes_xml.name}; manual update required"
            )
        else:
            text = new_text
            n_written += 1
            print(
                f"  [keyframe-update] updated {kn!r} qpos in {keyframes_xml.name}"
            )
    if n_written:
        keyframes_xml.write_text(text)
    return n_written > 0


def main() -> None:
    # Allow optional XML path via CLI, otherwise default to v2 model.
    default_xml = Path(__file__).parent / "v2" / "wildrobot.xml"
    xml_file = sys.argv[1] if len(sys.argv) > 1 else str(default_xml)
    print(f"start post process... (xml={xml_file})")
    # add_common_includes(xml_file)
    inject_additional_xml(xml_file)
    rename_legs_and_arms_to_semantic_names(xml_file)
    add_collision_names(xml_file)
    validate_foot_geoms(xml_file)
    ensure_root_body_pose(xml_file)
    add_option(xml_file)
    merge_default_blocks(xml_file)
    fix_collision_default_geom(xml_file)
    replace_upper_leg_collision_with_capsule(xml_file)
    replace_forearm_collision_with_box(xml_file)
    replace_htd_45h_collision_with_box(xml_file)
    replace_upper_leg_servo_connect_collision_with_box(xml_file)
    replace_foot_bottom_collision_with_box(xml_file)

    # Generate robot configuration next to the MJCF.
    # Primary artifact: mujoco_robot_config.json (human-readable + strict format).
    xml_path = Path(xml_file)
    out_json = str(xml_path.with_name("mujoco_robot_config.json"))
    generate_robot_config(xml_file, out_json)

    # Optional deeper validation (requires MuJoCo + repo imports). This runs best under:
    #   uv run python assets/post_process.py assets/v1/wildrobot.xml
    try:
        import mujoco  # noqa: F401

        # Import validator from assets/ (keeps model tooling near assets).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from validate_model import Thresholds, validate_model

        validate_model(
            robot_config_yaml=Path(out_json),
            scene_xml=xml_path.with_name("scene_flat_terrain.xml"),
            robot_xml=Path(xml_file),
            thresholds=Thresholds(
                pos_tol=1e-4,
                quat_tol=1e-4,
                friction_tol=1e-4,
                size_tol=1e-4,
                mass_rel_tol=0.05,
                inertia_rel_tol=0.10,
                mesh_pos_abs_max=0.10,
                contact_penetration_tol=0.005,
            ),
        )

        # Settle each named keyframe under PD ctrl; if the equilibrium
        # has drifted, rewrite the qpos in keyframes.xml.  CAD changes
        # (re-export, mate adjustments, mass tweaks) routinely shift
        # the standing equilibrium by mm-to-cm, and stale keyframes
        # bleed into closed-loop diagnostics as spurious correction
        # behavior at iter 0.
        validate_and_update_keyframes(
            scene_xml=xml_path.with_name("scene_flat_terrain.xml"),
            keyframes_xml=xml_path.with_name("keyframes.xml"),
        )
    except Exception as exc:
        print(f"Warning: skipping validate_model.py ({exc})")

    print("Post process completed")


if __name__ == "__main__":
    main()
