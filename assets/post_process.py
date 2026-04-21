import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def _geom_info(name: str):
        gid = _id(mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Compiled model missing geom '{name}'.")
        return {
            "gid": gid,
            "body": int(model.geom_bodyid[gid]),
            "pos": model.geom_pos[gid].copy(),
            "quat": model.geom_quat[gid].copy(),
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

    tol = 1e-4
    for key, a, b in (
        ("toe geom_pos", lt["pos"], rt["pos"]),
        ("heel geom_pos", lh["pos"], rh["pos"]),
        ("toe geom_quat", lt["quat"], rt["quat"]),
        ("heel geom_quat", lh["quat"], rh["quat"]),
        ("toe geom_size", lt["size"], rt["size"]),
        ("heel geom_size", lh["size"], rh["size"]),
        ("toe geom_friction", lt["friction"], rt["friction"]),
        ("heel geom_friction", lh["friction"], rh["friction"]),
    ):
        if not np.allclose(a, b, atol=tol, rtol=0.0):
            raise ValueError(f"Foot symmetry check failed for {key}: {a} vs {b}")

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

        mesh_origin_max = 0.10  # meters; toe/heel offsets are ~0.06
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

    # Ensure the root body sits at the expected pose now that the floating base wrapper is gone.
    waist.set("pos", "0 0 0.5")

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


def replace_upper_leg_collision_with_capsule(
    xml_file: str,
    radius: float = 0.02,
    half_length: float = 0.060188,
) -> None:
    """Replace upper_leg mesh collision geoms with capsule geoms.

    This targets only collision geoms where mesh="upper_leg" and keeps
    visual meshes untouched. The function is idempotent.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    replaced = 0
    size_str = f"{radius:.6f} {half_length:.6f}"
    # Match the resolved mesh frame for `upper_leg` so capsule overlaps visual.
    # Values come from compiled MuJoCo geom pose for the upper_leg mesh.
    pos_str = "0.000000029 0.099411835 0.026699927"
    quat_str = "0.5 0.5 -0.5 0.5"

    for geom in root.findall(".//geom[@class='collision']"):
        if geom.get("mesh") != "upper_leg":
            continue
        geom.set("type", "capsule")
        geom.set("size", size_str)
        geom.set("pos", pos_str)
        geom.set("quat", quat_str)
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            "Replaced upper_leg collision meshes with capsules: "
            f"{replaced} geoms (pos={pos_str}, size={size_str})"
        )


def replace_forearm_collision_with_box(xml_file: str) -> None:
    """Replace forearm mesh collision geoms with aligned box geoms.

    The forearm mesh is used twice per arm with different local frames.
    We map each source frame to a resolved primitive frame to keep overlap.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Resolved box half extents from compiled forearm mesh geom_size.
    size_str = "0.021754 0.042512 0.064691"
    pose_a = {
        "pos": "-0.000012244 0.082012300 -0.046051151",
        "quat": "0.509179290 0.490187470 -0.490663870 0.509609330",
    }
    pose_b = {
        "pos": "-0.000038559 0.081967005 -0.007357287",
        "quat": "0.490069430 0.510180960 0.509751350 -0.489592590",
    }

    replaced = 0
    for geom in root.findall(".//geom[@class='collision']"):
        if geom.get("mesh") != "forearm":
            continue

        # Two forearm collision entries are distinguished by original z sign.
        pos_vals = [float(x) for x in geom.get("pos", "0 0 0").split()]
        target = pose_a if pos_vals[2] >= 0.0 else pose_b

        geom.set("type", "box")
        geom.set("size", size_str)
        geom.set("pos", target["pos"])
        geom.set("quat", target["quat"])
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            "Replaced forearm collision meshes with boxes: "
            f"{replaced} geoms (size={size_str})"
        )


def replace_htd_45h_collision_with_box(xml_file: str) -> None:
    """Replace HTD-45H collision mesh geoms with aligned box geoms."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size_str = "0.010095 0.026704 0.030001"
    # Keep original export orientation (quat), but use resolved mesh-centered
    # positions so the primitive does not shift after removing mesh offsets.
    left_pose = {
        "pos": "-0.001295800 -0.008225190 -0.036494360",
        "quat": "0.000116219 -0.707107 0.000116219 0.707107",
    }
    right_pose = {
        "pos": "0.000495800 -0.007888980 -0.036511340",
        "quat": "0.000116219 -0.707107 -0.000116219 -0.707107",
    }

    replaced = 0
    for geom in root.findall(".//geom[@class='collision']"):
        if geom.get("mesh") != "htd_45h_collision":
            continue

        # Source export uses +x for left, -x for right for this mesh geom.
        pos_vals = [float(x) for x in geom.get("pos", "0 0 0").split()]
        target = left_pose if pos_vals[0] >= 0.0 else right_pose

        geom.set("type", "box")
        geom.set("size", size_str)
        geom.set("pos", target["pos"])
        geom.set("quat", target["quat"])
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            "Replaced HTD-45H collision meshes with boxes: "
            f"{replaced} geoms (size={size_str})"
        )


def replace_upper_leg_servo_connect_collision_with_box(xml_file: str) -> None:
    """Replace upper_leg_servo_connect collision mesh geoms with aligned box geoms."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size_str = "0.022780 0.033799 0.030266"
    pose = {
        "pos": "0.000000001 0.020521142 0.026090017",
        "quat": "-0.004637314 0.999989248 0.000000000 0.000000000",
    }

    replaced = 0
    for geom in root.findall(".//geom[@class='collision']"):
        if geom.get("mesh") != "upper_leg_servo_connect":
            continue

        geom.set("type", "box")
        geom.set("size", size_str)
        geom.set("pos", pose["pos"])
        geom.set("quat", pose["quat"])
        geom.attrib.pop("mesh", None)
        geom.attrib.pop("material", None)
        replaced += 1

    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            "Replaced upper_leg_servo_connect collision meshes with boxes: "
            f"{replaced} geoms (size={size_str})"
        )


def replace_foot_bottom_collision_with_box(xml_file: str) -> None:
    """Replace toe/heel bottom collision mesh geoms with aligned box geoms."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    toe_size_str = "0.002040 0.037100 0.045000"
    heel_size_str = "0.002062 0.021100 0.045000"

    toe_pose = {
        "pos": "0.056816350 0.060460060 0.026800000",
        "quat": "-0.707107 0.000000 0.000000 0.707107",
    }
    heel_pose = {
        "pos": "-0.032749720 0.060437950 0.026800000",
        "quat": "-0.707107 0.000000 0.000000 0.707107",
    }

    replaced_toe = 0
    replaced_heel = 0
    for geom in root.findall(".//geom[@class='collision']"):
        mesh_name = geom.get("mesh") or ""
        if mesh_name.startswith("toe_btm"):
            geom.set("type", "box")
            geom.set("size", toe_size_str)
            geom.set("pos", toe_pose["pos"])
            geom.set("quat", toe_pose["quat"])
            geom.attrib.pop("mesh", None)
            geom.attrib.pop("material", None)
            replaced_toe += 1
            continue

        if mesh_name.startswith("heel_btm"):
            geom.set("type", "box")
            geom.set("size", heel_size_str)
            geom.set("pos", heel_pose["pos"])
            geom.set("quat", heel_pose["quat"])
            geom.attrib.pop("mesh", None)
            geom.attrib.pop("material", None)
            replaced_heel += 1

    replaced = replaced_toe + replaced_heel
    if replaced > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(
            "Replaced foot bottom collision meshes with boxes: "
            f"toe={replaced_toe}, heel={replaced_heel} "
            f"(toe_size={toe_size_str}, heel_size={heel_size_str})"
        )


def add_mimic_bodies(xml_file):
    """Add dummy bodies at site positions for sites ending with '_mimic'.

    GMR (General Motion Retargeting) requires body elements to target for IK.
    WildRobot uses sites (e.g., left_knee_mimic, right_foot_mimic) for sensing,
    but GMR can't target sites directly.

    This function creates lightweight dummy bodies at the same position/orientation
    as each _mimic site, using the same name as the site.

    For example:
        - Site 'left_knee_mimic' -> Body 'left_knee_mimic'
        - Site 'pelvis_mimic' -> Body 'pelvis_mimic'
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Collect existing body names to avoid duplicates
    existing_bodies = {body.get("name") for body in root.findall(".//body")}

    # Find all sites ending with '_mimic'
    mimic_sites = root.findall(".//site[@name]")
    added_count = 0

    for site in mimic_sites:
        site_name = site.get("name")
        if not site_name or not site_name.endswith("_mimic"):
            continue

        # Use same name as site for the body
        body_name = site_name

        # Skip if body already exists
        if body_name in existing_bodies:
            continue

        # Get site position and orientation
        pos = site.get("pos", "0 0 0")
        quat = site.get("quat", "1 0 0 0")

        # Create dummy body element
        dummy_body = ET.Element("body")
        dummy_body.set("name", body_name)
        dummy_body.set("pos", pos)
        dummy_body.set("quat", quat)

        # Add minimal inertial to make it a valid body (massless marker)
        inertial = ET.SubElement(dummy_body, "inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", "0.001")
        inertial.set("diaginertia", "0.000001 0.000001 0.000001")

        # Insert dummy body as sibling after the site (in parent body)
        parent = None
        for potential_parent in root.iter():
            if site in potential_parent:
                parent = potential_parent
                break

        if parent is not None:
            # Find site index and insert body after it
            site_index = list(parent).index(site)
            parent.insert(site_index + 1, dummy_body)
            existing_bodies.add(body_name)
            added_count += 1
            print(f"Added dummy body '{body_name}' at site '{site_name}'")

    if added_count > 0:
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_file)
        print(f"Total: Added {added_count} dummy bodies for GMR compatibility")


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


def generate_actuated_joints_config(
    root: ET.Element,
    joints_details: List[Dict[str, Any]],
    existing_joint_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Generate actuated_joints configuration for CAL (Control Abstraction Layer).

    This function produces the configuration needed by CAL to:
    1. Map between policy actions and physical joint commands
    2. Normalize joint ranges for policy training

    v0.11.0: Added for Control Abstraction Layer

    Policy Action Sign Convention:
    - policy_action_sign = +1.0: Joint range maps directly to action range
    - policy_action_sign = -1.0: Joint range is inverted relative to paired limb motion

    Args:
        root: XML root element
        joints_details: List of joint details from generate_robot_config
        existing_joint_overrides: Optional per-joint overrides loaded from existing
            mujoco_robot_config.json, keyed by joint name.

    Returns:
        List of actuated joint configs with policy_action_sign
    """
    actuated_joints: List[Dict[str, Any]] = []

    # Build name-to-joint map for joints with ranges (actuated joints)
    actuated_joint_map = {
        j["name"]: j for j in joints_details if j.get("name") and j.get("range")
    }

    # Default policy-action sign configuration.
    explicit_policy_action_signs: Dict[str, float] = {
        "right_hip_pitch": -1.0,
        "right_hip_roll": -1.0,
        "right_shoulder_pitch": -1.0,
        "right_shoulder_roll": -1.0,
    }

    # Process each actuated joint
    for joint_name, joint_info in actuated_joint_map.items():
        joint_range = joint_info.get("range", [-1.57, 1.57])
        range_min, range_max = joint_range

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

        config = {
            "name": joint_name,
            "range": [
                int(round(math.degrees(range_min))),
                int(round(math.degrees(range_max))),
            ],
            "policy_action_sign": policy_action_sign,
            "max_velocity": float(existing_joint.get("max_velocity", 10.0)),
        }
        actuated_joints.append(config)

    # Sort by paired left/right order.
    # Unpaired joints keep their original discovery order.
    joint_order = [
        "left_hip_pitch",
        "right_hip_pitch",
        "left_hip_roll",
        "right_hip_roll",
        "left_knee_pitch",
        "right_knee_pitch",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
        "left_shoulder_roll",
        "right_shoulder_roll",
        "left_elbow_pitch",
        "right_elbow_pitch",
        "left_wrist_yaw",
        "right_wrist_yaw",
        "left_wrist_pitch",
        "right_wrist_pitch",
    ]
    original_order = {name: idx for idx, name in enumerate(actuated_joint_map.keys())}

    actuated_joints.sort(
        key=lambda x: (
            joint_order.index(x["name"])
            if x["name"] in joint_order
            else 1000 + original_order.get(x["name"], 999)
        )
    )

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

    actuated_joints = generate_actuated_joints_config(
        root,
        joints,
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


def main() -> None:
    # Allow optional XML path via CLI, otherwise default to v1 model.
    default_xml = Path(__file__).parent / "v1" / "wildrobot.xml"
    xml_file = sys.argv[1] if len(sys.argv) > 1 else str(default_xml)
    print(f"start post process... (xml={xml_file})")
    # add_common_includes(xml_file)
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
    add_mimic_bodies(xml_file)  # Add dummy bodies for GMR compatibility

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
    except Exception as exc:
        print(f"Warning: skipping validate_model.py ({exc})")

    print("Post process completed")


if __name__ == "__main__":
    main()
