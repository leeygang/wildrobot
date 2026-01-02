import json
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
    for clarity and to match AMP feature naming convention.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # v0.11.x: Some newer exports use generic foot body names ("foot", "foot_2").
    # Normalize to stable names used by training + robot_config.yaml.
    body_names = {b.get("name") for b in root.findall(".//body") if b.get("name")}
    if "left_foot" not in body_names and "foot" in body_names:
        for body in root.findall(".//body[@name='foot']"):
            body.set("name", "left_foot")
    if "right_foot" not in body_names and "foot_2" in body_names:
        for body in root.findall(".//body[@name='foot_2']"):
            body.set("name", "right_foot")

    # Map mesh names to semantic collision geom names per foot side.
    # Use the simplified shared meshes from newer exports: toe_btm / heel_btm
    mesh_to_name = {
        "toe_btm": {"left": "left_toe", "right": "right_toe"},
        "heel_btm": {"left": "left_heel", "right": "right_heel"},
    }

    for side in ("left", "right"):
        body_name = f"{side}_foot"
        for mesh, names in mesh_to_name.items():
            desired = names[side]
            xpath = f".//body[@name='{body_name}']/geom[@mesh='{mesh}'][@class='collision']"
            for geom in root.findall(xpath):
                geom.set("name", desired)

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
) -> List[Dict[str, Any]]:
    """Generate actuated_joints configuration for CAL (Control Abstraction Layer).

    This function produces the configuration needed by CAL to:
    1. Map between policy actions and physical joint commands
    2. Handle joint symmetry (left/right mirroring)
    3. Normalize joint ranges for policy training

    v0.11.0: Added for Control Abstraction Layer

    Mirror Sign Convention:
    - mirror_sign = +1.0: Joint range maps directly to action range
        Example: left_hip_pitch [-0.087, 1.571] → action 0 = center
    - mirror_sign = -1.0: Joint range is inverted relative to its pair
        Example: right_hip_pitch [-1.571, 0.087] is flipped vs left
        When policy outputs same action for L/R, we want symmetric motion

    Args:
        root: XML root element
        joints_details: List of joint details from generate_robot_config

    Returns:
        List of actuated joint configs with mirror_sign and symmetry_pair
    """
    actuated_joints: List[Dict[str, Any]] = []

    # Build name-to-joint map for joints with ranges (actuated joints)
    actuated_joint_map = {
        j["name"]: j for j in joints_details if j.get("name") and j.get("range")
    }

    # Define symmetry pairs (left ↔ right)
    symmetry_pairs = {
        "left_hip_pitch": "right_hip_pitch",
        "left_hip_roll": "right_hip_roll",
        "left_knee_pitch": "right_knee_pitch",
        "left_ankle_pitch": "right_ankle_pitch",
        "right_hip_pitch": "left_hip_pitch",
        "right_hip_roll": "left_hip_roll",
        "right_knee_pitch": "left_knee_pitch",
        "right_ankle_pitch": "left_ankle_pitch",
    }

    # Process each actuated joint
    for joint_name, joint_info in actuated_joint_map.items():
        joint_range = joint_info.get("range", [-1.57, 1.57])
        range_min, range_max = joint_range

        # Determine mirror_sign by comparing with symmetry pair
        symmetry_pair = symmetry_pairs.get(joint_name)
        mirror_sign = 1.0

        if symmetry_pair and symmetry_pair in actuated_joint_map:
            pair_info = actuated_joint_map[symmetry_pair]
            pair_range = pair_info.get("range", [-1.57, 1.57])
            pair_min, pair_max = pair_range

            # Explicit range checks for symmetry
            tol = 1e-3
            same_range = abs(range_min - pair_min) < tol and abs(range_max - pair_max) < tol
            mirrored = (
                abs(range_min + pair_max) < tol and abs(range_max + pair_min) < tol
            )

            if not (mirrored or same_range):
                raise ValueError(
                    "Unexpected joint range mismatch for symmetry pair "
                    f"'{joint_name}' ↔ '{symmetry_pair}': "
                    f"{range_min:.6f},{range_max:.6f} vs {pair_min:.6f},{pair_max:.6f}. "
                    "Update mirror detection or config."
                )

            if mirrored and not same_range:
                # Convention: joints starting with "right_" get -1.0 if inverted
                if joint_name.startswith("right_"):
                    mirror_sign = -1.0
                elif joint_name.startswith("left_"):
                    mirror_sign = 1.0
                else:
                    raise ValueError(
                        "Mirrored joint range requires explicit left/right naming for "
                        f"'{joint_name}' ↔ '{symmetry_pair}'."
                    )

        config = {
            "name": joint_name,
            "type": "position",  # All actuators are position-controlled
            "range": [round(range_min, 6), round(range_max, 6)],
            "symmetry_pair": symmetry_pair,
            "mirror_sign": mirror_sign,
            "max_velocity": 10.0,  # Conservative default for servos (rad/s)
        }
        actuated_joints.append(config)

    # Sort by actuator order (left leg first, then right leg, by joint order)
    joint_order = [
        "left_hip_pitch",
        "left_hip_roll",
        "left_knee_pitch",
        "left_ankle_pitch",
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee_pitch",
        "right_ankle_pitch",
    ]
    actuated_joints.sort(
        key=lambda x: joint_order.index(x["name"]) if x["name"] in joint_order else 999
    )

    return actuated_joints


def generate_robot_config(
    xml_file: str, output_file: str = "robot_config.yaml"
) -> Dict[str, Any]:
    """Generate robot configuration from MuJoCo XML.

    This function extracts all robot-specific information from the XML and
    saves it to a YAML config file. This config is then used by all Python
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
        output_file: Output YAML file path

    Returns:
        Dictionary containing the robot configuration
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    config: Dict[str, Any] = {
        "robot_name": root.get("model", "unknown"),
        "generated_from": xml_file,
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
    actuated_joints = generate_actuated_joints_config(root, joints)
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

    # AMP feature breakdown (what discriminator sees - motion signature)
    # Order: pose → root motion → contacts (for gait pattern matching)
    # Note: foot_contacts not in obs because policy infers contact from physics;
    # AMP needs explicit contacts to distinguish gait patterns
    amp_breakdown = {
        "joint_positions": num_actuators,
        "joint_velocities": num_actuators,
        "root_linear_velocity": 3,
        "root_angular_velocity": 3,
        "root_height": 1,
        "foot_contacts": 4,  # [left_toe, left_heel, right_toe, right_heel]
    }

    config["observation_breakdown"] = obs_breakdown
    config["amp_feature_breakdown"] = amp_breakdown

    # =========================================================================
    # Save to YAML
    # =========================================================================
    import yaml

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated robot config: {output_file}")
    print(f"  Actuated joints: {num_actuators}")
    print(f"  Total joints (incl. passive): {num_joints}")
    print(f"  Observation dim: {sum(obs_breakdown.values())}")
    print(f"  AMP feature dim: {sum(amp_breakdown.values())}")

    return config


def main() -> None:
    # Allow optional XML path via CLI, otherwise default to the XML next to this script.
    default_xml = Path(__file__).parent / "wildrobot.xml"
    xml_file = sys.argv[1] if len(sys.argv) > 1 else str(default_xml)
    print(f"start post process... (xml={xml_file})")
    # add_common_includes(xml_file)
    add_collision_names(xml_file)
    validate_foot_geoms(xml_file)
    ensure_root_body_pose(xml_file)
    add_option(xml_file)
    merge_default_blocks(xml_file)
    fix_collision_default_geom(xml_file)
    add_mimic_bodies(xml_file)  # Add dummy bodies for GMR compatibility

    # Generate robot configuration from XML
    # This config is used by all Python training code
    generate_robot_config(xml_file, "robot_config.yaml")

    # Optional deeper validation (requires MuJoCo + repo imports). This runs best under:
    #   uv run python assets/post_process.py assets/wildrobot.xml
    try:
        import mujoco  # noqa: F401

        # Import validator from assets/ (keeps model tooling near assets).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from validate_model import Thresholds, validate_model

        validate_model(
            robot_config_yaml=Path(__file__).parent / "robot_config.yaml",
            scene_xml=Path(__file__).parent / "scene_flat_terrain.xml",
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
