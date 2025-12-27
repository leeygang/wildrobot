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
    # Find left_foot body and name collision geoms
    # foot_btm_front = toe (front of foot)
    # foot_btm_back = heel (back of foot)
    for body in root.findall(".//body[@name='left_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "left_toe")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "left_heel")

    # Same for right foot
    for body in root.findall(".//body[@name='right_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "right_toe")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "right_heel")

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


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
    # Extract Actuators (these define the action space)
    # =========================================================================
    actuator_section = root.find("actuator")
    actuators: List[Dict[str, Any]] = []
    if actuator_section is not None:
        for actuator in actuator_section:
            act_info = {
                "name": actuator.get("name"),
                "joint": actuator.get("joint"),
                "type": actuator.tag,  # e.g., "position", "motor", etc.
                "class": actuator.get("class"),
            }
            actuators.append(act_info)

    config["actuators"] = {
        "names": [a["name"] for a in actuators],
        "joints": [a["joint"] for a in actuators],
        "count": len(actuators),
        "details": actuators,
    }

    # =========================================================================
    # Extract Joints (scan all bodies for joint definitions)
    # =========================================================================
    joints: List[Dict[str, Any]] = []
    floating_base: Optional[Dict[str, Any]] = None

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type", "hinge")  # Default is hinge

        # Check if this is a freejoint (floating base)
        if joint_type == "free" or joint.tag == "freejoint":
            floating_base = {
                "name": joint_name,
                "type": "free",
            }
            continue

        joint_info = {
            "name": joint_name,
            "type": joint_type,
            "axis": joint.get("axis", "0 0 1"),
            "class": joint.get("class"),
        }

        # Extract range if available
        range_str = joint.get("range")
        if range_str:
            range_vals = [float(x) for x in range_str.split()]
            joint_info["range"] = range_vals

        joints.append(joint_info)

    # Also check for freejoint elements (MuJoCo shorthand)
    for freejoint in root.findall(".//freejoint"):
        if floating_base is None:
            floating_base = {
                "name": freejoint.get("name"),
                "type": "free",
            }

    config["joints"] = {
        "names": [j["name"] for j in joints],
        "count": len(joints),
        "details": joints,
    }

    # =========================================================================
    # Extract Floating Base Info
    # =========================================================================
    if floating_base:
        # Find the body containing the floating base
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

    config["floating_base"] = floating_base

    # =========================================================================
    # Extract Sensors
    # =========================================================================
    sensor_section = root.find("sensor")
    sensors: Dict[str, List[Dict[str, Any]]] = {}

    if sensor_section is not None:
        for sensor in sensor_section:
            sensor_type = sensor.tag
            sensor_info = {
                "name": sensor.get("name"),
                "site": sensor.get("site"),
                "objname": sensor.get("objname"),
                "objtype": sensor.get("objtype"),
            }

            if sensor_type not in sensors:
                sensors[sensor_type] = []
            sensors[sensor_type].append(sensor_info)

    config["sensors"] = sensors

    # =========================================================================
    # Compute Dimensions
    # =========================================================================
    num_actuators = len(actuators)
    num_joints = len(joints)

    # Observation dimension breakdown (matching wildrobot_env.py)
    # This should match the actual observation construction
    obs_breakdown = {
        "gravity_vector": 3,
        "base_angular_velocity": 3,
        "base_linear_velocity": 3,
        "joint_positions": num_actuators,
        "joint_velocities": num_actuators,
        "previous_action": num_actuators,
        "velocity_command": 1,
        "padding": 1,  # For alignment
    }
    obs_dim = sum(obs_breakdown.values())

    # AMP feature dimension breakdown (matching amp_features.py)
    amp_breakdown = {
        "joint_positions": num_actuators,
        "joint_velocities": num_actuators,
        "root_linear_velocity": 3,
        "root_angular_velocity": 3,
        "root_height": 1,
        "foot_contacts": 4,
    }
    amp_feature_dim = sum(amp_breakdown.values())

    config["dimensions"] = {
        "action_dim": num_actuators,
        "observation_dim": obs_dim,
        "amp_feature_dim": amp_feature_dim,
        "num_actuated_joints": num_actuators,
        "num_total_joints": num_joints,
        "observation_breakdown": obs_breakdown,
        "amp_feature_breakdown": amp_breakdown,
    }

    # =========================================================================
    # Observation to AMP Feature Mapping (indices into observation vector)
    # =========================================================================
    # These indices allow AMP feature extraction from observations
    obs_idx = 0
    obs_indices = {}
    for key, size in obs_breakdown.items():
        obs_indices[key] = {"start": obs_idx, "end": obs_idx + size}
        obs_idx += size

    config["observation_indices"] = obs_indices

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

    # Build feet config with explicit keys (v0.5.0)
    config["feet"] = {
        "sites": feet_sites,
        # Foot body names (for position/velocity tracking)
        "left_foot_body": left_foot_body or "left_foot",
        "right_foot_body": right_foot_body or "right_foot",
        # Explicit semantic keys - no array order dependency
        "left_toe": left_toe or "left_toe",
        "left_heel": left_heel or "left_heel",
        "right_toe": right_toe or "right_toe",
        "right_heel": right_heel or "right_heel",
        # Legacy array format (for backward compatibility)
        "left_geoms": [g for g in [left_heel, left_toe] if g],
        "right_geoms": [g for g in [right_toe, right_heel] if g],
        "all_geoms": [g for g in [left_toe, left_heel, right_toe, right_heel] if g],
    }

    # =========================================================================
    # Save to YAML
    # =========================================================================
    import yaml

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated robot config: {output_file}")
    print(f"  Actuators: {num_actuators}")
    print(f"  Joints: {num_joints}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  AMP feature dim: {amp_feature_dim}")

    return config


def main() -> None:
    xml_file = "wildrobot.xml"
    print("start post process...")
    # add_common_includes(xml_file)
    add_collision_names(xml_file)
    ensure_root_body_pose(xml_file)
    add_option(xml_file)
    merge_default_blocks(xml_file)
    fix_collision_default_geom(xml_file)
    add_mimic_bodies(xml_file)  # Add dummy bodies for GMR compatibility

    # Generate robot configuration from XML
    # This config is used by all Python training code
    generate_robot_config(xml_file, "robot_config.yaml")

    print("Post process completed")


if __name__ == "__main__":
    main()
