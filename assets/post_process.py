import sys
import xml.etree.ElementTree as ET
from pathlib import Path


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
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find left_foot body and its foot_btm_front collision geom
    for body in root.findall(".//body[@name='left_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "left_foot_btm_front")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "left_foot_btm_back")

    # Same for right foot
    for body in root.findall(".//body[@name='right_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "right_foot_btm_front")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "right_foot_btm_back")

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
    print("Post process completed")


if __name__ == "__main__":
    main()
