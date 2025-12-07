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
    print("Post process completed")


if __name__ == "__main__":
    main()
