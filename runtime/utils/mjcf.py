from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class MjcfModelInfo:
    actuator_names: List[str]
    joint_limits: Dict[str, Tuple[float, float]]


def _parse_range(range_str: str) -> Tuple[float, float]:
    parts = (range_str or "").split()
    if len(parts) != 2:
        raise ValueError(f"Invalid joint range: {range_str!r}")
    return float(parts[0]), float(parts[1])


def load_mjcf_model_info(mjcf_path: str | Path) -> MjcfModelInfo:
    """Load the actuator order and joint limits from a MuJoCo MJCF XML.

    We treat `<actuator><position name=...>` order as the action order.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"MJCF missing <actuator>: {mjcf_path}")

    actuator_names: List[str] = []
    for pos_act in actuator.findall("position"):
        name = pos_act.get("name")
        if not name:
            continue
        actuator_names.append(name)

    if not actuator_names:
        raise ValueError(f"No <actuator><position name=...> found in: {mjcf_path}")

    joint_limits: Dict[str, Tuple[float, float]] = {}
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if not jname:
            continue
        if joint.get("type") != "hinge":
            continue
        rng = joint.get("range")
        if rng:
            joint_limits[jname] = _parse_range(rng)

    return MjcfModelInfo(actuator_names=actuator_names, joint_limits=joint_limits)
