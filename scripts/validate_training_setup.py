#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationPaths:
    scene_xml: Path
    robot_config_json: Path


def _validate_mujoco_load(scene_xml: Path) -> None:
    import mujoco

    mujoco.MjModel.from_xml_path(str(scene_xml))


def _validate_robot_config(scene_xml: Path, robot_config_json: Path) -> None:
    import mujoco
    from assets.robot_config import clear_robot_config_cache, load_robot_config

    clear_robot_config_cache()
    robot_config = load_robot_config(robot_config_json)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))

    # Foot bodies + geoms must resolve (used for contacts).
    for body_name in (robot_config.left_foot_body, robot_config.right_foot_body):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo body not found: {body_name}")

    left_toe, left_heel, right_toe, right_heel = robot_config.get_foot_geom_names()
    for geom_name in (left_toe, left_heel, right_toe, right_heel):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"MuJoCo geom not found: {geom_name}")


def main() -> int:
    # Ensure repo root is importable when running as a script.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="Validate WildRobot MuJoCo + config setup before training."
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=Path("assets/v2/scene_flat_terrain.xml"),
        help="Path to MuJoCo scene XML.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("assets/v2/mujoco_robot_config.json"),
        help="Path to assets mujoco_robot_config.json.",
    )
    args = parser.parse_args()

    paths = ValidationPaths(
        scene_xml=args.scene_xml, robot_config_json=args.robot_config
    )

    if not paths.scene_xml.exists():
        raise FileNotFoundError(paths.scene_xml)
    if not paths.robot_config_json.exists():
        raise FileNotFoundError(paths.robot_config_json)

    _validate_mujoco_load(paths.scene_xml)
    _validate_robot_config(paths.scene_xml, paths.robot_config_json)

    print("Validation OK")
    print(f"  scene_xml={paths.scene_xml}")
    print(f"  robot_config={paths.robot_config_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
