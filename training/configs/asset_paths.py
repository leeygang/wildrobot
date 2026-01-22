from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResolvedAssetPaths:
    assets_root: str
    scene_xml_path: str
    robot_config_path: str
    mjcf_path: str
    model_path: str


def resolve_env_asset_paths(env: Mapping[str, Any]) -> ResolvedAssetPaths:
    """Resolve asset paths with defaults derived from assets_root."""
    assets_root = str(env.get("assets_root", "assets"))
    scene_xml_path = env.get("scene_xml_path") or f"{assets_root}/scene_flat_terrain.xml"
    robot_config_path = env.get("robot_config_path") or f"{assets_root}/robot_config.yaml"
    mjcf_path = env.get("mjcf_path") or f"{assets_root}/wildrobot.xml"
    model_path = env.get("model_path") or scene_xml_path
    return ResolvedAssetPaths(
        assets_root=assets_root,
        scene_xml_path=scene_xml_path,
        robot_config_path=robot_config_path,
        mjcf_path=mjcf_path,
        model_path=model_path,
    )
