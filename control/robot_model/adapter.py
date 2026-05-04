"""WildRobot robot-model adapter scaffold for planner/controller integration.

This module provides a minimal compatibility contract between current WildRobot
assets (`assets/v2/*.xml`, `mujoco_robot_config.json`) and future control-stack
model requirements such as frame/contact conventions and optional controller
model artifacts.

It intentionally does not implement full model conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from assets.robot_config import RobotConfig


@dataclass(frozen=True)
class ModelArtifactStatus:
    """Compatibility status for required control-model artifacts."""

    has_mjcf: bool
    has_robot_config: bool
    has_urdf: bool
    has_frame_convention_note: bool
    has_contact_geometry_note: bool


@dataclass(frozen=True)
class WildRobotModelCompatibilityReport:
    """Compatibility report used by v0.17.2 adoption scaffolding."""

    robot_name: str
    source_mjcf: str
    source_robot_config: str
    missing_artifacts: tuple[str, ...]
    notes: tuple[str, ...]


def evaluate_model_artifacts(
    *,
    assets_root: Path,
    robot_cfg: RobotConfig,
) -> ModelArtifactStatus:
    """Evaluate presence of model artifacts useful for controller integration."""
    urdf_candidates: Iterable[Path] = (
        assets_root / "wildrobot.urdf",
        assets_root / "wildrobot_control.urdf",
    )
    has_urdf = any(path.exists() for path in urdf_candidates)

    # v0.17.2 note files (lightweight placeholders; can be replaced later).
    has_frame_note = (assets_root / "FRAME_CONVENTIONS.md").exists()
    has_contact_note = (assets_root / "CONTACT_GEOMETRY.md").exists()

    return ModelArtifactStatus(
        has_mjcf=(assets_root / "wildrobot.xml").exists(),
        has_robot_config=(assets_root / "mujoco_robot_config.json").exists(),
        has_urdf=has_urdf,
        has_frame_convention_note=has_frame_note,
        has_contact_geometry_note=has_contact_note,
    )


def build_compatibility_report(
    *,
    assets_root: str,
    robot_cfg: RobotConfig,
) -> WildRobotModelCompatibilityReport:
    """Build a concrete compatibility report for planning/integration scope."""
    root = Path(assets_root)
    status = evaluate_model_artifacts(assets_root=root, robot_cfg=robot_cfg)
    missing: list[str] = []
    notes: list[str] = []

    if not status.has_mjcf:
        missing.append(str(root / "wildrobot.xml"))
    if not status.has_robot_config:
        missing.append(str(root / "mujoco_robot_config.json"))
    if not status.has_urdf:
        missing.append(f"URDF control model export (e.g., {root / 'wildrobot.urdf'})")
        notes.append("URDF export is only needed if a future controller stack requires it.")
    if not status.has_frame_convention_note:
        missing.append(str(root / "FRAME_CONVENTIONS.md"))
        notes.append("Frame/sign conventions must be explicitly documented for controller integration.")
    if not status.has_contact_geometry_note:
        missing.append(str(root / "CONTACT_GEOMETRY.md"))
        notes.append("Controller-facing foot support/contact geometry note is missing.")

    return WildRobotModelCompatibilityReport(
        robot_name=robot_cfg.robot_name,
        source_mjcf=str(root / "wildrobot.xml"),
        source_robot_config=str(root / "mujoco_robot_config.json"),
        missing_artifacts=tuple(missing),
        notes=tuple(notes),
    )
