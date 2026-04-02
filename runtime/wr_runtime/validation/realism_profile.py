"""Runtime helpers for loading and validating digital-twin realism profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from control.realism_profile import DigitalTwinRealismProfile, load_realism_profile

from configs import WildRobotRuntimeConfig


def load_runtime_realism_profile(
    cfg: WildRobotRuntimeConfig,
) -> Optional[DigitalTwinRealismProfile]:
    """Load realism profile if runtime config declares one.

    This is a v0.19.1 integration hook only: the profile is validated and logged,
    while full sim/hardware application is deferred to follow-up milestones.
    """
    if not cfg.realism_profile_path:
        return None
    raw = str(cfg.realism_profile_path)
    repo_root = Path(__file__).resolve().parents[3]
    raw_path = Path(raw)
    suffix_from_assets: Optional[Path] = None
    if "assets" in raw_path.parts:
        parts = list(raw_path.parts)
        idx = parts.index("assets")
        suffix_from_assets = Path(*parts[idx:])
    candidate_paths = [
        Path(cfg.resolve_path(raw)),
        (repo_root / raw).resolve(),
        (Path.cwd() / raw).resolve(),
        (repo_root / suffix_from_assets).resolve() if suffix_from_assets is not None else None,
        Path(raw).expanduser().resolve(),
    ]
    for path in candidate_paths:
        if path is None:
            continue
        if path.exists():
            return load_realism_profile(path)
    raise FileNotFoundError(
        "Realism profile path not found. Checked: "
        + ", ".join(str(p) for p in candidate_paths)
    )
