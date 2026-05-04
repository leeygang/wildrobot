"""Training-side realism profile loader hooks for v0.19.1."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from control.realism_profile import DigitalTwinRealismProfile, load_realism_profile

from .training_runtime_config import TrainingConfig


def load_training_realism_profile(
    config: TrainingConfig,
) -> Optional[DigitalTwinRealismProfile]:
    profile_path = config.env.realism_profile_path
    if not profile_path:
        return None
    raw = str(profile_path)
    candidate_paths = [
        (Path.cwd() / raw).resolve(),
        (Path(__file__).resolve().parents[2] / raw).resolve(),
        Path(raw).expanduser().resolve(),
    ]
    for path in candidate_paths:
        if path.exists():
            return load_realism_profile(path)
    raise FileNotFoundError(
        "Training realism profile path not found. Checked: "
        + ", ".join(str(p) for p in candidate_paths)
    )
