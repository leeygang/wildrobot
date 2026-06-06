"""Loader for the exported ``runtime_policy_config.json`` bundle metadata.

This is the runtime-side mirror of ``training/exports/runtime_metadata.py``.
It carries the home-base residual action contract and the gait phase clock the
``wr_obs_v8_cmd3d`` policy needs — quantities that are NOT in
``policy_spec.json``.  Loading it (rather than re-parsing the full
``training_config.yaml``) keeps the runtime decoupled from the training config
schema and the JAX/``control`` stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class ReferencePhaseTable:
    """Bundled (bin-independent) gait phase clock.

    See ``training/exports/runtime_metadata.build_reference_phase_table``.
    """

    gait_n_cycle: int
    n_steps: int
    cmd_keys: np.ndarray          # (n_bins, 3) float32
    per_bin_n_steps: np.ndarray   # (n_bins,) int32
    phase_sin: np.ndarray         # (n_steps,) float32
    phase_cos: np.ndarray         # (n_steps,) float32

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferencePhaseTable":
        cmd_keys = np.asarray(data["cmd_keys"], dtype=np.float32).reshape(-1, 3)
        per_bin = np.asarray(data["per_bin_n_steps"], dtype=np.int32).reshape(-1)
        phase_sin = np.asarray(data["phase_sin"], dtype=np.float32).reshape(-1)
        phase_cos = np.asarray(data["phase_cos"], dtype=np.float32).reshape(-1)
        if cmd_keys.shape[0] != per_bin.shape[0]:
            raise ValueError(
                "reference.cmd_keys and reference.per_bin_n_steps length mismatch: "
                f"{cmd_keys.shape[0]} != {per_bin.shape[0]}"
            )
        if phase_sin.shape[0] != phase_cos.shape[0]:
            raise ValueError("reference.phase_sin/phase_cos length mismatch")
        n_steps = int(data.get("n_steps", phase_sin.shape[0]))
        if phase_sin.shape[0] != n_steps:
            raise ValueError(
                f"reference.n_steps={n_steps} != len(phase_sin)={phase_sin.shape[0]}"
            )
        return cls(
            gait_n_cycle=int(data.get("gait_n_cycle", 0)),
            n_steps=n_steps,
            cmd_keys=cmd_keys,
            per_bin_n_steps=per_bin,
            phase_sin=phase_sin,
            phase_cos=phase_cos,
        )


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Parsed ``runtime_policy_config.json``."""

    schema_version: int
    actor_obs_layout_id: str
    action_mapping_id: str
    ctrl_dt: float
    control_hz: float
    action_delay_steps: int
    action_filter_alpha: float
    loc_ref_residual_base: str
    loc_ref_residual_mode: str
    loc_ref_residual_scale: float
    loc_ref_residual_scale_per_joint: Dict[str, float]
    residual_scale_per_actuator: List[float]
    loc_ref_command_conditioned: bool
    loc_ref_command_axes_3d: bool
    default_velocity_cmd: List[float]
    reference: ReferencePhaseTable

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimePolicyConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"runtime_policy_config.json not found: {path}")
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"runtime_policy_config.json must be a dict: {path}")
        if "reference" not in data or not isinstance(data["reference"], dict):
            raise ValueError(
                f"runtime_policy_config.json missing 'reference' block: {path}"
            )
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            actor_obs_layout_id=str(data.get("actor_obs_layout_id", "")),
            action_mapping_id=str(data.get("action_mapping_id", "")),
            ctrl_dt=float(data["ctrl_dt"]),
            control_hz=float(data.get("control_hz", 1.0 / float(data["ctrl_dt"]))),
            action_delay_steps=int(data.get("action_delay_steps", 0)),
            action_filter_alpha=float(data.get("action_filter_alpha", 0.0)),
            loc_ref_residual_base=str(data.get("loc_ref_residual_base", "q_ref")),
            loc_ref_residual_mode=str(data.get("loc_ref_residual_mode", "absolute")),
            loc_ref_residual_scale=float(data.get("loc_ref_residual_scale", 0.18)),
            loc_ref_residual_scale_per_joint=dict(
                data.get("loc_ref_residual_scale_per_joint", {}) or {}
            ),
            residual_scale_per_actuator=[
                float(v) for v in data.get("residual_scale_per_actuator", [])
            ],
            loc_ref_command_conditioned=bool(
                data.get("loc_ref_command_conditioned", False)
            ),
            loc_ref_command_axes_3d=bool(data.get("loc_ref_command_axes_3d", False)),
            default_velocity_cmd=[
                float(v) for v in data.get("default_velocity_cmd", [0.0, 0.0, 0.0])
            ],
            reference=ReferencePhaseTable.from_dict(data["reference"]),
        )
