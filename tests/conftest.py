from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNTIME_ROOT = PROJECT_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


# ---------------------------------------------------------------------------
# Shared builders for the v0.21.0 hardware-runtime tests (wr_obs_v8_cmd3d).
# ---------------------------------------------------------------------------
_ROBOT_CONFIG = PROJECT_ROOT / "assets" / "v2" / "mujoco_robot_config.json"

# smoke9 per-joint residual scales (training_config.yaml
# loc_ref_residual_scale_per_joint); legs only, everything else uses the
# scalar fallback below.
SMOKE9_RESIDUAL_PER_JOINT = {
    "left_hip_pitch": 0.64,
    "left_hip_roll": 0.375,
    "left_knee_pitch": 0.978,
    "left_ankle_pitch": 0.33,
    "left_ankle_roll": 0.375,
    "right_hip_pitch": 0.64,
    "right_hip_roll": 0.375,
    "right_knee_pitch": 0.978,
    "right_ankle_pitch": 0.33,
    "right_ankle_roll": 0.375,
}
SMOKE9_RESIDUAL_SCALAR = 0.2


def actuated_joint_specs_rad():
    """Real WildRobot v2 actuated joint specs, ranges normalized to radians."""
    data = json.loads(_ROBOT_CONFIG.read_text())
    unit = str(data.get("joint_range_unit", "rad")).lower()
    out = []
    for s in data["actuated_joint_specs"]:
        rng = s.get("range", [0.0, 0.0])
        lo, hi = float(rng[0]), float(rng[1])
        if unit == "deg":
            lo, hi = math.radians(lo), math.radians(hi)
        out.append(
            {
                "name": str(s["name"]),
                "range": [lo, hi],
                "max_velocity": float(s.get("max_velocity", 10.0)),
            }
        )
    return out


def make_v8_spec(*, action_filter_alpha: float = 0.0):
    """Build a valid wr_obs_v8_cmd3d PolicySpec from the real robot config.

    home_ctrl_rad = per-joint midpoint (within range), which is a valid home
    pose for the home-base residual contract tests.
    """
    from policy_contract.spec_builder import build_policy_spec

    specs = actuated_joint_specs_rad()
    home = [0.5 * (j["range"][0] + j["range"][1]) for j in specs]
    return build_policy_spec(
        robot_name="wildrobot",
        actuated_joint_specs=specs,
        action_filter_alpha=action_filter_alpha,
        home_ctrl_rad=home,
        layout_id="wr_obs_v8_cmd3d",
        mapping_id="pos_target_rad_v1",
    )


def make_reference_dict(*, n_steps: int = 96, n_cycle: int = 48):
    """Small synthetic (bin-independent) phase table for runtime tests."""
    import numpy as np

    phase = (np.arange(n_steps) % n_cycle) / float(n_cycle)
    return {
        "gait_n_cycle": n_cycle,
        "n_steps": n_steps,
        "cmd_keys": [
            [0.13, 0.0, 0.0],
            [0.13, 0.065, 0.0],
            [0.13, -0.065, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "per_bin_n_steps": [n_steps, n_steps, n_steps, 1],
        "phase_sin": [float(v) for v in np.sin(2.0 * np.pi * phase)],
        "phase_cos": [float(v) for v in np.cos(2.0 * np.pi * phase)],
    }


def make_runtime_policy_config(
    spec,
    *,
    residual_base: str = "home",
    action_delay_steps: int = 1,
    action_filter_alpha: float = 0.0,
    reference: dict | None = None,
):
    """Build a RuntimePolicyConfig with smoke9-aligned residual scales."""
    from runtime.wr_runtime.control.runtime_policy_config import (
        ReferencePhaseTable,
        RuntimePolicyConfig,
    )

    names = list(spec.robot.actuator_names)
    per_actuator = [
        SMOKE9_RESIDUAL_PER_JOINT.get(n, SMOKE9_RESIDUAL_SCALAR) for n in names
    ]
    ref = reference if reference is not None else make_reference_dict()
    return RuntimePolicyConfig(
        schema_version=1,
        actor_obs_layout_id="wr_obs_v8_cmd3d",
        action_mapping_id="pos_target_rad_v1",
        ctrl_dt=0.02,
        control_hz=50.0,
        action_delay_steps=action_delay_steps,
        action_filter_alpha=action_filter_alpha,
        loc_ref_residual_base=residual_base,
        loc_ref_residual_mode="absolute",
        loc_ref_residual_scale=SMOKE9_RESIDUAL_SCALAR,
        loc_ref_residual_scale_per_joint=dict(SMOKE9_RESIDUAL_PER_JOINT),
        residual_scale_per_actuator=per_actuator,
        loc_ref_command_conditioned=True,
        loc_ref_command_axes_3d=True,
        default_velocity_cmd=[0.13, 0.0, 0.0],
        reference=ReferencePhaseTable.from_dict(ref),
    )


@pytest.fixture
def v8_spec():
    return make_v8_spec()


@pytest.fixture
def runtime_policy_config(v8_spec):
    return make_runtime_policy_config(v8_spec)
