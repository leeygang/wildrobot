from __future__ import annotations

from pathlib import Path

import pytest

from training.eval.verify_walking_stance_geometry import (
    analyze_stance_candidate,
    load_stance_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "training/configs/ppo_walking_v0210_17d8_hip_roll_margin.yaml"


def _analyze(offset_rad: float):
    (
        model,
        robot_config,
        home_qpos,
        home_foot_rotations,
        close_feet_threshold_m,
    ) = load_stance_inputs(CONFIG)
    return analyze_stance_candidate(
        model=model,
        robot_config=robot_config,
        home_qpos=home_qpos,
        home_foot_rotations=home_foot_rotations,
        offset_rad=offset_rad,
        close_feet_threshold_m=close_feet_threshold_m,
        max_support_torque_ratio=0.8,
        max_foot_orientation_delta_deg=1.0,
        max_sole_height_delta_m=0.002,
    )


def test_symmetric_roll_offset_reduces_support_leverage_and_keeps_feet_flat() -> None:
    home = _analyze(0.0)
    candidate = _analyze(0.03)

    assert candidate["foot_center_separation_m"] < home["foot_center_separation_m"]
    assert candidate["inner_foot_clearance_m"] > 0.0
    assert max(candidate["foot_orientation_delta_deg"].values()) < 1.0
    assert candidate["sole_height_delta_m"] < 0.002
    for side in ("left", "right"):
        assert (
            candidate["support"][side]["quasi_static_support_ratio"]
            < home["support"][side]["quasi_static_support_ratio"]
        )


def test_candidate_003_passes_static_geometry_gate_but_home_does_not() -> None:
    home = _analyze(0.0)
    candidate = _analyze(0.03)

    assert home["support"]["left"]["quasi_static_support_ratio"] == pytest.approx(
        0.8975, abs=0.002
    )
    assert not home["passed"]
    assert candidate["passed"]
    assert candidate["close_feet_margin_m"] > 0.01
