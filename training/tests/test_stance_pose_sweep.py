from types import SimpleNamespace

import pytest

from training.eval.sweep_stance_pose import (
    _configured_symmetric_home_roll_offset,
    _pose_scaled_close_feet_threshold,
)


def test_configured_symmetric_home_roll_offset_reads_absolute_magnitude() -> None:
    env_cfg = SimpleNamespace(
        home_joint_offsets_rad={
            "left_hip_roll": 0.03,
            "right_hip_roll": -0.03,
            "left_ankle_roll": -0.03,
            "right_ankle_roll": 0.03,
        }
    )

    assert _configured_symmetric_home_roll_offset(env_cfg) == pytest.approx(0.03)


def test_configured_symmetric_home_roll_offset_rejects_asymmetry() -> None:
    env_cfg = SimpleNamespace(
        home_joint_offsets_rad={
            "left_hip_roll": 0.03,
            "right_hip_roll": -0.02,
            "left_ankle_roll": -0.03,
            "right_ankle_roll": 0.03,
        }
    )

    with pytest.raises(ValueError, match="requires symmetric"):
        _configured_symmetric_home_roll_offset(env_cfg)


def test_pose_scaled_close_feet_threshold_preserves_toddlerbot_ratio() -> None:
    assert _pose_scaled_close_feet_threshold(0.13075) == pytest.approx(
        0.1060135, abs=1e-7
    )
