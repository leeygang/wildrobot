from __future__ import annotations

from pathlib import Path

import pytest

from control.realism_profile import DigitalTwinRealismProfile, load_realism_profile
from runtime.configs import WildRobotRuntimeConfig
from runtime.wr_runtime.validation.realism_profile import load_runtime_realism_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_realism_profile_loads_from_versioned_asset() -> None:
    profile = load_realism_profile(REPO_ROOT / "assets/v2/realism_profile_v0.19.1.json")
    assert profile.schema_version == "v0.19.1"
    assert profile.profile_name == "wildrobot_v2_baseline_sysid_m1"
    assert len(profile.actuators) >= 9
    assert profile.sensor_model.imu_gyro_noise_std >= 0.0


def test_realism_profile_rejects_duplicate_joint_entries() -> None:
    payload = {
        "schema_version": "v0.19.1",
        "profile_name": "dup",
        "asset_version": "v2",
        "sample_rate_hz": 50.0,
        "actuators": [
            {
                "joint_name": "left_knee_pitch",
                "control_delay_steps": 1,
                "backlash_rad": 0.01,
                "frictionloss": 0.01,
                "armature": 0.01,
                "effective_max_velocity_rad_s": 1.0,
                "effective_max_torque_scale": 1.0,
            },
            {
                "joint_name": "left_knee_pitch",
                "control_delay_steps": 1,
                "backlash_rad": 0.01,
                "frictionloss": 0.01,
                "armature": 0.01,
                "effective_max_velocity_rad_s": 1.0,
                "effective_max_torque_scale": 1.0,
            },
        ],
        "sensor_model": {
            "imu_gyro_noise_std": 0.01,
            "imu_gyro_bias_walk_std": 0.001,
            "imu_orientation_noise_std": 0.001,
            "foot_switch_latency_steps": 1,
            "foot_switch_dropout_prob": 0.0,
        },
        "metadata": {},
    }
    with pytest.raises(ValueError):
        DigitalTwinRealismProfile.from_dict(payload)


def test_runtime_config_loads_realism_profile_path_and_validation() -> None:
    cfg = WildRobotRuntimeConfig.load(REPO_ROOT / "runtime/configs/runtime_config_v2.json")
    profile = load_runtime_realism_profile(cfg)
    assert profile is not None
    assert profile.asset_version == "v2"
