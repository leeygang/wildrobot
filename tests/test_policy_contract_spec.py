from __future__ import annotations

import json

import pytest

from policy_contract.spec import PolicySpec, validate_spec


def _valid_spec_dict() -> dict:
    return {
        "contract_name": "wildrobot_policy",
        "contract_version": "1.0.0",
        "spec_version": 1,
        "model": {
            "format": "onnx",
            "input_name": "observation",
            "output_name": "action",
            "dtype": "float32",
            "obs_dim": 18,
            "action_dim": 2,
        },
        "robot": {
            "robot_name": "WildRobotDev",
            "actuator_names": ["joint_a", "joint_b"],
            "joints": {
                "joint_a": {
                    "range_min_rad": -0.5,
                    "range_max_rad": 0.5,
                    "mirror_sign": 1.0,
                    "max_velocity_rad_s": 10.0,
                },
                "joint_b": {
                    "range_min_rad": -1.0,
                    "range_max_rad": 1.0,
                    "mirror_sign": -1.0,
                    "max_velocity_rad_s": 12.0,
                },
            },
        },
        "observation": {
            "dtype": "float32",
            "layout_id": "wr_obs_v1",
            "linvel_mode": "zero",
            "layout": [
                {"name": "gravity_local", "size": 3, "frame": "local", "units": "unit_vector"},
                {"name": "angvel_heading_local", "size": 3, "frame": "heading_local", "units": "rad_s"},
                {"name": "joint_pos_normalized", "size": 2, "units": "normalized_-1_1"},
                {"name": "joint_vel_normalized", "size": 2, "units": "normalized_-1_1"},
                {"name": "foot_switches", "size": 4, "units": "bool_as_float"},
                {"name": "prev_action", "size": 2, "units": "normalized_-1_1"},
                {"name": "velocity_cmd", "size": 1, "units": "m_s"},
                {"name": "padding", "size": 1, "units": "unused"},
            ],
        },
        "action": {
            "dtype": "float32",
            "bounds": {"min": -1.0, "max": 1.0},
            "postprocess_id": "none",
            "postprocess_params": {},
            "mapping_id": "pos_target_rad_v1",
        },
    }


def test_valid_spec_parses_and_validates() -> None:
    spec = PolicySpec.from_json(json.dumps(_valid_spec_dict()))
    validate_spec(spec)


def test_missing_required_field_raises() -> None:
    data = _valid_spec_dict()
    data.pop("model")
    with pytest.raises(ValueError, match="model"):
        PolicySpec.from_json(json.dumps(data))


def test_obs_dim_mismatch_raises() -> None:
    data = _valid_spec_dict()
    data["model"]["obs_dim"] = 19
    spec = PolicySpec.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="obs_dim"):
        validate_spec(spec)


def test_unknown_ids_raise() -> None:
    data = _valid_spec_dict()
    data["observation"]["layout_id"] = "bad_layout"
    spec = PolicySpec.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="layout_id"):
        validate_spec(spec)

    data = _valid_spec_dict()
    data["observation"]["linvel_mode"] = "bad_linvel"
    spec = PolicySpec.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="linvel_mode"):
        validate_spec(spec)

    data = _valid_spec_dict()
    data["action"]["mapping_id"] = "bad_mapping"
    spec = PolicySpec.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="mapping_id"):
        validate_spec(spec)

    data = _valid_spec_dict()
    data["action"]["postprocess_id"] = "bad_post"
    spec = PolicySpec.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="postprocess_id"):
        validate_spec(spec)
