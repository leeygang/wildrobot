from __future__ import annotations

import json
import numpy as np
import pytest


def test_numpy_calib_interface() -> None:
    from policy_contract import spec as spec_mod
    from policy_contract.calib import NumpyCalibOps

    dummy_spec = spec_mod.PolicySpec.from_json(
        json.dumps(
            {
            "contract_name": "wildrobot_policy",
            "contract_version": "1.0.0",
            "spec_version": 1,
            "model": {
                "format": "onnx",
                "input_name": "observation",
                "output_name": "action",
                "dtype": "float32",
                "obs_dim": 15,
                "action_dim": 1,
            },
            "robot": {
                "robot_name": "WildRobotDev",
                "actuator_names": ["joint"],
                "joints": {
                    "joint": {
                        "range_min_rad": -0.5,
                        "range_max_rad": 0.5,
                        "mirror_sign": 1.0,
                        "max_velocity_rad_s": 10.0,
                    }
                },
            },
            "observation": {
                "dtype": "float32",
                "layout_id": "wr_obs_v1",
                "layout": [
                    {"name": "gravity_local", "size": 3},
                    {"name": "angvel_heading_local", "size": 3},
                    {"name": "joint_pos_normalized", "size": 1},
                    {"name": "joint_vel_normalized", "size": 1},
                    {"name": "foot_switches", "size": 4},
                    {"name": "prev_action", "size": 1},
                    {"name": "velocity_cmd", "size": 1},
                    {"name": "padding", "size": 1},
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
        )
    )

    action = np.array([0.1], dtype=np.float32)
    ctrl = NumpyCalibOps.action_to_ctrl(spec=dummy_spec, action=action)
    _ = NumpyCalibOps.ctrl_to_policy_action(spec=dummy_spec, ctrl_rad=ctrl)
    _ = NumpyCalibOps.normalize_joint_pos(spec=dummy_spec, joint_pos_rad=np.array([0.0], dtype=np.float32))
    _ = NumpyCalibOps.normalize_joint_vel(spec=dummy_spec, joint_vel_rad_s=np.array([0.0], dtype=np.float32))


def test_jax_calib_interface() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from policy_contract import spec as spec_mod
    from policy_contract.calib import JaxCalibOps

    dummy_spec = spec_mod.PolicySpec.from_json(
        json.dumps(
            {
            "contract_name": "wildrobot_policy",
            "contract_version": "1.0.0",
            "spec_version": 1,
            "model": {
                "format": "onnx",
                "input_name": "observation",
                "output_name": "action",
                "dtype": "float32",
                "obs_dim": 15,
                "action_dim": 1,
            },
            "robot": {
                "robot_name": "WildRobotDev",
                "actuator_names": ["joint"],
                "joints": {
                    "joint": {
                        "range_min_rad": -0.5,
                        "range_max_rad": 0.5,
                        "mirror_sign": 1.0,
                        "max_velocity_rad_s": 10.0,
                    }
                },
            },
            "observation": {
                "dtype": "float32",
                "layout_id": "wr_obs_v1",
                "layout": [
                    {"name": "gravity_local", "size": 3},
                    {"name": "angvel_heading_local", "size": 3},
                    {"name": "joint_pos_normalized", "size": 1},
                    {"name": "joint_vel_normalized", "size": 1},
                    {"name": "foot_switches", "size": 4},
                    {"name": "prev_action", "size": 1},
                    {"name": "velocity_cmd", "size": 1},
                    {"name": "padding", "size": 1},
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
        )
    )

    action = jnp.array([0.1], dtype=jnp.float32)
    ctrl = JaxCalibOps.action_to_ctrl(spec=dummy_spec, action=action)
    _ = JaxCalibOps.ctrl_to_policy_action(spec=dummy_spec, ctrl_rad=ctrl)
    _ = JaxCalibOps.normalize_joint_pos(spec=dummy_spec, joint_pos_rad=jnp.array([0.0], dtype=jnp.float32))
    _ = JaxCalibOps.normalize_joint_vel(spec=dummy_spec, joint_vel_rad_s=jnp.array([0.0], dtype=jnp.float32))
