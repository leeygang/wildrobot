from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from runtime.scripts.calibrate import resolve_config_path
from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
from runtime.wr_runtime.control.run_policy import _TargetBlendRobotIO
from runtime.wr_runtime.deployment_bundle import DeploymentBundle


_BUNDLE = Path(
    "runtime/bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90"
)


def test_deployment_bundle_has_shared_hardware_and_two_policy_contracts() -> None:
    deployment = DeploymentBundle.load(_BUNDLE)
    standing = deployment.policy_bundle("standing")
    walking = deployment.policy_bundle("walking")
    hardware = json.loads(deployment.hardware_config_path.read_text())

    assert standing.spec.model.obs_dim == 63
    assert standing.spec.model.action_dim == 17
    assert walking.spec.model.obs_dim == 1129
    assert walking.spec.model.action_dim == 21
    assert deployment.mjcf_path.name == "wildrobot.xml"
    assert deployment.robot_config_path.name == "mujoco_robot_config.json"
    assert "policy_onnx_path" not in hardware
    assert "velocity_cmd" not in hardware
    assert hardware["robot_config_path"] == "./mujoco_robot_config.json"
    assert len(hardware["externally_managed_actuator_names"]) == 4


def test_deployment_policy_timing_metadata_is_role_specific() -> None:
    deployment = DeploymentBundle.load(_BUNDLE)
    standing = json.loads(
        (deployment.policy_dir("standing") / "runtime_policy_config.json").read_text()
    )
    walking = json.loads(
        (deployment.policy_dir("walking") / "runtime_policy_config.json").read_text()
    )

    assert standing["policy_role"] == "standing"
    assert standing["action_delay_steps"] == 1
    assert standing["ctrl_dt"] == 0.02
    assert "reference" not in standing
    assert walking["action_delay_steps"] == 1
    assert walking["ctrl_dt"] == 0.02
    assert walking["reference"]["n_steps"] == 1104


def test_deployment_calibration_defaults_to_shared_hardware_config() -> None:
    class _Args:
        config = None
        bundle = str(_BUNDLE)

    assert resolve_config_path(_Args()).resolve() == (
        _BUNDLE / "hardware_config.json"
    ).resolve()


def test_deployment_checksums_cover_every_artifact() -> None:
    expected = json.loads((_BUNDLE / "checksums.json").read_text())
    actual = {
        str(path.relative_to(_BUNDLE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_BUNDLE.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    assert actual == expected


def test_walking_transition_blends_from_final_standing_target() -> None:
    base = MockRobotIO(
        actuator_names=["left_hip_pitch", "right_hip_pitch"],
        control_dt=0.02,
        home_q_rad=np.zeros(2, dtype=np.float32),
    )
    blended = _TargetBlendRobotIO(
        base,
        initial_target=np.array([0.2, -0.2], dtype=np.float32),
        blend_steps=2,
    )

    blended.write_ctrl(np.array([1.0, -1.0], dtype=np.float32))
    blended.write_ctrl(np.array([1.0, -1.0], dtype=np.float32))

    np.testing.assert_allclose(base.written[0], [0.6, -0.6], atol=1e-6)
    np.testing.assert_allclose(base.written[1], [1.0, -1.0], atol=1e-6)
