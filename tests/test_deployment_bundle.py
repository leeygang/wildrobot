from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.scripts.calibrate import resolve_config_path
from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
from runtime.wr_runtime.control.run_policy import _TargetBlendRobotIO
from runtime.wr_runtime.deployment_bundle import DeploymentBundle
from training.exports.export_policy_bundle import _export_hardware_config


_BUNDLE = Path(
    "runtime/bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90"
)
_CURRENT_BUNDLE = Path("runtime/bundles/standing_walk_v0222")


def test_historical_deployment_bundle_remains_an_archival_21d_artifact() -> None:
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


def test_canonical_hardware_config_is_natively_wrist_free() -> None:
    hardware = json.loads(Path("runtime/configs/hardware_config.json").read_text())
    servos = hardware["servo_controller"]["servos"]
    board_ids = {
        servo_id
        for board in hardware["servo_controller"]["boards"]
        for servo_id in board["servo_ids"]
    }

    assert len(servos) == 17
    assert not any("wrist" in name for name in servos)
    assert {servo["id"] for servo in servos.values()} == board_ids
    assert "externally_managed_actuator_names" not in hardware


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


@pytest.mark.parametrize("bundle", [_BUNDLE, _CURRENT_BUNDLE])
def test_deployment_checksums_cover_every_artifact(bundle: Path) -> None:
    expected = json.loads((bundle / "checksums.json").read_text())
    actual = {
        str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    assert actual == expected


def test_policy_export_uses_canonical_hardware_config(tmp_path: Path) -> None:
    template = json.loads(Path("runtime/configs/hardware_config.json").read_text())
    legacy_template = dict(template)
    legacy_template["realism_profile_path"] = "./realism_profile.json"
    source_path = tmp_path / "legacy_hardware_config.json"
    source_path.write_text(json.dumps(legacy_template))
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    output = _export_hardware_config(
        output_dir=output_dir,
        source_path=source_path,
    )
    exported = json.loads(output.read_text())

    assert exported["servo_controller"] == template["servo_controller"]
    assert exported["servo_read_schedule"] == template["servo_read_schedule"]
    assert exported["robot_config_path"] == "./mujoco_robot_config.json"
    assert "realism_profile_path" not in exported
    assert not (output_dir / "realism_profile.json").exists()


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
