from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.scripts.calibrate import resolve_config_path
from runtime.wr_runtime.control.mock_robot_io import MockRobotIO
from runtime.wr_runtime.control.run_policy import (
    _resolve_hardware_config_path,
    _TargetBlendRobotIO,
)
from runtime.wr_runtime.deployment_bundle import DeploymentBundle


_BUNDLE = Path(
    "runtime/bundles/deployment_walk_v0210_ckpt1650_stand_v0222_ckpt90"
)
_CURRENT_BUNDLE = Path("runtime/bundles/standing_walk_v0222")


def test_historical_deployment_bundle_remains_an_archival_21d_artifact() -> None:
    deployment = DeploymentBundle.load(_BUNDLE)
    standing = deployment.policy_bundle("standing")
    walking = deployment.policy_bundle("walking")

    assert standing.spec.model.obs_dim == 63
    assert standing.spec.model.action_dim == 17
    assert walking.spec.model.obs_dim == 1129
    assert walking.spec.model.action_dim == 21
    assert deployment.mjcf_path.name == "wildrobot.xml"
    assert deployment.robot_config_path.name == "mujoco_robot_config.json"


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


def test_deployment_calibration_defaults_to_canonical_hardware_config() -> None:
    class _Args:
        config = None
        bundle = str(_BUNDLE)

    assert resolve_config_path(_Args()).resolve() == Path(
        "runtime/configs/hardware_config.json"
    ).resolve()


def test_policy_runtime_defaults_to_canonical_hardware_config() -> None:
    canonical = Path("runtime/configs/hardware_config.json").resolve()

    assert _resolve_hardware_config_path(None).resolve() == canonical
    assert _resolve_hardware_config_path("custom.json") == Path("custom.json")


@pytest.mark.parametrize("bundle", [_BUNDLE, _CURRENT_BUNDLE])
def test_deployment_checksums_cover_every_artifact(bundle: Path) -> None:
    expected = json.loads((bundle / "checksums.json").read_text())
    actual = {
        str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    assert actual == expected


def test_policy_bundles_do_not_contain_hardware_configuration() -> None:
    bundled_configs = sorted(
        path
        for name in ("hardware_config.json", "wildrobot_config.json")
        for path in Path("runtime/bundles").rglob(name)
    )
    assert bundled_configs == []
    for manifest_path in Path("runtime/bundles").rglob("bundle_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        assert "hardware_config" not in manifest["shared"]


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
    np.testing.assert_allclose(blended.last_commanded_q_rad, [0.6, -0.6], atol=1e-6)
    blended.write_ctrl(np.array([1.0, -1.0], dtype=np.float32))

    np.testing.assert_allclose(base.written[0], [0.6, -0.6], atol=1e-6)
    np.testing.assert_allclose(base.written[1], [1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(blended.last_commanded_q_rad, [1.0, -1.0], atol=1e-6)
