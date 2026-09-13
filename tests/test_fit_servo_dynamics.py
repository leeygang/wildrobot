from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sysid.fit_servo_dynamics import ServoDynamics
from tools.sysid.fit_servo_dynamics import _unpack_parameters
from tools.sysid.fit_servo_dynamics import load_capture
from tools.sysid.fit_servo_dynamics import load_xml_dynamics
from tools.sysid.fit_servo_dynamics import update_xml_dynamics


def test_update_xml_dynamics_preserves_unidentified_limits(tmp_path: Path) -> None:
    path = tmp_path / "joints_properties.xml"
    path.write_text(
        "<default>\n"
        "  <default class=\"htd45hServo\">\n"
        "    <joint damping=\"0.1\" frictionloss=\"0.2\" armature=\"0.3\" />\n"
        "    <position kp=\"20\" kv=\"0.5\" forcerange=\"-4 4\" />\n"
        "  </default>\n"
        "</default>\n"
    )
    fitted = ServoDynamics(
        kp=17.25,
        kv=0.5,
        damping=0.12,
        armature=0.02,
        frictionloss=0.03,
        delay_steps=1,
    )

    update_xml_dynamics(path, fitted)

    text = path.read_text()
    assert 'damping="0.12"' in text
    assert 'frictionloss="0.03"' in text
    assert 'armature="0.02"' in text
    assert 'kp="17.25"' in text
    assert 'kv="0.5"' in text
    assert 'forcerange="-4 4"' in text
    assert load_xml_dynamics(path) == fitted.__class__(
        **{**fitted.__dict__, "delay_steps": 0}
    )


def test_load_capture_rejects_prepare_only(tmp_path: Path) -> None:
    path = tmp_path / "capture.npz"
    np.savez(
        path,
        command_rad=np.array([0.0, 0.1]),
        measured_position_rad=np.array([0.0, 0.05]),
        timestamps_s=np.array([0.0, 0.02]),
        position_valid=np.array([True, True]),
        segment_name=np.array(["initial_center", "step_positive_2deg"]),
    )
    path.with_suffix(".json").write_text(
        json.dumps({"outcome": "completed", "prepare_only": True})
    )

    with pytest.raises(ValueError, match="preparation-only"):
        load_capture(path)


def test_load_capture_reads_complete_profile(tmp_path: Path) -> None:
    path = tmp_path / "capture.npz"
    np.savez(
        path,
        command_rad=np.array([0.0, 0.1]),
        measured_position_rad=np.array([0.0, 0.05]),
        timestamps_s=np.array([0.0, 0.02]),
        position_valid=np.array([True, True]),
        segment_name=np.array(["initial_center", "step_positive_2deg"]),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "outcome": "completed",
                "prepare_only": False,
                "center_deg": 0.0,
                "sample_hz": 50.0,
                "fixture_direction": 1,
                "fixture_qpos_offset_deg": 0.0,
                "fixture_mjcf_sha256": "abc",
                "chirp_end_hz": 2.0,
            }
        )
    )

    trace = load_capture(path)

    assert trace.center_deg == 0.0
    assert trace.sample_hz == 50.0
    np.testing.assert_allclose(trace.command_rad, [0.0, 0.1])


def test_fit_parameter_unpack_keeps_unidentified_values_fixed() -> None:
    baseline = ServoDynamics(
        kp=21.1,
        kv=0.5,
        damping=0.08516,
        armature=0.024992,
        frictionloss=0.022275,
        delay_steps=0,
    )

    fitted = _unpack_parameters(
        np.log([31.902, 1.10618, 0.324094]),
        fixed=baseline,
        delay_steps=1,
    )

    assert fitted.kp == pytest.approx(31.902)
    assert fitted.damping == pytest.approx(1.10618)
    assert fitted.frictionloss == pytest.approx(0.324094)
    assert fitted.armature == baseline.armature
    assert fitted.kv == baseline.kv
    assert fitted.delay_steps == 1


def test_source_and_generated_htd_defaults_match() -> None:
    root = Path(__file__).resolve().parents[1]

    source = load_xml_dynamics(root / "assets/v2/joints_properties.xml")
    generated = load_xml_dynamics(root / "assets/v2/wildrobot.xml")

    assert source == generated
