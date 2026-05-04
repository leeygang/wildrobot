from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sysid.run_capture import build_capture_artifact
from tools.sysid.run_capture import _read_servo_position_units


def test_sysid_capture_artifact_shapes() -> None:
    artifact = build_capture_artifact(
        mode="step",
        joint_name="left_knee_pitch",
        sample_rate_hz=50.0,
        duration_s=2.0,
        amplitude_rad=0.2,
        hold_rad=0.0,
        step_start_s=0.5,
        chirp_start_hz=0.2,
        chirp_end_hz=2.0,
        model_delay_steps=1,
        model_backlash_rad=0.01,
        model_tau_s=0.08,
        model_noise_std=0.001,
        seed=1,
        metadata={"asset_version": "v2"},
    )
    artifact.validate()
    t = artifact.timestamps_s.shape[0]
    assert t > 10
    assert artifact.command_rad.shape == (t,)
    assert artifact.measured_position_rad.shape == (t,)
    assert artifact.measured_velocity_rad_s.shape == (t,)
    assert np.all(np.isfinite(artifact.command_rad))


class _FakeController:
    def __init__(self) -> None:
        self.calls = 0

    def read_servo_positions(self, ids):
        self.calls += 1
        if self.calls < 3:
            return None
        return [(int(ids[0]), 512)]


def test_read_servo_position_units_retries_and_succeeds() -> None:
    controller = _FakeController()
    units = _read_servo_position_units(
        controller,
        servo_id=7,
        retries=5,
        retry_sleep_s=0.0,
    )
    assert units == 512
