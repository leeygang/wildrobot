from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sim2real_eval.replay_compare import ReplayTrace, compare_replays


def _trace(delay_s: float) -> ReplayTrace:
    t = np.linspace(0.0, 2.0, 201)
    cmd = np.where(t >= 0.5, 0.2, 0.0)
    delayed_t = np.clip(t - delay_s, 0.0, t[-1])
    pos = np.where(delayed_t >= 0.5, 0.19, 0.0)
    vel = np.gradient(pos, t)
    gyro = 0.05 * np.sin(2.0 * np.pi * 0.8 * t)
    foot = (t > 0.6).astype(np.float64)
    trace = ReplayTrace(
        timestamps_s=t,
        command_rad=cmd,
        measured_position_rad=pos,
        measured_velocity_rad_s=vel,
        gyro_rad_s=gyro,
        foot_switches=foot,
    )
    trace.validate()
    return trace


def test_sim2real_summary_generation_with_synthetic_data() -> None:
    real = _trace(delay_s=0.03)
    sim = _trace(delay_s=0.01)
    summary = compare_replays(real=real, sim=sim)
    assert "delay_s" in summary
    assert "position_rmse_rad" in summary
    assert "overshoot_mismatch" in summary
    assert "steady_state_error_real_rad" in summary
    assert "imu_gyro_rmse_rad_s" in summary
    assert "foot_switch_timing_mismatch_frac" in summary
    assert summary["position_rmse_rad"] >= 0.0
