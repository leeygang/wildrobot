#!/usr/bin/env python3
"""Compare real capture and sim replay traces for v0.19.1 sim2real eval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ReplayTrace:
    timestamps_s: np.ndarray
    command_rad: np.ndarray
    measured_position_rad: np.ndarray
    measured_velocity_rad_s: np.ndarray
    gyro_rad_s: np.ndarray | None
    foot_switches: np.ndarray | None

    def validate(self) -> None:
        t = int(self.timestamps_s.shape[0])
        if t < 2:
            raise ValueError("trace must have at least two samples")
        for name, arr in (
            ("command_rad", self.command_rad),
            ("measured_position_rad", self.measured_position_rad),
            ("measured_velocity_rad_s", self.measured_velocity_rad_s),
        ):
            if arr.shape != (t,):
                raise ValueError(f"{name} must be shape (T,), got {arr.shape}")
        if self.gyro_rad_s is not None and self.gyro_rad_s.shape != (t,):
            raise ValueError(f"gyro_rad_s must be shape (T,), got {self.gyro_rad_s.shape}")
        if self.foot_switches is not None and self.foot_switches.shape != (t,):
            raise ValueError(f"foot_switches must be shape (T,), got {self.foot_switches.shape}")


def _load_trace(npz_path: Path, *, gyro_axis: int, foot_index: int) -> ReplayTrace:
    data = np.load(npz_path)
    files = set(data.files)
    timestamps = np.asarray(data["timestamps_s"], dtype=np.float64)
    command = np.asarray(data["command_rad"], dtype=np.float64)
    measured_pos = np.asarray(data["measured_position_rad"], dtype=np.float64)
    measured_vel = np.asarray(data.get("measured_velocity_rad_s"), dtype=np.float64)

    gyro_1d = None
    if "gyro_rad_s" in files:
        gyro = np.asarray(data["gyro_rad_s"], dtype=np.float64)
        if gyro.ndim == 2:
            if gyro_axis < 0 or gyro_axis >= gyro.shape[1]:
                raise ValueError(f"gyro-axis {gyro_axis} out of bounds for shape {gyro.shape}")
            gyro_1d = gyro[:, gyro_axis]
        elif gyro.ndim == 1:
            gyro_1d = gyro

    foot_1d = None
    if "foot_switches" in files:
        foot = np.asarray(data["foot_switches"], dtype=np.float64)
        if foot.ndim == 2:
            if foot_index < 0 or foot_index >= foot.shape[1]:
                raise ValueError(f"foot-index {foot_index} out of bounds for shape {foot.shape}")
            foot_1d = foot[:, foot_index]
        elif foot.ndim == 1:
            foot_1d = foot

    trace = ReplayTrace(
        timestamps_s=timestamps,
        command_rad=command,
        measured_position_rad=measured_pos,
        measured_velocity_rad_s=measured_vel,
        gyro_rad_s=gyro_1d,
        foot_switches=foot_1d,
    )
    trace.validate()
    return trace


def _resample(x_t: np.ndarray, x: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    return np.interp(ref_t, x_t, x)


def _estimate_delay_s(reference: np.ndarray, candidate: np.ndarray, dt: float) -> float:
    a = np.asarray(reference - np.mean(reference), dtype=np.float64)
    b = np.asarray(candidate - np.mean(candidate), dtype=np.float64)
    corr = np.correlate(a, b, mode="full")
    lag = int(np.argmax(corr) - (a.shape[0] - 1))
    return float(lag * dt)


def _step_metrics(command: np.ndarray, response: np.ndarray) -> dict[str, float]:
    c0 = float(command[0])
    c1 = float(command[-1])
    delta = c1 - c0
    if abs(delta) < 1e-8:
        return {"overshoot": 0.0, "settling_time_s": 0.0, "steady_state_error": float(np.mean(response - command))}

    max_resp = float(np.max(response) if delta > 0 else np.min(response))
    overshoot = (max_resp - c1) / delta
    abs_err = np.abs(response - c1)
    band = max(abs(delta) * 0.02, 1e-3)
    inside = np.where(abs_err <= band)[0]
    settling_idx = int(inside[0]) if inside.size > 0 else response.shape[0] - 1
    steady_state_error = float(np.mean(response[-max(5, response.shape[0] // 10):] - c1))
    return {
        "overshoot": float(overshoot),
        "settling_time_index": float(settling_idx),
        "steady_state_error": steady_state_error,
    }


def compare_replays(
    *,
    real: ReplayTrace,
    sim: ReplayTrace,
) -> dict[str, Any]:
    dt = float(np.median(np.diff(real.timestamps_s)))
    sim_pos = _resample(sim.timestamps_s, sim.measured_position_rad, real.timestamps_s)
    sim_cmd = _resample(sim.timestamps_s, sim.command_rad, real.timestamps_s)

    delay_s = _estimate_delay_s(real.measured_position_rad, sim_pos, dt)
    pos_rmse = float(np.sqrt(np.mean((real.measured_position_rad - sim_pos) ** 2)))
    cmd_rmse = float(np.sqrt(np.mean((real.command_rad - sim_cmd) ** 2)))

    real_step = _step_metrics(real.command_rad, real.measured_position_rad)
    sim_step = _step_metrics(sim_cmd, sim_pos)
    settling_time_s = float(real_step.get("settling_time_index", 0.0) * dt)

    out: dict[str, Any] = {
        "delay_s": delay_s,
        "position_rmse_rad": pos_rmse,
        "command_rmse_rad": cmd_rmse,
        "overshoot_real": float(real_step["overshoot"]),
        "overshoot_sim": float(sim_step["overshoot"]),
        "overshoot_mismatch": float(sim_step["overshoot"] - real_step["overshoot"]),
        "settling_time_s_real": settling_time_s,
        "steady_state_error_real_rad": float(real_step["steady_state_error"]),
        "steady_state_error_sim_rad": float(sim_step["steady_state_error"]),
    }

    if real.gyro_rad_s is not None and sim.gyro_rad_s is not None:
        sim_gyro = _resample(sim.timestamps_s, sim.gyro_rad_s, real.timestamps_s)
        out["imu_gyro_rmse_rad_s"] = float(np.sqrt(np.mean((real.gyro_rad_s - sim_gyro) ** 2)))

    if real.foot_switches is not None and sim.foot_switches is not None:
        sim_foot = _resample(sim.timestamps_s, sim.foot_switches, real.timestamps_s)
        real_bin = (real.foot_switches > 0.5).astype(np.float64)
        sim_bin = (sim_foot > 0.5).astype(np.float64)
        out["foot_switch_timing_mismatch_frac"] = float(np.mean(np.abs(real_bin - sim_bin)))

    return out


def _plot_overlay(
    *,
    real: ReplayTrace,
    sim: ReplayTrace,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(real.timestamps_s, real.command_rad, label="real command", linewidth=1.0)
    plt.plot(real.timestamps_s, real.measured_position_rad, label="real measured", linewidth=1.4)
    sim_pos = _resample(sim.timestamps_s, sim.measured_position_rad, real.timestamps_s)
    plt.plot(real.timestamps_s, sim_pos, label="sim measured (resampled)", linewidth=1.4)
    plt.xlabel("time [s]")
    plt.ylabel("joint position [rad]")
    plt.title("Sim-vs-real joint trace")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_error(
    *,
    real: ReplayTrace,
    sim: ReplayTrace,
    output_path: Path,
) -> None:
    sim_pos = _resample(sim.timestamps_s, sim.measured_position_rad, real.timestamps_s)
    err = sim_pos - real.measured_position_rad
    plt.figure(figsize=(10, 4))
    plt.plot(real.timestamps_s, err, label="sim - real", linewidth=1.3)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("time [s]")
    plt.ylabel("position error [rad]")
    plt.title("Sim-vs-real position error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sim replay against real capture.")
    parser.add_argument("--real-log", required=True, help="Real capture .npz")
    parser.add_argument("--sim-log", required=True, help="Sim replay .npz")
    parser.add_argument("--output-dir", default="runtime/logs/comparisons")
    parser.add_argument("--summary-json", default="summary.json")
    parser.add_argument("--gyro-axis", type=int, default=2, help="Gyro axis index for mismatch metric")
    parser.add_argument("--foot-index", type=int, default=0, help="Foot switch index for timing mismatch metric")
    parser.add_argument("--tag", default="compare")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    real_path = Path(args.real_log)
    sim_path = Path(args.sim_log)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = _load_trace(real_path, gyro_axis=int(args.gyro_axis), foot_index=int(args.foot_index))
    sim = _load_trace(sim_path, gyro_axis=int(args.gyro_axis), foot_index=int(args.foot_index))
    summary = compare_replays(real=real, sim=sim)
    summary["real_log"] = str(real_path)
    summary["sim_log"] = str(sim_path)
    summary["tag"] = args.tag

    summary_path = out_dir / args.summary_json
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    overlay_png = out_dir / f"{args.tag}_overlay.png"
    error_png = out_dir / f"{args.tag}_error.png"
    _plot_overlay(real=real, sim=sim, output_path=overlay_png)
    _plot_error(real=real, sim=sim, output_path=error_png)

    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote plot: {overlay_png}")
    print(f"Wrote plot: {error_png}")


if __name__ == "__main__":
    main()
