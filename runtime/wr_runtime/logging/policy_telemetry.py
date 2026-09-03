"""Structured per-step telemetry for hardware policy diagnostics."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np


class PolicyTelemetryRecorder:
    """Collect a policy run and save it as a compressed NumPy archive."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        actuator_names: Sequence[str],
        ctrl_dt: float,
        bundle_path: str | Path,
        hardware_config_path: str | Path,
    ) -> None:
        self.path = Path(path).expanduser()
        self.actuator_names = tuple(str(name) for name in actuator_names)
        self.ctrl_dt = float(ctrl_dt)
        self.bundle_path = str(Path(bundle_path).expanduser().resolve())
        self.hardware_config_path = str(
            Path(hardware_config_path).expanduser().resolve()
        )
        self._started_s = time.monotonic()
        self._rows: list[dict[str, Any]] = []

    @property
    def sample_count(self) -> int:
        return len(self._rows)

    def record(
        self,
        info: dict[str, Any],
        *,
        phase: str,
        loop_step: int,
        requested_velocity_cmd: np.ndarray | None = None,
    ) -> None:
        signals = info.get("hardware_signals")
        if signals is None:
            signals = info.get("signals")
        if signals is None:
            return

        n_act = len(self.actuator_names)
        commanded = _vector(
            info.get("commanded_q_rad", info.get("target_q_rad")), n_act
        )
        previous_commanded = _vector(info.get("previous_commanded_q_rad"), n_act)
        joint_pos = _vector(getattr(signals, "joint_pos_rad", None), n_act)
        obs_debug = info.get("obs_debug")
        if not isinstance(obs_debug, dict):
            obs_debug = {}
        velocity_cmd = _vector(obs_debug.get("velocity_cmd"), 3)
        requested_cmd = _vector(requested_velocity_cmd, 3)
        if not np.any(np.isfinite(requested_cmd)):
            requested_cmd = velocity_cmd.copy()
        servo_diagnostics = info.get("servo_diagnostics")
        if not isinstance(servo_diagnostics, dict):
            servo_diagnostics = {}

        self._rows.append(
            {
                "phase": str(phase),
                "control_mode": str(info.get("control_mode", "unknown")),
                "loop_step": int(loop_step),
                "policy_step_idx": int(info.get("step_idx", -1)),
                "host_monotonic_s": time.monotonic() - self._started_s,
                "sensor_timestamp_s": float(
                    getattr(signals, "timestamp_s", 0.0) or 0.0
                ),
                "quat_wxyz": _vector(getattr(signals, "quat_wxyz", None), 4),
                "gyro_rad_s": _vector(getattr(signals, "gyro_rad_s", None), 3),
                "joint_pos_rad": joint_pos,
                "joint_vel_rad_s": _vector(
                    getattr(signals, "joint_vel_rad_s", None), n_act
                ),
                "foot_switches": _vector(
                    getattr(signals, "foot_switches", None), 4
                ),
                "footswitch_available": bool(
                    info.get("footswitch_available", True)
                ),
                "observation": np.asarray(
                    info.get("obs", []), dtype=np.float32
                ).reshape(-1).copy(),
                "raw_action": _vector(info.get("raw_action"), n_act),
                "applied_action": _vector(info.get("applied_action"), n_act),
                "target_q_rad": _vector(info.get("target_q_rad"), n_act),
                "commanded_q_rad": commanded,
                "previous_commanded_q_rad": previous_commanded,
                "joint_tracking_error_rad": previous_commanded - joint_pos,
                "velocity_cmd": velocity_cmd,
                "requested_velocity_cmd": requested_cmd,
                "action_scale": float(info.get("action_scale", 1.0)),
                "command_ramp_scale": float(
                    info.get("command_ramp_scale", 1.0)
                ),
                "reference_bin_idx": int(obs_debug.get("reference_bin_idx", -1)),
                "phase_sin_cos": _vector(obs_debug.get("phase_sin_cos"), 2),
                "timing_s": _numeric_values(info.get("timing_s")),
                "servo_metrics": _numeric_values(info.get("servo_metrics")),
                "servo_position_units": _vector(
                    servo_diagnostics.get("position_units"), n_act
                ),
                "servo_velocity_units_s": _vector(
                    servo_diagnostics.get("velocity_units_s"), n_act
                ),
                "servo_position_age_s": _vector(
                    servo_diagnostics.get("position_age_s"), n_act
                ),
                "servo_read_fail_count": _vector(
                    servo_diagnostics.get("read_fail_count"), n_act
                ),
            }
        )

    def save(self, *, outcome: str, error: str | None = None) -> Path | None:
        if not self._rows:
            print("Telemetry requested, but no policy samples were captured.", flush=True)
            return None

        rows = self._rows
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.int32),
            "outcome": np.asarray(str(outcome)),
            "error": np.asarray(str(error or "")),
            "actuator_names": np.asarray(self.actuator_names),
            "ctrl_dt_s": np.asarray(self.ctrl_dt, dtype=np.float64),
            "bundle_path": np.asarray(self.bundle_path),
            "hardware_config_path": np.asarray(self.hardware_config_path),
            "phase": np.asarray([row["phase"] for row in rows]),
            "control_mode": np.asarray([row["control_mode"] for row in rows]),
            "loop_step": np.asarray([row["loop_step"] for row in rows], dtype=np.int64),
            "policy_step_idx": np.asarray(
                [row["policy_step_idx"] for row in rows], dtype=np.int64
            ),
            "host_monotonic_s": np.asarray(
                [row["host_monotonic_s"] for row in rows], dtype=np.float64
            ),
            "timestamp_s": np.asarray(
                [row["sensor_timestamp_s"] for row in rows], dtype=np.float64
            ),
            "footswitch_available": np.asarray(
                [row["footswitch_available"] for row in rows], dtype=bool
            ),
            "action_scale": np.asarray(
                [row["action_scale"] for row in rows], dtype=np.float32
            ),
            "command_ramp_scale": np.asarray(
                [row["command_ramp_scale"] for row in rows], dtype=np.float32
            ),
            "reference_bin_idx": np.asarray(
                [row["reference_bin_idx"] for row in rows], dtype=np.int32
            ),
        }
        for key in (
            "quat_wxyz",
            "gyro_rad_s",
            "joint_pos_rad",
            "joint_vel_rad_s",
            "foot_switches",
            "raw_action",
            "applied_action",
            "target_q_rad",
            "commanded_q_rad",
            "previous_commanded_q_rad",
            "joint_tracking_error_rad",
            "velocity_cmd",
            "requested_velocity_cmd",
            "phase_sin_cos",
            "servo_position_units",
            "servo_velocity_units_s",
            "servo_position_age_s",
            "servo_read_fail_count",
        ):
            arrays[key] = np.stack([row[key] for row in rows]).astype(np.float32)
        arrays["yaw_rate_cmd"] = arrays["velocity_cmd"][:, 2].copy()

        obs_size = np.asarray(
            [int(row["observation"].size) for row in rows], dtype=np.int32
        )
        max_obs = int(np.max(obs_size)) if obs_size.size else 0
        observations = np.full((len(rows), max_obs), np.nan, dtype=np.float32)
        for index, row in enumerate(rows):
            observations[index, : obs_size[index]] = row["observation"]
        arrays["observation"] = observations
        arrays["observation_size"] = obs_size
        arrays.update(_metric_arrays(rows, source="timing_s", prefix="timing_"))
        arrays.update(
            _metric_arrays(rows, source="servo_metrics", prefix="servo_")
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(self.path)
        print(
            f"Saved policy telemetry: {self.path} ({len(rows)} samples).",
            flush=True,
        )
        return self.path


def _vector(value: Any, size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != size:
        return np.full(size, np.nan, dtype=np.float32)
    return array.copy()


def _numeric_values(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float, np.integer, np.floating)):
            output[str(key)] = float(item)
    return output


def _metric_arrays(
    rows: Sequence[dict[str, Any]], *, source: str, prefix: str
) -> dict[str, np.ndarray]:
    keys = sorted({key for row in rows for key in row[source]})
    return {
        f"{prefix}{key}": np.asarray(
            [row[source].get(key, np.nan) for row in rows], dtype=np.float64
        )
        for key in keys
    }
