#!/usr/bin/env python3
"""Capture HTD-45H position-servo dynamics on a known-load lever fixture.

The script drives exactly one standalone fixture servo through conservative
step and chirp targets, records encoder feedback at the deployment control
rate, and writes an NPZ trace plus a JSON summary.  It does not fit a motor
model; the capture is the measured input required by the later
simulator-identification step.

The fixture must have a verified clear travel range, a load catcher, and a
reachable power cutoff. The script starts after a cancellable delay and returns
to a verified gravity-neutral pose before disabling torque.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from configs.config import ServoConfig  # noqa: E402
from wr_runtime.hardware.hiwonder_ttl_bus import (  # noqa: E402
    RawServoBus,
    RawServoBusConfig,
    SerialTransport,
    SerialTransportConfig,
)


SCHEMA_VERSION = 4
STANDARD_GRAVITY_M_S2 = 9.80665
DEFAULT_AMPLITUDES_DEG = (2.0, 5.0, 8.0)


def _yellow(text: str) -> str:
    if not sys.stderr.isatty() or "NO_COLOR" in os.environ:
        return text
    if os.environ.get("TERM", "") in {"", "dumb"}:
        return text
    return f"\x1b[33m{text}\x1b[0m"


def build_fixture_servo_config(servo_id: int) -> ServoConfig:
    """Return the standalone fixture's raw HTD servo-coordinate contract."""

    return ServoConfig(
        id=int(servo_id),
        servo_offset_unit=0,
        motor_unit_direction=1.0,
        joint_angle_at_servo_center_deg=0.0,
        rad_range=(-ServoConfig.RANGE_RAD / 2.0, ServoConfig.RANGE_RAD / 2.0),
    )


@dataclass(frozen=True)
class ProfileSegment:
    name: str
    kind: str
    targets_rad: tuple[float, ...]


@dataclass
class MujocoFixtureModel:
    path: Path
    joint_name: str
    direction: int
    qpos_offset_rad: float
    mujoco: Any
    model: Any
    data: Any
    joint_id: int
    qpos_address: int
    dof_address: int
    root_body_name: str
    moving_body_name: str
    moving_subtree_mass_kg: float

    def evaluate(
        self, hardware_joint_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(hardware_joint_rad, dtype=np.float64).reshape(-1)
        fixture_qpos = (
            float(self.qpos_offset_rad) + float(self.direction) * positions
        )
        if bool(self.model.jnt_limited[self.joint_id]):
            lower, upper = self.model.jnt_range[self.joint_id]
            if np.any(fixture_qpos < lower) or np.any(fixture_qpos > upper):
                raise ValueError(
                    f"fixture joint {self.joint_name!r} range [{lower}, {upper}] "
                    "does not contain the requested profile"
                )
        hold_torque = np.zeros(positions.shape, dtype=np.float32)
        body_inertia = np.zeros(positions.shape, dtype=np.float32)
        full_mass = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        for index, qpos in enumerate(fixture_qpos):
            self.mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qpos_address] = float(qpos)
            self.mujoco.mj_forward(self.model, self.data)
            self.mujoco.mj_fullM(self.model, full_mass, self.data.qM)
            hold_torque[index] = np.float32(
                self.data.qfrc_bias[self.dof_address]
            )
            body_inertia[index] = np.float32(
                full_mass[self.dof_address, self.dof_address]
                - self.model.dof_armature[self.dof_address]
            )
        if np.any(body_inertia <= 0.0):
            raise ValueError("fixture moving-body inertia must be positive")
        return fixture_qpos.astype(np.float32), hold_torque, body_inertia


def load_mujoco_fixture(
    path: Path,
    *,
    joint_name: str,
    direction: int,
    qpos_offset_rad: float,
) -> MujocoFixtureModel:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required when --fixture-mjcf is supplied"
        ) from exc

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Fixture MJCF not found: {resolved_path}")
    model = mujoco.MjModel.from_xml_path(str(resolved_path))
    if model.njnt != 1 or model.nq != 1 or model.nv != 1:
        raise ValueError(
            "fixture MJCF must contain exactly one fixed-base one-DOF joint; "
            f"got njnt={model.njnt}, nq={model.nq}, nv={model.nv}"
        )
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"fixture joint not found: {joint_name!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError(f"fixture joint {joint_name!r} must be a hinge")

    root_body_ids = [
        body_id
        for body_id in range(1, model.nbody)
        if int(model.body_parentid[body_id]) == 0
    ]
    if len(root_body_ids) != 1:
        raise ValueError(
            f"fixture MJCF must have one root body; found {len(root_body_ids)}"
        )
    root_body_id = root_body_ids[0]
    if int(model.body_jntnum[root_body_id]) != 0:
        raise ValueError("fixture root must be fixed to world and contain no joint")

    moving_body_id = int(model.jnt_bodyid[joint_id])
    moving_subtree_mass_kg = float(model.body_subtreemass[moving_body_id])
    if moving_subtree_mass_kg <= 0.0:
        raise ValueError("fixture moving subtree must have positive mass")

    return MujocoFixtureModel(
        path=resolved_path,
        joint_name=str(joint_name),
        direction=int(direction),
        qpos_offset_rad=float(qpos_offset_rad),
        mujoco=mujoco,
        model=model,
        data=mujoco.MjData(model),
        joint_id=int(joint_id),
        qpos_address=int(model.jnt_qposadr[joint_id]),
        dof_address=int(model.jnt_dofadr[joint_id]),
        root_body_name=str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)
        ),
        moving_body_name=str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, moving_body_id)
        ),
        moving_subtree_mass_kg=moving_subtree_mass_kg,
    )


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def build_profile_segments(
    *,
    center_rad: float,
    amplitudes_rad: Sequence[float],
    sample_hz: float,
    settle_s: float,
    step_hold_s: float,
    chirp_duration_s: float,
    chirp_start_hz: float,
    chirp_end_hz: float,
    chirp_decay_rate: float,
) -> tuple[ProfileSegment, ...]:
    """Build the ToddlerBot-style multi-amplitude step/chirp target profile."""

    def sample_count(duration_s: float) -> int:
        return max(1, int(round(float(duration_s) * float(sample_hz))))

    def hold(name: str, value: float, duration_s: float) -> ProfileSegment:
        return ProfileSegment(
            name=name,
            kind="hold",
            targets_rad=(float(value),) * sample_count(duration_s),
        )

    amplitudes = tuple(sorted({abs(float(value)) for value in amplitudes_rad}))
    if not amplitudes or amplitudes[0] <= 0.0:
        raise ValueError("amplitudes must contain positive values")

    segments: list[ProfileSegment] = [hold("initial_center", center_rad, settle_s)]
    for amplitude in amplitudes:
        label = f"{math.degrees(amplitude):g}deg"
        segments.extend(
            (
                hold(f"step_positive_{label}", center_rad + amplitude, step_hold_s),
                hold(f"settle_after_positive_{label}", center_rad, settle_s),
                hold(f"step_negative_{label}", center_rad - amplitude, step_hold_s),
                hold(f"settle_after_negative_{label}", center_rad, settle_s),
            )
        )

    chirp_count = sample_count(chirp_duration_s)
    t = np.arange(chirp_count, dtype=np.float64) / float(sample_hz)
    frequency_slope = (float(chirp_end_hz) - float(chirp_start_hz)) / float(
        chirp_duration_s
    )
    phase = 2.0 * np.pi * (
        float(chirp_start_hz) * t + 0.5 * frequency_slope * t * t
    )
    envelope = np.exp(-float(chirp_decay_rate) * t)
    for amplitude in amplitudes:
        label = f"{math.degrees(amplitude):g}deg"
        targets = center_rad + amplitude * envelope * np.sin(phase)
        segments.append(
            ProfileSegment(
                name=f"chirp_{label}",
                kind="chirp",
                targets_rad=tuple(float(value) for value in targets),
            )
        )
        segments.append(hold(f"settle_after_chirp_{label}", center_rad, settle_s))
    return tuple(segments)


def validate_profile(
    segments: Sequence[ProfileSegment],
    *,
    servo: ServoConfig,
    joint_limit_margin_rad: float,
    sample_hz: float,
    max_chirp_speed_rad_s: float,
) -> None:
    lower = float(servo.rad_range[0]) + float(joint_limit_margin_rad)
    upper = float(servo.rad_range[1]) - float(joint_limit_margin_rad)
    if lower >= upper:
        raise ValueError("joint-limit margin leaves no usable motion range")
    for segment in segments:
        targets = np.asarray(segment.targets_rad, dtype=np.float64)
        if np.any(targets < lower) or np.any(targets > upper):
            raise ValueError(
                f"segment {segment.name!r} exceeds the guarded joint range "
                f"[{math.degrees(lower):.2f}, {math.degrees(upper):.2f}] deg"
            )
        units = [servo.joint_target_rad_to_elect_unit(value) for value in targets]
        if min(units) <= servo.UNITS_MIN or max(units) >= servo.UNITS_MAX:
            raise ValueError(
                f"segment {segment.name!r} reaches the electrical servo boundary"
            )
        if segment.kind == "chirp" and len(targets) > 1:
            peak_rate = float(np.max(np.abs(np.diff(targets))) * float(sample_hz))
            if peak_rate > float(max_chirp_speed_rad_s):
                raise ValueError(
                    f"segment {segment.name!r} requests {peak_rate:.3f} rad/s, "
                    f"above --max-chirp-speed-rad-s={max_chirp_speed_rad_s:.3f}"
                )


def estimate_delay_metrics(
    target_rad: np.ndarray,
    position_rad: np.ndarray,
    *,
    sample_hz: float,
    max_delay_s: float,
) -> dict[str, object]:
    target = np.asarray(target_rad, dtype=np.float64).reshape(-1)
    position = np.asarray(position_rad, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(position)
    target = target[valid]
    position = position[valid]
    if target.size < 10 or float(np.std(target)) < 1e-8:
        return {
            "delay_samples": None,
            "delay_s": None,
            "delay_interpretation": (
                "cross_correlation_lag_including_servo_response_not_transport_only"
            ),
            "correlation": None,
            "gain": None,
            "fit_rmse_deg": None,
        }

    max_lag = min(
        max(0, int(round(float(max_delay_s) * float(sample_hz)))),
        max(0, target.size // 3),
    )
    best: tuple[float, int, float, float] | None = None
    for lag in range(max_lag + 1):
        x = target[: target.size - lag] if lag else target
        y = position[lag:] if lag else position
        if x.size < 8 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        x_centered = x - float(np.mean(x))
        y_centered = y - float(np.mean(y))
        gain = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
        offset = float(np.mean(y) - gain * np.mean(x))
        rmse = float(np.sqrt(np.mean((y - (gain * x + offset)) ** 2)))
        candidate = (correlation, lag, gain, rmse)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return {
            "delay_samples": None,
            "delay_s": None,
            "delay_interpretation": (
                "cross_correlation_lag_including_servo_response_not_transport_only"
            ),
            "correlation": None,
            "gain": None,
            "fit_rmse_deg": None,
        }
    correlation, lag, gain, rmse = best
    return {
        "delay_samples": int(lag),
        "delay_s": float(lag) / float(sample_hz),
        "delay_interpretation": (
            "cross_correlation_lag_including_servo_response_not_transport_only"
        ),
        "correlation": correlation,
        "gain": gain,
        "fit_rmse_deg": math.degrees(rmse),
    }


def estimate_step_response_metrics(
    target_rad: np.ndarray,
    position_rad: np.ndarray,
    *,
    baseline_position_rad: np.ndarray,
    sample_hz: float,
) -> dict[str, float | None]:
    """Estimate steady gain and 10/50/90% response times for one step."""

    target = np.asarray(target_rad, dtype=np.float64).reshape(-1)
    position = np.asarray(position_rad, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline_position_rad, dtype=np.float64).reshape(-1)
    if target.size == 0 or position.size == 0 or baseline_values.size == 0:
        return {}
    steady_count = min(position.size, max(1, int(round(0.4 * float(sample_hz)))))
    baseline = float(np.mean(baseline_values))
    command = float(np.mean(target))
    steady = float(np.mean(position[-steady_count:]))
    requested_delta = command - baseline
    realized_delta = steady - baseline
    gain = None
    if abs(requested_delta) > 1e-9:
        gain = realized_delta / requested_delta

    crossings: dict[str, float | None] = {
        "response_t10_s": None,
        "response_t50_s": None,
        "response_t90_s": None,
    }
    if abs(realized_delta) > 1e-9:
        fraction = (position - baseline) / realized_delta
        for label, threshold in (("t10", 0.1), ("t50", 0.5), ("t90", 0.9)):
            hits = np.flatnonzero(fraction >= threshold)
            crossings[f"response_{label}_s"] = (
                float(hits[0]) / float(sample_hz) if hits.size else None
            )
    rise_time = None
    if (
        crossings["response_t10_s"] is not None
        and crossings["response_t90_s"] is not None
    ):
        rise_time = float(crossings["response_t90_s"]) - float(
            crossings["response_t10_s"]
        )
    return {
        "baseline_position_deg": math.degrees(baseline),
        "command_deg": math.degrees(command),
        "steady_position_deg": math.degrees(steady),
        "steady_error_deg": math.degrees(command - steady),
        "steady_gain": gain,
        **crossings,
        "response_rise_10_90_s": rise_time,
    }


def summarize_capture(
    arrays: dict[str, np.ndarray],
    *,
    segments: Sequence[ProfileSegment],
    sample_hz: float,
    max_delay_s: float,
) -> dict[str, object]:
    valid = np.asarray(arrays["position_valid"], dtype=bool)
    target = np.asarray(arrays["applied_joint_rad"], dtype=np.float64)
    position = np.asarray(arrays["position_joint_rad"], dtype=np.float64)
    error = target[valid] - position[valid]
    position_elapsed = np.asarray(arrays["position_elapsed_s"], dtype=np.float64)
    sample_period = np.diff(position_elapsed)
    written = np.asarray(arrays["command_written"], dtype=bool)

    def timing_ms(values: np.ndarray) -> dict[str, float | None]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return {"mean": None, "p95": None, "p99": None, "max": None}
        return {
            "mean": 1000.0 * float(np.mean(finite)),
            "p95": 1000.0 * float(np.percentile(finite, 95.0)),
            "p99": 1000.0 * float(np.percentile(finite, 99.0)),
            "max": 1000.0 * float(np.max(finite)),
        }

    summary: dict[str, object] = {
        "samples": int(target.size),
        "valid_position_samples": int(np.sum(valid)),
        "missing_position_samples": int(target.size - np.sum(valid)),
        "command_write_fraction": (
            float(np.mean(arrays["command_written"])) if target.size else 0.0
        ),
        "tracking_rmse_deg": (
            math.degrees(float(np.sqrt(np.mean(error * error)))) if error.size else None
        ),
        "tracking_abs_p95_deg": (
            math.degrees(float(np.percentile(np.abs(error), 95.0)))
            if error.size
            else None
        ),
        "max_loop_lateness_ms": (
            1000.0 * float(np.max(arrays["loop_lateness_s"])) if target.size else None
        ),
        "timing_ms": {
            "sample_period": timing_ms(sample_period),
            "scheduler_wait": timing_ms(arrays["scheduler_wait_s"]),
            "command_write": timing_ms(arrays["command_write_s"][written]),
            "position_read": timing_ms(arrays["position_read_s"]),
            "command_age_at_read": timing_ms(arrays["command_age_at_read_s"]),
        },
        "segments": {},
    }
    segment_indices = np.asarray(arrays["segment_index"], dtype=np.int32)
    per_segment: dict[str, object] = {}
    for index, segment in enumerate(segments):
        mask = segment_indices == index
        segment_valid = mask & valid
        segment_error = target[segment_valid] - position[segment_valid]
        item: dict[str, object] = {
            "kind": segment.kind,
            "samples": int(np.sum(mask)),
            "valid_position_samples": int(np.sum(segment_valid)),
            "tracking_rmse_deg": (
                math.degrees(float(np.sqrt(np.mean(segment_error * segment_error))))
                if segment_error.size
                else None
            ),
            "command_write_fraction": (
                float(np.mean(written[mask])) if np.any(mask) else None
            ),
        }
        segment_write_times = np.asarray(
            arrays["command_write_elapsed_s"][mask], dtype=np.float64
        )
        segment_write_times = segment_write_times[np.isfinite(segment_write_times)]
        item["mean_command_write_interval_s"] = (
            float(np.mean(np.diff(segment_write_times)))
            if segment_write_times.size > 1
            else None
        )
        if (
            segment.kind == "hold"
            and segment.name.startswith("step_")
            and np.any(mask)
        ):
            indices = np.flatnonzero(mask)
            baseline_count = max(1, int(round(0.2 * float(sample_hz))))
            baseline_start = max(0, int(indices[0]) - baseline_count)
            item.update(
                estimate_step_response_metrics(
                    target[mask],
                    position[mask],
                    baseline_position_rad=position[baseline_start : int(indices[0])],
                    sample_hz=sample_hz,
                )
            )
        if segment.kind == "chirp":
            item.update(
                estimate_delay_metrics(
                    target[mask],
                    position[mask],
                    sample_hz=sample_hz,
                    max_delay_s=max_delay_s,
                )
            )
        per_segment[segment.name] = item
    summary["segments"] = per_segment
    return summary


def _empty_capture_arrays() -> dict[str, np.ndarray]:
    return {
        "profile_time_s": np.empty(0, dtype=np.float64),
        "scheduled_elapsed_s": np.empty(0, dtype=np.float64),
        "scheduler_wait_s": np.empty(0, dtype=np.float64),
        "command_elapsed_s": np.empty(0, dtype=np.float64),
        "command_write_elapsed_s": np.empty(0, dtype=np.float64),
        "command_age_at_read_s": np.empty(0, dtype=np.float64),
        "position_elapsed_s": np.empty(0, dtype=np.float64),
        "segment_index": np.empty(0, dtype=np.int32),
        "segment_name": np.empty(0, dtype="U1"),
        "target_joint_rad": np.empty(0, dtype=np.float32),
        "target_servo_units": np.empty(0, dtype=np.int32),
        "applied_joint_rad": np.empty(0, dtype=np.float32),
        "applied_servo_units": np.empty(0, dtype=np.int32),
        "position_joint_rad": np.empty(0, dtype=np.float32),
        "position_servo_units": np.empty(0, dtype=np.int32),
        "position_valid": np.empty(0, dtype=bool),
        "command_written": np.empty(0, dtype=bool),
        "command_write_s": np.empty(0, dtype=np.float64),
        "position_read_s": np.empty(0, dtype=np.float64),
        "loop_lateness_s": np.empty(0, dtype=np.float64),
    }


def _new_capture_fields() -> dict[str, list[object]]:
    return {key: [] for key in _empty_capture_arrays()}


def _arrays_from_fields(fields: dict[str, list[object]]) -> dict[str, np.ndarray]:
    return {
        "profile_time_s": np.asarray(fields["profile_time_s"], dtype=np.float64),
        "scheduled_elapsed_s": np.asarray(
            fields["scheduled_elapsed_s"], dtype=np.float64
        ),
        "scheduler_wait_s": np.asarray(
            fields["scheduler_wait_s"], dtype=np.float64
        ),
        "command_elapsed_s": np.asarray(
            fields["command_elapsed_s"], dtype=np.float64
        ),
        "command_write_elapsed_s": np.asarray(
            fields["command_write_elapsed_s"], dtype=np.float64
        ),
        "command_age_at_read_s": np.asarray(
            fields["command_age_at_read_s"], dtype=np.float64
        ),
        "position_elapsed_s": np.asarray(
            fields["position_elapsed_s"], dtype=np.float64
        ),
        "segment_index": np.asarray(fields["segment_index"], dtype=np.int32),
        "segment_name": np.asarray(fields["segment_name"], dtype="U64"),
        "target_joint_rad": np.asarray(fields["target_joint_rad"], dtype=np.float32),
        "target_servo_units": np.asarray(
            fields["target_servo_units"], dtype=np.int32
        ),
        "applied_joint_rad": np.asarray(
            fields["applied_joint_rad"], dtype=np.float32
        ),
        "applied_servo_units": np.asarray(
            fields["applied_servo_units"], dtype=np.int32
        ),
        "position_joint_rad": np.asarray(
            fields["position_joint_rad"], dtype=np.float32
        ),
        "position_servo_units": np.asarray(
            fields["position_servo_units"], dtype=np.int32
        ),
        "position_valid": np.asarray(fields["position_valid"], dtype=bool),
        "command_written": np.asarray(fields["command_written"], dtype=bool),
        "command_write_s": np.asarray(fields["command_write_s"], dtype=np.float64),
        "position_read_s": np.asarray(fields["position_read_s"], dtype=np.float64),
        "loop_lateness_s": np.asarray(fields["loop_lateness_s"], dtype=np.float64),
    }


def add_standard_capture_arrays(
    arrays: dict[str, np.ndarray],
    *,
    fixture_qpos_rad: np.ndarray,
    estimated_hold_torque_nm: np.ndarray,
    estimated_load_inertia_kg_m2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Add the established tools/sysid trace keys and known-load estimate."""

    timestamps = np.asarray(arrays["position_elapsed_s"], dtype=np.float64)
    position = np.asarray(arrays["position_joint_rad"], dtype=np.float32)
    for name, values in (
        ("fixture_qpos_rad", fixture_qpos_rad),
        ("estimated_hold_torque_nm", estimated_hold_torque_nm),
        ("estimated_load_inertia_kg_m2", estimated_load_inertia_kg_m2),
    ):
        if np.asarray(values).shape != position.shape:
            raise ValueError(f"{name} must match measured position shape")
    velocity = np.zeros(position.shape, dtype=np.float32)
    if position.size > 1:
        dt = np.diff(timestamps)
        valid_dt = dt > 1e-6
        velocity_delta = np.zeros(dt.shape, dtype=np.float32)
        velocity_delta[valid_dt] = (
            np.diff(position)[valid_dt] / dt[valid_dt]
        ).astype(np.float32)
        velocity[1:] = velocity_delta

    return {
        **arrays,
        "timestamps_s": timestamps,
        "command_rad": np.asarray(arrays["applied_joint_rad"], dtype=np.float32),
        "requested_command_rad": np.asarray(
            arrays["target_joint_rad"], dtype=np.float32
        ),
        "measured_position_rad": position,
        "measured_velocity_rad_s": velocity,
        "fixture_qpos_rad": np.asarray(fixture_qpos_rad, dtype=np.float32),
        "estimated_hold_torque_nm": np.asarray(
            estimated_hold_torque_nm, dtype=np.float32
        ),
        "estimated_gravity_torque_nm": np.asarray(
            estimated_hold_torque_nm, dtype=np.float32
        ),
        "estimated_load_inertia_kg_m2": np.asarray(
            estimated_load_inertia_kg_m2, dtype=np.float32
        ),
    }


def evaluate_manual_horizontal_fixture(
    hardware_joint_rad: np.ndarray,
    *,
    center_rad: float,
    signed_load_moment_kg_m: float,
    load_inertia_kg_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(hardware_joint_rad, dtype=np.float64).reshape(-1)
    hold_torque = (
        STANDARD_GRAVITY_M_S2
        * float(signed_load_moment_kg_m)
        * np.cos(positions - float(center_rad))
    ).astype(np.float32)
    inertia = np.full(
        positions.shape, float(load_inertia_kg_m2), dtype=np.float32
    )
    return positions.astype(np.float32), hold_torque, inertia


def _read_position_with_retries(
    bus: RawServoBus,
    *,
    servo_id: int,
    retries: int,
    retry_sleep_s: float,
) -> int:
    for attempt in range(max(1, int(retries))):
        position = bus.read_position(int(servo_id))
        if position is not None:
            return int(position)
        if attempt + 1 < max(1, int(retries)) and retry_sleep_s > 0.0:
            time.sleep(float(retry_sleep_s))
    raise RuntimeError(
        f"failed to read servo {int(servo_id)} position after "
        f"{max(1, int(retries))} attempts"
    )


def monitor_preparation_phase(
    bus: RawServoBus,
    *,
    servo_id: int,
    servo: ServoConfig,
    phase: str,
    attempt: int,
    target_units: int,
    commanded_move_ms: int,
    duration_s: float,
    sample_hz: float,
    preparation_started: float,
    samples: list[dict[str, object]],
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> dict[str, object]:
    """Poll the servo while it is being prepared, stopping on torque loss."""

    phase_started = monotonic_fn()
    sample_count = max(1, int(math.ceil(float(duration_s) * float(sample_hz))) + 1)
    last_sample: dict[str, object] = {}
    for sample_index in range(sample_count):
        scheduled = phase_started + min(
            float(sample_index) / float(sample_hz), float(duration_s)
        )
        remaining = scheduled - monotonic_fn()
        if remaining > 0.0:
            sleep_fn(remaining)
        read_started = monotonic_fn()
        position_units = bus.read_position(int(servo_id))
        loaded = bus.read_loaded(int(servo_id))
        voltage_v = bus.read_voltage_v(int(servo_id))
        temperature_c = bus.read_temperature_c(int(servo_id))
        read_finished = monotonic_fn()
        position_rad = (
            servo.servo_elect_units_to_joint_target_rad(int(position_units))
            if position_units is not None
            else None
        )
        last_sample = {
            "sample_index": len(samples),
            "attempt": int(attempt),
            "phase": str(phase),
            "preparation_elapsed_s": read_finished - preparation_started,
            "phase_elapsed_s": read_finished - phase_started,
            "target_units": int(target_units),
            "commanded_move_ms": int(commanded_move_ms),
            "position_units": (
                int(position_units) if position_units is not None else None
            ),
            "position_deg": (
                math.degrees(float(position_rad))
                if position_rad is not None
                else None
            ),
            "loaded": loaded,
            "voltage_v": voltage_v,
            "temperature_c": temperature_c,
            "read_duration_s": read_finished - read_started,
        }
        samples.append(last_sample)
        if loaded is False:
            break
    return last_sample


def preparation_trace_arrays(
    samples: Sequence[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Convert preparation monitoring samples into stable NPZ fields."""

    def optional_float(name: str) -> np.ndarray:
        return np.asarray(
            [
                float(sample[name]) if sample.get(name) is not None else float("nan")
                for sample in samples
            ],
            dtype=np.float64,
        )

    return {
        "preparation_elapsed_s": optional_float("preparation_elapsed_s"),
        "preparation_phase_elapsed_s": optional_float("phase_elapsed_s"),
        "preparation_attempt": np.asarray(
            [int(sample["attempt"]) for sample in samples], dtype=np.int32
        ),
        "preparation_phase": np.asarray(
            [str(sample["phase"]) for sample in samples], dtype="U32"
        ),
        "preparation_target_servo_units": np.asarray(
            [int(sample["target_units"]) for sample in samples], dtype=np.int32
        ),
        "preparation_commanded_move_ms": np.asarray(
            [int(sample["commanded_move_ms"]) for sample in samples],
            dtype=np.int32,
        ),
        "preparation_position_servo_units": np.asarray(
            [
                int(sample["position_units"])
                if sample.get("position_units") is not None
                else -1
                for sample in samples
            ],
            dtype=np.int32,
        ),
        "preparation_position_joint_rad": np.deg2rad(
            optional_float("position_deg")
        ).astype(np.float32),
        "preparation_loaded_state": np.asarray(
            [
                -1
                if sample.get("loaded") is None
                else int(bool(sample["loaded"]))
                for sample in samples
            ],
            dtype=np.int8,
        ),
        "preparation_voltage_v": optional_float("voltage_v").astype(np.float32),
        "preparation_temperature_c": optional_float("temperature_c").astype(
            np.float32
        ),
        "preparation_read_s": optional_float("read_duration_s"),
    }


def summarize_preparation_trace(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    loaded = np.asarray(arrays["preparation_loaded_state"], dtype=np.int8)
    voltage = np.asarray(arrays["preparation_voltage_v"], dtype=np.float64)
    position = np.asarray(
        arrays["preparation_position_joint_rad"], dtype=np.float64
    )
    torque = np.asarray(
        arrays["preparation_estimated_hold_torque_nm"], dtype=np.float64
    )
    valid_voltage = voltage[np.isfinite(voltage)]
    unload_indices = np.flatnonzero(loaded == 0)
    first_unload: dict[str, object] | None = None
    if unload_indices.size:
        index = int(unload_indices[0])
        first_unload = {
            "sample_index": index,
            "attempt": int(arrays["preparation_attempt"][index]),
            "phase": str(arrays["preparation_phase"][index]),
            "preparation_elapsed_s": float(
                arrays["preparation_elapsed_s"][index]
            ),
            "phase_elapsed_s": float(
                arrays["preparation_phase_elapsed_s"][index]
            ),
            "position_deg": (
                math.degrees(float(position[index]))
                if np.isfinite(position[index])
                else None
            ),
            "estimated_hold_torque_nm": (
                float(torque[index]) if np.isfinite(torque[index]) else None
            ),
            "voltage_v": (
                float(voltage[index]) if np.isfinite(voltage[index]) else None
            ),
            "temperature_c": (
                float(arrays["preparation_temperature_c"][index])
                if np.isfinite(arrays["preparation_temperature_c"][index])
                else None
            ),
        }
    return {
        "samples": int(loaded.size),
        "missing_position_samples": int(np.count_nonzero(~np.isfinite(position))),
        "unknown_load_state_samples": int(np.count_nonzero(loaded < 0)),
        "min_voltage_v": (
            float(np.min(valid_voltage)) if valid_voltage.size else None
        ),
        "max_voltage_v": (
            float(np.max(valid_voltage)) if valid_voltage.size else None
        ),
        "first_unload": first_unload,
    }


def minimum_voltage_sample_below(
    samples: Sequence[dict[str, object]], threshold_v: float
) -> dict[str, object] | None:
    voltage_samples = [
        sample for sample in samples if sample.get("voltage_v") is not None
    ]
    if not voltage_samples:
        return None
    minimum = min(voltage_samples, key=lambda sample: float(sample["voltage_v"]))
    return minimum if float(minimum["voltage_v"]) < float(threshold_v) else None


def capture_profile(
    bus: RawServoBus,
    *,
    servo_id: int,
    servo: ServoConfig,
    segments: Sequence[ProfileSegment],
    sample_hz: float,
    move_time_ms: int,
    write_deadband_units: int,
    max_position_error_rad: float,
    read_retries: int,
    read_retry_sleep_s: float,
    fields: dict[str, list[object]] | None = None,
) -> dict[str, np.ndarray]:
    fields = fields if fields is not None else _new_capture_fields()
    capture_start = time.monotonic()
    nominal_step = 0
    last_written_units: int | None = None
    last_write_done: float | None = None

    for segment_index, segment in enumerate(segments):
        print(
            f"  [{segment_index + 1:02d}/{len(segments):02d}] {segment.name} "
            f"({len(segment.targets_rad) / sample_hz:.1f}s)",
            flush=True,
        )
        segment_start = time.monotonic()
        for local_step, target_rad in enumerate(segment.targets_rad):
            scheduled = segment_start + float(local_step) / float(sample_hz)
            wait_start = time.monotonic()
            remaining = scheduled - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            loop_start = time.monotonic()
            scheduler_wait_s = loop_start - wait_start
            target_units = servo.joint_target_rad_to_elect_unit(float(target_rad))
            should_write = (
                last_written_units is None
                or abs(int(target_units) - int(last_written_units))
                > int(write_deadband_units)
            )
            write_start = time.monotonic()
            if should_write:
                bus.move_time_write(int(servo_id), int(target_units), int(move_time_ms))
                last_written_units = int(target_units)
            write_done = time.monotonic()
            if should_write:
                last_write_done = write_done
            assert last_written_units is not None
            assert last_write_done is not None
            applied_rad = servo.servo_elect_units_to_joint_target_rad(
                int(last_written_units)
            )
            read_start = time.monotonic()
            position_units = _read_position_with_retries(
                bus,
                servo_id=int(servo_id),
                retries=int(read_retries),
                retry_sleep_s=float(read_retry_sleep_s),
            )
            read_done = time.monotonic()
            position_rad = servo.servo_elect_units_to_joint_target_rad(
                int(position_units)
            )

            fields["profile_time_s"].append(float(nominal_step) / float(sample_hz))
            fields["scheduled_elapsed_s"].append(scheduled - capture_start)
            fields["scheduler_wait_s"].append(scheduler_wait_s)
            fields["command_elapsed_s"].append(write_start - capture_start)
            fields["command_write_elapsed_s"].append(
                write_done - capture_start if should_write else float("nan")
            )
            fields["command_age_at_read_s"].append(read_done - last_write_done)
            fields["position_elapsed_s"].append(read_done - capture_start)
            fields["segment_index"].append(segment_index)
            fields["segment_name"].append(segment.name)
            fields["target_joint_rad"].append(float(target_rad))
            fields["target_servo_units"].append(int(target_units))
            fields["applied_joint_rad"].append(float(applied_rad))
            fields["applied_servo_units"].append(int(last_written_units))
            fields["position_joint_rad"].append(position_rad)
            fields["position_servo_units"].append(int(position_units))
            fields["position_valid"].append(True)
            fields["command_written"].append(should_write)
            fields["command_write_s"].append(write_done - write_start)
            fields["position_read_s"].append(read_done - read_start)
            fields["loop_lateness_s"].append(max(0.0, loop_start - scheduled))
            nominal_step += 1

            if abs(float(applied_rad) - position_rad) > float(max_position_error_rad):
                raise RuntimeError(
                    "position tracking error exceeded safety limit: "
                    f"command={math.degrees(float(applied_rad)):+.2f}deg "
                    f"measured={math.degrees(position_rad):+.2f}deg "
                    f"limit={math.degrees(float(max_position_error_rad)):.2f}deg"
                )

        segment_end = segment_start + len(segment.targets_rad) / float(sample_hz)
        remaining = segment_end - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    return _arrays_from_fields(fields)


def read_health(
    bus: RawServoBus,
    *,
    servo_id: int,
    retries: int = 3,
    retry_sleep_s: float = 0.01,
) -> dict[str, float | None]:
    voltage: float | None = None
    temperature: float | None = None
    for attempt in range(max(1, int(retries))):
        if voltage is None:
            voltage = bus.read_voltage_v(int(servo_id))
        if temperature is None:
            temperature = bus.read_temperature_c(int(servo_id))
        if voltage is not None and temperature is not None:
            break
        if attempt + 1 < max(1, int(retries)) and retry_sleep_s > 0.0:
            time.sleep(float(retry_sleep_s))
    return {"voltage_v": voltage, "temperature_c": temperature}


def validate_health(
    health: dict[str, float | None],
    *,
    min_voltage_v: float,
    max_temperature_c: float,
) -> None:
    voltage = health.get("voltage_v")
    temperature = health.get("temperature_c")
    if voltage is None or temperature is None:
        raise RuntimeError(
            "servo voltage/temperature telemetry is unavailable; verify the TTL "
            "read path before running a loaded test"
        )
    if voltage is not None and float(voltage) < float(min_voltage_v):
        raise RuntimeError(
            f"servo voltage {float(voltage):.2f}V is below {float(min_voltage_v):.2f}V"
        )
    if temperature is not None and float(temperature) > float(max_temperature_c):
        raise RuntimeError(
            f"servo temperature {float(temperature):.1f}C exceeds "
            f"{float(max_temperature_c):.1f}C"
        )


def wait_for_cooldown(
    bus: RawServoBus,
    *,
    servo_id: int,
    target_temperature_c: float,
    timeout_s: float,
    poll_s: float,
    min_voltage_v: float,
    max_temperature_c: float,
    samples: list[dict[str, object]] | None = None,
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> tuple[list[dict[str, object]], float]:
    """Wait unloaded until the servo reaches a repeatable start temperature."""

    started = monotonic_fn()
    samples = samples if samples is not None else []
    while True:
        read_started = monotonic_fn()
        health = read_health(bus, servo_id=servo_id)
        read_finished = monotonic_fn()
        validate_health(
            health,
            min_voltage_v=min_voltage_v,
            max_temperature_c=max_temperature_c,
        )
        elapsed_s = read_finished - started
        sample = {
            "phase": "cooldown",
            "cooldown_elapsed_s": elapsed_s,
            "read_duration_s": read_finished - read_started,
            **health,
        }
        samples.append(sample)
        temperature_c = float(health["temperature_c"])
        print(
            f"Cooldown: elapsed={elapsed_s:.1f}s "
            f"temperature={temperature_c:.1f}C "
            f"target<={float(target_temperature_c):.1f}C "
            f"voltage={float(health['voltage_v']):.2f}V",
            flush=True,
        )
        if temperature_c <= float(target_temperature_c):
            return samples, elapsed_s
        if elapsed_s >= float(timeout_s):
            raise RuntimeError(
                f"cooldown timed out after {elapsed_s:.1f}s at "
                f"{temperature_c:.1f}C; target is {float(target_temperature_c):.1f}C"
            )
        sleep_fn(min(float(poll_s), max(0.0, float(timeout_s) - elapsed_s)))


def _verify_loaded_state(
    bus: RawServoBus,
    *,
    servo_id: int,
    expected: bool,
    retries: int = 3,
) -> None:
    state: bool | None = None
    for attempt in range(max(1, int(retries))):
        state = bus.read_loaded(int(servo_id))
        if state is expected:
            return
        if attempt + 1 < max(1, int(retries)):
            time.sleep(0.01)
    raise RuntimeError(
        f"servo torque-state verification failed: expected loaded={expected}, got {state}"
    )


def prepare_servo_center(
    bus: RawServoBus,
    *,
    servo_id: int,
    servo: ServoConfig,
    center_rad: float,
    initial_units: int,
    prepare_speed_deg_s: float,
    move_time_ms: int,
    settle_s: float,
    center_tolerance_deg: float,
    max_attempts: int,
    min_voltage_v: float,
    max_temperature_c: float,
    read_retries: int,
    read_retry_sleep_s: float,
    attempts: list[dict[str, object]],
    phase_label: str = "Center",
    monitor_hz: float = 0.0,
    monitor_samples: list[dict[str, object]] | None = None,
    monitor_started: float | None = None,
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> tuple[int, float, float, float]:
    """Move to the profile center with bounded retries and diagnostics."""

    target_units = servo.joint_target_rad_to_elect_unit(center_rad)
    current_units = int(initial_units)
    current_rad = servo.servo_elect_units_to_joint_target_rad(current_units)
    total_commanded_move_s = 0.0
    movement_started = monotonic_fn()
    preparation_started = (
        movement_started if monitor_started is None else float(monitor_started)
    )
    last_loaded: bool | None = None
    last_unload_sample: dict[str, object] | None = None
    phase_prefix = phase_label.lower().replace("-", "_").replace(" ", "_")
    minimum_prepare_ms = max(
        int(move_time_ms),
        int(
            math.ceil(
                abs(math.degrees(center_rad - current_rad))
                / float(prepare_speed_deg_s)
                * 1000.0
            )
        ),
    )
    minimum_prepare_ms = min(30000, minimum_prepare_ms)

    for attempt_index in range(1, int(max_attempts) + 1):
        loaded_before = bus.read_loaded(servo_id)
        reloaded_before_attempt = loaded_before is not True
        if reloaded_before_attempt:
            bus.move_time_write(
                servo_id,
                current_units,
                max(500, int(move_time_ms)),
            )
            bus.load(servo_id)
            _verify_loaded_state(bus, servo_id=servo_id, expected=True)
            print(
                f"{phase_label} attempt {attempt_index}: reload verified "
                f"at {math.degrees(current_rad):+.2f}deg.",
                flush=True,
            )
        loaded_after_reload = True if reloaded_before_attempt else loaded_before
        starting_error_deg = abs(math.degrees(center_rad - current_rad))
        prepare_ms = max(
            min(30000, minimum_prepare_ms * attempt_index),
            int(
                math.ceil(
                    starting_error_deg / float(prepare_speed_deg_s) * 1000.0
                )
            ),
        )
        prepare_ms = min(30000, prepare_ms)
        total_commanded_move_s += prepare_ms / 1000.0
        attempt_started = monotonic_fn()
        bus.move_time_write(servo_id, target_units, prepare_ms)
        accepted_move = bus.read_move_time(servo_id)
        attempt_sample_start = len(monitor_samples or ())
        final_monitor_sample: dict[str, object] | None = None
        if float(monitor_hz) > 0.0 and monitor_samples is not None:
            final_monitor_sample = monitor_preparation_phase(
                bus,
                servo_id=servo_id,
                servo=servo,
                phase=f"{phase_prefix}_move",
                attempt=attempt_index,
                target_units=target_units,
                commanded_move_ms=prepare_ms,
                duration_s=prepare_ms / 1000.0,
                sample_hz=float(monitor_hz),
                preparation_started=preparation_started,
                samples=monitor_samples,
                monotonic_fn=monotonic_fn,
                sleep_fn=sleep_fn,
            )
            if final_monitor_sample.get("loaded") is not False:
                final_monitor_sample = monitor_preparation_phase(
                    bus,
                    servo_id=servo_id,
                    servo=servo,
                    phase=f"{phase_prefix}_settle",
                    attempt=attempt_index,
                    target_units=target_units,
                    commanded_move_ms=prepare_ms,
                    duration_s=float(settle_s),
                    sample_hz=float(monitor_hz),
                    preparation_started=preparation_started,
                    samples=monitor_samples,
                    monotonic_fn=monotonic_fn,
                    sleep_fn=sleep_fn,
                )
        else:
            sleep_fn(prepare_ms / 1000.0 + float(settle_s))

        monitored_units = (
            final_monitor_sample.get("position_units")
            if final_monitor_sample is not None
            else None
        )
        measured_units = (
            int(monitored_units)
            if monitored_units is not None
            else _read_position_with_retries(
                bus,
                servo_id=servo_id,
                retries=int(read_retries),
                retry_sleep_s=float(read_retry_sleep_s),
            )
        )
        measured_rad = servo.servo_elect_units_to_joint_target_rad(measured_units)
        error_deg = abs(math.degrees(center_rad - measured_rad))
        health = read_health(
            bus,
            servo_id=servo_id,
            retries=int(read_retries),
            retry_sleep_s=float(read_retry_sleep_s),
        )
        monitored_loaded = (
            final_monitor_sample.get("loaded")
            if final_monitor_sample is not None
            else None
        )
        loaded_after = (
            bool(monitored_loaded)
            if monitored_loaded is not None
            else bus.read_loaded(servo_id)
        )
        last_loaded = loaded_after
        attempt_samples = (
            monitor_samples[attempt_sample_start:]
            if monitor_samples is not None
            else []
        )
        attempt_voltages = [
            float(sample["voltage_v"])
            for sample in attempt_samples
            if sample.get("voltage_v") is not None
        ]
        unload_sample = next(
            (sample for sample in attempt_samples if sample.get("loaded") is False),
            None,
        )
        if unload_sample is not None:
            last_unload_sample = unload_sample
        diagnostic = {
            "attempt": attempt_index,
            "start_position_units": current_units,
            "start_position_deg": math.degrees(current_rad),
            "requested_target_units": target_units,
            "requested_target_deg": math.degrees(center_rad),
            "commanded_move_ms": prepare_ms,
            "accepted_target_units": (
                int(accepted_move[0]) if accepted_move is not None else None
            ),
            "accepted_move_ms": (
                int(accepted_move[1]) if accepted_move is not None else None
            ),
            "measured_position_units": measured_units,
            "measured_position_deg": math.degrees(measured_rad),
            "error_deg": error_deg,
            "progress_deg": starting_error_deg - error_deg,
            "loaded_before": loaded_before,
            "reloaded_before_attempt": reloaded_before_attempt,
            "loaded_after_reload": loaded_after_reload,
            "loaded_after": loaded_after,
            "voltage_v": health["voltage_v"],
            "temperature_c": health["temperature_c"],
            "monitor_samples": len(attempt_samples),
            "monitor_min_voltage_v": (
                min(attempt_voltages) if attempt_voltages else None
            ),
            "unload_detected_phase": (
                str(unload_sample["phase"]) if unload_sample is not None else None
            ),
            "unload_detected_phase_elapsed_s": (
                float(unload_sample["phase_elapsed_s"])
                if unload_sample is not None
                else None
            ),
            "attempt_elapsed_s": monotonic_fn() - attempt_started,
        }
        attempts.append(diagnostic)
        accepted_text = (
            "unavailable"
            if accepted_move is None
            else f"{accepted_move[0]} units/{accepted_move[1]}ms"
        )
        print(
            f"{phase_label} attempt {attempt_index}/{int(max_attempts)}: "
            f"requested={target_units} units accepted={accepted_text} "
            f"measured={measured_units} units "
            f"({math.degrees(measured_rad):+.2f}deg) "
            f"error={error_deg:.2f}deg "
            f"progress={starting_error_deg - error_deg:+.2f}deg "
            f"loaded={loaded_after} {_format_health(health)}",
            flush=True,
        )
        validate_health(
            health,
            min_voltage_v=float(min_voltage_v),
            max_temperature_c=float(max_temperature_c),
        )
        if error_deg <= float(center_tolerance_deg) and loaded_after is True:
            return (
                measured_units,
                measured_rad,
                total_commanded_move_s,
                monotonic_fn() - movement_started,
            )
        current_units = measured_units
        current_rad = measured_rad

    if (
        abs(math.degrees(center_rad - current_rad))
        <= float(center_tolerance_deg)
        and last_loaded is not True
    ):
        raise RuntimeError(
            f"servo reached {phase_label.lower()} but could not remain loaded "
            f"after {int(max_attempts)} attempts"
        )
    if last_unload_sample is not None:
        position_text = (
            "unknown"
            if last_unload_sample.get("position_deg") is None
            else f"{float(last_unload_sample['position_deg']):+.2f}deg"
        )
        voltage_text = (
            "unknown"
            if last_unload_sample.get("voltage_v") is None
            else f"{float(last_unload_sample['voltage_v']):.3f}V"
        )
        raise RuntimeError(
            f"servo unloaded during {last_unload_sample['phase']} "
            f"at {position_text}, phase_elapsed="
            f"{float(last_unload_sample['phase_elapsed_s']):.3f}s, "
            f"voltage={voltage_text}"
        )
    raise RuntimeError(
        f"servo did not reach {phase_label.lower()} after "
        f"{int(max_attempts)} attempts: "
        f"measured={math.degrees(current_rad):+.2f}deg "
        f"target={math.degrees(center_rad):+.2f}deg "
        f"error={abs(math.degrees(center_rad - current_rad)):.2f}deg"
    )


def _default_output_path(servo_id: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        _REPO_ROOT
        / "runtime"
        / "calibration"
        / "servo_sysid"
        / f"htd45h_servo_{int(servo_id)}_{timestamp}.npz"
    )


def write_capture(
    output_path: Path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    npz_path = output_path.expanduser().resolve()
    if npz_path.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    json_path = npz_path.with_suffix(".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite existing capture: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        **arrays,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        **metadata,
        "summary": summary,
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one HTD-45H servo's known-load step/chirp response."
    )
    parser.add_argument(
        "--servo-id",
        type=int,
        required=True,
        help="Raw HTD servo ID on the isolated fixture bus.",
    )
    parser.add_argument(
        "--board-port",
        "--board_port",
        dest="board_port",
        required=True,
        help="Serial port for the TTL board connected to the fixture servo.",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--center-deg", type=float, required=True)
    parser.add_argument(
        "--fixture-mjcf",
        type=Path,
        default=None,
        help="Fixed-base one-DOF fixture MJCF used for load torque and inertia.",
    )
    parser.add_argument("--fixture-joint", default="pitch")
    parser.add_argument(
        "--fixture-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Map raw centered servo radians to the MJCF joint direction.",
    )
    parser.add_argument(
        "--fixture-qpos-offset-deg",
        type=float,
        default=0.0,
        help="MJCF joint position corresponding to raw servo unit 500.",
    )
    parser.add_argument("--lever-arm-m", type=float, default=None)
    parser.add_argument("--load-mass-kg", type=float, default=None)
    parser.add_argument("--lever-mass-kg", type=float, default=0.0)
    parser.add_argument("--lever-com-m", type=float, default=0.0)
    parser.add_argument(
        "--lever-inertia-kg-m2",
        type=float,
        default=None,
        help=(
            "Lever inertia about the servo shaft. If omitted, a uniform lever "
            "with length 2*--lever-com-m is assumed."
        ),
    )
    parser.add_argument(
        "--gravity-torque-sign",
        type=int,
        choices=(-1, 1),
        default=None,
        help="Sign of gravity torque in raw centered servo coordinates at center.",
    )
    parser.add_argument(
        "--amplitudes-deg",
        type=parse_float_list,
        default=DEFAULT_AMPLITUDES_DEG,
    )
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--step-hold-s", type=float, default=1.0)
    parser.add_argument("--chirp-duration-s", type=float, default=10.0)
    parser.add_argument("--chirp-start-hz", type=float, default=0.1)
    parser.add_argument("--chirp-end-hz", type=float, default=2.0)
    parser.add_argument("--chirp-decay-rate", type=float, default=0.05)
    parser.add_argument("--move-time-ms", type=int, default=20)
    parser.add_argument("--write-deadband-units", type=int, default=3)
    parser.add_argument("--prepare-speed-deg-s", type=float, default=20.0)
    parser.add_argument(
        "--prepare-monitor-hz",
        type=float,
        default=25.0,
        help="Polling rate for position/load/voltage during preparation moves.",
    )
    parser.add_argument(
        "--pre-move-hold-s",
        type=float,
        default=1.0,
        help="Verified loaded hold before moving toward the requested center.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after center verification and safe return; skip step/chirp capture.",
    )
    parser.add_argument("--center-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--center-max-attempts",
        type=int,
        default=3,
        help="Maximum automatic attempts to reach the requested center pose.",
    )
    parser.add_argument(
        "--startup-delay-s",
        type=float,
        default=3.0,
        help="Cancellable delay before opening the servo bus and applying torque.",
    )
    parser.add_argument(
        "--unload-pose-deg",
        type=float,
        default=0.0,
        help="Gravity-neutral fixture pose reached and verified before torque-off.",
    )
    parser.add_argument(
        "--max-unload-static-torque-nm",
        type=float,
        default=0.05,
        help="Maximum modeled gravity torque allowed at automatic torque-off.",
    )
    parser.add_argument("--servo-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--max-chirp-speed-rad-s", type=float, default=2.5)
    parser.add_argument("--max-position-error-deg", type=float, default=12.0)
    parser.add_argument("--max-static-torque-nm", type=float, default=2.5)
    parser.add_argument(
        "--min-voltage-v",
        type=float,
        default=9.6,
        help="Minimum HTD-45H operating voltage from the vendor specification.",
    )
    parser.add_argument("--max-temperature-c", type=float, default=60.0)
    parser.add_argument(
        "--cooldown-target-c",
        type=float,
        default=35.0,
        help="Wait unloaded until the servo is at or below this temperature.",
    )
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--cooldown-poll-s", type=float, default=5.0)
    parser.add_argument("--read-retries", type=int, default=3)
    parser.add_argument("--read-retry-sleep-s", type=float, default=0.002)
    parser.add_argument("--max-delay-s", type=float, default=0.6)
    parser.add_argument("--servo-label", default=None)
    parser.add_argument("--fixture-label", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.servo_id) < 1 or int(args.servo_id) > 253:
        raise ValueError("--servo-id must be between 1 and 253")
    if int(args.baudrate) <= 0:
        raise ValueError("--baudrate must be positive")
    positive = {
        "sample_hz": args.sample_hz,
        "settle_s": args.settle_s,
        "step_hold_s": args.step_hold_s,
        "chirp_duration_s": args.chirp_duration_s,
        "chirp_start_hz": args.chirp_start_hz,
        "chirp_end_hz": args.chirp_end_hz,
        "max_chirp_speed_rad_s": args.max_chirp_speed_rad_s,
        "max_position_error_deg": args.max_position_error_deg,
        "max_static_torque_nm": args.max_static_torque_nm,
        "prepare_speed_deg_s": args.prepare_speed_deg_s,
        "prepare_monitor_hz": args.prepare_monitor_hz,
    }
    for name, value in positive.items():
        if float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.chirp_end_hz < args.chirp_start_hz:
        raise ValueError("--chirp-end-hz must be >= --chirp-start-hz")
    if args.fixture_mjcf is not None:
        manual_values = (
            args.lever_arm_m,
            args.load_mass_kg,
            args.gravity_torque_sign,
        )
        if any(value is not None for value in manual_values) or any(
            value not in (None, 0.0)
            for value in (
                args.lever_mass_kg,
                args.lever_com_m,
                args.lever_inertia_kg_m2,
            )
        ):
            raise ValueError(
                "--fixture-mjcf cannot be combined with manual lever/load options"
            )
    else:
        if (
            args.lever_arm_m is None
            or args.load_mass_kg is None
            or args.gravity_torque_sign is None
        ):
            raise ValueError(
                "manual fixture mode requires --lever-arm-m, --load-mass-kg, "
                "and --gravity-torque-sign"
            )
        if args.load_mass_kg < 0.0 or args.lever_mass_kg < 0.0:
            raise ValueError("load and lever masses must be non-negative")
        if args.lever_arm_m <= 0.0 or args.lever_com_m < 0.0:
            raise ValueError(
                "lever distances must be non-negative and lever arm must be positive"
            )
        if args.lever_mass_kg > 0.0 and args.lever_com_m <= 0.0:
            raise ValueError(
                "--lever-com-m must be positive when --lever-mass-kg is non-zero"
            )
        if (
            args.lever_inertia_kg_m2 is not None
            and args.lever_inertia_kg_m2 < 0.0
        ):
            raise ValueError("--lever-inertia-kg-m2 must be non-negative")
    if args.move_time_ms < 0 or args.write_deadband_units < 0:
        raise ValueError("move time and write deadband must be non-negative")
    if args.center_max_attempts <= 0:
        raise ValueError("--center-max-attempts must be positive")
    if args.startup_delay_s < 0.0:
        raise ValueError("--startup-delay-s must be non-negative")
    if args.pre_move_hold_s < 0.0:
        raise ValueError("--pre-move-hold-s must be non-negative")
    if args.max_unload_static_torque_nm < 0.0:
        raise ValueError("--max-unload-static-torque-nm must be non-negative")
    if args.read_retries <= 0 or args.read_retry_sleep_s < 0.0:
        raise ValueError("read retries must be positive and retry sleep non-negative")
    if args.cooldown_target_c <= 0.0:
        raise ValueError("--cooldown-target-c must be positive")
    if args.cooldown_target_c > args.max_temperature_c:
        raise ValueError("--cooldown-target-c must not exceed --max-temperature-c")
    if args.cooldown_timeout_s < 0.0 or args.cooldown_poll_s <= 0.0:
        raise ValueError("cooldown timeout must be non-negative and poll must be positive")


def _format_health(health: dict[str, float | None]) -> str:
    voltage = health.get("voltage_v")
    temperature = health.get("temperature_c")
    voltage_text = "unsupported" if voltage is None else f"{float(voltage):.2f}V"
    temperature_text = (
        "unsupported" if temperature is None else f"{float(temperature):.1f}C"
    )
    return f"voltage={voltage_text} temperature={temperature_text}"


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    servo_id = int(args.servo_id)
    port = str(args.board_port)
    servo = build_fixture_servo_config(servo_id)
    output_path = args.output or _default_output_path(servo_id)
    center_rad = math.radians(float(args.center_deg))
    unload_pose_rad = math.radians(float(args.unload_pose_deg))
    lower_rad, upper_rad = servo.rad_range
    limit_margin_rad = math.radians(float(args.servo_limit_margin_deg))
    if not (
        lower_rad + limit_margin_rad
        <= unload_pose_rad
        <= upper_rad - limit_margin_rad
    ):
        raise ValueError("--unload-pose-deg is outside the guarded servo range")
    amplitudes_rad = tuple(math.radians(value) for value in args.amplitudes_deg)
    segments = build_profile_segments(
        center_rad=center_rad,
        amplitudes_rad=amplitudes_rad,
        sample_hz=float(args.sample_hz),
        settle_s=float(args.settle_s),
        step_hold_s=float(args.step_hold_s),
        chirp_duration_s=float(args.chirp_duration_s),
        chirp_start_hz=float(args.chirp_start_hz),
        chirp_end_hz=float(args.chirp_end_hz),
        chirp_decay_rate=float(args.chirp_decay_rate),
    )
    validate_profile(
        segments,
        servo=servo,
        joint_limit_margin_rad=math.radians(float(args.servo_limit_margin_deg)),
        sample_hz=float(args.sample_hz),
        max_chirp_speed_rad_s=float(args.max_chirp_speed_rad_s),
    )

    profile_targets = np.asarray(
        [target for segment in segments for target in segment.targets_rad],
        dtype=np.float64,
    )
    fixture_model: MujocoFixtureModel | None = None
    if args.fixture_mjcf is not None:
        fixture_model = load_mujoco_fixture(
            args.fixture_mjcf,
            joint_name=str(args.fixture_joint),
            direction=int(args.fixture_direction),
            qpos_offset_rad=math.radians(float(args.fixture_qpos_offset_deg)),
        )
        _, profile_hold_torque, _ = fixture_model.evaluate(
            profile_targets
        )
        _, center_hold_torque, center_inertia = fixture_model.evaluate(
            np.asarray([center_rad], dtype=np.float64)
        )
        _, unload_hold_torque, _ = fixture_model.evaluate(
            np.asarray([unload_pose_rad], dtype=np.float64)
        )
        static_torque_at_center_nm = float(center_hold_torque[0])
        load_inertia_at_center_kg_m2 = float(center_inertia[0])
        load_metadata: dict[str, object] = {
            "load_model": "mujoco_fixture",
            "fixture_mjcf": str(fixture_model.path),
            "fixture_mjcf_sha256": hashlib.sha256(
                fixture_model.path.read_bytes()
            ).hexdigest(),
            "fixture_joint": fixture_model.joint_name,
            "fixture_direction": fixture_model.direction,
            "fixture_qpos_offset_deg": float(args.fixture_qpos_offset_deg),
            "fixture_root_body": fixture_model.root_body_name,
            "fixture_moving_body": fixture_model.moving_body_name,
            "fixture_moving_subtree_mass_kg": (
                fixture_model.moving_subtree_mass_kg
            ),
            "fixture_excluded_mjcf_armature_kg_m2": float(
                fixture_model.model.dof_armature[fixture_model.dof_address]
            ),
        }
    else:
        assert args.load_mass_kg is not None
        assert args.lever_arm_m is not None
        assert args.gravity_torque_sign is not None
        load_moment_kg_m = (
            float(args.load_mass_kg) * float(args.lever_arm_m)
            + float(args.lever_mass_kg) * float(args.lever_com_m)
        )
        signed_load_moment_kg_m = int(args.gravity_torque_sign) * load_moment_kg_m
        if args.lever_inertia_kg_m2 is None:
            lever_inertia_kg_m2 = (
                (4.0 / 3.0)
                * float(args.lever_mass_kg)
                * float(args.lever_com_m) ** 2
            )
            lever_inertia_source = "uniform_lever_from_mass_and_com"
        else:
            lever_inertia_kg_m2 = float(args.lever_inertia_kg_m2)
            lever_inertia_source = "command_line"
        total_load_inertia_kg_m2 = (
            float(args.load_mass_kg) * float(args.lever_arm_m) ** 2
            + lever_inertia_kg_m2
        )
        _, profile_hold_torque, _ = (
            evaluate_manual_horizontal_fixture(
                profile_targets,
                center_rad=center_rad,
                signed_load_moment_kg_m=signed_load_moment_kg_m,
                load_inertia_kg_m2=total_load_inertia_kg_m2,
            )
        )
        static_torque_at_center_nm = float(
            STANDARD_GRAVITY_M_S2 * signed_load_moment_kg_m
        )
        _, unload_hold_torque, _ = evaluate_manual_horizontal_fixture(
            np.asarray([unload_pose_rad], dtype=np.float64),
            center_rad=center_rad,
            signed_load_moment_kg_m=signed_load_moment_kg_m,
            load_inertia_kg_m2=total_load_inertia_kg_m2,
        )
        load_inertia_at_center_kg_m2 = total_load_inertia_kg_m2
        load_metadata = {
            "load_model": "manual_horizontal_lever",
            "lever_arm_m": float(args.lever_arm_m),
            "load_mass_kg": float(args.load_mass_kg),
            "lever_mass_kg": float(args.lever_mass_kg),
            "lever_com_m": float(args.lever_com_m),
            "lever_inertia_kg_m2": lever_inertia_kg_m2,
            "lever_inertia_source": lever_inertia_source,
            "gravity_torque_sign": int(args.gravity_torque_sign),
        }

    static_torque_at_unload_pose_nm = float(unload_hold_torque[0])
    if abs(static_torque_at_unload_pose_nm) > float(
        args.max_unload_static_torque_nm
    ):
        raise ValueError(
            f"automatic unload pose has {static_torque_at_unload_pose_nm:+.3f}Nm "
            "modeled gravity torque, exceeding "
            f"--max-unload-static-torque-nm="
            f"{float(args.max_unload_static_torque_nm):.3f}"
        )

    max_profile_hold_torque_nm = (
        float(np.max(np.abs(profile_hold_torque)))
        if profile_hold_torque.size
        else 0.0
    )
    if max_profile_hold_torque_nm > float(args.max_static_torque_nm):
        raise ValueError(
            f"fixture profile reaches {max_profile_hold_torque_nm:.3f}Nm, exceeding "
            f"--max-static-torque-nm={float(args.max_static_torque_nm):.3f}"
        )
    sample_count = sum(len(segment.targets_rad) for segment in segments)
    duration_s = sample_count / float(args.sample_hz)

    print("HTD-45H single-servo SysID capture")
    print(f"  servo_id={servo_id} board_port={port}")
    print(
        "  coordinate=raw servo angle "
        "(unit 500 = 0deg, increasing units = positive)"
    )
    print(
        f"  center={float(args.center_deg):+.2f}deg "
        f"amplitudes={list(float(v) for v in args.amplitudes_deg)}deg"
    )
    if args.prepare_only:
        print("  mode=preparation-only diagnostic (step/chirp profile disabled)")
    else:
        print(
            f"  profile={duration_s:.1f}s at {float(args.sample_hz):.1f}Hz "
            f"move_time_ms={int(args.move_time_ms)} "
            f"deadband_units={int(args.write_deadband_units)}"
        )
    print(
        f"  preparation_monitor={float(args.prepare_monitor_hz):.1f}Hz "
        f"pre_move_hold={float(args.pre_move_hold_s):.1f}s "
        f"prepare_speed={float(args.prepare_speed_deg_s):.1f}deg/s"
    )
    print(
        f"  cooldown=unloaded to <= {float(args.cooldown_target_c):.1f}C "
        f"(poll={float(args.cooldown_poll_s):.1f}s, "
        f"timeout={float(args.cooldown_timeout_s):.1f}s)"
    )
    print(
        f"  automatic_start_delay={float(args.startup_delay_s):.1f}s "
        f"automatic_unload_pose={float(args.unload_pose_deg):+.2f}deg "
        f"unload_hold_torque={static_torque_at_unload_pose_nm:+.3f}Nm"
    )
    if fixture_model is not None:
        print(
            f"  fixture={fixture_model.path} joint={fixture_model.joint_name} "
            f"root={fixture_model.root_body_name} "
            f"moving_body={fixture_model.moving_body_name}"
        )
        print(
            f"  moving_mass={fixture_model.moving_subtree_mass_kg:.6f}kg "
            f"body_inertia={load_inertia_at_center_kg_m2:.6f}kg*m^2 "
            f"center_hold_torque={static_torque_at_center_nm:+.3f}Nm "
            f"profile_peak_hold_torque={max_profile_hold_torque_nm:.3f}Nm"
        )
    else:
        print(
            f"  load={float(args.load_mass_kg):.3f}kg at "
            f"{float(args.lever_arm_m):.3f}m "
            f"lever={float(args.lever_mass_kg):.3f}kg at "
            f"{float(args.lever_com_m):.3f}m "
            f"static_torque_at_center={static_torque_at_center_nm:+.3f}Nm"
        )
        print(
            f"  estimated_load_inertia={load_inertia_at_center_kg_m2:.6f}kg*m^2"
        )
    print(f"  output={output_path.expanduser().resolve()}")
    if fixture_model is not None:
        fixture_center_deg = (
            float(args.fixture_qpos_offset_deg)
            + int(args.fixture_direction) * float(args.center_deg)
        )
        print(
            "  fixture requirement: rigid fixed top, physical lever matching "
            f"MJCF pitch={fixture_center_deg:+.2f}deg at center, verified clear "
            "travel, load catcher, and reachable power cutoff"
        )
    else:
        print(
            "  fixture requirement: rigid servo mount, lever horizontal at center, "
            "verified clear travel, load catcher, and reachable power cutoff"
        )
    if args.dry_run:
        print("Dry run complete; no serial port was opened and no files were written.")
        return 0

    invocation_wall = datetime.now().astimezone().isoformat()
    startup_delay_started = time.monotonic()
    if float(args.startup_delay_s) > 0.0:
        print(
            "Automatic start: clear the fixture now; opening the servo bus in "
            f"{float(args.startup_delay_s):.1f}s. Press Ctrl-C to abort.",
            flush=True,
        )
        time.sleep(float(args.startup_delay_s))
    startup_delay_wait_s = time.monotonic() - startup_delay_started
    run_confirmation_wait_s = 0.0

    capture_fields = _new_capture_fields()
    preparation_samples: list[dict[str, object]] = []
    health_samples: list[dict[str, object]] = []
    cooldown_samples: list[dict[str, object]] = []
    outcome = "failed"
    error: str | None = None
    bus: RawServoBus | None = None
    exit_code = 1
    start_wall = datetime.now().astimezone().isoformat()
    operation_started = time.monotonic()
    cooldown_wait_s: float | None = None
    initial_loaded_state: bool | None = None
    final_unloaded_verified = False
    initial_units: int | None = None
    initial_rad: float | None = None
    centered_units: int | None = None
    centered_rad: float | None = None
    final_center_units: int | None = None
    final_center_rad: float | None = None
    prepare_move_s: float | None = None
    prepare_wait_s: float | None = None
    center_confirmation_wait_s = 0.0
    profile_duration_s: float | None = None
    return_to_center_wait_s: float | None = None
    unload_confirmation_wait_s = 0.0
    servo_angle_limits_units: tuple[int, int] | None = None
    servo_voltage_limits_v: tuple[float, float] | None = None
    servo_temperature_limit_c: int | None = None
    servo_motor_mode: tuple[int, int] | None = None
    servo_led_enabled: bool | None = None
    servo_alarm_mask: int | None = None
    initial_move_target: tuple[int, int] | None = None
    preparation_attempts: list[dict[str, object]] = []
    unload_pose_attempts: list[dict[str, object]] = []
    unload_pose_units: int | None = None
    unload_pose_position_rad: float | None = None
    unload_pose_move_s: float | None = None
    unload_pose_wait_s: float | None = None
    failure_diagnostics: dict[str, object] | None = None
    preparation_monitor_started: float | None = None
    preparation_voltage_violation: dict[str, object] | None = None
    try:
        transport = SerialTransport(
            SerialTransportConfig(
                port=port,
                baudrate=int(args.baudrate),
            )
        )
        bus = RawServoBus(transport, RawServoBusConfig())
        reported_id = bus.read_id(servo_id)
        if reported_id != servo_id:
            raise RuntimeError(
                f"servo ID probe failed on {port}: expected {servo_id}, got {reported_id}"
            )
        servo_angle_limits_units = bus.read_angle_limits(servo_id)
        servo_voltage_limits_v = bus.read_voltage_limits_v(servo_id)
        servo_temperature_limit_c = bus.read_temperature_limit_c(servo_id)
        servo_motor_mode = bus.read_motor_mode(servo_id)
        servo_led_enabled = bus.read_led_enabled(servo_id)
        servo_alarm_mask = bus.read_alarm_mask(servo_id)
        initial_move_target = bus.read_move_time(servo_id)
        if servo_angle_limits_units is None:
            print("Servo diagnostics: EEPROM angle limits unavailable")
        else:
            print(
                "Servo diagnostics: EEPROM angle limits="
                f"{servo_angle_limits_units[0]}..{servo_angle_limits_units[1]} units"
            )
            profile_servo_units = [
                servo.joint_target_rad_to_elect_unit(float(target))
                for target in profile_targets
            ]
            profile_min_units = min(profile_servo_units)
            profile_max_units = max(profile_servo_units)
            if (
                profile_min_units < servo_angle_limits_units[0]
                or profile_max_units > servo_angle_limits_units[1]
            ):
                raise RuntimeError(
                    "profile exceeds EEPROM angle limits: "
                    f"profile={profile_min_units}..{profile_max_units} units "
                    f"limits={servo_angle_limits_units[0]}.."
                    f"{servo_angle_limits_units[1]} units"
                )
        if initial_move_target is None:
            print("Servo diagnostics: active move target unavailable")
        else:
            print(
                "Servo diagnostics: active move target="
                f"{initial_move_target[0]} units/{initial_move_target[1]}ms"
            )
        print(
            "Servo diagnostics: "
            f"voltage_limits={servo_voltage_limits_v}V "
            f"temperature_limit={servo_temperature_limit_c}C "
            f"motor_mode={servo_motor_mode} "
            f"led_enabled={servo_led_enabled} alarm_mask={servo_alarm_mask}"
        )
        if servo_motor_mode is not None and servo_motor_mode[0] != 0:
            raise RuntimeError(
                f"servo is in motor mode {servo_motor_mode}; position mode is required"
            )
        initial_loaded_state = bus.read_loaded(servo_id)
        initial_units = _read_position_with_retries(
            bus,
            servo_id=servo_id,
            retries=int(args.read_retries),
            retry_sleep_s=float(args.read_retry_sleep_s),
        )
        initial_rad = servo.servo_elect_units_to_joint_target_rad(initial_units)
        print(
            f"Initial position={math.degrees(initial_rad):+.2f}deg "
            f"loaded={initial_loaded_state}"
        )

        bus.unload(servo_id)
        _verify_loaded_state(bus, servo_id=servo_id, expected=False)
        cooldown_started = time.monotonic()
        try:
            cooldown_samples, _ = wait_for_cooldown(
                bus,
                servo_id=servo_id,
                target_temperature_c=float(args.cooldown_target_c),
                timeout_s=float(args.cooldown_timeout_s),
                poll_s=float(args.cooldown_poll_s),
                min_voltage_v=float(args.min_voltage_v),
                max_temperature_c=float(args.max_temperature_c),
                samples=cooldown_samples,
            )
        finally:
            cooldown_wait_s = time.monotonic() - cooldown_started
        health_samples.extend(cooldown_samples)
        health_before = {
            "voltage_v": cooldown_samples[-1]["voltage_v"],
            "temperature_c": cooldown_samples[-1]["temperature_c"],
        }
        if servo_voltage_limits_v is not None:
            voltage_v = float(health_before["voltage_v"])
            if not (
                servo_voltage_limits_v[0]
                <= voltage_v
                <= servo_voltage_limits_v[1]
            ):
                raise RuntimeError(
                    f"servo voltage {voltage_v:.3f}V is outside its EEPROM limits "
                    f"[{servo_voltage_limits_v[0]:.3f}, "
                    f"{servo_voltage_limits_v[1]:.3f}]V"
                )
        if (
            servo_temperature_limit_c is not None
            and float(health_before["temperature_c"])
            > float(servo_temperature_limit_c)
        ):
            raise RuntimeError(
                f"servo temperature {float(health_before['temperature_c']):.1f}C "
                f"exceeds its EEPROM limit {servo_temperature_limit_c}C"
            )
        print(
            f"Starting profile preparation after {cooldown_wait_s:.1f}s cooldown; "
            f"{_format_health(health_before)}"
        )

        bus.move_time_write(
            servo_id,
            int(initial_units),
            max(500, int(args.move_time_ms)),
        )
        bus.load(servo_id)
        _verify_loaded_state(bus, servo_id=servo_id, expected=True)
        preparation_monitor_started = time.monotonic()
        print(
            f"Initial load verified; monitoring a "
            f"{float(args.pre_move_hold_s):.1f}s hold before motion.",
            flush=True,
        )
        prepare_initial_units = int(initial_units)
        if float(args.pre_move_hold_s) > 0.0:
            hold_sample = monitor_preparation_phase(
                bus,
                servo_id=servo_id,
                servo=servo,
                phase="post_load_hold",
                attempt=0,
                target_units=int(initial_units),
                commanded_move_ms=max(500, int(args.move_time_ms)),
                duration_s=float(args.pre_move_hold_s),
                sample_hz=float(args.prepare_monitor_hz),
                preparation_started=preparation_monitor_started,
                samples=preparation_samples,
            )
            if hold_sample.get("position_units") is not None:
                prepare_initial_units = int(hold_sample["position_units"])
            if hold_sample.get("loaded") is False:
                position_text = (
                    "unknown"
                    if hold_sample.get("position_deg") is None
                    else f"{float(hold_sample['position_deg']):+.2f}deg"
                )
                raise RuntimeError(
                    "servo unloaded during post_load_hold before any center "
                    f"motion at {position_text}, phase_elapsed="
                    f"{float(hold_sample['phase_elapsed_s']):.3f}s"
                )
            hold_voltage_violation = minimum_voltage_sample_below(
                preparation_samples, float(args.min_voltage_v)
            )
            if hold_voltage_violation is not None:
                raise RuntimeError(
                    "servo voltage dropped below the safety floor during "
                    f"{hold_voltage_violation['phase']}: "
                    f"{float(hold_voltage_violation['voltage_v']):.3f}V < "
                    f"{float(args.min_voltage_v):.3f}V"
                )
            print(
                "Post-load hold passed; torque remained enabled before motion.",
                flush=True,
            )
        print(
            f"Moving to fixture center with up to "
            f"{int(args.center_max_attempts)} attempts..."
        )
        (
            centered_units,
            centered_rad,
            prepare_move_s,
            prepare_wait_s,
        ) = prepare_servo_center(
            bus,
            servo_id=servo_id,
            servo=servo,
            center_rad=center_rad,
            initial_units=prepare_initial_units,
            prepare_speed_deg_s=float(args.prepare_speed_deg_s),
            move_time_ms=int(args.move_time_ms),
            settle_s=float(args.settle_s),
            center_tolerance_deg=float(args.center_tolerance_deg),
            max_attempts=int(args.center_max_attempts),
            min_voltage_v=float(args.min_voltage_v),
            max_temperature_c=float(args.max_temperature_c),
            read_retries=int(args.read_retries),
            read_retry_sleep_s=float(args.read_retry_sleep_s),
            attempts=preparation_attempts,
            monitor_hz=float(args.prepare_monitor_hz),
            monitor_samples=preparation_samples,
            monitor_started=preparation_monitor_started,
        )
        print(
            f"Center verified automatically at "
            f"{math.degrees(centered_rad):+.2f}deg."
        )

        center_health = read_health(bus, servo_id=servo_id)
        validate_health(
            center_health,
            min_voltage_v=float(args.min_voltage_v),
            max_temperature_c=float(args.max_temperature_c),
        )
        preparation_voltage_violation = minimum_voltage_sample_below(
            preparation_samples, float(args.min_voltage_v)
        )
        if args.prepare_only or preparation_voltage_violation is not None:
            final_center_units = centered_units
            final_center_rad = centered_rad
            health_samples.append(
                {
                    "phase": "center_verified",
                    "operation_elapsed_s": time.monotonic() - operation_started,
                    **center_health,
                }
            )
            if args.prepare_only:
                print(
                    "Preparation-only diagnostic complete; skipping step/chirp "
                    "profile.",
                    flush=True,
                )
            else:
                print(
                    _yellow(
                        "Preparation voltage crossed the safety floor; skipping "
                        "the step/chirp profile and returning to zero."
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        else:
            health_samples.append(
                {
                    "phase": "before_profile",
                    "operation_elapsed_s": time.monotonic() - operation_started,
                    **center_health,
                }
            )
            print("Running automatic profile...")
            profile_started = time.monotonic()
            capture_profile(
                bus,
                servo_id=servo_id,
                servo=servo,
                segments=segments,
                sample_hz=float(args.sample_hz),
                move_time_ms=int(args.move_time_ms),
                write_deadband_units=int(args.write_deadband_units),
                max_position_error_rad=math.radians(
                    float(args.max_position_error_deg)
                ),
                read_retries=int(args.read_retries),
                read_retry_sleep_s=float(args.read_retry_sleep_s),
                fields=capture_fields,
            )
            profile_duration_s = time.monotonic() - profile_started
            return_started = time.monotonic()
            bus.move_time_write(
                servo_id,
                servo.joint_target_rad_to_elect_unit(center_rad),
                1000,
            )
            time.sleep(1.0 + float(args.settle_s))
            return_to_center_wait_s = time.monotonic() - return_started
            final_center_units = _read_position_with_retries(
                bus,
                servo_id=servo_id,
                retries=int(args.read_retries),
                retry_sleep_s=float(args.read_retry_sleep_s),
            )
            final_center_rad = servo.servo_elect_units_to_joint_target_rad(
                final_center_units
            )
            health_after = read_health(bus, servo_id=servo_id)
            health_samples.append(
                {
                    "phase": "after_profile",
                    "operation_elapsed_s": time.monotonic() - operation_started,
                    **health_after,
                }
            )
            validate_health(
                health_after,
                min_voltage_v=float(args.min_voltage_v),
                max_temperature_c=float(args.max_temperature_c),
            )
            print(f"Profile complete; {_format_health(health_after)}")
        print(
            f"Returning automatically to the gravity-neutral unload pose "
            f"{float(args.unload_pose_deg):+.2f}deg..."
        )
        (
            unload_pose_units,
            unload_pose_position_rad,
            unload_pose_move_s,
            unload_pose_wait_s,
        ) = prepare_servo_center(
            bus,
            servo_id=servo_id,
            servo=servo,
            center_rad=unload_pose_rad,
            initial_units=int(final_center_units),
            prepare_speed_deg_s=float(args.prepare_speed_deg_s),
            move_time_ms=int(args.move_time_ms),
            settle_s=float(args.settle_s),
            center_tolerance_deg=float(args.center_tolerance_deg),
            max_attempts=int(args.center_max_attempts),
            min_voltage_v=float(args.min_voltage_v),
            max_temperature_c=float(args.max_temperature_c),
            read_retries=int(args.read_retries),
            read_retry_sleep_s=float(args.read_retry_sleep_s),
            attempts=unload_pose_attempts,
            phase_label="Unload-pose",
            monitor_hz=float(args.prepare_monitor_hz),
            monitor_samples=preparation_samples,
            monitor_started=preparation_monitor_started,
        )
        print(
            "Automatic unload pose verified at "
            f"{math.degrees(unload_pose_position_rad):+.2f}deg; disabling torque."
        )
        preparation_voltage_violation = minimum_voltage_sample_below(
            preparation_samples, float(args.min_voltage_v)
        )
        if preparation_voltage_violation is not None:
            raise RuntimeError(
                "servo voltage dropped below the safety floor during "
                f"{preparation_voltage_violation['phase']}: "
                f"{float(preparation_voltage_violation['voltage_v']):.3f}V < "
                f"{float(args.min_voltage_v):.3f}V"
            )
        outcome = "completed"
        exit_code = 0
    except KeyboardInterrupt:
        outcome = "aborted"
        error = "KeyboardInterrupt"
        exit_code = 130
        print(
            _yellow("\nCapture interrupted; stopping and unloading the servo."),
            file=sys.stderr,
        )
    except Exception as exc:
        outcome = "failed"
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1
        print(_yellow(f"Capture failed: {error}"), file=sys.stderr)
        if bus is not None:
            failure_diagnostics = {}
            try:
                failure_units = _read_position_with_retries(
                    bus,
                    servo_id=servo_id,
                    retries=int(args.read_retries),
                    retry_sleep_s=float(args.read_retry_sleep_s),
                )
                failure_diagnostics["position_units"] = failure_units
                failure_diagnostics["position_deg"] = math.degrees(
                    servo.servo_elect_units_to_joint_target_rad(failure_units)
                )
            except Exception as diagnostic_exc:
                failure_diagnostics["position_error"] = str(diagnostic_exc)
            try:
                failure_diagnostics["loaded"] = bus.read_loaded(servo_id)
            except Exception as diagnostic_exc:
                failure_diagnostics["loaded_error"] = str(diagnostic_exc)
            try:
                failure_move = bus.read_move_time(servo_id)
                failure_diagnostics["move_target_units"] = (
                    int(failure_move[0]) if failure_move is not None else None
                )
                failure_diagnostics["move_time_ms"] = (
                    int(failure_move[1]) if failure_move is not None else None
                )
            except Exception as diagnostic_exc:
                failure_diagnostics["move_error"] = str(diagnostic_exc)
            try:
                failure_diagnostics.update(read_health(bus, servo_id=servo_id))
            except Exception as diagnostic_exc:
                failure_diagnostics["health_error"] = str(diagnostic_exc)
    finally:
        if bus is not None:
            try:
                bus.move_stop(servo_id)
            except Exception as exc:
                print(_yellow(f"Warning: servo stop failed: {exc}"), file=sys.stderr)
            try:
                bus.unload(servo_id)
                _verify_loaded_state(bus, servo_id=servo_id, expected=False)
                final_unloaded_verified = True
                print("Servo unloaded and verified.")
            except Exception as exc:
                print(
                    _yellow(f"Warning: servo unload failed: {exc}"), file=sys.stderr
                )
                if exit_code == 0:
                    outcome = "failed"
                    error = f"servo unload verification failed: {exc}"
                    exit_code = 1
            try:
                bus.transport.close()
            except Exception as exc:
                print(_yellow(f"Warning: serial close failed: {exc}"), file=sys.stderr)

    finished_wall = datetime.now().astimezone().isoformat()
    operation_elapsed_s = time.monotonic() - operation_started
    arrays = _arrays_from_fields(capture_fields)
    arrays.update(
        {
            "cooldown_elapsed_s": np.asarray(
                [sample["cooldown_elapsed_s"] for sample in cooldown_samples],
                dtype=np.float64,
            ),
            "cooldown_temperature_c": np.asarray(
                [sample["temperature_c"] for sample in cooldown_samples],
                dtype=np.float32,
            ),
            "cooldown_voltage_v": np.asarray(
                [sample["voltage_v"] for sample in cooldown_samples],
                dtype=np.float32,
            ),
        }
    )
    measured_position = np.asarray(arrays["position_joint_rad"], dtype=np.float64)
    if fixture_model is not None:
        fixture_qpos, hold_torque, load_inertia = fixture_model.evaluate(
            measured_position
        )
    else:
        fixture_qpos, hold_torque, load_inertia = (
            evaluate_manual_horizontal_fixture(
                measured_position,
                center_rad=center_rad,
                signed_load_moment_kg_m=signed_load_moment_kg_m,
                load_inertia_kg_m2=load_inertia_at_center_kg_m2,
            )
        )
    arrays = add_standard_capture_arrays(
        arrays,
        fixture_qpos_rad=fixture_qpos,
        estimated_hold_torque_nm=hold_torque,
        estimated_load_inertia_kg_m2=load_inertia,
    )
    arrays.update(preparation_trace_arrays(preparation_samples))
    preparation_position = np.asarray(
        arrays["preparation_position_joint_rad"], dtype=np.float64
    )
    preparation_fixture_qpos = np.full(
        preparation_position.shape, np.nan, dtype=np.float32
    )
    preparation_hold_torque = np.full(
        preparation_position.shape, np.nan, dtype=np.float32
    )
    preparation_load_inertia = np.full(
        preparation_position.shape, np.nan, dtype=np.float32
    )
    valid_preparation_position = np.isfinite(preparation_position)
    if np.any(valid_preparation_position):
        if fixture_model is not None:
            prep_qpos, prep_torque, prep_inertia = fixture_model.evaluate(
                preparation_position[valid_preparation_position]
            )
        else:
            prep_qpos, prep_torque, prep_inertia = (
                evaluate_manual_horizontal_fixture(
                    preparation_position[valid_preparation_position],
                    center_rad=center_rad,
                    signed_load_moment_kg_m=signed_load_moment_kg_m,
                    load_inertia_kg_m2=load_inertia_at_center_kg_m2,
                )
            )
        preparation_fixture_qpos[valid_preparation_position] = prep_qpos
        preparation_hold_torque[valid_preparation_position] = prep_torque
        preparation_load_inertia[valid_preparation_position] = prep_inertia
    arrays.update(
        {
            "preparation_fixture_qpos_rad": preparation_fixture_qpos,
            "preparation_estimated_hold_torque_nm": preparation_hold_torque,
            "preparation_estimated_load_inertia_kg_m2": preparation_load_inertia,
        }
    )
    preparation_summary = summarize_preparation_trace(arrays)
    summary = summarize_capture(
        arrays,
        segments=segments,
        sample_hz=float(args.sample_hz),
        max_delay_s=float(args.max_delay_s),
    )
    health_by_phase = {
        str(sample["phase"]): sample
        for sample in health_samples
        if sample.get("phase") in ("before_profile", "after_profile")
    }
    before_profile_temperature = health_by_phase.get("before_profile", {}).get(
        "temperature_c"
    )
    after_profile_temperature = health_by_phase.get("after_profile", {}).get(
        "temperature_c"
    )
    temperature_rise_c = None
    temperature_rise_c_per_min = None
    if before_profile_temperature is not None and after_profile_temperature is not None:
        temperature_rise_c = float(after_profile_temperature) - float(
            before_profile_temperature
        )
        if profile_duration_s is not None and profile_duration_s > 0.0:
            temperature_rise_c_per_min = (
                temperature_rise_c * 60.0 / profile_duration_s
            )
    metadata: dict[str, object] = {
        "tool": "runtime/scripts/capture_servo_sysid.py",
        "mode": (
            "htd45h_center_preparation_diagnostic"
            if args.prepare_only
            else "htd45h_known_load_profile"
        ),
        "capture_source": "hardware",
        "invoked_at": invocation_wall,
        "captured_at": start_wall,
        "finished_at": finished_wall,
        "outcome": outcome,
        "error": error,
        "joint": str(args.servo_label or f"servo_{servo_id}"),
        "joint_name": str(args.servo_label or f"servo_{servo_id}"),
        "servo_id": servo_id,
        "board_port": port,
        "port": port,
        "baudrate": int(args.baudrate),
        "servo_coordinate": "raw_centered",
        "servo_center_unit": int(ServoConfig.UNITS_CENTER),
        "servo_units_per_rad": float(ServoConfig.UNITS_PER_RAD),
        "servo_unit_direction": 1,
        "servo_offset_unit": 0,
        "center_deg": float(args.center_deg),
        "unload_pose_deg": float(args.unload_pose_deg),
        "static_torque_at_unload_pose_nm": static_torque_at_unload_pose_nm,
        "amplitudes_deg": [float(value) for value in args.amplitudes_deg],
        "sample_hz": float(args.sample_hz),
        "sample_rate_hz": float(args.sample_hz),
        "num_samples": int(arrays["timestamps_s"].size),
        "move_time_ms": int(args.move_time_ms),
        "write_deadband_units": int(args.write_deadband_units),
        "prepare_only": bool(args.prepare_only),
        "prepare_monitor_hz": float(args.prepare_monitor_hz),
        "pre_move_hold_s": float(args.pre_move_hold_s),
        "timing_semantics": {
            "command_elapsed_s": "host monotonic time immediately before write decision",
            "command_write_elapsed_s": (
                "host monotonic time after serial flush; NaN when the deadband skips a write"
            ),
            "command_age_at_read_s": (
                "time from the latest serial-write completion to position-read completion"
            ),
            "position_elapsed_s": "host monotonic time after position response",
            "delay_s": (
                "cross-correlation lag including servo response; not pure transport latency"
            ),
        },
        "waits_s": {
            "run_confirmation": run_confirmation_wait_s,
            "automatic_start_delay": startup_delay_wait_s,
            "cooldown": cooldown_wait_s,
            "prepare_commanded_move": prepare_move_s,
            "prepare_move_and_settle_actual": prepare_wait_s,
            "center_confirmation": center_confirmation_wait_s,
            "profile_actual": profile_duration_s,
            "return_to_center_actual": return_to_center_wait_s,
            "unload_confirmation": unload_confirmation_wait_s,
            "return_to_unload_pose_commanded": unload_pose_move_s,
            "return_to_unload_pose_actual": unload_pose_wait_s,
            "operation_total": operation_elapsed_s,
        },
        "servo_diagnostics": {
            "angle_limits_units": (
                list(servo_angle_limits_units)
                if servo_angle_limits_units is not None
                else None
            ),
            "voltage_limits_v": (
                list(servo_voltage_limits_v)
                if servo_voltage_limits_v is not None
                else None
            ),
            "temperature_limit_c": servo_temperature_limit_c,
            "motor_mode": (
                {
                    "mode": int(servo_motor_mode[0]),
                    "speed": int(servo_motor_mode[1]),
                }
                if servo_motor_mode is not None
                else None
            ),
            "led_enabled": servo_led_enabled,
            "alarm_mask": servo_alarm_mask,
            "alarm_mask_interpretation": (
                "configured alarm sources: bit0=temperature, bit1=voltage, "
                "bit2=locked-rotor; not a live fault code"
            ),
            "initial_move_target_units": (
                int(initial_move_target[0])
                if initial_move_target is not None
                else None
            ),
            "initial_move_time_ms": (
                int(initial_move_target[1])
                if initial_move_target is not None
                else None
            ),
            "run_confirmation_required": False,
            "center_confirmation_required": False,
            "unload_confirmation_required": False,
            "center_max_attempts": int(args.center_max_attempts),
            "preparation_monitor": preparation_summary,
            "preparation_attempts": preparation_attempts,
            "unload_pose_attempts": unload_pose_attempts,
            "failure": failure_diagnostics,
        },
        "cooldown": {
            "target_temperature_c": float(args.cooldown_target_c),
            "timeout_s": float(args.cooldown_timeout_s),
            "poll_s": float(args.cooldown_poll_s),
            "samples": cooldown_samples,
        },
        "thermal_summary": {
            "before_profile_temperature_c": before_profile_temperature,
            "after_profile_temperature_c": after_profile_temperature,
            "profile_temperature_rise_c": temperature_rise_c,
            "profile_temperature_rise_c_per_min": temperature_rise_c_per_min,
        },
        "servo_state": {
            "initial_loaded": initial_loaded_state,
            "final_unloaded_verified": final_unloaded_verified,
            "initial_position_units": initial_units,
            "initial_position_deg": (
                math.degrees(initial_rad) if initial_rad is not None else None
            ),
            "centered_position_units": centered_units,
            "centered_position_deg": (
                math.degrees(centered_rad) if centered_rad is not None else None
            ),
            "final_center_position_units": final_center_units,
            "final_center_position_deg": (
                math.degrees(final_center_rad)
                if final_center_rad is not None
                else None
            ),
            "unload_pose_position_units": unload_pose_units,
            "unload_pose_position_deg": (
                math.degrees(unload_pose_position_rad)
                if unload_pose_position_rad is not None
                else None
            ),
        },
        "chirp_start_hz": float(args.chirp_start_hz),
        "chirp_end_hz": float(args.chirp_end_hz),
        "chirp_duration_s": float(args.chirp_duration_s),
        "chirp_decay_rate": float(args.chirp_decay_rate),
        **load_metadata,
        "load_inertia_at_center_kg_m2": load_inertia_at_center_kg_m2,
        "static_torque_at_center_nm": static_torque_at_center_nm,
        "max_abs_profile_hold_torque_nm": max_profile_hold_torque_nm,
        "servo_label": args.servo_label,
        "fixture_label": args.fixture_label,
        "notes": args.notes,
        "health_samples": health_samples,
        "profile_reference": {
            "project": "ToddlerBot",
            "paper": "https://arxiv.org/abs/2502.00893",
            "local_source": "~/projects/toddlerbot/toddlerbot/policies/sysID.py",
            "wildrobot_legacy_contract": "tools/sysid/run_capture.py",
            "servo_protocol_source": (
                "docs/HTD-45H Serial Bus Servo/2. Bus Servo Secondary Development/"
                "LSC Series Servo Controller Secondary Develpoment/Jetson Nano "
                "Development/Program/Library File/sdk/hiwonder_servo_controller.py"
            ),
        },
    }
    npz_path, json_path = write_capture(
        output_path,
        arrays=arrays,
        metadata=metadata,
        summary=summary,
    )
    print(f"Wrote trace:   {npz_path}")
    print(f"Wrote summary: {json_path}")
    first_unload = preparation_summary["first_unload"]
    print(
        "Preparation monitor: "
        f"samples={preparation_summary['samples']} "
        f"min_voltage={preparation_summary['min_voltage_v']}V",
        flush=True,
    )
    if first_unload is not None:
        print(
            "First torque loss: "
            f"phase={first_unload['phase']} "
            f"phase_elapsed={float(first_unload['phase_elapsed_s']):.3f}s "
            f"position={first_unload['position_deg']}deg "
            f"estimated_hold_torque="
            f"{first_unload['estimated_hold_torque_nm']}Nm "
            f"voltage={first_unload['voltage_v']}V",
            flush=True,
        )
    if summary["tracking_rmse_deg"] is not None:
        print(
            "Tracking: "
            f"RMSE={float(summary['tracking_rmse_deg']):.2f}deg "
            f"p95={float(summary['tracking_abs_p95_deg']):.2f}deg"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
