#!/usr/bin/env python3
"""Capture representative joint responses for SysID (v0.19.1)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
for _p in (str(_REPO_ROOT), str(_RUNTIME_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs import WildRobotRuntimeConfig
from runtime.wr_runtime.hardware.actuators import ServoModel
from runtime.wr_runtime.hardware.hiwonder_board_controller import HiwonderBoardController
from runtime.wr_runtime.validation.realism_profile import load_runtime_realism_profile


@dataclass(frozen=True)
class CaptureArtifact:
    mode: str
    joint_name: str
    sample_rate_hz: float
    command_rad: np.ndarray
    measured_position_rad: np.ndarray
    measured_velocity_rad_s: np.ndarray
    timestamps_s: np.ndarray
    metadata: dict[str, Any]

    def validate(self) -> None:
        t = int(self.timestamps_s.shape[0])
        if t <= 0:
            raise ValueError("capture must contain at least one sample")
        if self.command_rad.shape != (t,):
            raise ValueError("command_rad must be shape (T,)")
        if self.measured_position_rad.shape != (t,):
            raise ValueError("measured_position_rad must be shape (T,)")
        if self.measured_velocity_rad_s.shape != (t,):
            raise ValueError("measured_velocity_rad_s must be shape (T,)")
        if not np.all(np.isfinite(self.timestamps_s)):
            raise ValueError("timestamps_s contains non-finite values")
        if not np.all(np.isfinite(self.command_rad)):
            raise ValueError("command_rad contains non-finite values")
        if not np.all(np.isfinite(self.measured_position_rad)):
            raise ValueError("measured_position_rad contains non-finite values")
        if not np.all(np.isfinite(self.measured_velocity_rad_s)):
            raise ValueError("measured_velocity_rad_s contains non-finite values")

    def to_manifest(self, output_npz: str) -> dict[str, Any]:
        return {
            "schema_version": "v0.19.1",
            "tool": "tools/sysid/run_capture.py",
            "mode": self.mode,
            "joint_name": self.joint_name,
            "sample_rate_hz": self.sample_rate_hz,
            "num_samples": int(self.timestamps_s.shape[0]),
            "npz_file": output_npz,
            "metadata": self.metadata,
        }


def _command_signal(
    mode: str,
    timestamps_s: np.ndarray,
    *,
    amplitude_rad: float,
    hold_rad: float,
    step_start_s: float,
    chirp_start_hz: float,
    chirp_end_hz: float,
) -> np.ndarray:
    if mode == "step":
        return np.where(
            timestamps_s >= step_start_s,
            hold_rad + amplitude_rad,
            hold_rad,
        ).astype(np.float32)
    if mode == "hold":
        return np.full_like(timestamps_s, fill_value=hold_rad, dtype=np.float32)
    if mode == "chirp":
        t = np.asarray(timestamps_s, dtype=np.float64)
        duration = float(max(t[-1], 1e-9))
        k = (float(chirp_end_hz) - float(chirp_start_hz)) / duration
        phase = 2.0 * np.pi * (float(chirp_start_hz) * t + 0.5 * k * t * t)
        return (hold_rad + amplitude_rad * np.sin(phase)).astype(np.float32)
    raise ValueError(f"Unsupported mode: {mode}")


def _simulate_measured_response(
    command_rad: np.ndarray,
    sample_rate_hz: float,
    *,
    model_delay_steps: int,
    model_backlash_rad: float,
    model_tau_s: float,
    model_noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / float(sample_rate_hz)
    rng = np.random.default_rng(seed)
    delayed = np.roll(command_rad, model_delay_steps).astype(np.float32)
    if model_delay_steps > 0:
        delayed[:model_delay_steps] = delayed[model_delay_steps]

    pos = np.zeros_like(delayed, dtype=np.float32)
    vel = np.zeros_like(delayed, dtype=np.float32)
    alpha = float(dt / max(model_tau_s, dt))

    for i in range(1, delayed.shape[0]):
        err = float(delayed[i] - pos[i - 1])
        if abs(err) < model_backlash_rad:
            effective_err = 0.0
        else:
            effective_err = np.sign(err) * (abs(err) - model_backlash_rad)
        v = alpha * effective_err / dt
        vel[i] = np.float32(v + rng.normal(0.0, model_noise_std))
        pos[i] = np.float32(pos[i - 1] + vel[i] * dt)

    return pos, vel


def _read_servo_position_units(
    controller: HiwonderBoardController,
    *,
    servo_id: int,
    retries: int,
    retry_sleep_s: float,
) -> int:
    last_resp: Any = None
    for _ in range(retries):
        resp = controller.read_servo_positions([int(servo_id)])
        last_resp = resp
        if resp:
            for sid, pos in resp:
                if int(sid) == int(servo_id):
                    return int(pos)
        time.sleep(retry_sleep_s)
    raise RuntimeError(
        f"Failed to read servo position for id={servo_id}; last_resp={last_resp!r}"
    )


def _capture_measured_response_hardware(
    *,
    cfg: WildRobotRuntimeConfig,
    joint_name: str,
    command_rad: np.ndarray,
    sample_rate_hz: float,
    move_time_ms: int,
    read_retries: int,
    read_retry_sleep_s: float,
    settle_s: float,
    return_to_hold: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    servo = cfg.servo_controller.get_servo(joint_name)
    controller = HiwonderBoardController(
        port=cfg.servo_controller.port,
        baudrate=cfg.servo_controller.baudrate,
    )
    period_s = 1.0 / float(sample_rate_hz)
    num_samples = int(command_rad.shape[0])
    measured_position_rad = np.zeros((num_samples,), dtype=np.float32)
    timestamps_s = np.zeros((num_samples,), dtype=np.float64)

    try:
        hold_units = servo.joint_target_rad_to_elect_unit(float(command_rad[0]))
        controller.move_servos([(int(servo.id), int(hold_units))], time_ms=max(20, int(move_time_ms)))
        if settle_s > 0.0:
            time.sleep(float(settle_s))

        t_start = time.perf_counter()
        for i in range(num_samples):
            tick_target = t_start + (i * period_s)
            now = time.perf_counter()
            if tick_target > now:
                time.sleep(tick_target - now)

            cmd_units = servo.joint_target_rad_to_elect_unit(float(command_rad[i]))
            controller.move_servos([(int(servo.id), int(cmd_units))], time_ms=max(20, int(move_time_ms)))
            measured_units = _read_servo_position_units(
                controller,
                servo_id=int(servo.id),
                retries=max(1, int(read_retries)),
                retry_sleep_s=max(0.0, float(read_retry_sleep_s)),
            )
            measured_position_rad[i] = np.float32(
                servo.servo_elect_units_to_joint_target_rad(int(measured_units))
            )
            timestamps_s[i] = float(time.perf_counter() - t_start)

        if return_to_hold:
            controller.move_servos([(int(servo.id), int(hold_units))], time_ms=max(20, int(move_time_ms)))
    finally:
        controller.close()

    measured_velocity_rad_s = np.zeros_like(measured_position_rad, dtype=np.float32)
    nominal_dt = np.float32(1.0 / float(sample_rate_hz))
    for i in range(1, num_samples):
        dt = np.float32(timestamps_s[i] - timestamps_s[i - 1])
        if dt <= 1e-6:
            dt = nominal_dt
        measured_velocity_rad_s[i] = np.float32(
            (measured_position_rad[i] - measured_position_rad[i - 1]) / dt
        )
    return measured_position_rad, measured_velocity_rad_s, timestamps_s


def _load_servo_config(
    runtime_config: Path,
    joint_name: str,
) -> tuple[WildRobotRuntimeConfig, float]:
    cfg = WildRobotRuntimeConfig.load(runtime_config)
    servo = cfg.servo_controller.get_servo(joint_name)
    center = float(np.deg2rad(servo.motor_center_mujoco_deg))
    return cfg, center


def _write_capture(
    artifact: CaptureArtifact,
    output_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    artifact.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{artifact.mode}_{artifact.joint_name}"
    npz_path = output_dir / f"{base}.npz"
    json_path = output_dir / f"{base}.json"

    np.savez(
        npz_path,
        command_rad=artifact.command_rad.astype(np.float32),
        measured_position_rad=artifact.measured_position_rad.astype(np.float32),
        measured_velocity_rad_s=artifact.measured_velocity_rad_s.astype(np.float32),
        timestamps_s=artifact.timestamps_s.astype(np.float64),
    )
    json_path.write_text(json.dumps(artifact.to_manifest(npz_path.name), indent=2) + "\n")
    return npz_path, json_path


def build_capture_artifact(
    *,
    mode: str,
    joint_name: str,
    sample_rate_hz: float,
    duration_s: float,
    amplitude_rad: float,
    hold_rad: float,
    step_start_s: float,
    chirp_start_hz: float,
    chirp_end_hz: float,
    model_delay_steps: int,
    model_backlash_rad: float,
    model_tau_s: float,
    model_noise_std: float,
    seed: int,
    metadata: dict[str, Any],
) -> CaptureArtifact:
    num_samples = max(2, int(round(float(duration_s) * float(sample_rate_hz))))
    timestamps_s = np.arange(num_samples, dtype=np.float64) / float(sample_rate_hz)
    command = _command_signal(
        mode,
        timestamps_s,
        amplitude_rad=amplitude_rad,
        hold_rad=hold_rad,
        step_start_s=step_start_s,
        chirp_start_hz=chirp_start_hz,
        chirp_end_hz=chirp_end_hz,
    )
    measured_pos, measured_vel = _simulate_measured_response(
        command,
        sample_rate_hz,
        model_delay_steps=model_delay_steps,
        model_backlash_rad=model_backlash_rad,
        model_tau_s=model_tau_s,
        model_noise_std=model_noise_std,
        seed=seed,
    )
    return CaptureArtifact(
        mode=mode,
        joint_name=joint_name,
        sample_rate_hz=sample_rate_hz,
        command_rad=command,
        measured_position_rad=measured_pos,
        measured_velocity_rad_s=measured_vel,
        timestamps_s=timestamps_s,
        metadata=metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture joint response traces for SysID.")
    parser.add_argument("--mode", choices=("step", "hold", "chirp"), required=True)
    parser.add_argument(
        "--capture-source",
        choices=("hardware", "synthetic"),
        default="hardware",
        help="Use hardware actuator readback (default) or synthetic fallback for offline testing.",
    )
    parser.add_argument("--joint-name", required=True)
    parser.add_argument(
        "--runtime-config",
        default="runtime/configs/runtime_config_v2.json",
        help="Runtime config path for servo and metadata context.",
    )
    parser.add_argument("--output-dir", default="runtime/logs/sysid")
    parser.add_argument("--prefix", default="capture")
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--amplitude-rad", type=float, default=0.2)
    parser.add_argument("--hold-rad", type=float, default=0.0)
    parser.add_argument("--step-start-s", type=float, default=0.5)
    parser.add_argument("--chirp-start-hz", type=float, default=0.2)
    parser.add_argument("--chirp-end-hz", type=float, default=3.0)
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=20,
        help="Per-step board move command time (ms) for hardware capture.",
    )
    parser.add_argument("--read-retries", type=int, default=5)
    parser.add_argument("--read-retry-sleep-s", type=float, default=0.004)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument(
        "--no-return-to-hold",
        action="store_true",
        help="Do not command the joint back to the initial hold target after capture.",
    )
    parser.add_argument("--model-delay-steps", type=int, default=1)
    parser.add_argument("--model-backlash-rad", type=float, default=0.012)
    parser.add_argument("--model-tau-s", type=float, default=0.08)
    parser.add_argument("--model-noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_config = Path(args.runtime_config)
    cfg, servo_center_rad = _load_servo_config(runtime_config, args.joint_name)

    runtime_realism_profile = load_runtime_realism_profile(cfg)
    metadata = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "runtime_config_path": str(runtime_config),
        "asset_mjcf_path": cfg.mjcf_path,
        "asset_version": "v2",
        "sample_rate_hz": float(args.sample_rate_hz),
        "servo_center_rad": servo_center_rad,
        "servo_model": ServoModel().__dict__,
        "capture_source": str(args.capture_source),
        "simulated_capture": bool(args.capture_source == "synthetic"),
        "runtime_servo_port": cfg.servo_controller.port,
        "runtime_servo_baudrate": int(cfg.servo_controller.baudrate),
        "joint_servo_id": int(cfg.servo_controller.get_servo(args.joint_name).id),
    }
    if cfg.realism_profile_path is not None:
        metadata["realism_profile_path"] = cfg.realism_profile_path
    if runtime_realism_profile is not None:
        metadata["realism_profile_name"] = runtime_realism_profile.profile_name

    num_samples = max(2, int(round(float(args.duration_s) * float(args.sample_rate_hz))))
    nominal_timestamps = np.arange(num_samples, dtype=np.float64) / float(args.sample_rate_hz)
    command_rad = _command_signal(
        args.mode,
        nominal_timestamps,
        amplitude_rad=float(args.amplitude_rad),
        hold_rad=float(args.hold_rad),
        step_start_s=float(args.step_start_s),
        chirp_start_hz=float(args.chirp_start_hz),
        chirp_end_hz=float(args.chirp_end_hz),
    )
    if args.capture_source == "hardware":
        measured_pos, measured_vel, timestamps_s = _capture_measured_response_hardware(
            cfg=cfg,
            joint_name=args.joint_name,
            command_rad=command_rad,
            sample_rate_hz=float(args.sample_rate_hz),
            move_time_ms=int(args.move_time_ms),
            read_retries=int(args.read_retries),
            read_retry_sleep_s=float(args.read_retry_sleep_s),
            settle_s=float(args.settle_s),
            return_to_hold=not bool(args.no_return_to_hold),
        )
    else:
        measured_pos, measured_vel = _simulate_measured_response(
            command_rad,
            float(args.sample_rate_hz),
            model_delay_steps=int(args.model_delay_steps),
            model_backlash_rad=float(args.model_backlash_rad),
            model_tau_s=float(args.model_tau_s),
            model_noise_std=float(args.model_noise_std),
            seed=int(args.seed),
        )
        timestamps_s = nominal_timestamps

    artifact = CaptureArtifact(
        mode=args.mode,
        joint_name=args.joint_name,
        sample_rate_hz=float(args.sample_rate_hz),
        command_rad=command_rad.astype(np.float32),
        measured_position_rad=measured_pos.astype(np.float32),
        measured_velocity_rad_s=measured_vel.astype(np.float32),
        timestamps_s=timestamps_s.astype(np.float64),
        metadata=metadata,
    )
    npz_path, json_path = _write_capture(artifact, Path(args.output_dir), args.prefix)
    print(f"Wrote SysID capture: {npz_path}")
    print(f"Wrote SysID manifest: {json_path}")


if __name__ == "__main__":
    main()
