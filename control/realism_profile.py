"""Typed schema and loader for digital-twin realism profiles (v0.19.1)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


def _as_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value)!r}")
    return float(value)


def _as_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, field_name=field_name)


def _as_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int when provided, got {type(value)!r}")
    return int(value)


@dataclass(frozen=True)
class ActuatorRealismParams:
    joint_name: str
    control_delay_s: float | None
    control_delay_steps: int | None
    backlash_rad: float
    frictionloss: float
    armature: float
    effective_max_velocity_rad_s: float
    effective_max_torque_scale: float

    def __post_init__(self) -> None:
        if not self.joint_name:
            raise ValueError("joint_name must be non-empty")
        if self.control_delay_s is None and self.control_delay_steps is None:
            raise ValueError(
                f"{self.joint_name}: one of control_delay_s/control_delay_steps must be provided"
            )
        if self.control_delay_s is not None and self.control_delay_s < 0.0:
            raise ValueError(f"{self.joint_name}: control_delay_s must be >= 0")
        if self.control_delay_steps is not None and self.control_delay_steps < 0:
            raise ValueError(f"{self.joint_name}: control_delay_steps must be >= 0")
        if self.backlash_rad < 0.0:
            raise ValueError(f"{self.joint_name}: backlash_rad must be >= 0")
        if self.frictionloss < 0.0:
            raise ValueError(f"{self.joint_name}: frictionloss must be >= 0")
        if self.armature < 0.0:
            raise ValueError(f"{self.joint_name}: armature must be >= 0")
        if self.effective_max_velocity_rad_s <= 0.0:
            raise ValueError(f"{self.joint_name}: effective_max_velocity_rad_s must be > 0")
        if self.effective_max_torque_scale <= 0.0:
            raise ValueError(f"{self.joint_name}: effective_max_torque_scale must be > 0")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActuatorRealismParams":
        joint_name_raw = payload.get("joint_name", payload.get("actuator_name"))
        if not isinstance(joint_name_raw, str):
            raise TypeError("actuator entry must provide joint_name or actuator_name as string")
        return cls(
            joint_name=joint_name_raw,
            control_delay_s=_as_optional_float(
                payload.get("control_delay_s"),
                field_name=f"{joint_name_raw}.control_delay_s",
            ),
            control_delay_steps=_as_optional_int(
                payload.get("control_delay_steps"),
                field_name=f"{joint_name_raw}.control_delay_steps",
            ),
            backlash_rad=_as_float(
                payload.get("backlash_rad"),
                field_name=f"{joint_name_raw}.backlash_rad",
            ),
            frictionloss=_as_float(
                payload.get("frictionloss"),
                field_name=f"{joint_name_raw}.frictionloss",
            ),
            armature=_as_float(
                payload.get("armature"),
                field_name=f"{joint_name_raw}.armature",
            ),
            effective_max_velocity_rad_s=_as_float(
                payload.get("effective_max_velocity_rad_s"),
                field_name=f"{joint_name_raw}.effective_max_velocity_rad_s",
            ),
            effective_max_torque_scale=_as_float(
                payload.get("effective_max_torque_scale"),
                field_name=f"{joint_name_raw}.effective_max_torque_scale",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "control_delay_s": self.control_delay_s,
            "control_delay_steps": self.control_delay_steps,
            "backlash_rad": self.backlash_rad,
            "frictionloss": self.frictionloss,
            "armature": self.armature,
            "effective_max_velocity_rad_s": self.effective_max_velocity_rad_s,
            "effective_max_torque_scale": self.effective_max_torque_scale,
        }


@dataclass(frozen=True)
class SensorRealismParams:
    imu_gyro_noise_std: float
    imu_gyro_bias_walk_std: float
    imu_orientation_noise_std: float
    foot_switch_latency_s: float | None
    foot_switch_latency_steps: int | None
    foot_switch_dropout_prob: float

    def __post_init__(self) -> None:
        if self.imu_gyro_noise_std < 0.0:
            raise ValueError("imu_gyro_noise_std must be >= 0")
        if self.imu_gyro_bias_walk_std < 0.0:
            raise ValueError("imu_gyro_bias_walk_std must be >= 0")
        if self.imu_orientation_noise_std < 0.0:
            raise ValueError("imu_orientation_noise_std must be >= 0")
        if self.foot_switch_latency_s is None and self.foot_switch_latency_steps is None:
            raise ValueError(
                "one of foot_switch_latency_s/foot_switch_latency_steps must be provided"
            )
        if self.foot_switch_latency_s is not None and self.foot_switch_latency_s < 0.0:
            raise ValueError("foot_switch_latency_s must be >= 0")
        if self.foot_switch_latency_steps is not None and self.foot_switch_latency_steps < 0:
            raise ValueError("foot_switch_latency_steps must be >= 0")
        if not 0.0 <= self.foot_switch_dropout_prob <= 1.0:
            raise ValueError("foot_switch_dropout_prob must be in [0, 1]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorRealismParams":
        bias_std = payload.get("imu_gyro_bias_walk_std", payload.get("imu_gyro_bias_std"))
        return cls(
            imu_gyro_noise_std=_as_float(
                payload.get("imu_gyro_noise_std"),
                field_name="sensor_model.imu_gyro_noise_std",
            ),
            imu_gyro_bias_walk_std=_as_float(
                bias_std,
                field_name="sensor_model.imu_gyro_bias_walk_std",
            ),
            imu_orientation_noise_std=_as_float(
                payload.get("imu_orientation_noise_std", 0.0),
                field_name="sensor_model.imu_orientation_noise_std",
            ),
            foot_switch_latency_s=_as_optional_float(
                payload.get("foot_switch_latency_s"),
                field_name="sensor_model.foot_switch_latency_s",
            ),
            foot_switch_latency_steps=_as_optional_int(
                payload.get("foot_switch_latency_steps"),
                field_name="sensor_model.foot_switch_latency_steps",
            ),
            foot_switch_dropout_prob=_as_float(
                payload.get("foot_switch_dropout_prob"),
                field_name="sensor_model.foot_switch_dropout_prob",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "imu_gyro_noise_std": self.imu_gyro_noise_std,
            "imu_gyro_bias_walk_std": self.imu_gyro_bias_walk_std,
            "imu_orientation_noise_std": self.imu_orientation_noise_std,
            "foot_switch_latency_s": self.foot_switch_latency_s,
            "foot_switch_latency_steps": self.foot_switch_latency_steps,
            "foot_switch_dropout_prob": self.foot_switch_dropout_prob,
        }


@dataclass(frozen=True)
class DigitalTwinRealismProfile:
    schema_version: str
    profile_name: str
    asset_version: str
    sample_rate_hz: float
    actuators: tuple[ActuatorRealismParams, ...]
    sensor_model: SensorRealismParams
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        if not self.profile_name:
            raise ValueError("profile_name must be non-empty")
        if not self.asset_version:
            raise ValueError("asset_version must be non-empty")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be > 0")
        if not self.actuators:
            raise ValueError("actuators must be non-empty")

        names = [entry.joint_name for entry in self.actuators]
        dup_names = sorted({name for name in names if names.count(name) > 1})
        if dup_names:
            raise ValueError(f"duplicate actuator entries found: {dup_names}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DigitalTwinRealismProfile":
        actuators_raw = payload.get("actuators")
        if not isinstance(actuators_raw, list):
            raise TypeError("actuators must be a list")
        sensor_raw = payload.get("sensor_model")
        if not isinstance(sensor_raw, Mapping):
            raise TypeError("sensor_model must be a mapping")
        metadata_raw = payload.get("metadata", {})
        if not isinstance(metadata_raw, Mapping):
            raise TypeError("metadata must be a mapping when provided")

        return cls(
            schema_version=str(payload.get("schema_version", "")).strip(),
            profile_name=str(payload.get("profile_name", "")).strip(),
            asset_version=str(payload.get("asset_version", "")).strip(),
            sample_rate_hz=_as_float(payload.get("sample_rate_hz"), field_name="sample_rate_hz"),
            actuators=tuple(ActuatorRealismParams.from_dict(entry) for entry in actuators_raw),
            sensor_model=SensorRealismParams.from_dict(sensor_raw),
            metadata=dict(metadata_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "asset_version": self.asset_version,
            "sample_rate_hz": self.sample_rate_hz,
            "actuators": [entry.to_dict() for entry in self.actuators],
            "sensor_model": self.sensor_model.to_dict(),
            "metadata": self.metadata,
        }


def load_realism_profile(path: str | Path) -> DigitalTwinRealismProfile:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text())
    return DigitalTwinRealismProfile.from_dict(payload)


def save_realism_profile(profile: DigitalTwinRealismProfile, path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=False) + "\n")
