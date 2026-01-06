from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from assets.robot_config import RobotConfig
from playground_amp.sim_adapter.foot_switches import (
    contact_forces_from_mujoco,
    resolve_foot_geom_ids,
    switches_from_forces_np,
)
from policy_contract.io import NumpySignalsProvider
from policy_contract.numpy.signals import Signals
from policy_contract.spec import PolicySpec


@dataclass(frozen=True)
class _SensorInfo:
    adr: int
    dim: int


@dataclass(frozen=True)
class _TouchSensorInfo:
    adr: int
    dim: int


class MujocoSignalsAdapter(NumpySignalsProvider[Signals]):
    """Extract policy-facing signals from native MuJoCo (MjModel/MjData).

    Prefers explicit IMU sensors when present. Falls back to root qpos/qvel if
    sensors are missing. Foot switches are sourced from touch sensors when
    available; otherwise, contact normal forces are used as a fallback.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        robot_config: RobotConfig,
        policy_spec: PolicySpec,
        foot_switch_threshold: float,
        foot_geom_names: Optional[tuple[str, str, str, str]] = None,
    ) -> None:
        self._mj_model = mj_model
        self._robot_config = robot_config
        self._policy_spec = policy_spec
        self._foot_switch_threshold = float(foot_switch_threshold)

        self._orientation_sensor = _find_sensor(mj_model, robot_config.orientation_sensor)
        self._gyro_sensor = _find_sensor(mj_model, robot_config.root_gyro_sensor)

        if foot_geom_names is None:
            foot_geom_names = robot_config.get_foot_geom_names()
        self._foot_geom_ids = resolve_foot_geom_ids(mj_model, foot_geom_names)
        self._touch_sensors = _find_touch_sensors(mj_model, self._foot_geom_ids)

        if self._touch_sensors is None:
            self._contact_force = np.zeros(6, dtype=np.float64)
        else:
            self._contact_force = None

        self._joint_qpos_idx, self._joint_qvel_idx = _resolve_actuator_joint_indices(
            mj_model, list(policy_spec.robot.actuator_names)
        )

    def read(self, mj_data: mujoco.MjData) -> Signals:
        quat = _read_sensor_or_none(mj_data, self._orientation_sensor)
        if quat is None:
            quat = mj_data.qpos[3:7]

        gyro = _read_sensor_or_none(mj_data, self._gyro_sensor)
        if gyro is None:
            # Fallback: qvel-based angular velocity (frame may differ from IMU gyro).
            gyro = mj_data.qvel[3:6]

        joint_pos = np.take(mj_data.qpos, self._joint_qpos_idx)
        joint_vel = np.take(mj_data.qvel, self._joint_qvel_idx)

        if self._touch_sensors is not None:
            foot_switches = _read_touch_switches(mj_data, self._touch_sensors)
        else:
            foot_forces = contact_forces_from_mujoco(
                self._mj_model,
                mj_data,
                self._foot_geom_ids,
                self._contact_force,
            )
            foot_switches = switches_from_forces_np(
                foot_forces, self._foot_switch_threshold
            )

        return Signals(
            quat_xyzw=np.asarray(quat, dtype=np.float32),
            gyro_rad_s=np.asarray(gyro, dtype=np.float32),
            joint_pos_rad=np.asarray(joint_pos, dtype=np.float32),
            joint_vel_rad_s=np.asarray(joint_vel, dtype=np.float32),
            foot_switches=np.asarray(foot_switches, dtype=np.float32),
        )


def _find_sensor(mj_model: mujoco.MjModel, name: str) -> Optional[_SensorInfo]:
    sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        return None
    adr = int(mj_model.sensor_adr[sensor_id])
    dim = int(mj_model.sensor_dim[sensor_id])
    return _SensorInfo(adr=adr, dim=dim)


def _read_sensor_or_none(
    mj_data: mujoco.MjData, sensor: Optional[_SensorInfo]
) -> Optional[np.ndarray]:
    if sensor is None:
        return None
    return mj_data.sensordata[sensor.adr : sensor.adr + sensor.dim]


def _find_touch_sensors(
    mj_model: mujoco.MjModel, geom_ids: tuple[int, int, int, int]
) -> Optional[tuple[_TouchSensorInfo, _TouchSensorInfo, _TouchSensorInfo, _TouchSensorInfo]]:
    sensors = []
    for geom_id in geom_ids:
        sensor_id = _find_touch_sensor_for_geom(mj_model, geom_id)
        if sensor_id is None:
            return None
        adr = int(mj_model.sensor_adr[sensor_id])
        dim = int(mj_model.sensor_dim[sensor_id])
        sensors.append(_TouchSensorInfo(adr=adr, dim=dim))
    return tuple(sensors)  # type: ignore[return-value]


def _find_touch_sensor_for_geom(
    mj_model: mujoco.MjModel, geom_id: int
) -> Optional[int]:
    for sensor_id in range(int(mj_model.nsensor)):
        if mj_model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_TOUCH:
            continue
        if mj_model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        if int(mj_model.sensor_objid[sensor_id]) != int(geom_id):
            continue
        return int(sensor_id)
    return None


def _read_touch_switches(
    mj_data: mujoco.MjData,
    sensors: tuple[_TouchSensorInfo, _TouchSensorInfo, _TouchSensorInfo, _TouchSensorInfo],
) -> np.ndarray:
    values = []
    for sensor in sensors:
        value = mj_data.sensordata[sensor.adr : sensor.adr + sensor.dim]
        values.append(float(value[0]))
    return np.asarray(values, dtype=np.float32)


def _resolve_actuator_joint_indices(
    mj_model: mujoco.MjModel, actuator_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    qpos_idx = []
    qvel_idx = []
    for name in actuator_names:
        act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise ValueError(f"Actuator '{name}' not found in MJCF")

        trn_type = mj_model.actuator_trntype[act_id]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"Actuator '{name}' does not target a joint (trntype={int(trn_type)})"
            )

        joint_id = int(mj_model.actuator_trnid[act_id][0])
        qpos_idx.append(int(mj_model.jnt_qposadr[joint_id]))
        qvel_idx.append(int(mj_model.jnt_dofadr[joint_id]))

    return np.asarray(qpos_idx, dtype=np.int32), np.asarray(qvel_idx, dtype=np.int32)
