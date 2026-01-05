from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import mujoco
from mujoco import mjx

from assets.robot_config import RobotConfig
from policy_contract.io import SignalsProvider
from policy_contract.jax.signals import Signals


@dataclass(frozen=True)
class _SensorInfo:
    adr: int
    dim: int


@dataclass(frozen=True)
class _FootGeomInfo:
    geom_id: int


class MjxSignalsAdapter(SignalsProvider[Signals]):
    """Extract policy-facing signals directly from MJX without CAL."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        robot_config: RobotConfig,
        foot_switch_threshold: float,
    ) -> None:
        self._mj_model = mj_model
        self._robot_config = robot_config

        self._orientation_sensor = _find_sensor(mj_model, robot_config.orientation_sensor)
        self._gyro_sensor = _find_sensor(mj_model, robot_config.root_gyro_sensor)

        foot_geom_names = robot_config.get_foot_geom_names()
        self._foot_geoms = [_find_geom(mj_model, name) for name in foot_geom_names]
        self._foot_switch_threshold = float(foot_switch_threshold)

        self._joint_qpos = []
        self._joint_qvel = []
        for name in robot_config.actuator_names:
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF")
            self._joint_qpos.append(int(mj_model.jnt_qposadr[joint_id]))
            self._joint_qvel.append(int(mj_model.jnt_dofadr[joint_id]))

        self._joint_qpos_idx = jnp.asarray(self._joint_qpos, dtype=jnp.int32)
        self._joint_qvel_idx = jnp.asarray(self._joint_qvel, dtype=jnp.int32)

    def read(self, data: mjx.Data) -> Signals:
        quat = _read_sensor_or_none(data, self._orientation_sensor)
        if quat is None:
            # Fallback: root freejoint quaternion.
            quat = data.qpos[3:7]

        gyro = _read_sensor_or_none(data, self._gyro_sensor)
        if gyro is None:
            # Fallback: qvel-based angular velocity (frame may differ from IMU gyro).
            gyro = data.qvel[3:6]

        joint_pos = jnp.take(data.qpos, self._joint_qpos_idx)
        joint_vel = jnp.take(data.qvel, self._joint_qvel_idx)

        foot_switches = _get_foot_switches(
            data, self._foot_geoms, self._foot_switch_threshold
        )

        return Signals(
            quat_xyzw=quat.astype(jnp.float32),
            gyro_rad_s=gyro.astype(jnp.float32),
            joint_pos_rad=joint_pos.astype(jnp.float32),
            joint_vel_rad_s=joint_vel.astype(jnp.float32),
            foot_switches=foot_switches,
        )


def _find_sensor(mj_model: mujoco.MjModel, name: str) -> Optional[_SensorInfo]:
    sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        return None
    adr = int(mj_model.sensor_adr[sensor_id])
    dim = int(mj_model.sensor_dim[sensor_id])
    return _SensorInfo(adr=adr, dim=dim)


def _find_geom(mj_model: mujoco.MjModel, name: str) -> _FootGeomInfo:
    geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"Geom '{name}' not found in MJCF")
    return _FootGeomInfo(geom_id=int(geom_id))


def _read_sensor_or_none(data: mjx.Data, sensor: Optional[_SensorInfo]) -> Optional[jnp.ndarray]:
    if sensor is None:
        return None
    return data.sensordata[sensor.adr : sensor.adr + sensor.dim]


def _get_geom_contact_force(data: mjx.Data, geom_id: int) -> jnp.ndarray:
    if hasattr(data, "_impl"):
        contact = data._impl.contact
        efc_force = data._impl.efc_force
    else:
        contact = data.contact
        efc_force = data.efc_force

    geom1_match = contact.geom1 == geom_id
    geom2_match = contact.geom2 == geom_id
    is_our_contact = geom1_match | geom2_match

    normal_forces = efc_force[contact.efc_address]
    our_forces = jnp.where(is_our_contact, jnp.abs(normal_forces), 0.0)
    return jnp.sum(our_forces)


def _get_foot_switches(
    data: mjx.Data,
    foot_geoms: list[_FootGeomInfo],
    threshold: float,
) -> jnp.ndarray:
    forces = [ _get_geom_contact_force(data, g.geom_id) for g in foot_geoms ]
    raw = jnp.stack(forces)
    switches = jnp.where(raw > threshold, 1.0, 0.0)
    return switches.astype(jnp.float32)
