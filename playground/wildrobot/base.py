# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base classes for WildRobot environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco

from etils import epath
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from wildrobot import constants


def get_assets(root_path: str) -> Dict[str, bytes]:
    """Load all assets (STL files) for the environment."""
    assets = {}
    path = epath.Path(root_path)
    # Load all XML files from the models directory
    mjx_env.update_assets(assets, path, "*.xml")
    # Load all STL files from the assets directory
    mjx_env.update_assets(assets, path / "assets")
    return assets


class WildRobotEnv(mjx_env.MjxEnv):
    """Base class for WildRobot environments."""

    def __init__(
        self,
        task: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = constants.task_to_xml(task).as_posix()
        self.task = task
        self.robot_config = constants.ROBOT_CONFIGS[task]
        print(f"Loading WildRobot model from: {xml_path}")

        root_path = epath.Path(xml_path).parent  # models/v1 directory
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=get_assets(root_path)
        )
        self._mj_model.opt.timestep = self.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

        # Get floating base info
        self.floating_base_name = [
            self._mj_model.jnt(k).name
            for k in range(0, self._mj_model.njnt)
            if self._mj_model.jnt(k).type == 0
        ][
            0
        ]  # Assuming only one floating object (waist_freejoint)

        # Get actuator and joint names
        self.actuator_names = [
            self._mj_model.actuator(k).name for k in range(0, self._mj_model.nu)
        ]
        self.joint_names = [
            self._mj_model.jnt(k).name for k in range(0, self._mj_model.njnt)
        ]

        # Joint IDs and addresses
        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]

        self.actuator_qvel_addr = jp.array(
            [self._mj_model.jnt_dofadr[jid] for jid in self.actuator_joint_ids]
        )

        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }

        self._floating_base_qpos_addr = self._mj_model.jnt_qposadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][0]

        self._floating_base_qvel_addr = self._mj_model.jnt_dofadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][0]

        self._floating_base_id = self._mj_model.joint(self.floating_base_name).id

        print(f"WildRobot initialized with {len(self.actuator_names)} actuators")
        print(f"Actuators: {self.actuator_names}")
        print(f"Floating base: {self.floating_base_name}")

    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator."""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint."""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def get_joint_addr_from_name(self, name: str) -> int:
        """Return the address of a specified joint."""
        return self._mj_model.joint(name).qposadr

    def get_actuator_joint_qpos(self, data: jax.Array) -> jax.Array:
        """Return the qpos of actuator joints."""
        return data[jp.array(self.actuator_joint_qpos_addr)]

    def set_actuator_joints_qpos(
        self, new_qpos: jax.Array, qpos: jax.Array
    ) -> jax.Array:
        """Set the qpos only for the actuator joints."""
        return qpos.at[jp.array(self.actuator_joint_qpos_addr)].set(new_qpos)

    def get_actuator_joints_qvel(self, data: jax.Array) -> jax.Array:
        """Return the qvel of actuator joints."""
        return data[self.actuator_qvel_addr]

    def set_actuator_joints_qvel(
        self, new_qvel: jax.Array, qvel: jax.Array
    ) -> jax.Array:
        """Set the qvel only for the actuator joints."""
        return qvel.at[self.actuator_qvel_addr].set(new_qvel)

    def get_floating_base_qpos(self, data: jax.Array) -> jax.Array:
        """Return floating base position (7D: 3 pos + 4 quat)."""
        return data[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7]

    def get_floating_base_qvel(self, data: jax.Array) -> jax.Array:
        """Return floating base velocity (6D: 3 lin + 3 ang)."""
        return data[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6]

    def set_floating_base_qpos(self, new_qpos: jax.Array, qpos: jax.Array) -> jax.Array:
        """Set floating base position."""
        return qpos.at[
            self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7
        ].set(new_qpos)

    def set_floating_base_qvel(self, new_qvel: jax.Array, qvel: jax.Array) -> jax.Array:
        """Set floating base velocity."""
        return qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6
        ].set(new_qvel)

    # Sensor readings
    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Return the gravity vector in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, self.robot_config.gravity_sensor
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, self.robot_config.global_angvel_sensor
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, self.robot_config.local_linvel_sensor
        )

    def get_accelerometers(self, data: mjx.Data) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        acc_vecs = [
            mjx_env.get_sensor_data(self.mj_model, data, name)
            for name in self.robot_config.accelerometer_sensors
        ]
        return jp.concatenate(acc_vecs, axis=0)

    def get_gyros(self, data: mjx.Data) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        gyro_vecs = [
            mjx_env.get_sensor_data(self.mj_model, data, name)
            for name in self.robot_config.gyro_sensors
        ]
        return jp.concatenate(gyro_vecs, axis=0)

    # Accessors
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
