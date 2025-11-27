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
"""Constants for WildRobot - shared across all implementations."""

from etils import epath
from dataclasses import dataclass
from typing import List

# Point to the main models directory (common/ is at project root level)
PROJECT_ROOT = epath.Path(__file__).parent.parent
WILDROBOT_MODEL_XML = PROJECT_ROOT / "models" / "v1" / "wildrobot.xml"
WILDROBOT_FLAT_TERRAIN_XML = PROJECT_ROOT / "models" / "v1" / "scene_flat_terrain.xml"
WILDROBOT_ROUGH_TERRAIN_XML = PROJECT_ROOT / "models" / "v1" / "scene_rough_terrain.xml"

tasks = ["wildrobot_flat", "wildrobot_rough"]

def is_valid_task(task_name: str) -> bool:
    return task_name in tasks

def task_to_xml(task_name: str) -> epath.Path:
    return {
        "wildrobot_flat": WILDROBOT_FLAT_TERRAIN_XML,
        "wildrobot_rough": WILDROBOT_ROUGH_TERRAIN_XML,
    }[task_name]


@dataclass
class RobotConfig:
    """Configuration for WildRobot.

    Notes on feet_sites vs feet_geoms:
    - feet_sites: Named MuJoCo sites attached to foot bodies. Use these for
        kinematic queries (positions/orientations via data.site_xpos/site_xmat)
        and site-anchored sensors.

    - left_feet_geoms / right_feet_geoms: MuJoCo collision geoms that actually
        make contact with the floor. Use these to build contact flags.
    """
    # Feet and contact geoms
    feet_sites: List[str]
    left_feet_geoms: List[str]
    right_feet_geoms: List[str]

    # Joint names (11 DOFs total: 5 per leg + 1 waist)
    joint_names: List[str]

    # Root body
    root_body: str

    # Sensor names
    gravity_sensor: str
    global_linvel_sensor: str
    global_angvel_sensor: str
    local_linvel_sensor: str
    accelerometer_sensors: List[str]
    gyro_sensors: List[str]

    @property
    def feet_geoms(self) -> List[str]:
        """Convenience property that combines left and right feet geoms."""
        return self.left_feet_geoms + self.right_feet_geoms


# WildRobot configuration: 11 DOFs, no head
# Left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5 DOFs)
# Right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5 DOFs)
# Torso: waist (1 DOF)
WILDROBOT_CONFIG = RobotConfig(
    feet_sites=["left_foot_mimic", "right_foot_mimic"],
    left_feet_geoms=["left_foot_btm_front", "left_foot_btm_back"],
    right_feet_geoms=["right_foot_btm_front", "right_foot_btm_back"],
    joint_names=[
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee_pitch",
        "right_ankle_pitch",
        "right_foot_roll",
        "left_hip_pitch",
        "left_hip_roll",
        "left_knee_pitch",
        "left_ankle_pitch",
        "left_foot_roll",
        "waist_yaw",
    ],
    root_body="waist",
    gravity_sensor="pelvis_upvector",
    global_linvel_sensor="pelvis_global_linvel",
    global_angvel_sensor="pelvis_global_angvel",
    local_linvel_sensor="pelvis_local_linvel",
    accelerometer_sensors=[
        "chest_imu_accel",
        "left_knee_imu_accel",
        "right_knee_imu_accel",
    ],
    gyro_sensors=[
        "chest_imu_gyro",
        "left_knee_imu_gyro",
        "right_knee_imu_gyro",
    ],
)

ROBOT_CONFIGS = {
    "wildrobot_flat": WILDROBOT_CONFIG,
    "wildrobot_rough": WILDROBOT_CONFIG,
}
