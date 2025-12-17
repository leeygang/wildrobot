"""WildRobot environment extending mujoco_playground's MjxEnv.

This module provides a fully JAX-compatible WildRobot environment that:
- Extends mjx_env.MjxEnv for native Brax compatibility
- Is fully JIT-able with jax.jit, jax.vmap
- Uses pure JAX arrays (no numpy conversions)
- Supports mujoco_playground.wrapper.wrap_for_brax_training()

Architecture:
    WildRobotEnv(mjx_env.MjxEnv)
        └── reset(rng) -> WildRobotEnvState
        └── step(state, action) -> WildRobotEnvState
        └── Robot utilities (joint access, sensors)

Usage:
    from playground_amp.envs.wildrobot_env import WildRobotEnv, default_config
    from mujoco_playground import wrapper
    from brax.training.agents.ppo import train as ppo

    env = WildRobotEnv(config=default_config())

    # Train with Brax PPO
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from flax import struct
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env


# =============================================================================
# State Definition
# =============================================================================


@struct.dataclass
class WildRobotEnvState(mjx_env.State):
    """WildRobot environment state with Brax compatibility.

    Extends mujoco_playground's State to add pipeline_state field,
    which is required by Brax's training wrapper.
    """

    pipeline_state: mjx.Data = None


# =============================================================================
# Robot Configuration
# =============================================================================


@dataclass
class RobotConfig:
    """Configuration for WildRobot hardware."""

    feet_sites: List[str]
    left_feet_geoms: List[str]
    right_feet_geoms: List[str]
    joint_names: List[str]
    root_body: str
    gravity_sensor: str
    global_linvel_sensor: str
    global_angvel_sensor: str
    local_linvel_sensor: str
    accelerometer_sensors: List[str]
    gyro_sensors: List[str]

    @property
    def feet_geoms(self) -> List[str]:
        return self.left_feet_geoms + self.right_feet_geoms


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


# =============================================================================
# Configuration Factory
# =============================================================================


def default_config() -> config_dict.ConfigDict:
    """Return default configuration for WildRobotEnv."""
    config = config_dict.ConfigDict()

    # Simulation timing
    config.ctrl_dt = 0.02  # 50Hz control frequency
    config.sim_dt = 0.002  # 500Hz simulation frequency

    # Robot parameters
    config.target_height = 0.45  # Target standing height (meters)
    config.min_height = 0.2  # Termination height (too low)
    config.max_height = 0.7  # Termination height (too high)

    # Velocity command
    config.min_velocity = 0.5  # m/s
    config.max_velocity = 1.0  # m/s

    # Action filtering
    config.use_action_filter = True
    config.action_filter_alpha = 0.7

    # Episode length
    config.max_episode_steps = 500

    # Reward weights
    config.reward_weights = config_dict.ConfigDict()
    config.reward_weights.forward_velocity = 1.0
    config.reward_weights.healthy = 0.1
    config.reward_weights.action_rate = -0.01
    config.reward_weights.joint_velocity = -0.001

    return config


def load_config(path: str) -> config_dict.ConfigDict:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        ConfigDict with all required parameters.
    """
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config = config_dict.ConfigDict()

    # Required parameters - will raise KeyError if missing
    required_params = [
        "ctrl_dt",
        "sim_dt",
        "target_height",
        "min_height",
        "max_height",
        "min_velocity",
        "max_velocity",
        "use_action_filter",
        "action_filter_alpha",
        "max_episode_steps",
    ]

    for param in required_params:
        if param not in data:
            raise KeyError(f"Missing required config parameter: {param}")
        setattr(config, param, data[param])

    # Reward weights (nested dict)
    if "reward_weights" not in data:
        raise KeyError("Missing required config parameter: reward_weights")

    config.reward_weights = config_dict.ConfigDict()
    for key, value in data["reward_weights"].items():
        setattr(config.reward_weights, key, value)

    return config


# =============================================================================
# Asset Loading
# =============================================================================


def get_assets(root_path: str) -> Dict[str, bytes]:
    """Load all assets (STL files) for the environment."""
    assets = {}
    path = epath.Path(root_path)
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


def find_model_path() -> epath.Path:
    """Find the WildRobot model XML file."""
    # Try multiple locations
    candidates = [
        epath.Path(__file__).parent.parent.parent / "assets" / "wildrobot.xml",
        epath.Path(__file__).parent.parent.parent / "models" / "v1" / "scene_flat_terrain.xml",
        epath.Path(__file__).parent.parent.parent.parent / "assets" / "wildrobot.xml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find WildRobot model. Tried: {[str(p) for p in candidates]}"
    )


# =============================================================================
# Environment
# =============================================================================


class WildRobotEnv(mjx_env.MjxEnv):
    """WildRobot walking environment extending MjxEnv.

    This class provides:
    - Fully JAX/JIT compatible reset() and step()
    - Native Brax compatibility via pipeline_state
    - Robot utilities (joint access, sensor reading)

    Example:
        config = default_config()
        env = WildRobotEnv(config=config)

        # JIT-compiled reset and step
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)

        # Vectorized environments
        batched_reset = jax.vmap(env.reset)
        batched_step = jax.vmap(env.step)
    """

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize WildRobotEnv.

        Args:
            config: Configuration dict with ctrl_dt, sim_dt, etc.
            config_overrides: Optional overrides for config values.
        """
        if config is None:
            config = default_config()

        super().__init__(config, config_overrides)

        # Load MuJoCo model
        # Load model and setup robot infrastructure
        self._load_model()

        # Config parameters (required - defined in default_config())
        self._target_height = config.target_height
        self._min_height = config.min_height
        self._max_height = config.max_height
        self._min_velocity = config.min_velocity
        self._max_velocity = config.max_velocity
        self._use_action_filter = config.use_action_filter
        self._action_filter_alpha = config.action_filter_alpha
        self._max_episode_steps = config.max_episode_steps
        self._reward_weights = config.reward_weights

        print(f"WildRobotEnv initialized:")
        print(f"  Actuators: {len(self.actuator_names)}")
        print(f"  Floating base: {self.floating_base_name}")
        print(f"  Control dt: {self.dt}s, Sim dt: {self.sim_dt}s")

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_model(self) -> None:
        """Load MuJoCo model and setup robot infrastructure.

        This method:
        - Finds and loads the XML model file
        - Creates MJX model for JAX compatibility
        - Sets up actuator and joint mappings
        - Extracts initial qpos from keyframe or qpos0
        """
        xml_path = find_model_path()
        root_path = xml_path.parent

        print(f"Loading WildRobot model from: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=get_assets(root_path)
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = str(xml_path)

        # Robot configuration
        self.robot_config = WILDROBOT_CONFIG

        # Get floating base info
        self.floating_base_name = [
            self._mj_model.jnt(k).name
            for k in range(self._mj_model.njnt)
            if self._mj_model.jnt(k).type == 0
        ][0]

        # Actuator and joint mappings
        self.actuator_names = [
            self._mj_model.actuator(k).name for k in range(self._mj_model.nu)
        ]
        self.joint_names = [
            self._mj_model.jnt(k).name for k in range(self._mj_model.njnt)
        ]

        self.actuator_joint_ids = [
            self._get_joint_id(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = jp.array(
            [self._mj_model.joint(n).qposadr for n in self.actuator_names]
        )
        self.actuator_qvel_addr = jp.array(
            [self._mj_model.jnt_dofadr[jid] for jid in self.actuator_joint_ids]
        )

        self._floating_base_qpos_addr = self._mj_model.jnt_qposadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][0]
        self._floating_base_qvel_addr = self._mj_model.jnt_dofadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][0]

        # Get initial qpos from model's keyframe or qpos0
        # MuJoCo stores keyframe qpos in key_qpos (shape: nkey x nq)
        if self._mj_model.nkey > 0:
            # Use first keyframe (typically "home" pose)
            self._init_qpos = jp.array(self._mj_model.key_qpos[0])
            print(f"  Using keyframe qpos from model (nkey={self._mj_model.nkey})")
        else:
            # Fall back to qpos0 (default initial state)
            self._init_qpos = jp.array(self._mj_model.qpos0)
            print(f"  Using qpos0 from model")

        # Extract joint-only qpos for control default (exclude floating base: first 7 values)
        self._default_joint_qpos = self._init_qpos[7:]

    # =========================================================================
    # MjxEnv Interface
    # =========================================================================

    def reset(self, rng: jax.Array) -> WildRobotEnvState:
        """Reset environment to initial state.

        Args:
            rng: JAX random key for randomization.

        Returns:
            Initial WildRobotEnvState.
        """
        rng, key1, key2 = jax.random.split(rng, 3)

        # Sample velocity command
        velocity_cmd = jax.random.uniform(
            key1, shape=(), minval=self._min_velocity, maxval=self._max_velocity
        )

        # Use initial qpos from model (keyframe or qpos0)
        qpos = self._init_qpos.copy()
        qvel = jp.zeros(self._mj_model.nv)

        # Add small random noise to joint positions (not floating base)
        joint_noise = jax.random.uniform(
            key2, shape=self._default_joint_qpos.shape, minval=-0.05, maxval=0.05
        )
        qpos = qpos.at[7:7 + len(self._default_joint_qpos)].set(
            self._default_joint_qpos + joint_noise
        )

        # Create MJX data and run forward kinematics
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_joint_qpos)
        data = mjx.forward(self._mjx_model, data)

        # Build observation
        obs = self._get_obs(data, jp.zeros(self.action_size), velocity_cmd)

        # Initial reward and done
        reward = jp.zeros(())
        done = jp.zeros(())

        # Metrics (all must be JAX arrays for scan compatibility)
        height = self.get_floating_base_qpos(data.qpos)[2]
        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": jp.zeros(()),
            "reward/total": jp.zeros(()),
            "reward/forward": jp.zeros(()),
            "reward/healthy": jp.zeros(()),
            "reward/action_rate": jp.zeros(()),
        }

        # Info (can contain Python types)
        info = {
            "step_count": 0,
            "prev_action": jp.zeros(self.action_size),
        }

        return WildRobotEnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
            pipeline_state=data,  # Required for Brax compatibility
        )

    def step(
        self, state: WildRobotEnvState, action: jax.Array
    ) -> WildRobotEnvState:
        """Step environment forward.

        Args:
            state: Current environment state.
            action: Action to apply (joint position targets).

        Returns:
            New WildRobotEnvState.
        """
        velocity_cmd = state.metrics["velocity_command"]
        step_count = state.info["step_count"]
        prev_action = state.info["prev_action"]

        # Action filtering (low-pass)
        if self._use_action_filter:
            alpha = self._action_filter_alpha
            filtered_action = alpha * prev_action + (1.0 - alpha) * action
        else:
            filtered_action = action

        # Apply action as control
        data = state.data.replace(ctrl=filtered_action)

        # Physics simulation (multiple substeps)
        def substep_fn(data, _):
            return mjx.step(self._mjx_model, data), None

        n_substeps = int(self.dt / self.sim_dt)
        data, _ = jax.lax.scan(substep_fn, data, None, length=n_substeps)

        # Compute observation
        obs = self._get_obs(data, filtered_action, velocity_cmd)

        # Compute reward
        reward, reward_components = self._get_reward(
            data, filtered_action, prev_action, velocity_cmd
        )

        # Check termination
        done = self._get_done(data, step_count + 1)

        # Update metrics
        height = self.get_floating_base_qpos(data.qpos)[2]
        forward_vel = self.get_local_linvel(data)[0]

        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": forward_vel,
            "reward/total": reward,
            **reward_components,
        }

        # Update info
        info = {
            "step_count": step_count + 1,
            "prev_action": filtered_action,
        }

        return WildRobotEnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
            pipeline_state=data,
        )

    # =========================================================================
    # Observation, Reward, Done
    # =========================================================================

    def _get_obs(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
    ) -> jax.Array:
        """Build observation vector.

        Observation includes:
        - Base orientation (gravity vector in local frame): 3
        - Base angular velocity: 3
        - Base linear velocity (local): 3
        - Joint positions: 11
        - Joint velocities: 11
        - Previous action: 11
        - Velocity command: 1
        - Phase signal (sin/cos): 2 (optional)

        Total: 44 (without phase) or 46 (with phase)
        """
        # Gravity in local frame (from sensor)
        gravity = self.get_gravity(data)

        # Angular velocity
        angvel = self.get_global_angvel(data)

        # Local linear velocity
        linvel = self.get_local_linvel(data)

        # Joint positions and velocities
        joint_pos = self.get_actuator_joint_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        # Build observation
        obs = jp.concatenate([
            gravity,  # 3
            angvel,  # 3
            linvel,  # 3
            joint_pos,  # 11
            joint_vel,  # 11
            action,  # 11
            jp.array([velocity_cmd]),  # 1
            jp.zeros(1),  # padding to 44
        ])

        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        prev_action: jax.Array,
        velocity_cmd: jax.Array,
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute reward.

        Returns:
            (total_reward, reward_components_dict)
        """
        weights = self._reward_weights

        # Forward velocity tracking
        forward_vel = self.get_local_linvel(data)[0]
        vel_error = jp.abs(forward_vel - velocity_cmd)
        forward_reward = jp.exp(-vel_error * 2.0)  # Exponential reward

        # Healthy reward (staying upright)
        height = self.get_floating_base_qpos(data.qpos)[2]
        healthy = jp.where(
            (height > self._min_height) & (height < self._max_height),
            1.0,
            0.0,
        )

        # Action rate penalty
        action_diff = action - prev_action
        action_rate = jp.sum(jp.square(action_diff))

        # Joint velocity penalty
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        joint_vel_penalty = jp.sum(jp.square(joint_vel))

        # Combine rewards (weights from config, no defaults)
        w_forward = weights.forward_velocity
        w_healthy = weights.healthy
        w_action = weights.action_rate
        w_joint_vel = weights.joint_velocity

        total = (
            w_forward * forward_reward
            + w_healthy * healthy
            + w_action * action_rate
            + w_joint_vel * joint_vel_penalty
        )

        components = {
            "reward/forward": forward_reward,
            "reward/healthy": healthy,
            "reward/action_rate": action_rate,
        }

        return total, components

    def _get_done(self, data: mjx.Data, step_count: int) -> jax.Array:
        """Check termination conditions.

        Terminates if:
        - Height too low or too high
        - Episode length exceeded
        """
        height = self.get_floating_base_qpos(data.qpos)[2]

        # Height termination
        height_done = (height < self._min_height) | (height > self._max_height)

        # Episode length termination
        length_done = step_count >= self._max_episode_steps

        return jp.where(height_done | length_done, 1.0, 0.0)

    # =========================================================================
    # Robot Utilities (Joint/Sensor Access)
    # =========================================================================

    def _get_joint_id(self, name: str) -> int:
        """Get joint ID from name."""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def get_actuator_joint_qpos(self, qpos: jax.Array) -> jax.Array:
        """Get joint positions for actuated joints."""
        return qpos[self.actuator_joint_qpos_addr]

    def get_actuator_joints_qvel(self, qvel: jax.Array) -> jax.Array:
        """Get joint velocities for actuated joints."""
        return qvel[self.actuator_qvel_addr]

    def get_floating_base_qpos(self, qpos: jax.Array) -> jax.Array:
        """Get floating base position (7D: 3 pos + 4 quat)."""
        return qpos[self._floating_base_qpos_addr:self._floating_base_qpos_addr + 7]

    def get_floating_base_qvel(self, qvel: jax.Array) -> jax.Array:
        """Get floating base velocity (6D: 3 lin + 3 ang)."""
        return qvel[self._floating_base_qvel_addr:self._floating_base_qvel_addr + 6]

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Get gravity vector in local frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self.robot_config.gravity_sensor
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity in world frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self.robot_config.global_angvel_sensor
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Get linear velocity in local frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self.robot_config.local_linvel_sensor
        )

    # =========================================================================
    # Properties (MjxEnv interface)
    # =========================================================================

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

    @property
    def observation_size(self) -> int:
        return 44
