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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from flax import struct
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from playground_amp.configs.config import get_robot_config


# =============================================================================
# Asset Loading
# =============================================================================


def get_assets(root_path: Path) -> Dict[str, bytes]:
    """Load all assets from the WildRobot assets directory.

    Args:
        root_path: Path to assets directory (REQUIRED)

    Returns:
        Dict of asset name to bytes content
    """
    assets = {}

    # Load XML files
    mjx_env.update_assets(assets, root_path, "*.xml")

    # Load STL meshes from assets/assets subdirectory
    meshes_path = root_path / "assets"
    if meshes_path.exists():
        mjx_env.update_assets(assets, meshes_path, "*.stl")

    return assets


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
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize WildRobotEnv.

        Args:
            config: Configuration dict with ctrl_dt, sim_dt, etc.
                    Must be provided - load from training config YAML.
            config_overrides: Optional overrides for config values.
        """
        super().__init__(config, config_overrides)

        # Model path (required - must be in config)
        if not hasattr(config, 'model_path') or config.model_path is None:
            raise ValueError(
                "model_path is required in config. "
                "Add 'model_path: assets/scene_flat_terrain.xml' to your config."
            )
        self._model_path = Path(config.model_path)

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
        - Loads the XML model file from config path
        - Creates MJX model for JAX compatibility
        - Sets up actuator and joint mappings
        - Extracts initial qpos from keyframe or qpos0
        """
        # Resolve model path relative to project root
        project_root = Path(__file__).parent.parent.parent
        xml_path = project_root / self._model_path

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {xml_path}. "
                f"Check model_path in config: {self._model_path}"
            )

        root_path = xml_path.parent

        print(f"Loading WildRobot model from: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=get_assets(root_path)
        )
        self._mj_model.opt.timestep = self.sim_dt

        # Create MJX model for JAX compatibility
        self._mjx_model = mjx.put_model(self._mj_model)

        # Get robot config (loaded by train.py at startup)
        self._robot_config = get_robot_config()

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

        self.actuator_joint_ids = [self._get_joint_id(n) for n in self.actuator_names]
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
            # Debug: print initial height from keyframe
            init_height = float(self._init_qpos[2])
            print(f"  Initial height from keyframe: {init_height:.4f}m")
        else:
            # Fall back to qpos0 (default initial state)
            self._init_qpos = jp.array(self._mj_model.qpos0)
            print(f"  Using qpos0 from model")
            init_height = float(self._init_qpos[2])
            print(f"  Initial height from qpos0: {init_height:.4f}m")

        # Extract joint-only qpos for control default (exclude floating base: first 7 values)
        self._default_joint_qpos = self._init_qpos[7:]

    # =========================================================================
    # MjxEnv Interface
    # =========================================================================

    def _make_initial_state(
        self,
        qpos: jax.Array,
        velocity_cmd: jax.Array,
    ) -> WildRobotEnvState:
        """Create initial environment state from qpos (shared by reset and auto-reset).

        Args:
            qpos: Initial joint positions (including floating base)
            velocity_cmd: Velocity command for this episode

        Returns:
            Initial WildRobotEnvState
        """
        qvel = jp.zeros(self._mj_model.nv)

        # Create MJX data and run forward kinematics
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_joint_qpos)
        data = mjx.forward(self._mjx_model, data)

        # Build observation
        action = jp.zeros(self.action_size)
        obs = self._get_obs(data, action, velocity_cmd)

        # Initial reward and done
        reward = jp.zeros(())
        done = jp.zeros(())

        # Compute actual initial metrics (not zeros)
        height = self.get_floating_base_qpos(data.qpos)[2]

        # Forward reward: robot is stationary (vel=0), so miss target by velocity_cmd
        forward_reward = jp.exp(-velocity_cmd * 2.0)

        # Healthy reward: robot is upright at reset
        healthy_reward = jp.where(
            (height > self._min_height) & (height < self._max_height),
            1.0,
            0.0,
        )

        # Action rate: no previous action, so no penalty
        action_rate = jp.zeros(())

        # Compute total reward with weights
        weights = self._reward_weights
        total_reward = (
            weights.forward_velocity * forward_reward
            + weights.healthy * healthy_reward
            + weights.action_rate * action_rate
        )

        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": jp.zeros(()),  # Robot starts stationary
            "reward/total": total_reward,
            "reward/forward": forward_reward,
            "reward/healthy": healthy_reward,
            "reward/action_rate": action_rate,
        }

        # Info
        info = {
            "step_count": 0,
            "prev_action": jp.zeros(self.action_size),
            "truncated": jp.zeros(()),  # No truncation at reset
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

        # Add small random noise to joint positions (not floating base)
        joint_noise = jax.random.uniform(
            key2, shape=self._default_joint_qpos.shape, minval=-0.05, maxval=0.05
        )
        qpos = qpos.at[7 : 7 + len(self._default_joint_qpos)].set(
            self._default_joint_qpos + joint_noise
        )

        return self._make_initial_state(qpos, velocity_cmd)

    def step(self, state: WildRobotEnvState, action: jax.Array) -> WildRobotEnvState:
        """Step environment forward with auto-reset on termination.

        Args:
            state: Current environment state.
            action: Action to apply (joint position targets).

        Returns:
            New WildRobotEnvState. If episode terminated, state is auto-reset
            but done=True is still returned so the algorithm knows.
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

        # Check termination (get done, terminated, and truncated flags)
        done, terminated, truncated = self._get_termination(data, step_count + 1)

        # Update metrics (before potential reset)
        height = self.get_floating_base_qpos(data.qpos)[2]
        forward_vel = self.get_local_linvel(data)[0]

        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": forward_vel,
            "reward/total": reward,
            **reward_components,
        }

        # Update info with truncated flag for success rate tracking
        info = {
            "step_count": step_count + 1,
            "prev_action": filtered_action,
            "truncated": truncated,  # 1.0 if reached max steps (success), 0.0 otherwise
        }

        # =================================================================
        # Auto-reset on termination
        # When done=True, reset physics state to initial keyframe but keep
        # done=True so the algorithm knows an episode ended.
        # =================================================================
        reset_state = self._make_initial_state(self._init_qpos, velocity_cmd)

        final_data, final_obs, final_metrics, final_info = jax.lax.cond(
            done > 0.5,
            lambda: (reset_state.data, reset_state.obs, reset_state.metrics, reset_state.info),
            lambda: (data, obs, metrics, info),
        )

        return WildRobotEnvState(
            data=final_data,
            obs=final_obs,
            reward=reward,  # Keep original reward (from before reset)
            done=done,      # Keep done=True so algorithm knows episode ended
            metrics=final_metrics,
            info=final_info,
            pipeline_state=final_data,
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
        - Joint positions (actuated): 9
        - Joint velocities (actuated): 9
        - Previous action: 9
        - Velocity command: 1
        - Padding: 1

        Total: 38
        """
        # Gravity in local frame (from sensor)
        gravity = self.get_gravity(data)

        # Angular velocity
        angvel = self.get_global_angvel(data)

        # Local linear velocity
        linvel = self.get_local_linvel(data)

        # Joint positions and velocities (squeeze to handle vmap extra dims)
        joint_pos = self.get_actuator_joint_qpos(data.qpos).squeeze()
        joint_vel = self.get_actuator_joints_qvel(data.qvel).squeeze()

        # Build observation
        obs = jp.concatenate(
            [
                gravity,  # 3
                angvel,  # 3
                linvel,  # 3
                joint_pos,  # 9 (actuated joints only)
                joint_vel,  # 9 (actuated joints only)
                action,  # 9
                jp.array([velocity_cmd]),  # 1
                jp.zeros(1),  # padding to 38
            ]
        )

        # Ensure float32 output (x64 mode can produce float64)
        return obs.astype(jp.float32)

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

        # Forward velocity tracking (sharper exponential to penalize standing still)
        forward_vel = self.get_local_linvel(data)[0]
        vel_error = jp.abs(forward_vel - velocity_cmd)
        # TUNED: 2.0 → 4.0 (standing still: exp(-0.7*4)=0.06 vs exp(-0.7*2)=0.25)
        forward_reward = jp.exp(-vel_error * 4.0)

        # Forward movement bonus (incentivize any forward motion)
        # Clips negative velocity to 0, caps at 1.0 m/s
        forward_bonus = jp.clip(forward_vel, 0.0, 1.0) * 0.3

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
            w_forward * (forward_reward + forward_bonus)  # Include movement bonus
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

    def _get_termination(
        self, data: mjx.Data, step_count: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Check termination conditions.

        Returns:
            (done, terminated, truncated) where:
            - done: True if episode ended (terminated OR truncated)
            - terminated: True if ended due to failure (falling)
            - truncated: True if ended due to time limit (success)
        """
        height = self.get_floating_base_qpos(data.qpos)[2]

        # Height termination (failure - robot fell)
        terminated = (height < self._min_height) | (height > self._max_height)

        # Episode length truncation (success - reached max steps without falling)
        truncated = (step_count >= self._max_episode_steps) & ~terminated

        # Combined done flag
        done = terminated | truncated

        return (
            jp.where(done, 1.0, 0.0),
            jp.where(terminated, 1.0, 0.0),
            jp.where(truncated, 1.0, 0.0),
        )

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
        return qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7]

    def get_floating_base_qvel(self, qvel: jax.Array) -> jax.Array:
        """Get floating base velocity (6D: 3 lin + 3 ang)."""
        return qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6]

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Get gravity vector in local frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self._robot_config.gravity_sensor
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity in world frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self._robot_config.global_angvel_sensor
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Get linear velocity in local frame."""
        return mjx_env.get_sensor_data(
            self._mj_model, data, self._robot_config.local_linvel_sensor
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
        """Return observation size.

        Observation components:
        - Gravity vector: 3
        - Angular velocity: 3
        - Linear velocity: 3
        - Joint positions: 9 (actuated joints only)
        - Joint velocities: 9 (actuated joints only)
        - Previous action: 9
        - Velocity command: 1
        - Padding: 1

        Total: 38
        """
        return 38
