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

from playground_amp.configs.training_config import get_robot_config


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
        if not hasattr(config, "model_path") or config.model_path is None:
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

        # Foot contact geom IDs (from robot_config.yaml)
        # These are used for computing foot contact features for AMP discriminator
        # v0.5.0: Use explicit keys instead of array indexing (no order assumptions)
        feet_config = self._robot_config.raw_config.get("feet", {})

        # Get geom IDs using explicit config keys - order: [left_toe, left_heel, right_toe, right_heel]
        # Explicit keys - clear semantic meaning, no array order dependency
        left_toe_name = feet_config.get("left_toe", "left_toe")
        left_heel_name = feet_config.get("left_heel", "left_heel")
        right_toe_name = feet_config.get("right_toe", "right_toe")
        right_heel_name = feet_config.get("right_heel", "right_heel")

        try:
            self._left_toe_geom_id = self._mj_model.geom(left_toe_name).id
            self._left_heel_geom_id = self._mj_model.geom(left_heel_name).id
            self._right_toe_geom_id = self._mj_model.geom(right_toe_name).id
            self._right_heel_geom_id = self._mj_model.geom(right_heel_name).id
        except Exception as e:
            raise RuntimeError(
                f"Could not find foot geoms in MuJoCo model. "
                f"Expected: {left_toe_name}, {left_heel_name}, {right_toe_name}, {right_heel_name}. "
                f"Run 'cd assets && python post_process.py' to add foot geom names to XML. "
                f"Original error: {e}"
            )

        self._foot_geom_ids = jp.array(
            [
                self._left_toe_geom_id,
                self._left_heel_geom_id,
                self._right_toe_geom_id,
                self._right_heel_geom_id,
            ]
        )
        print(
            f"  Foot geom IDs: L_toe={self._left_toe_geom_id}, L_heel={self._left_heel_geom_id}, "
            f"R_toe={self._right_toe_geom_id}, R_heel={self._right_heel_geom_id}"
        )

        # Contact detection parameters (configurable from env config)
        # contact_threshold: Minimum force (N) to consider contact (not currently used)
        # contact_scale: Divisor for tanh normalization (10.0 means 10N → tanh(1) ≈ 0.76)
        self._contact_threshold = getattr(self._config, "contact_threshold", 1.0)
        self._contact_scale = getattr(self._config, "contact_scale", 10.0)

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
        base_info: dict | None = None,
    ) -> WildRobotEnvState:
        """Create initial environment state from qpos (shared by reset and auto-reset).

        Args:
            qpos: Initial joint positions (including floating base)
            velocity_cmd: Velocity command for this episode
            base_info: Optional base info dict to preserve (for auto-reset).
                       When provided, env fields are merged on top of base_info
                       to preserve wrapper-injected fields (truncation, episode_done, etc.)

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

        # Info - merge with base_info if provided (preserves wrapper fields during auto-reset)
        env_info = {
            "step_count": 0,
            "prev_action": jp.zeros(self.action_size),
            "truncated": jp.zeros(()),  # No truncation at reset
            "foot_contacts": self.get_foot_contacts(data),  # For AMP discriminator
            "root_height": height,  # v0.6.1: Actual root height for AMP features
            # v0.9.1: Track previous root pose for finite-diff velocity computation
            # This aligns policy velocity with reference velocity (both finite-diff)
            "prev_root_pos": self.get_floating_base_qpos(data.qpos)[0:3],  # (3,)
            "prev_root_quat": self.get_floating_base_qpos(data.qpos)[3:7],  # (4,) wxyz
        }

        # Merge: base_info (wrapper fields) + env_info (env fields override)
        if base_info is not None:
            info = dict(base_info)
            info.update(env_info)
        else:
            info = env_info

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

        # Compute observation with finite-diff velocities
        # v0.9.1: Pass previous root pose for FD velocity computation
        prev_root_pos = state.info["prev_root_pos"]
        prev_root_quat = state.info["prev_root_quat"]
        obs = self._get_obs(
            data, filtered_action, velocity_cmd, prev_root_pos, prev_root_quat
        )

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
        # v0.9.1: Update previous root pose for finite-diff velocity
        curr_root_pos = self.get_floating_base_qpos(data.qpos)[0:3]
        curr_root_quat = self.get_floating_base_qpos(data.qpos)[3:7]

        # v0.10.0: Preserve existing info fields from wrapper (Brax compatibility)
        # The Brax training wrapper adds fields like 'truncation', 'episode_done',
        # 'episode_metrics', 'first_obs', 'steps', 'first_state'. We need to preserve
        # these to maintain pytree structure compatibility with jax.lax.scan.
        info = dict(state.info)  # Copy existing fields
        info.update(
            {
                "step_count": step_count + 1,
                "prev_action": filtered_action,
                "truncated": truncated,  # 1.0 if reached max steps (success), 0.0 otherwise
                "foot_contacts": self.get_foot_contacts(data),  # For AMP discriminator
                "root_height": height,  # v0.6.1: Actual root height for AMP features
                # v0.9.1: Update previous root pose for finite-diff velocity
                "prev_root_pos": curr_root_pos,
                "prev_root_quat": curr_root_quat,
            }
        )

        # =================================================================
        # Auto-reset on termination
        # When done=True, reset physics state to initial keyframe but keep
        # done=True so the algorithm knows an episode ended.
        # =================================================================
        # Pass state.info as base_info to preserve wrapper fields during auto-reset
        reset_state = self._make_initial_state(
            self._init_qpos, velocity_cmd, base_info=state.info
        )

        final_data, final_obs, final_metrics, final_info = jax.lax.cond(
            done > 0.5,
            lambda: (
                reset_state.data,
                reset_state.obs,
                reset_state.metrics,
                reset_state.info,  # Already has wrapper fields preserved via base_info
            ),
            lambda: (data, obs, metrics, info),
        )

        return WildRobotEnvState(
            data=final_data,
            obs=final_obs,
            reward=reward,  # Keep original reward (from before reset)
            done=done,  # Keep done=True so algorithm knows episode ended
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
        prev_root_pos: Optional[jax.Array] = None,
        prev_root_quat: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Build observation vector.

        v0.9.1: CRITICAL FIX - Use finite-diff velocities for AMP parity.
        When prev_root_pos/quat are provided, velocities are computed from
        position changes (matching reference data). Otherwise, use qvel (for reset).

        v0.7.0: All velocities are now in heading-local frame for AMP parity.
        This is done transparently - the observation layout remains the same.

        Observation includes:
        - Base orientation (gravity vector in local frame): 3
        - Base angular velocity (heading-local, from FD or qvel): 3
        - Base linear velocity (heading-local, from FD or qvel): 3
        - Joint positions (actuated): 8
        - Joint velocities (actuated): 8
        - Previous action: 8
        - Velocity command: 1
        - Padding: 1

        Total: 35

        Args:
            data: MJX simulation data
            action: Previous action
            velocity_cmd: Current velocity command
            prev_root_pos: Previous root position for FD velocity (optional)
            prev_root_quat: Previous root quaternion for FD velocity (optional)

        Returns:
            Observation vector (35,)
        """
        # Gravity in local frame (from sensor)
        gravity = self.get_gravity(data)

        # v0.9.1: Use finite-diff velocities when prev_root_* are available
        # This is the key fix for AMP parity - matches reference data computation
        if prev_root_pos is not None and prev_root_quat is not None:
            # Finite-diff velocities (heading-local) - matches reference exactly
            linvel = self.get_heading_local_linvel_fd(data, prev_root_pos, self.dt)
            angvel = self.get_heading_local_angvel_fd(data, prev_root_quat, self.dt)
        else:
            # qvel-based velocities (for reset when no previous state)
            # These will be zero at reset anyway
            linvel = self.get_heading_local_linvel(data)
            angvel = self.get_heading_local_angvel(data)

        # Joint positions and velocities (squeeze to handle vmap extra dims)
        joint_pos = self.get_actuator_joint_qpos(data.qpos).squeeze()
        joint_vel = self.get_actuator_joints_qvel(data.qvel).squeeze()

        # Build observation
        obs = jp.concatenate(
            [
                gravity,  # 0-2: 3
                angvel,  # 3-5: 3 (heading-local, FD when available)
                linvel,  # 6-8: 3 (heading-local, FD when available)
                joint_pos,  # 9-16: 8 (actuated joints only)
                joint_vel,  # 17-24: 8 (actuated joints only)
                action,  # 25-32: 8
                jp.array([velocity_cmd]),  # 33: 1
                jp.zeros(1),  # 34: padding
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

    def get_root_quat(self, data: mjx.Data) -> jax.Array:
        """Get root quaternion (wxyz format from MuJoCo qpos)."""
        # MuJoCo qpos stores quaternion as [w, x, y, z] at indices 3-6
        return self.get_floating_base_qpos(data.qpos)[3:7]

    def get_heading_sin_cos(self, data: mjx.Data) -> jax.Array:
        """Get heading (yaw) as sin/cos for observation.

        Extracts yaw angle from root quaternion and returns [sin(yaw), cos(yaw)].
        This allows the policy and AMP features to compute heading-local frame
        velocities for rotation invariance.

        Returns:
            heading_sin_cos: (2,) array of [sin(yaw), cos(yaw)]
        """
        # Get quaternion in wxyz format
        quat = self.get_root_quat(data)  # [w, x, y, z]

        # v0.7.0: Normalize quaternion for numerical stability
        # MuJoCo quaternions are typically normalized but can drift slightly
        quat = quat / (jp.linalg.norm(quat) + 1e-8)

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Extract yaw (rotation around z-axis) from quaternion
        # Using standard euler extraction formula for ZYX convention
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        # Normalize to get sin/cos directly
        norm = jp.sqrt(siny_cosp * siny_cosp + cosy_cosp * cosy_cosp + 1e-8)
        sin_yaw = siny_cosp / norm
        cos_yaw = cosy_cosp / norm

        return jp.array([sin_yaw, cos_yaw])

    def get_heading_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Get linear velocity in heading-local frame.

        v0.7.0: Heading-local = world frame rotated by -yaw.
        This matches the reference data definition in GMR.

        The heading-local frame provides:
        - Invariance to global heading (yaw)
        - Preserves forward/lateral velocity semantics

        Returns:
            linvel_heading: (3,) linear velocity in heading-local frame
        """
        # Get world-frame linear velocity from qvel (floating base)
        # qvel[0:3] = world-frame linear velocity of floating base
        linvel_world = self.get_floating_base_qvel(data.qvel)[0:3]

        # Get heading sin/cos
        heading = self.get_heading_sin_cos(data)  # [sin_yaw, cos_yaw]
        sin_yaw, cos_yaw = heading[0], heading[1]

        # Rotate world velocity to heading-local frame
        # R_heading_from_world = R_z(-yaw)
        # v_heading = R_z(-yaw) @ v_world
        vx, vy, vz = linvel_world[0], linvel_world[1], linvel_world[2]

        # R_z(-yaw) @ [vx, vy, vz]
        vx_heading = cos_yaw * vx + sin_yaw * vy
        vy_heading = -sin_yaw * vx + cos_yaw * vy
        vz_heading = vz  # z component unchanged by z rotation

        return jp.array([vx_heading, vy_heading, vz_heading])

    def get_heading_local_angvel(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity in heading-local frame.

        Converts world-frame angular velocity to heading-local frame.
        This provides rotation invariance for the AMP discriminator.

        Returns:
            angvel_heading: (3,) angular velocity in heading-local frame
        """
        # Get world-frame angular velocity
        angvel_world = self.get_global_angvel(data)  # [wx, wy, wz] in world frame

        # Get heading sin/cos
        heading = self.get_heading_sin_cos(data)
        sin_yaw, cos_yaw = heading[0], heading[1]

        # Rotate world angular velocity to heading-local frame
        # R_heading_from_world = R_z(-yaw)
        # v_heading = R_z(-yaw) @ v_world
        wx, wy, wz = angvel_world[0], angvel_world[1], angvel_world[2]

        # R_z(-yaw) @ [wx, wy, wz]
        wx_heading = cos_yaw * wx + sin_yaw * wy
        wy_heading = -sin_yaw * wx + cos_yaw * wy
        wz_heading = wz  # z component unchanged by z rotation

        return jp.array([wx_heading, wy_heading, wz_heading])

    # =========================================================================
    # v0.9.1: Finite-Difference Velocity Computation for AMP Parity
    # =========================================================================
    # These methods compute velocities from finite differences of qpos,
    # matching the reference data computation method exactly.
    # This eliminates the distribution shift between:
    # - Reference: finite-diff of root_pos (kinematic)
    # - Policy (old): qvel from MuJoCo dynamics (different under contacts)
    #
    # Industry best practice: Use FD velocities for imitation features
    # even if the controller uses qvel internally.

    def get_heading_local_linvel_fd(
        self,
        data: mjx.Data,
        prev_root_pos: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Get linear velocity via finite difference (heading-local frame).

        v0.9.1: Computes velocity from position change, matching reference data.
        This is the key fix for AMP feature parity.

        Args:
            data: Current MJX data
            prev_root_pos: Previous root position (3,)
            dt: Time step for finite difference (control_dt, e.g., 0.02s)

        Returns:
            linvel_heading: (3,) linear velocity in heading-local frame
        """
        # Current root position
        curr_root_pos = self.get_floating_base_qpos(data.qpos)[0:3]

        # Finite difference in world frame (same as reference)
        linvel_world = (curr_root_pos - prev_root_pos) / dt

        # Convert to heading-local frame
        heading = self.get_heading_sin_cos(data)
        sin_yaw, cos_yaw = heading[0], heading[1]

        vx, vy, vz = linvel_world[0], linvel_world[1], linvel_world[2]
        vx_heading = cos_yaw * vx + sin_yaw * vy
        vy_heading = -sin_yaw * vx + cos_yaw * vy
        vz_heading = vz

        return jp.array([vx_heading, vy_heading, vz_heading])

    def get_heading_local_angvel_fd(
        self,
        data: mjx.Data,
        prev_root_quat: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Get angular velocity via finite difference (heading-local frame).

        v0.9.1: Computes angular velocity from quaternion change using axis-angle.
        This matches the reference data computation method exactly.

        Method: omega = (q_curr * q_prev^-1).as_rotvec() / dt
        This gives world-frame angular velocity, then converted to heading-local.

        Args:
            data: Current MJX data
            prev_root_quat: Previous root quaternion (4,) in wxyz format
            dt: Time step for finite difference (control_dt, e.g., 0.02s)

        Returns:
            angvel_heading: (3,) angular velocity in heading-local frame
        """
        # Current quaternion (wxyz format from MuJoCo)
        curr_quat = self.get_floating_base_qpos(data.qpos)[3:7]

        # Compute relative rotation: R_delta = R_curr * R_prev^-1
        # In quaternion: q_delta = q_curr * q_prev_conjugate
        # For unit quaternions, conjugate = inverse

        # Previous quaternion conjugate (wxyz format: [w, -x, -y, -z])
        w_prev, x_prev, y_prev, z_prev = (
            prev_root_quat[0],
            prev_root_quat[1],
            prev_root_quat[2],
            prev_root_quat[3],
        )
        q_prev_conj = jp.array([w_prev, -x_prev, -y_prev, -z_prev])

        # Quaternion multiplication: q_delta = q_curr * q_prev_conj
        w_curr, x_curr, y_curr, z_curr = (
            curr_quat[0],
            curr_quat[1],
            curr_quat[2],
            curr_quat[3],
        )

        # Hamilton product (wxyz)
        w_delta = (
            w_curr * q_prev_conj[0]
            - x_curr * q_prev_conj[1]
            - y_curr * q_prev_conj[2]
            - z_curr * q_prev_conj[3]
        )
        x_delta = (
            w_curr * q_prev_conj[1]
            + x_curr * q_prev_conj[0]
            + y_curr * q_prev_conj[3]
            - z_curr * q_prev_conj[2]
        )
        y_delta = (
            w_curr * q_prev_conj[2]
            - x_curr * q_prev_conj[3]
            + y_curr * q_prev_conj[0]
            + z_curr * q_prev_conj[1]
        )
        z_delta = (
            w_curr * q_prev_conj[3]
            + x_curr * q_prev_conj[2]
            - y_curr * q_prev_conj[1]
            + z_curr * q_prev_conj[0]
        )

        # Convert quaternion to rotation vector (axis * angle)
        # For small rotations: rotvec ≈ 2 * [x, y, z] (when w ≈ 1)
        # General formula: rotvec = 2 * acos(w) * [x, y, z] / |[x, y, z]|

        # Vector part of quaternion
        vec = jp.array([x_delta, y_delta, z_delta])
        vec_norm = jp.linalg.norm(vec) + 1e-8

        # Angle from quaternion: angle = 2 * acos(w)
        # For numerical stability with w near ±1:
        # angle = 2 * atan2(|vec|, w)
        angle = 2.0 * jp.arctan2(vec_norm, jp.abs(w_delta))

        # Handle sign of w (quaternion double cover)
        angle = jp.where(w_delta < 0, -angle, angle)

        # Rotation vector = angle * axis
        axis = vec / vec_norm
        rotvec = angle * axis

        # Angular velocity = rotvec / dt (world frame)
        angvel_world = rotvec / dt

        # Convert to heading-local frame
        heading = self.get_heading_sin_cos(data)
        sin_yaw, cos_yaw = heading[0], heading[1]

        wx, wy, wz = angvel_world[0], angvel_world[1], angvel_world[2]
        wx_heading = cos_yaw * wx + sin_yaw * wy
        wy_heading = -sin_yaw * wx + cos_yaw * wy
        wz_heading = wz

        return jp.array([wx_heading, wy_heading, wz_heading])

    def get_foot_contacts(self, data: mjx.Data) -> jax.Array:
        """Get foot contact features for AMP discriminator.

        Extracts contact forces from MJX and converts to soft-thresholded
        contact values in range [0, 1] using tanh normalization.

        This follows industry best practices (DeepMind, ETH Zurich, NVIDIA):
        - Continuous values (not binary) for smoother gradients
        - tanh normalization: tanh(force / scale)
        - 4-dim output: [left_toe, left_heel, right_toe, right_heel]

        Args:
            data: MJX simulation data containing contact information

        Returns:
            foot_contacts: (4,) array of contact values in [0, 1]
                [0] = left_toe (left_foot_btm_front)
                [1] = left_heel (left_foot_btm_back)
                [2] = right_toe (right_foot_btm_front)
                [3] = right_heel (right_foot_btm_back)
        """
        # v0.5.0: _foot_geom_ids is guaranteed to exist (exception raised in __init__ if not found)

        def get_contact_force_for_geom(geom_id: int) -> jax.Array:
            """Get total contact force for a single geom."""
            # Check which contacts involve this geom (either as geom1 or geom2)
            geom1_match = data.contact.geom1 == geom_id
            geom2_match = data.contact.geom2 == geom_id
            is_our_contact = geom1_match | geom2_match

            # Get normal forces from constraint solver
            # efc_address[i] is the index where contact i's forces start in efc_force
            # The first element at each address is the normal force
            normal_forces = data.efc_force[data.contact.efc_address]

            # Sum forces for contacts involving this geom
            our_forces = jp.where(is_our_contact, jp.abs(normal_forces), 0.0)
            total_force = jp.sum(our_forces)

            return total_force

        # Get contact forces for each foot geom
        left_toe_force = get_contact_force_for_geom(self._left_toe_geom_id)
        left_heel_force = get_contact_force_for_geom(self._left_heel_geom_id)
        right_toe_force = get_contact_force_for_geom(self._right_toe_geom_id)
        right_heel_force = get_contact_force_for_geom(self._right_heel_geom_id)

        # Soft thresholding with tanh: continuous value in [0, 1]
        # scale=10.0 means: 10N → tanh(1) ≈ 0.76, 20N → tanh(2) ≈ 0.96
        foot_contacts = jp.array(
            [
                jp.tanh(left_toe_force / self._contact_scale),
                jp.tanh(left_heel_force / self._contact_scale),
                jp.tanh(right_toe_force / self._contact_scale),
                jp.tanh(right_heel_force / self._contact_scale),
            ]
        )

        return foot_contacts

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

        v0.7.0: Velocities are now in heading-local frame for AMP parity.
        The observation layout is unchanged.

        Observation components:
        - Gravity vector: 3
        - Angular velocity (heading-local): 3
        - Linear velocity (heading-local): 3
        - Joint positions: 8 (actuated joints only)
        - Joint velocities: 8 (actuated joints only)
        - Previous action: 8
        - Velocity command: 1
        - Padding: 1

        Total: 35
        """
        return 35
