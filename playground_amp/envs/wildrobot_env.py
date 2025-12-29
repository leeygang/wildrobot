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
    from playground_amp.envs.wildrobot_env import WildRobotEnv
    from playground_amp.configs.training_config import load_training_config
    from mujoco_playground import wrapper
    from brax.training.agents.ppo import train as ppo

    training_cfg = load_training_config("configs/ppo_walking.yaml")
    runtime_cfg = training_cfg.to_runtime_config()
    env = WildRobotEnv(config=runtime_cfg)

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
from flax import struct
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env

from playground_amp.configs.training_config import get_robot_config, TrainingConfig
from playground_amp.training.heading_math import (
    heading_local_angvel_fd_from_quat_jax,
    heading_local_from_world_jax,
    heading_local_linvel_fd_from_pos_jax,
    heading_sin_cos_from_quat_jax,
    pitch_roll_from_quat_jax,
)
from playground_amp.envs.env_types import WildRobotInfo, WR_INFO_KEY
from playground_amp.training.metrics_registry import (
    METRIC_NAMES,
    METRICS_VEC_KEY,
    NUM_METRICS,
    build_metrics_vec,
)


# =============================================================================
# Observation Layout
# =============================================================================


@dataclass(frozen=True)
class ObsLayout:
    """Defines the observation vector structure.

    This is the single source of truth for observation dimensions.
    Used by both _get_obs() and observation_size property.

    Layout:
        [0:3]   gravity      - Gravity vector in local frame
        [3:6]   angvel       - Angular velocity (heading-local)
        [6:9]   linvel       - Linear velocity (heading-local)
        [9:9+n] joint_pos    - Joint positions (n = num_actuators)
        [...]   joint_vel    - Joint velocities (n = num_actuators)
        [...]   action       - Previous action (n = num_actuators)
        [...]   velocity_cmd - Velocity command (1)
        [...]   padding      - Padding for alignment (1)
    """

    # Fixed dimensions (independent of robot)
    GRAVITY_DIM: int = 3
    ANGVEL_DIM: int = 3
    LINVEL_DIM: int = 3
    VELOCITY_CMD_DIM: int = 1
    PADDING_DIM: int = 1

    @classmethod
    def _get_num_actuators(cls) -> int:
        """Get number of actuators from robot config (single source of truth)."""
        return get_robot_config().action_dim

    @classmethod
    def total_size(cls) -> int:
        """Compute total observation size.

        Number of actuators is read from robot_config.

        Returns:
            Total observation dimension
        """
        num_actuators = cls._get_num_actuators()
        fixed = (
            cls.GRAVITY_DIM
            + cls.ANGVEL_DIM
            + cls.LINVEL_DIM
            + cls.VELOCITY_CMD_DIM
            + cls.PADDING_DIM
        )
        dynamic = 3 * num_actuators  # joint_pos + joint_vel + action
        return fixed + dynamic

    @classmethod
    def get_slices(cls) -> Dict[str, slice]:
        """Get named slices for each observation component.

        Useful for debugging or extracting specific components.
        Number of actuators is read from robot_config.

        Returns:
            Dict mapping component names to slice objects
        """
        num_actuators = cls._get_num_actuators()
        idx = 0
        slices = {}

        slices["gravity"] = slice(idx, idx + cls.GRAVITY_DIM)
        idx += cls.GRAVITY_DIM

        slices["angvel"] = slice(idx, idx + cls.ANGVEL_DIM)
        idx += cls.ANGVEL_DIM

        slices["linvel"] = slice(idx, idx + cls.LINVEL_DIM)
        idx += cls.LINVEL_DIM

        slices["joint_pos"] = slice(idx, idx + num_actuators)
        idx += num_actuators

        slices["joint_vel"] = slice(idx, idx + num_actuators)
        idx += num_actuators

        slices["action"] = slice(idx, idx + num_actuators)
        idx += num_actuators

        slices["velocity_cmd"] = slice(idx, idx + cls.VELOCITY_CMD_DIM)
        idx += cls.VELOCITY_CMD_DIM

        slices["padding"] = slice(idx, idx + cls.PADDING_DIM)

        return slices

    @classmethod
    def build_obs(
        cls,
        gravity: jax.Array,
        angvel: jax.Array,
        linvel: jax.Array,
        joint_pos: jax.Array,
        joint_vel: jax.Array,
        action: jax.Array,
        velocity_cmd: jax.Array,
    ) -> jax.Array:
        """Build observation vector from components.

        This is the canonical way to construct observations, ensuring
        the layout matches get_slices() and total_size().

        Args:
            gravity: Gravity vector in local frame (3,)
            angvel: Angular velocity, heading-local (3,)
            linvel: Linear velocity, heading-local (3,)
            joint_pos: Joint positions (num_actuators,)
            joint_vel: Joint velocities (num_actuators,)
            action: Previous action (num_actuators,)
            velocity_cmd: Velocity command scalar

        Returns:
            Observation vector (total_size,) as float32

        Raises:
            ValueError: If input dimensions don't match expected layout
        """
        num_actuators = cls._get_num_actuators()

        # Validate fixed dimensions
        if gravity.shape != (cls.GRAVITY_DIM,):
            raise ValueError(
                f"gravity shape {gravity.shape} != expected ({cls.GRAVITY_DIM},)"
            )
        if angvel.shape != (cls.ANGVEL_DIM,):
            raise ValueError(
                f"angvel shape {angvel.shape} != expected ({cls.ANGVEL_DIM},)"
            )
        if linvel.shape != (cls.LINVEL_DIM,):
            raise ValueError(
                f"linvel shape {linvel.shape} != expected ({cls.LINVEL_DIM},)"
            )

        # Validate dynamic dimensions
        if joint_pos.shape != (num_actuators,):
            raise ValueError(
                f"joint_pos shape {joint_pos.shape} != expected ({num_actuators},)"
            )
        if joint_vel.shape != (num_actuators,):
            raise ValueError(
                f"joint_vel shape {joint_vel.shape} != expected ({num_actuators},)"
            )
        if action.shape != (num_actuators,):
            raise ValueError(
                f"action shape {action.shape} != expected ({num_actuators},)"
            )

        obs = jp.concatenate(
            [
                gravity,
                angvel,
                linvel,
                joint_pos,
                joint_vel,
                action,
                jp.atleast_1d(velocity_cmd),
                jp.zeros(cls.PADDING_DIM),
            ]
        )
        return obs.astype(jp.float32)


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

    Extends mujoco_playground's State to add:
    - pipeline_state: Required by Brax's training wrapper

    The packed metrics vector is stored in metrics[METRICS_VEC_KEY] following
    the same pattern as info[WR_INFO_KEY]. This avoids adding new fields
    to the state dataclass.
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
        training_cfg = load_training_config("configs/ppo_walking.yaml")
        runtime_cfg = training_cfg.to_runtime_config()
        env = WildRobotEnv(config=runtime_cfg)

        # JIT-compiled reset and step
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)

        # Vectorized environments
        batched_reset = jax.vmap(env.reset)
        batched_step = jax.vmap(env.step)
    """

    def __init__(
        self,
        config: TrainingConfig,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize WildRobotEnv.

        Args:
            config: TrainingConfig (frozen, JIT-compatible).
                    Call config.freeze() before passing.
            config_overrides: Optional overrides for config values.
        """
        # Create ConfigDict for parent class (MjxEnv expects ConfigDict)
        parent_config = config_dict.ConfigDict()
        parent_config.ctrl_dt = config.env.ctrl_dt
        parent_config.sim_dt = config.env.sim_dt
        super().__init__(parent_config, config_overrides)

        # Store frozen runtime config directly - no extraction needed!
        # All values accessed via self._config.env.*, self._config.ppo.*, etc.
        self._config = config

        # Model path (required - must be in config)
        if not config.env.model_path:
            raise ValueError(
                "model_path is required in config.env. "
                "Add 'model_path: assets/scene_flat_terrain.xml' to your config."
            )
        self._model_path = Path(config.env.model_path)

        # Load model and setup robot infrastructure
        self._load_model()

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

        # Cache hip/knee actuator indices for gait shaping rewards
        self._left_hip_pitch_idx, self._right_hip_pitch_idx = (
            self._robot_config.get_hip_pitch_indices()
        )
        self._left_knee_pitch_idx, self._right_knee_pitch_idx = (
            self._robot_config.get_knee_pitch_indices()
        )

        # Foot contact geom IDs (from robot_config.yaml)
        # These are used for computing foot contact features for AMP discriminator
        # v0.5.0: Use explicit keys instead of array indexing (no order assumptions)
        # Get geom IDs using explicit config keys - order: [left_toe, left_heel, right_toe, right_heel]
        left_toe_name, left_heel_name, right_toe_name, right_heel_name = (
            self._robot_config.get_foot_geom_names()
        )

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

        # Contact detection parameters (from env config)
        # contact_threshold: Minimum force (N) to consider contact (not currently used)
        # contact_scale: Divisor for tanh normalization (10.0 means 10N → tanh(1) ≈ 0.76)
        self._contact_threshold = self._config.env.contact_threshold_force
        self._contact_scale = self._config.env.contact_scale

        # v0.10.1: Foot body IDs for slip/clearance computation
        # Use actual foot bodies (not *_mimic sites which are for GMR retargeting)
        # Read from robot_config instead of hard-coding
        self._left_foot_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self._robot_config.left_foot_body
        )
        self._right_foot_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self._robot_config.right_foot_body
        )

        # v0.10.1: Actuator force limits for torque penalty normalization
        # MuJoCo stores force limits in actuator_forcerange (nu x 2)
        self._actuator_force_limit = jp.array(
            [self._mj_model.actuator_forcerange[i, 1] for i in range(self._mj_model.nu)]
        )  # Take upper limit (positive)

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

        # v0.10.1: Simplified initial reward (robot at rest, no slip/clearance yet)
        # Forward reward: robot is stationary (vel=0), so miss target by velocity_cmd
        forward_reward = jp.exp(
            -velocity_cmd * self._config.reward_weights.forward_velocity_scale
        )

        # Healthy reward: robot is upright at reset
        healthy_reward = jp.where(
            (height > self._config.env.min_height)
            & (height < self._config.env.max_height),
            1.0,
            0.0,
        )

        # Action rate: no previous action, so no penalty
        action_rate = jp.zeros(())

        # v0.10.1: Initialize all metric keys to match step() for JAX pytree compatibility
        # At reset, robot is stationary so most values are zero
        pitch_roll = pitch_roll_from_quat_jax(self.get_root_quat(data))
        pitch, roll = pitch_roll[0], pitch_roll[1]
        left_force, right_force = self.get_raw_foot_contacts(data)

        # Compute total reward with weights (from config)
        weights = self._config.reward_weights
        total_reward = (
            weights.tracking_lin_vel * forward_reward
            + weights.base_height * healthy_reward
            + weights.action_rate * action_rate
        )

        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": jp.zeros(()),  # Robot starts stationary
            "episode_step_count": jp.zeros(()),  # v0.10.4: Starts at 0 for new episode
            "reward/total": total_reward,
            # Primary rewards
            "reward/forward": forward_reward,
            "reward/lateral": jp.zeros(()),
            "reward/healthy": healthy_reward,
            "reward/orientation": jp.zeros(()),
            "reward/angvel": jp.zeros(()),
            # Effort rewards
            "reward/torque": jp.zeros(()),
            "reward/saturation": jp.zeros(()),
            # Smoothness rewards
            "reward/action_rate": action_rate,
            "reward/joint_vel": jp.zeros(()),
            # Foot stability rewards
            "reward/slip": jp.zeros(()),
            "reward/clearance": jp.zeros(()),
            "reward/gait_periodicity": jp.zeros(()),
            "reward/hip_swing": jp.zeros(()),
            "reward/knee_swing": jp.zeros(()),
            "reward/flight_phase": jp.zeros(()),
            # v0.10.4: Standing penalty
            "reward/standing": jp.zeros(()),
            # Debug metrics
            "debug/pitch": pitch,
            "debug/roll": roll,
            "debug/forward_vel": jp.zeros(()),
            "debug/lateral_vel": jp.zeros(()),
            "debug/left_force": left_force,
            "debug/right_force": right_force,
            # v0.10.2: Termination diagnostics (initialized to zero at reset)
            "term/height_low": jp.zeros(()),
            "term/height_high": jp.zeros(()),
            "term/pitch": jp.zeros(()),
            "term/roll": jp.zeros(()),
            "term/truncated": jp.zeros(()),
            "term/pitch_val": pitch,
            "term/roll_val": roll,
            "term/height_val": height,
            # v0.10.3: Tracking metrics for walking exit criteria
            "tracking/vel_error": jp.zeros(()),  # Starts at zero (stationary)
            "tracking/max_torque": jp.zeros(()),  # No torque at reset
        }

        # Info - use typed WildRobotInfo under "wr" namespace
        # This separates algorithm-critical fields from wrapper-injected fields
        left_foot_pos, right_foot_pos = self.get_foot_positions(data)
        wr_info = WildRobotInfo(
            step_count=jp.zeros(()),
            prev_action=jp.zeros(self.action_size),
            truncated=jp.zeros(()),  # No truncation at reset
            velocity_cmd=velocity_cmd,  # Target velocity for this episode
            prev_root_pos=self.get_floating_base_qpos(data.qpos)[0:3],  # (3,)
            prev_root_quat=self.get_floating_base_qpos(data.qpos)[3:7],  # (4,) wxyz
            prev_left_foot_pos=left_foot_pos,
            prev_right_foot_pos=right_foot_pos,
            foot_contacts=self.get_norm_foot_contacts(data),  # For AMP discriminator
            root_height=height,  # Actual root height for AMP features
        )

        # Merge: base_info (wrapper fields) + wr namespace (env fields)
        if base_info is not None:
            info = dict(base_info)
            info[WR_INFO_KEY] = wr_info
        else:
            info = {WR_INFO_KEY: wr_info}

        # Build packed metrics vector and store in metrics dict
        # This follows the same pattern as info[WR_INFO_KEY]
        metrics_vec = build_metrics_vec(metrics)
        metrics[METRICS_VEC_KEY] = metrics_vec

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
            key1,
            shape=(),
            minval=self._config.env.min_velocity,
            maxval=self._config.env.max_velocity,
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
        # Read from typed WildRobotInfo namespace (fail loudly if missing)
        wr = state.info[WR_INFO_KEY]
        velocity_cmd = wr.velocity_cmd
        step_count = wr.step_count
        prev_action = wr.prev_action

        # Action filtering (low-pass)
        if self._config.env.use_action_filter:
            alpha = self._config.env.action_filter_alpha
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
        # Use previous root pose from WildRobotInfo for FD velocity computation
        prev_root_pos = wr.prev_root_pos
        prev_root_quat = wr.prev_root_quat
        obs = self._get_obs(
            data, filtered_action, velocity_cmd, prev_root_pos, prev_root_quat
        )

        # Get previous foot positions for slip computation from WildRobotInfo
        prev_left_foot_pos = wr.prev_left_foot_pos
        prev_right_foot_pos = wr.prev_right_foot_pos

        # Compute reward
        reward, reward_components = self._get_reward(
            data,
            filtered_action,
            prev_action,
            velocity_cmd,
            prev_left_foot_pos,
            prev_right_foot_pos,
        )

        # Check termination (get done, terminated, truncated, and diagnostics)
        done, terminated, truncated, term_info = self._get_termination(data, step_count + 1)

        # Update metrics (before potential reset)
        height = self.get_floating_base_qpos(data.qpos)[2]
        forward_vel = self.get_local_linvel(data)[0]

        # v0.10.4: Store step count in metrics for episode length calculation
        # This gets preserved through auto-reset, unlike wr_info.step_count which resets
        episode_step_count = step_count + 1

        metrics = {
            "velocity_command": velocity_cmd,
            "height": height,
            "forward_velocity": forward_vel,
            "episode_step_count": episode_step_count,  # v0.10.4: For ep_len logging
            "reward/total": reward,
            **reward_components,
            # v0.10.2: Termination diagnostics
            **term_info,
        }
        # Add placeholder for metrics_vec (will be rebuilt after cond)
        # This ensures pytree structure matches preserved_metrics
        metrics[METRICS_VEC_KEY] = build_metrics_vec(metrics)

        # Compute current values for WildRobotInfo update
        curr_root_pos = self.get_floating_base_qpos(data.qpos)[0:3]
        curr_root_quat = self.get_floating_base_qpos(data.qpos)[3:7]
        curr_left_foot_pos, curr_right_foot_pos = self.get_foot_positions(data)

        # Create new WildRobotInfo with updated values
        new_wr_info = WildRobotInfo(
            step_count=step_count + 1,
            prev_action=filtered_action,
            truncated=truncated,  # 1.0 if reached max steps (success), 0.0 otherwise
            velocity_cmd=velocity_cmd,  # Preserved through episode
            prev_root_pos=curr_root_pos,
            prev_root_quat=curr_root_quat,
            prev_left_foot_pos=curr_left_foot_pos,
            prev_right_foot_pos=curr_right_foot_pos,
            foot_contacts=self.get_norm_foot_contacts(data),  # For AMP discriminator
            root_height=height,  # Actual root height for AMP features
        )

        # Preserve wrapper fields, update wr namespace
        # The Brax training wrapper adds fields like 'truncation', 'episode_done',
        # 'episode_metrics', 'first_obs', 'steps', 'first_state'. We preserve these.
        info = dict(state.info)  # Copy existing wrapper fields
        info[WR_INFO_KEY] = new_wr_info  # Update env-owned namespace

        # =================================================================
        # Auto-reset on termination
        # When done=True, reset physics state to initial keyframe but keep
        # done=True so the algorithm knows an episode ended.
        # v0.10.2: Preserve termination metrics for diagnostics
        # =================================================================
        # Pass state.info as base_info to preserve wrapper fields during auto-reset
        reset_state = self._make_initial_state(
            self._init_qpos, velocity_cmd, base_info=state.info
        )

        # v0.10.2: Preserve termination diagnostics in metrics even after reset
        # The reset_state.metrics has all zeros for term/*, but we want to keep
        # the actual termination reason for logging
        # v0.10.4: Also preserve episode_step_count for accurate ep_len
        # v0.10.4: Also preserve tracking metrics for accurate vel_err/torque logging
        preserved_metrics = {
            **reset_state.metrics,
            "episode_step_count": metrics["episode_step_count"],  # Terminal step count for ep_len
            # Tracking metrics (for topline monitoring)
            "tracking/vel_error": metrics["tracking/vel_error"],
            "tracking/max_torque": metrics["tracking/max_torque"],
            # Core metrics (for logging)
            "forward_velocity": metrics["forward_velocity"],
            "velocity_command": metrics["velocity_command"],
            # Termination diagnostics
            "term/height_low": metrics["term/height_low"],
            "term/height_high": metrics["term/height_high"],
            "term/pitch": metrics["term/pitch"],
            "term/roll": metrics["term/roll"],
            "term/truncated": metrics["term/truncated"],
            "term/pitch_val": metrics["term/pitch_val"],
            "term/roll_val": metrics["term/roll_val"],
            "term/height_val": metrics["term/height_val"],
        }

        # v0.10.2: Also preserve truncated flag in wr info for success rate calculation
        # We need to preserve the original wr_info.truncated (1.0 if success) through auto-reset
        #
        # v0.10.4 FIX: step_count must RESET to 0 for new episode, not accumulate!
        # The terminal step count is passed separately for logging.
        # Key insight: preserved_wr_info is what the NEXT episode sees, so step_count=0.
        reset_wr_info = reset_state.info[WR_INFO_KEY]
        preserved_wr_info = WildRobotInfo(
            step_count=reset_wr_info.step_count,  # Reset to 0 for new episode (NOT new_wr_info.step_count!)
            prev_action=reset_wr_info.prev_action,
            truncated=new_wr_info.truncated,  # Preserve original truncated flag for success rate
            velocity_cmd=reset_wr_info.velocity_cmd,  # New velocity for new episode
            prev_root_pos=reset_wr_info.prev_root_pos,
            prev_root_quat=reset_wr_info.prev_root_quat,
            prev_left_foot_pos=reset_wr_info.prev_left_foot_pos,
            prev_right_foot_pos=reset_wr_info.prev_right_foot_pos,
            foot_contacts=reset_wr_info.foot_contacts,
            root_height=reset_wr_info.root_height,
        )
        preserved_info = dict(reset_state.info)  # Copy wrapper fields
        preserved_info[WR_INFO_KEY] = preserved_wr_info  # Update wr namespace

        final_data, final_obs, final_metrics, final_info = jax.lax.cond(
            done > 0.5,
            lambda: (
                reset_state.data,
                reset_state.obs,
                preserved_metrics,  # Use preserved metrics with term/* intact
                preserved_info,     # Use preserved info with truncated intact
            ),
            lambda: (data, obs, metrics, info),
        )

        # Build packed metrics vector and store in metrics dict
        final_metrics_vec = build_metrics_vec(final_metrics)
        final_metrics[METRICS_VEC_KEY] = final_metrics_vec

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
            curr_root_pos = self.get_floating_base_qpos(data.qpos)[0:3]
            curr_root_quat = self.get_root_quat(data)
            linvel = heading_local_linvel_fd_from_pos_jax(
                curr_root_pos, prev_root_pos, curr_root_quat, self.dt
            )
            angvel = heading_local_angvel_fd_from_quat_jax(
                curr_root_quat, prev_root_quat, self.dt
            )
        else:
            # qvel-based velocities (for reset when no previous state)
            # These will be zero at reset anyway
            linvel = self.get_heading_local_linvel(data)
            angvel = self.get_heading_local_angvel(data)

        # Joint positions and velocities (squeeze to handle vmap extra dims)
        joint_pos = self.get_actuator_joint_qpos(data.qpos).squeeze()
        joint_vel = self.get_actuator_joints_qvel(data.qvel).squeeze()

        # Build observation using shared layout
        return ObsLayout.build_obs(
            gravity=gravity,
            angvel=angvel,
            linvel=linvel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=action,
            velocity_cmd=velocity_cmd,
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        prev_action: jax.Array,
        velocity_cmd: jax.Array,
        prev_left_foot_pos: Optional[jax.Array] = None,
        prev_right_foot_pos: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute reward following gold standard for bipedal locomotion.

        v0.10.1: Implements industry best practices:
        - Primary: velocity tracking, alive bonus, orientation, slip penalty
        - Secondary: torque penalty, saturation, action smoothness, clearance

        Returns:
            (total_reward, reward_components_dict)
        """

        # =====================================================================
        # PRIMARY REWARDS (Tier 1)
        # =====================================================================

        # 1. Forward velocity tracking (exp shaping)
        forward_vel = self.get_local_linvel(data)[0]
        vel_error = jp.abs(forward_vel - velocity_cmd)
        forward_reward = jp.exp(
            -vel_error * self._config.reward_weights.forward_velocity_scale
        )

        # 2. Lateral velocity penalty (penalize sideways drift)
        lateral_vel = self.get_local_linvel(data)[1]
        lateral_penalty = jp.square(lateral_vel)

        # 3. Alive/healthy reward (height-based)
        height = self.get_floating_base_qpos(data.qpos)[2]
        healthy = jp.where(
            (height > self._config.env.min_height)
            & (height < self._config.env.max_height),
            1.0,
            0.0,
        )

        # 4. Orientation penalty (pitch² + roll²)
        pitch_roll = pitch_roll_from_quat_jax(self.get_root_quat(data))
        pitch, roll = pitch_roll[0], pitch_roll[1]
        orientation_penalty = jp.square(pitch) + jp.square(roll)

        # 5. Angular velocity penalty (reduce body rotation)
        base_angvel = self.get_floating_base_qvel(data.qvel)[3:6]  # wx, wy, wz
        angvel_penalty = jp.sum(jp.square(base_angvel))

        # =====================================================================
        # EFFORT REWARDS (Tier 1 - critical for sim2real)
        # =====================================================================

        # 6. Torque penalty - penalize normalized torque usage
        torques = self.get_actuator_torques(data)
        normalized_torques = torques / (self._actuator_force_limit + 1e-6)
        torque_penalty = jp.sum(jp.square(normalized_torques))

        # 7. Saturation penalty - penalize actuators near limits
        saturation = jp.abs(normalized_torques) > 0.95
        saturation_penalty = jp.sum(saturation.astype(jp.float32))

        # =====================================================================
        # SMOOTHNESS REWARDS (Tier 2)
        # =====================================================================

        # 8. Action rate penalty (smoothness)
        action_diff = action - prev_action
        action_rate_penalty = jp.sum(jp.square(action_diff))

        # 9. Joint velocity penalty (efficiency)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        joint_vel_penalty = jp.sum(jp.square(joint_vel))

        # =====================================================================
        # FOOT STABILITY REWARDS (Tier 2)
        # =====================================================================

        # Get foot contact forces for slip/clearance gating
        left_force, right_force = self.get_raw_foot_contacts(data)
        left_pos, right_pos = self.get_foot_positions(data)

        # Robot weight for contact threshold (approximate: mass * gravity)
        # ~2kg robot * 9.81 ≈ 20N total, so ~5N per foot is "loaded"
        # v0.10.1: Read from config instead of hard-coding
        contact_threshold = self._config.env.contact_threshold_force

        # 10. Slip penalty (when foot is loaded, penalize tangential velocity)
        # Use finite-diff velocity if previous positions available
        if prev_left_foot_pos is not None and prev_right_foot_pos is not None:
            left_vel, right_vel = self.get_foot_velocities(
                data, prev_left_foot_pos, prev_right_foot_pos
            )
            # Tangential (xy) velocity magnitude
            left_slip = jp.sqrt(jp.square(left_vel[0]) + jp.square(left_vel[1]))
            right_slip = jp.sqrt(jp.square(right_vel[0]) + jp.square(right_vel[1]))

            # Gate by contact force (only penalize when loaded)
            left_loaded = left_force > contact_threshold
            right_loaded = right_force > contact_threshold
            slip_penalty = jp.where(left_loaded, jp.square(left_slip), 0.0) + jp.where(
                right_loaded, jp.square(right_slip), 0.0
            )
        else:
            slip_penalty = jp.zeros(())
            left_slip = jp.zeros(())
            right_slip = jp.zeros(())

        # 11. Swing clearance reward (when foot is unloaded, reward height)
        # Reward foot height above ground during swing phase
        min_clearance = 0.02  # 2cm minimum clearance during swing
        left_clearance = jp.clip(left_pos[2] - min_clearance, 0.0, 0.05)  # cap at 5cm
        right_clearance = jp.clip(right_pos[2] - min_clearance, 0.0, 0.05)

        # Gate by NOT loaded (only reward clearance during swing)
        left_swing = left_force < contact_threshold
        right_swing = right_force < contact_threshold
        clearance_reward = (
            jp.where(left_swing, left_clearance, 0.0)
            + jp.where(right_swing, right_clearance, 0.0)
        ) * 10.0  # Scale up small clearance values

        # 12. Gait periodicity (encourage alternating support)
        left_loaded = left_force > contact_threshold
        right_loaded = right_force > contact_threshold
        gait_periodicity = jp.where(left_loaded ^ right_loaded, 1.0, 0.0)

        # 13. Hip/knee swing (encourage leg articulation during swing)
        joint_pos = self.get_actuator_joint_qpos(data.qpos)
        left_hip_pitch = joint_pos[self._left_hip_pitch_idx]
        right_hip_pitch = joint_pos[self._right_hip_pitch_idx]
        left_knee_pitch = joint_pos[self._left_knee_pitch_idx]
        right_knee_pitch = joint_pos[self._right_knee_pitch_idx]

        hip_min = jp.maximum(weights.hip_swing_min, 1e-6)
        knee_min = jp.maximum(weights.knee_swing_min, 1e-6)

        def _swing_reward(angle: jax.Array, min_amp: jax.Array) -> jax.Array:
            amp = jp.abs(angle)
            return jp.clip((amp - min_amp) / min_amp, 0.0, 1.0)

        hip_swing = 0.5 * (
            _swing_reward(left_hip_pitch, hip_min) * left_swing
            + _swing_reward(right_hip_pitch, hip_min) * right_swing
        )
        knee_swing = 0.5 * (
            _swing_reward(left_knee_pitch, knee_min) * left_swing
            + _swing_reward(right_knee_pitch, knee_min) * right_swing
        )

        # 14. Flight phase penalty (discourage hopping)
        flight_phase = jp.where((~left_loaded) & (~right_loaded), 1.0, 0.0)

        # =====================================================================
        # COMBINE REWARDS
        # =====================================================================

        # Get weights from frozen config (type-safe access)
        weights = self._config.reward_weights

        # v0.10.4: Smooth, command-gated standing penalty
        # Only apply when |velocity_cmd| > velocity_cmd_min (don't penalize if asked to stop)
        # Only apply when healthy (prevents noise during falling/recovery)
        # Penalty is smooth: relu(threshold - |vel|) / threshold, ranges from 0 to 1
        standing_threshold = weights.velocity_standing_threshold
        standing_penalty_weight = weights.velocity_standing_penalty
        velocity_cmd_min = weights.velocity_cmd_min

        # Smooth penalty: how far below threshold (0 if at/above threshold)
        velocity_deficit = jp.maximum(standing_threshold - jp.abs(forward_vel), 0.0)
        standing_penalty_raw = velocity_deficit / (standing_threshold + 1e-6)

        # Gate by command magnitude: only penalize if asked to move (supports future backward cmds)
        cmd_gate = jp.where(jp.abs(velocity_cmd) > velocity_cmd_min, 1.0, 0.0)

        # Gate by healthy: don't penalize during falling/recovery (prevents twitch behaviors)
        healthy_gate = healthy

        standing_penalty = standing_penalty_raw * cmd_gate * healthy_gate

        total = (
            # Primary (Tier 1)
            weights.tracking_lin_vel * forward_reward
            + weights.lateral_velocity * lateral_penalty
            + weights.base_height * healthy
            + weights.orientation * orientation_penalty
            + weights.angular_velocity * angvel_penalty
            # v0.10.4: Smooth standing penalty (negative reward for standing still)
            - standing_penalty_weight * standing_penalty
            # Effort (Tier 1)
            + weights.torque * torque_penalty
            + weights.saturation * saturation_penalty
            # Smoothness (Tier 2)
            + weights.action_rate * action_rate_penalty
            + weights.joint_velocity * joint_vel_penalty
            # Foot stability (Tier 2)
            + weights.slip * slip_penalty
            + weights.clearance * clearance_reward
            + weights.gait_periodicity * gait_periodicity
            + weights.hip_swing * hip_swing
            + weights.knee_swing * knee_swing
            + weights.flight_phase_penalty * flight_phase
        )

        # =====================================================================
        # REWARD COMPONENTS FOR LOGGING
        # =====================================================================

        # v0.10.3: Compute tracking metrics for walking exit criteria
        # Max normalized torque (peak torque as fraction of limit, across all actuators)
        max_torque_ratio = jp.max(jp.abs(normalized_torques))

        components = {
            # Primary
            "reward/forward": forward_reward,
            "reward/lateral": lateral_penalty,
            "reward/healthy": healthy,
            "reward/orientation": orientation_penalty,
            "reward/angvel": angvel_penalty,
            # v0.10.4: Standing penalty
            "reward/standing": standing_penalty,
            # Effort
            "reward/torque": torque_penalty,
            "reward/saturation": saturation_penalty,
            # Smoothness
            "reward/action_rate": action_rate_penalty,
            "reward/joint_vel": joint_vel_penalty,
            # Foot stability
            "reward/slip": slip_penalty,
            "reward/clearance": clearance_reward,
            "reward/gait_periodicity": gait_periodicity,
            "reward/hip_swing": hip_swing,
            "reward/knee_swing": knee_swing,
            "reward/flight_phase": flight_phase,
            # Debug metrics
            "debug/pitch": pitch,
            "debug/roll": roll,
            "debug/forward_vel": forward_vel,
            "debug/lateral_vel": lateral_vel,
            "debug/left_force": left_force,
            "debug/right_force": right_force,
            # v0.10.3: Tracking metrics for walking exit criteria
            "tracking/vel_error": vel_error,  # |forward_vel - velocity_cmd|
            "tracking/max_torque": max_torque_ratio,  # max(|torque|/limit)
        }

        return total, components

    def _get_termination(
        self, data: mjx.Data, step_count: int
    ) -> tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]:
        """Check termination conditions.

        v0.10.2: Added termination diagnostics for debugging.

        Returns:
            (done, terminated, truncated, termination_info) where:
            - done: True if episode ended (terminated OR truncated)
            - terminated: True if ended due to failure (falling or bad orientation)
            - truncated: True if ended due to time limit (success)
            - termination_info: Dict with per-condition flags for diagnostics
        """
        height = self.get_floating_base_qpos(data.qpos)[2]

        # Height termination (failure - robot fell)
        height_too_low = height < self._config.env.min_height
        height_too_high = height > self._config.env.max_height
        height_fail = height_too_low | height_too_high

        # v0.10.1: Orientation termination (extreme pitch/roll)
        pitch_roll = pitch_roll_from_quat_jax(self.get_root_quat(data))
        pitch, roll = pitch_roll[0], pitch_roll[1]
        pitch_fail = jp.abs(pitch) > self._config.env.max_pitch
        roll_fail = jp.abs(roll) > self._config.env.max_roll
        orientation_fail = pitch_fail | roll_fail

        # Combined failure termination
        terminated = height_fail | orientation_fail

        # Episode length truncation (success - reached max steps without failing)
        truncated = (step_count >= self._config.env.max_episode_steps) & ~terminated

        # Combined done flag
        done = terminated | truncated

        # v0.10.2: Termination diagnostics
        termination_info = {
            "term/height_low": jp.where(height_too_low, 1.0, 0.0),
            "term/height_high": jp.where(height_too_high, 1.0, 0.0),
            "term/pitch": jp.where(pitch_fail, 1.0, 0.0),
            "term/roll": jp.where(roll_fail, 1.0, 0.0),
            "term/truncated": jp.where(truncated, 1.0, 0.0),
            # Raw values for debugging
            "term/pitch_val": pitch,
            "term/roll_val": roll,
            "term/height_val": height,
        }

        return (
            jp.where(done, 1.0, 0.0),
            jp.where(terminated, 1.0, 0.0),
            jp.where(truncated, 1.0, 0.0),
            termination_info,
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
        quat = self.get_root_quat(data)  # [w, x, y, z]
        return heading_sin_cos_from_quat_jax(quat)

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

        return heading_local_from_world_jax(linvel_world, sin_yaw, cos_yaw)

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

        return heading_local_from_world_jax(angvel_world, sin_yaw, cos_yaw)

    # =========================================================================
    # Reward Utility Methods
    # =========================================================================

    def get_actuator_torques(self, data: mjx.Data) -> jax.Array:
        """Get actuator forces/torques.

        Returns:
            actuator_force: (nu,) array of actuator forces in Nm
        """
        return data.qfrc_actuator[self.actuator_qvel_addr]

    # =========================================================================
    # Foot Utilities
    # =========================================================================

    def _get_raw_foot_contact_by_geom(self, data: mjx.Data, geom_id: int) -> jax.Array:
        """Get raw contact normal force for a single foot geom.

        Uses efc_force[efc_address] directly - the solver-native representation.
        This is the industry standard approach (dm_control, OpenAI Gym, Meta
        locomotion code, MJX/Brax pipelines).

        Note: This does NOT match mj_contactForce() output, which reconstructs
        forces in contact frame. For learning workloads, solver-native efc_force
        is preferred because:
        - It's deterministic and cheap
        - Locomotion rewards need load heuristics, not exact wrenches
        - It's what MJX exposes (no contactForce helper by design)

        Args:
            data: MJX simulation data
            geom_id: ID of the geom to query

        Returns:
            Sum of normal constraint forces on the geom (solver units)
        """
        geom1_match = data.contact.geom1 == geom_id
        geom2_match = data.contact.geom2 == geom_id
        is_our_contact = geom1_match | geom2_match

        # Get normal forces from constraint solver
        # efc_address[i] is the index where contact i's forces start in efc_force
        # The first element at each address is the normal force component
        normal_forces = data.efc_force[data.contact.efc_address]

        # Sum forces for contacts involving this geom
        our_forces = jp.where(is_our_contact, jp.abs(normal_forces), 0.0)
        return jp.sum(our_forces)

    def get_foot_positions(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Get foot body positions in world frame.

        Uses actual foot bodies (not *_mimic sites which are for GMR retargeting).

        Returns:
            (left_foot_pos, right_foot_pos): Each (3,) in world frame
        """
        left_pos = data.xpos[self._left_foot_body_id]
        right_pos = data.xpos[self._right_foot_body_id]
        return left_pos, right_pos

    def get_foot_velocities(
        self, data: mjx.Data, prev_left_pos: jax.Array, prev_right_pos: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get foot velocities via finite difference.

        Args:
            data: Current MJX data
            prev_left_pos: Previous left foot position (3,)
            prev_right_pos: Previous right foot position (3,)

        Returns:
            (left_vel, right_vel): Each (3,) in world frame
        """
        left_pos, right_pos = self.get_foot_positions(data)
        left_vel = (left_pos - prev_left_pos) / self.dt
        right_vel = (right_pos - prev_right_pos) / self.dt
        return left_vel, right_vel

    def get_norm_foot_contacts(self, data: mjx.Data) -> jax.Array:
        """Get normalized foot contact features for AMP discriminator.

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
        # Get contact forces for each foot geom
        left_toe_force = self._get_raw_foot_contact_by_geom(
            data, self._left_toe_geom_id
        )
        left_heel_force = self._get_raw_foot_contact_by_geom(
            data, self._left_heel_geom_id
        )
        right_toe_force = self._get_raw_foot_contact_by_geom(
            data, self._right_toe_geom_id
        )
        right_heel_force = self._get_raw_foot_contact_by_geom(
            data, self._right_heel_geom_id
        )

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

    def get_raw_foot_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Get raw foot contact forces (not normalized).

        Returns normal forces for left and right feet (sum of toe + heel).
        Used for reward computation where threshold comparison is needed.

        Returns:
            (left_force, right_force): Scalar normal forces in Newtons
        """
        left_toe_force = self._get_raw_foot_contact_by_geom(
            data, self._left_toe_geom_id
        )
        left_heel_force = self._get_raw_foot_contact_by_geom(
            data, self._left_heel_geom_id
        )
        right_toe_force = self._get_raw_foot_contact_by_geom(
            data, self._right_toe_geom_id
        )
        right_heel_force = self._get_raw_foot_contact_by_geom(
            data, self._right_heel_geom_id
        )

        left_force = left_toe_force + left_heel_force
        right_force = right_toe_force + right_heel_force

        return left_force, right_force

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
        """Return observation size using shared ObsLayout.

        v0.7.0: Velocities are now in heading-local frame for AMP parity.
        The observation layout is unchanged.

        See ObsLayout class for component breakdown.
        """
        return ObsLayout.total_size()
