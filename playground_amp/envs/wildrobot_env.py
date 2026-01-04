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
from policy_contract.jax.obs import build_observation_from_components
from policy_contract.jax.action import postprocess_action
from policy_contract.jax.calib import action_to_ctrl
from policy_contract.jax.state import PolicyState
from policy_contract.spec import (
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicySpec,
    RobotSpec,
)
from playground_amp.cal.cal import ControlAbstractionLayer
from playground_amp.cal.specs import Pose3D
from playground_amp.cal.types import CoordinateFrame

from playground_amp.configs.training_config import get_robot_config, TrainingConfig
from playground_amp.envs.env_info import WildRobotInfo, WR_INFO_KEY
from playground_amp.envs.disturbance import (
    DisturbanceSchedule,
    apply_push,
    sample_push_schedule,
)
from playground_amp.training.experiment_tracking import get_initial_env_metrics_jax
from playground_amp.training.metrics_registry import (
    build_metrics_vec,
    METRIC_NAMES,
    METRICS_VEC_KEY,
    NUM_METRICS,
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
        [...]   foot_switch  - Foot switches (4 = toe/heel L/R)
        [...]   action       - Previous action (n = num_actuators)
        [...]   velocity_cmd - Velocity command (1)
        [...]   padding      - Padding for alignment (1)
    """

    # Fixed dimensions (independent of robot)
    GRAVITY_DIM: int = 3
    ANGVEL_DIM: int = 3
    LINVEL_DIM: int = 3
    FOOT_SWITCH_DIM: int = 4
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
            + cls.FOOT_SWITCH_DIM
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

        slices["foot_switches"] = slice(idx, idx + cls.FOOT_SWITCH_DIM)
        idx += cls.FOOT_SWITCH_DIM

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
        foot_switches: jax.Array,
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
            foot_switches: Foot switch states (4,)
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
        if foot_switches.shape != (cls.FOOT_SWITCH_DIM,):
            raise ValueError(
                f"foot_switches shape {foot_switches.shape} != expected ({cls.FOOT_SWITCH_DIM},)"
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
                foot_switches,
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
    rng: jp.ndarray = None


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
        print(f"  Actuators: {self._mj_model.nu}")
        print(f"  Floating base: {self._robot_config.floating_base_body}")
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
        self._policy_spec = self._build_policy_spec()

        # =====================================================================
        # Initialize Control Abstraction Layer (CAL) - v0.11.0
        # =====================================================================
        # CAL provides the canonical action→ctrl transformation with:
        # - Symmetry correction (mirror_sign) for left/right joints
        # - Normalization from [-1, 1] to physical joint ranges
        # - Single source of truth for joint specifications
        # - Extended state for foot, root, and knee IMU access
        # - Contact scale is read from robot_config.feet.contact_scale
        self._cal = ControlAbstractionLayer(self._mj_model, self._robot_config)
        print(
            f"  CAL initialized with {self._cal.num_actuators} actuators (extended state included)"
        )

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
            # Fall back to qpos0 (model default)
            self._init_qpos = jp.array(self._mj_model.qpos0)
            print(f"  Using qpos0 from model")
            init_height = float(self._init_qpos[2])
            print(f"  Initial height from qpos0: {init_height:.4f}m")

        # Extract joint-only qpos for control default from CAL (single source of truth)
        self._default_joint_qpos = self._cal.get_ctrl_for_default_pose()

        # Disturbance configuration
        self._push_body_id = -1
        if self._config.env.push_enabled:
            self._push_body_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self._config.env.push_body
            )
            if self._push_body_id < 0:
                raise ValueError(
                    f"Push body '{self._config.env.push_body}' not found in model."
                )

    # =========================================================================
    # MjxEnv Interface
    # =========================================================================

    def _make_initial_state(
        self,
        qpos: jax.Array,
        velocity_cmd: jax.Array,
        base_info: dict | None = None,
        push_schedule: DisturbanceSchedule | None = None,
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

        if push_schedule is None:
            push_schedule = DisturbanceSchedule(
                start_step=jp.zeros(()),
                end_step=jp.zeros(()),
                force_xy=jp.zeros((2,)),
                rng=jp.zeros((2,)),
            )

        linvel_obs_mask, push_schedule = self._sample_linvel_obs_mask(push_schedule)

        # Build observation using default pose action (matches ctrl at reset)
        default_action = self._cal.ctrl_to_policy_action(self._default_joint_qpos)
        obs = self._get_obs(
            data, default_action, velocity_cmd, linvel_obs_mask=linvel_obs_mask
        )

        # Initial reward and done
        reward = jp.zeros(())
        done = jp.zeros(())

        # Cache root pose (used for height, pitch/roll, position, orientation)
        root_pose = self._cal.get_root_pose(data)
        height = root_pose.height

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
        roll, pitch, _ = root_pose.euler_angles()
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        foot_switches = self._cal.get_foot_switches(
            data, threshold=self._config.env.foot_switch_threshold
        )

        # Compute total reward with weights (from config)
        weights = self._config.reward_weights
        total_reward = (
            weights.tracking_lin_vel * forward_reward
            + weights.base_height * healthy_reward
            + weights.action_rate * action_rate
        )

        # This is the single source of truth for env metrics keys
        metrics = get_initial_env_metrics_jax(
            velocity_cmd=velocity_cmd,
            height=height,
            pitch=pitch,
            roll=roll,
            left_force=left_force,
            right_force=right_force,
            left_toe_switch=foot_switches[0],
            left_heel_switch=foot_switches[1],
            right_toe_switch=foot_switches[2],
            right_heel_switch=foot_switches[3],
            forward_reward=forward_reward,
            healthy_reward=healthy_reward,
            action_rate=action_rate,
            total_reward=total_reward,
        )

        # Info - use typed WildRobotInfo under "wr" namespace
        # This separates algorithm-critical fields from wrapper-injected fields
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False
        )
        wr_info = WildRobotInfo(
            step_count=jp.zeros(()),
            prev_action=default_action,
            truncated=jp.zeros(()),  # No truncation at reset
            velocity_cmd=velocity_cmd,  # Target velocity for this episode
            prev_root_pos=root_pose.position,  # (3,)
            prev_root_quat=root_pose.orientation,  # (4,) wxyz
            prev_left_foot_pos=left_foot_pos,
            prev_right_foot_pos=right_foot_pos,
            foot_contacts=self._cal.get_geom_based_foot_contacts(
                data, normalize=True
            ),  # For AMP
            root_height=root_pose.height,  # Actual root height for AMP features
            linvel_obs_mask=linvel_obs_mask,
            push_schedule=push_schedule,
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
            rng=push_schedule.rng,
        )

    def reset(self, rng: jax.Array) -> WildRobotEnvState:
        """Reset environment to initial state.

        Args:
            rng: JAX random key for randomization.

        Returns:
            Initial WildRobotEnvState.
        """
        rng, key1, key2, key3 = jax.random.split(rng, 4)

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

        schedule = sample_push_schedule(key3, self._config.env)

        return self._make_initial_state(
            qpos,
            velocity_cmd,
            push_schedule=schedule,
        )

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
        linvel_obs_mask = wr.linvel_obs_mask

        raw_action = action
        policy_state = PolicyState(prev_action=prev_action)
        filtered_action, policy_state = postprocess_action(
            spec=self._policy_spec,
            state=policy_state,
            action_raw=raw_action,
        )

        # =====================================================================
        # Apply action as control via CAL (v0.11.0)
        # =====================================================================
        # CAL applies symmetry correction (mirror_sign) for left/right joints
        # This ensures same action values produce symmetric motion on both sides
        ctrl = action_to_ctrl(spec=self._policy_spec, action=filtered_action)
        data = state.data.replace(ctrl=ctrl)

        # Apply external push disturbance (lateral force on body, if enabled)
        if self._config.env.push_enabled:
            data = apply_push(data, wr.push_schedule, step_count, self._push_body_id)

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
            data,
            filtered_action,
            velocity_cmd,
            prev_root_pos,
            prev_root_quat,
            linvel_obs_mask=linvel_obs_mask,
        )

        # Get previous foot positions for slip computation from WildRobotInfo
        prev_left_foot_pos = wr.prev_left_foot_pos
        prev_right_foot_pos = wr.prev_right_foot_pos

        # Compute reward
        reward, reward_components = self._get_reward(
            data,
            filtered_action,
            raw_action,
            prev_action,
            velocity_cmd,
            prev_left_foot_pos,
            prev_right_foot_pos,
        )

        # Check termination (get done, terminated, truncated, and diagnostics)
        done, terminated, truncated, term_info = self._get_termination(
            data, step_count + 1
        )

        # Cache root pose and velocity (used for metrics and WildRobotInfo update)
        curr_root_pose = self._cal.get_root_pose(data)
        forward_vel, _, _ = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        ).linear_xyz

        # v0.10.4: Store step count in metrics for episode length calculation
        # This gets preserved through auto-reset, unlike wr_info.step_count which resets
        episode_step_count = step_count + 1

        metrics = {
            "velocity_command": velocity_cmd,
            "height": curr_root_pose.height,
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

        # Compute current foot positions for WildRobotInfo update
        curr_left_foot_pos, curr_right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False
        )

        # Create new WildRobotInfo with updated values
        new_wr_info = WildRobotInfo(
            step_count=step_count + 1,
            prev_action=policy_state.prev_action,
            truncated=truncated,  # 1.0 if reached max steps (success), 0.0 otherwise
            velocity_cmd=velocity_cmd,  # Preserved through episode
            prev_root_pos=curr_root_pose.position,
            prev_root_quat=curr_root_pose.orientation,
            prev_left_foot_pos=curr_left_foot_pos,
            prev_right_foot_pos=curr_right_foot_pos,
            foot_contacts=self._cal.get_geom_based_foot_contacts(
                data, normalize=True
            ),  # For AMP
            root_height=curr_root_pose.height,  # Actual root height for AMP features
            linvel_obs_mask=linvel_obs_mask,
            push_schedule=wr.push_schedule,
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
        reset_schedule = sample_push_schedule(state.rng, self._config.env)
        reset_state = self._make_initial_state(
            self._init_qpos,
            velocity_cmd,
            base_info=state.info,
            push_schedule=reset_schedule,
        )

        # v0.10.2: Preserve termination diagnostics in metrics even after reset
        # The reset_state.metrics has all zeros for term/*, but we want to keep
        # the actual termination reason for logging
        # v0.10.4: Also preserve episode_step_count for accurate ep_len
        # v0.10.4: Also preserve tracking metrics for accurate vel_err/torque logging
        preserved_metrics = {
            **reset_state.metrics,
            "episode_step_count": metrics[
                "episode_step_count"
            ],  # Terminal step count for ep_len
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
            linvel_obs_mask=reset_wr_info.linvel_obs_mask,
            push_schedule=reset_wr_info.push_schedule,
        )
        preserved_info = dict(reset_state.info)  # Copy wrapper fields
        preserved_info[WR_INFO_KEY] = preserved_wr_info  # Update wr namespace

        final_data, final_obs, final_metrics, final_info, final_rng = jax.lax.cond(
            done > 0.5,
            lambda: (
                reset_state.data,
                reset_state.obs,
                preserved_metrics,  # Use preserved metrics with term/* intact
                preserved_info,  # Use preserved info with truncated intact
                reset_state.rng,
            ),
            lambda: (data, obs, metrics, info, state.rng),
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
            rng=final_rng,
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
        linvel_obs_mask: Optional[jax.Array] = None,
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
        - Foot switches (toe/heel L/R): 4
        - Previous action: 8
        - Velocity command: 1
        - Padding: 1

        Total: 39

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
        gravity = self._cal.get_gravity_vector(data)

        # v0.9.1: Use finite-diff velocities when prev_root_* are available
        # This is the key fix for AMP parity - matches reference data computation
        if prev_root_pos is not None and prev_root_quat is not None:
            # Finite-diff velocities (heading-local) - matches reference exactly
            curr_pose = self._cal.get_root_pose(data)
            prev_pose = Pose3D(
                position=prev_root_pos,
                orientation=prev_root_quat,
                frame=CoordinateFrame.WORLD,
            )
            fd_vel = curr_pose.velocity_fd(prev_pose, self.dt)
            linvel = fd_vel.linear
            angvel = fd_vel.angular
        else:
            # qvel-based velocities (for reset when no previous state)
            # These will be zero at reset anyway
            vel = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
            linvel, angvel = vel.linear, vel.angular

        if linvel_obs_mask is not None:
            linvel = linvel * linvel_obs_mask

        # Joint positions and velocities (normalized for NN input)
        joint_pos = self._cal.get_joint_positions(data.qpos, normalize=True).squeeze()
        joint_vel = self._cal.get_joint_velocities(data.qvel, normalize=True).squeeze()
        foot_switches = self._cal.get_foot_switches(
            data, threshold=self._config.env.foot_switch_threshold
        )

        # Build observation using shared layout
        return build_observation_from_components(
            spec=self._policy_spec,
            gravity_local=gravity,
            angvel_heading_local=angvel,
            linvel_heading_local=linvel,
            joint_pos_normalized=joint_pos,
            joint_vel_normalized=joint_vel,
            foot_switches=foot_switches,
            prev_action=action,
            velocity_cmd=velocity_cmd,
        )

    def _sample_linvel_obs_mask(
        self, push_schedule: DisturbanceSchedule
    ) -> tuple[jax.Array, DisturbanceSchedule]:
        """Sample per-episode linvel observation mask.

        This is a Sim2Real helper: real hardware may not have a reliable base
        linear velocity estimate, so we can train with linvel forced to zero
        (or dropped out) while keeping the observation layout unchanged.
        """
        mode = self._config.env.linvel_mode
        if mode == "sim":
            return jp.ones(()), push_schedule
        if mode == "zero":
            return jp.zeros(()), push_schedule
        if mode == "dropout":
            p = jp.clip(self._config.env.linvel_dropout_prob, 0.0, 1.0)
            rng, key = jax.random.split(push_schedule.rng)
            keep = jax.random.bernoulli(key, p=1.0 - p).astype(jp.float32)
            updated_schedule = DisturbanceSchedule(
                start_step=push_schedule.start_step,
                end_step=push_schedule.end_step,
                force_xy=push_schedule.force_xy,
                rng=rng,
            )
            return keep.reshape(()), updated_schedule
        raise ValueError(
            f"Unknown env.linvel_mode={mode!r} (expected: 'sim', 'zero', 'dropout')"
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        raw_action: jax.Array,
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

        # Cache root pose (used for height + orientation)
        root_pose = self._cal.get_root_pose(data)

        # Cache root velocity (used for forward/lateral vel + angular vel penalty)
        # HEADING_LOCAL: horizontal velocity in heading direction (robust to pitch/roll)
        root_vel = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )

        # 1. Forward velocity tracking (exp shaping)
        forward_vel, lateral_vel, _ = root_vel.linear_xyz
        vel_error = jp.abs(forward_vel - velocity_cmd)
        forward_reward = jp.exp(
            -vel_error * self._config.reward_weights.forward_velocity_scale
        )

        # 2. Lateral velocity penalty (penalize sideways drift)
        lateral_penalty = jp.square(lateral_vel)

        # 3. Alive/healthy reward (height-based)
        height = root_pose.height
        healthy = jp.where(
            (height > self._config.env.min_height)
            & (height < self._config.env.max_height),
            1.0,
            0.0,
        )

        # 4. Orientation penalty (pitch² + roll²)
        roll, pitch, _ = root_pose.euler_angles()
        orientation_penalty = jp.square(pitch) + jp.square(roll)

        # 5. Angular velocity penalty (reduce body rotation)
        # Note: Using cached root_vel (LOCAL frame) - reuse from above
        angvel_x, angvel_y, angvel_z = root_vel.angular_xyz
        # Only penalize yaw rate (z-axis rotation), not pitch/roll which are handled by orientation penalty
        angvel_penalty = jp.square(angvel_z)
        pitch_rate = angvel_y

        # =====================================================================
        # EFFORT REWARDS (Tier 1 - critical for sim2real)
        # =====================================================================

        # 6. Torque penalty - penalize normalized torque usage
        normalized_torques = self._cal.get_actuator_torques(
            data.qfrc_actuator, normalize=True
        )
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
        action_abs = jp.abs(action)
        action_abs_max = jp.max(action_abs)
        action_sat_frac = jp.mean((action_abs > 0.95).astype(jp.float32))
        raw_action_abs = jp.abs(raw_action)
        raw_action_abs_max = jp.max(raw_action_abs)
        raw_action_sat_frac = jp.mean((raw_action_abs > 0.95).astype(jp.float32))

        # 9. Joint velocity penalty (efficiency) - use normalized for consistent scaling
        joint_vel = self._cal.get_joint_velocities(data.qvel, normalize=True)
        joint_vel_penalty = jp.sum(jp.square(joint_vel))

        # =====================================================================
        # FOOT STABILITY REWARDS (Tier 2)
        # =====================================================================

        # Get foot contact forces for slip/clearance gating
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        # Robot weight for contact threshold (approximate: mass * gravity)
        # ~2kg robot * 9.81 ≈ 20N total, so ~5N per foot is "loaded"
        # v0.10.1: Read from config instead of hard-coding
        contact_threshold = self._config.env.contact_threshold_force

        # 10. Slip penalty (when foot is loaded, penalize tangential velocity)
        # Use finite-diff velocity if previous positions available
        if prev_left_foot_pos is not None and prev_right_foot_pos is not None:
            left_vel, right_vel = self._cal.get_foot_velocities(
                data,
                prev_left_foot_pos,
                prev_right_foot_pos,
                self.dt,
                normalize=False,
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
        left_clearance, right_clearance = self._cal.get_foot_clearances(
            data, normalize=False
        )
        left_clearance = jp.clip(
            left_clearance - min_clearance, 0.0, 0.05
        )  # cap at 5cm
        right_clearance = jp.clip(right_clearance - min_clearance, 0.0, 0.05)

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
        gait_periodicity = jp.asarray(gait_periodicity).reshape(())

        # Get weights from frozen config (type-safe access)
        weights = self._config.reward_weights

        # 13. Hip/knee swing (encourage leg articulation during swing)
        # Use CAL semantic accessors - joint names defined in robot_config (single source)
        left_hip_pitch, right_hip_pitch = self._cal.get_hip_pitch_positions(
            data.qpos, normalize=True
        )
        left_knee_pitch, right_knee_pitch = self._cal.get_knee_pitch_positions(
            data.qpos, normalize=True
        )

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
        hip_swing = jp.asarray(hip_swing).reshape(())
        knee_swing = jp.asarray(knee_swing).reshape(())

        # 14. Flight phase penalty (discourage hopping)
        flight_phase = jp.where((~left_loaded) & (~right_loaded), 1.0, 0.0)
        flight_phase = jp.asarray(flight_phase).reshape(())

        # 15. Height target shaping (encourage upright posture)
        # One-sided shaping: don't penalize being above target height.
        height_target = self._config.env.target_height
        height_sigma = jp.maximum(weights.height_target_sigma, 1e-6)
        height_deficit = jp.maximum(height_target - height, 0.0) / height_sigma
        height_target_reward = jp.exp(-jp.square(height_deficit))

        # 16. Stance width penalty (discourage wide split stance)
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        stance_width = jp.abs(left_foot_pos[1] - right_foot_pos[1])
        stance_sigma = jp.maximum(weights.stance_width_sigma, 1e-6)
        stance_excess = jp.maximum(stance_width - weights.stance_width_target, 0.0)
        stance_width_penalty = jp.square(stance_excess / stance_sigma)

        # =====================================================================
        # COMBINE REWARDS
        # =====================================================================

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
            + weights.height_target * height_target_reward
            + weights.stance_width_penalty * stance_width_penalty
        )

        # =====================================================================
        # REWARD COMPONENTS FOR LOGGING
        # =====================================================================

        # v0.10.3: Compute tracking metrics for walking exit criteria
        # Max normalized torque (peak torque as fraction of limit, across all actuators)
        max_torque_ratio = jp.max(jp.abs(normalized_torques))
        avg_torque = jp.mean(
            jp.abs(self._cal.get_actuator_torques(data.qfrc_actuator, normalize=False))
        )

        foot_switches = self._cal.get_foot_switches(
            data, threshold=self._config.env.foot_switch_threshold
        )

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
            "reward/height_target": height_target_reward,
            "reward/stance_width_penalty": stance_width_penalty,
            # Debug metrics
            "debug/pitch": pitch,
            "debug/roll": roll,
            "debug/forward_vel": forward_vel,
            "debug/lateral_vel": lateral_vel,
            "debug/left_force": left_force,
            "debug/right_force": right_force,
            "debug/left_toe_switch": foot_switches[0],
            "debug/left_heel_switch": foot_switches[1],
            "debug/right_toe_switch": foot_switches[2],
            "debug/right_heel_switch": foot_switches[3],
            "debug/pitch_rate": pitch_rate,
            "debug/action_abs_max": action_abs_max,
            "debug/action_sat_frac": action_sat_frac,
            "debug/raw_action_abs_max": raw_action_abs_max,
            "debug/raw_action_sat_frac": raw_action_sat_frac,
            # v0.10.3: Tracking metrics for walking exit criteria
            "tracking/vel_error": vel_error,  # |forward_vel - velocity_cmd|
            "tracking/max_torque": max_torque_ratio,  # max(|torque|/limit)
            "tracking/avg_torque": avg_torque,  # mean(|torque|) in Nm
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
        # Cache root pose (used for height + orientation)
        root_pose = self._cal.get_root_pose(data)
        height = root_pose.height

        # Height termination (failure - robot fell)
        height_too_low = height < self._config.env.min_height
        height_too_high = height > self._config.env.max_height
        height_fail = height_too_low | height_too_high

        # v0.10.1: Orientation termination (extreme pitch/roll)
        roll, pitch, _ = root_pose.euler_angles()
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
    # Properties (MjxEnv interface)
    # =========================================================================

    @property
    def xml_path(self) -> str:
        """Path to the xml file for the environment (required by MjxEnv)."""
        return str(self._model_path)

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
    def cal(self) -> ControlAbstractionLayer:
        """Access the Control Abstraction Layer for joint/actuator operations."""
        return self._cal

    @property
    def observation_size(self) -> int:
        """Return observation size using shared ObsLayout.

        v0.7.0: Velocities are now in heading-local frame for AMP parity.
        The observation layout is unchanged.

        See ObsLayout class for component breakdown.
        """
        return ObsLayout.total_size()

    def _build_policy_spec(self) -> PolicySpec:
        action_dim = self._robot_config.action_dim

        layout = [
            ObsFieldSpec(name="gravity_local", size=ObsLayout.GRAVITY_DIM, frame="local", units="unit_vector"),
            ObsFieldSpec(
                name="angvel_heading_local", size=ObsLayout.ANGVEL_DIM, frame="heading_local", units="rad_s"
            ),
            ObsFieldSpec(
                name="linvel_heading_local", size=ObsLayout.LINVEL_DIM, frame="heading_local", units="m_s"
            ),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=ObsLayout.FOOT_SWITCH_DIM, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=ObsLayout.VELOCITY_CMD_DIM, units="m_s"),
            ObsFieldSpec(name="padding", size=ObsLayout.PADDING_DIM, units="unused"),
        ]

        joints: dict[str, JointSpec] = {}
        for item in self._robot_config.actuated_joints:
            name = str(item.get("name"))
            rng = item.get("range") or [0.0, 0.0]
            if len(rng) != 2:
                raise ValueError(f"Invalid range for joint '{name}': {rng}")
            joints[name] = JointSpec(
                range_min_rad=float(rng[0]),
                range_max_rad=float(rng[1]),
                mirror_sign=float(item.get("mirror_sign", 1.0)),
                max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
            )

        postprocess_id = "lowpass_v1" if self._config.env.action_filter_alpha > 0.0 else "none"
        postprocess_params = {}
        if postprocess_id == "lowpass_v1":
            postprocess_params["alpha"] = float(self._config.env.action_filter_alpha)

        return PolicySpec(
            contract_name="wildrobot_policy",
            contract_version="1.0.0",
            spec_version=1,
            model=ModelSpec(
                format="onnx",
                input_name="observation",
                output_name="action",
                dtype="float32",
                obs_dim=ObsLayout.total_size(),
                action_dim=action_dim,
            ),
            robot=RobotSpec(
                robot_name=self._robot_config.robot_name,
                actuator_names=list(self._robot_config.actuator_names),
                joints=joints,
            ),
            observation=ObservationSpec(
                dtype="float32",
                layout_id="wr_obs_v1",
                linvel_mode=self._config.env.linvel_mode,
                layout=layout,
            ),
            action=ActionSpec(
                dtype="float32",
                bounds={"min": -1.0, "max": 1.0},
                postprocess_id=postprocess_id,
                postprocess_params=postprocess_params,
                mapping_id="pos_target_rad_v1",
            ),
            provenance=None,
        )
