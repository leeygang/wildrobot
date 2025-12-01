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
"""WildRobot walking with modular reward system for human-like gait."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from common import base_env
from amp.rewards import contact_rewards


class WildRobotWalkEnv(base_env.WildRobotEnvBase):
    """WildRobot walking task with modular rewards for human-like motion.

    This environment builds on the baseline locomotion with:
    - Modular reward system (easy to add/remove components)
    - Phase 1: Foot contact and sliding penalties
    - Phase 2: Motion imitation (future)
    - Phase 3: Hybrid approach (future)
    """

    def __init__(
        self,
        task: str = "wildrobot_flat",
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        if config is None:
            config = config_dict.ConfigDict()
            config.ctrl_dt = 0.02  # 50Hz control frequency
            config.sim_dt = 0.002  # 500Hz simulation frequency

        super().__init__(task, config, config_overrides)

        # Default joint positions for standing pose
        self._default_qpos = jp.array(
            [
                -0.4,  # right_hip_pitch
                0.0,  # right_hip_roll
                0.8,  # right_knee_pitch
                -0.4,  # right_ankle_pitch
                0.0,  # right_foot_roll
                -0.4,  # left_hip_pitch
                0.0,  # left_hip_roll
                0.8,  # left_knee_pitch
                -0.4,  # left_ankle_pitch
                0.0,  # left_foot_roll
                0.0,  # waist_yaw
            ]
        )

        self._target_height = 0.45  # Target standing height (meters)

        # Velocity command configuration
        self._velocity_mode = getattr(config, "velocity_command_mode", "range")

        if self._velocity_mode == "fixed":
            self._target_velocity = getattr(config, "target_velocity", 0.7)
            self._min_velocity = self._target_velocity
            self._max_velocity = self._target_velocity
        elif self._velocity_mode == "range":
            self._min_velocity = getattr(config, "min_velocity", 0.5)
            self._max_velocity = getattr(config, "max_velocity", 1.0)
            self._target_velocity = None
        else:
            raise ValueError(
                f"Invalid velocity_command_mode: {self._velocity_mode}. "
                "Must be 'fixed' or 'range'"
            )

        # Termination height range
        self._min_height = getattr(config, "min_height", 0.2)
        self._max_height = getattr(config, "max_height", 0.7)

        # Action filtering
        self._use_action_filter = getattr(config, "use_action_filter", True)
        self._action_filter_alpha = getattr(config, "action_filter_alpha", 0.7)

        # Phase signal (gait clock)
        self._use_phase_signal = getattr(config, "use_phase_signal", True)
        self._phase_period = getattr(config, "phase_period", 40)
        self._num_phase_clocks = getattr(config, "num_phase_clocks", 2)

        # === MODULAR REWARD SYSTEM ===
        # Load reward weights from config
        reward_weights = getattr(config, "reward_weights", {})
        self._reward_weights = reward_weights

        # Initialize reward components (Phase 1: Contact rewards)
        self._init_reward_components(reward_weights)

    def _init_reward_components(self, reward_weights: Dict[str, float]):
        """Initialize modular reward components."""
        # Phase 1: Foot contact rewards (optional, enabled via config)
        self._use_contact_rewards = reward_weights.get("foot_contact", 0.0) > 0
        self._use_sliding_penalty = reward_weights.get("foot_sliding", 0.0) > 0
        self._use_air_time_reward = reward_weights.get("foot_air_time", 0.0) > 0

        if self._use_contact_rewards:
            self._foot_contact_reward = contact_rewards.FootContactReward(
                model=self.mj_model,  # Use MuJoCo model, not MJX model
                left_foot_geoms=self.robot_config.left_feet_geoms,
                right_foot_geoms=self.robot_config.right_feet_geoms,
                contact_threshold=1.0,
            )

        if self._use_sliding_penalty:
            self._foot_sliding_penalty = contact_rewards.FootSlidingPenalty(
                model=self.mj_model,  # Use MuJoCo model, not MJX model
                left_foot_site=self.robot_config.feet_sites[0],  # left_foot_mimic
                right_foot_site=self.robot_config.feet_sites[1],  # right_foot_mimic
                left_foot_geoms=self.robot_config.left_feet_geoms,
                right_foot_geoms=self.robot_config.right_feet_geoms,
                contact_threshold=1.0,
            )

        if self._use_air_time_reward:
            self._foot_air_time_reward = contact_rewards.FootAirTimeReward(
                model=self.mj_model,
                left_foot_geoms=self.robot_config.left_feet_geoms,
                right_foot_geoms=self.robot_config.right_feet_geoms,
                contact_threshold=1.0,
                min_air_time=0.15,  # 150ms minimum flight time
                dt=self.dt,  # Control timestep
            )

    def reset(self, rng: jax.Array) -> base_env.WildRobotEnvState:
        """Reset the environment with random velocity command."""
        rng, key1, key2 = jax.random.split(rng, 3)

        # Sample random velocity command
        velocity_cmd = jax.random.uniform(
            key1, shape=(), minval=self._min_velocity, maxval=self._max_velocity
        )

        # Initialize with default standing pose
        qpos = jp.zeros(self.mj_model.nq)
        qvel = jp.zeros(self.mj_model.nv)

        # Set floating base to standing height
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, self._target_height]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))  # quaternion (w,x,y,z)

        # Set joint positions with small random noise
        joint_noise = jax.random.uniform(
            key2, shape=(len(self._default_qpos),), minval=-0.05, maxval=0.05
        )
        joint_positions = self._default_qpos + joint_noise
        qpos = qpos.at[7 : 7 + len(self._default_qpos)].set(joint_positions)

        # Create MJX data
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_qpos)
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(
            data, jp.zeros(self.action_size), velocity_cmd, step_count=0
        )
        reward, done = jp.zeros(2)

        actual_height = self.get_floating_base_qpos(data.qpos)[2]

        # Initialize ALL metrics with default values (required for JAX scan pytree consistency)
        metrics = {
            # Core metrics
            "velocity_command": velocity_cmd,
            "height": actual_height,
            "forward_velocity": jp.zeros(()),
            "distance_walked": jp.zeros(()),
            "success": jp.ones(()),
            # Reward components (all initialized to 0.0)
            "reward/total": jp.zeros(()),
            "reward/tracking_exp_xy": jp.zeros(()),
            "reward/tracking_lin_xy": jp.zeros(()),
            "reward/tracking_exp_yaw": jp.zeros(()),
            "reward/tracking_lin_yaw": jp.zeros(()),
            "reward/forward_velocity_bonus": jp.zeros(()),
            "reward/z_velocity": jp.zeros(()),
            "reward/roll_pitch_velocity": jp.zeros(()),
            "reward/roll_pitch_position": jp.zeros(()),
            "reward/nominal_joint_position": jp.zeros(()),
            "reward/joint_position_limit": jp.zeros(()),
            "reward/joint_velocity": jp.zeros(()),
            "reward/joint_acceleration": jp.zeros(()),
            "reward/action_rate": jp.zeros(()),
            "reward/joint_torque": jp.zeros(()),
            "reward/mechanical_power": jp.zeros(()),
            "reward/foot_contact": jp.zeros(()),
            "reward/foot_sliding": jp.zeros(()),
            "reward/foot_air_time": jp.zeros(()),
            "reward/existential": jp.zeros(()),
            # Contact metrics
            "contact/left_foot_force": jp.zeros(()),
            "contact/right_foot_force": jp.zeros(()),
            "contact/left_in_contact": jp.zeros(()),
            "contact/right_in_contact": jp.zeros(()),
            "contact/both_feet_contact": jp.zeros(()),
            "contact/no_feet_contact": jp.zeros(()),
            "contact/gait_phase": jp.zeros(()),
            "contact/left_sliding_vel": jp.zeros(()),
            "contact/right_sliding_vel": jp.zeros(()),
            "contact/left_air_time": jp.zeros(()),
            "contact/right_air_time": jp.zeros(()),
            "contact/avg_air_time": jp.zeros(()),
            "contact/left_meets_air_time_threshold": jp.zeros(()),
            "contact/right_meets_air_time_threshold": jp.zeros(()),
            "contact/both_meet_air_time_threshold": jp.zeros(()),
        }

        info = {
            "step_count": 0,
            "prev_action": jp.zeros(self.action_size),
            "prev_qvel": jp.zeros(self.mj_model.nv - 6),  # Track previous joint velocities (exclude floating base)
            "prev_x_position": jp.zeros(()),
            "left_air_time": jp.zeros(()),
            "right_air_time": jp.zeros(()),
        }

        state = base_env.WildRobotEnvState(
            data=data, obs=obs, reward=reward, done=done,
            metrics=metrics, info=info, pipeline_state=data
        )

        return state

    def step(self, state: base_env.WildRobotEnvState, action: jax.Array) -> base_env.WildRobotEnvState:
        """Step the environment."""
        velocity_cmd = state.metrics.get("velocity_command", 0.0)
        step_count = state.info.get("step_count", 0)
        prev_action = state.info.get("prev_action", jp.zeros(self.action_size))
        prev_qvel = state.info.get("prev_qvel", None)

        # Action filtering
        if self._use_action_filter:
            alpha = self._action_filter_alpha
            filtered_action = alpha * prev_action + (1.0 - alpha) * action
        else:
            filtered_action = action

        # Apply action
        data = state.data.replace(ctrl=filtered_action)

        # Physics simulation
        def step_fn(data, _):
            data = mjx.step(self.mjx_model, data)
            return data, None

        data, _ = jax.lax.scan(step_fn, data, None, length=int(self.dt / self.sim_dt))

        new_step_count = step_count + 1

        # Get current joint velocities (excluding floating base)
        current_qvel = self.get_actuator_joints_qvel(data.qvel)

        # Get observations and rewards
        obs = self._get_obs(
            data, filtered_action, velocity_cmd, step_count=new_step_count
        )
        reward, reward_components = self._get_reward(
            data, filtered_action, velocity_cmd, prev_action=prev_action,
            prev_qvel=prev_qvel, step_count=new_step_count
        )

        # === AIR TIME REWARD (Phase 1B) ===
        # Must be computed here to update state and add to reward
        if self._use_air_time_reward:
            left_air_time = state.info.get("left_air_time", 0.0)
            right_air_time = state.info.get("right_air_time", 0.0)
            air_time_state = (left_air_time, right_air_time)

            air_time_reward, new_air_time_state = self._foot_air_time_reward.compute(
                data, filtered_action, air_time_state
            )

            # Add to reward components and apply weight
            reward_components["foot_air_time"] = air_time_reward
            air_time_weight = float(self._reward_weights.get("foot_air_time", 0.0))
            reward = reward + air_time_weight * air_time_reward

            # Update state
            state.info["left_air_time"] = new_air_time_state[0]
            state.info["right_air_time"] = new_air_time_state[1]

        done = self._get_done(data)

        # Update metrics
        forward_velocity = self.get_local_linvel(data)[0]
        height = self.get_floating_base_qpos(data.qpos)[2]

        current_x_position = self.get_floating_base_qpos(data.qpos)[0]
        prev_x_position = state.info.get("prev_x_position", current_x_position)
        step_distance = jp.abs(current_x_position - prev_x_position)

        prev_distance = state.metrics.get("distance_walked", 0.0)
        total_distance = prev_distance + step_distance

        # Core metrics
        state.metrics["height"] = height
        state.metrics["forward_velocity"] = forward_velocity
        state.metrics["velocity_command"] = velocity_cmd
        state.metrics["distance_walked"] = total_distance
        state.metrics["success"] = jp.where(done > 0.5, 0.0, 1.0)

        # === REWARD COMPONENT METRICS ===
        state.metrics["reward/total"] = reward
        for name, value in reward_components.items():
            state.metrics[f"reward/{name}"] = value

        # === CONTACT METRICS (Phase 1) ===
        if self._use_contact_rewards:
            # Calculate phase
            phase = (new_step_count % self._phase_period) / self._phase_period

            # Get contact forces
            left_force = self._foot_contact_reward.get_contact_force(
                data, self._foot_contact_reward.left_foot_geom_ids
            )
            right_force = self._foot_contact_reward.get_contact_force(
                data, self._foot_contact_reward.right_foot_geom_ids
            )

            # Add to metrics
            state.metrics["contact/left_foot_force"] = left_force
            state.metrics["contact/right_foot_force"] = right_force
            state.metrics["contact/left_in_contact"] = jp.where(left_force > 1.0, 1.0, 0.0)
            state.metrics["contact/right_in_contact"] = jp.where(right_force > 1.0, 1.0, 0.0)
            state.metrics["contact/both_feet_contact"] = jp.where(
                (left_force > 1.0) & (right_force > 1.0), 1.0, 0.0
            )
            state.metrics["contact/no_feet_contact"] = jp.where(
                (left_force <= 1.0) & (right_force <= 1.0), 1.0, 0.0
            )
            state.metrics["contact/gait_phase"] = phase

        if self._use_sliding_penalty:
            # Get foot velocities
            left_foot_vel = self._foot_sliding_penalty.get_site_velocity(
                data,
                self._foot_sliding_penalty.left_foot_site_id,
                self._foot_sliding_penalty.left_foot_body_id
            )
            right_foot_vel = self._foot_sliding_penalty.get_site_velocity(
                data,
                self._foot_sliding_penalty.right_foot_site_id,
                self._foot_sliding_penalty.right_foot_body_id
            )

            # Horizontal sliding velocities
            state.metrics["contact/left_sliding_vel"] = jp.linalg.norm(left_foot_vel[..., :2])
            state.metrics["contact/right_sliding_vel"] = jp.linalg.norm(right_foot_vel[..., :2])

        if self._use_air_time_reward:
            # Air time metrics
            state.metrics["contact/left_air_time"] = state.info["left_air_time"]
            state.metrics["contact/right_air_time"] = state.info["right_air_time"]

            # Derived metrics for better visualization
            avg_air_time = (state.info["left_air_time"] + state.info["right_air_time"]) / 2.0
            state.metrics["contact/avg_air_time"] = avg_air_time

            # Check if meeting minimum threshold
            min_air_time_threshold = 0.15
            left_meets_threshold = jp.where(state.info["left_air_time"] >= min_air_time_threshold, 1.0, 0.0)
            right_meets_threshold = jp.where(state.info["right_air_time"] >= min_air_time_threshold, 1.0, 0.0)
            state.metrics["contact/left_meets_air_time_threshold"] = left_meets_threshold
            state.metrics["contact/right_meets_air_time_threshold"] = right_meets_threshold
            state.metrics["contact/both_meet_air_time_threshold"] = left_meets_threshold * right_meets_threshold

        # Update state info for next step
        state.info["step_count"] = new_step_count
        state.info["prev_action"] = filtered_action
        state.info["prev_qvel"] = current_qvel  # Save current velocity for next step's acceleration calculation
        state.info["prev_x_position"] = current_x_position

        return base_env.WildRobotEnvState(
            data=data, obs=obs, reward=reward, done=done,
            metrics=state.metrics, info=state.info, pipeline_state=data
        )

    def _get_obs(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
        step_count: int = 0,
    ) -> jax.Array:
        """Get observation vector including velocity command and phase signal."""
        # Joint state
        qpos = self.get_actuator_joint_qpos(data.qpos)
        qvel = self.get_actuator_joints_qvel(data.qvel)

        # Orientation and angular velocity
        gravity = self.get_gravity(data)
        angvel = self.get_global_angvel(data)

        # IMU readings
        gyro = self.get_gyros(data)
        accel = self.get_accelerometers(data)

        # Command
        cmd = jp.array([velocity_cmd])

        # Phase signal
        phase_obs = []
        if self._use_phase_signal:
            phase = (step_count % self._phase_period) / self._phase_period * 2 * jp.pi

            if self._num_phase_clocks == 1:
                phase_obs = [jp.sin(phase), jp.cos(phase)]
            elif self._num_phase_clocks == 2:
                left_phase = phase
                right_phase = phase + jp.pi
                phase_obs = [
                    jp.sin(left_phase),
                    jp.cos(left_phase),
                    jp.sin(right_phase),
                    jp.cos(right_phase),
                ]

        obs_components = [
            qpos,  # 11
            qvel,  # 11
            gravity,  # 3
            angvel,  # 3
            gyro,  # 9
            accel,  # 9
            action,  # 11
            cmd,  # 1
        ]

        if phase_obs:
            obs_components.append(jp.array(phase_obs))

        obs = jp.concatenate(obs_components)

        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
        prev_action: jax.Array = None,
        prev_qvel: jax.Array = None,
        step_count: int = 0,
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        """Calculate reward using modular reward system.

        Returns:
            tuple: (total_reward, reward_components_dict)
        """
        # Get basic state
        linvel = self.get_local_linvel(data)
        angvel = self.get_global_angvel(data)
        gravity = self.get_gravity(data)
        qpos = self.get_actuator_joint_qpos(data.qpos)
        qvel = self.get_actuator_joints_qvel(data.qvel)

        # Calculate phase for contact rewards
        phase = (step_count % self._phase_period) / self._phase_period

        # === MODULAR REWARD COMPONENTS ===
        reward_components = {}

        # 1. Velocity tracking (baseline rewards)
        reward_components.update(
            self._compute_velocity_tracking_reward(linvel, angvel, velocity_cmd)
        )

        # 2. Stability rewards (baseline)
        reward_components.update(
            self._compute_stability_reward(linvel, angvel, gravity)
        )

        # 3. Smoothness rewards (baseline) - NOW includes acceleration!
        reward_components.update(
            self._compute_smoothness_reward(qpos, qvel, action, prev_action, prev_qvel)
        )

        # 4. Energy rewards (baseline)
        reward_components.update(
            self._compute_energy_reward(action, qvel)
        )

        # 5. Phase 1: Contact rewards (NEW)
        if self._use_contact_rewards:
            contact_reward = self._foot_contact_reward.compute(data, action, phase=phase)
            reward_components["foot_contact"] = contact_reward

        if self._use_sliding_penalty:
            sliding_penalty = self._foot_sliding_penalty.compute(data, action)
            reward_components["foot_sliding"] = sliding_penalty

        if self._use_air_time_reward:
            # Get air time state from info dict (will be updated in step())
            # Here we just compute the reward based on current state
            # The actual state update happens in step() to avoid circular dependency
            reward_components["foot_air_time"] = jp.array(0.0)  # Placeholder, updated in step()

        # 6. Existential penalty
        reward_components["existential"] = jp.array(-5.0)

        # === APPLY WEIGHTS AND SUM ===
        total_reward = jp.array(0.0)
        for name, value in reward_components.items():
            # Convert weight to float (handles strings like "1e-6")
            weight = float(self._reward_weights.get(name, 0.0))
            total_reward = total_reward + weight * value

        return total_reward, reward_components

    def _compute_velocity_tracking_reward(
        self, linvel: jax.Array, angvel: jax.Array, velocity_cmd: jax.Array
    ) -> Dict[str, float]:
        """Compute velocity tracking rewards."""
        # XY velocity tracking
        xy_vel_error = jp.sqrt(
            jp.square(linvel[0] - velocity_cmd) + jp.square(linvel[1])
        )
        tracking_sigma = self._reward_weights.get("tracking_sigma", 0.25)
        tracking_exp_xy = jp.exp(-xy_vel_error / tracking_sigma)
        tracking_lin_xy = jp.exp(-jp.square(xy_vel_error))

        # Forward velocity bonus - ONLY reward actual forward motion
        # This prevents robot from getting tracking reward while moving backward
        # Robot must move forward to get this bonus (max(0, ...) clips negative velocities to 0)
        forward_velocity_bonus = jp.maximum(0.0, linvel[0])

        # Yaw velocity tracking
        yaw_vel_error = jp.abs(angvel[2])
        tracking_exp_yaw = jp.exp(-yaw_vel_error / tracking_sigma)
        tracking_lin_yaw = jp.exp(-jp.square(yaw_vel_error))

        return {
            "tracking_exp_xy": tracking_exp_xy,
            "tracking_lin_xy": tracking_lin_xy,
            "tracking_exp_yaw": tracking_exp_yaw,
            "tracking_lin_yaw": tracking_lin_yaw,
            "forward_velocity_bonus": forward_velocity_bonus,
        }

    def _compute_stability_reward(
        self, linvel: jax.Array, angvel: jax.Array, gravity: jax.Array
    ) -> Dict[str, float]:
        """Compute stability rewards."""
        z_vel_penalty = -jp.square(linvel[2])
        roll_pitch_vel_penalty = -jp.sum(jp.square(angvel[:2]))
        roll_pitch_pos_penalty = -jp.square(gravity[2] - 1.0)

        return {
            "z_velocity": z_vel_penalty,
            "roll_pitch_velocity": roll_pitch_vel_penalty,
            "roll_pitch_position": roll_pitch_pos_penalty,
        }

    def _compute_smoothness_reward(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        action: jax.Array,
        prev_action: Optional[jax.Array],
        prev_qvel: Optional[jax.Array],
    ) -> Dict[str, float]:
        """Compute smoothness rewards."""
        nominal_joint_penalty = -jp.sum(jp.square(qpos - self._default_qpos))
        joint_limit_penalty = -jp.sum(
            jp.square(jp.clip(jp.abs(qpos) - 1.5, 0.0, jp.inf))
        )

        # Joint velocity metric (for monitoring, not penalty)
        # This is useful to track even though we don't penalize it
        joint_vel_metric = -jp.sum(jp.square(qvel))

        # Joint acceleration penalty (PROPER smoothness penalty)
        # Penalizes jerky motion for smooth, natural movement
        if prev_qvel is not None:
            # acceleration = (qvel - prev_qvel) / dt
            # We use ctrl_dt as dt (typically 0.02s)
            dt = 0.02  # This should match env ctrl_dt
            joint_acc = (qvel - prev_qvel) / dt
            joint_acc_penalty = -jp.sum(jp.square(joint_acc))
        else:
            # First step, no previous velocity available
            joint_acc_penalty = jp.array(0.0)

        if prev_action is not None:
            action_rate_penalty = -jp.sum(jp.square(action - prev_action))
        else:
            action_rate_penalty = -jp.sum(jp.square(action))

        return {
            "nominal_joint_position": nominal_joint_penalty,
            "joint_position_limit": joint_limit_penalty,
            "joint_velocity": joint_vel_metric,  # METRIC ONLY (weight=0.0, for monitoring)
            "joint_acceleration": joint_acc_penalty,
            "action_rate": action_rate_penalty,
        }

    def _compute_energy_reward(
        self, action: jax.Array, qvel: jax.Array
    ) -> Dict[str, float]:
        """Compute energy efficiency rewards."""
        torque_penalty = -jp.sum(jp.square(action))
        energy_penalty = -jp.sum(jp.square(action * qvel))

        return {
            "joint_torque": torque_penalty,
            "mechanical_power": energy_penalty,
        }

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """Check if episode should terminate."""
        height = self.get_floating_base_qpos(data.qpos)[2]
        fallen = jp.where(
            (height < self._min_height) | (height > self._max_height), 1.0, 0.0
        )
        return fallen

    @property
    def observation_size(self) -> int:
        """Size of observation vector."""
        base_size = 58  # qpos(11) + qvel(11) + gravity(3) + angvel(3) + gyro(9) + accel(9) + action(11) + cmd(1)

        phase_size = 0
        if self._use_phase_signal:
            if self._num_phase_clocks == 1:
                phase_size = 2
            elif self._num_phase_clocks == 2:
                phase_size = 4

        return base_size + phase_size
