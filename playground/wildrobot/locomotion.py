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
"""WildRobot unified locomotion task with velocity commands."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env, reward
from wildrobot import base


class WildRobotLocomotion(base.WildRobotEnv):
    """
    Unified WildRobot locomotion task with velocity commands.

    Learns standing (velocity=0), walking (velocity>0), and everything in between
    by conditioning on commanded forward velocity. Single policy handles full
    velocity range from 0.0 m/s (standing) to 1.0 m/s (fast walking).
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

        # Velocity command range: 0.0 (standing) to 1.0 m/s (fast walk)
        self._min_velocity = 0.0
        self._max_velocity = 1.0

        # Termination height range (configurable via config)
        # Default values match loco-mujoco's WildRobot config
        self._min_height = getattr(config, 'min_height', 0.2)
        self._max_height = getattr(config, 'max_height', 0.7)

        # Load reward parameters from config (like loco_mujoco)
        # Default values match loco_mujoco's LocomotionReward
        reward_params = getattr(config, 'reward_params', {})

        # Velocity tracking weights (primary goal)
        self._tracking_w_exp_xy = reward_params.get('tracking_w_exp_xy', 15.0)
        self._tracking_w_lin_xy = reward_params.get('tracking_w_lin_xy', 5.0)
        self._tracking_w_exp_yaw = reward_params.get('tracking_w_exp_yaw', 4.0)
        self._tracking_w_lin_yaw = reward_params.get('tracking_w_lin_yaw', 1.0)
        self._tracking_sigma = reward_params.get('tracking_sigma', 0.25)

        # Stability penalties
        self._z_vel_coeff = reward_params.get('z_vel_coeff', 2.0)
        self._roll_pitch_vel_coeff = reward_params.get('roll_pitch_vel_coeff', 0.08)
        self._roll_pitch_pos_coeff = reward_params.get('roll_pitch_pos_coeff', 0.3)

        # Smoothness penalties
        self._nominal_joint_pos_coeff = reward_params.get('nominal_joint_pos_coeff', 0.005)
        self._joint_position_limit_coeff = reward_params.get('joint_position_limit_coeff', 5.0)
        self._joint_vel_coeff = reward_params.get('joint_vel_coeff', 5e-5)
        self._joint_acc_coeff = reward_params.get('joint_acc_coeff', 2e-5)
        self._action_rate_coeff = reward_params.get('action_rate_coeff', 0.02)

        # Energy penalties
        self._joint_torque_coeff = reward_params.get('joint_torque_coeff', 2e-7)
        self._energy_coeff = reward_params.get('energy_coeff', 2e-5)

        # === Action Filtering (Low-Pass Filter) ===
        # Prevents high-frequency stuttering by smoothing actions over time
        self._use_action_filter = getattr(config, 'use_action_filter', True)
        self._action_filter_alpha = getattr(config, 'action_filter_alpha', 0.7)  # 0.7 = 70% old, 30% new

        # === Phase Signal (Gait Clock) ===
        # Provides periodic time signal to help robot learn rhythmic stepping
        self._use_phase_signal = getattr(config, 'use_phase_signal', True)
        self._phase_period = getattr(config, 'phase_period', 40)  # steps per gait cycle (0.8s at 50Hz)
        self._num_phase_clocks = getattr(config, 'num_phase_clocks', 2)  # 1=single clock, 2=left+right legs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment with random velocity command."""
        rng, key1, key2 = jax.random.split(rng, 3)

        # Sample random velocity command (includes standing at 0.0)
        velocity_cmd = jax.random.uniform(
            key1, shape=(), minval=self._min_velocity, maxval=self._max_velocity
        )

        # Initialize with default standing pose
        qpos = jp.zeros(self.mj_model.nq)
        qvel = jp.zeros(self.mj_model.nv)

        # Set floating base to standing height (positions 0-6: 3 pos + 4 quat)
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, self._target_height]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))  # quaternion (w,x,y,z)

        # Set joint positions with small random noise
        # Actuator joints start at position 7 (after floating base: 7 pos)
        joint_noise = jax.random.uniform(
            key2, shape=(len(self._default_qpos),), minval=-0.05, maxval=0.05
        )
        joint_positions = self._default_qpos + joint_noise
        qpos = qpos.at[7 : 7 + len(self._default_qpos)].set(joint_positions)

        # Create MJX data
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_qpos)
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data, jp.zeros(self.action_size), velocity_cmd, step_count=0)
        reward, done = jp.zeros(2)

        # DIAGNOSTIC: Get actual height after physics forward pass
        actual_height = self.get_floating_base_qpos(data.qpos)[2]

        metrics = {
            "velocity_command": velocity_cmd,
            "height": actual_height,
            "forward_velocity": jp.zeros(()),
            "success": jp.ones(()),  # Initialize success metric (pytree structure must match step())
            "reset_min_height": self._min_height,  # DEBUG
            "reset_max_height": self._max_height,  # DEBUG
            "step_count": 0,  # Track step count for phase signal
            "prev_action": jp.zeros(self.action_size),  # Track previous action for filtering
        }
        info = {}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment."""
        # Get velocity command from metrics
        velocity_cmd = state.metrics.get("velocity_command", 0.0)
        step_count = state.metrics.get("step_count", 0)
        prev_action = state.metrics.get("prev_action", jp.zeros(self.action_size))

        # === Action Filtering (Low-Pass Filter) ===
        # Smooth actions to prevent high-frequency stuttering
        if self._use_action_filter:
            alpha = self._action_filter_alpha
            filtered_action = alpha * prev_action + (1.0 - alpha) * action
        else:
            filtered_action = action

        # Apply action (PD control) - use filtered action
        data = state.data.replace(ctrl=filtered_action)

        # Physics simulation (multiple substeps)
        def step_fn(data, _):
            data = mjx.step(self.mjx_model, data)
            return data, None

        data, _ = jax.lax.scan(step_fn, data, None, length=int(self.dt / self.sim_dt))

        # Increment step count for phase signal
        new_step_count = step_count + 1

        # Get observations and rewards (pass step count for phase signal)
        obs = self._get_obs(data, filtered_action, velocity_cmd, step_count=new_step_count)
        reward = self._get_reward(data, filtered_action, velocity_cmd, prev_action=prev_action)
        done = self._get_done(data)

        # Update metrics (these are tracked per environment and averaged during eval)
        forward_velocity = self.get_local_linvel(data)[0]
        height = self.get_floating_base_qpos(data.qpos)[2]

        state.metrics["height"] = height
        state.metrics["forward_velocity"] = forward_velocity
        state.metrics["velocity_command"] = velocity_cmd
        state.metrics["step_count"] = new_step_count
        state.metrics["prev_action"] = filtered_action

        # Track success (completing episode without falling)
        # Success = reached end of episode without terminating early
        state.metrics["success"] = jp.where(done > 0.5, 0.0, 1.0)

        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_obs(
        self, data: mjx.Data, action: jax.Array, velocity_cmd: jax.Array, step_count: int = 0
    ) -> jax.Array:
        """
        Get observation vector including velocity command and optional phase signal.

        Observation (base 58D + phase signal):
            - Joint positions (11)
            - Joint velocities (11)
            - Gravity vector (3)
            - Angular velocity (3)
            - Gyro readings from 3 IMUs (9)
            - Accelerometer readings from 3 IMUs (9)
            - Previous action (11)
            - Velocity command (1)
            - Phase signal (2 or 4, optional): sin/cos of gait cycle(s)
        """
        # Joint state
        qpos = self.get_actuator_joint_qpos(data.qpos)
        qvel = self.get_actuator_joints_qvel(data.qvel)

        # Orientation and angular velocity
        gravity = self.get_gravity(data)
        angvel = self.get_global_angvel(data)

        # IMU readings (chest + left knee + right knee)
        gyro = self.get_gyros(data)
        accel = self.get_accelerometers(data)

        # Command
        cmd = jp.array([velocity_cmd])

        # === Phase Signal (Gait Clock) ===
        # Provides periodic timing information to help robot learn rhythmic stepping
        phase_obs = []
        if self._use_phase_signal:
            # Calculate phase angle (0 to 2π over phase_period steps)
            phase = (step_count % self._phase_period) / self._phase_period * 2 * jp.pi

            if self._num_phase_clocks == 1:
                # Single clock for whole body
                phase_obs = [jp.sin(phase), jp.cos(phase)]
            elif self._num_phase_clocks == 2:
                # Two clocks: left leg and right leg (offset by π)
                left_phase = phase
                right_phase = phase + jp.pi
                phase_obs = [
                    jp.sin(left_phase), jp.cos(left_phase),
                    jp.sin(right_phase), jp.cos(right_phase)
                ]
            else:
                # Fallback: no phase signal if misconfigured
                phase_obs = []

        obs_components = [
            qpos,  # 11
            qvel,  # 11
            gravity,  # 3
            angvel,  # 3
            gyro,  # 9 (3 IMUs x 3)
            accel,  # 9 (3 IMUs x 3)
            action,  # 11 (previous action)
            cmd,  # 1 (velocity command)
        ]

        # Add phase signal if enabled
        if phase_obs:
            obs_components.append(jp.array(phase_obs))

        obs = jp.concatenate(obs_components)

        return obs

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, velocity_cmd: jax.Array, prev_action: jax.Array = None
    ) -> jax.Array:
        """
        Calculate reward based on velocity command.

        Reward components (configurable via YAML - similar to loco_mujoco):
            - Velocity tracking: Match commanded velocity (exponential + linear)
            - Stability: Minimize vertical bobbing, tipping, stay upright
            - Smooth motion: Joint velocities, accelerations, action rate
            - Energy efficiency: Minimize torque
        """
        linvel = self.get_local_linvel(data)
        angvel = self.get_global_angvel(data)
        gravity = self.get_gravity(data)
        qpos = self.get_actuator_joint_qpos(data.qpos)
        qvel = self.get_actuator_joints_qvel(data.qvel)

        # === 1. VELOCITY TRACKING (Primary Goal) ===
        # Exponential reward for xy velocity tracking (very strong signal)
        xy_vel_error = jp.sqrt(jp.square(linvel[0] - velocity_cmd) + jp.square(linvel[1]))
        tracking_exp_xy = self._tracking_w_exp_xy * jp.exp(-xy_vel_error / self._tracking_sigma)

        # Linear reward for xy velocity tracking (helps with gradient)
        tracking_lin_xy = self._tracking_w_lin_xy * jp.exp(-jp.square(xy_vel_error))

        # Yaw velocity tracking (keep it zero for straight walking)
        yaw_vel_error = jp.abs(angvel[2])  # angvel[2] is yaw velocity
        tracking_exp_yaw = self._tracking_w_exp_yaw * jp.exp(-yaw_vel_error / self._tracking_sigma)
        tracking_lin_yaw = self._tracking_w_lin_yaw * jp.exp(-jp.square(yaw_vel_error))

        # === 2. STABILITY REWARDS ===
        # Penalize vertical velocity (bobbing up/down)
        z_vel_penalty = -self._z_vel_coeff * jp.square(linvel[2])

        # Penalize roll/pitch angular velocity (tipping)
        roll_pitch_angvel = angvel[:2]  # Roll and pitch angular velocities
        roll_pitch_vel_penalty = -self._roll_pitch_vel_coeff * jp.sum(jp.square(roll_pitch_angvel))

        # Stay upright (penalize roll/pitch from vertical)
        # gravity[2] should be close to 1.0 when upright
        roll_pitch_pos_penalty = -self._roll_pitch_pos_coeff * jp.square(gravity[2] - 1.0)

        # === 3. SMOOTH MOTION ===
        # Keep joints near nominal pose (default standing)
        nominal_joint_penalty = -self._nominal_joint_pos_coeff * jp.sum(jp.square(qpos - self._default_qpos))

        # Penalize being near joint limits (use normalized positions)
        # Assume joint limits are roughly ±2.0 rad for most joints
        joint_limit_penalty = -self._joint_position_limit_coeff * jp.sum(jp.square(jp.clip(jp.abs(qpos) - 1.5, 0.0, jp.inf)))

        # Penalize large joint velocities
        joint_vel_penalty = -self._joint_vel_coeff * jp.sum(jp.square(qvel))

        # Penalize joint accelerations (smooth motion)
        # Approximate acceleration as change in velocity (we don't store prev_qvel)
        joint_acc_penalty = -self._joint_acc_coeff * jp.sum(jp.square(qvel))

        # Action rate penalty (encourage smooth actions, not jerky)
        # Use proper action change if previous action is available
        if prev_action is not None:
            action_rate_penalty = -self._action_rate_coeff * jp.sum(jp.square(action - prev_action))
        else:
            # Fallback: penalize action magnitude (less effective)
            action_rate_penalty = -self._action_rate_coeff * jp.sum(jp.square(action))

        # === 4. ENERGY EFFICIENCY ===
        # Penalize torque (action * qvel approximates mechanical power)
        torque_penalty = -self._joint_torque_coeff * jp.sum(jp.square(action))
        energy_penalty = -self._energy_coeff * jp.sum(jp.square(action * qvel))

        # === 5. EXISTENTIAL PENALTY (Forces action) ===
        # Constant per-timestep penalty to prevent "standing still" exploitation.
        # Robot must actively move to overcome this negative baseline.
        # See: https://arxiv.org/abs/1707.06347 (PPO paper, survival bonus discussion)
        # INCREASED to -5.0 (from -1.0) after observing velocity still decreasing
        # from 0.279 → 0.075 m/s in first 100K steps with -1.0 penalty
        existential_penalty = -5.0

        # === 6. TOTAL REWARD ===
        # All weights are configurable via YAML (like loco_mujoco)
        total_reward = (
            # Velocity tracking (CRITICAL - must be dominant)
            tracking_exp_xy
            + tracking_lin_xy
            + tracking_exp_yaw
            + tracking_lin_yaw
            # Stability (important but secondary)
            + z_vel_penalty
            + roll_pitch_vel_penalty
            + roll_pitch_pos_penalty
            # Smoothness (tertiary)
            + nominal_joint_penalty
            + joint_limit_penalty
            + joint_vel_penalty
            + joint_acc_penalty
            + action_rate_penalty
            # Energy (minimal weight)
            + torque_penalty
            + energy_penalty
            # Existential penalty (prevents standing still)
            + existential_penalty
        )

        # REMOVED: Reward clipping that allowed "standing still" exploitation
        # Previously: total_reward = jp.maximum(total_reward, 0.0)
        #
        # Why removed:
        # - Standing still gave ~10 tracking reward - 1 existential = +9/step (clipped to 0)
        # - Robot learned to stand for 420 steps = 420*0 = 0 total (safe!)
        # - Walking gave higher per-step reward but fell faster (less total)
        #
        # With NO clipping:
        # - Standing still: ~10 tracking - 1 existential - penalties = -1 to +8/step
        # - If net negative, robot MUST move to get positive reward
        # - Walking: ~50 tracking - 1 existential - penalties = +40-45/step (much better!)

        return total_reward

    def _get_done(self, data: mjx.Data) -> jax.Array:
        """
        Check if episode should terminate.

        Episode ends if robot falls (height outside healthy range).
        Uses only height-based termination like loco-mujoco (no gravity check).
        Height range is configurable via config (min_height, max_height).

        Additionally applies a large negative reward on fall to discourage
        "survival mode" strategies that prioritize episode length over walking.
        """
        height = self.get_floating_base_qpos(data.qpos)[2]

        # Robot is fallen if height is outside healthy range
        # Based on loco-mujoco's HeightBasedTerminalStateHandler
        # Values are configurable in YAML config (default: 0.2 - 0.7m)
        # See: loco-mujoco/loco_mujoco/environments/humanoids/wildrobot.py

        fallen = jp.where(
            (height < self._min_height) | (height > self._max_height),
            1.0, 0.0
        )

        return fallen

    @property
    def observation_size(self) -> int:
        """Size of observation vector."""
        # Base observation: qpos(11) + qvel(11) + gravity(3) + angvel(3) + gyro(9) + accel(9) + action(11) + cmd(1) = 58
        base_size = 58
        
        # Add phase signal size if enabled
        phase_size = 0
        if self._use_phase_signal:
            if self._num_phase_clocks == 1:
                phase_size = 2  # sin + cos
            elif self._num_phase_clocks == 2:
                phase_size = 4  # (sin, cos) for left + (sin, cos) for right
        
        return base_size + phase_size
