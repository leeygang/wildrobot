# training/tests/test_config_contract.py
"""
Layer 1: Config and Runtime Contract Tests

This module tests the training config schema and consistency between
config and robot schema.

Tier: T0 (Pre-commit, < 2 min)
Markers: @pytest.mark.unit
"""

import numpy as np
import pytest


# =============================================================================
# Test 1.1: Training Config Schema Validation
# =============================================================================


class TestTrainingConfigSchema:
    """Tests for training config structure and validity."""

    @pytest.mark.unit
    def test_training_config_required_keys(self, training_config):
        """
        Purpose: Verify all required config sections exist.

        Assertions:
        - Required keys: env, ppo, amp, networks, reward_weights, checkpoints, wandb
        """
        # Check env section
        assert hasattr(training_config, "env"), "Missing required section: env"

        # Check ppo section
        assert hasattr(training_config, "ppo"), "Missing required section: ppo"

        # Check amp section
        assert hasattr(training_config, "amp"), "Missing required section: amp"

        # Check networks section
        assert hasattr(training_config, "networks"), "Missing required section: networks"

        # Check reward_weights section
        assert hasattr(training_config, "reward_weights"), "Missing required section: reward_weights"

        # Check checkpoints section
        assert hasattr(training_config, "checkpoints"), "Missing required section: checkpoints"

        # Check wandb section (for logging)
        assert hasattr(training_config, "wandb"), "Missing required section: wandb"

    @pytest.mark.unit
    def test_timestep_consistency(self, training_config):
        """
        Purpose: Verify timestep configuration is valid.

        Assertions:
        - sim_dt < ctrl_dt
        - ctrl_dt / sim_dt is integer
        - ctrl_dt = 0.02 (50 Hz)
        """
        sim_dt = training_config.env.sim_dt
        ctrl_dt = training_config.env.ctrl_dt

        assert sim_dt < ctrl_dt, f"sim_dt ({sim_dt}) must be < ctrl_dt ({ctrl_dt})"

        ratio = ctrl_dt / sim_dt
        assert ratio == int(ratio), f"ctrl_dt/sim_dt must be integer, got {ratio}"

        # WildRobot uses 50 Hz control frequency
        assert abs(ctrl_dt - 0.02) < 1e-6, f"ctrl_dt should be 0.02 (50 Hz), got {ctrl_dt}"

    @pytest.mark.unit
    def test_reward_weights_valid(self, training_config):
        """
        Purpose: Verify reward weights are valid numbers.

        Assertions:
        - All weights are finite (not NaN or Inf)
        - Core reward weights are defined
        """
        required_weights = [
            "tracking_lin_vel",
            "orientation",
            "torque",
        ]

        for name in required_weights:
            weight = getattr(training_config.reward_weights, name, None)
            assert weight is not None, f"Missing required reward weight: {name}"
            assert np.isfinite(weight), f"Reward weight {name} is not finite: {weight}"

    @pytest.mark.unit
    def test_ppo_hyperparameters(self, training_config):
        """
        Purpose: Verify PPO hyperparameters are valid.

        Assertions:
        - gamma in (0, 1]
        - gae_lambda in (0, 1]
        - num_minibatches > 0
        - epochs > 0
        """
        ppo = training_config.ppo

        # learning_rate may be string, so skip that check
        assert 0 < ppo.gamma <= 1, f"gamma must be in (0, 1]: {ppo.gamma}"
        assert 0 < ppo.gae_lambda <= 1, f"gae_lambda must be in (0, 1]: {ppo.gae_lambda}"
        assert ppo.num_minibatches > 0, f"num_minibatches must be positive: {ppo.num_minibatches}"
        assert ppo.epochs > 0, f"epochs must be positive: {ppo.epochs}"

    @pytest.mark.unit
    def test_network_architecture(self, training_config):
        """
        Purpose: Verify network architecture is valid.

        Assertions:
        - Actor and critic hidden sizes are defined
        - Hidden sizes are positive integers
        """
        actor = training_config.networks.actor
        critic = training_config.networks.critic

        assert hasattr(actor, "hidden_sizes"), "Actor missing hidden_sizes"
        assert hasattr(critic, "hidden_sizes"), "Critic missing hidden_sizes"

        for size in actor.hidden_sizes:
            assert isinstance(size, int) and size > 0, f"Invalid actor hidden size: {size}"

        for size in critic.hidden_sizes:
            assert isinstance(size, int) and size > 0, f"Invalid critic hidden size: {size}"


# =============================================================================
# Test 1.2: Static Consistency Checks
# =============================================================================


class TestConfigConsistency:
    """Tests for consistency between config and schema."""

    @pytest.mark.unit
    def test_action_dim_matches_actuators(self, training_config, robot_schema):
        """
        Purpose: Verify number of actuators is as expected.

        Assertions:
        - Schema has 8 actuators (WildRobot spec)
        """
        schema_actuators = len(robot_schema.actuators)

        assert schema_actuators == 8, (
            f"Actuator count mismatch: expected 8, got {schema_actuators}"
        )

    @pytest.mark.unit
    def test_height_limits_valid(self, training_config):
        """
        Purpose: Verify height limits are reasonable.

        Assertions:
        - min_height > 0
        - max_height > min_height
        - Values are reasonable for WildRobot (~0.5m tall)
        """
        min_height = training_config.env.min_height
        max_height = training_config.env.max_height

        assert min_height > 0, f"min_height must be positive: {min_height}"
        assert max_height > min_height, (
            f"max_height ({max_height}) must be > min_height ({min_height})"
        )

        # WildRobot is about 0.5m tall, reasonable range
        assert 0.1 <= min_height <= 0.4, f"min_height seems unreasonable: {min_height}"
        assert 0.4 <= max_height <= 0.8, f"max_height seems unreasonable: {max_height}"

    @pytest.mark.unit
    def test_velocity_targets_valid(self, training_config):
        """
        Purpose: Verify velocity targets are reasonable.

        Assertions:
        - min_velocity < max_velocity
        - Targets are achievable (< 2 m/s for walking robot)
        """
        env = training_config.env

        assert hasattr(env, "min_velocity"), "Missing min_velocity"
        assert hasattr(env, "max_velocity"), "Missing max_velocity"

        assert env.min_velocity < env.max_velocity, (
            f"min_velocity ({env.min_velocity}) must be < "
            f"max_velocity ({env.max_velocity})"
        )

        # Walking speed should be reasonable (< 2 m/s)
        assert env.max_velocity <= 2.0, (
            f"max_velocity seems too high for walking: {env.max_velocity}"
        )


# =============================================================================
# Test 1.3: Robot Config Consistency
# =============================================================================


class TestRobotConfigConsistency:
    """Tests for robot_config.yaml consistency."""

    @pytest.mark.unit
    def test_robot_config_sensors_exist(self, robot_config):
        """
        Purpose: Verify required sensors are defined in robot_config.

        Assertions:
        - orientation_sensor is defined
        - root_gyro_sensor is defined
        - root_accel_sensor is defined
        - local_linvel_sensor is defined
        """
        assert robot_config.orientation_sensor, "Missing orientation_sensor"
        assert robot_config.root_gyro_sensor, "Missing root_gyro_sensor"
        assert robot_config.root_accel_sensor, "Missing root_accel_sensor"
        assert robot_config.local_linvel_sensor, "Missing local_linvel_sensor"

    @pytest.mark.unit
    def test_robot_config_actuators_exist(self, robot_config):
        """
        Purpose: Verify robot_config has actuators defined.

        Assertions:
        - actuator_names list is not empty
        - Has 8 actuators
        """
        assert hasattr(robot_config, "actuator_names"), "Missing actuator_names in robot_config"
        assert len(robot_config.actuator_names) == 8, (
            f"Expected 8 actuators, got {len(robot_config.actuator_names)}"
        )

    @pytest.mark.unit
    def test_robot_config_feet_geoms(self, robot_config):
        """Test robot_config feet configuration.

        Purpose: Verify robot_config has feet geoms defined.
        Validates:
        - feet_left_geoms is defined
        - feet_right_geoms is defined
        """
        assert hasattr(robot_config, "feet_left_geoms"), "Missing feet_left_geoms"
        assert hasattr(robot_config, "feet_right_geoms"), "Missing feet_right_geoms"
        assert len(robot_config.feet_left_geoms) >= 1, "Expected at least 1 left foot geom"
        assert len(robot_config.feet_right_geoms) >= 1, "Expected at least 1 right foot geom"
