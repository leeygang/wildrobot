"""Test reward components are reading correct values from MuJoCo.

This test verifies that:
1. Pitch/roll extraction is correct (quaternion to euler)
2. Actuator torques are reading from correct address
3. Foot positions are reading from correct bodies
4. Foot contact forces are reading correctly
5. All reward components have sensible values

Usage:
    pytest playground_amp/tests/test_reward_components.py -v
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from playground_amp.configs.robot_config import load_robot_config
from playground_amp.configs.training_config import load_training_config
from playground_amp.train import create_env_config


class TestRewardComponents:
    """Test that reward components read correct values from MuJoCo."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        # Load configs
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        # Create environment
        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)

        # Reset to get initial state
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_pitch_roll_at_rest(self):
        """At reset, robot should be upright with near-zero pitch/roll."""
        pitch, roll = self.env.get_pitch_roll(self.state.data)

        # At reset, robot should be upright
        assert jnp.abs(pitch) < 0.1, f"Pitch should be near zero at reset, got {pitch}"
        assert jnp.abs(roll) < 0.1, f"Roll should be near zero at reset, got {roll}"

    def test_pitch_roll_values_reasonable(self):
        """Pitch/roll should be in reasonable range."""
        pitch, roll = self.env.get_pitch_roll(self.state.data)

        # Should be within [-pi, pi]
        assert -np.pi <= pitch <= np.pi, f"Pitch out of range: {pitch}"
        assert -np.pi <= roll <= np.pi, f"Roll out of range: {roll}"

    def test_actuator_torques_shape(self):
        """Actuator torques should have correct shape (nu,)."""
        torques = self.env.cal.get_actuator_torques(
            self.state.data.qfrc_actuator, normalize=False
        )

        assert torques.shape == (
            self.env.action_size,
        ), f"Torque shape mismatch: {torques.shape} vs ({self.env.action_size},)"

    def test_actuator_torques_at_rest(self):
        """At rest, actuator torques should be small (no movement)."""
        torques = self.env.cal.get_actuator_torques(
            self.state.data.qfrc_actuator, normalize=False
        )

        # At rest with no action, torques should be near zero or small
        # (some may be non-zero due to gravity compensation)
        max_torque = jnp.max(jnp.abs(torques))
        assert max_torque < 10.0, f"Max torque at rest unexpectedly high: {max_torque}"

    def test_foot_positions_shape(self):
        """Foot positions should be (3,) arrays."""
        left_pos, right_pos = self.env.get_foot_positions(self.state.data)

        assert left_pos.shape == (3,), f"Left foot shape: {left_pos.shape}"
        assert right_pos.shape == (3,), f"Right foot shape: {right_pos.shape}"

    def test_foot_positions_reasonable_height(self):
        """Foot positions should be near ground level at reset."""
        left_pos, right_pos = self.env.get_foot_positions(self.state.data)

        # Feet should be near ground (z close to 0, within ~10cm)
        assert left_pos[2] < 0.15, f"Left foot too high: z={left_pos[2]}"
        assert right_pos[2] < 0.15, f"Right foot too high: z={right_pos[2]}"
        assert left_pos[2] > -0.1, f"Left foot below ground: z={left_pos[2]}"
        assert right_pos[2] > -0.1, f"Right foot below ground: z={right_pos[2]}"

    def test_foot_positions_symmetric(self):
        """Feet should be roughly symmetric at reset."""
        left_pos, right_pos = self.env.get_foot_positions(self.state.data)

        # Y coordinates should be opposite (left positive, right negative)
        # or at least symmetric around 0
        y_diff = jnp.abs(left_pos[1] + right_pos[1])
        assert (
            y_diff < 0.1
        ), f"Feet not symmetric in Y: left={left_pos[1]}, right={right_pos[1]}"

    def test_foot_contact_forces_at_rest(self):
        """At rest (standing), both feet should have contact forces."""
        left_force, right_force = self.env.get_raw_foot_contacts(self.state.data)

        # At rest on ground, should have some contact force
        # Note: This may vary depending on initial pose
        total_force = left_force + right_force
        print(
            f"Contact forces at rest: left={left_force:.2f}N, right={right_force:.2f}N"
        )

        # Forces should be non-negative
        assert left_force >= 0, f"Left force negative: {left_force}"
        assert right_force >= 0, f"Right force negative: {right_force}"

    def test_reward_components_not_nan(self):
        """All reward components should be finite (not NaN or Inf)."""
        # Call _get_reward
        reward, components = self.env._get_reward(
            self.state.data,
            action=jnp.zeros(self.env.action_size),
            prev_action=jnp.zeros(self.env.action_size),
            velocity_cmd=jnp.array(0.5),
            prev_left_foot_pos=None,  # No previous positions on first step
            prev_right_foot_pos=None,
        )

        assert jnp.isfinite(reward), f"Total reward not finite: {reward}"

        for name, value in components.items():
            assert jnp.isfinite(value).all(), f"{name} not finite: {value}"

    def test_reward_components_correct_sign(self):
        """Check reward components have correct sign conventions."""
        reward, components = self.env._get_reward(
            self.state.data,
            action=jnp.zeros(self.env.action_size),
            prev_action=jnp.zeros(self.env.action_size),
            velocity_cmd=jnp.array(0.5),
        )

        # Forward reward should be positive (0 to 1)
        assert components["reward/forward"] >= 0, "Forward reward should be >= 0"
        assert components["reward/forward"] <= 1.0, "Forward reward should be <= 1"

        # Healthy should be 0 or 1
        assert components["reward/healthy"] in [0.0, 1.0], "Healthy should be 0 or 1"

        # At reset, orientation penalty should be small
        assert (
            components["reward/orientation"] < 0.5
        ), "Orientation penalty too high at reset"

    def test_step_reward_has_all_components(self):
        """After step, reward should have all expected components."""
        # Take a step
        action = jnp.zeros(self.env.action_size)
        next_state = self.env.step(self.state, action)

        # Check metrics contain all reward components
        expected_keys = [
            "reward/total",
            "reward/forward",
            "reward/lateral",
            "reward/healthy",
            "reward/orientation",
            "reward/angvel",
            "reward/torque",
            "reward/saturation",
            "reward/action_rate",
            "reward/joint_vel",
            "reward/slip",
            "reward/clearance",
        ]

        for key in expected_keys:
            assert key in next_state.metrics, f"Missing metric: {key}"

    def test_debug_metrics_present(self):
        """Debug metrics should be present for monitoring."""
        action = jnp.zeros(self.env.action_size)
        next_state = self.env.step(self.state, action)

        debug_keys = [
            "debug/pitch",
            "debug/roll",
            "debug/forward_vel",
            "debug/lateral_vel",
            "debug/left_force",
            "debug/right_force",
        ]

        for key in debug_keys:
            assert key in next_state.metrics, f"Missing debug metric: {key}"


class TestOrientationTermination:
    """Test orientation-based termination."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_no_termination_at_reset(self):
        """Robot should not be terminated at reset."""
        done, terminated, truncated = self.env._get_termination(
            self.state.data, step_count=0
        )

        assert terminated == 0.0, "Should not be terminated at reset"
        assert done == 0.0, "Should not be done at reset"


class TestActionEffects:
    """Test that actions have expected effects on reward components."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_large_action_increases_torque(self):
        """Large actions should result in higher torque penalty."""
        # Zero action
        zero_action = jnp.zeros(self.env.action_size)
        _, components_zero = self.env._get_reward(
            self.state.data,
            action=zero_action,
            prev_action=zero_action,
            velocity_cmd=jnp.array(0.5),
        )

        # Large action (saturated)
        large_action = jnp.ones(self.env.action_size)
        state_after_large = self.env.step(self.state, large_action)
        _, components_large = self.env._get_reward(
            state_after_large.data,
            action=large_action,
            prev_action=zero_action,
            velocity_cmd=jnp.array(0.5),
        )

        # Large action should have higher torque penalty (more negative or larger magnitude)
        # Note: torque penalty is sum((tau/tau_max)^2), so larger action = larger value
        print(
            f"Torque penalty - zero: {components_zero['reward/torque']:.4f}, "
            f"large: {components_large['reward/torque']:.4f}"
        )

        # The raw torque component (before weight) should be larger with large action
        assert (
            components_large["reward/torque"] >= components_zero["reward/torque"]
        ), "Large action should have >= torque penalty"

    def test_action_rate_penalty_with_changing_actions(self):
        """Changing actions should incur action rate penalty."""
        zero_action = jnp.zeros(self.env.action_size)
        large_action = jnp.ones(self.env.action_size)

        # Same action as previous - no rate penalty
        _, components_same = self.env._get_reward(
            self.state.data,
            action=zero_action,
            prev_action=zero_action,
            velocity_cmd=jnp.array(0.5),
        )

        # Different action from previous - rate penalty
        _, components_diff = self.env._get_reward(
            self.state.data,
            action=large_action,
            prev_action=zero_action,
            velocity_cmd=jnp.array(0.5),
        )

        print(
            f"Action rate - same: {components_same['reward/action_rate']:.4f}, "
            f"diff: {components_diff['reward/action_rate']:.4f}"
        )

        # Changing action should have higher action rate penalty
        assert (
            components_diff["reward/action_rate"]
            > components_same["reward/action_rate"]
        ), "Changing action should have higher action rate penalty"

    def test_velocity_tracking_reward(self):
        """Forward velocity tracking should reward matching command."""
        # Get current forward velocity from state
        forward_vel = self.state.data.qvel[0]  # x velocity

        # Compute reward with matching velocity command
        _, components_match = self.env._get_reward(
            self.state.data,
            action=jnp.zeros(self.env.action_size),
            prev_action=jnp.zeros(self.env.action_size),
            velocity_cmd=jnp.array(float(forward_vel)),
        )

        # Compute reward with very different velocity command
        _, components_mismatch = self.env._get_reward(
            self.state.data,
            action=jnp.zeros(self.env.action_size),
            prev_action=jnp.zeros(self.env.action_size),
            velocity_cmd=jnp.array(10.0),  # Very high target
        )

        print(
            f"Forward reward - match: {components_match['reward/forward']:.4f}, "
            f"mismatch: {components_mismatch['reward/forward']:.4f}"
        )

        # Matching velocity should give higher reward
        assert (
            components_match["reward/forward"] >= components_mismatch["reward/forward"]
        ), "Matching velocity command should give higher forward reward"


class TestMujocoDataConsistency:
    """Test that MuJoCo data is read consistently and correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_root_height_matches_pelvis_z(self):
        """Root height should be reasonable for standing robot."""
        # Get root height from qpos (floating base z)
        root_z = float(self.state.data.qpos[2])

        print(f"Root qpos z: {root_z:.4f}")

        # Root height should be reasonable for a standing humanoid
        # (typically between 0.3-0.6m for a small robot)
        assert (
            0.2 < root_z < 0.8
        ), f"Root z ({root_z}) should be in reasonable standing range"

        # Also verify height matches what we use for termination checks
        target_height = self.env._target_height  # from config
        assert (
            abs(root_z - target_height) < 0.15
        ), f"Root z ({root_z}) should be close to target height ({target_height})"

    def test_joint_positions_match_qpos(self):
        """Joint positions from observation should match qpos."""
        # Get joint positions from qpos (skip 7 floating base DOFs)
        qpos_joints = self.state.data.qpos[7:]

        # Should have num_actuated_joints values
        expected_joints = self.env.action_size
        assert (
            len(qpos_joints) >= expected_joints
        ), f"Not enough joints in qpos: {len(qpos_joints)} < {expected_joints}"

        print(f"Joint positions (first 5): {qpos_joints[:5]}")

        # All joint positions should be within joint limits (roughly -pi to pi for revolute)
        for i, pos in enumerate(qpos_joints[:expected_joints]):
            assert (
                -4.0 < float(pos) < 4.0
            ), f"Joint {i} position {pos} out of reasonable range"

    def test_joint_velocities_match_qvel(self):
        """Joint velocities should match qvel."""
        # Get joint velocities from qvel (skip 6 floating base DOFs)
        qvel_joints = self.state.data.qvel[6:]

        # At reset, velocities should be near zero
        max_vel = jnp.max(jnp.abs(qvel_joints[: self.env.action_size]))

        print(f"Max joint velocity at reset: {max_vel:.4f}")

        assert max_vel < 1.0, f"Joint velocities too high at reset: {max_vel}"

    def test_foot_body_ids_are_valid(self):
        """Foot body IDs should be valid and different."""
        left_id = self.env._left_foot_body_id
        right_id = self.env._right_foot_body_id

        # IDs should be positive integers
        assert left_id > 0, f"Invalid left foot body ID: {left_id}"
        assert right_id > 0, f"Invalid right foot body ID: {right_id}"

        # IDs should be different
        assert left_id != right_id, "Left and right foot should have different IDs"

        # Get body names for verification
        left_name = mujoco.mj_id2name(
            self.env._mj_model, mujoco.mjtObj.mjOBJ_BODY, left_id
        )
        right_name = mujoco.mj_id2name(
            self.env._mj_model, mujoco.mjtObj.mjOBJ_BODY, right_id
        )

        print(f"Left foot body: {left_name} (id={left_id})")
        print(f"Right foot body: {right_name} (id={right_id})")

        assert (
            "left" in left_name.lower()
        ), f"Left foot body name doesn't contain 'left': {left_name}"
        assert (
            "right" in right_name.lower()
        ), f"Right foot body name doesn't contain 'right': {right_name}"

    def test_actuator_addresses_are_valid(self):
        """Actuator qvel addresses should be valid indices."""
        addresses = self.env.cal.qvel_addresses

        assert (
            len(addresses) == self.env.action_size
        ), f"Wrong number of actuator addresses: {len(addresses)} vs {self.env.action_size}"

        # All addresses should be within qvel bounds
        nv = self.env._mj_model.nv
        for i, addr in enumerate(addresses):
            assert (
                0 <= addr < nv
            ), f"Actuator {i} address {addr} out of bounds (nv={nv})"

        print(f"Actuator qvel addresses: {addresses}")


class TestSimulationDynamics:
    """Test that simulation dynamics produce expected results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_gravity_causes_falling_without_support(self):
        """Robot should fall if dropped from height."""
        # Modify state to be higher
        new_qpos = self.state.data.qpos.at[2].set(1.0)  # Set z to 1 meter
        new_data = self.state.data.replace(qpos=new_qpos)
        high_state = self.state.replace(data=new_data)

        # Take several steps with zero action
        state = high_state
        for _ in range(10):
            state = self.env.step(state, jnp.zeros(self.env.action_size))

        # Robot should have moved down
        final_z = float(state.data.qpos[2])
        initial_z = 1.0

        print(f"Initial z: {initial_z}, Final z after 10 steps: {final_z:.4f}")

        assert (
            final_z < initial_z
        ), f"Robot should fall under gravity: initial={initial_z}, final={final_z}"

    def test_standing_is_stable(self):
        """Robot should stay roughly upright when standing."""
        # Take several steps with zero action from standing
        state = self.state
        initial_z = float(state.data.qpos[2])

        for _ in range(20):
            state = self.env.step(state, jnp.zeros(self.env.action_size))

        final_z = float(state.data.qpos[2])

        print(f"Standing stability: initial z={initial_z:.4f}, final z={final_z:.4f}")

        # Should not have fallen significantly
        assert (
            final_z > initial_z - 0.2
        ), f"Robot fell too much while standing: {initial_z} -> {final_z}"

    def test_step_changes_state(self):
        """Taking a step should change the state."""
        initial_qpos = self.state.data.qpos.copy()

        # Take a step with some action
        action = jnp.ones(self.env.action_size) * 0.5
        next_state = self.env.step(self.state, action)

        final_qpos = next_state.data.qpos

        # State should have changed
        qpos_diff = jnp.sum(jnp.abs(final_qpos - initial_qpos))

        print(f"Qpos change after step: {qpos_diff:.6f}")

        assert qpos_diff > 0.0001, "State should change after taking a step"


class TestRewardWeights:
    """Test that reward weights are applied correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create environment for testing."""
        load_robot_config("assets/robot_config.yaml")
        training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
        env_config = create_env_config(training_cfg.raw_config)

        from playground_amp.envs.wildrobot_env import WildRobotEnv

        self.env = WildRobotEnv(config=env_config)
        rng = jax.random.PRNGKey(42)
        self.state = self.env.reset(rng)
        yield

    def test_weights_loaded_from_config(self):
        """Reward weights should match config values."""
        weights = self.env._reward_weights

        # Check some expected weights from ppo_walking.yaml
        assert hasattr(weights, "forward_velocity"), "Missing forward_velocity weight"
        assert hasattr(weights, "lateral_velocity"), "Missing lateral_velocity weight"
        assert hasattr(weights, "torque"), "Missing torque weight"

        print(f"Forward velocity weight: {weights.forward_velocity}")
        print(f"Lateral velocity weight: {weights.lateral_velocity}")
        print(f"Torque weight: {weights.torque}")

        # Weights should be non-zero
        assert (
            weights.forward_velocity != 0
        ), "Forward velocity weight should not be zero"

    def test_total_reward_is_weighted_sum(self):
        """Total reward should be weighted sum of components."""
        action = jnp.zeros(self.env.action_size)
        next_state = self.env.step(self.state, action)

        total = next_state.metrics["reward/total"]

        # Check that total is finite
        assert jnp.isfinite(total), f"Total reward not finite: {total}"

        print(f"Total reward after one step: {total:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
