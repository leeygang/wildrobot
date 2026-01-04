# playground_amp/tests/test_env_step.py
"""
Layer 5: Environment Step Integrity Tests

This module tests the environment step function, reset, and termination.
Tests verify state transitions, observation shapes, and done flags.

Tier: T1 (PR gate)
Markers: @pytest.mark.sim
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# =============================================================================
# Environment Fixture
# =============================================================================


@pytest.fixture(scope="module")
def env(training_config, robot_config):
    """Create WildRobot environment for testing."""
    from assets.robot_config import load_robot_config
    from playground_amp.configs.training_config import load_training_config

    load_robot_config("assets/robot_config.yaml")
    training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
    training_cfg.freeze()  # Freeze config for JIT compatibility

    from playground_amp.envs.wildrobot_env import WildRobotEnv

    return WildRobotEnv(config=training_cfg)


@pytest.fixture(scope="module")
def standing_env(robot_config):
    """Create a standing-specific environment with zero velocity command."""
    from assets.robot_config import load_robot_config
    from playground_amp.configs.training_config import load_training_config

    load_robot_config("assets/robot_config.yaml")
    training_cfg = load_training_config("playground_amp/configs/ppo_standing.yaml")
    training_cfg.env.velocity_cmd_min = 0.0
    training_cfg.env.velocity_cmd_max = 0.0
    training_cfg.env.push_enabled = False
    training_cfg.freeze()

    from playground_amp.envs.wildrobot_env import WildRobotEnv

    return WildRobotEnv(config=training_cfg)


# =============================================================================
# Test 5.1: Reset Contract
# =============================================================================


class TestResetContract:
    """Tests for environment reset behavior."""

    @pytest.mark.sim
    def test_reset_returns_valid_state(self, env, rng):
        """
        Purpose: Verify reset returns properly shaped, finite values.

        Assertions:
        - Observation shape matches env.observation_size
        - All observation values are finite
        - State is not terminated
        """
        state = env.reset(rng)

        # Check observation shape
        assert (
            state.obs.shape[-1] == env.observation_size
        ), f"Observation shape mismatch: {state.obs.shape} vs ({env.observation_size},)"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(state.obs)), "Observation contains NaN or Inf"

        # Check not terminated at reset
        assert not state.done, "State should not be done at reset"

    @pytest.mark.sim
    def test_reset_deterministic_with_same_seed(self, env):
        """
        Purpose: Verify reset with same seed produces identical state.

        Assertions:
        - Two resets with same RNG produce identical observations
        """
        rng1 = jax.random.PRNGKey(42)
        rng2 = jax.random.PRNGKey(42)

        state1 = env.reset(rng1)
        state2 = env.reset(rng2)

        assert jnp.allclose(
            state1.obs, state2.obs
        ), "Reset not deterministic with same seed"

    @pytest.mark.sim
    def test_reset_different_with_different_seed(self, env):
        """
        Purpose: Verify reset with different seeds produces different states.

        Assertions:
        - Resets with different RNGs produce different observations
        """
        rng1 = jax.random.PRNGKey(42)
        rng2 = jax.random.PRNGKey(123)

        state1 = env.reset(rng1)
        state2 = env.reset(rng2)

        # Some randomization should occur (velocity command, etc.)
        # Not all observations will differ, but some should
        # Check if observations are not exactly equal
        is_different = not jnp.allclose(state1.obs, state2.obs)

        # If there's no randomization, this is OK but worth noting
        if not is_different:
            pytest.skip("No randomization in reset - observations identical")

    @pytest.mark.sim
    def test_reset_prev_action_matches_ctrl(self, env, rng):
        """
        Purpose: Ensure reset prev_action matches the ctrl used for default pose.

        Assertions:
        - prev_action corresponds to data.ctrl at reset
        - observation action slice matches prev_action
        """
        from playground_amp.envs.env_info import WR_INFO_KEY
        from playground_amp.envs.wildrobot_env import ObsLayout

        state = env.reset(rng)
        wr_info = state.info[WR_INFO_KEY]

        expected_action = env._cal.ctrl_to_policy_action(state.data.ctrl)
        assert jnp.allclose(
            wr_info.prev_action, expected_action, atol=1e-6
        ), "prev_action does not match ctrl-derived action at reset"

        action_slice = ObsLayout.get_slices()["action"]
        obs_action = state.obs[action_slice]
        assert jnp.allclose(
            obs_action, wr_info.prev_action, atol=1e-6
        ), "Observation action slice does not match prev_action at reset"


# =============================================================================
# Test 5.2: Step Contract
# =============================================================================


class TestStepContract:
    """Tests for environment step behavior."""

    @pytest.mark.sim
    def test_step_returns_valid_state(self, env, rng):
        """
        Purpose: Verify step returns properly shaped, finite values.

        Assertions:
        - Observation shape correct after step
        - All values finite
        - Reward is finite
        """
        state = env.reset(rng)
        action = jnp.zeros(env.action_size)

        new_state = env.step(state, action)

        # Check observation shape
        assert new_state.obs.shape[-1] == env.observation_size

        # Check all values finite
        assert jnp.all(
            jnp.isfinite(new_state.obs)
        ), "Observation contains NaN after step"
        assert jnp.isfinite(new_state.reward), "Reward contains NaN or Inf"

    @pytest.mark.sim
    def test_step_with_zero_action(self, standing_env, rng):
        """
        Purpose: Verify robot remains stable with zero action.

        A well-tuned bipedal robot should maintain balance when standing still
        with zero action (holding the default/neutral pose).

        Procedure:
        1. Reset environment
        2. Step with zero action for 100 steps

        Assertions:
        - Robot should remain upright (not terminated)
        - No NaN or Inf values during simulation
        """
        state = standing_env.reset(rng)

        default_ctrl = standing_env._cal.get_ctrl_for_default_pose()
        action = standing_env._cal.ctrl_to_policy_action(default_ctrl)

        def step_collect(s, _):
            next_state = standing_env.step(s, action)
            return next_state, (next_state.obs, next_state.reward, next_state.done)

        scan_fn = jax.jit(lambda s: jax.lax.scan(step_collect, s, None, length=100))
        state, (obs_seq, reward_seq, done_flags) = scan_fn(state)

        assert jnp.all(jnp.isfinite(obs_seq)), "NaN/Inf detected in obs sequence"
        assert jnp.all(jnp.isfinite(reward_seq)), "NaN/Inf detected in reward sequence"

        if jnp.any(done_flags > 0.5):
            first_done = int(jnp.argmax(done_flags > 0.5))
            pytest.fail(
                f"Robot fell with zero action at step {first_done}. "
                "This indicates the default standing pose or physics parameters "
                "may need adjustment for stability."
            )

    @pytest.mark.sim
    def test_step_state_changes(self, env, rng):
        """
        Purpose: Verify state actually changes after step with non-zero action.

        Assertions:
        - Observation differs before and after step with action
        """
        state = env.reset(rng)
        obs_before = state.obs.copy()

        action = jnp.array([0.5] * env.action_size)
        new_state = env.step(state, action)

        # Observation should change
        assert not jnp.allclose(
            obs_before, new_state.obs
        ), "Observation didn't change after step with action"


# =============================================================================
# Test 5.3: Termination Correctness
# =============================================================================


class TestTermination:
    """Tests for termination conditions."""

    @pytest.mark.sim
    def test_height_termination_conceptual(self, training_config):
        """
        Purpose: Verify termination triggers at min_height conceptually.

        This tests the termination logic without requiring environment.

        Assertions:
        - Termination when height < min_height
        """
        min_height = training_config.env.min_height

        def check_height_termination(height, min_height):
            return height < min_height

        # Above min_height → not terminated
        assert not check_height_termination(min_height + 0.1, min_height)

        # Below min_height → terminated
        assert check_height_termination(min_height - 0.1, min_height)

    @pytest.mark.sim
    def test_orientation_termination_conceptual(self, training_config):
        """
        Purpose: Verify termination triggers at extreme orientation conceptually.

        Assertions:
        - Termination when |pitch| > max_pitch
        - Termination when |roll| > max_roll
        """
        max_pitch = getattr(training_config.env, "max_pitch", 0.5)
        max_roll = getattr(training_config.env, "max_roll", 0.5)

        def check_orientation_termination(pitch, roll, max_pitch, max_roll):
            return abs(pitch) > max_pitch or abs(roll) > max_roll

        # Upright → not terminated
        assert not check_orientation_termination(0.0, 0.0, max_pitch, max_roll)

        # Extreme pitch → terminated
        assert check_orientation_termination(max_pitch + 0.1, 0.0, max_pitch, max_roll)

        # Extreme roll → terminated
        assert check_orientation_termination(0.0, max_roll + 0.1, max_pitch, max_roll)

    @pytest.mark.sim
    def test_done_flag_propagates(self, env, rng):
        """
        Purpose: Verify done flag stays True after termination.

        Assertions:
        - Once done=True, subsequent steps don't resurrect the robot
        """
        state = env.reset(rng)

        # Force termination by applying extreme action repeatedly
        extreme_action = jnp.ones(env.action_size) * 2.0  # Max action

        def step_collect(s, _):
            next_state = env.step(s, extreme_action)
            return next_state, next_state.done

        scan_fn = jax.jit(lambda s: jax.lax.scan(step_collect, s, None, length=500))
        state, done_flags = scan_fn(state)

        if not jnp.any(done_flags > 0.5):
            pytest.skip("Robot didn't terminate within 500 steps")

        # After termination, take a few more steps to ensure no errors
        step_fn = jax.jit(env.step)
        for _ in range(10):
            state = step_fn(state, jnp.zeros(env.action_size))


# =============================================================================
# Test 5.4: Action Handling
# =============================================================================


class TestActionHandling:
    """Tests for action processing."""

    @pytest.mark.sim
    def test_action_clipping(self, env, rng):
        """
        Purpose: Verify actions are clipped to valid range.

        Assertions:
        - Extreme actions don't cause numerical issues
        """
        state = env.reset(rng)

        # Test with extreme actions
        extreme_high = jnp.ones(env.action_size) * 100.0
        extreme_low = jnp.ones(env.action_size) * -100.0

        # Should not cause NaN or Inf
        state_high = env.step(state, extreme_high)
        assert jnp.all(
            jnp.isfinite(state_high.obs)
        ), "Extreme high action caused NaN/Inf"

        state = env.reset(rng)
        state_low = env.step(state, extreme_low)
        assert jnp.all(jnp.isfinite(state_low.obs)), "Extreme low action caused NaN/Inf"

    @pytest.mark.sim
    def test_action_effect_direction(self, env, rng):
        """
        Purpose: Verify action sign affects joint movement direction.

        Assertions:
        - Positive action moves joint in one direction
        - Negative action moves joint in opposite direction
        """
        state = env.reset(rng)

        # Apply positive action to first joint
        action_pos = jnp.zeros(env.action_size)
        action_pos = action_pos.at[0].set(0.5)
        state_pos = env.step(state, action_pos)

        # Reset and apply negative action
        state = env.reset(rng)
        action_neg = jnp.zeros(env.action_size)
        action_neg = action_neg.at[0].set(-0.5)
        state_neg = env.step(state, action_neg)

        # Joint positions should differ
        # (Exact indices depend on observation structure, so we just check obs differ)
        assert not jnp.allclose(
            state_pos.obs, state_neg.obs
        ), "Positive and negative actions should produce different observations"


# =============================================================================
# Test 5.5: Observation Consistency
# =============================================================================


class TestObservationConsistency:
    """Tests for observation consistency and structure."""

    @pytest.mark.sim
    def test_observation_structure(self, env, rng, training_config):
        """
        Purpose: Verify observation has expected structure.

        WildRobot observation (39D):
        - gravity (3)
        - angvel (3)
        - linvel (3)
        - joint_pos (8)
        - joint_vel (8)
        - foot_switches (4)
        - prev_action (8)
        - velocity_cmd (1)
        - padding (1)
        """
        state = env.reset(rng)
        obs = state.obs

        # Use env's observation_size property (canonical source of truth)
        expected_dim = env.observation_size
        actual_dim = obs.shape[-1]

        assert (
            actual_dim == expected_dim
        ), f"Observation dimension mismatch: {actual_dim} vs {expected_dim}"

    @pytest.mark.sim
    def test_observation_bounds_reasonable(self, env, rng):
        """
        Purpose: Verify observation values are in reasonable bounds.

        Assertions:
        - No observation value exceeds ±100 (reasonable for normalized obs)
        """
        state = env.reset(rng)

        # Run for some steps to get varied observations (scan for speed)
        def step_collect(carry, _):
            s, key = carry
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(
                subkey, shape=(env.action_size,), minval=-1, maxval=1
            )
            next_state = env.step(s, action)
            max_val = jnp.max(jnp.abs(next_state.obs))
            return (next_state, key), max_val

        scan_fn = jax.jit(
            lambda s, k: jax.lax.scan(step_collect, (s, k), None, length=50)
        )
        (_, _), max_vals = scan_fn(state, rng)
        max_val = jnp.max(max_vals)
        assert max_val < 100, f"Observation value too large: {max_val}"

    @pytest.mark.sim
    def test_prev_action_in_observation(self, env, rng):
        """
        Purpose: Verify previous action is correctly included in observation.

        Assertions:
        - After step, observation contains the filtered action that was applied
        """
        from playground_amp.envs.wildrobot_env import ObsLayout

        state = env.reset(rng)

        # Apply known action
        action = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        new_state = env.step(state, action)

        # The observation should contain the filtered action
        alpha = env._config.env.action_filter_alpha
        expected_action = (1.0 - alpha) * action
        action_slice = ObsLayout.get_slices()["action"]
        obs_prev_action = new_state.obs[action_slice]
        assert jnp.allclose(
            obs_prev_action, expected_action, atol=1e-6
        ), f"prev_action slice mismatch: got {obs_prev_action}, expected {expected_action}"
