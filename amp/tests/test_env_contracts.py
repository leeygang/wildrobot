#!/usr/bin/env python3
"""Test JAX pytree contracts for WildRobot environment.

Critical tests that MUST PASS before training to prevent JAX scan errors.

Usage:
    python -m pytest tests/test_env_contracts.py -v
    python tests/test_env_contracts.py  # Direct run
"""

import sys
from pathlib import Path

# Add both amp directory and project root to path
amp_dir = Path(__file__).parent.parent
project_root = amp_dir.parent
sys.path.insert(0, str(amp_dir))
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jp
import pytest
from ml_collections import config_dict

from walk_env import WildRobotWalkEnv


def create_test_config(forward_velocity_bonus: float = 50.0) -> config_dict.ConfigDict:
    """Create test configuration with all reward weights.

    Centralized config builder to avoid duplicate code across test classes.

    Args:
        forward_velocity_bonus: Weight for forward velocity bonus (default 50.0)

    Returns:
        ConfigDict with all environment and reward settings
    """
    config = config_dict.ConfigDict()

    # Environment settings
    config.terrain = "flat"
    config.ctrl_dt = 0.02
    config.sim_dt = 0.002
    config.velocity_command_mode = "range"
    config.min_velocity = 0.5
    config.max_velocity = 1.0
    config.use_action_filter = True
    config.action_filter_alpha = 0.7
    config.use_phase_signal = True
    config.phase_period = 40
    config.num_phase_clocks = 2
    config.min_height = 0.2
    config.max_height = 0.7

    # Reward weights (Option 3 parameters included)
    config.reward_weights = config_dict.ConfigDict()
    config.reward_weights.tracking_sigma = 0.25
    config.reward_weights.tracking_steepness = 3.0  # OPTION 3 FIX #1
    config.reward_weights.velocity_threshold = 0.15  # OPTION 3 v2: REDUCED from 0.3
    config.reward_weights.tracking_exp_xy = 50.0
    config.reward_weights.tracking_lin_xy = 15.0
    config.reward_weights.tracking_exp_yaw = 5.0
    config.reward_weights.tracking_lin_yaw = 1.5
    config.reward_weights.forward_velocity_bonus = forward_velocity_bonus
    config.reward_weights.velocity_threshold_penalty = 25.0  # OPTION 3 v2: REDUCED from 50.0
    config.reward_weights.z_velocity = 0.1
    config.reward_weights.roll_pitch_velocity = 0.01
    config.reward_weights.roll_pitch_position = 0.05
    config.reward_weights.nominal_joint_position = 0.0
    config.reward_weights.joint_position_limit = 2.5
    config.reward_weights.joint_velocity = 0.0
    config.reward_weights.joint_acceleration = 1e-8
    config.reward_weights.action_rate = 0.001
    config.reward_weights.joint_torque = 7e-8
    config.reward_weights.mechanical_power = 7e-6
    config.reward_weights.foot_contact = 0.0
    config.reward_weights.foot_sliding = 0.0
    config.reward_weights.foot_air_time = 0.0
    config.reward_weights.existential = 1.0
    # Optional gating parameters for diagnostics
    config.reward_weights.tracking_gate_velocity = 0.20
    config.reward_weights.tracking_gate_scale = 0.20

    return config


def create_decay_test_config() -> config_dict.ConfigDict:
    """Create a config that enables velocity threshold penalty decay to test scheduling.

    Returns:
        ConfigDict with decay schedule parameters set so scale changes early.
    """
    cfg = create_test_config(forward_velocity_bonus=10.0)
    # Enable decay schedule immediately so tests trigger path that uses global_step.
    cfg.reward_weights.velocity_threshold_penalty_decay_start = 0
    cfg.reward_weights.velocity_threshold_penalty_decay_steps = 5  # Fast decay for test
    cfg.reward_weights.velocity_threshold_penalty_min_scale = 0.2
    return cfg

    return config


class TestJAXPytreeContracts:
    """Test JAX pytree structure consistency (critical for jax.lax.scan)."""

    @pytest.fixture
    def env_config(self):
        """Create test config."""
        return create_test_config(forward_velocity_bonus=50.0)

    @pytest.fixture
    def env(self, env_config):
        """Create environment instance."""
        return WildRobotWalkEnv(task="wildrobot_flat", config=env_config)

    def test_reset_step_metrics_structure_match(self, env):
        """CRITICAL: reset() and step() must return identical metrics structure.

        JAX scan requires the exact same pytree structure (dict keys and nesting).
        Mismatch causes: TypeError: scan body function carry input and output
        must have the same pytree structure.
        """
        rng = jax.random.PRNGKey(0)

        # Reset environment
        state_reset = env.reset(rng)

        # Take one step
        action = jp.zeros(env.action_size)
        state_step = env.step(state_reset, action)

        # Extract metrics dictionaries
        metrics_reset = state_reset.metrics
        metrics_step = state_step.metrics

        # Check keys match exactly
        keys_reset = set(metrics_reset.keys())
        keys_step = set(metrics_step.keys())

        missing_in_step = keys_reset - keys_step
        extra_in_step = keys_step - keys_reset

        assert not missing_in_step, (
            f"Keys in reset() but missing in step(): {missing_in_step}\n"
            "This will cause JAX scan to fail!"
        )

        assert not extra_in_step, (
            f"Keys in step() but missing in reset(): {extra_in_step}\n"
            "This will cause JAX scan to fail!"
        )

        # Verify pytree structure matches (recursively for nested dicts)
        def check_pytree_structure(tree1, tree2, path=""):
            """Recursively check pytree structure matches."""
            # Get pytree structure
            flat1, treedef1 = jax.tree_util.tree_flatten(tree1)
            flat2, treedef2 = jax.tree_util.tree_flatten(tree2)

            assert treedef1 == treedef2, (
                f"Pytree structure mismatch at {path}:\n"
                f"reset(): {treedef1}\n"
                f"step():  {treedef2}"
            )

        check_pytree_structure(metrics_reset, metrics_step, path="metrics")

        print(f"✅ PASS: metrics structure match ({len(keys_reset)} keys)")

    def test_reset_step_info_structure_match(self, env):
        """CRITICAL: reset() and step() info dicts must match structure."""
        rng = jax.random.PRNGKey(0)

        state_reset = env.reset(rng)
        action = jp.zeros(env.action_size)
        state_step = env.step(state_reset, action)

        info_reset = state_reset.info
        info_step = state_step.info

        keys_reset = set(info_reset.keys())
        keys_step = set(info_step.keys())

        missing_in_step = keys_reset - keys_step
        extra_in_step = keys_step - keys_reset

        assert not missing_in_step, (
            f"Info keys in reset() but missing in step(): {missing_in_step}"
        )

        assert not extra_in_step, (
            f"Info keys in step() but missing in reset(): {extra_in_step}"
        )

        print(f"✅ PASS: info structure match ({len(keys_reset)} keys)")

    def test_all_reward_components_initialized(self, env):
        """Verify all reward components from config are in metrics."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Get reward weights from config
        reward_weights = env._reward_weights

        # Parameters that are NOT reward components (skip these)
        parameters = {"tracking_sigma", "tracking_steepness", "velocity_threshold"}
        # Add gating parameters (diagnostic only, not reward components)
        parameters.update({"tracking_gate_velocity", "tracking_gate_scale"})

        # Check each component is initialized in metrics
        for component_name in reward_weights.keys():
            if component_name in parameters:
                continue  # Skip parameters (not reward components)

            metric_key = f"reward/{component_name}"
            assert metric_key in state.metrics, (
                f"Reward component '{component_name}' not initialized in reset()!\n"
                f"This will cause JAX scan error when step() adds it."
            )

        num_components = len([k for k in reward_weights.keys() if k not in parameters])
        print(f"✅ PASS: all {num_components} reward components initialized")

    def test_forward_velocity_bonus_in_metrics(self, env):
        """Specifically test forward_velocity_bonus is initialized.

        This was the bug that caused the initial failure.
        """
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        assert "reward/forward_velocity_bonus" in state.metrics, (
            "forward_velocity_bonus not initialized in reset()!\n"
            "This is the exact bug we fixed - it must be present."
        )

        # Also verify it's computed in step()
        action = jp.zeros(env.action_size)
        state_step = env.step(state, action)
        assert "reward/forward_velocity_bonus" in state_step.metrics

        print("✅ PASS: forward_velocity_bonus present in both reset() and step()")


class TestEnvironmentBasics:
    """Test basic environment functionality."""

    @pytest.fixture
    def env_config(self):
        """Create test config (reuse centralized builder)."""
        return create_test_config(forward_velocity_bonus=10.0)

    @pytest.fixture
    def env(self, env_config):
        """Create environment instance."""
        return WildRobotWalkEnv(task="wildrobot_flat", config=env_config)

    def test_reset_returns_valid_state(self, env):
        """Test reset() returns a valid state."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Check state has required attributes
        assert hasattr(state, "data")
        assert hasattr(state, "obs")
        assert hasattr(state, "reward")
        assert hasattr(state, "done")
        assert hasattr(state, "metrics")
        assert hasattr(state, "info")

        # Check observation size
        assert state.obs.shape == (env.observation_size,)

        # Check initial conditions
        assert state.reward == 0.0
        assert state.done == 0.0
        assert state.metrics["success"] == 1.0

        print(f"✅ PASS: reset() returns valid state (obs_size={env.observation_size})")

    def test_step_produces_valid_outputs(self, env):
        """Test step() produces valid outputs."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jp.zeros(env.action_size)
        state_next = env.step(state, action)

        # Check outputs
        assert state_next.obs.shape == (env.observation_size,)
        assert isinstance(float(state_next.reward), float)
        assert isinstance(float(state_next.done), float)
        assert 0.0 <= float(state_next.done) <= 1.0

        print("✅ PASS: step() produces valid outputs")

    def test_forward_velocity_bonus_computation(self, env):
        """Test forward_velocity_bonus is computed correctly."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Take a step (robot will likely fall or barely move)
        action = jp.zeros(env.action_size)
        state_next = env.step(state, action)

        # Get forward velocity and bonus
        forward_vel = state_next.metrics["forward_velocity"]
        bonus = state_next.metrics["reward/forward_velocity_bonus"]

        # Bonus should be max(0, velocity)
        expected_bonus = jp.maximum(0.0, forward_vel)

        # Allow small numerical errors
        assert jp.abs(bonus - expected_bonus) < 1e-6, (
            f"forward_velocity_bonus mismatch!\n"
            f"forward_vel: {forward_vel}\n"
            f"bonus: {bonus}\n"
            f"expected: {expected_bonus}"
        )

        print(f"✅ PASS: forward_velocity_bonus computed correctly "
              f"(vel={forward_vel:.3f}, bonus={bonus:.3f})")

    def test_multiple_steps_maintain_structure(self, env):
        """Test that multiple steps maintain pytree structure."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Take 10 steps
        for i in range(10):
            action = jp.zeros(env.action_size)
            state = env.step(state, action)

            # Verify structure hasn't changed
            assert "reward/forward_velocity_bonus" in state.metrics
            assert "forward_velocity" in state.metrics
            assert "step_count" in state.info

        print("✅ PASS: structure maintained over 10 steps")


class TestRewardComponents:
    """Test individual reward components."""

    @pytest.fixture
    def env_config(self):
        """Create test config (reuse centralized builder)."""
        return create_test_config(forward_velocity_bonus=10.0)

    @pytest.fixture
    def env(self, env_config):
        """Create environment instance."""
        return WildRobotWalkEnv(task="wildrobot_flat", config=env_config)

    def test_existential_penalty_applied(self, env):
        """Test existential penalty is -5.0 as expected."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        action = jp.zeros(env.action_size)
        state_next = env.step(state, action)

        existential = state_next.metrics["reward/existential"]
        assert existential == -5.0, (
            f"Existential penalty should be -5.0, got {existential}"
        )

        print("✅ PASS: existential penalty is -5.0")

    def test_forward_velocity_bonus_positive_for_forward_motion(self, env):
        """Test bonus is positive when moving forward."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Create a scenario where robot has positive forward velocity
        # (In practice this is hard without actually training, but we can check the formula)
        action = jp.zeros(env.action_size)
        state_next = env.step(state, action)

        forward_vel = state_next.metrics["forward_velocity"]
        bonus = state_next.metrics["reward/forward_velocity_bonus"]

        if forward_vel > 0:
            assert bonus > 0, "Bonus should be positive for forward motion"
            assert bonus == forward_vel, "Bonus should equal velocity"
        else:
            assert bonus == 0.0, "Bonus should be zero for non-forward motion"

        print(f"✅ PASS: bonus logic correct (vel={forward_vel:.3f}, bonus={bonus:.3f})")

    def test_gating_diagnostics_present(self, env):
        """Test that gating diagnostic metrics appear when gating configured."""
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)

        # Metrics should include diagnostic placeholders
        assert "reward/tracking_gate_active" in state.metrics, "tracking_gate_active metric missing in reset()"
        assert "reward/velocity_threshold_scale" in state.metrics, "velocity_threshold_scale metric missing in reset()"

        # Take a step and verify presence maintained
        action = jp.zeros(env.action_size)
        state_step = env.step(state, action)
        assert "reward/tracking_gate_active" in state_step.metrics
        assert "reward/velocity_threshold_scale" in state_step.metrics

        print("✅ PASS: gating diagnostics present in reset() and step()")

    def test_velocity_threshold_decay_schedule(self):
        """Test velocity threshold penalty decay scheduling scales down over steps.

        Ensures global_step-dependent decay path executes (previous bug: NameError when global_step not passed).
        """
        rng = jax.random.PRNGKey(0)
        cfg = create_decay_test_config()
        env = WildRobotWalkEnv(task="wildrobot_flat", config=cfg)

        # Reset environment
        state = env.reset(rng)
        scale_initial = float(state.metrics["reward/velocity_threshold_scale"])  # Should be 1.0 at step 0

        # Step multiple times to advance global_step and trigger decay
        for _ in range(6):  # > decay_steps to reach min scale
            action = jp.zeros(env.action_size)
            state = env.step(state, action)

        scale_final = float(state.metrics["reward/velocity_threshold_scale"])

        assert scale_initial == 1.0, f"Initial velocity_threshold_scale should be 1.0, got {scale_initial}" 
        assert scale_final < scale_initial, (
            f"Decay did not reduce scale: initial={scale_initial}, final={scale_final}"
        )
        assert scale_final <= 0.21, (
            f"Final scale should approach configured min (~0.2), got {scale_final}"
        )
        print(f"✅ PASS: decay schedule reduces velocity_threshold_scale (initial={scale_initial}, final={scale_final})")

    def test_global_step_dtype_consistent(self):
        """Ensure global_step metric dtype remains float32 across steps (scan carry stability)."""
        rng = jax.random.PRNGKey(1)
        cfg = create_test_config(forward_velocity_bonus=5.0)
        env = WildRobotWalkEnv(task="wildrobot_flat", config=cfg)
        state = env.reset(rng)
        dtype_initial = state.metrics["global_step"].dtype
        for _ in range(3):
            action = jp.zeros(env.action_size)
            state = env.step(state, action)
            assert state.metrics["global_step"].dtype == dtype_initial, (
                f"global_step dtype changed from {dtype_initial} to {state.metrics['global_step'].dtype}"
            )
        print(f"✅ PASS: global_step dtype stable ({dtype_initial}) over multiple steps")


def run_tests():
    """Run tests without pytest (for direct execution)."""
    print("\n" + "="*80)
    print("WILDROBOT AMP ENVIRONMENT CONTRACT TESTS")
    print("="*80 + "\n")

    # Create config using centralized builder
    config = create_test_config(forward_velocity_bonus=10.0)

    # Create environment
    print("Creating environment...")
    env = WildRobotWalkEnv(task="wildrobot_flat", config=config)
    print(f"Environment created: obs_size={env.observation_size}, action_size={env.action_size}\n")

    # Run tests
    test_suite = TestJAXPytreeContracts()
    basic_tests = TestEnvironmentBasics()
    reward_tests = TestRewardComponents()

    tests = [
        ("Metrics structure match", lambda: test_suite.test_reset_step_metrics_structure_match(env)),
        ("Info structure match", lambda: test_suite.test_reset_step_info_structure_match(env)),
        ("All reward components initialized", lambda: test_suite.test_all_reward_components_initialized(env)),
        ("forward_velocity_bonus in metrics", lambda: test_suite.test_forward_velocity_bonus_in_metrics(env)),
        ("Reset returns valid state", lambda: basic_tests.test_reset_returns_valid_state(env)),
        ("Step produces valid outputs", lambda: basic_tests.test_step_produces_valid_outputs(env)),
        ("forward_velocity_bonus computation", lambda: basic_tests.test_forward_velocity_bonus_computation(env)),
        ("Multiple steps maintain structure", lambda: basic_tests.test_multiple_steps_maintain_structure(env)),
        ("Existential penalty applied", lambda: reward_tests.test_existential_penalty_applied(env)),
        ("forward_velocity_bonus logic", lambda: reward_tests.test_forward_velocity_bonus_positive_for_forward_motion(env)),
        ("Gating diagnostics present", lambda: reward_tests.test_gating_diagnostics_present(env)),
        ("Velocity threshold decay schedule", lambda: reward_tests.test_velocity_threshold_decay_schedule()),
        ("Global step dtype consistent", lambda: reward_tests.test_global_step_dtype_consistent()),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"Running: {name}...", end=" ")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   {str(e)}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {name}")
            print(f"   {type(e).__name__}: {str(e)}\n")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80)

    if failed > 0:
        print("\n❌ TESTS FAILED - DO NOT START TRAINING!")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED - SAFE TO START TRAINING!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
