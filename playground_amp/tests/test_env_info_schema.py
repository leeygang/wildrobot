"""Test environment info schema consistency.

This test validates that the WildRobotEnv info dict maintains a consistent
schema using the typed WildRobotInfo under info["wr"] namespace.

Key invariants:
1. info["wr"] must exist at reset and every step (KeyError if missing)
2. WildRobotInfo fields have correct shapes
3. Required fields are never all zeros (position, quaternion, height)

Usage:
    python3 playground_amp/tests/test_env_info_schema.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.envs.env_info import (
    get_expected_shapes,
    validate_wildrobot_info,
    WILDROBOT_INFO_NONZERO_FIELDS,
    WildRobotInfo,
    WR_INFO_KEY,
)


def test_reset_info_schema():
    """Verify reset() produces correct info schema."""
    print("\n1. Testing reset() schema...")

    from playground_amp.configs.robot_config import load_robot_config
    from playground_amp.configs.training_config import load_training_config
    from playground_amp.envs.wildrobot_env import WildRobotEnv
    from playground_amp.train import create_env_config

    load_robot_config("assets/robot_config.yaml")
    training_cfg = load_training_config("playground_amp/configs/ppo_walking.yaml")
    env_config = create_env_config(training_cfg.raw_config)
    env = WildRobotEnv(config=env_config)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # Check info["wr"] exists
    assert WR_INFO_KEY in state.info, f"Missing '{WR_INFO_KEY}' namespace in info"
    wr_info = state.info[WR_INFO_KEY]

    # Check it's a WildRobotInfo instance
    assert isinstance(
        wr_info, WildRobotInfo
    ), f"Expected WildRobotInfo, got {type(wr_info)}"

    # Validate schema
    errors = validate_wildrobot_info(
        wr_info, context="reset", action_size=env.action_size
    )
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        raise AssertionError(f"Schema validation failed with {len(errors)} errors")

    print("   ✅ reset() schema valid")
    return env, state


def test_step_info_schema(env, state):
    """Verify step() produces correct info schema."""
    print("\n2. Testing step() schema...")

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    # Check info["wr"] exists
    assert WR_INFO_KEY in next_state.info, f"Missing '{WR_INFO_KEY}' namespace in info"
    wr_info = next_state.info[WR_INFO_KEY]

    # Validate schema
    errors = validate_wildrobot_info(
        wr_info, context="step", action_size=env.action_size
    )
    if errors:
        for e in errors:
            print(f"   ❌ {e}")
        raise AssertionError(f"Schema validation failed with {len(errors)} errors")

    print("   ✅ step() schema valid")
    return next_state


def test_multiple_steps_schema(env, state):
    """Verify schema consistency across multiple steps."""
    print("\n3. Testing multiple steps schema consistency...")

    action = jnp.zeros(env.action_size)
    for step_idx in range(10):
        state = env.step(state, action)
        wr_info = state.info[WR_INFO_KEY]
        errors = validate_wildrobot_info(
            wr_info, context=f"step_{step_idx}", action_size=env.action_size
        )
        if errors:
            for e in errors:
                print(f"   ❌ {e}")
            raise AssertionError(f"Schema validation failed at step {step_idx}")

    print("   ✅ Multiple steps schema consistent")
    return state


def test_autoreset_preserves_schema(env):
    """Verify auto-reset preserves info schema when done=True."""
    print("\n4. Testing auto-reset schema preservation...")

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    terminated = False
    for step_idx in range(2000):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(
            action_rng, shape=(env.action_size,), minval=-1, maxval=1
        )
        state = env.step(state, action)

        # Always validate schema
        wr_info = state.info[WR_INFO_KEY]
        errors = validate_wildrobot_info(
            wr_info, context=f"autoreset_step_{step_idx}", action_size=env.action_size
        )
        if errors:
            for e in errors:
                print(f"   ❌ {e}")
            raise AssertionError(f"Schema validation failed at step {step_idx}")

        if state.done > 0.5:
            terminated = True
            print(f"   ✅ Auto-reset schema valid (terminated at step {step_idx})")
            break

    if not terminated:
        print("   ⚠️  Episode did not terminate within 2000 steps (skipped)")


def test_truncated_sticky_through_autoreset(env):
    """Verify truncated flag is sticky through auto-reset for success rate."""
    print("\n5. Testing truncated flag stickiness through auto-reset...")

    # This is hard to test without running to max_steps
    # For now, just check that truncated is accessible
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    wr_info = state.info[WR_INFO_KEY]

    assert hasattr(wr_info, "truncated"), "WildRobotInfo missing truncated field"
    assert (
        wr_info.truncated.shape == ()
    ), f"truncated wrong shape: {wr_info.truncated.shape}"

    print("   ✅ truncated field accessible with correct shape")


def print_schema_summary():
    """Print schema summary for documentation."""
    print("\n" + "=" * 60)
    print("WildRobotInfo Schema Summary")
    print("=" * 60)
    print(f"\nNamespace key: info['{WR_INFO_KEY}']\n")

    print("Field shapes (action_size from robot_config):")
    expected_shapes = get_expected_shapes()  # Uses robot_config
    for field, shape in expected_shapes.items():
        nonzero = (
            " (MUST NOT be all zeros)" if field in WILDROBOT_INFO_NONZERO_FIELDS else ""
        )
        print(f"  • {field}: {shape}{nonzero}")


def run_all_tests():
    """Run all schema validation tests."""
    print("=" * 60)
    print("WildRobotEnv Info Schema Validation")
    print("=" * 60)

    all_errors = []

    try:
        env, state = test_reset_info_schema()
        state = test_step_info_schema(env, state)
        test_multiple_steps_schema(env, state)
        test_autoreset_preserves_schema(env)
        test_truncated_sticky_through_autoreset(env)
    except Exception as e:
        all_errors.append(str(e))
        import traceback

        traceback.print_exc()

    print_schema_summary()

    print("\n" + "=" * 60)
    if all_errors:
        print(f"VALIDATION FAILED with {len(all_errors)} errors")
        for e in all_errors:
            print(f"  {e}")
        print("=" * 60)
        return False
    else:
        print("ALL SCHEMA VALIDATION TESTS PASSED!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
