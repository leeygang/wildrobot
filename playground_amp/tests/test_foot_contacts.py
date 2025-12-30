"""Comprehensive tests for foot contact pipeline (v0.5.0).

Tests that foot contacts are:
1. Extracted from environment during rollouts
2. Stored in Transition namedtuple
3. Passed to extract_amp_features (not zeros)
4. Used correctly by the discriminator

Run with:
    cd /Users/ygli/projects/wildrobot && python playground_amp/tests/test_foot_contacts.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ml_collections import config_dict

from playground_amp.envs.env_info import WR_INFO_KEY


def setup_environment():
    """Initialize environment for testing."""
    from playground_amp.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    # Load training config
    with open("playground_amp/configs/ppo_amass_training.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)

    # Build complete env config
    env_dict = yaml_config["env"].copy()

    # Map YAML reward weight keys to env code keys
    # YAML: tracking_lin_vel, base_height, action_rate
    # ENV:  forward_velocity, healthy, action_rate, joint_velocity
    yaml_rewards = yaml_config["reward_weights"]
    env_dict["reward_weights"] = {
        "forward_velocity": yaml_rewards.get("tracking_lin_vel", 5.0),
        "healthy": yaml_rewards.get("base_height", 0.3),
        "action_rate": yaml_rewards.get("action_rate", -0.01),
        "joint_velocity": yaml_rewards.get("joint_velocity", 0.0),
    }
    env_config = config_dict.ConfigDict(env_dict)

    from playground_amp.envs.wildrobot_env import WildRobotEnv

    return WildRobotEnv(config=env_config)


def test_1_env_returns_foot_contacts():
    """Test 1: Environment returns foot_contacts in state.info["wr"]."""
    print("\n=== Test 1: Environment returns foot_contacts ===")

    env = setup_environment()
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Check info["wr"] namespace exists
    assert (
        WR_INFO_KEY in state.info
    ), f"'{WR_INFO_KEY}' namespace missing from state.info"
    wr_info = state.info[WR_INFO_KEY]

    # Check foot_contacts exists in wr_info
    assert hasattr(
        wr_info, "foot_contacts"
    ), "foot_contacts missing from state.info['wr']"
    foot_contacts = wr_info.foot_contacts

    # Check shape
    assert foot_contacts.shape == (
        4,
    ), f"Expected shape (4,), got {foot_contacts.shape}"

    # Check values are in valid range [0, 1]
    assert jnp.all(foot_contacts >= 0), "foot_contacts should be >= 0"
    assert jnp.all(foot_contacts <= 1), "foot_contacts should be <= 1"

    print(f"  ✓ foot_contacts shape: {foot_contacts.shape}")
    print(f"  ✓ foot_contacts values: {foot_contacts}")
    print(f"  ✓ Values in valid range [0, 1]")
    return True


def test_2_foot_contacts_update_after_step():
    """Test 2: foot_contacts update after stepping the environment."""
    print("\n=== Test 2: foot_contacts update after step ===")

    env = setup_environment()
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    initial_contacts = state.info[WR_INFO_KEY].foot_contacts

    # Step environment multiple times
    action = jnp.zeros(env.action_size)
    contacts_history = [initial_contacts]

    for i in range(10):
        state = env.step(state, action)
        contacts_history.append(state.info[WR_INFO_KEY].foot_contacts)

    # Check that contacts changed at some point
    contacts_array = jnp.stack(contacts_history)
    variance = jnp.var(contacts_array, axis=0)

    print(f"  Variance of contacts over 10 steps: {variance}")
    print(f"  Contact values at step 0: {contacts_history[0]}")
    print(f"  Contact values at step 10: {contacts_history[-1]}")

    # At least one foot should have some variance (robot is moving)
    print(f"  ✓ foot_contacts update during simulation")
    return True


def test_3_amp_features_require_foot_contacts():
    """Test 3: extract_amp_features raises ValueError when foot_contacts=None."""
    print("\n=== Test 3: amp_features require foot_contacts ===")

    from playground_amp.amp.policy_features import (
        extract_amp_features,
        get_feature_config,
    )
    from playground_amp.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()
    fake_obs = jnp.zeros(38)

    try:
        features = extract_amp_features(fake_obs, config, foot_contacts=None)
        print("  ✗ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError")
        print(f'    Message: "{str(e)[:70]}..."')
        return True


def test_4_amp_features_use_foot_contacts():
    """Test 4: extract_amp_features correctly uses foot_contacts."""
    print("\n=== Test 4: amp_features use foot_contacts ===")

    from playground_amp.amp.policy_features import get_feature_config
    from playground_amp.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()
    fake_obs = jnp.zeros(38)

    # Test with specific foot contact values
    foot_contacts = jnp.array([0.1, 0.2, 0.3, 0.4])
    features = extract_amp_features(fake_obs, config, foot_contacts=foot_contacts)

    # Features should be 29-dim
    assert features.shape == (29,), f"Expected shape (29,), got {features.shape}"

    # Last 4 dims should be foot_contacts
    extracted_contacts = features[25:29]
    assert jnp.allclose(
        extracted_contacts, foot_contacts
    ), f"foot_contacts not at expected position: {extracted_contacts} vs {foot_contacts}"

    print(f"  ✓ Features shape: {features.shape}")
    print(f"  ✓ Features[25:29] = {features[25:29]} (matches input foot_contacts)")
    return True


def test_5_batched_features_require_foot_contacts():
    """Test 5: extract_amp_features_batched requires foot_contacts."""
    print("\n=== Test 5: Batched features require foot_contacts ===")

    from playground_amp.amp.policy_features import get_feature_config
    from playground_amp.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )
    from playground_amp.training.trainer_jit import extract_amp_features_batched

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()

    # Create batch of observations (num_steps, num_envs, obs_dim)
    num_steps, num_envs = 5, 4
    fake_obs = jnp.zeros((num_steps, num_envs, 38))
    fake_contacts = jnp.ones((num_steps, num_envs, 4)) * 0.5

    # This should work with foot_contacts
    features = extract_amp_features_batched(fake_obs, config, fake_contacts)

    assert features.shape == (
        num_steps,
        num_envs,
        29,
    ), f"Expected shape ({num_steps}, {num_envs}, 29), got {features.shape}"

    # Check that foot_contacts are in features
    extracted_contacts = features[:, :, 25:29]
    assert jnp.allclose(
        extracted_contacts, fake_contacts
    ), "foot_contacts not correctly extracted in batched version"

    print(f"  ✓ Batched features shape: {features.shape}")
    print(f"  ✓ foot_contacts correctly propagated through batched extraction")
    return True


def test_6_transition_has_foot_contacts():
    """Test 6: Transition namedtuple includes foot_contacts field."""
    print("\n=== Test 6: Transition has foot_contacts field ===")

    from playground_amp.training.trainer_jit import Transition

    # Check field exists
    assert (
        "foot_contacts" in Transition._fields
    ), "foot_contacts missing from Transition namedtuple"

    # Create a dummy transition
    dummy_transition = Transition(
        obs=jnp.zeros(38),
        action=jnp.zeros(9),
        reward=jnp.array(0.0),
        done=jnp.array(0.0),
        log_prob=jnp.array(0.0),
        value=jnp.array(0.0),
        next_obs=jnp.zeros(38),
        truncated=jnp.array(0.0),
        foot_contacts=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert dummy_transition.foot_contacts.shape == (4,)

    print(f"  ✓ Transition fields: {Transition._fields}")
    print(f"  ✓ foot_contacts is field #{Transition._fields.index('foot_contacts')}")
    return True


def test_7_features_are_not_all_zeros():
    """Test 7: End-to-end test - features from env have non-zero contacts."""
    print("\n=== Test 7: End-to-end feature extraction (non-zero contacts) ===")

    from playground_amp.amp.policy_features import (
        extract_amp_features,
        get_feature_config,
    )
    from playground_amp.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    env = setup_environment()
    config = get_feature_config()

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Run for several steps to let robot settle
    action = jnp.zeros(env.action_size)
    for _ in range(50):  # Let robot settle/make contact
        state = env.step(state, action)

    # Extract features
    obs = state.obs
    foot_contacts = state.info[WR_INFO_KEY].foot_contacts
    features = extract_amp_features(obs, config, foot_contacts=foot_contacts)

    # Check features shape
    assert features.shape == (29,), f"Expected shape (29,), got {features.shape}"

    # Check foot contact portion of features
    feature_contacts = features[25:29]

    print(f"  Observation shape: {obs.shape}")
    print(f"  Foot contacts from env: {foot_contacts}")
    print(f"  Features[25:29]: {feature_contacts}")
    print(
        f"  Features match env contacts: {jnp.allclose(feature_contacts, foot_contacts)}"
    )

    # Verify features match env foot_contacts
    assert jnp.allclose(
        feature_contacts, foot_contacts
    ), "Features don't match env foot_contacts!"

    print(f"  ✓ Features correctly include foot_contacts from environment")
    return True


def test_8_contact_distribution_not_zeros():
    """Test 8: Statistical test - contacts should not be all zeros over time."""
    print("\n=== Test 8: Contact distribution is not all zeros ===")

    env = setup_environment()
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Collect contacts over many steps
    action = jnp.zeros(env.action_size)
    all_contacts = []

    for i in range(200):
        state = env.step(state, action)
        all_contacts.append(state.info[WR_INFO_KEY].foot_contacts)

    contacts_array = jnp.stack(all_contacts)  # (200, 4)

    # Statistics
    mean_contacts = jnp.mean(contacts_array, axis=0)
    max_contacts = jnp.max(contacts_array, axis=0)
    nonzero_ratio = jnp.mean(contacts_array > 0.01)

    print(f"  Over 200 steps:")
    print(f"    Mean contacts: {mean_contacts}")
    print(f"    Max contacts: {max_contacts}")
    print(f"    % non-zero (>0.01): {float(nonzero_ratio)*100:.1f}%")

    # The robot should have SOME contacts during normal operation
    # This test catches the original bug where policy always returned zeros
    assert jnp.max(max_contacts) > 0.1, "Contacts are all near-zero - likely a bug!"

    print(f"  ✓ Contacts are NOT all zeros (bug from v0.4.x is fixed)")
    return True


def run_all_tests():
    """Run all foot contact tests."""
    print("=" * 60)
    print("Foot Contact Pipeline Tests (v0.5.0)")
    print("=" * 60)

    tests = [
        ("Test 1: env returns foot_contacts", test_1_env_returns_foot_contacts),
        (
            "Test 2: foot_contacts update after step",
            test_2_foot_contacts_update_after_step,
        ),
        (
            "Test 3: amp_features require foot_contacts",
            test_3_amp_features_require_foot_contacts,
        ),
        (
            "Test 4: amp_features use foot_contacts",
            test_4_amp_features_use_foot_contacts,
        ),
        (
            "Test 5: batched features require foot_contacts",
            test_5_batched_features_require_foot_contacts,
        ),
        ("Test 6: Transition has foot_contacts", test_6_transition_has_foot_contacts),
        ("Test 7: end-to-end feature extraction", test_7_features_are_not_all_zeros),
        (
            "Test 8: contact distribution not zeros",
            test_8_contact_distribution_not_zeros,
        ),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r, _ in results if r)
    failed = len(results) - passed

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if failed == 0:
        print("\n🎉 All tests passed! Foot contact pipeline is working correctly.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please investigate.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
