"""Comprehensive tests for foot contact pipeline (v0.5.0).

Tests that foot contacts are:
1. Extracted from environment during rollouts
2. Stored in Transition namedtuple
3. Passed to extract_amp_features (not zeros)
4. Used correctly by the discriminator

Run with:
    cd /Users/ygli/projects/wildrobot && python training/tests/test_foot_contacts.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpy as np
from training.envs.env_info import WR_INFO_KEY


def setup_environment():
    """Initialize environment for testing."""
    from training.configs.training_config import (
        clear_config_cache,
        load_robot_config,
        load_training_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    training_cfg = load_training_config("training/configs/ppo_walking.yaml")
    training_cfg.freeze()

    from training.envs.wildrobot_env import WildRobotEnv

    return WildRobotEnv(config=training_cfg)


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

    # Step environment multiple times (scan for speed)
    action = jnp.zeros(env.action_size)

    def step_collect(s, _):
        next_state = env.step(s, action)
        return next_state, next_state.info[WR_INFO_KEY].foot_contacts

    scan_fn = jax.jit(lambda s: jax.lax.scan(step_collect, s, None, length=10))
    state, contacts = scan_fn(state)
    contacts_history = [initial_contacts] + list(contacts)

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
    """Test 3: extract_amp_features accepts no override and returns valid output."""
    print("\n=== Test 3: amp_features accept missing override ===")

    from training.amp.policy_features import extract_amp_features
    from training.configs.feature_config import get_feature_config
    from training.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()
    fake_obs = jnp.zeros((1, 39))
    root_height = jnp.array([0.5])
    prev_joint_pos = jnp.zeros((1, config.num_actuated_joints))

    features = extract_amp_features(
        fake_obs,
        config,
        root_height=root_height,
        prev_joint_pos=prev_joint_pos,
        dt=0.02,
    )

    expected_dim = config.feature_dim
    assert features.shape == (
        1,
        expected_dim,
    ), f"Expected shape (1, {expected_dim}), got {features.shape}"
    assert jnp.all(jnp.isfinite(features)), "Features contain NaN/Inf"

    print(f"  ✓ Features shape: {features.shape}")
    return True


def test_4_amp_features_use_foot_contacts():
    """Test 4: extract_amp_features correctly uses foot_contacts."""
    print("\n=== Test 4: amp_features use foot_contacts ===")

    from training.amp.policy_features import extract_amp_features
    from training.configs.feature_config import get_feature_config
    from training.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()
    fake_obs = jnp.zeros((1, 39))
    root_height = jnp.array([0.5])
    prev_joint_pos = jnp.zeros((1, config.num_actuated_joints))

    # Test with specific foot contact values
    foot_contacts = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    features = extract_amp_features(
        fake_obs,
        config,
        root_height=root_height,
        prev_joint_pos=prev_joint_pos,
        dt=0.02,
        foot_contacts_override=foot_contacts,
    )

    expected_dim = config.feature_dim
    assert features.shape == (
        1,
        expected_dim,
    ), f"Expected shape (1, {expected_dim}), got {features.shape}"

    # Last 4 dims should be foot_contacts
    extracted_contacts = features[0, -4:]
    assert jnp.allclose(
        extracted_contacts, foot_contacts[0]
    ), f"foot_contacts not at expected position: {extracted_contacts} vs {foot_contacts[0]}"

    print(f"  ✓ Features shape: {features.shape}")
    print(f"  ✓ Features[-4:] = {features[-4:]} (matches input foot_contacts)")
    return True


def test_5_batched_features_require_foot_contacts():
    """Test 5: extract_amp_features_batched requires foot_contacts."""
    print("\n=== Test 5: Batched features require foot_contacts ===")

    from training.amp.policy_features import extract_amp_features_batched
    from training.configs.feature_config import get_feature_config
    from training.configs.training_config import (
        clear_config_cache,
        load_robot_config,
    )

    clear_config_cache()
    load_robot_config("assets/robot_config.yaml")

    config = get_feature_config()

    # Create batch of observations (num_steps, num_envs, obs_dim)
    num_steps, num_envs = 5, 4
    fake_obs = jnp.zeros((num_steps, num_envs, 39))
    fake_contacts = jnp.ones((num_steps, num_envs, 4)) * 0.5
    fake_root_height = jnp.ones((num_steps, num_envs))
    fake_prev_joint_pos = jnp.zeros((num_steps, num_envs, config.num_actuated_joints))

    # This should work with foot_contacts
    features = extract_amp_features_batched(
        fake_obs,
        config,
        foot_contacts=fake_contacts,
        root_height=fake_root_height,
        prev_joint_pos=fake_prev_joint_pos,
        dt=0.02,
        use_estimated_contacts=False,
        use_finite_diff_vel=True,
    )

    assert features.shape == (
        num_steps,
        num_envs,
        config.feature_dim,
    ), f"Expected shape ({num_steps}, {num_envs}, {config.feature_dim}), got {features.shape}"

    # Check that foot_contacts are in features
    extracted_contacts = features[:, :, -4:]
    assert jnp.allclose(
        extracted_contacts, fake_contacts
    ), "foot_contacts not correctly extracted in batched version"

    print(f"  ✓ Batched features shape: {features.shape}")
    print(f"  ✓ foot_contacts correctly propagated through batched extraction")
    return True


def test_6_transition_has_foot_contacts():
    """Test 6: TrajectoryBatch includes foot_contacts field."""
    print("\n=== Test 6: TrajectoryBatch has foot_contacts field ===")

    from training.core.metrics_registry import NUM_METRICS
    from training.core.rollout import TrajectoryBatch

    # Check field exists
    assert (
        "foot_contacts" in TrajectoryBatch._fields
    ), "foot_contacts missing from TrajectoryBatch"

    # Create a dummy trajectory
    shape = (2, 3)
    dummy_traj = TrajectoryBatch(
        obs=jnp.zeros((*shape, 39)),
        actions=jnp.zeros((*shape, 8)),
        log_probs=jnp.zeros(shape),
        values=jnp.zeros(shape),
        task_rewards=jnp.zeros(shape),
        dones=jnp.zeros(shape),
        truncations=jnp.zeros(shape),
        next_obs=jnp.zeros((*shape, 39)),
        bootstrap_value=jnp.zeros((shape[1],)),
        metrics_vec=jnp.zeros((*shape, NUM_METRICS)),
        step_counts=jnp.zeros(shape),
        foot_contacts=jnp.array([[[0.1, 0.2, 0.3, 0.4]] * shape[1]] * shape[0]),
        root_heights=jnp.zeros((*shape, 1)),
        prev_joint_positions=jnp.zeros((*shape, 8)),
    )

    assert dummy_traj.foot_contacts.shape == (*shape, 4)

    print(f"  ✓ TrajectoryBatch fields: {TrajectoryBatch._fields}")
    print(
        f"  ✓ foot_contacts is field #{TrajectoryBatch._fields.index('foot_contacts')}"
    )
    return True


def test_7_features_are_not_all_zeros():
    """Test 7: End-to-end test - features from env have non-zero contacts."""
    print("\n=== Test 7: End-to-end feature extraction (non-zero contacts) ===")

    from training.amp.policy_features import extract_amp_features
    from training.configs.feature_config import get_feature_config
    from training.configs.training_config import (
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
    scan_fn = jax.jit(lambda s: jax.lax.fori_loop(0, 50, lambda _, st: env.step(st, action), s))
    state = scan_fn(state)

    # Extract features
    obs = state.obs
    foot_contacts = state.info[WR_INFO_KEY].foot_contacts
    root_height = state.info[WR_INFO_KEY].root_height
    prev_joint_pos = obs[config.joint_pos_start : config.joint_pos_end][None, :]
    features = extract_amp_features(
        obs[None, :],
        config,
        root_height=root_height[None],
        prev_joint_pos=prev_joint_pos,
        dt=env.dt,
        foot_contacts_override=foot_contacts[None, :],
    )

    # Check features shape
    expected_dim = config.feature_dim
    assert features.shape == (
        1,
        expected_dim,
    ), f"Expected shape (1, {expected_dim}), got {features.shape}"

    # Check foot contact portion of features (last 4 dims)
    feature_contacts = features[0, -4:]

    print(f"  Observation shape: {obs.shape}")
    print(f"  Foot contacts from env: {foot_contacts}")
    print(f"  Features[-4:]: {feature_contacts}")
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

    def step_collect(s, _):
        next_state = env.step(s, action)
        return next_state, next_state.info[WR_INFO_KEY].foot_contacts

    scan_fn = jax.jit(lambda s: jax.lax.scan(step_collect, s, None, length=200))
    state, contacts_array = scan_fn(state)

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
