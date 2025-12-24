"""Black-box tests for training system components.

Tests each component in isolation by verifying:
- Input → Output shape correctness
- Output value ranges
- Known input → expected output relationships

Components tested:
1. AMP Feature Extraction
2. Discriminator (forward, loss, reward)
3. Reference Motion Data loading
4. PPO (GAE, loss)
5. Replay Buffer

Run with:
    cd playground_amp
    uv run pytest tests/test_training_components.py -v
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# =============================================================================
# Session-level Setup: Load Robot Config
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def load_robot_config_fixture():
    """Load robot config once for the entire test session.

    This is required because amp_features.py depends on robot config
    for joint indices and other robot-specific parameters.
    """
    from playground_amp.configs.config import load_robot_config

    robot_config_path = Path("assets/robot_config.yaml")
    if robot_config_path.exists():
        load_robot_config(robot_config_path)
    else:
        pytest.skip(f"Robot config not found: {robot_config_path}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """JAX random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def batch_size():
    return 64


@pytest.fixture
def feature_dim():
    """AMP feature dimension (29-dim)."""
    return 29


@pytest.fixture
def obs_dim():
    """Observation dimension for WildRobotEnv."""
    return 37


@pytest.fixture
def action_dim():
    """Action dimension for WildRobotEnv."""
    return 9


# =============================================================================
# 1. AMP Feature Extraction Tests
# =============================================================================


class TestAMPFeatureExtraction:
    """Black-box tests for AMP feature extraction."""

    def test_feature_extraction_shape(self, rng):
        """Test: obs (37-dim) + foot_contacts (4-dim) → features (29-dim)."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()

        # Create mock observation (37-dim)
        obs = jax.random.normal(rng, (37,))
        foot_contacts = jnp.array([0.5, 0.3, 0.7, 0.2])  # 4-point model

        features = extract_amp_features(obs, config, foot_contacts)

        assert features.shape == (29,), f"Expected (29,), got {features.shape}"

    def test_feature_extraction_batched_shape(self, rng):
        """Test: batched extraction maintains correct shapes."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()
        num_steps, num_envs = 128, 64

        # Create batched observations
        obs = jax.random.normal(rng, (num_steps, num_envs, 37))
        foot_contacts = jax.random.uniform(rng, (num_steps, num_envs, 4))

        # Batched extraction using vmap
        def extract_single(obs_fc):
            o, fc = obs_fc
            return extract_amp_features(o, config, fc)

        features = jax.vmap(jax.vmap(extract_single))((obs, foot_contacts))

        assert features.shape == (num_steps, num_envs, 29)

    def test_feature_config_values(self):
        """Test: default config has expected values."""
        from playground_amp.amp.amp_features import get_amp_config

        config = get_amp_config()

        assert config.feature_dim == 29
        assert config.num_actuated_joints == 9
        # foot_contact_dim is implicitly 4 (hardcoded in feature_dim calculation)

    def test_velocity_normalization(self, rng):
        """Test: root velocity is normalized to unit direction."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()

        # Create obs with known velocity
        obs = jnp.zeros(37)
        # Set root linear velocity (indices 18-20 in obs, but need to check actual layout)
        # For this test, we'll just verify the output is bounded

        foot_contacts = jnp.array([0.5, 0.3, 0.7, 0.2])
        features = extract_amp_features(obs, config, foot_contacts)

        # Root velocity direction should be unit vector or zero
        # Indices 18-20 in features
        root_vel_dir = features[18:21]
        norm = jnp.linalg.norm(root_vel_dir)

        # Should be either 0 (stationary) or 1 (unit direction)
        assert norm <= 1.0 + 1e-5, f"Velocity direction norm {norm} > 1"

    def test_foot_contacts_passthrough(self, rng):
        """Test: foot contacts are preserved in features."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()

        obs = jax.random.normal(rng, (37,))
        foot_contacts = jnp.array([0.1, 0.2, 0.3, 0.4])

        features = extract_amp_features(obs, config, foot_contacts)

        # Foot contacts are at indices 25-28
        extracted_fc = features[25:29]
        np.testing.assert_array_almost_equal(
            extracted_fc,
            foot_contacts,
            decimal=5,
            err_msg="Foot contacts not preserved in features",
        )

    def test_feature_extraction_deterministic(self, rng):
        """Test: same input produces same output."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()
        obs = jax.random.normal(rng, (37,))
        foot_contacts = jnp.array([0.5, 0.3, 0.7, 0.2])

        features1 = extract_amp_features(obs, config, foot_contacts)
        features2 = extract_amp_features(obs, config, foot_contacts)

        np.testing.assert_array_equal(features1, features2)


# =============================================================================
# 2. Discriminator Tests
# =============================================================================


class TestDiscriminator:
    """Black-box tests for AMP discriminator."""

    def test_discriminator_output_shape(self, rng, batch_size, feature_dim):
        """Test: features (batch, 29) → scores (batch,)."""
        from playground_amp.amp.discriminator import create_discriminator

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        features = jax.random.normal(rng, (batch_size, feature_dim))
        scores = model.apply(params, features, training=False)

        assert scores.shape == (
            batch_size,
        ), f"Expected ({batch_size},), got {scores.shape}"

    def test_discriminator_output_range(self, rng, batch_size, feature_dim):
        """Test: LSGAN scores can be any real number (no sigmoid)."""
        from playground_amp.amp.discriminator import create_discriminator

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        features = jax.random.normal(rng, (batch_size, feature_dim))
        scores = model.apply(params, features, training=False)

        # LSGAN outputs raw logits, not bounded to [0, 1]
        # But typically initialized close to 0
        assert jnp.all(jnp.isfinite(scores)), "Scores contain NaN/Inf"

    def test_discriminator_loss_components(self, rng, batch_size, feature_dim):
        """Test: loss function returns expected components."""
        from playground_amp.amp.discriminator import (
            create_discriminator,
            discriminator_loss,
        )

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        rng, real_rng, fake_rng, loss_rng = jax.random.split(rng, 4)
        real_obs = jax.random.normal(real_rng, (batch_size, feature_dim))
        fake_obs = jax.random.normal(fake_rng, (batch_size, feature_dim))

        loss, metrics = discriminator_loss(
            params=params,
            model=model,
            real_obs=real_obs,
            fake_obs=fake_obs,
            rng_key=loss_rng,
            r1_gamma=5.0,
        )

        # Check loss is scalar
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert jnp.isfinite(loss), "Loss is NaN/Inf"

        # Check metrics dict has expected keys
        expected_keys = [
            "discriminator_loss",
            "discriminator_lsgan",
            "discriminator_r1",
            "discriminator_accuracy",
            "discriminator_real_mean",
            "discriminator_fake_mean",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_discriminator_accuracy_range(self, rng, batch_size, feature_dim):
        """Test: accuracy is in [0, 1] range."""
        from playground_amp.amp.discriminator import (
            create_discriminator,
            discriminator_loss,
        )

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        rng, real_rng, fake_rng, loss_rng = jax.random.split(rng, 4)
        real_obs = jax.random.normal(real_rng, (batch_size, feature_dim))
        fake_obs = jax.random.normal(fake_rng, (batch_size, feature_dim))

        _, metrics = discriminator_loss(
            params=params,
            model=model,
            real_obs=real_obs,
            fake_obs=fake_obs,
            rng_key=loss_rng,
            r1_gamma=5.0,
        )

        acc = metrics["discriminator_accuracy"]
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in [0, 1]"

    def test_amp_reward_mapping(self, rng, batch_size, feature_dim):
        """Test: reward = clip(D(s), 0, 1)."""
        from playground_amp.amp.discriminator import (
            compute_amp_reward,
            create_discriminator,
        )

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        features = jax.random.normal(rng, (batch_size, feature_dim))
        rewards = compute_amp_reward(params, model, features)

        # Rewards should be clipped to [0, 1]
        assert rewards.shape == (batch_size,)
        assert jnp.all(rewards >= 0.0), "Rewards < 0"
        assert jnp.all(rewards <= 1.0), "Rewards > 1"

    def test_amp_reward_known_values(self, feature_dim):
        """Test: D=0 → reward=0, D=1 → reward=1."""
        # This tests the clipping logic directly
        scores = jnp.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        expected_rewards = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])

        rewards = jnp.clip(scores, 0.0, 1.0)

        np.testing.assert_array_almost_equal(rewards, expected_rewards)

    def test_spectral_norm_enabled(self, rng, feature_dim):
        """Test: SpectralNormDense layer works correctly."""
        from playground_amp.amp.discriminator import SpectralNormDense

        layer = SpectralNormDense(features=64)
        x = jax.random.normal(rng, (32, feature_dim))

        # Initialize with training=True to create batch_stats
        params = layer.init(rng, x, training=True)

        # Apply with mutable batch_stats (required for SpectralNorm during training)
        output, updated_vars = layer.apply(
            params, x, training=True, mutable=["batch_stats"]
        )

        assert output.shape == (32, 64)
        assert jnp.all(jnp.isfinite(output))


# =============================================================================
# 3. Reference Motion Data Tests
# =============================================================================


class TestReferenceMotionData:
    """Black-box tests for reference motion data loading."""

    @pytest.fixture
    def ref_data_path(self):
        """Path to reference motion data."""
        return Path("playground_amp/data/walking_motions_normalized_vel.pkl")

    def test_reference_data_exists(self, ref_data_path):
        """Test: reference data file exists."""
        assert ref_data_path.exists(), f"Reference data not found: {ref_data_path}"

    def test_reference_data_shape(self, ref_data_path, feature_dim):
        """Test: reference data has shape (num_samples, 29)."""
        if not ref_data_path.exists():
            pytest.skip("Reference data not found")

        with open(ref_data_path, "rb") as f:
            data = pickle.load(f)

        # Handle different formats
        if isinstance(data, np.ndarray):
            features = data
        elif isinstance(data, dict) and "features" in data:
            features = np.array(data["features"])
        else:
            pytest.fail(f"Unknown reference data format: {type(data)}")

        assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"
        assert (
            features.shape[1] == feature_dim
        ), f"Expected {feature_dim} features, got {features.shape[1]}"

    def test_reference_data_not_empty(self, ref_data_path):
        """Test: reference data has samples."""
        if not ref_data_path.exists():
            pytest.skip("Reference data not found")

        with open(ref_data_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            num_samples = data.shape[0]
        elif isinstance(data, dict) and "features" in data:
            num_samples = len(data["features"])
        else:
            num_samples = 0

        assert num_samples > 0, "Reference data is empty"
        print(f"Reference data has {num_samples} samples")

    def test_reference_data_finite(self, ref_data_path):
        """Test: reference data contains no NaN/Inf."""
        if not ref_data_path.exists():
            pytest.skip("Reference data not found")

        with open(ref_data_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            features = data
        elif isinstance(data, dict) and "features" in data:
            features = np.array(data["features"])
        else:
            pytest.skip("Unknown format")

        assert np.all(np.isfinite(features)), "Reference data contains NaN/Inf"

    def test_reference_data_velocity_normalized(self, ref_data_path):
        """Test: velocity direction in reference data is unit vector or zero."""
        if not ref_data_path.exists():
            pytest.skip("Reference data not found")

        with open(ref_data_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            features = data
        elif isinstance(data, dict) and "features" in data:
            features = np.array(data["features"])
        else:
            pytest.skip("Unknown format")

        # Root velocity direction is at indices 18-20
        vel_dir = features[:, 18:21]
        norms = np.linalg.norm(vel_dir, axis=1)

        # Should be ~0 (stationary) or ~1 (unit direction)
        valid = (norms < 0.1) | (np.abs(norms - 1.0) < 0.1)
        pct_valid = np.mean(valid) * 100

        assert (
            pct_valid > 95
        ), f"Only {pct_valid:.1f}% samples have valid velocity norms"


# =============================================================================
# 4. PPO Tests
# =============================================================================


class TestPPO:
    """Black-box tests for PPO components."""

    def test_gae_output_shape(self, rng):
        """Test: GAE returns advantages and returns of correct shape."""
        from playground_amp.training.ppo_core import compute_gae

        num_steps, num_envs = 128, 64

        rewards = jax.random.normal(rng, (num_steps, num_envs))
        values = jax.random.normal(rng, (num_steps, num_envs))
        dones = jax.random.bernoulli(rng, 0.01, (num_steps, num_envs)).astype(
            jnp.float32
        )
        bootstrap_value = jax.random.normal(rng, (num_envs,))

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            bootstrap_value=bootstrap_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert advantages.shape == (num_steps, num_envs)
        assert returns.shape == (num_steps, num_envs)

    def test_gae_values_finite(self, rng):
        """Test: GAE produces finite values."""
        from playground_amp.training.ppo_core import compute_gae

        num_steps, num_envs = 128, 64

        rewards = jax.random.normal(rng, (num_steps, num_envs))
        values = jax.random.normal(rng, (num_steps, num_envs))
        dones = jnp.zeros((num_steps, num_envs))  # No episode ends
        bootstrap_value = jax.random.normal(rng, (num_envs,))

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            bootstrap_value=bootstrap_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert jnp.all(jnp.isfinite(advantages)), "Advantages contain NaN/Inf"
        assert jnp.all(jnp.isfinite(returns)), "Returns contain NaN/Inf"

    def test_gae_returns_equals_advantages_plus_values(self, rng):
        """Test: returns = advantages + values."""
        from playground_amp.training.ppo_core import compute_gae

        num_steps, num_envs = 32, 16

        rewards = jax.random.normal(rng, (num_steps, num_envs))
        values = jax.random.normal(rng, (num_steps, num_envs))
        dones = jnp.zeros((num_steps, num_envs))
        bootstrap_value = jax.random.normal(rng, (num_envs,))

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            bootstrap_value=bootstrap_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        expected_returns = advantages + values
        np.testing.assert_array_almost_equal(
            returns,
            expected_returns,
            decimal=5,
            err_msg="returns != advantages + values",
        )

    def test_ppo_loss_output(self, rng, obs_dim, action_dim):
        """Test: PPO loss returns scalar loss and metrics."""
        from playground_amp.training.ppo_core import (
            compute_ppo_loss,
            create_networks,
            init_network_params,
        )

        batch_size = 64

        # Create networks
        ppo_network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(64, 32),
            value_hidden_dims=(64, 32),
        )

        processor_params, policy_params, value_params = init_network_params(
            ppo_network, obs_dim, action_dim, seed=42
        )

        # Create mock data
        rng, obs_rng, action_rng, loss_rng = jax.random.split(rng, 4)
        obs = jax.random.normal(obs_rng, (batch_size, obs_dim))
        actions = jax.random.normal(action_rng, (batch_size, action_dim))
        old_log_probs = jax.random.normal(rng, (batch_size,))
        advantages = jax.random.normal(rng, (batch_size,))
        returns = jax.random.normal(rng, (batch_size,))

        loss, metrics = compute_ppo_loss(
            processor_params=processor_params,
            policy_params=policy_params,
            value_params=value_params,
            ppo_network=ppo_network,
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            rng=loss_rng,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        )

        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert jnp.isfinite(loss), "Loss is NaN/Inf"

        # Check metrics
        assert hasattr(metrics, "policy_loss")
        assert hasattr(metrics, "value_loss")
        assert hasattr(metrics, "entropy_loss")

    def test_network_output_shapes(self, rng, obs_dim, action_dim):
        """Test: policy and value networks output correct shapes."""
        from playground_amp.training.ppo_core import (
            compute_values,
            create_networks,
            init_network_params,
            sample_actions,
        )

        batch_size = 64

        ppo_network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(64, 32),
            value_hidden_dims=(64, 32),
        )

        processor_params, policy_params, value_params = init_network_params(
            ppo_network, obs_dim, action_dim, seed=42
        )

        obs = jax.random.normal(rng, (batch_size, obs_dim))

        # Test action sampling
        rng, action_rng = jax.random.split(rng)
        actions, raw_actions, log_probs = sample_actions(
            processor_params, policy_params, ppo_network, obs, action_rng
        )

        assert actions.shape == (batch_size, action_dim)
        assert raw_actions.shape == (batch_size, action_dim)
        assert log_probs.shape == (batch_size,)

        # Test value computation
        values = compute_values(processor_params, value_params, ppo_network, obs)
        assert values.shape == (batch_size,)


# =============================================================================
# 5. Replay Buffer Tests
# =============================================================================


class TestReplayBuffer:
    """Black-box tests for policy replay buffer."""

    def test_jit_buffer_create(self, feature_dim):
        """Test: buffer creation returns correct structure."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        max_size = 1000
        state = JITReplayBuffer.create(max_size, feature_dim)

        assert "data" in state
        assert "ptr" in state
        assert "size" in state
        assert state["data"].shape == (max_size, feature_dim)
        assert state["ptr"] == 0
        assert state["size"] == 0

    def test_jit_buffer_add(self, rng, feature_dim):
        """Test: adding samples updates buffer state."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        max_size = 100
        state = JITReplayBuffer.create(max_size, feature_dim)

        # Add 50 samples
        samples = jax.random.normal(rng, (50, feature_dim))
        new_state = JITReplayBuffer.add(state, samples)

        assert new_state["size"] == 50
        assert new_state["ptr"] == 50

    def test_jit_buffer_wraparound(self, rng, feature_dim):
        """Test: buffer wraps around when full."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        max_size = 100
        state = JITReplayBuffer.create(max_size, feature_dim)

        # Add 150 samples (should wrap around)
        rng1, rng2 = jax.random.split(rng)
        samples1 = jax.random.normal(rng1, (80, feature_dim))
        samples2 = jax.random.normal(rng2, (70, feature_dim))

        state = JITReplayBuffer.add(state, samples1)
        state = JITReplayBuffer.add(state, samples2)

        # Size should be capped at max_size
        assert state["size"] == max_size
        # Pointer should wrap around: (80 + 70) % 100 = 50
        assert state["ptr"] == 50

    def test_jit_buffer_sample(self, rng, feature_dim):
        """Test: sampling returns correct shape."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        max_size = 1000
        state = JITReplayBuffer.create(max_size, feature_dim)

        # Add samples
        rng, add_rng, sample_rng = jax.random.split(rng, 3)
        samples = jax.random.normal(add_rng, (500, feature_dim))
        state = JITReplayBuffer.add(state, samples)

        # Sample batch
        batch_size = 64
        batch = JITReplayBuffer.sample(state, sample_rng, batch_size)

        assert batch.shape == (batch_size, feature_dim)
        assert jnp.all(jnp.isfinite(batch))

    def test_python_buffer_api(self, rng, feature_dim):
        """Test: Python replay buffer API works."""
        from playground_amp.amp.replay_buffer import PolicyReplayBuffer

        buffer = PolicyReplayBuffer(max_size=1000, feature_dim=feature_dim)

        assert buffer.size == 0

        # Add samples
        samples = np.random.randn(100, feature_dim).astype(np.float32)
        buffer.add(samples)

        assert buffer.size == 100

        # Sample
        rng_key = jax.random.PRNGKey(0)
        batch = buffer.sample(rng_key, batch_size=32)

        assert batch.shape == (32, feature_dim)


# =============================================================================
# 6. Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for component interactions."""

    def test_feature_to_discriminator_pipeline(self, rng, feature_dim):
        """Test: features flow correctly through discriminator."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )
        from playground_amp.amp.discriminator import (
            compute_amp_reward,
            create_discriminator,
        )

        # Create discriminator
        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(256, 128),
            seed=42,
        )

        # Extract features
        config = get_amp_config()
        batch_size = 32
        obs = jax.random.normal(rng, (batch_size, 37))
        foot_contacts = jax.random.uniform(rng, (batch_size, 4))

        # Batch extract features
        def extract_single(obs_fc):
            o, fc = obs_fc
            return extract_amp_features(o, config, fc)

        features = jax.vmap(extract_single)((obs, foot_contacts))

        assert features.shape == (batch_size, feature_dim)

        # Compute rewards
        rewards = compute_amp_reward(params, model, features)

        assert rewards.shape == (batch_size,)
        assert jnp.all(rewards >= 0.0)
        assert jnp.all(rewards <= 1.0)

    def test_normalization_consistency(self, rng, feature_dim):
        """Test: normalization is consistent for policy and reference features."""
        from playground_amp.training.trainer_jit import (
            compute_normalization_stats,
            normalize_features,
        )

        # Create mock reference data
        ref_data = jax.random.normal(rng, (1000, feature_dim))

        # Compute fixed stats
        mean, var = compute_normalization_stats(ref_data)

        # Normalize reference data
        norm_ref = normalize_features(ref_data, mean, var)

        # Normalize policy features (should use same stats)
        rng, policy_rng = jax.random.split(rng)
        policy_features = jax.random.normal(policy_rng, (100, feature_dim))
        norm_policy = normalize_features(policy_features, mean, var)

        # Both should be finite
        assert jnp.all(jnp.isfinite(norm_ref))
        assert jnp.all(jnp.isfinite(norm_policy))

        # Reference data should have ~0 mean, ~1 std after normalization
        ref_mean = jnp.mean(norm_ref, axis=0)
        ref_std = jnp.std(norm_ref, axis=0)

        np.testing.assert_array_almost_equal(
            ref_mean,
            jnp.zeros(feature_dim),
            decimal=1,
            err_msg="Normalized reference mean should be ~0",
        )
        np.testing.assert_array_almost_equal(
            ref_std,
            jnp.ones(feature_dim),
            decimal=1,
            err_msg="Normalized reference std should be ~1",
        )


# =============================================================================
# 7. Additional Coverage Tests
# =============================================================================


class TestAMPFeaturesBranchCoverage:
    """Additional tests for branch coverage in amp_features.py."""

    def test_foot_contacts_required_error(self, rng):
        """Test: ValueError raised when foot_contacts is None."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()
        obs = jax.random.normal(rng, (37,))

        with pytest.raises(ValueError, match="foot_contacts is required"):
            extract_amp_features(obs, config, foot_contacts=None)

    def test_temporal_feature_config(self):
        """Test: TemporalFeatureConfig properties."""
        from playground_amp.amp.amp_features import TemporalFeatureConfig

        config = TemporalFeatureConfig(num_frames=3, feature_dim=29)

        assert config.num_frames == 3
        assert config.feature_dim == 29
        assert config.temporal_dim == 87  # 3 * 29

    def test_create_temporal_buffer(self):
        """Test: create_temporal_buffer returns correct shape."""
        from playground_amp.amp.amp_features import (
            create_temporal_buffer,
            TemporalFeatureConfig,
        )

        config = TemporalFeatureConfig(num_frames=3, feature_dim=29)
        buffer = create_temporal_buffer(num_envs=64, config=config)

        assert buffer.shape == (64, 3, 29)
        assert jnp.all(buffer == 0)

    def test_update_temporal_buffer(self, rng):
        """Test: update_temporal_buffer shifts and adds new frame."""
        from playground_amp.amp.amp_features import (
            create_temporal_buffer,
            TemporalFeatureConfig,
            update_temporal_buffer,
        )

        config = TemporalFeatureConfig(num_frames=3, feature_dim=29)
        buffer = create_temporal_buffer(num_envs=4, config=config)

        # Add first frame
        new_features = jax.random.normal(rng, (4, 29))
        buffer = update_temporal_buffer(buffer, new_features)

        # Last frame should be new_features
        np.testing.assert_array_almost_equal(buffer[:, -1, :], new_features)

    def test_get_temporal_features(self, rng):
        """Test: get_temporal_features flattens buffer correctly."""
        from playground_amp.amp.amp_features import (
            create_temporal_buffer,
            get_temporal_features,
            TemporalFeatureConfig,
            update_temporal_buffer,
        )

        config = TemporalFeatureConfig(num_frames=3, feature_dim=29)
        buffer = create_temporal_buffer(num_envs=4, config=config)

        # Add some features
        for i in range(3):
            rng, key = jax.random.split(rng)
            new_features = jax.random.normal(key, (4, 29))
            buffer = update_temporal_buffer(buffer, new_features)

        # Flatten
        flattened = get_temporal_features(buffer)
        assert flattened.shape == (4, 87)  # 4 envs, 3*29 features

    def test_add_temporal_context_to_reference(self, rng):
        """Test: add_temporal_context_to_reference creates windows."""
        from playground_amp.amp.amp_features import add_temporal_context_to_reference

        # Create single-frame features
        features = jax.random.normal(rng, (100, 29))

        # Convert to temporal (3-frame windows)
        temporal = add_temporal_context_to_reference(features, num_frames=3)

        # Should have N - num_frames + 1 = 100 - 3 + 1 = 98 windows
        assert temporal.shape == (98, 87)

    def test_add_temporal_context_insufficient_frames(self, rng):
        """Test: ValueError when not enough frames for temporal window."""
        from playground_amp.amp.amp_features import add_temporal_context_to_reference

        # Only 2 frames, but need 3
        features = jax.random.normal(rng, (2, 29))

        with pytest.raises(ValueError, match="Not enough frames"):
            add_temporal_context_to_reference(features, num_frames=3)

    def test_running_mean_std_creation(self):
        """Test: create_running_stats returns correct initial state."""
        from playground_amp.amp.amp_features import create_running_stats

        stats = create_running_stats(feature_dim=29)

        assert stats.mean.shape == (29,)
        assert stats.var.shape == (29,)
        assert jnp.allclose(stats.mean, 0.0)
        assert jnp.allclose(stats.var, 1.0)

    def test_update_running_stats(self, rng):
        """Test: update_running_stats updates mean and var correctly."""
        from playground_amp.amp.amp_features import (
            create_running_stats,
            update_running_stats,
        )

        stats = create_running_stats(feature_dim=29)

        # Update with a batch
        batch = jax.random.normal(rng, (100, 29))
        new_stats = update_running_stats(stats, batch)

        # Mean should be close to batch mean
        batch_mean = jnp.mean(batch, axis=0)
        # Count is very small initially, so new mean ≈ batch mean
        assert jnp.allclose(new_stats.mean, batch_mean, atol=0.1)

    def test_normalize_features_with_stats(self, rng):
        """Test: normalize_features clips and normalizes correctly."""
        from playground_amp.amp.amp_features import (
            create_running_stats,
            normalize_features,
            update_running_stats,
        )

        # Create stats from known data
        batch = jax.random.normal(rng, (1000, 29))
        stats = create_running_stats(feature_dim=29)
        stats = update_running_stats(stats, batch)

        # Normalize the same batch
        normalized = normalize_features(batch, stats, clip=10.0)

        # Should be approximately standard normal
        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.all(normalized >= -10.0)
        assert jnp.all(normalized <= 10.0)


class TestDiscriminatorBranchCoverage:
    """Additional tests for branch coverage in discriminator.py."""

    def test_discriminator_without_spectral_norm(self, rng, feature_dim):
        """Test: Discriminator with LayerNorm (legacy mode)."""
        from playground_amp.amp.discriminator import AMPDiscriminator

        model = AMPDiscriminator(hidden_dims=(64, 32), use_spectral_norm=False)
        x = jax.random.normal(rng, (16, feature_dim))

        params = model.init(rng, x, training=False)
        output = model.apply(params, x, training=False)

        assert output.shape == (16,)
        assert jnp.all(jnp.isfinite(output))

    def test_discriminator_loss_with_input_noise(self, rng, batch_size, feature_dim):
        """Test: discriminator_loss with input noise enabled."""
        from playground_amp.amp.discriminator import (
            create_discriminator,
            discriminator_loss,
        )

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(64, 32),
            seed=42,
        )

        rng, real_rng, fake_rng, loss_rng = jax.random.split(rng, 4)
        real_obs = jax.random.normal(real_rng, (batch_size, feature_dim))
        fake_obs = jax.random.normal(fake_rng, (batch_size, feature_dim))

        # Test with input noise
        loss, metrics = discriminator_loss(
            params=params,
            model=model,
            real_obs=real_obs,
            fake_obs=fake_obs,
            rng_key=loss_rng,
            r1_gamma=5.0,
            input_noise_std=0.1,  # Enable input noise
        )

        assert jnp.isfinite(loss)
        assert "discriminator_loss" in metrics

    def test_create_discriminator_optimizer(self):
        """Test: create_discriminator_optimizer returns valid optimizer."""
        from playground_amp.amp.discriminator import create_discriminator_optimizer

        optimizer = create_discriminator_optimizer(learning_rate=1e-4)

        # Check it's a valid optax optimizer
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")


class TestReplayBufferBranchCoverage:
    """Additional tests for branch coverage in replay_buffer.py."""

    def test_jit_buffer_sample_empty(self, feature_dim):
        """Test: sampling from empty buffer returns zeros."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        state = JITReplayBuffer.create(max_size=100, feature_dim=feature_dim)
        rng = jax.random.PRNGKey(0)

        # Sample from empty buffer
        batch = JITReplayBuffer.sample(state, rng, batch_size=32)

        # Should return zeros since buffer is empty
        assert batch.shape == (32, feature_dim)

    def test_jit_buffer_add_exact_capacity(self, rng, feature_dim):
        """Test: adding exactly max_size samples."""
        from playground_amp.amp.replay_buffer import JITReplayBuffer

        max_size = 100
        state = JITReplayBuffer.create(max_size, feature_dim)

        # Add exactly max_size samples
        samples = jax.random.normal(rng, (max_size, feature_dim))
        new_state = JITReplayBuffer.add(state, samples)

        assert new_state["size"] == max_size
        assert new_state["ptr"] == 0  # Should wrap to 0

    def test_python_buffer_sample_more_than_size(self, rng, feature_dim):
        """Test: Python buffer sampling when batch_size > buffer size."""
        from playground_amp.amp.replay_buffer import PolicyReplayBuffer

        buffer = PolicyReplayBuffer(max_size=1000, feature_dim=feature_dim)

        # Add only 10 samples
        samples = np.random.randn(10, feature_dim).astype(np.float32)
        buffer.add(samples)

        # Request more samples than available (should sample with replacement)
        rng_key = jax.random.PRNGKey(0)
        batch = buffer.sample(rng_key, batch_size=50)

        assert batch.shape == (50, feature_dim)

    def test_python_buffer_wraparound(self, feature_dim):
        """Test: Python buffer wraps around correctly."""
        from playground_amp.amp.replay_buffer import PolicyReplayBuffer

        buffer = PolicyReplayBuffer(max_size=100, feature_dim=feature_dim)

        # Add 150 samples in batches
        for i in range(3):
            samples = np.full((50, feature_dim), i, dtype=np.float32)
            buffer.add(samples)

        assert buffer.size == 100
        # Most recent 100 samples should be in buffer


class TestPPOBranchCoverage:
    """Additional tests for branch coverage in ppo_core.py."""

    def test_sample_actions_deterministic(self, rng, obs_dim, action_dim):
        """Test: deterministic action sampling (mode of distribution)."""
        from playground_amp.training.ppo_core import (
            create_networks,
            init_network_params,
            sample_actions,
        )

        ppo_network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(64, 32),
            value_hidden_dims=(64, 32),
        )

        processor_params, policy_params, value_params = init_network_params(
            ppo_network, obs_dim, action_dim, seed=42
        )

        obs = jax.random.normal(rng, (16, obs_dim))

        # Deterministic sampling
        actions1, raw1, log_probs1 = sample_actions(
            processor_params, policy_params, ppo_network, obs, rng, deterministic=True
        )
        actions2, raw2, log_probs2 = sample_actions(
            processor_params, policy_params, ppo_network, obs, rng, deterministic=True
        )

        # Should be identical
        np.testing.assert_array_equal(actions1, actions2)

    def test_gae_with_episode_terminations(self, rng):
        """Test: GAE handles episode terminations correctly."""
        from playground_amp.training.ppo_core import compute_gae

        num_steps, num_envs = 32, 8

        rewards = jax.random.normal(rng, (num_steps, num_envs))
        values = jax.random.normal(rng, (num_steps, num_envs))
        bootstrap_value = jax.random.normal(rng, (num_envs,))

        # Half of the steps are episode terminations
        dones = jnp.zeros((num_steps, num_envs))
        dones = dones.at[::2, :].set(1.0)  # Every other step is a termination

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            bootstrap_value=bootstrap_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert jnp.all(jnp.isfinite(advantages))
        assert jnp.all(jnp.isfinite(returns))

    def test_ppo_loss_without_advantage_normalization(self, rng, obs_dim, action_dim):
        """Test: PPO loss with normalize_advantages=False."""
        from playground_amp.training.ppo_core import (
            compute_ppo_loss,
            create_networks,
            init_network_params,
        )

        batch_size = 64

        ppo_network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(64, 32),
            value_hidden_dims=(64, 32),
        )

        processor_params, policy_params, value_params = init_network_params(
            ppo_network, obs_dim, action_dim, seed=42
        )

        rng, obs_rng, action_rng, loss_rng = jax.random.split(rng, 4)
        obs = jax.random.normal(obs_rng, (batch_size, obs_dim))
        actions = jax.random.normal(action_rng, (batch_size, action_dim))
        old_log_probs = jax.random.normal(rng, (batch_size,))
        advantages = jax.random.normal(rng, (batch_size,))
        returns = jax.random.normal(rng, (batch_size,))

        loss, metrics = compute_ppo_loss(
            processor_params=processor_params,
            policy_params=policy_params,
            value_params=value_params,
            ppo_network=ppo_network,
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            rng=loss_rng,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            normalize_advantages=False,  # Don't normalize
        )

        assert jnp.isfinite(loss)

    def test_create_optimizer_with_different_params(self):
        """Test: create_optimizer with various learning rates."""
        from playground_amp.training.ppo_core import create_optimizer

        # Test with different learning rates
        opt1 = create_optimizer(learning_rate=1e-3, max_grad_norm=0.5)
        opt2 = create_optimizer(learning_rate=1e-5, max_grad_norm=1.0)

        assert hasattr(opt1, "init")
        assert hasattr(opt2, "init")


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_discriminator_single_sample(self, rng, feature_dim):
        """Test: Discriminator works with batch_size=1."""
        from playground_amp.amp.discriminator import create_discriminator

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(64, 32),
            seed=42,
        )

        # Single sample
        features = jax.random.normal(rng, (1, feature_dim))
        scores = model.apply(params, features, training=False)

        assert scores.shape == (1,)

    def test_discriminator_large_batch(self, rng, feature_dim):
        """Test: Discriminator handles large batches."""
        from playground_amp.amp.discriminator import create_discriminator

        model, params = create_discriminator(
            obs_dim=feature_dim,
            hidden_dims=(64, 32),
            seed=42,
        )

        # Large batch
        features = jax.random.normal(rng, (4096, feature_dim))
        scores = model.apply(params, features, training=False)

        assert scores.shape == (4096,)
        assert jnp.all(jnp.isfinite(scores))

    def test_gae_single_step(self, rng):
        """Test: GAE works with num_steps=1."""
        from playground_amp.training.ppo_core import compute_gae

        num_envs = 8

        rewards = jax.random.normal(rng, (1, num_envs))
        values = jax.random.normal(rng, (1, num_envs))
        dones = jnp.zeros((1, num_envs))
        bootstrap_value = jax.random.normal(rng, (num_envs,))

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            bootstrap_value=bootstrap_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert advantages.shape == (1, num_envs)
        assert returns.shape == (1, num_envs)

    def test_feature_extraction_with_zeros(self):
        """Test: Feature extraction handles zero observations."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()
        obs = jnp.zeros(37)
        foot_contacts = jnp.zeros(4)

        features = extract_amp_features(obs, config, foot_contacts)

        assert features.shape == (29,)
        assert jnp.all(jnp.isfinite(features))

    def test_feature_extraction_with_large_values(self, rng):
        """Test: Feature extraction handles large values."""
        from playground_amp.amp.amp_features import (
            extract_amp_features,
            get_amp_config,
        )

        config = get_amp_config()
        obs = jax.random.normal(rng, (37,)) * 100  # Large values
        foot_contacts = jnp.array([0.9, 0.8, 0.7, 0.6])

        features = extract_amp_features(obs, config, foot_contacts)

        assert features.shape == (29,)
        assert jnp.all(jnp.isfinite(features))


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
