"""Tests for AMP feature extraction."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp

from playground_amp.amp.amp_features import (
    AMPFeatureConfig,
    create_running_stats,
    extract_amp_features,
    get_amp_config,
    normalize_features,
    update_running_stats,
)
from playground_amp.configs.config import load_robot_config


def test_feature_extraction():
    """Test basic feature extraction from observations."""
    # Load robot config first (required before get_amp_config)
    robot_config_path = project_root / "assets" / "robot_config.yaml"
    if not robot_config_path.exists():
        print(f"⚠️  Skipping test: robot_config.yaml not found at {robot_config_path}")
        print("   Run 'cd assets && python post_process.py' to generate it.")
        return

    load_robot_config(robot_config_path)
    config = get_amp_config()

    # Create dummy observation (38-dim as per robot_config)
    key = jax.random.PRNGKey(0)
    obs = jax.random.normal(key, (8, 38))  # Batch of 8, 38-dim obs

    # Extract features
    features = extract_amp_features(obs, config)

    print(f"Input obs shape: {obs.shape}")
    print(f"AMP features shape: {features.shape}")
    print(f"Expected feature dim: {config.feature_dim}")

    assert features.shape == (8, 29), f"Expected (8, 29), got {features.shape}"
    assert features.shape[-1] == config.feature_dim

    print("✅ Feature extraction test passed!")


def test_running_stats():
    """Test running statistics computation."""
    key = jax.random.PRNGKey(42)
    feature_dim = 29

    # Create initial stats
    stats = create_running_stats(feature_dim)
    assert stats.mean.shape == (feature_dim,)
    assert stats.var.shape == (feature_dim,)

    # Update with random batch
    batch = jax.random.normal(key, (100, feature_dim))
    stats = update_running_stats(stats, batch)

    print(f"Running stats count: {stats.count}")
    print(f"Running mean shape: {stats.mean.shape}")
    print(f"Running var shape: {stats.var.shape}")

    assert stats.count > 1.0
    assert stats.mean.shape == (feature_dim,)

    print("✅ Running stats test passed!")


def test_normalization():
    """Test feature normalization."""
    key = jax.random.PRNGKey(123)
    feature_dim = 29

    # Create stats and batch
    batch = jax.random.normal(key, (100, feature_dim)) * 10 + 5  # Mean ~5, std ~10
    stats = create_running_stats(feature_dim)
    stats = update_running_stats(stats, batch)

    # Normalize
    normalized = normalize_features(batch, stats)

    print(f"Original mean: {jnp.mean(batch):.4f}")
    print(f"Original std: {jnp.std(batch):.4f}")
    print(f"Normalized mean: {jnp.mean(normalized):.4f}")
    print(f"Normalized std: {jnp.std(normalized):.4f}")

    # Normalized should be roughly mean=0, std=1
    assert jnp.abs(jnp.mean(normalized)) < 0.5, "Normalized mean should be near 0"
    assert jnp.abs(jnp.std(normalized) - 1.0) < 0.5, "Normalized std should be near 1"

    print("✅ Normalization test passed!")


def test_velocity_normalization():
    """Test that root linear velocity is normalized to unit direction."""
    robot_config_path = project_root / "assets" / "robot_config.yaml"
    if not robot_config_path.exists():
        print(f"⚠️  Skipping test: robot_config.yaml not found")
        return

    load_robot_config(robot_config_path)
    config = get_amp_config()

    # Create observation with known velocity
    obs = jnp.zeros((1, 38))

    # Set root linear velocity to (3, 4, 0) - magnitude = 5
    obs = obs.at[0, config.root_linvel_start : config.root_linvel_end].set(
        jnp.array([3.0, 4.0, 0.0])
    )

    features = extract_amp_features(obs, config)

    # Extract velocity direction from features (indices 18-20)
    vel_dir = features[0, 18:21]
    vel_magnitude = jnp.linalg.norm(vel_dir)

    print(f"Input velocity: [3, 4, 0] (magnitude=5)")
    print(f"Output direction: {vel_dir}")
    print(f"Output magnitude: {vel_magnitude:.4f}")

    # Should be unit vector (0.6, 0.8, 0)
    assert jnp.abs(vel_magnitude - 1.0) < 0.01, "Velocity should be normalized to unit"
    assert jnp.allclose(vel_dir, jnp.array([0.6, 0.8, 0.0]), atol=0.01)

    print("✅ Velocity normalization test passed!")


def test_stationary_velocity():
    """Test that near-zero velocity returns zero direction."""
    robot_config_path = project_root / "assets" / "robot_config.yaml"
    if not robot_config_path.exists():
        print(f"⚠️  Skipping test: robot_config.yaml not found")
        return

    load_robot_config(robot_config_path)
    config = get_amp_config()

    # Create observation with very small velocity (below threshold)
    obs = jnp.zeros((1, 38))
    obs = obs.at[0, config.root_linvel_start : config.root_linvel_end].set(
        jnp.array([0.01, 0.02, 0.0])  # magnitude ~0.02, below 0.1 threshold
    )

    features = extract_amp_features(obs, config)
    vel_dir = features[0, 18:21]

    print(f"Input velocity: [0.01, 0.02, 0] (magnitude~0.02)")
    print(f"Output direction: {vel_dir}")

    # Should be zero vector for stationary
    assert jnp.allclose(vel_dir, jnp.zeros(3), atol=0.01), "Stationary should give zero"

    print("✅ Stationary velocity test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("AMP Feature Extraction Tests")
    print("=" * 60)
    print()

    test_feature_extraction()
    print()

    test_running_stats()
    print()

    test_normalization()
    print()

    test_velocity_normalization()
    print()

    test_stationary_velocity()
    print()

    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
