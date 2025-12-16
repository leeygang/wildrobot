"""AMP feature extraction for discriminator training.

This module extracts motion features from observations for use with
the AMP discriminator. Features are designed to capture the style
of motion (joint angles, velocities, base height/orientation) while
being invariant to task-specific aspects (goal position, commands).

Based on DeepMind AMP paper and 2024 best practices:
- Joint angles (root-relative)
- Joint velocities
- Root height + orientation
- Normalize features with running statistics
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class AMPFeatureConfig(NamedTuple):
    """Configuration for AMP feature extraction.

    Defines which parts of the observation to use for discriminator.
    Default values are for WildRobot 44-dim observation space.
    """

    # Joint position indices in observation
    joint_pos_start: int = 7  # Start index for joint positions
    joint_pos_end: int = 18  # End index (exclusive) - 11 joints

    # Joint velocity indices
    joint_vel_start: int = 32  # Start index for joint velocities
    joint_vel_end: int = 43  # End index (exclusive) - 11 joints

    # Base height index
    base_height_idx: int = 0  # Index of base height in observation

    # Base orientation indices (6D rotation or quaternion)
    base_orient_start: int = 1  # Start of orientation
    base_orient_end: int = 7  # End of orientation (6D repr)

    # Whether to use transition features (current + next)
    use_transition_features: bool = False

    # Feature dimension (computed from above)
    @property
    def feature_dim(self) -> int:
        dim = 0
        dim += self.joint_pos_end - self.joint_pos_start  # Joint positions
        dim += self.joint_vel_end - self.joint_vel_start  # Joint velocities
        dim += 1  # Base height
        dim += self.base_orient_end - self.base_orient_start  # Base orientation
        if self.use_transition_features:
            dim *= 2  # Double for transition features
        return dim


# Default config for WildRobot
DEFAULT_AMP_CONFIG = AMPFeatureConfig()


def extract_amp_features(
    obs: jnp.ndarray,
    next_obs: Optional[jnp.ndarray] = None,
    config: AMPFeatureConfig = DEFAULT_AMP_CONFIG,
) -> jnp.ndarray:
    """Extract motion features for discriminator from observation.

    Args:
        obs: Current observation (..., obs_dim)
        next_obs: Optional next observation for transition features
        config: Feature extraction configuration

    Returns:
        amp_features: Motion features (..., feature_dim)
    """
    # Extract components from current observation
    joint_pos = obs[..., config.joint_pos_start : config.joint_pos_end]
    joint_vel = obs[..., config.joint_vel_start : config.joint_vel_end]
    base_height = obs[..., config.base_height_idx : config.base_height_idx + 1]
    base_orient = obs[..., config.base_orient_start : config.base_orient_end]

    # Concatenate features
    features = jnp.concatenate(
        [
            joint_pos,
            joint_vel,
            base_height,
            base_orient,
        ],
        axis=-1,
    )

    # Add transition features if using next_obs
    if config.use_transition_features and next_obs is not None:
        next_joint_pos = next_obs[..., config.joint_pos_start : config.joint_pos_end]
        next_joint_vel = next_obs[..., config.joint_vel_start : config.joint_vel_end]
        next_base_height = next_obs[
            ..., config.base_height_idx : config.base_height_idx + 1
        ]
        next_base_orient = next_obs[
            ..., config.base_orient_start : config.base_orient_end
        ]

        next_features = jnp.concatenate(
            [
                next_joint_pos,
                next_joint_vel,
                next_base_height,
                next_base_orient,
            ],
            axis=-1,
        )

        features = jnp.concatenate([features, next_features], axis=-1)

    return features


# JIT-compiled version for performance
extract_amp_features_jit = jax.jit(extract_amp_features, static_argnames=("config",))


class RunningMeanStd(NamedTuple):
    """Running mean and standard deviation for feature normalization.

    Uses Welford's online algorithm for numerically stable updates.
    """

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def create_running_stats(feature_dim: int) -> RunningMeanStd:
    """Create initial running statistics."""
    return RunningMeanStd(
        mean=jnp.zeros(feature_dim, dtype=jnp.float32),
        var=jnp.ones(feature_dim, dtype=jnp.float32),
        count=jnp.array(
            1e-4, dtype=jnp.float32
        ),  # Small count to avoid division by zero
    )


def update_running_stats(
    stats: RunningMeanStd,
    batch: jnp.ndarray,
) -> RunningMeanStd:
    """Update running statistics with a new batch of features.

    Uses Welford's parallel algorithm for combining statistics.

    Args:
        stats: Current running statistics
        batch: New batch of features (batch_size, feature_dim)

    Returns:
        Updated running statistics
    """
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = jnp.array(batch.shape[0], dtype=jnp.float32)

    # Combine statistics using parallel Welford's algorithm
    total_count = stats.count + batch_count

    delta = batch_mean - stats.mean

    new_mean = stats.mean + delta * batch_count / total_count

    # Combine variances
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * stats.count * batch_count / total_count

    new_var = M2 / total_count

    return RunningMeanStd(
        mean=new_mean,
        var=new_var,
        count=total_count,
    )


def normalize_features(
    features: jnp.ndarray,
    stats: RunningMeanStd,
    clip: float = 10.0,
) -> jnp.ndarray:
    """Normalize features using running statistics.

    Args:
        features: Input features (..., feature_dim)
        stats: Running mean and std
        clip: Clipping range for normalized features

    Returns:
        Normalized features
    """
    std = jnp.sqrt(stats.var + 1e-8)
    normalized = (features - stats.mean) / std
    return jnp.clip(normalized, -clip, clip)


# JIT compiled versions
update_running_stats_jit = jax.jit(update_running_stats)
normalize_features_jit = jax.jit(normalize_features, static_argnames=("clip",))


def compute_feature_dim(obs_dim: int = 44) -> int:
    """Compute the AMP feature dimension for a given observation dimension.

    Default WildRobot observation (44-dim):
    - Joint positions: 11
    - Joint velocities: 11
    - Base height: 1
    - Base orientation: 6
    Total: 29 features

    Args:
        obs_dim: Observation dimension (default 44 for WildRobot)

    Returns:
        Feature dimension
    """
    return DEFAULT_AMP_CONFIG.feature_dim


# Alternative feature extraction using full observation (simpler)
def extract_amp_features_full(
    obs: jnp.ndarray,
    next_obs: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Extract AMP features using full observation.

    This is a simpler alternative that uses the full observation
    as features. Can work well if discriminator is large enough.

    Args:
        obs: Current observation (..., obs_dim)
        next_obs: Optional next observation (unused in this variant)

    Returns:
        Features (same as input observation)
    """
    return obs


# Select which feature extraction to use
def get_amp_feature_fn(mode: str = "selective"):
    """Get AMP feature extraction function.

    Args:
        mode: "selective" for motion-specific features, "full" for full obs

    Returns:
        Feature extraction function
    """
    if mode == "full":
        return extract_amp_features_full
    else:
        return extract_amp_features_jit


if __name__ == "__main__":
    # Test feature extraction
    key = jax.random.PRNGKey(0)

    # Create dummy observation
    obs = jax.random.normal(key, (8, 44))  # Batch of 8, 44-dim obs

    # Extract features
    features = extract_amp_features(obs)
    print(f"Input obs shape: {obs.shape}")
    print(f"AMP features shape: {features.shape}")
    print(f"Expected feature dim: {DEFAULT_AMP_CONFIG.feature_dim}")

    # Test running stats
    stats = create_running_stats(features.shape[-1])
    stats = update_running_stats(stats, features)
    print(f"\nRunning stats count: {stats.count}")
    print(f"Running mean shape: {stats.mean.shape}")

    # Test normalization
    normalized = normalize_features(features, stats)
    print(f"\nNormalized features mean: {jnp.mean(normalized):.4f}")
    print(f"Normalized features std: {jnp.std(normalized):.4f}")

    print("\n✅ AMP feature extraction test passed!")
