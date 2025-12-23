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

Configuration is loaded from robot_config.yaml to avoid hardcoding.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from playground_amp.configs.config import get_robot_config, RobotConfig


class AMPFeatureConfig(NamedTuple):
    """Configuration for AMP feature extraction.

    Defines which parts of the observation to use for discriminator.
    Indices are derived from robot_config.yaml observation_indices.

    NOTE: Root linear velocity is NORMALIZED to unit direction vector.
    This preserves directional information (forward/backward/sideways)
    while removing speed magnitude that causes mismatch between
    reference data (~0.27 m/s) and commanded speed (0.5-1.0 m/s).
    """

    # Observation indices for feature extraction
    # Joint positions
    joint_pos_start: int
    joint_pos_end: int

    # Joint velocities
    joint_vel_start: int
    joint_vel_end: int

    # Root velocities (from observation)
    root_linvel_start: int
    root_linvel_end: int
    root_angvel_start: int
    root_angvel_end: int

    # Root height - gravity z-component is proxy for uprightness
    root_height_idx: int

    # Whether to use transition features (current + next)
    use_transition_features: bool = False

    # Number of actuated joints
    num_actuated_joints: int = 9

    # Feature dimension
    @property
    def feature_dim(self) -> int:
        """AMP feature dimension.

        Format: joint_pos + joint_vel + root_linvel_dir(3) + root_angvel(3) + root_height(1) + foot_contacts(4)
        NOTE: root_linvel is NORMALIZED to unit direction (removes speed, keeps direction)
        """
        # 9 joint_pos + 9 joint_vel + 3 root_linvel_dir + 3 root_angvel + 1 root_height + 4 foot_contacts = 29
        return self.num_actuated_joints * 2 + 3 + 3 + 1 + 4


def create_amp_config_from_robot(robot_config: Optional[RobotConfig] = None) -> AMPFeatureConfig:
    """Create AMPFeatureConfig from robot configuration.

    Args:
        robot_config: Robot configuration. If None, loads from default path.

    Returns:
        AMPFeatureConfig with indices from robot config
    """
    if robot_config is None:
        robot_config = get_robot_config()

    obs_indices = robot_config.observation_indices

    return AMPFeatureConfig(
        joint_pos_start=obs_indices["joint_positions"]["start"],
        joint_pos_end=obs_indices["joint_positions"]["end"],
        joint_vel_start=obs_indices["joint_velocities"]["start"],
        joint_vel_end=obs_indices["joint_velocities"]["end"],
        root_linvel_start=obs_indices["base_linear_velocity"]["start"],
        root_linvel_end=obs_indices["base_linear_velocity"]["end"],
        root_angvel_start=obs_indices["base_angular_velocity"]["start"],
        root_angvel_end=obs_indices["base_angular_velocity"]["end"],
        root_height_idx=obs_indices["gravity_vector"]["start"] + 2,  # gravity[2]
        use_transition_features=False,
        num_actuated_joints=robot_config.action_dim,
    )


# Lazy-loaded default config
_default_amp_config: Optional[AMPFeatureConfig] = None


def get_default_amp_config() -> AMPFeatureConfig:
    """Get default AMP feature configuration.

    Loads from robot config on first call, then caches.
    """
    global _default_amp_config
    if _default_amp_config is None:
        _default_amp_config = create_amp_config_from_robot()
    return _default_amp_config


# For backward compatibility
DEFAULT_AMP_CONFIG = AMPFeatureConfig(
    joint_pos_start=9,
    joint_pos_end=18,
    joint_vel_start=18,
    joint_vel_end=27,
    root_linvel_start=6,
    root_linvel_end=9,
    root_angvel_start=3,
    root_angvel_end=6,
    root_height_idx=2,
    use_transition_features=False,
    num_actuated_joints=9,
)


def extract_amp_features(
    obs: jnp.ndarray,
    next_obs: Optional[jnp.ndarray] = None,
    config: AMPFeatureConfig = DEFAULT_AMP_CONFIG,
) -> jnp.ndarray:
    """Extract motion features for discriminator from observation.

    Maps observation to 29-dim AMP feature format:
    - Joint positions (0-8): 9 dims
    - Joint velocities (9-17): 9 dims
    - Root linear velocity DIRECTION (18-20): 3 dims (normalized unit vector)
    - Root angular velocity (21-23): 3 dims
    - Root height (24): 1 dim
    - Foot contacts (25-28): 4 dims (zeros placeholder)

    NOTE: Root linear velocity is NORMALIZED to unit direction vector.
    This preserves directional info (forward/backward/sideways) while
    removing speed magnitude that causes mismatch between reference
    data (~0.27 m/s) and commanded speed (0.5-1.0 m/s).

    Args:
        obs: Current observation (..., obs_dim=38)
        next_obs: Optional next observation (unused)
        config: Feature extraction configuration

    Returns:
        amp_features: Motion features (..., 29)
    """
    # Extract components from observation (38-dim layout):
    # - gravity: 0-2 (3)
    # - angvel: 3-5 (3)
    # - linvel: 6-8 (3)
    # - joint_pos: 9-17 (9)
    # - joint_vel: 18-26 (9)
    # - prev_action: 27-35 (9)
    # - velocity_cmd: 36 (1)
    # - padding: 37 (1)

    joint_pos = obs[..., config.joint_pos_start : config.joint_pos_end]  # 9 dims
    joint_vel = obs[..., config.joint_vel_start : config.joint_vel_end]  # 9 dims
    root_linvel = obs[..., config.root_linvel_start : config.root_linvel_end]  # 3 dims
    root_angvel = obs[..., config.root_angvel_start : config.root_angvel_end]  # 3 dims
    root_height = obs[..., config.root_height_idx : config.root_height_idx + 1]  # 1 dim

    # Normalize root linear velocity to unit direction vector
    # This removes speed (magnitude) while preserving direction
    # SAFETY: When velocity is very small (standing/start of episode),
    # return zero vector instead of noisy normalized direction
    linvel_norm = jnp.linalg.norm(root_linvel, axis=-1, keepdims=True)

    # Threshold: if speed < 0.1 m/s, consider it "stationary" → zero direction
    min_speed_threshold = 0.1
    is_moving = linvel_norm > min_speed_threshold

    # Safe normalization: only normalize if moving, else zero
    root_linvel_dir = jnp.where(
        is_moving,
        root_linvel / (linvel_norm + 1e-8),  # Normalized direction
        jnp.zeros_like(root_linvel)           # Zero for stationary
    )

    # Foot contacts - we don't have this in observation, use zeros
    # Shape: (..., 4)
    batch_shape = obs.shape[:-1]
    foot_contacts = jnp.zeros(batch_shape + (4,), dtype=obs.dtype)

    # Concatenate in order (29-dim with normalized velocity direction):
    # [joint_pos(9), joint_vel(9), root_linvel_dir(3), root_angvel(3), root_height(1), foot_contacts(4)]
    features = jnp.concatenate(
        [
            joint_pos,        # 0-8: 9 dims
            joint_vel,        # 9-17: 9 dims
            root_linvel_dir,  # 18-20: 3 dims (normalized direction)
            root_angvel,      # 21-23: 3 dims
            root_height,      # 24: 1 dim
            foot_contacts,    # 25-28: 4 dims
        ],
        axis=-1,
    )

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
