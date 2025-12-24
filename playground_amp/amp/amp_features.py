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

from typing import NamedTuple, Tuple

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

    # Number of actuated joints (from robot_config.action_dim)
    num_actuated_joints: int

    # Whether to use transition features (current + next)
    # Fields with defaults must come after fields without defaults
    use_transition_features: bool = False

    # Feature dimension
    @property
    def feature_dim(self) -> int:
        """AMP feature dimension.

        Format: joint_pos + joint_vel + root_linvel_dir(3) + root_angvel(3) + root_height(1) + foot_contacts(4)
        NOTE: root_linvel is NORMALIZED to unit direction (removes speed, keeps direction)
        """
        # 9 joint_pos + 9 joint_vel + 3 root_linvel_dir + 3 root_angvel + 1 root_height + 4 foot_contacts = 29
        return self.num_actuated_joints * 2 + 3 + 3 + 1 + 4


def create_amp_config_from_robot(robot_config: RobotConfig) -> AMPFeatureConfig:
    """Create AMPFeatureConfig from robot configuration.

    Args:
        robot_config: Robot configuration (REQUIRED).

    Returns:
        AMPFeatureConfig with indices from robot config
    """
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


def get_amp_config() -> AMPFeatureConfig:
    """Get AMP feature configuration from cached robot config.

    Raises:
        RuntimeError: If robot_config hasn't been loaded yet.
    """
    robot_config = get_robot_config()
    return create_amp_config_from_robot(robot_config)


def extract_amp_features(
    obs: jnp.ndarray,
    config: AMPFeatureConfig,
    foot_contacts: jnp.ndarray,
    root_height: jnp.ndarray,
) -> jnp.ndarray:
    """Extract motion features for discriminator from observation.

    Maps observation to 29-dim AMP feature format:
    - Joint positions (0-8): 9 dims
    - Joint velocities (9-17): 9 dims
    - Root linear velocity DIRECTION (18-20): 3 dims (normalized unit vector)
    - Root angular velocity (21-23): 3 dims
    - Root height (24): 1 dim (actual height in meters)
    - Foot contacts (25-28): 4 dims

    v0.5.0: foot_contacts is REQUIRED (no silent fallback to zeros).
    v0.6.1: root_height is REQUIRED (must be actual height, not gravity[2]).

    NOTE: Root linear velocity is NORMALIZED to unit direction vector.
    This preserves directional info (forward/backward/sideways) while
    removing speed magnitude that causes mismatch between reference
    data (~0.27 m/s) and commanded speed (0.5-1.0 m/s).

    NOTE: Root height is the ACTUAL robot height (meters), not gravity[2].
    This must match the reference data format which uses actual height.

    Args:
        obs: Current observation (..., obs_dim=38)
        config: Feature extraction configuration (REQUIRED)
        foot_contacts: Foot contact values from env (..., 4) (REQUIRED)
        root_height: Root height from env (...,) in meters (REQUIRED)

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

    # Root height: ensure shape is (..., 1) for concatenation
    # v0.6.1: root_height is REQUIRED - must be actual height from env.info["root_height"]
    if root_height.ndim == 0 or (root_height.ndim > 0 and root_height.shape[-1] != 1):
        root_height_feat = root_height[..., jnp.newaxis]
    else:
        root_height_feat = root_height

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
        jnp.zeros_like(root_linvel),  # Zero for stationary
    )

    # Concatenate in order (29-dim with normalized velocity direction):
    # [joint_pos(9), joint_vel(9), root_linvel_dir(3), root_angvel(3), root_height(1), foot_contacts(4)]
    features = jnp.concatenate(
        [
            joint_pos,  # 0-8: 9 dims
            joint_vel,  # 9-17: 9 dims
            root_linvel_dir,  # 18-20: 3 dims (normalized direction)
            root_angvel,  # 21-23: 3 dims
            root_height_feat,  # 24: 1 dim (actual height in meters)
            foot_contacts,  # 25-28: 4 dims
        ],
        axis=-1,
    )

    return features


# JIT-compiled version for performance
extract_amp_features_jit = jax.jit(extract_amp_features, static_argnames=("config",))


# =============================================================================
# Temporal Feature Buffer (v0.5.0: 3-frame history for discriminator)
# =============================================================================


class TemporalFeatureConfig(NamedTuple):
    """Configuration for temporal feature extraction.

    v0.5.0: Added temporal context for AMP discriminator.
    The discriminator can now see 2-3 frames of motion to detect:
    - Acceleration and jerk (smoothness indicators)
    - Gait phase continuity
    - Natural motion dynamics
    """

    num_frames: int = 3  # Number of frames in temporal window
    feature_dim: int = 29  # Per-frame feature dimension

    @property
    def temporal_dim(self) -> int:
        """Total dimension of temporal features (num_frames × feature_dim)."""
        return self.num_frames * self.feature_dim


def create_temporal_buffer(
    num_envs: int,
    config: TemporalFeatureConfig,
) -> jnp.ndarray:
    """Create initial temporal buffer filled with zeros.

    Args:
        num_envs: Number of parallel environments
        config: Temporal feature configuration

    Returns:
        buffer: Shape (num_envs, num_frames, feature_dim)
    """
    return jnp.zeros(
        (num_envs, config.num_frames, config.feature_dim),
        dtype=jnp.float32,
    )


def update_temporal_buffer(
    buffer: jnp.ndarray,
    new_features: jnp.ndarray,
) -> jnp.ndarray:
    """Update temporal buffer with new frame (shift left, add new at end).

    This implements a sliding window:
    - Old: [t-2, t-1, t]
    - New features for t+1
    - Result: [t-1, t, t+1]

    Args:
        buffer: Current buffer, shape (num_envs, num_frames, feature_dim)
        new_features: New features, shape (num_envs, feature_dim)

    Returns:
        Updated buffer, shape (num_envs, num_frames, feature_dim)
    """
    # Shift left (drop oldest frame at index 0)
    shifted = jnp.roll(buffer, shift=-1, axis=1)

    # Add new frame at the end
    return shifted.at[:, -1, :].set(new_features)


def get_temporal_features(
    buffer: jnp.ndarray,
) -> jnp.ndarray:
    """Flatten temporal buffer to feature vector.

    Args:
        buffer: Temporal buffer, shape (num_envs, num_frames, feature_dim)

    Returns:
        Flattened features, shape (num_envs, num_frames * feature_dim)
    """
    num_envs = buffer.shape[0]
    return buffer.reshape(num_envs, -1)


def add_temporal_context_to_reference(
    features: jnp.ndarray,
    num_frames: int = 3,
) -> jnp.ndarray:
    """Convert single-frame features to temporal features for reference data.

    Creates overlapping windows of consecutive frames:
    - Input: (N, feature_dim)
    - Output: (N - num_frames + 1, num_frames * feature_dim)

    The last `num_frames - 1` samples are lost because they don't have
    enough future frames to form a complete window.

    Args:
        features: Single-frame features, shape (N, feature_dim)
        num_frames: Number of frames per temporal window

    Returns:
        Temporal features, shape (N - num_frames + 1, num_frames * feature_dim)
    """
    N = features.shape[0]
    feature_dim = features.shape[1]

    # Number of valid windows
    num_windows = N - num_frames + 1

    if num_windows <= 0:
        raise ValueError(
            f"Not enough frames ({N}) for temporal window of size {num_frames}"
        )

    # Create indices for each window
    # window[i] contains frames [i, i+1, ..., i+num_frames-1]
    indices = jnp.arange(num_windows)[:, None] + jnp.arange(num_frames)

    # Gather frames and flatten each window
    windows = features[indices]  # (num_windows, num_frames, feature_dim)
    temporal_features = windows.reshape(
        num_windows, -1
    )  # (num_windows, num_frames * feature_dim)

    return temporal_features


# =============================================================================
# Running Statistics for Feature Normalization
# =============================================================================


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
