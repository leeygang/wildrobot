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

    v0.6.2: Added Golden Rule parameters for Mathematical Parity.
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

    # v0.6.2: Joint indices for contact estimation (from robot_config actuator order)
    # These map actuator names to indices in joint_pos array
    left_hip_pitch_idx: int  # Index of left_hip_pitch in joint_pos
    left_knee_pitch_idx: int  # Index of left_knee_pitch in joint_pos
    right_hip_pitch_idx: int  # Index of right_hip_pitch in joint_pos
    right_knee_pitch_idx: int  # Index of right_knee_pitch in joint_pos

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

    # v0.6.2: Derive joint indices from actuator names
    # These are used for contact estimation to match reference data
    actuator_names = robot_config.actuator_names

    def get_joint_index(name: str) -> int:
        """Get index of joint in actuator_names list."""
        try:
            return actuator_names.index(name)
        except ValueError:
            raise ValueError(
                f"Joint '{name}' not found in actuator_names: {actuator_names}"
            )

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
        # v0.6.2: Joint indices for contact estimation (derived from robot_config)
        left_hip_pitch_idx=get_joint_index("left_hip_pitch"),
        left_knee_pitch_idx=get_joint_index("left_knee_pitch"),
        right_hip_pitch_idx=get_joint_index("right_hip_pitch"),
        right_knee_pitch_idx=get_joint_index("right_knee_pitch"),
    )


def get_amp_config() -> AMPFeatureConfig:
    """Get AMP feature configuration from cached robot config.

    Raises:
        RuntimeError: If robot_config hasn't been loaded yet.
    """
    robot_config = get_robot_config()
    return create_amp_config_from_robot(robot_config)


def estimate_foot_contacts_from_joints(
    joint_pos: jnp.ndarray,
    config: AMPFeatureConfig,
    threshold_angle: float = 0.1,
    knee_scale: float = 0.5,
    min_confidence: float = 0.3,
) -> jnp.ndarray:
    """Estimate foot contacts from joint positions (matches reference data).

    This uses the same heuristic as the reference data generation:
    - When hip pitch is negative (leg behind body), foot is likely in contact
    - When knee is more bent, contact confidence decreases

    v0.6.2: All parameters are now configurable from training config.
    Joint indices are derived from robot_config.yaml.

    Args:
        joint_pos: Joint positions (9,)
        config: AMP feature config with joint indices
        threshold_angle: Hip pitch angle threshold (rad). When hip_pitch < threshold,
            the leg is behind the body and foot is likely in contact. Default 0.1 rad (~6°).
        knee_scale: Knee angle at which confidence starts decreasing (rad).
            When |knee_angle| > knee_scale, the leg is bent and less likely to have
            ground contact. Default 0.5 rad (~28°). Confidence is linearly scaled
            from 1.0 at knee=0 to min_confidence at knee=knee_scale.
        min_confidence: Minimum confidence value (0-1) when hip indicates contact
            but knee is bent. Default 0.3 (30%). Prevents complete contact loss
            during bent-knee stance phases.

    Returns:
        Estimated foot contacts (4,) - [left_toe, left_heel, right_toe, right_heel]
    """
    # Joint indices from config (derived from robot_config.yaml actuator order)
    left_hip_pitch = joint_pos[config.left_hip_pitch_idx]
    left_knee = joint_pos[config.left_knee_pitch_idx]
    right_hip_pitch = joint_pos[config.right_hip_pitch_idx]
    right_knee = joint_pos[config.right_knee_pitch_idx]

    # Left foot contact estimation:
    # - Base contact: hip_pitch < threshold (leg behind body)
    # - Confidence scaling: reduces when knee is bent
    #   - At knee=0: confidence = 1.0 (fully extended)
    #   - At knee=knee_scale: confidence = min_confidence
    #   - Formula: clip(1.0 - |knee| / knee_scale, min_confidence, 1.0)
    left_contact = (left_hip_pitch < threshold_angle).astype(jnp.float32)
    left_contact = left_contact * jnp.clip(
        1.0 - jnp.abs(left_knee) / knee_scale, min_confidence, 1.0
    )

    # Right foot contact (same logic)
    right_contact = (right_hip_pitch < threshold_angle).astype(jnp.float32)
    right_contact = right_contact * jnp.clip(
        1.0 - jnp.abs(right_knee) / knee_scale, min_confidence, 1.0
    )

    # Return 4 contacts: [left_toe, left_heel, right_toe, right_heel]
    # Toe and heel share the same contact estimate for each foot
    return jnp.array([left_contact, left_contact, right_contact, right_contact])


def extract_amp_features(
    obs: jnp.ndarray,
    config: AMPFeatureConfig,
    foot_contacts: jnp.ndarray,
    root_height: jnp.ndarray,
    prev_joint_pos: jnp.ndarray,
    dt: float,
    use_estimated_contacts: bool,
    use_finite_diff_vel: bool,
    contact_threshold_angle: float = 0.1,
    contact_knee_scale: float = 0.5,
    contact_min_confidence: float = 0.3,
) -> jnp.ndarray:
    """Extract motion features for discriminator from observation.

    Maps observation to 29-dim AMP feature format:
    - Joint positions (0-8): 9 dims
    - Joint velocities (9-17): 9 dims
    - Root linear velocity DIRECTION (18-20): 3 dims (normalized unit vector)
    - Root angular velocity (21-23): 3 dims
    - Root height (24): 1 dim (actual height in meters)
    - Foot contacts (25-28): 4 dims

    v0.6.2 GOLDEN RULES for Mathematical Parity:
    - foot_contacts: Use joint-based estimation (matches reference data)
    - joint_vel: Use finite differences (matches reference data)

    These settings ensure the discriminator cannot "cheat" by detecting
    calculation method differences between reference and policy data.

    Args:
        obs: Current observation (..., obs_dim=38)
        config: Feature extraction configuration (REQUIRED)
        foot_contacts: Physics-based foot contacts (only used if use_estimated_contacts=False)
        root_height: Root height from env (...,) in meters (REQUIRED)
        prev_joint_pos: Previous joint positions for finite difference velocity (..., 9)
        dt: Time step for finite difference calculation (e.g., 0.02 = 50Hz) (REQUIRED)
        use_estimated_contacts: If True, estimate contacts from joint angles (REQUIRED)
        use_finite_diff_vel: If True, use finite difference velocities (REQUIRED)
        contact_threshold_angle: Hip pitch threshold for contact detection (rad)
        contact_knee_scale: Knee angle for confidence scaling (rad)
        contact_min_confidence: Minimum confidence when hip indicates contact (0-1)

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
    root_linvel = obs[..., config.root_linvel_start : config.root_linvel_end]  # 3 dims
    root_angvel = obs[..., config.root_angvel_start : config.root_angvel_end]  # 3 dims

    # === GOLDEN RULE FIX 1: Joint Velocities ===
    # Use finite differences to match reference data calculation method
    if use_finite_diff_vel and prev_joint_pos is not None:
        # Finite difference velocity (matches AMASS reference processing)
        joint_vel = (joint_pos - prev_joint_pos) / dt
    else:
        # Fallback to analytical velocities from MuJoCo (not recommended for AMP)
        joint_vel = obs[..., config.joint_vel_start : config.joint_vel_end]

    # === GOLDEN RULE FIX 2: Foot Contacts ===
    # Use joint-based estimation to match reference data calculation method
    if use_estimated_contacts:
        # Estimate from joint angles (matches AMASS reference processing)
        foot_contacts_feat = estimate_foot_contacts_from_joints(
            joint_pos,
            config,
            threshold_angle=contact_threshold_angle,
            knee_scale=contact_knee_scale,
            min_confidence=contact_min_confidence,
        )
    else:
        # Use physics contacts (may cause distribution mismatch)
        if foot_contacts is None:
            raise ValueError(
                "foot_contacts is required when use_estimated_contacts=False"
            )
        foot_contacts_feat = foot_contacts

    # Root height: ensure shape is (..., 1) for concatenation
    if root_height is None:
        raise ValueError("root_height is required")
    if root_height.ndim == 0 or (root_height.ndim > 0 and root_height.shape[-1] != 1):
        root_height_feat = root_height[..., jnp.newaxis]
    else:
        root_height_feat = root_height

    # Normalize root linear velocity to unit direction vector
    # This removes speed (magnitude) while preserving direction
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
            joint_vel,  # 9-17: 9 dims (finite diff or analytical)
            root_linvel_dir,  # 18-20: 3 dims (normalized direction)
            root_angvel,  # 21-23: 3 dims
            root_height_feat,  # 24: 1 dim (actual height in meters)
            foot_contacts_feat,  # 25-28: 4 dims (estimated or physics)
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
