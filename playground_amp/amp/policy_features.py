"""Online AMP feature extraction from policy observations (JAX).

This module extracts motion features from policy observations for use with
the AMP discriminator during training. Features are designed to capture the
style of motion (joint angles, velocities, base height/orientation) while
being invariant to task-specific aspects (goal position, commands).

For offline reference data extraction, see ref_features.py.
For shared feature configuration, see feature_config.py.

Based on DeepMind AMP paper and 2024 best practices:
- Joint angles (root-relative)
- Joint velocities
- Root height + orientation
- Normalize features with running statistics
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from playground_amp.configs.feature_config import FeatureConfig


def estimate_foot_contacts_from_joints(
    joint_pos: jnp.ndarray,
    config: FeatureConfig,
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
    left_hip_pitch = joint_pos[..., config.left_hip_pitch_idx]
    left_knee = joint_pos[..., config.left_knee_pitch_idx]
    right_hip_pitch = joint_pos[..., config.right_hip_pitch_idx]
    right_knee = joint_pos[..., config.right_knee_pitch_idx]

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
    return jnp.stack(
        [left_contact, left_contact, right_contact, right_contact], axis=-1
    )


def extract_amp_features(
    obs: jnp.ndarray,
    config: FeatureConfig,
    root_height: jnp.ndarray,
    prev_joint_pos: jnp.ndarray,
    dt: float,
    contact_threshold_angle: float = 0.1,
    contact_knee_scale: float = 0.5,
    contact_min_confidence: float = 0.3,
    foot_contacts_override: jnp.ndarray = None,
    use_obs_joint_vel: bool = False,
) -> jnp.ndarray:
    """Extract motion features for discriminator from observation.

    v0.7.0: Simplified API - no fallback options.
    - All velocities in HEADING-LOCAL frame (Contract A)
    - Joint velocities: ALWAYS use finite differences (matches reference)
    - Foot contacts: ALWAYS use joint-based estimation (matches reference)
    - Ankle calibration: REMOVED (foot_mimic sites relocated in robot)

    v0.10.0: Industry Golden Rule - TRUE qvel support.
    - Added foot_contacts_override for physics reference data.
    - Added use_obs_joint_vel to use TRUE qvel from observation.
    - When using physics reference data, both features maintain parity
      because they use the same velocity source (MuJoCo qvel).

    Args:
        obs: Current observation (..., obs_dim=35)
             Layout: gravity(3), angvel(3), linvel(3), joint_pos(8),
             joint_vel(8), prev_action(8), velocity_cmd(1), padding(1)
             Note: angvel and linvel are already in heading-local frame
        config: Feature extraction configuration (REQUIRED)
        root_height: Root height from env (...,) in meters (REQUIRED)
        prev_joint_pos: Previous joint positions for finite diff (..., 8)
                        (REQUIRED if use_obs_joint_vel=False)
        dt: Time step for finite diff (e.g., 0.02 = 50Hz) (REQUIRED)
        contact_threshold_angle: Hip pitch threshold for contact detection (rad)
        contact_knee_scale: Knee angle for confidence scaling (rad)
        contact_min_confidence: Minimum confidence when hip indicates contact
        foot_contacts_override: Optional ground-truth contacts from physics (4 dims).
                                If provided, bypasses joint-based estimation.
        use_obs_joint_vel: If True, use joint velocities from observation (TRUE qvel
                           from MuJoCo). If False, compute via finite diff.
                           Default False for backward compatibility.
                           Industry golden rule: use TRUE qvel for physics reference.

    Returns:
        amp_features: Motion features (..., 27)
    """
    joint_pos = obs[..., config.joint_pos_start : config.joint_pos_end]  # 8 dims
    root_linvel = obs[..., config.root_linvel_start : config.root_linvel_end]  # 3 dims
    root_angvel = obs[..., config.root_angvel_start : config.root_angvel_end]  # 3 dims

    # Joint velocities: v0.10.0 - Industry Golden Rule
    # Option 1: Use TRUE qvel from observation (physics reference mode)
    # Option 2: Compute finite diff (GMR reference mode, backward compatible)
    if use_obs_joint_vel:
        # Use TRUE qvel from observation (what policy sees during training)
        # This matches physics reference data which also uses MuJoCo qvel
        joint_vel = obs[..., config.joint_vel_start : config.joint_vel_end]  # 8 dims
    else:
        # Finite diff (v0.7.0 default, matches GMR reference)
        if prev_joint_pos is None:
            raise ValueError(
                "prev_joint_pos is REQUIRED for finite difference velocity. "
                "Pass the previous frame's joint positions, or set use_obs_joint_vel=True."
            )
        joint_vel = (joint_pos - prev_joint_pos) / dt

    # Foot contacts: Use override if provided (v0.10.0 - physics reference support)
    # Otherwise, use joint-based estimation (v0.7.0 default)
    if foot_contacts_override is not None:
        # Use ground-truth contacts from physics simulation
        foot_contacts_feat = foot_contacts_override
    else:
        # Estimate from joint angles (matches GMR reference data)
        foot_contacts_feat = estimate_foot_contacts_from_joints(
            joint_pos,
            config,
            threshold_angle=contact_threshold_angle,
            knee_scale=contact_knee_scale,
            min_confidence=contact_min_confidence,
        )

    # Root linear velocity: optionally normalize to direction
    if config.normalize_velocity:
        root_linvel_feat = normalize_velocity(root_linvel)
    else:
        root_linvel_feat = root_linvel  # raw m/s

    # Root height: ensure shape is (N, 1) for concatenation
    if root_height.ndim == 0:
        root_height_feat = root_height[None]
    elif root_height.ndim == 1:
        root_height_feat = root_height[:, jnp.newaxis]
    else:
        root_height_feat = root_height

    # Use FeatureLayout for consistent ordering (SINGLE SOURCE OF TRUTH)
    # Layout respects drop flags from config
    layout = config.get_layout()
    ordered = layout.get_features(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_linvel=root_linvel_feat,
        root_angvel=root_angvel,
        root_height=root_height_feat,
        foot_contacts=foot_contacts_feat,
    )
    features = jnp.concatenate(ordered, axis=-1)

    return features


def extract_amp_features_batched(
    obs: jnp.ndarray,
    config: FeatureConfig,
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
    """Extract AMP features from observations (batched over steps and envs).

    Wraps extract_amp_features with double vmap for efficient batch processing.

    Args:
        obs: Observations, shape (num_steps, num_envs, obs_dim)
        config: Feature extraction configuration (REQUIRED)
        foot_contacts: Foot contacts from env, shape (num_steps, num_envs, 4)
        root_height: Root height from env, shape (num_steps, num_envs) (REQUIRED)
        prev_joint_pos: Previous joint positions, shape (num_steps, num_envs, num_joints)
        dt: Time step for finite difference calculation
        use_estimated_contacts: Use joint-based contact estimation
        use_finite_diff_vel: Use finite difference velocities
        contact_threshold_angle: Hip pitch threshold for contact detection (rad)
        contact_knee_scale: Knee angle for confidence scaling (rad)
        contact_min_confidence: Minimum confidence when hip indicates contact (0-1)

    Returns:
        AMP features, shape (num_steps, num_envs, feature_dim)
    """

    def extract_single(inputs):
        obs_single, fc_single, rh_single, prev_jp_single = inputs
        return extract_amp_features(
            obs_single,
            config=config,
            root_height=rh_single,
            prev_joint_pos=prev_jp_single,
            dt=dt,
            foot_contacts_override=fc_single if not use_estimated_contacts else None,
            use_obs_joint_vel=not use_finite_diff_vel,
            contact_threshold_angle=contact_threshold_angle,
            contact_knee_scale=contact_knee_scale,
            contact_min_confidence=contact_min_confidence,
        )

    return jax.vmap(jax.vmap(extract_single))(
        (obs, foot_contacts, root_height, prev_joint_pos)
    )



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
# Stage 0 Preprocessing Utilities (JAX)
# =============================================================================


def normalize_velocity(
    root_linvel: jnp.ndarray,
    min_speed_threshold: float = 0.05,
) -> jnp.ndarray:
    """Normalize root linear velocity to unit direction (JAX version).

    When speed is below threshold, returns zero vector to avoid noisy
    normalized direction from near-stationary states.

    Args:
        root_linvel: Root linear velocity (..., 3)
        min_speed_threshold: Below this speed (m/s), return zero vector

    Returns:
        Normalized velocity direction (..., 3), unit vectors or zeros
    """
    speed = jnp.linalg.norm(root_linvel, axis=-1, keepdims=True)
    is_moving = speed > min_speed_threshold
    return jnp.where(
        is_moving, root_linvel / (speed + 1e-8), jnp.zeros_like(root_linvel)
    )
