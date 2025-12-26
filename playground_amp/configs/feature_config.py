"""Shared AMP feature configuration for online and offline extraction.

This module defines the feature layout used by both:
- policy_features.py (online JAX extraction from policy observations)
- ref_features.py (offline NumPy extraction from reference data)

Configuration is derived from robot_config.yaml.

IMPORTANT: The FeatureLayout class is the SINGLE SOURCE OF TRUTH for feature
ordering and dimensions. Both online and offline extraction must use this
to ensure feature parity.
"""

from __future__ import annotations

from typing import NamedTuple, Dict, List, Tuple

from playground_amp.configs.training_config import (
    get_robot_config,
    get_training_config,
    RobotConfig,
    TrainingConfig,
)


# =============================================================================
# Feature Layout - SINGLE SOURCE OF TRUTH (Private, created only by FeatureConfig)
# =============================================================================


class _FeatureLayout:
    """Centralized feature layout definition (PRIVATE - use FeatureConfig.get_layout()).

    This class defines the AMP feature vector structure used by both:
    - Online extraction (policy_features.py, JAX)
    - Offline extraction (ref_features.py, NumPy)

    SINGLE SOURCE OF TRUTH: The COMPONENT_DEFS tuple defines everything.
    Drop flags are set at construction and immutable.

    NOTE: Do not instantiate directly. Use FeatureConfig.get_layout() or
    FeatureConfig.get_full_layout() instead.
    """

    # Fixed dimensions (not robot-dependent)
    ROOT_LINVEL_DIM = 3
    ROOT_ANGVEL_DIM = 3
    ROOT_HEIGHT_DIM = 1
    FOOT_CONTACTS_DIM = 4

    # SINGLE SOURCE OF TRUTH: Component definitions
    # Format: (name, dim_attr_or_value, droppable)
    # dim_attr_or_value: either a class constant name (str) or "num_joints" for robot-dependent
    COMPONENT_DEFS = [
        ("joint_pos", "num_joints", False),
        ("joint_vel", "num_joints", False),
        ("root_linvel", "ROOT_LINVEL_DIM", False),
        ("root_angvel", "ROOT_ANGVEL_DIM", False),
        ("root_height", "ROOT_HEIGHT_DIM", True),  # droppable
        ("foot_contacts", "FOOT_CONTACTS_DIM", True),  # droppable
    ]

    def __init__(self, num_joints: int, drop: Dict[str, bool]):
        """Initialize feature layout (PRIVATE - use FeatureConfig methods).

        Args:
            num_joints: Number of actuated joints (from FeatureConfig)
            drop: Dict mapping droppable component names to bool.
                  e.g., {} for full features, {"root_height": True} to drop height.
                  Immutable after construction.
        """
        self.num_joints = num_joints
        self._drop = drop.copy()  # Immutable copy
        self._build_layout()
        self._validate_drop()

    def _get_component_dim(self, dim_attr: str) -> int:
        """Get dimension for a component."""
        if dim_attr == "num_joints":
            return self.num_joints
        return getattr(self, dim_attr)

    def _build_layout(self):
        """Build indices and dimensions from COMPONENT_DEFS."""
        self._dims = {}
        self._order = []
        self._droppable = {}

        for name, dim_attr, droppable in self.COMPONENT_DEFS:
            dim = self._get_component_dim(dim_attr)
            self._dims[name] = dim
            self._order.append(name)
            self._droppable[name] = droppable

    def _validate_drop(self):
        """Validate drop flags at construction time."""
        for name in self._drop.keys():
            if name not in self._droppable:
                raise ValueError(f"Unknown component in drop: {name}")
            if not self._droppable[name]:
                raise ValueError(f"Component '{name}' is not droppable")

    def get_feature_list(self) -> List[str]:
        """Get ordered list of active feature names (respecting drops)."""
        return [
            name for name in self._order
            if not (self._droppable[name] and self._drop.get(name, False))
        ]

    def get_features(self, **components) -> List:
        """Get feature arrays in correct order (respecting drops).

        Validates that:
        - All required (non-dropped) components are provided
        - Component dimensions match expected dimensions

        Args:
            **components: Keyword arguments mapping component names to arrays.
                          e.g., joint_pos=arr1, joint_vel=arr2, ...
                          Dropped components can be omitted.

        Returns:
            List of arrays in correct order for concatenation

        Raises:
            ValueError: If missing required components or wrong dimensions

        Example:
            ordered = layout.get_features(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                root_linvel=root_linvel,
                root_angvel=root_angvel,
                root_height=root_height,
                foot_contacts=foot_contacts,
            )
            features = np.concatenate(ordered, axis=-1)
        """
        result = []

        for name, dim_attr, drop in self.COMPONENT_DEFS:
            # Skip dropped components
            if drop and self._drop.get(name, False):
                continue

            # Check component is provided
            if name not in components:
                raise ValueError(f"Missing required component: {name}")

            arr = components[name]
            expected_dim = self._get_component_dim(dim_attr)

            # Validate dimension
            if arr.ndim == 1:
                if expected_dim != 1:
                    raise ValueError(
                        f"Component '{name}' is 1D array but expected dim={expected_dim} ({dim_attr}). "
                        f"Should be shape (N, {expected_dim})."
                    )
                actual_dim = 1
            else:
                actual_dim = arr.shape[-1]

            if actual_dim != expected_dim:
                raise ValueError(
                    f"Component '{name}' has wrong dimension: "
                    f"expected {expected_dim} ({dim_attr}), got {actual_dim}"
                )

            result.append(arr)

        return result

    @property
    def dim(self) -> int:
        """Feature dimension (respecting drops)."""
        return sum(self._dims[name] for name in self.get_feature_list())


# =============================================================================
# Feature Config - Runtime Configuration
# =============================================================================


class FeatureConfig(NamedTuple):
    """Configuration for AMP feature extraction.

    Defines which parts of the observation to use for discriminator.
    Indices are derived from robot_config.yaml observation_indices.

    NOTE: Root linear velocity uses RAW values (m/s), NOT normalized.
    v0.6.6 fix: Both policy and reference must use raw velocity for
    feature parity. Normalization was causing discriminator collapse.

    v0.6.2: Added Golden Rule parameters for Mathematical Parity.
    v0.6.3: Added feature cleaning (velocity filter, ankle calibration).
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

    # v0.6.3: Ankle indices for calibration offset
    left_ankle_pitch_idx: int  # Index of left_ankle_pitch in joint_pos
    right_ankle_pitch_idx: int  # Index of right_ankle_pitch in joint_pos

    # Whether to use transition features (current + next)
    # Fields with defaults must come after fields without defaults
    use_transition_features: bool = False

    # Stage 0 feature dropping (prevent discriminator shortcuts)
    drop_contacts: bool = False  # Drop foot contacts (4 dims) from discriminator
    drop_height: bool = False  # Drop root height (1 dim) from discriminator
    normalize_velocity: bool = False  # Normalize root_linvel to unit direction

    def get_layout(self) -> _FeatureLayout:
        """Get FeatureLayout from this config.

        Returns:
            FeatureLayout instance with num_joints and drop flags from this config
        """
        drop = {}
        if self.drop_height:
            drop["root_height"] = True
        if self.drop_contacts:
            drop["foot_contacts"] = True

        return _FeatureLayout(num_joints=self.num_actuated_joints, drop=drop)

    @property
    def feature_dim(self) -> int:
        """Get feature dimension (respecting drop flags)."""
        return self.get_layout().dim



def create_config_from_robot(
    robot_config: RobotConfig,
    drop_contacts: bool = False,
    drop_height: bool = False,
    normalize_velocity: bool = False,
) -> FeatureConfig:
    """Create FeatureConfig from robot configuration.

    Args:
        robot_config: Robot configuration (REQUIRED).
        drop_contacts: v0.8.0 - Drop foot contacts from features (4 dims)
        drop_height: v0.8.0 - Drop root height from features (1 dim)
        normalize_velocity: v0.8.0 - Normalize root_linvel to unit direction

    Returns:
        FeatureConfig with indices from robot config
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

    return FeatureConfig(
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
        # v0.6.3: Ankle indices for calibration offset
        left_ankle_pitch_idx=get_joint_index("left_ankle_pitch"),
        right_ankle_pitch_idx=get_joint_index("right_ankle_pitch"),
        # v0.8.0: Feature dropping (prevent discriminator shortcuts)
        drop_contacts=drop_contacts,
        drop_height=drop_height,
        normalize_velocity=normalize_velocity,
    )


def get_feature_config() -> FeatureConfig:
    """Get AMP feature configuration from cached configs.

    Reads v0.8.0 feature drop flags from training config if loaded,
    otherwise uses defaults (no dropping).

    Raises:
        RuntimeError: If robot_config hasn't been loaded yet.
    """
    robot_config = get_robot_config()

    # Try to read v0.8.0 flags from training config
    try:
        training_config = get_training_config()
        # v0.10.0: Flags are now under training_config.amp (MutableAMPConfig)
        drop_contacts = training_config.amp.drop_contacts
        drop_height = training_config.amp.drop_height
        normalize_velocity = training_config.amp.normalize_velocity
    except (RuntimeError, AttributeError):
        # Training config not loaded yet, use defaults
        drop_contacts = False
        drop_height = False
        normalize_velocity = False

    return create_config_from_robot(
        robot_config,
        drop_contacts=drop_contacts,
        drop_height=drop_height,
        normalize_velocity=normalize_velocity,
    )
