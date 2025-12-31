"""Unit tests for Control Abstraction Layer (CAL).

v0.11.0: Tests for the Control Abstraction Layer module.

These tests verify:
1. CAL initialization from MuJoCo model and config
2. Policy action to ctrl transformation (with symmetry correction)
3. Physical angles to ctrl (passthrough)
4. Normalized joint position/velocity getters
5. Symmetry handling (mirror_sign)
6. Command generation for Sim2Real
7. Extended state (foot, root) - v0.11.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground_amp.cal.cal import ControlAbstractionLayer
from playground_amp.cal.specs import (
    ActuatorSpec,
    ControlCommand,
    FootSpec,
    JointSpec,
    Pose3D,
    RootSpec,
    Velocity3D,
)
from playground_amp.cal.types import ActuatorType, CoordinateFrame


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cal_instance(mj_model, robot_config):
    """Create CAL instance for testing."""
    return ControlAbstractionLayer(mj_model, robot_config)


# =============================================================================
# Basic Initialization Tests
# =============================================================================


class TestCALInitialization:
    """Tests for CAL initialization."""

    def test_cal_initializes_with_correct_num_actuators(
        self, cal_instance, robot_config
    ):
        """CAL should have same number of actuators as robot config."""
        assert cal_instance.num_actuators == robot_config.action_dim

    def test_cal_has_joint_specs(self, cal_instance):
        """CAL should have joint specifications."""
        assert len(cal_instance.joint_specs) == cal_instance.num_actuators
        for spec in cal_instance.joint_specs:
            assert isinstance(spec, JointSpec)
            assert spec.name is not None

    def test_cal_has_actuator_specs(self, cal_instance):
        """CAL should have actuator specifications."""
        assert len(cal_instance.actuator_specs) == cal_instance.num_actuators
        for spec in cal_instance.actuator_specs:
            assert isinstance(spec, ActuatorSpec)
            assert spec.actuator_type == ActuatorType.POSITION

    def test_joint_spec_names_match_config(self, cal_instance, robot_config):
        """Joint spec names should match actuator names from config."""
        cal_names = [spec.name for spec in cal_instance.joint_specs]
        config_names = robot_config.actuator_names
        assert cal_names == config_names


# =============================================================================
# Action Transformation Tests
# =============================================================================


class TestActionTransformation:
    """Tests for action→ctrl transformation."""

    def test_policy_action_to_ctrl_shape(self, cal_instance):
        """policy_action_to_ctrl should return correct shape."""
        action = jnp.zeros(cal_instance.num_actuators)
        ctrl = cal_instance.policy_action_to_ctrl(action)
        assert ctrl.shape == (cal_instance.num_actuators,)

    def test_policy_action_zero_maps_to_center(self, cal_instance):
        """Action=0 should map to center of ctrl range."""
        action = jnp.zeros(cal_instance.num_actuators)
        ctrl = cal_instance.policy_action_to_ctrl(action)

        for i, spec in enumerate(cal_instance.actuator_specs):
            expected_center = spec.ctrl_center
            # Allow small tolerance for floating point
            assert (
                jnp.abs(ctrl[i] - expected_center) < 1e-5
            ), f"Joint {spec.name}: expected center {expected_center}, got {ctrl[i]}"

    def test_policy_action_plus_one_maps_to_max(self, cal_instance):
        """Action=+1 should map to max of ctrl range (for mirror_sign=+1 joints)."""
        action = jnp.ones(cal_instance.num_actuators)
        ctrl = cal_instance.policy_action_to_ctrl(action)

        for i, spec in enumerate(cal_instance.actuator_specs):
            joint_spec = cal_instance.joint_specs[i]
            if joint_spec.mirror_sign > 0:
                # For positive mirror_sign, action=+1 → ctrl_max
                expected = spec.ctrl_range[1]
                assert (
                    jnp.abs(ctrl[i] - expected) < 1e-5
                ), f"Joint {spec.name}: expected max {expected}, got {ctrl[i]}"
            else:
                # For negative mirror_sign, action=+1 → ctrl_min
                expected = spec.ctrl_range[0]
                assert (
                    jnp.abs(ctrl[i] - expected) < 1e-5
                ), f"Joint {spec.name}: expected min {expected}, got {ctrl[i]}"

    def test_physical_angles_to_ctrl_is_passthrough(self, cal_instance):
        """physical_angles_to_ctrl should pass angles through unchanged."""
        angles = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
        ctrl = cal_instance.physical_angles_to_ctrl(angles)
        np.testing.assert_array_almost_equal(ctrl, angles)

    def test_ctrl_to_policy_action_is_inverse(self, cal_instance):
        """ctrl_to_policy_action should be inverse of policy_action_to_ctrl."""
        # Generate random action in [-1, 1]
        original_action = jnp.array([0.5, -0.3, 0.7, -0.1, 0.2, -0.8, 0.4, -0.6])
        ctrl = cal_instance.policy_action_to_ctrl(original_action)
        recovered_action = cal_instance.ctrl_to_policy_action(ctrl)

        np.testing.assert_array_almost_equal(
            recovered_action,
            original_action,
            decimal=5,
            err_msg="ctrl_to_policy_action should be inverse of policy_action_to_ctrl",
        )


# =============================================================================
# Symmetry Tests
# =============================================================================


class TestSymmetry:
    """Tests for symmetry handling (mirror_sign)."""

    def test_left_right_symmetry_with_same_action(self, cal_instance, robot_config):
        """Same action for left/right should produce symmetric physical motion.

        This is the key test for mirror_sign correctness:
        - When policy outputs action[left_hip]=0.5 and action[right_hip]=0.5
        - The physical ctrl values should be symmetric (same physical motion)

        For joints with inverted ranges (e.g., left_hip_pitch vs right_hip_pitch),
        the mirror_sign correction should handle this automatically.
        """
        # Create action with same value for left and right hip pitch
        action = jnp.zeros(cal_instance.num_actuators)

        # Find left and right hip pitch indices
        left_hip_idx = robot_config.get_actuator_index("left_hip_pitch")
        right_hip_idx = robot_config.get_actuator_index("right_hip_pitch")

        # Set same action for both
        action = action.at[left_hip_idx].set(0.5)
        action = action.at[right_hip_idx].set(0.5)

        ctrl = cal_instance.policy_action_to_ctrl(action)

        # Get joint specs
        left_spec = cal_instance.joint_specs[left_hip_idx]
        right_spec = cal_instance.joint_specs[right_hip_idx]

        # Normalize both ctrl values to their respective ranges
        left_normalized = (
            ctrl[left_hip_idx] - left_spec.range_center
        ) / left_spec.range_span
        right_normalized = (
            ctrl[right_hip_idx] - right_spec.range_center
        ) / right_spec.range_span

        # For mirrored joint ranges, symmetric motion means:
        # - Same action value (0.5) produces opposite normalized positions
        # - left_normalized should equal -right_normalized
        # This is because mirror_sign flips the action for mirrored joints
        np.testing.assert_almost_equal(
            left_normalized,
            -right_normalized,  # Explicitly verify signs are opposite
            decimal=5,
            err_msg=(
                f"Symmetric action should produce opposite normalized positions. "
                f"Expected left ({left_normalized}) == -right ({-right_normalized})"
            ),
        )

    def test_mirror_sign_values(self, cal_instance, robot_config):
        """Verify mirror_sign values are set correctly from config."""
        for i, joint_spec in enumerate(cal_instance.joint_specs):
            config_entry = robot_config.actuated_joints[i]
            expected_mirror_sign = config_entry.get("mirror_sign", 1.0)
            assert joint_spec.mirror_sign == expected_mirror_sign, (
                f"Joint {joint_spec.name}: expected mirror_sign={expected_mirror_sign}, "
                f"got {joint_spec.mirror_sign}"
            )

    def test_get_symmetric_action_swaps_left_right(self, cal_instance, robot_config):
        """get_symmetric_action should swap left/right joint values."""
        # Create asymmetric action
        action = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        mirrored = cal_instance.get_symmetric_action(action, swap_left_right=True)

        # Check that left and right are swapped
        left_hip_idx = robot_config.get_actuator_index("left_hip_pitch")
        right_hip_idx = robot_config.get_actuator_index("right_hip_pitch")

        assert (
            mirrored[left_hip_idx] == action[right_hip_idx]
        ), "Left should have right's value"
        assert (
            mirrored[right_hip_idx] == action[left_hip_idx]
        ), "Right should have left's value"


# =============================================================================
# State Access Tests
# =============================================================================


class TestStateAccess:
    """Tests for normalized state access methods."""

    def test_get_joint_positions_normalized_shape(self, cal_instance, mj_model):
        """get_joint_positions(normalize=True) should return correct shape."""
        qpos = jnp.array(mj_model.qpos0)
        normalized = cal_instance.get_joint_positions(qpos, normalize=True)
        assert normalized.shape == (cal_instance.num_actuators,)

    def test_get_joint_positions_normalized_range(self, cal_instance, mj_model):
        """Normalized positions should be in [-1, 1] range for default pose."""
        qpos = jnp.array(mj_model.qpos0)
        normalized = cal_instance.get_joint_positions(qpos, normalize=True)

        # Default pose should give values within [-1, 1] (or close to it)
        assert jnp.all(normalized >= -1.5), "Normalized positions should be >= -1"
        assert jnp.all(normalized <= 1.5), "Normalized positions should be <= 1"

    def test_get_joint_positions_physical_matches_qpos(self, cal_instance, mj_model):
        """Physical positions should extract correct values from qpos."""
        qpos = jnp.array(mj_model.qpos0)
        physical = cal_instance.get_joint_positions(qpos, normalize=False)

        # Verify shape
        assert physical.shape == (cal_instance.num_actuators,)

        # Verify values match expected qpos addresses
        for i, spec in enumerate(cal_instance.joint_specs):
            expected = qpos[spec.qpos_addr]
            assert (
                physical[i] == expected
            ), f"Joint {spec.name}: expected {expected}, got {physical[i]}"

    def test_get_joint_velocities_normalized_shape(self, cal_instance, mj_model):
        """get_joint_velocities(normalize=True) should return correct shape."""
        qvel = jnp.zeros(mj_model.nv)
        normalized = cal_instance.get_joint_velocities(qvel, normalize=True)
        assert normalized.shape == (cal_instance.num_actuators,)

    def test_get_joint_velocities_physical_shape(self, cal_instance, mj_model):
        """get_joint_velocities(normalize=False) should return correct shape."""
        qvel = jnp.zeros(mj_model.nv)
        physical = cal_instance.get_joint_velocities(qvel, normalize=False)
        assert physical.shape == (cal_instance.num_actuators,)


# =============================================================================
# Sim2Real Command Tests
# =============================================================================


class TestSim2RealCommands:
    """Tests for Sim2Real command generation."""

    def test_policy_action_to_command_returns_control_command(self, cal_instance):
        """policy_action_to_command should return ControlCommand."""
        action = jnp.zeros(cal_instance.num_actuators)
        cmd = cal_instance.policy_action_to_command(action)
        assert isinstance(cmd, ControlCommand)

    def test_policy_action_to_command_has_position(self, cal_instance):
        """ControlCommand should have target_position."""
        action = jnp.zeros(cal_instance.num_actuators)
        cmd = cal_instance.policy_action_to_command(action)
        assert cmd.target_position is not None
        assert cmd.target_position.shape == (cal_instance.num_actuators,)

    def test_policy_action_to_command_no_velocity_without_qpos(self, cal_instance):
        """ControlCommand should have no velocity when current_qpos not provided."""
        action = jnp.zeros(cal_instance.num_actuators)
        cmd = cal_instance.policy_action_to_command(action)
        assert cmd.target_velocity is None

    def test_policy_action_to_command_with_velocity(self, cal_instance, mj_model):
        """ControlCommand should include velocity when current_qpos is provided."""
        action = jnp.array([0.5] * cal_instance.num_actuators)
        qpos = jnp.array(mj_model.qpos0)
        cmd = cal_instance.policy_action_to_command(action, current_qpos=qpos)
        assert cmd.target_velocity is not None
        assert cmd.target_velocity.shape == (cal_instance.num_actuators,)


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for validation methods."""

    def test_validate_action_accepts_valid(self, cal_instance):
        """validate_action should accept actions in [-1, 1]."""
        valid_action = jnp.array([0.5, -0.5, 0.0, 1.0, -1.0, 0.3, -0.7, 0.9])
        assert cal_instance.validate_action(valid_action)

    def test_validate_action_rejects_invalid(self, cal_instance):
        """validate_action should reject actions outside [-1, 1]."""
        invalid_action = jnp.array([1.5, -0.5, 0.0, 1.0, -1.0, 0.3, -0.7, 0.9])
        assert not cal_instance.validate_action(invalid_action)

    def test_get_ctrl_for_default_pose(self, cal_instance):
        """get_ctrl_for_default_pose should return ctrl for keyframe pose."""
        default_ctrl = cal_instance.get_ctrl_for_default_pose()
        assert default_ctrl.shape == (cal_instance.num_actuators,)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_actions(self, cal_instance):
        """Extreme actions [-1, 1] should produce valid ctrl."""
        action_min = jnp.full(cal_instance.num_actuators, -1.0)
        action_max = jnp.full(cal_instance.num_actuators, 1.0)

        ctrl_min = cal_instance.policy_action_to_ctrl(action_min)
        ctrl_max = cal_instance.policy_action_to_ctrl(action_max)

        # Both should be finite
        assert jnp.all(jnp.isfinite(ctrl_min))
        assert jnp.all(jnp.isfinite(ctrl_max))

    def test_joint_spec_properties(self, cal_instance):
        """JointSpec computed properties should be correct."""
        for spec in cal_instance.joint_specs:
            # range_center = (min + max) / 2
            expected_center = (spec.range_min + spec.range_max) / 2
            assert abs(spec.range_center - expected_center) < 1e-10

            # range_span = (max - min) / 2
            expected_span = (spec.range_max - spec.range_min) / 2
            assert abs(spec.range_span - expected_span) < 1e-10

    def test_actuator_spec_properties(self, cal_instance):
        """ActuatorSpec computed properties should be correct."""
        for spec in cal_instance.actuator_specs:
            # ctrl_center = (min + max) / 2
            expected_center = (spec.ctrl_range[0] + spec.ctrl_range[1]) / 2
            assert abs(spec.ctrl_center - expected_center) < 1e-10

            # ctrl_span = (max - min) / 2
            expected_span = (spec.ctrl_range[1] - spec.ctrl_range[0]) / 2
            assert abs(spec.ctrl_span - expected_span) < 1e-10


# =============================================================================
# Extended State Tests (v0.11.0)
# =============================================================================


@pytest.fixture(scope="function")
def cal_with_extended_state(mj_model, robot_config):
    """Create CAL instance with extended state initialized.

    Note: With v0.11.0, extended state is initialized in __init__,
    and contact_scale is read from robot_config.feet.contact_scale.
    """
    return ControlAbstractionLayer(mj_model, robot_config)


@pytest.fixture(scope="function")
def mjx_data(mj_model):
    """Create MJX data for testing extended state methods."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    # Convert to MJX
    mjx_model = mjx.put_model(mj_model)
    data = mjx.put_data(mj_model, mj_data)
    return mjx.forward(mjx_model, data)


class TestExtendedStateInit:
    """Tests for extended state initialization."""

    def test_init_extended_state_creates_foot_specs(self, cal_with_extended_state):
        """init_extended_state should create FootSpec list."""
        foot_specs = cal_with_extended_state.foot_specs
        assert len(foot_specs) == 2  # left and right foot
        assert all(isinstance(s, FootSpec) for s in foot_specs)

    def test_foot_specs_have_correct_names(self, cal_with_extended_state):
        """FootSpecs should have correct foot names."""
        foot_specs = cal_with_extended_state.foot_specs
        names = [s.name for s in foot_specs]
        assert "left_foot" in names
        assert "right_foot" in names

    def test_foot_specs_have_symmetry_pairs(self, cal_with_extended_state):
        """FootSpecs should have symmetry pair references."""
        foot_specs = cal_with_extended_state.foot_specs
        for spec in foot_specs:
            assert spec.symmetry_pair is not None

        # Left foot should pair with right foot
        left_foot = next(s for s in foot_specs if s.name == "left_foot")
        right_foot = next(s for s in foot_specs if s.name == "right_foot")
        assert left_foot.symmetry_pair == "right_foot"
        assert right_foot.symmetry_pair == "left_foot"

    def test_init_extended_state_creates_root_spec(self, cal_with_extended_state):
        """init_extended_state should create RootSpec."""
        root_spec = cal_with_extended_state.root_spec
        assert root_spec is not None
        assert isinstance(root_spec, RootSpec)

    def test_root_spec_has_sensor_names(self, cal_with_extended_state):
        """RootSpec should have sensor names from config."""
        root_spec = cal_with_extended_state.root_spec
        assert root_spec.orientation_sensor is not None
        assert root_spec.gyro_sensor is not None
        assert root_spec.accel_sensor is not None


class TestFootState:
    """Tests for foot state access methods."""

    def test_get_geom_based_foot_contacts_shape_normalized(
        self, cal_with_extended_state, mjx_data
    ):
        """get_geom_based_foot_contacts(normalize=True) should return (4,) array."""
        contacts = cal_with_extended_state.get_geom_based_foot_contacts(
            mjx_data, normalize=True
        )
        assert contacts.shape == (4,)

    def test_get_geom_based_foot_contacts_shape_raw(
        self, cal_with_extended_state, mjx_data
    ):
        """get_geom_based_foot_contacts(normalize=False) should return (4,) array."""
        contacts = cal_with_extended_state.get_geom_based_foot_contacts(
            mjx_data, normalize=False
        )
        assert contacts.shape == (4,)

    def test_get_geom_based_foot_contacts_normalized_range(
        self, cal_with_extended_state, mjx_data
    ):
        """Normalized foot contacts should be in [0, 1] range."""
        contacts = cal_with_extended_state.get_geom_based_foot_contacts(
            mjx_data, normalize=True
        )
        # tanh output is in (-1, 1), but contact forces are positive, so [0, 1)
        assert jnp.all(contacts >= 0.0)
        assert jnp.all(contacts <= 1.0)

    def test_get_foot_positions_shape(self, cal_with_extended_state, mjx_data):
        """get_foot_positions should return tuple of (3,) arrays."""
        left_pos, right_pos = cal_with_extended_state.get_foot_positions(
            mjx_data, normalize=False
        )
        assert left_pos.shape == (3,)
        assert right_pos.shape == (3,)

    def test_get_foot_positions_world_frame(self, cal_with_extended_state, mjx_data):
        """get_foot_positions in WORLD frame should return world coordinates."""
        left_pos, right_pos = cal_with_extended_state.get_foot_positions(
            mjx_data,
            normalize=False,
            frame=CoordinateFrame.WORLD,
        )
        # Positions should be finite
        assert jnp.all(jnp.isfinite(left_pos))
        assert jnp.all(jnp.isfinite(right_pos))

    def test_get_foot_clearances_shape(self, cal_with_extended_state, mjx_data):
        """get_foot_clearances should return tuple of scalar arrays."""
        left_clearance, right_clearance = cal_with_extended_state.get_foot_clearances(
            mjx_data, normalize=False
        )
        assert left_clearance.shape == ()
        assert right_clearance.shape == ()

    def test_get_foot_clearances_non_negative(self, cal_with_extended_state, mjx_data):
        """Foot clearances should be non-negative (above ground)."""
        left_clearance, right_clearance = cal_with_extended_state.get_foot_clearances(
            mjx_data, normalize=False
        )
        # At rest, feet should be at or above ground level
        # Allow small negative values due to contact penetration
        assert left_clearance >= -0.01
        assert right_clearance >= -0.01


class TestRootState:
    """Tests for root state access methods."""

    def test_get_root_pose_returns_pose3d(self, cal_with_extended_state, mjx_data):
        """get_root_pose should return Pose3D dataclass."""
        pose = cal_with_extended_state.get_root_pose(mjx_data)
        assert isinstance(pose, Pose3D)

    def test_root_pose_has_position(self, cal_with_extended_state, mjx_data):
        """Pose3D should have position (3,) array."""
        pose = cal_with_extended_state.get_root_pose(mjx_data)
        assert pose.position.shape == (3,)
        assert jnp.all(jnp.isfinite(pose.position))

    def test_root_pose_has_orientation(self, cal_with_extended_state, mjx_data):
        """Pose3D should have orientation quaternion (4,) array."""
        pose = cal_with_extended_state.get_root_pose(mjx_data)
        assert pose.orientation.shape == (4,)
        # Quaternion should be unit norm
        norm = jnp.linalg.norm(pose.orientation)
        assert jnp.abs(norm - 1.0) < 1e-5

    def test_root_pose_has_frame(self, cal_with_extended_state, mjx_data):
        """Pose3D should have frame attribute."""
        pose = cal_with_extended_state.get_root_pose(
            mjx_data, frame=CoordinateFrame.WORLD
        )
        assert pose.frame == CoordinateFrame.WORLD

    def test_root_pose_height_property(self, cal_with_extended_state, mjx_data):
        """Pose3D.height should return Z coordinate."""
        pose = cal_with_extended_state.get_root_pose(mjx_data)
        expected_height = pose.position[2]
        assert pose.height == expected_height

    def test_get_root_velocity_returns_velocity3d(
        self, cal_with_extended_state, mjx_data
    ):
        """get_root_velocity should return Velocity3D dataclass."""
        velocity = cal_with_extended_state.get_root_velocity(mjx_data)
        assert isinstance(velocity, Velocity3D)

    def test_root_velocity_has_linear(self, cal_with_extended_state, mjx_data):
        """Velocity3D should have linear (3,) array."""
        velocity = cal_with_extended_state.get_root_velocity(mjx_data)
        assert velocity.linear.shape == (3,)
        assert jnp.all(jnp.isfinite(velocity.linear))

    def test_root_velocity_has_angular(self, cal_with_extended_state, mjx_data):
        """Velocity3D should have angular (3,) array."""
        velocity = cal_with_extended_state.get_root_velocity(mjx_data)
        assert velocity.angular.shape == (3,)
        assert jnp.all(jnp.isfinite(velocity.angular))

    def test_get_gravity_vector_shape(self, cal_with_extended_state, mjx_data):
        """get_gravity_vector should return (3,) array."""
        gravity = cal_with_extended_state.get_gravity_vector(mjx_data)
        assert gravity.shape == (3,)

    def test_get_gravity_vector_normalized(self, cal_with_extended_state, mjx_data):
        """Gravity vector should be approximately unit length."""
        gravity = cal_with_extended_state.get_gravity_vector(mjx_data)
        norm = jnp.linalg.norm(gravity)
        # Should be close to 1.0 (normalized)
        assert jnp.abs(norm - 1.0) < 0.1

    def test_get_root_height_scalar(self, cal_with_extended_state, mjx_data):
        """get_root_height should return scalar height value."""
        height = cal_with_extended_state.get_root_height(mjx_data)
        assert height.shape == ()  # scalar
        assert jnp.isfinite(height)


class TestCoordinateFrames:
    """Tests for coordinate frame transformations."""

    def test_root_pose_heading_local_frame(self, cal_with_extended_state, mjx_data):
        """get_root_pose in HEADING_LOCAL should work."""
        pose = cal_with_extended_state.get_root_pose(
            mjx_data, frame=CoordinateFrame.HEADING_LOCAL
        )
        assert pose.frame == CoordinateFrame.HEADING_LOCAL
        assert jnp.all(jnp.isfinite(pose.position))

    def test_root_velocity_local_frame(self, cal_with_extended_state, mjx_data):
        """get_root_velocity in LOCAL frame should work."""
        velocity = cal_with_extended_state.get_root_velocity(
            mjx_data, frame=CoordinateFrame.LOCAL
        )
        assert velocity.frame == CoordinateFrame.LOCAL
        assert jnp.all(jnp.isfinite(velocity.linear))

    def test_foot_positions_heading_local(self, cal_with_extended_state, mjx_data):
        """get_foot_positions in HEADING_LOCAL should work."""
        left_pos, right_pos = cal_with_extended_state.get_foot_positions(
            mjx_data,
            normalize=False,
            frame=CoordinateFrame.HEADING_LOCAL,
        )
        assert jnp.all(jnp.isfinite(left_pos))
        assert jnp.all(jnp.isfinite(right_pos))
