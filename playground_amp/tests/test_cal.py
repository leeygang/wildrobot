"""Unit tests for Control Abstraction Layer (CAL).

v0.11.0: Tests for the Control Abstraction Layer module.

These tests verify:
1. CAL initialization from MuJoCo model and config
2. Policy action to ctrl transformation (with symmetry correction)
3. Physical angles to ctrl (passthrough)
4. Normalized joint position/velocity getters
5. Symmetry handling (mirror_sign)
6. Command generation for Sim2Real
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground_amp.control.cal import ControlAbstractionLayer
from playground_amp.control.specs import ActuatorSpec, ControlCommand, JointSpec
from playground_amp.control.types import ActuatorType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cal_instance(mj_model, robot_config):
    """Create CAL instance for testing."""
    actuated_joints_config = robot_config.actuated_joints
    return ControlAbstractionLayer(mj_model, actuated_joints_config)


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

        # Both should produce the same normalized position (symmetric motion)
        # The actual ctrl values will differ due to different ranges, but normalized
        # positions should be equal
        np.testing.assert_almost_equal(
            left_normalized,
            right_normalized,
            decimal=5,
            err_msg=(
                f"Symmetric action should produce symmetric motion. "
                f"Left normalized: {left_normalized}, Right normalized: {right_normalized}"
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
