# training/tests/test_schema_contract.py
"""
Layer 0: Schema Contract Tests

This module tests the schema contract between wildrobot.xml and robot_config.yaml.
These are the foundation tests - if they fail, no other tests should run.

Tier: T0 (Pre-commit, < 2 min)
Markers: @pytest.mark.unit
"""

import hashlib
import json
from pathlib import Path

import mujoco
import pytest

from training.tests.robot_schema import WildRobotSchema


# =============================================================================
# Test 0.1: Generation and Snapshot Tests
# =============================================================================


class TestSchemaGeneration:
    """Tests for schema generation correctness."""

    @pytest.mark.unit
    def test_deterministic_generation(self, project_root):
        """
        Purpose: Verify schema generation is deterministic.

        Procedure:
        1. Generate robot_config from the same XML twice
        2. Compare outputs

        Assertions:
        - Byte-identical or semantically identical output
        """
        xml_path = project_root / "assets" / "scene_flat_terrain.xml"

        schema1 = WildRobotSchema.from_xml(xml_path)
        schema2 = WildRobotSchema.from_xml(xml_path)

        # Compare as dictionaries for semantic equality
        assert schema1.to_dict() == schema2.to_dict(), "Generation not deterministic"

    @pytest.mark.unit
    def test_snapshot_consistency(self, project_root, robot_schema):
        """
        Purpose: Detect unintentional XML changes.

        Procedure:
        1. Regenerate schema from current XML
        2. Diff against committed robot_schema.json (if exists)

        Assertions:
        - Any difference fails with actionable diff message
        """
        schema_path = project_root / "assets" / "robot_schema.json"

        if not schema_path.exists():
            pytest.skip(
                f"Schema snapshot missing at {schema_path}. "
                "Run: python -c 'from training.tests.robot_schema import extract_and_save_schema; "
                "extract_and_save_schema()' if you want a snapshot."
            )

        try:
            robot_schema.assert_matches_saved(schema_path)
        except AssertionError as e:
            pytest.fail(
                f"Schema mismatch! XML has changed.\n"
                f"If intentional, run: python -c 'from training.tests.robot_schema import extract_and_save_schema; extract_and_save_schema()'\n"
                f"Details: {e}"
            )

    @pytest.mark.unit
    def test_schema_dimensions(self, robot_schema):
        """
        Purpose: Verify schema contains expected dimensions.

        Assertions:
        - nq > 0 (qpos dimension)
        - nv > 0 (qvel dimension)
        - nu > 0 (actuator count)
        - nu == 8 (WildRobot has 8 actuators)
        """
        assert robot_schema.nq > 0, "nq should be positive"
        assert robot_schema.nv > 0, "nv should be positive"
        assert robot_schema.nu > 0, "nu should be positive"
        assert robot_schema.nu == 8, f"Expected 8 actuators, got {robot_schema.nu}"


# =============================================================================
# Test 0.2: Contract Content Tests
# =============================================================================


class TestSchemaContent:
    """Tests for schema content correctness."""

    # -------------------------------------------------------------------------
    # 0.2.1 Joints (Real Joints Only)
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_no_mimic_joints_in_schema(self, robot_schema):
        """
        Purpose: Verify _mimic joints are excluded from schema.

        Assertions:
        - No joint name contains "_mimic"
        """
        for joint in robot_schema.joints:
            assert "_mimic" not in joint.joint_name.lower(), (
                f"Mimic joint leaked into schema: {joint.joint_name}"
            )

    @pytest.mark.unit
    def test_joints_have_valid_addresses(self, robot_schema):
        """
        Purpose: Verify all joints have valid qpos and dof addresses.

        Assertions:
        - qpos_adr >= 0
        - dof_adr >= 0
        - qpos_adr < nq
        - dof_adr < nv
        """
        for joint in robot_schema.joints:
            assert joint.qpos_adr >= 0, f"Joint {joint.joint_name} has invalid qpos_adr"
            assert joint.dof_adr >= 0, f"Joint {joint.joint_name} has invalid dof_adr"
            assert joint.qpos_adr < robot_schema.nq, f"Joint {joint.joint_name} qpos_adr exceeds nq"
            assert joint.dof_adr < robot_schema.nv, f"Joint {joint.joint_name} dof_adr exceeds nv"

    @pytest.mark.unit
    def test_joint_limits_valid(self, robot_schema):
        """
        Purpose: Verify all joints have valid limits.

        Assertions:
        - range_lower < range_upper for all joints
        - Limits are finite
        """
        import numpy as np

        for joint in robot_schema.joints:
            assert np.isfinite(joint.range_lower), f"Joint {joint.joint_name} has non-finite lower limit"
            assert np.isfinite(joint.range_upper), f"Joint {joint.joint_name} has non-finite upper limit"
            assert joint.range_lower < joint.range_upper, (
                f"Joint {joint.joint_name} has invalid limits: [{joint.range_lower}, {joint.range_upper}]"
            )

    # -------------------------------------------------------------------------
    # 0.2.2 Actuators
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_actuators_target_valid_joints(self, robot_schema):
        """
        Purpose: Verify no actuator targets a _mimic joint.

        Assertions:
        - All actuator target joints exist in schema joints
        - No actuator targets a _mimic joint
        """
        joint_names = {j.joint_name for j in robot_schema.joints}

        for actuator in robot_schema.actuators:
            assert "_mimic" not in actuator.joint_name.lower(), (
                f"Actuator '{actuator.actuator_name}' targets _mimic joint '{actuator.joint_name}'"
            )
            assert actuator.joint_name in joint_names, (
                f"Actuator '{actuator.actuator_name}' targets unknown joint '{actuator.joint_name}'"
            )

    @pytest.mark.unit
    def test_actuator_count_matches_joints(self, robot_schema):
        """
        Purpose: Verify actuator count matches expected joint count.

        WildRobot has 8 actuated joints:
        - left_hip_pitch, left_hip_roll, left_knee_pitch, left_ankle_pitch
        - right_hip_pitch, right_hip_roll, right_knee_pitch, right_ankle_pitch

        Assertions:
        - 8 actuators present
        - Each actuator has unique target joint
        """
        assert len(robot_schema.actuators) == 8, f"Expected 8 actuators, got {len(robot_schema.actuators)}"

        target_joints = [a.joint_name for a in robot_schema.actuators]
        assert len(target_joints) == len(set(target_joints)), "Duplicate actuator targets found"

    @pytest.mark.unit
    def test_actuator_force_limits(self, robot_schema):
        """
        Purpose: Verify actuator force limits are set correctly.

        WildRobot uses 4 Nm force limit (HTD-45H servos).

        Assertions:
        - All actuators have symmetric force limits
        - Force limit is 4.0 Nm
        """
        for actuator in robot_schema.actuators:
            assert actuator.force_range_lower == -4.0, (
                f"Actuator {actuator.actuator_name} has wrong lower force limit: {actuator.force_range_lower}"
            )
            assert actuator.force_range_upper == 4.0, (
                f"Actuator {actuator.actuator_name} has wrong upper force limit: {actuator.force_range_upper}"
            )

    # -------------------------------------------------------------------------
    # 0.2.3 Foot Contact Sets
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_foot_geoms_valid(self, robot_schema):
        """
        Purpose: Verify foot geoms are correctly identified.

        Assertions:
        - Left and right foot geom sets are disjoint
        - All foot geoms are collision-enabled
        - No foot geom belongs to a _mimic body
        """
        left_ids = {g.geom_id for g in robot_schema.left_foot_geoms}
        right_ids = {g.geom_id for g in robot_schema.right_foot_geoms}

        assert left_ids.isdisjoint(right_ids), "Foot geom sets overlap!"

        for geom in robot_schema.left_foot_geoms + robot_schema.right_foot_geoms:
            assert "_mimic" not in geom.body_name.lower(), (
                f"Foot geom '{geom.geom_name}' belongs to _mimic body '{geom.body_name}'"
            )
            assert geom.is_collision, (
                f"Foot geom '{geom.geom_name}' is not collision-enabled"
            )

    @pytest.mark.unit
    def test_foot_geom_names(self, robot_schema):
        """
        Purpose: Verify expected foot geom names are present.

        Expected geoms:
        - Left: left_toe, left_heel
        - Right: right_toe, right_heel

        Assertions:
        - All expected geoms present
        """
        left_names = {g.geom_name for g in robot_schema.left_foot_geoms}
        right_names = {g.geom_name for g in robot_schema.right_foot_geoms}

        assert "left_toe" in left_names, "Missing left_toe geom"
        assert "left_heel" in left_names, "Missing left_heel geom"
        assert "right_toe" in right_names, "Missing right_toe geom"
        assert "right_heel" in right_names, "Missing right_heel geom"

    # -------------------------------------------------------------------------
    # 0.2.4 Base Definition
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_base_definition_exists(self, robot_schema):
        """
        Purpose: Verify floating base is defined.

        Assertions:
        - base is not None
        - base has valid body name
        - base has valid free joint name
        """
        assert robot_schema.base is not None, "Floating base not defined"
        assert robot_schema.base.base_body_name, "Base body name is empty"
        assert robot_schema.base.base_free_joint_name, "Base free joint name is empty"

    @pytest.mark.unit
    def test_base_qpos_slice(self, robot_schema):
        """
        Purpose: Verify base qpos slice is correct (7D: pos + quat).

        Assertions:
        - base_qpos_adr = 0 (floating base at start)
        - base_dof_adr = 0 (6D velocity at start)
        """
        assert robot_schema.base.base_qpos_adr == 0, (
            f"Base qpos_adr should be 0, got {robot_schema.base.base_qpos_adr}"
        )
        assert robot_schema.base.base_dof_adr == 0, (
            f"Base dof_adr should be 0, got {robot_schema.base.base_dof_adr}"
        )


# =============================================================================
# Test 0.3: Schema Validation Methods
# =============================================================================


class TestSchemaValidation:
    """Tests for schema validation methods."""

    @pytest.mark.unit
    def test_validate_passes_for_valid_schema(self, robot_schema):
        """
        Purpose: Verify validate() doesn't raise for valid schema.

        Assertions:
        - No exception raised
        """
        robot_schema.validate()  # Should not raise

    @pytest.mark.unit
    def test_assert_no_mimic_in_actuators(self, robot_schema):
        """
        Purpose: Verify assert_no_mimic_in_actuators() passes.

        Assertions:
        - No exception raised
        """
        robot_schema.assert_no_mimic_in_actuators()  # Should not raise

    @pytest.mark.unit
    def test_assert_no_mimic_in_foot_geoms(self, robot_schema):
        """
        Purpose: Verify assert_no_mimic_in_foot_geoms() passes.

        Assertions:
        - No exception raised
        """
        robot_schema.assert_no_mimic_in_foot_geoms()  # Should not raise


# =============================================================================
# Test 0.4: Accessor Methods
# =============================================================================


class TestSchemaAccessors:
    """Tests for schema accessor methods."""

    @pytest.mark.unit
    def test_get_actuated_joint_qpos_indices(self, robot_schema):
        """
        Purpose: Verify qpos indices are returned correctly.

        Assertions:
        - Returns list of 8 indices (one per actuator)
        - All indices are valid (within nq)
        """
        indices = robot_schema.get_actuated_joint_qpos_indices()
        assert len(indices) == 8, f"Expected 8 indices, got {len(indices)}"
        for idx in indices:
            assert 0 <= idx < robot_schema.nq, f"Invalid qpos index: {idx}"

    @pytest.mark.unit
    def test_get_actuated_joint_dof_indices(self, robot_schema):
        """
        Purpose: Verify dof indices are returned correctly.

        Assertions:
        - Returns list of 8 indices (one per actuator)
        - All indices are valid (within nv)
        """
        indices = robot_schema.get_actuated_joint_dof_indices()
        assert len(indices) == 8, f"Expected 8 indices, got {len(indices)}"
        for idx in indices:
            assert 0 <= idx < robot_schema.nv, f"Invalid dof index: {idx}"

    @pytest.mark.unit
    def test_get_foot_geom_ids(self, robot_schema, mj_model):
        """
        Purpose: Verify foot geom IDs are valid.

        Assertions:
        - All IDs are within valid range
        - IDs match names in model
        """
        left_ids = robot_schema.get_left_foot_geom_ids()
        right_ids = robot_schema.get_right_foot_geom_ids()

        assert len(left_ids) == 2, f"Expected 2 left foot geoms, got {len(left_ids)}"
        assert len(right_ids) == 2, f"Expected 2 right foot geoms, got {len(right_ids)}"

        for geom_id in left_ids + right_ids:
            assert 0 <= geom_id < mj_model.ngeom, f"Invalid geom ID: {geom_id}"

    @pytest.mark.unit
    def test_get_base_qpos_slice(self, robot_schema):
        """
        Purpose: Verify base qpos slice is correct.

        Assertions:
        - Slice covers 7 elements (pos + quat)
        """
        s = robot_schema.get_base_qpos_slice()
        assert s.start == 0, f"Base qpos slice should start at 0, got {s.start}"
        assert s.stop == 7, f"Base qpos slice should end at 7, got {s.stop}"

    @pytest.mark.unit
    def test_get_base_qvel_slice(self, robot_schema):
        """
        Purpose: Verify base qvel slice is correct.

        Assertions:
        - Slice covers 6 elements (linvel + angvel)
        """
        s = robot_schema.get_base_qvel_slice()
        assert s.start == 0, f"Base qvel slice should start at 0, got {s.start}"
        assert s.stop == 6, f"Base qvel slice should end at 6, got {s.stop}"
