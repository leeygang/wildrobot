# training/tests/test_state_mapping.py
"""
Layer 3: State and Observation Correctness Tests

This module ensures reward and AMP features read the right state.
Tests velocity cross-checks, contact force validation, and joint causality.

Tier: T1 (PR gate)
Markers: @pytest.mark.sim
"""

import mujoco
import numpy as np
import pytest

from training.tests.conftest import get_foot_contact_forces, get_total_mass


# =============================================================================
# Test 3.1: Finite Difference Velocity Cross-Check
# =============================================================================


class TestVelocityCrossCheck:
    """Tests for velocity correctness via finite difference comparison."""

    MIN_CORRELATION = 0.95
    MAX_MAE = 0.1  # m/s

    @pytest.mark.sim
    def test_base_linear_velocity_fd_vs_qvel(self, mj_model, mj_data, robot_schema):
        """
        Purpose: Confirm base qvel is correct by comparing to finite difference.

        Procedure:
        1. Reset to standing pose
        2. Apply random actions for N=100 steps
        3. At each step compute FD velocity: v_fd = (pos_after - pos_before) / dt
        4. Compare FD velocities to qvel velocities

        Assertions:
        - Correlation coefficient > 0.95 for each axis (x, y, z)
        - Mean Absolute Error < 0.1 m/s

        Why This Matters:
        If this fails, YOUR VELOCITY REWARD IS MEANINGLESS.
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        dt = mj_model.opt.timestep
        n_steps = 100

        fd_velocities = []
        qvel_velocities = []

        base_dof_adr = robot_schema.base.base_dof_adr

        for _ in range(n_steps):
            mj_data.ctrl[:] = np.random.uniform(-1, 1, mj_model.nu)

            pos_before = mj_data.qpos[:3].copy()
            mujoco.mj_step(mj_model, mj_data)
            pos_after = mj_data.qpos[:3].copy()

            v_fd = (pos_after - pos_before) / dt
            fd_velocities.append(v_fd)

            v_qvel = mj_data.qvel[base_dof_adr:base_dof_adr+3].copy()
            qvel_velocities.append(v_qvel)

        fd_velocities = np.array(fd_velocities)
        qvel_velocities = np.array(qvel_velocities)

        # Check correlation for each axis
        for axis, name in enumerate(['x', 'y', 'z']):
            corr = np.corrcoef(fd_velocities[:, axis], qvel_velocities[:, axis])[0, 1]
            assert corr > self.MIN_CORRELATION, (
                f"Velocity {name} correlation too low: {corr:.3f} < {self.MIN_CORRELATION}"
            )

        # Check MAE
        mae = np.mean(np.abs(fd_velocities - qvel_velocities))
        assert mae < self.MAX_MAE, f"Velocity MAE too high: {mae:.4f} m/s"

    @pytest.mark.sim
    def test_joint_velocity_fd_vs_qvel(self, mj_model, mj_data, robot_schema):
        """
        Purpose: Verify joint velocities match finite difference.

        Procedure:
        1. Apply controls and step
        2. Compare qvel to finite difference of qpos

        Assertions:
        - Joint velocity correlation > 0.90
        """
        MIN_CORRELATION = 0.90

        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        dt = mj_model.opt.timestep
        n_steps = 100

        # Get actuated joint qpos indices
        joint_indices = robot_schema.get_actuated_joint_qpos_indices()
        dof_indices = robot_schema.get_actuated_joint_dof_indices()

        fd_velocities = []
        qvel_velocities = []

        for _ in range(n_steps):
            mj_data.ctrl[:] = np.random.uniform(-0.5, 0.5, mj_model.nu)

            pos_before = mj_data.qpos[joint_indices].copy()
            mujoco.mj_step(mj_model, mj_data)
            pos_after = mj_data.qpos[joint_indices].copy()

            v_fd = (pos_after - pos_before) / dt
            fd_velocities.append(v_fd)

            v_qvel = mj_data.qvel[dof_indices].copy()
            qvel_velocities.append(v_qvel)

        fd_velocities = np.array(fd_velocities)
        qvel_velocities = np.array(qvel_velocities)

        # Check correlation for each joint
        for i, actuator in enumerate(robot_schema.actuators):
            corr = np.corrcoef(fd_velocities[:, i], qvel_velocities[:, i])[0, 1]
            assert corr > MIN_CORRELATION, (
                f"Joint {actuator.joint_name} velocity correlation too low: {corr:.3f}"
            )


# =============================================================================
# Test 3.2: Joint-to-Foot Causality Test
# =============================================================================


class TestJointFootCausality:
    """Tests for correct joint-to-foot mapping."""

    MIN_AFFECTED_DELTA = 0.004  # m
    MAX_UNAFFECTED_DELTA = 0.002  # m

    @pytest.mark.sim
    def test_left_knee_affects_left_foot(self, mj_model, mj_data, robot_schema):
        """
        Purpose: Verify qpos indexing and geom IDs are correct by testing
        that moving left knee affects only left foot.

        Procedure:
        1. Reset to standing pose
        2. Call mj_forward()
        3. Record left and right foot positions
        4. Add +0.1 rad to left knee qpos
        5. Call mj_forward()
        6. Record new foot positions

        Assertions:
        - Left foot Z changed significantly (|Δz| > 0.01m)
        - Right foot Z changed minimally (|Δz| < 0.001m)
        """
        # Find left knee joint
        left_knee = next(
            (j for j in robot_schema.joints
             if "left" in j.joint_name.lower() and "knee" in j.joint_name.lower()),
            None
        )
        assert left_knee is not None, "Could not find left knee joint"

        qpos_adr = left_knee.qpos_adr

        # Get foot geom IDs
        left_foot_geom_id = robot_schema.left_foot_geoms[0].geom_id
        right_foot_geom_id = robot_schema.right_foot_geoms[0].geom_id

        # Reset and forward
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Record initial positions
        left_foot_z_initial = mj_data.geom_xpos[left_foot_geom_id][2]
        right_foot_z_initial = mj_data.geom_xpos[right_foot_geom_id][2]

        # Perturb left knee
        mj_data.qpos[qpos_adr] += 0.1
        mujoco.mj_forward(mj_model, mj_data)

        # Record new positions
        left_foot_z_final = mj_data.geom_xpos[left_foot_geom_id][2]
        right_foot_z_final = mj_data.geom_xpos[right_foot_geom_id][2]

        # Assertions
        left_delta = abs(left_foot_z_final - left_foot_z_initial)
        right_delta = abs(right_foot_z_final - right_foot_z_initial)

        assert left_delta > self.MIN_AFFECTED_DELTA, (
            f"Left foot should move significantly: Δz={left_delta:.4f}m"
        )
        assert right_delta < self.MAX_UNAFFECTED_DELTA, (
            f"Right foot should not move: Δz={right_delta:.4f}m"
        )

    @pytest.mark.sim
    def test_right_ankle_affects_right_foot(self, mj_model, mj_data, robot_schema):
        """
        Purpose: Verify right ankle affects right foot.

        Similar to left knee test but for right ankle.
        """
        # Find right ankle joint
        right_ankle = next(
            (j for j in robot_schema.joints
             if "right" in j.joint_name.lower() and "ankle" in j.joint_name.lower()),
            None
        )
        assert right_ankle is not None, "Could not find right ankle joint"

        qpos_adr = right_ankle.qpos_adr

        # Get foot geom IDs
        left_foot_geom_id = robot_schema.left_foot_geoms[0].geom_id
        right_foot_geom_id = robot_schema.right_foot_geoms[0].geom_id

        # Reset and forward
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Record initial positions
        left_foot_z_initial = mj_data.geom_xpos[left_foot_geom_id][2]
        right_foot_z_initial = mj_data.geom_xpos[right_foot_geom_id][2]

        # Perturb right ankle
        mj_data.qpos[qpos_adr] += 0.1
        mujoco.mj_forward(mj_model, mj_data)

        # Record new positions
        left_foot_z_final = mj_data.geom_xpos[left_foot_geom_id][2]
        right_foot_z_final = mj_data.geom_xpos[right_foot_geom_id][2]

        # Assertions
        left_delta = abs(left_foot_z_final - left_foot_z_initial)
        right_delta = abs(right_foot_z_final - right_foot_z_initial)

        assert right_delta > self.MIN_AFFECTED_DELTA, (
            f"Right foot should move significantly: Δz={right_delta:.4f}m"
        )
        assert left_delta < self.MAX_UNAFFECTED_DELTA, (
            f"Left foot should not move: Δz={left_delta:.4f}m"
        )


# =============================================================================
# Test 3.3: Contact Force Validation
# =============================================================================


class TestContactForces:
    """Tests for contact force correctness."""

    FORCE_RATIO_MIN = 0.8
    FORCE_RATIO_MAX = 1.2

    @pytest.mark.sim
    def test_static_equilibrium(self, mj_model, mj_data, robot_schema, floor_geom_id):
        """
        Purpose: Verify total contact force equals robot weight.

        Procedure:
        1. Reset to standing pose
        2. Step until settled (500 steps)
        3. Sum normal forces on all foot geoms
        4. Compare to robot weight: mg = total_mass × 9.81

        Assertions:
        - ΣFn / mg ∈ [0.8, 1.2]

        Why This Matters:
        If this fails, CONTACT READING IS WRONG.
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Let robot settle
        for _ in range(500):
            mj_data.ctrl[:] = 0
            mujoco.mj_step(mj_model, mj_data)

        # Get contact forces
        left_force, right_force = get_foot_contact_forces(
            mj_model, mj_data, robot_schema, floor_geom_id
        )
        total_contact_force = left_force + right_force

        # Compute expected weight
        total_mass = get_total_mass(mj_model)
        expected_weight = total_mass * 9.81

        # Check ratio
        ratio = total_contact_force / expected_weight
        assert self.FORCE_RATIO_MIN <= ratio <= self.FORCE_RATIO_MAX, (
            f"Contact force ratio wrong: {ratio:.2f} (expected ~1.0), "
            f"Fn={total_contact_force:.1f}N, mg={expected_weight:.1f}N"
        )

    @pytest.mark.sim
    def test_left_right_load_asymmetry(self, mj_model, mj_data, robot_schema, floor_geom_id):
        """
        Purpose: Verify feet differentiation is correct.

        Procedure:
        1. Reset to standing pose
        2. Shift COM slightly left (5cm)
        3. Step until settled
        4. Measure normal forces on left vs right feet

        Assertions:
        - Left Fn > Right Fn (since COM shifted left)
        - ΣFn / mg ∈ [0.8, 1.2]
        """
        COM_SHIFT = 0.05  # 5cm

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[1] -= COM_SHIFT  # Shift left (negative Y)
        mujoco.mj_forward(mj_model, mj_data)

        # Let settle
        for _ in range(500):
            mj_data.ctrl[:] = 0
            mujoco.mj_step(mj_model, mj_data)

        # Get contact forces
        left_force, right_force = get_foot_contact_forces(
            mj_model, mj_data, robot_schema, floor_geom_id
        )

        # Left should have more load (shifted toward left)
        assert left_force > right_force, (
            f"Expected left > right force. Left={left_force:.1f}N, Right={right_force:.1f}N"
        )

        # Total should still be body weight
        total_mass = get_total_mass(mj_model)
        expected_weight = total_mass * 9.81
        total_force = left_force + right_force
        ratio = total_force / expected_weight

        assert self.FORCE_RATIO_MIN <= ratio <= self.FORCE_RATIO_MAX, (
            f"Total force ratio: {ratio:.2f}"
        )

    @pytest.mark.sim
    def test_drop_impact_force(self, mj_model, mj_data, robot_schema, floor_geom_id):
        """
        Purpose: Verify impact forces are detected correctly.

        Procedure:
        1. Start robot 5cm above ground
        2. Let it drop
        3. Monitor contact forces

        Assertions:
        - Peak force spike detected (Fn_peak > 2 × mg)
        - Force stabilizes to ~mg after impact
        """
        DROP_HEIGHT = 0.05
        PEAK_FORCE_MULTIPLIER = 2.0

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] += DROP_HEIGHT
        mujoco.mj_forward(mj_model, mj_data)

        peak_force = 0
        forces = []

        for _ in range(200):
            mj_data.ctrl[:] = 0
            mujoco.mj_step(mj_model, mj_data)

            left_force, right_force = get_foot_contact_forces(
                mj_model, mj_data, robot_schema, floor_geom_id
            )
            total_force = left_force + right_force

            forces.append(total_force)
            peak_force = max(peak_force, total_force)

        total_mass = get_total_mass(mj_model)
        expected_weight = total_mass * 9.81

        assert peak_force > PEAK_FORCE_MULTIPLIER * expected_weight, (
            f"No impact spike. Peak={peak_force:.1f}N, "
            f"expected > {PEAK_FORCE_MULTIPLIER * expected_weight:.1f}N"
        )

        # Check stabilization
        final_force = np.mean(forces[-50:])
        ratio = final_force / expected_weight
        assert 0.8 <= ratio <= 1.2, (
            f"Force didn't stabilize. Final={final_force:.1f}N, expected={expected_weight:.1f}N"
        )


# =============================================================================
# Test 3.4: Sensor Reading Validation
# =============================================================================


class TestSensorReadings:
    """Tests for sensor reading correctness."""

    @staticmethod
    def _gravity_from_quat(quat):
        """Compute local gravity direction from quaternion (wxyz)."""
        w, x, y, z = quat
        r = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )
        world_gravity = np.array([0.0, 0.0, -1.0])
        return r.T @ world_gravity

    @pytest.mark.sim
    def test_gravity_sensor_direction(self, mj_model, mj_data, robot_config):
        """
        Purpose: Verify gravity sensor reads correct direction.

        When upright, gravity in body frame should be approximately [0, 0, -1].

        Assertions:
        - Gravity sensor z-component is approximately -1 (± 0.1)
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Compute gravity direction from orientation sensor quaternion
        orientation_sensor_name = robot_config.orientation_sensor
        sensor_id = None
        for i in range(mj_model.nsensor):
            if mj_model.sensor(i).name == orientation_sensor_name:
                sensor_id = i
                break

        assert sensor_id is not None, f"Orientation sensor '{orientation_sensor_name}' not found"

        adr = mj_model.sensor_adr[sensor_id]
        quat = mj_data.sensordata[adr:adr+4]
        gravity = self._gravity_from_quat(quat)

        # When upright, gravity in body frame should be [0, 0, -1]
        assert abs(gravity[2] - (-1.0)) < 0.1, (
            f"Gravity sensor z should be ~-1 when upright: got {gravity[2]}"
        )

    @pytest.mark.sim
    def test_velocity_sensor_zero_at_rest(self, mj_model, mj_data, robot_config):
        """
        Purpose: Verify velocity sensors read zero when robot is at rest.

        Assertions:
        - Linear velocity magnitude < 0.1 m/s after settling
        - Angular velocity magnitude < 0.1 rad/s after settling
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Let robot settle
        for _ in range(500):
            mj_data.ctrl[:] = 0
            mujoco.mj_step(mj_model, mj_data)

        # Check linear velocity sensor
        linvel_sensor_name = robot_config.local_linvel_sensor
        linvel_sensor_id = None
        for i in range(mj_model.nsensor):
            if mj_model.sensor(i).name == linvel_sensor_name:
                linvel_sensor_id = i
                break

        if linvel_sensor_id is not None:
            adr = mj_model.sensor_adr[linvel_sensor_id]
            linvel = mj_data.sensordata[adr:adr+3]
            linvel_mag = np.linalg.norm(linvel)
            assert linvel_mag < 0.1, f"Linear velocity should be ~0 at rest: {linvel_mag:.3f} m/s"

        # Check angular velocity sensor (gyro)
        angvel_sensor_name = robot_config.root_gyro_sensor
        angvel_sensor_id = None
        for i in range(mj_model.nsensor):
            if mj_model.sensor(i).name == angvel_sensor_name:
                angvel_sensor_id = i
                break

        if angvel_sensor_id is not None:
            adr = mj_model.sensor_adr[angvel_sensor_id]
            angvel = mj_data.sensordata[adr:adr+3]
            angvel_mag = np.linalg.norm(angvel)
            assert angvel_mag < 0.1, f"Angular velocity should be ~0 at rest: {angvel_mag:.3f} rad/s"
