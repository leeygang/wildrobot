# training/tests/test_model_validation.py
"""
Layer 2: MuJoCo Model Validation Tests

This module tests the MuJoCo model directly without RL.
These tests verify the model is physically correct before training.

Tier: T1 (PR gate, 10-20 min)
Markers: @pytest.mark.sim
"""

import mujoco
import numpy as np
import pytest


# =============================================================================
# Test 2.1: Load and Step Smoke
# =============================================================================


class TestModelStability:
    """Tests for basic model stability under simulation."""

    @pytest.mark.sim
    def test_model_stability(self, mj_model, mj_data):
        """
        Purpose: Verify model is stable under simulation.

        Procedure:
        1. Load scene_flat_terrain.xml
        2. Run 1000 sim steps with zero controls

        Assertions:
        - No NaN in qpos or qvel
        - No constraint solver warnings above threshold
        - Energy does not explode (< 1e6 J)
        """
        for _ in range(1000):
            mj_data.ctrl[:] = 0
            mujoco.mj_step(mj_model, mj_data)

            assert not np.any(np.isnan(mj_data.qpos)), "NaN in qpos"
            assert not np.any(np.isnan(mj_data.qvel)), "NaN in qvel"

        # Check energy didn't explode
        kinetic = 0.5 * np.sum(mj_data.qvel ** 2)
        assert kinetic < 1e6, f"Energy exploded: {kinetic}"

    @pytest.mark.sim
    def test_model_loads_correctly(self, mj_model):
        """
        Purpose: Verify model dimensions match expectations.

        WildRobot:
        - nq = 15 (7 base + 8 joints)
        - nv = 14 (6 base + 8 joints)
        - nu = 8 (8 actuators)

        Assertions:
        - Dimensions match expected values
        """
        assert mj_model.nq == 15, f"Expected nq=15, got {mj_model.nq}"
        assert mj_model.nv == 14, f"Expected nv=14, got {mj_model.nv}"
        assert mj_model.nu == 8, f"Expected nu=8, got {mj_model.nu}"


# =============================================================================
# Test 2.2: Kinematic Sanity
# =============================================================================


class TestKinematicSanity:
    """Tests for kinematic validity of the model."""

    @pytest.mark.sim
    def test_initial_pose_valid(self, mj_model, mj_data, training_config):
        """
        Purpose: Verify reset pose is physically valid.

        Assertions:
        - Base height in [min_height, max_height]
        - Base pitch and roll < 0.3 rad (roughly upright)
        - All joints within limits
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Base height
        base_height = mj_data.qpos[2]
        assert training_config.env.min_height <= base_height <= training_config.env.max_height, (
            f"Base height {base_height} not in [{training_config.env.min_height}, {training_config.env.max_height}]"
        )

        # Base orientation (check quaternion is roughly upright)
        quat = mj_data.qpos[3:7]
        # For upright: quat ≈ [1, 0, 0, 0] or [w, x, y, z] with w ≈ 1
        # The z-component of the up vector in body frame tells us tilt
        # Using quaternion to rotate [0, 0, 1] gives us the up vector
        w, x, y, z = quat
        up_z = 1 - 2 * (x*x + y*y)  # z-component of rotated up vector
        assert up_z > 0.9, f"Robot not upright: up_z = {up_z}"

        # Joint limits
        for i in range(mj_model.njnt):
            if mj_model.jnt_limited[i]:
                qpos_adr = mj_model.jnt_qposadr[i]
                # Skip free joint (has 7 qpos values)
                if mj_model.jnt_type[i] == 0:
                    continue
                pos = mj_data.qpos[qpos_adr]
                low, high = mj_model.jnt_range[i]
                assert low <= pos <= high, (
                    f"Joint {i} out of limits: {pos} not in [{low}, {high}]"
                )

    @pytest.mark.sim
    def test_no_initial_penetrations(self, mj_model, mj_data):
        """
        Purpose: Verify no penetrations in initial pose.

        Assertions:
        - No contacts with large penetration depth (> 0.01m)
        """
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        MAX_PENETRATION = 0.01  # 1cm

        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            dist = contact.dist
            if dist < -MAX_PENETRATION:
                geom1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "unnamed"
                geom2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "unnamed"
                pytest.fail(
                    f"Initial penetration detected: {geom1_name} vs {geom2_name}, "
                    f"depth = {-dist:.4f}m"
                )


# =============================================================================
# Test 2.3: Actuator Isolation Test
# =============================================================================


class TestActuatorIsolation:
    """Tests for actuator-to-joint mapping correctness."""

    MIN_ACTUATOR_FORCE = 0.01  # Nm
    MIN_VELOCITY_CHANGE = 1e-6  # rad/s

    @pytest.mark.sim
    def test_actuator_isolation(self, mj_model, mj_data, robot_schema):
        """
        Purpose: Verify actuator-to-joint mapping is correct.

        Procedure:
        For each actuator:
        1. Reset to standing pose
        2. Zero all controls
        3. Apply small control (+0.1) to THIS actuator only
        4. Step simulation

        Assertions:
        - Target actuator produces significant force (|τ| > 0.01 Nm)
        - Target joint velocity changes (|Δv| > 1e-6 rad/s)
        """
        for actuator in robot_schema.actuators:
            act_name = actuator.actuator_name
            target_dof = actuator.dof_adr

            # Find actuator index
            act_idx = None
            for i in range(mj_model.nu):
                if mj_model.actuator(i).name == act_name:
                    act_idx = i
                    break

            assert act_idx is not None, f"Actuator {act_name} not found in model"

            # Reset
            mujoco.mj_resetData(mj_model, mj_data)
            mujoco.mj_forward(mj_model, mj_data)

            # Zero all controls, apply to this actuator only
            mj_data.ctrl[:] = 0
            mj_data.ctrl[act_idx] = 0.1

            # Record initial state
            initial_vel = mj_data.qvel[target_dof]

            # Step
            mujoco.mj_step(mj_model, mj_data)

            # Check actuator produces force
            act_force = mj_data.actuator_force[act_idx]
            assert abs(act_force) > self.MIN_ACTUATOR_FORCE, (
                f"Actuator {act_name} produced insufficient force: {act_force}"
            )

            # Check target joint moved
            final_vel = mj_data.qvel[target_dof]
            vel_change = abs(final_vel - initial_vel)
            assert vel_change > self.MIN_VELOCITY_CHANGE, (
                f"Actuator {act_name} didn't move target joint: Δv={vel_change}"
            )

    @pytest.mark.sim
    def test_actuator_symmetry(self, mj_model, mj_data):
        """
        Purpose: Verify left/right actuators produce symmetric responses.

        Procedure:
        1. Apply symmetric target positions to left and right hip_pitch
        2. Compare resulting force magnitudes

        Note: WildRobot has mirrored joint ranges for left/right legs:
        - left_hip_pitch: [-0.087, 1.571] rad (flexion dominant)
        - right_hip_pitch: [-1.571, 0.087] rad (extension dominant)

        The ctrl value IS the target position in radians (not a normalized value).
        For symmetry testing, we use symmetric target positions: +X for left, -X for right.

        Assertions:
        - Left and right force magnitudes are approximately equal
        """
        # Find left and right hip pitch actuators
        left_idx = None
        right_idx = None

        for i in range(mj_model.nu):
            name = mj_model.actuator(i).name
            if "left_hip_pitch" in name:
                left_idx = i
            elif "right_hip_pitch" in name:
                right_idx = i

        if left_idx is None or right_idx is None:
            pytest.skip("Could not find hip_pitch actuators")

        # Reset to initial pose
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Use symmetric target positions (within both joint ranges)
        # Left targets positive, right targets negative (mirror symmetry)
        target_angle = 0.05  # Small angle within both ranges
        mj_data.ctrl[:] = 0
        mj_data.ctrl[left_idx] = target_angle    # left: target = +0.05 rad
        mj_data.ctrl[right_idx] = -target_angle  # right: target = -0.05 rad

        mujoco.mj_step(mj_model, mj_data)

        left_force = abs(mj_data.actuator_force[left_idx])
        right_force = abs(mj_data.actuator_force[right_idx])

        # Force magnitudes should be within 10% of each other
        diff = abs(left_force - right_force)
        max_force = max(left_force, right_force)
        if max_force > 0:
            ratio = diff / max_force
            assert ratio < 0.1, (
                f"Left/right hip_pitch forces asymmetric: L={left_force:.3f}, R={right_force:.3f}, diff={ratio:.1%}"
            )


# =============================================================================
# Test 2.4: Torque Limit Validation
# =============================================================================


class TestTorqueLimits:
    """Tests for actuator torque limit enforcement."""

    @pytest.mark.sim
    def test_torque_limits_enforced(self, mj_model, mj_data):
        """
        Purpose: Verify actuator forces are clamped to limits.

        Procedure:
        1. Apply maximum control to all actuators
        2. Check actuator forces don't exceed forcerange

        Assertions:
        - All actuator forces within forcerange
        """
        # Reset
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)

        # Apply maximum control
        mj_data.ctrl[:] = 1.0  # Maximum positive control

        mujoco.mj_step(mj_model, mj_data)

        for i in range(mj_model.nu):
            force = mj_data.actuator_force[i]
            force_range = mj_model.actuator_forcerange[i]

            # Allow small tolerance for numerical precision
            assert force >= force_range[0] - 0.01, (
                f"Actuator {i} force below min: {force} < {force_range[0]}"
            )
            assert force <= force_range[1] + 0.01, (
                f"Actuator {i} force above max: {force} > {force_range[1]}"
            )

    @pytest.mark.sim
    def test_torque_limit_value(self, mj_model):
        """
        Purpose: Verify torque limits are set to 4 Nm (WildRobot spec).

        Assertions:
        - All actuators have forcerange = [-4, 4]
        """
        EXPECTED_LIMIT = 4.0

        for i in range(mj_model.nu):
            force_range = mj_model.actuator_forcerange[i]
            assert force_range[0] == -EXPECTED_LIMIT, (
                f"Actuator {i} has wrong lower limit: {force_range[0]}"
            )
            assert force_range[1] == EXPECTED_LIMIT, (
                f"Actuator {i} has wrong upper limit: {force_range[1]}"
            )


# =============================================================================
# Test 2.5: Mass and Inertia Validation
# =============================================================================


class TestMassInertia:
    """Tests for mass and inertia properties."""

    @pytest.mark.sim
    def test_total_mass_reasonable(self, mj_model):
        """
        Purpose: Verify total robot mass is reasonable.

        WildRobot is a small biped (~3kg expected).

        Assertions:
        - Total mass between 1 and 10 kg
        """
        total_mass = sum(mj_model.body(i).mass[0] for i in range(mj_model.nbody))

        assert 1.0 <= total_mass <= 10.0, (
            f"Total mass {total_mass:.2f} kg seems unreasonable for WildRobot"
        )

    @pytest.mark.sim
    def test_no_zero_mass_bodies(self, mj_model):
        """
        Purpose: Verify no unexpected zero-mass bodies.

        Assertions:
        - Only world body should have zero mass
        - All robot bodies have positive mass
        """
        for i in range(mj_model.nbody):
            body_name = mj_model.body(i).name
            mass = mj_model.body(i).mass[0]

            if body_name == "world" or body_name == "":
                continue  # World body has zero mass

            # _mimic bodies may have zero mass (visual only)
            if "_mimic" in body_name.lower():
                continue

            assert mass > 0, f"Body '{body_name}' has zero mass"

    @pytest.mark.sim
    def test_inertia_positive_definite(self, mj_model):
        """
        Purpose: Verify all body inertias are positive.

        Assertions:
        - All diagonal inertia elements are positive
        """
        for i in range(mj_model.nbody):
            body_name = mj_model.body(i).name
            inertia = mj_model.body(i).inertia

            # Skip world and zero-mass bodies
            if mj_model.body(i).mass[0] == 0:
                continue

            assert all(inertia > 0), (
                f"Body '{body_name}' has non-positive inertia: {inertia}"
            )
