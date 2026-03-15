# training/tests/test_reward_terms.py
"""
Layer 4: Reward Component Correctness Tests

This module tests individual reward terms using deterministic micro-scenarios.
Tests use controlled setups, not training, to validate each term.

Tier: T1 (PR gate)
Markers: @pytest.mark.sim
"""

import mujoco
import numpy as np
import pytest
import jax.numpy as jp

from training.envs.wildrobot_env import (
    compute_dense_progress_reward,
    compute_cycle_progress_terms,
    compute_phase_gated_clearance_reward,
    compute_propulsion_quality_gate,
    compute_step_length_touchdown_reward,
    compute_zero_baseline_forward_reward,
)


# =============================================================================
# Test 4.1: Velocity Reward Monotonicity
# =============================================================================


class TestVelocityReward:
    """Tests for velocity tracking reward correctness."""

    @pytest.mark.sim
    def test_velocity_reward_exponential_formula(self):
        """
        Purpose: Verify velocity reward formula is correct.

        The velocity tracking reward uses exponential form:
        reward = exp(-error^2 / sigma^2)

        Assertions:
        - Reward is 1.0 when error = 0
        - Reward decreases as error increases
        - Reward is always positive
        """

        def velocity_reward(v_actual, v_target, sigma=0.25):
            """Standard exponential tracking reward."""
            error = v_actual - v_target
            return np.exp(-error**2 / sigma**2)

        target = 1.0  # m/s
        sigma = 0.25

        # At target, reward should be 1.0
        r_at_target = velocity_reward(target, target, sigma)
        assert np.isclose(r_at_target, 1.0), f"Reward at target should be 1.0: {r_at_target}"

        # Test monotonicity
        velocities = np.linspace(-0.5, 2.5, 50)
        rewards = [velocity_reward(v, target, sigma) for v in velocities]

        max_idx = np.argmax(rewards)
        # Max should be near target velocity
        assert np.isclose(velocities[max_idx], target, atol=0.1), (
            f"Max reward not at target: max at v={velocities[max_idx]}, target={target}"
        )

        # Rewards should decrease away from target
        for i in range(max_idx):
            assert rewards[i] < rewards[i+1], f"Not increasing toward target at idx {i}"
        for i in range(max_idx, len(rewards)-1):
            assert rewards[i] > rewards[i+1], f"Not decreasing away from target at idx {i}"

    @pytest.mark.sim
    def test_velocity_reward_always_positive(self):
        """
        Purpose: Verify velocity reward is always positive.

        Assertions:
        - Reward > 0 for all velocities (within reasonable range)
        """

        def velocity_reward(v_actual, v_target, sigma=0.25):
            error = v_actual - v_target
            return np.exp(-error**2 / sigma**2)

        # Test reasonable velocities (exponential approaches 0 for extreme values)
        reasonable_velocities = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
        for v in reasonable_velocities:
            r = velocity_reward(v, 0.5, sigma=0.25)
            assert r > 0, f"Reward should be positive: v={v}, r={r}"
            assert r <= 1.0, f"Reward should not exceed 1.0: v={v}, r={r}"


# =============================================================================
# Test 4.2: Slip Gating Test (Critical)
# =============================================================================


class TestSlipPenalty:
    """Tests for slip penalty gating correctness.

    Without proper gating, robot learns to avoid lifting feet.
    """

    CONTACT_THRESHOLD = 50.0  # N
    MIN_PENALTY = 0.01

    @pytest.mark.sim
    def test_slip_penalty_gating_concept(self):
        """
        Purpose: Verify slip penalty only applies when foot is loaded.

        Test Cases:
        A. Fn > threshold, tangential velocity > 0 → penalty applies
        B. Fn < threshold, tangential velocity > 0 → penalty suppressed
        C. Fn > threshold, tangential velocity = 0 → no penalty

        Why This Matters:
        Without proper gating, robot learns to avoid lifting feet.
        """

        def compute_slip_penalty(normal_force, tangential_velocity, threshold=50.0):
            """Slip penalty with contact gating."""
            # Gate by contact force
            is_contact = normal_force > threshold

            # Only penalize slip when foot is in contact
            if is_contact:
                return tangential_velocity ** 2
            else:
                return 0.0

        # Case A: Loaded foot with slip → penalty applies
        penalty_a = compute_slip_penalty(
            normal_force=100.0,
            tangential_velocity=0.5,
            threshold=self.CONTACT_THRESHOLD
        )
        assert penalty_a > self.MIN_PENALTY, (
            f"Case A: Expected penalty > {self.MIN_PENALTY}, got {penalty_a}"
        )

        # Case B: Unloaded foot with motion (swing phase) → penalty suppressed
        penalty_b = compute_slip_penalty(
            normal_force=10.0,
            tangential_velocity=1.0,
            threshold=self.CONTACT_THRESHOLD
        )
        assert penalty_b < self.MIN_PENALTY, (
            f"Case B: Penalty should be gated, got {penalty_b}"
        )

        # Case C: Loaded foot, no slip → no penalty
        penalty_c = compute_slip_penalty(
            normal_force=100.0,
            tangential_velocity=0.0,
            threshold=self.CONTACT_THRESHOLD
        )
        assert penalty_c < self.MIN_PENALTY, (
            f"Case C: No slip, expected no penalty, got {penalty_c}"
        )


# =============================================================================
# Test 4.3: Orientation Penalty
# =============================================================================


class TestOrientationPenalty:
    """Tests for orientation penalty correctness."""

    @pytest.mark.sim
    def test_orientation_penalty_increases_with_tilt(self):
        """
        Purpose: Verify orientation penalty increases with pitch/roll.

        Procedure:
        1. Test with pitch = 0 (upright)
        2. Test with pitch = 10°
        3. Test with pitch = 30°

        Assertions:
        - penalty(0°) < penalty(10°) < penalty(30°)
        - penalty(0°) ≈ 0
        """
        MAX_UPRIGHT_PENALTY = 0.01

        def compute_orientation_penalty(pitch, roll):
            """Orientation penalty based on squared angles."""
            return pitch**2 + roll**2

        penalty_0 = compute_orientation_penalty(pitch=0.0, roll=0.0)
        penalty_10 = compute_orientation_penalty(pitch=np.radians(10), roll=0.0)
        penalty_30 = compute_orientation_penalty(pitch=np.radians(30), roll=0.0)

        assert penalty_0 < MAX_UPRIGHT_PENALTY, (
            f"Upright penalty should be ~0: {penalty_0}"
        )
        assert penalty_0 < penalty_10 < penalty_30, (
            f"Penalty should increase with tilt: {penalty_0} < {penalty_10} < {penalty_30}"
        )

    @pytest.mark.sim
    def test_orientation_penalty_symmetric(self):
        """
        Purpose: Verify orientation penalty is symmetric for pitch/roll.

        Assertions:
        - penalty(+pitch) == penalty(-pitch)
        - penalty(pitch, roll) == penalty(roll, pitch) when using sum of squares
        """

        def compute_orientation_penalty(pitch, roll):
            return pitch**2 + roll**2

        # Symmetric for positive/negative
        assert np.isclose(
            compute_orientation_penalty(0.1, 0.0),
            compute_orientation_penalty(-0.1, 0.0)
        ), "Penalty should be symmetric for ±pitch"

        # Symmetric for pitch/roll swap
        assert np.isclose(
            compute_orientation_penalty(0.1, 0.2),
            compute_orientation_penalty(0.2, 0.1)
        ), "Penalty should be symmetric for pitch/roll swap"


# =============================================================================
# Test 4.4: Torque Penalty
# =============================================================================


class TestTorquePenalty:
    """Tests for torque penalty correctness."""

    @pytest.mark.sim
    def test_torque_penalty_sum_of_squares(self):
        """
        Purpose: Verify torque penalty uses sum of squares.

        Assertions:
        - penalty(0) = 0
        - penalty increases quadratically with torque
        """

        def compute_torque_penalty(torques):
            """Standard torque penalty: sum of squared torques."""
            return np.sum(torques**2)

        # Zero torque → zero penalty
        penalty_zero = compute_torque_penalty(np.zeros(8))
        assert penalty_zero == 0, f"Zero torque should have zero penalty: {penalty_zero}"

        # Quadratic increase
        torques_1 = np.ones(8)
        torques_2 = np.ones(8) * 2

        penalty_1 = compute_torque_penalty(torques_1)
        penalty_2 = compute_torque_penalty(torques_2)

        # penalty_2 should be 4x penalty_1 (quadratic)
        assert np.isclose(penalty_2, 4 * penalty_1), (
            f"Torque penalty should be quadratic: {penalty_2} vs 4*{penalty_1}"
        )

    @pytest.mark.sim
    def test_saturation_penalty(self):
        """
        Purpose: Verify saturation penalty penalizes torque near limits.

        Saturation penalty: sum of max(0, |τ| - τ_soft)^2

        Assertions:
        - Zero penalty when torques below soft limit
        - Increasing penalty as torques approach hard limit
        """
        SOFT_LIMIT = 3.0  # Nm
        HARD_LIMIT = 4.0  # Nm

        def compute_saturation_penalty(torques, soft_limit=3.0):
            """Saturation penalty for torques exceeding soft limit."""
            excess = np.maximum(0, np.abs(torques) - soft_limit)
            return np.sum(excess**2)

        # Below soft limit → zero penalty
        torques_low = np.ones(8) * 2.0  # 2 Nm each
        penalty_low = compute_saturation_penalty(torques_low, SOFT_LIMIT)
        assert penalty_low == 0, f"Below soft limit should have zero penalty: {penalty_low}"

        # At soft limit → zero penalty (just below threshold)
        torques_at_soft = np.ones(8) * SOFT_LIMIT
        penalty_at_soft = compute_saturation_penalty(torques_at_soft, SOFT_LIMIT)
        assert penalty_at_soft == 0, f"At soft limit should have zero penalty: {penalty_at_soft}"

        # Above soft limit → penalty
        torques_high = np.ones(8) * 3.5  # 0.5 Nm above soft limit
        penalty_high = compute_saturation_penalty(torques_high, SOFT_LIMIT)
        assert penalty_high > 0, f"Above soft limit should have penalty: {penalty_high}"


# =============================================================================
# Test 4.5: Healthy Reward
# =============================================================================


class TestHealthyReward:
    """Tests for healthy (survival) reward correctness."""

    @pytest.mark.sim
    def test_healthy_reward_binary(self):
        """
        Purpose: Verify healthy reward is binary (0 or 1).

        Assertions:
        - healthy_reward = 1 when robot is upright and within height limits
        - healthy_reward = 0 when terminated
        """

        def compute_healthy_reward(height, pitch, roll,
                                   min_height=0.2, max_height=0.6,
                                   max_pitch=0.5, max_roll=0.5):
            """Binary healthy reward."""
            height_ok = min_height <= height <= max_height
            orientation_ok = abs(pitch) < max_pitch and abs(roll) < max_roll
            return 1.0 if height_ok and orientation_ok else 0.0

        # Healthy state
        healthy = compute_healthy_reward(
            height=0.45, pitch=0.0, roll=0.0,
            min_height=0.2, max_height=0.6
        )
        assert healthy == 1.0, f"Should be healthy: {healthy}"

        # Fallen (low height)
        fallen_low = compute_healthy_reward(
            height=0.1, pitch=0.0, roll=0.0,
            min_height=0.2, max_height=0.6
        )
        assert fallen_low == 0.0, f"Should be unhealthy (low): {fallen_low}"

        # Fallen (tilted)
        fallen_tilt = compute_healthy_reward(
            height=0.45, pitch=0.6, roll=0.0,  # Exceeds max_pitch
            min_height=0.2, max_height=0.6
        )
        assert fallen_tilt == 0.0, f"Should be unhealthy (tilted): {fallen_tilt}"


# =============================================================================
# Test 4.6: Gait Periodicity Reward
# =============================================================================


class TestGaitPeriodicityReward:
    """Tests for gait periodicity reward correctness."""

    @pytest.mark.sim
    def test_gait_periodicity_xor_logic(self):
        """
        Purpose: Verify gait periodicity uses XOR logic.

        XOR(left_contact, right_contact) = 1 when exactly one foot is in contact.

        Assertions:
        - XOR = 1 when one foot grounded, one lifted
        - XOR = 0 when both grounded or both lifted
        """

        def compute_gait_periodicity(left_contact, right_contact):
            """Gait periodicity: XOR of foot contacts."""
            return float(left_contact != right_contact)

        # Case 1: Left grounded, right lifted → XOR = 1
        assert compute_gait_periodicity(True, False) == 1.0

        # Case 2: Right grounded, left lifted → XOR = 1
        assert compute_gait_periodicity(False, True) == 1.0

        # Case 3: Both grounded → XOR = 0 (double support)
        assert compute_gait_periodicity(True, True) == 0.0

        # Case 4: Both lifted → XOR = 0 (flight phase)
        assert compute_gait_periodicity(False, False) == 0.0

    @pytest.mark.sim
    def test_gait_periodicity_soft_version(self):
        """
        Purpose: Verify soft gait periodicity handles continuous values.

        For continuous contact forces, use:
        periodicity = |F_left - F_right| / max(F_left + F_right, ε)

        Assertions:
        - Periodicity = 1 when one foot has 100% weight
        - Periodicity = 0 when weight is evenly distributed
        """

        def compute_soft_gait_periodicity(left_force, right_force, eps=1e-6):
            """Soft gait periodicity based on force asymmetry."""
            total = left_force + right_force
            if total < eps:
                return 0.0  # Both feet in air
            return abs(left_force - right_force) / total

        # All weight on left → periodicity = 1
        assert np.isclose(compute_soft_gait_periodicity(100.0, 0.0), 1.0)

        # All weight on right → periodicity = 1
        assert np.isclose(compute_soft_gait_periodicity(0.0, 100.0), 1.0)

        # Even distribution → periodicity = 0
        assert np.isclose(compute_soft_gait_periodicity(50.0, 50.0), 0.0)

        # Partial asymmetry → 0 < periodicity < 1
        p = compute_soft_gait_periodicity(70.0, 30.0)
        assert 0 < p < 1, f"Partial asymmetry should give 0 < p < 1: {p}"


# =============================================================================
# Test 4.7: Foot Clearance Reward
# =============================================================================


class TestFootClearanceReward:
    """Tests for foot clearance reward correctness."""

    @pytest.mark.sim
    def test_foot_clearance_increases_with_height(self):
        """
        Purpose: Verify foot clearance reward increases with swing foot height.

        Assertions:
        - reward(h=0) < reward(h=0.03) < reward(h=0.05)
        """

        def compute_clearance_reward(foot_height, target_height=0.05):
            """Clearance reward: encourage foot height during swing."""
            # Saturating reward: approaches 1 as height increases
            return min(1.0, foot_height / target_height)

        r_0 = compute_clearance_reward(0.0)
        r_03 = compute_clearance_reward(0.03)
        r_05 = compute_clearance_reward(0.05)

        assert r_0 < r_03 < r_05 or np.isclose(r_03, r_05), (
            f"Clearance should increase with height: {r_0} < {r_03} <= {r_05}"
        )
        # At target, reward should saturate
        r_10 = compute_clearance_reward(0.10)
        assert np.isclose(r_05, r_10), f"Reward should saturate at target: r_05={r_05}, r_10={r_10}"

    @pytest.mark.sim
    def test_foot_clearance_gated_by_swing(self):
        """
        Purpose: Verify clearance only rewards swing leg, not stance leg.

        Assertions:
        - Stance leg (loaded) should not get clearance reward
        - Swing leg (unloaded) should get clearance reward based on height
        """
        CONTACT_THRESHOLD = 50.0  # N
        WIDTH = 0.22

        # Left swing phase center (pi/2): unloaded left foot should get reward.
        r_left_phase = compute_phase_gated_clearance_reward(
            left_clearance=jp.asarray(0.03),
            right_clearance=jp.asarray(0.00),
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(CONTACT_THRESHOLD),
            phase=jp.asarray(0.5 * np.pi),
            phase_gate_width=jp.asarray(WIDTH),
        )
        assert float(r_left_phase) > 0.0, f"Expected left-swing clearance reward > 0: {r_left_phase}"

        # Wrong phase (right swing center): same foot should not be rewarded.
        r_wrong_phase = compute_phase_gated_clearance_reward(
            left_clearance=jp.asarray(0.03),
            right_clearance=jp.asarray(0.00),
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(CONTACT_THRESHOLD),
            phase=jp.asarray(1.5 * np.pi),
            phase_gate_width=jp.asarray(WIDTH),
        )
        assert np.isclose(float(r_wrong_phase), 0.0), (
            f"Expected zero reward in wrong swing window: {r_wrong_phase}"
        )

        # Loaded foot should get no reward even in correct phase.
        r_loaded = compute_phase_gated_clearance_reward(
            left_clearance=jp.asarray(0.03),
            right_clearance=jp.asarray(0.00),
            left_force=jp.asarray(100.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(CONTACT_THRESHOLD),
            phase=jp.asarray(0.5 * np.pi),
            phase_gate_width=jp.asarray(WIDTH),
        )
        assert np.isclose(float(r_loaded), 0.0), f"Loaded foot should not get clearance reward: {r_loaded}"


# =============================================================================
# Test 4.8: v0.15.9 Propulsion Rewards
# =============================================================================


class TestPropulsionRewardsV0159:
    """Focused tests for v0.15.9 step-length and cycle-progress semantics."""

    @pytest.mark.sim
    def test_step_length_reward_scales_with_cmd_and_clamps_negative_step(self):
        """Touchdown step-length target should scale with command and clamp negative landings."""

        # Perfect touchdown at target should be near-max.
        cmd = 0.2
        target = 0.035 + 0.30 * cmd
        r_match = compute_step_length_touchdown_reward(
            left_x=jp.asarray(target),
            right_x=jp.asarray(-1.0),
            touchdown_left=jp.asarray(True),
            touchdown_right=jp.asarray(False),
            velocity_cmd=jp.asarray(cmd),
            step_reward_gate=jp.asarray(1.0),
            target_base=jp.asarray(0.035),
            target_scale=jp.asarray(0.30),
            sigma=jp.asarray(0.03),
        )
        assert np.isclose(r_match, 1.0), f"Expected near-1 reward at target, got {r_match}"

        # Backward touchdown should be clamped to 0 and strongly penalized for moving command.
        r_backward = compute_step_length_touchdown_reward(
            left_x=jp.asarray(-0.05),
            right_x=jp.asarray(-1.0),
            touchdown_left=jp.asarray(True),
            touchdown_right=jp.asarray(False),
            velocity_cmd=jp.asarray(cmd),
            step_reward_gate=jp.asarray(1.0),
            target_base=jp.asarray(0.035),
            target_scale=jp.asarray(0.30),
            sigma=jp.asarray(0.03),
        )
        r_forward = r_match
        assert r_backward < r_forward, (
            f"Backward touchdown should score lower than target landing: {r_backward} < {r_forward}"
        )

        # Negative command should not reduce target below base (max(cmd, 0) behavior).
        r_neg_cmd = compute_step_length_touchdown_reward(
            left_x=jp.asarray(0.035),
            right_x=jp.asarray(-1.0),
            touchdown_left=jp.asarray(True),
            touchdown_right=jp.asarray(False),
            velocity_cmd=jp.asarray(-0.2),
            step_reward_gate=jp.asarray(1.0),
            target_base=jp.asarray(0.035),
            target_scale=jp.asarray(0.30),
            sigma=jp.asarray(0.03),
        )
        r_zero_cmd = compute_step_length_touchdown_reward(
            left_x=jp.asarray(0.035),
            right_x=jp.asarray(-1.0),
            touchdown_left=jp.asarray(True),
            touchdown_right=jp.asarray(False),
            velocity_cmd=jp.asarray(0.0),
            step_reward_gate=jp.asarray(1.0),
            target_base=jp.asarray(0.035),
            target_scale=jp.asarray(0.30),
            sigma=jp.asarray(0.03),
        )
        assert np.isclose(r_neg_cmd, r_zero_cmd), (
            f"Negative cmd should clamp to base target: {r_neg_cmd} vs {r_zero_cmd}"
        )

    @pytest.mark.sim
    def test_cycle_progress_reward_only_on_cycle_complete(self):
        """Cycle progress should integrate heading-local vx*dt and pay only on cycle boundary."""

        dt = 0.02
        stride_steps = 4
        velocity_cmd = 0.2
        forward_vel = 0.2

        progress = 0.0
        rewards = []
        for step in [1, 2, 3, 4]:
            reward, cycle_delta, complete, progress = compute_cycle_progress_terms(
                cycle_progress_accum=jp.asarray(progress),
                forward_vel=jp.asarray(forward_vel),
                dt=dt,
                current_step_count=jp.asarray(step),
                stride_steps=jp.asarray(stride_steps),
                velocity_cmd=jp.asarray(velocity_cmd),
                target_scale=jp.asarray(1.0),
                sigma=jp.asarray(0.06),
            )
            rewards.append(float(reward))

        # No reward before boundary, then reward at completion.
        assert rewards[0] == 0.0 and rewards[1] == 0.0 and rewards[2] == 0.0, (
            f"Cycle reward must be zero before completion: {rewards}"
        )
        assert rewards[3] > 0.9, f"Cycle-complete reward should be high at target progress: {rewards[3]}"
        assert np.isclose(progress, 0.0), f"Cycle accumulator should reset on completion, got {progress}"


class TestRewardEconomicsV01510:
    """Focused tests for v0.15.10 reward-economics helpers."""

    @pytest.mark.sim
    def test_zero_baseline_forward_reward_for_commanded_walking(self):
        """Standing should be near-zero reward for commanded walking after baseline subtraction."""
        scale = jp.asarray(6.0)
        cmd_min = jp.asarray(0.08)
        velocity_cmd = jp.asarray(0.16)

        r_stand = compute_zero_baseline_forward_reward(
            forward_vel=jp.asarray(0.0),
            velocity_cmd=velocity_cmd,
            scale=scale,
            velocity_cmd_min=cmd_min,
        )
        r_move = compute_zero_baseline_forward_reward(
            forward_vel=jp.asarray(0.16),
            velocity_cmd=velocity_cmd,
            scale=scale,
            velocity_cmd_min=cmd_min,
        )
        assert np.isclose(float(r_stand), 0.0, atol=1e-6), f"Standing baseline should be ~0: {r_stand}"
        assert float(r_move) > 0.5, f"Matched command should keep strong reward: {r_move}"
        r_backward = compute_zero_baseline_forward_reward(
            forward_vel=jp.asarray(-0.16),
            velocity_cmd=velocity_cmd,
            scale=scale,
            velocity_cmd_min=cmd_min,
        )
        assert float(r_backward) < 0.0, f"Backward tracking should be negative vs standing baseline: {r_backward}"

    @pytest.mark.sim
    def test_dense_progress_reward_heading_local_semantics(self):
        """Dense progress should be command-gated and increase with positive heading-local velocity."""
        cmd_min = jp.asarray(0.08)
        velocity_cmd = jp.asarray(0.16)

        r_zero = compute_dense_progress_reward(
            forward_vel=jp.asarray(0.0),
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=cmd_min,
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(50.0),
        )
        r_half = compute_dense_progress_reward(
            forward_vel=jp.asarray(0.08),
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=cmd_min,
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(50.0),
        )
        r_match = compute_dense_progress_reward(
            forward_vel=jp.asarray(0.16),
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=cmd_min,
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(50.0),
        )
        r_small_cmd = compute_dense_progress_reward(
            forward_vel=jp.asarray(0.10),
            velocity_cmd=jp.asarray(0.02),
            velocity_cmd_min=cmd_min,
            left_force=jp.asarray(0.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(50.0),
        )
        r_no_support = compute_dense_progress_reward(
            forward_vel=jp.asarray(0.16),
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=cmd_min,
            left_force=jp.asarray(100.0),
            right_force=jp.asarray(100.0),
            contact_threshold=jp.asarray(50.0),
        )
        assert np.isclose(float(r_zero), 0.0), f"No forward speed should give zero dense progress: {r_zero}"
        assert float(r_zero) < float(r_half) < float(r_match), (
            f"Dense reward should rise with forward speed: {r_zero}, {r_half}, {r_match}"
        )
        assert np.isclose(float(r_small_cmd), 0.0), f"Below cmd gate should disable dense progress: {r_small_cmd}"
        assert np.isclose(float(r_no_support), 0.0), (
            f"Dense progress should not open without single-support stepping evidence: {r_no_support}"
        )

    @pytest.mark.sim
    def test_propulsion_quality_gate_depends_on_propulsion_signals(self):
        """Propulsion gate should follow propulsion quality max(step_length, dense_progress, cycle_progress)."""
        gate_low = compute_propulsion_quality_gate(
            step_length_reward=jp.asarray(0.0),
            dense_progress_reward=jp.asarray(0.0),
            cycle_progress_reward=jp.asarray(0.0),
        )
        gate_dense = compute_propulsion_quality_gate(
            step_length_reward=jp.asarray(0.0),
            dense_progress_reward=jp.asarray(0.6),
            cycle_progress_reward=jp.asarray(0.1),
        )
        gate_step = compute_propulsion_quality_gate(
            step_length_reward=jp.asarray(0.7),
            dense_progress_reward=jp.asarray(0.2),
            cycle_progress_reward=jp.asarray(0.1),
        )
        assert np.isclose(float(gate_low), 0.0), f"No propulsion signals should keep gate closed: {gate_low}"
        assert np.isclose(float(gate_dense), 0.6), f"Gate should follow dense progress max: {gate_dense}"
        assert np.isclose(float(gate_step), 0.7), f"Gate should follow step-length max: {gate_step}"
