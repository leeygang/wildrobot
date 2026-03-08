"""Unit tests for M3 foot-placement step controller (training/envs/step_controller.py).

Tests cover:
- need_step gate values
- swing-foot selection
- step-target clamping
- swing-trajectory interpolation
- FSM state transitions (STANCE→SWING, SWING→TOUCHDOWN_RECOVER, RECOVER→STANCE)
- ctrl_base shape and bounds
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jp

# Make sure the project root is on sys.path (handled by conftest.py)
from training.envs import step_controller as sc


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DEFAULT_FSM = dict(
    phase=jp.int32(sc.STANCE),
    swing_foot=jp.int32(0),
    phase_ticks=jp.int32(0),
    frozen_tx=jp.float32(0.0),
    frozen_ty=jp.float32(0.0),
    swing_sx=jp.float32(0.0),
    swing_sy=jp.float32(0.0),
    touch_hold=jp.int32(0),
    trigger_hold=jp.int32(0),
)

_FSM_KWARGS = dict(
    trigger_threshold=0.45,
    recover_threshold=0.20,
    trigger_hold_ticks=2,
    touch_hold_ticks=1,
    swing_timeout_ticks=12,
    y_nominal_m=0.115,
    k_lat_vel=0.15,
    k_roll=0.10,
    k_fwd_vel=0.05,
    k_pitch=0.05,
    x_nominal_m=0.0,
    x_step_min_m=-0.08,
    x_step_max_m=0.12,
    y_step_inner_m=0.08,
    y_step_outer_m=0.20,
    step_max_delta_m=0.12,
)

_N_ACT = 9  # minimal actuator count for tests


def _baseline_action(n: int = _N_ACT) -> jp.ndarray:
    return jp.zeros((n,), dtype=jp.float32)


# ---------------------------------------------------------------------------
# compute_need_step
# ---------------------------------------------------------------------------

class TestComputeNeedStep:
    def test_upright_robot_zero(self):
        """Near-zero state → need_step near 0."""
        ns = sc.compute_need_step(
            pitch=0.0, roll=0.0, lateral_vel=0.0, pitch_rate=0.0, healthy=1.0,
            g_pitch=0.35, g_roll=0.35, g_lat=0.30, g_pr=1.00,
        )
        assert float(ns) < 0.05

    def test_large_roll_raises(self):
        """Large roll → need_step close to 1."""
        ns = sc.compute_need_step(
            pitch=0.0, roll=0.70, lateral_vel=0.0, pitch_rate=0.0, healthy=1.0,
            g_pitch=0.35, g_roll=0.35, g_lat=0.30, g_pr=1.00,
        )
        assert float(ns) > 0.4

    def test_unhealthy_zeroes_out(self):
        """When healthy=0, need_step must be 0 regardless of state."""
        ns = sc.compute_need_step(
            pitch=1.0, roll=1.0, lateral_vel=2.0, pitch_rate=3.0, healthy=0.0,
            g_pitch=0.35, g_roll=0.35, g_lat=0.30, g_pr=1.00,
        )
        assert float(ns) == pytest.approx(0.0, abs=1e-5)

    def test_clipped_to_one(self):
        """Very large disturbance → need_step capped at 1.0."""
        ns = sc.compute_need_step(
            pitch=10.0, roll=10.0, lateral_vel=10.0, pitch_rate=20.0, healthy=1.0,
            g_pitch=0.35, g_roll=0.35, g_lat=0.30, g_pr=1.00,
        )
        assert float(ns) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# select_swing_foot
# ---------------------------------------------------------------------------

class TestSelectSwingFoot:
    def test_loads_right_more_swing_left(self):
        """Right foot more loaded → swing left."""
        foot = sc.select_swing_foot(
            roll=0.0, lateral_vel=0.0,
            prev_left_loaded=jp.float32(0.0),
            prev_right_loaded=jp.float32(1.0),
        )
        assert int(foot) == 0  # left

    def test_loads_left_more_swing_right(self):
        """Left foot more loaded → swing right."""
        foot = sc.select_swing_foot(
            roll=0.0, lateral_vel=0.0,
            prev_left_loaded=jp.float32(1.0),
            prev_right_loaded=jp.float32(0.0),
        )
        assert int(foot) == 1  # right

    def test_bias_positive_lateral_swing_right(self):
        """Equal loads + positive lateral vel → swing right (catch step)."""
        foot = sc.select_swing_foot(
            roll=0.0, lateral_vel=0.5,
            prev_left_loaded=jp.float32(0.5),
            prev_right_loaded=jp.float32(0.5),
        )
        assert int(foot) == 1  # right

    def test_bias_negative_lateral_swing_left(self):
        """Equal loads + negative lateral vel → swing left."""
        foot = sc.select_swing_foot(
            roll=0.0, lateral_vel=-0.5,
            prev_left_loaded=jp.float32(0.5),
            prev_right_loaded=jp.float32(0.5),
        )
        assert int(foot) == 0  # left


# ---------------------------------------------------------------------------
# compute_step_target
# ---------------------------------------------------------------------------

class TestComputeStepTarget:
    def _call(self, swing_foot, lat_vel=0.0, roll=0.0, fwd_vel=0.0, pitch=0.0):
        return sc.compute_step_target(
            lateral_vel=jp.float32(lat_vel),
            roll=jp.float32(roll),
            forward_vel=jp.float32(fwd_vel),
            pitch=jp.float32(pitch),
            swing_foot=jp.int32(swing_foot),
            y_nominal_m=0.115,
            k_lat_vel=0.15,
            k_roll=0.10,
            k_fwd_vel=0.05,
            k_pitch=0.05,
            x_nominal_m=0.0,
            x_step_min_m=-0.08,
            x_step_max_m=0.12,
            y_step_inner_m=0.08,
            y_step_outer_m=0.20,
        )

    def test_left_foot_positive_y(self):
        x, y = self._call(swing_foot=0)
        assert float(y) > 0.0, "Left foot should be on +y side"

    def test_right_foot_negative_y(self):
        x, y = self._call(swing_foot=1)
        assert float(y) < 0.0, "Right foot should be on -y side"

    def test_x_clamped(self):
        """X target should never exceed bounds."""
        x, _ = self._call(swing_foot=0, fwd_vel=100.0, pitch=10.0)
        assert float(x) <= 0.12 + 1e-5

        x, _ = self._call(swing_foot=0, fwd_vel=-100.0, pitch=-10.0)
        assert float(x) >= -0.08 - 1e-5

    def test_lateral_correction_left(self):
        """Positive lateral vel → moves left foot more +y (wider catch)."""
        _, y_base = self._call(swing_foot=0, lat_vel=0.0)
        _, y_lat = self._call(swing_foot=0, lat_vel=0.5)
        assert float(y_lat) > float(y_base)

    def test_y_bounds_respected(self):
        """Y target should stay within [inner, outer] for both feet."""
        for swing in (0, 1):
            for lat in (-5.0, 0.0, 5.0):
                _, y = self._call(swing_foot=swing, lat_vel=lat)
                if swing == 0:
                    assert 0.08 - 1e-5 <= float(y) <= 0.20 + 1e-5
                else:
                    assert -0.20 - 1e-5 <= float(y) <= -0.08 + 1e-5

    def test_step_max_delta_clamp(self):
        """step_max_delta_m must clamp target to within delta of current foot pos."""
        # Place current foot at origin; target would otherwise be at (0, 0.115).
        # With delta=0.05, y should be clamped to 0.0+0.05=0.05.
        x, y = sc.compute_step_target(
            lateral_vel=jp.float32(0.0),
            roll=jp.float32(0.0),
            forward_vel=jp.float32(0.0),
            pitch=jp.float32(0.0),
            swing_foot=jp.int32(0),
            y_nominal_m=0.115,
            k_lat_vel=0.15, k_roll=0.10, k_fwd_vel=0.05, k_pitch=0.05,
            x_nominal_m=0.0,
            x_step_min_m=-0.08, x_step_max_m=0.12,
            y_step_inner_m=0.08, y_step_outer_m=0.20,
            current_foot_x=jp.float32(0.0),
            current_foot_y=jp.float32(0.0),
            step_max_delta_m=0.05,
        )
        assert float(y) <= 0.0 + 0.05 + 1e-5, (
            f"y target {float(y):.4f} exceeds current_foot_y + step_max_delta_m (0.05)"
        )

    def test_step_max_delta_not_applied_when_None(self):
        """When current_foot_x/y are None (default), no delta clamp is applied."""
        # Large nominal y=0.115 is reachable with no current foot supplied
        x, y = sc.compute_step_target(
            lateral_vel=jp.float32(0.0),
            roll=jp.float32(0.0),
            forward_vel=jp.float32(0.0),
            pitch=jp.float32(0.0),
            swing_foot=jp.int32(0),
            y_nominal_m=0.115,
            k_lat_vel=0.15, k_roll=0.10, k_fwd_vel=0.05, k_pitch=0.05,
            x_nominal_m=0.0,
            x_step_min_m=-0.08, x_step_max_m=0.12,
            y_step_inner_m=0.08, y_step_outer_m=0.20,
            # current_foot_x/y not provided → delta clamp inactive
        )
        assert float(y) == pytest.approx(0.115, abs=0.01)

# ---------------------------------------------------------------------------
# compute_swing_trajectory
# ---------------------------------------------------------------------------

class TestComputeSwingTrajectory:
    def _call(self, ticks, start_x=0.0, start_y=0.0, tgt_x=0.10, tgt_y=0.15, need_step=0.5):
        return sc.compute_swing_trajectory(
            phase_ticks=jp.int32(ticks),
            swing_start_x=jp.float32(start_x),
            swing_start_y=jp.float32(start_y),
            frozen_target_x=jp.float32(tgt_x),
            frozen_target_y=jp.float32(tgt_y),
            swing_height_m=0.04,
            swing_duration_ticks=10,
            need_step=jp.float32(need_step),
        )

    def test_at_start_near_swing_start(self):
        """At tick 0 the foot should be close to the swing start."""
        x, y, z = self._call(0)
        assert float(x) == pytest.approx(0.0, abs=0.01)
        assert float(y) == pytest.approx(0.0, abs=0.01)
        assert float(z) == pytest.approx(0.0, abs=0.01)

    def test_at_end_near_target(self):
        """At tick = duration the foot should be close to target."""
        x, y, z = self._call(10)
        assert float(x) == pytest.approx(0.10, abs=0.01)
        assert float(y) == pytest.approx(0.15, abs=0.01)
        assert float(z) == pytest.approx(0.0, abs=0.01)

    def test_peak_z_at_midpoint(self):
        """Z should peak near midswing."""
        _, _, z_mid = self._call(5)
        _, _, z_start = self._call(0)
        _, _, z_end = self._call(10)
        assert float(z_mid) > float(z_start)
        assert float(z_mid) > float(z_end)

    def test_need_step_increases_height(self):
        """Higher need_step → higher peak swing height."""
        _, _, z_low = self._call(5, need_step=0.0)
        _, _, z_high = self._call(5, need_step=1.0)
        assert float(z_high) >= float(z_low)


# ---------------------------------------------------------------------------
# update_fsm: transition tests
# ---------------------------------------------------------------------------

def _run_fsm(n_steps, phase_in, swing_foot_in, phase_ticks_in=0,
             frozen_tx=0.0, frozen_ty=0.0,
             swing_sx=0.0, swing_sy=0.0,
             touch_hold=0, trigger_hold=0,
             need_step=0.6,
             loaded_left=0, loaded_right=1,
             left_foot_x=0.0, left_foot_y=0.115,
             right_foot_x=0.0, right_foot_y=-0.115,
             lateral_vel=0.0, roll=0.0, forward_vel=0.0, pitch=0.0):
    """Run FSM for `n_steps` ticks starting from given state, returning final state tuple."""
    state = (
        jp.int32(phase_in), jp.int32(swing_foot_in), jp.int32(phase_ticks_in),
        jp.float32(frozen_tx), jp.float32(frozen_ty),
        jp.float32(swing_sx), jp.float32(swing_sy),
        jp.int32(touch_hold), jp.int32(trigger_hold),
    )
    for _ in range(n_steps):
        state = sc.update_fsm(
            *state,
            need_step=jp.float32(need_step),
            loaded_left=jp.int32(loaded_left),
            loaded_right=jp.int32(loaded_right),
            left_foot_x=jp.float32(left_foot_x),
            left_foot_y=jp.float32(left_foot_y),
            right_foot_x=jp.float32(right_foot_x),
            right_foot_y=jp.float32(right_foot_y),
            lateral_vel=jp.float32(lateral_vel),
            roll=jp.float32(roll),
            forward_vel=jp.float32(forward_vel),
            pitch=jp.float32(pitch),
            **_FSM_KWARGS,
        )
    return state


class TestFSMTransitions:
    def test_stance_no_trigger(self):
        """Stays in STANCE when need_step is below threshold."""
        state = _run_fsm(5, phase_in=sc.STANCE, swing_foot_in=0, need_step=0.10)
        assert int(state[0]) == sc.STANCE

    def test_stance_to_swing_after_hold(self):
        """STANCE → SWING after trigger_hold_ticks consecutive ticks above threshold."""
        # trigger_hold_ticks = 2 in _FSM_KWARGS
        state = _run_fsm(3, phase_in=sc.STANCE, swing_foot_in=0, need_step=0.60)
        assert int(state[0]) == sc.SWING, (
            f"Expected SWING after trigger hold, got phase={int(state[0])}"
        )

    def test_swing_to_touchdown_recover_on_load(self):
        """SWING → TOUCHDOWN_RECOVER when swing foot is loaded."""
        # Start in SWING phase, swing_foot=0 (left), loaded_left=1 → touchdown
        state = _run_fsm(
            2,
            phase_in=sc.SWING, swing_foot_in=0, phase_ticks_in=3,
            frozen_tx=0.05, frozen_ty=0.115, swing_sx=0.0, swing_sy=0.115,
            need_step=0.60,
            loaded_left=1, loaded_right=1,  # both loaded → touchdown left detected
        )
        assert int(state[0]) == sc.TOUCHDOWN_RECOVER, (
            f"Expected TOUCHDOWN_RECOVER, got phase={int(state[0])}"
        )

    def test_swing_timeout_triggers_touchdown(self):
        """SWING → TOUCHDOWN_RECOVER on swing timeout even if foot not loaded."""
        # swing_timeout_ticks=12; run 15 ticks in SWING with swing foot unloaded
        state = _run_fsm(
            15,
            phase_in=sc.SWING, swing_foot_in=0, phase_ticks_in=0,
            frozen_tx=0.05, frozen_ty=0.115, swing_sx=0.0, swing_sy=0.115,
            need_step=0.70,
            loaded_left=0, loaded_right=1,  # swing foot (left) NOT loaded
        )
        assert int(state[0]) == sc.TOUCHDOWN_RECOVER, (
            f"Expected TOUCHDOWN_RECOVER after timeout, got phase={int(state[0])}"
        )

    def test_touchdown_recover_to_stance(self):
        """TOUCHDOWN_RECOVER → STANCE when need_step drops below recover_threshold."""
        # recover_threshold=0.20; run with low need_step
        state = _run_fsm(
            3,
            phase_in=sc.TOUCHDOWN_RECOVER, swing_foot_in=0, phase_ticks_in=0,
            need_step=0.10,  # below 0.20
            loaded_left=1, loaded_right=1,
        )
        assert int(state[0]) == sc.STANCE, (
            f"Expected STANCE after recovery with low need_step, got phase={int(state[0])}"
        )

    def test_touchdown_recover_stays_if_need_step_high(self):
        """TOUCHDOWN_RECOVER does not reset to STANCE while need_step > recover_threshold.

        With need_step=0.50 >> recover_threshold=0.20, the FSM must NOT transition
        back to STANCE in a single tick.  It may re-enter SWING (need_step also
        exceeds trigger_threshold=0.45 + 2-tick hold), but STANCE is not valid.
        """
        # Run only 1 tick so we can observe the immediate RECOVER→* decision
        # before a possible re-trigger into SWING completes its hold counter.
        state = _run_fsm(
            1,
            phase_in=sc.TOUCHDOWN_RECOVER, swing_foot_in=0, phase_ticks_in=0,
            need_step=0.50,  # above recover_threshold=0.20
            loaded_left=1, loaded_right=1,
        )
        assert int(state[0]) != sc.STANCE, (
            f"TOUCHDOWN_RECOVER with need_step=0.50 > recover_threshold=0.20 "
            f"must not reset to STANCE; got phase={int(state[0])}"
        )


# ---------------------------------------------------------------------------
# compute_ctrl_base: shape and bounds
# ---------------------------------------------------------------------------

class TestComputeCtrlBase:
    def _call(self, phase=sc.STANCE, swing_foot=0, phase_ticks=0, need_step=0.5):
        return sc.compute_ctrl_base(
            phase=jp.int32(phase),
            swing_foot=jp.int32(swing_foot),
            phase_ticks=jp.int32(phase_ticks),
            frozen_tx=jp.float32(0.10),
            frozen_ty=jp.float32(0.12),
            swing_sx=jp.float32(0.0),
            swing_sy=jp.float32(0.115),
            pitch=jp.float32(0.1),
            roll=jp.float32(0.05),
            pitch_rate=jp.float32(0.2),
            roll_rate=jp.float32(0.1),
            need_step=jp.float32(need_step),
            left_foot_x=jp.float32(0.0),
            left_foot_y=jp.float32(0.115),
            right_foot_x=jp.float32(0.0),
            right_foot_y=jp.float32(-0.115),
            default_action=_baseline_action(_N_ACT),
            # Map a few actuators (use simple 0-based indices for test)
            idx_left_hip_pitch=0,
            idx_right_hip_pitch=1,
            idx_left_hip_roll=2,
            idx_right_hip_roll=3,
            idx_left_knee=4,
            idx_right_knee=5,
            idx_left_ankle=6,
            idx_right_ankle=7,
            idx_waist=8,
            base_pitch_kp=0.20, base_pitch_kd=0.04,
            base_roll_kp=0.20, base_roll_kd=0.04,
            hip_pitch_gain=1.0, ankle_pitch_gain=0.7, hip_roll_gain=1.0,
            base_action_clip=0.25,
            swing_x_to_hip_pitch=0.30, swing_y_to_hip_roll=0.30,
            swing_z_to_knee=0.40, swing_z_to_ankle=0.20,
            swing_height_m=0.04, swing_duration_ticks=10,
            swing_height_need_step_mult=0.5,
            arm_enabled=True, arm_need_step_threshold=0.35,
            arm_k_roll=0.10, arm_k_roll_rate=0.05, arm_k_pitch_rate=0.03,
            arm_max_delta_rad=0.25,
        )

    def test_shape(self):
        out = self._call()
        assert out.shape == (_N_ACT,)

    def test_bounds(self):
        """Output must stay within policy-action bounds [-1, 1]."""
        for phase in (sc.STANCE, sc.SWING, sc.TOUCHDOWN_RECOVER):
            out = self._call(phase=phase, need_step=1.0)
            assert jp.all(out >= -1.0), f"Phase {phase}: output below -1"
            assert jp.all(out <= 1.0), f"Phase {phase}: output above +1"

    def test_missing_joints_no_op(self):
        """Passing idx=-1 for all joints should give default_action back."""
        out = sc.compute_ctrl_base(
            phase=jp.int32(sc.STANCE),
            swing_foot=jp.int32(0),
            phase_ticks=jp.int32(0),
            frozen_tx=jp.float32(0.0), frozen_ty=jp.float32(0.0),
            swing_sx=jp.float32(0.0), swing_sy=jp.float32(0.0),
            pitch=jp.float32(0.0), roll=jp.float32(0.0),
            pitch_rate=jp.float32(0.0), roll_rate=jp.float32(0.0),
            need_step=jp.float32(0.0),
            left_foot_x=jp.float32(0.0), left_foot_y=jp.float32(0.0),
            right_foot_x=jp.float32(0.0), right_foot_y=jp.float32(0.0),
            default_action=jp.zeros((_N_ACT,), dtype=jp.float32),
            # All joints absent
            idx_left_hip_pitch=-1, idx_right_hip_pitch=-1,
            idx_left_hip_roll=-1, idx_right_hip_roll=-1,
            idx_left_knee=-1, idx_right_knee=-1,
            idx_left_ankle=-1, idx_right_ankle=-1,
            idx_waist=-1,
            base_pitch_kp=0.20, base_pitch_kd=0.04,
            base_roll_kp=0.20, base_roll_kd=0.04,
            hip_pitch_gain=1.0, ankle_pitch_gain=0.7, hip_roll_gain=1.0,
            base_action_clip=0.25,
            swing_x_to_hip_pitch=0.30, swing_y_to_hip_roll=0.30,
            swing_z_to_knee=0.40, swing_z_to_ankle=0.20,
            swing_height_m=0.04, swing_duration_ticks=10,
            swing_height_need_step_mult=0.5,
            arm_enabled=False, arm_need_step_threshold=0.35,
            arm_k_roll=0.10, arm_k_roll_rate=0.05, arm_k_pitch_rate=0.03,
            arm_max_delta_rad=0.25,
        )
        assert jp.allclose(out, jp.zeros((_N_ACT,))), (
            "With all idx=-1 and zero input, output should equal default_action (zeros)"
        )
