"""Deterministic tests for finalized recovery summary metrics."""

import jax.numpy as jnp
import pytest

from training.core.metrics_registry import (
    METRIC_INDEX,
    METRIC_SPECS,
    Reducer,
    aggregate_metrics,
)
from training.envs.disturbance import DisturbanceSchedule
from training.envs.wildrobot_env import (
    RECOVERY_WINDOW_STEPS,
    _update_recovery_tracking,
    compute_com_velocity_damping_reward,
    compute_disturbed_window_reward_weights,
    compute_height_floor_penalty,
)


def _schedule(start: int = 50, end: int = 60) -> DisturbanceSchedule:
    return DisturbanceSchedule(
        start_step=jnp.asarray(start, dtype=jnp.int32),
        end_step=jnp.asarray(end, dtype=jnp.int32),
        force_xy=jnp.zeros((2,), dtype=jnp.float32),
        body_id=jnp.asarray(0, dtype=jnp.int32),
        rng=jnp.asarray([0, 0], dtype=jnp.uint32),
    )


def _initial_recovery_state() -> dict[str, jnp.ndarray]:
    return {
        "recovery_active": jnp.asarray(False, dtype=jnp.bool_),
        "recovery_age": jnp.asarray(0, dtype=jnp.int32),
        "recovery_last_support_foot": jnp.asarray(-1, dtype=jnp.int32),
        "recovery_first_liftoff_recorded": jnp.asarray(False, dtype=jnp.bool_),
        "recovery_first_liftoff_latency": jnp.asarray(0, dtype=jnp.int32),
        "recovery_first_touchdown_recorded": jnp.asarray(False, dtype=jnp.bool_),
        "recovery_first_step_latency": jnp.asarray(0, dtype=jnp.int32),
        "recovery_first_touchdown_age": jnp.asarray(-1, dtype=jnp.int32),
        "recovery_visible_step_recorded": jnp.asarray(False, dtype=jnp.bool_),
        "recovery_touchdown_count": jnp.asarray(0, dtype=jnp.int32),
        "recovery_support_foot_changes": jnp.asarray(0, dtype=jnp.int32),
        "recovery_pitch_rate_at_push_end": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_pitch_rate_at_touchdown": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_pitch_rate_after_10t": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_capture_error_at_push_end": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_capture_error_at_touchdown": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_capture_error_after_10t": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_first_step_dx": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_first_step_dy": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_first_step_target_err_x": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_first_step_target_err_y": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_min_height": jnp.asarray(0.0, dtype=jnp.float32),
        "recovery_max_knee_flex": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _step_recovery(
    state: dict[str, jnp.ndarray],
    *,
    step: int,
    touchdown_left: bool = False,
    touchdown_right: bool = False,
    liftoff_left: bool = False,
    liftoff_right: bool = False,
    done: bool = False,
    failed: bool = False,
    speed: float = 0.0,
    pitch_rate: float = 0.0,
    capture_error_norm: float = 0.0,
    first_step_dx: float = 0.0,
    first_step_dy: float = 0.0,
    first_step_target_err_x: float = 0.0,
    first_step_target_err_y: float = 0.0,
    current_height: float = 0.0,
    current_knee_flex: float = 0.0,
    track: bool = True,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    return _update_recovery_tracking(
        track_recovery=jnp.asarray(track, dtype=jnp.bool_),
        push_schedule=_schedule(),
        current_step=jnp.asarray(step, dtype=jnp.int32),
        episode_done=jnp.asarray(done, dtype=jnp.bool_),
        episode_failed=jnp.asarray(failed, dtype=jnp.bool_),
        liftoff_left=jnp.asarray(liftoff_left, dtype=jnp.bool_),
        liftoff_right=jnp.asarray(liftoff_right, dtype=jnp.bool_),
        touchdown_left=jnp.asarray(touchdown_left, dtype=jnp.bool_),
        touchdown_right=jnp.asarray(touchdown_right, dtype=jnp.bool_),
        horizontal_speed=jnp.asarray(speed, dtype=jnp.float32),
        pitch_rate=jnp.asarray(pitch_rate, dtype=jnp.float32),
        capture_error_norm=jnp.asarray(capture_error_norm, dtype=jnp.float32),
        first_touchdown_dx=jnp.asarray(first_step_dx, dtype=jnp.float32),
        first_touchdown_dy=jnp.asarray(first_step_dy, dtype=jnp.float32),
        first_touchdown_target_err_x=jnp.asarray(first_step_target_err_x, dtype=jnp.float32),
        first_touchdown_target_err_y=jnp.asarray(first_step_target_err_y, dtype=jnp.float32),
        current_height=jnp.asarray(current_height, dtype=jnp.float32),
        current_knee_flex=jnp.asarray(current_knee_flex, dtype=jnp.float32),
        recovery_active=state["recovery_active"],
        recovery_age=state["recovery_age"],
        recovery_last_support_foot=state["recovery_last_support_foot"],
        recovery_first_liftoff_recorded=state["recovery_first_liftoff_recorded"],
        recovery_first_liftoff_latency=state["recovery_first_liftoff_latency"],
        recovery_first_touchdown_recorded=state["recovery_first_touchdown_recorded"],
        recovery_first_step_latency=state["recovery_first_step_latency"],
        recovery_first_touchdown_age=state["recovery_first_touchdown_age"],
        recovery_visible_step_recorded=state["recovery_visible_step_recorded"],
        recovery_touchdown_count=state["recovery_touchdown_count"],
        recovery_support_foot_changes=state["recovery_support_foot_changes"],
        recovery_pitch_rate_at_push_end=state["recovery_pitch_rate_at_push_end"],
        recovery_pitch_rate_at_touchdown=state["recovery_pitch_rate_at_touchdown"],
        recovery_pitch_rate_after_10t=state["recovery_pitch_rate_after_10t"],
        recovery_capture_error_at_push_end=state["recovery_capture_error_at_push_end"],
        recovery_capture_error_at_touchdown=state["recovery_capture_error_at_touchdown"],
        recovery_capture_error_after_10t=state["recovery_capture_error_after_10t"],
        recovery_first_step_dx=state["recovery_first_step_dx"],
        recovery_first_step_dy=state["recovery_first_step_dy"],
        recovery_first_step_target_err_x=state["recovery_first_step_target_err_x"],
        recovery_first_step_target_err_y=state["recovery_first_step_target_err_y"],
        recovery_min_height=state["recovery_min_height"],
        recovery_max_knee_flex=state["recovery_max_knee_flex"],
    )


def test_recovery_metric_reducers_use_finalized_summary_contract():
    expected = {
        "recovery/first_step_latency": Reducer.SUM,
        "recovery/first_liftoff_latency": Reducer.SUM,
        "recovery/first_touchdown_latency": Reducer.SUM,
        "recovery/touchdown_count": Reducer.SUM,
        "recovery/support_foot_changes": Reducer.SUM,
        "recovery/post_push_velocity": Reducer.SUM,
        "recovery/completed": Reducer.SUM,
        "recovery/no_touchdown_frac": Reducer.SUM,
        "recovery/touchdown_then_fail_frac": Reducer.SUM,
        "recovery/touchdown_to_term_steps": Reducer.SUM,
        "recovery/visible_step_rate": Reducer.SUM,
        "recovery/min_height": Reducer.SUM,
        "recovery/max_knee_flex": Reducer.SUM,
        "recovery/first_step_dist_abs": Reducer.SUM,
        "visible_step_rate_hard": Reducer.SUM,
        "teacher/step_required_event_rate": Reducer.SUM,
        "unnecessary_step_rate": Reducer.MEAN,
    }
    for spec in METRIC_SPECS:
        reducer = expected.get(spec.name)
        if reducer is not None:
            assert spec.reducer == reducer


def test_no_touchdown_recovery_finalizes_zero_summary():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        state, metrics = _step_recovery(
            state,
            step=step,
            speed=0.25,
            current_height=0.235 - 0.001 * (step - 60),
            current_knee_flex=0.15 + 0.005 * (step - 60),
        )

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 0.0
    assert float(metrics["recovery/no_touchdown_frac"]) == 1.0
    assert float(metrics["recovery/touchdown_then_fail_frac"]) == 0.0
    assert float(metrics["recovery/touchdown_count"]) == 0.0
    assert float(metrics["recovery/visible_step_rate"]) == 0.0
    assert float(metrics["recovery/support_foot_changes"]) == 0.0
    assert float(metrics["recovery/post_push_velocity"]) == pytest.approx(0.25)
    assert float(metrics["recovery/min_height"]) == pytest.approx(0.186)
    assert float(metrics["recovery/max_knee_flex"]) == pytest.approx(0.395)
    assert float(metrics["recovery/first_step_dist_abs"]) == 0.0


def test_single_touchdown_records_first_latency_once():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        touchdown_left = step == 64
        state, metrics = _step_recovery(
            state,
            step=step,
            liftoff_left=(step == 62),
            touchdown_left=touchdown_left,
            pitch_rate=1.2 if step == 60 else (1.0 if step == 64 else 0.4 if step == 74 else 0.0),
            capture_error_norm=0.8 if step == 60 else (0.6 if step == 64 else 0.2 if step == 74 else 0.0),
            first_step_dx=0.11 if step == 64 else 0.0,
            first_step_dy=0.05 if step == 64 else 0.0,
            first_step_target_err_x=0.02 if step == 64 else 0.0,
            first_step_target_err_y=-0.01 if step == 64 else 0.0,
            current_height=0.24 if step == 60 else 0.205 if step == 68 else 0.22,
            current_knee_flex=0.25 if step == 60 else 0.47 if step == 66 else 0.31,
        )

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 4.0
    assert float(metrics["recovery/first_liftoff_latency"]) == 2.0
    assert float(metrics["recovery/first_touchdown_latency"]) == 4.0
    assert float(metrics["recovery/touchdown_count"]) == 1.0
    assert float(metrics["recovery/visible_step_rate"]) == 1.0
    assert float(metrics["recovery/support_foot_changes"]) == 0.0
    assert float(metrics["recovery/first_step_dx"]) == pytest.approx(0.11)
    assert float(metrics["recovery/first_step_dy"]) == pytest.approx(0.05)
    assert float(metrics["recovery/first_step_target_err_x"]) == pytest.approx(0.02)
    assert float(metrics["recovery/first_step_target_err_y"]) == pytest.approx(-0.01)
    assert float(metrics["recovery/pitch_rate_at_push_end"]) == pytest.approx(1.2)
    assert float(metrics["recovery/pitch_rate_at_touchdown"]) == pytest.approx(1.0)
    assert float(metrics["recovery/pitch_rate_reduction_10t"]) == pytest.approx(0.6)
    assert float(metrics["recovery/capture_error_at_push_end"]) == pytest.approx(0.8)
    assert float(metrics["recovery/capture_error_at_touchdown"]) == pytest.approx(0.6)
    assert float(metrics["recovery/capture_error_reduction_10t"]) == pytest.approx(0.4)
    assert float(metrics["recovery/min_height"]) == pytest.approx(0.205)
    assert float(metrics["recovery/max_knee_flex"]) == pytest.approx(0.47)
    assert float(metrics["recovery/first_step_dist_abs"]) == pytest.approx((0.11**2 + 0.05**2) ** 0.5)


def test_visible_step_requires_prior_liftoff_not_same_tick_chatter():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        state, metrics = _step_recovery(
            state,
            step=step,
            liftoff_left=(step == 64),
            touchdown_left=(step == 64),
        )

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/visible_step_rate"]) == 0.0


def test_support_changes_count_only_true_alternations():
    state = _initial_recovery_state()
    sequence = {
        63: "left",
        66: "right",
        69: "right",
        72: "left",
    }
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        foot = sequence.get(step)
        state, metrics = _step_recovery(
            state,
            step=step,
            touchdown_left=foot == "left",
            touchdown_right=foot == "right",
        )

    assert metrics is not None
    assert float(metrics["recovery/touchdown_count"]) == 4.0
    assert float(metrics["recovery/visible_step_rate"]) == 0.0
    assert float(metrics["recovery/support_foot_changes"]) == 2.0
    assert float(metrics["recovery/first_step_latency"]) == 3.0


def test_early_termination_finalizes_partial_recovery():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 67):
        state, metrics = _step_recovery(
            state,
            step=step,
            touchdown_left=(step == 62),
            touchdown_right=(step == 65),
            done=(step == 66),
            failed=(step == 66),
            speed=0.4,
        )

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 2.0
    assert float(metrics["recovery/touchdown_count"]) == 2.0
    assert float(metrics["recovery/support_foot_changes"]) == 1.0
    assert float(metrics["recovery/post_push_velocity"]) == pytest.approx(0.4)
    assert float(metrics["recovery/touchdown_then_fail_frac"]) == 1.0
    assert float(metrics["recovery/touchdown_to_term_steps"]) == pytest.approx(4.0)
    assert float(metrics["recovery/visible_step_rate"]) == 0.0


def test_aggregate_metrics_normalizes_by_completed_recoveries():
    num_metrics = len(METRIC_SPECS)
    metrics_vec = jnp.zeros((4, 2, num_metrics), dtype=jnp.float32)

    def set_metric(name: str, t: int, n: int, value: float) -> None:
        idx = METRIC_INDEX[name]
        nonlocal metrics_vec
        metrics_vec = metrics_vec.at[t, n, idx].set(value)

    set_metric("recovery/first_step_latency", 0, 0, 4.0)
    set_metric("recovery/first_liftoff_latency", 0, 0, 2.0)
    set_metric("recovery/no_touchdown_frac", 0, 0, 0.0)
    set_metric("recovery/touchdown_then_fail_frac", 0, 0, 1.0)
    set_metric("recovery/touchdown_to_term_steps", 0, 0, 7.0)
    set_metric("recovery/visible_step_rate", 0, 0, 1.0)
    set_metric("recovery/min_height", 0, 0, 0.19)
    set_metric("recovery/max_knee_flex", 0, 0, 0.42)
    set_metric("recovery/first_step_dist_abs", 0, 0, 0.12)
    set_metric("visible_step_rate_hard", 0, 0, 1.0)
    set_metric("teacher/step_required_event_rate", 0, 0, 1.0)
    set_metric("recovery/touchdown_count", 0, 0, 2.0)
    set_metric("recovery/support_foot_changes", 0, 0, 1.0)
    set_metric("recovery/post_push_velocity", 0, 0, 0.3)
    set_metric("recovery/completed", 0, 0, 1.0)

    set_metric("recovery/first_step_latency", 2, 1, 6.0)
    set_metric("recovery/first_liftoff_latency", 2, 1, 1.0)
    set_metric("recovery/no_touchdown_frac", 2, 1, 1.0)
    set_metric("recovery/touchdown_then_fail_frac", 2, 1, 0.0)
    set_metric("recovery/touchdown_to_term_steps", 2, 1, 0.0)
    set_metric("recovery/visible_step_rate", 2, 1, 0.0)
    set_metric("recovery/min_height", 2, 1, 0.21)
    set_metric("recovery/max_knee_flex", 2, 1, 0.36)
    set_metric("recovery/first_step_dist_abs", 2, 1, 0.08)
    set_metric("visible_step_rate_hard", 2, 1, 0.0)
    set_metric("teacher/step_required_event_rate", 2, 1, 0.0)
    set_metric("recovery/touchdown_count", 2, 1, 4.0)
    set_metric("recovery/support_foot_changes", 2, 1, 3.0)
    set_metric("recovery/post_push_velocity", 2, 1, 0.5)
    set_metric("recovery/completed", 2, 1, 1.0)

    aggregated = aggregate_metrics(metrics_vec)

    assert float(aggregated["recovery/completed"]) == 2.0
    assert float(aggregated["recovery/first_step_latency"]) == 5.0
    assert float(aggregated["recovery/first_liftoff_latency"]) == 1.5
    assert float(aggregated["recovery/no_touchdown_frac"]) == 0.5
    assert float(aggregated["recovery/touchdown_then_fail_frac"]) == 0.5
    assert float(aggregated["recovery/touchdown_to_term_steps"]) == 3.5
    assert float(aggregated["recovery/visible_step_rate"]) == 0.5
    assert float(aggregated["recovery/min_height"]) == pytest.approx(0.2)
    assert float(aggregated["recovery/max_knee_flex"]) == pytest.approx(0.39)
    assert float(aggregated["recovery/first_step_dist_abs"]) == pytest.approx(0.1)
    assert float(aggregated["visible_step_rate_hard"]) == 0.5
    assert float(aggregated["teacher/step_required_event_rate"]) == 0.5
    assert float(aggregated["recovery/touchdown_count"]) == 3.0
    assert float(aggregated["recovery/support_foot_changes"]) == 2.0
    assert float(aggregated["recovery/post_push_velocity"]) == pytest.approx(0.4)


def test_teacher_aggregation_normalizes_by_active_count_and_clean_count():
    num_metrics = len(METRIC_SPECS)
    metrics_vec = jnp.zeros((4, 1, num_metrics), dtype=jnp.float32)

    def set_metric(name: str, t: int, value: float) -> None:
        idx = METRIC_INDEX[name]
        nonlocal metrics_vec
        metrics_vec = metrics_vec.at[t, 0, idx].set(value)

    # Active on t=0,1 only.
    set_metric("teacher/active_frac", 0, 1.0)
    set_metric("teacher/active_frac", 1, 1.0)
    set_metric("teacher/active_count", 0, 1.0)
    set_metric("teacher/active_count", 1, 1.0)
    set_metric("teacher/step_required_mean", 0, 0.5)
    set_metric("teacher/step_required_mean", 1, 1.0)
    set_metric("teacher/target_step_x_mean", 0, 0.02)
    set_metric("teacher/target_step_x_mean", 1, 0.06)
    set_metric("teacher/target_step_x_sq_mean", 0, 0.02 * 0.02)
    set_metric("teacher/target_step_x_sq_mean", 1, 0.06 * 0.06)
    set_metric("teacher/reachable_frac", 0, 1.0)
    set_metric("teacher/reachable_frac", 1, 0.0)

    # Clean on t=2,3 only.
    set_metric("teacher/clean_step_count", 2, 1.0)
    set_metric("teacher/clean_step_count", 3, 1.0)
    set_metric("teacher/raw_step_required_during_clean_mean", 2, 0.3)
    set_metric("teacher/raw_step_required_during_clean_mean", 3, 0.5)
    set_metric("teacher/teacher_active_during_clean_frac", 2, 0.0)
    set_metric("teacher/teacher_active_during_clean_frac", 3, 0.0)
    # Push on t=0,1 only; teacher active both times.
    set_metric("teacher/push_step_count", 0, 1.0)
    set_metric("teacher/push_step_count", 1, 1.0)
    set_metric("teacher/teacher_active_during_push_frac", 0, 1.0)
    set_metric("teacher/teacher_active_during_push_frac", 1, 1.0)

    aggregated = aggregate_metrics(metrics_vec)

    assert float(aggregated["teacher/active_frac"]) == pytest.approx(0.5)
    assert float(aggregated["teacher/active_count"]) == pytest.approx(2.0)
    assert float(aggregated["teacher/step_required_mean"]) == pytest.approx(0.75)
    assert float(aggregated["teacher/target_step_x_mean"]) == pytest.approx(0.04)
    assert float(aggregated["teacher/reachable_frac"]) == pytest.approx(0.5)
    assert float(aggregated["teacher/target_step_x_std"]) == pytest.approx(0.02)
    assert float(aggregated["teacher/raw_step_required_during_clean_mean"]) == pytest.approx(0.4)
    assert float(aggregated["teacher/teacher_active_during_clean_frac"]) == pytest.approx(0.0)
    assert float(aggregated["teacher/teacher_active_during_push_frac"]) == pytest.approx(1.0)


def test_whole_body_teacher_aggregation_normalizes_expected_fields():
    num_metrics = len(METRIC_SPECS)
    metrics_vec = jnp.zeros((4, 1, num_metrics), dtype=jnp.float32)

    def set_metric(name: str, t: int, value: float) -> None:
        idx = METRIC_INDEX[name]
        nonlocal metrics_vec
        metrics_vec = metrics_vec.at[t, 0, idx].set(value)

    set_metric("teacher/whole_body_active_frac", 0, 1.0)
    set_metric("teacher/whole_body_active_frac", 1, 1.0)
    set_metric("teacher/whole_body_active_count", 0, 1.0)
    set_metric("teacher/whole_body_active_count", 1, 1.0)
    set_metric("teacher/recovery_height_target_mean", 0, 0.40)
    set_metric("teacher/recovery_height_target_mean", 1, 0.41)
    set_metric("teacher/recovery_height_error", 0, 0.01)
    set_metric("teacher/recovery_height_error", 1, 0.03)
    set_metric("teacher/recovery_height_in_band_frac", 0, 1.0)
    set_metric("teacher/recovery_height_in_band_frac", 1, 0.0)
    set_metric("teacher/com_velocity_target_mean", 0, 0.12)
    set_metric("teacher/com_velocity_target_mean", 1, 0.12)
    set_metric("teacher/com_velocity_error", 0, 0.08)
    set_metric("teacher/com_velocity_error", 1, 0.02)
    set_metric("teacher/com_velocity_target_hit_frac", 0, 0.0)
    set_metric("teacher/com_velocity_target_hit_frac", 1, 1.0)
    set_metric("teacher/clean_step_count", 2, 1.0)
    set_metric("teacher/clean_step_count", 3, 1.0)
    set_metric("teacher/push_step_count", 0, 1.0)
    set_metric("teacher/push_step_count", 1, 1.0)
    set_metric("teacher/whole_body_active_during_clean_frac", 2, 0.0)
    set_metric("teacher/whole_body_active_during_clean_frac", 3, 0.0)
    set_metric("teacher/whole_body_active_during_push_frac", 0, 1.0)
    set_metric("teacher/whole_body_active_during_push_frac", 1, 1.0)

    aggregated = aggregate_metrics(metrics_vec)
    assert float(aggregated["teacher/whole_body_active_frac"]) == pytest.approx(0.5)
    assert float(aggregated["teacher/recovery_height_target_mean"]) == pytest.approx(0.405)
    assert float(aggregated["teacher/recovery_height_error"]) == pytest.approx(0.02)
    assert float(aggregated["teacher/recovery_height_in_band_frac"]) == pytest.approx(0.5)
    assert float(aggregated["teacher/com_velocity_target_mean"]) == pytest.approx(0.12)
    assert float(aggregated["teacher/com_velocity_error"]) == pytest.approx(0.05)
    assert float(aggregated["teacher/com_velocity_target_hit_frac"]) == pytest.approx(0.5)
    assert float(aggregated["teacher/whole_body_active_during_clean_frac"]) == pytest.approx(0.0)
    assert float(aggregated["teacher/whole_body_active_during_push_frac"]) == pytest.approx(1.0)


def test_disturbed_window_reward_weights_switch_only_in_disturbed_window():
    gate, orient_w, height_w, posture_w = compute_disturbed_window_reward_weights(
        base_orientation_weight=jnp.asarray(-1.0, dtype=jnp.float32),
        base_height_target_weight=jnp.asarray(1.5, dtype=jnp.float32),
        base_posture_weight=jnp.asarray(0.3, dtype=jnp.float32),
        push_active=jnp.asarray(0.0, dtype=jnp.float32),
        recovery_active=jnp.asarray(0.0, dtype=jnp.float32),
        disturbed_orientation_weight=jnp.asarray(-0.3, dtype=jnp.float32),
        disturbed_height_target_scale=jnp.asarray(0.0, dtype=jnp.float32),
        disturbed_posture_scale=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert float(gate) == 0.0
    assert float(orient_w) == pytest.approx(-1.0)
    assert float(height_w) == pytest.approx(1.5)
    assert float(posture_w) == pytest.approx(0.3)

    gate, orient_w, height_w, posture_w = compute_disturbed_window_reward_weights(
        base_orientation_weight=jnp.asarray(-1.0, dtype=jnp.float32),
        base_height_target_weight=jnp.asarray(1.5, dtype=jnp.float32),
        base_posture_weight=jnp.asarray(0.3, dtype=jnp.float32),
        push_active=jnp.asarray(1.0, dtype=jnp.float32),
        recovery_active=jnp.asarray(0.0, dtype=jnp.float32),
        disturbed_orientation_weight=jnp.asarray(-0.3, dtype=jnp.float32),
        disturbed_height_target_scale=jnp.asarray(0.0, dtype=jnp.float32),
        disturbed_posture_scale=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert float(gate) == 1.0
    assert float(orient_w) == pytest.approx(-0.3)
    assert float(height_w) == pytest.approx(0.0)
    assert float(posture_w) == pytest.approx(0.0)


def test_disturbed_height_floor_and_com_damping_terms():
    height_floor = compute_height_floor_penalty(
        height=jnp.asarray(0.18, dtype=jnp.float32),
        threshold=jnp.asarray(0.20, dtype=jnp.float32),
        sigma=jnp.asarray(0.02, dtype=jnp.float32),
    )
    assert float(height_floor) == pytest.approx(1.0)

    height_floor_zero = compute_height_floor_penalty(
        height=jnp.asarray(0.22, dtype=jnp.float32),
        threshold=jnp.asarray(0.20, dtype=jnp.float32),
        sigma=jnp.asarray(0.02, dtype=jnp.float32),
    )
    assert float(height_floor_zero) == pytest.approx(0.0)

    damping_small = compute_com_velocity_damping_reward(
        horizontal_speed=jnp.asarray(0.10, dtype=jnp.float32),
        scale=jnp.asarray(1.0, dtype=jnp.float32),
    )
    damping_large = compute_com_velocity_damping_reward(
        horizontal_speed=jnp.asarray(0.30, dtype=jnp.float32),
        scale=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert float(damping_small) > float(damping_large)
