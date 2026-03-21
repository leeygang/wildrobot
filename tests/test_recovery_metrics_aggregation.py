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
from training.envs.wildrobot_env import RECOVERY_WINDOW_STEPS, _update_recovery_tracking


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
        "recovery_first_touchdown_recorded": jnp.asarray(False, dtype=jnp.bool_),
        "recovery_first_step_latency": jnp.asarray(0, dtype=jnp.int32),
        "recovery_touchdown_count": jnp.asarray(0, dtype=jnp.int32),
        "recovery_support_foot_changes": jnp.asarray(0, dtype=jnp.int32),
    }


def _step_recovery(
    state: dict[str, jnp.ndarray],
    *,
    step: int,
    touchdown_left: bool = False,
    touchdown_right: bool = False,
    done: bool = False,
    speed: float = 0.0,
    track: bool = True,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    return _update_recovery_tracking(
        track_recovery=jnp.asarray(track, dtype=jnp.bool_),
        push_schedule=_schedule(),
        current_step=jnp.asarray(step, dtype=jnp.int32),
        episode_done=jnp.asarray(done, dtype=jnp.bool_),
        touchdown_left=jnp.asarray(touchdown_left, dtype=jnp.bool_),
        touchdown_right=jnp.asarray(touchdown_right, dtype=jnp.bool_),
        horizontal_speed=jnp.asarray(speed, dtype=jnp.float32),
        recovery_active=state["recovery_active"],
        recovery_age=state["recovery_age"],
        recovery_last_support_foot=state["recovery_last_support_foot"],
        recovery_first_touchdown_recorded=state["recovery_first_touchdown_recorded"],
        recovery_first_step_latency=state["recovery_first_step_latency"],
        recovery_touchdown_count=state["recovery_touchdown_count"],
        recovery_support_foot_changes=state["recovery_support_foot_changes"],
    )


def test_recovery_metric_reducers_use_finalized_summary_contract():
    expected = {
        "recovery/first_step_latency": Reducer.SUM,
        "recovery/touchdown_count": Reducer.SUM,
        "recovery/support_foot_changes": Reducer.SUM,
        "recovery/post_push_velocity": Reducer.SUM,
        "recovery/completed": Reducer.SUM,
    }
    for spec in METRIC_SPECS:
        reducer = expected.get(spec.name)
        if reducer is not None:
            assert spec.reducer == reducer


def test_no_touchdown_recovery_finalizes_zero_summary():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        state, metrics = _step_recovery(state, step=step, speed=0.25)

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 0.0
    assert float(metrics["recovery/touchdown_count"]) == 0.0
    assert float(metrics["recovery/support_foot_changes"]) == 0.0
    assert float(metrics["recovery/post_push_velocity"]) == pytest.approx(0.25)


def test_single_touchdown_records_first_latency_once():
    state = _initial_recovery_state()
    metrics = None
    for step in range(60, 60 + RECOVERY_WINDOW_STEPS):
        touchdown_left = step == 64
        state, metrics = _step_recovery(state, step=step, touchdown_left=touchdown_left)

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 4.0
    assert float(metrics["recovery/touchdown_count"]) == 1.0
    assert float(metrics["recovery/support_foot_changes"]) == 0.0


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
            speed=0.4,
        )

    assert metrics is not None
    assert float(metrics["recovery/completed"]) == 1.0
    assert float(metrics["recovery/first_step_latency"]) == 2.0
    assert float(metrics["recovery/touchdown_count"]) == 2.0
    assert float(metrics["recovery/support_foot_changes"]) == 1.0
    assert float(metrics["recovery/post_push_velocity"]) == pytest.approx(0.4)


def test_aggregate_metrics_normalizes_by_completed_recoveries():
    num_metrics = len(METRIC_SPECS)
    metrics_vec = jnp.zeros((4, 2, num_metrics), dtype=jnp.float32)

    def set_metric(name: str, t: int, n: int, value: float) -> None:
        idx = METRIC_INDEX[name]
        nonlocal metrics_vec
        metrics_vec = metrics_vec.at[t, n, idx].set(value)

    set_metric("recovery/first_step_latency", 0, 0, 4.0)
    set_metric("recovery/touchdown_count", 0, 0, 2.0)
    set_metric("recovery/support_foot_changes", 0, 0, 1.0)
    set_metric("recovery/post_push_velocity", 0, 0, 0.3)
    set_metric("recovery/completed", 0, 0, 1.0)

    set_metric("recovery/first_step_latency", 2, 1, 6.0)
    set_metric("recovery/touchdown_count", 2, 1, 4.0)
    set_metric("recovery/support_foot_changes", 2, 1, 3.0)
    set_metric("recovery/post_push_velocity", 2, 1, 0.5)
    set_metric("recovery/completed", 2, 1, 1.0)

    aggregated = aggregate_metrics(metrics_vec)

    assert float(aggregated["recovery/completed"]) == 2.0
    assert float(aggregated["recovery/first_step_latency"]) == 5.0
    assert float(aggregated["recovery/touchdown_count"]) == 3.0
    assert float(aggregated["recovery/support_foot_changes"]) == 2.0
    assert float(aggregated["recovery/post_push_velocity"]) == pytest.approx(0.4)
