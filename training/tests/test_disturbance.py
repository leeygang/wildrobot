"""Tests for disturbance scheduling and application."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from training.configs.training_runtime_config import EnvConfig
from training.envs.disturbance import (
    DisturbanceSchedule,
    apply_push,
    sample_push_schedule,
)


def test_sample_push_schedule_ranges():
    cfg = EnvConfig(
        push_enabled=True,
        push_start_step_min=5,
        push_start_step_max=15,
        push_duration_steps=7,
        push_force_min=3.0,
        push_force_max=9.0,
        push_body="waist",
    )
    schedule = sample_push_schedule(jax.random.PRNGKey(0), cfg)
    assert 5 <= int(schedule.start_step) <= 15
    assert int(schedule.end_step) == int(schedule.start_step) + 7
    force_mag = float(jp.linalg.norm(schedule.force_xy))
    assert 3.0 <= force_mag <= 9.0
    assert schedule.rng.shape == (2,)


def test_sample_push_schedule_disabled():
    cfg = EnvConfig(push_enabled=False)
    schedule = sample_push_schedule(jax.random.PRNGKey(1), cfg)
    assert float(schedule.start_step) == 0.0
    assert float(schedule.end_step) == 0.0
    assert jp.allclose(schedule.force_xy, 0.0)
    assert schedule.rng.shape == (2,)


def test_apply_push_applies_force(mj_model):
    mjx_model = mjx.put_model(mj_model)
    data = mjx.make_data(mjx_model)
    waist_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "waist")
    assert waist_id >= 0

    schedule = DisturbanceSchedule(
        start_step=jp.asarray(5),
        end_step=jp.asarray(8),
        force_xy=jp.asarray([3.0, -4.0]),
        rng=jp.zeros((2,)),
    )

    active_data = apply_push(data, schedule, jp.asarray(5), waist_id)
    active_force = active_data.xfrc_applied[waist_id][:2]
    assert jp.allclose(active_force, schedule.force_xy)

    inactive_data = apply_push(data, schedule, jp.asarray(8), waist_id)
    inactive_force = inactive_data.xfrc_applied[waist_id][:2]
    assert jp.allclose(inactive_force, 0.0)
