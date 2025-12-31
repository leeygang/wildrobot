"""Disturbance scheduling utilities for WildRobot."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from playground_amp.configs.training_runtime_config import EnvConfig

try:
    import flax.struct as struct

    @struct.dataclass
    class DisturbanceSchedule:
        """Episode-constant push schedule."""

        start_step: jp.ndarray  # shape=()
        end_step: jp.ndarray  # shape=()
        force_xy: jp.ndarray  # shape=(2,)
        rng: jp.ndarray  # shape=(2,)

except ImportError:
    class DisturbanceSchedule(NamedTuple):
        """Episode-constant push schedule."""

        start_step: jp.ndarray
        end_step: jp.ndarray
        force_xy: jp.ndarray
        rng: jp.ndarray


def sample_push_schedule(rng: jax.Array, cfg: EnvConfig) -> DisturbanceSchedule:
    """Sample a single lateral push schedule for an episode."""
    if not cfg.push_enabled:
        _, next_rng = jax.random.split(rng)
        return DisturbanceSchedule(
            start_step=jp.zeros(()),
            end_step=jp.zeros(()),
            force_xy=jp.zeros((2,)),
            rng=next_rng,
        )

    rng, key_start, key_force, key_angle, next_rng = jax.random.split(rng, 5)
    start_step = jax.random.randint(
        key_start,
        shape=(),
        minval=cfg.push_start_step_min,
        maxval=cfg.push_start_step_max + 1,
    )
    force_mag = jax.random.uniform(
        key_force,
        shape=(),
        minval=cfg.push_force_min,
        maxval=cfg.push_force_max,
    )
    angle = jax.random.uniform(key_angle, shape=(), minval=0.0, maxval=2 * jp.pi)
    force_xy = force_mag * jp.array([jp.cos(angle), jp.sin(angle)])
    duration = jp.asarray(cfg.push_duration_steps)
    start_step = jp.asarray(start_step)
    end_step = start_step + duration
    return DisturbanceSchedule(
        start_step=start_step,
        end_step=end_step,
        force_xy=force_xy,
        rng=next_rng,
    )


def apply_push(
    data,
    schedule: DisturbanceSchedule,
    step_count: jp.ndarray,
    body_id: int,
):
    """Apply push force to the specified body for the active window."""
    if body_id < 0:
        return data

    push_active = (step_count >= schedule.start_step) & (step_count < schedule.end_step)
    push_force_xy = jp.where(push_active, schedule.force_xy, 0.0)
    xfrc_applied = jp.zeros_like(data.xfrc_applied)
    push_force = jp.array(
        [push_force_xy[0], push_force_xy[1], 0.0, 0.0, 0.0, 0.0]
    )
    xfrc_applied = xfrc_applied.at[body_id].set(push_force)
    return data.replace(xfrc_applied=xfrc_applied)
