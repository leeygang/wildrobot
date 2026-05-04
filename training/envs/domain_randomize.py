"""v0.17.3b domain randomization — disabled by default.

Sampling functions for per-episode physics randomization.
Randomized parameters are stored in WildRobotInfo and applied at reset/step.

This module is gated by `domain_randomization_enabled` in the env config.
When disabled (default for v0.17.3a), all functions return nominal values.
"""

from __future__ import annotations

import jax
import jax.numpy as jp


def sample_domain_rand_params(
    rng: jax.Array,
    *,
    num_bodies: int,
    num_actuators: int,
    friction_range: tuple[float, float] = (0.5, 1.0),
    mass_scale_range: tuple[float, float] = (0.9, 1.1),
    kp_scale_range: tuple[float, float] = (0.9, 1.1),
    frictionloss_scale_range: tuple[float, float] = (0.9, 1.1),
    joint_offset_rad: float = 0.03,
) -> dict[str, jax.Array]:
    """Sample randomized physics parameters for one episode.

    Returns a dict of JAX arrays to be stored in env info and applied
    at reset/step time.
    """
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)

    friction_scale = jax.random.uniform(
        k1, shape=(), minval=friction_range[0], maxval=friction_range[1]
    )
    mass_scales = jax.random.uniform(
        k2, shape=(num_bodies,), minval=mass_scale_range[0], maxval=mass_scale_range[1]
    )
    kp_scales = jax.random.uniform(
        k3, shape=(num_actuators,), minval=kp_scale_range[0], maxval=kp_scale_range[1]
    )
    frictionloss_scales = jax.random.uniform(
        k4,
        shape=(num_actuators,),
        minval=frictionloss_scale_range[0],
        maxval=frictionloss_scale_range[1],
    )
    joint_offsets = jax.random.uniform(
        k5, shape=(num_actuators,), minval=-joint_offset_rad, maxval=joint_offset_rad
    )

    return {
        "friction_scale": friction_scale,
        "mass_scales": mass_scales,
        "kp_scales": kp_scales,
        "frictionloss_scales": frictionloss_scales,
        "joint_offsets": joint_offsets,
    }


def nominal_domain_rand_params(
    *,
    num_bodies: int,
    num_actuators: int,
) -> dict[str, jax.Array]:
    """Return nominal (no-randomization) parameter values."""
    return {
        "friction_scale": jp.ones(()),
        "mass_scales": jp.ones((num_bodies,)),
        "kp_scales": jp.ones((num_actuators,)),
        "frictionloss_scales": jp.ones((num_actuators,)),
        "joint_offsets": jp.zeros((num_actuators,)),
    }
