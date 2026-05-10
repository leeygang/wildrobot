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
    backlash_range: tuple[float, float] = (0.0, 0.0),
) -> dict[str, jax.Array]:
    """Sample randomized physics parameters for one episode.

    Returns a dict of JAX arrays to be stored in env info and applied
    at reset/step time.

    ``backlash_range`` (Phase 3 of walking_training.md Appendix B.2):
    per-joint episode-constant deadband magnitude in radians, mirror
    of TB ``DomainRandConfig.backlash_range`` (mjx_config.py:210).  At
    obs time the policy sees ``motor_pos += 0.5 * backlash *
    tanh(qfrc_actuator / backlash_activation)`` — observed joint
    position can be displaced by ±backlash/2 depending on the sign of
    the actuator torque (the gear sits on one side of the deadband or
    the other).  Default range (0.0, 0.0) keeps backlash disabled.
    """
    rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)

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
    backlash = jax.random.uniform(
        k6,
        shape=(num_actuators,),
        minval=backlash_range[0],
        maxval=backlash_range[1],
    )

    return {
        "friction_scale": friction_scale,
        "mass_scales": mass_scales,
        "kp_scales": kp_scales,
        "frictionloss_scales": frictionloss_scales,
        "joint_offsets": joint_offsets,
        "backlash": backlash,
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
        "backlash": jp.zeros((num_actuators,)),
    }


def apply_backlash_to_joint_pos(
    joint_pos: jax.Array,
    qfrc_actuator: jax.Array,
    backlash: jax.Array,
    backlash_activation: float,
) -> jax.Array:
    """Apply TB-style smooth backlash to the observed joint position.

    Mirrors toddlerbot/locomotion/mjx_env.py:2073-2080:
        motor_pos_noisy += 0.5 * backlash * tanh(qfrc_actuator / activation)

    The tanh gives a smooth signed transition so PPO sees a continuous
    displacement that flips sign when the actuator torque flips sign
    (gear sitting on one side of the deadband or the other).
    Activation (in Nm) sets the torque magnitude at which the backlash
    is fully engaged.  When ``backlash`` is all zeros this is a no-op.
    """
    safe_act = jp.maximum(jp.float32(backlash_activation), jp.float32(1e-6))
    return joint_pos + 0.5 * backlash * jp.tanh(qfrc_actuator / safe_act)
