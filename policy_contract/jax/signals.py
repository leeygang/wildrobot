from __future__ import annotations

from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Signals:
    quat_xyzw: jnp.ndarray
    gyro_rad_s: jnp.ndarray
    joint_pos_rad: jnp.ndarray
    joint_vel_rad_s: jnp.ndarray
    foot_switches: jnp.ndarray
    timestamp_s: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(0.0, dtype=jnp.float32)
    )
