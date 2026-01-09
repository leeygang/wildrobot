import types

import jax
import jax.numpy as jnp

from policy_contract.jax.signals import Signals
from training.envs.wildrobot_env import _apply_imu_noise_and_delay


def _make_cfg(*, imu_gyro_noise_std: float, imu_quat_noise_deg: float, imu_latency_steps: int):
    return types.SimpleNamespace(
        env=types.SimpleNamespace(
            imu_gyro_noise_std=imu_gyro_noise_std,
            imu_quat_noise_deg=imu_quat_noise_deg,
            imu_latency_steps=imu_latency_steps,
        )
    )


def _make_signals():
    return Signals(
        quat_xyzw=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        gyro_rad_s=jnp.zeros((3,), dtype=jnp.float32),
        joint_pos_rad=jnp.zeros((1,), dtype=jnp.float32),
        joint_vel_rad_s=jnp.zeros((1,), dtype=jnp.float32),
        foot_switches=jnp.zeros((4,), dtype=jnp.float32),
    )


def test_imu_noise_latency_no_rng_consumption_when_noise_disabled():
    cfg = _make_cfg(imu_gyro_noise_std=0.0, imu_quat_noise_deg=0.0, imu_latency_steps=0)
    rng = jax.random.PRNGKey(0)
    signals = _make_signals()

    _, _, _, rng_out = _apply_imu_noise_and_delay(signals, rng, cfg, None, None)

    assert jnp.array_equal(rng_out, rng)


def test_imu_latency_no_rng_consumption_when_noise_disabled():
    cfg = _make_cfg(imu_gyro_noise_std=0.0, imu_quat_noise_deg=0.0, imu_latency_steps=2)
    rng = jax.random.PRNGKey(1)
    signals = _make_signals()

    _, _, _, rng_out = _apply_imu_noise_and_delay(signals, rng, cfg, None, None)

    assert jnp.array_equal(rng_out, rng)
