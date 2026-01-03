import jax
import jax.numpy as jp
import pytest
from playground_amp.envs.disturbance import DisturbanceSchedule

# We'll import the method object without instantiating WildRobotEnv to avoid heavy init.
from playground_amp.envs.wildrobot_env import WildRobotEnv


class Dummy:
    def __init__(self, linvel_mode: str, linvel_dropout_prob: float = 0.0):
        class Env:
            pass

        self._config = type("C", (), {})()
        self._config.env = Env()
        self._config.env.linvel_mode = linvel_mode
        self._config.env.linvel_dropout_prob = linvel_dropout_prob


def make_schedule(seed: int = 0):
    rng = jax.random.PRNGKey(seed)
    return DisturbanceSchedule(
        start_step=jp.zeros(()), end_step=jp.zeros(()), force_xy=jp.zeros((2,)), rng=rng
    )


def test_linvel_mode_sim_returns_one():
    d = Dummy("sim")
    schedule = make_schedule(1)
    mask, _ = WildRobotEnv._sample_linvel_obs_mask(d, schedule)
    assert float(mask) == pytest.approx(1.0)


def test_linvel_mode_zero_returns_zero():
    d = Dummy("zero")
    schedule = make_schedule(2)
    mask, _ = WildRobotEnv._sample_linvel_obs_mask(d, schedule)
    assert float(mask) == pytest.approx(0.0)


def test_linvel_mode_dropout_with_prob_one_returns_zero():
    # dropout_prob=1.0 -> keep prob = 0.0 -> always zero
    d = Dummy("dropout", linvel_dropout_prob=1.0)
    schedule = make_schedule(3)
    mask, updated = WildRobotEnv._sample_linvel_obs_mask(d, schedule)
    assert float(mask) == pytest.approx(0.0)
    # Ensure rng was advanced on update
    assert hasattr(updated, "rng")
