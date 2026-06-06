"""Runtime action composition matches training's smoke9 home-base residual.

CRITICAL (per the runtime objective): smoke9 commands residuals around
``home_q_rad`` with per-joint scales, NOT the generic ``pos_target_rad_v1``
midpoint mapping.  These tests pin that contract + the 1-step action delay.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SMOKE9_RESIDUAL_SCALAR
from runtime.wr_runtime.control.policy_runner import RuntimePolicyRunner


class _ZeroPolicy:
    def __init__(self, n):
        self._n = n

    def predict(self, obs):
        return np.zeros(self._n, dtype=np.float32)


def _runner(v8_spec, runtime_policy_config):
    return RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_ZeroPolicy(v8_spec.model.action_dim),
        robot_io=None,
    )


def test_zero_action_maps_exactly_to_home(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    n = v8_spec.model.action_dim
    target, applied = runner.compose_and_apply(np.zeros(n, dtype=np.float32))
    np.testing.assert_allclose(target, runner.home_q_rad, atol=1e-6)
    np.testing.assert_allclose(applied, 0.0, atol=1e-6)


def test_one_step_delay_applies_previous_filtered(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    n = v8_spec.model.action_dim
    a = np.full(n, 0.5, dtype=np.float32)
    b = np.full(n, -0.3, dtype=np.float32)

    _, applied1 = runner.compose_and_apply(a)
    # With action_delay_steps=1 the first applied action is zero (pending was 0).
    np.testing.assert_allclose(applied1, 0.0, atol=1e-6)

    _, applied2 = runner.compose_and_apply(b)
    # Second step applies the PREVIOUS filtered action (= a, since alpha=0).
    np.testing.assert_allclose(applied2, a, atol=1e-6)


def test_nonzero_uses_per_joint_residual_scale_not_scalar(
    v8_spec, runtime_policy_config
):
    runner = _runner(v8_spec, runtime_policy_config)
    n = v8_spec.model.action_dim
    names = list(v8_spec.robot.actuator_names)
    home = runner.home_q_rad

    # Two composes so the delay lets a full +1.0 action actually apply.
    runner.compose_and_apply(np.ones(n, dtype=np.float32))
    target, applied = runner.compose_and_apply(np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(applied, 1.0, atol=1e-6)

    knee = names.index("left_knee_pitch")
    waist = names.index("waist_yaw")

    # left_knee_pitch uses its per-joint scale 0.978 (range [0, 2.094],
    # home 1.047 -> 2.025, no clip).
    assert (target[knee] - home[knee]) == pytest.approx(0.978, abs=1e-4)
    # waist_yaw is NOT in the per-joint dict -> scalar fallback 0.2,
    # NOT the legacy action_scale_rad (0.35) and NOT the midpoint mapping.
    assert (target[waist] - home[waist]) == pytest.approx(
        SMOKE9_RESIDUAL_SCALAR, abs=1e-4
    )
    assert abs((target[waist] - home[waist]) - 0.35) > 0.1


def test_wrong_sized_action_raises_not_broadcasts(v8_spec, runtime_policy_config):
    """Regression: a length-1 action must NOT broadcast into a 21-joint target.

    Reproduces the review finding (length-1 action -> full joint target via NumPy
    broadcasting on the second delayed step).  The runner guards the broadcast
    site and fails loudly instead.
    """
    runner = _runner(v8_spec, runtime_policy_config)
    n = v8_spec.model.action_dim
    # First step: pending is zeros so the broadcast wouldn't be observable yet,
    # but the guard rejects the wrong-sized raw immediately.
    with pytest.raises(ValueError, match="expected"):
        runner.compose_and_apply(np.array([0.5], dtype=np.float32))

    # And after a valid first step, a wrong-sized second action is still rejected
    # (this is the exact second-delayed-step path the review reproduced).
    runner.compose_and_apply(np.zeros(n, dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        runner.compose_and_apply(np.array([0.5], dtype=np.float32))
