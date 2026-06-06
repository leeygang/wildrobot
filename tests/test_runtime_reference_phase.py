"""Runtime gait phase service: nearest-bin selection + absorbing-boundary clamp."""

from __future__ import annotations

import numpy as np

from conftest import make_reference_dict
from runtime.wr_runtime.control.reference_phase import ReferencePhaseService
from runtime.wr_runtime.control.runtime_policy_config import ReferencePhaseTable


def _service(n_steps=96, n_cycle=48):
    return ReferencePhaseService(ReferencePhaseTable.from_dict(
        make_reference_dict(n_steps=n_steps, n_cycle=n_cycle)
    ))


def test_select_bin_nearest_l2():
    svc = _service()
    # cmd_keys = [[0.13,0,0],[0.13,0.065,0],[0.13,-0.065,0],[0,0,0]]
    assert svc.select_bin(np.array([0.13, 0.0, 0.0])) == 0
    assert svc.select_bin(np.array([0.13, 0.06, 0.0])) == 1
    assert svc.select_bin(np.array([0.13, -0.06, 0.0])) == 2
    assert svc.select_bin(np.array([0.0, 0.0, 0.0])) == 3
    # scalar broadcasts to (vx, 0, 0)
    assert svc.select_bin(np.array([0.13])) == 0


def test_phase_is_periodic_clock():
    svc = _service(n_steps=96, n_cycle=48)
    p0 = svc.phase_sin_cos(bin_idx=0, step_idx=0)
    # phase 0 -> sin 0, cos 1
    np.testing.assert_allclose(p0, [0.0, 1.0], atol=1e-6)
    # one full cycle later (step 48) -> back to phase 0
    p48 = svc.phase_sin_cos(bin_idx=0, step_idx=48)
    np.testing.assert_allclose(p48, [0.0, 1.0], atol=1e-6)
    # quarter cycle (step 12) -> phase 0.25 -> sin 1, cos 0
    p12 = svc.phase_sin_cos(bin_idx=0, step_idx=12)
    np.testing.assert_allclose(p12, [1.0, 0.0], atol=1e-6)


def test_static_bin_clamps_to_single_frame():
    svc = _service()
    # bin 3 is the static (0,0,0) bin with per_bin_n_steps=1 -> always frame 0.
    p = svc.phase_sin_cos(bin_idx=3, step_idx=50)
    np.testing.assert_allclose(p, [0.0, 1.0], atol=1e-6)


def test_step_idx_clamped_to_array_bound():
    svc = _service(n_steps=96, n_cycle=48)
    # Way past the end -> clamped to last frame, no IndexError.
    p = svc.phase_sin_cos(bin_idx=0, step_idx=10_000)
    assert p.shape == (2,)
    assert np.all(np.isfinite(p))
