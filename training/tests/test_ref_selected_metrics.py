"""Pin the tracking/ref_selected_vy + ref_selected_wz instrumentation
(commit 3926660): the per-step heading-corrected reference lookup, and the
metric wiring that surfaces it.

Two contracts:
  1. Metrics wiring: the keys are registered, and with a forward command +
     ~zero yaw drift (reset + a few zero-action steps) the selected vy / wz
     bins are 0 (the lookup stays on the vy=0, wz=0 row).
  2. Lookup contract: a forward command at HIGH vx with a yawed heading makes
     the heading-frame rotation inject a lateral component, so the nearest-bin
     lookup selects a vy!=0 bin — while selected_yaw_rate stays 0 (wz is
     unaffected by the heading rotation).  An aligned heading keeps vy=0.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from training.configs.training_config import load_training_config
from training.core.metrics_registry import (
    METRIC_NAMES,
    METRICS_VEC_KEY,
    unpack_metrics,
)
from training.envs.wildrobot_env import WildRobotEnv

_CFG = Path("training/configs/ppo_walking_v0210_smoke4_forward_first.yaml")


@pytest.fixture(scope="module")
def env():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return WildRobotEnv(config=load_training_config(str(_CFG)))


def test_ref_selected_vy_wz_registered() -> None:
    assert "tracking/ref_selected_vy" in METRIC_NAMES
    assert "tracking/ref_selected_wz" in METRIC_NAMES


def test_ref_selected_vy_wz_zero_at_zero_yaw(env) -> None:
    """Forward cmd, ~zero yaw drift (reset + a few zero-action steps) → the
    selected vy/wz bins stay 0 (lookup on the vy=0, wz=0 row)."""
    st = env.reset(jax.random.PRNGKey(0))
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    assert float(m["tracking/ref_selected_vy"]) == pytest.approx(0.0, abs=1e-6)
    assert float(m["tracking/ref_selected_wz"]) == pytest.approx(0.0, abs=1e-6)

    act = jp.zeros(env.action_size)
    for _ in range(3):
        st = env.step(st, act)
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    assert abs(float(m["tracking/ref_selected_vy"])) < 1e-6
    assert abs(float(m["tracking/ref_selected_wz"])) < 1e-6


def _yaw_quat_wxyz(yaw: float) -> jp.ndarray:
    """Quaternion (wxyz) for a rotation of ``yaw`` rad about +z."""
    return jp.array(
        [jp.cos(yaw / 2.0), 0.0, 0.0, jp.sin(yaw / 2.0)], dtype=jp.float32
    )


def test_lookup_aligned_heading_stays_on_vy0(env) -> None:
    """Forward cmd, no heading error → lookup selects vy=0, wz=0."""
    cmd = jp.array([0.26, 0.0, 0.0], dtype=jp.float32)
    identity = _yaw_quat_wxyz(0.0)
    win = env._lookup_offline_window(
        jp.asarray(0, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=identity,
        heading_rot_wxyz=identity,
    )
    assert float(win["selected_vy"]) == pytest.approx(0.0, abs=1e-6)
    assert float(win["selected_yaw_rate"]) == pytest.approx(0.0, abs=1e-6)


def test_lookup_yawed_heading_high_vx_flips_to_lateral_bin(env) -> None:
    """Forward cmd (0.26,0,0) + heading error δ≈20° injects a lateral
    component 0.26·sin(δ)≈0.089 > half the 0.065-bin spacing, so the
    heading-corrected lookup selects a vy!=0 bin.  selected_yaw_rate stays 0
    (the rotation does not touch wz)."""
    cmd = jp.array([0.26, 0.0, 0.0], dtype=jp.float32)
    path_rot = _yaw_quat_wxyz(0.0)          # path heading 0
    # heading rotated by -δ so delta_yaw = yaw_path - yaw_heading = +δ
    delta = 0.35  # ~20°
    heading_rot = _yaw_quat_wxyz(-delta)
    win = env._lookup_offline_window(
        jp.asarray(0, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=path_rot,
        heading_rot_wxyz=heading_rot,
    )
    assert abs(float(win["selected_vy"])) > 1e-6   # flipped onto a lateral bin
    assert float(win["selected_yaw_rate"]) == pytest.approx(0.0, abs=1e-6)
