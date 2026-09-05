from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from training.core.metrics_registry import METRIC_INDEX
from training.eval.eval_policy import _write_failure_trace


def test_failure_trace_records_only_first_episode_falls(tmp_path) -> None:
    steps, envs, obs_dim, action_dim = 5, 2, 7, 3
    metric_dim = max(METRIC_INDEX.values()) + 1
    dones = np.zeros((steps, envs), dtype=np.float32)
    truncations = np.zeros_like(dones)
    dones[3, 0] = 1.0
    dones[4, 1] = 1.0
    truncations[4, 1] = 1.0
    metrics_vec = np.zeros((steps, envs, metric_dim), dtype=np.float32)
    metrics_vec[:, 0, METRIC_INDEX["debug/pitch"]] = np.linspace(0.0, 0.5, steps)
    metrics_vec[2, 0, METRIC_INDEX["torque/left_hip_roll/sat_frac"]] = 1.0
    traj = SimpleNamespace(
        dones=dones,
        truncations=truncations,
        obs=np.arange(steps * envs * obs_dim, dtype=np.float32).reshape(
            steps, envs, obs_dim
        ),
        actions=np.zeros((steps, envs, action_dim), dtype=np.float32),
        metrics_vec=metrics_vec,
    )
    output = tmp_path / "failure_trace.npz"

    cases = _write_failure_trace(traj, output, ctrl_dt=0.02, window_s=0.04)

    assert [case["env_index"] for case in cases] == [0]
    assert cases[0]["failure_step"] == 3
    assert cases[0]["trace_steps"] == 2
    assert cases[0]["pre_fall_max_actuator_torque_sat_frac"] == 1.0
    saved = np.load(output)
    assert saved["failed_env_indices"].tolist() == [0]
    assert saved["observations"].shape == (1, 2, obs_dim)
    assert saved["actions"].shape == (1, 2, action_dim)
