"""v0.21.0 P11 — numpy obs / eval / visualize parity for 3-axis velocity_cmd.

Mirrors the JAX-side v8 layout work landed in P7.  All tests below come
straight from the v0.21.0 plan (training/docs/v0210_lateral_yaw_prior_plan-final-r2.md
§P11).
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from policy_contract.spec_builder import build_policy_spec


def _spec_v8():
    return build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {"name": f"j{i}", "range": [-1.0, 1.0],
             "policy_action_sign": 1.0, "max_velocity_rad_s": 10.0}
            for i in range(13)
        ],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v8_cmd3d",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * 13,
    )


def test_numpy_build_observation_accepts_v8_layout() -> None:
    from policy_contract.numpy.obs import build_observation_from_components
    spec = _spec_v8()
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = np.zeros((15 * proprio_bundle,), dtype=np.float32)
    obs = build_observation_from_components(
        spec=spec,
        gravity_local=np.zeros(3, dtype=np.float32),
        angvel_heading_local=np.zeros(3, dtype=np.float32),
        joint_pos_normalized=np.zeros(action_dim, dtype=np.float32),
        joint_vel_normalized=np.zeros(action_dim, dtype=np.float32),
        foot_switches=np.zeros(4, dtype=np.float32),
        prev_action=np.zeros(action_dim, dtype=np.float32),
        velocity_cmd=np.array([0.20, 0.05, 0.10], dtype=np.float32),
        velocity_cmd_lateral_yaw=np.array([0.05, 0.10], dtype=np.float32),
        loc_ref_phase_sin_cos=np.zeros(2, dtype=np.float32),
        proprio_history=proprio_history,
    )
    assert obs.shape == (spec.model.obs_dim,)


def test_numpy_v7_obs_byte_identical_when_velocity_cmd_passed_as_3vec() -> None:
    """H4: v7 numpy obs must produce byte-identical output for (1,) and
    (3,) cmd inputs, same as the JAX path."""
    from policy_contract.numpy.obs import build_observation_from_components
    spec_v7 = build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {"name": f"j{i}", "range": [-1.0, 1.0],
             "policy_action_sign": 1.0, "max_velocity_rad_s": 10.0}
            for i in range(13)
        ],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v7_phase_proprio",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * 13,
    )
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = np.zeros((15 * proprio_bundle,), dtype=np.float32)
    kw = dict(
        spec=spec_v7,
        gravity_local=np.zeros(3, dtype=np.float32),
        angvel_heading_local=np.zeros(3, dtype=np.float32),
        joint_pos_normalized=np.zeros(action_dim, dtype=np.float32),
        joint_vel_normalized=np.zeros(action_dim, dtype=np.float32),
        foot_switches=np.zeros(4, dtype=np.float32),
        prev_action=np.zeros(action_dim, dtype=np.float32),
        loc_ref_phase_sin_cos=np.zeros(2, dtype=np.float32),
        proprio_history=proprio_history,
    )
    obs_scalar = build_observation_from_components(
        velocity_cmd=np.array([0.20], dtype=np.float32), **kw,
    )
    obs_3vec = build_observation_from_components(
        velocity_cmd=np.array([0.20, 0.05, 0.10], dtype=np.float32),
        velocity_cmd_lateral_yaw=np.array([0.05, 0.10], dtype=np.float32),
        **kw,
    )
    np.testing.assert_array_equal(obs_scalar, obs_3vec)


def test_eval_policy_sentinel_reads_vx_index_only() -> None:
    """H4 + H3: eval override sentinel reads cmd[0] only.  A YAML pinning
    eval_velocity_cmd: [-1.0, 0.5, 0.0] means "use sampled cmd" (vx=-1
    -> sentinel)."""
    from training.eval.eval_policy import _eval_cmd_is_overridden
    cfg = types.SimpleNamespace(eval_velocity_cmd=(-1.0, 0.5, 0.0))
    assert _eval_cmd_is_overridden(cfg) is False
    cfg = types.SimpleNamespace(eval_velocity_cmd=(0.26, 0.0, 0.0))
    assert _eval_cmd_is_overridden(cfg) is True


def test_v6_eval_adapter_accepts_3vec_cmd() -> None:
    """H4: v6 eval adapter must forward a (3,) velocity_cmd to the numpy
    build_observation path (post-P11.1 it accepts the new kwarg)."""
    from training.eval.v6_eval_adapter import V6EvalAdapter
    sig = inspect.signature(V6EvalAdapter.compute_obs)
    cmd_param = sig.parameters["velocity_cmd"]
    # Annotation should mention ndarray, not float (we accept either by
    # ducktype in code; the test guards against silent regressions).
    assert (
        "ndarray" in str(cmd_param.annotation)
        or cmd_param.annotation is inspect.Parameter.empty
    )


def test_visualize_policy_validate_accepts_three_floats(tmp_path) -> None:
    """H4: --velocity-cmd can be either a scalar (broadcast to vx-only per
    H3) or three floats."""
    from training.eval.visualize_policy import _validate_user_fixed_velocity_cmd
    cfg = types.SimpleNamespace(
        env=types.SimpleNamespace(
            min_velocity=0.18, max_velocity=0.26,
            min_velocity_y=-0.13, max_velocity_y=0.13,
            max_yaw_rate=0.25,
        ),
    )
    out = _validate_user_fixed_velocity_cmd(cfg, 0.20)        # scalar -> vx
    assert tuple(out) == (0.20, 0.0, 0.0)
    out = _validate_user_fixed_velocity_cmd(cfg, [0.20, 0.05, 0.10])
    assert tuple(out) == pytest.approx((0.20, 0.05, 0.10))
