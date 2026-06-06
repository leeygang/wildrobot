"""Runtime observation: wr_obs_v8_cmd3d obs shape + command slot semantics."""

from __future__ import annotations

import numpy as np

from policy_contract.numpy.signals import Signals
from policy_contract.spec import PROPRIO_HISTORY_FRAMES
from runtime.wr_runtime.control.policy_runner import RuntimePolicyRunner


class _ZeroPolicy:
    def __init__(self, action_dim: int) -> None:
        self._n = action_dim

    def predict(self, obs):
        return np.zeros(self._n, dtype=np.float32)


def _signals(spec, *, joint_pos=None) -> Signals:
    n = spec.model.action_dim
    return Signals(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.arange(3, dtype=np.float32) * 0.01,
        joint_pos_rad=(
            np.zeros(n, dtype=np.float32) if joint_pos is None else joint_pos
        ),
        joint_vel_rad_s=np.zeros(n, dtype=np.float32),
        foot_switches=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        timestamp_s=0.0,
    )


def _runner(v8_spec, runtime_policy_config):
    return RuntimePolicyRunner(
        spec=v8_spec,
        runtime_config=runtime_policy_config,
        policy=_ZeroPolicy(v8_spec.model.action_dim),
        robot_io=None,
    )


def test_obs_shape_is_1129(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    obs = runner.build_obs(_signals(v8_spec), np.array([0.13, 0.065, 0.0]))
    assert obs.shape == (v8_spec.model.obs_dim,)
    assert obs.shape == (1129,)  # 21-actuator v8 contract


def _slot_offsets(action_dim: int) -> dict:
    o = {}
    i = 0
    for name, size in [
        ("gravity_local", 3),
        ("angvel_heading_local", 3),
        ("joint_pos_normalized", action_dim),
        ("joint_vel_normalized", action_dim),
        ("foot_switches", 4),
        ("prev_action", action_dim),
        ("velocity_cmd", 1),
        ("loc_ref_phase_sin_cos", 2),
        ("proprio_history", PROPRIO_HISTORY_FRAMES * (3 + 4 + 3 * action_dim)),
        ("velocity_cmd_lateral_yaw", 2),
        ("padding", 1),
    ]:
        o[name] = (i, i + size)
        i += size
    return o


def test_velocity_cmd_slot_carries_vx(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    cmd = np.array([0.13, 0.065, -0.02], dtype=np.float32)
    obs = runner.build_obs(_signals(v8_spec), cmd)
    off = _slot_offsets(v8_spec.model.action_dim)
    lo, hi = off["velocity_cmd"]
    assert obs[lo:hi] == np.float32(0.13)


def test_lateral_yaw_slot_carries_vy_wz(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    cmd = np.array([0.13, 0.065, -0.02], dtype=np.float32)
    obs = runner.build_obs(_signals(v8_spec), cmd)
    off = _slot_offsets(v8_spec.model.action_dim)
    lo, hi = off["velocity_cmd_lateral_yaw"]
    np.testing.assert_allclose(obs[lo:hi], [0.065, -0.02], atol=1e-6)


def test_proprio_history_rolls_oldest_to_newest(v8_spec, runtime_policy_config):
    runner = _runner(v8_spec, runtime_policy_config)
    action_dim = v8_spec.model.action_dim
    off = _slot_offsets(action_dim)
    hlo, hhi = off["proprio_history"]
    bundle_size = 3 + 4 + 3 * action_dim

    # First obs: history all zeros (pre-roll, env iter-1 semantics).
    obs0 = runner.build_obs(_signals(v8_spec), np.array([0.13, 0.0, 0.0]))
    assert np.allclose(obs0[hlo:hhi], 0.0)

    # Apply a distinctive action and roll a POST bundle with nonzero gyro.
    runner.compose_and_apply(np.ones(action_dim, dtype=np.float32))
    post = _signals(v8_spec)
    post = Signals(
        quat_xyzw=post.quat_xyzw,
        gyro_rad_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        joint_pos_rad=post.joint_pos_rad,
        joint_vel_rad_s=post.joint_vel_rad_s,
        foot_switches=post.foot_switches,
    )
    runner.roll_history(post)

    # After two more build_obs calls the bundle becomes visible at the NEWEST
    # frame (env 1-iteration lag promotes pending -> proprio on read).
    runner.build_obs(_signals(v8_spec), np.array([0.13, 0.0, 0.0]))
    obs2 = runner.build_obs(_signals(v8_spec), np.array([0.13, 0.0, 0.0]))
    history = obs2[hlo:hhi].reshape(PROPRIO_HISTORY_FRAMES, bundle_size)
    newest = history[-1]
    # newest frame's first 3 channels are the POST gyro we injected.
    np.testing.assert_allclose(newest[:3], [1.0, 2.0, 3.0], atol=1e-6)
    # the frame before it is still zero (only one bundle rolled in).
    assert np.allclose(history[-2], 0.0)
