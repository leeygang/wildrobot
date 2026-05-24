from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.eval.visualize_policy import (
    _adapter_layout_enabled,
    _build_sample_velocity_cmd,
    _is_terminated_from_pose,
    _network_activation_name,
    _validate_user_fixed_velocity_cmd,
)
from training.policy_spec_utils import build_policy_spec_from_training_config
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE11_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke11.yaml"
)
SMOKE12B_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12b.yaml"
)
SMOKE1_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0210_smoke1_lateral_yaw.yaml"
)
SMOKE14_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke14.yaml"
)


def test_visualizer_adapter_layout_supports_v6_and_v7() -> None:
    assert _adapter_layout_enabled("wr_obs_v6_offline_ref_history") is True
    assert _adapter_layout_enabled("wr_obs_v7_phase_proprio") is True
    assert _adapter_layout_enabled("wr_obs_v4") is False


def test_visualizer_threads_actor_activation_from_training_config() -> None:
    cfg = load_training_config(str(SMOKE12B_CFG))
    assert _network_activation_name(cfg) == "elu"
    cfg.networks.critic.activation = "silu"
    with pytest.raises(ValueError, match="must match"):
        _network_activation_name(cfg)


def test_visualizer_rejects_out_of_range_fixed_velocity() -> None:
    # v0.21.0 P11: _validate_user_fixed_velocity_cmd now returns a (3,)
    # ndarray (vx, vy, wz); scalar input broadcasts to vx-only.
    cfg = load_training_config(str(SMOKE12B_CFG))
    out = _validate_user_fixed_velocity_cmd(cfg, 0.10)
    assert tuple(out) == pytest.approx((0.10, 0.0, 0.0))
    with pytest.raises(ValueError, match="outside configured range"):
        _validate_user_fixed_velocity_cmd(cfg, float(cfg.env.min_velocity) - 1e-3)
    with pytest.raises(ValueError, match="outside configured range"):
        _validate_user_fixed_velocity_cmd(cfg, float(cfg.env.max_velocity) + 1e-3)


def test_visualizer_relaxed_termination_ignores_pitch_roll() -> None:
    cfg = load_training_config(str(SMOKE12B_CFG))
    assert cfg.env.use_relaxed_termination is True
    assert (
        _is_terminated_from_pose(
            height=0.46,
            roll_rad=0.0,
            pitch_rad=1.2,
            step_count=10,
            training_cfg=cfg,
        )
        is False
    )
    assert (
        _is_terminated_from_pose(
            height=float(cfg.env.min_height) - 0.01,
            roll_rad=0.0,
            pitch_rad=0.0,
            step_count=10,
            training_cfg=cfg,
        )
        is True
    )


def test_visualizer_v7_layout_uses_native_eval_adapter_contract() -> None:
    cfg = load_training_config(str(SMOKE11_CFG))
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(str(robot_cfg_path))

    scene_path = Path(cfg.env.scene_xml_path)
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)

    policy_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
        action_filter_alpha=float(cfg.env.action_filter_alpha),
    )
    signals_adapter = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=cfg,
        mj_model=mj_model,
        policy_spec=policy_spec,
        signals_adapter=signals_adapter,
        action_dim=int(policy_spec.model.action_dim),
    )
    adapter.reset_native_mj_state(mj_data, apply_noise=False, rng=None)
    obs = adapter.compute_obs(
        mj_data=mj_data,
        # v0.21.0 P3 / H3: ``eval_velocity_cmd`` is (vx, vy, wz); the
        # visualize-policy v6/v7 obs adapter still takes a scalar vx.
        velocity_cmd=float(cfg.env.eval_velocity_cmd[0]),
    )
    assert obs.shape == (policy_spec.model.obs_dim,)


# =============================================================================
# Reviewer bug 2: visualize sampler must mirror env branched 3D sampler
# under cmd_sampler_3d_branched opt-in.
# =============================================================================


def test_visualize_sampler_uses_branched_3d_when_smoke1_opt_in() -> None:
    """Reviewer bug 2: when cfg.env.cmd_sampler_3d_branched=True the
    visualize sampler must mirror the env's branched sampler -- at
    least one of (vy, wz) must vary across many samples.  Without this
    fix, smoke1 visualization without --velocity-cmd / --demo defaults
    to scalar-vx only and never exercises lateral / turning bins."""
    if not SMOKE1_CFG.exists():
        pytest.skip(f"{SMOKE1_CFG.name} not found")
    cfg = load_training_config(str(SMOKE1_CFG))
    assert bool(getattr(cfg.env, "cmd_sampler_3d_branched", False)) is True, (
        "smoke1 yaml must set cmd_sampler_3d_branched=True for this test"
    )
    rng = np.random.default_rng(0)
    sampler = _build_sample_velocity_cmd(cfg, rng)
    cmds = np.stack([sampler() for _ in range(512)])
    assert cmds.shape == (512, 3) and cmds.dtype == np.float32
    assert np.any(np.abs(cmds[:, 1]) > 1e-3), (
        "vy was always zero under branched 3D mode; sampler did not route "
        "through the walk-ellipse / pure-turn branches"
    )
    assert np.any(np.abs(cmds[:, 2]) > 1e-3), (
        "wz was always zero under branched 3D mode; sampler did not route "
        "through the pure-turn branch"
    )


def test_visualize_sampler_stays_scalar_vx_for_smoke14() -> None:
    """Back-compat: smoke14 (cmd_sampler_3d_branched=False) must still
    sample (vx, 0, 0).  Catches an over-aggressive flip that would have
    smoke14 / smoke12b / smoke7 visualization start emitting lateral
    or yaw commands the underlying 1D reference library can't honor."""
    if not SMOKE14_CFG.exists():
        pytest.skip(f"{SMOKE14_CFG.name} not found")
    cfg = load_training_config(str(SMOKE14_CFG))
    assert bool(getattr(cfg.env, "cmd_sampler_3d_branched", False)) is False
    rng = np.random.default_rng(0)
    sampler = _build_sample_velocity_cmd(cfg, rng)
    cmds = np.stack([sampler() for _ in range(64)])
    assert cmds.shape == (64, 3) and cmds.dtype == np.float32
    assert np.all(np.abs(cmds[:, 1]) < 1e-9), (
        f"vy must be exactly zero under legacy scalar-vx sampler; got "
        f"max |vy| = {float(np.abs(cmds[:, 1]).max()):.6e}"
    )
    assert np.all(np.abs(cmds[:, 2]) < 1e-9), (
        f"wz must be exactly zero under legacy scalar-vx sampler; got "
        f"max |wz| = {float(np.abs(cmds[:, 2]).max()):.6e}"
    )

