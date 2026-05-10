"""Parity smoke for V6EvalAdapter — keeps the native-MuJoCo viewer's
v6 obs / action / history paths honest against the env's JAX/MJX
implementation.

Why this guard exists: pre-Plan-A visualize_policy.py reimplemented the
obs / action / ref-preview / proprio-history paths in NumPy alongside
the JAX env.  Each v0.20.x contract change required matching visualizer
patches, and they eventually drifted to broken on
``wr_obs_v6_offline_ref_history`` (proprio_history was simply not
threaded; plus the residual action contract, action delay, and POST-
physics bundle assembly were missing).

Plan A added training/eval/v6_eval_adapter.py to host the v6 logic in
one place.  These tests pin the env-aligned semantics:

  - obs_dim parity at step 0 (catches schema drift).
  - reset puts step_idx=0, pending_action=zeros, last_applied_action=
    zeros, proprio_history=zeros, loc_ref_history=None.
  - apply_action advances step_idx BEFORE the q_ref lookup so iter 1
    targets q_ref[1], not q_ref[0] (env line 1686).
  - apply_action with action_delay_steps=1 applies pending_action
    (zeros at iter 1) regardless of raw_action; the new pending_action
    becomes the filtered raw_action for next step (env line 1719).
  - post_physics builds the bundle from POST-physics signals + the
    action APPLIED this step, and rolls it into history (env lines
    1905-1908).
  - obs at iter 1 (after iter-1 apply+physics+post_physics) sees the
    rolled history with bundle in slot -1.
  - adapter / env parity on the deterministic obs slice (ref window +
    proprio_history zeros at reset) matches.
"""

from __future__ import annotations

from pathlib import Path

import jax
import mujoco
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from policy_contract.spec import PROPRIO_HISTORY_FRAMES
from training.configs.training_config import load_training_config
from training.eval.v6_eval_adapter import V6_LAYOUT_ID, V6EvalAdapter
from training.policy_spec_utils import build_policy_spec_from_training_config
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_YAML = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke.yaml"
)


@pytest.fixture(scope="function")
def smoke_setup():
    """Build the (cfg, mj_model, mj_data, adapter, action_dim) fixture.

    Function-scoped so each test gets a fresh adapter; the adapter holds
    per-episode state (proprio_history, step_idx, pending_action) that
    would otherwise leak across tests.
    """
    cfg = load_training_config(str(SMOKE_YAML))
    robot_cfg_path = cfg.env.robot_config_path
    if not Path(robot_cfg_path).is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(str(robot_cfg_path))

    # Disable DR + push so the parity test is deterministic.
    cfg.env.domain_randomization_enabled = False
    cfg.env.push_enabled = False
    cfg.freeze()

    scene_path = cfg.env.scene_xml_path
    if not Path(scene_path).is_absolute():
        scene_path = str(PROJECT_ROOT / scene_path)
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    policy_spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg, action_filter_alpha=0.0
    )
    sig = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=cfg,
        mj_model=mj_model,
        policy_spec=policy_spec,
        signals_adapter=sig,
        action_dim=int(policy_spec.model.action_dim),
    )
    return {
        "cfg": cfg,
        "mj_model": mj_model,
        "mj_data": mj_data,
        "adapter": adapter,
        "policy_spec": policy_spec,
        "action_dim": int(policy_spec.model.action_dim),
    }


def test_obs_dim_matches_policy_spec(smoke_setup) -> None:
    s = smoke_setup
    assert s["policy_spec"].observation.layout_id == V6_LAYOUT_ID
    obs = s["adapter"].compute_obs(s["mj_data"], velocity_cmd=0.20)
    assert obs.shape == (s["policy_spec"].model.obs_dim,)
    assert obs.dtype == np.float32


def test_reset_state_matches_env_initial_conditions(smoke_setup) -> None:
    """Reset must put the adapter in the env's _make_initial_state shape:
    proprio_history zeros, loc_ref_history None, step_idx 0,
    pending_action and last_applied_action zeros."""
    s = smoke_setup
    a = s["adapter"]
    a.reset()
    assert a.step_idx == 0
    assert a._state.loc_ref_history is None
    np.testing.assert_array_equal(
        a._state.pending_action,
        np.zeros(s["action_dim"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        a._state.last_applied_action,
        np.zeros(s["action_dim"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        a._state.proprio_history,
        np.zeros((PROPRIO_HISTORY_FRAMES, a._bundle_size), dtype=np.float32),
    )


def test_apply_action_advances_step_idx_before_q_ref_lookup(smoke_setup) -> None:
    """At iter 1, the residual target must use q_ref[1], not q_ref[0]
    (env line 1686: ``next_step_idx = wr.loc_ref_offline_step_idx + 1``).

    Catches the off-by-one this commit fixes.
    """
    s = smoke_setup
    a = s["adapter"]
    a.reset()
    assert a.step_idx == 0

    # Force a non-zero raw_action so applied != q_ref-only and we can
    # check the lookup index unambiguously.  With delay=1, applied is
    # still pending_action=zeros, so target_q == q_ref at the new index.
    raw = 0.5 * np.ones(s["action_dim"], dtype=np.float32)
    a.apply_action(s["mj_data"], raw)
    assert a.step_idx == 1, "step_idx must be 1 after first apply_action"

    expected_q_ref = np.asarray(a._service.lookup_np(1).q_ref, dtype=np.float32)
    # Read the ctrl back out via the inverse permutation by building the
    # adapter's expected target and comparing through the same mapper.
    actual_ctrl = np.asarray(s["mj_data"].ctrl, dtype=np.float32)
    expected_ctrl = a._ctrl_mapper.to_mj_np(expected_q_ref)
    np.testing.assert_allclose(actual_ctrl, expected_ctrl, atol=1e-6,
        err_msg="apply_action must compose target_q from q_ref[step_idx + 1]")


def test_apply_action_with_delay_uses_pending_not_raw(smoke_setup) -> None:
    """With action_delay_steps=1 (smoke yaml), iter 1's applied action
    must be ``pending_action`` (zeros at reset), not the raw action.
    iter 2 then applies what was filtered at iter 1.

    Catches the missing pending_action / filter pipeline this commit fixes.
    """
    s = smoke_setup
    if not s["adapter"]._action_delay_enabled:
        pytest.skip("smoke yaml has action_delay_steps != 1; can't test delay")
    a = s["adapter"]
    a.reset()
    raw_iter1 = 0.5 * np.ones(s["action_dim"], dtype=np.float32)
    applied_iter1 = a.apply_action(s["mj_data"], raw_iter1)
    np.testing.assert_array_equal(
        applied_iter1, np.zeros(s["action_dim"], dtype=np.float32),
        err_msg="iter 1 applied must be pending=zeros under 1-step delay",
    )
    # pending_action now holds filtered_iter1 (= raw_iter1 with alpha=0).
    np.testing.assert_array_equal(
        a._state.pending_action, raw_iter1.astype(np.float32),
        err_msg="pending_action must equal filtered_iter1 after iter 1",
    )

    raw_iter2 = -0.25 * np.ones(s["action_dim"], dtype=np.float32)
    applied_iter2 = a.apply_action(s["mj_data"], raw_iter2)
    np.testing.assert_array_equal(
        applied_iter2, raw_iter1.astype(np.float32),
        err_msg="iter 2 applied must be the filtered raw from iter 1",
    )


def test_post_physics_bundle_uses_post_signals_and_applied_action(
    smoke_setup,
) -> None:
    """post_physics must build the bundle from POST-physics signals
    (read after mj_step) and the action APPLIED this step (not raw).

    Catches the pre-step bundle bug this commit fixes (history was being
    rolled with PRE-step data + the wrong prev_action slot).
    """
    s = smoke_setup
    a = s["adapter"]
    a.reset()
    raw = 0.3 * np.ones(s["action_dim"], dtype=np.float32)
    applied = a.apply_action(s["mj_data"], raw)
    # Step physics so post-signals differ from pre-signals.
    for _ in range(int(s["cfg"].env.ctrl_dt / s["cfg"].env.sim_dt)):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])
    # Newest slot (-1) of proprio_history is the bundle just rolled.
    bundle_newest = a._state.proprio_history[-1]
    # Bundle's prev_action slice is the last 21 entries.
    n = s["action_dim"]
    bundle_prev_action = bundle_newest[-n:]
    np.testing.assert_array_equal(
        bundle_prev_action, applied,
        err_msg="bundle prev_action slot must equal last_applied_action "
                "(env line 1906); not raw_action / not zeros",
    )


def test_proprio_history_rolls_after_post_physics_only(smoke_setup) -> None:
    """Iter 1's obs sees PRE-roll history (zeros).  Iter 2's obs sees
    history with iter-1's bundle in slot -1, after post_physics has run.

    Catches the wrong-timing roll bug this commit fixes (history was
    rolled before iter 2's obs read it, so iter 1's obs already saw the
    just-built bundle).
    """
    s = smoke_setup
    a = s["adapter"]
    a.reset()

    # Iter 1 obs.  Must be all-zero proprio history.
    obs_iter1 = a.compute_obs(s["mj_data"], velocity_cmd=0.20)
    layout = s["policy_spec"].observation.layout
    cursor = 0
    name_to_offset = {}
    for f in layout:
        name_to_offset[f.name] = (cursor, cursor + f.size)
        cursor += f.size
    hist_start, hist_end = name_to_offset["proprio_history"]
    np.testing.assert_array_equal(
        obs_iter1[hist_start:hist_end],
        np.zeros(hist_end - hist_start, dtype=np.float32),
        err_msg="iter 1 obs proprio_history slice must be zeros",
    )

    # Apply, step, post_physics.
    raw = 0.1 * np.ones(s["action_dim"], dtype=np.float32)
    a.apply_action(s["mj_data"], raw)
    for _ in range(int(s["cfg"].env.ctrl_dt / s["cfg"].env.sim_dt)):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])

    # Iter 2 obs.  Now history slot -1 has iter-1's bundle; older slots
    # still zeros.
    obs_iter2 = a.compute_obs(s["mj_data"], velocity_cmd=0.20)
    flat_hist = obs_iter2[hist_start:hist_end].reshape(
        PROPRIO_HISTORY_FRAMES, a._bundle_size
    )
    assert flat_hist[-1].any(), "iter 2 obs history slot -1 must be non-zero"
    np.testing.assert_array_equal(
        flat_hist[-2],
        np.zeros(a._bundle_size, dtype=np.float32),
        err_msg="iter 2 obs history slot -2 must still be zeros",
    )


def test_step_idx_clamps_at_end_of_trajectory(smoke_setup) -> None:
    s = smoke_setup
    a = s["adapter"]
    a.reset()
    a._state.step_idx = a.n_steps - 1
    a.apply_action(s["mj_data"], np.zeros(s["action_dim"], dtype=np.float32))
    assert a.step_idx == a.n_steps - 1, "step_idx must clamp at n_steps - 1"


def test_v6_eval_adapter_env_parity_deterministic_slice(smoke_setup) -> None:
    """Adapter and env must agree on the *deterministic* obs slice at
    step 0: reference-window channels and the proprio_history (zeros at
    reset).

    The full obs is NOT compared because env's reset path applies
    sources of stochasticity that the native-MuJoCo path can't replicate
    bit-for-bit:
      - IMU noise + delay (env._apply_imu_noise_and_delay) → gravity/gyro
      - joint reset noise (env.reset, ±0.05 rad) → joint_pos / joint_vel
      - DR backlash on observed joint_pos (when DR enabled)
    Those slices are inherently env-stochastic; the adapter would have
    to reimplement the noise pipeline to match them, which defeats the
    "use shared library" goal.

    What we DO pin: the reference window (loc_ref_q_ref, loc_ref_pelvis_*,
    loc_ref_*_foot_*, loc_ref_contact_mask, loc_ref_phase_*, etc.) is a
    deterministic lookup into the same offline ReferenceLibrary.  And the
    proprio_history must be all zeros at step 0 in both implementations
    (env_info.py contract: "buffer holds PAST bundles only, never the
    current frame; zero-filled at reset").
    """
    from training.envs.wildrobot_env import WildRobotEnv

    s = smoke_setup
    cfg = s["cfg"]

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(0)
    env_state = env.reset_for_eval(rng)
    env_obs = np.asarray(env_state.obs)

    mj_model = s["mj_model"]
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[:] = np.asarray(env_state.data.qpos)
    mj_data.qvel[:] = np.asarray(env_state.data.qvel)
    mujoco.mj_forward(mj_model, mj_data)

    adapter = s["adapter"]
    adapter.reset()
    adapter_obs = adapter.compute_obs(
        mj_data, velocity_cmd=float(cfg.env.eval_velocity_cmd)
    )

    assert adapter_obs.shape == env_obs.shape, (
        f"obs shape mismatch: adapter {adapter_obs.shape} vs env {env_obs.shape}"
    )

    layout = s["policy_spec"].observation.layout
    cursor = 0
    name_to_offset = {}
    for f in layout:
        name_to_offset[f.name] = (cursor, cursor + f.size)
        cursor += f.size
    ref_start = name_to_offset["loc_ref_q_ref"][0]
    ref_end = name_to_offset["loc_ref_contact_mask"][1]
    ref_diff = np.abs(adapter_obs[ref_start:ref_end] - env_obs[ref_start:ref_end])
    if float(ref_diff.max()) > 1e-5:
        bad = np.where(ref_diff > 1e-5)[0][:5]
        details = ", ".join(
            f"local_idx={int(i)} adapter={adapter_obs[ref_start + i]:.6f} "
            f"env={env_obs[ref_start + i]:.6f} diff={ref_diff[i]:.6f}"
            for i in bad
        )
        pytest.fail(
            f"reference-window obs slice [{ref_start}:{ref_end}] disagrees "
            f"in {int((ref_diff > 1e-5).sum())} indices "
            f"(max={float(ref_diff.max()):.4e}).  First few: {details}"
        )

    hist_start, hist_end = name_to_offset["proprio_history"]
    np.testing.assert_array_equal(
        adapter_obs[hist_start:hist_end],
        np.zeros(hist_end - hist_start, dtype=np.float32),
        err_msg="adapter proprio_history must be zero at step 0 (env contract)",
    )
    np.testing.assert_array_equal(
        env_obs[hist_start:hist_end],
        np.zeros(hist_end - hist_start, dtype=np.float32),
        err_msg="env proprio_history must be zero at step 0",
    )
