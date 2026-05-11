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
from policy_contract.calib import NumpyCalibOps
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
    (read after mj_step) and the action APPLIED this step, then roll it
    into ``pending_history`` (NOT ``proprio_history`` — the lag fix
    moved the write target to the pending buffer).

    Drives TWO iterations so iter 2's ``applied`` is the filtered raw
    from iter 1 — non-zero — and the bundle's prev_action slot is a
    meaningful check.  (Iter 1's applied is always zero under
    action_delay_steps=1; reading the slot then would pass vacuously.)

    Catches the pre-step bundle bug from commit 1a467c1 (history was
    rolled with PRE-step data + wrong prev_action slot) AND the wrong-
    buffer bug had the test inspected proprio_history[-1] (where post_
    physics no longer writes since the lag fix in 49be22e).
    """
    s = smoke_setup
    if not s["adapter"]._action_delay_enabled:
        pytest.skip("test assumes 1-step delay; smoke yaml has it on")
    a = s["adapter"]
    a.reset()
    n = s["action_dim"]
    n_substeps = int(s["cfg"].env.ctrl_dt / s["cfg"].env.sim_dt)

    # ---- Iter 1: applied = pending = zeros (delay).  Drive the cycle
    # so iter 2's apply_action sees a non-zero pending_action (=
    # filtered_iter1).  We don't inspect iter 1's bundle here — its
    # applied is zero, so the prev_action slot would be all-zero and
    # not prove anything.
    raw_iter1 = 0.3 * np.ones(n, dtype=np.float32)
    applied_iter1 = a.apply_action(s["mj_data"], raw_iter1)
    np.testing.assert_array_equal(
        applied_iter1, np.zeros(n, dtype=np.float32),
        err_msg="iter 1 applied must be zero under 1-step delay (precondition)",
    )
    for _ in range(n_substeps):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])
    # After iter 1's compute_obs would normally promote pending_history
    # → proprio_history; the assertion below targets iter 2's
    # post_physics output which writes to pending_history again, so the
    # promotion in between doesn't change what we check.
    a.compute_obs(s["mj_data"], velocity_cmd=0.20)

    # ---- Iter 2: applied = pending = filtered_iter1 = raw_iter1 (alpha=0).
    # Now the bundle's prev_action slot will be a meaningful non-zero
    # value, so checking it actually proves the bundle uses applied
    # (not raw, not zeros).
    raw_iter2 = -0.6 * np.ones(n, dtype=np.float32)
    applied_iter2 = a.apply_action(s["mj_data"], raw_iter2)
    np.testing.assert_array_equal(
        applied_iter2, raw_iter1,
        err_msg=("iter 2 applied must equal filtered_iter1 (= raw_iter1 with "
                 "alpha=0); precondition for the bundle prev_action check"),
    )
    assert not np.allclose(applied_iter2, 0.0), (
        "applied_iter2 must be non-zero so the prev_action slot check is "
        "non-vacuous"
    )

    for _ in range(n_substeps):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])

    # post_physics writes the rolled history into pending_history (lag
    # fix 49be22e).  Inspect THAT buffer's newest slot.
    bundle_newest = a._state.pending_history[-1]
    bundle_prev_action = bundle_newest[-n:]
    np.testing.assert_array_equal(
        bundle_prev_action, applied_iter2,
        err_msg=("bundle prev_action slot must equal last_applied_action "
                 "(env line 1906); not raw_iter2, not zeros"),
    )

    # Also pin POST-vs-PRE physics: the bundle's joint_pos_norm slice
    # (offset 7..7+n in the bundle, since layout = gyro(3) + sw(4) +
    # joint_pos(n) + joint_vel(n) + prev_action(n)) must reflect the
    # POST-physics joint state, not the pre-step state.  Sanity-check
    # by confirming joint_pos_norm corresponds to the *current* mj_data
    # qpos (which is post-physics at this point).
    expected_joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
        spec=s["policy_spec"],
        joint_pos_rad=a._signals_adapter.read(s["mj_data"]).joint_pos_rad,
    ).astype(np.float32)
    bundle_joint_pos = bundle_newest[7 : 7 + n]
    np.testing.assert_allclose(
        bundle_joint_pos, expected_joint_pos_norm, atol=1e-6,
        err_msg=("bundle joint_pos_norm slot must reflect POST-physics qpos; "
                 "if this fails, post_physics is reading mj_data BEFORE "
                 "the physics step or signals_adapter is misconfigured"),
    )


def test_proprio_history_lags_one_iteration(smoke_setup) -> None:
    """Bundle rolled at end of iter N's post_physics must FIRST appear in
    the obs returned by compute_obs at iter N+2 (not iter N+1).

    Why one extra iteration of lag: env.step at iter N stores the rolled
    history in ``new_wr.proprio_history`` but explicitly returns its obs
    using the PRE-roll ``wr.proprio_history`` (env line 1979).  PPO at
    iter N+1 acts on env.step iter N's RETURNED obs → sees PRE-roll
    history → bundle_N is NOT visible yet.  Bundle_N is only consumed by
    env.step iter N+1's roll input, becoming the pre-roll for iter N+2's
    returned obs.

    Mapped to visualizer iterations:
      - iter 1 obs (= reset obs): zero history.
      - iter 2 obs (= env.step iter 1 returned obs): zero history.
      - iter 3 obs (= env.step iter 2 returned obs): bundle_1 in slot -1.

    The previous version of this test pinned the WRONG behavior
    (asserting iter 2 obs sees bundle_1 in slot -1) — because the
    adapter was promoting the rolled history into compute_obs's view
    one iteration too early.  This rewrite pins the correct env-aligned
    behavior.
    """
    s = smoke_setup
    a = s["adapter"]
    a.reset()

    layout = s["policy_spec"].observation.layout
    cursor = 0
    name_to_offset = {}
    for f in layout:
        name_to_offset[f.name] = (cursor, cursor + f.size)
        cursor += f.size
    hist_start, hist_end = name_to_offset["proprio_history"]
    n_substeps = int(s["cfg"].env.ctrl_dt / s["cfg"].env.sim_dt)

    def obs_history_view(obs_vec: np.ndarray) -> np.ndarray:
        return obs_vec[hist_start:hist_end].reshape(
            PROPRIO_HISTORY_FRAMES, a._bundle_size
        )

    # ---- iter 1 (reset obs): zero history.
    obs_iter1 = a.compute_obs(s["mj_data"], velocity_cmd=0.20)
    np.testing.assert_array_equal(
        obs_history_view(obs_iter1),
        np.zeros((PROPRIO_HISTORY_FRAMES, a._bundle_size), dtype=np.float32),
        err_msg="iter 1 obs proprio_history must be zeros (reset state)",
    )
    # Apply iter 1's action, physics, post_physics.
    raw_iter1 = 0.1 * np.ones(s["action_dim"], dtype=np.float32)
    a.apply_action(s["mj_data"], raw_iter1)
    for _ in range(n_substeps):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])

    # ---- iter 2 obs: STILL zero history (one-iteration lag).
    obs_iter2 = a.compute_obs(s["mj_data"], velocity_cmd=0.20)
    np.testing.assert_array_equal(
        obs_history_view(obs_iter2),
        np.zeros((PROPRIO_HISTORY_FRAMES, a._bundle_size), dtype=np.float32),
        err_msg=(
            "iter 2 obs proprio_history must STILL be zeros — env's iter 1 "
            "returned obs uses pre-roll history; the bundle rolled at iter 1's "
            "post_physics is only visible in env's iter 2 returned obs (= "
            "visualizer iter 3 compute_obs)"
        ),
    )
    # Apply iter 2's action, physics, post_physics.
    raw_iter2 = -0.05 * np.ones(s["action_dim"], dtype=np.float32)
    a.apply_action(s["mj_data"], raw_iter2)
    for _ in range(n_substeps):
        mujoco.mj_step(s["mj_model"], s["mj_data"])
    a.post_physics(s["mj_data"])

    # ---- iter 3 obs: bundle_1 in slot -1 (FIRST appearance).
    obs_iter3 = a.compute_obs(s["mj_data"], velocity_cmd=0.20)
    hist3 = obs_history_view(obs_iter3)
    assert hist3[-1].any(), (
        "iter 3 obs history slot -1 must be non-zero (bundle_1 first visible)"
    )
    np.testing.assert_array_equal(
        hist3[-2],
        np.zeros(a._bundle_size, dtype=np.float32),
        err_msg="iter 3 obs history slot -2 must still be zeros (bundle_2 not visible yet)",
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


def test_adapter_obs_history_matches_env_for_three_steps(smoke_setup) -> None:
    """End-to-end: drive both env and adapter for 3 steps with the same
    zero actions and confirm the adapter's compute_obs proprio_history
    SLOT pattern (which frames are zero vs populated) tracks env's
    returned state.obs proprio_history at every iteration.

    This is the load-bearing test that catches the proprio_history
    timing class of bug.  Earlier revisions had compute_obs reading
    history one iteration too early (rolled bundle visible immediately
    after post_physics, instead of one extra iteration later).  This
    test pins iteration N's obs against env's state_{N-1}.obs for N in
    {1, 2, 3, 4} so any future re-introduction of the lag bug fails
    here loudly.

    Compares only the SLOT-OCCUPANCY pattern (per-frame "is this slot
    all zeros?"), not exact bundle values.  Numerical values within a
    bundle differ between adapter and env because env applies IMU noise
    + backlash to its bundle's gyro / joint_pos slices and the adapter
    uses raw post-physics signals (same rationale as
    test_v6_eval_adapter_env_parity_deterministic_slice).  The
    occupancy pattern is the timing-sensitive part this test exists to
    pin.
    """
    from training.envs.wildrobot_env import WildRobotEnv

    s = smoke_setup
    cfg = s["cfg"]

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(0)
    env_state = env.reset_for_eval(rng)

    mj_model = s["mj_model"]
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    # Sync env's qpos / qvel into native MJ data so signals match
    # bit-for-bit at iteration 1 (and approximately thereafter, modulo
    # tiny solver differences between mjx and native mujoco).
    mj_data.qpos[:] = np.asarray(env_state.data.qpos)
    mj_data.qvel[:] = np.asarray(env_state.data.qvel)
    mujoco.mj_forward(mj_model, mj_data)

    adapter = s["adapter"]
    adapter.reset()

    layout = s["policy_spec"].observation.layout
    cursor = 0
    name_to_offset = {}
    for f in layout:
        name_to_offset[f.name] = (cursor, cursor + f.size)
        cursor += f.size
    hist_start, hist_end = name_to_offset["proprio_history"]
    n_substeps = int(cfg.env.ctrl_dt / cfg.env.sim_dt)

    def slot_occupancy(obs_vec) -> np.ndarray:
        """Per-frame mask: True if that frame slot has any non-zero value."""
        history = np.asarray(obs_vec[hist_start:hist_end]).reshape(
            PROPRIO_HISTORY_FRAMES, adapter._bundle_size
        )
        return np.any(history != 0.0, axis=1)

    # ---- iter 1: both at reset; both must show all-zero history.
    adapter_obs = adapter.compute_obs(
        mj_data, velocity_cmd=float(cfg.env.eval_velocity_cmd)
    )
    np.testing.assert_array_equal(
        slot_occupancy(adapter_obs),
        slot_occupancy(env_state.obs),
        err_msg="iter 1: adapter and env proprio_history must both be all-zero",
    )

    zeros_action = np.zeros(s["action_dim"], dtype=np.float32)
    for iter_idx in (2, 3, 4):
        # Step env forward.
        env_state = env.step(
            env_state,
            jax.numpy.zeros(s["action_dim"], dtype=jax.numpy.float32),
            disable_cmd_resample=True,
        )
        # Step adapter forward.
        adapter.apply_action(mj_data, zeros_action)
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)
        adapter.post_physics(mj_data)
        # Re-sync mj_data qpos/qvel from env to keep slot-occupancy
        # deterministic across many steps (mjx and native mj_step diverge
        # numerically over time; this test only cares about timing).
        mj_data.qpos[:] = np.asarray(env_state.data.qpos)
        mj_data.qvel[:] = np.asarray(env_state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        adapter_obs = adapter.compute_obs(
            mj_data, velocity_cmd=float(cfg.env.eval_velocity_cmd)
        )
        adapter_occ = slot_occupancy(adapter_obs)
        env_occ = slot_occupancy(env_state.obs)
        np.testing.assert_array_equal(
            adapter_occ,
            env_occ,
            err_msg=(
                f"iter {iter_idx}: adapter and env proprio_history slot "
                "occupancy must agree.  Most likely cause: adapter is "
                "promoting the rolled bundle into compute_obs's view at the "
                "wrong time (one iteration too early or too late).  See test "
                "docstring for the env contract.\n"
                f"  adapter occupancy (oldest -> newest): {adapter_occ.tolist()}\n"
                f"  env     occupancy (oldest -> newest): {env_occ.tolist()}"
            ),
        )


# =============================================================================
# Smoke8 (TB-aligned home-base residual) parity
# =============================================================================
# These tests guard the adapter's home-base contract: smoke8 checkpoints
# eval'd via visualize_policy.py must compose target_q with the same
# base the env used during training.  Without these the adapter
# silently falls back to q_ref base for ANY checkpoint, masking the
# smoke8 architectural change at eval time.

SMOKE8_YAML = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke8.yaml"
)


@pytest.fixture(scope="function")
def smoke8_setup():
    """smoke_setup analog for smoke8 (env.loc_ref_residual_base: home).

    Skipped if the smoke8 yaml hasn't been added to the repo yet.
    """
    if not SMOKE8_YAML.exists():
        pytest.skip(f"{SMOKE8_YAML.name} not found (smoke8 not added)")
    cfg = load_training_config(str(SMOKE8_YAML))
    robot_cfg_path = cfg.env.robot_config_path
    if not Path(robot_cfg_path).is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(str(robot_cfg_path))

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


def test_smoke8_adapter_residual_base_flag_threaded(smoke8_setup) -> None:
    """The adapter must read env.loc_ref_residual_base at init.  Catches
    a silent fallback to q_ref base when the yaml asks for home — which
    would mean visualize_policy.py applies the smoke7 control contract
    to a smoke8 checkpoint.
    """
    s = smoke8_setup
    assert s["adapter"]._residual_base_mode == "home", (
        f"adapter._residual_base_mode = "
        f"{s['adapter']._residual_base_mode!r}; smoke8 yaml sets "
        "loc_ref_residual_base: home — flag not threaded into adapter."
    )


def test_smoke8_adapter_zero_action_writes_home_ctrl(smoke8_setup) -> None:
    """Under home base + zero raw action, apply_action must write
    home_q_rad to mj_data.ctrl (modulo PolicySpec→MJ ctrl-order
    permutation).  This is the adapter's analog of the env's
    ``test_smoke8_zero_action_applied_target_q_equals_home_under_full_step``.

    With action_delay_steps=1 the very first apply_action sees
    pending_action=zeros (filter zero in, zero out), so applied=zeros
    and the composed target should be exactly home — independent of the
    raw action passed in.  We pass raw=0.5*ones to make the assertion
    sharper: if the adapter were silently using q_ref as base, the
    output would not be byte-equal to home.
    """
    s = smoke8_setup
    a = s["adapter"]
    a.reset()

    raw = 0.5 * np.ones(s["action_dim"], dtype=np.float32)
    a.apply_action(s["mj_data"], raw)

    # apply_action delayed mode: applied = pending_action = zeros, so
    # residual_delta = 0 and target_q = home (regardless of raw).
    home = a._home_q_rad
    expected_ctrl = a._ctrl_mapper.to_mj_np(home)
    actual_ctrl = np.asarray(s["mj_data"].ctrl, dtype=np.float32)
    np.testing.assert_allclose(
        actual_ctrl, expected_ctrl, atol=1e-5,
        err_msg=(
            "smoke8 adapter: zero applied-action under home base must "
            "write home_q_rad to ctrl.  If actual_ctrl looks like "
            "q_ref[1] instead, the adapter is using the wrong residual "
            "base — visualize_policy.py would silently apply smoke7 "
            "semantics to a smoke8 checkpoint."
        ),
    )


def test_smoke7_adapter_residual_base_default_q_ref(smoke_setup) -> None:
    """Smoke7 yaml (loc_ref_residual_base unset → defaults to 'q_ref')
    must NOT regress to home base.  Catches an over-aggressive default
    flip when extending the flag."""
    s = smoke_setup
    assert s["adapter"]._residual_base_mode == "q_ref", (
        f"smoke7 adapter._residual_base_mode = "
        f"{s['adapter']._residual_base_mode!r}; expected 'q_ref' (the "
        "back-compat default).  Did the dataclass default flip?"
    )
