"""Parity smoke for V6EvalAdapter — keeps the native-MuJoCo viewer's
v6 obs builder honest against the env's JAX/MJX obs builder.

Why this guard exists: pre-Plan-A visualize_policy.py reimplemented the
obs / action / ref-preview / proprio-history paths in NumPy alongside
the JAX env.  Each v0.20.x contract change required matching visualizer
patches, and they eventually drifted to broken on
``wr_obs_v6_offline_ref_history`` (proprio_history was simply not
threaded, plus the residual action contract was missing).

Plan A added training/eval/v6_eval_adapter.py to host the v6 logic in
one place.  These tests pin:

  - obs_dim parity at step 0 (catches schema drift).
  - residual action contract at zero action (target_q == q_ref).
  - proprio history rolling cadence (newest slot fills with the bundle
    compute_obs() built for THIS step; oldest slot drops).
  - reference-window indexing (step_idx advances and clamps at
    end-of-trajectory).

For the deeper "adapter obs == env obs at step 0" numerical-parity
test, see test_v6_eval_adapter_env_parity below; the adapter obs and the
env obs should agree to single-precision tolerance on the same initial
state.
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
    """Build the (cfg, mj_model, mj_data, adapter, action_dim) fixture once."""
    cfg = load_training_config(str(SMOKE_YAML))
    robot_cfg_path = cfg.env.robot_config_path
    if not Path(robot_cfg_path).is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(str(robot_cfg_path))

    # Disable DR + no push so the test is deterministic.
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
    """The v6 obs vector size must match the policy spec.

    Catches: contract drift where the env / adapter / spec disagree on the
    obs layout (most common cause: someone bumps PROPRIO_HISTORY_FRAMES
    or adds a ref-preview channel without updating both sides).
    """
    s = smoke_setup
    assert s["policy_spec"].observation.layout_id == V6_LAYOUT_ID
    prev_action = np.zeros(s["action_dim"], dtype=np.float32)
    obs = s["adapter"].compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
    assert obs.shape == (s["policy_spec"].model.obs_dim,)
    assert obs.dtype == np.float32


def test_residual_zero_action_yields_q_ref(smoke_setup) -> None:
    """Zero residual must produce ``target_q == q_ref`` exactly.

    This is the env's "zero-residual invariant"
    (test_v0201_env_zero_action.py).  If the visualizer's residual path
    drifts from the env's, the policy will see different ctrl than what
    training saw at the same q_ref.
    """
    s = smoke_setup
    s["adapter"].reset()
    prev_action = np.zeros(s["action_dim"], dtype=np.float32)
    # compute_obs() must run first so step_idx is consistent.
    s["adapter"].compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
    target_q = s["adapter"].apply_action(
        s["mj_data"], np.zeros(s["action_dim"], dtype=np.float32)
    )
    q_ref = np.asarray(
        s["adapter"]._service.lookup_np(0).q_ref, dtype=np.float32
    )
    np.testing.assert_allclose(target_q, q_ref, atol=0.0)


def test_proprio_history_rolls_in_post_step(smoke_setup) -> None:
    """post_step() must roll the buffer: drop oldest, append the bundle
    that produced THIS step's obs.

    Catches: forgotten post_step() call in the viewer loop, or wrong
    roll direction (newest in slot 0 instead of slot -1, etc.).
    """
    s = smoke_setup
    adapter = s["adapter"]
    adapter.reset()
    prev_action = np.zeros(s["action_dim"], dtype=np.float32)

    # At reset, history is all zeros.
    np.testing.assert_array_equal(
        adapter._state.proprio_history,
        np.zeros((PROPRIO_HISTORY_FRAMES, adapter._bundle_size), dtype=np.float32),
    )

    # Step 1: compute_obs builds a bundle (non-zero gyro + joint pos).
    _ = adapter.compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
    bundle_step1 = adapter._pending_bundle.copy()
    adapter.post_step()
    # Newest slot must equal bundle_step1; second-newest still zeros.
    np.testing.assert_array_equal(
        adapter._state.proprio_history[-1], bundle_step1
    )
    np.testing.assert_array_equal(
        adapter._state.proprio_history[-2],
        np.zeros(adapter._bundle_size, dtype=np.float32),
    )


def test_step_idx_advances_and_clamps(smoke_setup) -> None:
    """step_idx must advance by 1 per post_step() and clamp at n_steps - 1
    so end-of-trajectory doesn't crash."""
    s = smoke_setup
    adapter = s["adapter"]
    adapter.reset()
    assert adapter.step_idx == 0

    prev_action = np.zeros(s["action_dim"], dtype=np.float32)
    for expected in (1, 2, 3):
        adapter.compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
        adapter.post_step()
        assert adapter.step_idx == expected

    # Force-fast-forward past end-of-trajectory.
    n = adapter.n_steps
    adapter._state.step_idx = n - 1
    adapter.compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
    adapter.post_step()
    assert adapter.step_idx == n - 1, "step_idx must clamp at n_steps - 1"


def test_reset_clears_history_and_step_idx(smoke_setup) -> None:
    s = smoke_setup
    adapter = s["adapter"]
    prev_action = np.zeros(s["action_dim"], dtype=np.float32)

    # Drive the adapter past zero state.
    adapter.compute_obs(s["mj_data"], prev_action, velocity_cmd=0.20)
    adapter.post_step()
    assert adapter.step_idx == 1
    assert adapter._state.proprio_history.any()

    adapter.reset()
    assert adapter.step_idx == 0
    np.testing.assert_array_equal(
        adapter._state.proprio_history,
        np.zeros((PROPRIO_HISTORY_FRAMES, adapter._bundle_size), dtype=np.float32),
    )
    assert adapter._state.prev_phase_sin_cos is None


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

    Schema offsets are derived from the policy_spec layout; if the layout
    changes (e.g. a ref-preview channel is added), this test breaks
    loudly with the new offsets and an "expected vs got" diff.
    """
    from training.envs.wildrobot_env import WildRobotEnv

    s = smoke_setup
    cfg = s["cfg"]

    # Build env from the SAME cfg the adapter uses.
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
    prev_action = np.zeros(s["action_dim"], dtype=np.float32)
    adapter_obs = adapter.compute_obs(
        mj_data, prev_action, velocity_cmd=float(cfg.env.eval_velocity_cmd)
    )

    assert adapter_obs.shape == env_obs.shape, (
        f"obs shape mismatch: adapter {adapter_obs.shape} vs env {env_obs.shape}"
    )

    # Reference-window slice.  Compute offsets from the layout to stay
    # robust against future channel additions: skip the v4 base
    # (gravity..velocity_cmd) and the v4-compat block, and start at
    # loc_ref_q_ref.  End just before proprio_history.
    layout = s["policy_spec"].observation.layout
    name_to_offset = {}
    cursor = 0
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
            f"(max={float(ref_diff.max()):.4e}).  First few: {details}.  "
            "Likely cause: adapter / env disagree on ZMP library lookup or "
            "v6 ref-preview channel layout."
        )

    # Proprio history slice must be exactly zero on both at reset.
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
