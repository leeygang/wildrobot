"""V6EvalAdapter parity for v0.21.0 P5/P11 3D-reference-library mode.

These tests pin the adapter's per-step bin selection against
``WildRobotEnv._lookup_offline_window`` when
``env.loc_ref_command_axes_3d=True`` (smoke1 opt-in).  Without these,
the adapter silently picks the forward ``(offline_vx, 0, 0)`` bin
regardless of the (vy, wz) command axes — a contract mismatch for any
``wr_obs_v8_cmd3d`` checkpoint trained against the 3D env.

The bug the test catches:

  - ``_init_offline_service`` builds a single trajectory from the
    forward offline_vx and stores ``self._service``.
  - ``compute_obs`` always calls ``self._service.lookup_np(step_idx)``.
  - ``velocity_cmd[1:]`` only flows through the obs builder's
    ``velocity_cmd_lateral_yaw`` slot; the phase / stance / swing /
    pelvis_targets channels stay anchored to the forward bin.

Smoke1 trains v8 against the env's 3-axis nearest-bin selector, so a
``(0.20, 0.10, 0.0)`` cmd MUST surface a different reference window
than a pure-vx ``(0.20, 0, 0)`` cmd.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.policy_spec_utils import build_policy_spec_from_training_config
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE1_YAML = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0210_smoke1_lateral_yaw.yaml"
)
SMOKE14_YAML = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke14.yaml"
)


def _build_adapter(yaml_path: Path):
    """Build (cfg, mj_model, mj_data, adapter, action_dim, policy_spec).

    Disables DR + push for deterministic obs.  Returns ``None`` if the
    config can't be loaded (e.g. asset paths missing in the test env).
    """
    cfg = load_training_config(str(yaml_path))
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


@pytest.fixture(scope="module")
def smoke1_setup():
    if not SMOKE1_YAML.exists():
        pytest.skip(f"{SMOKE1_YAML.name} not found")
    return _build_adapter(SMOKE1_YAML)


def test_v6_adapter_3d_mode_initializes_per_bin_services(smoke1_setup) -> None:
    """Under loc_ref_command_axes_3d=True the adapter must build the
    full 3D library (per-bin services + cmd_keys) — not just a single
    forward-vx service.  Catches the bug where the adapter silently
    falls back to ``[offline_vx]`` 1D mode under the 3D opt-in YAML."""
    s = smoke1_setup
    a = s["adapter"]
    cfg = s["cfg"]
    assert bool(getattr(cfg.env, "loc_ref_command_axes_3d", False)) is True, (
        "smoke1 yaml must set loc_ref_command_axes_3d=True for this test"
    )
    assert hasattr(a, "_services_by_bin"), (
        "V6EvalAdapter must expose _services_by_bin under 3D mode"
    )
    assert hasattr(a, "_cmd_keys"), (
        "V6EvalAdapter must expose _cmd_keys (n_bins, 3) under 3D mode"
    )
    assert len(a._services_by_bin) >= 2, (
        f"expected multiple per-bin services under smoke1's 3D library; "
        f"got {len(a._services_by_bin)}"
    )
    assert a._cmd_keys.shape[1] == 3
    # vy_grid is non-trivial in smoke1 — at least one bin must have
    # non-zero vy.
    assert np.any(np.abs(a._cmd_keys[:, 1]) > 1e-6), (
        "smoke1 3D library should include lateral bins (vy != 0); "
        f"got cmd_keys vy column = {a._cmd_keys[:, 1].tolist()}"
    )


def test_v6_adapter_3d_mode_selects_different_bin_for_lateral_cmd(
    smoke1_setup,
) -> None:
    """Reviewer bug 1: under loc_ref_command_axes_3d=True, compute_obs
    must drive the reference window from the 3-axis cmd.  A pure-vx cmd
    and a (vx, vy, 0) cmd MUST select DIFFERENT bins (and therefore
    different obs slices that read from ``win``)."""
    s = smoke1_setup
    a = s["adapter"]

    cmd_fwd = np.array([0.20, 0.0, 0.0], dtype=np.float32)
    cmd_lat = np.array([0.20, 0.10, 0.0], dtype=np.float32)

    # Pin step_idx and lateral selection at the SAME step so any obs
    # delta proves the per-bin selection differs.
    bin_fwd = a._select_bin_idx(cmd_fwd)
    bin_lat = a._select_bin_idx(cmd_lat)
    assert bin_fwd != bin_lat, (
        f"3D adapter must select different bins for forward vs lateral "
        f"cmd; got bin_fwd={bin_fwd}, bin_lat={bin_lat} from cmd_keys="
        f"{a._cmd_keys.tolist()}"
    )

    # And the obs slices that read from ``win`` must reflect the
    # difference end-to-end (catch a regression where the bin selector
    # is correct but ``compute_obs`` still calls the old single service).
    layout = s["policy_spec"].observation.layout
    cursor = 0
    name_to_offset: dict[str, tuple[int, int]] = {}
    for f in layout:
        name_to_offset[f.name] = (cursor, cursor + f.size)
        cursor += f.size

    # v8 layout only includes loc_ref_phase_sin_cos from the reference
    # window (the rich q_ref / foot pose slots are v6-only).  Advance a
    # few steps so phase has time to diverge across bins, then compare.
    a.reset()
    obs_fwd_seq = [a.compute_obs(s["mj_data"], velocity_cmd=cmd_fwd)]
    for _ in range(50):
        a._state.step_idx += 1
        obs_fwd_seq.append(
            a.compute_obs(s["mj_data"], velocity_cmd=cmd_fwd)
        )

    a.reset()
    obs_lat_seq = [a.compute_obs(s["mj_data"], velocity_cmd=cmd_lat)]
    for _ in range(50):
        a._state.step_idx += 1
        obs_lat_seq.append(
            a.compute_obs(s["mj_data"], velocity_cmd=cmd_lat)
        )

    # Use the per-bin services to confirm that the (vx=0.20, vy=0.10, 0)
    # trajectory has different phase than (vx=0.20, vy=0, 0) at some
    # step — if so, the obs slice must reflect that delta when
    # compute_obs is actually routing through the per-bin selector.
    win_fwd_50 = a._services_by_bin[bin_fwd].lookup_np(50)
    win_lat_50 = a._services_by_bin[bin_lat].lookup_np(50)
    bins_differ_on_phase = not (
        abs(win_fwd_50.phase_sin - win_lat_50.phase_sin) < 1e-6
        and abs(win_fwd_50.phase_cos - win_lat_50.phase_cos) < 1e-6
    )
    phase_start, phase_end = name_to_offset["loc_ref_phase_sin_cos"]
    if bins_differ_on_phase:
        phase_fwd = obs_fwd_seq[-1][phase_start:phase_end]
        phase_lat = obs_lat_seq[-1][phase_start:phase_end]
        assert not np.allclose(phase_fwd, phase_lat, atol=1e-6), (
            "loc_ref_phase_sin_cos slice must differ between fwd and "
            "lateral cmd under 3D mode (the per-bin trajectories disagree "
            "on phase at step 50); if equal, compute_obs is reading from "
            "the single forward-bin service."
        )
    else:
        # Phase tracks the gait clock — for two bins with the same
        # cycle_time the phase at any step is identical.  In that case
        # the obs phase slice IS expected to match; the bin-index
        # difference is enough to confirm the per-bin selector wired
        # through correctly (already pinned above).
        pass


def test_v6_adapter_3d_mode_matches_env_bin_selection(smoke1_setup) -> None:
    """End-to-end: for several (vx, vy, wz) cmd probes, the adapter's
    selected bin (and resulting q_ref) must match what
    ``WildRobotEnv._lookup_offline_window`` picks under the same cmd.
    """
    import jax
    import jax.numpy as jnp
    from training.envs.wildrobot_env import WildRobotEnv

    s = smoke1_setup
    cfg = s["cfg"]
    a = s["adapter"]

    env = WildRobotEnv(config=cfg)

    cmds = [
        np.array([0.20, 0.0, 0.0], dtype=np.float32),
        np.array([0.20, 0.10, 0.0], dtype=np.float32),
        np.array([0.20, -0.10, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.20], dtype=np.float32),
        np.array([0.0, 0.0, -0.20], dtype=np.float32),
        # Reviewer 2026-05-24 follow-up #2: probe vx != offline_vx so
        # a single-bin adapter library (the bug being fixed) would
        # snap to (0.20, *, *) while env would correctly use (0.18, *, *)
        # / (0.26, *, *).  Locks the full env vx_grid parity.
        np.array([0.18, 0.05, 0.0], dtype=np.float32),
        np.array([0.18, -0.05, 0.0], dtype=np.float32),
        np.array([0.26, 0.10, 0.0], dtype=np.float32),
        np.array([0.26, -0.10, 0.0], dtype=np.float32),
        np.array([0.22, 0.05, 0.0], dtype=np.float32),
    ]
    for cmd in cmds:
        adapter_bin_idx = a._select_bin_idx(cmd)
        adapter_key = a._cmd_keys[adapter_bin_idx]

        env_win = env._lookup_offline_window(
            jnp.asarray(0, dtype=jnp.int32),
            velocity_cmd=jnp.asarray(cmd, dtype=jnp.float32),
        )
        env_key = np.array(
            [
                float(env_win["selected_vx"]),
                float(env_win["selected_vy"]),
                float(env_win["selected_yaw_rate"]),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            adapter_key,
            env_key,
            atol=1e-5,
            err_msg=(
                f"Adapter selected bin {adapter_key.tolist()} for cmd "
                f"{cmd.tolist()} but env selected {env_key.tolist()}"
            ),
        )

        # q_ref at step 0 must match (within solver float tolerance).
        adapter_qref = np.asarray(
            a._services_by_bin[adapter_bin_idx].lookup_np(0).q_ref,
            dtype=np.float32,
        )
        env_qref = np.asarray(env_win["q_ref"], dtype=np.float32)
        np.testing.assert_allclose(
            adapter_qref, env_qref, atol=1e-5,
            err_msg=(
                f"q_ref at step 0 disagrees for cmd {cmd.tolist()}: "
                f"max |Δ| = {float(np.abs(adapter_qref - env_qref).max()):.6e}"
            ),
        )


def test_v6_adapter_legacy_mode_keeps_single_service(smoke1_setup) -> None:
    """Back-compat: smoke14 (loc_ref_command_axes_3d=False) MUST still
    have ``self._service`` as the canonical lookup; ``_services_by_bin``
    if exposed must reduce to a single bin so callers keep their fast
    path.  Skipped if smoke14 yaml is missing."""
    if not SMOKE14_YAML.exists():
        pytest.skip(f"{SMOKE14_YAML.name} not found")
    s14 = _build_adapter(SMOKE14_YAML)
    a = s14["adapter"]
    cfg = s14["cfg"]
    assert bool(getattr(cfg.env, "loc_ref_command_axes_3d", False)) is False
    # Legacy mode: _service is still the source of truth.
    assert a._service is not None
    # If services_by_bin is exposed for uniformity, it should hold exactly
    # one entry pointing at _service so legacy callers don't multiply work.
    services = getattr(a, "_services_by_bin", None)
    if services is not None:
        assert len(services) == 1
        assert services[0] is a._service
