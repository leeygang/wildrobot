"""Native-MuJoCo eval adapter for the active v0.20.1 locomotion actor layouts.

The active v0.20.1 PPO recipes use one of two actor layouts:

  - ``wr_obs_v6_offline_ref_history``: full offline-ref actor channels
  - ``wr_obs_v7_phase_proprio``: de-hybridized actor with only phase +
    proprio history on top of the shared proprio base

Both layouts still share the same native-MuJoCo runtime requirements:

  - residual action composition around the configured base pose
  - optional 1-step action delay + low-pass filter
  - offline gait phase / reference service for step indexing
  - a flat past-proprio stack of ``PROPRIO_HISTORY_FRAMES`` frames of
    ``(gyro, foot_switches, joint_pos_norm, joint_vel_norm, applied_action)``

and consumes a residual action with optional 1-step delay + filter:

    filtered  = lowpass(prev_filtered, raw_action)        # alpha=0 ⇒ no-op
    applied   = prev_filtered if delay else filtered      # 1-step delay
    target_q  = clip(base_q(step_idx + 1) + applied * scale, joint_range)

Notice three subtleties that earlier adapter revisions got wrong:

1. ``applied`` (not ``raw_action``) is the value that lands in the obs's
   ``prev_action`` slot AND in the rolled proprio bundle's prev_action
   slot.  The bundle's other channels come from POST-physics signals.
2. The control reference advances BEFORE the lookup: at env step N the
   target uses ``q_ref[step_idx + 1]``, not ``q_ref[step_idx]``.
3. The proprio history obs uses the PRE-roll buffer (zeros at step 1),
   and the POST-physics bundle is rolled in *after* the obs is built so
   it shows up at step 2.

This adapter mirrors env's per-step semantics exactly:

  - ``training/envs/wildrobot_env.py:_compute_proprio_bundle``
  - ``training/envs/wildrobot_env.py:_v4_compat_channels_from_window``
  - ``training/envs/wildrobot_env.py:_compose_target_q_from_residual``
  - ``training/envs/wildrobot_env.py:_to_mj_ctrl``
  - ``training/envs/wildrobot_env.py:_roll_proprio_history``
  - ``training/envs/wildrobot_env.py.step`` lines 1682-1727 (delay/filter)

so visualize_policy.py can run v6/v7 checkpoints without re-implementing
each piece inline.

Lifecycle (per visualizer iteration; mirrors env.step end-to-end):

    adapter.reset()                           # once at episode start
    obs = adapter.compute_obs(mj_data, velocity_cmd)  # uses step_idx
    action_raw = policy(obs)
    adapter.apply_action(mj_data, action_raw) # advances step_idx,
                                              # writes ctrl
    step_physics(mj_model, mj_data, n_substeps)
    adapter.post_physics(mj_data)             # builds bundle from POST
                                              # signals, rolls history
    # next iteration: compute_obs reads the NEW step_idx + rolled history

Single source of truth for the schema is the env; if the contract changes,
update both env and adapter together (``test_v6_eval_adapter.py`` is the
guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.action import postprocess_action
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PROPRIO_HISTORY_FRAMES, PolicySpec
from training.utils.ctrl_order import CtrlOrderMapper

V6_LAYOUT_ID = "wr_obs_v6_offline_ref_history"
V7_LAYOUT_ID = "wr_obs_v7_phase_proprio"
# v0.21.0 P11: v8 is a strict superset of v7 — the adapter handles v8 by
# routing (vy, wz) through ``velocity_cmd_lateral_yaw`` at the obs build
# site (compute_obs below).  Without v8 in the allow-list a v8-trained
# checkpoint cannot be evaluated or visualized off-policy.
V8_LAYOUT_ID = "wr_obs_v8_cmd3d"
ADAPTER_LAYOUT_IDS = frozenset({V6_LAYOUT_ID, V7_LAYOUT_ID, V8_LAYOUT_ID})


@dataclass
class V6AdapterState:
    """Per-episode mutable state held by V6EvalAdapter.

    proprio_history vs pending_history (Strategy A — see compute_obs /
    post_physics docstrings):

    - ``proprio_history`` is what the CURRENT compute_obs reads.  Mirror
      of env's ``wr.proprio_history`` at the start of env.step (i.e.,
      pre-roll for the obs being built).
    - ``pending_history`` is the post-roll buffer that the previous
      post_physics produced.  It only becomes the new ``proprio_history``
      AFTER the current compute_obs returns.  Mirror of env's
      ``new_wr.proprio_history`` (post-roll), which env stores in the
      returned state.info but doesn't put into the returned state.obs.

    This lag is load-bearing: PPO at iteration N+1 sees ``state_N.obs``
    which uses ``state_{N-1}.info.wr.proprio_history`` (= pre-roll at
    step N).  The bundle rolled at the end of step N first becomes
    visible in obs at step N+2.
    """
    proprio_history: np.ndarray  # (PROPRIO_HISTORY_FRAMES, bundle_size); used by compute_obs
    pending_history: np.ndarray  # (PROPRIO_HISTORY_FRAMES, bundle_size); promoted at end of compute_obs
    loc_ref_history: Optional[np.ndarray]  # (4,) rolled phase_sin_cos; None at reset
    step_idx: int  # offline-trajectory frame for the CURRENT obs (next ctrl uses step_idx + 1)
    pending_action: np.ndarray  # (action_dim,) last-filtered value; also "applied" when delay enabled
    last_applied_action: np.ndarray  # (action_dim,) action that was applied to physics this step


class V6EvalAdapter:
    """Native-MuJoCo eval adapter for the active locomotion actor layouts.

    See module docstring for the lifecycle and parity rationale.
    """

    def __init__(
        self,
        *,
        training_cfg,
        mj_model: mujoco.MjModel,
        policy_spec: PolicySpec,
        signals_adapter,
        action_dim: int,
    ) -> None:
        if policy_spec.observation.layout_id not in ADAPTER_LAYOUT_IDS:
            raise ValueError(
                "V6EvalAdapter expects layout_id in "
                f"{sorted(ADAPTER_LAYOUT_IDS)!r}, got "
                f"{policy_spec.observation.layout_id!r}"
            )
        self._cfg = training_cfg
        self._mj_model = mj_model
        self._policy_spec = policy_spec
        self._signals_adapter = signals_adapter
        self._action_dim = int(action_dim)

        self._bundle_size = 3 + 4 + 3 * self._action_dim  # gyro + sw + 3*N

        # Action delay flag: env supports {0, 1}; v0.20.1 smoke uses 1.
        delay_steps = int(getattr(self._cfg.env, "action_delay_steps", 0))
        if delay_steps not in (0, 1):
            raise ValueError(
                f"V6EvalAdapter supports action_delay_steps in {{0, 1}}, "
                f"got {delay_steps}"
            )
        self._action_delay_enabled = (delay_steps == 1)

        # smoke8/smoke9c — residual base selector.  Mirrors WildRobotEnv.__init__
        # so native-MuJoCo eval composes target_q with the same base the
        # training env used (q_ref(t) for smoke7, home for smoke8,
        # ref_init for smoke9c).
        # Without this branch the adapter silently falls back to q_ref
        # base regardless of the cfg, and a home/ref_init-base checkpoint gets eval'd
        # under the wrong control contract.
        self._residual_base_mode = str(
            getattr(self._cfg.env, "loc_ref_residual_base", "q_ref")
        ).lower()
        if self._residual_base_mode not in ("q_ref", "home", "ref_init"):
            raise ValueError(
                f"V6EvalAdapter: env.loc_ref_residual_base must be "
                f"'q_ref', 'home', or 'ref_init'; got {self._residual_base_mode!r}"
            )
        self._reset_base_mode = str(
            getattr(self._cfg.env, "loc_ref_reset_base", "home")
        ).lower()
        if self._reset_base_mode not in ("home", "ref_init"):
            raise ValueError(
                f"V6EvalAdapter: env.loc_ref_reset_base must be "
                f"'home' or 'ref_init'; got {self._reset_base_mode!r}"
            )

        self._init_offline_service()
        self._init_residual_scale()
        self._init_joint_ranges()
        self._init_ref_init_q_rad()
        self._init_home_q_rad()
        self._init_ctrl_mapper()
        self._init_reset_perturbation()
        self._init_dr_joint_offsets()
        self.reset()

    # ------------------------------------------------------------------ init

    def _init_offline_service(self) -> None:
        """Build the offline reference service(s) for this episode.

        Three modes, mirroring ``WildRobotEnv._init_offline_reference``
        (env.py lines 800-1080):

          - **Legacy / 1D mode** (``loc_ref_command_axes_3d=False`` or
            unset): single forward-vx trajectory; ``_service`` is the
            canonical lookup, ``_services_by_bin`` collapses to a
            single-entry list pointing at ``_service`` for adapter-side
            uniformity, ``_cmd_keys`` is ``[[offline_vx, 0, 0]]``.

          - **3D mode** (``loc_ref_command_axes_3d=True``; v0.21.0 P5
            opt-in, exercised by smoke1): build the full TB-split 3D
            library via ``build_library_for_3d_values`` over the YAML
            vy / yaw_rate grids.  Per-bin services are stored in
            ``_services_by_bin``; ``_cmd_keys`` is the (n_bins, 3)
            ``(vx, vy, wz)`` matrix driven off the stored trajectories'
            command_key (matches env P5 plumbing).

          - **On-disk library** (``loc_ref_offline_library_path``): load
            from disk; iterate stored trajectories under 3D mode, or
            single-vx lookup under 1D.

        ``_service`` is selected by L2 distance to the canonical
        straight-walk anchor ``(offline_vx, 0, 0)`` so ``_ref_init_q_rad``
        / ``_init_ref_init_q_rad`` reads the same anchor frame the env
        uses (env P5-fix at line 1039-1050).

        Degenerate static bins (``(0, 0, 0)`` reduces to a 1-step
        trajectory under the ZMP planner) are handled by
        ``RuntimeReferenceService.lookup_np``'s built-in
        ``np.clip(step_idx, 0, n_steps - 1)`` -- the numpy lookup just
        re-reads the same frame for every step.  Env tiles them via
        ``jp.tile`` for the stacked-arrays contract; the numpy adapter
        doesn't need that because each bin keeps its own service.
        """
        from control.references.runtime_reference_service import (
            RuntimeReferenceService,
        )
        offline_path = getattr(self._cfg.env, "loc_ref_offline_library_path", None)
        offline_vx = float(getattr(self._cfg.env, "loc_ref_offline_command_vx", 0.20))
        axes_3d = bool(
            getattr(self._cfg.env, "loc_ref_command_axes_3d", False)
        )

        if axes_3d:
            # 3D library opt-in.  Build (or load) the full TB-split
            # library and stand up one service per bin.
            if offline_path:
                from control.references.reference_library import ReferenceLibrary
                lib = ReferenceLibrary.load(offline_path)
                stored_trajectories = list(lib._entries.values())
            else:
                from control.zmp.zmp_walk import ZMPWalkGenerator
                vy_grid_cfg = list(
                    getattr(
                        self._cfg.env, "loc_ref_offline_command_vy_grid", ()
                    )
                ) or [0.0]
                wz_grid_cfg = list(
                    getattr(
                        self._cfg.env,
                        "loc_ref_offline_command_yaw_rate_grid",
                        (),
                    )
                ) or [0.0]
                # Mirror env _init_offline_service vx_grid derivation
                # (training/envs/wildrobot_env.py:835-855): arange over
                # [min_velocity, max_velocity] at
                # loc_ref_command_grid_interval, unioned with
                # loc_ref_offline_command_vx so the eval-cmd anchor
                # snaps to its own bin exactly.  Previously the adapter
                # built [offline_vx] only, so any 3D cmd whose vx
                # axis was NOT offline_vx selected the wrong bin off-JAX
                # (reviewer 2026-05-24 follow-up #2).
                min_vx = float(self._cfg.env.min_velocity)
                max_vx = float(self._cfg.env.max_velocity)
                interval = float(
                    getattr(
                        self._cfg.env, "loc_ref_command_grid_interval", 0.05
                    )
                )
                if interval <= 0.0:
                    raise ValueError(
                        f"loc_ref_command_grid_interval must be positive; "
                        f"got {interval!r}"
                    )
                arange_vals = np.arange(
                    min_vx, max_vx + 1e-6, interval, dtype=np.float64
                )
                vx_grid = sorted(
                    {round(float(v), 6) for v in arange_vals}
                    | {round(offline_vx, 6)}
                )
                lib = ZMPWalkGenerator().build_library_for_3d_values(
                    vx_values=vx_grid,
                    vy_values=vy_grid_cfg,
                    yaw_rate_values=wz_grid_cfg,
                )
                stored_trajectories = list(lib._entries.values())

            services: list[RuntimeReferenceService] = []
            cmd_keys: list[tuple[float, float, float]] = []
            for traj in stored_trajectories:
                services.append(RuntimeReferenceService(traj, n_anchor=2))
                cmd_keys.append(
                    (
                        float(traj.command_vx),
                        float(traj.command_vy),
                        float(traj.command_yaw_rate),
                    )
                )
            self._services_by_bin = services
            self._cmd_keys = np.asarray(cmd_keys, dtype=np.float32)

            # Pick the canonical straight-walk anchor as the primary
            # service (drives _ref_init_q_rad and any single-service
            # callsite).  L2 distance over (vx, vy, wz) — matches env
            # line 1039-1050.
            anchor = np.array([offline_vx, 0.0, 0.0], dtype=np.float32)
            diffs = self._cmd_keys - anchor[np.newaxis, :]
            primary_idx = int(np.argmin(np.linalg.norm(diffs, axis=-1)))
            self._service = services[primary_idx]
            # Use the MAX n_steps across bins as the canonical horizon
            # so static (0, 0, 0) bin (n_steps=1) doesn't shorten the
            # advertised episode length.  Per-bin lookup_np clamps the
            # idx internally so the short bin re-reads its single frame.
            self._n_steps = max(int(s.n_steps) for s in services)
            return

        # Legacy / 1D path: single forward-vx service.
        if offline_path:
            from control.references.reference_library import ReferenceLibrary
            lib = ReferenceLibrary.load(offline_path)
        else:
            from control.zmp.zmp_walk import ZMPWalkGenerator
            lib = ZMPWalkGenerator().build_library_for_vx_values([offline_vx])
        traj = lib.lookup(offline_vx)
        self._service = RuntimeReferenceService(traj, n_anchor=2)
        self._services_by_bin = [self._service]
        self._cmd_keys = np.array(
            [[offline_vx, 0.0, 0.0]], dtype=np.float32
        )
        self._n_steps = int(self._service.n_steps)

    def _select_bin_idx(self, velocity_cmd: np.ndarray) -> int:
        """Return the per-bin index nearest to ``velocity_cmd``.

        Mirrors ``WildRobotEnv._lookup_offline_window`` 3D path
        (env line 1585-1588): L2-argmin over the (vx, vy, wz) cmd_keys.
        Heading-frame correction is NOT applied here -- the env default
        is also without correction unless both ``path_rot_wxyz`` and
        ``heading_rot_wxyz`` are supplied, and the adapter is the
        eager-eval / visualizer path where the path-frame anchor isn't
        available at the obs site.

        Under 1D / legacy mode this collapses to index 0 because
        ``_cmd_keys`` is a single-row array.
        """
        cmd = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
        if cmd.size == 1:
            cmd = np.array([float(cmd[0]), 0.0, 0.0], dtype=np.float32)
        elif cmd.size != 3:
            raise ValueError(
                "V6EvalAdapter._select_bin_idx: velocity_cmd must be "
                f"scalar or length-3; got size {cmd.size}"
            )
        diffs = self._cmd_keys - cmd[np.newaxis, :]
        return int(np.argmin(np.linalg.norm(diffs, axis=-1)))

    def _init_residual_scale(self) -> None:
        per_joint = dict(
            getattr(self._cfg.env, "loc_ref_residual_scale_per_joint", {}) or {}
        )
        scalar = float(getattr(self._cfg.env, "loc_ref_residual_scale", 0.18))
        self._scale_per_joint = np.array(
            [
                float(per_joint.get(name, scalar))
                for name in self._policy_spec.robot.actuator_names
            ],
            dtype=np.float32,
        )

    def _init_joint_ranges(self) -> None:
        actuator_names = list(self._policy_spec.robot.actuator_names)
        joint_ranges: list[tuple[float, float]] = []
        actuator_qpos_addrs: list[int] = []
        for name in actuator_names:
            act_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = int(self._mj_model.actuator_trnid[act_id][0])
            jrange = self._mj_model.jnt_range[joint_id]
            joint_ranges.append((float(jrange[0]), float(jrange[1])))
            actuator_qpos_addrs.append(int(self._mj_model.jnt_qposadr[joint_id]))
        self._joint_min = np.array([r[0] for r in joint_ranges], dtype=np.float32)
        self._joint_max = np.array([r[1] for r in joint_ranges], dtype=np.float32)
        self._actuator_qpos_addrs = np.asarray(actuator_qpos_addrs, dtype=np.int32)

    def _init_home_q_rad(self) -> None:
        """Cache the home-pose ctrl values, pre-clipped to joint range.

        Mirrors WildRobotEnv._home_q_rad construction.  The env builds
        ``_home_q_rad = clip(_cal.get_ctrl_for_default_pose(), j_min,
        j_max)``; the equivalent for the adapter (which doesn't have
        the calibration layer) is the policy spec's ``home_ctrl_rad``,
        which the env populated from MJCF's home keyframe via
        ``get_home_ctrl_from_mj_model + clamp_home_ctrl``.  Both sources
        derive from MJCF key_qpos[0]; pre-clipping makes them
        byte-identical at the zero-action invariant (test
        ``test_smoke8_eval_adapter_zero_action_matches_env_home``).
        """
        if self._policy_spec.robot.home_ctrl_rad is None:
            qpos0 = (
                self._mj_model.key_qpos[0]
                if self._mj_model.nkey > 0
                else self._mj_model.qpos0
            )
            self._home_q_rad = np.clip(
                np.asarray(qpos0[self._actuator_qpos_addrs], dtype=np.float32),
                self._joint_min,
                self._joint_max,
            ).astype(np.float32)
            return
        home_rad = np.asarray(
            self._policy_spec.robot.home_ctrl_rad, dtype=np.float32
        )
        self._home_q_rad = np.clip(
            home_rad, self._joint_min, self._joint_max
        ).astype(np.float32)

    def _init_ref_init_q_rad(self) -> None:
        win0 = self._service.lookup_np(0)
        self._ref_init_q_rad = np.clip(
            np.asarray(win0.q_ref, dtype=np.float32),
            self._joint_min,
            self._joint_max,
        ).astype(np.float32)

    def _init_ctrl_mapper(self) -> None:
        self._ctrl_mapper = CtrlOrderMapper(
            self._mj_model, list(self._policy_spec.robot.actuator_names)
        )

    def _init_dr_joint_offsets(self) -> None:
        """Cache the DR joint-offset half-range so the adapter reset can
        mirror training's per-episode per-joint calibration error
        (env line 1918: ``self._default_joint_qpos + joint_noise +
        dr_params["joint_offsets"]``).

        The training env samples ``joint_offsets`` uniformly in
        ``[-env.domain_rand_joint_offset_rad, +env.domain_rand_joint_offset_rad]``
        (training/envs/domain_randomize.py:59-61) only when
        ``env.domain_randomization_enabled`` is True; we cache the same
        gate so the adapter is a no-op when the env's DR loop is off.
        """
        self._dr_enabled = bool(
            getattr(self._cfg.env, "domain_randomization_enabled", False)
        )
        self._dr_joint_offset_rad = float(
            getattr(self._cfg.env, "domain_rand_joint_offset_rad", 0.0)
        )

    def _apply_dr_joint_offsets_np(
        self, qpos: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Numpy parity copy of the env's per-episode DR joint-offset
        application (env line 1918).

        Samples uniform per-joint offsets in
        ``[-dr_joint_offset_rad, +dr_joint_offset_rad]`` and adds them
        into the actuator-joint qpos slots, then re-clips against the
        joint range to match env's clipping.  Statistical distribution
        matches training; per-seed bit equality is not asserted
        (JAX vs numpy PRNGs differ).
        """
        if not self._dr_enabled or self._dr_joint_offset_rad <= 0.0:
            return qpos
        qpos = qpos.copy()
        offsets = rng.uniform(
            -self._dr_joint_offset_rad,
            +self._dr_joint_offset_rad,
            size=self._action_dim,
        ).astype(np.float32)
        joint_q = qpos[self._actuator_qpos_addrs].astype(np.float32) + offsets
        joint_q = np.clip(joint_q, self._joint_min, self._joint_max)
        qpos[self._actuator_qpos_addrs] = joint_q.astype(qpos.dtype)
        return qpos

    def _init_reset_perturbation(self) -> None:
        """Cache leg-pitch metadata + reset perturbation ranges for the
        numpy parity copy of ``WildRobotEnv._apply_reset_perturbation``.

        Without this the visualizer / native-MuJoCo eval reset would
        only see joint noise around home, while training reset
        additionally rotates the torso by uniform roll/pitch in
        ``env.reset_torso_{roll,pitch}_range`` and partitions the
        pitch across hip/knee/ankle.  That parity gap silently makes
        visual eval easier than training, especially under
        ``loc_ref_reset_base=home``.
        """
        leg_pitch_joint_names = (
            "left_hip_pitch",
            "left_knee_pitch",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee_pitch",
            "right_ankle_pitch",
        )
        addrs: list[int] = []
        mins: list[float] = []
        maxs: list[float] = []
        for jname in leg_pitch_joint_names:
            jid = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            if jid < 0:
                # Leg-pitch joint missing.  Disable the perturbation
                # silently — the env raises here, but the adapter is
                # used in some narrow contexts (legacy robot variants)
                # where the perturbation is not needed.  Mark the
                # ranges as zero so _apply_reset_perturbation_np is a
                # no-op.
                self._leg_pitch_qpos_addrs = np.zeros((0,), dtype=np.int32)
                self._leg_pitch_joint_mins = np.zeros((0,), dtype=np.float32)
                self._leg_pitch_joint_maxs = np.zeros((0,), dtype=np.float32)
                self._reset_torso_roll_range = (0.0, 0.0)
                self._reset_torso_pitch_range = (0.0, 0.0)
                return
            addrs.append(int(self._mj_model.jnt_qposadr[jid]))
            jr = self._mj_model.jnt_range[jid]
            mins.append(float(jr[0]))
            maxs.append(float(jr[1]))
        self._leg_pitch_qpos_addrs = np.asarray(addrs, dtype=np.int32)
        self._leg_pitch_joint_mins = np.asarray(mins, dtype=np.float32)
        self._leg_pitch_joint_maxs = np.asarray(maxs, dtype=np.float32)
        # Mirror-symmetric TB sign pattern (env line 510).
        self._leg_pitch_joint_signs = np.array(
            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32
        )

        roll_range = list(
            getattr(self._cfg.env, "reset_torso_roll_range", [0.0, 0.0])
        )
        pitch_range = list(
            getattr(self._cfg.env, "reset_torso_pitch_range", [0.0, 0.0])
        )
        self._reset_torso_roll_range = (float(roll_range[0]), float(roll_range[1]))
        self._reset_torso_pitch_range = (float(pitch_range[0]), float(pitch_range[1]))

    def _reset_perturbation_enabled(self) -> bool:
        return (
            self._reset_torso_roll_range[1] > self._reset_torso_roll_range[0]
            or self._reset_torso_pitch_range[1] > self._reset_torso_pitch_range[0]
        )

    def _apply_reset_perturbation_np(
        self, qpos: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Numpy parity copy of ``WildRobotEnv._apply_reset_perturbation``.

        Samples uniform torso roll/pitch in the configured ranges,
        partitions ``|pitch|`` into hip/knee/ankle nonnegative parts
        summing to it, applies signed deltas via the mirror-symmetric
        leg-pitch sign pattern, and composes ``R_xyz(roll, pitch, 0)``
        onto the root quat.  Statistical distribution matches the
        training env reset; per-seed bit-equality is not asserted
        because JAX and numpy PRNGs differ.
        """
        if self._leg_pitch_qpos_addrs.size == 0:
            return qpos
        roll_lo, roll_hi = self._reset_torso_roll_range
        pitch_lo, pitch_hi = self._reset_torso_pitch_range

        torso_roll = float(rng.uniform(roll_lo, roll_hi))
        torso_pitch = float(rng.uniform(pitch_lo, pitch_hi))

        # Partition |torso_pitch| across hip / knee / ankle.
        pitch_abs = abs(torso_pitch)
        hip_delta = float(rng.uniform(0.0, pitch_abs)) if pitch_abs > 0 else 0.0
        knee_max = max(pitch_abs - hip_delta, 0.0)
        knee_delta = float(rng.uniform(0.0, knee_max)) if knee_max > 0 else 0.0
        ankle_delta = max(pitch_abs - hip_delta - knee_delta, 0.0)

        leg_pitch_mag = np.array(
            [hip_delta, knee_delta, ankle_delta,
             hip_delta, knee_delta, ankle_delta],
            dtype=np.float32,
        )
        sign_pitch = 1.0 if torso_pitch >= 0.0 else -1.0
        leg_pitch_signed = leg_pitch_mag * self._leg_pitch_joint_signs * sign_pitch

        qpos = qpos.copy()
        leg_pitch_curr = qpos[self._leg_pitch_qpos_addrs]
        leg_pitch_new = np.clip(
            leg_pitch_curr + leg_pitch_signed,
            self._leg_pitch_joint_mins,
            self._leg_pitch_joint_maxs,
        )
        qpos[self._leg_pitch_qpos_addrs] = leg_pitch_new.astype(qpos.dtype)

        # Compose R_xyz(roll, pitch, 0) onto the existing root quat.
        # MJCF stores root qpos[3:7] as [w, x, y, z].
        wx, ix, iy, iz = (
            float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        )
        # XYZ extrinsic Euler -> quaternion (matches env helper).
        cr, sr = np.cos(torso_roll * 0.5), np.sin(torso_roll * 0.5)
        cp, sp = np.cos(torso_pitch * 0.5), np.sin(torso_pitch * 0.5)
        # R_xyz(roll, pitch, 0): q = q_x * q_y * q_z; with z=0 only q_x * q_y.
        # q_x = (sr, 0, 0, cr); q_y = (0, sp, 0, cp); product (xyzw):
        delta_x = sr * cp
        delta_y = cr * sp
        delta_z = -sr * sp
        delta_w = cr * cp
        # quat_mul(delta_xyzw, root_xyzw):
        a, b, c, d = delta_x, delta_y, delta_z, delta_w
        e, f, g, h = ix, iy, iz, wx
        new_x = d * e + a * h + b * g - c * f
        new_y = d * f - a * g + b * h + c * e
        new_z = d * g + a * f - b * e + c * h
        new_w = d * h - a * e - b * f - c * g
        # Normalize.
        norm = float(np.sqrt(new_x * new_x + new_y * new_y + new_z * new_z + new_w * new_w))
        if norm > 1e-12:
            new_x, new_y, new_z, new_w = (
                new_x / norm, new_y / norm, new_z / norm, new_w / norm
            )
        qpos[3] = np.float32(new_w)
        qpos[4] = np.float32(new_x)
        qpos[5] = np.float32(new_y)
        qpos[6] = np.float32(new_z)
        return qpos

    # ------------------------------------------------------------ public API

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def step_idx(self) -> int:
        return int(self._state.step_idx)

    def _reset_joint_qpos_policy_order(
        self, *, apply_noise: bool, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        """Return actuator-joint qpos in policy order for reset.

        Mirrors env.reset:
        - loc_ref_reset_base == "ref_init": exact ref_init_q, no reset noise.
        - loc_ref_reset_base == "home": home pose with optional +/-0.05 rad
          reset noise (visualizer parity behavior).
        """
        if self._reset_base_mode == "ref_init":
            return self._ref_init_q_rad.copy()

        joint_q = self._home_q_rad.copy()
        if apply_noise:
            if rng is None:
                raise ValueError("rng is required when apply_noise=True")
            noise = rng.uniform(-0.05, 0.05, size=self._action_dim).astype(np.float32)
            joint_q = np.clip(joint_q + noise, self._joint_min, self._joint_max)
        return joint_q.astype(np.float32)

    def _ctrl_init_policy_order(self) -> np.ndarray:
        if self._residual_base_mode == "home":
            return self._home_q_rad.copy()
        # q_ref and ref_init both use the frame-0 offline reference.
        return self._ref_init_q_rad.copy()

    def reset_native_mj_state(
        self,
        mj_data: mujoco.MjData,
        *,
        apply_noise: bool,
        rng: Optional[np.random.Generator],
        perturb_pose: bool = True,
        apply_dr: bool = True,
    ) -> None:
        """Reset native MuJoCo state with env-aligned reset semantics:
        per-joint reset noise, per-episode DR joint offsets, and the
        TB-style torso pose perturbation.

        ``perturb_pose`` mirrors the training env's
        ``WildRobotEnv.reset(rng, perturb_pose=...)`` argument:
          - True (default; visualizer / native-MuJoCo eval that wants
            to mirror training reset): applies torso roll/pitch
            perturbation when ``apply_noise`` is also True and the
            configured ranges are non-degenerate.
          - False: skips the perturbation regardless of range/config.
            Mirrors ``env.reset_for_eval`` which calls
            ``env.reset(rng, perturb_pose=False)`` so eval-mode
            rollouts stay deterministic on the torso side even when
            joint noise is otherwise applied.

        ``apply_dr`` controls the per-episode DR joint-offset
        application (env line 1918: ``default_joint_qpos + joint_noise
        + dr_params["joint_offsets"]``):
          - True (default; visualizer / native-MuJoCo eval that wants
            to mirror training reset distribution): samples per-joint
            offsets in ``[-domain_rand_joint_offset_rad, +rad]`` when
            ``apply_noise`` is True AND
            ``env.domain_randomization_enabled`` is True.
          - False: skips the DR offset application even if otherwise
            configured.  Use for strict no-DR demo runs.

        The training env reset was previously the *only* place where
        the torso perturbation and the DR offsets fired — the
        visualizer adapter omitted both, which made visual eval easier
        than training under ``loc_ref_reset_base=home`` + non-zero
        torso ranges + DR enabled.  Defaults
        ``perturb_pose=True``/``apply_dr=True`` close both parity gaps.

        Statistical distribution matches training; per-seed bit
        equality is not asserted (JAX vs numpy PRNG differ).
        """
        if self._mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._mj_model, mj_data, 0)
            qpos = self._mj_model.key_qpos[0].copy()
        else:
            mujoco.mj_resetData(self._mj_model, mj_data)
            qpos = self._mj_model.qpos0.copy()

        joint_qpos = self._reset_joint_qpos_policy_order(
            apply_noise=apply_noise,
            rng=rng,
        )
        qpos[self._actuator_qpos_addrs] = joint_qpos

        # Mirror training's per-episode DR joint offsets when caller
        # opted in (apply_dr=True), randomness is on (apply_noise=True),
        # the env's DR loop is enabled, and the reset base is "home"
        # (env line 1909-1918: under ``loc_ref_reset_base=ref_init``
        # the joint_qpos branch is exact ref_init with no joint noise
        # and no DR offsets, so the adapter must not apply them
        # either).  apply_dr=False bypasses the DR for strict no-DR
        # demo runs even if the env config has DR on.  Applied before
        # the torso perturbation so the final leg-pitch qpos sees
        # both (matches training reset ordering at env line 1918 + 1932).
        if (
            apply_dr
            and apply_noise
            and self._dr_enabled
            and self._reset_base_mode != "ref_init"
        ):
            if rng is None:
                raise ValueError("rng is required when apply_noise=True")
            qpos = self._apply_dr_joint_offsets_np(qpos, rng)

        # Mirror training's torso pose perturbation when the caller
        # opted into it (perturb_pose=True), randomness is on
        # (apply_noise=True), and the configured ranges are
        # non-degenerate.  perturb_pose=False matches
        # env.reset_for_eval semantics (torso deterministic even if
        # joint noise is otherwise applied).
        if (
            perturb_pose
            and apply_noise
            and self._reset_perturbation_enabled()
        ):
            if rng is None:
                raise ValueError("rng is required when apply_noise=True")
            qpos = self._apply_reset_perturbation_np(qpos, rng)

        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = 0.0
        if hasattr(mj_data, "xfrc_applied"):
            mj_data.xfrc_applied[:] = 0.0

        ctrl_init = self._ctrl_init_policy_order()
        mj_data.ctrl[:] = self._ctrl_mapper.to_mj_np(ctrl_init)
        mujoco.mj_forward(self._mj_model, mj_data)
        self.reset()

    def reset(self) -> None:
        """Re-zero per-episode state.  Mirrors env's ``_make_initial_state``:
        proprio_history zero-filled, pending_action and last_applied_action
        zero (so iter-1 with action=0 keeps target_q == base_q exactly).

        Both ``proprio_history`` and ``pending_history`` start as zeros so
        the first two compute_obs() calls (= reset's obs and the obs
        returned by env.step iter 1) both see zero history — matching env.
        """
        zeros_history = np.zeros(
            (PROPRIO_HISTORY_FRAMES, self._bundle_size), dtype=np.float32
        )
        self._state = V6AdapterState(
            proprio_history=zeros_history.copy(),
            pending_history=zeros_history.copy(),
            loc_ref_history=None,
            step_idx=0,
            pending_action=np.zeros(self._action_dim, dtype=np.float32),
            last_applied_action=np.zeros(self._action_dim, dtype=np.float32),
        )

    def compute_obs(
        self,
        mj_data: mujoco.MjData,
        velocity_cmd: np.ndarray | float,
    ) -> np.ndarray:
        """Build the v6 obs at the current ``step_idx``.

        Mirrors env's ``_get_obs`` call from ``step()``:
          - obs's prev_action slot = ``last_applied_action`` (the action
            that was applied THIS step, written by ``apply_action`` and
            picked up by the bundle / current proprio prev_action slot).
          - proprio_history slot is the PRE-roll buffer (zeros at iter 1
            because ``post_physics`` of iter 1 hasn't run yet).
          - v4_compat["history"] is computed from the previous
            ``loc_ref_history``, then stored as the new
            ``loc_ref_history`` for the next obs (env's wr.loc_ref_history).

        v0.21.0 P11 follow-up — 3D bin selection: when the env opts into
        ``loc_ref_command_axes_3d=True``, the reference window MUST come
        from the per-bin service nearest the incoming ``(vx, vy, wz)``
        cmd (mirrors env ``_lookup_offline_window`` 3D path).  Under
        legacy / 1D mode ``_select_bin_idx`` returns 0 and the lookup
        falls back to the single forward-vx service.
        """
        signals = self._signals_adapter.read(mj_data)
        # Normalize velocity_cmd EARLY so the bin selector and the obs
        # builder see the same canonical (3,) form.
        cmd_arr = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
        if cmd_arr.size == 1:
            cmd_arr = np.array(
                [float(cmd_arr[0]), 0.0, 0.0], dtype=np.float32
            )
        elif cmd_arr.size != 3:
            raise ValueError(
                "V6EvalAdapter.compute_obs: velocity_cmd must be scalar or "
                f"length-3 (vx, vy, wz); got size {cmd_arr.size}"
            )
        bin_idx = self._select_bin_idx(cmd_arr)
        win = self._services_by_bin[bin_idx].lookup_np(self._state.step_idx)

        phase_sin_cos = np.array(
            [float(win.phase_sin), float(win.phase_cos)], dtype=np.float32
        )
        is_left_stance = int(win.stance_foot_id) == 0
        if is_left_stance:
            next_foothold_xyz = win.right_foot_pos
            swing_pos_xyz = win.right_foot_pos
            swing_vel_xyz = win.right_foot_vel
        else:
            next_foothold_xyz = win.left_foot_pos
            swing_pos_xyz = win.left_foot_pos
            swing_vel_xyz = win.left_foot_vel
        next_foothold_xy = np.asarray(next_foothold_xyz[:2], dtype=np.float32)
        swing_pos = np.asarray(swing_pos_xyz[:3], dtype=np.float32)
        swing_vel = np.asarray(swing_vel_xyz[:3], dtype=np.float32)
        pelvis_targets = np.array(
            [float(win.pelvis_pos[2]), 0.0, 0.0], dtype=np.float32
        )

        # v4_compat history: env's _v4_compat_channels_from_window uses
        # tile(phase_sc, 2) when prev_history is None (reset path) and
        # concat([phase_sc, prev_history[:2]]) otherwise.  After this
        # step's compute_obs, store v4_compat["history"] as the NEW
        # loc_ref_history for the next compute_obs.
        prev_loc_ref_history = self._state.loc_ref_history
        if prev_loc_ref_history is None:
            history_4 = np.tile(phase_sin_cos, 2).astype(np.float32)
        else:
            history_4 = np.concatenate(
                [phase_sin_cos, prev_loc_ref_history[:2]]
            ).astype(np.float32)

        # PolicyState.prev_action for the obs's prev_action slot is the
        # action APPLIED this step (env: PolicyState(prev_action=action)
        # where action == applied_action).  At reset, last_applied_action
        # is zeros.
        policy_state = PolicyState(
            prev_action=np.asarray(self._state.last_applied_action, dtype=np.float32)
        )
        # v0.21.0 P11: ``cmd_arr`` was normalized up-front above so the
        # bin selector and the obs builder share a single canonical (3,)
        # representation.  The numpy obs builder slices index 0 for the
        # shared scalar slot so v1-v7 layouts stay byte-identical; v8
        # additionally consumes the (vy, wz) tail via
        # ``velocity_cmd_lateral_yaw``.
        obs_kwargs: dict[str, object] = dict(
            spec=self._policy_spec,
            state=policy_state,
            signals=signals,
            velocity_cmd=cmd_arr,
            loc_ref_phase_sin_cos=phase_sin_cos,
            loc_ref_stance_foot=np.array(
                [float(win.stance_foot_id)], dtype=np.float32
            ),
            loc_ref_next_foothold=next_foothold_xy,
            loc_ref_swing_pos=swing_pos,
            loc_ref_swing_vel=swing_vel,
            loc_ref_pelvis_targets=pelvis_targets,
            loc_ref_history=history_4,
            loc_ref_q_ref=np.asarray(win.q_ref, dtype=np.float32),
            loc_ref_pelvis_pos=np.asarray(win.pelvis_pos, dtype=np.float32),
            loc_ref_pelvis_vel=np.asarray(win.pelvis_vel, dtype=np.float32),
            loc_ref_left_foot_pos=np.asarray(win.left_foot_pos, dtype=np.float32),
            loc_ref_right_foot_pos=np.asarray(win.right_foot_pos, dtype=np.float32),
            loc_ref_left_foot_vel=np.asarray(win.left_foot_vel, dtype=np.float32),
            loc_ref_right_foot_vel=np.asarray(win.right_foot_vel, dtype=np.float32),
            loc_ref_contact_mask=np.asarray(win.contact_mask, dtype=np.float32),
            proprio_history=self._state.proprio_history.reshape(-1),
        )
        # v0.21.0 P11: route (vy, wz) only when the policy spec is v8 —
        # v6/v7 layouts ignore the kwarg, but passing it unconditionally
        # would break the build_observation_from_components contract.
        if self._policy_spec.observation.layout_id == "wr_obs_v8_cmd3d":
            obs_kwargs["velocity_cmd_lateral_yaw"] = cmd_arr[1:]
        obs = build_observation(**obs_kwargs)

        # Update for next compute_obs.
        self._state.loc_ref_history = history_4
        # Strategy A promote: pending_history was set by the most recent
        # post_physics().  After this compute_obs has READ the current
        # proprio_history (= pre-roll for THIS obs), promote pending into
        # proprio_history so the NEXT compute_obs reads it.  This matches
        # env's flow: state.obs at iter N uses pre-roll history;
        # state.info.wr.proprio_history (post-roll) is what the next step's
        # obs reads as ITS pre-roll.  Idempotent if compute_obs is called
        # twice without an intervening post_physics (pending_history is
        # unchanged in that case).
        self._state.proprio_history = self._state.pending_history
        return obs

    def apply_action(
        self,
        mj_data: mujoco.MjData,
        raw_action: np.ndarray,
    ) -> np.ndarray:
        """Filter, delay, advance step_idx, compose target_q, write ctrl.

        Mirrors env.step lines 1682-1727 exactly:

            policy_state = PolicyState(prev_action=pending_action)
            filtered, _ = postprocess_action(spec, policy_state, raw_action)
            applied = pending_action if delay else filtered
            next_step_idx = clip(step_idx + 1, n - 1)
            q_ref = service.lookup_np(next_step_idx).q_ref
            target_q = clip(q_ref + clip(applied) * scale, joint_range)
            ctrl_mj = ctrl_mapper.to_mj_np(target_q)
            mj_data.ctrl[:] = ctrl_mj
            new pending_action = filtered
            new last_applied_action = applied

        Returns ``applied_action`` for callers that want it; the adapter
        also stores it on ``self._state.last_applied_action`` so the next
        ``compute_obs`` and ``post_physics`` use the correct value.
        """
        raw = np.asarray(raw_action, dtype=np.float32)

        # Filter (alpha=0 ⇒ no-op; lowpass_v1 otherwise).  Filter input
        # is pending_action (= last filtered), NOT last_applied_action,
        # per env line 1713.
        filtered, _ = postprocess_action(
            spec=self._policy_spec,
            state=PolicyState(prev_action=self._state.pending_action),
            action_raw=raw,
        )
        filtered = np.asarray(filtered, dtype=np.float32)

        # 1-step delay: applied = previous filtered (= pending_action).
        # No delay: applied = current filtered.
        if self._action_delay_enabled:
            applied = self._state.pending_action.copy()
        else:
            applied = filtered.copy()

        # Advance step_idx.  Env clamps at n_steps - 1 (lookup beyond is
        # an absorbing-boundary frame).
        self._state.step_idx = min(self._state.step_idx + 1, self._n_steps - 1)

        win = self._service.lookup_np(self._state.step_idx)
        q_ref = np.asarray(win.q_ref, dtype=np.float32)
        clipped = np.clip(applied, -1.0, 1.0)
        residual = clipped * self._scale_per_joint
        if self._residual_base_mode == "home":
            base_q = self._home_q_rad
        elif self._residual_base_mode == "ref_init":
            base_q = self._ref_init_q_rad
        else:
            base_q = q_ref
        target_q = np.clip(
            base_q + residual, self._joint_min, self._joint_max
        ).astype(np.float32)
        ctrl_mj = self._ctrl_mapper.to_mj_np(target_q)
        mj_data.ctrl[:] = np.asarray(ctrl_mj, dtype=np.float32)

        # Save for next iteration.
        self._state.pending_action = filtered
        self._state.last_applied_action = applied
        return applied

    def post_physics(self, mj_data: mujoco.MjData) -> None:
        """Read POST-physics signals, build the per-step bundle, roll into
        ``pending_history`` (NOT into ``proprio_history``).

        Mirrors env.step lines 1893-1908: bundle uses POST-step signals
        and the action APPLIED this step (``last_applied_action``).  The
        rolled buffer becomes ``pending_history``; the next compute_obs
        will return its obs using the OLD ``proprio_history`` and THEN
        promote ``pending_history`` into ``proprio_history`` for the
        compute_obs after that.

        This 1-iteration lag is the key fix: env's wr.proprio_history
        update at the END of step N becomes visible in obs only at
        step N+2's returned state.obs.  Without the lag, the visualizer's
        compute_obs at iter N+1 would see bundle_N immediately, while
        env's iter N+1 returned obs would still see pre-roll (= pre-bundle_N)
        history.  Pinned by ``test_proprio_history_lags_one_iteration``.

        Roll input is ``proprio_history`` (the value compute_obs at this
        iteration just read), not ``pending_history``.  This matches env:
        roll input = wr.proprio_history at start of step = obs's pre-roll
        view = same value compute_obs read.
        """
        signals_post = self._signals_adapter.read(mj_data)
        joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
            spec=self._policy_spec, joint_pos_rad=signals_post.joint_pos_rad
        ).astype(np.float32)
        joint_vel_norm = NumpyCalibOps.normalize_joint_vel(
            spec=self._policy_spec, joint_vel_rad_s=signals_post.joint_vel_rad_s
        ).astype(np.float32)
        bundle = np.concatenate(
            [
                np.asarray(signals_post.gyro_rad_s, dtype=np.float32),
                np.asarray(signals_post.foot_switches, dtype=np.float32),
                joint_pos_norm,
                joint_vel_norm,
                np.asarray(self._state.last_applied_action, dtype=np.float32),
            ]
        )
        history = self._state.proprio_history
        self._state.pending_history = np.concatenate(
            [history[1:], bundle[None, :]], axis=0
        ).astype(np.float32)
