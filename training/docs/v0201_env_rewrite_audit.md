# v0.20.1 Env Rewrite — Reuse Audit (Step 1)

**Status:** Historical audit completed.  Companion to `v0201_env_wiring.md`
§10 (8-step rewrite plan).
**Source:** Git revision ``a943831:training/envs/wildrobot_env.py`` (the
last revision before the placeholder; 6533 lines).
**Goal:** map what to extract verbatim into the v3-only rewrite, and what
to drop wholesale.

> **Note (high-confidence prep, 2026-04-19):** the active layout is
> `wr_obs_v6_offline_ref_history`, not `wr_obs_v5_offline_ref`.  v5 was
> deprecated and removed; treat the `wr_obs_v5_offline_ref` references
> below as historical naming for what is now v6's reference-window
> block.  See CHANGELOG `v0.20.1-prep`.

This audit is reading-only — no code moves until the rewrite commit.

---

## A. Pre-rewrite checklist results (gates the rewrite)

| Item | Outcome |
|---|---|
| Env interface (consumer audit) | DONE — see Task #1 |
| Smoke YAML (`ppo_walking_v0201_smoke.yaml`) | **MISSING** — Task #49 hasn't landed it.  The rewrite needs at least a stub YAML to validate construction.  Recommendation: bundle a stub YAML into the rewrite commit; Task #49 then tunes weights/PPO hyperparams rather than creating the file.  Confirm with user before committing. |
| `policy_contract.calib.JaxCalibOps` reuse | DONE — env-agnostic Protocol wrapper, reusable as-is |
| `compute_*_reward` helpers | All 11 are v0.19.5c-specific dead code; extract NONE.  Per §10 step 5 the rewrite uses `r = alive` placeholder; Task 3 designs the imitation reward family fresh. |

**Public env interface the rewrite must preserve** (consumer audit):

- Constructor: `WildRobotEnv(config: TrainingConfig)`
- Properties: `observation_size`, `action_size`
- Methods: `reset(rng) -> WildRobotEnvState`,
  `step(state, action, *, disable_pushes: bool = False) -> WildRobotEnvState`
- Private but consumed: `_default_joint_qpos` (train.py uses it to seed
  `policy_init_action`)

`visualize_policy.py` does **not** construct `WildRobotEnv` (it loads
`MjModel` directly and reimplements the loop with NumPy + policy_contract),
so the rewrite is decoupled from viewer concerns.

---

## B. Extract-verbatim list (carry forward into the rewrite)

These are the specific code regions in ``a943831:training/envs/wildrobot_env.py``
that contain pure, env-shape-agnostic logic worth extracting verbatim
(or with light parameter renames).  Line numbers are from commit
``a943831``.

### B.1 Module-level helper imports (lines 60-115)

The rewrite keeps these imports verbatim:

```python
from training.configs.training_config import get_robot_config, TrainingConfig
from training.envs.env_info import (
    IMU_HIST_LEN, IMU_MAX_LATENCY, PRIVILEGED_OBS_DIM,
    WildRobotInfo, WR_INFO_KEY,
)
from training.envs.disturbance import (
    DisturbanceSchedule, apply_push, sample_push_schedule,
)
from training.envs.domain_randomize import (
    nominal_domain_rand_params, sample_domain_rand_params,
)
from training.sim_adapter.mjx_signals import MjxSignalsAdapter
from training.cal.cal import ControlAbstractionLayer
from training.cal.specs import Pose3D
from training.cal.types import CoordinateFrame
from training.utils.ctrl_order import CtrlOrderMapper
from policy_contract.calib import JaxCalibOps
from policy_contract.jax import frames as jax_frames
from policy_contract.jax.action import postprocess_action
from policy_contract.jax.obs import build_observation
from policy_contract.jax.state import PolicyState
from policy_contract.layout import get_slices as get_obs_slices
from policy_contract.spec import PolicySpec
from policy_contract.spec_builder import build_policy_spec
from training.policy_spec_utils import clamp_home_ctrl, get_home_ctrl_from_mj_model
from training.core.experiment_tracking import get_initial_env_metrics_jax
from training.core.metrics_registry import (
    build_metrics_vec, METRIC_NAMES, METRICS_VEC_KEY, NUM_METRICS,
)
```

**Drop**:
- `training.envs.step_controller as sc` (FSM helper)
- `control.mpc.standing.compute_mpc_standing_action` (MPC standing)
- `control.kinematics.leg_ik` (v1/v2 IK; offline library does this)
- `control.references.walking_ref_v1` (already deleted)
- `control.references.walking_ref_v2` (already deleted)
- `training.envs.teacher_step_target` (drop list)
- `training.envs.teacher_whole_body_target` (drop list)

### B.2 `WildRobotEnvState` (lines 1247-1262)

Verbatim extract.  Already preserved in the placeholder.

```python
class WildRobotEnvState(mjx_env.State):
    pipeline_state: mjx.Data = None
    rng: jp.ndarray = None
```

### B.3 `_load_model` model + CAL + signals adapter setup (lines 1350-1430)

Lines 1350-1430 are the canonical robot-load sequence.  Extract verbatim:

1. Resolve `xml_path` from `config.env.scene_xml_path`
2. `mujoco.MjModel.from_xml_string(...)` with `get_assets(root_path)`
3. `self._mj_model.opt.timestep = self.sim_dt`
4. `self._mjx_model = mjx.put_model(self._mj_model)`
5. `self._robot_config = get_robot_config()`
6. `self._signals_adapter = MjxSignalsAdapter(...)` with `foot_switch_threshold`
7. `self._cal = ControlAbstractionLayer(self._mj_model, self._robot_config)`
8. Keyframe / qpos0 → `self._init_qpos`, `self._default_joint_qpos`

Lines 1430-1455 build the `PolicySpec` via `build_policy_spec(...)` —
keep verbatim, but for v3 the `layout_id` MUST be `wr_obs_v5_offline_ref`
(validated against the smoke YAML).

**Drop everything past line 1455**: the giant block at lines 1455-2029
inside `_load_model` configures `WalkingRefV1Config`, `WalkingRefV2Config`,
LegIK, FSM step controller, MPC standing, and the support-posture B
precomputation.  All in §10 drop list.

### B.4 `get_assets` (line 1212)

Verbatim extract — pure utility that walks an XML directory tree
resolving mesh / material asset bytes for `mujoco.MjModel.from_xml_string`.
The v0.20.1 cleanup (commit ``61ed948``) preserved external consumers
(`training/data/gmr_to_physics_ref_data.py`, `debug_gmr_physics.py`),
so the function signature stays the same.

### B.5 Domain randomization wiring (lines 2098-2150)

`_sample_domain_rand_params` and `_get_randomized_mjx_model` extract
verbatim.  Both are pure delegators to `training.envs.domain_randomize`
helpers (which survived deletion).  No FSM / v1 / v2 dependency.

### B.6 `_to_mj_ctrl` (lines 2150-2153)

Verbatim:

```python
def _to_mj_ctrl(self, ctrl_policy_order: jax.Array) -> jax.Array:
    return self._ctrl_mapper.to_mj_jax(ctrl_policy_order)
```

`CtrlOrderMapper` (`training/utils/ctrl_order.py`) survived deletion.
M2.5 ctrl-ordering bug-fix lineage; mandatory.

### B.7 Action filter + ctrl write (step() inner block, lines 3160-3175)

The action-filter contract is 8 lines and applies as-is to v3:

```python
policy_state = PolicyState(prev_action=pending_action)
filtered_action, policy_state = postprocess_action(
    spec=self._policy_spec,
    state=policy_state,
    action_raw=raw_action,
)
applied_action = pending_action if self._action_delay_enabled else filtered_action
applied_target_q = JaxCalibOps.action_to_ctrl(
    spec=self._policy_spec, action=applied_action
).astype(jp.float32)
data = data.replace(ctrl=self._to_mj_ctrl(applied_target_q))
```

For v3, `raw_action` is the output of `_compose_loc_ref_residual_action`
(see B.8).  The filter, the optional 1-step delay, and the JaxCalibOps
ctrl mapping are unchanged.

### B.8 `_compose_loc_ref_residual_action` (lines 5031-5052)

Extract with the §4.1 absolute-mode change applied.  Old (half-span):

```python
residual_delta_q = (
    jp.clip(policy_action, -1.0, 1.0)
    * self._residual_q_scale
    * self._joint_half_spans
)
target_q = jp.clip(nominal_q_ref + residual_delta_q, joint_mins, joint_maxs)
raw_action = JaxCalibOps.ctrl_to_policy_action(spec=..., ctrl_rad=target_q)
```

New v3-absolute (per §4.1 + step-3a per-joint scale already wired):

```python
residual_delta_q = jp.clip(policy_action, -1.0, 1.0) * self._residual_q_scale_per_joint
target_q = jp.clip(nominal_q_ref + residual_delta_q, joint_mins, joint_maxs)
raw_action = JaxCalibOps.ctrl_to_policy_action(spec=..., ctrl_rad=target_q)
```

The `_residual_q_scale_per_joint` array is already configured by the
step-3a `__init__` infrastructure (commit ``a3101cd``).  In absolute
mode, scale × half_span is replaced by scale (rad) directly.

### B.9 Termination (`_get_termination`, lines 6436-6498)

Verbatim extract.  Pure: reads `root_pose.height`, pitch, roll;
compares against `self._config.env.{min,max}_height`,
`max_pitch`, `max_roll`; supports `use_relaxed_termination` (height-only
mode).  Returns the same `(done, terminated, truncated, info)` tuple
the trainer expects.

The `term/*` diagnostic dict keys (`term/height_low`, `term/height_high`,
`term/pitch`, `term/roll`, `term/truncated`, `term/{pitch,roll,height}_val`)
are part of W&B logging — preserve key names exactly.

### B.10 Properties block (lines 6503-6533)

Verbatim:

```python
@property
def xml_path(self) -> str: return str(self._model_path)
@property
def action_size(self) -> int: return self._mjx_model.nu
@property
def mj_model(self) -> mujoco.MjModel: return self._mj_model
@property
def mjx_model(self) -> mjx.Model: return self._mjx_model
@property
def cal(self) -> ControlAbstractionLayer: return self._cal
@property
def observation_size(self) -> int: return self._policy_spec.model.obs_dim
```

### B.11 IMU noise + delay (lines 842-908, helper `_apply_imu_noise_and_delay`)

Module-level helper.  Verbatim extract.  Used in both `_make_initial_state`
(reset) and `step`.  Pure JAX, no v1/v2 dependency.

### B.12 `_get_obs` skeleton (lines 5403-5472) — needs v5 branch added

The current `_get_obs` dispatches to `wr_obs_v2` (capture_point_error)
and `wr_obs_v3` (gait_clock).  The v5 layout requires extending the
dispatch to forward the eight new channels listed in
`v0201_env_wiring.md` §5.2:

```python
elif self._policy_spec.observation.layout_id == "wr_obs_v5_offline_ref":
    # All channels populated from the v3 service window.
    pass  # plumbed via build_observation kwargs below
return build_observation(
    spec=self._policy_spec,
    state=policy_state,
    signals=signals,
    velocity_cmd=velocity_cmd,
    capture_point_error=capture_point_error,
    gait_clock=gait_clock,
    # v4 channels (filled from offline window for v3):
    loc_ref_phase_sin_cos=...,
    loc_ref_stance_foot=...,
    loc_ref_next_foothold=...,
    loc_ref_swing_pos=...,
    loc_ref_swing_vel=...,
    loc_ref_pelvis_targets=...,
    loc_ref_history=...,
    # v5-only channels (per §5.2):
    loc_ref_q_ref=...,
    loc_ref_pelvis_pos=...,
    loc_ref_pelvis_vel=...,
    loc_ref_left_foot_pos=...,
    loc_ref_right_foot_pos=...,
    loc_ref_left_foot_vel=...,
    loc_ref_right_foot_vel=...,
    loc_ref_contact_mask=...,
)
```

The `build_observation` call signature already accepts the v5 kwargs
(landed in commit ``0cf47e7`` as Task 2 step 2/6).

### B.13 Layer-2 service init (already landed in commit ``a3101cd``)

The step-3a commit landed `__init__` infrastructure for the v3 mode:

- `self._loc_ref_residual_mode in {"half_span", "absolute"}`
- `self._residual_q_scale_per_joint` (per-actuator scale array)
- `self._offline_service: RuntimeReferenceService` (loaded from disk
  or built via ZMPWalkGenerator)
- `self._offline_jax_arrays = service.to_jax_arrays()` (pre-stacked
  for in-JIT lookup)

These do **not** need re-extraction; the rewrite preserves them and
adds the `reset` / `step` v3 branches that consume them.

---

## C. Drop list (do NOT carry forward)

Per `v0201_env_wiring.md` §10 step 2, plus what the audit found beyond
the original list:

### C.1 Subsystems

| Region (line range, commit ``a943831``) | What it does | Why drop |
|---|---|---|
| `_step_walking_reference_jax` (125-238) | v1 parametric reference step | v1 path deleted |
| `_loc_ref_brake_scales` (244-277) | v1/v2 brake-scale heuristic | v1/v2 only |
| `_loc_ref_support_scales` (278-324) | v1/v2 support-scale heuristic | v1/v2 only |
| `_loc_ref_support_health` (325-364) | v1/v2 support-health heuristic | v1/v2 only |
| `_swing_state_in_stance_frame` (365-380) | v1/v2 swing kinematics | v1/v2 only |
| `_update_recovery_tracking` (381-840) | recovery-metrics tracking | recovery_* in drop list |
| `_update_loc_ref_history` (239-243) | v4 phase history rotation | KEEP — needed for v5 (which is a strict superset of v4) |
| `compute_*_reward` × 11 (910-1212) | v0.19.5c reward family | drop list |
| `_precompute_support_posture_B` (2029-2098) | M2.5 support posture | start-from-keyframe in v3 |
| `_compute_nominal_q_ref_from_loc_ref` (4799-5031) | v1/v2 IK | offline library does this |
| `_apply_startup_support_rate_limiter` (4681-4798) | FSM-mode rate limiter | offline q_ref is smooth |
| `_compute_need_step` (5126-5152) | M2 need-step heuristic | M2 dropped |
| `_mpc_standing_compute_ctrl` (5153-5211) | MPC standing pivot | drop list |
| `_fsm_compute_ctrl` (5220-5402) | M3 FSM controller | drop list |
| `_mix_base_and_residual_action` (5054-5125) | M2 base+residual mix | M2 dropped (v3 uses pure residual via B.8) |
| `_get_capture_point_error` (5473-5489) | wr_obs_v2 channel | v2 obs dropped |
| `_get_gait_clock` (5490-5510) | wr_obs_v3 channel | v3 obs dropped |
| `_joint_tracking_error` (4621-4632) | v1/v2 reward helper | drop with reward family |
| `_stance_leg_tracking_errors` (4633-4656) | v1/v2 reward helper | drop with reward family |
| `_limit_joint_target_step` (4657-4680) | v1/v2 startup helper | startup not needed for v3 |
| `_get_privileged_critic_obs` (4580-4620) | privileged critic | re-eval for v3 — see Q below |

### C.2 `step()` and `reset()` v1/v2 branches

The `step()` body is ~1700 lines (2865-4580) because it splits on
`self._loc_ref_version == "v2"` and on `self._controller_stack`.  The
rewrite keeps **only** the v3 branch (~150 lines target):

1. Read `wr` from `state.info[WR_INFO_KEY]`
2. Apply pending push / sample new push if scheduled (B-grade reuse from
   existing `apply_push` in `disturbance.py`)
3. Lookup window from service:
   `win = self._offline_service.lookup_jax(wr.loc_ref_offline_step_idx + 1, self._offline_jax_arrays)`
4. `nominal_q_ref = win["q_ref"]`
5. `raw_action, residual_delta = self._compose_loc_ref_residual_action(policy_action=action, nominal_q_ref=nominal_q_ref)`
6. Action filter + delay (B.7)
7. `data = data.replace(ctrl=self._to_mj_ctrl(applied_target_q))`
8. Inner physics loop: `mjx.step(model, data)` repeated `n_substeps` times
9. IMU noise + delay (B.11)
10. `_get_obs` (v5 layout, B.12)
11. Placeholder reward `r = alive` (§10 step 5)
12. Termination (B.9)
13. Build new `WildRobotInfo` (incrementing `step_count`,
    `loc_ref_offline_step_idx`, rotating IMU history, etc.)
14. Auto-reset on `done == 1.0` via the existing pattern (sample new
    push + velocity_cmd + qpos noise; v3 doesn't need
    `_make_initial_state`'s v2 ref init)

The same compression applies to `reset()` (target ~50 lines).

### C.3 `WildRobotInfo` field cleanup (separate commit per §10 step 4)

Audit confirms what to drop.  The current `WildRobotInfo` (in
`training/envs/env_info.py`) has fields for FSM, MPC, teacher, recovery,
and v1/v2 ref state.  The rewrite consumes the existing `WildRobotInfo`
as-is (so the rewrite commit is one file); the pruning commit follows
per §10 step 4.

---

## D. Open questions surfaced by the audit

**D.1 Privileged critic obs.**  `_get_privileged_critic_obs`
(lines 4580-4620) feeds the asymmetric-actor-critic critic head with
sim-only fields (linear velocity, contact forces, etc.).  Drop entirely
for v3 placeholder, or keep as-is?  Recommendation: **keep**, since the
critic uses it during PPO training even with the placeholder reward,
and rebuilding it is a v0.19.5c lineage decision unrelated to the v3
imitation reward family (Task 3).  It's pure: reads from `data` and
`signals`, no FSM / v1 / v2 dependency.

**D.2 v4-channel population for v5.**  v5 is a strict superset of v4
(per §5.2).  For the v4 channels (phase_sin_cos, stance_foot,
next_foothold, swing_pos, swing_vel, pelvis_targets, history) — what
does the env feed in v3?  The offline library window contains
analogous fields, but the names don't line up 1:1:
  - `phase_sin_cos` ← derive from `step_idx / n_steps` (the library
    has no explicit phase, but the trajectory is periodic)
  - `stance_foot` ← `argmax(window.contact_mask)` (or similar)
  - `next_foothold` ← look ahead one stride in
    `window.{left,right}_foot_pos`
  - `swing_pos`, `swing_vel` ← swing-side foot pos/vel from window
  - `pelvis_targets` ← `concat(window.pelvis_pos, window.pelvis_rpy)`
  - `history` ← rolling buffer of phase_sin_cos, same as v4

This is mechanical but needs Task 3 confirmation that the smoke YAML
expects exactly this mapping.  Defer to the rewrite commit; flag in
the smoke YAML PR.

**D.3 Push schedule in v3.**  Per `walking_training.md` v0.20.1 plan,
the smoke is **push-disabled** initially (focus on imitation tracking).
The rewrite preserves the `push_schedule` plumbing (sample at reset,
apply during step) but the smoke YAML sets `env.push_enabled: false`,
so the apply path is gated off.  Cheaper than ripping out the wiring
and re-adding it later.

**D.4 `disable_pushes=True` kwarg.**  train.py uses this on the
eval-clean path.  v3 honors it the same way: when set, skip
`apply_push` regardless of `push_schedule.start_step <= step_count`.
One-line gate.

---

## E. Estimated rewrite commit size

| Region | Lines |
|---|---|
| Module imports + helpers (B.1, B.4, B.11) | ~150 |
| `WildRobotEnvState` (B.2) | ~10 |
| `WildRobotEnv.__init__` (B.3 + step-3a infra) | ~200 |
| Domain rand wiring (B.5) | ~50 |
| Action filter + ctrl write (B.7) | ~30 |
| Residual action composition (B.8) | ~25 |
| `_get_obs` v5 (B.12) | ~80 |
| `_get_termination` (B.9) | ~70 |
| `_get_privileged_critic_obs` (D.1) | ~40 |
| `reset` (v3 branch) | ~80 |
| `step` (v3 branch + auto-reset) | ~250 |
| `_make_initial_state` (v3-only) | ~120 |
| Properties block (B.10) | ~30 |
| Module docstring + comments | ~70 |
| **Total target** | **~1200-1400 lines** |

Tighter than the §10 estimate of 1500-2000 because dropping the v1/v2
branches removes a lot of conditional plumbing in `step()` and
`_make_initial_state`.

---

## F. Recommended next steps

1. **Resolve smoke YAML blocker.**  Either bundle a stub
   `ppo_walking_v0201_smoke.yaml` into the rewrite commit, or wait
   for Task #49.  Confirm with user.
2. **Write the rewrite** — single commit per §10 step 3.  Use this
   audit as the section-by-section plan: extract from B.1-B.13 in
   order, drop everything in C.
3. **Validate** per §10 step 6: env constructs cleanly with the smoke
   YAML; `jax.jit(env.reset)(rng)` works; `jax.jit(env.step)(state,
   zero_action)` advances `loc_ref_offline_step_idx` 0 → N; the
   wiring test (`tests/test_v0201_env_zero_action.py`) passes.
4. **Stale-test fanout** (separate commit, per §10 step 7): delete
   the FSM/MPC/teacher/recovery/AMP-debug consumers listed in Task #1
   findings.  Most are imports of dropped subsystems — they'll fail
   loudly post-rewrite.  Canonical list lives in
   ``v0201_env_wiring.md`` §8 (split into "source modules and configs"
   + "tests bound to dropped subsystems"); includes
   ``training/eval/eval_loc_ref_probe.py`` and its test
   ``tests/test_eval_loc_ref_probe.py`` (added 2026-04-19 — the v3 env
   doesn't compute v2's hybrid-state diagnostics that the probe
   summarizes).
