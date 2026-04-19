# v0.20.1 Env Wiring Design Note

**Status:** Design proposal — pending review before implementation
**Scope:** `training/envs/wildrobot_env.py` changes for the v0.20.1 PPO smoke
**Companion docs:** `walking_training.md` (v0.20.1 section), `reference_design.md`

This note is the implementation plan for wiring the offline `ReferenceLibrary`
into `WildRobotEnv` so the v0.20.1 smoke (single command `vx=0.15`,
imitation-dominant residual PPO) can run. It replaces the current `walking_ref_v2`
parametric reference path **only inside a new `loc_ref_version=v3_offline_library`
mode**; the existing v1/v2 paths are not touched, so no existing training config
is invalidated by this change.

The design is reviewed before code lands so the surgery on a 2300-line JAX env
is bounded and predictable.

---

## 1. What's already in the env that we keep

The existing residual-action plumbing is reusable as-is. We do not need to write
a new action contract.

- **`_compose_loc_ref_residual_action`** (line 4952): already does
  `residual_delta_q = clip(policy_action, -1, 1) * _residual_q_scale * _joint_half_spans`
  then `target_q = clip(nominal_q_ref + residual_delta_q, joint_mins, joint_maxs)`.
  This IS the action contract `q_target = q_ref_from_library + delta_q_policy`.
- **`_residual_q_scale`** (line 1758): scalar, configurable via
  `loc_ref_residual_scale`.  Currently 0.18 (v0.19.5c default) which gives
  ~±0.18·half_span ≈ ±0.13 rad on a typical leg joint — too small for the
  G1 smoke spec.  See §4 for the change.
- **`_joint_half_spans`** (line 1916): `0.5 * (joint_range_maxs - joint_range_mins)` —
  per-joint span used as the residual scale base.  Reused.
- **`build_observation`** (line 5378): already takes a set of `loc_ref_*` fields
  as kwargs.  Most fields map cleanly to the offline library; a small set
  (the FSM-state fields) become unused in v3.
- **`_apply_startup_support_rate_limiter`** (line 4602): rate-limits big jumps in
  `nominal_q_ref` during the FSM-mode startup ramp.  In v3 the offline library's
  q_ref is already smooth (validated by the FK gate), so this becomes a no-op
  for v3.  Plumb a v3-bypass flag rather than removing.

## 2. What we replace

**Replace** the parametric reference computation pipeline:

- `step_walking_reference_v2_jax(...)` → `service.lookup_jax(step_idx, jax_arrays)`
- `_compute_nominal_q_ref_from_loc_ref(...)` (line 4720, ~230 lines of IK +
  support-health logic): **fully bypassed** in v3 — `nominal_q_ref` comes
  directly from the offline library's `q_ref` field.  The IK is offline,
  computed once during library generation by `ZMPWalkGenerator`.

**Bypass** (not remove) the FSM-mode plumbing:

- `loc_ref_mode_id`, `loc_ref_mode_time`, `loc_ref_startup_route_*` fields in
  `WildRobotInfo`: keep field declarations (so the existing v1/v2 code paths
  still compile) but set them to fixed sentinel values (e.g., `mode_id=0`,
  `mode_time=0.0`) in v3 reset/step.  Don't read them in the v3 reward branch.

## 3. New runtime state

Three new fields on `WildRobotInfo` (or one composite dict to minimize churn):

| Field | Shape | Purpose |
|---|---|---|
| `loc_ref_offline_step_idx` | `()` int32 | Current step index into the offline trajectory.  Increments by 1 each env step.  Clamped to `n_steps - 1` by the service. |
| `loc_ref_offline_command_id` | `()` int32 | Index into the per-command service registry.  For the smoke this is constant (single command), but the field is included so multi-command future work is forward-compatible. |
| (no field needed) | — | Pre-stacked JAX arrays live on the env object (`self._offline_jax_arrays_by_command`), not in `WildRobotInfo`, because they're constants across env steps. |

Reset writes `loc_ref_offline_step_idx = 0`.  Step increments and clamps via
the service.

## 4. G1 residual bounds

The current `_residual_q_scale = 0.18` is a single scalar applied to all joints'
half-span.  For G1 (`±0.50 rad` on leg joints, `±0.20 rad` on others) we need
either (a) a per-joint scale array, or (b) a two-tier scalar split.

**Decision: per-joint scale array.**  Cleanest, doesn't add new config primitives,
and matches how `_joint_half_spans` is already shaped.  Specifically:

- New env config field: `loc_ref_residual_scale_per_joint` (optional dict mapping
  joint name → scalar in [0, 1]).  When present, **overrides** the scalar
  `loc_ref_residual_scale` for those joints; missing joints fall back to the
  scalar.
- For the v0.20.1 smoke, the YAML sets:
  ```yaml
  loc_ref_residual_scale: 0.20         # default (non-leg joints)
  loc_ref_residual_scale_per_joint:
    left_hip_pitch: 0.50
    left_hip_roll: 0.50
    left_knee_pitch: 0.50
    left_ankle_pitch: 0.50
    right_hip_pitch: 0.50
    right_hip_roll: 0.50
    right_knee_pitch: 0.50
    right_ankle_pitch: 0.50
  ```
  These are scaled internally by `_joint_half_spans`, so the actual rad bound
  is `scale * half_span`.  Spec says G1 wants `±0.50 rad` directly, NOT
  `±0.50·half_span`.  See §4.1 below.

### 4.1 Scale interpretation: half-span vs absolute

The existing v2 code multiplies `policy_action × scale × half_span`.  For
joints with `half_span > 1.0 rad` (most legs), `scale=0.50` gives `>0.5 rad`
of residual authority — over-shoots the G1 spec.  For joints with
`half_span < 1.0 rad`, it under-shoots.

**Decision: introduce a v3-only `loc_ref_residual_mode` enum**:

- `mode = "half_span"` (current default; preserves v2 behavior; used by v1/v2)
- `mode = "absolute"` (v3 default; `target_q = nominal_q_ref + clip(action, -1, 1) * scale`)

In absolute mode the per-joint `scale` value IS the rad bound directly.  This
matches G1's spec verbatim and removes a confusing layer of indirection.

The v0.20.1 smoke YAML uses `loc_ref_residual_mode: absolute`.  v1/v2
configurations stay on the existing implicit `half_span` mode (no behavior
change).

## 5. Observation layout — REVISED per round-1 design review

### 5.0 Why `wr_obs_v4` reuse was rejected

The first version of this note proposed reusing `wr_obs_v4` unchanged.
Round-1 review (correctly) flagged this as a blocker: the v0.20.1 smoke
contract (``walking_training.md`` v0.20.1 Observation §) requires the actor
to see a current reference slice that **includes the current `q_ref`**, but
`wr_obs_v4`'s `loc_ref_*` channels (line 5378 of `wildrobot_env.py`) carry
only `phase_sin_cos`, `stance_foot`, `next_foothold`, `swing_pos`,
`swing_vel`, `pelvis_targets`, `history` — **no path for `q_ref`** at all.
Reusing the layout would silently violate the smoke contract.

### 5.1 Decision: new `wr_obs_v5_offline_ref` layout for v3 only

Add a new layout id `wr_obs_v5_offline_ref` that extends `wr_obs_v4` with
the missing channels.  v1/v2 paths keep `wr_obs_v4` (no behavior change for
v0.19.5c).  The smoke YAML sets:

```yaml
env:
  loc_ref_version: v3_offline_library
  actor_obs_layout_id: wr_obs_v5_offline_ref
```

### 5.2 Channels added vs `wr_obs_v4`

| New channel | Shape | Source from `RuntimeReferenceWindow` |
|---|---|---|
| `loc_ref_q_ref`           | `[n_joints]` | `window.q_ref` |
| `loc_ref_pelvis_pos`      | `[3]`        | `window.pelvis_pos` |
| `loc_ref_pelvis_vel`      | `[3]`        | `window.pelvis_vel` (finite-diff per G2) |
| `loc_ref_left_foot_pos`   | `[3]`        | `window.left_foot_pos` |
| `loc_ref_right_foot_pos`  | `[3]`        | `window.right_foot_pos` |
| `loc_ref_left_foot_vel`   | `[3]`        | `window.left_foot_vel` (finite-diff) |
| `loc_ref_right_foot_vel`  | `[3]`        | `window.right_foot_vel` (finite-diff) |
| `loc_ref_contact_mask`    | `[2]`        | `window.contact_mask` |

This is a strict superset of `wr_obs_v4`'s `loc_ref_*` content: the existing
`v4` channels are still computed and concatenated, and the v5 additions are
appended at the end so that `wr_obs_v4`-shaped consumers (e.g., the
`build_observation` builder) remain compatible if we re-route them.

### 5.3 Channels NOT added (deferred)

Per the smoke obs contract ("do not start the first smoke with a large
dense future preview window"), the following `RuntimeReferenceWindow` fields
are NOT included in `wr_obs_v5_offline_ref`:

- `future_q_ref`, `future_phase_sin/cos`, `future_contact_mask`
  (the 1-2 future anchor frames)

These are reserved for a `wr_obs_v6_*` upgrade if the smoke shows the
policy needs anticipation.  `RuntimeReferenceService` already computes
them; the env just doesn't read them in v5.

### 5.4 Deferred: `loc_ref_pelvis_rpy` channel

The library currently emits `pelvis_rpy = zeros` (yaw-stationary, no
explicit roll/pitch shaping).  The reward (`ref/body_quat_track`) reads
pelvis RPY directly from the trajectory, not via the obs.  **No
`loc_ref_pelvis_rpy` obs channel for v5** — adding it would be a noisy zero
input for the policy.  If a future prior emits non-zero pelvis_rpy targets,
add the channel then.

### 5.5 Implementation surface — REVISED per round-2 review

The previous version of this section understated the code surface.  The
actual observation builders are split across two files (JAX and NumPy
parity is mandatory because `policy_contract` is consumed both at training
time and at export / runtime), and both must accept the new channels and
the new layout id.

**Mandatory dual-builder changes** (these are the round-2 review's
explicitly-named files; both need synchronized edits):

- **`policy_contract/jax/obs.py`**:
  - line 32: extend the layout id whitelist
    `{"wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_v4"}` →
    add `"wr_obs_v5_offline_ref"`
  - line 12 (`build_observation_from_components`): add new kwargs
    `loc_ref_q_ref`, `loc_ref_pelvis_pos`, `loc_ref_pelvis_vel`,
    `loc_ref_left_foot_pos`, `loc_ref_right_foot_pos`,
    `loc_ref_left_foot_vel`, `loc_ref_right_foot_vel`,
    `loc_ref_contact_mask` (each defaulting to `None` with a `zeros(...)`
    fallback identical in pattern to the existing `loc_ref_*` fallbacks)
  - line 94-95 (the per-layout `parts.extend(...)` block): add a new
    branch for `wr_obs_v5_offline_ref` that extends with both the
    `wr_obs_v4` parts (phase, stance, foothold, swing_pos, swing_vel,
    pelvis, history) AND the eight new v5 channels in the order listed
    in §5.2
  - line 102 (`build_observation` wrapper): add the same eight new
    kwargs and forward them to `build_observation_from_components`
- **`policy_contract/numpy/obs.py`**: identical edits (mirror the JAX
  changes line-for-line; the existing parity test will fail if they
  diverge).  The NumPy builder is what the export bundle and runtime
  policy adapter consume.

**Other implementation surface** (orchestration, config, env-side):

- `training/envs/wildrobot_env.py`:
  - `_get_obs` (line 5378 area): when
    `layout_id == "wr_obs_v5_offline_ref"`, read the v3 service window
    and pass the eight new channels to `build_observation`
  - the existing `wr_obs_v4` callsite stays unchanged (it just doesn't
    pass the new kwargs)
- `training/policy_spec_utils.py` (line 130 area): allow
  `actor_obs_layout_id = "wr_obs_v5_offline_ref"`
- `training/configs/training_runtime_config.py` (line 119 area): no
  validation change required (the field is already a free-form string),
  but add a comment noting the v5 layout exists and what mode it pairs
  with
- `training/exports/export_policy_bundle.py`: defer until smoke actually
  ships a checkpoint worth exporting (export bundle just needs to know
  the obs dim — driven by the spec, not hard-coded)

**Parity test obligation**: there is a JAX/NumPy parity test for the
observation builders.  Whatever it asserts for `wr_obs_v4` (concatenation
ordering, dtype, shape) must hold for `wr_obs_v5_offline_ref` too;
the implementer should extend the parity test fixture to include the
new layout id and the new channel set.

**Net change footprint**: 2 builder files (synchronized), 1 env file
(one new branch), 1 spec-utils file (one whitelist entry), 1 config-doc
update, 1 parity-test extension.  No checkpoint format change; existing
`v0.19.5c` configs / checkpoints unaffected because they continue to
use `wr_obs_v4`.

## 6. Reset & step deltas

**Reset (v3 mode only):**

```python
# Existing: build runtime parametric reference state
# v3:        load offline trajectory + build service + write step_idx=0

if loc_ref_version == "v3_offline_library":
    # Service is pre-built at env __init__ (one per command in the smoke set)
    win = self._offline_service.lookup_np(0)  # for reset only — NumPy is fine
    nominal_q_ref = jp.asarray(win.q_ref, dtype=jp.float32)
    # Skip _compute_nominal_q_ref_from_loc_ref entirely.
    # Skip _apply_startup_support_rate_limiter (offline q_ref is already smooth).
    # Use win.* fields to populate the loc_ref_* slots in WildRobotInfo.
    wr_info_extras = dict(loc_ref_offline_step_idx=jp.int32(0), ...)
```

**Step (v3 mode only):**

```python
if loc_ref_version == "v3_offline_library":
    step_idx = wr.loc_ref_offline_step_idx + 1
    win = self._offline_service.lookup_jax(step_idx, self._offline_jax_arrays)
    nominal_q_ref = win["q_ref"]
    # Populate loc_ref_* fields from win for reward + obs.
    # _compose_loc_ref_residual_action used unchanged but with "absolute" residual mode.
```

Both reset and step preserve the existing `WildRobotEnvState` shape — the new
fields are added to `WildRobotInfo` (the dataclass already has spare slots; the
NamedTuple fallback gets new fields too).

## 7. Pre-smoke "zero-action" wiring check — REFINED per round-1 review

The check is **wiring-focused**, not locomotion-quality-focused.  Round-1
review noted that mixing the two would let a balance failure mask a wiring
failure (or vice versa).

Implementation (`tests/test_v0201_env_zero_action.py`):

1. **Wiring assertion (primary)**: with `policy_action = zeros`, the env
   computes `target_q == nominal_q_ref == q_ref_from_library[step_idx]`
   exactly (or within float32 round-off), at every step in a 50-step
   probe.  This is testable without running physics — just call the env's
   action-composition path and inspect `target_q`.
2. **Residual-zero assertion**: `_compose_loc_ref_residual_action`
   returns `residual_delta_q` that is byte-zero when `policy_action = 0`
   (verifies the absolute-mode plumbing in §4.1).
3. **Trajectory-advance assertion**: across 50 steps,
   `wr.loc_ref_offline_step_idx` advances `0, 1, 2, ..., 49`; the
   `loc_ref_*` obs channels read from the service track those step
   indices (not stuck on step 0 due to a missed increment).
4. **PD-trackability probe (informational, NOT a balance test)**: optional
   short physics rollout (≤20 steps) that records `data.qpos -
   q_ref_lib`.  The test does not assert any specific tracking RMSE bound
   — that's the kinematic/fixed-base closeout's job.  We log the value so
   regressions in actuator/PD config show up, but failure to track
   tightly is NOT the wiring test failing.

The test catches: wrong joint ordering, wrong residual mode, q_ref
shifted by one step, missed step-idx increment, FSM-mode plumbing
leaking into the v3 path.  It does NOT catch: balance instability,
reward bugs, learning failures (those are the smoke runner's job).

## 8. Files touched — REVISED (rewrite scope, round-3)

After committing steps 1-3a (config / WR info / dual obs builders / `__init__`
infrastructure), it became clear the original "surgical add v3 branches"
plan was infeasible: ``training/envs/wildrobot_env.py`` is 6533 lines
deeply entangled with several deprecated subsystems beyond v1/v2
(M3 FSM, MPC standing controller, teacher targets, recovery metrics,
v0.19.5c reward family).  Stripping them piecemeal in 6 commits would
leave the env in indeterminate states between commits and risk silent
regressions in the v0.19.5c reward pipeline.

User direction (round-3): **rewrite ``wildrobot_env.py`` from scratch
as a v3-only module**, dropping the deprecated subsystems entirely
rather than carrying them forward through deprecation cycles.  No
backward-compat with v0.19.5c.

**Already deleted** (commit ``0ffc473``):

- ``control/references/walking_ref_v1.py`` (173 lines)
- ``control/references/walking_ref_v2.py`` (1523 lines)
- ``control/references/locomotion_contract.py`` (146 lines)
- ``tests/test_walking_ref_v1.py``, ``tests/test_walking_ref_v2.py``,
  ``tests/test_walking_v0193_integration.py``,
  ``training/tests/test_v0195_walking_controller.py``,
  ``tests/test_runtime_loc_ref_builder.py``

**To delete in the next session** (consequent on the rewrite):

*Source modules and configs:*

| File | Why |
|---|---|
| ``runtime/wr_runtime/control/loc_ref_runtime.py`` | wraps walking_ref_v1 |
| ``runtime/wr_runtime/control/loc_ref_runtime_v2.py`` | wraps walking_ref_v2 |
| ``runtime/wr_runtime/control/run_walking.py`` | uses both |
| ``tools/reference_smoke/`` (whole directory) | v0.19.2 smoke |
| ``training/eval/run_m25_v2_probe_sweep.py`` | M2.5 v2 probe |
| ``training/eval/visualize_nominal_ref.py`` | v1/v2 plotter |
| ``training/eval/eval_loc_ref_probe.py`` | v2 probe runner — emits v2-only summary keys (``debug_loc_ref_swing_x_scale_mean``, ``tracking_loc_ref_progression_permission_mean``, hybrid mode ids, etc.).  The v3 env doesn't compute any of these signals (no v2 hybrid state machine, no swing-x brake heuristic).  A v3 nominal-tracking probe can be re-introduced later if Task #49 needs one — different shape, different metrics. |
| ``control/locomotion/nominal_ik_adapter.py`` | v1/v2 IK adapter |
| ``control/locomotion/walking_controller.py`` | v0.19 walking ctrl |
| ``training/configs/ppo_walking_v0193a.yaml`` | v1-bound |
| ``training/configs/ppo_walking_v0195.yaml`` | v2-bound |
| ``training/configs/ppo_walking_v0195c.yaml`` | v2-bound (kept as design reference; can be archived) |
| ``training/envs/teacher_step_target.py`` | M3 teacher |
| ``training/envs/teacher_whole_body_target.py`` | M3 teacher |

*Tests bound to dropped subsystems* (audited 2026-04-19; many still
import cleanly because the removed names live in the deleted env's
internals, but they will fail at construction or assertion time
because the v3 env no longer offers FSM / MPC / teacher / recovery /
v2 ref state).  Delete in the same cleanup commit as the source modules:

| Test file | Bound to |
|---|---|
| ``tests/test_eval_loc_ref_probe.py`` | v2 probe runner above; asserts v2-specific summary keys |
| ``tests/test_v0171_boundary_push_eval.py`` | v0.17.1 push-recovery scaffold |
| ``tests/test_mpc_standing_v0173.py`` | MPC standing pivot (drop list) |
| ``tests/test_teacher_step_target.py`` | M3 teacher targets (drop list) |
| ``tests/test_fsm_integration.py`` | M3 step FSM (drop list) |
| ``tests/test_recovery_metrics_aggregation.py`` | recovery_* fields + RECOVERY_WINDOW_STEPS constant (already broken: ImportError) |
| ``tests/test_imu_noise_latency.py`` | imports the env's ``_apply_imu_noise_and_delay`` helper; helper survived but the test setup uses v0.19.5c env construction |
| ``training/tests/test_env_step.py`` | v0.19.5c env-step contract |
| ``training/tests/test_env_info_schema.py`` | WildRobotInfo full-schema check (depends on M3/recovery fields that §10 step 4 will drop) |
| ``training/tests/test_physics_validation.py`` | v0.19.5c physics smoke |
| ``training/tests/test_actuator_ordering.py`` | M2.5 ctrl-ordering scaffold (now belongs in policy_contract tests) |
| ``training/tests/test_ppo_training.py`` | v0.19.5c trainer smoke |
| ``training/tests/test_foot_contacts.py`` | v0.19.5c foot-contact aggregation reward |
| ``training/tests/run_validation.py`` | non-pytest smoke runner that constructs v0.19.5c env |
| ``training/scripts/debug_amp_features.py`` | already broken: imports nonexistent ``get_feature_config`` and ``EnvConfig`` from removed modules |
| ``training/scripts/diagnose_amp_features.py`` | AMP-debug coupled to v0.19.5c env metrics |

Tests that **stay** (post-rewrite verified):
``tests/test_policy_contract_parity.py``,
``tests/test_runtime_reference_service.py``,
``tests/test_policy_contract_migration.py`` —
these decoupled from the v0.19.5c env-internal contract.

**To update in the next session** (drop loc_ref builder dependency):

- ``runtime/wr_runtime/control/run_policy.py`` — strip
  ``RuntimeLocRefBuilder`` import; wire to a v3 service when the
  hardware-deployable runtime adapter lands (v0.20.2 milestone)
- ``runtime/wr_runtime/validation/replay_policy.py`` — same

**To rewrite** (the big one):

- ``training/envs/wildrobot_env.py`` — see §10 below for scope

## 9. Open design questions (decide before code)

**Q1.** `loc_ref_pelvis_targets` from offline library has `pelvis_rpy = zeros`
(the prior is yaw-stationary and currently emits zero roll/pitch targets).  Is
that a problem for `m3_pelvis_orientation_tracking`?  
**Proposed answer**: no — the smoke uses the new imitation reward family (Task 3),
not the v0.19.5c m3 family.  The new `ref/body_quat_track` reads pelvis_rpy
from the library directly and rewards small deviation from those zero targets.

**Q2.** What if the policy reaches the end of the 22-second offline trajectory
within an episode?  
**Proposed answer**: per the library's clamp contract, the service freezes at
the terminal frame.  Episode horizon `max_episode_steps = 500` × `ctrl_dt = 0.02`
= 10 s, well under the trajectory's 22 s.  No clamp triggered in normal
operation.  ``RuntimeReferenceService.lookup_np`` emits a one-shot
``RuntimeWarning`` per service instance if a query lands past the end
(added in commit ``<TBD>`` after round-1 review; covered by
``test_terminal_clamp_emits_warning_once``).  The JAX path can't issue
warnings inside JIT, so callers running JAX-only loops must either size the
horizon to ``service.n_steps`` or run a NumPy-side overrun probe out of
band.

**Q3.** Multi-command smoke later — is the per-command service registry
forward-compatible?  
**Revised answer (round-1 review)**: NO, not as originally proposed.  A
Python dict keyed by an integer cannot be indexed inside a JAX JIT'd step
using a JAX-traced integer; the lookup would have to happen outside the
JIT, defeating the per-step efficiency the service is built for.

For the v0.20.1 smoke (single command), the dict-of-services approach is
fine — ``loc_ref_offline_command_id`` is a Python constant (always 0), the
JIT folds it, and the JAX arrays for that one command are threaded through
the step exactly as designed.  This is a **smoke-only simplification**, not
forward-compatible multi-command machinery.

If multi-command training resumes (later milestone), the right design is
**stacked JAX arrays** at the env level:

- ``self._offline_jax_arrays_stacked`` shape ``[n_cmd, n_step, ...]``
  for each field, built once at env init by stacking the per-command
  service outputs
- ``loc_ref_offline_command_id`` is a JAX-traced int, used as the leading
  index in the stacked arrays inside the JIT
- the per-command Python ``RuntimeReferenceService`` instances are kept
  for NumPy-side use (eval, viewer, tests) but the JAX path no longer
  routes through them

That is additive over this design (the service interface doesn't change;
only the env-side storage does).  Explicitly out of scope for v0.20.1.

**Q4.** What about reset RNG?  Should the offline trajectory have a per-env
random start offset (e.g., to avoid all envs walking the same prefix)?  
**Proposed answer**: NOT for the smoke — single-command, single-trajectory,
all envs start at step 0.  This makes the pre-smoke zero-action sanity check
cleanly reproducible.  Random start offsets become useful for multi-command
training; defer.

**Q5.** Per-joint residual scale: should leg joints get directional asymmetry
(e.g., +0.5 rad / -0.3 rad to constrain backward-leaning motions)?  
**Proposed answer**: NOT for the smoke.  Symmetric clip is simpler and lets the
G5 anti-exploit metric do its job (catch the policy when it leans too far).
Asymmetric clipping is a v3.1 lever if needed.

## 10. Implementation sequence — REVISED (rewrite, round-3)

**Status at the session boundary** (commit ``7c81dff``, post round-3
cleanup):

- Steps 1-3a of the original surgical plan are landed:
  config fields + WR info field declarations (``8c69a06``); dual obs
  builders + parity test (``0cf47e7``); env ``__init__`` infrastructure
  (``a3101cd``).
- Deletion wave landed in 4 commits:
  - ``0ffc473`` — v1/v2 source files + 5 direct test files
  - ``61ed948`` — 12 v1/v2-bound consumers (runtime/, control/, training/eval/,
    tools/) + 4 dependent tests + 4 ``__init__.py`` cleanups
  - ``a943831`` — config defaults (loc_ref_version, loc_ref_residual_mode)
    + 6 v1/v2-bound YAML configs deleted + train.py default updated
  - ``7c81dff`` — replaced ``wildrobot_env.py`` with an 80-line
    placeholder; deleted ``test_reward_terms.py`` (tested deleted
    helpers); env imports cleanly, construction raises
    ``NotImplementedError`` pointing to this design note
- **Branch is now safe for shared use**: any module imports
  cleanly; tests that try to construct ``WildRobotEnv`` fail
  loudly with a clear pointer to the rewrite plan.

**Next-session execution plan (rewrite-shaped, replaces the original
6-commit surgical plan)**:

1. **Reuse audit** — read the existing 6533-line ``wildrobot_env.py``
   and identify what to extract verbatim into the new env:
   - model load, calibration ops, signals adapter, MJX physics step
   - action filter (lowpass v1) + residual action composition
   - domain randomization sampling
   - push schedule sampling
   - termination conditions (height + pitch/roll bounds)
   - obs builder dispatch (v5 layout)
   - core ``WildRobotEnvState`` shape

2. **Drop list** — explicit list of subsystems to NOT carry forward:
   - ``_fsm_compute_ctrl`` + ``fsm_*`` state (M3 era)
   - ``_mpc_standing_compute_ctrl`` + ``mpc_*`` state (v0.17.3 pivot)
   - teacher targets (``teacher_*`` state, helper modules)
   - recovery-metrics tracking (``recovery_*`` state)
   - the v0.19.5c reward family (``compute_*_reward`` helpers, m3_*
     tracking, dense_progress, slip nuance) — replaced by the
     imitation-dominant family in Task 3
   - ``_compute_nominal_q_ref_from_loc_ref`` (v1/v2 IK)
   - ``_apply_startup_support_rate_limiter`` (FSM-mode helper)
   - ``_step_walking_reference_jax`` (v1 helper)
   - all v1/v2 brake-scale / support-scale / support-health helpers

3. **New env structure** (target ~1500-2000 lines, single file):
   - ``WildRobotEnv.__init__``: model load + calib + signals adapter
     + Layer-2 service init (already designed in §1 / commit ``a3101cd``)
   - ``WildRobotEnv.reset``: keyframe init → seed step 0 of the offline
     trajectory → build initial ``WildRobotEnvState``
   - ``WildRobotEnv.step``: read window from service →
     ``q_target = q_ref + residual(policy_action)`` (absolute mode) →
     MJX physics step → obs / placeholder reward / termination
   - ``WildRobotInfo``: pruned to only the fields the rewrite uses
     (existing M3/MPC/recovery/teacher fields removed in step 4 below)

4. **WildRobotInfo cleanup** (separate commit after the rewrite):
   - keep: ``step_count``, ``prev_action``, ``pending_action``,
     ``truncated``, ``velocity_cmd``, ``prev_root_*``, ``imu_*_hist``,
     ``foot_contacts``, ``root_height``, ``prev_left/right_loaded``,
     ``cycle_start_forward_x``, ``last_touchdown_*``, ``critic_obs``,
     ``loc_ref_*`` (the obs-feeding ones), ``nominal_q_ref``,
     ``loc_ref_offline_step_idx``, ``loc_ref_offline_command_id``,
     ``push_schedule``, ``domain_rand_*``
   - drop: ``fsm_*`` (9 fields), ``mpc_*`` (5 fields), ``teacher_*``
     (7 fields), ``recovery_*`` (~25 fields), ``loc_ref_mode_*``,
     ``loc_ref_startup_route_*`` (FSM-era), ``loc_ref_com_x0_at_*``
     (v2-specific)

5. **Placeholder reward** for the rewrite commit:
   - ``r = alive`` only (1.0 per step until termination)
   - This lets the env JIT, reset, and step end-to-end so we can validate
     wiring before Task 3 designs the real imitation reward family.

6. **Validation** before declaring the rewrite done:
   - env constructs cleanly with the v0.20.1 smoke config (Task 3 YAML)
   - ``jax.jit(env.reset)(rng)`` returns a valid state
   - ``jax.jit(env.step)(state, zero_action)`` runs N steps without
     error and ``state.info[WR_INFO_KEY].loc_ref_offline_step_idx``
     advances 0 → N
   - the §7 wiring test (``tests/test_v0201_env_zero_action.py``)
     passes: ``target_q == q_ref_lib`` exactly when
     ``policy_action == 0``
   - parity tests (``tests/test_policy_contract_parity.py`` 7 cases)
     still pass

7. **Cleanup commits** after the rewrite:
   - delete deprecated runtime helpers (``loc_ref_runtime[_v2].py``,
     ``run_walking.py``) + update ``run_policy.py`` /
     ``replay_policy.py`` to drop ``RuntimeLocRefBuilder``
   - delete obsolete configs / tools / control/locomotion helpers
     (per §8 list)
   - clean dead config fields and dead ``WildRobotInfo`` fields per §4 above

8. **Hand-off** to Task 3 (imitation-dominant reward + smoke YAML +
   PPO runner).

**Estimated scope of the rewrite commit**: ~1500-2000 lines of new
``wildrobot_env.py``, with ~6533 lines deleted (the old file).
Net repo line change: roughly -4500.

**Pre-rewrite checklist for the next session**:

- Read ``training/eval/visualize_policy.py`` and ``train.py`` to confirm
  what env interface they call into (``env.reset``, ``env.step``,
  ``env.observation_size``, ``env.action_size``, ``env.dt``,
  ``env.mj_model``, ``env.mjx_model``)
- Read the v0.20.1 smoke YAML (Task 3 will write it before the rewrite
  begins) so the env is built to consume the right config fields
- Verify ``policy_contract.calib.JaxCalibOps`` can be reused as-is (it
  was designed to be env-agnostic)
- Identify which ``compute_*_reward`` helpers in the existing env file
  are pure functions worth keeping in a shared ``training/envs/rewards.py``
  vs which ones are v0.19.5c-specific dead code

## 11. What this note does NOT cover

- The new imitation reward family (Task 3 — separate design pass)
- The smoke YAML config (Task 3)
- The PPO runner / launch script (Task 3)
- Multi-command training (deferred — explicitly out of scope per the smoke)
- Stop / yaw / resume commands (deferred to v0.20.4)
- Hardware-deployable runtime adapter (deferred to v0.20.2)

---

**Reviewer ask:** confirm the §2 bypass list, the §4.1 residual-mode introduction,
the §5 obs-layout reuse decision, and the §7 zero-action sanity check before
code lands.  Open questions Q1-Q5 also benefit from explicit yes/no.
