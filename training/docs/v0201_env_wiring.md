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

## 5. Observation layout

The existing `wr_obs_v4` builds obs from these `loc_ref_*` kwargs to
`build_observation`:

- `loc_ref_phase_sin_cos` `[2]`
- `loc_ref_stance_foot` `[]` (int)
- `loc_ref_next_foothold` `[2]`
- `loc_ref_swing_pos` `[3]`
- `loc_ref_swing_vel` `[3]`
- `loc_ref_pelvis_targets` (height/roll/pitch tuple)
- `loc_ref_history` `[STANCE_REF_HISTORY_LEN * 2]`

**For the smoke we keep `wr_obs_v4` as-is** — every field maps cleanly to the
offline library:

| Existing field | v3 source |
|---|---|
| `loc_ref_phase_sin_cos` | `service.window.phase_sin/cos` |
| `loc_ref_stance_foot` | `service.window.stance_foot_id` |
| `loc_ref_next_foothold` | derived: `traj.left_foot_pos[idx_at_next_touchdown]` (computed once at env init via the existing `ReferenceLibrary._find_next_foothold` helper) |
| `loc_ref_swing_pos` | `service.window.left_foot_pos` or `right_foot_pos`, selected by stance |
| `loc_ref_swing_vel` | `service.window.left_foot_vel` or `right_foot_vel`, selected by stance |
| `loc_ref_pelvis_targets` | `(pelvis_pos[2], pelvis_rpy[0], pelvis_rpy[1])` — note `pelvis_rpy` is currently zero in the offline library; OK for the smoke |
| `loc_ref_history` | maintained by `_update_loc_ref_history` (unchanged — uses phase_sin/cos which we already have) |

**Optional G7-style anchor preview** (`n_anchor=2` future frames of `q_ref` +
`contact_mask`): per the doc explicitly NOT in the first smoke obs.  Reserved
for a v3.1 obs upgrade.  Service exposes the future fields in `RuntimeReferenceWindow`
already; env just doesn't read them in v3 init.

**No new obs layout** — keep `wr_obs_v4` so the policy network shape matches
existing code.  Obs *contents* change (sourced from offline library), but the
shape is identical.

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

## 7. Pre-smoke "zero-action reproduces q_ref" check

Implementation:

1. Add `tests/test_v0201_env_zero_action.py`: builds env in v3 mode, runs 50
   steps with `policy_action = zeros`, asserts that
   `data.qpos[actuator_qpos_addrs] - q_ref_from_library` stays bounded by the
   PD tracking budget (e.g., max abs error < 0.05 rad steady-state).
2. The control flow: zero action → residual_delta_q=0 → target_q=nominal_q_ref →
   PD tracks within budget.  This is the env-side equivalent of the closeout's
   kinematic/fixed-base smoke — a precondition for any meaningful PPO run.

The test fails CI if the env wiring is wrong (e.g., wrong joint ordering,
wrong residual mode, q_ref shifted).  Lands as part of the implementation PR.

## 8. Files touched

| File | Change | Risk |
|---|---|---|
| `training/envs/wildrobot_env.py` | New v3 branch in `__init__`, `reset`, `step`; new helper `_lookup_offline_reference`; bypass IK + rate limiter in v3; per-joint residual scale + `absolute` mode | Medium — new branches, but isolated by `if loc_ref_version == "v3_offline_library"` |
| `training/envs/env_info.py` | 2 new fields on `WildRobotInfo` (both versions: dataclass + NamedTuple) | Low — additive |
| `training/configs/training_runtime_config.py` | New env config fields: `loc_ref_residual_scale_per_joint`, `loc_ref_residual_mode`, `loc_ref_offline_library_path`, `loc_ref_offline_command_vx` | Low — additive defaults |
| `tests/test_v0201_env_zero_action.py` | New test (above) | New file |

**Untouched**: `walking_ref_v1.py`, `walking_ref_v2.py`, the v1/v2 code paths
in `wildrobot_env.py`, all existing configs (`ppo_walking_v0195c.yaml` etc.).
v0.19.5c remains runnable byte-for-byte after this change.

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
operation.  Service emits a one-shot warning if it does (already implemented).

**Q3.** Multi-command smoke later — is the per-command service registry
forward-compatible?  
**Proposed answer**: yes.  `self._offline_services_by_command_id` is a dict
keyed by an integer command_id; the env carries `loc_ref_offline_command_id` in
WR info; lookup picks the right service's jax_arrays.  For the smoke this dict
has one entry; expansion is additive.

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

## 10. Implementation sequence (post-review)

1. Add config fields (training_runtime_config.py + env_info.py) — no behavior
   change.  Commit.
2. Add v3 branch to `__init__` (load library, build service, build jax_arrays).
   Add v3 branch to reset.  Commit.
3. Add v3 branch to step (lookup, populate loc_ref_* slots, call existing
   residual composition).  Commit.
4. Add `loc_ref_residual_mode = "absolute"` plumbing in
   `_compose_loc_ref_residual_action`.  Commit.
5. Write `tests/test_v0201_env_zero_action.py` and confirm it passes.  Commit.
6. Hand off to Task 3 (imitation reward + smoke YAML).

Each commit is independently runnable; the existing v0.19.5c training path
keeps working at every step.

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
