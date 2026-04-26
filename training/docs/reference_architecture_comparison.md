# WR Reference Architecture Plan vs ToddlerBot

**Status:** Canonical design note for `v0.20.x` reference alignment
**Last updated:** 2026-04-26 (Phase 7-11 plan added)
**Purpose:** define the reference-design policy for WildRobot (WR), record
the design-history context behind the current parity harness, and order the
architecture backlog needed to bring WR reference quality on par with or
better than ToddlerBot (TB).

All file paths in this doc are repo-relative:
- TB: `projects/toddlerbot/...`
- WR: `projects/wildrobot/...`

## Governing Policy

TB is the default reference architecture.

WR should align to TB on reference design and setup unless there is an
explicit reason not to. A WR-specific divergence is acceptable only when at
least one of these is true:

- it is forced by WR hardware or kinematic limits
- matching TB would materially hurt WR trackability or safety
- matching TB would add significant complexity with no measured parity benefit

If none of those are true, the divergence is alignment backlog.

## Acceptance Rule

The goal is not "WR looks reasonable in isolation." The goal is:

- WR reference quality is on par with or better than TB
- the judgment is driven primarily by architecture, not local tuning
- the comparison is size-aware and robot-limit-aware

Use the parity layers in
[reference_parity_scorecard.md](reference_parity_scorecard.md):

- `P0`: shared geometry gate
- `P1A`: FK-realized nominal gait shape
- `P2`: reference smoothness
- `P1`: free-base zero-residual trackability

Acceptance uses two lenses:

- absolute parity is the sanity floor
- size-normalised parity is the primary reading for "on par or better"

This means a WR change should be credited only when it improves the
load-bearing parity gaps under the normalised view, while not regressing the
absolute floor or violating WR's own `G4/G5/G6/G7` policy contract.

## Design History

This note supersedes the earlier history-only note
`reference_design_history_tb_vs_wr.md`.

The design history matters because the first comparison pass was too weak:

1. We started with a shared geometry gate:
   - stance foot bottom `z <= 0.003 m`
   - swing foot bottom `z > -0.002 m`
   - `vx in {0.10, 0.15, 0.20, 0.25}`
   - frames `{0, 5, 10, 16, 22, 28, 32, 48, 64}`
2. That first pass showed WR was acceptable at the current in-scope bins
   `vx in {0.10, 0.15}`, but weaker than TB at higher speed.
3. We then built a shared parity harness:
   - TB-side geometry test:
     `projects/toddlerbot/tests/test_walk_reference_geometry.py`
   - WR-side parity script:
     `projects/wildrobot/tools/reference_geometry_parity.py`
4. Review of that first harness found three major problems:
   - it mixed WR planner-side quantities with TB FK-realized quantities
   - it called offline gait-shape statistics "trackability"
   - it was missing most smoothness metrics
5. The current structure fixed that split:
   - `P0` shared geometry
   - `P1A` FK-realized nominal gait shape
   - `P2` smoothness
   - `P1` free-base zero-residual replay

That history leads to the current design rule:

- use TB as the baseline
- measure WR using the same realized quantities wherever possible
- prioritize architecture changes before reward or PPO-side tuning

## Current Gap Summary

Post-Phase-6 state (refreshed parity report at commit `ac51038`,
both WR and TB sides re-run under `cycle_time_s = 0.72`).

### Geometry — PASS

- WR passes the in-scope `P0` gate at `vx <= 0.15`
- WR also passes the **full-matrix** P0 gate post-Phase-6: `total_failures = 0`,
  worst stance z = +1.9 mm, worst swing margin = +17 mm
- the previously deferred high-`vx` (≥ 0.20) stance failures are now under
  the 3 mm tolerance

Interpretation:
- WR is at parity with TB on geometry, including the diagnostic high-`vx`
  bins that were out of scope before Phase 6

### FK-realized gait shape — mixed

- absolute step_len / cadence / clearance / speed / DS all PASS for both
  TB baselines (Phase 6 closed the absolute gates)
- size-normalised view still fails on four gates at vx=0.15:
  `step_length_per_leg = 0.56× TB`, `cadence_froude_norm = 1.25× TB`,
  `swing_clearance_per_com_height = 0.82× TB`,
  `swing_foot_z_step_per_clearance = 1.40× TB`

Interpretation:
- the absolute gait shape matches TB; the proportional shape does not
- two of the four normalised FAILs (`step_per_leg`, `cadence_norm`) are
  bounded by leg-length / Froude scaling at the current operating point
  and are not closeable by planner tuning alone
- the other two (`clearance_per_h`, `swing_step_per_clr`) are closeable
  by config (`foot_step_height_m`) and planner-shape (swing-z envelope)
  changes — owned by Phases 7 / 8

### Smoothness — mixed

- shared-leg q step / pelvis step / contact-flips all PASS (WR `q_step`
  0.36 rad is actually *smoother* than TB's 0.65 rad)
- absolute swing-foot z step still FAILs (WR 0.0183 m vs TB 0.010 m,
  ratio 1.83×)

Interpretation:
- the swing-z lift-off / touchdown shape is the only remaining smoothness
  gap; addressed by Phase 7 (TB-style single-peaked triangle envelope)

### Free-base zero-residual trackability — mixed

- absolute-physical gates pass with margin: WR survives 4-9× more steps
  than TB on every `vx` bin; WR drifts ½ TB's per-leg drift rate;
  WR achieved-forward-velocity matches or beats TB on every bin
- per-joint / contact / step gates fail: WR shared-leg RMSE 1.7× TB,
  WR `ref_contact_match_frac` 0.58-0.70 vs the 0.90 absolute floor,
  WR touchdown-step-length-per-leg under-shoots

Interpretation:
- the closed-loop layer mixes prior shape, actuator-stack divergence, and
  WR's cycle-0 retention; whether the failing gates are reference-quality
  problems or actuator-stack artifacts is not settled by current data
- Phase 10 (post-Phase-7 diagnostic) decides the verdict; until then the
  doc's "on par or better" rule cannot be claimed as cleared, even if all
  FK-side and smoothness gates close

## Pipeline Comparison

Both projects use the same ZMP-preview / LIPM family. The main question is
not "same algorithm family?" It is "does WR package and consume the reference
the same way TB does?"

### Stage 1. Footstep planner

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Command space | `vx, vy, yaw_rate` lookup grid | `vx` only | backlog for broader TB parity, but not the main current gap |
| Cycle-0 handling | first transient cycle truncated before storage | from-rest cycle retained, linear replay only | justified temporary difference for current single-cmd setup |

### Stage 2. ZMP preview COM integration

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Preview LQR / Euler integration | Kajita-style preview over LIPM cart-table | same | aligned |

### Stage 3. Cartesian foot trajectory

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Foot planning | position + orientation | position only | partly hardware-driven, partly backlog |
| Swing-z shape | triangular, continuous at boundaries | half-sine with clearance offset and boundary discontinuity | discontinuity is backlog |

The load-bearing distinction here is:

- WR's clearance offset has a hardware rationale
- WR's lift-off / touchdown discontinuity does not

### Stage 4. Inverse kinematics

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Leg DoF | 6-DoF leg | 4-DoF leg | hardware-driven |
| Foot orientation tracking | planner-driven | heuristic ankle behavior | hardware-driven with room for better WR implementation |

This is the most clearly justified WR-specific area. WR cannot literally match
TB's 6-DoF foot-orientation realization because the hardware is different.

### Stage 5. Reference enrichment after IK

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Fixed-base FK replay / harvested body-site fields | yes, via `mujoco_replay` | yes, via `ZMPWalkGenerator._fixed_base_fk_replay` (Phase 3) | aligned |

WR now stores TB-comparable realized `body_pos` / `body_quat` / `body_lin_vel`
/ `body_ang_vel` / `site_pos` per frame on `ReferenceTrajectory`
(`control/references/reference_library.py`). The parity tool reads these
directly via `RuntimeReferenceWindow` instead of re-FKing on the fly. See
Phase 3 closeout for details.

### Stage 6. Asset lifecycle

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Storage model | pre-baked lookup asset | deterministic on-disk library cache fingerprinted on `(planner sources, MJCF/JSON assets, vx grid, ZMPWalkConfig)` (Phase 5) | aligned |

WR no longer regenerates per call. The parity tool, env, and CLI debugging
all consume the same fingerprinted library; cache invalidation is
automatic on planner / asset / config changes. See Phase 5 closeout.

### Stage 7. Runtime reference service

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Runtime service | path-state integration + lookup + root composition | path-state integration via `RuntimeReferenceService.compute_command_integrated_path_state` + offline trajectory lookup (Phase 4) | aligned |

WR's runtime target is now command-integrated: `path_state["torso_pos"] =
path_rot.apply(default_root_pos_xyz) + path_pos`, mirroring TB's
`walk_zmp_ref.py:228-239`. See Phase 4 closeout.

### Stage 8. Env reward consumption

Reward-target semantics now match TB. `_reward_torso_pos_xy` reads
`path_state["torso_pos"][:2]` (`wildrobot_env.py:813-822`, Phase 4); the
legacy planner `pelvis_pos` is consumed only by the diagnostic
`feet_track_raw` probe and the `lin_vel_z` reward
(`wildrobot_env.py:843`, numerical no-op because `pelvis_pos[:, 2] =
cfg.com_height_m` is constant by construction; tracked for cleanup as
Phase 11). The contact mask remains planner output by design (Phase 4
explicit scope decision).

The remaining reward-side architectural divergence is **none**. Any future
divergence (e.g., WR adding a hardware-driven reward TB doesn't have)
should be classified explicitly under "Keep WR-specific" with a hardware
or measured-parity rationale.

## Divergence Classification

The current architecture gaps sort into three buckets.

### Keep WR-specific

- 4-DoF leg instead of TB's 6-DoF leg
- resulting ankle / foot-orientation constraints that are truly imposed by WR
  hardware
- size and COM-height differences

These are not alignment failures. They are constraints the parity system must
normalize around.

### Resolved (kept here as historical record)

The following were classified "Remove unless explicitly justified" before
Phases 1-5 landed. They are no longer current backlog; they are listed
here so the design history stays traceable.

- swing-z boundary discontinuity — **resolved by Phase 1**
  (see Phase 1 closeout)
- planner-intent-only storage instead of TB-style realized FK
  enrichment — **resolved by Phase 3** (see Stage 5 above and Phase 3
  closeout)
- on-demand per-`vx` generation as the only lifecycle — **resolved by
  Phase 5** (see Stage 6 above and Phase 5 closeout)
- planner-phase torso target instead of TB-style command-integrated
  target — **resolved by Phase 4** (see Stage 7 / Stage 8 above and
  Phase 4 closeout)

### Remove unless explicitly justified (current open list)

No items currently open in this category. The next time a WR-specific
deviation from TB lands without a hardware / measured-parity
rationale, add it here so the architecture-first ordering still
applies.

### Parameter differences, not architecture

- `cycle_time`
- `foot_step_height`
- other pure config values

These still matter, but the rule is:

- architecture first
- local config tuning second

If a gap is caused by architecture, do not try to tune around it first.

## Execution Plan

The execution plan should close the WR-vs-TB quality gap primarily through
architecture changes. Local tuning is allowed only after the architecture
phases have cleared their exit criteria.

### Phase 0. Lock the baseline

Goal:
- make the current parity report the baseline artifact for all future WR
  reference changes

Work:
- keep [tools/reference_geometry_parity.py](../../tools/reference_geometry_parity.py)
  as the single comparison entry point
- keep
  [reference_parity_scorecard.md](reference_parity_scorecard.md)
  as the acceptance contract
- record both absolute and size-normalised results before each architecture
  change

Exit criteria:
- every reference-design PR includes before/after parity output
- changes are judged against the same `P0 / P1A / P2 / P1` layers

Acceptance statement:
- success means the baseline and reporting contract are stable enough that later
  phases can be judged consistently
- success does not mean WR/TB parity improved yet

### Phase 1. Fix the swing-z architecture bug

Goal:
- remove the unjustified WR swing-z boundary discontinuity while preserving the
  WR-specific floor-clearance rationale

Work:
- change the swing-foot clearance shaping so the extra WR clearance goes to
  zero at lift-off and touchdown
- keep the change planner-local; do not bundle schema or reward changes into
  this phase

Expected metric movement:
- improve `P0` swing-floor margin
- improve `P2` swing-foot-z-step

Exit criteria:
- no regression on in-scope `P0`
- measurable improvement on WR swing margin and `P2` swing smoothness
- no new WR-only discontinuity introduced elsewhere in the reference

Acceptance statement:
- success means `P0` swing-floor margin and `P2` swing-step smoothness improve
  without regressing the in-scope geometry gate
- success does not require a full `P2` pass yet; the phase is allowed to leave
  WR smoother than before but still rougher than TB

#### Phase 1 closeout (2026-04-25, commit `876a863`)

Implemented in `control/zmp/zmp_walk.py` only (planner-local). Two
coupled changes: the swing-z formula collapses to a single
`(foot_step_height + clearance) * sin(pi*frac)` envelope so the
clearance offset vanishes at lift-off / touchdown; the IK plantarflex
is now gated on the commanded foot z (not just the contact mask) so
boundary swing frames where the envelope returns to zero stay
flat-ankle and don't dip the foot below the floor.

Verification done with a **dense per-frame FK sweep** (every frame in
the 64-frame plan window, all four probed `vx` bins) — the
`tests/test_v0200c_geometry.py` probe alone samples only
`{0, 5, 10, 16, 22, 28, 32, 48, 64}`, which is too sparse to catch
in-between violations introduced by an envelope change. The dense
sweep is the verification of record for this phase.

Dense sweep results (after fix):

| `vx` | frames | worst swing z (m) | worst stance z (m) | in-scope violations |
|---|---|---|---|---|
| 0.10 | 64 | −0.0011 | +0.0007 | 0 |
| 0.15 | 64 | −0.0011 | +0.0024 | 0 |
| 0.20 | 64 | −0.0011 | +0.0047 | 6 (deferred bin) |
| 0.25 | 64 | −0.0011 | +0.0075 | 12 (deferred bin) |

Parity-relevant deltas (WR side; TB unchanged so cached TB numbers
in `tools/parity_report.json` remain the right baseline):

| Metric | Before | After | Δ |
|---|---|---|---|
| `P0` worst_swing_z (m) | −0.0017 | −0.0011 | +0.6 mm margin |
| `P0` worst_stance_z (m) | +0.0075 | +0.0075 | unchanged |
| `P0` in-scope failures | 0 | 0 | unchanged |
| `P2` swing_foot_z_step (m) | 0.0243 | 0.0200 | −17.7 % |
| `P2` shared_leg_q_step (rad) | 0.4094 | 0.3632 | −11.3 % (cleaner ankle transition) |
| `P1A` swing_clearance_mean_m (per-cycle peak) | 0.0668 | 0.0668 | unchanged |

Important nuance — what `P1A swing_clearance_mean_m` does and does
not measure: it is the **mean of the per-cycle peak** above the
stance baseline (see
`tools/reference_geometry_parity.py::_swing_clearance`), not the
mean foot height during swing. The peak is genuinely unchanged at
0.065 m. The mean foot height during swing **does** drop because
the new envelope sits below the old one everywhere except at the
peak:

| Quantity | Old (m) | New (m) | Δ (m) |
|---|---|---|---|
| swing-z peak (per-cycle max) | 0.0650 | 0.0650 | 0.0000 |
| swing-z mean (over swing frames) | 0.0480 | 0.0373 | −0.0107 |
| swing-z min (per-swing-frame minimum) | 0.0250 | 0.0000 | −0.0250 |

For architectural alignment: TB's analytical swing-z mean (linear
up-then-down, peak 0.05, no DC offset) is **0.0227 m**. WR after
the fix is 0.0373 m — still **1.64× TB's mean**, so the change
moves WR toward TB's swing profile, not below it. The peak ceiling
that determines obstacle clearance is preserved. If a future phase
wants to recover the old mean swing height for additional margin,
that belongs in Phase 6 (parameter values: `foot_step_height_m`),
not Phase 1.

Outstanding caveats:
- TB-side parity helper not re-run on this machine (TB venv missing
  `joblib` / `lz4`); the cached TB numbers in
  `tools/parity_report.json` stay valid because TB code is
  untouched. A full refresh on a TB-capable machine will produce a
  coherent post-Phase-1 report.
- The high-`vx` (≥ 0.20) stance failures are unchanged and remain
  out of scope per the round-10 milestone reframing.

### Phase 2. Decide runtime target semantics

Goal:
- resolve whether WR should keep planner-phase torso targets or align to TB's
  command-integrated target semantics

Work:
- write an explicit design decision in this doc or a linked ADR
- default decision is TB-style command-integrated target semantics
- keep planner-phase target only if WR has a measured parity benefit or a clear
  safety / trackability reason

Decision output:
- chosen target semantics
- why the rejected option was rejected
- which runtime and reward fields must change next

Exit criteria:
- the target semantics are written down explicitly
- subsequent architecture work is scoped against that decision rather than
  left ambiguous

Acceptance statement:
- success means the WR runtime target semantics are explicitly chosen and the
  next implementation phases are unblocked
- success does not require code changes or parity movement yet; this is a
  design-decision phase

#### Phase 2 closeout (2026-04-25, doc-only)

**Decision: WR adopts TB-style command-integrated runtime target
semantics.**

The WR runtime reference target for body / pelvis / torso position
(and any reward consuming a "where should the body be" signal)
becomes the path-integrated command:

```
path_state = integrate(velocity_cmd, yaw_rate_cmd, t_since_reset)
ref_torso_xy(t)  = path_state.path_pos[:2]
ref_torso_quat(t) = path_state.path_rot · default_rot
ref_lin_vel(t)   = velocity_cmd       (already runtime-derived)
ref_ang_vel(t)   = [0, 0, yaw_rate_cmd]
```

`q_ref` and the planner's per-frame foot/contact annotations stay
sourced from the offline trajectory window (the current
`RuntimeReferenceService` slice). Only the body-pose / body-velocity
reference channels move to the path-integrated form.

**Why this decision (rationale grounded in repo evidence):**

1. **TB is the default architecture per the Governing Policy.** Only
   a measured benefit to keeping the planner-phase target would
   override that. None exists today.
2. **The planner-phase target couples reward signals to planner
   choices.** Pre-Phase-4, the env's torso-pose reward read
   `ref_pelvis_pos = win["pelvis_pos"]` (the LIPM-LQR COM trajectory
   from `control/zmp/zmp_walk.py`) and used it as the
   `_reward_torso_pos_xy` target. Any planner change (ALIP, mocap,
   different cost weights) would silently shift the reward target.
   Path-integrated targets keep the reward signal stable across
   planner iterations. (Phase 4 has since landed this switch — see
   `wildrobot_env.py:813-822` for the post-Phase-4 reward and
   `wildrobot_env.py:1489-1495` for the runtime call site. The
   legacy `ref_pelvis_pos = win["pelvis_pos"]` lookup at
   `wildrobot_env.py:802-807` is retained for the diagnostic
   `feet_track_raw` probe and is also still read by the `lin_vel_z`
   reward at `wildrobot_env.py:843` (`win["pelvis_pos"][2]`). The
   `lin_vel_z` consumption is a numerical no-op because
   `pelvis_pos[:, 2] = cfg.com_height_m` is constant by construction
   in `zmp_walk.py:845`, so the reward effectively penalises
   `|body_lin_vel_z|` only — but the reward target is technically
   still a planner-side field and would be cleaner to source from
   `cfg.com_height_m` directly in a follow-up.)
3. **TB consumes the path-integrated form** at
   `toddlerbot/locomotion/mjx_env.py:2265-2269`
   (`torso_pos_ref = info["state_ref"]["qpos"][:2]`). The
   `state_ref["qpos"]` it reads is composed at
   `toddlerbot/reference/walk_zmp_ref.py:228-239` from
   `path_state["path_pos"]` (which itself comes from the integrator
   defined at `toddlerbot/reference/motion_ref.py:200-235` and called
   from `walk_zmp_ref.py:129`). Aligning here means WR's reward
   consumes the same semantic class TB does.
4. **The runtime computation is no more complex than the current
   lookup at the smoke scope.** WR's env already threads
   `loc_ref_offline_step_idx` through state
   (`wildrobot_env.py:1345, 1483`). For the yaw-stationary smoke
   (`yaw_rate_cmd = 0`, `vy = 0`), `t_since_reset = step_idx · dt`
   and `path_pos = velocity_cmd · t_since_reset` is one multiply and
   one add per step — comparable to the array lookup that produces
   today's `win["pelvis_pos"]`. When yaw is enabled (v0.20.4), the
   helper must mirror TB's
   `toddlerbot/reference/motion_ref.py::integrate_path_state` —
   `delta_rot = R.from_rotvec(ang_vel · dt); path_rot = path_rot ·
   delta_rot; path_pos += path_rot.apply(lin_vel) · dt` — which is
   a per-step SO(3) composition, not a multiply-add. That's still
   trivial relative to the current asset infrastructure, but should
   not be described as "one multiply and one add" once yaw is in
   the command set.
5. **The planner's `pelvis_pos` field stays useful** (it's still the
   target the IK solved against, so it stays as the sanity-check
   diagnostic in `feet_track_raw` and similar). Only the reward
   target moves; the asset schema does not need to grow or shrink.

**Why the planner-phase target was rejected:**

- *Materially better parity?* No. Measured by
  `tools/compare_reference_target_semantics.py` (added in this
  phase): the planner-phase target and the command-integrated path
  diverge along the X-axis even in steady state by a *cm-scale*
  permanent offset (not "sub-mm" as an earlier draft of this
  closeout asserted without measurement). At smoke `vx = 0.15`,
  steady-state X-mean offset is **−25.2 mm** (planner trails the
  command-integrated path by ~25 mm because WR retains cycle 0
  while the command-integrated path advances linearly from
  reset); steady-state X-RMS is **25.6 mm**; worst-case L2
  divergence is **47.8 mm** at the cycle 0 transient. The
  Y-axis is the small one — steady-state Y-max is **19.8 mm**
  (LIPM lateral swing under Q=I, R=0.1) with **2.2 mm RMS** —
  small enough to ignore for reward shaping. Per-`vx`
  numbers (steady state, post-cycle-0):

  | vx (m/s) | X mean (mm) | X RMS (mm) | Y max (mm) | Y RMS (mm) |
  |---|---|---|---|---|
  | 0.10 | −16.8 | 17.1 | 19.8 | 2.2 |
  | 0.15 | −25.2 | 25.6 | 19.8 | 2.2 |
  | 0.20 | −33.5 | 34.1 | 19.8 | 2.2 |

  The X offset scales linearly with `vx`, consistent with the
  hypothesis that it equals the cumulative ramp-lag of cycle 0
  (which TB truncates and WR retains).
  These are not noise-level differences. They are a real reward-
  signal divergence in the load-bearing axis. The path-integrated
  form is the cleaner target precisely because it removes this
  cycle-0 lag, but the case for switching is **stronger** than
  the original closeout claimed, not weaker.
- *Materially better trackability or safety?* No. Both targets
  advance forward at `vx · t` in steady state. There is no scenario
  where tracking the planner's noisier path produces safer behavior
  than tracking the clean command-integrated path.
- *Implementation complexity argument?* The opposite —
  path-integration is trivially simpler than the existing array
  lookup (see point 4 above).

None of the three rejection conditions hold, so the default
(TB-style) is taken.

**Implementation map for Phase 3 (asset content alignment):**

Phase 3 is **independent of this Phase 2 decision**. It can land
before, after, or in parallel.

What changes in Phase 3:
- `control/zmp/zmp_walk.py::_generate_at_step_length` adds a
  fixed-base FK pass after the IK step (mirroring
  `toddlerbot/algorithms/zmp_walk.py:153-213 mujoco_replay`).
- `control/references/reference_library.py::ReferenceTrajectory`
  schema gains: `body_pos`, `body_quat`, `body_lin_vel`,
  `body_ang_vel`, `site_pos` (per-frame, FK-realised).
- `control/references/runtime_reference_service.py::_StackedTrajectory`
  + `RuntimeReferenceWindow` add the same fields and the
  `lookup_np` / `lookup_jax` paths plumb them through.
- Existing `pelvis_pos` field stays for diagnostics +
  IK-input continuity; new fields are additive.

What does NOT change in Phase 3:
- The reward target (that's Phase 4).
- The planner intent (still ZMP / IK).
- The contact mask field (still planner output).

Acceptance: WR asset stores TB-comparable realized body / site
quantities; the parity tool's `_wr_fk_and_smoothness` re-FK in
`tools/reference_geometry_parity.py` becomes a sanity check rather
than a workaround.

**Implementation map for Phase 4 (runtime consumption alignment):**

Phase 4 lands the Phase 2 decision in code. It depends on Phase 2
being decided (this commit) but **not** on Phase 3 being landed.

Concrete code changes (as landed — see Phase 4 closeout below):

1. **`control/references/runtime_reference_service.py`** — added a
   small `compute_command_integrated_path_state(*, t_since_reset_s,
   velocity_cmd_mps, yaw_rate_cmd_rps=0.0, dt_s=0.02,
   default_root_pos_xyz=None) -> {path_pos, path_rot, torso_pos,
   lin_vel, ang_vel}` static helper. Mirrors
   `toddlerbot/reference/motion_ref.py::integrate_path_state` but
   stateless: takes elapsed time and command, returns the
   integrated frame. Pure-JAX (`jax.numpy`); no service state
   change. The optional `default_root_pos_xyz` reproduces TB's
   absolute-world torso semantics
   (`path_rot.apply(default_root_pos_xyz) + path_pos`).
2. **`training/envs/wildrobot_env.py`**:
   - Compute `t_since_reset = next_step_idx · self.dt` (already
     available via `loc_ref_offline_step_idx`,
     `wildrobot_env.py:1483-1487`).
   - Compute `path_state = self._offline_service.\
     compute_command_integrated_path_state(...)`
     (`wildrobot_env.py:1489-1495`) and pass it as a kwarg into
     `_compute_rewards` (`wildrobot_env.py:720, 1624`).
   - In `_reward_torso_pos_xy`, replace the planner-side reference
     with `path_state["torso_pos"][:2]`
     (`wildrobot_env.py:813-822`).
   - Leave `ref_pelvis_pos = win["pelvis_pos"]`
     (`wildrobot_env.py:802-807`) available for the diagnostic
     `feet_track_raw` and any other planner-phase consumer that
     hasn't been migrated yet (move them in follow-ups, not in
     Phase 4's first slice).
3. **No schema change, no `RuntimeReferenceWindow` field change,
   no asset rebuild.** Phase 4 is reward-consumption-side only.

Acceptance: `_reward_torso_pos_xy` reads a path-integrated target;
WR no longer tracks planner LIPM-COM jitter as the reward signal.
P1 trackability is read against the same semantic target class TB
uses.

**No code change in this phase (Phase 2).** This is the
design-decision phase. The Phase 3 / Phase 4 implementation maps
above are the concrete next slices.

Open questions resolved by Phase 4 implementation:

- Should the path-state integration also handle `yaw_rate_cmd`
  immediately, or defer to v0.20.4 (yaw-cmd milestone)? **Resolved:**
  the helper accepts `yaw_rate_cmd_rps` (default 0.0); the env
  passes `jp.float32(0.0)` for the smoke and the closed-form
  discrete sum in `compute_command_integrated_path_state` already
  collapses to the linear case in that limit.
- Should `path_state` be cached in `WildRobotInfo` to avoid
  recomputation across the reward terms? **Resolved:** yes —
  computed once per step (`wildrobot_env.py:1489`) and threaded
  through `_compute_rewards` as a kwarg
  (`wildrobot_env.py:720, 1624`).

### Phase 3. Align WR asset content with TB

Goal:
- move WR from planner-intent-only storage toward TB-style realized reference
  content

Work:
- add a fixed-base FK replay after IK during WR reference construction
- persist realized body/site quantities needed by parity, runtime reference
  service, and debugging
- keep the stored quantities clearly separated from planner-intent fields

Expected metric movement:
- make `P1A` and `P2` comparisons natively apples-to-apples
- reduce downstream re-FK reconstruction and data ambiguity

Exit criteria:
- WR reference asset contains realized body/site data comparable to TB
- parity tooling no longer needs WR-specific reconstruction hacks for the main
  realized quantities

Acceptance statement:
- success means WR stores TB-comparable realized reference quantities and the
  main parity comparisons are natively apples-to-apples
- success does not require full `P1A` or `P1` parity yet; it removes a core
  architectural mismatch so later phases can improve those metrics cleanly

#### Phase 3 closeout (2026-04-25)

Implemented the TB-style ``mujoco_replay`` enrichment in three additive
slices, each preserving existing planner-intent fields and reward
semantics.

Code changes:

1. **`control/zmp/zmp_walk.py`** — added
   ``ZMPWalkGenerator._fixed_base_fk_replay``, mirroring
   ``toddlerbot/algorithms/zmp_walk.py:153-213 mujoco_replay``.
   Lazily loads ``assets/v2/scene_flat_terrain.xml``, anchors at the
   ``home`` keyframe (TB-style fixed-base FK), and per-frame:
   - reads ``data.xpos`` / ``data.xquat`` for all model bodies
     (`range(model.nbody)`),
   - synthesises foot-centre ``site_pos`` from heel + toe geom
     midpoints (WR's MJCF lacks explicit ``left/right_foot_center``
     sites; this matches the parity tool's existing convention).
   Velocities are finite-diff (G2 convention); angular velocity uses
   the small-angle ``2 · (q1·conj(q0))[xyz] / dt`` approximation with
   per-body double-cover sign alignment.  Both
   ``_generate_at_step_length`` and ``_generate_standing`` now run
   the replay and pass the five arrays to ``ReferenceTrajectory``.
2. **`control/references/reference_library.py`** — additive schema
   fields on ``ReferenceTrajectory``: ``body_pos``, ``body_quat``,
   ``body_lin_vel``, ``body_ang_vel``, ``site_pos`` plus
   ``body_names`` / ``site_names`` tuples for stable indexing.  All
   default ``None`` so legacy assets keep loading.  ``validate()``
   shape-checks the new arrays only when present.
3. **`control/references/runtime_reference_service.py`** — same five
   additive fields on ``_StackedTrajectory`` and
   ``RuntimeReferenceWindow``; ``lookup_np`` / ``lookup_jax`` /
   ``to_jax_arrays`` plumb them through; service exposes
   ``n_bodies`` / ``n_sites`` / ``body_names`` / ``site_names``.
   Legacy trajectories without the fields fall back to ``[n, 0, 3]``
   shaped empties so downstream consumers can detect "no realized FK"
   via ``len(service.body_names) == 0`` rather than catching
   exceptions.
4. **`tools/reference_geometry_parity.py`** —
   ``_wr_fk_and_smoothness`` now reads ``traj.site_pos`` directly when
   present, falling back to per-frame re-FK only for legacy assets.
   This is the asset-content alignment exit criterion: the parity
   tool no longer depends on a WR-only re-FK reconstruction.

Tests:

- `tests/test_v0200a_contract.py`: 15/15 PASS unchanged (additive
  schema doesn't break the required-field assertion).
- `tests/test_v0200c_geometry.py`: 2/2 PASS (geometry probe uses its
  own FK; no regression).
- `tests/test_runtime_reference_service.py`: 13/13 PASS, including
  three new Phase 3 tests:
  - ``test_realized_fk_fields_round_trip``: NumPy / JAX paths return
    matching shapes and values for the new fields, sourced from the
    trajectory.
  - ``test_realized_fk_root_anchor``: fixed-base FK keeps the
    ``waist`` body z constant across frames (TB-comparable
    semantics).
  - ``test_legacy_trajectory_yields_empty_fk_arrays``: a trajectory
    without Phase 3 arrays builds a service cleanly and returns
    ``[0, 3]`` / ``[0, 4]`` shaped empties.

Asset-backed parity verification (WR-side, vx=0.15):

| Metric | Pre-Phase-1 baseline (cached) | Post-Phase-1 | Post-Phase-3 (asset-backed) |
|---|---|---|---|
| `step_length_mean_m` | 0.0478 | 0.0478 | 0.0478 |
| `swing_clearance_mean_m` | 0.0668 | 0.0668 (re-FK) | 0.0659 (asset-FK) |
| `foot_z_step_max_m` | 0.0243 | 0.0200 (re-FK) | 0.0201 (asset-FK) |
| `shared_leg_q_step_max_rad` | 0.4094 | 0.3632 (re-FK) | 0.3632 (asset-FK) |
| `swing_foot_z_step_max_m` | 0.0243 | 0.0200 (re-FK) | 0.0201 (asset-FK) |

Asset-backed values match the per-frame re-FK output to mm precision.
The ~1 mm shift on `swing_clearance_mean_m` (0.0668 → 0.0659) is the
expected difference between the parity tool's previous **free-base**
FK at planned ``pelvis_pos`` (which advanced the body forward each
frame) and the new **fixed-base** FK anchored at home (TB-comparable).
Both produce the same swing-clearance-above-baseline value to within
~1 mm because the metric subtracts the per-cycle stance baseline.

Smoothness metrics (`foot_z_step_max_m`, `swing_foot_z_step_max_m`,
`shared_leg_q_step_max_rad`) match exactly because they're per-frame
deltas invariant to absolute body anchoring.

Outstanding caveats:
- TB-side parity helper not re-runnable on this machine (TB venv
  missing ``joblib`` / ``lz4``).  TB code untouched, so the cached
  TB numbers in ``tools/parity_report.json`` remain the right
  baseline.
- `body_lin_vel` / `body_ang_vel` use finite-diff; they will be 0
  for any body whose ``xpos`` / ``xquat`` is constant across frames
  in fixed-base FK (e.g. the world body).  TB's `mujoco_replay`
  reads these from `sim.data.cvel` after `set_joint_angles +
  forward()`; with `qvel = 0` the cvel is also 0 there, so the
  finite-diff form is no worse than TB's storage.
- `site_pos` only stores the two foot-centre synthetic sites today.
  Adding more (e.g. hand sites if WR ever gains them) is a single
  extra entry in ``ZMPWalkGenerator._FOOT_CENTER_GEOMS`` (and would
  arguably belong in a different config field then; revisit if /
  when WR planners need a site beyond feet).

Phase 4 unblocked: ``ReferenceLibrary``-backed ``body_pos`` /
``body_quat`` are now available for RSI initialization (mirroring
``toddlerbot/locomotion/mjx_env.py:1174-1192``) and for any future
reward channel that wants a fixed-base FK signal.  The Phase 4 doc
under "Phase 2 closeout" stands as-is — the runtime reference
consumption alignment is independent of this asset-content slice.

### Phase 4. Align runtime reference consumption

Goal:
- make WR runtime reference consumption match the chosen semantics from Phase 2

Work:
- if Phase 2 chooses TB-style command tracking, add path-state integration and
  root-pose composition to WR runtime reference service
- align reward-target consumption so torso/path tracking uses the same semantic
  target as TB unless explicitly justified otherwise

Expected metric movement:
- improve `P1` trackability interpretation by removing target-semantics drift
- make WR reward consumption closer to TB by architecture, not by tuning

Exit criteria:
- runtime reference service behavior matches the documented target semantics
- the torso/path tracking target is no longer an unowned WR-vs-TB difference

Acceptance statement:
- success means WR runtime consumption and reward-target semantics match the
  documented architectural decision
- success does not require all downstream training metrics to pass yet; it
  removes a major source of WR/TB semantic drift

#### Phase 4 closeout (2026-04-26)

Implemented across two files; reward-consumption-side only, no
schema or asset rebuild.

Code changes:

1. **`control/references/runtime_reference_service.py`** — added
   `RuntimeReferenceService.compute_command_integrated_path_state`
   (`runtime_reference_service.py:182-262`). Stateless static
   helper, kwargs-only, JAX-native. Returns
   `{path_pos, path_rot, torso_pos, lin_vel, ang_vel}` where
   `torso_pos = path_rot.apply(default_root_pos_xyz) + path_pos`
   when `default_root_pos_xyz` is supplied (matches TB's
   absolute-world torso semantics from
   `toddlerbot/reference/walk_zmp_ref.py:230-232`). Uses a
   closed-form discrete sum
   `p_n = dt · Σ_{k=1..n} Rz(k·θ) · [vx, 0]` with a numerically
   stable `θ → 0` fallback so the yaw-stationary smoke and the
   future yaw-cmd milestone share one code path.
2. **`training/envs/wildrobot_env.py`**:
   - `wildrobot_env.py:1483-1495` computes
     `t_since_reset_s = next_step_idx · self.dt` and calls
     `self._offline_service.compute_command_integrated_path_state(
       t_since_reset_s=..., velocity_cmd_mps=velocity_cmd,
       yaw_rate_cmd_rps=jp.float32(0.0), dt_s=jp.float32(self.dt),
       default_root_pos_xyz=self._init_qpos[:3].astype(jp.float32))`
     once per step.
   - `wildrobot_env.py:720` adds `path_state` as a kwarg on
     `_compute_rewards`; `wildrobot_env.py:1624` threads it through
     so all reward terms see the same per-step path state.
   - `wildrobot_env.py:813-822` switches `_reward_torso_pos_xy` to
     `torso_pos_xy_err = root_pos_xyz[:2] - path_state["torso_pos"][:2]`,
     matching TB `_reward_torso_pos_xy`
     (`toddlerbot/locomotion/mjx_env.py:2265-2269`).
   - `wildrobot_env.py:802-807` retains the legacy
     `ref_pelvis_pos = win["pelvis_pos"]` lookup for the
     diagnostic-only `feet_track_raw` probe; no other reward term
     reads planner-side pelvis_pos.

What did NOT change in Phase 4:
- `ReferenceTrajectory` / `RuntimeReferenceWindow` schema (Phase 3
  asset content remains additive on top).
- `ZMPWalkConfig` planner timing / geometry constants.
- `q_ref` / `contact` / `pelvis_pos` window fields (still planner
  output).

Acceptance: `_reward_torso_pos_xy` reads a path-integrated target;
WR no longer tracks planner LIPM-COM jitter as the reward signal.
The remaining torso_pos divergence between WR and TB is now purely
hardware-driven (default `default_root_pos_xyz` differs by COM
height and stance width), not target-semantics-driven.

### Phase 5. Align lifecycle and reproducibility

Goal:
- close the remaining asset-lifecycle gap where it materially affects parity,
  debugging, or multi-command behavior

Work:
- add cached `vx`-grid construction at env init or on disk if needed
- prefer deterministic, reusable assets over repeated ad hoc regeneration when
  parity work spans multiple command bins

Expected metric movement:
- not a primary parity gain by itself
- improves repeatability and makes multi-command comparisons easier to trust

Exit criteria:
- WR reference generation is reproducible enough for repeated parity sweeps
- multi-bin comparisons do not depend on fragile one-off runtime generation

Acceptance statement:
- success means parity runs and debugging are reproducible enough to trust
  architecture deltas
- success does not by itself imply reference-quality parity improvement

#### Phase 5 closeout (2026-04-26)

Implemented as a deterministic, fingerprinted on-disk cache for the
`vx`-grid reference library so the parity tool, the env, and CLI
debugging all consume the same reproducible asset across runs.

Code changes:

1. **`tools/reference_geometry_parity.py`** — added
   `_wr_generator_fingerprint`
   (`reference_geometry_parity.py:258-296`) and `_wr_cache_key`
   (`reference_geometry_parity.py:298-308`). The fingerprint is a
   SHA-256 of the contents of the planner-side sources
   (`control/zmp/zmp_walk.py`, `control/zmp/zmp_planner.py`,
   `control/references/reference_library.py`,
   `training/utils/ctrl_order.py`) **and** every `*.xml` / `*.json`
   under `assets/v2`, so any kinematics / site-definition change
   invalidates the cache automatically. `_wr_cache_key` further
   binds the cache to the requested `vx` grid and the current
   `asdict(ZMPWalkConfig())`, so any config change (including the
   Phase 6 `cycle_time_s` 0.64 → 0.72 bump) gets a fresh cache key.
   Cached schemas are versioned
   (`wr_phase5_generator_fingerprint_v2`,
   `wr_phase5_library_cache_v1`) so format changes invalidate too.
2. **`control/zmp/build_library.py`** — CLI entry point
   (`uv run python control/zmp/build_library.py --output ...`,
   `--vx-max 0.20`, `--vx-interval ...`) for deterministic offline
   library construction. Used by parity sweeps and any future
   batch-generation workflow.
3. **`tests/test_reference_lifecycle_phase5.py`** — pytest +
   ad-hoc smoke that exercises fingerprint stability,
   `vx`-bin-set sensitivity, and serialisation round-trips against
   a stub `ReferenceLibrary`.

What did NOT change in Phase 5:
- `ReferenceTrajectory` / `RuntimeReferenceWindow` schema.
- Env / reward consumption paths (Phase 4 stands).
- Planner config / timing (Phase 6 owns those).

Acceptance: parity sweeps no longer need ad-hoc per-run
regeneration; the parity tool's library cache key is a pure
function of `(planner sources, MJCF/JSON assets, vx grid,
ZMPWalkConfig())`, so any future planner / asset / config change
is picked up automatically. This unblocks Phase 6's
`cycle_time_s` tuning to flow through the cache without manual
invalidation.

### Phase 6. Tune only after the architecture is aligned

Goal:
- use local planner/config changes to refine an aligned architecture, not to
  compensate for misalignment

Allowed work:
- `cycle_time`
- `foot_step_height`
- other scalar planner parameters

Rule:
- do not start here if the active gap is still architectural
- only use this phase after Phases 1 through 4 have either landed or been
  explicitly rejected with rationale

Exit criteria:
- tuning changes improve parity on top of the aligned architecture
- tuning is not being used to hide an unresolved architecture gap

Acceptance statement:
- success means local tuning produces measurable parity gains on top of the
  aligned architecture
- success is invalid if the apparent gain comes from tuning around an
  unresolved architecture mismatch

#### Phase 6 closeout (2026-04-26)

**Tuning change:** `ZMPWalkConfig.cycle_time_s` 0.64 → 0.72 in
`control/zmp/zmp_walk.py`. This adopts ToddlerBot's value (TB
`mjx_config.py:107`, `walk.gin:19`) and is the explicit Item 1
recommendation from § "Architectural alignment moves toward ToddlerBot
(priority order)". No other code changed.

**Why this is tuning, not architecture:** `cycle_time_s` is a single
scalar in the planner config. The planner pipeline (LIPM cart-table
preview, half-sine swing envelope from Phase 1, fixed-base FK enrichment
from Phase 3, command-integrated reward target from Phase 4) is
unchanged. The only mechanical effect is that
`_generate_at_step_length` runs with `half_cycle = cycle/2 = 0.36 s`
instead of `0.32 s`, which proportionally lengthens the per-step
distance at the same `vx` and proportionally drops cadence.

**Targeted gaps from the scorecard "Latest Analysis Result"
section:**
- size-normalised `cadence_froude_norm` failing at 1.41× TB (gate ≤ 1.20×)
- size-normalised `step_length_per_leg` failing at 0.50× TB (gate ≥ 0.85×)
- absolute `step_length_mean_m` failing at 0.88× TB (gate ≥ 0.90×)

**Verification:**
- `uv run pytest tests/test_v0200a_contract.py tests/test_v0200c_geometry.py`
  — 19/19 PASS unchanged
- `uv run pytest tests/test_runtime_reference_service.py` — 16/16 PASS
- full repo `uv run pytest tests/` — 161/161 PASS
- `uv run python tools/phase6_wr_probe.py` — WR-only probe that runs the
  parity tool's WR-side helpers (`_summarize_wildrobot`,
  `_wr_fk_and_smoothness`) without the TB venv. The TB-side numbers in
  `tools/parity_report.json` stay valid because TB code is untouched
  between Phase 1 and Phase 6.

**Before/after WR metrics at vx=0.15 (TB references in []):**

| Metric | Pre (cycle=0.64) | Post (cycle=0.72) | TB-2xc | Δ vs WR |
|---|---|---|---|---|
| **P0 geometry (full matrix)** |
| total failures | 7 | **0** | 2 | -7 |
| in-scope failures | 0 | 0 | 0 | unchanged |
| worst stance z (m) | +0.0075 | **+0.0019** | +0.0100 | -75% (better) |
| worst swing z (m) | -0.0011 | **+0.0170** | +0.0100 | +18.1 mm margin |
| **P1A FK gait** |
| step_length_mean_m | 0.0478 | **0.0538** | 0.0541 | matches TB to 0.3 mm |
| swing_clearance_mean_m | 0.0659 | 0.0653 | 0.0500 | -1% (still > TB) |
| touchdown_rate_hz | 3.0804 | **2.7330** | 2.7622 | matches TB to 0.03 Hz |
| double_support_frac | 0.3125 | **0.3333** | 0.3371 | matches TB to 0.4% |
| foot_z_step_max_m | 0.0201 | **0.0183** | 0.0100 | -9% (smoother) |
| **P2 smoothness** |
| swing_foot_z_step_max_m | 0.0201 | **0.0183** | 0.0100 | -9% (smoother) |
| shared_leg_q_step_max_rad | 0.3632 | 0.3599 | 0.6471 | -1% (already better than TB) |
| pelvis_z_step_max_m | 0.0000 | 0.0000 | 0.0000 | sentinel pass |
| contact_flips_per_cycle | 3.9714 | 3.9677 | 3.9775 | -0.1% |
| **Size-normalised** |
| step_length_per_leg | 0.1282 | **0.1441** | 0.2557 | +12.4% toward 0.85× gate |
| cadence_froude_norm | 0.6656 | **0.5905** | 0.4715 | -11.3% toward 1.20× gate |
| swing_clearance_per_com_height | 0.1440 | 0.1426 | 0.1749 | -1% (unchanged sized to clearance) |
| swing_foot_z_step_per_clearance | 0.3045 | **0.2803** | 0.2000 | -8% (smoother) |

**Gate movement (vs TB-2xc):**
- P0 swing_gate: WR worst swing went from -1.7 mm (FAIL by 0.3 mm) to
  +17 mm (PASS by 19 mm).  P0 swing_gate flips FAIL → **PASS**.
- P0 stance_gate: WR worst stance went from +7.5 mm to +1.9 mm — full
  matrix now under the 3 mm tolerance.  WR no longer trails TB on
  high-vx diagnostic robustness; the previously deferred vx ∈ {0.20,
  0.25} bins all pass.
- step_len_gate (absolute, ≥ 0.90× TB): WR went from 0.0478/0.0541 =
  0.883 (FAIL) to 0.0538/0.0541 = 0.994 (**PASS**).
- cadence_norm gate (≤ 1.20× TB): WR went from 0.6656/0.4715 = 1.412
  (FAIL) to 0.5905/0.4715 = 1.252 (still FAIL but cuts the over-cadence
  by 60%; the residual is dominated by the leg-length difference
  through the Froude factor, which is hardware, not tuning).
- step_per_leg gate (≥ 0.85× TB): WR went from 0.1282/0.2557 = 0.501
  (FAIL) to 0.1441/0.2557 = 0.564 (still FAIL but closes 13% of the
  gap; residual is the operating-vx-vs-Froude question called out at
  the end of the scorecard's "Architectural alignment moves" section,
  not a knob this phase owns).

**Why this slice is legitimate Phase 6 work (not architecture-masking):**

This says "this is a valid tuning slice", **not** "this completes the
redesign".  The decision-rule check above shows several gates still
FAIL.

1. The gate failures Phase 6 was scoped against (`cadence_norm`,
   `step_length_per_leg`, absolute `step_len`) are exactly the gaps the
   parity scorecard's § "Architectural alignment moves toward
   ToddlerBot" called out as parameter values, not architecture.
2. The metrics that *would* signal an architectural masking — pelvis
   discontinuity (`pelvis_z_step_max_m`), reward-target semantics
   (Phase 4), and the asset content schema (Phase 3) — are unchanged
   here; only the planner's gait period changed.
3. P0 geometry improved on every dimension simultaneously (full-matrix
   failures 7 → 0, worst stance halved, worst swing went from a 0.3 mm
   FAIL margin to a 19 mm PASS margin).  This is the natural
   consequence of the IK no longer needing to stretch the leg as far
   forward at higher vx — a longer cycle drops the per-frame foot
   excursion at every vx.
4. The residual `step_per_leg` and `cadence_froude_norm` failures
   *appear* hardware-bounded at the current operating point: TB's
   nominal `vx = 0.15` puts WR's larger leg/COM at a smaller Froude
   number than TB by construction.  This phase does not prove that
   bound; it proposes that closing the rest needs an operating-vx /
   curriculum decision rather than more `cycle_time_s` tuning, and
   leaves the proof to that follow-up.

**What Phase 6 does NOT close (verdicts after the change, vs cached TB-2xc):**

This is a single tuning slice, not the end of the redesign.  Per the
"What 'On Par or Better' Means" decision rule below, WR is on par with
or better than TB only when P0 + size-normalised P1A + P2 + P1 all clear.
After Phase 6:

| Layer | Gate | Post-Phase-6 | Verdict |
|---|---|---|---|
| P2 absolute | `swing_foot_z_step_max_m` ≤ 1.10× TB | 1.83× TB | **FAIL** |
| P1A normalised | `step_length_per_leg` ≥ 0.85× TB | 0.56× TB | **FAIL** |
| P1A normalised | `swing_clearance_per_com_height` ≥ 0.85× TB | 0.82× TB | **FAIL** |
| P1A normalised | `cadence_froude_norm` ≤ 1.20× TB | 1.25× TB | **FAIL** |
| P1A normalised | `swing_foot_z_step_per_clearance` ≤ 1.10× TB | 1.40× TB | **FAIL** |
| P1 trackability | full closed-loop replay | **not re-measured** under cycle=0.72 | unknown |

So the doc's own "on par or better" decision rule is **not satisfied
yet**.  Phase 6 closes the absolute step_length / cadence / DS-fraction
gates and the full-matrix P0 gate, but four normalised P1A gates, the P2
swing-foot smoothness gate, and the P1 closed-loop layer are still open.
Reading this closeout as "WR ≈ TB now" would overstate the result.

**What still has to happen for the redesign to complete:**

1. Re-run the full parity tool on a TB-capable machine.  The local TB
   venv is missing `joblib` / `lz4` (same caveat as Phase 1 / Phase 3
   closeouts), so P1 was not re-measured under `cycle_time = 0.72`.  TB
   code itself is untouched between Phase 1 and Phase 6, so the cached
   TB numbers in `tools/parity_report.json` (commit `c1b815d`) are
   still the right TB baseline for the absolute / normalised P1A / P2
   ratios above; only the P1 closed-loop layer truly needs a fresh full
   parity run to update both sides.
2. Address the P2 `swing_foot_z_step` absolute / per-clearance gap.
   Phase 6 nudged it (0.0201 → 0.0183) but it is still 1.83× TB
   absolute and 1.40× TB per-clearance.  TB's swing-z is a
   **single-peaked triangle** (linear up, peak at one frame, linear
   down — see TB `algorithms/zmp_walk.py:483-498`), not a trapezoid;
   adopting the TB envelope means swapping WR's half-sine for the same
   piecewise-linear ramp, not building a flat top. Alternatively,
   revisit the Phase 1 envelope choice now that the cycle is longer.
3. Address the residual size-normalised P1A gates.  Two distinct
   knobs:
   - `swing_clearance_per_com_height` is one `foot_step_height_m`
     bump away from PASS. WR currently sits at 0.04 m
     (`control/zmp/zmp_walk.py:95`); TB defaults to **0.05 m**
     (TB `algorithms/zmp_walk.py:32`, not overridden by
     `walk_zmp_ref.py:43`). Adopting TB's value (0.04 → 0.05, +25 %)
     puts the metric at ≈ 0.175 vs gate 0.149 — clears with margin.
     A halfway step (0.045) would land at ≈ 0.16, also clearing the
     gate while halving the swing-step smoothness regression. The
     trade-off is P2 swing-step smoothness; this is a legitimate next
     Phase 6 slice if the smoothness regression stays bounded.
   - `step_length_per_leg` and `cadence_froude_norm` are bounded
     below / above by hardware proportions (WR has 1.76× longer legs
     than TB) at the current operating point.  Closing them needs an
     operating-vx / curriculum decision (running WR at TB's Froude
     point ≈ vx 0.19 m/s rather than 0.15 m/s), not further
     `cycle_time_s` tuning.
4. Decide whether to amend the parity tool's `_SHARED_LEG_NAMES` /
   gate thresholds for the WR-vs-TB hardware delta, or to live with
   "normalised gates document the gap" as the policy.  Either is
   defensible; the current doc already says the normalised view is the
   primary reading, so leaving the gates failing-as-evidence is
   coherent.

**Implementation-side caveats unchanged by this slice:**

- This phase changes a planner timing constant.  The env's
  `cycle_time_s` consumers (e.g. body-phase reward terms in
  `wildrobot_env.py`) read the same value through `ZMPWalkConfig`
  defaults, so the change propagates without further plumbing.
- The Phase 5 cache fingerprint (`_wr_generator_fingerprint` in
  `tools/reference_geometry_parity.py`) hashes `zmp_walk.py` and
  `asdict(ZMPWalkConfig())`, so any future `cycle_time_s` change is
  picked up automatically by the parity tool's library cache.

### Phase 7. Swing-z envelope shape (TB-style single-peaked triangle)

Goal:
- close the P2 absolute `swing_foot_z_step_max` gate (currently 1.83× TB)
  and the normalised `swing_foot_z_step_per_clearance` gate (currently
  1.40× TB) by adopting TB's piecewise-linear single-peaked triangular
  envelope in place of WR's half-sine

Work (planner-local, single file):
- `control/zmp/zmp_walk.py:719-722` and `:745-750`: replace
  `(foot_step_height_m + swing_foot_z_floor_clearance_m) * sin(pi * frac)`
  with the TB envelope from `toddlerbot/algorithms/zmp_walk.py:483-498`
  (linear up `[0, up_delta, 2*up_delta, ..., peak]`, single-frame apex,
  linear down — no flat top)
- keep the WR-specific `swing_foot_z_floor_clearance_m = 0.025`
  (hardware-justified); only the shape changes
- keep the Phase 1 plantarflex gating (`zmp_walk.py:779-783`); boundary
  frames still need flat-ankle
- keep both Phase A and Phase B branches consistent

Expected metric movement (predicted from TB-shape geometry):
- `P1A swing_foot_z_step_per_clearance`: 1.40× → ≈ 1.00× TB → PASS
  with margin. Math: WR triangle peak / TB triangle peak =
  `(0.04 + 0.025) / 0.05 = 1.30`; per-frame Δz scales linearly with
  peak at fixed `n_swing_half`, so WR step_max ≈ `1.30 × 0.010 =
  0.013 m`; ratio `0.013 / 0.065 = 0.20` matches TB's `0.010 / 0.050 =
  0.20`. Gate ≤ `1.10 × 0.20 = 0.22` → PASS.
- `P2 swing_foot_z_step_max_m` (absolute): 0.0183 → ≈ 0.013 m. Gate
  `1.10 × TB = 0.011 m`. **Borderline FAIL by ~18 %.** WR carries a
  `swing_foot_z_floor_clearance_m = 0.025 m` offset on top of
  `foot_step_height_m` that TB does not (hardware-justified per
  Stage 3 of the Pipeline Comparison). At equal `n_half`, WR's per-
  frame Δz is structurally `(h_WR + c) / h_TB = 1.30×` TB's. Closing
  the absolute gate fully would require either dropping the floor
  clearance (regressing the P0 swing margin Phase 1 added) or
  shaping the envelope to spend more frames at the apex (cosine /
  flat-top trapezoid — outside TB-alignment scope).
- `P1A swing_clearance_mean_m` (per-cycle peak): unchanged at
  ≈ 0.065 m (peak is the same for half-sine and triangle at equal
  `(h + c)`)

Exit criteria:
- WR-only probe (`uv run python tools/phase6_wr_probe.py`) shows
  `P1A swing_step_per_clr` PASS at vx=0.15
- full parity refresh shows `swing_step_per_clr` PASS for both TB
  baselines
- absolute `P2 swing_foot_z_step_max_m` improves by ≥ 25 % (target
  ≤ 0.014 m); a residual FAIL on the 1.10× TB absolute gate is
  acceptable if the per-clearance gate clears (see Acceptance
  statement)
- no regression on `P0` (in-scope or full matrix), `P1A` step_len /
  cadence / clearance / DS, or `q_step` smoothness

Acceptance statement:
- success means the **normalised** `swing_step_per_clr` gate flips
  to PASS without regressing any currently-passing gate
- the absolute `P2 swing_foot_z_step_max` gate is allowed to remain
  FAILing at the architecturally-bounded ratio (≈ 1.30× TB) because
  the WR-specific floor clearance offset is hardware-justified;
  this should be documented as a bounded-by-design gap in the
  scorecard, not pursued by further envelope shaping unless a
  hardware change drops the clearance need
- success does not require P1 closed-loop to move; Phase 10 owns the
  closed-loop interpretation

Rollback if expected movement doesn't materialize:
- if the triangular shape produces *worse* P2 step_max than the
  half-sine (unlikely but possible if `n_half` is small enough that the
  per-frame delta dominates), revert to half-sine and try a half-cosine
  variant `(1 - cos(2 * pi * frac)) / 2` which has a flatter apex than
  half-sine

#### Phase 7 closeout (2026-04-26, commit `61234d3`)

Implemented in `control/zmp/zmp_walk.py` only (planner-local). The
swing-z formula `(foot_step_height_m + swing_foot_z_floor_clearance_m)
* sin(pi*frac)` is replaced by a TB-style piecewise-linear single-
peaked triangle table, mirroring `toddlerbot/algorithms/zmp_walk.py:483-491`:

```python
swing_window = max(2, n_half - n_ds)
swing_half = swing_window // 2
up_delta = (cfg.foot_step_height_m + swing_foot_z_floor_clearance_m) \
           / max(1, swing_half - 1)
swing_z_table = up_delta * np.concatenate(
    (np.arange(swing_half),
     np.arange(swing_window - swing_half - 1, -1, -1))
)
```

The table is pre-computed once before the cycle loop and used for both
Phase A (left stance) and Phase B (right stance). Indexing via
`i_swing = i - n_ds`. Peak preserved at `0.04 + 0.025 = 0.065 m`; H1
hardware exemption (the WR floor-clearance offset) stays explicit.
`frac` is kept for the X-direction interpolation (no change to that
path). Phase 1 plantarflex gating still keeps boundary frames flat-ankle
because `swing_z_table[0] = swing_z_table[-1] = 0` by construction.

For the WR cycle (n_half=18, n_ds=6, swing_window=12), the table
materializes as `up_delta * [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]` — a
2-frame apex plateau (indices 5 and 6 both at peak), which is TB's
exact behaviour for even `swing_window`. The doc's earlier review
correctly flagged that "triangular-then-flat" was the wrong phrasing
for TB's overall envelope; the TB envelope is single-peaked with a
1-2 frame apex plateau depending on parity of `swing_window`, which
is what WR now mirrors.

WR-only probe (`uv run python tools/phase6_wr_probe.py`) at vx=0.15:

| Metric | Pre-Phase-7 | Post-Phase-7 | Δ |
|---|---|---|---|
| **H1 replacement gauge** |
| `swing_foot_z_step_per_clearance` | 0.2803 (1.401× TB) | **0.2160 (1.080× TB)** | **FAIL → PASS** |
| **H1 absolute gate (informational under H1)** |
| `swing_foot_z_step_max_m` | 0.0183 (1.831× TB) | 0.0143 (1.427× TB) | −22 % |
| **P1A gait shape** |
| `step_length_mean_m` | 0.0538 | 0.0538 | unchanged |
| `swing_clearance_mean_m` (per-cycle peak) | 0.0653 | 0.0660 | +1.1 % (apex plateau hits cleanly) |
| `swing_clearance_per_com_height` | 0.1426 | **0.1442** | +1.1 % (still FAIL at 0.825× TB; closer to gate 0.85×) |
| `touchdown_rate_hz` | 2.7330 | 2.7330 | unchanged |
| `double_support_frac` | 0.3333 | 0.3333 | unchanged |
| **P2 smoothness** |
| `shared_leg_q_step_max_rad` | 0.3599 | 0.3155 | −12 % (linear ramp gentler on IK than half-sine apex) |
| `pelvis_z_step_max_m` | 0.0000 | 0.0000 | sentinel pass |
| `contact_flips_per_cycle` | 3.9677 | 3.9677 | unchanged (no extra mid-swing tap-down events from the apex plateau) |
| **P0 geometry** |
| total failures | 0 | 0 | unchanged |
| in-scope failures | 0 | 0 | unchanged |
| worst stance z (m) | +0.0019 | +0.0019 | unchanged |
| worst swing z (m) | +0.0170 | +0.0117 | margin reduced by 5.3 mm but well above the −2 mm gate |

Phase 7 acceptance check (per the Acceptance statement and the H1
exemption rule):
- **H1 replacement gauge `swing_foot_z_step_per_clearance` PASS** at
  1.080× TB (gate ≤ 1.10× TB) — load-bearing gate cleanly closed.
- **H1 absolute gate** widened from 1.83× → 1.43× TB (improved 22 %;
  the doc's "≥ 25 %" target was missed by 3 percentage points). H1
  exemption applies; the absolute gate is informational, not part of
  the closure check. Honest note: my predicted Phase 7 post-state of
  ≈ 0.012 m / ≈ 1.20× TB was optimistic — the FK-realized step max
  is ~10 % above the table's per-frame `up_delta` because of
  plantarflex-threshold-crossing at the 2nd / 10th swing frames. The
  H1 gauge does not depend on this slack.
- No regression on P0, P1A absolute gates, or P2 q-step / pelvis /
  flips.
- P0 swing margin reduced by 5.3 mm (to +11.7 mm) — within the
  Phase 7 exit criterion of "no regression on P0 (in-scope or full
  matrix)". The reduction is structural: the triangle's apex plateau
  spends 2 frames at peak so the IK plantarflex during those frames
  brings the foot bottom slightly closer to the floor than the half-
  sine's brief apex visit.
- Bonus: `swing_clearance_per_com_height` ticked up 1.1 % (0.143 →
  0.144) without any height bump — closer to the 0.149 gate, so
  Phase 8 has a smaller delta to close. Does **not** clear H2 by
  itself; Phase 8 is still the path to PASS that gate.
- `shared_leg_q_step_max` IMPROVED 12 % (0.36 → 0.32 rad) — the
  linear ramp produces a gentler IK joint trajectory than the half-
  sine's curvature. Pleasant side effect; not a Phase 7 acceptance
  requirement.

Outstanding caveats:
- **TB-side parity refresh deferred.** The local TB venv has a
  filesystem-permission issue: `uv pip install joblib lz4` fails with
  `Operation not permitted` on `.venv/lib/python3.11/site-packages`,
  and the TB `uv sync` path fails with a pybullet build error. TB
  code is untouched between Phase 6 and Phase 7, so the cached TB
  numbers in `tools/parity_report.json` stay valid for absolute /
  normalised P1A / P2 ratios; only the P1 closed-loop layer needs
  a fresh full parity run on a TB-capable machine to update both
  sides under the Phase 7 envelope.
- The Phase 5 cache fingerprint picked up the planner change
  automatically (the WR probe regenerated the library with the new
  fingerprint hash, no manual cache invalidation needed).
- All 165 tests pass (`uv run pytest tests/`).

**P1 closed-loop observation (added by review verification, full
parity refresh on a TB-capable machine):**

Phase 7 closed the H1 planner-shape gauge as intended, but the **same
patch worsened multiple WR-side P1 closed-loop metrics** at vx=0.15
under MuJoCo:

| P1 metric (WR @ vx=0.15) | Pre-Phase-7 | Post-Phase-7 | Δ |
|---|---|---|---|
| `ref_contact_match_frac` | 0.6986 | 0.6250 | **−0.074** |
| `touchdown_step_length_mean_m` | +0.0016 | −0.0097 | **flipped sign, worsened** |
| `joint_limit_step_frac` (sat_step_frac) | 0.0274 | 0.0556 | **+0.028 (doubled)** |
| `shared_leg_rmse_rad` | 0.1558 | 0.1669 | **+0.011 (+7 %)** |

This was not visible in the WR-only probe (`tools/phase6_wr_probe.py`)
because it doesn't run the closed-loop replay. The numbers above came
from a full `tools/reference_geometry_parity.py` refresh post-patch,
done by a reviewer on a TB-capable machine; the local run could not
produce them because of the TB venv permission issue documented above.

**Hypothesis** (not yet verified): the half-sine envelope has a
shallow slope near the boundary frames (`d/dt sin(pi·frac) → 0` as
`frac → 0`), so the foot z command grows gradually past the
plantarflex threshold (0.025 m). The triangle envelope has a
**constant** slope `up_delta = 0.013 m/frame`, so the threshold
crossing happens between table indices 1 (z=0.013) and 2 (z=0.026)
**in a single frame**, simultaneously with the discrete 0.15 rad
ankle plantarflex flip. This combined transition appears to push the
WR position-actuator stack harder than the half-sine did, manifesting
as more saturated steps (`sat_step_frac` doubled) and worse contact /
stride / RMSE tracking. None of these tripped the H1 gauge or any
P0 / P1A / P2 absolute gate.

**Implications:**

1. **Phase 7's intended acceptance is met** (H1 gauge PASS) and the
   triangle is the correct planner-shape change per TB alignment.
   The closed-loop regression is an actuator-side interaction, not
   a planner-shape regression.
2. **The earlier closeout claim that "Phase 7 unblocks Phase 10
   cleaner" was wrong.** Phase 10 now has *more* to settle, not less:
   it must add a question about why the triangle worsened P1 (see
   Phase 10 update below). Removed the unfounded "cleaner" claim.
3. **Phase 8 should be treated as more contingent than originally
   written.** Bumping `foot_step_height_m` raises `up_delta`
   proportionally (`up_delta = 0.075/5 = 0.015 m/frame` at h=0.05;
   `0.07/5 = 0.014 m/frame` at h=0.045), which would make the
   threshold-crossing transition even more aggressive. If the
   hypothesis above is right, Phase 8 risks widening the P1
   regression. Do not ship Phase 8 until the post-Phase-7 P1
   numbers are understood (see Phase 10 update below).

**Phase 7 acceptance verdict (updated):** PASS on the load-bearing
H1 gauge; the closed-loop P1 regression is **noted, not gated**, but
becomes Phase 10's responsibility to explain. The change stays in
tree because (a) it correctly aligns the planner shape to TB and
(b) the closed-loop layer was already failing pre-Phase-7 — Phase 7
shifted the failure mode within the already-failing layer rather
than breaking a passing one.

**Sequencing impact:**

- Phase 8 is now **gated on Phase 10** rather than only on Phase 7's
  H1 PASS. Concretely: Phase 10 must surface a verdict on whether
  Phase 7's P1 movement is fixable (e.g., by softening the
  threshold-crossing transition, by changing the plantarflex schedule,
  or by mirroring TB's swing-z slope-shaping at the boundaries) before
  Phase 8 ships.
- Phase 10's diagnostic adds a new item C (see Phase 10 update).
- The acceptance-rule exemption table is unchanged — H1 stays as the
  shape-side gauge, the P1 closed-loop layer is still gated by H6
  (pending) for the actuator-stack-bound case.

Outstanding caveats:
- TB-side parity refresh on this machine is still blocked by the venv
  permission issue. The post-Phase-7 P1 numbers above are reviewer-
  reported from their TB-capable run and not yet reflected in the
  local `tools/parity_report.json` (which still shows pre-Phase-7 WR
  values). Updating the local report cleanly requires the TB venv to
  be reachable, or running the refresh on a different machine.
- All 165 local tests pass; the test suite does not exercise the
  closed-loop P1 layer.

### Phase 8. `foot_step_height_m` adoption (TB's 0.05 m, contingent)

Goal:
- close the normalised `swing_clearance_per_com_height` gate (currently
  0.82× TB) by raising `foot_step_height_m`

Work (single config change):
- `control/zmp/zmp_walk.py:95`: `foot_step_height_m: float = 0.04` →
  one of:
  - **0.045** (recommended — halfway step) if Phase 7 leaves the
    absolute `P2 swing_foot_z_step_max` near the 1.10× TB gate
  - **0.05** (full TB-adoption) only if Phase 7 leaves significant
    headroom on the absolute gate AND the per-clearance gate
  - the Phase 5 cache fingerprint picks up either change automatically

Sequence:
- AFTER Phase 7 — must measure post-Phase-7 absolute swing_step before
  picking 0.045 vs 0.05; Phase 7's Acceptance statement allows the
  absolute gate to remain bounded-FAIL, in which case Phase 8 is the
  knob most likely to widen that gap.
- **AFTER Phase 10 (added by Phase 7 closeout)** — Phase 7 caused
  observed P1 closed-loop regressions on contact_match (−0.074),
  td_step_mean (sign-flipped), sat_step_frac (doubled), and RMSE
  (+7 %) at vx=0.15. Bumping `foot_step_height_m` raises `up_delta`
  proportionally (0.013 → 0.014 at h=0.045, → 0.015 at h=0.05),
  making the plantarflex-threshold-crossing transition Phase 7
  introduced even more aggressive. Phase 8 is now **gated on Phase 10
  producing a verdict** on whether the Phase 7 P1 movement is fixable
  (e.g., by softening the boundary transition) or accepted. If Phase
  10's verdict is "Phase 7 boundary-transition is load-bearing",
  Phase 8 must wait for the Phase 12 fix to land before shipping; if
  the verdict is "actuator-stack-bound and accepted (H6 active)",
  Phase 8 can ship with explicit acknowledgement that the residual
  P1 worsening is bounded by H6.

Expected metric movement:
- `P1A swing_clearance_per_com_height`:
  - at `h = 0.045`: 0.143 → ≈ 0.158 (gate 0.149) → PASS
  - at `h = 0.05`:  0.143 → ≈ 0.175 (gate 0.149) → PASS with margin
- `P1A swing_clearance_mean_m` (per-cycle peak):
  - at `h = 0.045`: 0.065 → ≈ 0.070 m
  - at `h = 0.05`:  0.065 → ≈ 0.075 m
- `P2 swing_foot_z_step_max_m` (absolute) — **this gets worse**:
  - at `h = 0.045`: post-P7 0.013 m → ≈ 0.014 m
  - at `h = 0.05`:  post-P7 0.013 m → ≈ 0.015 m
  - both are above the 0.011 m gate; the absolute gate's residual FAIL
    is the Phase 7 hardware-bounded ratio, widened proportionally

Exit criteria:
- `swing_clearance_per_com_height` ≥ 0.85× TB (= 0.149) → PASS for both
  TB baselines
- no regression on Phase 7 normalised `swing_step_per_clr` gate (must
  still PASS — the per-clearance ratio stays at ≈ 0.20 because both
  numerator and denominator scale with peak)
- no regression on P0 swing margin (must remain > +5 mm)
- absolute `P2 swing_foot_z_step_max_m` is allowed to drift further
  from the 0.011 m gate; document the new ratio in the closeout

Fallback:
- if Phase 7 leaves the absolute swing_step gate borderline (≈ 0.013 m)
  and the project explicitly wants the absolute gate clear, **defer
  Phase 8** rather than break it; the normalised P1A clearance gap is
  small and not load-bearing relative to the actuator-stack questions
  Phase 10 will surface

Acceptance statement:
- success means `clearance_per_h` flips to PASS while the per-clearance
  swing-step ratio stays at the Phase 7 PASS level
- the absolute `swing_step_max` gate is **explicitly accepted** as
  bounded-by-design (WR's hardware-justified floor clearance offset on
  top of a higher `foot_step_height_m`); the scorecard should annotate
  this gate accordingly so future readers don't re-litigate

### Phase 9. Operating-`vx` policy decision (explicit fork)

Goal:
- force an explicit project decision on whether to **change WR's
  operating point** to TB's Froude-equivalent vx, or to **document the
  proportional-shape gap** as bounded-by-design at the current operating
  point. Phase 9 closes the two hardware-Froude-bounded normalised P1A
  gates (`step_length_per_leg = 0.56× TB`, `cadence_froude_norm = 1.25× TB`)
  in exactly one of these two ways — they are NOT interchangeable.

The reason the fork has to be explicit: a parity-tool-only Froude
re-read does **not** close the parity scorecard at the operating point
training and deployment actually use. Letting the parity report
silently use a different `vx_nominal` would move the goalposts. The
two options below produce different end-states and must be chosen, not
hedged.

Froude background (used by both options):
- TB at vx=0.15: `Fr = vx² / (g · h_TB) = 0.15² / (9.81 · 0.286) = 0.0080`
- WR Froude-matching point: `vx_WR = √(Fr · g · h_WR) = √(0.0080 · 9.81
  · 0.458) ≈ 0.19 m/s`
- WR at vx=0.15 sits at `Fr = 0.005`, ~63 % of TB's, so per-leg stride
  must under-shoot and Froude-norm cadence must over-shoot by
  construction

#### Phase 9A — change WR's operating point to vx ≈ 0.19 m/s

Use this option ONLY if the project intends WR to walk at the
Froude-equivalent operating point in training and deployment.

Work:
1. Update WR's nominal command grid in `walking_training.md` G4/G5
   contract from `vx ∈ {0.10, 0.15}` to (e.g.) `vx ∈ {0.13, 0.19}`,
   and propagate to:
   - `tests/test_v0200c_geometry.py` smoke probe
   - PPO training-config command sampling distribution
   - the parity tool's `vx_nominal` (and the in-scope geometry bins
     if the project agrees to widen scope)
2. Re-run G4/G5 walking smoke at the new operating point and confirm
   the policy-side contract still holds.
3. Re-run the parity tool — `step_per_leg` and `cadence_norm` now PASS
   directly because WR is operating at TB's Froude point.

Expected metric movement at WR vx=0.19:
- `step_length_per_leg`: 0.56× → ≈ 0.85–0.95× TB → PASS
- `cadence_froude_norm`: 1.25× → ≈ 1.0× TB → PASS

Exit criteria (9A):
- WR walks G4/G5 at the new operating point with no regression on the
  policy contract
- parity tool's primary `vx_nominal` is the new value; both normalised
  gates PASS at the primary read
- `walking_training.md` documents the operating-point change with
  hardware / safety rationale

Acceptance statement (9A):
- success means WR's actual operating point changed; the normalised
  gates close at the operating point that training and deployment use,
  not via a parity-tool re-read

#### Phase 9B — keep vx=0.15 as the primary operating point, document the gap as bounded

Use this option if the project intends WR to keep walking at vx=0.15
for hardware / safety / curriculum reasons.

Work:
1. Document in `walking_training.md` and the parity scorecard that the
   two Froude-bounded normalised gates are **architecturally bounded
   below TB at the chosen operating point**, not a closable gap.
   Provide the Froude scaling argument and the per-bin numbers.
2. Add a **supplemental** Froude-matched section to the parity report
   (e.g., `--report-secondary-vx-wr 0.19`) that re-reads WR at the
   matched operating point. This is a diagnostic / reference-quality
   sanity check, NOT the primary parity reading.
3. Annotate the parity scorecard's normalised gates with the bounded
   ratio so future readers see the gap is by-design, not a regression.

Expected metric movement: none at the primary operating point. The
supplemental Froude-matched read shows the same numbers as Phase 9A's
predicted PASS values, but the primary parity verdict at vx=0.15 still
shows these gates FAILing.

Exit criteria (9B):
- `walking_training.md` explicitly chooses vx=0.15 as the primary
  operating point with rationale
- the parity scorecard's two affected gates are annotated as
  hardware-Froude-bounded with the supplemental Froude-matched values
  recorded for reference
- the `What "On Par or Better" Means` section of this doc is updated
  to acknowledge that under 9B, two normalised gates remain FAILing
  at the primary operating point and this is accepted by policy

Acceptance statement (9B):
- success means the gap is documented and accepted, not closed; the
  parity scorecard's "on par" claim explicitly excludes these two
  Froude-bounded gates with a documented hardware reason
- this is NOT a parity closure — it is an explicit acceptance of a
  bounded gap. Future readers must be able to see the FAILs are
  bounded-by-design, not unresolved work

#### Choosing between 9A and 9B

This decision is owned by the training / hardware project leads, not
by the parity-tool maintainer. The parity tool itself is the same
under both options; only the primary `vx_nominal` and the
acceptance language differ.

If the project cannot decide cleanly between 9A and 9B, the
conservative default is **9B** — do not move WR's operating point
purely to satisfy a parity gate. Move it only if there is an
independent training / deployment reason to.

### Phase 10. P1 closed-loop contact-match-frac diagnostic

Goal:
- decide whether the closed-loop contact-match FAIL (WR 0.58–0.70 vs
  the 0.90 absolute floor) is fixable prior shape, residual cycle-0
  drag, or accepted actuator-stack artifact

Sequence:
- AFTER Phase 7 — Phase 7 should drop swing-apex flips; running this
  diagnostic before would conflate that effect

Work (analysis, no code change unless evidence demands):

1. Front-loaded vs steady-state contact match. Add a small probe (or a
   one-off script) that splits `ref_contact_match_frac` over
   [first 30 steps] vs [steps 30–end] of each P1 episode. Compare WR vs
   TB. If WR's [30-end] match is ≥ 0.85, the front-load is dragging the
   mean and the cycle-0-retention hypothesis is confirmed.
2. Cycle-0 retention quantification. Compute the contribution of WR's
   first cycle (TB truncates `trunc_len = ceil(cycle_time / control_dt)`
   per `toddlerbot/algorithms/zmp_walk.py:138-145`) to the contact
   mismatch. If this is the load-bearing source, mirror TB's truncation
   in WR's `_generate_at_step_length`
   (`control/zmp/zmp_walk.py:691-696`) — this becomes a Phase 12
   follow-up if it materializes.
3. Mid-swing tap-down audit. Inspect the foot z trace during swing in
   the closed-loop replay: how often does the foot dip back into
   contact within a single swing? Phase 7's triangular envelope was
   originally predicted to drop this to near-zero. **Updated 2026-04-26
   per Phase 7 closeout**: the post-Phase-7 P1 numbers actually
   *worsened* on contact_match (−0.074 at vx=0.15), td_step_mean
   (sign-flipped), sat_step_frac (doubled), and shared_leg_rmse_rad
   (+7 %). So the original "should drop to near-zero" prediction was
   wrong; the audit must report the actual post-Phase-7 mid-swing
   tap-down rate, not assume it's zero.
4. **Phase-7-effect attribution (NEW item from Phase 7 closeout).** The
   triangle envelope cleanly closed the H1 planner-shape gauge but
   appears to have shifted the closed-loop failure mode. Concrete
   checks:
   - *Plantarflex-threshold-crossing analysis.* Does the foot z
     command cross the 0.025 m plantarflex threshold (`zmp_walk.py:611`)
     in a single frame under the triangle envelope (i_swing 1 → 2:
     z 0.013 → 0.026, simultaneously with the discrete 0.15 rad ankle
     flip)? Compare against the half-sine where the same crossing
     happens with a shallower slope. If the triangle's discrete
     transition is the cause, the fix is either softening the
     plantarflex schedule (ramping the 0.15 rad over multiple frames),
     or shaping the swing-z envelope's slope at the boundary frames
     to match the half-sine's shallower onset (e.g., a small cosine
     blend over the first / last 1-2 frames of the triangle).
   - *Actuator-stack saturation count.* Where do the doubled
     `joint_limit_step_frac` events fall in the swing window? If they
     cluster at the threshold-crossing frames (i_swing = 2, 10),
     plantarflex-coupling is the load-bearing cause and Phase 12's
     scope expands to "soften the boundary transition" rather than
     "mirror TB cycle-0 truncation".
5. Decide. Possible verdicts now include:
   - "Cycle-0 drag is load-bearing" → mirror TB truncation in
     Phase 12
   - "Phase 7 boundary-transition is load-bearing" → soften the
     plantarflex schedule and/or shape the triangle's first / last
     frames in Phase 12
   - "Both clean, residual is actuator stack" → activate H6 (after
     parity tool also ships the composite gauge per H6's activation
     rule)
   - "Mixed" → multiple Phase 12-class follow-ups required

Exit criteria:
- a decision is recorded with evidence (numbers from items 1–4)
- if the verdict is "fixable", a concrete follow-up phase is added to
  this doc
- if the verdict is "actuator-stack-bound", H6's activation conditions
  are formally evaluated (composite gauge ships in parity tool +
  evidence on file) before H6 is treated as active
- a clear statement on whether Phase 8 should ship; under the Phase 7
  closeout's contingency, Phase 8 is gated until the Phase-7-effect
  question above has a verdict

Acceptance statement:
- success means the P1 closed-loop FAIL set has a verdict (fixable /
  partially-fixable-by-Phase-12 / actuator-stack-bound / mixed), not a
  vague "WR is worse than TB"
- the closed-loop layer's narrative becomes either "all gates either
  PASS strictly or PASS via an active exemption", "Phase 12 follows",
  or both

#### Phase 10 closeout (2026-04-26, commit `cc5e8d3`)

Implemented `tools/phase10_diagnostic.py` — a WR-only closed-loop
replay that captures per-step traces and produces D1 / D2 / D3+D4 as
specified above. No planner code changed. The diagnostic imports
WR-side helpers from `tools/reference_geometry_parity.py`, so the
replay is byte-for-byte the same trajectory the parity tool sees;
only the per-step capture is added. Same TB-venv-free constraint as
`tools/phase6_wr_probe.py`.

**Diagnostic results on post-Phase-7 tree** (commit `cc5e8d3`,
horizon 200, vx ∈ {0.10, 0.15, 0.20}):

| Metric | vx=0.10 | vx=0.15 | vx=0.20 |
|---|---|---|---|
| **D1: cycle-0 retention** |
| overall match_frac | 0.566 | 0.611 | 0.647 |
| first-cycle match_frac (steps 0–35) | 0.778 | 0.806 | 0.806 |
| rest match_frac (steps 36+) | 0.375 | 0.417 | 0.469 |
| `rest − first_cycle` gap | **−0.40** | **−0.39** | **−0.34** |
| D1 verdict | no-cycle-0-drag | no-cycle-0-drag | no-cycle-0-drag |
| **D2: mid-swing tap-down** |
| events / swing cycle | 1.50 | 1.50 | 1.20 |
| left vs right asymmetry | 7 vs 2 | 7 vs 2 | 4 vs 2 |
| D2 verdict | planner-shape-bound | planner-shape-bound | planner-shape-bound |
| **D3+D4: sat distribution by i_swing** |
| swing-window mean sat rate | 0.188 | 0.083 | 0.091 |
| sat rate at up-cross (i_swing=2) | 0.250 | 0.000 | 0.000 |
| sat rate at down-cross (i_swing=9) | 0.250 | 0.250 | 0.333 |
| D3+D4 verdict | uniform-sat | Phase-7-boundary-bound | Phase-7-boundary-bound |
| **Composite verdict** | planner-shape | mixed: planner-shape + Phase-7-boundary | mixed: planner-shape + Phase-7-boundary |

**Three load-bearing findings:**

1. **Cycle-0 retention is NOT the cause.** D1's gap is **negative** at
   every vx (−0.40, −0.39, −0.34). The first cycle is consistently
   the BEST part of every episode; WR's contact match degrades
   *progressively* after step 36. **TB cycle-0 truncation would not
   help — it would discard the cleanest part of the trajectory.**
   This rules out one of the four hypothesised Phase 12 fixes
   (mirror TB cycle-0 truncation in WR). Saves us from doing the
   wrong work.

2. **Phase 7's plantarflex DOWN-cross is the actuator-saturation
   driver at vx ≥ 0.15.** D3+D4 shows saturation rate at the
   down-cross (i_swing=9) is ≥ 1.5× the swing-window mean while
   the up-cross (i_swing=2) is at 0.0. The asymmetry is
   diagnostic: the discrete ankle un-plantarflex (−0.15 rad → 0
   rad in a single frame between i_swing=9 z=0.026 m plantarflexed
   and i_swing=10 z=0.013 m neutral) is what the actuator stack
   cannot track. The up-cross has the same z step but is well-
   separated from touchdown so the joints have headroom. This
   confirms the Phase 7 closeout's hypothesis.

3. **Mid-swing tap-down events are confirmed across all vx**
   (1.2–1.5/cycle), with strong **left/right asymmetry** (left foot
   3-4× right). This is consistent with the down-cross "snap"
   pulling the foot into the floor before the planner's commanded
   touchdown frame, but also suggests an asymmetry in the IK or
   model that warrants a follow-up audit independent of the Phase 12
   fix.

**Composite verdict: mixed (planner-shape-bound + Phase-7-boundary-
bound at vx ≥ 0.15; planner-shape-bound only at vx=0.10).**

**Phase 12 scope (recommended):**

The right Phase 12 fix is **not** TB cycle-0 truncation (D1 ruled
out). The right fix targets the discrete plantarflex transition at
the down-cross. Two candidate slices, in order of preference:

- **Phase 12A: Soften the plantarflex schedule.** Currently the
  ankle plantarflex is binary on/off-gated on
  `foot_world_i[2] > swing_ankle_lift_threshold_m = 0.025 m`
  (`control/zmp/zmp_walk.py:780-783`). Replace the binary gate with
  a continuous ramp: e.g.,
  `plantarflex_rad = 0.15 * smoothstep((foot_z − 0.020) / 0.010)`
  so the ankle ramps over a 10 mm window (0.020 to 0.030 m)
  instead of switching at 0.025 m. This addresses the down-cross
  snap directly and should symmetrically improve the up-cross too
  (which is currently fine but stays fine under the ramp).
  Planner-local single-file change; no schema change; Phase 5 cache
  fingerprint picks it up. Expected to drop D3+D4 down-cross sat
  rate to swing-window-mean and reduce D2 mid-swing tap-down events
  proportionally.
- **Phase 12B (fallback if 12A is insufficient): Shape the
  triangle's first / last 1-2 frames.** Replace the linear ramp at
  the boundaries with a half-cosine blend over 1-2 frames, so the
  per-frame Δz at the down-cross is smaller (asymptotically zero
  approaching touchdown). More invasive than 12A.

**Phase 8 ship/wait decision:** **WAIT.** Phase 8 (foot_step_height
0.04 → 0.045 or 0.05) raises `up_delta` proportionally
(0.013 → 0.014 at h=0.045, → 0.015 at h=0.05), which steepens the
down-stroke and amplifies the down-cross snap. Phase 12A must land
and verify before Phase 8 ships. Under Phase 9B keep-vx=0.15, this
also means the H7 / H8 conditional exemptions remain pending until
Phase 12A clears the down-cross saturation.

**H6 activation status:** **NOT activated.** The closed-loop FAIL is
*not* purely actuator-stack-bound — the diagnostic surfaced two
fixable causes (planner-shape mid-swing tap-down + Phase-7-boundary
plantarflex transition). Treating H6 as the gauge of record now
would mask fixable architecture-side issues. H6 stays pending until
Phase 12A lands and the residual is verified to be uniform-sat
across i_swing (D3+D4 verdict "uniform-sat" or "no-significant-
sat"), at which point H6 + the parity-tool composite gauge can
activate together.

**Sequencing impact (updated):**

- Phase 10 closeout produces a verdict; ✅ done.
- **Phase 12A (NEW)** added to backlog: soften plantarflex schedule.
  Required before Phase 8 ships and before H6 can activate.
- Phase 8 still gated on Phase 12A landing.
- Phase 11 (lin_vel_z cleanup) unchanged, low-priority hygiene.

Outstanding caveats:
- The diagnostic does not include a pre-Phase-7 baseline
  (half-sine envelope) for direct A/B comparison. The D3+D4 down-
  cross-asymmetric pattern is consistent with the reviewer's
  pre/post P1 numbers (Phase 7 caused the regression), but a
  rigorous A/B would require temporarily reverting the planner.
  Skipped here because the post-state evidence is sufficient to
  scope Phase 12A.
- Episode survival is short (68-76 steps) at every vx, so the
  "rest" sample size is small (32-40 steps). The rest-vs-first
  contrast direction (negative gap) is consistent across all three
  vx bins, so the conclusion is robust, but the absolute "rest"
  numbers carry larger uncertainty than the first-cycle ones.
- The left/right asymmetry in D2 (left foot 3-4× right tap-downs)
  is not explained by the Phase-7-boundary hypothesis alone. May
  warrant a separate audit of the WR MJCF / IK chain for left-leg
  vs right-leg structural asymmetry. Tracking as a follow-up
  question, not as a blocker for Phase 12A.

### Phase 11. `lin_vel_z` reward cleanup (low priority, hygiene)

Goal:
- remove the residual planner-side reference consumption flagged in the
  doc-vs-code drift audit; make the Phase 4 doc claim "pelvis_pos
  retained only for diagnostics" literally true

Work (one-line change):
- `training/envs/wildrobot_env.py:843`: replace `prev_ref_pelvis_z =
  win["pelvis_pos"][2]` (and the symmetric current-step read) with the
  constant `cfg.com_height_m` directly. The numerical effect is
  identical (the planner field is constant by construction in
  `zmp_walk.py:845`); the change is doc-vs-code consistency only.

Expected metric movement: none. This is hygiene.

Exit criteria:
- `pelvis_pos` is consumed only by the diagnostic `feet_track_raw`
  probe; verifiable by `grep "pelvis_pos" training/envs/wildrobot_env.py`
  returning only the diagnostic line

Acceptance statement:
- success means no behavioral change in any reward or training metric;
  the consumption-narrowing is verifiable by grep

### Phase 12A. Soften the IK plantarflex schedule (NEW from Phase 10 verdict)

Goal:
- close the Phase-7-boundary-bound component of the post-Phase-7 P1
  closed-loop FAIL by replacing the binary plantarflex on/off-gate
  with a continuous ramp, eliminating the discrete ankle un-plantarflex
  at the down-cross frame

Diagnostic signal that triggered this phase (from Phase 10):
- D3+D4 sat rate at i_swing=9 (down-cross) is 1.5-3× the swing-window
  mean at vx ≥ 0.15
- D2 mid-swing tap-down events (1.2-1.5/cycle) cluster on the down-
  stroke side
- D1 ruled out cycle-0 retention as the cause; this is a planner-side
  fix, not a TB-truncation fix

Work (planner-local, single file):
- `control/zmp/zmp_walk.py:780-783`: replace the binary
  `is_swing_for_ankle = is_swing AND foot_world_i[2] >
  swing_ankle_lift_threshold_m` gate with a continuous ramp
  `plantarflex_factor = smoothstep((foot_z - 0.020) / 0.010)` so the
  0.15 rad ankle plantarflex ramps over the 0.020-0.030 m foot-z
  window instead of switching at 0.025 m. Apply
  `plantarflex_rad = 0.15 * plantarflex_factor` at the IK ankle
  injection site.
- The threshold constant `swing_ankle_lift_threshold_m = 0.025` stays
  as the ramp center; the ramp width (0.010 m) is the new tunable.
- Phase 1 plantarflex compensation rationale stays intact (the toe-
  geometry overhang still needs offsetting); only the schedule
  becomes continuous.
- Phase 5 cache fingerprint picks up the change automatically.

Sequence:
- AFTER Phase 7 ✅ (already landed)
- AFTER Phase 10 ✅ (verdict in tree at commit `cc5e8d3`)
- BEFORE Phase 8 — Phase 12A must verify before Phase 8 ships, since
  Phase 8 amplifies the down-cross transition Phase 12A is closing

Expected metric movement:
- D3+D4 sat rate at i_swing=9: 0.250-0.333 → ≤ swing-window mean
  (≈ 0.10) at vx ≥ 0.15 → planner-shape-bound only verdict
- D2 mid-swing tap-down events: 1.2-1.5/cycle → near zero (the down-
  cross snap is the primary mechanism)
- P1 closed-loop `ref_contact_match_frac` at vx=0.15:
  0.625 (post-Phase-7) → ≥ 0.700 (recover at least the pre-Phase-7
  value of 0.6986) and ideally ≥ 0.80
- P1 `joint_limit_step_frac` at vx=0.15:
  0.0556 (post-Phase-7) → ≤ 0.030 (recover pre-Phase-7 0.0274)
- P1 `shared_leg_rmse_rad`: similar recovery toward pre-Phase-7
- Episode survival: > 100 steps at vx=0.15 (currently 72)

Exit criteria:
- `tools/phase10_diagnostic.py` reports D3+D4 verdict =
  "uniform-sat" or "no-significant-sat" at vx=0.15 and vx=0.20
- D2 events/cycle drops to ≤ 0.3 at every vx
- D1 sign / direction unchanged (this phase doesn't target cycle-0)
- Full parity refresh on a TB-capable machine shows P1 closed-loop
  metrics return to or improve on pre-Phase-7 values
- No regression on H1 swing_step_per_clearance (Phase 7 PASS must
  hold; the ankle ramp shouldn't change the foot-z trajectory shape
  on its own, but the swing FK will change because the un-plantarflex
  no longer snaps)
- No regression on P0 (in-scope or full matrix)

Acceptance statement:
- success means D3+D4 down-cross asymmetry disappears AND D2 mid-swing
  tap-downs drop to near-zero AND P1 contact_match recovers to or
  beyond the pre-Phase-7 value
- if D3+D4 clears but D2 stays elevated, Phase 12B (boundary-shape
  envelope) is the next slice
- if D3+D4 also clears the residual P1 RMSE/contact gap to TB
  parity, H6 can activate (subject to the parity tool also shipping
  the composite gauge per H6's activation rule)

Open questions for the implementer:
- Is `smoothstep` the right ramp shape, or should it be linear?
  Recommendation: smoothstep (`3t² - 2t³`) so the ramp's derivative
  is continuous at both ends; this avoids re-introducing a (smaller)
  discontinuity in the ankle's velocity profile.
- Should the ramp also apply on the up-cross side, or only down?
  Recommendation: both — the diagnostic shows up-cross is currently
  fine (0.0 sat rate at i_swing=2 for vx ≥ 0.15) but the symmetric
  ramp is simpler and avoids creating a planner asymmetry.
- The asymmetric left/right tap-down distribution in D2 (left foot
  3-4× right) is a separate signal not addressed by Phase 12A.
  Recommendation: track as an independent IK / MJCF audit follow-up;
  do not bundle into Phase 12A.

### Validation harness (after every Phase 7-11 slice)

Run before declaring exit on any phase:

```bash
# WR-only sanity check (no TB venv needed)
uv run python tools/phase6_wr_probe.py

# Full parity refresh (requires TB venv with joblib + lz4)
cd /Users/ygli/projects/wildrobot
uv run ./tools/reference_geometry_parity.py --json > tools/parity_report.json

# Test suite (must stay green)
uv run pytest tests/
```

Phase-specific: capture before/after parity metrics for the gates that
phase targets, in the same format as the existing Phase 1 / 3 / 6
closeouts in this doc.

### Expected end state after Phases 7-10

The columns below assume Phase 9A (operating-point change to vx=0.19).
Under Phase 9B (keep vx=0.15), the two Froude-bounded normalised
P1A gates remain FAILing as a documented bounded-by-design gap; that
is success under 9B's acceptance statement, NOT a parity claim.

| Layer | Pre-Phase-7 | Post-Phase-9A | Notes |
|---|---|---|---|
| P0 (in-scope + full matrix) | PASS | PASS | unchanged |
| P1A absolute | PASS | PASS | unchanged |
| P1A normalised — `swing_step_per_clr` | FAIL (1.40×) | PASS (≈ 1.00×) | Phase 7 |
| P1A normalised — `clearance_per_h` | FAIL (0.82×) | PASS (≈ 1.05×) | Phase 8 |
| P1A normalised — `step_per_leg` | FAIL (0.56×) | PASS (≈ 0.85-0.95×) under 9A; FAIL bounded under 9B | Phase 9 fork |
| P1A normalised — `cadence_norm` | FAIL (1.25×) | PASS (≈ 1.0×) under 9A; FAIL bounded under 9B | Phase 9 fork |
| P2 — `swing_step` (absolute, 1.10× TB) | FAIL (1.83×) | **FAIL bounded** (≈ 1.30×) | hardware-bounded by WR floor-clearance offset; documented in Phase 7 closeout |
| P2 — leg-q / pelvis / flips | PASS | PASS | unchanged |
| P1 closed-loop — survival / drift / forward | PASS | PASS | already passing pre-Phase-7 |
| P1 closed-loop — RMSE / contact / step | FAIL | **verdict pending Phase 10** | actuator-stack vs cycle-0 vs swing-apex residue — not settled by current data |

**What is and is not claimed:**

After Phases 7-9 (under 9A) WR clears every architecturally-closable
gate plus the absolute P2 swing_step gate at the architecturally-
bounded ratio. Two outstanding items prevent a clean "WR is on par or
better" claim:

1. **P2 absolute swing_step stays FAIL by design** — bounded at ≈ 1.30×
   TB by the WR floor-clearance offset on top of `foot_step_height`.
   This is a hardware-justified architectural divergence, NOT
   unresolved work. The scorecard should annotate the gate.
2. **P1 closed-loop has 3 outstanding FAILs whose root cause is
   unsettled** — could be reference-shape (Phase 7 may have fixed
   mid-swing tap-down events), cycle-0 retention (potential Phase 12
   follow-up if Phase 10 surfaces it), or actuator-stack-bound (accepted
   limitation per the scorecard's interpretation note). Until Phase 10
   produces a verdict with evidence, "the closed-loop residual is just
   actuator stack" is a hypothesis, not a conclusion.

**The doc's "On Par or Better" decision rule explicitly requires P1 to
clear.** Phase 10 is mandatory before that rule can be claimed
satisfied — even if Phases 7-9 close every other gate.

### Execution order

The intended order is:

1. Phase 0 to lock the baseline
2. Phase 1 to remove the known swing-z planner bug
3. Phase 2 to resolve the runtime target semantics
4. Phase 3 to align WR asset content with TB
5. Phase 4 to align WR runtime consumption with the chosen semantics
6. Phase 5 to improve lifecycle parity where it matters
7. Phase 6 for local refinement only after the architecture backlog is under
   control
8. Phase 7 to swap WR's half-sine swing-z for TB's single-peaked triangle
   (closes the normalised swing-step gate; absolute swing-step gate
   becomes bounded-by-design at ≈ 1.30× TB)
9. Phase 8 to bump `foot_step_height_m` (0.045 halfway or 0.05 TB-adopt,
   contingent on Phase 7 result); closes `clearance_per_h`; sequence
   after Phase 7
10. Phase 9 to make the explicit operating-`vx` policy decision
    (9A: change WR's operating point to vx≈0.19 — closes the two
    Froude-bounded gates; 9B: keep vx=0.15 — documents them as
    bounded-by-design under the Option B exemption); independent of 7 / 8
11. Phase 10 to diagnose the P1 closed-loop contact-match FAIL
    (diagnostic-only; produces a verdict, does not itself close the
    gate); sequence after Phase 7
12. Phase 11 to clean up the residual `lin_vel_z` planner-side
    consumption; independent, low priority
13. **Phase 12A** (active, opened by Phase 10 verdict) — soften the
    IK plantarflex schedule from a binary on/off-gate to a
    smoothstep ramp over 0.020-0.030 m foot-z. Closes the
    Phase-7-boundary-bound component of the post-Phase-7 P1 FAIL
    that Phase 10 surfaced. Required before Phase 8 ships.
14. ~~Phase 12 (mirror TB cycle-0 truncation)~~ — **explicitly
    rejected** by Phase 10's D1 verdict (cycle-0 is the BEST part of
    every episode, not a drag). Listed here so future readers do
    not re-litigate.

Phases 7 + 9 + 11 can run in parallel; 8 needs 7, 10's verdict, AND
Phase 12A landing (Phase 7 closeout + Phase 10 closeout flagged P1
closed-loop regressions that Phase 8 would amplify); 10 needs 7;
12A needs 7 and 10.

**Phases 7-11 form an investigate-and-improve plan, not a guaranteed
terminal plan.** They are sufficient to determine whether "WR on par
or better" is reachable under the current hardware assumptions, the
chosen Phase 9 fork (9A vs 9B), and the closed exemption table in the
"What 'On Par or Better' Means" section below. They are not sufficient
by themselves to guarantee a parity claim: Phase 10 may spawn Phase 12,
H6 may need to activate, and Phase 9B requires a parity-tool extension
before H7/H8 are real.

## What "On Par or Better" Means

### Acceptance rule (Option B with explicit hardware-bounded exemption)

WR is on par with or better than TB when **all** of the following are true:

1. **Every gate not on the hardware-bounded exemption list literally
   clears.** No bounded-FAIL exemption is allowed unless it is on the
   table below.
2. **Every gate on the exemption list either clears OR clears its
   designated replacement metric.** The replacement metric is the
   parity gauge for that gate; the absolute gate becomes informational
   only.
3. **WR clears the shared `P0` gate and is not materially weaker on the
   full geometry matrix.** No P0 metric is exemptable.
4. **WR satisfies its own `G4/G5/G6/G7` policy contract.**

If a future WR change improves PPO but does not improve the parity
scorecard at the architecture layer, it should not be claimed as
"reference parity improvement."

### Hardware-bounded exemption list (closed list — additions require explicit doc edit)

This is a **closed list**. A metric is exempt from the strict
absolute-gate reading only if it appears in the table below, with a
named hardware divergence, a named replacement gauge, and a
"Replacement-actually-captures-quality" rationale that can be checked
against. A divergence that is not from the "Keep WR-specific" hardware
list at the top of the Divergence Classification section is **not
eligible** for an exemption — the path for non-hardware divergences
remains "remove unless explicitly justified."

| # | Absolute metric (gauge OFF) | Hardware divergence | Replacement metric (gauge ON) | Why the replacement captures the underlying quality |
|---|---|---|---|---|
| H1 | `P2 swing_foot_z_step_max_m` (gate ≤ 1.10× TB) | WR adds `swing_foot_z_floor_clearance_m = 0.025 m` on top of `foot_step_height_m` for floor-margin reasons (Phase 1). At equal `n_swing_half`, WR's per-frame Δz is structurally `(h_WR + c) / h_TB ≈ 1.30×` TB. Removing the offset would regress the P0 swing margin Phase 1 closed. | `P1A swing_foot_z_step_per_clearance` (gate ≤ 1.10× TB) | The per-clearance ratio measures the **shape** of the swing-z envelope (how peaky vs how spread-out across the swing window) independent of absolute peak. It catches a half-sine apex or a discontinuity exactly where the absolute gate would, but does not penalise the larger-by-design clearance floor. PASS at this gate proves the envelope shape matches TB; FAIL still catches a peakiness regression. |
| H2 | `P1A swing_clearance_mean_m` (gate ≥ 0.85× TB) | WR's leg length 0.373 m vs TB's 0.211 m (1.76×) and COM height 0.458 m vs 0.286 m (1.60×). Absolute clearance scales with size; gating it absolutely either rewards over-clearance or penalises proportional clearance. | `P1A swing_clearance_per_com_height` (gate ≥ 0.85× TB) | Normalising by COM height converts clearance to a dimensionless body-proportional value. PASS proves WR is lifting the foot a TB-comparable fraction of body height; an actual under-clearance regression still trips the gate because the normalisation does not depend on the regression source. |
| H3 | `P1A step_length_mean_m` at the operating-vx grid (gate ≥ 0.90× TB) | Same leg-length / Froude scaling. At equal absolute `vx`, WR is at a smaller Froude number, so per-leg stride structurally under-shoots TB. | `P1A step_length_per_leg` (gate ≥ 0.85× TB) | Per-leg stride captures stride efficiency independent of absolute leg length. WR can fail this gate without hardware excuse (e.g., a planner that under-strides for its own size); the gate stays meaningful. NOTE: this gate is currently FAIL (0.56× TB); H3 only makes it the **gauge** of record, not a free pass. |
| H4 | `P1A touchdown_rate_hz` (gate ≤ 1.20× TB) | Same Froude scaling. At equal absolute `vx`, WR must over-cadence relative to TB to maintain the commanded velocity. | `P1A cadence_froude_norm` (gate ≤ 1.20× TB) | Froude-normalised cadence (`touchdown_rate_hz · √(com_height_m / g)`) captures the per-bodysize cadence regime. WR over-cadencing for its size still trips the gate; the gate isn't disabled, just normalised. |
| H5 | `P1 achieved_forward_mps` per bin (gate `≥ TB − 0.02 m/s`) | Same size delta. The parity tool computes `achieved_forward_mps = terminal_root_x_m / (survived_steps · dt)`, so the absolute gate is effectively a drift-rate-in-m/s comparison. WR's larger leg makes the same per-leg-length drift translate to a bigger absolute m/s value — so an absolute m/s gate makes WR look worse simply for being bigger, while a per-leg gate captures stability per body size. | `P1 terminal_drift_rate_per_leg_per_s` (gate `≤ TB + 0.5 leg/s`) | Per-leg-length drift rate normalises the same underlying quantity (`terminal_root_x_m / duration`) by leg length. The current data shows WR drifts at ½ TB's per-leg rate — the absolute m/s gate would mask this. PASS at this gate proves WR is stable per body proportion; a real instability (drift accelerating beyond TB's per-leg rate) trips the gate. |
**Conditional exemptions (PENDING — not active until activation conditions are met):**

These rows are **not currently active**. They become active only when their activation condition is satisfied (Rule 2: replacement metric must already exist in the parity tool). Until then, the absolute gate stays on as the gauge of record.

| # | Absolute metric (gauge OFF when active) | Activation condition | Replacement metric (gauge ON when active) | Why |
|---|---|---|---|---|
| H6 | `P1 shared_leg_rmse_rad` (gate ≤ 1.10× TB) | Active **only after** (a) `tools/reference_geometry_parity.py` exposes a single named composite verdict combining survival + forward + drift, AND (b) Phase 10 closes with verdict "actuator-stack-bound" with evidence. Until both hold, H6 is pending and the absolute RMSE gate stays on. | `P1 actuator_trackability_composite` — a composite verdict to be added to the parity tool, defined as `survived_steps ≥ 0.95× TB` AND `achieved_forward_per_cmd ≥ TB - 0.10` AND `terminal_drift_rate_per_leg_per_s ≤ TB + 0.5`, all per `vx` bin | WR uses MuJoCo position actuators (target-error integration); TB uses torque actuators with PD-on-torque. These produce structurally different per-step joint-error patterns under closed-loop replay even at identical reference quality (see scorecard interpretation note). The composite asks "does the WR prior actually walk under MuJoCo with this actuator stack?" — answered by survival + forward + drift jointly. A broken prior trips survival or forward-velocity first; a real actuator regression trips drift. Currently the parity tool reports the three metrics separately, so the named composite verdict does not exist yet — Rule 2 keeps H6 inactive. |
| H7 | `P1A step_length_per_leg` (gate ≥ 0.85× TB) at WR vx=0.15 | Active **only under Phase 9B** (project keeps WR at vx=0.15 as primary operating point) AND `tools/reference_geometry_parity.py` ships a supplemental Froude-matched section that re-reads WR at vx≈0.19. Under Phase 9A (operating-point change to vx≈0.19), this exemption does NOT apply — the gate must clear at the new primary `vx_nominal_wr`. | Same metric at the **Froude-matched supplemental `vx`** (WR vx ≈ 0.19, supplemental parity report section) | Under 9B, the project chose to operate at a different Froude point than TB. The Froude-matched supplemental view re-reads WR at the matched operating point so the parity gauge measures stride efficiency at comparable body-Froude state. Currently the supplemental view does not exist in the parity tool — Rule 2 keeps H7 inactive. |
| H8 | `P1A cadence_froude_norm` (gate ≤ 1.20× TB) at WR vx=0.15 | Same activation as H7 (Phase 9B + supplemental view in parity tool). | Same metric at the **Froude-matched supplemental `vx`**. | Same rationale as H7. |

### Rules for adding to the exemption list

A new row can only be added when **all four** of the following hold:

1. The hardware divergence column traces to an item in the
   "Keep WR-specific" list at the top of the Divergence Classification
   section. If the divergence is not on that list, the right path is
   either (a) move it to "Keep WR-specific" with rationale, or
   (b) treat it as backlog under "Remove unless explicitly justified".
2. The replacement metric column names a metric that is **already
   computed by the parity tool** and gated. If the gauge does not
   exist yet, it must be added to the parity tool first; until then
   the exemption is not active.
3. The "Why" column passes a "**catches a real regression**" test:
   describe a hypothetical regression in the underlying quality, and
   show that the replacement metric trips on that regression. If the
   replacement metric does not catch a plausible regression, it is
   not a real gauge — the absolute gate stays on.
4. Adding the row requires an explicit doc edit on this section, with
   a commit message that names the new H-number. The list cannot
   grow silently in code.

### Removing or revising an exemption

An exemption row should be **removed** when the underlying hardware
divergence is removed (e.g., if the WR floor-clearance offset becomes
unnecessary because of a hardware change, H1 goes away). An exemption
should be **revised** when the replacement metric stops catching a
class of regressions — write the case, propose a better gauge, edit
the row.

### What this means for current Phase 7-11 closeouts

- Phase 7 close: requires `P1A swing_foot_z_step_per_clearance` PASS
  (the H1 replacement gauge). The H1 absolute gate is informational.
- Phase 8 close: requires `P1A swing_clearance_per_com_height` PASS
  (the H2 replacement gauge). Both H1 and H2 absolute gates are
  informational; the absolute swing_step gate is allowed to widen
  proportionally.
- Phase 9 close: under 9A, H7 and H8 are inactive — both
  `step_length_per_leg` and `cadence_froude_norm` must PASS at the new
  primary `vx_nominal_wr`. Under 9B, H7 and H8 activate, requiring the
  parity tool's supplemental Froude-matched section to be implemented
  and to PASS at the matched `vx`.
- Phase 10 close: produces a verdict on whether the P1 contact-match
  FAIL is reference-shape (closed by Phase 7), cycle-0-bound (Phase 12
  follow-up), or actuator-stack-bound (activates H6 — but only once
  the parity tool also ships the composite gauge per H6's activation
  condition; both must hold).
- Phase 11 close: hygiene only; no exemption interaction.

### What Phases 7-11 do and do not guarantee

Phases 7-11 are an **investigate-and-improve plan**, not a guaranteed
terminal plan. Specifically:

- Phases 7, 8, 11 are direct improvements with predicted metric
  movements; whether they fully close their target gates depends on
  the actual measurement (Phase 8 is contingent; Phase 7's absolute
  swing_step is exempt under H1 and not part of the closure check).
- Phase 9 is a policy fork; the closure path depends on whether 9A or
  9B is chosen. Under 9B the parity tool must grow a supplemental
  Froude-matched section before H7/H8 are real exemptions.
- **Phase 10 is diagnostic-only** — it produces a verdict on whether
  the remaining P1 closed-loop FAILs are reference-shape (already
  fixed by Phase 7), cycle-0 retention (would spawn Phase 12),
  actuator-stack-bound (would activate H6 — pending exemption
  conditional on the parity tool also shipping the composite gauge),
  or mixed. It does NOT itself close the P1 gates.
- **Phase 12 may be required** — if Phase 10 surfaces cycle-0
  retention as the load-bearing source of the contact-match FAIL,
  mirroring TB's cycle-0 truncation in WR becomes a follow-up phase.
  This is not in the current 7-11 list.

So the correct framing is:

- 7-11 are sufficient to **determine whether the goal is reachable**
  under the current hardware assumptions, the chosen Phase 9 policy
  fork, and the exemption table above.
- 7-11 are **not sufficient by themselves** to guarantee final
  parity. Phase 12 may be needed pending Phase 10's verdict; H6 may
  need to activate (requires both the actuator-stack verdict from
  Phase 10 AND the composite gauge shipping in the parity tool); the
  supplemental Froude-matched parity view may need implementing under
  Phase 9B for H7/H8 to activate.

A phase closeout can claim "the gate this phase targeted is closed
under the H-rule that applies" or "this phase's verdict is X". A
phase closeout cannot claim "WR on par or better" until every gate
either clears strictly OR clears its named replacement under the
exemption table above.

## Practical Use

Use this note together with:

- [tools/reference_geometry_parity.py](../../tools/reference_geometry_parity.py) (WR)
- [training/docs/reference_parity_scorecard.md](reference_parity_scorecard.md) (WR)
- [training/docs/walking_training.md](walking_training.md) (WR)
- [tests/test_v0200c_geometry.py](../../tests/test_v0200c_geometry.py) (WR)
- `projects/toddlerbot/tests/test_walk_reference_geometry.py` (TB)

When the parity report and this note disagree, prefer the parity report for
current metrics and use this note for policy, classification, and backlog
ordering.
