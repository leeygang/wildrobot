# WR Reference Architecture Plan vs ToddlerBot

**Status:** Canonical design note for `v0.20.x` reference alignment
**Last updated:** 2026-04-25
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
[reference_parity_scorecard.md](/home/leeygang/projects/wildrobot/training/docs/reference_parity_scorecard.md):

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
     [tests/test_walk_reference_geometry.py](/home/leeygang/projects/toddlerbot/tests/test_walk_reference_geometry.py)
   - WR-side parity script:
     [tools/reference_geometry_parity.py](/home/leeygang/projects/wildrobot/tools/reference_geometry_parity.py)
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

The current parity harness reports four load-bearing gaps.

### Geometry

- WR passes the current in-scope `P0` gate at `vx <= 0.15`
- WR still trails TB on high-`vx` diagnostic robustness
- WR still trails TB on swing-floor margin

Interpretation:
- WR is acceptable for the current contract
- WR is not yet TB-level on geometry robustness outside that narrow scope

### FK-realized gait shape

- WR remains shorter-step than TB
- WR compensates with faster cadence
- WR's absolute clearance can look generous, but the size-normalised view is
  less favorable

Interpretation:
- the current WR prior is not yet operating at a TB-like proportional gait
  point

### Smoothness

- WR is still rougher than TB on swing-foot lift-off / touchdown behavior

Interpretation:
- this is an architecture-level planner-shape issue, not a PPO issue

### Free-base zero-residual trackability

- WR still trails TB on some replay metrics
- some absolute replay differences are actuator-stack artifacts
- the important point is that WR still does not yet clear the combined
  reference-quality bar strongly enough to claim parity

Interpretation:
- the reference still needs architecture work before PPO-side gains should be
  credited as "reference parity"

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
| Fixed-base FK replay / harvested body-site fields | yes, via `mujoco_replay` | no, planner output stored directly | backlog |

This is still an important architecture gap. TB stores realized body/site
quantities; WR historically stored planner intent and reconstructed realized
quantities downstream. The parity harness now re-FKs WR for fairness, which is
useful, but it also shows the underlying architecture is still misaligned.

### Stage 6. Asset lifecycle

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Storage model | pre-baked lookup asset | generated on demand per `vx` | backlog if multi-cmd parity and repeatability matter |

This is not the main cause of the current reference-quality gap, but it is
still a TB-architecture mismatch.

### Stage 7. Runtime reference service

| Stage | ToddlerBot | WildRobot | Classification |
|---|---|---|---|
| Runtime service | path-state integration + lookup + root composition | trajectory slice / preview only | unresolved design decision |

This is the one remaining architectural question that needs an explicit answer.

TB's runtime target is command-integrated: "where should the body be by now if
the command had been followed since reset?"

WR's runtime target is planner-phase-based: "what does the planner say the
body should look like at this gait phase?"

For this project, the default answer should be:

- align to TB's command-integrated target semantics unless WR has an explicit,
  measured reason to keep the current planner-phase target

Until that justification exists, this is backlog, not a neutral alternative.

### Stage 8. Env reward consumption

Most reward terms are conceptually aligned. The main semantic difference is
still the torso/body path target:

- TB compares against the command-integrated body target
- WR compares against planner-side pelvis trajectory

This should be described narrowly as a reward-target semantics difference, not
as proof that WR and TB consume the reference identically.

## Divergence Classification

The current architecture gaps sort into three buckets.

### Keep WR-specific

- 4-DoF leg instead of TB's 6-DoF leg
- resulting ankle / foot-orientation constraints that are truly imposed by WR
  hardware
- size and COM-height differences

These are not alignment failures. They are constraints the parity system must
normalize around.

### Remove unless explicitly justified

- swing-z boundary discontinuity
- planner-intent-only storage instead of TB-style realized FK enrichment
- on-demand per-`vx` generation as the only lifecycle
- planner-phase torso target instead of TB-style command-integrated target

These should be treated as backlog by default.

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
- keep [tools/reference_geometry_parity.py](/home/leeygang/projects/wildrobot/tools/reference_geometry_parity.py)
  as the single comparison entry point
- keep
  [reference_parity_scorecard.md](/home/leeygang/projects/wildrobot/training/docs/reference_parity_scorecard.md)
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
   choices.** Today
   `wildrobot_env.py:801-818` reads
   `ref_pelvis_pos = win["pelvis_pos"]` (the LIPM-LQR COM trajectory
   from `control/zmp/zmp_walk.py`) and uses it as the
   `_reward_torso_pos_xy` target. Any planner change (ALIP, mocap,
   different cost weights) silently shifts the reward target.
   Path-integrated targets keep the reward signal stable across
   planner iterations.
3. **TB consumes the path-integrated form** at
   `toddlerbot/locomotion/mjx_env.py:2264-2270`
   (`torso_pos_ref = info["state_ref"]["qpos"][:2]`, where
   `state_ref["qpos"][:2]` is built from `path_state.path_pos`
   integrated at runtime per
   `toddlerbot/reference/walk_zmp_ref.py:228-236`). Aligning here
   means WR's reward consumes the same semantic class TB does.
4. **The runtime computation is simpler, not more complex.** WR's
   env already threads `loc_ref_offline_step_idx` through state
   (`wildrobot_env.py:1342, 1480`). Computing
   `t_since_reset = step_idx · dt` and
   `path_pos = velocity_cmd · t_since_reset` is one multiply and one
   add per step — strictly less work than the array lookup that
   produces today's `win["pelvis_pos"]`.
5. **The planner's `pelvis_pos` field stays useful** (it's still the
   target the IK solved against, so it stays as the sanity-check
   diagnostic in `feet_track_raw` and similar). Only the reward
   target moves; the asset schema does not need to grow or shrink.

**Why the planner-phase target was rejected:**

- *Materially better parity?* No. The planner's COM trajectory and
  the command-integrated path agree to within sub-mm in steady
  state at the smoke command (LIPM Q=I, R=0.1 produces a tightly-
  tracking LQR with mm-scale lateral oscillation), so the reward
  signal differs by noise in steady state. The planner-phase target
  carries the from-rest startup transient and the LIPM lateral
  swing into the reward; the command-integrated target does not.
  Neither has been shown to produce a measurable reward-quality
  benefit; the path-integrated form is cleaner.
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
  `toddlerbot/algorithms/zmp_walk.py:153-212 mujoco_replay`).
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

Concrete code changes:

1. **`control/references/runtime_reference_service.py`** — add a
   small `compute_path_state(t, velocity_cmd, yaw_rate_cmd) ->
   {path_pos, path_rot, lin_vel, ang_vel}` static helper. Mirrors
   `toddlerbot/reference/motion_ref.py::integrate_path_state` but
   stateless: takes elapsed time and command, returns the
   integrated frame. Pure-NumPy + JAX-friendly; no service state
   change.
2. **`training/envs/wildrobot_env.py`**:
   - Compute `t_since_reset = step_idx · self.dt` (already
     available via `loc_ref_offline_step_idx`).
   - Compute `path_state = compute_path_state(t_since_reset,
     velocity_cmd, yaw_rate_cmd_or_zero)`.
   - In `_reward_torso_pos_xy`, replace
     `ref_pelvis_pos[:2]` with `path_state.path_pos[:2]`.
   - In any future torso_quat reward, use
     `path_state.path_rot · default_rot`.
   - Leave `ref_pelvis_pos` available for the diagnostic
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

Open questions for the implementer of Phase 4:

- Should the path-state integration also handle `yaw_rate_cmd`
  immediately, or defer to v0.20.4 (yaw-cmd milestone)?
  Recommendation: build the helper to accept `yaw_rate_cmd` but
  set it to 0 for the smoke; the helper is identical for the
  yaw-stationary case, just future-proofed.
- Should `path_state` be cached in `WildRobotInfo` to avoid
  recomputation across the reward terms? Recommendation: yes —
  compute once per step, attach to `state_info`, read from
  reward functions.

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

## What "On Par or Better" Means

WR is on par with or better than TB only when all of the following are true:

- WR clears the shared `P0` gate and is not materially weaker on the full
  geometry matrix
- WR's size-normalised `P1A` gait shape is on par with or better than TB
- WR's `P2` smoothness is on par with or better than TB
- WR's `P1` free-base replay is on par with or better than TB, after reading
  the normalised metrics first and the absolute metrics as a sanity floor
- WR still satisfies its own `G4/G5/G6/G7` policy contract

If a future WR change improves PPO but does not improve the parity scorecard at
the architecture layer, it should not be claimed as "reference parity
improvement."

## Practical Use

Use this note together with:

- [tools/reference_geometry_parity.py](/home/leeygang/projects/wildrobot/tools/reference_geometry_parity.py)
- [training/docs/reference_parity_scorecard.md](/home/leeygang/projects/wildrobot/training/docs/reference_parity_scorecard.md)
- [training/docs/walking_training.md](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)
- [tests/test_v0200c_geometry.py](/home/leeygang/projects/wildrobot/tests/test_v0200c_geometry.py)
- [tests/test_walk_reference_geometry.py](/home/leeygang/projects/toddlerbot/tests/test_walk_reference_geometry.py)

When the parity report and this note disagree, prefer the parity report for
current metrics and use this note for policy, classification, and backlog
ordering.
