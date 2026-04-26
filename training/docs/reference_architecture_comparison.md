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
- `P2 swing_foot_z_step_max_m`: 0.0183 → ≈ 0.012 m
- `P1A swing_foot_z_step_per_clearance`: 1.40× → ≈ 0.92× TB → PASS
- `P1A swing_clearance_mean_m`: small ~−1 % drop (triangle has lower
  swing-frame mean than half-sine at same peak), acceptable

Exit criteria:
- WR-only probe (`uv run python tools/phase6_wr_probe.py`) shows P2
  `swing_gate` PASS at vx=0.15
- full parity refresh shows both `swing_step_per_clr` and `P2 swing_gate`
  PASS for both TB baselines
- no regression on `P0` (in-scope or full matrix), `P1A` step_len /
  cadence / clearance / DS, or `q_step` smoothness

Acceptance statement:
- success means the two swing-shape gates flip to PASS without
  regressing any currently-passing gate
- success does not require P1 closed-loop to move; Phase 10 owns the
  closed-loop interpretation

Rollback if expected movement doesn't materialize:
- if the triangular shape produces *worse* P2 step_max than the
  half-sine (unlikely but possible if `n_half` is small enough that the
  per-frame delta dominates), revert to half-sine and try a half-cosine
  variant `(1 - cos(2 * pi * frac)) / 2` which has a flatter apex than
  half-sine

### Phase 8. `foot_step_height_m` adoption (TB's 0.05 m)

Goal:
- close the normalised `swing_clearance_per_com_height` gate (currently
  0.82× TB) by adopting TB's `foot_step_height` value

Work (single config change):
- `control/zmp/zmp_walk.py:95`: `foot_step_height_m: float = 0.04` → `0.05`
- the Phase 5 cache fingerprint will pick up the change automatically

Sequence:
- AFTER Phase 7 — the triangular envelope absorbs the +0.01 m peak with
  less per-frame Δz than the half-sine would

Expected metric movement:
- `P1A swing_clearance_per_com_height`: 0.143 → ≈ 0.175 (gate 0.149) → PASS
- `P1A swing_clearance_mean_m`: 0.065 → ≈ 0.075 m
- `P2 swing_foot_z_step_max_m`: with triangle envelope, +0.01 m apex
  spreads across the swing half-window, so per-frame Δz scales by
  `(0.05 + 0.025) / (0.04 + 0.025) = 1.15×`. Combined with Phase 7
  effect: ≈ 0.014 m absolute, still under the 1.10× TB ≈ 0.011 m gate
  by ~30 % — borderline, must be verified after running

Exit criteria:
- `swing_clearance_per_com_height` ≥ 0.85× TB (= 0.149) → PASS for both
  TB baselines
- no regression on Phase 7 swing-step gates (both must still PASS)
- no regression on P0 swing margin (must remain > +5 mm)

Fallback:
- if Phase 8's smoothness regression breaks the Phase 7 gates, use
  0.045 as a halfway step (lifts the metric to ≈ 0.16, still PASS,
  with smaller smoothness cost)

Acceptance statement:
- success means `clearance_per_h` flips to PASS while preserving
  Phase 7's gate wins

### Phase 9. Operating-`vx` curriculum decision (Froude equivalence)

Goal:
- close the two hardware-Froude-bounded normalised P1A gates
  (`step_length_per_leg = 0.56× TB`, `cadence_froude_norm = 1.25× TB`)
  that no planner tuning can address at vx=0.15

Work (curriculum / training-config decision, not a planner change):

1. Decide and document in `training/docs/walking_training.md` whether
   the v0.20.x walking smoke commands `vx ∈ {0.10, 0.15}` should shift
   to `vx ∈ {0.13, 0.19}` (Froude-equivalent to TB's `{0.10, 0.15}`)
   - Froude scaling: TB uses `vx² / (g · h_TB) = 0.15² / (9.81 · 0.286)
     = 0.0080`. WR's matching point: `vx_WR = sqrt(0.0080 · 9.81 · 0.458)
     = 0.190 m/s`
2. Update the parity scorecard's `vx_nominal` in
   `tools/reference_geometry_parity.py` (currently 0.15). Either:
   - add a `--vx-nominal-wr` flag so WR is read at 0.19 while TB stays
     at 0.15
   - or split the report into "absolute-vx parity" (current) vs
     "Froude-matched parity" (new section), keeping both visible
3. Decide the training-side smoke vx. If the smoke command grid moves,
   `tests/test_v0200c_geometry.py` and the walking_training.md G4/G5
   contract need to follow. If only the parity tool's `vx_nominal`
   moves, training stays at 0.15 and the parity report carries the
   Froude-matched read for "is the prior on par with TB at the matched
   operating point"

Expected metric movement (at WR vx=0.19, predicted):
- `step_length_per_leg`: 0.56× → ≈ 0.85–0.95× TB → PASS (gate ≥ 0.85×)
- `cadence_froude_norm`: 1.25× → ≈ 1.0× TB → PASS (gate ≤ 1.20×)

Exit criteria:
- an explicit decision is in `walking_training.md` with rationale for
  the chosen `vx_nominal_wr`
- parity tool reports both absolute and Froude-matched normalised P1A;
  Froude-matched view shows both gates PASS
- training-side curriculum (if moved) is reflected in the smoke test
  command grid

Acceptance statement:
- success means Froude-matched normalised P1A clears both gates with
  explicit doc rationale
- if absolute-vx parity is kept as the primary read, the rationale must
  say *why* and accept the residual gap as a documented hardware-bounded
  limitation

This is a curriculum decision, not a parity bug fix. If the project
chooses to keep WR at vx=0.15 for safety / hardware reasons, document
that and accept the two normalised FAILs as bounded. The plan should
not force a knob-twist that violates a training-side constraint.

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
   contact within a single swing? Phase 7's triangular envelope should
   drop this to near-zero. If it doesn't, that means the actuator stack
   is producing the dips, not the planner.
4. Decide. Three possible verdicts:
   - "Cycle-0 drag is load-bearing" → mirror TB truncation in a
     follow-up Phase 12
   - "Mid-swing tap-down was load-bearing, Phase 7 fixed it" → no
     further work; gate may now PASS
   - "Both clean, residual is actuator stack" → accept as P1-test-design
     limitation per the scorecard's own note; document explicitly so
     future readers don't re-litigate

Exit criteria:
- a decision is recorded with evidence (numbers from items 1–3)
- if the verdict is "fixable", a concrete follow-up phase is added to
  this doc
- if the verdict is "accepted artifact", the parity scorecard's gate is
  annotated to reflect the architectural reason

Acceptance statement:
- success means the contact-match FAIL has a verdict (fixable /
  fixed-by-Phase-7 / actuator-bound), not a vague "WR is worse than TB"
- the closed-loop layer's narrative becomes either "all gates either
  PASS or have documented hardware reasons" or "Phase 12 follows"

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

### Expected end state after Phases 7-9

| Layer | Pre-Phase-7 | Post-Phase-9 |
|---|---|---|
| P0 (in-scope + full matrix) | PASS | PASS |
| P1A absolute | PASS | PASS |
| P1A normalised (4 gates) | 0/4 PASS | **4/4 PASS** (`clearance_per_h` via P8; `step_per_leg` + `cadence_norm` via P9 Froude-matched read; `swing_step_per_clr` via P7) |
| P2 (4 gates) | 3/4 PASS | **4/4 PASS** (`swing_step` via P7) |
| P1 closed-loop | 3/6 PASS | 3/6 PASS + Phase 10 verdict on remaining 3 |

After Phases 7-9 land, WR is on par or better than TB on every gate
that doesn't depend on actuator-stack architecture. Phase 10 closes the
interpretation of the closed-loop residual.

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
   (closes 2 gates with one planner-local change)
9. Phase 8 to adopt TB's `foot_step_height_m = 0.05` (closes
   `clearance_per_h`); sequence after Phase 7
10. Phase 9 to make the operating-`vx` curriculum decision (closes the
    two hardware-Froude-bounded normalised gates); independent of 7 / 8
11. Phase 10 to diagnose the P1 closed-loop contact-match FAIL;
    sequence after Phase 7
12. Phase 11 to clean up the residual `lin_vel_z` planner-side
    consumption; independent, low priority

Phases 7 + 9 + 11 can run in parallel; 8 needs 7; 10 needs 7.

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

- [tools/reference_geometry_parity.py](../../tools/reference_geometry_parity.py) (WR)
- [training/docs/reference_parity_scorecard.md](reference_parity_scorecard.md) (WR)
- [training/docs/walking_training.md](walking_training.md) (WR)
- [tests/test_v0200c_geometry.py](../../tests/test_v0200c_geometry.py) (WR)
- `projects/toddlerbot/tests/test_walk_reference_geometry.py` (TB)

When the parity report and this note disagree, prefer the parity report for
current metrics and use this note for policy, classification, and backlog
ordering.
