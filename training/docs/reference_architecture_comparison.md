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

## Architecture-Driven Plan

The current plan should close the WR-vs-TB quality gap primarily through
architecture changes, not local optimization.

### Priority 1. Remove unjustified swing-z discontinuity

Change WR's swing-foot clearance shaping so the WR-specific floor-clearance
compensation goes to zero at lift-off and touchdown.

Why first:
- directly targets the current `P0` swing-margin weakness
- directly targets the current `P2` swing-step weakness
- no schema or env contract change required

### Priority 2. Decide and document runtime target semantics

Make an explicit architectural decision:

- default to TB-style command-integrated body target
- keep planner-phase target only if WR has a written, measured reason

Why second:
- this controls whether Stage 7 and Stage 8 should align more closely to TB
- it prevents further local tuning on top of an unresolved target philosophy

### Priority 3. Add TB-style realized-reference enrichment to WR

After IK, run fixed-base FK replay and persist realized body/site quantities as
part of the WR reference asset.

Why third:
- aligns WR asset content with TB
- removes the planner-intent vs realized-data mismatch
- makes parity measurement and downstream consumers use the same content type

### Priority 4. Align WR runtime service to the chosen target semantics

If Priority 2 chooses TB-style command tracking, add the runtime path-state
integration and root-pose composition needed to make WR consume the reference
like TB.

Why fourth:
- it is the main architectural move needed to make WR's body target semantics
  TB-like

### Priority 5. Improve lifecycle parity where it matters

Build a repeatable cached `vx` grid at env init or on disk when multi-command
coverage, reproducibility, or debugging makes it worth doing.

Why fifth:
- useful for broader parity and experiment repeatability
- not the main driver of the current normalized quality gap

### Priority 6. Use local tuning only after the architecture backlog is under control

Examples:
- `cycle_time`
- `foot_step_height`
- other scalar planner config

These are still important, but they should refine the aligned architecture, not
stand in for it.

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
