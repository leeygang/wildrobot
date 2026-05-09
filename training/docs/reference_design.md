# WildRobot Nominal Reference Design

**Status:** Stable architecture note for `v0.20.x` prior reference layer.
**Last updated:** 2026-05-08 (scope reshape — milestone detail moved to comparison.md + CHANGELOG).

---

## Purpose

This document defines the intended architecture for the WildRobot
locomotion prior-reference layer after the failed `v0.19.5`
reward-first PPO attempt.  It is the **stable architectural commitment**
record — what we're building and why.

`v0.20.0` exists because `v0.19.5` showed that residual PPO is not a good
mainline when the base walking reference does not yet own propulsion, step
release, and command tracking strongly enough.

The goal of this document is to make the design explicit enough that:

- the reference architecture has one source of truth
- downstream layers remain adapters, not competing controllers
- validation is repeatable and gate-based
- future refactors reduce code debt rather than add more local patches
- prior selection is explicit rather than implicit

### Companion docs (where to find what)

- **Active phase-by-phase TB-alignment work** (Phases 0-12, the
  tactical decomposition of the milestones below):
  [`reference_architecture_comparison.md`](reference_architecture_comparison.md).
- **Completed-phase closeouts** with measured gate movements:
  [`../CHANGELOG.md`](../CHANGELOG.md).
- **Training-side plan + G-gate contracts** (smoke YAMLs, PPO
  acceptance gates, threshold audits):
  [`walking_training.md`](walking_training.md).

---

## Scope

This document is about:

- locomotion prior / reference architecture
- `walking_ref_v1` / `walking_ref_v2` limitations and lessons
- offline prior / reference library design
- the validation **approach** (5-layer ladder + C2 stabilizer scope) —
  per-milestone task lists are tracked in `reference_architecture_comparison.md`
- the milestone framework (v0.20.0-A through D as headers) — detailed
  exit criteria and per-phase tactical work are in comparison.md;
  closeouts are in CHANGELOG

This document is not about:

- PPO reward tuning
- long-run training strategy
- hardware deployment steps
- full MPC adoption
- per-phase tactical execution (tracked in comparison.md)
- per-phase closeout measurements (tracked in CHANGELOG)

---

## Main Design Decision

The locomotion stack should follow this contract:

```text
command + robot state
    ->
walking prior / reference
    ->
offline reference library lookup / preview
    ->
nominal targets + preview annotations
    ->
policy
    ->
joint targets
```

The intended ownership is:

- the prior reference owns locomotion semantics
- IK / nominal target shaping owns execution adaptation
- PPO owns correction, robustness, and adaptation only after the prior is
  validated

Active first contract:

- ToddlerBot-style bounded residual policy around library-backed `q_ref`

Planned upgrade compatibility:

- keep the same offline prior asset and preview interface if later policy
  versions move toward a Cassie-style broader-authority contract

Only one layer should own locomotion semantics, and for the active `v0.20`
path it does so at generation time:

- whether progression opens at all
- which gait phase / contact schedule is encoded
- how much forward swing release is planned
- how foothold progression is shaped across the command family

That layer is the offline prior generator, not the runtime adapter.

---

## Pivot Decision

`v0.19.5` did not fail because PPO was configured slightly wrong. It failed
because the walking prior still did not own enough of the locomotion task.

Observed lesson from `v0.19.4` and `v0.19.5`:

- gait existence is not enough
- visible step release is not enough
- the prior must already own meaningful command tracking, step amplitude, and
  multi-seed stability before PPO resumes

So the `v0.20.0` design decision is:

- stop treating reward tuning as the mainline next step
- choose a solid prior family first
- validate that prior visually and metrically before PPO work resumes

---

## Prior Options

### Comparison

| Option | Development cost | Debuggability | Validation difficulty | Generalization upside | Fit for WildRobot v2 | Recommendation |
|---|---|---|---|---|---|---|
| `ZMP` / preview-style flat-ground prior | low-medium | high | medium | medium | good | recommended first prior |
| `ALIP` / capture-point / angular-momentum-aware prior | medium-high | medium | high | high | very good | recommended upgrade path |
| `Placo` / Mini-Duck-style whole-body kinematic prior | medium | medium-low | high | low-medium | mixed | use as a prototyping aid, not first source of truth |

### `ZMP`

Strengths:

- easiest prior to reason about on flat ground
- easiest to debug with COM, foothold, and phase plots
- easiest to verify visually on a position-controlled biped
- lowest implementation risk for stop, resume, and slow forward walk

Weaknesses:

- conservative
- less expressive for dynamic catch steps and momentum-rich locomotion
- likely to saturate before the full adaptation goal is met

WildRobot fit:

- good first fit because WildRobot v2 is small, servo-driven, and uses
  identical-class servos (htd45hServo, kp=21.1, ±4 Nm) on every leg
  joint including the ankle_roll DOF added by the v20 ankle_roll merge
- a `ZMP`-shaped prior is easier to keep inside reachable geometry and visible
  stepping than a more aggressive model on day one
- the ZMP prior continues to plan ankle_roll = 0 (TB-aligned convention);
  lateral correction lives in the C2 stabilizer (offline) and the PPO
  residual (closed-loop), not in the offline trajectory itself

### `ALIP`

Strengths:

- better dynamic step-placement logic
- better long-term fit for speed adaptation and disturbance handling
- stronger ceiling for broader locomotion behavior

Weaknesses:

- harder to tune and diagnose than `ZMP`
- easier to inject forward-drive channels that look plausible in plots but are
  unstable in full env dynamics

WildRobot fit:

- best medium-term direction once the first prior family is proven
- especially relevant if `ZMP` clears flat-ground walking but stalls on command
  breadth, push response, or dynamic step placement

### `Placo` / Mini-Duck-style prior

Strengths:

- fast way to author coherent whole-body motion
- useful for visualizing startup, stop, and turn sequences
- useful as a prototyping environment for candidate references

Weaknesses:

- easiest option to over-trust visually
- kinematic coherence does not prove dynamic viability
- harder to make it the single source of truth for support permission and step
  release under real contact dynamics

WildRobot fit:

- useful as a sidecar tool for prototyping and inspection
- not the recommended mainline prior source of truth on current hardware

### Recommendation

For WildRobot `v0.20.x`:

1. start with a `ZMP`-shaped command-conditioned prior family
2. freeze the contract so the same PPO stack can consume an `ALIP` upgrade
   later
3. use `Placo` only for rapid kinematic prototyping and visual authoring
4. do not start from mocap or pure kinematic clips as the mainline walking
   prior

---

## Active `v0.20` Architecture

### Runtime-FSM decision

The runtime walking FSM is deprecated for the active `v0.20` path.

This means:

- no active runtime `support_health -> progression_permission -> mode-transition`
  controller in the deployed walking stack
- no runtime `walking_ref_v2`-style gatekeeper deciding when swing may release
- no split-brain semantics between a reference FSM and a policy/runtime adapter

If old `walking_ref_v1` / `walking_ref_v2` code is reused, it is allowed only
for:

- offline trajectory generation
- historical diagnostics
- side-by-side comparison against the new prior path

It is not the active runtime controller.

### Mainline architecture

The active `v0.20` architecture is:

```text
command
    ->
offline prior generator
    ->
command-indexed reference library
    ->
runtime lookup / interpolation / preview service
    ->
reference preview + nominal `q_ref`
    ->
policy
    ->
joint targets
    ->
runtime / hardware
```

The mainline policy contract is ToddlerBot-style first, Cassie-compatible
later:

- `v0.20.x` first policy: bounded residual around library-backed `q_ref`
- later upgrade path: broader direct policy authority using the same reference
  preview asset

### Layer responsibilities

**Layer 0: Offline prior generation**

Responsibilities:

- generate command-conditioned reference trajectories offline
- compute gait phase, footholds, pelvis / COM targets, and contact annotations
- validate the trajectory family before policy training starts

Allowed implementations:

- `ZMP` / preview-style planner
- later `ALIP` upgrade
- optional `Placo`-assisted authoring or prototyping

Reference starting point:

- algorithm family: Kajita et al. `ZMP` preview control over a cart-table /
  `LIPM` model
- public implementation reference: ToddlerBot
  `toddlerbot.algorithms.zmp_planner` and
  `toddlerbot.algorithms.zmp_walk`
- intended WildRobot pipeline:
  - command bin / footstep plan
  - `ZMP` preview-control COM plan
  - swing-foot trajectory generation
  - IK / nominal joint target generation
  - offline lookup-table export indexed by command

**Layer 1: Reference library asset**

Responsibilities:

- store reference trajectories indexed by command
- support interpolation between nearby command bins
- expose both task-space targets and nominal joint-space targets where useful

This is the active source of locomotion semantics.

Minimum schema per library entry:

- command key:
  - `vx` for the first branch
  - later extension points for `vy` and `yaw_rate`
- timing:
  - `dt`
  - step period / cycle length
  - normalized phase
- nominal targets:
  - `q_ref`
  - optional `dq_ref`
- task-space preview:
  - pelvis pose target
  - COM target
  - left/right foot pose targets
- annotations:
  - stance foot / support side
  - contact state or contact mask
  - stop / resume markers where applicable
- metadata:
  - generation version
  - command-bin bounds
  - feasibility / validation flags

**Layer 2: Runtime reference service**

Responsibilities:

- lookup / interpolate the library at runtime
- expose a short preview window to the policy
- emit current nominal `q_ref`, future targets, phase, and contact annotations

This layer must not contain a new locomotion FSM.

**Layer 3: Policy**

`v0.20` first contract:

- bounded residual policy
- imitation-dominant hybrid reward
- ToddlerBot-style reference-conditioned training

Future upgrade path:

- larger action authority
- longer observation / action history
- Cassie-style reference-guided direct policy if the residual contract
  saturates

**Layer 4: Target adaptation and runtime**

Responsibilities:

- lightweight target shaping only
- actuator-safe clipping
- optional small smoothing or delay compensation
- no new gait semantics

### Single source of truth

For the active `v0.20` path, the single source of truth is:

- the offline reference library asset

Not:

- a runtime FSM
- an adapter-side gating controller
- a PPO policy inventing phase structure

### ToddlerBot-first, Cassie-compatible path

This architecture is chosen specifically to support both phases:

1. **ToddlerBot-first**
   - offline reference asset
   - bounded residual policy around `q_ref`
   - imitation-dominant hybrid reward
2. **Cassie-compatible later**
   - keep the same reference preview asset
   - expand policy authority
   - add longer history and more direct control if the residual contract limits
     adaptation

The upgrade path works only if the prior is a reusable offline asset.
That is why the runtime FSM is removed from the active path instead of being
kept as the mainline controller.

---

## Adapter Design

The active runtime adapter is intentionally simple.

Its job is to convert offline-prior outputs into executable targets without
reintroducing gait semantics.

It should:

- consume the library lookup output directly
- run IK where the library stores task-space targets
- emit joint targets from stored `q_ref` where joint-space targets are already
  available
- clip to actuator and joint limits
- apply lightweight smoothing or rate limiting when required for actuator
  safety
- apply explicit safe fallbacks for unreachable targets

It should not:

- infer or reopen gait phase
- recompute support-health, progression, or release gates
- reinterpret stop / resume semantics
- introduce a second locomotion controller through hidden scaling rules

If runtime shaping exists, it must be:

- mechanical
- safety-oriented
- explicitly bounded and documented
- unable to change the gait family encoded by the offline prior

Historical adapter-side support-scaling logic from the `walking_ref_v2` era now
belongs to
[reference_design_history.md](reference_design_history.md), not to the active
`v0.20` architecture.

---

## Validation Approach

Validation should follow a layered ladder.

### Active `v0.20.0` validation ladder

This is the active validation contract for the prior pivot.

The prior is not considered valid because it "looks reasonable" in a viewer.
For the active offline-prior path, it must pass all five layers:

1. `offline reference asset quality`
2. `reference + IK playback`
3. `short-horizon MuJoCo feasibility`
4. `ToddlerBot policy learning gate`
5. `policy multi-command sweep`

### Layer 1: `offline reference asset quality`

What to verify visually:

- step timing is regular
- alternating step-out is visible
- stop commands settle into quiet standing
- resume commands re-open stepping without a jump
- pelvis / COM preview stays bounded

What to verify metrically:

- commanded foothold mean at the operating-point cmd matches
  the kinematic identity `step = vx · cycle_time / 2` within
  numerical precision (Phase 9A: vx=0.265 → step≈0.095 m).
- previewed step period stays within `0.45-0.75 s` (Phase 6
  selected `cycle_time=0.72 s` to match TB).
- pelvis and COM preview are continuous
- target-rate limits are not violated
- library coverage includes:
  - quiet stand
  - stop / resume
  - operating-point walk (Phase 9A: vx=0.265)
  - bracket bins around the operating point for robustness probes
    (`tools/v0200c_closeout.py::_VX_BINS` controls the matrix)

### Layer 2: `reference + IK playback`

What to verify visually:

- feet land where the prior intended
- stance switching is clearly visible
- no foot scissoring, severe toe drag, or obvious self-collision posture
- stop, resume, and low-speed stepping remain coherent after IK

What to verify metrically:

- realized nominal touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`
- IK reachability stays near `1.0`
- joint targets remain inside configured safety margins
- nominal foot-clearance floor remains positive throughout swing

### Layer 3: `short-horizon MuJoCo feasibility` (v0.20.0-C closeout territory)

This replaces the old "open-loop must walk 400+ steps" gate for the active
offline-prior path.  Uses the existing MuJoCo actuator model for feasibility
checking.  Hardware SysID verification is deferred to `v0.20.2` and gates
sim-to-real deployment, not ZMP library generation.

> **Acceptance reframe (2026-05-09, post-Phase-9A v0.20.0-C closeout
> at vx=0.265).**  The original Layer 3 used absolute pass gates
> ("survive ≥1 full gait cycle open-loop", "0% systematic joint-limit
> hits") that were designed in the v0.20.0-C era when the operating
> point was vx=0.15 and the prior was supposed to be open-loop-
> rescuable by a bounded stabilizer.  Empirical measurement at the
> May-08 closeout (CHANGELOG `v0.20.1-c1c2-closeout-phase9A`) shows:
> 1. **No position-PD-only small biped survives long open-loop at any
>    walking vx.**  TB-2xc / TB-2xm at TB's own operating point
>    (vx=0.15) survives 21-23 ctrl steps; WR @ vx=0.265 survives 54
>    ctrl steps.  Both fall via pitch termination.
> 2. **WR strictly outlasts TB** at the analogous operating point
>    (2.3× longer survival, 1/2 the drift, WR produces forward
>    touchdowns where TB does not).
>
> The right Layer 3 gate is therefore **comparative**, not absolute:
> "is WR at least as good as TB on each closed-loop gate when both
> are measured at their respective operating points?"  This matches
> the parity tool's `--tb-vx` flag (CHANGELOG
> `v0.20.1-parity-tool-asymmetric-vx-comparison`).

What to verify visually:

- the MuJoCo actuator model can track the reference for short rollouts without
  immediate geometric collapse
- startup, stop, and one-to-two gait cycles look trackable rather than rigid or
  obviously impossible
- failures should look like tracking limits or contact mismatch, not like an
  incoherent reference

What to verify metrically — by gate type:

**Absolute prior-sanity gates (must pass):**
- Kinematic playback at the in-scope command bins: q_ref produces
  forward foot-translation matching cmd (realised/cmd ratio ≥ 0.95
  in kinematic mode at the operating point).
- Fixed-base PD tracking at the in-scope command bins: per-leg-joint
  RMSE within the actuator-feasibility budget; any-joint saturation
  rate < 5%.
- Pre-PPO sanity: kinematic + fixed-base both PASS.  This confirms
  the prior is sane and the actuator stack can track it when balance
  is held.

**Comparative closed-loop gates (must be ≥ TB at the analogous
operating point, not "must PASS an absolute floor"):**
- WR closed-loop survival at WR's operating vx ≥ TB closed-loop
  survival at TB's operating vx.
- WR closed-loop forward-velocity sign / drift better than or
  comparable to TB.
- WR closed-loop contact-phase match ≥ TB at the analogous operating
  point.
- WR produces real touchdown events (positive `td_step_mean`) at
  least as often as TB.

**Closed-loop absolute floors (informational, NOT gating):**
- Open-loop survival in ctrl steps — useful as a regression detector
  vs prior runs, but NOT a PASS gate.  No small position-PD biped
  meets the original "≥ 64 ctrl steps" budget at its operating
  point.
- C2 stabilizer step length / ratio — useful for catching stabilizer
  bugs (e.g. the post-merge cp_gain sign regression) but NOT a PASS
  gate at the operating point.

### Layer 4: `ToddlerBot policy learning gate`

This is the main unblock gate for the active `v0.20` path.

> **Acceptance reframe (2026-05-09, post-Phase-9A).**  The
> per-metric thresholds below were calibrated for vx=0.15.  Phase 9A
> shifted the operating point to vx=0.265 (TB-step/leg-matched);
> threshold magnitudes scale with operating-point vx and need to be
> updated by the smoke7+ track.  The original `eval_walk/success_rate`
> gate was already removed (CHANGELOG `v0.20.1-smoke2`); see
> `walking_training.md` Appendix A for the live G-gate contract used
> by the smoke runs.

> **Current status (2026-05-09).**  There is no promotion candidate
> PPO checkpoint for the current 21-DOF model.  The April 25 smoke7
> checkpoint is a 19-action pre-ankle-roll artifact and is obsolete.
> A fresh Phase 9A smoke must use a `vx=0.265` q_ref trajectory and
> `eval_velocity_cmd=0.265`; using a `vx=0.15` q_ref with a `0.265`
> eval command is a contract mismatch, not a valid test of prior
> quality.  The live G4 stride floor is now command-scaled: at
> `vx=0.265`, require touchdown step length `>= 0.050 m` before
> claiming the learned policy is tracking a non-shuffle prior.

What to verify visually:

- within a short training horizon, the learned policy starts to track the
  reference and produce visible stepping
- the learned gait still looks like the reference family, not a new exploit
- stop / resume behavior remains recognizable

What to verify metrically:

- reference-tracking metrics improve materially from initialization:
  - `ref/joint_track_rmse`
  - `ref/pelvis_pose_error`
  - `ref/foot_site_error`
  - `ref/contact_phase_match`
- by the early smoke horizon, the run shows directional improvement
  on `env/forward_velocity`, `eval_walk/episode_length`, and
  `term_pitch_frac` vs initialisation — exact thresholds tracked in
  `walking_training.md` Appendix A and the smoke YAML, scaled to
  the current operating-point vx.
- by the promotion horizon, the run achieves the
  per-vx-scaled `forward_velocity`, episode-length, and
  touchdown-step-length floors documented in
  `walking_training.md` G4 / G5.  In particular, the old absolute
  `0.030 m` stride floor is retained only as a lower bound; the
  Phase 9A operating point uses `max(0.030, 0.50 * vx * cycle_time/2)`.

### Layer 5: `policy multi-command sweep`

What to verify visually:

- zero command means standing, not creeping
- low command means slow but visible stepping
- increasing command does not immediately collapse into pitch dive
- stop/resume works without a discontinuous posture jump

What to verify metrically:

- zero-command drift `< 0.01 m/s`
- stop latency to quiet posture `< 1.0 s`
- resume-to-first-step latency `< 1.0 s`
- forward-speed response is monotonic non-decreasing over the tested range
- no single command bin is propped up by a qualitatively different exploit gait

### Option-specific validation emphasis

`ZMP`

- strongest on layers 1, 3, and 4
- easiest to validate with COM, foothold, stop/resume, and imitation tracking
  diagnostics

`ALIP`

- requires extra scrutiny on pitch-rate spikes, overspeed, and dynamic
  catch-step plausibility
- only upgrade here after `ZMP` clears the main gate or clearly saturates

`Placo`

- never accept viewer quality alone as validation
- always require full layer-3 through layer-5 checks before trusting it as a
  walking prior

## Execution Milestones for `v0.20.0`

The milestone framework v0.20.0-A through D defines the high-level
sequence the prior reference work must follow:

- **`v0.20.0-A` Prior contract freeze** — schema + interface decisions.
- **`v0.20.0-B` ZMP prior generator and viewer** — first concrete prior.
- **`v0.20.0-C` Reference + IK + short-horizon feasibility validation** —
  prior-sanity gates (kinematic + fixed-base PD tracking) PASS as
  absolute floors; closed-loop gates use the comparative-vs-TB rule
  (see Validation Approach Layer 3 above).
- **`v0.20.0-D` PPO unblock decision** — go/no-go gate.  Unblock
  criteria: v0.20.0-C absolute prior-sanity gates PASS + comparative
  closed-loop gates show WR ≥ TB at the analogous operating point.
  The original "C1 PASS + C2 PASS at all bins" criterion was
  aspirational and has been demonstrated unmeetable by either WR or
  TB (CHANGELOG `v0.20.1-c1c2-closeout-phase9A`).

Detailed per-milestone task lists, exit criteria, and post-shipped
closeouts are tracked in:
- [`reference_architecture_comparison.md`](reference_architecture_comparison.md)
  for active phase-by-phase TB-alignment work (Phases 0-12 are the
  finer-grained tactical decomposition of milestones A through D).
- [`../CHANGELOG.md`](../CHANGELOG.md) for completed-phase closeouts
  with measured gate movements.

C1 + C2 closeout measurement infrastructure (the script + matrix +
parsed verdicts) lives in [`tools/v0200c_closeout.py`](../../tools/v0200c_closeout.py).
Per-milestone task lists at the training granularity live in
[`walking_training.md`](walking_training.md) §v0.20.0-A through
§v0.20.0-D.  The harness scope and forbidden-input list (the
architectural commitments) live in this doc's [Validation Approach](#validation-approach)
section above.

## Historical Companion

Historical `walking_ref_v1/v2`, `M2.5`, `v0.19.x`, and `v0.19.4` DCM COM
trajectory notes now live in
[reference_design_history.md](reference_design_history.md) so this
document stays focused on the active `v0.20.x` offline-prior path.

## Relationship to other docs

- [`walking_training.md`](walking_training.md) — training-side
  milestones, smoke contracts, G-gates, threshold audits.  Carries
  the per-milestone task lists at training granularity.
- [`reference_architecture_comparison.md`](reference_architecture_comparison.md)
  — active TB-alignment phase tracker (Phases 0-12 are the tactical
  decomposition of milestones v0.20.0-A through D).
- [`../CHANGELOG.md`](../CHANGELOG.md) — closeouts with measured
  gate movements for completed phases.

This file (`reference_design.md`) carries the **stable architectural
commitments** — what we're building and why.  It changes rarely;
when an architectural decision is revisited (e.g. ZMP vs ALIP
selection, layer responsibilities, validation ladder framework) the
update lands here.  Tactical work and per-phase results land in the
other three places above.
