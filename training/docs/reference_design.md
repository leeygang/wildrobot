# WildRobot Nominal Reference Design

**Status:** Active design note for `v0.20.x` prior reference validation  
**Last updated:** 2026-04-17

---

## Purpose

This document defines the intended architecture and validation plan for the
WildRobot locomotion prior-reference layer after the failed `v0.19.5`
reward-first PPO attempt.

`v0.20.0` exists because `v0.19.5` showed that residual PPO is not a good
mainline when the base walking reference does not yet own propulsion, step
release, and command tracking strongly enough.

The goal of this document is to make the design explicit enough that:

- the reference architecture has one source of truth
- downstream layers remain adapters, not competing controllers
- validation is repeatable and gate-based
- future refactors reduce code debt rather than add more local patches
- prior selection is explicit rather than implicit

---

## Scope

This document is about:

- locomotion prior / reference architecture
- `walking_ref_v1` limitations
- `walking_ref_v2` target design
- validation of prior assets, short-horizon feasibility, and first-policy
  learning
- `v0.20.0` execution milestones

This document is not about:

- PPO reward tuning
- long-run training strategy
- hardware deployment steps
- full MPC adoption

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

Only one layer should decide:

- whether progression is allowed
- which hybrid gait mode is active
- how much forward swing release is permitted
- how foothold progression is shaped under instability

That layer is the reference layer.

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

- good first fit because WildRobot v2 is small, servo-driven, and lacks active
  ankle roll
- a `ZMP`-shaped prior is easier to keep inside reachable geometry and visible
  stepping than a more aggressive model on day one

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

**Layer 1: Reference library asset**

Responsibilities:

- store reference trajectories indexed by command
- support interpolation between nearby command bins
- expose both task-space targets and nominal joint-space targets where useful

This is the active source of locomotion semantics.

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

The nominal target adapter is responsible for converting reference intent into
executable joint targets.

It should:

- use reference outputs directly
- solve IK
- clip to actuator/joint limits
- apply safe fallbacks for unreachable targets
- preserve home-centered defaults

It should not:

- recompute its own gait mode
- recompute a different support-health controller
- invent separate progression logic

If additional execution safeguards are needed, they should be:

- mechanical
- safety-oriented
- explicitly documented as adapter logic

not implicit locomotion semantics.

### Adapter-side support scaling rule

The env currently contains adapter-side helper logic such as
`_loc_ref_support_scales()` that scales swing-x and pelvis-pitch channels from
overspeed and pitch signals.

This is risky because it can become a second locomotion controller:

- the reference already computes `support_health`
- the reference already computes `progression_permission`
- the reference already decides whether swing should release

So the design position for `M2.5` is:

- the reference remains the authoritative owner of progression semantics
- adapter-side support scaling is allowed only as an execution-safety layer
- adapter-side support scaling may only further restrict a command
- adapter-side support scaling must never reopen or reinterpret progression

In practice, this means:

- `walking_ref_v2` decides release
- the adapter may apply a bounded multiplicative brake for actuator safety
- the brake should consume reference-owned signals where possible
- if adapter-side braking continues to affect debugging materially, the next
  cleanup step should remove independent instability estimation and consume only
  reference-owned support signals

The preferred end state is still single-source-of-truth:

- one authoritative support-health / progression signal in the reference
- adapter logic reduced to mechanical execution safeguards

---

## Validation Approach

Validation should follow a layered ladder.

### Active `v0.20.0` validation ladder

This is the active validation contract for the prior pivot.

The prior is not considered valid because it "looks reasonable" in a viewer.
For the active offline-prior path, it must pass all five layers:

1. `offline reference asset quality`
2. `reference + IK playback`
3. `servo-model and short-horizon feasibility`
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

- commanded foothold mean at `cmd=0.15` is `>= 0.045 m`
- previewed step period stays within `0.45-0.75 s`
- pelvis and COM preview are continuous
- target-rate limits are not violated
- library coverage includes:
  - quiet stand
  - stop / resume
  - low-speed walk
  - nominal command sweep points

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

### Layer 3: `servo-model and short-horizon feasibility`

This replaces the old "open-loop must walk 400+ steps" gate for the active
offline-prior path.

What to verify visually:

- the simulated servo model can track the reference for short rollouts without
  immediate geometric collapse
- startup, stop, and one-to-two gait cycles look trackable rather than rigid or
  obviously impossible
- failures should look like tracking limits or contact mismatch, not like an
  incoherent reference

What to verify metrically:

- simulated leg-joint tracking RMSE stays within the configured actuator budget
- repeated target saturation does not dominate the rollout
- no systematic joint-limit hits occur in the validated command range
- short-horizon replay survives at least one full gait cycle without immediate
  pitch-faceplant termination
- actuator SysID replay error is good enough to trust the simulated servo model

### Layer 4: `ToddlerBot policy learning gate`

This is the main unblock gate for the active `v0.20` path.

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
- by the early smoke horizon, the run achieves all of:
  - `env/forward_velocity >= 0.04 m/s`
  - `eval_walk/episode_length >= 150`
  - `term_pitch_frac < 0.50`
- by the promotion horizon, the run achieves all of:
  - `env/forward_velocity >= 0.075 m/s`
  - `eval_walk/success_rate >= 0.60`
  - `tracking/cmd_vs_achieved_forward <= 0.075 m/s`
  - touchdown step length mean `>= 0.03 m`

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

## Execution Milestones For `v0.20.0`

### `v0.20.0-A`: Prior contract freeze

Goal:

- define one stable offline-prior contract that both `ZMP` and later `ALIP`
  variants can satisfy
- make the contract ToddlerBot-first and Cassie-compatible later

Required outputs:

- command-indexed library schema
- reference preview window contract
- nominal `q_ref` contract
- task-space preview contract
- phase / contact annotation contract
- stop / resume handling contract

Exit criteria:

- one documented prior schema exists
- one viewer and one trace logger can consume that schema
- rate limits and bounds are explicitly defined
- active runtime FSM path is marked deprecated in the design docs

### `v0.20.0-B`: `ZMP` prior implementation

Goal:

- implement the first production prior as a `ZMP`-shaped offline reference
  library
- verify that the servo model is good enough to trust the library in MuJoCo

Visual gate:

- at `cmd=0.15`, feet clearly step out instead of shuffling
- stop command settles to quiet standing
- resume command restarts stepping without a jump

Metric gate:

- commanded foothold mean `>= 0.045 m`
- step period remains within `0.45-0.75 s`
- preview continuity and rate limits pass on command sweeps
- actuator SysID replay error is within the accepted servo-model budget for the
  tracked leg joints
- no validated reference trajectory requires systematic joint-limit violation

### `v0.20.0-C`: Prior + IK playback validation

Goal:

- prove the prior remains coherent after IK and nominal target adaptation

Visual gate:

- touchdown locations match the intended stepping pattern
- no obvious toe drag, scissoring, or pathological posture

Metric gate:

- realized touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`
- IK reachability stays near `1.0`
- safety margin violations remain zero

### `v0.20.0-D`: Servo-feasibility and short-horizon env gate

Goal:

- prove the offline prior is trackable enough in full MuJoCo dynamics to be a
  valid imitation target for PPO
- do not require long-horizon open-loop walking from the offline prior

Visual gate:

- startup, stop, and one-to-two gait cycles remain coherent in the full env
- motion looks trackable rather than instantly impossible under the simulated
  servo model
- failures look like bounded tracking limits, not incoherent reference design

Metric gate:

- short-horizon env replay survives at least one gait cycle without immediate
  catastrophic pitch termination
- simulated tracking RMSE stays within the actuator-feasibility budget
- no repeated hard saturation dominates the replay
- touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`

### `v0.20.1`: ToddlerBot-style PPO

Goal:

- train the first policy as a bounded residual, imitation-dominant
  reference-conditioned PPO
- keep the architecture compatible with a later Cassie-style direct-policy
  upgrade, but do not take that step yet

Visual gate:

- by the smoke horizon, the learned policy produces visible reference-like
  stepping rather than standing or inventing a new gait
- stop / resume remains recognizable
- the learned gait still resembles the reference family

Metric gate:

- reference-tracking metrics improve materially from initialization:
  - `ref/joint_track_rmse`
  - `ref/pelvis_pose_error`
  - `ref/foot_site_error`
  - `ref/contact_phase_match`
- by the early smoke horizon:
  - `env/forward_velocity >= 0.04 m/s`
  - `eval_walk/episode_length >= 150`
  - `term_pitch_frac < 0.50`
- by the promotion horizon:
  - `env/forward_velocity >= 0.075 m/s`
  - `eval_walk/success_rate >= 0.60`
  - `tracking/cmd_vs_achieved_forward <= 0.075 m/s`
  - touchdown step length mean `>= 0.03 m`

### `v0.20.2`: Policy coverage and breadth gate

Goal:

- make sure the first ToddlerBot-style policy is not only valid at one exact
  operating point

Visual gate:

- zero command means standing, not creeping
- low command means slow but visible stepping
- stop/resume looks controlled

Metric gate:

- zero-command drift `< 0.01 m/s`
- stop latency `< 1.0 s`
- resume-to-first-step latency `< 1.0 s`
- forward speed response is monotonic non-decreasing over the tested range

### `v0.20.3`: Cassie-upgrade decision gate

Goal:

- decide whether the ToddlerBot-style bounded residual contract is sufficient
- move to a Cassie-style broader-authority policy only if the residual contract
  is the limiting factor

Promote to Cassie-style only if:

- the offline prior library is already working
- the imitation-dominant residual policy has cleared `v0.20.2`
- remaining limitations are clearly adaptation / authority limits rather than
  prior-quality limits

If promoted, keep:

- the same offline reference library
- the same preview interface
- the same validation tools

What the first policy should own:

- tracking correction
- disturbance recovery
- contact adaptation
- delay / backlash / model-mismatch robustness

What the first policy should not own:

- basic step release
- core propulsion generation
- stop semantics
- command-conditioned gait structure

---

## Historical Companion

Historical `walking_ref_v1/v2`, `M2.5`, `v0.19.x`, and `v0.19.4` DCM COM
trajectory notes now live in
[reference_design_history.md](reference_design_history.md) so this
document stays focused on the active `v0.20.x` offline-prior path.

## Relationship To Main Walking Plan

This document is a focused design companion to:

- [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

`walking_training.md` remains the main milestone plan.
This file exists to keep the nominal reference architecture and `M2.5`
validation approach explicit and stable while the locomotion stack is still
being refactored.
