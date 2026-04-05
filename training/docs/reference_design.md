# WildRobot Nominal Reference Design

**Status:** Active design note for `M2.5` nominal reference validation  
**Last updated:** 2026-04-04

---

## Purpose

This document defines the intended architecture and validation plan for the
WildRobot locomotion nominal reference layer during `M2.5`.

`M2.5` exists because `v0.19.3` training integration is implemented, but PPO is
currently blocked by insufficient confidence in the nominal prior under full env
dynamics.

The goal of this document is to make the design explicit enough that:

- the reference architecture has one source of truth
- downstream layers remain adapters, not competing controllers
- validation is repeatable and gate-based
- future refactors reduce code debt rather than add more local patches

---

## Scope

This document is about:

- locomotion reference architecture
- `walking_ref_v1` limitations
- `walking_ref_v2` target design
- validation of nominal-only behavior
- `M2.5` execution milestones

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
walking reference
    ->
task-space nominal targets
    ->
IK / nominal target adapter
    ->
q_ref
    ->
policy(obs, ref, history) -> delta_q
    ->
q_target = q_ref + delta_q
```

The intended ownership is:

- `walking_ref_v2` owns locomotion semantics
- IK / nominal target shaping owns execution adaptation
- PPO owns residual robustness and recovery

Only one layer should decide:

- whether progression is allowed
- which hybrid gait mode is active
- how much forward swing release is permitted
- how foothold progression is shaped under instability

That layer is the reference layer.

---

## Architecture Design

### Layer responsibilities

**Layer 1: Reference generation**

Responsibilities:

- compute support health
- compute progression permission
- maintain hybrid gait mode/state
- generate foothold targets
- generate swing-foot targets
- generate pelvis targets

This is the controller-shaped nominal intent layer.

**Layer 2: Nominal target adaptation**

Responsibilities:

- convert task-space targets to `q_ref`
- perform IK
- apply joint-limit safety
- apply reachability fallback
- keep outputs executable on the robot morphology

This layer must not invent new gait semantics.

**Layer 3: Residual policy**

Responsibilities:

- compensate for model mismatch, delay, contact error, and disturbance
- track and adapt around `q_ref`

This layer is the deployed artifact.

**Layer 4: Env/runtime wiring**

Responsibilities:

- provide state inputs to the reference
- call the reference step function
- log diagnostics
- preserve training/runtime/replay contract parity

---

## Single Source Of Truth Rule

Best-practice architecture for this stack is:

- one subsystem owns locomotion semantics
- all downstream layers are adapters or safety layers

For WildRobot, this means:

- `walking_ref_v2` should be the single source of truth for:
  - support health
  - progression permission
  - hybrid gait mode
  - swing release logic
  - foothold progression logic

Downstream layers may still do:

- clipping
- IK reachability fallback
- smoothing
- actuator-safe target shaping

But downstream layers should not independently decide:

- when support is good enough
- when swing should release
- how fast progression should advance

If both the reference layer and the nominal-target layer answer those
questions, the system becomes split-brain and hard to debug.

---

## Reference Evolution

### `walking_ref_v1`

`walking_ref_v1` was the first usable reduced-order reference.

Main properties:

- `LIPM / DCM / capture-point` inspired
- bounded foothold targets
- bounded swing-foot trajectory
- simple pelvis targets
- primarily phase-driven progression

`walking_ref_v1` was appropriate for Milestone 2, but `M2.5` showed that it is
not yet dynamically valid under the full training env.

Observed limitations:

- too phase-driven under full dynamics
- forward progression remains active before support is stable
- local braking patches reduce aggression but do not solve the core issue

### `walking_ref_v2`

`walking_ref_v2` is the intended support-first hybrid reference.

Its role is not to become a heavy runtime controller.
Its role is to become a stronger nominal prior for residual RL.

In ToddlerBot-style terms, `walking_ref_v2` should act as a reference-motion
scaffold:

- the reference should already contain a coherent startup transition
- the reference should already contain a viable support posture
- the reference should already contain a conservative support-to-step release
- PPO should improve robustness around that scaffold, not invent it from
  scratch

Core properties:

- explicit hybrid gait modes
- support-conditioned progression
- conservative support-first stance behavior
- bounded foothold progression
- bounded swing release
- morphology-aware nominal posture

---

## `walking_ref_v2` Design

### Startup transition

The nominal reference must explicitly own the transition from reset posture into
loaded support posture.

This is part of reference design, not PPO design.

If the robot resets in an upright keyframe but the nominal support posture uses
a more compressed stance leg, the reference should not jump there in one step.
It should provide a bounded transition path.

Recommended startup sequence:

1. `startup_support_ramp`
2. `support_stabilize`
3. `swing_release`
4. `touchdown_capture`
5. `post_touchdown_settle`

The startup transition should:

- start close to the reset / keyframe-0 posture
- preserve planted-foot contact and nominal stance width
- gradually compress the stance leg
- gradually move pelvis height toward the loaded support target
- keep swing release fully suppressed
- complete only when the robot has settled into a viable support posture

The startup transition should not:

- instantly command a deep crouch-like support posture
- use PPO residuals as the primary mechanism for entering support
- open stepping progression before support posture is established

Recommended implementation shape:

- define a startup interpolation scalar `startup_alpha in [0, 1]`
- blend from reset/home posture toward support posture over a short bounded
  interval
- keep `swing_x_release = 0` and `lateral_release_y = 0` during this phase
- keep `swing_y_target = base_support_y`
- only allow transition into `support_stabilize` once startup posture error,
  pitch rate, and support health are within acceptable bounds

The purpose of this phase is not to "walk slowly."
Its purpose is to provide a dynamically plausible path from reset posture to the
first real support posture.

### Hybrid modes

The intended `walking_ref_v2` mode sequence is:

1. `startup_support_ramp`
2. `support_stabilize`
3. `swing_release`
4. `touchdown_capture`
5. `post_touchdown_settle`

Mode transitions should depend on state, not just phase.

### Support health

Support health is the main gating signal for progression.

It should be computed from bounded signals such as:

- root pitch
- root pitch rate
- overspeed relative to command
- stance/contact status
- optional stance support geometry margin

The reference should derive:

- `support_health`
- `support_instability`
- `progression_permission`

from those inputs.

### Progression permission

Progression must be support-conditioned.

When support is weak:

- phase progression should slow or freeze
- swing-x release should slow or freeze
- forward foothold progression should shrink

When support is healthy:

- controlled forward progression may resume

### Foothold progression

Foothold logic should be conservative.

Recommended structure:

- start from bounded nominal step length
- optionally add a small capture-point or Raibert-style correction
- suppress forward placement when support is failing

`DCM` or `capture-point` logic may be used, but should not be the sole driver.

### Swing release

Swing release should be staged.

Recommended behavior:

- hold safe swing-x while support is poor
- release forward swing-x only when progression permission is high enough
- keep vertical and lateral motion bounded

### Pelvis targets

Pelvis targets are secondary.

They may help posture, but should not be major forward-drive channels.
Pelvis behavior should remain small, bounded, and support-aware.

### Support posture construction

The support posture should be designed as a realizable loaded stance, not just
an IK output that looks reasonable in isolation.

That means:

- stance-leg compression should be introduced gradually
- pelvis height must be consistent with reachable stance-leg geometry
- stance hip pitch, knee pitch, and ankle pitch should stay in a range the
  position servos can actually enter and hold under load
- baseline lateral support width must remain nonzero in support mode

If a grounded nominal-qref initialization can instantiate the support posture
but the robot immediately collapses under contact dynamics, the remaining issue
is still in support-posture design or the transition into it.

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

---

## Validation Approach

Validation should follow a layered ladder.

### Level 1: Reference-generator correctness

Verify in isolation:

- phase stays bounded
- hybrid mode transitions are coherent
- startup transition stays bounded and completes coherently
- foothold targets stay bounded
- swing trajectories stay bounded
- pelvis targets stay bounded

### Level 2: IK and nominal target correctness

Verify:

- reachability stays high
- no systematic joint-limit hugging
- left/right symmetry remains sane
- `q_ref` varies smoothly enough to be trackable
- startup-to-support interpolation does not inject large instantaneous
  joint-target jumps

### Level 2.5: Grounded support-posture viability

Before treating a nominal-only walking failure as a stepping failure, verify the
support posture itself.

Recommended checks:

- grounded nominal-qref initialization starts with stance foot on the ground
- support-only mode can hold the nominal support posture materially longer than
  an immediate collapse
- stance-leg target-vs-actual mismatch remains bounded enough to explain the
  resulting motion

The nominal viewer is the canonical local tool for this check:

- `PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu uv run python training/eval/visualize_nominal_ref.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 32 --headless --print-every 1 --force-support-only --init-from-nominal-qref --log`

### Level 3: Nominal-only full-env viability

This is the main gate.

Run canonical nominal-only probes with:

- residual disabled
- fixed command
- fixed horizon
- reproducible config

Canonical sweep command:

- `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`

Expected artifacts:

- per-speed JSON summaries under `/tmp/m25_v2_probe_sweep/`
- per-speed traces under `/tmp/m25_v2_probe_sweep/`
- aggregate sweep summary at `/tmp/m25_v2_probe_sweep/summary.json`

The nominal reference is not considered validated until this level is credible.

### Level 4: Channel-isolation diagnostics

Use bounded ablations and traces to identify dominant failure sources.

This is a root-cause tool, not the final success criterion.

### Level 5: Small robustness sweep

After one-point success, run a small command sweep:

- `0.06 m/s`
- `0.10 m/s`
- `0.14 m/s`

Preferred command:

- `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`

If the prior only works at one exact operating point, it is not ready for PPO.

---

## Support-First And Dynamically Valid Checklist

### Support-first means

The nominal reference should:

- establish a viable stance/support posture first
- keep pitch growth bounded
- allow forward progression only when support is healthy enough
- suppress forward-drive channels when instability grows

### Dynamically valid means

The nominal reference should survive the full env dynamics well enough to serve
as a real policy prior.

That means nominal-only rollout should:

- survive materially beyond the current low-30-step regime
- avoid immediate `term/pitch`
- keep achieved speed in the same regime as command
- show swing tracking above numerical noise
- show foothold consistency above the current near-zero regime
- keep nominal-vs-applied target gap small

### Required trace-level checks

For a fixed nominal-only probe, inspect:

- hybrid mode over time
- phase progress
- stance foot
- root pitch
- root pitch rate
- forward velocity
- swing-x target vs actual
- foothold target vs realized proxy
- support health
- progression permission
- nominal-vs-applied `q` gap

---

## Current `M2.5` Readout

Current ablation evidence says:

- `no-dcm`: no meaningful effect
- `zero-pelvis-pitch`: no meaningful effect
- `freeze-swing-x`: reduces overspeed, but no survival gain
- `slow-phase`: reduces aggression slightly, but no survival gain
- `tight-stance`: improves survival, but worsens overspeed

Interpretation:

- the blocker is a coupled sagittal support + forward progression problem
- support geometry matters
- progression is still too aggressive relative to support recovery
- PPO should remain paused

---

## Execution Milestone For `M2.5`

### `M2.5-A0`: Architecture cleanup

Goals:

- make `walking_ref_v2` the single source of truth
- remove duplicated locomotion semantics from env code
- keep the adapter layer execution-focused

Expected tasks:

- unify support-health semantics
- delete dead hybrid-transition helpers
- remove or gate legacy `v1` debug metrics when `v2` is active
- narrow the env reference interface

### `M2.5-A1`: Reference validation

Goals:

- validate `walking_ref_v2` in isolation and nominal-only env mode

Expected tasks:

- unit tests for mode transitions
- unit tests for support-health bounds
- nominal-only fixed probe runs
- canonical multi-speed sweep:
  - `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`
- trace inspection against the support-first checklist

### `M2.5-A2`: Support-first tuning

Goals:

- make `walking_ref_v2` conservative and dynamically plausible

Expected tasks:

- tune support-conditioned progression
- tune support-first stance behavior
- tune bounded foothold and swing release
- re-run nominal-only probes at `0.06 / 0.10 / 0.14`

### `M2.5-A3`: Exit decision

Resume PPO only if all are true:

- nominal-only survives materially longer
- immediate pitch-fall is gone
- speed is in the same regime as command
- swing tracking is above noise
- foothold consistency is clearly nontrivial
- no major contract drift between training, runtime, and replay

If these gates are not met, do not resume PPO.
At that point, treat the nominal-layer design as requiring a larger redesign,
not more local patches.

---

## Immediate Next Actions

1. Finish architecture cleanup so `walking_ref_v2` fully owns progression semantics.
2. Remove duplicated support/gating logic from env-side nominal shaping.
3. Re-run nominal-only probes under the cleaned architecture.
   Preferred command:
   `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`
4. Use the support-first checklist as the only decision gate for resuming PPO.

---

## Relationship To Main Walking Plan

This document is a focused design companion to:

- [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

`walking_training.md` remains the main milestone plan.
This file exists to keep the nominal reference architecture and `M2.5`
validation approach explicit and stable while the locomotion stack is still
being refactored.
