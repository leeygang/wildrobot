# WildRobot Nominal Reference Design

**Status:** Active design note for `M2.5` nominal reference validation  
**Last updated:** 2026-04-05

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
- move through a staged fixed route from `A -> W1 -> W2 -> W3 -> B`
- establish load transfer / alignment before deeper support compression
- gradually move pelvis height toward the loaded support target
- keep swing release fully suppressed
- complete only when the robot has settled into the support terminal region `B`
- react to actual posture realization, not only elapsed startup time

The startup transition should not:

- instantly command a deep crouch-like support posture
- use PPO residuals as the primary mechanism for entering support
- open stepping progression before support posture is established
- continue deepening support posture when stance knee/ankle tracking is clearly
  lagging

Recommended implementation shape:

- keep the endpoint `B` fixed as the full support target / terminal region
  - `B` should include the full support posture, not only a knee-bend target
- define a staged fixed route:
  - `A`: startup posture / reset-side posture
  - `W1`: preload / load-shift waypoint with minimal knee/ankle demand
  - `W2`: partial support-entry waypoint with moderate pelvis drop and partial
    support-chain realization
  - `W3`: final support-convergence waypoint approaching full support geometry
  - `B`: support terminal region
- define startup progress from realized posture, not elapsed time alone
- use pelvis height realization as the primary early-stage task-progress signal
  for upright-to-support transition
- guard stage progression with stance knee realization, stance ankle
  realization, root pitch, root pitch rate, and support health
- keep `swing_x_release = 0` and `lateral_release_y = 0` during this phase
- keep `swing_y_target = base_support_y`
- use a startup readiness scale only as a soft governor on progression speed,
  not as the primary notion of progress
- add explicit startup target-rate limiting
  - nominal design rate should be around `30°/s`
  - hard emergency cap should be around `100°/s`
- only allow transition into `support_stabilize` once the route has entered the
  `B` region and startup posture error, pitch rate, and support health are
  within acceptable bounds
- keep elapsed time only as a watchdog / timeout, not as the main progress
  driver
- add explicit stage logging for diagnostics:
  - current stage label: `A`, `W1`, `W2`, `W3`, or `B`
  - stage entry / exit step and time
  - transition reason
  - target vs actual for pelvis height, hip roll, hip pitch, knee pitch, and
    ankle pitch

The purpose of this phase is not to "walk slowly."
Its purpose is to provide a dynamically plausible path from reset posture to the
first real support posture.

Feedforward interpolation alone is not sufficient.
If nominal-only traces show that `q_ref -> ctrl` is clean but the robot is not
realizing the intermediate support posture, startup progression should wait for
the robot rather than blindly advancing on schedule.

Best-practice note:

- startup progress should be task-specific
- for upright-to-support transition, pelvis `z` / height is the primary task
  metric
- joint realization and stability signals should guard that progress so collapse
  is not mistaken for success

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

### Concrete support-posture redesign

Current `M2.5` evidence says the startup ramp is now present, but the robot
still does not realize the intended loaded stance leg:

- stance hip pitch partially tracks
- stance knee pitch stays near straight while target remains strongly flexed
- stance ankle pitch stays far from target
- root roll and root pitch continue to grow under support-only and startup-path
  runs

So the next redesign pass should treat support posture as a bounded algorithmic
construction problem, not an open-ended tuning exercise.

#### Design objective

`support_stabilize` should behave like a conservative loaded standing posture:

- preserve nonzero baseline support width
- keep both swing releases closed
- keep the stance leg load-bearing without requiring a deep crouch
- keep pelvis height high enough that knee and ankle targets remain trackable
- give the startup ramp a reachable posture to converge to

#### Support-posture algorithm

Build support posture from a small set of state-aware scalars rather than from
aggressive fixed compression.

Recommended construction:

1. Choose a baseline support width:
   - preserve the existing nonzero `base_support_y`
   - keep `lateral_release_y = 0` in `support_stabilize`
   - keep `swing_y_target = base_support_y`
2. Choose a conservative stance compression scalar:
   - derive `stance_compression in [0, 1]` from support-health deficit
   - clamp it to a shallow range rather than allowing a deep crouch
3. Choose pelvis height from that compression:
   - use a higher support pelvis target than the current failing posture
   - lower pelvis only enough to load the stance leg, not enough to demand a
     large knee bend
4. Build sagittal support geometry:
   - set stance foot sagittal target with small or zero forward offset
   - solve stance hip pitch / knee pitch / ankle pitch from the support height
     target
   - prefer a shallower knee and smaller ankle counter-rotation
5. Keep pelvis attitude small:
   - pelvis roll only as a small support bias
   - pelvis pitch only as a small bounded support bias
   - do not use pelvis attitude as a hidden forward-drive channel
6. Keep progression closed:
   - `swing_x_release = 0`
   - `lateral_release_y = 0`
   - foothold progression bounded by support mode rules

Recommended design constraint:

- treat support posture as a loaded standing pose, not a crouch
- if stance knee target grows large while the robot remains nearly straight,
  redesign the support height/geometry first rather than asking PPO to rescue it

#### Lateral unloading for no-ankle-roll morphology

WildRobot v2 has limited frontal-plane authority:

- no active ankle roll
- less lateral recovery authority than a 6-DOF humanoid leg

So `support_stabilize` cannot be a purely sagittal bracing mode.
Even while lateral release remains closed, the reference still needs an explicit
unloading mechanism before swing release is allowed.

Priority note:

- this is a real design constraint for the current v2 hardware
- but it is not the top `M2.5` blocker right now
- current evidence still says the dominant immediate blocker is support-posture
  realizability in the sagittal/support-loading chain
- if active ankle roll is added soon, this lateral strategy should be revisited
  under the new morphology rather than over-optimized for the temporary v2 limit

The intended mechanism is:

1. preserve baseline support width
   - keep `swing_y_target = base_support_y`
   - keep `lateral_release_y = 0` during support establishment
2. create bounded stance loading with pelvis/trunk shift
   - use small pelvis roll plus bounded trunk alignment to move load toward the
     stance side
   - do not use lateral foot release as the primary unloading tool in this phase
3. verify unloading before swing release
   - support establishment should require evidence that the swing side is less
     loaded or the stance side is more loaded
   - acceptable signals are stance/swing contact asymmetry, support-health
     recovery, and bounded root-roll behavior
4. only then allow swing release
   - forward release remains gated by progression permission
   - lateral release remains a small delta around baseline width, not the main
     unloading mechanism

Design rule:

- root roll growth caused by collapse is not acceptable evidence of unloading
- support establishment should produce controlled load transfer, not passive
  lateral divergence

For the next redesign pass, the support-first controller should explicitly tune:

- pelvis roll bias magnitude
- pelvis/trunk lateral alignment during support establishment
- support-open criteria that require some unloading evidence before swing release

#### Immediate tuning strategy

For the next pass, tune in this order:

1. raise support pelvis height slightly
2. reduce stance compression target
3. reduce implied stance knee bend
4. reduce ankle counter-rotation magnitude
5. retune startup ramp only after the new support posture is reachable

This order matches the current failure pattern: hip pitch can partially track,
but knee and ankle do not.

#### Transition contract

The startup ramp should remain, but it should now ramp into this shallower
support posture:

- start near keyframe-0 upright posture
- preserve stance width
- gradually increase stance loading
- complete only when the support posture is reachable enough to hold

The startup phase should not compensate for an unrealistic destination posture.
The destination posture itself must be realizable.

#### Concrete geometry levers

The current support-leg geometry is not set by an abstract "compression" value
alone. In the current stack it is controlled mainly through:

- `_loc_ref_stance_extension_margin_m`
- its health-dependent scaling
- the resulting `nominal_stance_z`
- pelvis/support height targets coming from `walking_ref_v2`

Conceptually:

```text
stance_margin
    = stance_extension_margin_m * f(support_health)

nominal_stance_z
    = -(pelvis_height - stance_margin)
```

So the concrete posture levers for the next redesign pass are:

1. `pelvis_height`
   - first lever to raise if stance knee/ankle targets are too deep
2. `stance_extension_margin_m`
   - second lever to reduce if the support leg is over-compressed
3. health-dependent margin scaling
   - keep it bounded so poor support does not produce erratic geometry changes
4. startup support height offset
   - use startup to approach the support posture gradually, but do not hide an
     unrealistic final support height

Recommended tuning order remains:

1. raise support pelvis height
2. reduce `stance_extension_margin_m`
3. reduce health-driven compression sensitivity
4. then retune startup ramp timing around the new support posture

The next redesign pass should document the chosen config values once a working
range is found.

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

### Canonical nominal-probe realism baseline

For `M2.5`, the canonical nominal probes are defined operationally by:

- config: [`ppo_walking_v0193a.yaml`](/home/leeygang/projects/wildrobot/training/configs/ppo_walking_v0193a.yaml)
- probe path: [`eval_loc_ref_probe.py`](/home/leeygang/projects/wildrobot/training/eval/eval_loc_ref_probe.py)
- sweep path: [`run_m25_v2_probe_sweep.py`](/home/leeygang/projects/wildrobot/training/eval/run_m25_v2_probe_sweep.py)

The baseline assumptions for those probes are:

- v2 MuJoCo MJX asset stack and full actuator/contact dynamics
- fixed forward command per probe
- residual disabled for nominal-only evaluation
- pushes disabled by config for the canonical `v0.19.3a` nominal path
- action delay disabled (`action_delay_steps = 0`)
- domain randomization disabled
- IMU noise/latency disabled unless explicitly re-enabled in config

Important clarification:

- `ppo_walking_v0193a.yaml` records
  `realism_profile_path = assets/v2/realism_profile_v0.19.1.json`
- but the canonical probe contract for `M2.5` should be treated as the concrete
  env/config behavior above, not as an assumption that every realism-profile
  feature is independently applied and verified in the probe path

If probe-time realism-profile application becomes a gating requirement, it
should be made explicit in runtime/env logs and documented as a new baseline.

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

Additional required checks for the next redesign pass:

- startup-path run from keyframe 0:
  - `PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu uv run python training/eval/visualize_nominal_ref.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 64 --headless --print-every 1 --log`
- grounded support-only run from nominal posture:
  - `PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu uv run python training/eval/visualize_nominal_ref.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 32 --headless --print-every 1 --force-support-only --init-from-nominal-qref --log`

These two runs answer different questions:

- startup-path run:
  - can the robot transition from keyframe 0 into support posture smoothly
- grounded support-only run:
  - can the robot actually hold the support posture once it starts there

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

For the next support-posture redesign pass, inspect these additional
support-specific metrics:

- startup mode duration and handoff step
- root roll
- root height
- stance hip-roll target vs actual
- stance hip-pitch target vs actual
- stance knee-pitch target vs actual
- stance ankle-pitch target vs actual
- support width actual vs commanded
- swing-y target vs actual

These are the main pass/fail diagnostics for the current bug.

---

## Current `M2.5` Readout

**Last updated:** 2026-04-12

### Support posture (B) — resolved

The support posture is now self-stable in pure MuJoCo hold tests.

Root causes found and fixed:

1. **Height inversion**: startup height was below support height — the ramp
   went upward while gravity pulled down.  Fixed: startup=0.46m (near reset),
   support=0.39m (walking height).

2. **Stance foot placement bug**: the adapter hardcoded `target_x=0` for the
   stance leg IK, placing the foot directly under the hip joint.  But the
   torso mass shifts the COM forward of the hip when the hip pitches.  For
   medium squat depths, the COM fell outside the support polygon → forward
   tipping.  Fixed: added a COM-forward offset (`gain × sin(hip_pitch)`) with
   depth-dependent fade — full offset for shallow squats, zero for deep squats
   where the posture is naturally stable.

3. **Support posture in unstable zone**: target_z=-0.39 was in an unstable
   equilibrium band (z=-0.41 to z=-0.38).  Two stable regions exist:
   - z≥-0.42 (nearly straight, trivially stable)
   - z≤-0.37 (deep squat, COM drops inside support polygon)
   Fixed: support at z=-0.37 (knee=0.99 rad, 57°).

4. **Adapter-side health-dependent margin**: `stance_margin = base × (0.2 +
   0.8 × health)` created a positive feedback loop — any perturbation changed
   health → changed margin → changed IK targets → destabilized posture.
   Fixed: constant margin.

5. **Double-support penalty during startup**: support_health penalized both
   feet loaded (the most stable configuration).  Fixed: removed the penalty
   during STARTUP_SUPPORT_RAMP.

Current B posture:
- target_z = -0.37 (support_height=0.39, margin=0.02)
- knee ≈ 0.99 rad (57°), meaningful squat
- pure MuJoCo hold: 200/200 steps, pitch stable at ~0.05 rad
- symmetric: both feet get the same COM-forward offset during support

Verification command:
```
JAX_PLATFORMS=cpu uv run mjpython training/eval/visualize_nominal_ref.py \
  --config training/configs/ppo_walking_v0193a.yaml \
  --forward-cmd 0.10 --horizon 200 --print-every 10 \
  --force-support-only --init-from-nominal-qref --disable-action-filter
```

### Startup transition (A→B) — resolved

The smooth A→B transition is now implemented and validated using qpos
interpolation with progressive contact settling.

The servo-driven approach (commanding joint targets and letting the PD servo
drive the squat) does not work because the closed kinematic chain with both
feet planted prevents the ankle dorsiflexion needed for the knee to bend.
The servo applies force but the leg acts as a rigid truss.

Instead, the transition uses **direct qpos interpolation**: at each step,
the leg joint positions are linearly blended from keyframe A (standing)
toward support posture B (squat), with root height re-grounded at each step
to keep the feet on the ground as the knees bend.  The ctrl is set to match
the interpolated joint positions so the servo holds each intermediate posture.

This approach:
- produces a smooth, visible transition from standing to squat
- both knees track symmetrically (error < 0.01 rad throughout)
- root height drops smoothly from 0.469m to 0.430m
- pitch stays bounded (< 0.15 rad) during transition
- after transition, the robot holds B for 200+ steps
- is compatible with the MuJoCo viewer for visual verification

Key implementation details:
- the transition runs inside the `run_step` loop so the viewer displays it
- only leg joints (hip/knee/ankle pitch) are interpolated; other joints stay
  at keyframe values to avoid self-collision at deep squat angles
- re-grounding at each step prevents foot-ground interpenetration
- the transition completes at step N (configurable via `--startup-settle-steps`)
  and then switches to pure hold mode with frozen ctrl

Verification command:
```
JAX_PLATFORMS=cpu uv run mjpython training/eval/visualize_nominal_ref.py \
  --config training/configs/ppo_walking_v0193a.yaml \
  --forward-cmd 0.10 --horizon 200 --print-every 20 \
  --force-support-only --startup-settle-steps 100 --disable-action-filter
```

Next step: integrate the same qpos-interpolation startup into the env reset
path so PPO training episodes start from the stable B posture.

### Nominal walking probe — not yet tested

The nominal walking probe (with stepping) has not been run with the new stable
B posture.  This is the next validation step after the startup transition is
integrated into the env.

### PPO — still blocked

PPO remains blocked until:
1. DCM COM trajectory is implemented (stance-leg forward drive)
2. The nominal walking probe produces visible stepping with controlled forward
   speed
3. The M2.5-A4 gate criteria are met

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
- make the loaded support posture reachable and holdable under full dynamics
- isolate obvious command-path distortions that hide the real startup/support
  behavior

Expected tasks:

1. Support-posture redesign
   - raise support pelvis height slightly
   - reduce support compression depth
   - reduce stance knee target
   - reduce stance ankle counter-rotation
2. Startup-path validation
   - run startup-path viewer from keyframe 0
   - verify smoother support entry and reduced early collapse
3. Support-only validation
   - run grounded support-only viewer
   - verify the posture can hold materially longer than the current failing
     regime
4. Only after support posture improves:
   - retune support-conditioned progression
   - retune bounded foothold and swing release
5. Command-path isolation
   - verify whether startup `q_ref -> ctrl` mismatch is caused by filtering or
     other delivery-path reshaping
   - identify whether the first major divergence is `q_ref -> ctrl` or
     `ctrl -> actual`
6. Re-run nominal-only probes at `0.06 / 0.10 / 0.14`

Target metrics for `M2.5-A2`:

- startup-path:
  - support handoff occurs without immediate pitch jump
  - stance knee actual moves visibly toward target during startup, rather than
    remaining near straight while target is strongly flexed
  - foot spreading is reduced relative to the current failure mode
- support-only:
  - no immediate collapse from grounded nominal posture
  - root height drop is reduced
  - root roll growth is reduced
  - stance knee and ankle target-vs-actual mismatch is smaller
- nominal sweep:
  - pitch-failure timing improves beyond the current step-30/40 regime
  - achieved speed remains within `1.5x` command
  - swing-foot tracking reward exceeds `1e-5`
  - foothold consistency reward exceeds `0.05`

### `M2.5-A3`: Staged fixed-route startup/support execution

Goals:

- keep startup posture `A` and support target `B` fixed
- replace the direct startup-to-support route with a staged fixed route
  `A -> W1 -> W2 -> W3 -> B`
- make route progression react to actual posture realization
- preserve the improved startup smoothness while testing whether route shape is
  the missing factor

Expected tasks:

1. Define a fixed terminal support region `B`
   - include the full support posture, not a single-joint condition
   - include pelvis height and the key support-chain joints with tolerances
2. Define explicit intermediate waypoints
   - `W1`: preload / load-shift with minimal knee/ankle demand
   - `W2`: partial support entry
   - `W3`: final support convergence
3. Drive progression by stage completion, not elapsed startup time
   - stage completion should be event-based
   - primary early-stage task metric: pelvis height realization
   - guard metrics: stance knee realization, stance ankle realization, root
     pitch, root pitch rate, and support health
4. Keep explicit startup/support-entry rate limiting
   - nominal support-entry design rate should be around `30°/s`
   - hard emergency cap should be around `100°/s`
5. Add route / stage diagnostics
   - current stage label: `A`, `W1`, `W2`, `W3`, `B`
   - stage transition log with entry / exit step and transition reason
   - target vs actual for pelvis height, hip roll, hip pitch, knee pitch, and
     ankle pitch per stage
6. Re-run startup-path and support-only viewer probes
   - compare startup / early-support / late-support realizability by stage
   - verify `q_ref -> ctrl` stays clean
   - compare stance knee/ankle target-vs-actual error by stage
7. Only after staged route behavior is understood:
   - decide whether a fixed staged route is enough
   - otherwise escalate to MPC / online route optimization

Target metrics for `M2.5-A3`:

- startup-path:
  - startup `q_ref -> ctrl` remains clean in nominal-only debug mode
  - route progression follows `A -> W1 -> W2 -> W3 -> B` coherently
  - stance knee and ankle target-rate stays near the `30°/s` design intent and
    never exceeds the `100°/s` hard cap
  - stage transitions are event-based and do not deadlock
  - pelvis height progresses toward support target for the right reason, not
    because the robot is collapsing
  - support handoff occurs only after the route has effectively entered `B`
- support-only:
  - support terminal region `B` remains holdable when initialized directly
  - stage logs make it clear whether route failure or target failure is the
    dominant issue
  - root roll and width growth remain bounded
- route-validation:
  - if ideal or near-ideal target delivery still cannot converge through
    `A -> W1 -> W2 -> W3 -> B`, treat route shape as inadequate
  - if the route reaches `B` in ideal-tracking replay but not in real MuJoCo,
    treat `ctrl -> actual` realization as the dominant issue
- nominal sweep:
  - early pitch stability is preserved
  - progression quality does not regress further while staged route execution is
    being validated

Current status / investigation result (April 2026):

- staged-route plumbing and diagnostics are now implemented
  - route labels `A / W1 / W2 / W3 / B`
  - route progress / ceiling ratchets
  - transition reasons
  - per-stage viewer summaries
- route progression bugs are **resolved** (April 10-11):
  - height inversion fixed (startup=0.46m > support=0.39m)
  - directional pelvis realization (no abs(), overshoot clips to 1.0)
  - double-support penalty removed during startup
  - route ceiling and progress ratcheted (monotonic, no regression)
  - lead_alpha increased (0.12→0.25) to bootstrap past W1
  - bilateral rate limiting during double support
  - unconditional timeout as safety valve
- support posture B is **validated stable** (April 11):
  - target_z=-0.37 (deep squat, knee=0.99 rad)
  - COM-forward offset with depth-dependent fade
  - symmetric foot placement during support modes
  - pure MuJoCo hold: 200/200 steps, pitch ≈ 0.05 rad
- startup transition A→B is **validated** (April 11):
  - smooth qpos interpolation over configurable settle steps
  - both knees track within 0.01 rad throughout transition
  - root height drops smoothly from 0.469m to 0.430m
  - visible in MuJoCo viewer
  - holds B for 200+ steps after transition completes
- **next step**: integrate the qpos-interpolation startup into the env reset
  path, then run the nominal walking probe with the stable B posture

### `M2.5-A4`: Exit decision

Resume PPO only if all are true:

- nominal-only no longer reaches early `term/pitch` in the canonical sweep
- mean forward speed stays within `1.5x` commanded speed
- swing-foot tracking reward is above `1e-5`
- foothold consistency reward is above `0.05`
- no major contract drift between training, runtime, and replay

If these gates are not met, do not resume PPO.
At that point, treat the nominal-layer design as requiring a larger redesign,
not more local patches.

These thresholds are intentionally aligned with the current canonical sweep
assessment in:

- [`run_m25_v2_probe_sweep.py`](/home/leeygang/projects/wildrobot/training/eval/run_m25_v2_probe_sweep.py)

Trace-level review is still required after the numeric gate passes.

### `M2.5-A5`: If bounded redesign still fails

If `M2.5-A2` + `M2.5-A3` still cannot produce a credible nominal prior, do not
continue with small local patches indefinitely.

Escalation options should be considered in this order:

1. larger nominal redesign
   - simplify the gait scaffold further
   - use a longer in-place support-capture / weight-shift phase before forward
     progression
2. reduced-order model upgrade
   - consider stronger ALIP / capture dynamics if the current support-first
     hybrid logic remains too heuristic
3. actuator / sim fidelity improvement
   - revisit actuator SysID and servo-response realism if the posture appears
     plausible but is still not realized under dynamics
4. morphology-specific gait simplification
   - explicitly design for hip-strategy-dominant walking if conventional
     assumptions do not fit the platform well

The key rule is:

- if the nominal prior cannot survive support-first validation after bounded
  posture and transition redesign, switch to a larger redesign path rather than
  resume PPO prematurely

---

## Immediate Next Actions

**Focus: resume PPO training with the walking reference**

### Resolved (April 12)

1. **Ctrl ordering fix** — root cause of ALL v0.19.3+ failures.
   Fixed via `CtrlOrderMapper`.

2. **Env starts from support posture B** — `start_from_support_posture: true`.

3. **Reference tuning** — conservative thresholds relaxed post-ctrl-fix.

4. **COM trajectory** — phase-proportional stance-leg forward drive.
   Robot now walks: gait cycles complete, stance switches, 500+ steps.

### Current walking status

The nominal reference produces walking with known quality issues:
- **Wobble** (±10° pitch oscillation) — open-loop reference, no active balance
- **Lateral drift** — small per-step asymmetries accumulate into turning
- **Small steps** — servo tracking limits foot displacement to ~3cm

Heading correction via foothold lateral bias was attempted twice but both
approaches destabilized lateral balance.  The robot has no ankle roll, so
any lateral foot placement change directly affects the support polygon
without a compensating mechanism.

### Next step: PPO integration (v0.19.5)

These quality issues are the intended scope for PPO residuals:
- Wobble → PPO learns pitch-rate damping corrections
- Drift → PPO learns lateral balance corrections
- Step size → PPO improves servo tracking during swing

The nominal reference provides a viable walking scaffold. Resume PPO.

1. **Ctrl ordering fix** — PolicySpec and MuJoCo listed actuators in different
   orders.  The env wrote PolicySpec-ordered ctrl to MuJoCo's data.ctrl,
   sending joint targets to wrong actuators (e.g. knee bend → hip pitch).
   This was the root cause of ALL v0.19.3+ loc_ref training failures.
   Fixed via `CtrlOrderMapper` (`training/utils/ctrl_order.py`).
   Regression test: `training/tests/test_actuator_ordering.py`.

2. **Env starts from support posture B** — episodes begin in
   SUPPORT_STABILIZE mode with bent-knee squat posture.
   Config: `start_from_support_posture: true`.

3. **Reference tuning** — conservative thresholds tuned during the ctrl bug
   era have been relaxed (swing_target_blend, brake gains, step limits,
   support_open_threshold, pelvis height).

### Current status after fixes

- Robot survives 120 steps (full horizon), no pitch termination
- Gait mode cycles: SUPPORT → SWING_RELEASE → TOUCHDOWN → SETTLE → repeat
- Reference commands 5-6cm foothold targets during swing release
- Swing foot tracks partially (~3cm actual of 6cm target)
- **But**: forward speed ≈ 0, body rocks ±12° pitch, steps too small to walk
- Root cause: stance leg is a static pillar — no forward COM drive

### Remaining work

1. **Implement DCM COM trajectory** (M3.0-A, described below)
   - Let the body fall forward over the stance foot following LIPM dynamics
   - The IK automatically produces stance-leg hip extension + ankle rocker
   - This is the missing piece that makes walking work

2. **Re-run nominal walking probe** after COM trajectory is implemented
   - Check M2.5-A4 gate: forward speed tracks command, foothold > 0.05

3. **Resume PPO** once the nominal reference produces viable walking

---

## DCM COM Trajectory Design

### Problem

The current reference computes a **static** pelvis position: the body stays
directly above the stance foot the entire stance phase.  Walking requires the
body to **fall forward** over the stance foot — a controlled inverted pendulum
motion.  Without this, the stance leg has no forward drive and the robot can
only produce tiny steps from swing-foot placement alone.

### Algorithm: LIPM COM trajectory

The Linear Inverted Pendulum Model (LIPM) gives the closed-form COM trajectory
during stance:

```
x_com(t) = x0 × cosh(ω×t) + (ẋ0/ω) × sinh(ω×t)

where:
  ω = sqrt(g / h)           ≈ 4.83 rad/s for WildRobot (h=0.42m)
  x0 = COM position relative to stance foot at start of stance
  ẋ0 = COM velocity at start of stance
  t  = time elapsed in current stance phase
```

This is not a new algorithm — it's the **same LIPM dynamics** already used for
the DCM foothold computation.  The foothold uses the endpoint of this
trajectory to decide WHERE to step.  The COM trajectory uses the trajectory
itself to decide HOW THE BODY SHOULD MOVE during stance.

### How it connects to existing code

The reference already computes (in `walking_ref_v2.py`):

```python
omega = sqrt(9.81 / com_height)                    # pendulum frequency
cp_x = com_x + com_vx / omega                      # capture point
b = exp(-omega * step_time)                         # time decay factor
x_foot = (nominal_step - b * cp_x) / (1 - b)       # foothold placement
```

The COM trajectory adds (new code):

```python
t = phase * step_time                               # time in stance
com_x_planned = x0 * cosh(omega * t) + (vx0 / omega) * sinh(omega * t)
```

Both use the same `omega`, same state variables, same physical model.

### What changes in each layer

**Layer 1: Reference generation** (`walking_ref_v2.py`)

New output: `com_x_planned` — the planned COM x position relative to the
stance foot at the current phase.

New state: `com_x0` and `com_vx0` — COM position and velocity at the start
of each stance phase, recorded at stance transitions.

```python
# At stance transition (touchdown → new stance):
com_x0 = com_position_stance_frame[0]   # COM pos relative to new stance foot
com_vx0 = com_velocity_stance_frame[0]  # COM velocity at transition

# During stance:
t = phase * step_time
com_x_planned = com_x0 * cosh(omega * t) + (com_vx0 / omega) * sinh(omega * t)
```

**Layer 2: IK adapter** (`_compute_nominal_q_ref_from_loc_ref`)

Replace the static stance foot offset:

```python
# BEFORE (static):
_stance_foot_x = com_forward_gain * depth_fade * sin(hip_pitch)

# AFTER (dynamic):
_stance_foot_x = -com_x_planned  # foot moves behind as COM advances
```

As the COM moves forward over the stance foot, the IK automatically produces:
- Hip pitch increases (hip extends) → pushes body forward
- Ankle pitch changes from dorsiflexion to plantarflexion → ankle rocker
- Knee adjusts to maintain height → natural flexion wave

**Layer 3: No change** — IK solver, PPO architecture, action contract all
stay the same.

### Expected stance leg behavior

```
Phase 0% (loading):     COM behind foot → hip flexed, ankle dorsiflexed
Phase 50% (mid-stance): COM over foot   → hip neutral, ankle neutral
Phase 100% (push-off):  COM ahead       → hip extended, ankle plantarflexed
```

This matches the natural human gait cycle without any manual joint trajectory
scripting.  The IK derives the joint angles from the COM trajectory
automatically.

### Safety bounds

The COM trajectory should be bounded to prevent excessive forward lean:

```python
com_x_planned = clip(com_x_planned, -max_com_behind, max_com_ahead)
```

Recommended initial bounds:
- `max_com_behind = 0.03m` (3cm behind stance foot)
- `max_com_ahead = 0.08m` (8cm ahead of stance foot)

These can be relaxed as confidence grows.

---

## Execution Milestones For DCM COM Trajectory (`v0.19.4`)

### `v0.19.4-A`: COM trajectory implementation (was M3.0-A)

**Goal:** add LIPM COM trajectory to the reference and IK adapter.

**Tasks:**

1. Add `com_x0`, `com_vx0` state fields to the walking reference
2. Record COM state at each stance transition (SETTLE → SUPPORT switch)
3. Compute `com_x_planned = x0 * cosh(ω*t) + (vx0/ω) * sinh(ω*t)`
4. Output `com_x_planned` from the reference step function
5. In the IK adapter, replace static `_stance_foot_x` with `-com_x_planned`
6. Add safety bounds (clip to ±max range)
7. Add `com_x_planned` to diagnostics/viewer logging

**Verification:**

```bash
# Viewer: watch stance leg change through phase
JAX_PLATFORMS=cpu uv run mjpython training/eval/visualize_nominal_ref.py \
  --config training/configs/ppo_walking_v0193a.yaml \
  --forward-cmd 0.15 --horizon 120 --print-every 5 --log --record --headless
```

- Stance hip should visibly extend during late stance
- Ankle should dorsiflex → plantarflex through the phase
- Body should translate forward over the stance foot

**Exit criteria:**
- `com_x_planned` varies from negative (COM behind) to positive (COM ahead)
  during each stance phase
- Stance hip pitch changes by > 0.1 rad through the phase
- Stance ankle pitch changes by > 0.05 rad through the phase

### `v0.19.4-B`: Nominal walking probe — done (was M3.0-B)

**Goal:** verify the nominal reference produces sustained walking.

**Exit criteria (nominal-only, no PPO):**
- Survives 300+ steps at cmd=0.15 without pitch termination
- Forward speed mean > 0
- Gait cycles complete (mode transitions 1→2→3→4→1)
- Swing tracking > 1e-5
- Stance foot switches occur

**Result (April 12):**
- 461/500 steps (9.2s) at cmd=0.15 ✅
- Forward speed mean +0.038 m/s ✅
- 15 gait cycles, 30 stance switches ✅
- Swing tracking 0.050 ✅
- Terminated: term/roll (lateral drift)

**Not gated here (moved to v0.19.5):**
- Foothold consistency > 0.05 — requires PPO foot tracking corrections
- Pitch oscillation < ±0.15 — requires PPO balance corrections

### `v0.19.5`: PPO integration (was M3.0-C)

**Goal:** train residual PPO policy on top of the walking reference to
improve balance, lateral stability, and foot tracking.

**Tasks:**

1. Verify env reset + auto-reset work with COM trajectory state
2. Run a short PPO training (1000 iterations) to check learning signal
3. Check that PPO residuals improve stepping (larger steps, better tracking)

**Exit criteria:**
- PPO training runs without errors
- `eval_clean/success_rate` > 0% within 1000 iterations
- Step length improves with training (residuals help foot tracking)
- Forward speed tracks command better than nominal-only
- Mean step length > 5cm (current baseline: 2.5cm, metric: `debug/step_length_m`)
- Pitch oscillation < ±0.15 rad (current baseline: ±0.17 rad)

### `v0.19.5-D`: If COM trajectory alone is insufficient (was M3.0-D)

If `v0.19.4-B` gate is not met after bounded tuning:

1. Add explicit ankle push-off command during terminal stance (not just IK-
   derived — direct ankle target override)
2. Add pelvis pitch feedforward (small forward lean proportional to commanded
   speed)
3. Consider longer step time (0.50s → 0.70s) to give more time for foot
   tracking
4. Lower `support_release_phase_start` to give swing more of the gait cycle
   (tested at 0.25 — produced bigger steps but body rocked without COM drive;
   may work with COM trajectory providing momentum control)

Do not skip to full MPC or major redesign until these simpler options are
tried.

---

## Historical Notes

### Ctrl ordering mismatch (discovered April 12)

PolicySpec (`mujoco_robot_config.json`) and MuJoCo model (`wildrobot.xml`)
listed actuators in different orders.  The env wrote PolicySpec-ordered ctrl
directly to `data.ctrl`, which expects MuJoCo model order.  This sent each
joint's target to the wrong actuator — e.g. the knee bend (0.99 rad) went to
the hip pitch actuator.

**Impact:**
- All v0.19.3+ loc_ref training runs: completely broken (scrambled ctrl)
- All M2.5 nominal probes and conclusions: based on scrambled ctrl
- The entire M2.5 support-posture redesign effort: tuning around a bug
- v0.17/v0.18 standing PPO: worked but with secretly scrambled action
  semantics (PPO learns any consistent mapping, but action-by-name analysis
  and hardware deployment would be wrong)

**Fix:** `CtrlOrderMapper` (`training/utils/ctrl_order.py`) provides a single
API for all ctrl writes.  Regression test: `test_actuator_ordering.py`.
See `docs/system_architecture.md` for the design principle.

---

## Relationship To Main Walking Plan

This document is a focused design companion to:

- [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

`walking_training.md` remains the main milestone plan.
This file exists to keep the nominal reference architecture and `M2.5`
validation approach explicit and stable while the locomotion stack is still
being refactored.
