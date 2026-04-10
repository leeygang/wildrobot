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

Current ablation evidence says:

- `no-dcm`: no meaningful effect
- `zero-pelvis-pitch`: no meaningful effect
- `freeze-swing-x`: reduces overspeed, but no survival gain
- `slow-phase`: reduces aggression slightly, but no survival gain
- `tight-stance`: improves survival, but worsens overspeed

Latest nominal-viewer evidence also says:

- disabling the action filter removes startup `q_ref -> ctrl` distortion
- startup knee/ankle target steps become very abrupt once that filter is removed
- in `support_stabilize`, `q_ref -> ctrl` is mostly clean
- in `support_stabilize`, stance knee/ankle still show large `ctrl -> actual`
  mismatch while stance hip tracks much better

Interpretation:

- the blocker is still a coupled support-realization + forward-progression
  problem
- support geometry matters, but startup/support execution is still too
  feedforward
- startup target delivery distortion was partly caused by the action filter
- once that filter is removed, startup target changes are still too abrupt for
  the current realization path
- the next bounded pass should make startup/support progression feedback-aware
  before more foothold/progression retuning
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

1. Keep the current support-first posture branch as the baseline.
2. Replace the direct startup-to-support route with a staged fixed route:
   `A -> W1 -> W2 -> W3 -> B`.
   - keep `B` fixed as the full support target / terminal region
   - use pelvis height realization as the primary early-stage task metric
   - guard stage progression with stance knee/ankle realization plus
     pitch/support stability
   - keep a nominal startup rate target around `30°/s` and a hard cap around
     `100°/s`
3. Add explicit stage logs for `A`, `W1`, `W2`, `W3`, `B` and every stage
   transition.
4. Validate both:
   - startup-path transition from keyframe 0
   - grounded support-only holding
5. Re-run nominal-only probes under the updated staged-route execution.
   Preferred command:
   `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`
6. If staged fixed-route still cannot drive the robot into `B`, escalate to a
   longer-term MPC / online route-optimization design rather than continuing to
   tune a failing direct route.
7. Use the support-first checklist as the only decision gate for resuming PPO.

---

## Relationship To Main Walking Plan

This document is a focused design companion to:

- [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

`walking_training.md` remains the main milestone plan.
This file exists to keep the nominal reference architecture and `M2.5`
validation approach explicit and stable while the locomotion stack is still
being refactored.
