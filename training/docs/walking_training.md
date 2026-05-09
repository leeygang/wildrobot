# WildRobot Walking Training Plan

**Status:** Active locomotion plan for `v0.20.x`  
**Last updated:** 2026-04-17

---

## Purpose

This document defines the concrete walking plan after the failed `v0.19.5`
reward-first residual PPO attempt and the pivot toward a prior-reference-first
walking stack on current WildRobot v2 hardware.

The key architecture decision is now:

- the deployed runtime artifact remains a learned policy
- a solid prior reference must own locomotion semantics before PPO is trusted
- PPO is trained with that prior as structured guidance, not as the inventor of
  gait generation

This is not:

- pure reward-first gait discovery
- a full runtime MPC stack
- a mocap-first locomotion program

---

## Product Goal

The first realistic locomotion goal on current WildRobot v2 hardware is:

- zero-shot or near-zero-touch sim-to-real forward walking
- reliable stop and quiet standing
- in-place yaw turning
- moderate push recovery during standing and slow walking

The project is not targeting full ToddlerBot locomotion capability on day one.
The target is a ToddlerBot-style architecture and training recipe adapted to
WildRobot v2 morphology.

---

## Why This Training Approach

WildRobot's standing branch showed that PPO can learn quiet standing and partial
recovery strategies. It did not show reliable emergence of the full locomotion
and whole-body recovery mechanism from rewards alone.

The current training approach therefore changes from:

- reward shaping + bounded residual rescue

to:

- prior-guided PPO with explicit reference validation gates

The source of prior becomes:

- reduced-order locomotion dynamics, not mocap
- command-conditioned reference families, not a single hand-tuned gait
- morphology-aware nominal motion, not a copied external humanoid walk

---

## Mainline Runtime Contract

The intended runtime contract is:

```text
command
    ->
offline prior generator
    ->
command-indexed reference library
    ->
runtime lookup / interpolation / preview
    ->
reference preview + nominal `q_ref`
    ->
policy(obs, reference_preview, history) -> bounded correction
    ->
joint position targets
    ->
runtime / hardware
```

The policy remains the deployed artifact. The reference and IK layers are part
of the locomotion stack, but they must be strong enough that PPO is refining a
real walk rather than fabricating one.

Active decision:

- runtime FSM is deprecated and removed from the active path
- if old `walking_ref_v2` logic is reused, it is offline-generation only
- the reusable asset is the offline reference library, not a runtime gatekeeper

---

## Prior Strategy

For the current reference architecture, single-source-of-truth rule,
offline-prior design direction, and detailed reference validation gates, see:

- [`reference_design.md`](reference_design.md)

### Prior options comparison

The next phase must choose a solid prior family instead of continuing to patch
`v0.19.5` reward economics.

| Prior option | Development cost | Debuggability | Validation burden | Generalization upside | Fit for WildRobot v2 | Recommendation |
|---|---|---|---|---|---|---|
| `ZMP` / preview-style flat-ground reference | low-medium | high | medium | medium | good | best first executable prior |
| `ALIP` / capture-point / angular-momentum-aware prior | medium-high | medium | high | high | very good if tuned well | target upgrade after first prior clears gates |
| `Placo` / Mini-Duck-style whole-body kinematic references | medium | medium-low | high | low-medium | mixed on current hardware | useful as a prototyping tool, not mainline prior |

Interpretation:

- `ZMP` is the best first step because it is the easiest to verify visually and
  metrically on a servo-driven biped.  The v20 ankle_roll merge added a
  dedicated ankle_roll DOF, but the ZMP family still plans ankle_roll = 0;
  the new joint is exercised by the C2 stabilizer and the PPO residual,
  not by the offline prior.
- `ALIP` is the best medium-term direction because it has better upside for
  dynamic stepping, push adaptation, and broader command coverage.
- `Placo` is helpful for quickly generating or visualizing coherent whole-body
  motions, but it is a weaker single-source-of-truth prior on WildRobot because
  it is harder to prove dynamic validity and easier to drift into kinematic-only
  comfort.

### Recommended sequence

1. `v0.20.0`: build a `ZMP`-shaped offline prior library and validate it
   rigorously.
2. `v0.20.1`: train a ToddlerBot-style bounded residual PPO policy against that
   library.
3. `v0.20.2`: broaden command coverage and transfer hardening.
4. `v0.20.3`: upgrade the prior toward `ALIP` if `ZMP` saturates on dynamic
   stepping, push recovery, or command breadth.
5. Use `Placo` selectively for:
   - fast kinematic prototyping
   - style checks
   - startup / stop / turn reference authoring
   not as the first production prior.

### How to validate each option

`ZMP`

- visually: clear alternating step-out, no standing-shuffle, no pelvis dive,
  consistent stop/resume behavior, no large scissoring or toe drag
- metrically: easiest to validate with foothold preview, COM track, touchdown
  step length, and multi-seed survival

`ALIP`

- visually: same as `ZMP`, plus credible dynamic catch steps under speed
  changes and bounded push disturbances
- metrically: requires stronger checks on overspeed, pitch-rate spikes, and
  step-placement correction quality

`Placo`

- visually: kinematic motion may look good even when dynamics are wrong, so
  playback alone is not enough
- metrically: must always be validated in closed-loop MuJoCo, not only in
  kinematic viewer output

### What We Are Not Using First

- `AMASS` or mocap-retargeting as the first walking prior
- reward-first PPO without a validated prior
- `Placo`-only kinematic references without full env validation
- runtime `walking_ref_v2` FSM as the deployed locomotion controller

---

## IK Role

In the active `v0.20` path, IK is allowed in two places:

- offline, while generating the reference library
- optionally at runtime, as a lightweight adapter from reference targets to
  nominal `q_ref`

IK is not the source of locomotion semantics.

IK is used to:

- produce morphology-aware nominal joint targets
- keep nominal references executable on WildRobot v2
- support the first bounded-residual policy contract

IK is not used to:

- become the standalone locomotion controller
- reintroduce a runtime walking FSM
- replace the learned policy under contact mismatch
- hard-script all body coordination

The policy's job is still critical:

- correct for contact error
- correct for servo delay and backlash
- stabilize touchdown
- adapt under disturbances
- handle sim2real mismatch

---

## Observation Contract

The walking policy observation should include:

- IMU orientation and angular velocity
- joint positions and velocities
- previous action
- short contact / action history
- command forward speed
- command yaw rate
- reference preview:
  - future nominal joint targets
  - future foothold / foot-site targets
  - future pelvis / COM targets
  - phase as `sin/cos`
  - contact-phase annotations

ToddlerBot-first requirement:

- the first policy should be reference-conditioned with a dense preview

Cassie-compatible later:

- keep the preview interface reusable if the policy is later upgraded to higher
  authority and longer history

If a privileged path is needed during training, it should stay bounded and
remain removable at deployment.

---

## Action Contract

The first locomotion action contract should be:

- policy outputs bounded residual joint deltas around nominal `q_ref`

Form:

```text
q_target = q_ref_from_library + delta_q_policy
```

This is an explicit architectural decision, not an inherited default.

Why this first:

- safer for sim-to-real
- easier to debug than a direct policy
- preserves the reference semantics while PPO learns correction

Future upgrade path:

- if residual authority is clearly the limiting factor, later policy versions
  may move toward a Cassie-style direct action contract using the same reference
  preview asset
- do not take that step until the offline prior and ToddlerBot-style policy are
  already working

---

## Reward Design

The `v0.20.1` reward family should be imitation-dominant hybrid reward.

Primary terms:

- reference joint-target tracking
- pelvis pose tracking against the reference
- foot / site tracking against the reference
- contact-phase consistency
- survival

Secondary terms:

- forward command tracking
- stop / zero-drift behavior
- later yaw command tracking

Regularizers:

- action smoothness
- torque / effort
- residual magnitude
- slip penalty
- excessive impact penalty

Not the active plan:

- pure task-space reward without strong imitation terms
- pure unconstrained imitation with no task or command terms

Reason:

- pure task-space reward repeats the ambiguity that enabled `v0.19.5` exploits
- pure imitation over-trusts the prior and weakens command semantics
- the right first recipe is: strong reference tracking, plus enough task and
  regularization terms to preserve useful behavior and robustness

---

## Domain Randomization and Transfer Hardening

The locomotion stack should start with sim2real hardening as part of the main
plan, not a late add-on.

Mainline randomization families:

- friction
- damping
- mass
- Kp/Kd
- backlash
- frictionloss
- armature
- motor torque or speed limits
- action delay
- IMU noise and bias
- foot-switch timing noise or dropout

The mainline deployment loop should also include:

- scripted standing and squat replay
- scripted step-response replay
- sim-vs-real trace comparison
- versioned actuator realism parameters

SysID is a gate for sim-to-real deployment (`v0.20.2`), not for prior
generation or PPO training in simulation:

- before `v0.20.2` sim-to-real deployment, verify that the simulated servo
  model can plausibly track the reference on WildRobot v2
- if the servo model is wrong, do not trust PPO results trained against it
  for hardware deployment
- ZMP prior generation and PPO training in simulation may proceed using the
  existing MuJoCo actuator model, but hardware deployment requires verified
  SysID

---

## Software Stack Deliverables

### `control/`

Expected additions:

- offline prior generator / exporter
- command-indexed reference library format
- `control/reduced_model/lipm.py`
- `control/reduced_model/dcm.py`
- `control/kinematics/leg_ik.py`
- `control/adapters/reference_to_joint_targets.py`
- `control/references/locomotion_contract.py` (Milestone 0 schema)
- `control/kinematics/interfaces.py` (Milestone 0 interface)

### `training/`

Expected additions:

- locomotion env wrapper that consumes command velocity and reference state
- PPO config for ToddlerBot-style reference-conditioned locomotion
- fixed walking evaluation ladder
- rollout logging for reference tracking and touchdown quality
- reference-tracking metrics:
  - `ref/joint_track_rmse`
  - `ref/pelvis_pose_error`
  - `ref/foot_site_error`
  - `ref/contact_phase_match`

### `tools/`

Expected additions:

- `tools/sysid/`
- `tools/sim2real_eval/`
- replay and plotting scripts for sim-vs-real comparison
- Milestone 1 usable CLIs:
  - `tools/sysid/run_capture.py`
  - `tools/sim2real_eval/replay_compare.py`

### `runtime/`

Expected additions:

- locomotion policy runner with history and delay support
- unified log schema for sim and real execution
- `runtime/wr_runtime/logging/locomotion_log_schema.py` (Milestone 0 schema)

---

## Milestones

> **Three-level scope:** the milestone framework here describes the
> training-side plan (G-gates, smoke contracts, reward shape) at
> milestone granularity (v0.19.x → v0.20.x).  For:
> - **Architecture decisions** (prior selection, layer contracts,
>   validation ladder framework) see
>   [`reference_design.md`](reference_design.md).
> - **Active phase-by-phase TB-alignment work** (Phases 0-12, the
>   tactical decomposition of milestones v0.20.0-A through D) see
>   [`reference_architecture_comparison.md`](reference_architecture_comparison.md).
> - **Completed-phase closeouts** with measured gate movements see
>   [`../CHANGELOG.md`](../CHANGELOG.md).
>
> Per-milestone task lists below stay here because they carry
> training-specific G-gates, smoke contracts, and reward design
> commitments that are not architecture-level.

### Current status

- `v0.19.0` complete: locomotion-first contract, schema, and runtime/logging
  split are in place.
- `v0.19.1` complete: digital-twin realism profile, SysID capture, and
  sim-vs-real comparison tooling are implemented.
- `v0.19.2` complete: `walking_ref_v1`, bounded sagittal IK, and nominal
  reference smoke tests are implemented.
- `v0.19.3` implemented but first screening run failed:
  - run: `training/wandb/offline-run-20260402_082711-xn1wgkap`
  - failure mode: forward-dive / skate exploit
  - `eval_clean/success_rate = 0%`
  - `term_pitch_frac = 100%`
  - episode length collapsed to about `40` steps
  - achieved forward speed overshot command by about `2x`
  - `m3_swing_foot_tracking` and `m3_foothold_consistency` stayed nearly
    inactive
- `v0.19.3a` failed to clear the same gate:
  - run: `training/wandb/offline-run-20260402_104958-qhb3w0zc`
  - `term_pitch_frac = 100%`
  - `eval_clean/success_rate = 0%`
  - reduced command range and stronger tracking penalties did not create a
    viable stepping prior
- `v0.19.3b` established the root blocker:
  - nominal-only probe with zero residual still pitch-fails early
  - `q_ref` is not dynamically viable under the full M3 env
- `v0.19.3c` and `v0.19.3d` improved semantics and braking diagnostics, but
  did not materially improve the nominal-only probe gate
- `v0.19.3e` added support-first local clamping for swing-x and pelvis pitch:
  - clamp is active
  - survival still does not improve
  - PPO must remain paused
- `v0.19.4` complete: nominal reference validation (M2.5) and walking
  reference achieved (M3.0-A/B).  Ctrl ordering bug found and fixed.
  DCM COM trajectory implemented.  Nominal walking: 461 steps, 15 gait
  cycles, forward speed +0.038 m/s at cmd=0.15.
- `v0.19.5` attempted but handoff was premature (April 13-14):
  - PPO found lean-back exploit (v0.19.5b) and move-less exploit (v0.19.5c)
  - Root cause: nominal only delivered 25% of commanded speed, 30-50% of
    commanded step amplitude — propulsion was mis-assigned to PPO
  - Handoff gate tightened: now requires ≥50% command tracking, multi-seed
    robustness, step-amplitude diagnosis
  - Result: do not continue `v0.19.5x` reward surgery as the mainline path
- Current active stage is now **`v0.20.0`: Prior reference pivot**.
  - a solid prior reference must be chosen and validated before PPO resumes
  - `ZMP` is the recommended first executable prior
  - `ALIP` is the intended upgrade path if `ZMP` proves too conservative
  - `Placo` is a prototyping aid, not the mainline prior source of truth

### `v0.19.0` Platform Pivot

Software stack:

- define the locomotion-first runtime contract
- define the shared observation, action, and log schema
- add this walking plan as the active locomotion roadmap

Training approach:

- stop using standing-only PPO as the main product path
- define the locomotion task around forward speed and yaw commands

Sim2real:

- create one reproducible calibration checklist
- create one standing and squat replay capture on hardware

Exit gate:

- one documented train/eval/deploy contract exists for locomotion

### `v0.19.1` Digital Twin and SysID

Software stack:

- add actuator realism parameters under version control
- add sim-vs-real replay comparison tools

Training approach:

- no full walking target yet beyond smoke tests
- validate that the simulator reproduces measured actuator behavior

Sim2real:

- collect representative joint step responses
- fit delay, backlash, frictionloss, armature, and effective output limits
- model IMU noise and foot-switch imperfections

Exit gate:

- scripted sim-vs-real traces are close enough to justify policy transfer work

### `v0.19.2` Reference Generator v1

Software stack:

- implement `walking_ref_v1`
- implement `LIPM / DCM / capture-point` stepping logic
- implement simple swing-foot and pelvis target generation
- implement the first leg IK path

Training approach:

- use the reference only for smoke-test rollouts at first
- verify feasibility before training a full policy

Sim2real:

- replay the nominal reference slowly on hardware with safety limits
- verify foot clearance, touchdown geometry, and joint limit margin

Exit gate:

- the reference generator produces coherent stepping in simulation
- the nominal motion is safe enough to serve as a policy prior
- initial smoke command:
  - `uv run python tools/reference_smoke/run_reference_smoke.py --robot-config assets/v2/mujoco_robot_config.json --steps 300 --dt-s 0.02 --forward-speed-mps 0.12`

### `v0.20.0` Prior Reference Pivot

Purpose:

- stop treating reward tuning as the main walking lever
- choose a reference prior family that can own gait semantics
- define hard visual and metric validation before PPO resumes
- fully remove the runtime FSM from the active locomotion path

Status: **active**

Decision:

- start from a `ZMP`-shaped offline command-indexed prior library
- train the first policy in a ToddlerBot-style bounded residual setup
- keep the architecture Cassie-compatible later by preserving a reusable
  reference preview interface
- use `Placo` only for prototyping and visual authoring, not as the mainline
  validation truth

Why:

- `ZMP` is cheapest to build, easiest to debug, and easiest to validate on a
  small position-controlled biped
- `ALIP` has better long-term dynamic coverage, but should upgrade a proven
  prior rather than replace an unvalidated stack
- `Placo` is valuable for authoring and inspection, but on WildRobot it is too
  easy to get kinematically plausible motions that are not dynamically viable
- bounded residual PPO is the safest first policy contract
- Cassie-style direct control remains an upgrade path, not the first step

Execution milestones:

1. `v0.20.0-A`: Prior contract freeze
2. `v0.20.0-B`: `ZMP` prior library generation and viewer
3. `v0.20.0-C`: Reference + IK + short-horizon feasibility validation
4. `v0.20.0-D`: PPO unblocks only after the prior gate clears

These milestone labels are canonical and should match
[reference_design.md](reference_design.md).

SysID note:

- servo-model verification is required before sim-to-real transfer
  (`v0.20.2`), not before ZMP prior generation
- the ZMP library can be generated and validated in simulation using the
  existing MuJoCo actuator model
- hardware SysID data collection should happen in parallel and must gate
  `v0.20.2` deployment, not `v0.20.0` prior work

Exit rule:

- no `v0.20.1` PPO training until the `v0.20.0` prior gate passes

Primary design note:

- [`reference_design.md`](reference_design.md)

### `v0.20.0-A` Prior contract freeze

Expectation:

- this milestone defines a stable reference contract, not walking quality

Quick visual check:

- replaying the same command bin always produces the same reference trajectory
- no target jumps or phase discontinuities are visible in the viewer traces

How this advances the end goal:

- removes interface ambiguity so later milestones measure motion quality, not
  contract mismatch

Tasks:

- define one prior interface:
  - command-indexed library entry
  - preview output
  - nominal `q_ref`
  - foothold targets
  - pelvis / COM targets
  - phase / contact annotations
- deprecate the runtime FSM from the active runtime path
- define one validation schema for reference-only, feasibility, and policy
  probes
- freeze the first reference family as `ZMP`-shaped, with later `ALIP`
  compatibility in the contract

Required schema contents:

- command key:
  - `vx` in the first branch
  - reserved extension fields for `vy` and `yaw_rate`
- timing:
  - `dt`
  - normalized phase
  - step period / cycle length
- nominal targets:
  - `q_ref`
  - optional `dq_ref`
- task-space targets:
  - pelvis pose
  - COM target
  - left/right foot pose targets
- annotations:
  - stance side
  - contact mask / contact state
  - stop / resume markers
- metadata:
  - generator version
  - command-bin bounds
  - validation and feasibility flags

Visual validation:

- prior playback shows alternating step release
- stop command visibly settles into quiet standing
- no visible stepping-in-place when forward command is zero

Metric validation:

- command-conditioned outputs are deterministic and bounded
- preview trajectories are continuous
- no discontinuous target jumps above the configured rate limit
- the same library entry can be consumed by both:
  - bounded residual policy
  - future broader-authority policy

### `v0.20.0-B` `ZMP` prior generator and viewer

Expectation:

- the prior itself should already look like a walk family, before PPO is
  involved

Quick visual check:

- at walk command, feet clearly step out with alternating left/right pattern
- at stop command, posture settles to quiet standing
- resume restarts stepping without a jump

How this advances the end goal:

- proves locomotion semantics are encoded in the prior rather than invented by
  PPO

Tasks:

- implement the first `ZMP`-shaped offline prior library
- use a concrete, reviewable `ZMP` generation pipeline rather than a vague
  “preview-like” approximation
- provide a viewer for:
  - prior-only playback
  - prior + IK playback
  - prior traces over command sweeps

Reference implementation target:

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
  - lookup-table export indexed by command

Visual validation:

- at `cmd=0.15`, the feet clearly step out rather than shuffle
- pelvis stays bounded and does not dive forward
- left/right stepping is visibly alternating and repeatable
- stop/resume transitions are visible and non-chaotic

Metric validation:

- commanded foothold mean at `cmd=0.15` is `>= 0.045 m`
- commanded step period remains within `0.45-0.75 s`
- previewed COM and pelvis targets stay bounded and phase-consistent

### `v0.20.0-C` Reference + IK + short-horizon feasibility validation

Expectation:

- prior motion remains trackable after IK and full env dynamics for short
  horizons

Quick visual check:

- startup and one-to-two gait cycles stay coherent in full env replay
- if failure occurs, it looks like bounded balance/tracking limit, not
  incoherent reference behavior

How this advances the end goal:

- confirms the prior is physically usable as a learning target for PPO

Tasks:

- run the prior with IK / nominal targets in full MuJoCo env
- verify one-to-two gait cycles, startup, and stop/resume remain trackable
- do not require long-horizon open-loop walking from the offline prior

Measurement contract for this gate:

- `control steps` means env control-loop ticks (`ctrl_dt`), not MuJoCo
  substeps
- `footfalls` means touchdown events (left/right contact transitions), not
  control-step count
- use both:
  - control steps for short-horizon survival timing
  - touchdown events for gait-cycle progression

Visual validation:

- startup, stop, and one-to-two gait cycles remain coherent in full env replay
- the motion looks trackable rather than instantly impossible
- failures look like bounded tracking or contact limits, not incoherent
  reference design

Metric validation:

- short-horizon env replay survives at least `45` control steps before
  catastrophic pitch termination (equivalent to `>= 0.9 s` at `ctrl_dt=0.02`)
- touchdown progression includes at least one left touchdown and one right
  touchdown in the same replay
- simulated tracking RMSE stays within the actuator-feasibility budget
- touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`

### `v0.20.0-D` PPO unblock decision

Expectation:

- this is a gate decision, not a new behavior milestone

Quick visual check:

- only unblock if prior replay already looks like a real walk family and not a
  balance exploit

How this advances the end goal:

- prevents restarting PPO on an unready prior and repeating `v0.19.5` failure
  modes

PPO is unblocked only if:

- `v0.20.0-A` through `v0.20.0-C` pass
- the offline prior visually looks like a real walk family, not a balance
  exploit
- the remaining issues are plausibly correction-layer problems

### `v0.20.1` ToddlerBot-style PPO

Expectation:

- PPO improves robustness and tracking while preserving the prior’s gait
  structure
- the smoke validates that the offline prior is learnable by PPO at the
  current Phase 9D operating point, `vx = 0.20` (paired with the
  WR-pendulum-scaled `cycle_time = 0.96 s`; see CHANGELOG
  `v0.20.1-phase9D-cycle-time-scaling`)
- the smoke policy contract is a temporary validation bridge, not the final
  locomotion policy endpoint
- **current status (2026-05-09):** no valid current-model PPO checkpoint
  exists.  The April 25 smoke7 checkpoint is a 19-action pre-ankle-roll
  artifact; the current 21-DOF model needs a fresh smoke at the Phase 9D
  operating point.

Quick visual check:

- learned motion still resembles the prior family
- steps become more stable and less fragile, rather than becoming a different
  exploit gait
- PPO should stabilize and track the prior better, not invent a new propulsion
  strategy that the prior should own

How this advances the end goal:

- adds closed-loop correction so walking survives longer and tracks commands
  better
- decides whether the current offline prior is good enough to justify the next
  policy upgrade

Smoke scope:

- fixed q_ref operating point: `vx = 0.20` (Phase 9D)
- single seed
- smoke objective is prior validation, not command-breadth validation
- command randomization is enabled as TB-inspired robustness pressure, but
  the runtime reference service is still a fixed one-trajectory q_ref.  Do
  not claim full ToddlerBot multi-command reference conditioning until q_ref
  lookup is command-conditioned.

Smoke contract:

- Prior:
  - offline lookup-table prior remains the only locomotion source of truth
  - no runtime FSM or online gait-mode logic is reintroduced
- Observation:
  - use a ToddlerBot-like current-state observation first:
    - IMU orientation and angular velocity
    - joint positions and velocities
    - previous action
    - command forward speed
    - command yaw rate
    - phase as `sin/cos`
    - current reference slice:
      - current `q_ref`
      - current pelvis / torso target
      - current foot / site target
      - current contact / stance annotation
  - if anticipation is needed, allow only a very short preview:
    - one or two future anchor frames
  - do not start the first smoke with a large dense future preview window
- Action:
  - use a higher-authority residual action around the offline reference
  - form:
    - `q_target = q_ref_from_library + delta_q_policy`
  - residual authority should be materially larger than a tiny rescue delta,
    especially on the leg joints
  - this higher-authority residual is a smoke-stage debugging compromise, not
    the final intended contract
  - **G1 — concrete starting bounds for the smoke** (per-joint clip on
    `delta_q_policy`, applied symmetrically; tighten if anti-exploit metric
    G5 trips):
    - leg joints (hip pitch / hip roll / knee / ankle pitch / ankle roll):
      `±0.25 rad`
    - all other joints: `±0.20 rad`
    - rationale: post-ToddlerBot alignment, residual authority must stay a
      correction layer.  ToddlerBot uses action scale 0.25; matching that
      scale keeps G5 meaningful and prevents PPO from owning propulsion that
      the prior should own.  The old ±0.50 smoke debug bound is retired.
- Reward:
  - use an imitation-dominant ToddlerBot-like reward family
  - primary terms:
    - `ref/joint_track`
    - `ref/body_quat_track`
    - `ref/torso_pos_xy` (smoke4 onward — body-position parity with TB
      `_reward_torso_pos_xy`; smoke YAML α=90)
    - `ref/lin_vel_z` (smoke6 onward — TB-aligned vertical body velocity
      tracking.  Continuous phase signal.  Actual rotated to true
      body-local via `inv(root_quat)`; reference computed from
      finite-diff of prior `pelvis_pos[2]` per Appendix A.3 G2.)
    - `ref/ang_vel_xy` (smoke6 onward — TB-aligned body roll/pitch
      angular velocity.  Continuous phase signal.  Reference is zero
      because the yaw-stationary prior has no pitch/roll oscillation —
      term reduces to penalizing body sway during each gait phase.)
    - `ref/contact_phase_match` (smoke6 onward — TB boolean equality
      count, re-enabled at TB weight 1.0; previously gated to 0 since
      smoke3 with a WR-specific Gaussian shape.  Switched to TB form
      `sum(stance_mask == ref_stance_mask)` to resolve a doc/code
      mismatch on the Gaussian kernel formula and to align strictly
      with TB.  Discrete event anchor for gait phase at
      touchdown/takeoff transitions.)
    - `ref/site_track`
    - `ref/body_lin_vel_track`
    - `ref/body_ang_vel_track`
    - `survival`
  - secondary terms:
    - `cmd/forward_velocity_track`
    - `ref/contact_phase_match` (diagnostic-only for the first smoke; see G3)
  - regularizers:
    - `action_rate`
    - `torque`
    - `slip`
    - `impact`
    - `residual_magnitude`
  - compute body / site / velocity tracking in torso-relative or root-relative
    coordinates where applicable
  - do not reuse the `v0.19.x` task-space reward family for this smoke
  - **G2 — reference velocity fields**:
    - the offline `ReferenceLibrary` schema currently stores only positions
      (`pelvis_pos`, `com_pos`, `left_foot_pos`, `right_foot_pos`); body
      and site linear / angular velocities are not stored
    - decision: **compute velocity references on-the-fly in the env** from
      finite differences of the position fields stored in the library;
      angular velocity references default to zero (the prior is yaw-stationary)
    - rationale: keeps the library asset narrow and avoids growing the
      schema before we know the velocity-tracking reward is actually
      pulling its weight; finite-diff velocity is good enough for a
      stable Gaussian-shaped tracking reward at our `dt = 0.02 s`
    - **NOT a "matches ToddlerBot" decision**: ToddlerBot's library
      generator (``toddlerbot/algorithms/zmp_walk.py:174``,
      ``mujoco_replay``) records ``body_lin_vel`` and ``body_ang_vel``
      from the kinematic forward step explicitly; we are choosing
      finite-diff because we don't currently run that mujoco_replay
      step and don't want the schema growth right now.  This is a
      WildRobot-specific simplification, not a ToddlerBot inheritance.
    - revisit if a future prior generator (ALIP, etc.) emits velocity
      directly with better fidelity than finite-diff, or if the smoke
      shows velocity-tracking reward gradients are noisy at the
      finite-diff numerical-derivative quality
  - **G3 — `ref/contact_phase_match` definition** (smoke6 onward):
    - boolean equality count per foot, matching ToddlerBot's
      `_reward_feet_contact` exactly:
      - `r = sum(stance_mask == ref_stance_mask)` ∈ `{0, 1, 2}`
      - `cmd_contact ∈ {0, 1}` from `traj.contact_mask`
      - `actual_contact ∈ {0, 1}` from MuJoCo foot-floor contact
      - per-step value 0/1/2; multiplied by reward weight then `dt`.
    - The diagnostic `ref/contact_phase_match` continues to log a
      0..1-normalized fraction (`r / 2`) so smoke3-5 logs remain
      qualitatively comparable.
    - rationale: TB demonstrates this boolean form works empirically on
      a similar small biped — the gradient flows through the policy's
      continuous joint targets to the foot trajectory, and small action
      shifts cross the binary contact threshold differently.  Earlier
      smokes used a WR-specific Gaussian shape (`exp(-x²/(2σ²))` with
      σ=0.5) out of an over-cautious "boolean has no gradient" concern,
      which created two issues: (1) the doc spec wrote a different
      formula (`exp(-x²/σ²)`) than the code implemented, and (2) the
      WR-specific Gaussian + sigma was a divergence from TB without
      concrete justification.  Switching to TB's form resolves both.
    - **smoke6 disposition**:
      - enabled at TB's weight 1.0 (previously gated to 0).
      - the original smoke-stage gate (`mean ref/contact_phase_match
        < 0.90` → weight 0) was written for the Gaussian form and
        argued that a known-misaligned contact schedule was "structured
        reward noise."  With TB's boolean form at TB's weight, we are
        explicitly choosing TB-alignment over the WR-specific
        graduation policy, on the rationale that TB's empirical track
        record on a similar robot is stronger evidence than WR's a
        priori noise concern.  If smoke6 shows the contact term
        actively hurting (e.g., contact_phase_match drops below smoke5
        baseline 0.67 with no other gain), revisit.

Tasks:

- implement a `v0.20.1` env path that reads the offline prior directly
- expose the ToddlerBot-like observation fields listed above
- wire the higher-authority residual action around `q_ref_from_library`
- add a dedicated `v0.20.1` reward branch with explicit `ref/*` metrics
- lock the smoke q_ref/eval operating point to `vx = 0.20` (Phase 9D)
- reject runs where PPO improvement comes mainly from inventing propulsion that
  the prior should own

Pre-smoke checks:

- zero-action policy reproduces the nominal `q_ref` target path
- current reference slice and optional short preview advance correctly with the
  replay index
- reference horizon never overruns the trajectory length
- reward logs expose the primary `ref/*` tracking metrics from iteration 1
- if the contact-alignment probe fails the G3 bar, the smoke YAML must set
  `reward_weights.ref_contact_match = 0.0` before launch; do not carry a
  known-noisy contact term into the first PPO run
- **G7 — prior-vs-body sanity at the smoke command**:
  - run `tools/v0200c_per_frame_probe.py --horizon 200` (default
    `--vx 0.20` post Phase 9D — TB-step/leg-matched operating point at
    the WR-pendulum-scaled cycle_time=0.96 s) and record the
    deterministic baselines that will be used to judge whether PPO is
    recovering vs degrading.  Operating-point shifts change these
    baselines materially; re-run after every cycle/vx change.
    - first lever-sign-flip step (vx=0.20: step 2;
      historical vx=0.265: step 2; historical vx=0.15: step 2)
    - pelvis-x lag at step 30 (vx=0.20: -13.5 cm;
      historical vx=0.265: -13.0 cm; historical vx=0.15: -10.2 cm)
    - pelvis-x lag at step 70 (vx=0.20: n/a — body falls before step 70;
      historical vx=0.265: n/a; historical vx=0.15: -22.0 cm)
    - free-float survival ctrl steps (vx=0.20: ~55 ctrl steps with pitch
      term; historical vx=0.265: ~54; historical vx=0.15: ≥200/200 —
      both Phase 9A and Phase 9D operating points are above WR's
      open-loop static-stability envelope, so PPO must own the
      closed-loop balance entirely)
    - free-float terminal aPLVx (vx=0.20: ~-0.22 m at step 55;
      historical vx=0.265: ~-0.15 m at step 54; historical vx=0.15:
      -0.0104 m at step 199)
  - the smoke succeeds if PPO measurably reduces these gaps (positive
    pelvis-x progress, longer survival); it fails if PPO collapses below
    these baselines or matches them after meaningful training compute
  - this is a non-blocking sanity probe, not a reward signal — it just
    pins the "what we're trying to beat" baseline before training starts

Visual validation:

- by the smoke horizon, the learned policy produces visible reference-like
  stepping
- the learned gait still resembles the prior family rather than a new exploit
- the smoke does not need stop / resume breadth or broader command coverage
- the learned motion should look more stable than the pure prior replay, not
  qualitatively different

Metric validation:

- reference-tracking metrics improve materially from initialization:
  - `ref/joint_track`
  - `ref/body_quat_track`
  - `ref/site_track`
  - `ref/body_lin_vel_track`
  - `ref/body_ang_vel_track`
  - `ref/contact_phase_match` (diagnostic in the first smoke unless G3 probe
    passes; do not require it to improve when the term is intentionally at
    zero weight)
- **G4 — early-horizon gate (informational only)**:
  - the current Phase 9D bare-q_ref baseline at `vx=0.20` falls by pitch
    at step ~55 over a 200-step deterministic run.  PPO is being asked
    to keep the body upright and track the prior; the historical
    `vx=0.15` baseline where free-float survived to the horizon is no
    longer the operating point.  Survival is essentially unchanged from
    the prior `vx=0.265` operating point under cycle_time=0.72 s
    (~54 vs ~55 ctrl steps), so PPO's balance-recovery budget is the
    same under either operating-point choice.
  - softened early-horizon targets (informational, do not abort runs that
    miss these — only abort if collapsing below the open-loop baseline):
    - `Evaluate/forward_velocity >= 0.0 m/s`
    - `Evaluate/mean_episode_length >= 100`
    - `term_pitch_frac < 0.80`
  - the originally-stated stricter early-horizon thresholds
    (`forward_velocity >= 0.04 m/s, episode_length >= 150,
    term_pitch_frac < 0.50`) move to the **mid-horizon** gate; only the
    promotion horizon below stays hard
- by the promotion horizon (hard gate — pass or fail the smoke).
  v0.20.1-smoke2 update: ``eval_walk/success_rate >= 0.60`` was removed
  in favour of two ToddlerBot-style ``Evaluate/*`` floors on the
  deterministic eval rollout.  Rationale: ToddlerBot has no
  ``success_rate`` concept anywhere in
  ``toddlerbot/locomotion/{train_mjx,walk_env,mjx_env}.py`` —
  checkpoint quality is read off ``Evaluate/mean_reward`` +
  ``Evaluate/mean_episode_length`` + per-term ``Evaluate/<term>``
  tracking.  Once we adopt the same reward shape (which we now have:
  ``lin_vel_xy=5.0`` at α=200, ``motor_pos=5.0``, strict
  ``-done`` survival), eval reward + eval episode length already
  encode "alive AND tracking cmd_vx AND tracking q_ref"; a separate
  binary success bool just adds a degenerate signal that the
  truncation-based ``env/success_rate`` already showed (stuck at 0
  through smoke1).  CLAUDE.md § "follows ToddlerBot" applies — we
  no longer have a rationale to keep a WildRobot-specific gate when
  the reward block matches.
  - `env/forward_velocity >= 0.075 m/s` (train rollout, weak floor only;
    `tracking/cmd_vs_achieved_forward` carries the command-scale signal)
  - `Evaluate/forward_velocity >= 0.075 m/s` (deterministic eval
    rollout — weak forward-progress floor; at Phase 9D this is
    deliberately paired with the stricter command-error and ratio gates)
  - `Evaluate/mean_episode_length >= 475` (95% of 500-step horizon;
    second half of the dropped success_rate row)
  - `Evaluate/cmd_vs_achieved_forward <= 0.075 m/s`
  - `tracking/cmd_vs_achieved_forward <= 0.075 m/s` (train rollout)
  - touchdown step length mean `>= max(0.030 m, 0.50 * vx_eval *
    cycle_time / 2)`.  At the Phase 9D operating point (`vx_eval=0.20`,
    `cycle_time=0.96`), this is `>= 0.0480 m`; use `>= 0.050 m` in the
    launch checklist.  (Same launch-checklist value as the historical
    Phase 9A operating point at `vx=0.265, cycle=0.72`, by coincidence
    of the kinematic identity `step = vx × cycle / 2` — Phase 9D
    preserves step length while scaling cycle and vx.)  The old
    absolute `0.030 m` floor is too weak at any post-shuffle operating
    point because it is only ~32% of the nominal prior step.
- **G5 — anti-exploit metric (hard gate)**:
  - to reject the "policy invents propulsion the prior should own" failure
    mode, log and gate on:
    - median absolute residual on hip-pitch + knee + ankle-roll channels:
      `|delta_q_policy[hip_pitch_L,R, knee_L,R, ankle_roll_L,R]| p50`
      ≤ `0.20 rad`
      across the eval horizon (residual stays a correction, not a
      replacement gait)
    - realized-vs-commanded forward speed ratio:
      `0.6 ≤ Evaluate/forward_velocity / eval_velocity_cmd ≤ 1.5`
      on the pinned deterministic eval rollout.  For multi-command
      training rollouts, do not gate on the mean of
      `tracking/forward_velocity_cmd_ratio`; sampled near-zero commands make
      that average numerically unstable.  Use
      `tracking/cmd_vs_achieved_forward` for train-rollout command tracking.
  - if either bound is violated at the promotion horizon, the smoke fails
    even if the forward-velocity gate passes — explicitly to prevent a
    false-positive caused by a residual-driven exploit gait
- **G6 — policy init details**:
  - actor: hidden `[256, 256, 128]`, ELU
  - critic: same shape, ELU
  - `log_std_init = -1.0` (matches v0.19.5c; not changing exploration
    pressure for the smoke)
  - residual head zero-initialized so the iter-0 policy is exactly
    bare-q_ref replay (this is what the pre-smoke "zero-action policy
    reproduces nominal q_ref" check verifies)

Post-smoke policy direction:

- if the `v0.20.1` smoke succeeds, the next policy milestone should explicitly
  move toward a full ToddlerBot / Cassie-style higher-authority contract
- the smoke-stage higher-authority residual should not be treated as the final
  architecture
- the offline prior asset, reference-tracking reward family, and observation
  interface should be reusable in that upgrade

**M2 — compute budget cap**:

- single bounded smoke run: ~20M env steps (150 PPO iterations at
  `num_envs=1024 × rollout_steps=128` = 1024 × 128 × 150 ≈ 1.97×10⁷
  policy decisions), cap configured in the smoke YAML
- (an earlier revision of this section mis-stated the budget as ~5M env
  steps — that derivation conflated env steps with sub-physics steps;
  the YAML's 150-iter cap is the authoritative budget)
- if the promotion-horizon gates haven't been met by ~20M env steps, the
  smoke is considered failed (do not extend the run hoping it converges
  later — prefer to come back to the prior or contract design)
- single seed for the smoke; if it passes, multi-seed validation belongs
  to the next milestone, not this one

**M1 — failure-mode decision tree**:

If the smoke fails (does not meet the promotion-horizon gates within the
M2 compute budget), the next action depends on which gate failed:

- `env/forward_velocity` flat near zero AND `episode_length` short
  → **prior issue**: the LQR prior is not producing executable forward
    push; the body recovers balance but doesn't translate.  Action:
    return to v0.20.0-x prior surgery — specifically the prior-vs-body
    gap at the smoke command (`tools/v0200c_per_frame_probe.py`).  Do
    not relax G5 to "fix" this with more residual authority.
- `forward_velocity` reaches the gate BUT G5 fails (residual magnitude
  too large or realized/commanded ratio out of band)
  → **action-authority leak**: the policy is doing the propulsion, not
    the prior.  Action: tighten the G1 residual bound (start with -50%),
    rerun.  If still failing, the prior is too weak to be a residual
    target — return to prior surgery, not reward surgery.
- `forward_velocity` reaches the gate, G5 passes, BUT
  `term_pitch_frac >= 0.50` at promotion horizon
  → **balance issue**: closed-loop policy has learned to translate but
    not to stay upright.  Action: increase regularizer weight on
    `pitch_rate` and `slip`; consider widening the residual bound on
    ankle pitch only (the joint that owns body-pitch correction).
- `ref/contact_phase_match` decreases over training while
  `forward_velocity` increases
  → **gait drift**: the policy is achieving forward motion by abandoning
    the prior's contact schedule.  Action:
    - if `reward_weights.ref_contact_match == 0.0`, first fix the contact
      baseline (prior / sigma / acceptance decision) before promoting the term
      into the reward
    - if the term is already enabled, increase weight on
      `ref/contact_phase_match` and `ref/joint_track`; do not relax other
      ref/* terms.
- mixed / unclear failure
  → defer to manual diagnosis with the per-frame probe and the eval
    rollout video before deciding next milestone
- **shuffle exploit** (added 2026-04-24 after smoke3/4/5/6 evidence):
  `forward_velocity` reaches gate, G5 passes, balance is fine, but
  `tracking/step_length_touchdown_event_m` parks well below the 0.030 m
  gate (typical 0.018-0.022 m, ~33-44% of the prior's 0.054 m nominal).
  Per-foot stride is bilaterally short, not one-sided.  PPO satisfies
  cmd_vx via micro-cadence rather than tracking the prior's foothold
  geometry.  Action: **do NOT add another imitation reward term**.  Four
  smokes (3/4/5/6) ruled out reward-only fixes.  TB has the same reward
  family and avoids shuffle via the training curriculum (multi-cmd
  resample + zero_chance + DR), not a stride-amplitude reward.  Promote
  the next deferred TB curriculum mechanism (smoke7 below).

**M1 — shuffle-exploit context (the missing TB lever)**:

The shuffle exploit is the dominant single-point failure mode through
smoke6.  It is structural to single-point training without
domain randomization, not a reward-shape problem.  TB's recipe avoids it
via:

- **Multi-velocity sampling** (`vx ∈ [-0.2, 0.3]`, `resample_time = 3.0
  s`).  At vx=0.3 the body's max stepping cadence is exceeded by what
  cadence-only would require, **forcing** stride extension.  Once
  learned for high vx, the long-stride basin carries down to lower
  commands within the same policy.
- **`zero_chance = 0.2` + `deadzone = [0.05, 0.05, 0.2]`** gates the
  cadence reward (`feet_air_time *= ||cmd|| > 1e-6`,
  `walk_env.py:302`).  20% of episodes have cmd≈0 → policy must learn
  that shuffling without a command pays nothing.  Breaks the "always
  shuffle" basin from the low-cmd end.
- **Domain randomization** (backlash, friction, kp, push_torso) makes
  micro-shuffles fragile (small steps amplify backlash and can't recover
  from pushes); long stride is more disturbance-robust.

**TB stride evidence at vx=0.15** (analytical, from
`toddlerbot/algorithms/zmp_walk.py:258`):

```
stride_per_step ≈ vx × cycle_time / 2
                = 0.15 × 0.72 / 2
                = 0.054 m
```

(`cycle_time = 0.72 s` from `toddlerbot/locomotion/mjx_config.py:98`.)

WildRobot's prior at vx=0.15 (cycle=0.72) delivered 0.045-0.053 m
steady-state per `tools/v0200c_per_frame_probe.py --vx 0.15` — same
design point.  That is why the old `0.030 m` G4 floor was reasonable
before Phase 9A.  Under the Phase 9D operating point (vx=0.20,
cycle=0.96), nominal step is `0.20 × 0.96 / 2 = 0.096 m`, so the
"at least half the prior" floor is about `0.048 m` — same launch
checklist value `>= 0.050 m`, by coincidence of the kinematic identity
that Phase 9D preserved step length while scaling cycle and vx.

### `v0.20.1-smoke6-prep3` Loader fix + actually-test smoke6

Status: **active**

Smoke6 (`offline-run-20260423_213122-v1kclz2w`) was a **null run**.  A
YAML loader bug in
`training/configs/training_config.py:_parse_reward_weights_config`
silently dropped the `lin_vel_z` and `ang_vel_xy` weights set in the
smoke YAML — both fell through to the dataclass default of 0.0, so the
TB phase-signal hypothesis was never tested.  Only `ref_contact_match`
(boolean, w=1.0) was actually enabled; the "stride still fails" outcome
is essentially smoke5 + an inert boolean contact bonus.

Tasks:

1. Fix `_parse_reward_weights_config`: add
   `lin_vel_z=rewards.get("lin_vel_z", 0.0)`,
   `ang_vel_xy=rewards.get("ang_vel_xy", 0.0)`,
   `lin_vel_z_alpha=rewards.get("lin_vel_z_alpha", 200.0)`,
   `ang_vel_xy_alpha=rewards.get("ang_vel_xy_alpha", 0.5)`.
2. Add `tests/test_config_load_smoke6.py`: load the smoke YAML, assert
   every key under `reward_weights:` round-trips to the dataclass field
   with the same value.  Catches this class of bug at config-load time
   instead of at training-time-result interpretation.
3. Re-run smoke6 unchanged.

Decision rule after the re-run:

- if `reward/lin_vel_z` and `reward/ang_vel_xy` are alive at
  training-time and stride moves → smoke6 succeeds late; promote.
- if both terms are alive but stride is still parked → smoke6
  hypothesis falsified; phase signals are not the missing piece;
  proceed to smoke7 (multi-cmd curriculum).
- if either term is dead at training-time err scale → recalibrate α
  per the smoke6-prep2 commitment, single rerun.

### `v0.20.1-smoke7` Promote TB-inspired command/DR pressure

Status: **active, Phase 9A-rescoped** (smoke6-prep3 result falsified
phase-signal hypothesis; five reward-only smokes 3/4/5/6/6-prep3
exhausted reward-only space; April 25 smoke7 is obsolete for the current
21-DOF model).

**Scope revision (2026-04-24):** the original smoke7 plan was "multi-cmd
alone, no DR" with the hypothesis "vx=0.20 hits a cadence ceiling →
forces stride extension."  An open-loop probe at vx ∈ {0.10, 0.15, 0.20,
0.25} via `tools/v0200c_per_frame_probe.py` falsified that mechanism:
WR's prior uses **fixed cycle_time = 0.640 s** across all vx bins
(cadence 1.56 Hz/foot regardless of cmd; only stride per step varies).
PPO's current shuffle at 6.5 Hz/foot already runs 4× the prior's cadence,
and at vx=0.20 needs only 9.5 Hz/foot — well within servo capability.
**No cadence ceiling exists in the WR prior at any vx ∈ [0.10, 0.25].**

TB's prior (`toddlerbot/locomotion/mjx_config.py:98`) is also fixed-cadence
(cycle_time = 0.72 s).  TB's anti-shuffle mechanism is therefore not a
cadence ceiling either — it's the combined effect of multi-cmd
distribution + domain randomization (backlash + friction + kp + IMU
noise make micro-shuffles fragile; pushes are NOT the load-bearing piece
since TB's `add_push` defaults to False for walk).

Smoke7 enables TB-aligned command/DR pressure to test which combination
of mechanisms is load-bearing.  Challenge: this is not yet full
ToddlerBot multi-command reference conditioning, because WR's
`RuntimeReferenceService` still serves one fixed q_ref trajectory.  Treat
this as a fixed-operating-point prior smoke with randomized command
pressure, not proof that the multi-command reference stack is complete.

Scope:

- **Multi-cmd curriculum** (TB anti-shuffle lever #1):
  - `cmd_resample_steps: 150` (3.0 s at ctrl_dt=0.02 — TB resample_time)
  - `cmd_zero_chance: 0.2` (TB default)
  - `cmd_deadzone: 0.05` (TB vx-channel default)
  - vx range: `[0.0, 0.30]` — covers the Phase 9D operating point
    (`0.20`) with generous upper-bracket robustness pressure (1.5×
    operating point) while still deferring negative cmd to v0.20.4.
    Range was historically `[0.0, 0.20]` pre-Phase-9A, then `[0.0,
    0.30]` post-Phase-9A; kept at `[0.0, 0.30]` through Phase 9D so
    smoke results stay directly comparable across the operating-point
    transitions.
- **Domain randomization** (TB anti-shuffle lever #2):
  - `domain_rand_friction_range: [0.4, 1.0]` (TB)
  - `domain_rand_frictionloss_scale_range: [0.8, 1.2]` (TB)
  - `domain_rand_kp_scale_range: [0.9, 1.1]` (TB)
  - `domain_rand_mass_scale_range: [0.9, 1.1]` (WR-specific multiplicative;
    TB uses additive `body_mass_range: [-0.2, 0.2]` kg — different
    semantics, kept WR's existing form)
  - `domain_rand_joint_offset_rad: 0.03` (WR-specific reset-time joint
    offset; TB has no equivalent — TB uses `rand_init_state_indices`
    instead)
- **IMU noise** (sim2real hardening, secondary):
  - `imu_gyro_noise_std: 0.05 rad/s` (conservative vs TB's gyro_std=0.25
    under bias-walk RW)
  - `imu_quat_noise_deg: 2.0` (conservative vs TB's quat_std=0.10 rad ≈
    5.7°)
- **Eval-cmd override** (G4 readout protection):
  - `eval_velocity_cmd: 0.20` — pins eval rollouts at the Phase 9D
    operating point.  Without this, eval would sample uniformly across
    `[0.0, 0.30]` and dilute G4.  The fixed q_ref trajectory is also
    pinned to `loc_ref_offline_command_vx: 0.20`; q_ref/eval-cmd
    mismatch is a null smoke.
- **NOT enabling**: pushes (TB walk default add_push: False), yaw cmd
  (defer to v0.20.4 — WR prior lacks yaw), negative vx (prior lacks
  negative-vx generation).
- Reward family unchanged from smoke6-prep3 (TB-complete and verified
  alive in smoke6-prep3).

Why DR + command pressure jointly (Option C) and not isolated:

- Per the probe results, the cadence-ceiling story is wrong; multi-cmd
  alone has weak a-priori support.
- DR alone would test "is robustness pressure load-bearing for shuffle?"
  cleanly, but the M1 fail-mode tree has no precedent for which
  TB-inspired knob is load-bearing.
- Per CLAUDE.md "follow ToddlerBot before WR-specific design", TB enables
  both by default — Option C is the closest to TB's actual recipe.
- Trade-off accepted: if smoke7 succeeds, we don't know which mechanism
  was load-bearing without a follow-up ablation (smoke7-ablate-DR or
  smoke7-ablate-multicmd).  That's a known follow-up cost, not a
  blocker.  If smoke7 fails, we have stronger evidence the bottleneck
  is the prior, not the training distribution.

Pre-smoke checks (all completed in smoke7-prep1):

- ✅ DR plumbing audit: `wildrobot_env.py` `_sample_domain_rand_params`
  → `_get_randomized_mjx_model` applied per ctrl step.
- ✅ Multi-cmd resample plumbing audit: `wildrobot_env.py:1644-1664`
  resamples velocity_cmd at `cmd_resample_steps` interval, honors
  zero_chance + deadzone via shared `_sample_velocity_cmd`.
- ✅ YAML loader audit: all DR + cmd_* keys read by
  `_parse_env_config` (training_config.py:480-504).
- ✅ Round-trip test extended to assert smoke7 critical env settings.
- ✅ G6 invariant: 7/7 pass with smoke7 YAML.
- ✅ Eval-cmd override behaviorally verified: `reset_for_eval` pins
  cmd to the configured eval command across all seeds; training reset
  samples within the configured `[min_velocity, max_velocity]` range
  with proper zero_chance/deadzone distribution.
- ✅ Open-loop prior probe at vx ∈ {0.10, 0.15, 0.20, 0.25}: prior
  generates 1120-step trajectories at all bins; no immediate collapse.
- ✅ Phase 9D q_ref/eval alignment: `eval_velocity_cmd` and
  `loc_ref_offline_command_vx` both pin `0.20`, and the env builds an
  explicit one-bin library for that value so it cannot snap to a stale
  default-grid bin.

Decision rule after smoke7:

- **stride at eval (vx=0.20) ≥ 0.050 m** → TB-inspired command/DR hypothesis
  confirmed; promote to v0.20.2 (SysID hardening) and run ablation
  follow-ups (DR-only, multi-cmd-only) to isolate the load-bearing
  mechanism.
- **stride still parked below ~0.030 m** → TB-inspired command/DR pressure
  is not sufficient on WR.  Two candidates:
  - (a) prior is the actual blocker — return to v0.20.0-x prior
    surgery (extend prior to negative vx, re-evaluate prior fidelity
    under the closed-loop dynamics PPO actually faces).
  - (b) smoke compute budget exhausted before convergence — TB trains
    on 2× the episode length (1000 vs 500) and broader command range;
    extend M2 cap to ~300 iters and rerun before declaring failure.
- **mixed signal at intermediate stride (0.030-0.050)** → smoke7 is on
  the right track but needs more compute or true command-conditioned
  q_ref lookup before declaring the prior blocked.

What we will NOT do in smoke7:

- not enable pushes (TB walk default; defer to v0.20.5).
- not enable yaw cmd (defer to v0.20.4).
- not add stride-amplitude reward terms (TB doesn't have one; five
  smokes ruled out the reward space).
- not widen the residual bound past ±0.25 rad (G5 has ample headroom:
  smoke6-prep3 residuals 0.097 vs 0.20 cap).
- not relax G5.
- not extend the M2 compute budget yet (judge first at the standard
  150 iters; if intermediate result, then extend).

### `v0.20.2` SysID verification + command breadth + transfer hardening

Expectation:

- behavior works beyond one command point and the actuator model is trustworthy
  for hardware transfer

Quick visual check:

- zero command stands quietly
- low command walks slowly but visibly
- higher commands increase speed without immediate pitch dive

How this advances the end goal:

- converts a single-point sim result into a transferable command-conditioned
  walking capability

SysID is a hard gate for sim-to-real deployment. No hardware deployment
until the servo model is verified.

SysID tasks:

- collect hardware step-response data (step, hold, chirp captures) using
  `tools/sysid/run_capture.py`
- fit delay, backlash, frictionloss, armature, and effective output limits
  from hardware captures
- update `assets/v2/realism_profile_v0.19.1.json` with fitted values
  (or create a new versioned profile)
- apply fitted parameters to `wildrobot.xml` actuator model
- run sim-vs-real comparison via `tools/sim2real_eval/replay_compare.py`
- verify leg-joint replay RMSE stays within the accepted servo-model budget

SysID metric gate:

- sim-vs-real step-response RMSE per leg joint is within 15% of commanded
  amplitude
- delay estimate is stable across capture sessions
- backlash estimate is versioned and matches the MJCF backlash range
- the fitted servo model does not invalidate the ZMP prior library
  (re-run v0.20.0-C feasibility check with fitted parameters if the
  parameters change materially)

If SysID reveals that the servo model is wrong enough to invalidate the
prior library, regenerate the ZMP library with corrected parameters
before deploying.

Required diagnostics before sign-off:

- `debug/nominal_step_length_cmd_mean_m`
- `debug/nominal_foothold_x_cmd_mean_m`
- `debug/step_length_touchdown_mean_m`
- do not use the current `debug/step_length_m` alone because it is diluted by
  zeros on non-touchdown steps

Software stack:

- add regression plots for sim-vs-real mismatch
- add failure classification for slip, pitch fall, lateral fall, and missed
  touchdown

Training approach:

- broaden the command curriculum
- expand domain randomization
- tune the balance between reference tracking and PPO adaptation freedom

Sim2real:

- repeat hardware trials across multiple surfaces
- require repeatable transfer for forward walk and stop

Exit gate:

- zero-shot or near-zero-touch forward walking is reliable on hardware

### `v0.20.3` `ALIP` / Cassie-upgrade decision

Expectation:

- decide architecture upgrade only from measured bottlenecks, not preference

Quick visual check:

- if current policy is stable but capped in adaptation range, consider upgrade
- if motion quality is still prior-limited, keep improving the prior first

How this advances the end goal:

- avoids premature complexity and upgrades only when it increases real walking
  coverage

Decision rule:

- upgrade the prior toward `ALIP` if `ZMP` is the clear limiting factor
- upgrade the policy toward a Cassie-style broader-authority contract only if
  residual authority is the clear limiting factor

Do not promote either upgrade if the real blocker is:

- poor prior quality
- poor servo model fidelity
- weak imitation tracking
- reward ambiguity

### `v0.20.4` Yaw and Command Breadth

Software stack:

- extend the reference generator to in-place yaw and mild turn commands

Training approach:

- add yaw command tracking
- add command transitions and stop-go behavior

Sim2real:

- verify stable turn-in-place and stop behavior on hardware

Exit gate:

- walk, stop, and yaw are all repeatable on current hardware

### `v0.20.5` Recovery and Robustness

Software stack:

- extend the reference generator with disturbance-aware step adjustment
- upgrade to `ALIP` only if the simpler reference clearly saturates

Training approach:

- add moderate push recovery during standing and slow walking
- keep the same prior-guided PPO architecture

Sim2real:

- benchmark moderate push recovery
- track failure modes by class, not only overall success rate

Exit gate:

- WildRobot v2 can walk, stop, yaw, and recover from moderate disturbances

---

## Out Of Scope For The First Walking Branch

These are explicitly not the first implementation target:

- full omnidirectional locomotion
- mocap retargeting pipeline
- runtime-heavy MPC controller
- full-body style optimization
- major hardware changes

Those can be revisited after the first locomotion stack works on current v2
hardware.

---

## Decision Summary

WildRobot's locomotion pivot should use:

- offline command-indexed prior library
- `ZMP` first, `ALIP` later if needed
- ToddlerBot-style bounded residual PPO first
- Cassie-compatible preview interface for later upgrade
- imitation-dominant hybrid reward
- digital twin and SysID
- explicit sim2real replay and evaluation

That is the most practical ToddlerBot-style route for current WildRobot v2
hardware.

---

## Appendix A — ToddlerBot vs WildRobot v0.20.1 Threshold Audit

**Status:** captured 2026-04-19 before launching the v0.20.1 smoke. Driver:
align reward shape, termination, and observation contract to the
ToddlerBot recipe that has trained successfully on a similar small
position-controlled biped (`toddlerbot/locomotion/walk.gin`,
`toddlerbot/locomotion/mjx_config.py`,
`toddlerbot/locomotion/walk_env.py`).

The audit covers everything that materially shapes training convergence:
episode/termination, command sampling, reward weights, tracking-sigma
kernel widths, behavior-detection thresholds, action contract, and
domain randomization.  Each row cites WildRobot file/line and
ToddlerBot file/line so future drift is reviewable.

### A.1 Episode / termination thresholds

| Threshold | WildRobot v0.20.1 (pre-alignment) | ToddlerBot walk | Decision |
|---|---|---|---|
| `sim_dt` | 0.002 s (`ppo_walking_v0201_smoke.yaml:46`) | 0.005 s (`mjx_config.py:22`) | keep WR — finer solver step is safe |
| `ctrl_dt` | 0.020 s | 0.020 s (`n_frames=4 × 0.005`) | match |
| `n_substeps` | 10 | 4 | inherent from the dt pair; no change |
| `max_episode_steps` | 500 (10 s) | gin `episode_length` ≈ 1000 | keep WR for the smoke |
| `min_height` / `max_height` | 0.30 / 0.70 (`yaml:56`) | `healthy_z_range = [0.2, 1.0]` (`mjx_config.py:103`) | keep WR (tight v2 robot) |
| `max_pitch` / `max_roll` term | 0.8 rad each, **active** (`wildrobot_env.py:891`) | **none** — only height terminates (`mjx_env.py:1029`) | **change** — relax to height-only via `use_relaxed_termination: true` |
| `action_delay_steps` | 0 (`yaml:146`) | `n_steps_delay = 1` (`mjx_config.py:96`) | **change** — set to 1 |
| `action_filter_alpha` | 0.0 | active exp filter | keep WR=0 for the smoke (G6 invariant); revisit at v0.20.2 |
| `contact_threshold_force` | 1.0 N | 1.0 N (`mjx_config.py:95`) | match |
| Frame stack (history) | `PROPRIO_HISTORY_FRAMES = 3` (`env_info.py:47`) | `frame_stack = 15`, `c_frame_stack = 15` (`mjx_config.py:74`) | **change** — bump to 15 |

### A.2 Command / curriculum thresholds

| Item | WildRobot v0.20.1 | ToddlerBot | Decision |
|---|---|---|---|
| `min_velocity` / `max_velocity` | 0.0 / 0.30 (smoke7 command pressure); fixed q_ref/eval op point 0.265 | `command_range[5] = [-0.2, 0.3]` for vx | TB-inspired range on positive vx only; true command-conditioned q_ref lookup remains backlog |
| `resample_steps` | none | `resample_time = 3.0 s` ⇒ `resample_steps = 150` (`mjx_config.py:159`) | **add** — wire `resample_steps`, default 150; degenerate range still resamples to the same vx |
| `zero_chance` / `turn_chance` | none | 0.2 / 0.2 (`walk.gin:11`) | **add** as configurable, default to ToddlerBot values; smoke keeps single-cmd range |
| `deadzone` | none | `[0.05, 0.05, 0.2]` | **add** as configurable, default to ToddlerBot values |

### A.3 Reward weights — ToddlerBot vs WildRobot v0.20.1

ToddlerBot weights from `toddlerbot/locomotion/walk.gin`; reward fn
shapes from `toddlerbot/locomotion/{mjx_env,walk_env}.py`.  WR weights
from `training/configs/ppo_walking_v0201_smoke.yaml:240-258` and
`training/envs/wildrobot_env.py:715-877`.

> **Reward integration (load-bearing)**: ToddlerBot's PPO consumes
> `reward = sum(reward_dict.values()) * self.dt`
> (`toddlerbot/locomotion/mjx_env.py:1048`).  WR's `_aggregate_reward`
> applies the same `* self.dt` rescale uniformly across all weighted
> contributions before returning the per-term and total values.  This
> is what makes the `Term ↔ Term` weight comparison below
> apples-to-apples; if you remove the `* dt` from WR the published
> ToddlerBot weights are NOT directly portable here (event-based
> terms like `feet_air_time` get inflated ~50x at `ctrl_dt = 0.02`).
>
> **Survival semantics (load-bearing)**: ToddlerBot's
> `_reward_survival = -done` (`mjx_env.py:1897-1914`) at weight 10
> means the survival term contributes `0` while alive and `-10 * dt`
> on the terminating step.  WR mirrors this in `_aggregate_reward`
> (`wildrobot_env.py:_aggregate_reward`) — the `alive` weight is a
> failure penalty only, NOT a dense per-step bonus.  An earlier
> revision of this section was misleadingly summarised as "asymmetric
> +alive_w / -alive_w" semantics; that turns survival into a dense
> positive reward that biases PPO toward any long-lived behavior
> regardless of imitation tracking.  The current implementation is
> the strict ToddlerBot contract.

| Term (ToddlerBot ↔ WR name) | ToddlerBot | WR pre-alignment | WR post-alignment |
|---|---|---|---|
| `motor_pos` ↔ `ref_q_track` | 5.0 | 0.65 | 5.0 |
| `torso_quat` ↔ `ref_body_quat_track` | 5.0 | 0.10 | 5.0 |
| `torso_pos_xy` | 2.0 | — | **add 2.0** (match ToddlerBot `_reward_torso_pos_xy`) |
| `torso_roll` (soft, gate `[-0.1, 0.1]`) | 0.5 | — | **add 0.5** |
| `torso_pitch` (soft, gate `[-0.2, 0.2]`) | 0.5 | — | **add 0.5** |
| `lin_vel_xy` ↔ `cmd_forward_velocity_track` | 5.0 | 0.30 | 5.0 |
| `lin_vel_z` | 1.0 | — | **smoke6: 1.0** (vertical body bobbing — TB-aligned continuous phase signal; actual rotated to true body-local via `inv(root_quat)` matching TB; ref is finite-diff of prior `pelvis_pos[2]` per A.3 G2 — equals body-local for our yaw-stationary prior since pelvis quat is identity throughout) |
| `ang_vel_xy` | 2.0 | — | **smoke6: 2.0** (body roll/pitch rate — TB-aligned continuous phase signal; ref is zero for yaw-stationary prior; actual is `gyro_rad_s[:2]` which is already body-frame in MuJoCo, no rotation needed) |
| `ang_vel_z` | 5.0 | — | defer (smoke has no yaw cmd) |
| `feet_contact` ↔ `ref_contact_match` | 1.0 (boolean) | 0.0 (Gaussian, gated through smoke5) | **smoke6: 1.0 boolean** (TB form `sum(stance_mask == ref_stance_mask)` ∈ {0,1,2}; smoke3-5 used WR-specific Gaussian with σ=0.5 + doc/code mismatch on the kernel formula; switching to TB form resolves both and aligns strictly with TB) |
| `feet_air_time` | 500.0 | — | **add 500.0** (cmd-gated) |
| `feet_clearance` | 1.0 | — | **add 1.0** (cmd-gated) |
| `feet_distance` (`min=0.07`, `max=0.13`) | 1.0 | — | **add 1.0** |
| `feet_slip` ↔ `slip` | 0.05 | 0.0 (M1 hook) | **add 0.05** |
| `survival` ↔ `alive` | 10.0 (`-1 * done`, paid `-10 * dt` on the terminating step) | 0.05 (positive only) | **change to 10.0** with strict ToddlerBot `-done` semantics (no dense per-step bonus) |
| `motor_torque` ↔ `torque` | 0.1 (× neg-MSE) | -0.001 (× sum-sq) | keep — different scaling but similar effective penalty |
| `energy` | 0.01 | — | defer |
| `action_rate` | 1.0 (× neg-MSE) | -0.01 (× sum-sq) | **bump to -1.0** to match ToddlerBot grad scale |
| `collision` | 0.1 | — | defer (no self-collision shaping yet) |
| `feet_pos_track` (WR legacy diagnostic) | (env-specific) | 0.15 | remove from reward objective; keep raw debug metric only (`ref/feet_pos_track_raw`) |
| `joint_velocity` | — | -0.0005 | keep |

### A.4 Tracking-sigma / kernel widths

Both projects use the DeepMimic-style numerator-α form
`r = exp(-α · Σx²)`.

| Kernel | ToddlerBot α (`mjx_config.py:104`) | WR α (`training_runtime_config.py:747`) | Decision |
|---|---|---|---|
| `pos_tracking_sigma` (xy/z) | 200 | default 200; smoke override 90 (`torso_pos_xy_alpha`) | temporary divergence: keep reward alive through step 50 under WR open-loop drift; restore toward 200 after prior cleanup |
| `rot_tracking_sigma` | 20 | 20 (`ref_body_quat_alpha`) | match |
| `motor_pos` sigma (effective) | 1.0 | 1.0 (`ref_q_track_alpha`) | match |
| `lin_vel_tracking_sigma` | 200 | 4.0 (`cmd_forward_velocity_alpha`) | **change** — bump to 200 |
| `ang_vel_tracking_sigma` | 0.5 | — | n/a (no ang_vel reward yet) |
| `feet_pos` α (legacy world-frame) | 200 | debug-only fixed α=200 (`ref/feet_pos_track_raw`) | diagnostic only (not optimized); deleted from reward in smoke4 — kept as a logged signal so the world-frame metric trace stays comparable across smoke generations |
| `ref_foot_pos_body` (v0.20.2 body-frame, smoke5 only) | (n/a; TB has no foot-pos imitation) | dropped after smoke5 | Smoke5 verdict: term was DEAD throughout training (raw exp(-100·sum_sq) ≈ 0) because actual training-time per-foot err was 5-7× the open-loop probe prediction. Smoke3 also proved that lowering α to bring it alive doesn't help — the metric is phase-sensitive and pulls in the wrong direction at phase mismatch. Term and per-foot diagnostic err removed entirely in smoke6 (legacy `ref/feet_pos_err_l2` continues to log the equivalent world-frame measurement). |
| `ref_contact_match_sigma` | n/a (boolean) | 0.5 | WR-specific Gaussian; keep |

### A.5 Behavior-detection thresholds

| Signal | WR pre-alignment | ToddlerBot | Decision |
|---|---|---|---|
| Stance / contact mask | `force > 1.0 N` (`wildrobot_env.py:1242`) | `force > 1.0 N` (`mjx_config.py:95`) | match |
| `min_feet_y_dist` / `max` | — | 0.07 / 0.13 (`mjx_config.py:108`) | **add** (with feet_distance reward) |
| `torso_roll_range` (reward gate) | — | `[-0.1, 0.1]` | **add** (with torso_roll soft reward) |
| `torso_pitch_range` (reward gate) | — | `[-0.2, 0.2]` | **add** (with torso_pitch soft reward) |
| Cmd-magnitude gating on stepping rewards | inert (`step_need_*` defined but not consumed) | air_time/clearance/standstill all gate on `||cmd_obs|| > 1e-6` | apply on the new feet_* terms |
| Touchdown event | physical foot/floor contact transition (`view_zmp_in_mujoco.py:519`; env G4 in `wildrobot_env.py:1251`) | `stance_mask xor last_stance_mask` | equivalent |
| G4 success metrics | `forward_velocity ≥ 0.075`, `Evaluate/mean_episode_length ≥ 475`, `Evaluate/cmd_vs_achieved ≤ 0.075`, command-scaled step floor `max(0.030, 0.50 * vx_eval * cycle_time/2)` (Phase 9A: `>= 0.050 m`); smoke2 dropped `eval_walk/success_rate` because ToddlerBot has no success_rate concept | mean_reward / mean_episode_length / per-term `Evaluate/<term>` (`train_mjx.py:385-410`) | aligned via `Evaluate/*` namespace |
| G5 anti-exploit | `|residual_p50|_{hip_pitch,knee,ankle_roll} ≤ 0.20 rad`, `0.6 ≤ vx/cmd ≤ 1.5` | n/a | WR-specific; keep.  ankle_roll added post-merge alongside the residual-scale promotion below. |

### A.6 Action contract

| Item | WR pre-alignment | ToddlerBot | Decision |
|---|---|---|---|
| Form | `q = q_ref + clip(action) · scale_per_joint` | `q = default_q + action · 0.25` | semantically similar |
| Leg `action_scale` | ±0.50 rad | ±0.25 rad | **change** — clip legs to ±0.25 rad to match ToddlerBot and reduce G5 trip risk.  Post v20 ankle_roll merge: the leg list is `{hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll}` × L/R (10 joints).  ankle_roll uses ±0.25 to match the rest of the leg chain; range margin is the same as ankle_pitch (±0.25 / ±π/4 ≈ 32%) so headroom is identical. |
| Other-joint scale | ±0.20 rad | ±0.25 rad | keep ±0.20 (close enough; conservative) |

### A.7 Domain randomization — active in smoke7

WR smoke7 enables the TB-aligned DR subset after five reward-only smokes
failed to clear the shuffle gate.  ToddlerBot defaults include friction
`[0.4, 1.0]`, frictionloss `[0.8, 1.2]`, kp `[0.9, 1.1]`, mass
randomization, `add_domain_rand=True`, and `add_push=False` for walk.
WR matches the friction / frictionloss / kp ranges, keeps WR-specific
multiplicative mass scaling `[0.9, 1.1]`, and adds conservative IMU noise.
Pushes remain disabled because ToddlerBot walk also defaults
`add_push=False`.

### A.8 Required code changes (this audit)

The following changes land before the v0.20.1 smoke is launched.
Commit-by-commit:

1. **Config dataclass (`training_runtime_config.py`)**: add new
   `RewardWeightsConfig` fields for `feet_air_time`, `feet_clearance`,
   `feet_distance`, `torso_pitch_soft`, `torso_roll_soft`, plus
   `EnvConfig` fields for `min_feet_y_dist`, `max_feet_y_dist`,
   `torso_roll_range`, `torso_pitch_range`, `feet_air_time_threshold`,
   `cmd_resample_steps`, `cmd_zero_chance`, `cmd_turn_chance`,
   `cmd_deadzone`.  Defaults are inert (zero weight / no-op resample)
   so non-smoke configs are unaffected.
2. **Env code (`wildrobot_env.py`)**: extend `WildRobotInfo` with
   per-foot `feet_air_time` / `feet_air_dist` / `feet_height_init`
   and `cmd_resample_rng`; add `_reward_feet_air_time`,
   `_reward_feet_clearance`, `_reward_feet_distance`,
   `_reward_torso_pitch_soft`, `_reward_torso_roll_soft`; thread
   results through `_aggregate_reward`; relax termination per
   `use_relaxed_termination`; resample velocity_cmd every
   `cmd_resample_steps` ticks (degenerate when min == max).
3. **Frame stack (`env_info.py`, `policy_contract/spec.py`)**: bump
   `PROPRIO_HISTORY_FRAMES` 3 → 15.  Smoke checkpoint is not yet
   trained, so no migration burden.
4. **Smoke YAML**: rescale weights/sigmas per A.3/A.4, set
   `use_relaxed_termination: true`, `action_delay_steps: 1`, leg
   residual scale 0.50 → 0.25, enable the new feet_* / torso_* /
   slip terms with ToddlerBot-aligned weights.

After landing, the open-loop pre-smoke probe
(`tools/v0201_contact_alignment_probe.py`) and the env zero-action
test (`tests/test_v0201_env_zero_action.py`) must still pass — the
G6 invariant (zero residual ⇒ `target_q == q_ref`) is unchanged by
this audit.
