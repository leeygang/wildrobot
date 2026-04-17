# WildRobot Walking Training Plan

**Status:** Active locomotion plan for `v0.20.x`  
**Last updated:** 2026-04-15

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
prior reference / preview
    ->
task-space or joint-space nominal targets
    ->
policy(obs, prior, history) -> action or bounded correction
    ->
joint position targets
    ->
runtime / hardware
```

The policy remains the deployed artifact. The reference and IK layers are part
of the locomotion stack, but they must be strong enough that PPO is refining a
real walk rather than fabricating one.

---

## Prior Strategy

For the current reference architecture, single-source-of-truth rule,
support-first design direction, and detailed reference validation gates, see:

- [`reference_design.md`](/home/leeygang/projects/wildrobot/training/docs/reference_design.md)

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
  metrically on a servo-driven, no-ankle-roll platform.
- `ALIP` is the best medium-term direction because it has better upside for
  dynamic stepping, push adaptation, and broader command coverage.
- `Placo` is helpful for quickly generating or visualizing coherent whole-body
  motions, but it is a weaker single-source-of-truth prior on WildRobot because
  it is harder to prove dynamic validity and easier to drift into kinematic-only
  comfort.

### Recommended sequence

1. `v0.20.0`: build a `ZMP`-shaped prior family and validate it rigorously.
2. `v0.20.1`: expose prior preview and reference-conditioned observations to PPO.
3. `v0.20.2`: upgrade the prior toward `ALIP` if `ZMP` saturates on dynamic
   stepping, push recovery, or command breadth.
4. Use `Placo` selectively for:
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

---

## IK Role

IK is used to:

- convert swing-foot and pelvis targets into nominal joint targets
- produce a morphology-aware baseline stepping motion
- keep the policy search space narrow enough to train efficiently

IK is not used to:

- become the standalone locomotion controller
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
- foot contact or foot-switch state
- command forward speed
- command yaw rate
- gait phase as `sin/cos`
- stance foot ID
- desired next foothold in stance-foot frame
- desired swing-foot state
- desired pelvis height
- desired pelvis roll/pitch
- short observation history

If a privileged path is needed during training, it should stay bounded and
remain removable at deployment.

---

## Action Contract

The first locomotion action contract should be:

- policy outputs residual joint deltas around IK nominal motion

Form:

```text
q_target = q_ref_from_ik + delta_q_policy
```

This should remain home-centered and range-limited. The residual should be
smaller than the unconstrained joint spans used in earlier standing branches.

---

## Reward Design

The reward family should be task-space-first.

Primary terms:

- command velocity tracking
- pelvis orientation tracking
- pelvis height tracking
- swing-foot tracking
- foothold or touchdown tracking
- contact timing consistency
- survival

Secondary regularizers:

- slip penalty
- excessive impact penalty
- action smoothness
- residual magnitude penalty

De-emphasized terms:

- dense joint-space pose imitation
- large families of ad hoc standing-only penalties

The main idea is to reward:

- the robot doing the right locomotion task

not:

- the robot matching an arbitrary pose everywhere

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

---

## Software Stack Deliverables

### `control/`

Expected additions:

- `control/references/walking_ref_v1.py`
- `control/reduced_model/lipm.py`
- `control/reduced_model/dcm.py`
- `control/kinematics/leg_ik.py`
- `control/adapters/reference_to_joint_targets.py`
- `control/references/locomotion_contract.py` (Milestone 0 schema)
- `control/kinematics/interfaces.py` (Milestone 0 interface)

### `training/`

Expected additions:

- locomotion env wrapper that consumes command velocity and reference state
- PPO config for reference-guided residual locomotion
- fixed walking evaluation ladder
- rollout logging for reference tracking and touchdown quality

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

Status: **active**

Decision:

- start from a `ZMP`-shaped command-conditioned prior family
- design the contract so the same PPO stack can later consume an `ALIP` upgrade
- use `Placo` only for prototyping and visual authoring, not as the mainline
  validation truth

Why:

- `ZMP` is cheapest to build, easiest to debug, and easiest to validate on a
  small position-controlled biped
- `ALIP` has better long-term dynamic coverage, but should upgrade a proven
  prior rather than replace an unvalidated stack
- `Placo` is valuable for authoring and inspection, but on WildRobot it is too
  easy to get kinematically plausible motions that are not dynamically viable

Execution milestones:

1. `v0.20.0-A`: Prior contract freeze
2. `v0.20.0-B`: `ZMP` prior generator and viewer
3. `v0.20.0-C`: Prior-in-env nominal validation
4. `v0.20.0-D`: Prior library coverage and stop/resume validation
5. `v0.20.0-E`: PPO unblocks only after the prior gate clears

Exit rule:

- no `v0.20.1` PPO training until the `v0.20.0` prior gate passes

Primary design note:

- [`reference_design.md`](/home/leeygang/projects/wildrobot/training/docs/reference_design.md)

### `v0.20.0-A` Prior contract freeze

Tasks:

- define one prior interface:
  - command input
  - preview output
  - foothold targets
  - pelvis / COM targets
  - phase / mode outputs
- define one validation schema for reference-only and nominal-in-env probes
- freeze the first reference family as `ZMP`-shaped, with later `ALIP`
  compatibility in the contract

Visual validation:

- prior playback shows alternating step release
- stop command visibly settles into quiet standing
- no visible stepping-in-place when forward command is zero

Metric validation:

- command-conditioned outputs are deterministic and bounded
- preview trajectories are continuous
- no discontinuous target jumps above the configured rate limit

### `v0.20.0-B` `ZMP` prior generator and viewer

Tasks:

- implement the first `ZMP`-shaped prior family
- provide a viewer for:
  - prior-only playback
  - prior + IK playback
  - prior traces over command sweeps

Visual validation:

- at `cmd=0.15`, the feet clearly step out rather than shuffle
- pelvis stays bounded and does not dive forward
- left/right stepping is visibly alternating and repeatable
- stop/resume transitions are visible and non-chaotic

Metric validation:

- commanded foothold mean at `cmd=0.15` is `>= 0.045 m`
- commanded step period remains within `0.45-0.75 s`
- previewed COM and pelvis targets stay bounded and phase-consistent

### `v0.20.0-C` Prior-in-env nominal validation

Tasks:

- run the prior with IK / nominal targets in full MuJoCo env
- zero PPO residual or disable PPO entirely
- perform multi-seed validation at `cmd=0.15`

Visual validation:

- robot walks forward with clear step-out and stance switching
- mild wobble is acceptable, but no recurring forward-dive and no stand-shuffle
- failures, if any, are intermittent correction problems, not immediate
  gait-generation failures

Metric validation:

- at least `80%` of seeds survive `>= 400 / 500` steps
- no seed fails before `300` steps
- mean survival steps `>= 430`
- mean `env/forward_velocity >= 0.075 m/s`
- mean `tracking/cmd_vs_achieved_forward <= 0.075 m/s`
- touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`
- pitch `p95(|pitch|) <= 0.20 rad`
- roll `p95(|roll|) <= 0.12 rad`
- `term_pitch_frac <= 0.20`
- `term_roll_frac <= 0.20`

### `v0.20.0-D` Prior library coverage and stop/resume validation

Tasks:

- extend the prior across:
  - quiet stand
  - stop/resume
  - low-speed forward walk
  - in-place stepping
  - small yaw commands if available
- record command sweep diagnostics and visual summaries

Visual validation:

- zero command produces standing, not creeping
- low forward command produces slow but visible stepping
- resume from stop does not require a jump or stumble

Metric validation:

- zero-command drift stays `< 0.01 m/s`
- stop latency to quiet posture `< 1.0 s`
- resume-to-first-step latency `< 1.0 s`
- command sweep shows monotonic non-decreasing forward speed across the tested
  forward range

### `v0.20.0-E` PPO unblock decision

PPO is unblocked only if:

- all `v0.20.0-C` and `v0.20.0-D` gates pass
- the prior visually looks like a real walk rather than a balance exploit
- the remaining failures are plausibly correction-layer problems

### `v0.20.1` Prior-conditioned PPO

**Goal:** train PPO against a validated prior, not against missing gait
generation.

**Precondition:** `v0.20.0` prior gate passes.

Tasks:

- expose prior preview in observations
- keep PPO focused on robustness, tracking correction, and contact adaptation
- reject runs where PPO improvement comes mainly from inventing propulsion that
  the prior should own

Exit criteria:

- PPO improves stability and tracking relative to prior-only
- PPO does not materially change commanded step semantics
- command tracking improves without reopening `v0.19.5` lean-back / move-less
  exploit families

Required diagnostics before `v0.20.0` sign-off:

- `debug/nominal_step_length_cmd_mean_m`
- `debug/nominal_foothold_x_cmd_mean_m`
- `debug/step_length_touchdown_mean_m`
- do not use the current `debug/step_length_m` alone because it is diluted by
  zeros on non-touchdown steps

### `v0.20.2` Transfer Hardening (forward walk)

Software stack:

- add regression plots for sim-vs-real mismatch
- add failure classification for slip, pitch fall, lateral fall, and missed
  touchdown

Training approach:

- broaden the command curriculum
- expand domain randomization
- tune the balance between prior tracking and PPO adaptation freedom

Sim2real:

- repeat hardware trials across multiple surfaces
- require repeatable transfer for forward walk and stop

Exit gate:

- zero-shot or near-zero-touch forward walking is reliable on hardware

### `v0.20.3` Yaw and Command Breadth

Software stack:

- extend the reference generator to in-place yaw and mild turn commands

Training approach:

- add yaw command tracking
- add command transitions and stop-go behavior

Sim2real:

- verify stable turn-in-place and stop behavior on hardware

Exit gate:

- walk, stop, and yaw are all repeatable on current hardware

### `v0.20.4` Recovery and Robustness

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

- reduced-order reference generation
- IK nominal motion
- residual PPO policy
- digital twin and SysID
- explicit sim2real replay and evaluation

That is the most practical ToddlerBot-style route for current WildRobot v2
hardware.
