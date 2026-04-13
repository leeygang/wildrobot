# WildRobot Walking Training Plan

**Status:** Active locomotion plan for `v0.19.x`  
**Last updated:** 2026-04-03

---

## Purpose

This document defines the concrete walking plan after the pivot toward a
ToddlerBot-style direction on current WildRobot v2 hardware.

The key architecture decision is:

- the deployed runtime artifact remains a learned policy
- reduced-order walking references and IK provide nominal motion
- RL learns residual robustness and adaptation around that nominal motion

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

- reward shaping + bounded teacher cues

to:

- reference-guided RL + IK nominal motion + residual policy

The source of prior becomes:

- reduced-order locomotion dynamics, not mocap
- task-space references, not a hardcoded joint trajectory
- morphology-aware nominal motion, not a copied ToddlerBot gait

---

## Mainline Runtime Contract

The intended runtime contract is:

```text
command
    ->
walking reference
    ->
IK nominal motion
    ->
q_ref
    ->
policy(obs, ref, history) -> delta_q
    ->
q_target = q_ref + delta_q
    ->
joint position targets
    ->
runtime / hardware
```

The policy remains the deployed artifact. The reference and IK layers are part
of the locomotion stack, but not the final controller product by themselves.

---

## Reference Strategy

For the current nominal-reference architecture, single-source-of-truth rule,
support-first design direction, and `M2.5` validation gates, see:

- [`reference_design.md`](/home/leeygang/projects/wildrobot/training/docs/reference_design.md)

### Reference v1: `LIPM / DCM / capture-point`

The first reference generator should use:

- constant-height `LIPM`
- `DCM / capture-point` step placement logic
- fixed or slowly varying step timing
- swing-foot trajectory generation
- pelvis height target
- small pelvis roll/pitch targets for weight shift

This is chosen because it is:

- easier to implement than full ZMP preview control
- more directly useful for WildRobot's balance and step-placement problem
- easier to adapt to a no-ankle-roll morphology

### Reference v2: `ALIP` upgrade if needed

If the first reference saturates because hip/trunk momentum effects matter
strongly, the next upgrade is:

- `ALIP`-inspired task-space reference generation

This keeps the same policy architecture while improving the reduced-order model.

### What We Are Not Using First

- full ZMP preview walking as the first implementation
- full runtime MPC as the locomotion controller
- mocap retargeting as the initial prior

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
- Current active stage is now **`v0.19.5`: Walking PPO v2 (residual policy)**.
  - the nominal reference is validated and walks on its own
  - PPO should now improve balance, step quality, and lateral stability

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

### `v0.19.4` Nominal Reference Validation & Walking Reference

Purpose:

- validate that the nominal reference is good enough to unblock walking PPO
- localize the root cause when the nominal prior fails under full env dynamics
- block further PPO work until the nominal-only gate is cleared

Status: **complete** (April 2026)

Key results:

- ctrl ordering bug found and fixed (`CtrlOrderMapper`)
- support posture B validated stable (200/200 steps)
- startup transition A→B validated (smooth qpos interpolation)
- DCM COM trajectory implemented (stance-leg forward drive)
- nominal walking probe: 461/500 steps, 15 gait cycles, +0.038 m/s

Primary design note:

- [`reference_design.md`](/home/leeygang/projects/wildrobot/training/docs/reference_design.md)

### `v0.19.5` Walking PPO v2 — Residual Policy

**Goal:** train residual PPO policy on top of the walking reference to
improve balance, lateral stability, and foot tracking.

Status: **next**

Tasks:

1. Verify env reset + auto-reset work with COM trajectory state
2. Run a short PPO training (1000 iterations) to check learning signal
3. Check that PPO residuals improve stepping (larger steps, better tracking)

Exit criteria:

- PPO training runs without errors
- `eval_clean/success_rate` > 0% within 1000 iterations
- Step length improves with training (residuals help foot tracking)
- Forward speed tracks command better than nominal-only
- Mean step length > 5cm (current baseline: 2.5cm, metric: `debug/step_length_m`)
- Pitch oscillation < ±0.15 rad (current baseline: ±0.17 rad)

### `v0.19.6` Transfer Hardening (forward walk)

Software stack:

- add regression plots for sim-vs-real mismatch
- add failure classification for slip, pitch fall, lateral fall, and missed
  touchdown

Training approach:

- broaden the command curriculum
- expand domain randomization
- tune the balance between nominal reference and residual freedom

Sim2real:

- repeat hardware trials across multiple surfaces
- require repeatable transfer for forward walk and stop

Exit gate:

- zero-shot or near-zero-touch forward walking is reliable on hardware

### `v0.19.7` Yaw and Command Breadth

Software stack:

- extend the reference generator to in-place yaw and mild turn commands

Training approach:

- add yaw command tracking
- add command transitions and stop-go behavior

Sim2real:

- verify stable turn-in-place and stop behavior on hardware

Exit gate:

- walk, stop, and yaw are all repeatable on current hardware

### `v0.19.8` Recovery and Robustness

Software stack:

- extend the reference generator with disturbance-aware step adjustment
- upgrade to `ALIP` only if the simpler reference clearly saturates

Training approach:

- add moderate push recovery during standing and slow walking
- keep the same reference-guided residual-policy architecture

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
