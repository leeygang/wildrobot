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
- Current active stage is now **`M2.5`: nominal reference validation**.
  - `v0.19.3` remains implemented, but it is blocked by insufficient confidence
    in the nominal prior under full env dynamics
  - active work is no longer PPO config tuning
  - active work is a fixed nominal-reference validation ladder and ablation
    program before any further M3 training

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

### `M2.5` Nominal Reference Validation

Purpose:

- validate that the `v0.19.2` nominal reference is good enough for M3
- localize the root cause when the nominal prior fails under full env dynamics
- block further PPO work until the nominal-only gate is cleared

Status:

- active stage
- `v0.19.3` integration exists, but it exposed that the nominal prior is not
  yet dynamically viable
- this is therefore a validation and root-cause stage, not a reward-tuning
  stage

Primary design note:

- [`reference_design.md`](/home/leeygang/projects/wildrobot/training/docs/reference_design.md)
  This is the canonical design note for the nominal-reference architecture,
  support-first `walking_ref_v2` direction, validation ladder, and `M2.5`
  execution milestones.

Current findings:

- `v0.19.3b` nominal-only probe established the blocker:
  - zero residual still pitch-fails early
  - `q_ref` is not currently viable under the full M3 env
- `v0.19.3c` and `v0.19.3d` improved semantics and braking diagnostics, but
  did not materially improve the nominal-only gate
- `v0.19.3e` improved the validation harness and isolated the current failure:
  - `no-dcm`: no meaningful effect
  - `zero-pelvis-pitch`: no meaningful effect
  - `freeze-swing-x`: reduces overspeed, but does not improve survival
  - `tight-stance`: improves survival, but worsens overspeed

Interpretation:

- the current blocker is not PPO optimization
- the current blocker is not primarily DCM or pelvis-pitch feedforward
- the dominant remaining issue appears to be coupled sagittal support and
  forward-progression timing under full env dynamics

Immediate next step:

- run the canonical `M2.5` multi-speed nominal-only sweep for `walking_ref_v2`
- inspect per-speed traces and the aggregate summary
- use that result to decide whether `walking_ref_v2` is:
  - too conservative and ready for one bounded release-threshold tuning pass
  - or still not dynamically valid and in need of a larger redesign

Canonical commands:

- baseline probe:
  - `uv run python training/eval/eval_loc_ref_probe.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 120 --num-envs 1 --residual-scale 0.0`
- canonical multi-speed sweep:
  - `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`
- next ablation:
  - `uv run python training/eval/eval_loc_ref_probe.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 120 --num-envs 1 --residual-scale 0.0 --ablation slow-phase`

### `v0.19.3` Walking Policy v1

Software stack:

- add reference state into the locomotion env
- add history stacking and one-step delay support
- add policy export and offline replay tools

Training approach:

- train reference-guided PPO for forward walking and stop
- use residual joint targets around IK nominal motion
- reward task-space tracking first

Sim2real:

- begin guarded hardware tests
- compare action statistics and reference-tracking mismatch between sim and real

Exit gate:

- stable forward walking in simulation
- first hardware rollouts show repeatable stepping and controlled stopping

Result:

- implemented end-to-end, but the first screening run did not clear the gate
- the initial config overpaid uncontrolled forward motion relative to nominal
  reference tracking
- no checkpoint from the first `v0.19.3` run should be treated as a promotion
  candidate

### `v0.19.3a` Walking Policy v1 Recovery Pass

Software stack:

- keep the `v0.19.3` code path unchanged
- apply a bounded config-only recovery pass in
  `training/configs/ppo_walking_v0193a.yaml`

Training approach:

- reduce `loc_ref_residual_scale` so PPO cannot overpower nominal motion as
  easily
- reduce maximum commanded forward speed for the first recovery screen
- increase `m3_swing_foot_tracking` and `m3_foothold_consistency` so the policy
  is paid for following the reference instead of free-running forward
- strengthen slip, impact, and residual-magnitude penalties
- set generic `clearance` reward to zero for this recovery run

Sim2real:

- no hardware walking promotion from `v0.19.3`
- regain stable simulation behavior first, then re-open guarded hardware tests

Exit gate:

- `term_pitch_frac` drops materially below the failed `v0.19.3` regime
- clean eval episode length rises well above `40` steps
- achieved forward speed tracks command instead of overshooting it
- `m3_swing_foot_tracking` and `m3_foothold_consistency` become meaningfully
  active
- slip falls sharply relative to the failed `v0.19.3` run

Result:

- `v0.19.3a` did not clear the gate
- subsequent `v0.19.3b` through `v0.19.3e` debug branches show that the
  blocker is the nominal prior, not PPO optimization
- from this point on, M3 work is gated by the nominal reference validation
  ladder below

### `M2.5` Validation Ladder

Purpose:

- prove the nominal reference is good enough to unblock `v0.19.3`
- localize the root cause when it is not
- avoid wasting PPO runs on a broken prior

This validation ladder defines the active `M2.5` stage and applies to all
remaining `v0.19.3x` work.

#### Level 1: Reference Generator Correctness

Goal:

- verify `walking_ref_v1` is internally consistent before full-env rollout

Checks:

- phase stays in `[0, 1]`
- stance alternates correctly
- step length stays within configured bounds
- lateral placement stays within bounds
- swing trajectory is continuous and bounded
- pelvis targets are bounded and smooth

Typical validation:

- unit tests around `control/references/walking_ref_v1.py`
- bounded test cases for phase progression, foothold limits, and swing shape

#### Level 2: IK And Nominal Target Correctness

Goal:

- verify `q_ref` is kinematically coherent and not already bad in joint space

Checks:

- reachability stays high
- no systematic clipping or limit hugging
- left/right symmetry remains sane
- unsolved joints stay in safe home-centered defaults
- `q_ref` evolves smoothly over phase
- stance/swing geometry maps to plausible sagittal targets

Typical validation:

- integration tests around the env-side nominal target path
- explicit checks on `tracking/loc_ref_left_reachable`,
  `tracking/loc_ref_right_reachable`, and nominal target magnitude

#### Level 3: Nominal-Only Full-Env Viability

Goal:

- verify the nominal prior survives full actuator/contact dynamics without PPO

Canonical probe command:

- `uv run python training/eval/eval_loc_ref_probe.py --config training/configs/ppo_walking_v0193a.yaml --forward-cmd 0.10 --horizon 120 --num-envs 1 --residual-scale 0.0`

Canonical multi-speed sweep:

- `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`

Required outputs:

- `done_step`
- dominant termination
- `forward_velocity_mean`
- `forward_velocity_last`
- `tracking_nominal_q_abs_mean`
- `tracking_loc_ref_left_reachable`
- `tracking_loc_ref_right_reachable`
- `tracking_loc_ref_phase_progress_*`
- `reward_m3_swing_foot_tracking`
- `reward_m3_foothold_consistency`
- `debug_loc_ref_nominal_vs_applied_q_l1_mean`

Current status:

- this level is still failing
- PPO must not resume until this level improves materially

#### Level 4: Channel-Isolation Ablations

Goal:

- identify which nominal channel causes the forward pitch-dive

Required ablation families:

- freeze or heavily damp swing-x progression
- zero or heavily damp nominal pelvis pitch
- remove or greatly reduce DCM contribution
- slow phase progression
- increase stance/support conservatism

Rules:

- run one ablation at a time
- always rerun the same canonical nominal-only probe
- compare against a fixed baseline rather than reward totals

Current ablation status:

- `no-dcm`: no meaningful effect
- `zero-pelvis-pitch`: no meaningful effect
- `freeze-swing-x`: lowers overspeed, but does not improve survival
- `tight-stance`: improves survival, but worsens overspeed
- `slow-phase`: lowers overspeed slightly, but does not improve survival

#### Support-First And Dynamically Valid Checklist

Purpose:

- define what the nominal prior must do before PPO is allowed to resume
- make the next nominal-layer work measurable and repeatable

`Support-first` means:

- stance/support geometry is prioritized before forward progression
- forward swing advance slows or freezes when support is already failing
- nominal progression is conditional on support state, not just phase time

`Dynamically valid` means:

- the nominal prior survives the full training env dynamics without PPO rescue
- contacts, actuator realization, and body motion remain in the same regime as
  the commanded task

Required support-first checks:

- root pitch remains bounded enough that support can still be recovered
- root pitch rate remains bounded enough that the nominal prior is not
  accelerating into collapse
- support-gate activation correlates with reduced swing-x progression and/or
  reduced phase advance when instability is present
- forward progression slows or stops when overspeed and pitch indicate support
  is already failing

Required dynamic-validity checks:

- nominal-only rollout survives materially longer than the current low-30-step
  regime
- immediate `term/pitch` collapse is gone
- achieved forward speed stays in the same regime as command
- swing-foot tracking is clearly above numerical noise
- foothold consistency is clearly above the current near-zero regime
- nominal-vs-applied target gap remains small enough to explain realized motion

Required trace-level checks:

- inspect one fixed probe trace for:
  - phase progress
  - stance foot
  - root pitch and pitch rate
  - forward velocity
  - swing-x target vs actual
  - foothold target vs realized touchdown
  - support-gate activity
  - nominal `q_ref` vs applied target gap
- expected interpretation:
  - support is established first
  - then controlled progression is permitted
  - not the other way around

Required robustness checks after one-point success:

- rerun nominal-only probes at:
  - `0.06 m/s`
  - `0.10 m/s`
  - `0.14 m/s`
- optionally add one or two mild friction variations
- do not treat the prior as validated if it only works at one exact command

Decision rule:

- if the prior passes the support-first and dynamic-validity checks above,
  resume `v0.19.3` PPO work
- if it fails after one more bounded nominal redesign pass, stop local tuning
  and redesign the nominal layer more substantially

#### Level 5: Robustness Sweep

Goal:

- verify the nominal prior is not only working at one exact operating point

Minimum sweep after Level 3 improves:

- forward command `0.06`
- forward command `0.10`
- forward command `0.14`
- multiple seeds if runtime cost is acceptable

Preferred command:

- `JAX_PLATFORMS=cpu uv run python training/eval/run_m25_v2_probe_sweep.py`

Optional:

- one or two mild friction variations

#### Resume-PPO Gate

PPO training under `v0.19.3` remains blocked until all of these are true:

- nominal-only survives materially longer than the current low-30-step failure
  regime
- immediate pitch-dive is gone
- achieved forward speed is in the same regime as command, not large overspeed
- swing tracking is clearly above numerical noise
- foothold consistency is no longer near zero
- nominal-vs-applied target gap is small enough to explain the realized motion

If this ladder still fails after bounded channel-isolation work, the next step
is not more `v0.19.3x` tuning. It is a larger redesign of the nominal layer
before resuming M3.

### `v0.19.4` Transfer Hardening

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

### `v0.19.5` Yaw and Command Breadth

Software stack:

- extend the reference generator to in-place yaw and mild turn commands

Training approach:

- add yaw command tracking
- add command transitions and stop-go behavior

Sim2real:

- verify stable turn-in-place and stop behavior on hardware

Exit gate:

- walk, stop, and yaw are all repeatable on current hardware

### `v0.19.6` Recovery and Robustness

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
