# Training Plan

**Version:** v0.16.3
**Status:** Active
**Last Updated:** 2026-03-20

## Overview

This document is the execution plan for WildRobot locomotion training.

The current conclusion is:
- Stage `1a` delivered the control/runtime foundation.
- Stage `1b` PPO-only walking is exhausted as the main branch.
- Stage `1c` is the active branch: bootstrap a robot-native gait with a MuJoCo motion-tracking teacher, then transfer it to a student locomotion policy.

The key shift is methodological:
- stop asking PPO-from-scratch to invent stable biped gait structure from task rewards alone
- use a reference-guided teacher to create a stable gait basin first

## Current Stage

**Current active branch:** Stage `1c` (`v0.16.x`)

**Current objective:** verify the upstream GMR retarget itself before any more teacher training, then either fix/regenerate the source motion or export a corrected teacher clip.

**Current status:**
- The Stage `1c` pipeline is still structurally sound enough to keep:
  - reference clip loading works
  - loop-unwrapped teacher tracking works
  - contact-template phase estimation works
  - teacher rollout export for `v0.16.1` is real
- But the current blocker has moved upstream again:
  - the `real retarget v1` source motion is not verified as a physically valid WildRobot walk
  - downstream teacher results cannot be trusted until the source retarget passes a MuJoCo verification gate
- `v0.16.1` remains blocked.

**Evidence:**
- [`offline-run-20260316_202553-04hb74z3`](/home/leeygang/projects/wildrobot/training/wandb/offline-run-20260316_202553-04hb74z3):
  - stable, but converged to standing
  - `tracking/root_tracking_error ≈ 0.52`
  - `reward/teacher_foot_position ≈ 0.002`
  - `env/forward_velocity ≈ -0.002`
- [`offline-run-20260316_214827-dxmrxuc2`](/home/leeygang/projects/wildrobot/training/wandb/offline-run-20260316_214827-dxmrxuc2):
  - materially better tracking
  - `tracking/root_tracking_error ≈ 0.20`
  - `reward/teacher_foot_position ≈ 0.019`
  - still `env/forward_velocity ≈ -0.004`

So the teacher moved from “stand anywhere” to “quasi-static reference matching,” but not to walking.

Additional retarget verification on March 20, 2026 showed the deeper issue:
- [`walking_slow01_gmr_report.json`](/home/leeygang/projects/wildrobot/training/reference_motion/verification/walking_slow01_gmr_report.json)
  - `verdict: fail`
  - failed checks:
    - `kinematic_grounding`
    - `full_contact_support`
    - `capped_dynamic_tracking`
  - support foot median height in kinematic replay was still about `+1.4 cm`
  - full replay load support was only about `67% mg`
  - capped replay failed by frame `82/305`

That means the current source retarget itself is not yet a valid walking target.

---

## Stage 1a: Control Abstraction Layer - Archived

**Goal:** decouple policy training from MuJoCo actuator primitives and fix action semantics.

### What Stage 1a Delivered

- normalized policy action semantics
- actuator/joint metadata as a single source of truth
- cleaner symmetry handling
- a training/export/runtime contract that later stages can build on

### Why It Matters

Without Stage `1a`, later locomotion debugging would have been ambiguous because action semantics, actuator order, and joint sign conventions were not stable enough.

### Stage 1a Status

Complete enough for the locomotion program. No further major work planned here unless a later stage exposes a low-level control bug.

---

## Stage 1b: PPO-Only Robot-Native Walking - Exhausted

**Goal:** learn stable nominal walking from task rewards alone, with no reference motion and no teacher.

### Final Verdict

Stage `1b` is exhausted as the primary branch.

The branch proved that WildRobot can generate forward speed in simulation, but it did **not** produce a stable walking basin. Across `v0.15.10` to `v0.15.12`, the policy repeatedly converged to the same failure mode:
- forward speed emerges
- pitch failure rises
- eval-clean survival collapses

This means PPO-from-scratch plus local reward shaping was sufficient to find translation, but not sufficient to reliably discover and hold a stable biped gait.

### Condensed History

Use this as the short diagnostic record.

#### Early `v0.15.x`

- `v0.15.1` / `v0.15.2`
  - mostly standing or near-standing
  - warm-started standing basin remained too attractive
- `v0.15.3` / `v0.15.4`
  - fresh-start walking increased speed slightly
  - but mostly through leaning/pitching harder
  - clear posture-exploit basin appeared

#### Mid `v0.15.x`

- `v0.15.5`
  - conservative reset
  - improved stability but still no useful propulsion
- `v0.15.6`
  - contact-gated velocity and step-oriented rewards induced stable stepping
  - but converged to stepping in place / no propulsion
- `v0.15.7`
  - propulsion rebalance
  - still stepping without useful translation
- `v0.15.8`
  - added `wr_obs_v3` gait clock
  - improved timing structure, but still no real propulsion breakthrough
- `v0.15.9`
  - explicit per-step propulsion rewards
  - structurally better, but too hard/sparse early and still did not dominate training

#### Late `v0.15.x`

- `v0.15.10`
  - first true speed breakthrough
  - but speed came mainly from pitch-driven collapse
- `v0.15.11`
  - posture-aware gating delayed collapse
  - did not change the terminal basin
- `v0.15.12`
  - contact-driven `step_progress` improved early structured behavior
  - still ended in the same pitch-collapse outcome

### What Stage 1b Actually Taught Us

These are the parts worth keeping.

1. PPO-only walking is not totally broken.
   - The robot can step.
   - The robot can produce speed.
   - Reward structure matters a lot.

2. But PPO-only walking is not enough here.
   - It does not reliably discover a stable biped gait basin in this setup.

3. Useful reward ideas from `1b` that may still help later:
   - zero-baseline forward reward for commanded walking
   - explicit pitch-rate shaping
   - command curriculum
   - structured step-length / step-progress terms
   - phase-aware observations (`wr_obs_v3`)

4. Less useful directions to keep spending time on as the main path:
   - more small reward-coefficient edits
   - more short PPO-only retries with the same overall structure
   - trying to “wait longer” for a branch that is already in pitch-collapse

### Why We Are Leaving Stage 1b

The branch no longer fails because of an obvious missing local lever.
It fails because gait structure itself is under-specified relative to the optimization incentives.

That is the signal to move to a teacher-guided bootstrap.

---

## Stage 1c: MuJoCo Motion-Tracking Teacher Bootstrap - Current

**Goal:** create a stable robot-native gait basin first, then transfer it into a command-conditioned locomotion student.

### Why This Stage Exists

Stage `1b` proved the policy can find speed, but not a stable gait basin.

The right next step is to follow mature locomotion practice:
- retarget a simple walk reference to the robot
- train a phase-conditioned motion-tracking teacher
- use the teacher to bootstrap a student locomotion policy

This is closer to:
- DeepMimic
- HumanoidVerse
- ASAP

than continuing local reward surgery.

### Current Gate Before More Teacher Training

Before exporting or training on any new teacher clip:

1. Verify the upstream GMR/WildRobot motion in MuJoCo.
2. Reject any motion that floats, penetrates badly, or cannot sustain support in kinematic replay.
3. Only then convert/export it into the teacher reference-motion contract.

The new verification tools are:
- [`verify_gmr_retarget.py`](/home/leeygang/projects/wildrobot/scripts/verify_gmr_retarget.py)
- [`verify_reference_retarget.py`](/home/leeygang/projects/wildrobot/scripts/verify_reference_retarget.py)
- [`debug_gmr_physics.py`](/home/leeygang/projects/wildrobot/training/data/debug_gmr_physics.py)

This gate now precedes the next `v0.16.0` teacher run.

### External Patterns To Follow

- DeepMimic: reference-motion imitation in physics before broader task control
  - Project: https://xbpeng.github.io/projects/DeepMimic/index.html
  - Code: https://github.com/xbpeng/DeepMimic
- HumanoidVerse: humanoid locomotion recipes with curriculum and large-scale training
  - https://github.com/LeCAR-Lab/HumanoidVerse
- ASAP: retargeting + phase-based motion tracking + later handoff/fine-tune
  - https://github.com/LeCAR-Lab/ASAP

### Scope

Stage `1c` stays:
- MuJoCo-first
- WildRobot-first
- flat-terrain first

It does **not** mean:
- direct H1 checkpoint reuse
- moving the training stack to Isaac Lab
- broad robustness before nominal gait exists

### Stage 1c Architecture

1. Reference clip
   - start with one nominal forward-walk clip
   - robot-native or retargeted to robot-native form

2. Motion-tracking teacher
   - phase-conditioned
   - flat terrain only
   - track pose, velocity, foot timing, and contact timing

3. Teacher-to-student handoff
   - collect teacher rollouts into student-format data
   - behavior-clone or distill a student
   - fine-tune the student on narrow command-conditioned locomotion

4. Return to broader locomotion
   - only after the student preserves the gait basin

---

## `v0.16.0`: Teacher Bootstrap

### Objective

Create the minimum viable teacher branch:
- one nominal walk clip
- one reference-motion format
- one teacher config
- one teacher reward/observation path
- one short proof-of-life teacher run

### Reference Motion Data Format

Use one `.npz` clip plus one `.json` metadata file.

#### Clip file

Path example:
- `training/reference_motion/wildrobot_walk_forward_v001.npz`

Required arrays:
- `dt: ()`
- `phase: (T,)`
- `root_pos: (T, 3)`
- `root_quat: (T, 4)` in `wxyz`
- `root_lin_vel: (T, 3)`
- `root_ang_vel: (T, 3)`
- `joint_pos: (T, A)`
- `joint_vel: (T, A)`
- `left_foot_pos: (T, 3)`
- `right_foot_pos: (T, 3)`
- `left_foot_contact: (T,)`
- `right_foot_contact: (T,)`

Optional arrays:
- `com_pos: (T, 3)`
- `com_vel: (T, 3)`
- `left_foot_quat: (T, 4)`
- `right_foot_quat: (T, 4)`

#### Metadata file

Path example:
- `training/reference_motion/wildrobot_walk_forward_v001.json`

Required fields:
- `name`
- `source_name`
- `source_format`
- `frame_count`
- `dt`
- `actuator_names`
- `loop_mode`
- `phase_mode`

Recommended fields:
- `retarget_version`
- `notes`
- `nominal_forward_velocity_mps`
- `pelvis_height_m`
- `stance_width_m`

#### Loader contract

Use a typed structure such as `ReferenceMotionClip` with:
- `name`
- `dt`
- `phase`
- `root_pos`
- `root_quat`
- `root_lin_vel`
- `root_ang_vel`
- `joint_pos`
- `joint_vel`
- `left_foot_pos`
- `right_foot_pos`
- `left_foot_contact`
- `right_foot_contact`
- `metadata`

Loader must validate:
- consistent frame count
- consistent actuator dimension
- quaternion shape
- actuator-name alignment with WildRobot config

### Teacher Reward / Observation Contract

Teacher observations should include:
- phase encoding
- teacher target features needed for control
- robot state features needed for feedback

Teacher reward should include:
- root pose tracking
- root velocity tracking
- joint pose tracking
- foot position tracking
- contact timing agreement
- upright stability
- effort / smoothness penalties

### `v0.16.0` Current Status

Teacher pipeline fixes are in place and two proof-of-life runs have already answered the next question.

Current conclusion:
- `v0.16.0` teacher infrastructure is viable enough to continue using
- the bootstrap procedural clip is not a good enough locomotion target
- the next `v0.16.0` task is no longer reward surgery or teacher-plumbing fixes
- the next `v0.16.0` task is `real retarget v1`

### `v0.16.0` Immediate Next Milestone: `real retarget v1`

#### Objective

Replace the placeholder procedural clip with the first real WildRobot-feasible retargeted nominal walk clip.

The target does not need final production quality.
It needs to be:
- clearly walking forward
- kinematically feasible for WildRobot
- smooth enough that teacher PPO can track it
- informative enough that standing still is no longer a good local optimum

#### Source Motion Assumption

Use one simple nominal forward-walk source clip only.

Acceptable sources:
- a local AMASS / GMR / project motion clip if already available
- a local robot-native source animation
- a manually authored fallback only if no suitable local source exists

Do not block on a motion library or multi-clip system.
One good nominal walk is enough for `real retarget v1`.

#### Retargeting Priorities

Retarget task-space gait structure first.

Primary targets:
- root/pelvis forward progression
- pelvis height
- torso orientation
- left/right foot world positions
- stance width
- forward step length
- contact timing
- knee bend direction and minimum flexion

Secondary targets:
- exact upper-body style
- perfect joint-angle imitation of the source

#### Minimal Solver Shape

Implement a lightweight offline retargeting pass in `training/reference_motion/retarget.py`:
- per-frame optimization or IK solve
- first acceptable implementation:
  - SciPy `least_squares`
  - WildRobot-specific mapping, not a giant framework

Optimization priorities:
- match foot positions
- keep pelvis/root feasible
- enforce joint limits
- preserve smoothness frame-to-frame
- keep stance width and knee configuration valid

#### Output Contract

Keep the existing reference motion file format.
Export a new clip version, for example:
- `training/reference_motion/wildrobot_walk_forward_v002.npz`
- `training/reference_motion/wildrobot_walk_forward_v002.json`

Required `.npz` arrays remain:
- `dt`
- `phase`
- `root_pos`
- `root_quat`
- `root_lin_vel`
- `root_ang_vel`
- `joint_pos`
- `joint_vel`
- `left_foot_pos`
- `right_foot_pos`
- `left_foot_contact`
- `right_foot_contact`

Metadata should keep the current fields and update:
- `source_name`
- `source_format`
- `retarget_version`
- `nominal_forward_velocity_mps`
- `notes`

#### `real retarget v1` (Implemented)

Source clip used:
- `training/data/gmr/walking_slow01.pkl`

Implemented assumptions:
- source is a local 8-DOF lower-body GMR clip (`left/right hip pitch/roll, knee, ankle`) with floating-base root motion
- source `root_rot` is consumed as `xyzw` and converted to `wxyz` for MuJoCo/reference export consistency
- retarget path is WildRobot-specific and intentionally minimal
- one nominal gait cycle is selected from the source, aligned to world +x progression, then solved frame-by-frame
- per-frame solve uses SciPy `least_squares` with:
  - foot world-position tracking
  - joint-limit bounds
  - minimum knee flexion regularization
  - frame-to-frame smoothness regularization

Exported clip:
- `training/reference_motion/wildrobot_walk_forward_v002.npz`
- `training/reference_motion/wildrobot_walk_forward_v002.json`

Known limitations in `real retarget v1`:
- only one source clip and one nominal gait cycle are used (no multi-clip coverage yet)
- because the source clip is already WildRobot-retargeted, the v1 solver is mostly a feasibility/adaptation pass (not a full de novo retarget reconstruction)
- upper body is mostly neutral (style fidelity is not a target in this step)
- stance width is conservative and near the lower feasible bound
- this is a kinematic nominal clip; dynamic realism quality is still gated by teacher PPO results

#### Validation Before Training

Before rerunning teacher training, verify:
- joint positions stay inside limits
- joint velocities are finite and not excessively spiky
- pelvis height stays in nominal range
- foot trajectories show alternating swing/stance
- contact labels match foot height/phase well enough
- nominal forward velocity is positive and nontrivial

#### What Success Looks Like

`real retarget v1` is good enough if the next teacher run shows:
- positive `env/forward_velocity`
- active `reward/teacher_foot_position`
- improving `tracking/root_tracking_error`
- stable or improving `tracking/contact_timing_agreement`
- no immediate return to standing or pitch-collapse

### `v0.16.0` Exit Criteria

`v0.16.0` is successful if:
- the teacher can track a nominal walk without immediate collapse
- stepping is visibly periodic
- forward translation is achieved without `term_pitch_frac` exploding
- teacher tracking metrics improve over training

This stage does **not** require:
- wide command coverage
- pushes
- rough terrain
- natural human style

### `v0.16.0` Decision Rule

- If the teacher cannot stabilize even one nominal walk clip, fix the clip or tracking reward before building a student.
- That question is now answered more precisely:
  - teacher mechanics are adequate
  - clip quality is the next bottleneck
- Do not go back to PPO-only walking unless `1c` fails for a clear infrastructure reason.

---

## `v0.16.1`: Teacher-to-Student Handoff

### Objective

Convert the stable teacher gait into a command-conditioned locomotion student without losing the gait basin.

### Teacher Rollout Dataset Format

Use sharded `.npz` files plus `metadata.json`.

#### Dataset shard

Path example:
- `training/data/teacher_rollouts/wildrobot_walk_teacher_v001/shard_0000.npz`

Required arrays:
- `obs: (N, O)`
- `actions: (N, A)`
- `phase: (N, P)`
- `velocity_cmd: (N,)`
- `forward_velocity: (N,)`
- `done: (N,)`
- `left_foot_contact: (N,)`
- `right_foot_contact: (N,)`

Strongly recommended arrays:
- `teacher_action_mean: (N, A)`
- `teacher_action_std: (N, A)`
- `teacher_value: (N,)`
- `root_pitch: (N,)`
- `root_pitch_rate: (N,)`

#### Metadata

Required fields:
- `teacher_checkpoint`
- `teacher_config`
- `observation_layout`
- `action_dim`
- `num_shards`
- `num_samples`
- `phase_dim`
- `velocity_cmd_range`

### Student Bootstrap Contract

Minimum first path:
1. collect teacher rollouts
2. behavior-clone `obs -> actions`
3. save a PPO-compatible warm-start checkpoint
4. fine-tune the student on narrow locomotion commands

### `v0.16.1` Current Status

Implemented and review-fixed, but intentionally blocked.

Current block:
- `v0.16.1` should not be run until the teacher is actually walking
- the current teacher still exports a quasi-static basin, not a locomotion basin

### Student Training Rules

- do not train the student from scratch
- keep flat terrain only
- keep narrow commands first, around the teacher nominal gait
- delay pushes and broad coverage until the student preserves the gait basin

### `v0.16.1` Exit Criteria

`v0.16.1` is successful if:
- the student preserves stable nominal walking after warm start
- the student can handle a narrow command range without immediate pitch-collapse regression
- inference no longer depends on direct reference-motion targets

### `v0.16.1` Decision Rule

- If offline imitation fails, fix teacher/student observation-action contract mismatch before more PPO.
- If fine-tuning immediately destroys the gait, increase teacher regularization or lengthen imitation warm start before widening commands.

---

## Stage 2: Command-Conditioned Nominal Locomotion Expansion

**Goal:** turn the narrow student gait basin into a useful nominal locomotion policy across a modest command range on flat terrain.

### Entry Criteria

Start Stage 2 only after:
- `v0.16.0` teacher is stable
- `v0.16.1` student preserves the teacher gait basin
- narrow command tracking works without immediate regression to pitch-collapse

### Scope

Stage 2 remains intentionally narrow:
- flat terrain only
- no pushes
- no rough terrain
- no style objectives as the main target

### Main Work

1. Widen command range gradually
   - start near teacher nominal speed
   - then expand to a practical flat-ground band

2. Reduce teacher dependence
   - decay imitation / teacher regularization
   - keep enough regularization to avoid falling back into the old `v0.15.x` basin

3. Improve control quality
   - reduce velocity error
   - increase gait consistency
   - improve episode length and survival

4. Keep structured diagnostics
   - step quality
   - pitch failure
   - tracking error
   - torque / saturation

### Exit Criteria

Stage 2 is successful if:
- flat-ground command tracking works over a useful speed band
- gait remains stable across that band
- eval-clean survival remains high
- the student no longer needs strong teacher regularization to stay in a walking basin

### Decision Rule

- If widening commands breaks the basin, narrow again and strengthen teacher regularization temporarily.
- If the student only works near one nominal speed, remain in Stage 2 until command generalization is real.

---

## Stage 3: Robustness and Disturbance Training

**Goal:** add the real-world robustness requirements after nominal walking is stable.

### Entry Criteria

Start Stage 3 only after Stage 2 demonstrates:
- stable flat-ground walking
- usable command tracking
- no obvious pitch-collapse regression under nominal conditions

### Scope

Stage 3 adds:
- pushes / external disturbances
- command transitions
- limited terrain variation if needed

Stage 3 still does **not** prioritize style. It prioritizes robustness and deployability.

### Main Work

1. Push robustness
   - enable controlled push curriculum
   - measure recovery success and post-push gait retention

2. Command-transition robustness
   - acceleration / deceleration
   - stop / restart
   - modest command reversals only if appropriate for the robot

3. Terrain broadening
   - only after push robustness is acceptable on flat ground
   - start with small perturbations, not full rough-terrain ambition immediately

4. Safety / hardware-oriented regularization
   - torque and saturation discipline
   - contact stability
   - smoother recovery behavior

### Exit Criteria

Stage 3 is successful if:
- the policy retains nominal gait quality under moderate disturbances
- push recovery is reliable
- command changes do not immediately destabilize the gait
- robustness improvements do not destroy the nominal gait basin

### Decision Rule

- If robustness work breaks nominal gait, reduce disturbance difficulty and restore nominal basin first.
- Do not mask a nominal locomotion failure by claiming robustness progress.

---

## Stage 4: Style, Retargeting, and Motion-Prior Refinement

**Goal:** improve motion quality, style, or human-likeness only after the robot already has a stable and robust locomotion foundation.

### Why This Is Later

Early GMR-first work failed because:
- human gait does not map directly to robot dynamics
- IK retargeting alone preserves pose, not control feasibility
- stable robot-native gait had not been established yet

So style comes after function, not before.

### Scope

Possible Stage 4 directions:
- better reference clips
- richer retargeting
- multiple gait families
- style losses or motion priors
- AMP-style refinement if still useful

### Main Work

1. Improve reference quality
   - move from bootstrap placeholder clip to better retargeted motions
   - support more than one nominal gait if needed

2. Add style refinement
   - imitation regularizers
   - motion-prior losses
   - optional AMP-like objectives if justified

3. Preserve deployability
   - style must not undermine stability, command tracking, or robustness

### Exit Criteria

Stage 4 is successful if:
- gait quality improves visibly
- nominal and robust locomotion performance stay intact
- retargeted/style objectives are helping rather than fighting control

---

## Immediate Next Steps

Work in this order.

1. Implement `real retarget v1`.
2. Export a new retargeted nominal walk clip.
3. Repoint `v0.16.0` teacher training to the new clip.
4. Run a short `v0.16.0` teacher proof-of-life on the real clip.
5. Only if the teacher shows positive forward tracking, unblock `v0.16.1`.
6. Run teacher rollout export, BC warm start, and narrow student PPO.
7. Promote to Stage 2 only after the student preserves the gait basin under narrow commands.

## Success Definition For The Program

The program is back on track when all of the following are true:
- a teacher can produce stable nominal walking in MuJoCo
- a student can preserve that gait under narrow command-conditioned RL
- forward speed is no longer explained mainly by pitch-collapse

After that:
- Stage 2 widens flat-ground command coverage
- Stage 3 adds robustness and disturbance handling
- Stage 4 improves style / retargeted motion quality without sacrificing deployability
