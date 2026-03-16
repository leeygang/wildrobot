# Training Plan

**Version:** v0.16.1
**Status:** Active
**Last Updated:** 2026-03-16

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

**Current objective:** get a stable nominal forward-walk teacher in MuJoCo, then hand it off to a narrow command-conditioned student.

**Current blockers from PR review:**
- `v0.16.0` teacher clip/reward path still has a looping forward-reference issue:
  - wrapped clip phase currently implies wrapped absolute root position unless fixed in code
  - teacher root tracking must be cycle-relative or displacement-aware
- `v0.16.0` teacher phase-consistency signal is not yet trustworthy if it is derived from step index equality with the reference clip rather than state/contact-derived phase
- `v0.16.1` rollout export is not complete until it can actually run a teacher checkpoint in-env and emit the student dataset, not just shard prebuilt arrays

So Stage `1c` is the right direction, but the two overnight PRs are not yet ready to merge without fixes.

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

Implemented in local PR form, but **not review-complete**.

Open review findings that must be fixed before merge:
1. looping forward-reference bug:
   - wrapped phase cannot imply wrapped absolute root `x`
   - teacher root/foot tracking must support forward progression across cycles
2. phase-consistency signal must not collapse to a trivial step-index equality
3. teacher proof-of-life should use metrics that actually reflect tracking quality, not just reference indexing

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

Implemented in local PR form, but **not review-complete**.

Open review finding that must be fixed before merge:
- the current rollout tool only shards prebuilt arrays
- it must actually:
  - load a teacher checkpoint
  - run the teacher in the env
  - build student-layout observations
  - record teacher actions and required metadata
  - emit the dataset contract directly

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

1. Fix the `v0.16.0` teacher PR review findings.
2. Fix the `v0.16.1` real rollout-collection gap.
3. Merge `v0.16.0` only after the teacher branch is structurally sound.
4. Merge `v0.16.1` only after it can export real teacher data and warm-start a student.
5. Run a short `v0.16.0` teacher proof-of-life.
6. If the teacher is stable, run the `v0.16.1` student bootstrap.
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
