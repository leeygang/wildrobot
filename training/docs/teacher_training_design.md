# v0.17 Whole-Body Recovery Design

**Status:** Active implementation design for post-`v0.17.4t` standing branches  
**Applies to:** Standing push recovery on the current MuJoCo / JAX / single-policy stack  
**Last updated:** 2026-03-29

---

## Outcome Note

`v0.17.4t` completed successfully on the fixed ladder.

Winning checkpoint:
- `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`

Fixed ladder:
- `eval_medium = 93.8%`
- `eval_hard = 70.6%`
- `eval_hard/term_height_low_frac = 29.4%`

This means:
- the first bounded teacher branch was sufficient to clear the benchmark gate
- but visual inspection suggests the policy still does not show a strong
  conditional recovery step under hard pushes
- `v0.17.4t-step` should still be ladder-validated before it is used as
  evidence for larger architecture decisions
- if visible stepping and whole-body recovery when bracing fails are product
  requirements, the design remains active for the next sequence:
  `v0.17.4t-crouch` -> `v0.17.4t-crouch-teacher` -> `v0.18` if needed

The rest of this document remains useful as the design record for what
`v0.17.4t` and `v0.17.4t-step` established, and as the detailed implementation
plan for the next standing branches.

---

## Purpose

This document defines the implementation-level plan for the follow-on standing
branches after `v0.17.4t` cleared the hard gate but did not yet show the full
desired mechanism.

This doc is the detailed companion to
[standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md),
which stays the canonical high-level plan. Use this document when the work
requires concrete implementation scope, milestones, file ownership, and
decision gates.

Active sequence covered here:
1. `v0.17.4t-crouch`
2. `v0.17.4t-crouch-teacher`
3. `v0.18` position-level model-based control, only if the bounded RL path is
   exhausted

Historical context:
- `v0.17.3a`: fixed-ladder `eval_hard = 51.4%`
- `v0.17.3a-pitchfix`: `51.8%`
- `v0.17.3a-arrest`: `52.2%`
- `v0.17.4t`: `70.6%`
- `v0.17.4t-step`: mechanism branch; best internal point currently needs the
  same fixed-ladder validation before it is used for architecture decisions

---

## Goal

Teach or engineer the robot to use the intended recovery mechanism under hard
push:
- brace when bracing is enough
- step when bracing is no longer enough
- if needed, use modest knee flexion / CoM lowering / trunk motion during the
  recovery window
- keep quiet standing unchanged

Success means:
- preserve or exceed the current standing benchmark (`v0.17.4t` at
  `eval_hard = 70.6%`)
- preserve `eval_clean = 100%`
- under hard pushes, the robot visibly initiates a recovery step when bracing
  alone is not enough
- if large pushes exceed upright brace capacity, the robot is allowed to use a
  bounded whole-body recovery response instead of falling tall and straight
- quiet standing keeps unnecessary stepping low
- keep deployment as one policy for the `v0.17.x` RL branches, with no runtime
  teacher dependency

---

## Non-Goals

The `v0.17.x` follow-on branches are **not**:
- a runtime planner
- a second deployed controller
- a high-level policy plus low-level policy architecture
- a requirement to add teacher targets to actor observations
- a sim2real hardening branch (`v0.17.3b` remains later)

`v0.18` is explicitly a new branch and a different controller architecture.
That branch is allowed to change the deployed artifact if the bounded RL path
fails to deliver the mechanism.

---

## Active Design Principles

1. Disturbed-window-only changes
- all recovery-specific permissions or teacher guidance remain bounded to
  `push_active || recovery_active`
- quiet standing should keep the normal standing priors

2. Compact task-space guidance, not per-joint supervision
- if RL needs extra structure, guide a small number of recovery-relevant
  quantities:
  - step required
  - swing foot
  - touchdown target
  - recovery height band
  - horizontal CoM-velocity reduction
- do not add teacher targets for every joint angle

3. Runtime contract stability for `v0.17.x`
- actor observations remain unchanged
- exported policy remains a single policy with no runtime teacher dependency

4. Quantitative gate before architecture pivot
- no architecture change should be triggered by viewer impressions alone
- use fixed-ladder comparison plus behavior metrics plus visual inspection

5. Diagnosability over complexity
- each branch should answer one concrete question:
  - did removing reward barriers unlock whole-body recovery?
  - if not, does compact whole-body teacher guidance unlock it?
  - if not, is a controller-stack change justified?

---

## Current Evidence

---

`v0.17.4t` result:
- fixed ladder:
  - `eval_medium = 93.8%`
  - `eval_hard = 70.6%`
  - `eval_hard/term_height_low_frac = 29.4%`
- outcome gate is solved
- but visual inspection suggests large-push failures still look too tall and
  stiff

---

## Metrics

The follow-on branches should use the existing recovery metrics and add any
extra branch-specific metrics needed to answer the current question.

Existing recovery metrics that matter:
- `recovery/no_touchdown_frac`
- `recovery/touchdown_then_fail_frac`
- `recovery/first_touchdown_latency`
- `recovery/pitch_rate_reduction_10t`
- `recovery/capture_error_reduction_10t`
- `eval_clean/unnecessary_step_rate`
- `recovery/visible_step_rate`
- `recovery/first_step_dx`
- `recovery/first_step_dy`

Teacher-specific metrics to add:
- `teacher/active_frac`
- `teacher/step_required_mean`
- `teacher/step_required_hard_frac`
- `teacher/target_step_x_mean`
- `teacher/target_step_y_mean`
- `teacher/swing_left_frac`
- `teacher/foot_match_frac`
- `teacher/target_xy_error`
- `teacher/teacher_active_during_clean_frac`
- `teacher/teacher_active_during_push_frac`

Additional metrics for whole-body recovery branches:
- `recovery/min_height`
  minimum robot height during disturbed recovery window
- `recovery/max_knee_flex`
  max recovery knee flexion magnitude, averaged over the recovery window
- `recovery/first_step_dist_abs`
  mean absolute first-step distance:
  `sqrt(dx^2 + dy^2)` averaged without sign cancellation
- `recovery/trunk_pitch_excursion`
  absolute trunk/waist contribution during recovery if instrumented

Interpretation:
- `visible_step_rate` rising alone is not enough
- the desired progression is:
  - more clear recovery steps when needed
  - lower `no_touchdown_frac`
  - lower `touchdown_then_fail_frac`
  - better `pitch_rate_reduction_10t`
  - modest recovery height lowering or knee flexion only during disturbed
    windows
- `eval_clean/unnecessary_step_rate` remains a hard regression gate

---

## Active Execution Roadmap

### Branch A: `v0.17.4t-crouch`

Objective:
- test whether removing reward barriers is sufficient for visible whole-body
  recovery, while building on the current best step-initiation mechanism

Branch principle:
- keep the current `v0.17.4t-step` teacher signals
- change only disturbed-window reward logic
- do not add new observation dimensions or runtime dependencies

Resume point:
- current best `v0.17.4t-step` checkpoint:
  `training/checkpoints/ppo_standing_v0174t_step_v0174t-step_20260328_155031-vtrx3sxe/checkpoint_440_57671680.pkl`
  or the current best validated checkpoint from that branch if the ladder
  result changes the winner

Concrete implementation scope:
- new config:
  - `training/configs/ppo_standing_v0174t_crouch.yaml`
- reward/config plumbing:
  - `training/configs/training_runtime_config.py`
  - `training/configs/training_config.py`
- env reward gating and metrics:
  - `training/envs/wildrobot_env.py`
  - `training/envs/env_info.py`
  - `training/core/metrics_registry.py`
  - `training/core/experiment_tracking.py`
  - `training/core/training_loop.py`

Concrete changes:
- during `push_active || recovery_active`:
  - gate `height_target` to `0.0`
  - gate `posture` to `0.0`
  - reduce `orientation` from `-1.0` to `-0.3`
  - add `height_floor` penalty below `0.20 m`
  - add `com_velocity_damping` on horizontal CoM speed
- outside disturbed windows:
  - keep existing `v0.17.4t-step` standing rewards unchanged

Milestones:

#### A0: Reward Plumbing
Deliver:
- config loads cleanly
- disturbed-window gating is implemented and covered by deterministic tests
- new metrics exist:
  - `recovery/min_height`
  - `recovery/first_step_dist_abs`
  - `recovery/max_knee_flex` if feasible from current joint naming

Exit:
- tests pass
- actor observation contract unchanged
- no runtime/export contract drift

#### A1: Screening Run
Deliver:
- one resumed screening run from the best `v0.17.4t-step` checkpoint
- stop at the best internal checkpoint, not necessarily the final iteration

Quantitative gate:
- run fixed ladder on the best checkpoint
- compare against `v0.17.4t` (`eval_hard = 70.6%`)
- must not regress below `65%`
- `eval_clean/unnecessary_step_rate <= 0.10`
- inspect:
  - `recovery/visible_step_rate`
  - `recovery/no_touchdown_frac`
  - `recovery/touchdown_then_fail_frac`
  - `recovery/pitch_rate_reduction_10t`
  - `recovery/min_height`
  - `recovery/first_step_dist_abs`

#### A2: Decision
If:
- `eval_hard >= 70%`
- visible crouch + step appears
- clean standing remains clean

Then:
- stay on the RL path and hand off to `v0.17.3b`

If:
- `eval_hard >= 65%` but crouch/whole-body response is still absent or weak

Then:
- proceed to Branch B: `v0.17.4t-crouch-teacher`

If:
- `eval_hard < 65%`

Then:
- restore `v0.17.4t` as the best RL baseline
- proceed to Branch B anyway, because reward gating alone was insufficient

### Branch B: `v0.17.4t-crouch-teacher`

Objective:
- add one compact whole-body recovery scaffold inside RL before changing the
  deployed controller architecture

Branch principle:
- keep step teacher for compact recovery intent:
  - `teacher_step_required`
  - `teacher_swing_foot`
  - `teacher_target_step_xy`
- add one or two compact whole-body targets during disturbed windows
- do not add per-joint supervision

Default compact targets:
1. `teacher_recovery_height`
   - bounded recovery height target or target band during disturbed windows
2. `teacher_com_velocity_reduction`
   - teacher-assisted pressure toward lower horizontal CoM velocity during
     recovery

Optional only if still needed:
3. `teacher_knee_flex_min`
   - bounded knee-flex indicator or target band

Concrete implementation scope:
- new config:
  - `training/configs/ppo_standing_v0174t_crouch_teacher.yaml`
- teacher extension:
  - extend `training/envs/teacher_step_target.py` or add
    `training/envs/teacher_whole_body_target.py`
- env/reward plumbing:
  - `training/envs/wildrobot_env.py`
  - `training/envs/env_info.py`
  - `training/core/metrics_registry.py`
  - `training/core/experiment_tracking.py`
  - `training/core/training_loop.py`

Concrete changes:
- retain current step teacher, but keep it bounded
- add teacher reward/supervision terms only during `push_active || recovery_active`
- first preferred whole-body teacher is task-space, not joint-space:
  - recovery height target / band
  - CoM-velocity reduction target
- keep quiet standing unchanged outside disturbed windows

Milestones:

#### B0: Teacher Target Dry-Run
Deliver:
- teacher signals logged with no reward changes first
- document distribution / reachability / gating:
  - height target distribution
  - teacher active in disturbed windows only
  - clean-window leakage near zero

Exit:
- targets are bounded and interpretable
- actor contract unchanged

#### B1: First Whole-Body Teacher Reward
Deliver:
- turn on one compact whole-body teacher term only
- recommended first term:
  - `teacher_recovery_height`
- keep step teacher active but do not increase its weights

Exit:
- internal metrics improve without clean-standing regression
- visible crouch + step appears more clearly than Branch A

#### B2: Optional Second Teacher Term
Deliver:
- only if B1 is still too weak, add:
  - `teacher_com_velocity_reduction`
  or
  - `teacher_knee_flex_min`
- add one term at a time

Exit:
- fixed ladder is competitive (`eval_hard >= 65%` screening bar, target `>= 70%`)
- visible crouch + step appears
- `eval_clean/unnecessary_step_rate <= 0.10`

#### B3: RL Path Final Decision
If:
- fixed ladder stays competitive and the mechanism is present

Then:
- bounded RL whole-body guidance succeeded
- proceed to `v0.17.3b`

If:
- compact whole-body teacher guidance still fails to produce the mechanism

Then:
- the bounded RL whole-body guidance path is exhausted
- proceed to Branch C: `v0.18`

### Branch C: `v0.18` Position-Level Model-Based Control

Objective:
- replace end-to-end RL joint control with a position-level model-based stack
  that reduces the whole-body coordination problem from RL exploration to
  structured task-space IK design

This is the fallback after Branch A and Branch B fail to deliver the desired
mechanism.

Concrete implementation scope:
- new code under `control/`:
  - `control/ik/jacobian_ik.py`
  - `control/planner/capture_point.py`
  - `control/planner/standing_planner.py`
  - `control/standing_controller.py`
  - `control/sim_adapter.py`
- validation scripts:
  - `scripts/validate_ik.py`
  - `scripts/validate_standing_controller.py`

Milestones:

#### C0: Kinematics And IK Validation
Deliver:
- Jacobian-based IK on the MuJoCo model
- static validation of CoM height tracking + foot placement
- joint-limit and singularity handling

Exit:
- commanded CoM lowering produces bounded knee flexion on the static model
- swing-foot targets remain reachable within configured bounds

#### C1: Standing Controller In Simulation
Deliver:
- standing controller with:
  - quiet standing
  - recovery mode trigger
  - CoM lowering target
  - step target
- contact-consistent task prioritization

Exit:
- quiet standing stable in sim
- visible crouch + step under scripted pushes
- survives `eval_medium` pushes

#### C2: Fixed-Ladder Validation
Deliver:
- same ladder used by RL branches
- direct comparison against `v0.17.4t`

Exit:
- `eval_medium > 75%`
- `eval_hard > 60%`
- visual inspection confirms coordinated whole-body recovery

#### C3: Walking Extension
Deliver:
- same planner + IK stack extended to slow walking
- standing recovery remains a regression gate

Exit:
- slow forward walking
- stop / hold
- standing regression passes

#### C4: Sim2Real
Deliver:
- hardware-calibrated kinematics
- planner + IK deployed to the robot

Exit:
- standing recovery works on hardware with acceptable drift from sim
- same runtime stack remains viable for walking

### Historical Completed Teacher Milestones

The original `v0.17.4t` teacher milestones are complete historical context:
- dry-run teacher instrumentation
- touchdown XY teacher
- step-required teacher
- swing-foot teacher

They established the current RL baseline and are not the active implementation
focus of this document anymore.

---

## Suggested Experiment Order

1. fixed-ladder validate `v0.17.4t-step` best checkpoint
2. `v0.17.4t-crouch`
3. `v0.17.4t-crouch-teacher`
4. `v0.18` only if the bounded RL whole-body guidance path still fails

---

## File-Level Plan

Primary files:
- `training/configs/ppo_standing_v0174t_crouch.yaml`
- `training/configs/ppo_standing_v0174t_crouch_teacher.yaml`
- `training/envs/teacher_step_target.py`
- optional:
  - `training/envs/teacher_whole_body_target.py`
- `training/envs/wildrobot_env.py`
- `training/envs/env_info.py`
- `training/configs/training_runtime_config.py`
- `training/configs/training_config.py`
- `training/core/metrics_registry.py`
- `training/core/experiment_tracking.py`
- `training/core/training_loop.py`
- `control/ik/jacobian_ik.py`
- `control/planner/capture_point.py`
- `control/planner/standing_planner.py`
- `control/standing_controller.py`

Tests / validation:
- add deterministic tests for:
  - recovery-gated reward masking
  - whole-body teacher target generation if used
  - new recovery metrics
  - env-step / auto-reset persistence
- verify export/runtime contract remains unchanged for all `v0.17.x` branches
- add standalone validation scripts for `v0.18` IK and standing controller

---

## Decision Rules

Proceed to Branch A if:
- `v0.17.4t-step` fixed-ladder result is available
- the mechanism requirement is still open after viewer + metrics review

Proceed from Branch A to Branch B if:
- reward gating alone did not create the mechanism
- but the RL path remains benchmark-competitive enough to justify one last
  bounded whole-body guidance branch

Proceed from Branch B to Branch C if:
- compact whole-body teacher guidance also fails to produce the mechanism
- or the RL path becomes clearly less competitive than the validated
  `v0.17.4t` baseline

Escalate to `v0.18` only after:
- Branch A gate is closed
- Branch B gate is closed
- the evidence says the bounded RL whole-body guidance path is exhausted

---

## Bottom Line

`v0.17.4t` is not a controller rewrite.

It is a bounded training-time hint system meant to teach a better first rescue
step:
- when to step
- which foot should step
- where that step should go

If it works, the final robot still runs the same way:
- one policy
- no runtime teacher
- better hard-push standing recovery
