# v0.17 Whole-Body Recovery Design

**Status:** Active implementation design for post-`v0.17.4t-crouch-teacher` B1 standing branches  
**Applies to:** Standing push recovery on the current MuJoCo / JAX / single-policy stack  
**Last updated:** 2026-03-30

---

## Outcome Note

`v0.17.4t` completed successfully on the fixed ladder.

Winning checkpoint:
- `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`

Fixed ladder:
- `eval_medium = 93.8%`
- `eval_hard = 70.6%`
- `eval_hard/term_height_low_frac = 29.4%`

Comparison across key branches:

| Branch | eval_medium | eval_hard | eval_hard h_low |
| --- | ---: | ---: | ---: |
| `v0.17.1` | 71.6% | 39.2% | 60.8% |
| `v0.17.3a` | 83.1% | 51.4% | 48.6% |
| `v0.17.3a-pitchfix` | — | ~51.8% | — |
| `v0.17.4t` | 93.8% | 70.6% | 29.4% |
| `v0.17.4t-step` | 83.1% | 52.2% | 47.8% |
| `v0.17.4t-crouch` | 83.3% | 54.2% | 45.8% |
| `v0.17.4t-crouch-teacher` | 84.4% | 55.1% | 44.9% |

This means:
- the first bounded teacher branch was sufficient to clear the benchmark gate
- but visual inspection suggests the policy still does not show a strong
  conditional recovery step under hard pushes
- `v0.17.4t-step` now has critical-suite ladder validation:
  - `eval_medium = 83.1%`
  - `eval_hard = 52.2%`
  - `eval_hard_long = 27.6%`
  - `eval_hard/term_height_low_frac = 47.8%`
  - `eval_easy` was skipped in that pass, so the script's printed `0.0%`
    target line should not be treated as an actual measured result
- the step-initiation mechanism branch did not beat the validated `v0.17.4t`
  baseline on fixed-ladder robustness
- Branch A (`v0.17.4t-crouch`) has now been run and closed:
  - fixed ladder:
    - `eval_clean = 100.0%`
    - `eval_easy = 100.0%`
    - `eval_medium = 83.3%`
    - `eval_hard = 54.2%`
    - `eval_hard_long = 32.1%`
  - it failed the `>= 65%` screening bar
  - it was worse than the validated `v0.17.4t` baseline
  - it also failed to improve on the `v0.17.4t-step` branch for the intended
    mechanism evidence
- Branch B B1 (`v0.17.4t-crouch-teacher`) has now been run:
  - fixed ladder:
    - `eval_clean = 100.0%`
    - `eval_easy = 100.0%`
    - `eval_medium = 84.4%`
    - `eval_hard = 55.1%`
    - `eval_hard_long = 31.3%`
  - it improved slightly over `v0.17.4t-step` and `v0.17.4t-crouch`
  - but it still failed the `>= 65%` screening bar
  - training-time diagnostics still did not show a convincing whole-body
    crouch-mediated recovery mechanism
- the active next sequence is now:
  optional `v0.17.4t-crouch-teacher` B2 -> `v0.18` if needed

The rest of this document remains useful as the design record for what
`v0.17.4t`, `v0.17.4t-step`, `v0.17.4t-crouch`, and `v0.17.4t-crouch-teacher`
B1 established, and as the detailed implementation plan for the next standing
branches.

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
1. optional `v0.17.4t-crouch-teacher` B2
2. `v0.18` position-level model-based control, only if the bounded RL path is
   exhausted

Historical context:
- `v0.17.3a`: fixed-ladder `eval_hard = 51.4%`
- `v0.17.3a-pitchfix`: `51.8%`
- `v0.17.3a-arrest`: `52.2%`
- `v0.17.4t`: `70.6%`
  - best checkpoint:
    - `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`
- `v0.17.4t-step`: `52.2%` (`eval_medium = 83.1%`, `eval_hard_long = 27.6%`)
  - best checkpoint:
    - `training/checkpoints/ppo_standing_v0174t_step_v0174t-step_20260328_155031-vtrx3sxe/checkpoint_440_57671680.pkl`
- `v0.17.4t-crouch`: `54.2%` (`eval_hard_long = 32.1%`)
- `v0.17.4t-crouch-teacher` B1: `55.1%` (`eval_medium = 84.4%`, `eval_hard_long = 31.3%`)
  - best checkpoint:
    - `training/checkpoints/ppo_standing_v0174t_crouch_teacher_v0174t-crouch-teacher_20260329_203032-rocg9xk1/checkpoint_600_78643200.pkl`
- `v0.17.4t-step` confirmed that internal visible-step evidence was not enough
  to outperform `v0.17.4t` on the fixed ladder

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

Observed result:
- screening run:
  - `run-20260329_134942-k4xaef65`
- best checkpoint:
  - `training/checkpoints/ppo_standing_v0174t_crouch_v0174t-crouch_20260329_134944-k4xaef65/checkpoint_550_72089600.pkl`
- fixed ladder:
  - `eval_clean = 100.0%`
  - `eval_easy = 100.0%`
  - `eval_medium = 83.3%`
  - `eval_hard = 54.2%`
  - `eval_hard_long = 32.1%`
- mechanism readout:
  - no convincing crouch-specific lowering
  - no convincing knee-flex increase
  - no clear improvement over `v0.17.4t-step` on useful step-trait evidence
- verdict:
  - this branch landed in the `< 65%` case
  - Branch A is closed
  - Branch B is now the active implementation path

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

Recommended Branch B implementation plan:
- resume from the best `v0.17.4t-crouch` checkpoint, not from `v0.17.4t-step`
- keep the disturbed-window reward gating from Branch A unchanged:
  - `height_target` relaxed in disturbed windows
  - `posture` relaxed in disturbed windows
  - `orientation` reduced in disturbed windows
  - `height_floor` and `com_velocity_damping` remain active
- add compact teacher targets on top of that existing disturbed-window reward
  scaffold rather than replacing it
- preserve actor observation / policy contract:
  - no new actor observation dimensions
  - no runtime teacher dependency at inference

Preferred initial target definitions:
1. `teacher_recovery_height`
   - purpose:
     - explicitly ask for modest body lowering during disturbed windows, since
       Branch A showed that permission alone does not create the behavior
   - gating:
     - active only when `push_active || recovery_active`
     - additionally gate on current step teacher demand:
       `teacher.step_required_hard > 0.5`
   - form:
     - bounded target band, not a hard point target
     - recommended initial band:
       - `target_min = 0.39 m`
       - `target_max = 0.42 m`
     - reward should be high inside the band and decay smoothly outside it
   - interpretation:
     - low enough to create visible knee flexion / body lowering
     - high enough to avoid teaching collapse or deep squat behavior

2. `teacher_com_velocity_reduction`
   - purpose:
     - explicitly ask for horizontal arrest after the disturbance, since
       Branch A preserved stepping but did not improve arrest quality
   - gating:
     - active only when `push_active || recovery_active`
     - preferred additional gate:
       - only after push onset or only when horizontal speed exceeds a small
         threshold, to avoid paying reward for already-static states
   - form:
     - scalar target on heading-local horizontal speed
     - recommended initial target:
       - encourage `horizontal_speed <= 0.10-0.15 m/s`
     - reward should be smooth and saturating, not binary
   - interpretation:
     - this is whole-body guidance for arrest, not footstep micromanagement

3. `teacher_knee_flex_min` (deferred; optional only)
   - only add if recovery-height guidance is too indirect
   - treat as a bounded indicator:
     - reward modest knee flexion in disturbed windows
     - do not reward unbounded crouching
   - this should remain a fallback behind recovery height and CoM-velocity
     targets

Suggested config additions:
- env:
  - `whole_body_teacher_enabled: true`
  - `whole_body_teacher_height_target_min: 0.39`
  - `whole_body_teacher_height_target_max: 0.42`
  - `whole_body_teacher_height_hard_gate: true`
  - `whole_body_teacher_com_vel_target: 0.12`
  - `whole_body_teacher_com_vel_active_speed_min: 0.10`
- reward_weights:
  - `teacher_recovery_height`
  - `teacher_recovery_height_sigma`
  - `teacher_com_velocity_reduction`
  - `teacher_com_velocity_reduction_sigma`
  - optional only:
    - `teacher_knee_flex_min`
    - `teacher_knee_flex_target`
    - `teacher_knee_flex_sigma`

Preferred code shape:
- keep `training/envs/teacher_step_target.py` responsible for step-intent
  signals only
- add a separate `training/envs/teacher_whole_body_target.py` for Branch B
  whole-body targets
- wire the whole-body teacher into `training/envs/wildrobot_env.py` beside the
  existing step teacher, not inside the policy or rollout code
- extend config/runtime parsing in:
  - `training/configs/training_runtime_config.py`
  - `training/configs/training_config.py`
- extend logging / reduction in:
  - `training/core/metrics_registry.py`
  - `training/core/experiment_tracking.py`
  - `training/core/training_loop.py`

Required metrics for Branch B:
- teacher activation / leakage:
  - `teacher/whole_body_active_frac`
  - `teacher/whole_body_active_during_clean_frac`
  - `teacher/whole_body_active_during_push_frac`
- recovery height teacher:
  - `teacher/recovery_height_target_mean`
  - `teacher/recovery_height_error`
  - `teacher/recovery_height_in_band_frac`
- CoM-velocity teacher:
  - `teacher/com_velocity_target_mean`
  - `teacher/com_velocity_error`
  - `teacher/com_velocity_target_hit_frac`
- reward terms:
  - `reward/teacher_recovery_height`
  - `reward/teacher_com_velocity_reduction`
  - optional only:
    - `reward/teacher_knee_flex_min`
- preserve existing Branch A recovery diagnostics:
  - `recovery/min_height`
  - `recovery/max_knee_flex`
  - `recovery/first_step_dist_abs`
  - `recovery/visible_step_rate`
  - `recovery/no_touchdown_frac`
  - `recovery/touchdown_then_fail_frac`
  - `recovery/pitch_rate_reduction_10t`

Tests required for implementation:
- deterministic unit tests for whole-body teacher target generation:
  - disturbed-window gating only
  - hard-gated activation through `teacher.step_required_hard`
  - clean-window leakage stays zero
- env aggregation tests for new teacher metrics and reward terms
- env-step persistence / auto-reset coverage for new info fields
- config parse tests for new env and reward keys
- verification that actor observation shape and policy contract hash remain
  unchanged

Recommended milestone-to-code mapping:
- B0:
  - add config/runtime plumbing
  - add `teacher_whole_body_target.py`
  - surface metrics only, with reward weights set to zero by default
  - run one dry-run verify / short smoke to confirm bounded signals
- B1:
  - turn on `teacher_recovery_height` only
  - keep `teacher_com_velocity_reduction = 0.0`
  - use this as the first actual Branch B screening run
- B2:
  - only if B1 still produces strong step-without-crouch behavior, add
    `teacher_com_velocity_reduction`
  - do not add `teacher_knee_flex_min` unless both task-space teachers are too
    weak
- B3:
  - fixed-ladder evaluation + viewer review
  - decide whether Branch B closes the whole-body mechanism gap or whether the
    RL path is exhausted

Milestones:

#### B0: Teacher Target Dry-Run
Deliver:
- teacher signals logged with no reward changes first
- document distribution / reachability / gating:
  - height target distribution
  - teacher active in disturbed windows only
  - clean-window leakage near zero
- concrete implementation:
  - add `training/envs/teacher_whole_body_target.py`
  - add whole-body teacher config keys with zeroed reward weights by default
  - surface the new `teacher/*` metrics and `reward/teacher_*` placeholders
  - keep total reward numerically unchanged in this milestone

Exit:
- targets are bounded and interpretable
- actor contract unchanged
- verify:
  - `teacher/whole_body_active_during_clean_frac ~= 0`
  - recovery-height targets stay in `[0.39, 0.42]`
  - no new reward term changes training behavior when weights are zero

#### B1: First Whole-Body Teacher Reward
Deliver:
- turn on one compact whole-body teacher term only
- recommended first term:
  - `teacher_recovery_height`
- keep step teacher active but do not increase its weights
- initial config intent:
  - resume from best `v0.17.4t-crouch` checkpoint
  - keep Branch A disturbed-window reward gating unchanged
  - set `teacher_recovery_height > 0`
  - keep `teacher_com_velocity_reduction = 0`
  - keep `teacher_knee_flex_min = 0`

Exit:
- internal metrics improve without clean-standing regression
- visible crouch + step appears more clearly than Branch A
- quantitative screening expectations:
  - `eval_clean/unnecessary_step_rate <= 0.10`
  - `recovery/min_height` decreases modestly relative to Branch A
  - `recovery/max_knee_flex` increases modestly relative to Branch A
  - `recovery/visible_step_rate` stays at least competitive with Branch A

Observed result:
- screening run:
  - `run-20260329_203029-rocg9xk1`
- best checkpoint:
  - `training/checkpoints/ppo_standing_v0174t_crouch_teacher_v0174t-crouch-teacher_20260329_203032-rocg9xk1/checkpoint_600_78643200.pkl`
- fixed ladder:
  - `eval_clean = 100.0%`
  - `eval_easy = 100.0%`
  - `eval_medium = 84.4%`
  - `eval_hard = 55.1%`
  - `eval_hard_long = 31.3%`
- training-time mechanism readout:
  - clean standing stayed clean at the best checkpoint
  - the height teacher stayed tightly gated with no clean leakage
  - but recovery-height in-band fraction remained low and the observed lowering
    / knee-flex metrics stayed weak
- verdict:
  - B1 did not clear the `>= 65%` screening bar
  - B1 recovered slightly over Branch A but did not close the whole-body
    mechanism gap
  - if the bounded RL path is continued, proceed only to B2

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
- expected signature versus B1:
  - `recovery/pitch_rate_reduction_10t` improves toward positive values
  - `recovery/no_touchdown_frac` and `recovery/touchdown_then_fail_frac` do not
    regress materially

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
