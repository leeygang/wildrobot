# v0.17.4t Teacher Training Design

**Status:** Active design for the next standing branch after `v0.17.3a-arrest`  
**Applies to:** Standing push recovery on the current MuJoCo / JAX / single-policy stack  
**Last updated:** 2026-03-24

---

## Purpose

This document defines the design for `v0.17.4t`, the first bounded
teacher-assisted standing branch.

The branch exists because the pure-RL standing sequence improved materially but
plateaued below the hard-push gate:
- `v0.17.3a`: `eval_hard = 51.4%`
- `v0.17.3a-pitchfix`: `eval_hard = 51.8%`
- `v0.17.3a-arrest`: `eval_hard = 52.2%`

Interpretation:
- the policy already steps under disturbance
- reward-only tuning is no longer producing meaningful gains
- the remaining gap looks like recovery-step geometry / step-quality, not
  “never steps at all”

The goal of `v0.17.4t` is to improve hard-push recovery **without** changing
the deployed architecture.

---

## Goal

Teach the policy to make a better first recovery step under hard pushes:
- step when bracing is no longer enough
- choose a plausible swing foot
- place the foot in a more useful recovery location
- keep quiet standing unchanged

Success means:
- improve fixed-ladder `eval_hard` beyond the `52.2%` plateau
- target `eval_hard > 60%`
- preserve `eval_clean = 100%`
- keep deployment as one policy with no runtime teacher dependency

---

## Non-Goals

`v0.17.4t` is **not**:
- a runtime planner
- a second deployed controller
- a high-level policy plus low-level policy architecture
- a requirement to add teacher targets to actor observations
- a sim2real hardening branch

`v0.17.3b` remains the transfer-hardening milestone and should stay after the
sim competence gate is solved.

---

## Core Strategy

Use a small training-time teacher that is active only during disturbed windows:
- `push_active || recovery_active`

The teacher does not output joint actions. It outputs compact recovery-step
targets:
- `step_required`
- `swing_foot`
- `target_step_x`
- `target_step_y`

The RL policy still produces all motor actions. The teacher only supplies an
extra learning signal that says:
- when a step is likely necessary
- which leg should make the first rescue step
- where that first rescue step should roughly land

This improves credit assignment. Instead of learning only from delayed outcomes
like `term_height_low` or `term_pitch`, the policy receives a more immediate
signal about what a useful recovery step looks like.

---

## Why Teacher Is Justified Now

The current evidence is:
- `v0.17.3a-arrest` made post-touchdown arrest rewards active
- `recovery/no_touchdown_frac` is near zero at the best point
- fixed-ladder improvement from reward-only tuning is marginal

So the unresolved question is no longer “should the robot step at all?”
It is:
- “is the first rescue step timed and placed well enough to arrest the fall?”

That is exactly the kind of gap a bounded teacher can address.

---

## Teacher Type

The first teacher should be a **heuristic step-target teacher**, not a learned
teacher and not a runtime planner.

Why heuristic first:
- cheapest to implement
- easiest to inspect and debug
- uses state signals the env already computes
- low risk of creating runtime dependency

Likely teacher source:
- capture-point-style recovery heuristic
- plus simple foot-selection and reachable-step clipping rules

Later escalation, only if needed:
- stronger privileged teacher-student path
- model-based or MJPC teacher in simulation

---

## How The Teacher Works

During `push_active || recovery_active`, the teacher inspects the current fall
state:
- body tilt
- body angular velocity
- heading-local base velocity
- capture-point-style error
- current foot positions / contact state

Then it produces four targets:

### 1. `step_required`

Meaning:
- should the policy treat the current state as requiring a rescue step?

Teacher intent:
- low during quiet standing
- high only when bracing is likely no longer enough

### 2. `swing_foot`

Meaning:
- which foot should likely make the first rescue step?

Teacher intent:
- choose a plausible foot based on lateral fall direction, current stance
  geometry, and contact / loading context

### 3. `target_step_x`

Meaning:
- approximate forward/back landing target in heading-local coordinates

### 4. `target_step_y`

Meaning:
- approximate lateral landing target in heading-local coordinates

The teacher is not required to be perfect. It only needs to provide a bounded,
useful prior that is better than delayed failure-only learning.

---

## Training Mechanism

The first `v0.17.4t` version should use the teacher for **reward shaping /
training supervision only**.

Do:
- compute teacher targets in the env during disturbed windows
- add small reward terms that encourage agreement with those targets
- log teacher activity and policy/teacher agreement

Do not:
- feed teacher targets into actor observations
- make the exported policy depend on teacher inputs
- replace PPO with a separate imitation pipeline

The training loop remains PPO on the same standing task.

---

## First Reward Design

The first teacher branch should add small, bounded terms:
- `teacher_step_required`
  reward when a first recovery touchdown occurs in states where teacher
  confidence says a step is needed
- `teacher_target_step_xy`
  reward when first touchdown lands near the teacher target
- optional `teacher_swing_foot`
  reward when touchdown foot matches teacher foot choice

Design rules:
- keep weights small compared with survival and posture terms
- gate them to disturbed windows only
- compute them on first recovery step / touchdown, not all touch events
- keep clipping and bounds explicit

Initial priority:
1. `teacher_target_step_xy`
2. `teacher_step_required`
3. optional `teacher_swing_foot`

Reason:
- the current evidence says step quality matters more than step occurrence

---

## Safety / Failure Containment

The teacher can be imperfect. `v0.17.4t` should be designed so that imperfect
teacher thresholds do not corrupt the final controller.

Protections:
- teacher only affects training
- teacher is active only in disturbed windows
- teacher weights stay small and bounded
- quiet-standing metrics remain a hard regression gate
- fixed-ladder outcome decides success, not teacher agreement alone

If teacher is noisy or wrong:
- `eval_clean/unnecessary_step_rate` will rise
- `touchdown_then_fail_frac` will not improve
- fixed-ladder `eval_hard` will not improve

That is how the branch should be judged.

---

## Metrics

`v0.17.4t` should use the existing recovery metrics and add teacher-specific
ones.

Existing recovery metrics that matter:
- `recovery/no_touchdown_frac`
- `recovery/touchdown_then_fail_frac`
- `recovery/first_touchdown_latency`
- `recovery/pitch_rate_reduction_10t`
- `recovery/capture_error_reduction_10t`
- `eval_clean/unnecessary_step_rate`

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

Interpretation:
- teacher should be mostly inactive in clean standing
- teacher should engage mostly in disturbed windows
- touchdown agreement should improve before fixed-ladder survival improves

---

## Execution Roadmap

### Milestone T0: Teacher Dry-Run

Objective:
- implement the teacher target generator and logging without changing the
  standing reward yet

Scope:
- add `training/envs/teacher_step_target.py`
- compute teacher outputs in `wildrobot_env.py`
- log teacher signals and agreement metrics
- no actor observation changes
- no runtime export changes

Entry criteria:
- `v0.17.3a-arrest` complete and accepted as the reward-only plateau

Exit criteria:
- teacher targets are emitted only in disturbed windows
- `eval_clean/unnecessary_step_rate` does not regress
- teacher debug plots are visually sane in sampled rollouts
- teacher-active fraction in clean standing is near zero

Failure criteria:
- teacher triggers broadly in quiet standing
- teacher targets are unstable / discontinuous enough to be unusable

### Milestone T1: Teacher XY Guidance

Objective:
- add first-teacher reward shaping using touchdown proximity to teacher target

Scope:
- add `teacher_target_step_xy` reward
- gate to first recovery touchdown only
- keep weight small and bounded

Exit criteria:
- internal recovery metrics improve:
  - lower `touchdown_then_fail_frac`
  - better `pitch_rate_reduction_10t`
  - better `capture_error_reduction_10t`
- fixed-ladder `eval_hard` shows a clear upward move over `52.2%`
- `eval_clean` remains saturated

Failure criteria:
- no meaningful hard-ladder improvement
- clear overstepping in clean standing
- teacher target agreement improves but survival does not

### Milestone T2: Teacher Step-Required / Foot Choice

Objective:
- add small supervision for whether a step is needed and optionally which foot
  should swing

Scope:
- add `teacher_step_required`
- optionally add `teacher_swing_foot`
- keep both weaker than `teacher_target_step_xy`

Reason:
- use only if T1 is directionally right but still insufficient

Exit criteria:
- fixed-ladder `eval_hard > 60%`, or
- clear improvement trajectory with no clean-standing regression

Failure criteria:
- no material improvement beyond T1
- rising `eval_clean/unnecessary_step_rate`
- teacher starts acting like a global stepping prior rather than a recovery aid

### Milestone T3: Confirm And Freeze

Objective:
- confirm the teacher branch is a real win and does not create runtime
  dependency

Scope:
- rerun from scratch if a resumed screening run succeeded
- compare directly against `v0.17.3a`, `pitchfix`, and `arrest`
- verify runtime/export contract is unchanged

Exit criteria:
- best checkpoint beats the `52.2%` plateau convincingly
- target milestone is `eval_hard > 60%`
- deployed policy still requires no teacher inputs
- branch is ready to hand off to later `v0.17.3b` sim2real hardening

If T3 fails:
- stop adding local heuristic complexity
- escalate to stronger teacher forms only if justified by the data

---

## Suggested Experiment Order

1. `v0.17.4t-dryrun`
   teacher targets + metrics only
2. `v0.17.4t-xy`
   add touchdown target reward
3. `v0.17.4t-step`
   add step-required reward
4. `v0.17.4t-foot`
   add swing-foot reward only if still needed

This keeps the branch debuggable and avoids bundling too many new levers into
the first run.

---

## File-Level Plan

Primary files:
- `training/configs/ppo_standing_v0174t.yaml`
- `training/envs/teacher_step_target.py`
- `training/envs/wildrobot_env.py`
- `training/configs/training_runtime_config.py`
- `training/configs/training_config.py`
- `training/core/metrics_registry.py`
- `training/core/experiment_tracking.py`

Tests / validation:
- add deterministic tests for teacher target generation
- add env-step coverage for teacher gating
- verify export/runtime contract remains unchanged

---

## Decision Rules

Proceed with `v0.17.4t` if:
- reward-only tuning is clearly plateaued
- teacher remains bounded to disturbed windows
- clean standing stays intact

Stop the branch and reconsider if:
- teacher causes overstepping in clean standing
- teacher metrics look active everywhere
- fixed-ladder performance does not move despite good teacher agreement

Escalate beyond heuristic teacher only if:
- `v0.17.4t` is cleanly implemented and measured
- the branch still fails to break the plateau

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
