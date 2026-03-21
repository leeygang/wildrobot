# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0  
**Status:** Planned  
**Created:** 2026-03-20  

---

## Purpose

This document defines the return to the original standing-first training goal for WildRobot.

Goal:
- stand stably under external pushes
- take corrective steps only when bracing is no longer enough
- return to an upright neutral posture after recovery

This branch is intentionally **not** a walking branch.

It does not try to:
- learn nominal walking
- use a gait clock
- use velocity tracking rewards
- use a motion-prior or teacher pipeline as the first step

The plan is designed to follow a small number of public recipes closely, so the branch is driven by established practice rather than ad hoc exploration.

---

## Two External Recipes

v0.17 uses only two primary external references.

### Recipe A: `legged_gym` / Rudin et al. 2022

- Paper: https://proceedings.mlr.press/v164/rudin22a.html
- Code: https://github.com/leggedrobotics/legged_gym

This is the **implementation recipe** for v0.17.

We follow it for:
- PPO setup and rollout structure
- curriculum as the main training lever
- compact reward design centered on stability and control quality
- structured evals at fixed difficulty levels
- later domain randomization and robustness passes

What we do **not** copy literally:
- quadruped-specific terms
- walking-specific command tracking objectives
- terrain curriculum

WildRobot mapping:
- terrain curriculum -> push curriculum
- velocity tracking -> zero-velocity standing target
- gait rewards -> none in the first branch

### Recipe B: Li et al. 2024 Cassie

- Project: https://xbpeng.github.io/projects/Cassie_Locomotion/index.html

This is the **biped behavior recipe** for v0.17.

We follow it for:
- treating standing and push recovery as first-class skills
- using task randomization and disturbance exposure to build robustness
- using observation history when contact timing and disturbance response matter
- keeping the recovery policy general instead of scripting one exact recovery

What we take from it most directly:
- standing is a valid locomotion skill, not just a pretraining artifact
- push recovery should emerge from training under disturbances
- short-term and long-term history are legitimate upgrades if contact handling stalls

### What Is Not A Primary Recipe

The following are not mainline references for v0.17:
- DeepMimic
- ASAP
- the existing M3/FSM document as a first-line controller plan

Reason:
- they either require validated reference motions,
- or add a larger control-design burden than we need for the first standing reset,
- or were already tested in a weaker form and underperformed the pure-PPO standing baseline.

---

## Repo Starting Point

The standing history already gives a clear base to build from.

### What Worked

- `v0.13.8` established strong standing robustness at moderate pushes with collapse-prevention shaping and dual evals.
- `v0.13.10` added explicit posture return after disturbance.
- `v0.14.6` pure PPO at the calibrated hard-push regime reached the best hard-push standing checkpoint:
  - `eval_push/success_rate = 60.84%` at `10N x 10` after resume
  - dominant failure remained `term_height_low`

### What Did Not Work

- `v0.14.7` privileged critic did not beat the pure PPO baseline.
- `v0.14.8` actor-side capture-point observation improved quality but did not beat the pure PPO baseline.
- `v0.14.9` guide-only FSM underperformed decisively and remained weakly engaged / timeout-driven.

### Quantitative Boundary Data

The standing curriculum should be anchored to the force sweep already established in the changelog:

| Push Regime | Success Rate | Episode Length | Interpretation |
|---|---:|---:|---|
| `8N x 10` | `74.6%` | `431.8` | mostly bracing-manageable |
| `9N x 10` | `58.0%` | `382.9` | boundary / transition regime |
| `10N x 10` | `47.3%` | `346.7` | hard-push regime |

This is the empirical basis for the v0.17 curriculum:
- `8N x 10` is below the stepping boundary
- `9N x 10` is where stepping should begin to matter
- `10N x 10` is the first hard-push target

Capture-point geometry points in the same direction.

Back-of-the-envelope standing estimate for WildRobot:
- nominal CoM height `z ≈ 0.30 m`
- `omega_0 = sqrt(g / z) ≈ 5.7 rad/s`
- a `9N` push for `0.2s` on a `~1.2kg` body gives `delta_v ≈ 1.5 m/s`
- capture-point offset is then about `delta_v / omega_0 ≈ 0.26 m`

That offset is larger than the nominal support-width scale the current standing branch works around, so stepping at `9N+` is not just empirically observed. It is geometrically expected.

### Standing Branch Conclusion

The unresolved problem is:
- not disturbance detection
- not basic posture recovery
- not whether stepping pressure exists

It is:
- step quality
- touchdown quality
- post-touchdown arrest of collapse

That means v0.17 should restart from the best standing PPO recipe, not from walking and not from the disproven FSM branch.

---

## Strategy

v0.17 uses a single mainline strategy:

1. Start from a standing-only PPO baseline aligned to `legged_gym`.
2. Train with a push curriculum centered on the known bracing boundary.
3. Keep the reward simple and stability-focused.
4. Keep the actor observation contact-aware.
5. Keep mild need-gated stepping auxiliaries as bootstrap signals, not as the main objective.
6. Escalate by failure signature:
   - Li-style history for step timing / event sensitivity problems
   - guide-only M3-lite for touchdown geometry / foot-placement quality problems
7. Delay any motion-prior path until the RL baseline and the two bounded escalations clearly fail.

This avoids the two old failure modes:
- turning standing into a walking problem
- turning the training loop into an open-ended reward-design experiment

---

## Core Design

### Task Definition

Task:
- command is always zero velocity
- robot starts from neutral standing
- random pushes occur during the episode
- success means surviving the push and returning to upright quiet standing

There is no reward for:
- forward motion
- periodic stepping
- gait style
- making steps for their own sake

### Observation Design

The first observation layout should stay close to the current standing stack and add only the information justified by the two recipes.

Required actor observations:
- projected gravity
- base angular velocity
- heading-local base linear velocity
- joint positions
- joint velocities
- previous action
- left / right foot contact or load signals
- `capture_point_error`

Guideline:
- do **not** remove contact information from the actor
- do **not** add a gait clock
- do **not** include velocity commands, because the command is always zero

Reason:
- `legged_gym` rewards and curricula assume the policy can see enough proprioceptive state to manage contacts
- Li et al. specifically motivate richer temporal context for contact-sensitive biped behaviors

### Health And Termination Defaults

Unless hard evidence says otherwise, v0.17 inherits the proven standing thresholds:
- `target_height: 0.44`
- `min_height: 0.40`
- `max_pitch: 0.8`
- `max_roll: 0.8`
- `collapse_height_buffer: 0.03`
- `collapse_vz_gate_band: 0.07`

Rule:
- do **not** relax the fall thresholds into a deep-collapse regime
- the branch should preserve the current standing definition of "still alive"

### Reward Design

Use a small standing-focused reward stack.

Primary terms:
- upright orientation
- target height
- zero horizontal velocity
- posture return toward neutral
- pre-collapse shaping near the height threshold

Quality terms:
- angular velocity penalty
- slip penalty
- action-rate penalty
- torque / saturation penalty

Keep:
- `collapse_height`
- `collapse_vz`
- posture return
- need-gated stepping auxiliaries at low weight

Remove from the mainline:
- all walking reward terms
- gait periodicity
- dense progress
- step-progress propulsion terms
- gait clock dependent rewards

Default v0.17 position on stepping rewards:
- keep `step_event` and `foot_place` **on at low weight** in the mainline
- keep them strictly gated by the existing `need_step` logic from the standing branch
- treat them as bootstrap and diagnostic signals, not primary task terms

Recommended starting weights:
- `step_event: 0.10-0.15`
- `foot_place: 0.25-0.35`

Rule:
- if they produce fidgeting or gratuitous stepping, zero them quickly
- if they stay well-behaved, keep them because the standing history suggests they are a safe bootstrap signal here

This matches the external recipes better than inventing a new reward family.

### PPO and Training Geometry

Do not redesign PPO on day one.

Start from the current stable standing geometry, then scale only after the new branch is behaving sensibly.

Initial default:
- keep the current rollout geometry close to `training/configs/ppo_standing_push.yaml`
- keep conservative PPO trust-region settings
- keep deterministic evals and rollback

Scaling rule:
- only move toward larger `num_envs` after the baseline branch is stable and the metrics are trustworthy

This is more defensible than assuming a `4096`-env jump up front.

---

## What We Follow From Each Recipe

### From `legged_gym`

Use directly:
- curriculum is the main lever
- reward stack stays small and generic
- fixed eval ladders at known difficulty
- later domain randomization is a robustness stage, not a day-one feature

Concrete WildRobot translation:
- curriculum variable is push force / duration, not terrain
- objective is standing robustness, not velocity tracking
- checkpoint selection is driven by push evals, not reward

### From Li et al. 2024

Use directly:
- standing and recovery are legitimate target skills
- disturbance exposure is part of the task, not an afterthought
- history is a principled escalation path when contact-sensitive behavior stalls

Concrete WildRobot translation:
- keep standing as a first-class branch
- use push training to make recovery emerge
- if hard-push stepping is low-quality, add history or a guide path based on the failure signature before inventing new rewards

---

## Push Curriculum

The curriculum should stay centered around the known WildRobot bracing boundary.

Known boundary from the standing history:
- `8N x 10` is mostly bracing-manageable
- `9N x 10` is the transition regime
- `10N x 10` is the hard-push regime
- `9N x 15` is a strong long-impulse stress test

v0.17 curriculum stages:

1. Quiet standing
   - no pushes
   - verify posture baseline and logging

2. Easy pushes
   - roughly `3N` to `7N`
   - goal: ankle / hip strategy quality

3. Boundary pushes
   - roughly `7N` to `10N`
   - goal: first reliable recovery steps at the capturability boundary

4. Hard pushes
   - fixed and randomized pushes centered on `10N x 10`
   - include `9N x 15`
   - goal: improve hard-push success beyond the old pure-PPO ceiling

5. Robustness pushes
   - add direction diversity, duration diversity, and later multi-push episodes
   - only after hard-push single-recovery behavior is stable

Rule:
- do not jump immediately to `14N+` or multi-push training
- first solve the boundary regime where stepping should begin

---

## Evaluation Ladder

Every checkpoint should run a fixed eval ladder.

Required eval suites:
- `eval_clean`
  - no pushes
- `eval_easy`
  - `5N x 10`
- `eval_medium`
  - `8N x 10`
- `eval_hard`
  - `10N x 10`
- `eval_hard_long`
  - `9N x 15`

Optional later eval:
- `eval_extreme`
  - `14N x 12`

Primary metrics:
- `success_rate`
- `episode_length`
- `term_height_low_frac`
- `term_pitch_frac`
- `term_roll_frac`
- `tracking/max_torque`
- `debug/torque_sat_frac`

Required new recovery metrics:
- time from push end to upright recovery
- post-push residual body velocity
- number of touchdown events after each push
- first-step latency after push onset
- support-foot change count

Decision rule:
- checkpoint selection is driven by fixed evals, especially `eval_hard` and `eval_hard_long`
- do not use aggregate reward as the main selection signal

---

## Concrete Roadmap

### v0.17.0: Standing Reset

Objective:
- create a clean standing config aligned to the two recipes

Changes:
- fork from the standing branch, not the walking branch
- disable `fsm_enabled`
- disable `base_ctrl_enabled`
- remove walking-specific reward terms
- keep collapse-prevention and posture-return terms
- keep contact-aware actor observations
- keep `capture_point_error` in the actor observation from day one
- keep low-weight need-gated `step_event` / `foot_place`
- keep current PPO geometry and eval/rollback guardrails

Deliverables:
- new standing config
- new or updated standing observation layout
- fixed eval ladder
- recovery metrics plumbing

Exit criteria:
- `eval_clean/success_rate = 100%`
- quiet-standing posture is stable for full episodes
- all eval and recovery metrics log correctly

### v0.17.1: Recipe Baseline at The Boundary

Objective:
- reproduce and exceed the old hard-push pure-PPO baseline using the cleaned standing recipe

Training setup:
- curriculum through easy and medium pushes
- hard evals present from the start
- no controller guide
- low-weight need-gated step auxiliaries remain on

Main question:
- can the cleaned standing recipe beat the old `v0.14.6` line at `10N x 10`

Exit criteria:
- `eval_easy > 95%`
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` below the old hard-push baseline

### v0.17.2: Bounded Escalation Fork

Objective:
- improve hard-push recovery if the baseline plateaus near the old standing ceiling

This stage is a controlled fork, not a single hard-sequenced escalation.

#### Track A: Li-Style History

Use this track if:
- stepping rarely happens at all, or
- first-step timing is delayed, or
- contact event handling looks noisy / inconsistent

Changes:
- stack the most recent `4` control ticks of the contact-sensitive channels:
  - projected gravity
  - base angular velocity
  - joint positions
  - joint velocities
  - left / right foot contact or load
  - `capture_point_error`
- this adds roughly `28 x 4 = 112` history features on top of the instantaneous observation
- keep the reward stack and curriculum unchanged

Reason:
- this is the simplest Li-style temporal upgrade that preserves the current PPO stack

Expected outcome:
- lower first-step latency
- cleaner event timing
- better recovery initiation

#### Track B: M3-Lite Guide Path

Use this track if:
- stepping is present, but touchdown quality is poor, or
- the robot changes support but still collapses through `term_height_low`, or
- left / right step geometry is visibly wrong or asymmetric

Changes:
- reintroduce only a guide-only foot-placement prior
- keep full residual authority
- do **not** reduce policy authority by phase
- do **not** use arm logic, since WildRobot has no arms
- keep the touchdown target conservative and standing-specific

Reason:
- this addresses foot geometry directly without returning to the old constrained FSM branch

Expected outcome:
- better touchdown placement
- lower post-touchdown collapse
- improved hard-push survival without rewriting the training recipe

Exit criteria for either winning track:
- `eval_hard` improves materially over `v0.17.1`
- `term_height_low_frac` falls
- either first-step latency improves, or post-touchdown collapse improves, depending on the chosen track

### v0.17.3: Hard-Push Reliability

Objective:
- make hard-push recovery reliable across directions

Changes:
- broaden push direction distribution
- include diagonal and fore-aft pushes
- keep training centered on single-recovery episodes

Exit criteria:
- `eval_hard > 80%`
- `eval_hard_long > 65%`
- recovery-to-upright time stays bounded
- left and right recovery stepping both appear when appropriate

### v0.17.4: Robustness Pass

Objective:
- add standard robustness tools from the `legged_gym` recipe

Changes:
- friction randomization
- body-mass randomization
- actuator-strength randomization
- observation noise
- only after the nominal hard-push behavior is stable, add limited multi-push episodes

Exit criteria:
- hard-push success remains high under randomization
- policy degrades gradually, not catastrophically
- checkpoint performance is stable across training

### v0.17.5: Optional Walking Handoff

This is outside the core v0.17 standing goal.

Only consider this after:
- hard-push standing recovery is reliable
- the step response is reusable and symmetric
- robustness randomization does not destroy the standing basin

At that point, walking can be revisited either as:
- a command-conditioned extension of the standing policy, or
- a separate walking branch bootstrapped from the standing recovery policy

This milestone is intentionally deferred.

---

## Decision Rules

These rules are part of the plan and should be followed strictly.

### If Quiet Standing Breaks

- do not add new reward terms
- return to the `legged_gym`-style baseline stack
- check observation normalization, action filtering, and PPO stability first

### If Hard-Push Success Plateaus Near The Old Baseline

- do not start a new motion-prior branch
- diagnose the failure mode first
- choose the bounded escalation track that matches it:
  - history for timing / event sensitivity
  - M3-lite for touchdown geometry / placement quality

### If Step Count Stays Near Zero At The Boundary

- keep the reward stack stable
- do not start a new motion-prior branch
- first apply the Li-style history track

### If Step Count Rises But `term_height_low` Does Not Fall

Interpretation:
- the robot is attempting steps
- the problem is step quality, not step intent

Action:
- keep the reward stack stable
- inspect first-step latency, touchdown timing, and post-push residual velocity
- try the M3-lite guide path before adding reward complexity

### If Both Bounded Escalations Fail

Only then consider a second-line branch:
- increase the low-weight step auxiliaries modestly, or
- start a separate motion-prior / recovery-teacher investigation

These are fallback branches, not the mainline plan.

### If Robustness Work Breaks Nominal Recovery

- reduce randomization strength
- reduce disturbance diversity
- restore the hard-push single-recovery basin first

This follows the current repo-wide rule that robustness must not mask a broken nominal skill.

---

## What We Are Explicitly Avoiding

To keep confidence high, v0.17 will not begin with:
- a walking objective
- a gait clock
- a new teacher or imitation pipeline
- a large custom reward family for stepping
- a fresh controller architecture as the mainline
- a broad literature-driven exploration branch

The branch should look like a disciplined execution of two recipes:
- `legged_gym` for the RL training recipe
- Li et al. for the biped behavior and escalation path

---

## Relationship To The Main Training Plan

This document defines the return to the original standing push-recovery goal.

`training/docs/learn_first_plan.md` remains the walking-first roadmap described in `CLAUDE.md`.

For the standing objective, v0.17 should be treated as the primary plan.

v0.17 is narrower than the walking roadmap:
- it tests whether WildRobot can solve the original standing push-recovery goal cleanly
- it does so using a small, recipe-driven branch rather than a broad research program

If v0.17 succeeds, it gives the project:
- a deployable standing-recovery policy
- a stronger basis for later walking handoff
- a clearer answer about whether WildRobot needs a motion prior for recovery at all
