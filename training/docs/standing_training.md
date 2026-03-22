# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0  
**Status:** `v0.17.1` evaluated, `v0.17.2` planned  
**Created:** 2026-03-20  

---

## Purpose

This document defines the return to the original standing-first training goal for WildRobot.

Goal:
- stand stably under external pushes
- take corrective steps only when bracing is no longer enough
- return to an upright neutral posture after recovery

Validation remains standing-first, but starting with `v0.17.2` the target architecture must support both:
- standing push recovery
- nominal walking

The early `v0.17.0` / `v0.17.1` branch was intentionally **not** a walking branch.

It did not try to:
- learn nominal walking
- use a gait clock
- use velocity tracking rewards
- use a motion-prior or teacher pipeline as the first step

The plan is designed to follow a small number of public recipes closely, so the branch is driven by established practice rather than ad hoc exploration.

Current standing result:
- `v0.17.1` completed as the recipe baseline run
- fixed ladder evaluation did **not** beat the old `v0.14.6` hard-push baseline
- `v0.17.2` is therefore the architecture-pivot stage:
  - primary reference stack: OCS2 / humanoid MPC family
  - concrete humanoid reference: 1X `wb-humanoid-mpc`
  - OpenLoong remains a secondary MuJoCo implementation reference

---

## External References

v0.17 now has two phases with different primary references.

### Phase 1 References: `v0.17.0` - `v0.17.1`

These were the references used for the PPO standing baseline.

#### Recipe A: `legged_gym` / Rudin et al. 2022

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

#### Recipe B: Li et al. 2024 Cassie

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

### Phase 2 References: `v0.17.2+` Architecture Pivot

These are the main references for the post-`v0.17.1` stack change.

#### Recipe C: OCS2 / MPC Stack

- Code: https://github.com/leggedrobotics/ocs2
- Examples: https://leggedrobotics.github.io/ocs2/robotic_examples.html

This is the new **control architecture foundation**.

We follow it for:
- optimal-control problem structure
- switched contact planning structure
- model, cost, and constraint decomposition
- receding-horizon control as the main stabilizing mechanism

#### Recipe D: 1X `wb-humanoid-mpc`

- Code: https://github.com/1x-technologies/wb-humanoid-mpc

This is the new **humanoid implementation reference**.

We follow it for:
- humanoid-specific MPC structure
- whole-body execution layer expectations
- standing and walking under one control architecture
- practical robot-model integration patterns

#### Secondary Reference: OpenLoong Dyn-Control

- Code: https://github.com/loongOpen/OpenLoong-Dyn-Control

OpenLoong is a secondary implementation reference for:
- MuJoCo integration patterns
- practical humanoid module boundaries
- working MPC/WBC-style control in simulation

It is **not** the primary adoption target because it is more tightly coupled to its own robot and controller conventions.

### What Is Not A Primary Recipe

The following are not mainline references for the post-`v0.17.1` plan:
- DeepMimic
- ASAP
- the existing M3/FSM document as the main architecture
- PPO-only standing escalation as the mainline

Reason:
- they either require validated reference motions,
- or add a larger control-design burden than needed for the initial baseline,
- or were already tested in weaker form and underperformed the true hard-push baseline,
- or do not provide the broader standing+walking architecture now required.

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

That means `v0.17.0` / `v0.17.1` were the correct PPO baseline reset, but `v0.17.2+` should pivot away from PPO-only escalation and toward a more established model-based humanoid stack.

---

## Strategy

v0.17 now uses a two-stage strategy:

1. Establish a standing-only PPO baseline aligned to `legged_gym`.
2. Use that baseline to determine whether pure PPO is enough for the true hard gate.
3. Because `v0.17.1` failed the true hard gate, pivot to an OCS2 / humanoid MPC stack for `v0.17.2+`.
4. Keep standing push recovery as the first validation task on the new stack.
5. Extend the same stack to nominal walking after standing recovery is credible.
6. Keep RL augmentation as a later option, not the new foundation.

This avoids the two old failure modes:
- turning standing into a walking problem
- turning the training loop into an open-ended reward-design experiment
- continuing PPO-only escalation after the fixed ladder already rejected it

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

Result:
- training run `run-20260321_123307-0flajxzh` completed with stable PPO and perfect `eval_clean`
- internal `eval_push` rose to `84.4%`, but this did **not** transfer to the fixed ladder
- best checked fixed-ladder checkpoint was `checkpoint_160_20971520.pkl`
- fixed ladder result:
  - `eval_medium = 71.6%`
  - `eval_hard = 39.2%`
  - `eval_hard/term_height_low_frac = 60.8%`

Verdict:
- `v0.17.1` failed the true gate
- it did **not** beat the old `v0.14.6` hard-push baseline of `60.84%` at `10N x 10`
- the internal `eval_push/*` probe overstated true hard-push performance

Diagnosis:
- clean standing and PPO stability were not the problem
- the dominant failure remained `term_height_low`
- stepping pressure existed, but hard-push recovery quality was still insufficient
- this points to touchdown geometry / post-touchdown arrest, not to a basic disturbance-detection failure

Decision:
- do **not** extend `v0.17.1`
- use it as the baseline result for the branch
- use it as the justification for the `v0.17.2` architecture pivot to OCS2 / humanoid MPC

### v0.17.2: Architecture Pivot To OCS2 / Humanoid MPC

Objective:
- replace PPO-only escalation with a higher-confidence model-based control stack that can cover both standing push recovery and walking

Reason:
- `v0.17.1` failed the true hard-push gate
- the next priority is not more PPO reward/controller patching
- the next priority is adopting a more established humanoid control architecture with broader standing+walking coverage

Primary references:
- OCS2 for the optimal-control foundation
- 1X `wb-humanoid-mpc` for the humanoid control-stack structure
- OpenLoong only as a secondary MuJoCo integration reference

Deliverables:
- WildRobot adoption plan for OCS2 / humanoid MPC stack
- robot-model gap analysis:
  - URDF/MJCF and inertial data requirements
  - contact model and foot geometry mapping
  - state-estimation / kinematics / frame conventions
- controller decomposition for WildRobot:
  - reduced-order model
  - MPC footstep / CoM planner
  - whole-body or joint-space execution layer
- minimal simulation bring-up plan for standing

Exit criteria:
- the target architecture is defined clearly enough that implementation does not depend on ad hoc controller invention
- WildRobot-specific integration risks are listed explicitly
- first implementation milestone is scoped

### v0.17.3: Standing Recovery Bring-Up On The New Stack

Objective:
- get the new model-based stack to hold quiet standing and survive moderate pushes

Changes:
- integrate the robot model into the chosen control stack
- implement the minimal stabilizing controller path needed for quiet standing
- add moderate push tests before hard-push stepping
- keep validation in simulation first

Exit criteria:
- quiet standing is stable
- moderate push survival is credible
- controller signals are inspectable and physically coherent

### v0.17.4: Hard-Push Standing Recovery

Objective:
- solve the original standing push-recovery task on the new stack

Changes:
- activate step planning / step switching under the model-based controller
- evaluate against the same fixed ladder used for `v0.17.1`
- compare directly against both `v0.17.1` and `v0.14.6`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` improves materially over `v0.17.1`

### v0.17.5: Walking Bring-Up On The Same Stack

Objective:
- extend the same architecture from standing recovery to nominal walking

Changes:
- add walking references / velocity control within the model-based stack
- preserve standing recovery as a regression suite
- do not split into a separate unrelated walking architecture

Exit criteria:
- nominal walking exists without breaking standing recovery
- the same stack supports both behaviors

### v0.17.6: Optional RL Augmentation

Objective:
- add learning only after the model-based standing+walking stack is already credible

Changes:
- consider RL augmentation of footstep planning or tracking
- evaluate whether a `2407.17683`-style hybrid stack adds value on top of the adopted control foundation

Exit criteria:
- RL improves the adopted architecture rather than replacing it
- standing and walking both remain intact

### Deferred Side Branch: M3-Lite / Li-Style PPO Escalations

These are no longer the mainline plan.

They remain optional only if:
- the OCS2 / humanoid MPC adoption stalls for practical reasons, or
- a quick side experiment is needed while the new architecture is being integrated

---

## Decision Rules

These rules are part of the plan and should be followed strictly.

### If Quiet Standing Breaks

- do not add new reward terms
- return to the `legged_gym`-style baseline stack
- check observation normalization, action filtering, and PPO stability first

### If Hard-Push Success Plateaus Near The Old Baseline

- do not continue PPO-only escalation as the mainline
- use the result as the justification to pivot to the model-based stack

Applied result:
- this happened in `v0.17.1`
- the chosen next branch is the OCS2 / humanoid MPC architecture pivot

### If Step Count Stays Near Zero At The Boundary

- in the PPO baseline stage, keep the reward stack stable
- do not start a new motion-prior branch inside the same baseline
- if the architecture pivot has already been chosen, do not spend the mainline effort here

### If Step Count Rises But `term_height_low` Does Not Fall

Interpretation:
- the robot is attempting steps
- the problem is step quality, not step intent

Action:
- treat it as evidence that direct-action PPO is not closing the hard-recovery geometry gap
- do not keep adding local PPO/controller complexity as the mainline answer
- pivot to the model-based architecture

Applied result:
- this is the closest match to the `v0.17.1` failure signature

### If Both Bounded Escalations Fail

Only then consider them as side branches:
- M3-lite
- Li-style history
- motion-prior / recovery-teacher investigation

These are no longer the mainline plan.

### If Robustness Work Breaks Nominal Recovery

- reduce randomization strength
- reduce disturbance diversity
- restore the hard-push single-recovery basin first

This follows the current repo-wide rule that robustness must not mask a broken nominal skill.

---

## What We Are Explicitly Avoiding

To keep confidence high, v0.17 will not continue with:
- a walking objective
- endless PPO reward / controller patching after the hard gate already failed
- a large custom reward family for stepping
- a broad literature-driven exploration branch

The branch should now look like a disciplined architecture pivot:
- OCS2 for the control foundation
- 1X `wb-humanoid-mpc` for humanoid stack structure
- OpenLoong only as a secondary MuJoCo implementation reference

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
