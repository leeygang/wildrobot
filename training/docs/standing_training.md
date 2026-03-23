# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0  
**Status:** `v0.17.1` evaluated, `v0.17.2` scaffolding complete, `v0.17.3` next  
**Created:** 2026-03-20  
**Last updated:** 2026-03-22

---

## Purpose

This document defines the standing-first roadmap for WildRobot.

This is the **canonical training plan** for the active branch. Older
walking-first roadmap material is archived under
`training/docs/archive/learn_first_plan.md` and should be treated as historical
context, not the current execution plan.

Standing success means:
- stand quietly under zero command
- survive pushes
- step only when bracing is no longer enough
- return to upright neutral standing after recovery

Walking is still part of the broader goal, but for `v0.17` it comes only after
standing recovery is credible.

The key project-level decision after `v0.17.1` is:
- keep the current MJCF / MuJoCo / RL stack as the mainline
- follow the practical industry recipe for deployment:
  - simulator-native RL
  - direct policy deployment
  - transfer hardening through training, not controller replacement
- fix the proven RL recipe gaps from the closest hardware-adjacent baselines
  before adding teacher logic
- use step-target or capture-point logic only as bounded training-time
  assistance when needed

Active architecture docs:
- [docs/system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)
- [open_duck_op3_comparison.md](/home/leeygang/projects/wildrobot/training/docs/open_duck_op3_comparison.md)

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Current Standing Result

`v0.17.1` is now the final PPO standing baseline for this branch.

Best checked fixed-ladder result:
- `eval_medium = 71.6%`
- `eval_hard = 39.2%`
- `eval_hard/term_height_low_frac = 60.8%`

Verdict:
- `v0.17.1` failed the true gate
- it did **not** beat the old `v0.14.6` hard-push baseline of `60.84%`
- internal `eval_push/*` overstated true hard-push performance

Diagnosis:
- quiet standing and PPO stability were not the problem
- dominant failure remained `term_height_low`
- the gap is still step quality and post-touchdown arrest

That means the next step is not:
- endless PPO-only patching, or
- a full controller-stack pivot the hardware cannot exploit

The active answer is:
- **keep the deployed controller as a policy**
- **fix the RL recipe first**
- **add bounded teacher/scaffold signals only if the recipe-fixed rerun still
  needs them**

---

## External References

v0.17 now uses two categories of references:
- historical references for the completed PPO baseline
- active references for the industry-style RL mainline

### Historical Baseline References: `v0.17.0` - `v0.17.1`

#### Recipe A: `legged_gym` / Rudin et al. 2022

- Paper: https://proceedings.mlr.press/v164/rudin22a.html
- Code: https://github.com/leggedrobotics/legged_gym

Used for:
- PPO structure
- curriculum-first training
- compact reward design
- fixed eval discipline

#### Recipe B: Li et al. 2024 Cassie

- Project: https://xbpeng.github.io/projects/Cassie_Locomotion/index.html

Used for:
- treating standing recovery as a first-class skill
- disturbance exposure
- history as a later escalation if needed

### Active Mainline References: `v0.17.3+`

#### Direct RL recipe references

- MuJoCo Playground:
  - https://github.com/google-deepmind/mujoco_playground
- Unitree RL Lab:
  - https://github.com/unitreerobotics/unitree_rl_lab

These matter because they define the active recipe shape:
- simulator-native training
- direct policy deployment
- transfer hardening through training

#### Teacher / scaffold reference

- ModelBasedFootstepPlanning-IROS2024:
  - https://github.com/hojae-io/ModelBasedFootstepPlanning-IROS2024

Relevant because:
- it is the closest open-source reference for model-based step geometry helping
  model-free RL execution
- it matches the structural idea of bounded teacher assistance better than a
  full controller-stack pivot

#### Primary hardware-adjacent implementation reference

- adjacent Open Duck Mini project / runtime

Relevant because:
- it is much closer to WildRobot's hardware class than large torque-controlled
  humanoid stacks
- it keeps attention on what can actually deploy on current hardware

#### Long-term architecture references

Useful, but not the active mainline:
- OCS2
- `wb_humanoid_mpc`
- OpenLoong
- MJPC

---

## Repo Starting Point

The standing history still gives a clear baseline.

### What Worked

- `v0.13.8` established strong moderate-push standing robustness
- `v0.13.10` added explicit posture return
- `v0.14.6` pure PPO at the calibrated hard-push regime reached the best
  historical hard-push checkpoint:
  - `eval_push/success_rate = 60.84%` at `10N x 10`
  - dominant failure still `term_height_low`

### What Did Not Work

- `v0.14.7` privileged critic did not beat pure PPO
- `v0.14.8` actor-side capture-point observation did not beat pure PPO
- `v0.14.9` guide-only FSM underperformed and remained weakly engaged
- `v0.17.1` cleaned PPO baseline still failed the true hard gate

### Quantitative Boundary Data

Existing force sweep:

| Push Regime | Success Rate | Episode Length | Interpretation |
|---|---:|---:|---|
| `8N x 10` | `74.6%` | `431.8` | mostly bracing-manageable |
| `9N x 10` | `58.0%` | `382.9` | boundary / transition regime |
| `10N x 10` | `47.3%` | `346.7` | hard-push regime |

Planning implication:
- `8N x 10` is often recoverable by bracing
- `9N x 10` is where stepping should start to matter
- `10N x 10` is the real hard-push target

---

## Hardware Reality

Current hardware matters to the architecture choice.

Current constraints:
- position-controlled hobby servos
- `50 Hz` runtime
- one IMU
- four binary foot switches
- no ankle roll

Implications:
- keep the deployed controller simple
- avoid torque-control assumptions
- avoid planner-dependent runtime architectures
- use training-time structure before controller-stack replacement

Aggressive whole-body dynamic recovery is **not** the immediate target on this
hardware.

---

## Strategy

v0.17 now uses this sequence:

1. Use `v0.17.1` as the final PPO standing baseline result.
2. Keep MJCF / MuJoCo and the current RL infrastructure.
3. Fix the proven recipe gaps from the closest small-servo RL baselines:
   - residual action around a standing default pose
   - recovery-friendly termination
   - dominant alive / survival reward
   - domain randomization
   - action / IMU delay and sensor noise
4. Re-evaluate the hard-push gate on the recipe-fixed pure-RL branch.
5. Add bounded teacher or scaffold signals only if the geometry gap remains
   after that rerun.
6. Validate standing push recovery first.
7. Extend the same deployed policy stack to conservative walking later.

This avoids four failure modes:
- endless PPO-only reward patching without recipe discipline
- premature teacher / hierarchy work before the execution layer is fixed
- a controller pivot the hardware cannot use well
- splitting standing and walking into unrelated deployment stacks

---

## Core Design

### Task Definition

Task:
- command starts at zero velocity
- robot starts from neutral standing
- pushes arrive during the episode
- success means surviving the push and returning to upright standing

Later walking extension:
- add conservative forward / turn / stop commands on the same policy stack

### Observation Design

The deployed actor should stay runtime-compatible.

Required runtime observations:
- projected gravity
- base angular velocity
- heading-local base linear velocity
- joint positions
- joint velocities
- previous action
- foot contact signals

Optional training-only signals:
- teacher labels
- privileged capture-margin signals
- target-step annotations

Rule:
- do not make runtime deployment depend on planner-only signals unless the
  architecture decision is revisited explicitly

### Teacher / Scaffold Design

If used, teacher logic should stay small and removable.

Allowed first teacher forms:
- `need_step` labels
- target foot placement
- capture-point / LIP targets
- auxiliary reward alignment

Not the first branch:
- full runtime footstep planner
- full MPC
- full WBC

### Policy Design

The policy remains the deployed controller.

Responsibilities:
- joint-space execution
- posture stabilization
- recovery transients
- compensation for actuation limits and model mismatch
- later conservative walking behavior

### Training Recipe Design

The mainline should emphasize:
- residual actions around a standing default pose
- recovery-friendly termination
- dominant survival reward
- curriculum
- domain randomization
- actuator/noise/delay hardening
- fixed eval ladder
- optional privileged/teacher channels only after the recipe-fixed rerun if
  justified

### Evaluation

Keep the same standing ladder as the first gate:
- `eval_clean`
- `eval_easy`
- `eval_medium`
- `eval_hard`
- `eval_hard_long`

Primary decision metrics:
- `eval_medium`
- `eval_hard`
- `term_height_low_frac`

---

## Milestones

### `v0.17.2`: Planning / Scaffolding

Objective:
- architecture review, folder split, and doc/scaffolding work

Status:
- complete

### `v0.17.3`: Recipe-Fixed Pure RL

Objective:
- align the active branch with the closest proven RL recipe on similar hardware

Changes:
- switch standing action representation to residual targets around a default
  standing pose
- relax recovery-related termination so near-fall states remain trainable
- raise the alive / survival reward so survival dominates the reward landscape
- enable domain randomization on the standing branch
- enable action delay, IMU delay, and sensor noise
- make the runtime policy contract explicit
- separate runtime actor inputs from any privileged training inputs
- preserve export/runtime parity
- rerun the fixed ladder on the pure-RL branch before any teacher escalation

Exit criteria:
- the key recipe gaps are addressed on the pure-RL branch
- deployment remains policy-only
- the hard-push gate has been rerun without teacher logic
- the branch is ready for a teacher decision only if the pure-RL rerun still
  misses the gate

### `v0.17.4`: Conditional Step-Target Teacher / Scaffold Integration

Objective:
- add bounded geometry assistance only if the recipe-fixed pure-RL branch still
  misses the hard gate

Changes:
- optional capture-point or similar step-target heuristic
- teacher labels or auxiliary targets
- metrics and debug signals for teacher engagement

Exit criteria:
- teacher signals are bounded and inspectable
- the policy can train with or without them
- the teacher branch is compared directly against the recipe-fixed pure-RL
  baseline

### `v0.17.5`: Standing Push Recovery

Objective:
- solve standing push recovery on the RL mainline

Changes:
- train and evaluate the hard-push branch using the industry-style recipe
- compare directly against both `v0.17.1` and `v0.14.6`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` materially below `v0.17.1`

### `v0.17.6`: Walking On The Same Policy Stack

Objective:
- extend the same deployed stack to slow / conservative walking

Changes:
- forward walking
- stop / hold
- gentle turn
- keep standing push recovery as a regression suite

Exit criteria:
- one policy stack supports both standing recovery and conservative walking

### `v0.17.7`: Optional Stronger Teachers

Objective:
- add complexity only if the simpler branch clearly saturates

Possible directions:
- privileged teacher-student path
- imitation / motion-reference assistance
- MJPC-generated teacher data in simulation

---

## Decision Rules

### If Quiet Standing Breaks

- do not jump to bigger teacher logic first
- restore runtime-compatible standing behavior first

### If Recipe-Fixed Pure RL Meets The Hard Gate

- do not add teacher logic just because it was planned
- keep the simpler policy-only path as the mainline

### If Recipe-Fixed Pure RL Still Fails The Hard Gate

- add bounded step-target teacher signals before considering runtime hierarchy
  or a new controller architecture

### If Teacher Logic Helps In Training But Creates Runtime Dependency

- treat that as a design failure
- the active branch must keep the deployed artifact as a policy

### If The RL Mainline Still Fails After Bounded Teacher Support

Only then consider bigger escalations:
- stronger privileged teacher-student paths
- MJPC teacher in simulation
- revisit heavier controller stacks after a hardware change

---

## What We Are Explicitly Avoiding

For the active branch, do not make these the mainline:
- full OCS2 / humanoid MPC adoption on current hardware
- URDF-first migration as a prerequisite
- planner-dependent runtime controller execution
- runtime hierarchy before the recipe-fixed pure-RL branch is evaluated
- open-ended reward-family growth without recipe discipline

---

## Canonical Plan Status

This document is the active source of truth for training planning on the current
branch:
- standing push recovery is the first acceptance task
- walking is added only after the standing stack is credible
- archived walking-first plans are historical references only

If v0.17 succeeds, it gives the project:
- a deployable standing-recovery policy for current hardware
- a realistic path to conservative walking
- a much clearer answer about whether stronger teachers or heavier controllers
  are actually necessary
