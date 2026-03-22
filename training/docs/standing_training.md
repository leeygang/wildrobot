# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0  
**Status:** `v0.17.1` evaluated, `v0.17.2` scaffolding complete, `v0.17.3` next  
**Created:** 2026-03-20  
**Last updated:** 2026-03-22

---

## Purpose

This document defines the standing-first roadmap for WildRobot.

Standing success means:
- stand quietly under zero command
- survive pushes
- step only when bracing is no longer enough
- return to upright neutral standing after recovery

Walking is still part of the broader goal, but for `v0.17` it comes only after
standing recovery is credible.

The key project-level decision after `v0.17.1` is:
- keep the current MJCF / MuJoCo / RL stack
- add explicit step geometry through a lightweight planner
- do **not** make full OCS2 / humanoid MPC adoption the current mainline on
  this hardware

Active architecture docs:
- repo-level architecture:
  [docs/system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)
- active hybrid plan:
  [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)

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

This result invalidated two different ideas:
- continuing PPO-only escalation as the mainline
- jumping straight to full humanoid MPC on current hobby hardware

The active answer is now the middle path:
- **simple planner for step geometry**
- **RL for execution through the existing position-servo stack**

---

## External References

v0.17 now uses two categories of references:
- historical references for the completed PPO baseline
- active references for the hybrid planner + RL branch

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

#### Practical control pattern: simple footstep planner + learned tracking

What we take from this pattern:
- planner provides when/where to step
- RL handles execution on messy real actuators

This is the active architecture choice for current WildRobot hardware.

#### Servo-humanoid implementation style: OP3-like planner + kinematic execution

Relevant because:
- it is much closer to current WildRobot hardware than torque-control humanoid
  MPC stacks
- it matches the reality of position-servo execution and simpler sensing

#### Secondary simulation-side references

Useful as reference only:
- OpenLoong for MuJoCo humanoid module structure
- MuJoCo MPC as a possible simulation teacher later

#### Long-term architecture references

Useful, but not the current mainline:
- OCS2
- `wb_humanoid_mpc`

They remain valuable as long-term architecture inspiration, not as the active
implementation contract for `v0.17.3+`.

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
- full whole-body MPC / WBC is not a good near-term fit
- URDF / Pinocchio / OCS2 plumbing is too much cost for the immediate branch
- pure PPO still needs help with explicit step geometry

So the active plan is deliberately narrower:
- planner gives conservative recovery-step intent
- RL tracks that intent through the existing actuation stack

Aggressive whole-body dynamic recovery is **not** the immediate target on this
hardware.

---

## Strategy

v0.17 now uses this sequence:

1. Use `v0.17.1` as the final PPO baseline result.
2. Keep MJCF / MuJoCo and the current RL infrastructure.
3. Add a lightweight reduced-order planner for step geometry.
4. Validate quiet standing first.
5. Validate standing push recovery next.
6. Extend the same stack to slow / conservative walking.

This avoids three failure modes:
- endless PPO-only reward patching
- architecture pivots the hardware cannot exploit
- splitting standing and walking into unrelated controller stacks

---

## Core Design

### Task Definition

Task:
- command starts at zero velocity
- robot starts from neutral standing
- pushes arrive during the episode
- success means surviving the push and returning to upright standing

Later walking extension:
- add conservative forward / turn / stop commands on the same stack

### Observation Design

Keep the actor close to the current standing stack and add only planner signals
that are genuinely useful.

Required observations:
- projected gravity
- base angular velocity
- heading-local base linear velocity
- joint positions
- joint velocities
- previous action
- foot contact signals
- planner state / mode
- step target or local touchdown target when active

Guidelines:
- do not remove contact information from the actor
- do not add a gait clock as the first fix
- do not explode the observation space with speculative planner internals

### Planner Design

First planner should be small and inspectable.

Required abilities:
- detect when the no-step basin is exceeded
- pick the swing foot
- compute a conservative landing target
- expose planner activation and step intent clearly

First model family:
- capture-point / LIP

Allowed escalation only if required:
- ALIP-lite

Not the first branch:
- full OCS2
- full humanoid MPC
- full WBC

### Policy / Execution Design

The RL side remains the execution engine.

Responsibilities:
- turn planner intent into servo-friendly joint behavior
- stabilize the body around the planned step
- absorb model mismatch and actuation quirks

This is still an RL branch, but not an RL-from-scratch stepping-geometry branch.

### Reward Design

Keep reward changes bounded.

Allow:
- planner alignment terms
- execution-quality terms
- recovery success terms already consistent with the standing task

Avoid:
- large new reward families for every phase of stepping
- turning the branch into another open-ended reward-design project

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

Important note:
- the earlier OCS2-mainline framing from this stage is now superseded by the
  hardware-fit review

### `v0.17.3`: Planner / Policy Integration Groundwork

Objective:
- integrate the reduced-order planner into the existing stack without claiming
  stable standing yet

Changes:
- planner state extraction
- planner outputs in metrics / debug signals
- mode manager skeleton
- actor observation plumbing for planner targets
- keep execution servo-friendly

Exit criteria:
- planner path initializes cleanly
- planner outputs are inspectable
- RL path can consume planner outputs without crashing

### `v0.17.4`: Quiet Standing

Objective:
- quiet standing on the hybrid planner + RL stack

Changes:
- validate stable neutral posture hold
- keep planner mostly inactive during nominal standing
- verify no regression in clean-standing behavior

Exit criteria:
- quiet standing is repeatable
- planner does not destabilize idle standing

### `v0.17.5`: Standing Push Recovery

Objective:
- solve standing push recovery with a real step trait on the hybrid stack

Changes:
- planner-triggered stepping under moderate pushes
- extend to hard-push boundary
- compare against both `v0.17.1` and `v0.14.6`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` materially better than `v0.17.1`

### `v0.17.6`: Walking On The Same Stack

Objective:
- extend the same stack to slow / conservative walking

Changes:
- forward walking
- stop / hold
- gentle turn
- keep standing push recovery as a regression suite

Exit criteria:
- one stack supports both standing recovery and slow commanded walking

### `v0.17.7`: Optional Teacher / Privileged Extensions

Objective:
- add complexity only if the simpler hybrid stack works but saturates

Possible directions:
- privileged teacher-student training
- MJPC-generated references in simulation
- more structured planner-state supervision

Not a license to abandon the active branch early.

---

## Decision Rules

### If Quiet Standing Breaks

- do not add a large new reward family first
- check planner gating, observation plumbing, and sign conventions first
- restore clean-standing stability before continuing

### If Planner Activity Stays Near Zero At The Boundary

Interpretation:
- the planner threshold or state abstraction is wrong

Action:
- debug planner state and triggering first
- do not jump straight to bigger controllers or broader rewards

### If Planner Activity Rises But `term_height_low` Does Not Fall

Interpretation:
- step intent exists
- execution quality is still insufficient

Action:
- improve planner target quality or RL tracking
- do not conclude that the active architecture is wrong after one occurrence
- only escalate after repeated evidence across the fixed ladder

### If The Hybrid Branch Still Fails The Hard Gate

Only then consider bigger escalations:
- privileged teacher-student path
- MJPC teacher in simulation
- revisit stronger model-based stacks after a hardware change

### If Walking Breaks Standing Recovery

- walking is subordinate to standing recovery in this roadmap
- reduce walking ambition before weakening the standing recovery stack

---

## What We Are Explicitly Avoiding

For the active branch, do not make these the mainline:
- full OCS2 / humanoid MPC adoption on current hardware
- URDF-first migration as a prerequisite
- endless PPO-only reward/controller patching
- full custom WBC stack
- walking-first redesign before standing recovery improves

---

## Relationship To The Main Training Plan

This document defines the standing-first roadmap.

It remains narrower than the broader walking-first plans elsewhere in the repo:
- standing push recovery is the first acceptance task
- walking is added only after the standing stack is credible

If v0.17 succeeds, it gives the project:
- a deployable standing-recovery strategy for current hardware
- a realistic path to slow commanded walking
- a clearer basis for deciding whether stronger teachers or bigger controllers
  are actually needed
