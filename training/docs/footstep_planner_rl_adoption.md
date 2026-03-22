# WildRobot Footstep Planner + RL Adoption Plan

**Status:** Active mainline plan for `v0.17.3+`  
**Last updated:** 2026-03-22

---

## Purpose

This document defines the active post-`v0.17.1` implementation path for
WildRobot:
- keep the existing MJCF / MuJoCo / JAX training stack
- add a lightweight reduced-order footstep planner
- use RL to execute the plan through the existing position-servo action path

This replaces the earlier idea of making OCS2 / `wb_humanoid_mpc` the immediate
mainline. Those stacks remain useful references, but not the current
implementation target for this hardware.

Read together with:
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Why This Is The New Mainline

`v0.17.1` showed:
- PPO can stabilize quiet standing
- PPO can learn useful recovery behavior
- the true hard-push gate still fails
- the remaining failure is still mostly `term_height_low`

The practical diagnosis is:
- pure PPO did not discover good enough step geometry under the true hard pushes
- current WildRobot hardware is a poor fit for full humanoid MPC / WBC

That makes the active solution:
- add a small planner that provides stepping geometry
- keep RL in charge of messy servo-compatible execution

This is the highest-confidence path that still fits:
- position-controlled hobby servos
- `50 Hz` runtime
- one IMU
- binary foot switches
- MJCF / MuJoCo simulation

---

## Hardware-Fit Constraints

Current hardware assumptions:
- position-controlled hobby servos
- no torque control
- one IMU
- four binary foot switches
- no ankle roll
- lightweight runtime budget

Implications:
- do not build the active branch around torque commands
- do not require whole-body force control
- do not assume precise contact wrench estimation
- do not assume a `200-1000 Hz` controller stack

This is why the new branch is:
- planner-guided
- servo-friendly
- inspectable
- conservative

---

## Target Architecture

```text
MuJoCo state and contacts
    ->
State abstraction
    ->
Reduced-order footstep planner
    ->
Planner outputs:
  need_step / support foot / touchdown target / optional CoM target
    ->
RL tracking policy
    ->
Joint position targets
    ->
Existing MuJoCo + runtime actuation path
```

### 1. State abstraction

Purpose:
- convert MuJoCo state into planner-friendly quantities
- keep the planner small and explicit

Expected signals:
- base orientation and angular velocity
- heading-local base velocity
- support foot / foot-contact state
- nominal stance frame
- capture-point or equivalent balance quantity

### 2. Reduced-order planner

Purpose:
- decide whether the current state is inside or outside the no-step basin
- choose the swing foot
- produce a conservative landing target
- optionally provide simple CoM guidance

Expected first model:
- capture-point / LIP

Allowed escalation only if needed:
- ALIP-lite

Not the first target:
- full humanoid MPC
- full WBC
- large custom contact optimizer

### 3. Mode manager

Modes:
- `stand`
- `recover`
- `walk`

Purpose:
- keep behavior explicit and debuggable
- let standing and walking share the same controller contract later

### 4. RL tracking policy

Purpose:
- execute planner intent through the current action interface
- compensate for the mismatch between the simple model and real servo dynamics
- preserve the parts of stabilization that PPO already learned well

The policy should not be responsible for inventing the footstep geometry from
scratch anymore.

---

## What The Planner Must Provide

Minimum planner outputs:
- `planner_active`
- `mode`
- `need_step`
- `swing_foot`
- touchdown target in a local frame
- optional scalar or vector indicating capture urgency

Nice-to-have outputs:
- target step length / width
- target support-foot switch timing
- optional CoM target or velocity target

These outputs should be:
- logged
- testable
- easy to visualize

---

## What The RL Policy Still Owns

The RL side remains responsible for:
- joint-space execution quality
- posture stabilization around the plan
- landing softness and recovery transients
- compensating for actuation limits and model mismatch
- eventually commanded walking style on the same hardware

This is why the branch remains an RL branch, not a pure controller rewrite.

---

## Reuse From The Current Repo

Reuse directly:
- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- current MuJoCo scenes
- `training/envs/wildrobot_env.py`
- disturbance generation and fixed eval ladder
- current runtime calibration/config path
- current policy export path

Reuse as baselines:
- `v0.14.6`
- `v0.17.1`

Do not delete:
- PPO-only training path
- current eval ladder
- current runtime path

---

## Implementation Plan

### `v0.17.3` - Planner / Policy Integration Groundwork

Deliver:
- reduced-order planner skeleton
- planner state extraction from the existing env
- planner outputs in `env_info` / metrics
- mode manager skeleton
- actor observation plumbing for planner targets
- no stable-standing claim yet

Exit criteria:
- planner path initializes cleanly
- planner outputs are inspectable in simulation
- the RL path can consume planner outputs without crashing

### `v0.17.4` - Quiet Standing

Deliver:
- quiet standing on the hybrid stack
- planner mostly inactive in nominal standing
- stable posture hold without breaking clean-standing regression

Exit criteria:
- clean-standing success remains high
- planner does not destabilize the idle regime

### `v0.17.5` - Standing Push Recovery With Step Trait

Deliver:
- planner-triggered stepping under moderate pushes
- recovery stepping at the hard-push boundary
- fixed-ladder comparison against `v0.17.1` and `v0.14.6`

Exit criteria:
- `eval_medium > 75%`
- `eval_hard > 60%`
- `term_height_low_frac` materially below `v0.17.1`

### `v0.17.6` - Walking On The Same Stack

Deliver:
- slow / conservative forward walking
- stop / hold
- gentle turn
- standing push recovery stays intact

Exit criteria:
- one stack handles both standing recovery and slow commanded walking

### `v0.17.7` - Optional Teacher / Privileged Extensions

Possible directions only if needed:
- privileged planner-state training
- teacher-student distillation
- MJPC-generated references in simulation

Not a license to abandon the simpler mainline early.

---

## Diagnostics To Add

Planner diagnostics:
- planner active fraction
- `need_step` count
- swing-foot choice
- touchdown target
- support-foot switches

Recovery diagnostics:
- first step latency
- touchdown count
- support-foot changes
- post-push velocity
- `term_height_low_frac`

Walking diagnostics later:
- command tracking
- stop/turn transitions
- planner activation during gait changes

---

## Immediate Next Actions

1. Define the reduced-order planner interface in `control/`.
2. Add planner-state extraction from the current MuJoCo env.
3. Decide which planner outputs enter the actor observation.
4. Add logging and tests for planner engagement.
5. Keep the standing fixed ladder as the first hard acceptance test.

---

## Out Of Scope For The Active Branch

Do not make these the mainline right now:
- full OCS2 adoption
- URDF-first migration
- full humanoid MPC / WBC
- endless PPO reward-family expansion without planner structure
- walking-first redesign before standing recovery improves
