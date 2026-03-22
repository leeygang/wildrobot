# WildRobot System Architecture

**Status:** Active mainline updated at `v0.17.3`  
**Last updated:** 2026-03-22

---

## Purpose

This document captures the repo-level architecture decision after the `v0.17.1`
standing result and the subsequent hardware-fit review.

The repo keeps two different paths:
- the existing PPO/AMP stack in `training/` as baseline and regression
- a new hybrid controller path under `control/` for current-hardware standing
  recovery and walking

The active mainline for current WildRobot hardware is now:
- **MJCF / MuJoCo remains the main simulation path**
- **a lightweight reduced-order footstep planner provides stepping geometry**
- **an RL tracking policy or servo-friendly execution layer handles the actual
  position-servo behavior**
- **standing push recovery is validated first**
- **walking is added on the same stack later**

Long-term references remain useful, but they are no longer the active
implementation target for `v0.17.3+`:
- OCS2
- `wb_humanoid_mpc`
- URDF-first migration

Active planning docs:
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)

Deferred reference docs:
- [ocs2_humanoid_mpc_adoption.md](/home/leeygang/projects/wildrobot/training/docs/ocs2_humanoid_mpc_adoption.md)
- [URDF_migration_plan.md](/home/leeygang/projects/wildrobot/docs/URDF_migration_plan.md)

---

## Why The Architecture Changed

`v0.17.1` answered the PPO baseline question:
- quiet standing was stable
- PPO optimization itself was not the blocker
- internal push probes looked strong
- the fixed ladder still failed the true hard-push gate

The remaining gap was:
- hard-push step quality
- touchdown geometry
- post-touchdown arrest of collapse

The first follow-up plan was a full OCS2 / humanoid MPC pivot. After review,
that is no longer the active mainline for current hardware because WildRobot is
not in the same robot class as the intended MPC references.

Current hardware constraints that matter:
- position-controlled hobby servos, not torque control
- `50 Hz` runtime loop
- one IMU and binary foot switches
- no ankle roll
- limited dynamic torque margin

So the architecture change is now more specific:
- **do not ask PPO to rediscover stepping geometry from scratch**
- **do not jump to a torque-control humanoid MPC stack the hardware cannot use**
- **add a small model-based planner inside the existing MuJoCo / RL workflow**

---

## Current Mainline Architecture

The new mainline is a hardware-fit hybrid.

```text
MJCF / MuJoCo robot model
    ->
State extraction and contact abstraction
    ->
Reduced-order planner (capture-point / LIP / ALIP-lite)
    ->
Mode manager (stand / recover / walk)
    ->
RL tracking policy or servo-friendly execution layer
    ->
Simulation evaluation and hardware runtime
```

### Layer 1: Simulation and robot model

Responsibilities:
- keep `assets/v2/wildrobot.xml` as the main simulation model
- preserve current MuJoCo contact tuning and actuator semantics
- provide state and contact signals to the controller path

### Layer 2: Reduced-order planner

Responsibilities:
- decide whether a step is needed
- choose the swing foot
- compute a conservative touchdown target
- provide simple CoM / capture-point style stabilization guidance

Expected model family:
- capture-point / LIP first
- ALIP-lite only if the simpler model is clearly insufficient

### Layer 3: Mode manager

Responsibilities:
- explicit operating modes:
  - `stand`
  - `recover`
  - `walk`
- gate when the planner is active
- keep behavior conservative and inspectable on hobby hardware

### Layer 4: RL tracking / execution

Responsibilities:
- turn planner intent into actual servo-friendly joint behavior
- absorb the mismatch between the simple planner and the real position-servo
  robot dynamics
- preserve learned stabilization where direct planning would be brittle

This is the key difference from the old PPO-only path:
- the planner provides step geometry
- RL provides execution and robustness on the existing actuation model

### Layer 5: Evaluation harness

Responsibilities:
- reuse the standing ladder already in `training/eval/`
- compare directly against:
  - `v0.14.6`
  - `v0.17.1`
- later add walking regression without replacing the standing tests

### Layer 6: Optional teachers / advanced references

Not the mainline.

Possible later roles:
- MJPC as a simulation-only teacher
- privileged teacher-student training
- long-term OCS2 / humanoid MPC revisit after a hardware upgrade

---

## Architecture Principles

1. **Hardware-fit first**
   - The mainline controller must match the current servo, sensing, and runtime
     constraints.

2. **Planner provides geometry, RL provides execution**
   - Stepping geometry should not be rediscovered from scratch by PPO.
   - Position-servo execution should not be forced into a torque-control stack.

3. **Keep MJCF / MuJoCo as the mainline sim path**
   - Do not force a URDF-first migration for the active branch.

4. **Keep the old PPO stack as a baseline**
   - `training/` remains the comparison/control line.

5. **Use heavyweight model-based stacks as references, not dependencies**
   - OCS2, `wb_humanoid_mpc`, OpenLoong, and MJPC remain useful references.
   - They do not define the immediate implementation contract for current
     hardware.

---

## What We Reuse From The Current Repo

These parts remain mainline assets:
- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- current scene files in `assets/v2/`
- `training/envs/`
- `training/core/`
- `training/eval/`
- `runtime/configs/runtime_config_v2.json`
- the standing regression targets in
  [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)

These remain baseline comparison lines:
- `v0.14.6` hard-push standing baseline
- `v0.17.1` final PPO standing baseline

These remain useful but are no longer active mainline dependencies:
- OCS2
- `wb_humanoid_mpc`
- URDF-first migration work

---

## What Stops Being Mainline

These paths stay documented, but they are not the active implementation plan:
- full OCS2 / humanoid MPC adoption on current hardware
- URDF-first migration as the mainline path
- PPO-only standing escalation after `v0.17.1`
- M3-lite as the mainline branch
- unbounded reward-family growth as the main solution

---

## Folder Ownership

The controller path should still live in **top-level `control/`**, not
`training/control/`.

Ownership rule:
- `control/` owns planner/controller logic
- `training/` owns experiments, PPO baselines, evals, and integrations

Practical boundary:
- `training/` may depend on `control/`
- `control/` should not depend on PPO-specific training code

What belongs in `control/`:
- robot model interfaces
- frame and contact conventions
- reduced-order planner code
- mode management
- servo-friendly execution helpers
- simulation/runtime adapters
- reserved room for future advanced planning experiments

What stays in `training/`:
- PPO baseline stack
- reward and curriculum experiments
- training configs
- evaluation ladder
- W&B / logging / checkpointing
- wrappers that call into `control/`

---

## Expected Repo Shape

```text
docs/
  system_architecture.md
  URDF_migration_plan.md            # deferred / conditional

training/docs/
  standing_training.md
  footstep_planner_rl_adoption.md
  ocs2_humanoid_mpc_adoption.md     # deferred / reference only

control/
  README.md
  robot_model/
  reduced_model/
  mpc/                              # reserved for advanced planner work
  execution/
  adapters/
```

The current `control/mpc/` placeholder can remain, but the active `v0.17.3+`
work should start with a simpler reduced-order planner, not full humanoid MPC.

---

## Milestone Mapping

### `v0.17.2`
- architecture review
- folder split and scaffolding
- documentation
- explicit rejection of full OCS2 adoption as the near-term mainline

### `v0.17.3`
- hybrid planner + RL integration groundwork
- planner state, mode, and target signals wired into simulation
- no stable-standing claim yet

### `v0.17.4`
- quiet standing on the hybrid stack

### `v0.17.5`
- standing push recovery with step trait on the hybrid stack

### `v0.17.6`
- slow / conservative walking on the same stack

### `v0.17.7`
- optional teacher or privileged extensions only if the hybrid stack works

---

## Decision Record

Mainline decision as of `2026-03-22`:
- **Use a simple footstep planner + RL tracking as the active path**
- **Keep MJCF / MuJoCo as the mainline sim and evaluation stack**
- **Keep OCS2 / `wb_humanoid_mpc` as long-term references only**
- **Keep standing push recovery as the first acceptance task**
- **Use walking as the next extension on the same stack**
