# WildRobot System Architecture

**Status:** Transitioning at `v0.17.2`  
**Last updated:** 2026-03-22

---

## Purpose

This document captures the repo-level architecture shift for WildRobot.

The old mainline architecture was centered on:
- MuJoCo/MJX simulation
- JAX environment code
- direct-action PPO policies
- reward- and curriculum-driven standing and walking behaviors

That architecture remains in the repo and stays useful for:
- baseline training
- regression comparison
- evaluation tooling
- policy export/runtime experiments

But starting at `v0.17.2`, it is no longer the mainline answer for the core locomotion problem.

The new target architecture is:
- **model-based humanoid control as the foundation**
- **standing push recovery as the first validation task**
- **walking on the same control stack**
- **RL augmentation only after the model-based stack is already credible**

Primary references:
- OCS2: <https://github.com/leggedrobotics/ocs2>
- OCS2 robotic examples: <https://leggedrobotics.github.io/ocs2/robotic_examples.html>
- 1X `wb-humanoid-mpc`: <https://github.com/1x-technologies/wb-humanoid-mpc>
- OpenLoong Dyn-Control: <https://github.com/loongOpen/OpenLoong-Dyn-Control>

---

## Why The Architecture Is Changing

`v0.17.1` answered the PPO baseline question:
- the cleaned standing PPO recipe was stable
- internal `eval_push/*` looked strong
- fixed-ladder hard-push evaluation still failed the true gate

The remaining gap was not:
- PPO instability
- poor quiet standing
- lack of disturbance exposure

The remaining gap was:
- hard-push step quality
- touchdown geometry
- post-touchdown arrest of collapse

That makes continued PPO-only escalation a lower-confidence path than adopting a more established humanoid control architecture.

---

## Current Mainline Architecture

### Training path today

Current active training stack:
- `training/envs/wildrobot_env.py`
- `training/core/training_loop.py`
- `training/algos/ppo/ppo_core.py`
- YAML-driven task configs in `training/configs/`

Current control assumption:
- policy outputs direct actuator targets or residual-style action commands
- stabilization is learned primarily through reward shaping and curriculum
- stepping behavior is induced, not planned

### Assets and runtime path today

Current robot/model path:
- active robot assets in `assets/v2/`
- main model source: `assets/v2/wildrobot.xml`
- generated MuJoCo config: `assets/v2/mujoco_robot_config.json`
- runtime calibration/config path: `runtime/configs/runtime_config_v2.json`

This means the repo already has:
- MJCF geometry
- inertias embedded in MJCF
- actuator ordering
- contact sensor path
- runtime actuator calibration

But it does **not** yet have a model-based locomotion stack organized around:
- reduced-order dynamics
- contact-phase planning
- MPC
- whole-body execution

---

## Target Mainline Architecture

The target stack is layered.

```text
Robot model + kinematics/dynamics
    ->
Reduced-order locomotion model
    ->
MPC planner (CoM / footsteps / contact timing)
    ->
Whole-body or joint-space execution layer
    ->
Simulation + evaluation harness
    ->
Optional RL augmentation
```

### Layer 1: Robot model

Responsibilities:
- canonical robot description
- inertias and geometry
- joint limits and actuation limits
- foot/contact geometry
- frame definitions

Expected outputs:
- a model path usable by the control stack
- frame-consistent kinematics and dynamics

### Layer 2: Reduced-order model

Responsibilities:
- simplified dynamics for locomotion planning
- CoM / momentum / support reasoning
- contact-mode abstraction

Likely candidate families:
- LIP / capture-point
- ALIP
- centroidal approximations depending on the chosen stack

### Layer 3: MPC planner

Responsibilities:
- receding-horizon stabilization
- footstep / contact planning
- CoM and momentum regulation
- standing push-recovery and walking planning under one framework

### Layer 4: Execution layer

Responsibilities:
- realize planned motion on the full robot
- map planner outputs to joint-level commands
- enforce robot and contact constraints

Expected forms:
- whole-body control if the adopted stack requires it
- or a narrower joint-space tracking layer if that is the cleanest viable first step for WildRobot

### Layer 5: Evaluation harness

Responsibilities:
- fixed standing push ladder
- quiet-standing regression
- walking regression later
- direct comparison against the old PPO baselines

### Layer 6: Optional RL augmentation

Not the foundation.

Only add learning here after the model-based stack already works for:
- standing recovery
- nominal walking

Likely roles for RL later:
- footstep refinement
- tracking compensation
- model mismatch correction

---

## Architecture Principles

1. **Standing-first validation, shared architecture**
   - Standing recovery is the first task.
   - Walking must reuse the same main controller stack.

2. **Model-based first, learning second**
   - Planning and stabilization should not depend on PPO rediscovering contact mechanics.

3. **Keep the old PPO stack as a baseline path**
   - Do not delete it.
   - It remains the comparison/control line.

4. **Use external architecture references directly**
   - OCS2 for structure
   - 1X for humanoid stack conventions
   - OpenLoong only for MuJoCo implementation ideas

5. **Do not invent a custom hybrid stack prematurely**
   - First adopt the established model-based structure.
   - Only then decide whether RL augmentation is worth adding.

---

## What We Reuse From The Current Repo

These parts remain valuable:

- `assets/v2/wildrobot.xml`
  - current source of geometry and inertials
- `assets/v2/mujoco_robot_config.json`
  - generated MuJoCo robot metadata
- `training/eval/`
  - especially the standing fixed-ladder evaluation logic
- `training/wandb/` and current metrics/eval conventions
- `runtime/configs/runtime_config_v2.json`
  - actuator calibration and hardware-side constraints
- standing push definitions and regression targets from `training/docs/standing_training.md`

These should be preserved as regression infrastructure:
- `v0.14.6` hard-push baseline
- `v0.17.1` PPO standing baseline

---

## What Stops Being Mainline

These paths stay in the repo but are no longer the mainline answer:

- PPO-only standing escalation after `v0.17.1`
- M3-lite as the mainline architecture
- Li-style history as the mainline architecture
- reward-family expansion as the mainline solution to locomotion

They become:
- side experiments
- local diagnostics
- fallback branches only if the model-based adoption stalls

---

## Expected Repo Shape After The Pivot

This is the intended organization direction, not a claim that it already exists.

```text
docs/
  system_architecture.md

training/docs/
  standing_training.md
  ocs2_humanoid_mpc_adoption.md

control/ or training/control/
  robot_model/
  reduced_model/
  mpc/
  execution/
  adapters/
```

Suggested boundary:
- planning/control code should not be buried inside PPO-specific modules
- simulation adapters should remain reusable
- training code should become a client of the controller path, not the owner of the controller design

---

## Milestone Mapping

### `v0.17.2`
- architecture adoption plan
- WildRobot gap analysis
- controller decomposition

### `v0.17.3`
- quiet standing / moderate-push bring-up on the new stack

### `v0.17.4`
- hard-push standing recovery on the new stack

### `v0.17.5`
- walking bring-up on the same stack

### `v0.17.6`
- optional RL augmentation only if the model-based stack is already credible

---

## Decision Record

Mainline decision as of `2026-03-22`:
- **Adopt OCS2 / humanoid MPC architecture as the new mainline**
- **Use 1X `wb-humanoid-mpc` as the closest concrete humanoid reference**
- **Use OpenLoong as a secondary MuJoCo reference only**
- **Keep standing recovery as the first validation task**
- **Do not continue PPO-only escalation as the mainline after `v0.17.1`**
