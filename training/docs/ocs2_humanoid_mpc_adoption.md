# WildRobot OCS2 / Humanoid MPC Adoption Plan

**Status:** Planned at `v0.17.2`  
**Last updated:** 2026-03-22

---

## Purpose

This document defines the WildRobot-specific adoption plan for the new mainline control architecture:
- OCS2 as the optimal-control foundation
- 1X `wb-humanoid-mpc` as the primary humanoid implementation reference
- OpenLoong as a secondary MuJoCo implementation reference

This is the concrete follow-up to the `v0.17.1` standing baseline failure and should be read together with:
- [standing_training.md](/home/leeygang/projects/wildrobot/training/docs/standing_training.md)
- [system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)

---

## Why `v0.17.1` Was Insufficient

`v0.17.1` was the correct PPO baseline to run, but it was not sufficient as the mainline solution.

What it proved:
- quiet standing can be kept stable
- PPO optimization was not the dominant blocker
- the cleaned standing recipe can survive nontrivial pushes

What it failed to prove:
- reliable hard-push recovery at the true fixed gate
- a credible path from standing recovery to walking on the same stack

Observed result:
- internal `eval_push/*` looked strong
- fixed ladder still failed the true hard-push gate
- dominant failure remained `term_height_low`

Interpretation:
- the remaining problem is not “make PPO more stable”
- the remaining problem is “use a control architecture that handles stepping, contact switching, and recovery geometry more directly”

---

## Why OCS2 / 1X Is The Chosen Mainline

### OCS2

OCS2 is the chosen control foundation because it provides a more established structure for:
- switched-system optimal control
- contact-phase reasoning
- model/cost/constraint decomposition
- receding-horizon control

This is a better fit for WildRobot’s current standing+walking goal than continuing direct-action PPO as the only mainline.

### 1X `wb-humanoid-mpc`

1X `wb-humanoid-mpc` is the chosen humanoid implementation reference because it is closer to the actual target stack:
- humanoid-specific MPC organization
- whole-body execution assumptions
- one stack for both standing-like balance and locomotion

This is more directly aligned with WildRobot’s medium-term objective than a standing-only PPO branch.

### Why Not OpenLoong As The Main Reference

OpenLoong remains useful, but as a secondary reference only.

Reason:
- it is more tightly coupled to its own robot and controller conventions
- it is more useful as a MuJoCo implementation reference than as the architectural source of truth

---

## WildRobot-Specific Model And Integration Gaps

WildRobot already has a lot of usable robot data, but not in the exact form this stack will want.

### What exists today

Current active robot/model assets:
- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- `assets/v2/scene.xml`
- `assets/v2/scene_flat_terrain.xml`
- `runtime/configs/runtime_config_v2.json`

These provide:
- MJCF geometry and inertials
- current actuator ordering
- runtime calibration/config path
- current foot/contact instrumentation path

### Main gaps to close

1. **Canonical model path for the new controller stack**
   - WildRobot currently has a checked-in MJCF path in `assets/v2/`
   - It does **not** yet have a clearly maintained URDF/control-model path for the new architecture

2. **Frame and convention audit**
   - base frame
   - heading-local frame
   - foot frames
   - contact frame conventions
   - sign conventions for roll/pitch/yaw, foot lateral offsets, and joint directions

3. **Contact geometry audit**
   - exact support polygon definition
   - toe/heel geometry mapping
   - contact points used by the controller vs by MuJoCo

4. **Actuation mapping audit**
   - joint names and order
   - limits
   - nominal joint posture
   - runtime actuator constraints vs simulation actuator assumptions

5. **Execution-layer choice**
   - determine whether the first WildRobot implementation should use:
     - whole-body control, or
     - a narrower joint-space tracking layer

6. **State-estimation interface definition**
   - what the controller reads from simulation first
   - what the runtime equivalent would be later

---

## Required Model Artifacts

These artifacts are required before the adopted stack can be implemented cleanly.

### Required source artifacts

1. **Canonical robot description for control**
   - preferred: URDF or equivalent model path suitable for the target control stack
   - must stay synchronized with `assets/v2/wildrobot.xml`

2. **Inertial data**
   - link masses
   - COM locations
   - inertias
   - validated against the active MJCF

3. **Joint metadata**
   - names
   - order
   - limits
   - neutral standing pose
   - actuator mapping

4. **Contact geometry**
   - left/right foot support geometry
   - toe/heel contact points
   - controller-facing support polygon definition

5. **Frame conventions**
   - base frame
   - pelvis/root frame
   - world frame
   - heading-local frame
   - left/right foot frames

6. **Simulation/controller adapter spec**
   - what kinematic and dynamic quantities the new controller path expects
   - where they come from in MuJoCo
   - how they will map later to runtime signals

### Immediate WildRobot action items

1. Define whether `assets/v2/wildrobot.xml` remains the single source of truth for inertials.
2. Add or generate the control-model artifact expected by the adopted stack.
3. Create a written frame-convention map for WildRobot.
4. Document the support geometry used for standing and stepping.

---

## Proposed Controller Decomposition

The WildRobot controller stack should be decomposed explicitly.

### 1. Reduced-order model layer

Responsibilities:
- represent locomotion-relevant simplified dynamics
- support balance and step planning
- expose quantities like support state, CoM evolution, and contact phase

Candidate forms:
- LIP/capture-point style for first reasoning
- ALIP or centroidal approximation if required by the chosen implementation path

WildRobot requirement:
- this layer must be simple enough to validate first in simulation
- but strong enough to support both standing recovery and walking planning

### 2. MPC footstep / CoM planner

Responsibilities:
- plan or refine CoM motion
- plan support switching
- generate footstep targets
- stabilize standing under pushes
- serve as the main locomotion-planning mechanism for walking later

Standing-first expectation:
- quiet standing
- moderate push rejection
- recovery-step planning under hard pushes

Walking expectation later:
- nominal forward walking
- preserved standing recovery behavior

### 3. Whole-body or joint-space execution layer

Responsibilities:
- realize planner outputs on the full WildRobot model
- respect joint limits and contact constraints
- maintain stable posture while tracking the planner

Decision needed:
- start with a narrower joint-space execution layer if that materially lowers adoption risk
- only adopt full WBC immediately if the chosen reference stack makes that the smaller-risk path

Default recommendation:
- choose the smallest execution layer that preserves the planner/controller contract cleanly
- do not prematurely implement a custom full-body controller if a simpler execution layer is enough for the first standing bring-up

---

## What Will Be Reused From The Current Repo

These parts should be reused directly or treated as regression infrastructure.

### Reuse directly

- `assets/v2/wildrobot.xml`
- `assets/v2/mujoco_robot_config.json`
- current scene files in `assets/v2/`
- `runtime/configs/runtime_config_v2.json`
- MuJoCo simulation path
- evaluation ladder and standing regression targets
- run logging and experiment comparison workflow

### Reuse as validation infrastructure

- `training/docs/standing_training.md`
- `v0.14.6` as hard-push historical baseline
- `v0.17.1` as the final PPO standing baseline
- current fixed ladder concepts:
  - `eval_clean`
  - `eval_easy`
  - `eval_medium`
  - `eval_hard`
  - `eval_hard_long`

---

## What Will Not Be Reused As Mainline

These stay in the repo but stop being the mainline control answer.

- PPO-only escalation after `v0.17.1`
- M3-lite as the primary architecture
- Li-style history as the primary architecture
- large reward-family growth as the main way to solve stepping

---

## Proposed Repository Impact

The architecture pivot should be visible in the repo layout.

Recommended new areas:

```text
docs/
  system_architecture.md

training/docs/
  ocs2_humanoid_mpc_adoption.md

training/control/ or control/
  robot_model/
  reduced_model/
  mpc/
  execution/
  adapters/
```

Principle:
- controller code should not live inside PPO-specific files
- planner/execution layers should be separable from the training loop
- the current PPO path should remain intact for comparison

---

## Implementation Plan

### `v0.17.2` - Adoption And Gap Analysis

Deliver:
- architecture docs
- WildRobot gap analysis
- repo scaffolding direction
- first implementation scope

### `v0.17.3` - Standing Bring-Up

Deliver:
- quiet standing on the new stack
- inspectable controller signals
- moderate-push readiness

### `v0.17.4` - Hard-Push Standing Recovery

Deliver:
- fixed-ladder evaluation on the new stack
- comparison against `v0.17.1` and `v0.14.6`

### `v0.17.5` - Walking Bring-Up

Deliver:
- same architecture supports nominal walking
- standing recovery remains a regression requirement

### `v0.17.6` - Optional RL Augmentation

Deliver:
- RL only if it clearly improves the established model-based stack

---

## Top WildRobot Integration Risks

1. **Model-path mismatch**
   - MJCF exists, but the new stack may require a different canonical model path and stricter frame conventions.

2. **Execution-layer overreach**
   - trying to implement too much WBC/custom control at once could recreate the same “repo-specific controller invention” problem we are trying to avoid.

3. **Simulation/controller convention drift**
   - if support geometry, contact frames, or joint directions are inconsistent between model, planner, and execution layer, standing and stepping behavior will fail for reasons that are hard to diagnose.

---

## Immediate Next Actions

1. Confirm the control-model source of truth for WildRobot.
2. Define the frame and contact-geometry conventions in writing.
3. Decide the minimum execution layer for the first standing bring-up.
4. Create the implementation scaffolding for the new controller path.
5. Keep the fixed standing ladder as the first acceptance test for the new stack.
