# WildRobot URDF Migration Plan

**Status:** Deferred / conditional  
**Last updated:** 2026-03-22

---

## Purpose

This document records the URDF-first migration path that was considered during
the OCS2 / humanoid MPC investigation.

It is **not** the active mainline for `v0.17.3+`.

Current mainline:
- [system_architecture.md](/home/leeygang/projects/wildrobot/docs/system_architecture.md)
- [footstep_planner_rl_adoption.md](/home/leeygang/projects/wildrobot/training/docs/footstep_planner_rl_adoption.md)

This document stays in the repo because a URDF-first control model may still be
useful later:
- after a hardware upgrade
- if a stronger model-based controller becomes practical
- if the project revisits OCS2 / Pinocchio-based stacks

---

## Why It Is Deferred

The original motivation was:
- align the control model with OCS2 / Pinocchio tooling
- keep MJCF for MuJoCo simulation
- move toward a more standard humanoid control stack

That path is deferred for one reason:
- **the current hardware cannot make good use of the controller stack that
  motivated the migration**

Given:
- position servos
- `50 Hz` runtime
- sparse sensing
- no ankle roll

the project gets more value from:
- keeping MJCF / MuJoCo as the mainline model path
- adding a lightweight planner on top of the existing RL stack

than from paying the URDF migration cost now.

---

## Current Policy

For the active branch:
- `assets/v2/wildrobot.xml` remains the practical mainline robot model
- MJCF / MuJoCo remains the main simulation and evaluation path
- URDF is not required to complete `v0.17.3+`

That means:
- do not block planner / RL work on URDF creation
- do not make Pinocchio integration a prerequisite for current standing
  recovery work

---

## What Still Makes URDF Valuable Later

URDF is still worth revisiting if the project later needs:
- Pinocchio-based kinematics/dynamics tooling
- stronger model-based controllers
- cleaner controller-side robot description ownership
- simulator portability beyond MuJoCo

At that point, the right policy would likely be:
- URDF as controller-facing model
- MJCF as simulation-facing model
- explicit consistency checks between them

---

## Triggers To Revisit The Migration

Revisit this plan only if one or more of these become true:

1. Hardware is upgraded materially:
   - higher-bandwidth actuators
   - torque or impedance control
   - richer state/contact sensing

2. The project commits to a stronger model-based stack:
   - OCS2
   - Pinocchio-heavy tooling
   - whole-body control or similar execution layers

3. MuJoCo-only model ownership becomes a concrete bottleneck:
   - model maintenance pain
   - controller integration pain
   - cross-simulator needs

---

## If The Migration Is Revived

Use this staged plan.

### Phase 0: Freeze Baselines

Keep:
- `v0.14.6`
- `v0.17.1`
- the fixed standing ladder

### Phase 1: Define Model Ownership

Decide:
- what is canonical for control
- what is canonical for simulation
- how URDF and MJCF stay synchronized

### Phase 2: Build The URDF Control Model

Deliver:
- link hierarchy
- inertias
- joint limits
- frame conventions
- foot/contact metadata

### Phase 3: Add URDF <-> MJCF Validation

Check:
- total mass
- inertias
- nominal pose
- support geometry
- frame conventions

### Phase 4: Only Then Integrate Controller Code

Do not start controller adoption before the model path is trustworthy.

---

## MuJoCo And URDF

MuJoCo can load URDF, but URDF support is a subset of what MuJoCo can represent.
For serious MuJoCo work, MJCF remains the better native format.

Practical conclusion:
- if the migration is revived, keep MJCF for MuJoCo anyway
- do not expect URDF alone to replace the existing sim model cleanly

---

## Bottom Line

URDF migration is no longer the near-term action item.

For current WildRobot:
- keep MJCF / MuJoCo as the mainline model path
- focus on the hybrid planner + RL branch first
- revisit URDF only if the hardware or controller ambition changes
