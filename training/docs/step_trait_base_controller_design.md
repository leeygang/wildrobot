# Step-Trait Base Controller Design (Foot + Arms Placement)

Status: Draft  
Last updated: 2026-03-07

## Goal

Train and deploy policies that:
- **Recover from hard torso pushes by stepping** (not just bracing/crouching).
- **Return to an upright/neutral posture** after recovery (avoid “stuck crouch”).
- Scale from standing push-recovery → **walking** without rewriting the whole stack.

Constraints / non-goals (initially):
- Keep the **policy contract** stable (observations/actions remain usable in runtime).
- Prefer **resume-safe** training iterations (avoid contract-breaking changes like `env.action_filter_alpha`).
- Avoid full-blown MPC; start with a **simple, debuggable** base controller.

## Approaches

### 1) Pure RL (end-to-end)

Policy outputs actuator targets directly; reward/curriculum must induce stepping.

Pros
- Minimal engineering; simplest conceptual pipeline.
- Can discover non-obvious strategies.

Cons
- Often stalls in local optima (brace/crouch, fidget, hop).
- Needs careful curriculum and reward shaping for stepping.
- More fragile sim→real due to exploration and contact discontinuities.

When to use
- If you can afford long iteration cycles and prefer minimal hand design.

---

### 2) Joint-space heuristic base controller + residual RL

Compute a stabilizing joint-target baseline from body state (height, pitch/roll, gyro),
then add RL residuals on top.

Pros
- Fastest path to “doesn’t fall” and “returns upright”.
- Residual RL explores safely (fewer catastrophic episodes).

Cons
- Doesn’t encode “where to step”; RL must still learn stepping conceptually.
- If too strong, base controller can cap agility or fight learned strategy.

When to use
- As a first step to stabilize learning and hardware execution.

---

### 3) Foot/arms placement-driven base controller + residual RL (recommended)

Compute **desired support-foot placement** (Raibert/capture-point style) and a simple
**arm momentum strategy**; track those targets with joint-space targets (or lightweight IK),
and let RL supply residuals and/or modulate targets.

Pros
- Encodes the stepping concept directly (support polygon recovery).
- Strong path from push recovery → walking (same foot placement principle).
- Residual RL focuses on timing and coordination, not rediscovering balance.

Cons
- More engineering: frames, touchdown detection, foot position targets, tracking.
- Needs careful sim↔runtime parity for kinematics and sign conventions.

When to use
- If the primary goal is a robust stepping trait and fast transfer to hardware.

## Recommended direction: Foot/Arms Placement Base + Residual RL

### High-level control decomposition

At each control tick:
1) Read signals/state (gravity vector, gyro, joint pos/vel, foot contacts).
2) Compute a **need-to-step** scalar `need_step ∈ [0, 1]`.
3) Compute **foot placement targets** (x/y) for the next touchdown.
4) Compute **arm targets** to counter angular velocity / body tilt.
5) Convert targets into a baseline joint target `ctrl_base_rad`.
6) Apply RL residual `ctrl_resid_rad = resid_scale * action`.
7) Final command: `ctrl = clip(ctrl_base_rad + ctrl_resid_rad, joint_limits)`.

Key idea:
- The base controller provides “reasonable” stepping and momentum responses.
- RL learns *when* and *how much* to deviate, and can take over if it finds better strategies.

### Base controller outputs (API-level)

Add a training/runtime control module conceptually like:

```text
BaseController.compute(signals, internal_state) -> ctrl_base_rad, debug_dict
```

This base controller must be implemented in both:
- Training: `training/envs/wildrobot_env.py` (or imported helper).
- Runtime: `runtime/wr_runtime/control/` (mirrored logic).

#### Proposed config/API surface

Add a base controller config block in training YAML (and bundle/runtime JSON as needed):

```yaml
base_controller:
  enabled: true
  mode: foot_arms_placement   # or joint_heuristic
  resid_scale_rad: 0.35       # max RL residual authority per joint
  resid_scale_need_step_mult: 1.0   # optional: scale residual authority by need_step

  # Need-to-step gate
  need_step:
    pitch: 0.35
    roll: 0.35
    lateral_vel: 0.30
    pitch_rate: 1.00

  # Foot placement heuristic
  foot_place:
    sigma_m: 0.12
    k_lat_vel: 0.15
    k_roll: 0.10
    k_pitch: 0.05
    k_fwd_vel: 0.05

  # Arm momentum heuristic
  arms:
    enabled: true
    k_pitch_rate: ...
    k_roll_rate: ...
    max_delta_rad: ...
```

Notes:
- Keep **policy observation/action** the same; only change the internal control composition.
- The runtime config should be able to disable base control or clamp residual authority.

### Reward shaping changes (industry-style stepping incentives)

Use stepping rewards that are:
- **Event-based** (touchdown), not only continuous “feet moving” rewards.
- **Gated** by `need_step` to avoid marching-in-place.
- **Placement-based** to teach “step where it helps”.

Recommended reward components (already implemented in v0.13.11 training code):
- `reward/step_event`: touchdown event (gated by `need_step`).
- `reward/foot_place`: exp(-placement error / sigma²) at touchdown (gated by `need_step`).
- Keep `reward/clearance`, `reward/gait_periodicity`, `reward/hip_swing`, `reward/knee_swing` as auxiliary terms.
- Keep `reward/slip` strong to avoid shuffling.

Recommended posture return shaping:
- `reward/posture` (gated by uprightness), so “after recovery” it returns to neutral.

### Evaluation plan (sim + hardware)

#### Sim (offline W&B) signals

Always track:
- `eval_push/survival_rate`, `eval_push/survival_steps` (probe horizon)
- `eval_push/success_rate`, `eval_push/episode_length` (confirm horizon = 500)
- Dominant terminations: `eval_push/term_height_low_frac`, `term_pitch_frac`, `term_roll_frac`
- Stepping trait metrics (new):
  - `reward/step_event`, `reward/foot_place`
  - `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right`
- Stress: `debug/torque_sat_frac`, `tracking/max_torque`

Add explicit eval stages (curriculum):
- Stage 1 “hard”: `push_force_max=9`, `push_duration_steps=15`
- Stage 2 “very hard”: `push_force_max=12`, `push_duration_steps=15`

Success criteria per stage (confirm eval):
- `eval_push/success_rate` near 1.0 and `episode_length` near 500.
- Termination fractions near 0.
- Stepping present when needed:
  - `reward/step_event` > 0 and touchdown flags observed when disturbed.

#### Hardware

Minimum hardware checks per milestone:
- Run bundle with calibrated runtime config.
- Apply controlled pushes; check:
  - does it step instead of collapsing?
  - does it return to neutral posture?
  - does it avoid prolonged crouch/oscillation?

### Milestones

#### M0: Baseline measurement (current PPO)
- Lock baseline checkpoint and run hard/very-hard push eval.
- Record step-event/foot-place metrics (should be low/near 0 if no stepping).

#### M1: Reward-only stepping trait (no base controller)
- Enable stepping-trait shaping (touchdown + foot placement).
- Use push curriculum.
- Goal: step-event metrics increase; survival improves.

#### M2: Add base controller (joint-heuristic first)
- Implement `ctrl = ctrl_base + ctrl_resid` (no foot placement yet).
- Gate residual authority by `need_step`.
- Goal: faster learning + fewer catastrophic terminations.

#### M3: Foot placement + arms base controller
- Add foot placement targets for touchdown and arm momentum targets.
- Ensure runtime parity and stable safety limits.
- Goal: reliable stepping under hard pushes and consistent return upright.

#### M4: Walking extension
- Expand command range (`env.max_velocity`) with curriculum.
- Keep same stepping mechanics; foot placement becomes primary walking driver.

## Risks and mitigations

- **Reward hacking / marching-in-place**
  - Gate stepping rewards by `need_step`.
  - Keep slip penalty strong; keep action_rate/torque penalties (but allow bursts when needed).

- **Sim→real mismatch**
  - Keep observation set hardware-consistent (IMU gravity/gyro).
  - Mirror base controller logic in runtime.
  - Validate actuator order and sign conventions via bundle validator.

- **Overpowering base controller**
  - Keep base gains conservative.
  - Let RL residual authority increase with `need_step`.
  - Provide a runtime knob to cap residual/arm motion.

