# Step-Trait Base Controller Design (Foot + Arms Placement)

Status: Draft  
Last updated: 2026-03-08

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
3) Update the **step state machine** (support leg, swing leg, trigger/commit/touchdown).
4) If stepping, compute a **frozen touchdown target** for the active swing foot.
5) Compute a simple **swing trajectory** toward that frozen touchdown target.
6) Compute **arm targets** as secondary damping for torso angular motion.
7) Convert targets into a baseline joint target `ctrl_base_rad`.
8) Apply RL residual `ctrl_resid_rad = resid_scale * action`.
9) Final command: `ctrl = clip(ctrl_base_rad + ctrl_resid_rad, joint_limits)`.

Key idea:
- The base controller provides “reasonable” stepping and momentum responses.
- RL learns *when* and *how much* to deviate, but should not have to rediscover the basic step sequence.

### M3 controller contract

M3 should be implemented as a **hybrid controller**, not a fully continuous target generator.

Required properties:
- Only **one swing foot** is active at a time in M3.
- A step target is **committed/frozen** when the controller enters swing; it is not recomputed every tick.
- Arms are **secondary stabilizers**; they must not carry the recovery by themselves.
- Residual RL is additive and bounded; it can refine the step, not replace the step state machine.

### M3 controller state machine

The base controller should maintain explicit internal state:

```text
ControllerMode:
  STANCE
  STEP_TRIGGERED
  SWING
  TOUCHDOWN_RECOVER
  REPLAN   # optional safety state; can be deferred if not needed
```

Per-step internal state:

```text
support_leg: {left, right, none}
swing_leg: {left, right, none}
mode_step_count: int
step_target_xy: [x, y]          # heading-local touchdown target for active swing foot
step_start_xy: [x, y]           # heading-local swing-foot position when swing begins
step_height_m: float
step_duration_ticks: int
touchdown_confirmed: bool
```

State semantics:
- `STANCE`: both feet considered support candidates; maintain neutral/upright baseline.
- `STEP_TRIGGERED`: choose swing leg and freeze a touchdown target.
- `SWING`: track the committed swing-foot trajectory until touchdown or timeout.
- `TOUCHDOWN_RECOVER`: blend support posture back toward neutral after touchdown.
- `REPLAN`: optional fallback if swing timeout, target infeasible, or robot state diverges badly.

Transition rules for the first M3 implementation:
- `STANCE -> STEP_TRIGGERED` when `need_step > step_trigger_threshold` for `N` consecutive control ticks.
- `STEP_TRIGGERED -> SWING` immediately after choosing `swing_leg`, `support_leg`, and freezing `step_target_xy`.
- `SWING -> TOUCHDOWN_RECOVER` when the swing foot touchdown is detected and remains loaded for `M` ticks.
- `SWING -> REPLAN` on swing timeout or if torso tilt grows beyond a safety threshold while the target is still not reached.
- `TOUCHDOWN_RECOVER -> STANCE` when `need_step < recover_threshold` and torso orientation returns within upright bounds.
- `REPLAN -> SWING` if a new target is successfully generated; otherwise `REPLAN -> STANCE` as a conservative fallback.

Recommended initial values:
- `step_trigger_threshold`: 0.45
- `recover_threshold`: 0.20
- `trigger_hold_ticks`: 2-3
- `touchdown_hold_ticks`: 1-2
- `swing_timeout_ticks`: 8-14 control ticks

### Support-leg and swing-leg selection

M3 should keep this simple and deterministic:
- Prefer the currently more-loaded foot as `support_leg`.
- Choose the opposite foot as `swing_leg`.
- If both feet are similarly loaded, choose the foot that steps **toward** the disturbance direction:
  - positive roll / positive lateral velocity -> swing left foot
  - negative roll / negative lateral velocity -> swing right foot

Do not support crossover steps or double-step plans in the first M3 version.

### Foot placement target generation

For M3, target generation should be intentionally conservative:
- Primary objective: **lateral catch step**.
- Secondary objective: mild fore-aft correction.
- Compute target in **heading-local frame**.
- Freeze the target at `STEP_TRIGGERED`; do not continuously retarget during `SWING`.

Recommended touchdown target:

```text
y_des = y_nominal(sign by foot side)
      + k_lat_vel * lateral_vel
      + k_roll * roll

x_des = x_nominal
      + k_fwd_vel * forward_vel
      + k_pitch * pitch
```

Recommended constraints:
- Keep `x_nominal` near zero in standing push recovery.
- Clamp `x_des` and `y_des` to reachable per-foot bounds.
- Make lateral correction dominant over fore-aft correction in M3.
- Reuse the same `stance_width_target` concept as the nominal lateral separation.

Recommended first-pass bounds:
- `x_step_min_m`, `x_step_max_m`
- `y_step_inner_m`, `y_step_outer_m`
- `step_max_delta_m` relative to current swing-foot position

### Swing trajectory generation

M3 should not jump directly from touchdown target to joint deltas. It should define a simple swing-foot trajectory first.

Required swing trajectory outputs:
- `swing_foot_xy(t)`: interpolated horizontal path from `step_start_xy` to `step_target_xy`
- `swing_foot_z(t)`: fixed-height arc with optional mild scaling by `need_step`

Recommended M3 simplification:
- Use a fixed-duration swing trajectory.
- Use a fixed arc height with a small `need_step` multiplier.
- Use a simple trajectory shape:
  - XY: linear or smoothstep interpolation
  - Z: parabola or half-sine

Recommended initial values:
- `swing_height_m`: 0.03-0.05
- `swing_duration_ticks`: 10-12

### Target tracking to joints

The design must explicitly separate:
1) foot target generation
2) swing trajectory generation
3) conversion from trajectory to joint targets

For the first M3 version, use the simplest feasible tracker:
- Keep stance leg near default support posture.
- Apply lightweight foot-position tracking to the swing leg using:
  - either a small analytic IK helper if available,
  - or a hand-tuned joint heuristic that maps foot x/y/z errors into hip/knee/ankle deltas.

The first M3 implementation should favor:
- predictable tracking,
- low gain,
- conservative limits,
- debuggable sign conventions.

Avoid full-body IK or optimization-based tracking in M3.

### Arm momentum strategy

Arm control should be intentionally weaker and narrower in scope than foot placement.

M3 arm requirements:
- Arms are **assistive only**.
- Arms should be gated by `need_step` and/or enabled only in `SWING` and `TOUCHDOWN_RECOVER`.
- Arm authority must be capped separately from leg/base authority.

Recommended first M3 arm strategy:
- Prioritize **roll stabilization** first.
- Optionally add a small pitch-rate damping term after roll behavior is stable.
- Do not try to learn or hand-design a complex multi-axis arm swing pattern in M3.

Recommended target form:

```text
arm_delta =
    k_roll_rate * roll_rate
  + k_roll * roll
  + optional small k_pitch_rate * pitch_rate
```

Recommended limits:
- `arm_max_delta_rad`
- `arm_need_step_threshold`
- `arm_gain_scale` < foot placement authority

### Residual RL interaction

Residual RL should remain additive, bounded, and phase-aware.

Recommended policy:
- In `STANCE`, allow modest residual authority for posture refinement.
- In `SWING`, preserve enough residual authority to refine timing/placement, but do not allow it to fully override the committed swing motion.
- In `TOUCHDOWN_RECOVER`, allow residual authority to help settle back to neutral.

The key M3 rule is:
- Residual RL can modulate execution quality, not replace the hybrid stepping logic.

### M3 success criteria

The first M3 training run should be judged against explicit mechanism and outcome criteria.

Mechanism criteria:
- `debug/bc_in_swing` should be the primary occupancy signal for `SWING` during disturbed episodes.
- `debug/bc_phase` remains useful as a coarse aggregate phase summary, but should not be used alone to infer swing occupancy.
- `debug/bc_phase_ticks` should not be dominated by timeout-driven swings.
- `reward/step_event` must be measurably above zero under hard-push evaluation.
- `reward/foot_place` must rise with stepping, indicating the touchdown target is not random.
- `debug/touchdown_left` / `debug/touchdown_right` should occur when `debug/need_step` is elevated, not only during idle motion.

Outcome criteria:
- Under hard-push eval (`push_force_max=9`, `push_duration_steps=15`), `eval_push/survival_rate` should match or exceed the M2 baseline.
- `eval_push/episode_length` should trend toward the full 500-step horizon.
- Dominant terminations should shift away from immediate collapse after disturbance.
- After the push, the robot should return toward neutral posture:
  - `reward/posture` should recover after the disturbance window.
  - roll/pitch termination fractions should remain low.
- Visual inspection should confirm:
  - the robot steps instead of only bracing,
  - the step widens support in the disturbance direction,
  - the robot does not remain stuck in a crouched or oscillatory recovery pose.

Failure signatures for the first M3 run:
- `SWING` rarely occurs even under hard pushes.
- Swing is mostly timeout-driven rather than touchdown-driven.
- The policy survives by residual-only bracing while the FSM contributes little.
- The robot steps but fails to regain upright posture afterward.
- Arms or waist compensation dominate while foot placement remains weak.

**Revised weak-engagement response** (supersedes the original "reduce residual authority" branch):

The original recommendation was to lower trigger thresholds and reduce residual authority to force FSM engagement. Training runs v0.14.3–v0.14.5 showed this approach is counterproductive: cutting residual authority fights the optimizer, degrading the policy's best known strategy (brace/crouch at 98% success) without the FSM being accurate enough to compensate. The result was a monotonic regression from 98% → 86% → 64% → 57% success.

If the first M3 run shows weak FSM engagement but decent posture recovery:
1. **Establish the bracing ceiling first.** Run the bracing-only baseline (FSM disabled, full residual authority) through a force sweep (9–25N) using `training/eval/force_sweep.py`. Find the force level where success drops below 60%.
2. **Train at the bracing-failure regime.** Set `push_force_max` to the bracing ceiling + 20%. Keep full residual authority (`resid_scale = 1.0` for all phases). At this force level, bracing alone cannot survive, so the optimizer has a gradient signal toward stepping.
3. **Try pure PPO first.** If the policy discovers stepping on its own under the harder pushes, FSM complexity is unnecessary.
4. **Add information, not constraints.** If pure RL does not step, add capture-point error `[x_com - x_cp, y_com - y_cp]` to the actor observation and/or use a privileged critic that sees push force and CoM velocity. This gives the policy *information* about where to step without constraining *how*.
5. **Re-introduce FSM as a guide, not a controller.** Only if information alone is insufficient, re-enable the FSM with full residual override authority (`resid_scale = 1.0`). The FSM output becomes a mild bias that the RL can fully override when it has a better strategy.
6. **Do not reduce residual authority** to force FSM engagement. If the FSM is not engaged at full residual authority, the FSM is not helping — fix the FSM or the force regime, not the policy's freedom.

### Base controller outputs (API-level)

Add a training/runtime control module conceptually like:

```text
BaseController.compute(signals, internal_state) -> ctrl_base_rad, next_state, debug_dict
```

This base controller must be implemented in both:
- Training: `training/envs/wildrobot_env.py` (or imported helper).
- Runtime: `runtime/wr_runtime/control/` (mirrored logic).

Recommended shared helper boundary:

```text
compute_need_step(signals, cfg) -> scalar
update_step_fsm(signals, state, cfg) -> next_state
compute_step_target(signals, state, cfg) -> target_xy
compute_swing_traj(signals, state, cfg) -> swing_target
compute_arm_targets(signals, state, cfg) -> arm_target
targets_to_ctrl_base(signals, state, targets, cfg) -> ctrl_base_rad
```

The training env and runtime should both call the same helper logic, with only thin adapter code for simulator/runtime-specific state access.

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
    enabled: true
    trigger_threshold: 0.45
    recover_threshold: 0.20
    trigger_hold_ticks: 2
    touchdown_hold_ticks: 1
    swing_timeout_ticks: 12

    x_nominal_m: 0.0
    y_nominal_m: 0.115
    k_lat_vel: 0.15
    k_roll: 0.10
    k_pitch: 0.05
    k_fwd_vel: 0.05

    x_step_min_m: -0.08
    x_step_max_m: 0.12
    y_step_inner_m: 0.08
    y_step_outer_m: 0.20
    step_max_delta_m: 0.12

    swing_height_m: 0.04
    swing_height_need_step_mult: 0.5
    swing_duration_ticks: 10

  # Arm momentum heuristic
  arms:
    enabled: true
    need_step_threshold: 0.35
    k_roll: ...
    k_roll_rate: ...
    k_pitch_rate: ...
    max_delta_rad: ...
```

Notes:
- Keep **policy observation/action** the same; only change the internal control composition.
- The runtime config should be able to disable base control or clamp residual authority.
- The runtime config should be able to disable arms independently of foot placement.
- `stance_width_target` remains useful as the nominal support geometry, but should not be used as a separate reward penalty for M3.

### Reward shaping changes (industry-style stepping incentives)

Use stepping rewards that are:
- **Event-based** (touchdown), not only continuous “feet moving” rewards.
- **Gated** by `need_step` to avoid marching-in-place.
- **Placement-based** to teach “step where it helps”.

Recommended reward components (already implemented in v0.13.11 training code):
- `reward/step_event`: touchdown event (gated by `need_step`).
- `reward/foot_place`: exp(-placement error / sigma²) at touchdown (gated by `need_step`).
- Keep `reward/slip` strong to avoid shuffling.
- Keep deprecated gait-style terms only as debug metrics during the first M3 validation run; they should not shape the M3 objective.

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

### Prerequisite: Bracing Ceiling Calibration

Before investing in FSM tuning or stepping reward shaping, the project must establish at what force level the bracing/crouch strategy fundamentally fails. Without this calibration, stepping work may target a force regime where stepping is unnecessary (as happened with M3 at 9N).

**Why this matters:**
The v0.13.8 checkpoint achieves 98.44% success at `push_force_max=9`, `push_duration_steps=15` with pure PPO (no FSM, no base controller). This means 9N is well within the bracing-recoverable regime. All M3 stepping work at 9N was trying to induce a behavior the optimizer correctly concluded was unnecessary.

**How to calibrate:**

```bash
uv run python training/eval/force_sweep.py \
    --checkpoint <bracing_baseline_checkpoint.pkl> \
    --config training/configs/ppo_standing_push.yaml \
    --forces 9,12,15,18,22,25 \
    --num-envs 128 \
    --num-steps 500 \
    --output training/eval/bracing_ceiling.json
```

This sweeps push force from 9N to 25N with FSM disabled and reports `success_rate`, `episode_length`, and termination fractions at each level. The **bracing ceiling** is the force level where `success_rate` drops below 60%.

**Decision tree after calibration:**

1. If bracing ceiling is above 25N: the robot may not need stepping for practical push recovery. Consider whether walking (M4) is a better next investment than push-recovery stepping.
2. If bracing ceiling is 12–20N: this is the stepping regime. Set `push_force_max` to ceiling + 20% and proceed to M3 at that force level with full residual authority.
3. If bracing ceiling is below 12N: the bracing policy may be undertrained. Consider further training at the current force level before escalating.

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

#### M2.5: Bracing ceiling calibration (prerequisite for M3)
- Run the best bracing-only checkpoint through a force sweep (9–25N) using `training/eval/force_sweep.py`.
- Establish the bracing ceiling: the force where `success_rate` drops below 60%.
- Define the stepping regime: `push_force_max = ceiling + 20%`.
- At the stepping regime, train pure PPO with full residual authority to see if RL discovers stepping on its own.
- If pure RL does not step: add capture-point observations and/or privileged critic, then re-evaluate.
- Only proceed to M3 after confirming the force regime and that information alone is insufficient.

#### M3: Foot placement + arms base controller
- Operates at the calibrated stepping-regime force level (from M2.5), not at 9N.
- FSM acts as a **guide**: residual authority starts at 1.0 for all phases. The policy can fully override the FSM.
- Add explicit step-state logic, frozen touchdown targets, swing trajectory tracking, and secondary arm damping.
- If the FSM's open-loop joint tracking (scalar gains) is insufficient, replace with simple analytical leg IK or expose desired foot targets in the policy observation.
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
