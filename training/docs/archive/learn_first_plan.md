# Learn First, Retarget Later: Robot-Native Walking Policy

**Version:** v0.15.6
**Status:** Archived / superseded by `training/docs/standing_training.md` for the
active training plan
**Last Updated:** 2026-03-14

> Archive note: this document is preserved as historical context for the older
> walking-first roadmap (`v0.10` - `v0.15`). The canonical active training plan
> is now `training/docs/standing_training.md`.

## Overview

This document outlines a phased approach to training a walking policy for WildRobot. Instead of starting with human motion retargeting (GMR), we first discover the robot's natural gait manifold through reinforcement learning, then progressively add style refinement.

### Why This Approach?

The GMR-first approach failed because:
1. Human gait ≠ Robot gait (different mass, leg length, joint limits)
2. IK retargeting preserves pose, not dynamics
3. Physics validation rejected most retargeted motion (0% Tier 0+)
4. The robot's feasible motion manifold was unknown

This approach discovers what the robot *can* do before trying to make it look human-like.

---

## Stage 1a: Control Abstraction Layer (v0.11.x) - NEW

**Goal:** Implement CAL to decouple training from MuJoCo primitives.

⚠️ **BREAKING CHANGE**: This requires retraining from scratch. Existing checkpoints are incompatible.

### Exit Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| CAL unit tests | 100% pass | pytest test_cal.py |
| Policy action semantics | Normalized [-1, 1] | Code review |
| Symmetry correction | Verified | Left/right hip produce symmetric motion |
| Config migration | Complete | actuated_joints in mujoco_robot_config.json |

### Value Delivered

- Clear action semantics (policy actions vs physical angles)
- Correct joint symmetry handling
- Sim2Real ready architecture
- Single source of truth for joint properties

### Version Milestones

#### v0.11.0 - CAL Infrastructure
**Objective:** Implement Control Abstraction Layer core

Tasks:
- [ ] Create `training/control/` module
- [ ] Implement `cal.py` with ControlAbstractionLayer class
- [ ] Implement `specs.py` with JointSpec, ActuatorSpec, ControlCommand
- [ ] Implement `types.py` with ActuatorType, VelocityProfile enums
- [ ] Create `test_cal.py` unit tests
- [ ] Create `assets/keyframes.xml` with home pose

Exit: CAL module implemented and tested

#### v0.11.1 - Environment Integration
**Objective:** Integrate CAL into wildrobot_env.py

Tasks:
- [ ] Update `wildrobot_env.py` to use CAL
- [ ] Update `robot_config.py` for actuated_joints support
- [ ] Update `mujoco_robot_config.json` with actuated_joints section
- [ ] Update `post_process.py` with generate_actuated_joints_config()

Exit: Environment uses CAL for all action/state conversion

#### v0.11.2 - Validation
**Objective:** Verify CAL works correctly end-to-end

Tasks:
- [ ] Run standing policy with CAL
- [ ] Verify symmetric actions produce symmetric motion
- [ ] Validate keyframe loading from MuJoCo
- [ ] Update documentation

Exit: CAL fully integrated and validated

---

## Stage 1b: Robot-Native Walking Policy (v0.15.x) - CURRENT

**Goal:** Learn a stable, repeatable, controllable gait using task rewards only (no AMP, no reference motion).

### Current Guidance (2026-03-14)

Recent `v0.15.3` and `v0.15.4` walking runs show that simply increasing velocity reward and stability penalties at the same time is not sufficient. The observed failure mode is a **posture exploit**: the robot gains a small amount of forward speed by leaning/pitching harder, while training and eval stability collapse.

The confirmed direction for Stage 1b is therefore:
- keep flat-terrain PPO-only walking as the primary path
- keep velocity tracking as the dominant task objective
- use a **command curriculum** for gait emergence instead of training immediately on the full target walking range
- use a **reward / penalty curriculum** rather than full-strength stability / effort penalties from iteration 1
- widen the command range only after a nominal stable gait exists
- select checkpoints using locomotion quality plus stability, not survival-only eval signals

This aligns with common modern legged-locomotion practice from massively parallel RL systems and curriculum-based locomotion training: command curricula first, then robustness and broader speed coverage.

### References / Rationale

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (CoRL 2022): emphasizes massively parallel RL plus a game-inspired curriculum for locomotion training.
  - Link: https://proceedings.mlr.press/v164/rudin22a.html
- Margolis et al., *Rapid Locomotion via Reinforcement Learning* (RSS 2022): identifies adaptive curriculum on velocity commands as a key component of agile locomotion learning.
  - Link: https://www.roboticsproceedings.org/rss18/p022.html
- Margolis and Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior* (CoRL 2023): argues against repeated ad hoc reward redesign and toward structured behavior spaces / commandable locomotion families.
  - Link: https://proceedings.mlr.press/v205/margolis23a.html
- Isaac Lab Documentation, *Curriculum Utilities*: documents reward-weight and environment-parameter curricula as standard tooling for RL training.
  - Link: https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- Ciebielski and Khadiv, *Contact-conditioned learning of locomotion policies* (arXiv 2024): shows, for bipeds, that contact-conditioned representations can outperform simple velocity-conditioned policies for robustness and multi-gait learning.
  - Link: https://arxiv.org/abs/2408.00776

### Exit Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Episode length | > 400 steps (8s) | Average over 100 episodes |
| Forward velocity | Stage gate: 0.10-0.35 m/s, final target: 0.5-1.0 m/s | Tracking error < 0.2 m/s |
| Fall rate | < 5% | Episodes ending in fall |
| Gait periodicity | Visible alternating steps | Visual inspection |
| Torque usage | < 80% of limits | Max torque / torque limit |

### Value Delivered

- Proof that robot can walk in simulation
- Understanding of robot's natural gait dynamics
- Baseline policy for comparison
- Foundation for all subsequent stages

### Stage 1b Operating Rules

- **Do not** treat `eval_push/*` as meaningful for walking checkpoint selection when `push_enabled: false`.
- A checkpoint is not deployable walking if forward velocity remains near zero, even if eval survival is perfect.
- If a run reaches roughly `80-120` iterations with `env/forward_velocity < 0.1 m/s` and rising pitch failures, treat it as trapped or exploiting posture rather than "still training".
- If a run gains some speed but `term_pitch_frac` and eval-clean failures rise sharply, stop and retune; do not continue on the assumption it will stabilize later.
- Stage 1b should solve nominal translation first. Pushes, terrain variation, and broader speed coverage come only after a stable nominal gait exists.

### Stage 1b Lever Backlog

The current `v0.15.5` evidence suggests the task is still trapped between the stand-still basin and the lean-fall basin. Stage 1b should therefore explore the following levers in order, from smallest structural fix to larger contract changes.

1. **Contact-gated velocity reward** (`v0.15.6`, implemented)
   - Make most forward reward contingent on stepping evidence instead of unconditional body translation.
   - Goal: remove the reward path where the policy earns progress by leaning without stepping.
2. **Explicit pitch-rate penalty** (`v0.15.6`, implemented)
   - Penalize slow forward-diving behavior directly; the existing `angular_velocity` term only damps yaw.
3. **Activate existing stepping rewards** (`v0.15.6`, implemented)
   - Use the already-implemented `step_event` and `foot_place` terms so stepping has a direct positive path.
4. **Low-speed command curriculum** (`v0.15.6`, implemented)
   - Keep commands around `0.05-0.20 m/s` until stable translation exists.
5. **Command-conditioned propulsion bias** (`v0.15.7`, implemented)
   - If the policy learns to step without moving, reduce undirected touchdown reward and tie forward foot placement to `velocity_cmd`, not just current `forward_vel`.
6. **Reward-path diagnostics** (next if `v0.15.7` still stalls)
   - If the step gate rarely opens, promote a minimal subset of currently logging-only gait terms such as `clearance` and `gait_periodicity` into the total reward.
7. **Clocked observation layout** (`wr_obs_v3`, implemented in `v0.15.8`)
   - After `v0.15.7`, reward-only tuning had clearly run out: the policy could step and stay upright but still could not produce propulsion. `wr_obs_v3` adds an explicit gait clock so the actor does not need to infer periodic timing from scratch.
8. **Two-phase curriculum** (deferred)
   - If nominal translation still does not emerge, explicitly train stepping / weight transfer before walking commands.
9. **Scale env count / rollout length** (deferred)
   - Increase to larger batches only after the reward path produces real stepping. More samples should accelerate a good objective, not a bad one.

### Stage 1b Decision Rules After `v0.15.6`

- Use `80` iterations as the hard stop for the incremental `v0.15.6` probe. Historical `v0.15.3` to `v0.15.5` runs all showed their basin clearly by `80`; `120` only made the failure more obvious, not more informative.
- Treat `40` iterations as the first read:
  - if `debug/velocity_step_gate` is still near zero and stepping rewards remain flat, the objective is still not opening a stepping path
  - do not stop yet, because `v0.15.5` had a misleading early transient before settling
- Treat `60` iterations as the decision-prep checkpoint:
  - if forward velocity is still near zero and stepping metrics are flat, prepare to stop at `80`
  - if stepping metrics rise but `term_pitch_frac` also rises sharply, keep going only to `80` to confirm the basin
- Stop at `80` unless the run shows both stepping engagement and material forward translation without eval-clean collapse.
- If `debug/velocity_step_gate` stays near zero through `80`, the next step is `wr_obs_v3` clock plus minimal gait terms in total reward.
- If stepping metrics rise but `term_pitch_frac` still climbs, tighten the pitch-rate / posture branch before widening the command range.
- Only scale to longer runs or larger batches once the policy shows both stepping engagement and non-trivial forward translation without eval-clean collapse.

### `v0.15.8` Fast-Read Rule

- `v0.15.8` is the first clocked walking probe, so keep the same `60`-iteration fast-read window.
- Read it at `20`, `40`, and `60`:
  - by `20`, forward velocity should move off the floor more clearly than `v0.15.7`
  - by `40`, there should be a visible separation from the reward-only stepping-in-place basin
- by `60`, stop unless forward velocity is materially improving and velocity error is dropping
- If `v0.15.8` still cannot turn stepping into propulsion by `60`, stop doing local reward tuning and move to a more explicit curriculum / architecture change.

### `v0.15.9` Explicit Per-Step Propulsion Branch

- `v0.15.9` keeps `wr_obs_v3` and stays a fresh-run branch (not resume-safe), but changes reward structure to make propulsion explicit per step cycle.
- Branch changes:
  - touchdown step-length reward scaled by `velocity_cmd`
  - per-cycle forward displacement reward using `clock_stride_period_steps`
  - phase-gated clearance reward (left clearance only in left swing window, right only in right swing window)
- Read it at `20`, `40`, and `60`:
  - by `20`, forward velocity should rise off the floor more clearly than `v0.15.8`
  - by `40`, there should be visible separation from stepping-in-place
  - by `60`, if forward velocity remains near zero, treat this branch as failed and move to a larger curriculum/architecture lever

### `v0.15.10` Reward-Economics Fix Branch

- `v0.15.9` showed that the main blocker is now reward economics, not missing stepping machinery.
- Root-cause conclusions from `v0.15.9`:
  - dense forward tracking still pays too much even when forward velocity is near zero
  - generic stepping is sufficient to open the forward reward gate
  - `step_length` and `cycle_progress` are directionally right but too sparse and too strict to dominate early learning
- Implemented `v0.15.10` structure (fresh-run, `wr_obs_v3`):
  - forward reward uses a zero-baseline commanded-walking form so standing / near-standing is near-zero reward
  - forward reward gate is propulsion-quality based (`step_length`, dense progress, cycle progress), not generic stepping
  - dense heading-local per-step forward-progress shaping is active (cycle-complete bonus retained)
  - Stage-A propulsion targets are relaxed for early-learning gradient
  - legacy touchdown shaping is minimized (`foot_place` near/off, `step_event` negligible)
- Concrete tuning direction:
  - lower `step_length_target_scale` from `0.30` toward `0.12-0.18`
  - widen `step_length_sigma` from `0.03` toward `0.05-0.07`
  - lower `cycle_progress_target_scale` from `1.0` toward `0.5-0.7`
  - widen `cycle_progress_sigma` from `0.06` toward `0.10-0.14`
- Probe policy:
  - fresh run, not resume
  - `ppo.iterations: 60`
  - read at `20`, `40`, and `60`
  - by `60`, if `env/forward_velocity <= 0.02`, treat the branch as failed

### `v0.15.11` Stability-Constrained Propulsion Branch

- `v0.15.10` confirmed a major milestone: the low-propulsion basin is now broken.
- But it also showed the next failure mode clearly:
  - speed rises past command
  - `dense_progress` and `propulsion_gate` keep rising
  - `step_length` plateaus
  - `term_pitch_frac` goes to `1.0`
  - `eval_clean` collapses
- So `v0.15.11` should preserve the `v0.15.10` reward-economics foundation while constraining propulsion by posture quality.
- Implemented branch changes:
  - keep zero-baseline forward reward
  - keep propulsion-quality gating
  - keep relaxed Stage-A step targets
  - make `dense_progress` posture-aware with smooth pitch / pitch-rate gating
  - upright-gate the commanded forward reward with configurable strength
  - bias propulsion gate toward `step_length` / `cycle_progress` and cap dense-only influence
  - reduce `dense_progress` weight so it cannot dominate by pure leaning
  - increase `orientation` / `pitch_rate` penalties and modestly strengthen pre-collapse shaping
- Probe policy:
  - fresh run, not resume
  - `ppo.iterations: 40`
  - read at `20` and `40`
  - by `20`, expect `env/forward_velocity > 0.03` with `term_pitch_frac < 0.25`
  - by `40`, expect `env/forward_velocity > 0.05` while `eval_clean/success_rate` remains meaningfully alive
  - if speed rises only together with pitch collapse again, stop immediately

### `v0.15.12` Contact-Driven Step-to-Step Propulsion Branch

- `v0.15.11` improved the early trajectory but did not change the final basin:
  - speed still rises into the commanded range and beyond
  - `term_pitch_frac` still ends at `1.0`
  - `eval_clean` still collapses
  - `step_length` stays mostly flat while `dense_progress` remains the easiest path to speed
- So `v0.15.12` should stop making dense forward velocity the main source of reward credit and instead make contact-driven step outcomes the main source of reward credit.
- Implemented branch direction:
  - keep zero-baseline forward reward
  - keep `wr_obs_v3`
  - keep phase-gated clearance and stronger pitch stability shaping
  - remove `dense_progress` from the main propulsion-gate path
  - add a new structured `step_progress` reward based on touchdown-to-touchdown forward displacement
  - make the forward-reward gate depend on structured signals:
    - `step_length`
    - `step_progress`
  - demote or remove `cycle_progress` as the primary propulsion driver if it remains too sparse / clock-bound
  - runtime choices in `v0.15.12`:
    - command range narrowed to `0.05-0.15 m/s`
    - `dense_progress: 0.0` (not a main optimization path)
    - `step_length` and `step_progress` are the primary structured propulsion terms
    - `cycle_progress` retained only as low auxiliary shaping
- Probe policy:
  - fresh run, not resume
  - `ppo.iterations: 40`
  - narrow command range back toward `0.05-0.15 m/s`
  - treat this as a bigger structural branch, not a local weight retune
- Success criteria:
  - by `20`, `env/forward_velocity > 0.03` without rapid pitch divergence
  - by `40`, `env/forward_velocity > 0.06` with meaningful `eval_clean` survival
  - `step_length` and `step_progress` should rise together
  - if speed rises again while structured step rewards stay flat, stop and escalate to a new training paradigm

### Version Milestones

#### v0.10.0 - PPO Infrastructure Setup ✅
**Objective:** Clean PPO training without AMP dependencies

Tasks:
- [x] Create `ppo_walking.yaml` config (PPO-only, no AMP section)
- [x] Verify train.py works with AMP disabled via config
- [x] Validate simulation setup passes all checks
- [x] Smoke test: 10 iterations, policy updates without errors
- [x] Unified trainer: `trainer_unified.py` supports both PPO-only and AMP modes
- [x] Invariance tests: Verify PPO behavior is identical regardless of AMP mode

Exit: Training loop runs, loss decreases, no crashes

**Completed:** 2024-12-26
- Created `ppo_walking.yaml` with PPO-only config
- Created `trainer_unified.py` - single trainer for Stage 1 and Stage 3
- Created `rollout.py` - unified rollout collector
- Added 11 invariance tests in `test_trainer_invariance.py`
- Verified smoke test passes with AMP disabled via config

#### v0.10.1 - Reward Shaping
**Objective:** Design rewards that encourage stable walking

Tasks:
- [ ] Implement velocity tracking reward
- [ ] Implement upright reward (penalize pitch/roll)
- [ ] Implement contact reward (feet on ground)
- [ ] Implement energy penalty (minimize torque)
- [ ] Implement action smoothness penalty

Exit: Rewards are computed correctly, logged to W&B

#### v0.10.2 - Basic Standing
**Objective:** Robot maintains balance without falling

Tasks:
- [ ] Train with zero velocity command
- [ ] Robot stays upright for 500 steps
- [ ] Tune PD gains if needed

Exit: Standing policy with < 5% fall rate

#### v0.10.3 - Forward Walking
**Objective:** Robot walks forward at commanded velocity

Tasks:
- [ ] Start with a conservative command curriculum such as `0.10-0.30` or `0.12-0.35 m/s`
- [ ] Tune reward weights so velocity tracking dominates standing still
- [ ] Widen the command range only after stable translation emerges
- [ ] Add gait symmetry reward (optional)

Exit: Robot walks forward, tracks velocity within 0.2 m/s

#### v0.10.4 - Gait Quality
**Objective:** Improve gait consistency and efficiency

Tasks:
- [ ] Analyze gait pattern (is it shuffling? hopping?)
- [ ] Add periodic gait reward if needed
- [ ] Use a reward / penalty curriculum:
  - start with lighter orientation / angular-velocity / torque / saturation penalties
  - ramp them up after translation emerges
- [ ] Tune action smoothing / damping if pitch divergence persists
- [ ] Reduce torque usage without collapsing forward speed

Exit: Consistent alternating gait, reasonable energy usage

#### v0.10.5 - Robustness
**Objective:** Policy handles perturbations

Notes:
- v0.10.5 (current): added gait periodicity reward, reduced velocity dominance,
  increased orientation/torque/saturation penalties, shortened run to 400 iters.
 - v0.10.6 (current): add swing-gated hip/knee rewards + flight penalty to reduce ankle-dominant gait.
 - v0.10.6 failure gate: after 300 iters, abort if knee range <30% and flight phase >15%.
 - v0.15.3: fresh-start walking confirmed that standing-resume was counterproductive.
 - v0.15.4: stronger velocity and stability shaping improved forward speed somewhat, but still produced a posture exploit with rising pitch failures.
 - Current conclusion: move from fixed full-strength reward forcing to curriculum-based gait emergence.

Tasks:
- [ ] Add push disturbances during training
- [ ] Add terrain variations (small bumps)
- [ ] Test velocity command changes

Exit: Policy recovers from small pushes, handles velocity changes

---

## Stage 2: Collect Robot-Native Reference Data (v0.12.x)

**Goal:** Generate high-quality reference motion dataset from trained policy.

### Exit Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total duration | > 60 seconds | Sum of all segments |
| Segment count | > 100 segments | After filtering |
| Velocity coverage | 0.3-1.2 m/s | Range of velocities |
| Fall rate in dataset | 0% | No falls in kept segments |

### Value Delivered

- Physics-valid reference motion database
- Perfectly matched feature statistics
- No retargeting error
- Guaranteed learnable by policy class

### Version Milestones

#### v0.12.0 - Rollout Collection Infrastructure
**Objective:** Automated rollout collection and storage

Tasks:
- [ ] Create rollout collector script
- [ ] Define data format (matches AMP features)
- [ ] Implement multi-velocity collection
- [ ] Add perturbation diversity
- [ ] **Unify PPO trainer**: Make trainer_jit.py support amp_weight=0 (optional but recommended)
      - Currently Stage 1 uses Brax PPO, Stage 3 uses custom JIT trainer
      - Unifying ensures consistent PPO behavior across stages
      - See TODO in trainer_jit.py for implementation notes

Exit: Can collect and save rollouts in correct format

#### v0.12.1 - Quality Filtering
**Objective:** Filter out unstable segments

Tasks:
- [ ] Implement fall detection filter
- [ ] Implement velocity consistency filter
- [ ] Implement symmetry filter (optional)
- [ ] Generate quality statistics

Exit: Clean dataset with 0% falls, consistent quality

#### v0.12.2 - Dataset Validation
**Objective:** Verify dataset is suitable for AMP

Tasks:
- [ ] Compute feature statistics (mean, std)
- [ ] Verify dimensions match policy observation
- [ ] Test discriminator can distinguish (overfit test)

Exit: Dataset passes all validation checks

---

## Stage 3: Self-Imitation AMP (v0.13.x)

**Goal:** Use robot-native rollouts as AMP reference to regularize gait.

### Exit Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Discriminator accuracy | 0.55-0.75 | Healthy adversarial game |
| Gait smoothness | Improved vs Stage 1 | Visual + action rate metric |
| Gait consistency | Lower variance | Std of joint trajectories |
| Performance maintained | > 90% of Stage 1 | Velocity tracking, episode length |

### Value Delivered

- Smoother, more consistent gait
- Reduced mode switching
- AMP working as designed (style, not physics)
- Foundation for human-style injection

### Version Milestones

#### v0.13.0 - AMP with Robot Reference
**Objective:** Train AMP using Stage 2 dataset

Tasks:
- [ ] Configure AMP with robot-native dataset
- [ ] Verify discriminator trains correctly
- [ ] Monitor disc_acc (target: 0.55-0.75)

Exit: AMP training runs, discriminator is balanced

#### v0.13.1 - Style Regularization
**Objective:** Policy becomes more consistent

Tasks:
- [ ] Compare gait variance: Stage 1 vs Stage 3
- [ ] Evaluate smoothness metrics
- [ ] Tune AMP weight if needed

Exit: Measurable improvement in consistency/smoothness

#### v0.13.2 - Performance Validation
**Objective:** Ensure AMP didn't hurt task performance

Tasks:
- [ ] Compare velocity tracking
- [ ] Compare episode length
- [ ] Compare robustness to pushes

Exit: Performance >= 90% of Stage 1

---

## Stage 4: Human Style Blending (v0.14.x)

**Goal:** Gradually inject human motion characteristics while maintaining physical feasibility.

### Exit Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Human-likeness | Subjective improvement | Visual comparison |
| Physical stability | Maintained | Fall rate < 5% |
| Discriminator balance | 0.55-0.75 | With mixed dataset |

### Value Delivered

- Human-like gait aesthetics
- Physically feasible motion
- Tuneable human/robot style balance

### Version Milestones

#### v0.14.0 - Mixed Dataset (Option A)
**Objective:** Blend robot + GMR reference data

Tasks:
- [ ] Prepare GMR dataset (reuse existing)
- [ ] Create mixed dataset loader (configurable ratio)
- [ ] Start with 80% robot / 20% human
- [ ] Monitor discriminator stability

Exit: Training runs with mixed dataset

#### v0.14.1 - Feature-Level Bias (Option B)
**Objective:** Use human data as soft constraints

Tasks:
- [ ] Extract joint angle distributions from GMR
- [ ] Add distribution matching reward
- [ ] Add posture regularization

Exit: Policy biased toward human-like posture

#### v0.14.2 - Phase-Aligned Priors (Option C)
**Objective:** Align human style to gait phase

Tasks:
- [ ] Extract phase from robot gait
- [ ] Align human joint statistics to phase
- [ ] Add phase-conditioned style reward

Exit: Phase-aware human style injection

#### v0.14.3 - Final Tuning
**Objective:** Balance style and performance

Tasks:
- [ ] Tune human/robot blend ratio
- [ ] Optimize for deployment
- [ ] Final evaluation suite

Exit: Deployment-ready policy

---

## Quick Reference

| Stage | Version | Focus | Key Output |
|-------|---------|-------|------------|
| 1a | v0.11.x | Control Abstraction Layer (CAL) | CAL module, action semantics |
| 1b | v0.10.x | Task rewards, PPO only | Walking policy |
| 2 | v0.12.x | Data collection | Robot-native dataset |
| 3 | v0.13.x | Self-imitation AMP | Regularized policy |
| 4 | v0.14.x | Human style blend | Human-like policy |

---

## Pre-Training Checklist

Before starting any training:

```bash
# Validate MuJoCo setup
uv run python scripts/validate_training_setup.py

# Expected output: "All checks passed! Ready for training."
```

---

## Related Files

- Config: `training/configs/ppo_walking.yaml`
- Training: `training/train.py --config configs/ppo_walking.yaml`
- Validation: `scripts/validate_training_setup.py`
- Changelog: `training/CHANGELOG.md`
- CAL Proposal: `training/docs/CONTROL_ABSTRACTION_LAYER_PROPOSAL.md`
