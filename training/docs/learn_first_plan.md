# Learn First, Retarget Later: Robot-Native Walking Policy

**Version:** v0.15.4
**Status:** Active
**Last Updated:** 2026-03-14

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
