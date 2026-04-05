# ToddlerBot Direction Analysis

**Status:** Active reference note for the ToddlerBot-style WildRobot pivot
**Created:** 2026-03-31
**Last updated:** 2026-04-05
**Context:** After `v0.17.4t-crouch` failed the gate (eval_hard = 54.2%), evaluating
whether ToddlerBot's ZMP + reference-tracking recipe is a better path forward than
continuing bounded RL teacher iterations.

---

## Route Call

If WildRobot follows ToddlerBot's direction, the main architecture rule is:

- the deployed runtime artifact remains a learned policy
- model-based pieces move into the reference / teacher / validation layer

That means:

- reduced-order planning, capture-point logic, and IK remain useful
- but they should not become the long-term runtime controller
- PPO should be structured around a nominal scaffold rather than asked to
  discover locomotion from scratch

This is the practical meaning of a ToddlerBot-style pivot on current hardware.

## Three-Way Comparison: ToddlerBot vs LeRobot vs WildRobot

### What Each Project Is

| | ToddlerBot | LeRobot | WildRobot |
|---|---|---|---|
| **Type** | Full-stack bipedal humanoid | Robotics ML framework/toolkit | Full-stack bipedal humanoid |
| **Primary task** | Walking + manipulation | Manipulation (arms, grippers) | Standing push recovery → walking |
| **Hardware class** | 30-DOF, 3.4kg, Dynamixel servos | Robot arms (SO-100, Koch, etc.) | 9-DOF, ~1.5kg, HTD-45H servos |
| **Height** | 0.56m | N/A | ~0.47m |
| **Locomotion?** | Yes — core feature | No — no bipedal locomotion | Yes — core feature |
| **Open source** | Yes (code + hardware) | Yes (code + models + datasets) | Private |

### Architecture

| | ToddlerBot | LeRobot | WildRobot |
|---|---|---|---|
| **Simulator** | MuJoCo MJX/JAX | MuJoCo (via Gymnasium) | MuJoCo MJX/JAX |
| **Training backend** | JAX (Brax PPO) + PyTorch (RSL-RL) | PyTorch only | JAX (custom PPO) |
| **Parallelism** | 1024 envs (MJX vmap) | Single env or small batch | 512 envs (MJX vmap) |
| **Policy arch** | MLP (512,256,128) ELU | ACT (Transformer), Diffusion, VQ-BeT, VLAs | MLP (custom) |
| **Action space** | Joint position targets (residual) | Joint positions or end-effector | Joint position targets (residual) |
| **Control freq** | 50 Hz | 10-30 Hz (task-dependent) | 50 Hz |
| **Deployed artifact** | Single policy (no runtime teacher) | Single policy | Single policy (no runtime teacher) |

### Training Approach

| | ToddlerBot | LeRobot | WildRobot |
|---|---|---|---|
| **Primary method** | **RL (PPO) + reference tracking** | **Imitation learning** (demos) | **RL (PPO) + bounded teacher** |
| **Reference motion** | ZMP-generated walking trajectory | Human demonstrations (teleop) | None (heuristic teacher targets) |
| **Reward design** | Imitation + regularization + survival | N/A (behavioral cloning loss) | Survival + shaping + teacher rewards |
| **RL algorithm** | PPO (Brax) or SAC (RSL-RL) | HIL-SERL (SAC, on-robot), TDMPC | PPO (custom JAX) |
| **Training data** | Sim-generated from ZMP reference | Human teleop demonstrations | Sim-generated from curriculum |
| **Domain rand** | Comprehensive (friction, mass, Kp/Kd, backlash, armature, torque limits) | Minimal (real-robot focused) | Planned but not yet active (v0.17.3b) |
| **Curriculum** | Reference motion difficulty | N/A | Push force curriculum |
| **Key innovation** | Motor sysID + ZMP reference | Shared datasets + pretrained VLAs | Bounded teacher for step geometry |

### Sim-to-Real

| | ToddlerBot | LeRobot | WildRobot |
|---|---|---|---|
| **Transfer method** | **Zero-shot sim2real** | **No sim2real** (trains on real robot) | Planned zero-shot (not yet attempted) |
| **Sim fidelity** | Motor sysID (chirp signal), backlash modeling | N/A | Standard MuJoCo (no motor sysID) |
| **Domain rand for transfer** | Yes — comprehensive | N/A | Planned (v0.17.3b) |
| **Sensor noise modeling** | AR(1) IMU bias walk, gyro filtering | N/A | Planned (simple Gaussian) |
| **Action delay** | 1-step buffer | N/A | Planned (1-step) |
| **Backlash modeling** | Yes, [0.02, 0.1] rad | N/A | Not planned |
| **Motor torque rand** | Yes, x[0.9, 1.1] | N/A | Not planned |
| **Demonstrated transfer** | Walking at 0.25 m/s, zero-shot | Direct real-world training | Not yet attempted |

### Domain Randomization Detail

| Parameter | ToddlerBot | WildRobot (v0.17.3b plan) |
|---|---|---|
| Friction | [0.4, 1.0] | [0.5, 1.0] |
| Damping | x[0.8, 1.2] | x[0.9, 1.1] |
| Mass | +/-20% body, +10% hand | x[0.9, 1.1] |
| Kp/Kd | x[0.9, 1.1] | x[0.9, 1.1] |
| Frictionloss | x[0.8, 1.2] | x[0.9, 1.1] |
| Armature | x[0.8, 1.2] | not planned |
| Backlash | [0.02, 0.1] rad | not planned |
| Motor torque limits | x[0.9, 1.1] | not planned |
| Action delay | 1-step buffer | 1-step planned |
| Push forces | [1.0, 3.0]N, 0.2s | [5N-10N], ongoing |
| IMU noise | AR(1) bias walk, gyro fc=0.35 | simple Gaussian planned |

### PPO Hyperparameters

| Parameter | ToddlerBot | WildRobot |
|---|---|---|
| Learning rate | 3e-5 | 1e-4 |
| Num envs | 1024 | 512 |
| Discount (gamma) | 0.97 | 0.99 |
| GAE lambda | 0.95 | — |
| Clip epsilon | 0.2 | 0.15 |
| Entropy cost | 1e-3 | 5e-3 |
| Episode length | 1000 | 500 |
| Total steps | 500M | ~75M |
| Network | (512,256,128) ELU | custom MLP |
| Action scale | 0.25 | per-joint span (0.5-1.5) |
| Initial noise std | 0.5 | — |

### Reward Weights (Walking)

| Component | ToddlerBot | WildRobot (standing) |
|---|---|---|
| survival / alive | 10.0 | 15.0 |
| motor_pos (ref tracking) | 5.0 | N/A (no reference) |
| torso_quat (ref tracking) | 5.0 | N/A |
| lin_vel_xy (ref tracking) | 5.0 | N/A |
| ang_vel_z (ref tracking) | 5.0 | N/A |
| torso_pos_xy | 2.0 | N/A |
| ang_vel_xy | 2.0 | N/A |
| feet_air_time | **500.0** | N/A |
| feet_distance | 1.0 | N/A |
| feet_clearance | 1.0 | N/A |
| action_rate | 1.0 | -0.01 |
| motor_torque | 0.1 | -0.001 |
| energy | 0.01 | N/A |
| collision | 0.1 | N/A |
| feet_slip | 0.05 | -0.5 |
| orientation | N/A (in quat) | -1.0 |
| posture | N/A (in motor_pos) | 0.5 |
| height_target | N/A (in torso_pos) | 0.5 |
| teacher_target_step_xy | N/A | 0.3 |

---

## What WildRobot Can Learn

### From ToddlerBot (directly relevant)

1. **Motor system identification** — Characterize HTD-45H dynamics via chirp
   signal. ToddlerBot does this once per motor model. Cheap and high-impact for
   sim2real.

2. **Backlash simulation** — ToddlerBot randomizes gear backlash [0.02, 0.1]
   rad. Hobby servos have significant backlash; modeling this is likely critical
   for sim2real.

3. **Reference motion as training scaffold** — ToddlerBot's walking success
   comes from ZMP reference tracking, not free exploration. This aligns with the
   `v0.17.4t-crouch-teacher` direction. Consider generating simple capture-point
   recovery trajectories as reference motions.

4. **AR(1) IMU noise model** — More realistic than simple Gaussian for cheap
   IMUs.

5. **Lower discount factor** (0.97 vs 0.99) — Emphasizes near-term rewards,
   potentially better for reactive recovery.

6. **Tighter action scale** (0.25 vs 0.5-1.5) — Constrains exploration near
   default pose, faster convergence.

7. **Much longer training** (500M vs 75M steps) — More experience budget.

8. **Reference motion as a scaffold, not a crutch** — The nominal trajectory
   should already contain a coherent transition into support and stepping.
   PPO improves robustness around it instead of inventing that sequence.

### From LeRobot (different problem class)

1. **HIL-SERL for real-robot fine-tuning** — Once a sim-trained policy deploys,
   human-in-the-loop SAC with safety bounds could fine-tune on hardware.

2. **Vision-based success detection** — Train a classifier from ~10 demos to
   detect "recovered from push" as reward for on-robot training.

3. **LeRobot is NOT relevant for locomotion.** No bipedal walking, no push
   recovery, no balance control. Don't look to it for standing/walking
   methodology.

---

## Switching to ToddlerBot's Approach: Feasibility

### What "Following ToddlerBot" Actually Means

ToddlerBot should not be read as "pure PPO discovered locomotion from scratch."
Its public direction is better understood as:

- calibrated robot platform
- high-fidelity digital twin
- actuator system identification
- structured locomotion reference or prior
- history / delay / realism modeling
- learned runtime policy as the deployed artifact

For WildRobot, "follow ToddlerBot" means:

- build a robot-learning platform, not only a standing experiment
- make the runtime artifact a deployable learned policy
- treat reduced-order planners and IK as nominal-reference infrastructure
- make sim-to-real an explicit engineering loop

This is compatible with capture-point, `LIPM`, `ALIP`, keyframes, and hybrid
walking references. It is not compatible with using heavy runtime MPC as the
center of the stack.

### What Would Change

| Component | Current WildRobot | ToddlerBot-Style |
|---|---|---|
| Reference motion | None | ZMP-generated standing/recovery trajectories |
| Reward structure | Survival + shaping | Imitation tracking + regularization + survival |
| Action mapping | Home-centered residual | Default-pose residual (same concept, tighter scale) |
| Domain rand | Not active | Comprehensive (add backlash, motor sysID) |
| Sensor noise | Not active | AR(1) IMU bias walk |
| Training budget | ~75M steps | 500M steps |
| Observation | No phase signal | Phase signal for gait cycle |

### What Can Be Reused

- MuJoCo MJX/JAX stack (same simulator)
- PPO training loop (similar structure)
- Env vmap architecture (same pattern)
- Policy contract / export pipeline
- Eval ladder infrastructure
- Hardware runtime (50 Hz position control, same pattern)

### What Needs New Implementation

1. **ZMP / capture-point reference generator** for standing recovery
2. **Reference tracking reward** (exponential tracking with sigma tuning)
3. **Phase signal** in observations (for gait cycle timing)
4. **Motor sysID** pipeline (chirp signal + model fitting for HTD-45H)
5. **Backlash simulation** in env
6. **AR(1) IMU noise model**
7. **Extended domain randomization** (armature, torque limits, passive/active ratio)

### Difficulty Assessment

**Overall: Medium.** The core simulator and training infrastructure are the same
(MuJoCo MJX + JAX PPO). The main new work is the reference motion generator and
the reward restructuring. The sim2real hardening (backlash, sysID, noise) is
incremental.

| Work Item | Effort | Risk |
|---|---|---|
| ZMP reference for standing recovery | Medium | Low — simpler than walking ZMP |
| Reference tracking reward | Low | Low — well-understood exponential form |
| Motor sysID (HTD-45H) | Medium | Low — one-time hardware characterization |
| Backlash + domain rand | Low | Low — additive to existing env |
| AR(1) IMU noise | Low | Low — straightforward signal processing |
| Observation restructuring (phase) | Low | Low — add one dimension |
| Training budget increase (500M) | None (just time) | None — ~10x current wall time |
| Reward weight tuning | Medium | Medium — iterative, but ToddlerBot weights give a starting point |

### Key Differences That Make It Simpler for WildRobot

1. **Standing recovery is simpler than walking.** ToddlerBot needs a full ZMP
   gait cycle. WildRobot's standing recovery needs a simpler reference: "stay
   upright, and if pushed hard, execute a capture-point step then return to
   standing." This is a subset of what ToddlerBot solves.

2. **Fewer DOFs.** 9 actuated joints vs 30 means smaller action space, faster
   training, and less coordination to learn.

3. **No manipulation.** WildRobot doesn't need arm coordination during recovery,
   simplifying the reference motion.

### Concrete Hardware Delta vs WildRobot v2

The most important difference is not only "ToddlerBot uses RL." The more
important difference is that ToddlerBot has more mechanically useful authority
for locomotion and a platform built around learning.

For WildRobot v2, the main practical deltas are:

- no ankle roll in the current lower-body design
- less frontal-plane authority
- less whole-body redundancy for recovery
- weaker platform tooling around actuator realism and sim-to-real evaluation

The most valuable software-only moves toward ToddlerBot are:

- policy-centered runtime architecture
- actuator SysID for `HTD-45H`
- backlash, delay, and torque-limit realism
- history / delay observations
- reference-guided locomotion rather than free PPO search
- explicit sim-vs-real replay and evaluation tooling

The most valuable future hardware move, if a v3 is considered, is:

- add ankle roll to each leg

### Key Risks

1. **ZMP may be too simple for push recovery.** ZMP assumes flat-foot contact
   and quasi-static balance. Hard pushes create dynamic states where ZMP is less
   applicable. Capture-point / LIPM extensions handle this better.

2. **Standing ≠ walking.** ToddlerBot's reward structure is designed for
   continuous periodic gait. Standing recovery is aperiodic and event-driven.
   The phase signal and feet_air_time rewards don't directly transfer.

3. **Hardware gap.** WildRobot has no ankle roll (passive only), which limits
   the bracing strategies available compared to ToddlerBot's 6-DOF legs.

---

## Recommendation

The ToddlerBot comparison validates two things:

1. **Reference-tracking RL works for small humanoids.** Free exploration RL
   struggles (WildRobot's experience confirms this). Reference tracking
   dramatically reduces the search space.

2. **Comprehensive sim2real hardening is essential.** Backlash, motor sysID,
   and AR(1) noise are not optional extras — they're core to zero-shot transfer.

### Suggested Path

**Option A: Adopt ToddlerBot's reference-tracking approach within RL
(`v0.17.4t-crouch-teacher` evolution)**

- Generate capture-point recovery reference trajectories (simpler than ZMP gait)
- Use exponential tracking rewards (ToddlerBot's `exp(-sigma * error^2)` form)
- Add ToddlerBot-style domain rand (backlash, motor sysID)
- Keep the existing PPO + MJX stack
- This is essentially what `v0.17.4t-crouch-teacher` was going to do, but with
  a more structured reference trajectory instead of ad-hoc teacher targets

**Option B: Directly use ToddlerBot's codebase**

- Fork ToddlerBot's training pipeline
- Replace their MJCF model with WildRobot's
- Adapt walk_env.py → standing_recovery_env.py
- Use their domain rand, noise, and sysID infrastructure
- Benefit: proven sim2real pipeline
- Cost: migration effort, unfamiliar codebase, walking-centric assumptions

**Option C: `v0.18` model-based (already in the plan)**

- Position-level model-based controller generating recovery trajectories
- RL tracks those trajectories (similar to ToddlerBot but with explicit MPC)
- Higher confidence for complex recovery but more implementation work

**Recommended: Option A** — it's the smallest delta from the current codebase,
uses ToddlerBot's proven ideas (reference tracking, domain rand, motor sysID)
without the migration cost, and aligns with the existing plan trajectory.

### Recommended Goal Shift

If this pivot is accepted, the top-level goal should shift from:

- "find a controller that can survive pushes while standing"

to:

- "build an ML-compatible locomotion platform for WildRobot"

On current WildRobot v2 hardware, the first realistic product goal is:

- zero-shot or low-touch sim-to-real forward walking
- reliable stop and quiet standing
- in-place yaw turning
- moderate push recovery during standing and slow walking

This is a better first target than trying to match ToddlerBot's full locomotion
class directly.

## Concrete Milestones

### Milestone 0: Platform Pivot

- freeze the standing-only branch as a completed exploration path
- define a locomotion-first repo structure around runtime, assets, references,
  training, policies, and tools
- standardize one shared sim/real log schema
- define reference-guided RL as the mainline direction

### Milestone 1: Digital Twin and Actuator Realism

- add `HTD-45H` actuator realism files and SysID workflow
- add replay tools for sim-vs-real comparison
- model delay, backlash, frictionloss, armature, and IMU realism

### Milestone 2: Reference Generator

- implement reduced-order locomotion references
- generate bounded task-space foot and pelvis targets
- convert them into nominal joint targets via IK
- validate the nominal prior before resuming PPO

### Milestone 3: Walking Policy

- train residual PPO around the nominal reference
- add history, delay, and reference-conditioned observations
- target forward walking, stop, and quiet standing first

### Milestone 4: Transfer Hardening

- broaden domain randomization
- strengthen sim-vs-real replay and failure analysis
- add stop/start transitions and in-place yaw

### Milestone 5: Recovery and Robustness

- add moderate disturbance recovery during standing and slow walking
- improve the nominal reference rather than replacing the policy-centered stack

### Milestone 6: Platformization

- standardize train/eval/replay/deploy entry points
- add reusable policy packaging and logging workflows
- keep the repo usable as a robot-learning platform rather than a one-off
  experiment branch
