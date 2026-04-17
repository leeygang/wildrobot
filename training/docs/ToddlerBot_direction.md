# ToddlerBot Direction Analysis

**Status:** Active reference note for the ToddlerBot-style WildRobot pivot
**Created:** 2026-03-31
**Last updated:** 2026-04-15
**Context:** After `v0.17.4t-crouch` failed the gate (eval_hard = 54.2%), evaluating
whether ToddlerBot's ZMP + reference-tracking recipe is a better path forward than
continuing bounded RL teacher iterations.

---

## 2026-04-15 Update: ToddlerBot, Cassie, and Residual-RL Boundary

This update reflects a fresh review of:

- ToddlerBot repo: `hshi74/toddlerbot`
- ToddlerBot 2025 paper / site
- Locomotion Beyond Feet 2026 paper / site
- Cassie 2024 versatile locomotion paper from the Berkeley Hybrid Robotics lab
- current WildRobot `v0.19.x` walking design and training stack

### ToddlerBot 2025/2026: what they are actually doing

ToddlerBot is not a good example of "PPO discovers walking from scratch."

The 2025 walking stack is better described as:

- a strong ZMP-based motion reference (`WalkZMPReference`)
- reference-conditioned walking rewards
- PPO learning robustness and execution around that prior
- broad domain randomization and sim-to-real realism

The 2026 "Locomotion Beyond Feet" system pushes this further:

- physics-grounded keyframes / reference motions
- reinforcement learning to make them robust and dynamically feasible
- terrain-specific motion-tracking policies
- failure recovery + vision-based skill selection on top

So ToddlerBot's public success is "reference first, RL robustifies," not
"weak nominal gait plus residual PPO invents locomotion."

### External references for "controller first, RL corrects"

There are real success cases for a hybrid stack where a conventional controller
or motion prior owns the base behavior and RL learns a corrective term:

- **Residual Reinforcement Learning for Robot Control (2019)**  
  canonical residual-RL paper; strongest evidence on manipulation
- **Agile and Versatile Robot Locomotion via Kernel-based Residual Learning
  (2023)**  
  closest quadruped locomotion example of controller/prior + RL residual
- **HiLMa-Res (2024)**  
  general locomotion + manipulation hierarchy, with RL corrections layered on
  top of a stronger nominal locomotion stack
- **RuN: Residual Policy for Natural Humanoid Locomotion (2025)**  
  strongest recent humanoid example; a motion prior provides natural gait and
  RL learns a lightweight residual correction
- **RL2AC (RSS 2024)**  
  not a pure residual policy, but same hybrid lesson: a base locomotion policy
  is complemented by a fast adaptive controller for robustness

Important caveat:

- these examples only support residual RL when the base controller / prior
  already owns most locomotion semantics
- they do **not** support expecting a small residual policy to create missing
  gait generation from a weak nominal controller

#### External proof points: what is actually proven

The cleanest statement is:

- controller / prior + residual **RL** is well supported
- controller / prior + residual **PPO** is supported by newer examples, but the
  evidence is still thinner than the broader residual-RL literature

| Project / Paper | Pattern | Domain | Real-world evidence | Why it matters for WildRobot | PPO note |
|---|---|---|---|---|---|
| Residual RL for Robot Control (2019) | conventional feedback controller + additive RL residual | contact-rich manipulation | yes | canonical proof that additive residual control works on hardware | not presented as a locomotion or PPO result |
| Kernel-based Residual Learning (2023) | frozen MPC-derived kernel + RL residual | quadruped locomotion | paper proves robust locomotion against perturbations and terrains | strongest locomotion proof that a model-based base policy can be widened by a residual learner | RL is explicit; PPO should not be inferred unless checked separately |
| HiLMa-Res (2024) | operational-space locomotion controller + residual RL for downstream loco-manip tasks | quadruped loco-manipulation | yes | shows hybrid controller + residual learning can scale beyond pure walking | residual RL is explicit; PPO is not the key claim in the abstract |
| RuN (2025) | pre-trained motion generator + lightweight residual correction policy | humanoid locomotion | yes, simulation and reality on Unitree G1 | closest humanoid analogue to "strong motion prior + residual correction" | strongest recent residual-humanoid example; use as evidence for the pattern more than for PPO specifically |
| RL2AC (2024) | learned locomotion policy + fast adaptive controller | quadruped locomotion | yes | not residual PPO, but confirms the broader hybrid lesson: split nominal skill from fast robustness correction | adjacent, not a residual-policy example |

Practical interpretation:

- if the question is "does hybrid control + learned correction work?" the
  answer is clearly yes
- if the question is "does small-residual PPO reliably rescue weak nominal
  bipeds?" the evidence is much weaker
- the external record supports using residual PPO only when the nominal prior is
  already strong enough that RL is genuinely correcting, not inventing, gait
  semantics

#### Closest PPO-flavored comparison points

For WildRobot, the most relevant "controller/prior + PPO-family training"
examples are:

- **ToddlerBot 2025**  
  PPO trained with a strong ZMP-based walking prior; not a small residual-action
  setup, but clear evidence that PPO works well when the prior already owns the
  locomotion semantics
- **RuN (2025)**  
  closest humanoid example of a motion prior plus lightweight learned
  correction; strongest external reference for "strong prior + learned
  correction" in humanoid walking, even though the main lesson is the residual
  boundary more than PPO branding
- **Cassie 2024**  
  not residual PPO, but the most important biped comparison because it shows a
  stronger reference-conditioned policy can outperform a tightly constrained
  residual action space

So the most defensible statement is:

- PPO with a strong locomotion prior is well supported
- residual learning with a strong controller/prior is well supported
- strict "small residual PPO rescues weak nominal biped gait" is still not
  strongly supported by external evidence

### Cassie 2024: the important comparison for WildRobot

The most relevant Cassie reference is:

- **Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal
  Locomotion Control** (Li et al., 2024)

This paper is highly relevant because it is a strong real-world biped result,
but its design boundary is **not** the same as current WildRobot.

#### What Cassie does

Cassie uses:

- direct RL control for a torque-controlled human-sized biped
- a dual-history policy with both short and long robot I/O history
- a skill-specific reference-motion preview as policy input
- joint-level PD control under the learned policy

For walking specifically:

- the walking skill is parameterized by commands
- the policy receives a preview of a reference motion `q_t^r`
- that reference is built from a gait library
- the paper describes a walking reference set of **1331** Bezier gait motions
- running uses motion-capture retargeting
- jumping uses hand-authored animation

So Cassie is **not** "no prior." It is:

- strong motion prior
- direct adaptive RL policy
- rich temporal context
- broad task randomization

#### What Cassie says about residual learning

Cassie's ablations are especially important for WildRobot because the paper does
**not** generally endorse residual action spaces for dynamic biped locomotion.

Their discussion argues that performance is best when RL can jointly learn
trajectory optimization and real-time control, and they note that this holds
as long as the action space is not corrupted by the added reference motion, as
in their residual-action ablation. In practical terms:

- they found strong benefit from reference motion as **input / prior**
- but not from tightly constraining the action space as a small residual around
  that reference

This is the clearest external warning sign for the current WildRobot boundary.

#### Cassie vs WildRobot

| Dimension | Cassie 2024 | WildRobot `v0.19.x` |
|---|---|---|
| Robot class | Human-sized, torque-controlled biped | Small servo-driven position-controlled biped |
| Base gait prior | Large gait library / mocap / animation priors | Online analytical `walking_ref_v2` controller |
| Policy authority | Direct locomotion policy with broad adaptive authority | Residual around nominal `q_ref` |
| Policy memory | Short + long I/O history | Current walking PPO is mostly feedforward, no comparable long-history adaptive structure |
| Reference role | Reference preview guides policy | Nominal controller already commits to gait semantics and PPO only corrects |
| RL burden | Robustify and adapt a strong prior | Currently still tempted to rescue nominal gait-generation gaps |

The key implication:

- Cassie succeeds because the policy is allowed to be the main locomotion
  controller while using a strong reference prior
- WildRobot currently asks a **smaller** residual policy to fix a nominal gait
  that is still not fully owning propulsion and visible step release

That is a much harder and less supported problem.

### Updated interpretation for WildRobot

ToddlerBot and Cassie both support the same strategic lesson:

- a strong locomotion prior is valuable
- RL is excellent at robustness, adaptation, and dynamic feasibility
- but RL should not be asked to invent missing gait semantics from a weak base
  under small residual authority

For WildRobot, that means:

1. `walking_ref_v2` still has to pass the `v0.19.4-C/D` nominal handoff gate.
2. Residual PPO is justified only after nominal visibly owns:
   - swing release
   - step amplitude
   - propulsion
3. If WildRobot wants to move closer to Cassie, the next architectural step is
   not "bigger residual weight." It is:
   - stronger reference priors
   - richer history-conditioned policy inputs
   - possibly a later transition from strict residual correction toward a more
     direct reference-conditioned locomotion policy

### Strategic options if WildRobot wants broader adaptation

If the goal is broader adaptation across speed changes, disturbances, and later
terrain variations, there are three realistic architecture paths.

| Option | Runtime structure | What the "base" is | What PPO learns | Broader adaptation upside | Development cost | Debug cost | Maintenance cost |
|---|---|---|---|---|---|---|---|
| A. Stay controller + small residual PPO | analytic walking controller + bounded residual action | online controller (`walking_ref_v2` + IK) | correction only | lowest | low-medium | medium-high | medium |
| B. Stronger prior + residual / tracking PPO | stronger reference family + policy trained to track / correct around it | command-conditioned prior or reference library | tracking robustness + moderate adaptation | medium-high | medium | medium | medium-high |
| C. Cassie-style reference-conditioned direct locomotion policy | reference preview as input, policy owns main action generation | gait library / keyframes / mocap / planner-generated reference | main locomotion control with prior guidance | highest | high | high | medium-high |

Interpretation:

- **Option A** is the easiest to keep stable and interpretable, but it inherits
  the controller's assumptions. It is the weakest path for broad adaptation.
- **Option B** is the best middle ground for WildRobot if the goal is "broader
  than current residual PPO" without taking on the full risk of a direct policy.
- **Option C** has the highest long-term upside, but it requires the strongest
  reference assets, observation design, training budget, and debugging
  discipline.

My professional recommendation:

- for near-term delivery, keep `v0.19.x` disciplined enough to prove nominal
  gait ownership
- for the next architecture step, target **Option B**
- treat **Option C** as a later-generation platform move, not the next patch

### What "controller" vs "prior" means for WildRobot

This distinction matters because it changes both training behavior and long-term
 adaptability.

- **Controller**: can run the robot by itself and produce actual actions
  online.  
  Example for WildRobot: `walking_ref_v2` + phase logic + IK.
- **Prior / reference**: provides structured motion guidance, but the learned
  policy still has substantial control authority.  
  Example: command-conditioned gait library, keyframe sequence, motion preview,
  or reduced-order trajectory family.

Practical difference:

- controller-first systems are easier to debug one failure at a time
- prior-first systems are usually better for scaling to broader conditions,
  because the policy is allowed to become the main controller instead of a small
  correction layer

### Options to build a prior / reference for WildRobot

If WildRobot wants broader adaptation, the next question is not only
"controller or prior?" It is also "what kind of prior is realistic to build and
maintain on this robot?"

#### Concrete prior candidates: ZMP, ALIP, Open Duck Mini-style references, AMASS

These are not equally good fits for WildRobot.

| Candidate | What it provides | Fit for WildRobot | Main upside | Main downside | Recommended role |
|---|---|---|---|---|---|
| ZMP reference family | stable flat-ground COM / footstep / pelvis trajectories | good for first deployable prior | simple, interpretable, good for stop/start and quiet walking | conservative, less dynamic, weaker for broad disturbance adaptation | good first prior layer |
| ALIP / capture-point / step-placement prior | dynamic COM and footstep planning prior | best medium-term fit | better disturbance handling and speed adaptation, stronger locomotion semantics | harder to tune and realize on a small servo robot | best core prior direction |
| Open Duck Mini-style reference motions / imitation clips | hand-authored or generated motion clips used for imitation / reference tracking | useful for style bootstrapping, not enough alone | quick way to define gait phase and behavior clips | high manual burden, dynamic mismatch risk, limited coverage | secondary tool, not core prior |
| AMASS / human mocap prior | large human motion dataset | poor first fit | broad motion diversity and style | major morphology-retargeting mismatch for a 9-DOF small robot | later-stage research tool, not first walking prior |

Practical recommendation:

- do **not** start from AMASS
- do **not** rely only on hand-authored imitation clips
- start from a **ZMP / reduced-order analytical reference family**
- then move that family toward **ALIP-like step-placement and COM dynamics**
- use motion clips only as supplemental assets for start / stop / turn / style
  shaping if needed

#### Option 1: stronger analytical reference family

Build on the current controller path:

- keep reduced-order structure (`LIPM` / `ALIP` / capture-point / DCM)
- generate a cleaner command-conditioned family of nominal trajectories
- expose more of that reference to PPO as inputs

What it looks like:

- speed-conditioned step length, step time, pelvis, and COM trajectories
- explicit preview of future nominal states
- still no external dataset requirement

Cost profile:

- development: **medium**
- debug: **medium**
- maintenance: **medium**

Pros:

- closest to the current WildRobot codebase
- easiest bridge from current residual PPO
- preserves interpretability and debugging leverage

Cons:

- still limited by the analytical model family
- harder to cover very diverse scenarios cleanly
- can become brittle if too many heuristics accumulate

Best fit:

- next step after `v0.19.x` if the goal is broader adaptation without a full
  architecture reset

Concrete guidance:

- if the team wants the safest first prior, start with **ZMP-like flat-ground
  reference families**
- if the team wants the best long-term locomotion prior, evolve that family
  toward **ALIP / capture-point-driven step placement**

#### Option 2: offline gait library from the analytical planner

Use the analytical controller to generate many trajectories offline, then train
against that library as a prior.

What it looks like:

- sweep command space offline
- store nominal state/action trajectories
- give PPO current + previewed reference states from the library

Cost profile:

- development: **medium-high**
- debug: **medium**
- maintenance: **medium-high**

Pros:

- cleaner separation between "reference asset" and "runtime control"
- easier to inspect coverage gaps in command space
- closer to ToddlerBot / Cassie reference-preview training

Cons:

- requires tooling for trajectory generation, indexing, interpolation, and
  validation
- library refresh becomes a maintenance task whenever the robot or model
  changes

Best fit:

- strongest near-term route to a real prior without needing mocap or a large
  data pipeline

Concrete guidance:

- this is the most natural place to use **ZMP** or **ALIP** for WildRobot
- build the library from your own analytical planner rather than from human
  mocap first
- make the library command-conditioned over:
  - forward speed
  - stop / resume
  - in-place stepping
  - yaw turning
  - moderate push-recovery responses

#### Option 3: keyframe / animation / hand-authored motion prior

Build a small set of target motions first, then use RL to make them physically
feasible and robust.

What it looks like:

- authored gait cycles
- skill-specific sequences for start, stop, walk, turn, recovery
- reference tracking reward around those motions

Cost profile:

- development: **medium-high**
- debug: **high**
- maintenance: **high**

Pros:

- fast way to define desired style and phase structure
- useful if the team wants explicit control over gait appearance

Cons:

- easy to create references that are kinematically nice but dynamically wrong
- large manual burden as behaviors multiply
- high risk of style lock-in unless the library becomes broad

Best fit:

- style-sensitive work, demos, or a small skill set with strong manual design
  control

Concrete guidance:

- this is the closest category to what projects like **Open Duck Mini** are
  experimenting with when they mention reference-motion generation for
  imitation-learning-style training
- useful for bootstrapping:
  - start
  - stop
  - turning
  - recovery gestures
- not the best single source of truth for WildRobot's whole walking stack
  because coverage and maintenance cost grow quickly

#### Option 4: trajectory-optimization or teacher-generated reference dataset

Generate references from a stronger teacher:

- trajectory optimization
- MPC
- offline optimal control
- higher-fidelity planning stack

Then use those trajectories as the training prior for the deployable policy.

Cost profile:

- development: **high**
- debug: **high**
- maintenance: **high**

Pros:

- potentially strongest prior quality
- can cover disturbances and richer scenarios if the teacher is strong

Cons:

- highest infrastructure cost
- hardest to keep synchronized with robot/model changes
- debugging failures becomes a multi-layer problem: teacher, dataset, student,
  runtime

Best fit:

- later-stage platform investment, not the cheapest next move for WildRobot

Concrete guidance:

- if WildRobot later wants a stronger teacher than hand-tuned ZMP / ALIP,
  teacher-generated datasets are a better next step than jumping directly to
  human mocap
- for a small servo biped, this is usually a better fit than **AMASS** because
  the generated references already respect the robot's morphology and control
  limits

### Recommendation if broader adaptation is the goal

For WildRobot, the best path is:

1. keep the current controller-first branch strict enough to prove nominal gait
   ownership
2. then move toward a **prior/reference family**, not just a larger residual
3. specifically, prefer:
   - **Option 1** as the shortest path
   - **Option 2** as the best medium-term architecture target
4. avoid jumping directly from today's `walking_ref_v2` residual setup to a
   fully direct policy unless the team is ready for a larger training and
   tooling investment

In short:

- if you want the cheapest near-term progress, improve the controller
- if you want broader adaptation, build a real prior/reference family
- if you want the highest eventual ceiling, move toward reference-conditioned
  direct control, but expect higher development and debugging cost

### Practical takeaway

ToddlerBot and Cassie do **not** invalidate the current WildRobot direction.
They sharpen the boundary:

- **good use of residual RL:** correction around an already valid gait
- **bad use of residual RL:** trying to repair a nominal controller that still
  cannot reliably produce visible stepping and sufficient propulsion

This is exactly why the tightened `v0.19.4-C/D` handoff gate is the right
discipline for WildRobot.

### References

- ToddlerBot repo: `https://github.com/hshi74/toddlerbot`
- ToddlerBot 2025 site: `https://toddlerbot.github.io/`
- Locomotion Beyond Feet 2026 site: `https://locomotion-beyond-feet.github.io/`
- Cassie 2024 paper: `https://hybrid-robotics.berkeley.edu/publications/Cassie_RL_Versatile_Locomotion.pdf`
- Residual RL 2019: `https://residualrl.github.io/`
- Kernel-based Residual Learning 2023: `https://arxiv.org/abs/2302.07343`
- HiLMa-Res 2024: `https://arxiv.org/abs/2407.06584`
- RL2AC 2024: `https://www.roboticsproceedings.org/rss20/p060.html`
- RuN 2025: `https://arxiv.org/abs/2509.20696`

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
