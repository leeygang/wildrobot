# ToddlerBot Direction for WildRobot

## Purpose

This note is for discussing what it would mean to pivot WildRobot toward the
general direction of ToddlerBot, both in architecture and in project goals.

The key question is not just "RL or MPC?" The more important question is:

- What is the deployed controller artifact?
- What role do model-based components play?
- What hardware capability is required for the target locomotion class?

## Route Call

If WildRobot wants to follow ToddlerBot's direction, the main change is:

- the long-term runtime controller should be a learned policy
- model-based components should move into the training/reference/tooling layer

This is different from using planner + IK as the final controller. It is also
different from trying to solve locomotion by pure reward-tuned PPO with no
structure.

The ToddlerBot pattern is closer to:

- calibrated robot platform
- high-fidelity digital twin
- system identification for actuators and passive effects
- structured reference or motion prior where needed
- learned policy as the deployed skill
- explicit sim-to-real validation tooling

For WildRobot, that means `v0.18` should be reinterpreted if this pivot is
accepted:

- keep reduced-order planning / capture-point / IK work
- use it as a reference generator, teacher, validator, or motion prior
- do not treat planner + IK as the long-term final control architecture

## What "Following ToddlerBot" Actually Means

ToddlerBot should not be read as "pure PPO discovered locomotion from scratch."
Its public stack is structured around a digital twin, motor identification,
reference-guided locomotion, domain randomization, history/delay modeling, and
deployment of a learned runtime policy.

So "follow ToddlerBot" for WildRobot means:

- build a robot-learning platform, not only a standing experiment
- make the runtime artifact a deployable learned policy
- make sim-to-real a first-class engineering loop
- allow model-based methods to support learning instead of competing with it

This is compatible with using capture-point, ALIP, keyframes, or simple gait
references. It is not compatible with making heavy runtime MPC the center of
the stack.

## Recommended Goal Shift

The current standing push-recovery ladder was a good way to explore the limits
of direct RL on the existing system. If the project pivots to ToddlerBot's
direction, the top-level goal should change from:

- "find a controller that can survive pushes while standing"

to:

- "build an ML-compatible locomotion platform for WildRobot"

The first locomotion goal should then be hardware-realistic.

For current WildRobot v2, the recommended first goal is:

- zero-shot or low-touch sim-to-real forward walking
- in-place yaw turning
- reliable stopping and quiet standing
- moderate push recovery during standing and slow walking

This is a better first target than "match ToddlerBot omnidirectional walking"
because WildRobot v2 has clear hardware limits relative to ToddlerBot.

## Architecture Changes

### 1. Keep a Learned Runtime Policy as the Main Controller

If the project pivots, the deployed controller should remain a learned policy.
That aligns with ToddlerBot's direction and keeps the long-term architecture
compatible with richer learned skills later.

The practical implication is:

- policy output remains the runtime command interface
- planner / IK / capture-point logic becomes training infrastructure
- references can shape learning without becoming a hard online dependency

### 2. Split the Stack into Platform Layers

WildRobot should move toward clearer platform boundaries:

- `robot/` or `runtime/`: hardware interface, servo calibration, safety limits
- `sim/` or `assets/`: digital twin and robot model maintenance
- `references/`: gait priors, capture-point tools, reduced-order planners,
  keyframes, nominal motion generators
- `training/`: RL, imitation, curriculum, domain randomization
- `policies/`: inference wrappers, exported policy runners
- `tools/`: SysID, replay, sim2real evaluation, logging, visualization

This is the main architecture move toward ToddlerBot.

### 3. Add a Sim-to-Real Tooling Loop

ToddlerBot is strong because sim-to-real is not left as a vague hope. The stack
includes explicit motor identification and sim-vs-real comparison tooling.

WildRobot should add:

- actuator characterization for `HTD-45H`
- backlash and passive-effect modeling
- delay modeling
- log replay tools
- sim-vs-real trace comparison for joint position, velocity, action, and IMU
- one reproducible calibration flow for all joints and sensors

### 4. Use Structured RL Rather Than Free PPO Search

The training direction should become:

- command-conditioned locomotion
- reference-guided or prior-guided learning
- residual action around a nominal pose or nominal gait target
- observation history and delay
- stronger transfer hardening

This is the clearest training change relative to the current standing-only
branching path.

## Concrete Hardware Delta vs WildRobot v2

The biggest control difference is not just "ToddlerBot uses RL." The bigger
difference is that ToddlerBot has more mechanically useful authority for
locomotion and a platform built around learning.

### WildRobot v2 Today

From the current repo and runtime config, WildRobot v2 has:

- 19 actuated joints in the model
- locomotion-critical lower body of 8 leg actuators plus 1 waist yaw actuator
- no ankle roll in the current lower-body design
- position-controlled `HTD-45H` servos, with a documented `4.41 Nm` torque
  limit in project docs
- `50 Hz` runtime control
- one `BNO085` IMU
- four foot switches: left toe, left heel, right toe, right heel

In practice, the locomotion-relevant body is much less redundant than
ToddlerBot's.

### ToddlerBot Delta That Matters Most

ToddlerBot's public platform is a small humanoid with:

- 30 DoFs total
- 6-DoF legs, including ankle roll
- richer whole-body redundancy through arms, neck, and a 2-DoF waist
- a digital-twin and system-identification workflow built into the project

For locomotion, the most important hardware delta is frontal-plane and
whole-body authority.

WildRobot v2 currently gives up several things ToddlerBot has:

- ankle-roll authority for lateral balance and foot placement shaping
- more torso/waist authority for body reconfiguration
- more mechanical redundancy for coordinated whole-body recovery

That directly affects the locomotion class you can expect.

### Software-Only Delta

These changes do not require a new robot:

- move to a policy-centered runtime architecture
- add actuator SysID for `HTD-45H`
- add backlash, torque-limit, frictionloss, and armature randomization
- improve IMU noise and delay modeling
- add observation history
- add sim2real trace evaluation and replay
- add nominal gait/reference infrastructure
- add logging and dataset tooling for future imitation / keyframe workflows

This is the lowest-risk way to move toward ToddlerBot's direction while keeping
WildRobot v2.

### Medium Hardware Delta

These changes improve the platform without fully redesigning the legs:

- improve waist authority if another waist DoF is feasible
- improve foot/contact sensing robustness
- improve servo mounting stiffness and reduce backlash variation
- add cleaner calibration fixtures and repeatable assembly references
- add more convenient logging / teleop / observation hardware if later skills
  need it

These changes help platform quality even if the leg kinematics stay the same.

### High Hardware Delta

If the goal is to approach ToddlerBot's locomotion class more directly, the
highest-value mechanical change is:

- add ankle roll to each leg

After that, the next most meaningful changes are:

- more lower-body DoF for frontal-plane control
- more torso/waist authority
- preserving and using upper body DoFs for whole-body balance

If a future WildRobot v3 is possible, ankle roll is the clearest single change
that would move the morphology closer to ToddlerBot in locomotion relevance.

## MPC vs RL Under This Pivot

If the project follows ToddlerBot's direction, the practical answer is:

- do not make heavy runtime MPC the center of the stack
- do not keep betting on unstructured end-to-end PPO either
- use model-based methods as structure around learning

So the intended architecture becomes:

- reference generator or reduced-order planner
- optional IK or pose projection
- learned policy that tracks, adapts, and generalizes

This is the most consistent interpretation of both ToddlerBot's platform and
WildRobot's current hardware constraints.

## Recommendation

If the pivot is accepted, WildRobot should follow a "ToddlerBot-lite" path for
v2 hardware:

- keep current robot hardware for now
- change the project goal from standing recovery to locomotion platform
- keep learned policy as the deployed artifact
- demote `v0.18` planner + IK from final controller to training/reference
  infrastructure
- build the sim-to-real tooling and actuator realism loop first
- target a narrower locomotion benchmark that matches current hardware

If the goal is specifically to match ToddlerBot-like locomotion capability, then
the project should also plan a hardware revision. The most important candidate
change is ankle roll.

## Proposed Next Steps

1. Decide whether the goal is:
   - ToddlerBot-style architecture on current hardware, or
   - ToddlerBot-style locomotion capability, which likely implies hardware
     change
2. If the answer is "current hardware first," start a new locomotion roadmap:
   - platform tooling
   - actuator SysID
   - sim2real evaluation
   - reference-guided walking env
   - learned runtime policy
3. If the answer is "locomotion capability match," begin a WildRobot v3 concept
   discussion with ankle roll as the first hardware candidate.

## References

- ToddlerBot website: https://toddlerbot.github.io/
- ToddlerBot paper: https://arxiv.org/abs/2502.00893
- ToddlerBot follow-up: https://arxiv.org/abs/2601.03607
- WildRobot system architecture: `/home/leeygang/projects/wildrobot/docs/system_architecture.md`
- WildRobot standing plan: `/home/leeygang/projects/wildrobot/training/docs/standing_training.md`
- WildRobot v2 model config:
  `/home/leeygang/projects/wildrobot/assets/v2/mujoco_robot_config.json`
- WildRobot v2 runtime config:
  `/home/leeygang/projects/wildrobot/runtime/configs/runtime_config_v2.json`
