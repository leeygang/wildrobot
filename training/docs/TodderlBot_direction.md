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

## Concrete Milestones

The pivot needs one clear rule up front:

- the deployed runtime artifact is a learned policy
- reduced-order planning, capture-point logic, and IK are support tools for
  reference generation, teacher signals, safety checks, and debugging

The first realistic product goal on current WildRobot v2 hardware is:

- zero-shot or near-zero-touch sim-to-real forward walking
- reliable stop and quiet standing
- in-place yaw turning
- moderate push recovery during standing and slow walking

### Milestone 0: Platform Pivot

Software stack:

- freeze the standing-only branch as a completed exploration path
- define a locomotion-first repo structure around `runtime`, `assets`,
  `references`, `training`, `policies`, and `tools`
- add a standard log format shared by sim and real runs
- define one canonical observation and action schema for locomotion

Training approach:

- stop treating free PPO standing recovery as the main path
- define the new training direction as reference-guided RL
- define the first command space as forward speed and yaw rate, not full
  omnidirectional motion

Sim2real:

- add a replayable standing test script
- add a reproducible calibration checklist for joints, IMU, and foot switches
- add one baseline real-vs-sim log capture for standing and squat motions

Exit gate:

- one documented train/eval/deploy flow exists for locomotion work
- one shared log schema is used in both simulation and hardware

### Milestone 1: Digital Twin and Actuator Realism

Software stack:

- add `HTD-45H` actuator parameter files for delay, backlash, frictionloss,
  armature, and torque/speed limits
- add `tools/sysid/` for chirp, step, and hold-response characterization
- add `tools/sim2real_eval/` for plotting real-vs-sim joint and IMU traces

Training approach:

- no walking training yet beyond smoke tests
- validate that the simulator reacts plausibly to the measured actuator model
- optionally train a very small standing policy with history as a realism check

Sim2real:

- measure joint step response on representative joints
- fit a first usable servo model
- model IMU bias/noise and foot-switch latency or dropout

Exit gate:

- sim-vs-real traces for scripted motions are close enough to support policy
  transfer work
- actuator and sensor realism parameters are under version control

### Milestone 2: Reference Generator v1

Software stack:

- implement `walking_ref_v1` as a reduced-order task-space reference generator
- use `LIPM/DCM/capture-point` for step timing and foot placement
- generate swing-foot trajectory, pelvis height target, and pelvis roll/pitch
  target
- add simple IK that converts the task-space reference into nominal joint
  targets

Training approach:

- train a residual policy around IK-generated nominal targets
- start with forward walking only
- include gait phase, next foothold, and pelvis targets in observation
- keep the first reference narrow and morphology-aware rather than copying a
  human or ToddlerBot gait directly

Sim2real:

- verify the reference generator produces kinematically feasible motions in sim
- replay the nominal reference slowly on hardware with safety constraints
- confirm joint limits, foot clearance, and touchdown geometry are sane

Exit gate:

- the reference generator alone produces stable nominal walking in simulation or
  at least a coherent stepping sequence
- the reference is safe enough to use as a prior for RL

### Milestone 3: Walking Policy v1

Software stack:

- add a locomotion env that consumes command velocity and the reference state
- add runtime support for history stacking and one-step action delay
- add policy export and replay tools for offline inspection

Training approach:

- train reference-guided PPO for forward walking and stop
- reward command tracking, pelvis tracking, foothold tracking, low slip, and
  survival
- keep joint-space imitation light; prioritize task-space tracking
- use residual actions around the nominal reference instead of unrestricted
  joint targets

Sim2real:

- add domain randomization v1 for friction, damping, mass, Kp/Kd, backlash,
  armature, torque limits, and IMU noise
- test short hardware rollouts with tether or guarded safety setup
- compare the policy's action statistics in sim and real

Exit gate:

- stable forward walking in simulation over a meaningful command range
- first hardware rollouts show recognizable stepping and controlled stopping

### Milestone 4: Transfer Hardening

Software stack:

- add automatic regression plots for sim-vs-real mismatch
- add richer observation history and delay randomization
- add failure-case tagging for slip, pitch fall, lateral fall, and missed
  touchdown

Training approach:

- broaden the command curriculum
- train on richer domain randomization and contact variation
- add start/stop transitions and in-place yaw
- tune the reference and residual balance rather than adding many ad hoc reward
  terms

Sim2real:

- iterate between real logs and simulator calibration
- require repeatable hardware evaluation across multiple floors or friction
  conditions
- quantify zero-shot success rate for walk, stop, and turn

Exit gate:

- zero-shot or near-zero-touch sim-to-real forward walking is reliable
- stop and turn behaviors are repeatable on hardware

### Milestone 5: Recovery and Robustness

Software stack:

- extend the reference generator to support disturbance-aware step adjustment
- optionally move from plain `LIPM/DCM` to an `ALIP`-inspired reference if
  angular-momentum effects matter strongly
- add runtime safety monitors and fall logging

Training approach:

- add moderate push recovery during standing and slow walking
- keep the same policy-centered runtime artifact
- treat planner logic as a better prior, not as a controller replacement

Sim2real:

- benchmark moderate push recovery
- benchmark recovery from command changes and stop transitions
- track failure modes by class rather than only aggregate success rate

Exit gate:

- WildRobot v2 can walk, stop, turn, and recover from moderate disturbances on
  current hardware

### Milestone 6: Platformization

Software stack:

- add reusable policy packaging and deployment tools
- add dataset recording and replay for future imitation or keyframe workflows
- add one command entry point for train, eval, replay, and deploy

Training approach:

- keep RL as the main locomotion training path
- make room for future data-driven skills on top of the same platform
- separate locomotion priors from robot IO and evaluation tooling

Sim2real:

- make calibration, evaluation, and deployment reproducible enough for repeated
  hardware iterations
- keep one standard benchmark suite for every new policy

Exit gate:

- WildRobot operates as a small robot-learning platform rather than a single
  experiment branch

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
