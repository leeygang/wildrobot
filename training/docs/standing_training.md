# v0.17: Standing Stabilization with Reactive Stepping

**Version:** v0.17.0
**Status:** Planned
**Created:** 2026-03-20

---

## Executive Summary

The original training goal was a robot that **stands stably and uses corrective footsteps to recover from pushes**, returning to an upright neutral posture. The v0.15.x branch evolved this into a walking problem, spending 12 iterations (v0.15.0 through v0.15.12) trying to achieve forward locomotion. Every walking attempt converged to the same failure: pitch-driven collapse. Dense velocity reward is too easy to exploit via leaning, while contact-driven step rewards remain too sparse to dominate credit assignment.

**v0.17 returns to the original goal: standing push recovery with reactive stepping.**

This is not a walking problem. It is a balance problem. The reward landscape, observation space, curriculum, and success criteria are fundamentally different.

---

## Research & Analysis

### Why Walking Training Failed for Standing Stabilization

The v0.15.x history reveals a clear pattern:

| Version | Approach | Outcome |
|---------|----------|---------|
| v0.15.3-4 | Direct velocity reward | Posture exploit (lean for speed) |
| v0.15.5 | Penalty strengthening | Trapped between stand-still and lean-fall basins |
| v0.15.6-7 | Contact-gated velocity, stepping rewards | Step gate opens but no propulsion |
| v0.15.8 | Gait clock (`wr_obs_v3`) | Stepping without propulsion |
| v0.15.9 | Per-step propulsion terms | Low-propulsion basin (stepping in place) |
| v0.15.10 | Zero-baseline forward reward | Propulsion unlocked, pitch collapse |
| v0.15.11 | Posture-gated dense progress | Delayed collapse, same terminal basin |
| v0.15.12 | Contact-driven step-to-step | Structural shift, outcome pending |

**Root cause**: Walking rewards create a velocity-tracking basin that competes with stability. The policy can always earn more reward by leaning forward than by stepping carefully. This is a fundamental tension in the reward landscape that no amount of gating, weighting, or curriculum has resolved.

**Key insight**: Standing push recovery has no velocity-tracking reward. The dominant reward is "stay upright and still." There is no competing velocity basin. Stepping emerges because it is the only strategy that prevents termination under strong pushes.

### Industry Best Practices for Reactive Stepping

#### The Golden Rule: Standing + Stepping ≠ Walking

| Property | Standing + Reactive Stepping | Walking |
|----------|------------------------------|---------|
| Default behavior | Stand still, zero velocity | Move at commanded velocity |
| When to step | Only when balance is threatened | Continuously |
| Success metric | Return to quiet standing | Track velocity command |
| Step purpose | Widen support polygon | Propulsion |
| Reward dominant term | Posture/upright maintenance | Velocity tracking |
| Clock/phase signal | Not needed | Essential |

#### Capture Point Theory (Classical Foundation)

The theoretical backbone of all successful bipedal balance controllers (Boston Dynamics, IHMC, Agility):

```
Capture Point = x_CoM + v_CoM / omega_0
where omega_0 = sqrt(g / z_CoM)
```

- If the capture point is inside the current support polygon: no step needed (ankle/hip strategy suffices)
- If outside: must step, and the step target ≈ capture point + margin
- For WildRobot at nominal height ~0.30m: `omega_0 ≈ 5.7 rad/s`, capture point offset ≈ `v_CoM * 0.175s`

The existing `step_trait_base_controller_design.md` M2.5 design already recommends capture point as an observation. This is sound.

#### The Three Balance Strategies (Progressive Difficulty)

Human and bipedal robot balance research identifies three progressive recovery strategies:

1. **Ankle strategy**: Small perturbations. Torque at ankles shifts CoP within existing support polygon. Works when capture point stays inside support polygon. WildRobot's bracing-only baseline already uses this (v0.13.9: 58% success at 9N x 10 steps).

2. **Hip strategy**: Medium perturbations. Rapid hip/waist angular acceleration creates reaction torques. Uses angular momentum to shift CoP. WildRobot can do this via the waist actuator.

3. **Stepping strategy**: Large perturbations. Move the support polygon to recapture the capture point. The only strategy that works when the capture point exits the current base of support. This is what v0.17 must learn.

These strategies emerge **naturally in order of difficulty** when push magnitude increases. No explicit mode-switching is needed — the reward landscape and push curriculum drive the progression.

#### Modern RL Approach: Push Curriculum with Zero Velocity Command

The most effective approach from recent locomotion research (ETH Zurich, Berkeley, Agility Robotics):

1. **Train with `v_cmd = 0` always** — standing, not walking
2. **Progressive push curriculum** — starts where ankle strategy works, ramps to where only stepping survives
3. **No gait clock or phase signal** — stepping is aperiodic and event-driven
4. **Capture point as observation** — tells the policy *where* balance is threatened without constraining *how* to recover
5. **Termination pressure drives stepping** — the policy steps because falling terminates the episode, not because stepping is explicitly rewarded

#### Key References

##### Classical Foundations

1. **Raibert (1986)**, *Legged Robots That Balance*, MIT Press.
   - Original foot placement heuristic: `x_step = x_com + k * v_com + k_p * (v - v_des)`.
   - Foundation of all modern bipedal stepping controllers including capture point theory.
   - **Relevance to v0.17:** The Raibert heuristic is the basis for the capture-point observation we add to the actor. It tells the policy *where* to step without constraining *how*.

2. **Pratt et al. (2006)**, *Capture Point: A Step toward Humanoid Push Recovery*, IEEE-RAS Humanoids.
   - Defines the capture point as `x_cp = x_com + v_com / omega_0` where `omega_0 = sqrt(g/z_com)`.
   - Proves that if you can place your foot at the capture point, you can stop in one step.
   - Extended by Pratt & Tedrake (2012) to multi-step capture regions.
   - **Relevance to v0.17:** Capture point error is our primary balance-threat signal in the observation space. The WildRobot capture point analysis (0.26m offset at 9N push) confirms stepping is geometrically necessary.

3. **Horak & Nashner (1986)**, *Central Programming of Postural Movements: Adaptation to Altered Support-Surface Configurations*, Journal of Neurophysiology.
   - Original characterization of the three human balance strategies: ankle, hip, and stepping.
   - Shows that humans naturally progress through these strategies as perturbation magnitude increases.
   - **Relevance to v0.17:** Our push curriculum is designed to replicate this progression. Phase 2 (3-7N) targets ankle/hip strategy, Phase 3-4 (9-14N) forces stepping emergence.

##### Modern RL Locomotion (Directly Applicable)

4. **Rudin et al. (2022)**, *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, CoRL 2022.
   - Link: https://arxiv.org/abs/2109.11978
   - Massively parallel PPO training (4096+ envs) on a single GPU.
   - Introduces a "game-inspired curriculum" for locomotion: progressive difficulty ramps for terrain, velocity commands, and disturbances.
   - Trains ANYmal quadruped to walk on flat terrain in <4 minutes, uneven terrain in <20 minutes.
   - Open-source training code.
   - **Relevance to v0.17:** Our massively parallel PPO setup (4096 envs) and progressive push curriculum directly follow this paradigm. The key insight is that curriculum difficulty — not reward redesign — is the primary training lever.

5. **Li, Peng, Abbeel, Levine, Berseth, Sreenath (2024)**, *Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control*, International Journal of Robotics Research.
   - Link: https://arxiv.org/abs/2401.16889
   - Demonstrates standing, walking, running, and jumping on the Cassie bipedal robot using a single RL framework.
   - **Novel dual-history architecture**: uses both long-term and short-term I/O history for handling dynamics changes and contact events.
   - **Task randomization** as a key source of robustness — forces generalization and compliance to disturbances.
   - Successfully deploys on real Cassie robot including push recovery, standing balance, and terrain traversal.
   - **Relevance to v0.17:** This paper demonstrates the exact paradigm we follow — a single RL policy that handles standing, push recovery, and locomotion through task/disturbance randomization rather than explicit mode switching or stepping rewards. The dual-history architecture (short-term for contact events, long-term for dynamics) is a potential observation enhancement if our simpler approach stalls.

6. **Haarnoja et al. (2024)**, *Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning*, Science Robotics.
   - Link: https://arxiv.org/abs/2304.13653
   - 28 authors, DeepMind. Trains a 20-joint miniature humanoid for soccer with zero-shot sim-to-real.
   - Key sim-to-real ingredients: **high-frequency control**, **targeted dynamics randomization**, and **perturbation training** (push disturbances during training).
   - Push recovery emerges naturally from perturbation training — fall recovery is 63% faster than scripted baselines.
   - Behavioral regularization ensures safe, smooth movements.
   - **Relevance to v0.17:** Validates that push disturbance training during RL produces robust balance recovery without explicit stepping/balance rewards. Their perturbation training is analogous to our push curriculum. Their behavioral regularization maps to our action_rate and energy penalties.

7. **Radosavovic, Zhang, Shi, Rajasegaran, Kamat, Darrell, Sreenath, Malik (2024)**, *Humanoid Locomotion as Next Token Prediction*, arXiv 2024.
   - Link: https://arxiv.org/abs/2402.19469
   - Frames humanoid control as autoregressive sequence prediction using a causal transformer.
   - Trained on diverse data: simulated trajectories, motion capture, YouTube videos.
   - Achieves zero-shot generalization to unseen commands (including backward walking) with only 27 hours of walking data.
   - Deployed on real humanoid robot walking through San Francisco.
   - **Relevance to v0.17:** Demonstrates that velocity-conditioned humanoid control (where v=0 produces standing with reactive balance) can transfer to real hardware. While we use PPO rather than sequence prediction, the principle of velocity-conditioned behavior — where standing and stepping are special cases of a general locomotion policy — motivates our v0.17.5 walking extension.

8. **Schumacher, Geijtenbeek, Caggiano, Kumar, Schmitt, Martius, Haeufle (2023)**, *Natural and Robust Walking using Reinforcement Learning without Demonstrations in High-Dimensional Musculoskeletal Models*, arXiv 2023.
   - Link: https://arxiv.org/abs/2309.02976
   - Achieves natural bipedal walking with RL **without any expert demonstrations or reference motions**.
   - Optimizes for stability, robustness, and energy efficiency simultaneously.
   - Key insight: RL without demonstrations can produce more robust locomotion than demonstration-dependent methods, because the policy is not constrained to replicate potentially brittle reference trajectories.
   - **Relevance to v0.17:** Directly validates our "no AMP, no reference motion" approach. Our v0.15.x failures with walking rewards don't mean RL can't work — they mean the reward structure was wrong (velocity tracking vs. standing). This paper shows that the right multi-objective reward (stability + efficiency + robustness) produces natural behavior without demonstrations.

##### Complementary Approaches (Context & Fallbacks)

9. **Margolis, Yang, Paigwar, Chen, Agrawal (2022)**, *Rapid Locomotion via Reinforcement Learning*, RSS 2022.
   - Link: https://arxiv.org/abs/2205.02824
   - Identifies adaptive curriculum on velocity commands as a key component of agile locomotion learning.
   - Progressive velocity curriculum: start simple, ramp difficulty.
   - **Relevance to v0.17:** Our push curriculum mirrors their velocity curriculum concept — progressive difficulty is the training lever, not reward weight tuning.

10. **Margolis and Agrawal (2023)**, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*, CoRL 2023.
    - Link: https://arxiv.org/abs/2212.03238
    - Argues against repeated ad hoc reward redesign (exactly what v0.15.x did) and toward structured behavior spaces / commandable locomotion families.
    - Single policy with commandable parameters (velocity, gait frequency, stance width, etc.).
    - **Relevance to v0.17:** Directly diagnoses why v0.15.x failed — we were doing exactly the ad hoc reward redesign this paper warns against. v0.17 follows their advice: one clean objective (stand upright) with a structured curriculum (push difficulty), not iterative reward weight tuning.

11. **Li, Junheng, Ma, Kolt, Shah, Nguyen (2023)**, *Dynamic Loco-manipulation on HECTOR*, arXiv 2023.
    - Link: https://arxiv.org/abs/2312.11868
    - MPC-based (not RL) humanoid control with explicit push recovery and disturbance rejection.
    - Demonstrates balance recovery during perturbations, 3D walking on varied terrains, and object manipulation.
    - Uses Force-and-Moment MPC with simplified rigid body dynamics.
    - **Relevance to v0.17:** Provides a complementary non-RL perspective. If our pure RL approach fails to discover stepping, the HECTOR-style MPC approach maps to our FSM base controller fallback (M3 design in `step_trait_base_controller_design.md`).

12. **Ren et al. (2025)**, *VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception*, arXiv 2025.
    - Link: https://arxiv.org/abs/2502.14814
    - Composite framework combining vision-based and proprioception-only (blind) policies for humanoid locomotion.
    - Proprioception-only (blind) policies are "highly robust" — exactly the modality v0.17 uses (no vision, proprioception + capture point only).
    - Demonstrates that blind policies can handle challenging perturbations, trading locomotion speed for robustness.
    - **Relevance to v0.17:** Validates our proprioception-only observation space. For standing push recovery, vision is unnecessary — proprioceptive signals (gravity, gyro, joint state) plus capture point are sufficient. VB-Com shows this is not a limitation but a robustness advantage.

##### End-to-End Recipe References (Primary Blueprints for v0.17+)

The following two references together cover the **complete training recipe** from standing through push recovery to walking. They are the primary blueprints v0.17 follows.

**Blueprint 1: legged_gym / Rudin et al. — The Open-Source Training Recipe**

- **Repository:** https://github.com/leggedrobotics/legged_gym (now migrated to Isaac Lab)
- **Paper:** Rudin et al. (2022), *Learning to Walk in Minutes* (Ref #4 above)
- **Why this is the primary blueprint:** This is the only fully open-source, production-proven training recipe for legged locomotion with RL. It provides the exact reward terms, domain randomization parameters, push disturbance protocol, and curriculum that v0.17 adapts.

The legged_gym recipe that v0.17 directly maps to:

| legged_gym Component | v0.17 Mapping |
|----------------------|---------------|
| `_reward_tracking_lin_vel`: exp(-error/sigma) velocity tracking | `zero_velocity`: exp(-k * v²) — same form, but target is v=0 |
| `_reward_orientation`: penalize projected gravity xy² | `upright`: exp(-k * (pitch² + roll²)) — equivalent |
| `_reward_base_height`: MSE from target height | `height`: exp(-k * (z - z_nom)²) — equivalent |
| `_reward_ang_vel_xy`: penalize pitch/roll angular velocity | `angular_velocity`: -k * \|\|omega\|\|² — equivalent |
| `_reward_torques`: sum of squared torques | `energy`: -k * sum(\|tau * dq\|) — similar, power-based |
| `_reward_action_rate`: change between consecutive actions | `action_rate`: -k * \|\|a_t - a_{t-1}\|\|² — identical |
| `_reward_stand_still`: motion penalty at zero command | `zero_velocity` + `posture` — same intent |
| `_reward_dof_pos_limits`: soft joint limit penalty | (handled by CAL clipping) |
| `_reward_feet_contact_forces`: excessive GRF penalty | `slip`: foot velocity penalty — related |
| Push disturbance: random velocity impulses every `push_interval` steps | Push curriculum: phased force ramp 0→18N |
| Friction randomization: `friction_range` across 64 buckets | Domain randomization in v0.17.4 |
| Base mass randomization: `added_mass_range` | Domain randomization in v0.17.4 |
| Observation noise: per-channel scaled uniform noise | Planned for v0.17.4 |
| Terrain curriculum: progressive difficulty by distance traveled | Push curriculum: progressive force by training iteration |
| Command curriculum: velocity range grows when tracking > 80% | Push force range grows across phases |
| P-control action: `tau = Kp*(action + default - q) - Kd*dq` | CAL position targets with PD control — equivalent |

Key differences from legged_gym:
- v0.17 uses **push force curriculum** instead of terrain curriculum (standing, not walking)
- v0.17 adds **capture point error** to observation (balance-specific signal)
- v0.17 uses **zero velocity target** (standing) instead of velocity tracking (walking)
- v0.17 removes gait-related rewards (`feet_air_time`, `tracking_ang_vel`) — not needed for standing

**Blueprint 2: Li et al. (2024) — Standing + Stepping + Walking on Real Bipedal Hardware**

- **Paper:** Li, Peng, Abbeel, Levine, Berseth, Sreenath (2024), *Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control*, IJRR (Ref #5 above)
- **Link:** https://arxiv.org/abs/2401.16889
- **Why this is the primary real-hardware blueprint:** This is the most comprehensive demonstration of a single RL policy achieving standing, walking, running, jumping, and push recovery on a real bipedal robot (Cassie). It covers the full v0.17 → v0.17.5 trajectory.

The Li et al. recipe that maps to the v0.17+ roadmap:

| Li et al. Component | v0.17+ Mapping |
|---------------------|----------------|
| Single unified RL policy for all locomotion modes | v0.17 (standing) → v0.17.5 (walking) — same policy, extended |
| Dual-history architecture (short-term for contacts, long-term for dynamics) | v0.17: simple history via `previous_action`; potential upgrade if needed |
| Task randomization for robustness | v0.17.4: domain randomization (friction, mass, motor strength) |
| Standing demonstrated as a skill within the unified policy | v0.17.0-v0.17.4: standing is the primary/only skill |
| Push recovery emerges from task randomization, not explicit rewards | v0.17 core principle: push curriculum drives stepping emergence |
| Walking and running as velocity-conditioned extensions | v0.17.5: add velocity command and tracking reward |
| Sim-to-real on real Cassie hardware | v0.17.4+: hardware validation planned |

Key insight from Li et al. that validates v0.17:
> "Task randomization [is] another key source of robustness, fostering better task generalization and compliance to disturbances."

This directly supports our approach: **push curriculum (a form of task randomization) produces stepping compliance without explicit stepping rewards.**

The dual-history architecture is a potential upgrade path if v0.17's simpler observation space proves insufficient. Short-term I/O history captures contact events (when the foot hits the ground during stepping), while long-term history captures dynamics shifts (changed friction, added mass).

##### Reference Summary Table

| Ref # | Paper | Year | Key Technique Used in v0.17 |
|--------|-------|------|----------------------------|
| 1 | Raibert | 1986 | Foot placement heuristic → capture point obs |
| 2 | Pratt et al. | 2006 | Capture point theory → observation space |
| 3 | Horak & Nashner | 1986 | Ankle→hip→step progression → push curriculum |
| 4 | Rudin et al. | 2022 | Massively parallel PPO + curriculum |
| 5 | Li et al. | 2024 | Single RL policy for standing+stepping+walking |
| 6 | Haarnoja et al. | 2024 | Perturbation training → push recovery emerges |
| 7 | Radosavovic et al. | 2024 | Velocity-conditioned policy, v=0 → standing |
| 8 | Schumacher et al. | 2023 | RL without demonstrations, multi-objective |
| 9 | Margolis et al. | 2022 | Adaptive command curriculum |
| 10 | Margolis & Agrawal | 2023 | Against ad hoc reward redesign |
| 11 | Li et al. (HECTOR) | 2023 | MPC-based push recovery (fallback reference) |
| 12 | Ren et al. (VB-Com) | 2025 | Proprioception-only robustness |

#### Why Pure RL + Push Curriculum Works Here

The v0.15.x walking attempts failed because the reward landscape had a parasitic attractor (lean-for-speed). Standing push recovery does not have this problem:

1. **No velocity reward to exploit** — `v_cmd = 0`, the only way to earn reward is staying still and upright
2. **Push curriculum creates stepping pressure automatically** — at low forces, bracing works. Above ~9N, bracing fails (established in v0.13.9). The optimizer has a clear gradient toward stepping.
3. **Stepping is inherently aperiodic** — no clock needed. Step when pushed, stand when not. This eliminates the gait-clock alignment problems from v0.15.8+.
4. **The success criterion is unambiguous** — survive the push and return to quiet standing. No velocity tracking, no gait periodicity, no step length targets.

### WildRobot-Specific Analysis

#### Bracing Ceiling (Established v0.13.9)

The force sweep from the standing-push baseline established:

| Force (N) | Duration (steps) | Success Rate | Strategy |
|-----------|-------------------|--------------|----------|
| 7 | 10 | ~85% | Ankle/hip bracing |
| 8 | 10 | ~75% | Bracing at limit |
| **9** | **10** | **58%** | **Bracing ceiling** |
| 10 | 10 | ~47% | Bracing fails reliably |
| 9 | 15 | <50% | Longer impulse, bracing insufficient |

**The stepping regime begins at ~9N x 10 steps.** This is where the policy must discover stepping because bracing alone no longer reliably prevents termination.

#### Robot Constraints

- **9 actuated DOFs**: 4 per leg (hip_pitch, hip_roll, knee, ankle) + waist
- **No arms**: WildRobot has no arm actuators, so arm momentum strategies (M3 design) do not apply. All recovery must come from legs + waist.
- **4.41 Nm torque limit**: Stepping must be efficient. Large, fast steps that saturate actuators will fail on hardware.
- **50 Hz control**: Reactive stepping at 50 Hz is feasible but timing-sensitive. The policy has ~0.5s to complete a recovery step at the stepping regime force level.
- **Passive foot_roll joints**: These provide some passive compliance during stepping but cannot be actively controlled.

#### Capture Point for WildRobot

At nominal standing height z_CoM ≈ 0.30m:
- `omega_0 = sqrt(9.81 / 0.30) ≈ 5.72 rad/s`
- Capture point offset = `v_CoM / 5.72`
- At 9N push for 0.2s (10 steps @ 50Hz) on ~1.2kg robot: `delta_v ≈ 9*0.2/1.2 ≈ 1.5 m/s`
- Capture point offset ≈ `1.5 / 5.72 ≈ 0.26m` — well beyond the support polygon (~0.23m total width)
- This confirms stepping is geometrically necessary at 9N+ for this robot.

---

## v0.17 Training Plan

### Approach: Pure RL with Push Curriculum and Capture Point Observation

Based on the analysis above, v0.17 uses:

1. **Pure RL (end-to-end PPO)** — no FSM base controller, no residual architecture
2. **Zero velocity command** — standing only, no walking objective
3. **Progressive push curriculum** — the key training lever
4. **Capture point error as observation** — gives the policy balance-threat information
5. **Termination-driven stepping pressure** — stepping emerges because it prevents falling, not because it is explicitly rewarded

This is the simplest possible approach that addresses the fundamental problem. No gait clocks, no dense progress rewards, no propulsion gates, no velocity tracking.

### Observation Space: `wr_obs_standing`

```
Observation                    Dim    Source
─────────────────────────────────────────────
gravity_vector                  3     IMU (sim: projected gravity)
angular_velocity                3     IMU gyro
joint_positions                 9     Encoders (normalized to [-1, 1])
joint_velocities                9     Encoders (normalized)
previous_action                 9     Policy output t-1
capture_point_error             2     x_cp - x_com, y_cp - y_com (heading-local)
─────────────────────────────────────────────
Total                          35
```

**Key differences from v0.15.x (`wr_obs_v3`, 46-dim):**
- **Removed**: `velocity_cmd` (2), `gait_clock` (2), foot contacts (2), `clock_progress` (1), `foot_height_left/right` (2)
- **Added**: `capture_point_error` (2)
- **No clock signal** — stepping is aperiodic
- **No velocity command** — always zero, not worth observing
- **Capture point error** — the single most informative signal for "do I need to step, and where?"

**Contract**: This is a new `policy_spec_hash`. Not resume-safe from any v0.15.x checkpoint. Fresh training required.

### Action Space

Unchanged from current: 9-dim normalized `[-1, 1]`, mapped to joint position targets via CAL.

### Reward Function

```yaml
# PRIMARY — what the robot should care about most
rewards:
  upright:          4.0    # exp(-k * (pitch² + roll²)), k=20
  height:           2.0    # exp(-k * (z - z_nominal)²), k=100
  zero_velocity:    2.0    # exp(-k * (v_x² + v_y²)), k=10
  posture:          1.5    # exp(-k * ||q - q_default||²), k=2.0

# QUALITY — secondary objectives
  angular_velocity: -0.5   # -k * ||omega||² (all 3 axes, not just yaw)
  action_rate:      -0.3   # -k * ||a_t - a_{t-1}||²
  energy:           -0.2   # -k * sum(|tau * dq|)
  slip:             -1.0   # -k * ||v_foot_xy||² when foot in contact

# RECOVERY QUALITY — matters during/after pushes (optional, milestone-gated)
  recovery_posture: 0.5    # bonus for returning to default posture after perturbation
```

**Key design decisions:**
- **No velocity tracking reward** — `v_cmd = 0`, reward is for staying still
- **No stepping event reward** — stepping is not explicitly rewarded; it emerges from termination avoidance
- **No foot placement reward** — no capture-point placement shaping (that is in the observation, not the reward). The policy learns *where* to step from survival pressure.
- **No gait periodicity** — standing has no periodic gait
- **No forward/dense_progress** — the v0.15.x poison pill is completely absent
- **Strong posture return** — after a push, the policy should return to neutral, not settle in a crouch

**Why no explicit stepping rewards?**

The v0.15.x experience showed that explicit stepping rewards (step_event, foot_place, step_length, step_progress) create complex reward landscapes with many local optima (marching in place, fidgeting, etc.). The push curriculum naturally creates stepping pressure without these complications:
- Below bracing ceiling: bracing earns full survival reward. No stepping needed.
- Above bracing ceiling: bracing leads to termination. Stepping is the only path to continued reward. The gradient is clear and unambiguous.

### Termination Conditions

```yaml
termination:
  height_low:        0.15   # m — below this, consider fallen
  pitch_limit:       0.8    # rad (~46°) — forward/backward fall
  roll_limit:        0.6    # rad (~34°) — lateral fall
  max_episode_steps: 500    # 10s at 50 Hz
```

Termination is the **primary learning signal for stepping**. The policy learns to step because not stepping leads to termination (zero future reward).

### Push Curriculum (The Key Training Lever)

```yaml
push_curriculum:
  enabled: true

  # Phase 1: No pushes (learn quiet standing)
  phase_1:
    start_iter: 0
    end_iter: 30
    push_probability: 0.0    # no pushes

  # Phase 2: Easy pushes (ankle/hip strategy regime)
  phase_2:
    start_iter: 30
    end_iter: 80
    push_probability: 0.5
    force_min: 3.0           # N
    force_max: 7.0           # N
    duration_min: 5           # control steps (0.1s)
    duration_max: 10          # control steps (0.2s)
    directions: omnidirectional
    push_interval_min: 50     # steps between pushes

  # Phase 3: Medium pushes (transition to stepping)
  phase_3:
    start_iter: 80
    end_iter: 150
    push_probability: 0.7
    force_min: 5.0
    force_max: 10.0
    duration_min: 8
    duration_max: 12
    directions: omnidirectional
    push_interval_min: 40

  # Phase 4: Hard pushes (stepping regime)
  phase_4:
    start_iter: 150
    end_iter: 300
    push_probability: 0.8
    force_min: 7.0
    force_max: 14.0
    duration_min: 8
    duration_max: 15
    directions: omnidirectional
    push_interval_min: 30

  # Phase 5: Stress test (multi-push, extreme)
  phase_5:
    start_iter: 300
    end_iter: 500
    push_probability: 0.9
    force_min: 8.0
    force_max: 18.0
    duration_min: 10
    duration_max: 20
    directions: omnidirectional
    push_interval_min: 25
    multi_push: true          # multiple pushes per episode
```

**Design rationale:**
- Phase 1 (0-30 iters): Learn quiet standing. Build the posture baseline.
- Phase 2 (30-80 iters): Easy pushes. Ankle/hip strategy develops naturally.
- Phase 3 (80-150 iters): Cross the bracing ceiling (~9N). Stepping pressure begins.
- Phase 4 (150-300 iters): Hard pushes. Stepping must be reliable.
- Phase 5 (300-500 iters): Stress test with extreme forces and multi-push scenarios.

### Training Configuration

```yaml
version: "0.17.0"
version_name: "Standing Push Recovery with Reactive Stepping"

ppo:
  iterations: 500
  num_envs: 4096
  episode_length: 500        # 10s at 50 Hz
  learning_rate: 3e-4
  num_minibatches: 32
  update_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  entropy_cost: 0.01
  normalize_observations: true

env:
  velocity_cmd: 0.0           # ALWAYS zero
  actor_obs_layout_id: wr_obs_standing
  dt: 0.02                    # 50 Hz control
  action_filter_alpha: 0.3    # smooth actions

  # Push config (see curriculum above)
  push_enabled: true
  push_curriculum: true
```

### Evaluation Protocol

#### Eval Stages (Run at Every Checkpoint)

| Eval Name | Push Force | Duration | Purpose |
|-----------|-----------|----------|---------|
| `eval_quiet` | 0 N | — | Quiet standing baseline |
| `eval_easy` | 5 N | 10 steps | Ankle strategy regime |
| `eval_medium` | 8 N | 10 steps | Transition regime |
| `eval_hard` | 10 N | 10 steps | Stepping regime |
| `eval_hard_long` | 9 N | 15 steps | Long impulse stepping |
| `eval_extreme` | 14 N | 12 steps | Stress test |

#### Key Metrics

```
# Survival
eval_*/success_rate          → target: > 0.95 for easy/medium, > 0.80 for hard
eval_*/episode_length        → target: near 500 (10s)

# Termination breakdown
eval_*/term_height_low_frac  → should decrease with training
eval_*/term_pitch_frac       → should remain low
eval_*/term_roll_frac        → should remain low

# Balance quality
env/z_height                 → stable at nominal
env/pitch_mean               → near zero
env/roll_mean                → near zero

# Recovery quality (new metrics for v0.17)
recovery/time_to_upright     → time from push end to upright posture
recovery/max_displacement    → max CoM displacement during recovery
recovery/posture_at_end      → posture quality at episode end
recovery/num_steps_taken     → 0 for bracing, 1-3 for stepping

# Stepping behavior
debug/step_count_per_push    → increases with push difficulty
debug/foot_displacement      → foot movement during recovery
```

---

## Execution Milestones

### v0.17.0 — Infrastructure & Quiet Standing

**Objective:** Set up the new training pipeline and verify quiet standing.

Tasks:
- [ ] Create `wr_obs_standing` observation layout (35-dim, with capture point error)
- [ ] Implement capture point computation in `wildrobot_env.py`
- [ ] Create `ppo_standing.yaml` config (v0.17 reward function, no pushes initially)
- [ ] Remove all walking-specific reward terms (forward, dense_progress, step_length, cycle_progress, gait_periodicity, step_event, foot_place, clearance)
- [ ] Remove gait clock from observation
- [ ] Implement push curriculum infrastructure (phase-based force ramping)
- [ ] Create eval stages (quiet, easy, medium, hard, extreme)
- [ ] Implement recovery quality metrics (time_to_upright, max_displacement, step_count)
- [ ] Smoke test: 10 iterations, no pushes, robot stands quietly

**Exit criteria:**
- Robot stands for 500 steps (10s) with 100% success, no pushes
- `upright` and `height` rewards are near maximum
- `zero_velocity` reward is near maximum
- `posture` reward is stable
- All new metrics logging correctly

### v0.17.1 — Ankle/Hip Strategy (Easy Pushes)

**Objective:** Robot survives easy pushes using bracing (no stepping expected yet).

Tasks:
- [ ] Enable push curriculum Phase 2 (3-7N, 5-10 step duration)
- [ ] Train for 80 iterations
- [ ] Verify bracing behavior matches v0.13.9 baseline quality
- [ ] Tune reward weights if needed

**Readout schedule:**
- Iter 20: Robot should stand quietly with occasional bracing
- Iter 40: `eval_easy` success > 90%
- Iter 60: `eval_medium` success > 70%
- Iter 80: Ankle/hip strategy should be visible in visualization

**Exit criteria:**
- `eval_quiet`: success > 99%
- `eval_easy` (5N): success > 95%
- `eval_medium` (8N): success > 70%
- Robot returns to upright posture after easy pushes
- No stepping observed (bracing only)

### v0.17.2 — Stepping Emergence (Medium/Hard Pushes)

**Objective:** Stepping emerges as a recovery strategy for pushes above the bracing ceiling.

Tasks:
- [ ] Enable push curriculum Phase 3 (5-10N)
- [ ] Train for 150 iterations total (continue from v0.17.1 checkpoint)
- [ ] Monitor `debug/step_count_per_push` — should start rising at iter ~80-100
- [ ] Monitor `debug/foot_displacement` — should show meaningful foot movement
- [ ] Visualize recovery episodes — look for first stepping behaviors

**Readout schedule:**
- Iter 100: First signs of stepping should appear under 9N+ pushes
- Iter 120: `eval_hard` (10N) success should be improving over bracing-only baseline
- Iter 150: Stepping should be the dominant strategy for 10N+ pushes

**Decision rules:**
- If by iter 120, `debug/step_count_per_push` is still zero under hard pushes and `eval_hard` success is at or below the v0.13.9 bracing ceiling (~47%): add capture point error to **critic** observation (asymmetric actor-critic) and re-run from v0.17.1 checkpoint. This gives privileged balance information without changing the actor contract.
- If stepping appears but the robot falls after stepping (poor placement): continue training — placement quality should improve with more experience.
- If the robot steps but doesn't return upright (stuck crouch): increase `posture` reward weight and add `recovery_posture` bonus.

**Exit criteria:**
- `eval_hard` (10N): success > 60% (beating bracing ceiling)
- `eval_hard_long` (9N x 15): success > 50%
- `debug/step_count_per_push` > 0.5 for hard pushes (stepping happens most of the time)
- Robot returns to neutral posture within 2s after push ends
- Stepping is visible in visualization for hard pushes

### v0.17.3 — Stepping Reliability (Hard Pushes)

**Objective:** Reliable stepping recovery for all push directions and magnitudes.

Tasks:
- [ ] Enable push curriculum Phase 4 (7-14N)
- [ ] Train for 300 iterations total
- [ ] Test omnidirectional push recovery (lateral, fore-aft, diagonal)
- [ ] Tune curriculum pacing if needed
- [ ] Verify stepping in both directions (left foot / right foot)

**Decision rules:**
- If stepping works for fore-aft pushes but not lateral: check if hip_roll authority is sufficient. May need to increase hip_roll action scale.
- If stepping works for one direction only: check for asymmetry in reward or observation. May indicate a sign convention issue.
- If stepping quality plateaus: try adding `recovery_posture` reward to incentivize post-step upright return.

**Exit criteria:**
- `eval_hard` (10N): success > 85%
- `eval_hard_long` (9N x 15): success > 75%
- `eval_extreme` (14N): success > 50%
- Omnidirectional stepping (lateral, fore-aft, diagonal)
- Both-foot stepping (left and right)
- Recovery time < 2s for most pushes
- Torque usage < 80% of limits during stepping

### v0.17.4 — Stress Test & Robustness

**Objective:** Handle extreme pushes and multi-push sequences.

Tasks:
- [ ] Enable push curriculum Phase 5 (8-18N, multi-push)
- [ ] Train for 500 iterations total
- [ ] Run force sweep evaluation (3N to 20N in 1N increments)
- [ ] Test multi-push recovery (2-3 pushes per episode)
- [ ] Domain randomization: friction, mass, motor strength
- [ ] Final checkpoint selection

**Exit criteria:**
- `eval_hard` (10N): success > 90%
- `eval_extreme` (14N): success > 65%
- Multi-push recovery: > 70% survival for 2 pushes per episode
- Force sweep shows graceful degradation (not cliff-edge failure)
- Domain randomization doesn't collapse performance by more than 15%
- Policy is checkpoint-stable (performance doesn't regress)

### v0.17.5 — Walking Extension (Future, After Standing is Solved)

**Objective:** Add velocity-conditioned walking on top of the standing/stepping foundation.

This milestone is **deferred** until v0.17.4 is complete and validated.

Tasks:
- [ ] Add `velocity_cmd` back to observation (`v_cmd ∈ [0, 0.3]`)
- [ ] Add velocity tracking reward (gated by `v_cmd > 0`)
- [ ] Add gait clock to observation (only for `v_cmd > 0`)
- [ ] Resume from v0.17.4 checkpoint with expanded observation
- [ ] Curriculum: ramp velocity from 0 to 0.3 m/s

**Rationale:** A robot that already knows how to step for balance should transition to walking much more easily than one trained from scratch. The stepping reflex transfers directly to walking — walking is just periodic stepping with a velocity objective.

---

## Key Design Decisions & Rationale

### 1. No FSM Base Controller (Pure RL)

The v0.14.x FSM experiments showed:
- FSM with reduced residual authority degraded performance (98% → 57% success)
- The FSM was not accurate enough to compensate for reduced RL freedom
- Pure RL at the bracing ceiling already has a clear gradient toward stepping

**Decision:** Start with pure RL. Add FSM only if pure RL fails to discover stepping by v0.17.2 (see decision rules).

### 2. No Explicit Stepping Rewards

The v0.15.x experience showed that explicit stepping rewards (`step_event`, `foot_place`, `step_length`) create complex reward landscapes with many local optima:
- Marching in place
- Fidgeting / oscillating
- Stepping without propulsion purpose

**Decision:** Let stepping emerge from termination pressure. The push curriculum naturally creates the gradient. Survival is the reward for stepping; falling is the penalty for not stepping.

### 3. Capture Point as Observation, Not Controller

**Decision:** Add capture point error `[x_cp - x_com, y_cp - y_com]` (2-dim) to the actor observation. This gives the policy information about *where* balance is threatened without constraining *how* to recover.

Why observation, not controller:
- The policy can learn non-obvious recovery strategies
- No sim-runtime parity issues from IK or trajectory generation
- The capture point may not be the optimal step target (it's the minimum — the policy may learn to overshoot for robustness)

### 4. No Gait Clock

**Decision:** Remove gait clock entirely. Standing push recovery is aperiodic. Steps happen in response to disturbances, not on a fixed schedule.

The v0.15.8+ gait clock was designed for walking. For standing, it would create unnecessary bias toward periodic stepping (marching in place).

### 5. Push Curriculum as Primary Lever (Not Reward Shaping)

**Decision:** The main training lever is push difficulty, not reward weight tuning.

This avoids the v0.15.x trap of endlessly retuning reward weights. The curriculum creates an automatic gradient:
- Easy regime → bracing works → policy learns bracing
- Hard regime → bracing fails → policy discovers stepping
- No reward redesign needed between phases

---

## Risk Analysis

### Risk 1: Policy Doesn't Discover Stepping (Braces Harder Instead)

**Likelihood:** Medium. The bracing ceiling at 9N is well-established, but the policy might find creative bracing strategies that extend the ceiling.

**Mitigation:**
1. If by v0.17.2 iter 120 stepping hasn't emerged: add capture point to critic (asymmetric actor-critic) for privileged balance information
2. If still no stepping: add lightweight stepping reward (`step_event` gated by `need_step > 0.5`) as a mild bias
3. If still no stepping: reintroduce the FSM base controller from M3 design as a guide (not controller)

### Risk 2: Policy Steps but Doesn't Return Upright (Stuck Crouch)

**Likelihood:** Medium-high. This was observed in v0.14.x experiments.

**Mitigation:**
- Strong `posture` reward (weight 1.5) drives return to neutral
- `recovery_posture` bonus can be added if needed
- `height` reward penalizes crouching
- Episode is 500 steps — the policy has 8+ seconds of reward pressure to return upright

### Risk 3: Stepping Creates Sim-Real Gap

**Likelihood:** Medium. Stepping involves dynamic contact transitions that are harder to simulate accurately than quiet standing.

**Mitigation:**
- Domain randomization on friction, mass, motor strength (v0.17.4)
- Conservative stepping (low step height, slow swing) is inherently more robust
- 50 Hz control is the same for training and hardware
- Validate on hardware at v0.17.3 if possible

### Risk 4: Torque Saturation During Stepping

**Likelihood:** Low-medium. Fast stepping requires burst torque that may exceed the 4.41 Nm limit.

**Mitigation:**
- `energy` penalty discourages high-torque actions
- Monitor `debug/torque_sat_frac` throughout training
- The push curriculum ramps gradually — the policy has time to learn efficient stepping
- Capture point observation helps the policy step early (before the disturbance grows large), reducing the required torque

---

## Relationship to Other Plans

| Document | Relationship |
|----------|-------------|
| `learn_first_plan.md` | v0.17 replaces the v0.15.x walking branch as the active execution plan. Walking (v0.17.5) comes after standing is solved. |
| `step_trait_base_controller_design.md` | v0.17 implements the M2.5 recommendation (capture point observation + pure RL at bracing ceiling). FSM (M3) is deferred unless M2.5 fails. |
| `CHANGELOG.md` | v0.17.0 entries should be added when implementation begins. |

---

## Quick Reference

```bash
# v0.17 Training
uv run python training/train.py --config training/configs/ppo_standing.yaml

# Smoke test (10 iterations, no pushes)
uv run python training/train.py --config training/configs/ppo_standing.yaml --verify

# Force sweep evaluation
uv run python training/eval/force_sweep.py \
    --checkpoint training/checkpoints/<v17_checkpoint>.pkl \
    --config training/configs/ppo_standing.yaml \
    --forces 3,5,7,9,10,12,14,18 \
    --push-duration 10 \
    --num-envs 128 \
    --num-steps 500
```
