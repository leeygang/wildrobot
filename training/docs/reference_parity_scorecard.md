# Reference Parity Scorecard

Goal: make WildRobot reference quality demonstrably on par with ToddlerBot
before giving credit to PPO-side improvements.

## Scorecard

### P0. Shared Geometry Gate

Use the same reference-only geometry gate in both repos:

- stance foot bottom z must be `<= 0.003 m`
- swing foot bottom z must be `> -0.002 m`
- probe velocities: `vx in {0.10, 0.15, 0.20, 0.25}`
- probe frames: `{0, 5, 10, 16, 22, 28, 32, 48, 64}`
- in-scope pass gate: `vx in {0.10, 0.15}`
- higher `vx` bins stay diagnostic until both repos explicitly widen scope

Pass criteria:

- ToddlerBot must pass the in-scope gate on
  `tests/test_walk_reference_geometry.py`
- WildRobot must pass the in-scope gate on
  `tests/test_v0200c_geometry.py`

Parity target:

- On the full shared matrix, WildRobot worst stance-foot miss must be no more
  than ToddlerBot worst stance-foot miss `+ 0.002 m`
- On the full shared matrix, WildRobot worst swing-foot dip must be no worse
  than ToddlerBot worst swing-foot dip `- 0.002 m`

Interpretation:

- stance failures mean the reference is asking for contacts the model cannot
  realize cleanly
- swing failures mean the reference is not providing enough clearance

### P1. Open-Loop Trackability

Run a free-base nominal-only replay with no learned residual across
`vx in {0.10, 0.15, 0.20}`.

Required metrics:

- per-joint tracking RMSE over the 8 leg joints
- achieved forward velocity per `vx` bin
- survived episode steps per `vx` bin
- first termination mode
- contact-phase match against the physical foot-floor contact schedule
- physical touchdown step length mean/min per `vx` bin
- joint-limit saturation fraction per `vx` bin

Pass criteria:

- no NaN, no joint runaway, no repeated hard-limit clipping
- contact-phase match `>= 0.90` over the probe horizon

Parity target:

- WildRobot leg-joint RMSE `<= 1.10 x` ToddlerBot on each `vx` bin
- WildRobot achieved forward velocity `>=` ToddlerBot `- 0.02 m/s` on each `vx` bin
- WildRobot survived steps `>= 0.95 x` ToddlerBot on each `vx` bin
- WildRobot physical touchdown step length mean `>= 0.90 x` ToddlerBot on each `vx` bin
- WildRobot joint-limit saturation fraction `<=` ToddlerBot `+ 0.05` on each `vx` bin

Interpretation:

- failing here means the reference is not physically trackable enough, even if
  the pure FK geometry looks acceptable
- this is the layer that should predict whether PPO is failing because the
  prior is not trackable enough under real MuJoCo dynamics
- this layer measures **prior + each robot's own actuator stack**, not pure
  prior quality. WildRobot drives MuJoCo position actuators directly;
  ToddlerBot drives torque actuators with PD-on-torque. Comparable RMSE on
  both sides plus a P1 fail points to actuator / control mismatch rather than
  the prior; a P1 fail with materially worse WR RMSE points to the prior.
- termination thresholds (`pitch ≤ 0.8 rad`, `roll ≤ 0.6 rad`,
  `root_z ≥ 0.15 m`) are a shared physical-failure floor for both robots.
  ToddlerBot's PPO walk env terminates only on height (per the A.1 audit in
  `walking_training.md`), so this layer applies a stricter floor to TB than
  its training env uses; the intent is "same definition of physical failure",
  not "TB's training env defaults".

### P1A. FK-Realized Nominal Gait Shape

Check the gait shape at nominal `vx=0.15` from FK-realized foot trajectories,
not from planner-side intent.

Required metrics:

- touchdown step length mean/min
- touchdown rate (cadence)
- touchdown-speed proxy
  `step_length_mean * touchdown_rate / 2`
- swing-clearance mean/min
- double-support fraction
- foot-z max single-frame step

Parity target:

- WildRobot touchdown step length mean `>= 0.90 x` ToddlerBot
- WildRobot touchdown rate `<= 1.20 x` ToddlerBot
- WildRobot touchdown-speed proxy within `0.02 m/s` of ToddlerBot
- WildRobot swing-clearance mean `>= 0.85 x` ToddlerBot
- WildRobot double-support fraction within `0.05` absolute of ToddlerBot

Interpretation:

- this is a gait-shape check, not a trackability check
- the widened cadence cap is intentional: WR and TB use different fixed cycle
  times, so cadence should be treated as a bounded design difference rather
  than a hard equality target

### P2. Reference Smoothness

Check the reference asset itself, without physics.

Required metrics:

- max `|q_ref[t] - q_ref[t-1]|` over the shared 8 leg joints
- max pelvis height step
- max swing-foot height step
- contact-mask flip count per cycle

Pass criteria:

- no single-frame discontinuity that is visually or physically implausible
- no extra contact flips beyond the intended gait schedule

Parity target:

- WildRobot should be within `10%` of ToddlerBot on each smoothness metric

Interpretation:

- if WildRobot only matches ToddlerBot after heavy filtering, the reference is
  still weaker
- `pelvis_z_step_max` is a **sentinel** gate: both ZMP-style priors store a
  constant commanded pelvis height by design, so both sides legitimately
  report ~0 here and the gate trivially passes. A nonzero value would
  indicate a planner discontinuity bug (which is what the gate catches). Do
  not read a 0/0 PASS as "WR is smoother on pelvis height."

### P3. WildRobot Policy-Contract Gates

After P0-P2 are acceptable, WildRobot must still satisfy its own policy-side
contract from `training/docs/walking_training.md`.

Required gates:

- `G4`: promotion-horizon walking gate
- `G5`: anti-exploit residual bound
- `G6`: zero residual implies `target_q == q_ref`
- `G7`: trained policy beats bare-`q_ref` replay

Pass criteria:

- pass `G4`
- do not violate `G5`
- pass `G6`
- pass `G7`

Interpretation:

- `G4/G6/G7` confirm the reference works in the actual WR training stack
- `G5` confirms PPO is correcting the prior rather than replacing it

## Decision Rule

WildRobot is "on par with ToddlerBot" only when all of the following are true:

1. both repos pass P0 in-scope
2. WildRobot meets the P0 full-matrix parity target against ToddlerBot
3. WildRobot matches ToddlerBot on P1A nominal FK-realized gait shape
4. WildRobot meets the P1 parity target against ToddlerBot
5. WildRobot is not materially rougher than ToddlerBot on P2
6. WildRobot passes `G4`, `G6`, and `G7` without violating `G5`

If P0 fails, fix geometry first.
If P0 passes but P1A fails, fix the nominal gait shape / FK realization.
If P1A passes but P1 fails, fix trackability.
If P1 passes but `G5` fails, the prior is too weak and PPO is compensating too
much.
