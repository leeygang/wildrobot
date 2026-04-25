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

Run a nominal-only replay with no learned residual.

Required metrics:

- per-joint tracking RMSE over the 8 leg joints
- forward velocity at `vx=0.15`
- mean episode length
- first termination mode
- contact-phase match against the reference contact schedule
- touchdown step length mean/min at `vx=0.15`
- touchdown rate (cadence) at `vx=0.15`
- stride-speed proxy at `vx=0.15`
  `step_length_mean * touchdown_rate / 2`
- swing-clearance mean/min at `vx=0.15`
- double-support fraction at `vx=0.15`

Pass criteria:

- no NaN, no joint runaway, no repeated hard-limit clipping
- contact-phase match `>= 0.90` over the probe horizon

Parity target:

- WildRobot leg-joint RMSE `<= 1.10 x` ToddlerBot
- WildRobot forward velocity `>=` ToddlerBot `- 0.02 m/s`
- WildRobot episode length `>= 0.95 x` ToddlerBot
- WildRobot touchdown step length mean `>= 0.90 x` ToddlerBot
- WildRobot touchdown rate `<= 1.10 x` ToddlerBot
- WildRobot stride-speed proxy within `0.01 m/s` of ToddlerBot
- WildRobot swing-clearance mean `>= 0.85 x` ToddlerBot
- WildRobot double-support fraction within `0.05` absolute of ToddlerBot

Interpretation:

- failing here means the reference is not physically trackable enough, even if
  the pure FK geometry looks acceptable
- specifically for the current WR short-stride / shuffle risk:
  if step length is low while touchdown rate is high but the stride-speed proxy
  is still near TB, then WR is matching nominal speed by taking shorter, faster
  steps rather than matching TB-like stride geometry

### P2. Reference Smoothness

Check the reference asset itself, without physics.

Required metrics:

- max `|q_ref[t] - q_ref[t-1]|`
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
3. WildRobot meets the P1 parity target against ToddlerBot
4. WildRobot is not materially rougher than ToddlerBot on P2
5. WildRobot passes `G4`, `G6`, and `G7` without violating `G5`

If P0 fails, fix geometry first.
If P0 passes but P1 fails, fix trackability.
If P1 passes but `G5` fails, the prior is too weak and PPO is compensating too
much.
