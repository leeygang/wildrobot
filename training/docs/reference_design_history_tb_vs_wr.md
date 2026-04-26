# TB vs WR Reference Design Note

Purpose: capture the design-comparison history between ToddlerBot (TB) and
WildRobot (WR) reference evaluation so future WR reference changes can be
measured against a stable baseline.

## Goal

Make WR reference quality demonstrably on par with TB before giving credit to
PPO-side improvements.

The working definition became:

1. WR and TB should pass the same shared geometry gate.
2. WR should be no worse than TB on reference realization quality.
3. WR should be no worse than TB on nominal trackability.
4. Only after that should WR policy-side gates be used to judge training.

## Initial Comparison

The first shared check was the geometry gate:

- stance foot bottom `z <= 0.003 m`
- swing foot bottom `z > -0.002 m`
- probe `vx in {0.10, 0.15, 0.20, 0.25}`
- probe frames `{0, 5, 10, 16, 22, 28, 32, 48, 64}`

Initial result:

- WR passed the current in-scope gate at `vx={0.10, 0.15}`.
- TB also passed the current in-scope gate.
- At higher `vx`, WR had more failures than TB.
- WR’s main issue was stance/swing robustness at higher speed, not the current
  in-scope gate.

Interpretation:

- WR was acceptable for the current contract.
- WR was not yet TB-level outside the current scope.

## First Parity Harness

We added a shared parity harness so WR and TB could be compared with one report.

Key additions:

- TB-side geometry test:
  [tests/test_walk_reference_geometry.py](/home/leeygang/projects/toddlerbot/tests/test_walk_reference_geometry.py)
- WR-side comparison script:
  [tools/reference_geometry_parity.py](/home/leeygang/projects/wildrobot/tools/reference_geometry_parity.py)
- scorecard docs:
  [training/docs/reference_parity_scorecard.md](/home/leeygang/projects/wildrobot/training/docs/reference_parity_scorecard.md)
  [reference_parity_scorecard.md](/home/leeygang/projects/toddlerbot/reference_parity_scorecard.md)

The first version of the scorecard had:

- `P0`: shared geometry gate
- `P1`: nominal stride/contact metrics
- `P2`: smoothness
- `P3`: WR policy gates (`G4/G5/G6/G7`)

## Report Formatting Decisions

The comparison report was then reformatted to be easier to read.

Final table style decisions:

- first column is always the robot name
- rows are:
  - `wildrobot`
  - `toddlerbot_2xc`
  - `toddlerbot_2xm`
- pass/fail columns were removed from the main tables
- WR-vs-TB parity verdicts were moved to footer lines below each table

This made the report read as:

- raw metrics table first
- explicit `WR vs toddlerbot_*` parity result after the table

## Design Review Findings

After the first parity harness, a review found that the script was useful but
not yet measuring reference quality correctly enough.

Main findings:

### 1. `P0` geometry was sound

The shared geometry gate was considered accurate and apples-to-apples.

### 2. The original nominal gait table mixed different quantities

The first version compared:

- WR planner-side foot trajectories
- TB FK-realized foot trajectories from `mujoco_replay`

This made WR-vs-TB clearance and step-length comparison weaker than intended,
because WR was being measured on planner intent while TB was being measured on
realized FK output.

### 3. `P1` trackability was not actually implemented

The scorecard said `P1` should include:

- leg-joint RMSE
- forward speed
- episode length
- termination mode
- contact-phase match

But the original script only computed offline gait-shape statistics. That made
it a geometry/kinematics tool, not a true trackability tool.

### 4. `P2` was incomplete

Only one smoothness-like metric existed in the original script:

- `foot_z_step_max`

Missing from implementation:

- max leg-joint `q_ref` step
- pelvis height step
- swing-foot height step
- contact-mask flips per cycle

### 5. Cadence and speed semantics needed correction

Two design issues were called out:

- `stride_speed_proxy_mps` was misnamed: algebraically it was half-forward-speed
  proxy
- cadence parity was too strict because WR and TB use different fixed cycle
  times by design

## Revised Design

Based on that review, the parity design was changed.

The current structure is:

### `P0`: Shared Geometry Gate

Unchanged in principle.

Use the same WR/TB geometry gate and compare:

- in-scope failures
- full-matrix failures
- worst stance-foot height
- worst swing-foot dip

### `P1A`: FK-Realized Nominal Gait Shape

This was split out from trackability to avoid mixing concerns.

Measure nominal gait shape at `vx=0.15` from FK-realized foot trajectories:

- touchdown step length mean/min
- touchdown rate
- touchdown-speed proxy
- swing clearance mean/min
- double-support fraction
- foot-z single-frame step

Important design decision:

- WR is now measured from FK-realized reference replay, not planner-side foot
  positions

### `P2`: Reference Smoothness

Measure the reference asset itself:

- max shared-leg `q_ref` step
- max pelvis-z step
- max swing-foot-z step
- contact flips per cycle

### `P1`: Free-Base Zero-Residual Replay

Trackability was moved into its own physical replay layer.

Measure across `vx in {0.10, 0.15, 0.20}`:

- shared-leg RMSE
- achieved forward speed
- survived steps
- termination mode
- reference/physical contact match
- physical touchdown step length mean/min
- joint-limit saturation fraction

This is the layer intended to answer:

- is the prior physically trackable enough?
- is PPO likely to fail because the reference itself is weak?

## Current Interpretation Rules

When reading the current parity results:

- `P0` answers: is the reference geometrically valid and roughly TB-like?
- `P1A` answers: does WR realize a TB-like gait shape in FK?
- `P2` answers: is WR materially rougher than TB?
- `P1` answers: can the bare reference be tracked in free-base dynamics?

This means:

- a good `P0` result alone is not enough
- a good `P1A` result alone is not enough
- `P1` is the closest layer to predicting PPO failure caused by the prior

## Current Findings

At the current state of the tools:

- `P0`: WR still fails parity against both TB variants on full-matrix failure
  count and swing margin
- `P1A`: WR is still shorter-step than TB, though speed proxy is close
- `P2`: WR is rougher on swing-foot z step
- `P1`: WR still fails parity on several free-base replay metrics, especially
  contact match, touchdown step length, and saturation

One useful outcome of the new `P1` layer is that it also shows a broader
problem:

- both bare references are weak in free-base replay
- WR is not strong enough relative to TB to claim parity

So the parity tool now gives a more defensible answer than the original script:

- not just “does the reference look like a walk?”
- but “is the reference realized, smooth, and trackable enough to compare WR
  fairly against TB?”

## Key Design Decisions To Preserve

These are the main decisions future changes should not accidentally undo:

1. Keep `P0` shared between WR and TB.
2. Do not compare WR planner-space foot trajectories against TB FK-realized
   foot trajectories.
3. Keep nominal gait shape (`P1A`) separate from physical trackability (`P1`).
4. Keep smoothness (`P2`) explicit instead of hiding it inside one foot-step
   metric.
5. Treat cadence as a bounded design difference, not a forced equality target.
6. Use TB as the parity baseline before giving PPO-side credit to WR.

## Related Files

- [tools/reference_geometry_parity.py](/home/leeygang/projects/wildrobot/tools/reference_geometry_parity.py)
- [training/docs/reference_parity_scorecard.md](/home/leeygang/projects/wildrobot/training/docs/reference_parity_scorecard.md)
- [tests/test_v0200c_geometry.py](/home/leeygang/projects/wildrobot/tests/test_v0200c_geometry.py)
- [tests/test_walk_reference_geometry.py](/home/leeygang/projects/toddlerbot/tests/test_walk_reference_geometry.py)
- [reference_parity_scorecard.md](/home/leeygang/projects/toddlerbot/reference_parity_scorecard.md)

## Practical Use

Use this design note when:

- changing WR ZMP / IK / gait timing design
- deciding whether a WR reference improvement is real or just PPO compensation
- explaining why a WR training gain should or should not be credited to the
  reference
- extending the parity script with new WR/TB comparison metrics
