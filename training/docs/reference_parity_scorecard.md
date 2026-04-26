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

### Size-normalised parity (Option A — companion to P0–P2)

WildRobot is a meaningfully bigger robot than ToddlerBot (leg ≈ 0.37 m
vs 0.21 m, COM height ≈ 0.46 m vs 0.29 m). Several absolute metrics in
P1A / P1 / P2 are size-sensitive — most importantly step length, swing
clearance, cadence, and drift rate scale with leg length or √(L/g).

The script reports a parallel set of size-normalised gates so the parity
reading is fair across the size delta, while keeping the absolute gates
in place as sanity checks.

Characteristic dimensions used:

- WildRobot: `leg_length_m = ZMPWalkConfig.upper_leg_m + lower_leg_m`,
  `com_height_m = ZMPWalkConfig.com_height_m`
- ToddlerBot 2xc / 2xm: `leg_length_m = robot.yml hip_to_ankle_pitch_z`,
  `com_height_m = robot.yml com_z`

Normalised metrics (all dimensionless):

- `step_length_per_leg = step_length_mean_m / leg_length_m`
- `swing_clearance_per_com_height = swing_clearance_mean_m / com_height_m`
- `cadence_froude_norm = touchdown_rate_hz · √(com_height_m / g)`
- `swing_foot_z_step_per_clearance = swing_foot_z_step_max_m / swing_clearance_mean_m`
- `forward_speed_froude = vx² / (g · com_height_m)` (operating-point characterisation)
- `achieved_forward_per_cmd = achieved_forward_mps / vx` (P1)
- `terminal_drift_rate_per_leg_per_s = (terminal_root_x_m / duration_s) / leg_length_m` (P1)
- `td_step_length_per_leg = touchdown_step_length_mean_m / leg_length_m` (P1)

Normalised parity targets:

- WildRobot `step_length_per_leg ≥ 0.85 ×` ToddlerBot
- WildRobot `swing_clearance_per_com_height ≥ 0.85 ×` ToddlerBot
- WildRobot `cadence_froude_norm ≤ 1.20 ×` ToddlerBot
- WildRobot `swing_foot_z_step_per_clearance ≤ 1.10 ×` ToddlerBot
- WildRobot `achieved_forward_per_cmd ≥` ToddlerBot `− 0.10` per `vx` bin
- WildRobot `|terminal_drift_rate_per_leg_per_s| ≤` ToddlerBot `+ 0.5 leg/s` per bin
- WildRobot `td_step_length_per_leg ≥ 0.85 ×` ToddlerBot per bin

Interpretation:

- **Both verdicts (absolute and normalised) are reported**. Absolute is
  the literal "WR matches TB's numbers" sanity floor; normalised is the
  fair "after-size-correction parity" reading.
- A passing absolute gate with a failing normalised gate means WR happens
  to land near TB's number but is operating at a proportionally
  different gait point (e.g. step length matches but as a much smaller
  fraction of leg length).
- A failing absolute gate with a passing normalised gate means WR is
  proportionally on par; the absolute gap is purely a size artifact.
- `--strict` requires BOTH verdicts to pass.

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
6. WildRobot meets the size-normalised parity targets on FK and P1
7. WildRobot passes `G4`, `G6`, and `G7` without violating `G5`

If P0 fails, fix geometry first.
If P0 passes but P1A fails, fix the nominal gait shape / FK realization.
If P1A passes but P1 fails, fix trackability.
If P1 passes but `G5` fails, the prior is too weak and PPO is compensating too
much.

## Latest Analysis Result (2026-04-25)

Source: `tools/parity_report.json` from
`uv run ./tools/reference_geometry_parity.py` at commit `c1b815d`. The
read below leads with the **size-normalised** view because absolute
units are confounded by the WR/TB size delta (WR leg ≈ 0.37 m vs TB
0.21 m, COM ≈ 0.46 m vs 0.29 m).

Phase 5 lifecycle note: the parity tool now resolves WR references
through a deterministic reusable library path (cache key from
`ZMPWalkConfig` + required `vx` bins + generator-source fingerprint)
instead of ad-hoc per-call generation. Default cache root is
`.cache/wr_reference_libraries`.
The fingerprint includes WR generator code plus `assets/v2/*.xml` and
`assets/v2/*.json` inputs used by Phase 3 fixed-base FK replay.
Use `--wr-library-path <dir>` to pin parity/debug runs to a specific
saved WR asset.

### Top-line verdicts

| | Status |
|---|---|
| Absolute (`overall_pass`) | FAIL |
| Size-normalised (`normalized_overall_pass`) | FAIL |

### Where the absolute and normalised lenses **disagree**

Three P1A / P1 gates flip when read normalised — these are the
load-bearing readings:

| Metric | Absolute | Normalised | Why the flip matters |
|---|---|---|---|
| P1A `step_length` | 88 % of TB (close) | **50 % of TB-per-leg** (gulf) | WR's stride is half of TB's as a fraction of leg length. The absolute reading hides this. |
| P1A `swing_clearance` | 134 % of TB (generous) | **83 % of TB-per-h** (under) | WR's higher absolute clearance is a size artifact; proportionally WR is below TB. |
| P1A `cadence` | 1.12× TB (PASS) | **141 %** of Froude-norm (FAIL) | A 1.76×-longer-leg robot should walk **slower**, not faster. WR is over-cadenced. |
| P1 `terminal_drift` | WR more negative (looks worse) | **WR drifts half as much per leg/s** as TB | The "WR walks backward worse than TB" reading was a size artifact. |

### Where both lenses agree

- **P0 swing margin**: WR worst swing z = −1.7 mm vs TB +9.96 mm — 12 mm
  margin gap. Real geometry weakness, not size-related.
- **P2 swing-foot z step**: 24 mm/frame WR vs 10 mm/frame TB. Even
  normalised by clearance, WR is **1.81× rougher** at the swing
  lift-off / touchdown frames. Real planner discontinuity bug.
- **P0 high-vx (≥ 0.20) full matrix**: 7 stance failures vs TB's 2.
  Out of in-scope (vx ≤ 0.15), so deferred to a later prior-extension
  milestone.

### What this changes in the gap diagnosis

1. **The prior asset has a real proportional-stride deficit.** WR's
   stride at 50 % of TB's (per leg) and clearance at 83 % (per COM
   height) are the load-bearing FK-side gaps.
2. **Cadence is over-Froude (141 %), the inverse of the stride
   problem.** WR is taking faster, shorter steps where a longer-legged
   robot should take slower, longer steps. Both follow from the same
   `cycle_time` choice.
3. **Swing-foot lift-off discontinuity is a planner bug, independent of
   size.** Holds at 1.81× TB even after normalisation.
4. **Closed-loop "WR walks backward worse than TB" was a size artifact.**
   Per-leg drift rate shows WR is **better** than both TB variants:
   −0.48 leg/s (WR) vs −0.92 leg/s (TB-2xc) vs −1.12 leg/s (TB-2xm) at
   vx=0.10. There is **no proportional dynamic-trackability gap** for
   PPO to overcome beyond what fixing the prior asset will address.
5. **The absolute P1 RMSE / contact gaps are mostly actuator-architecture
   artifacts** (WR position actuators vs TB torque + PD) plus a sample-
   size artifact (TB falls in 17–35 ctrl steps; WR survives 63–191).
   These don't indicate a prior-quality problem.

### Architectural alignment moves toward ToddlerBot (priority order)

Following CLAUDE.md "follow ToddlerBot before WR-specific design": every
deviation from TB needs an explicit rationale. The current WR ZMP prior
has at least three values that were chosen WR-specifically without a
load-bearing rationale and now show as the load-bearing parity gaps.

1. **`cycle_time_s`** — adopt TB's value. Required to bring
   `cadence_norm` toward parity (drives the 141 % over-cadence finding).
2. **`foot_step_height_m`** — adopt TB's value. Closes the
   `swing_clearance_per_com_height` and `swing_clearance` parity gaps.
3. **Swing-foot z trajectory shape** — apply WR's floor-clearance offset
   smoothly so it vanishes at lift-off / touchdown, matching TB's
   continuous swing curve. Closes the P0 swing-margin gap and the P2
   swing-step gap.

Items 1 and 2 are TB-default adoption; item 3 is a TB-shape alignment
that keeps the WR-specific compensation but applies it without the
discontinuity. After these land, the remaining FK-side gap is
`step_length_per_leg`, which is partly a curriculum question (Froude-
equivalent operating vx) rather than a prior knob.
