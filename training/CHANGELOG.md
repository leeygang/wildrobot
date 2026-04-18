# AMP Training Changelog
# CHANGELOG

This changelog tracks capability changes, configuration updates, and training results for the WildRobot AMP training pipeline.

**Ordering:** Newest releases first (reverse chronological order).

---

## [v0.20.0-C] - 2026-04-18: ToddlerBot-style multi-cycle ZMP plan

### Problem
Cyclic single-cycle replay of the ZMP reference at `vx=0.15` made the
robot lift its legs in the right order but never translate, then fall.

Root cause: the one-cycle library entry was not steady-state periodic.
Phase A's right-foot trajectory started at world `x=0` and swung only
`sl` (a "from-rest" half-step), while Phase B's left swing was a full
`2*sl` steady-state stride.  Diagnostic dump at the wrap boundary
showed right hip pitch jumping `+0.082 rad` (≈4.7°), right foot
position jumping `-0.034 m` (COM-relative), and the contact mask
flipping `(0,1) → (1,1)` — the legs got rewound forward each wrap, so
no net world-frame translation accumulated.

### Decision
Adopt ToddlerBot's `toddlerbot.algorithms.zmp_walk` shape: one
trajectory per command bin spans `cfg.total_plan_time_s = 22 s` of
monotonic forward walking (cycle 0 from rest, cycles 1+ LIPM
steady-state) and is replayed linearly.  No periodicity contract on
the cycle boundary, so no asymmetric Phase A vs Phase B to keep in
sync.

### Changes
- `control/zmp/zmp_walk.py`: rewrote `_generate_at_step_length` to emit
  `n_cycles * 32` steps with cycle-offset world positions, a per-cycle
  LIPM x0/vx0 sagittal pattern, and a cosine lateral weight-shift.
  Pelvis pose is now monotonic (matches the IK frame).  Removed the
  unused `n_warmup_cycles` knob.
- `training/eval/view_zmp_in_mujoco.py`: replaced `step_idx % n_cycle`
  with linear playback clamped to `n_steps`.  Logs phase instead of
  cycle index.
- `control/references/reference_library.py`: docstring now states
  trajectories may span multiple gait cycles and must be played
  linearly (no wrap to index 0 — first cycle is from-rest, not
  periodic with the last steady-state cycle).

### Validation
- v0.20.0-A schema contract: 14/14 tests still pass.
- Kinematic replay (`--kinematic`) at `vx=0.15`: 100 control steps →
  `root_x` `-0.024 → +0.246 m` → realized speed `0.150 m/s` (matches
  command exactly).
- Fixed-base replay (`--fixed-base`) at `vx=0.15`: 100 steps,
  `root_z` flat at `0.4683`, `root_x = +0.301 m`, pitch/roll at
  zero — servos track the multi-cycle q_ref with no drift.
- Free-floating replay still falls inside one gait cycle as expected
  (no balance feedback yet — that is v0.20.1's PPO job; the v0.20.0-C
  gate is q_ref trackability, not open-loop walking).

### Strict-pass diagnostics (2026-04-18, revised after review)

The first strict-pass attempt was invalidated by reviewer findings:

- **F1**: `ReferenceLibrary.get_preview` still wrapped indices with
  modulo, contradicting the multi-cycle "no wrap" contract.  Fixed:
  `get_preview` and `_find_next_foothold` now clamp to the last frame
  and the contract test was rewritten as `test_preview_clamps_at_end`.
- **F2**: fixed-base mode hard-coded `qpos[0] += dt * command_vx`
  every step, so the previously reported "exact 0.10/0.15/0.20/0.25
  m/s tracking" was synthetic, not a PD result.  Fixed: pelvis is now
  pinned at the standing pose; root_x stays at 0.0006 m across 200
  control steps.  Fixed-base is restricted to RMSE / saturation
  evidence; stride length must come from free-floating.
- **F3**: top-of-file docstring still said trajectories "are stored
  per gait cycle and meant to be looped" — replaced with the linear
  multi-cycle contract.
- **F4**: per-joint RMSE / saturation / touchdowns now logged in
  free-floating replay so the strict-pass evidence comes from one
  consistent mode.

Re-run sweep with the corrected viewer:

**Fixed-base (pelvis pinned, PD legs only — RMSE / saturation only)**

| vx | overall RMSE | knee RMSE L/R | hip-roll RMSE L/R | sat % |
|----|--------------|---------------|-------------------|-------|
| 0.10 | 0.189 | 0.345 / 0.335 | 0.080 / 0.044 | 3.5 |
| 0.15 | 0.190 | 0.345 / 0.346 | 0.080 / 0.042 | 3.0 |
| 0.20 | 0.193 | 0.352 / 0.351 | 0.082 / 0.043 | 2.0 |
| 0.25 | 0.194 | 0.356 / 0.357 | 0.071 / 0.034 | 2.0 |

Pelvis pitch p95 = 0.0018 rad, height flat 0.4683 m.

**Free-floating (full physics, no balance feedback — survival-bounded)**

Touchdowns now detected from real MuJoCo `data.contact` foot/floor
geom pairs (R1 fix); previous reference-mask numbers were biased
because the q_ref ticks normally during a fall.

| vx | survival ctrl steps | L+R touchdowns (physics) | L step / ratio | R step / ratio | overall RMSE | any-joint sat % |
|----|---------------------|--------------------------|-----------------|-----------------|---------------|-----------------|
| 0.10 | 88 (1.76 s) | 6+4 | -0.016 / -0.51 | -0.019 / -0.59 | 0.191 | 39.8 |
| 0.15 | 124 (2.48 s) | 7+4 | -0.025 / -0.53 | -0.018 / -0.37 | 0.190 | 28.2 |
| 0.20 | 127 (2.54 s) | 7+6 | -0.010 / -0.16 | -0.006 / -0.09 | 0.188 | 30.7 |
| 0.25 | 108 (2.16 s) | 6+6 | -0.001 / -0.01 | +0.009 / +0.11 | 0.197 | 61.1 |

Real-contact counts are **higher** than the reference-mask counts
recorded earlier (e.g. 6+4 vs 2+3 at vx=0.10) — the feet bounce
during the fall, generating touchdown events the q_ref does not
encode.  Strides remain negative / near-zero, confirming the
open-loop direction is not forward.

Termination cause is consistently a pitch / roll fall, not a q_ref
discontinuity.  Library validate() clean for all 5 entries; zero
joint-limit violations encoded in q_ref.

### v0.20.0-C gate scoring against `reference_design.md` L716-737 (revised)

| Gate | free-floating | fixed-base | kinematic | Verdict |
|------|---------------|------------|-----------|---------|
| Touchdown locations match intended pattern | ✓ count, but in fall direction | ✓ count | ✓ | ⚠ |
| No toe drag / scissoring / pathological posture | ✓ until fall | ✓ | ✓ | ✓ |
| Startup / 1-2 cycles coherent in env | ⚠ falls in 1.7-2.5 s | ✓ | n/a | ⚠ |
| Failures = tracking limits, not incoherent ref | ✓ (balance loss, not q_ref) | ✓ | n/a | ✓ |
| Realized step ≥ 0.03 m | ❌ negative / near-zero | n/a (pinned) | ✓ exact sl | ❌ free-floating |
| Realized/cmd step ratio ≥ 0.5 | ❌ negative or ≤ 0.06 | n/a | ✓ 1.00 | ❌ free-floating |
| IK reachability ≈ 1.0 | ✓ | ✓ | ✓ | ✓ |
| Safety margin violations = 0 | ✓ at q_ref level | ✓ | ✓ | ✓ |
| Survives ≥ 45 ctrl steps | ✓ 88-127 | ✓ 200/200 | n/a | ✓ |
| ≥ 1L + 1R touchdown in same replay | ✓ all bins | ✓ 6+6 | ✓ | ✓ |
| Tracking RMSE within actuator budget | ⚠ knee 0.34, no budget | ⚠ knee 0.35 | n/a | ⚠ |
| No repeated hard saturation | ❌ 28-61% in free-floating | ✓ 2-3.5% pinned | n/a | ❌ free-floating |

### Honest assessment

Three gates fail under strict free-floating measurement:
realized step length, realized/cmd ratio, and saturation %.
All three failures share a single cause: open-loop LIPM is unstable
without a state-feedback controller.  Evidence:

- Pinned pelvis (fixed-base) shows the q_ref is trackable at
  3 % saturation — the saturation in free-floating (28-61 %) is a
  **consequence of the fall**, not a property of q_ref.
- Kinematic playback (no physics) shows the q_ref encodes exactly
  the commanded forward speed and step length.
- Free-floating tumbles in 1.7-2.5 s with negative strides because
  the actual robot starts at standing (vx=0) while the q_ref expects
  LIPM steady-state initial conditions (vx≈+0.18 m/s).  No mechanism
  in v0.20.0-C exists to bridge this — that is `v0.20.1`'s PPO job.

The remaining real signal is the **knee tracking RMSE 0.34 rad**
(max_abs 0.94 rad ≈ 54°): present in both fixed-base and
free-floating, so it is a servo-vs-q_ref issue, not balance.
It does not block v0.20.0-C (no explicit RMSE budget in the gate),
but it is a hard flag for `v0.20.2` SysID and for `v0.20.1`
PPO authority on the knee channel.

### Findings flagged for later milestones

- **`v0.20.2` (SysID)**: knee servo lags ~0.94 rad peak under PD;
  verify htd45hServo gains and rotor inertia in MJCF match hardware.
- **`v0.20.1` (PPO)**: lateral asymmetry — L hip-roll RMSE is 1.8-2.0×
  R across all vx in fixed-base, growing to 3-4× in free-floating
  (which falls left-first).  Likely root cause: cosine lateral COM
  starts at `+lat_amplitude` (toward left), loading the left hip
  roll first.  Worth A/B-testing a phase-shifted or sin-based
  lateral pattern.
- **Gate-language**: the v0.20.0-C metric gates assume a closed-loop
  context; with a pure offline prior they are unmeetable.  Either
  add minimal stabilization to the validation harness or reword the
  gate to acknowledge the open-loop limitation.

### Reviewer round 2 (2026-04-18 PM): R1-R3 closeout

- **R1 (medium)**: touchdowns were inferred from the q_ref's
  contact mask, so a falling robot still appeared to land on
  schedule.  Replaced with `data.contact` / floor-geom pair detection
  in `view_zmp_in_mujoco.py`.  Real-contact counts are 1.5-3× higher
  than ref-mask counts (feet bounce during the fall), and strides
  shift modestly (e.g. vx=0.10 R ratio -0.90 → -0.59).  Forward-stride
  conclusion is unchanged.
- **R2 (medium)**: `get_preview` now emits a one-shot RuntimeWarning
  per command_key when `step_index >= n_steps`.  Test
  `test_preview_warns_on_overrun` asserts the warning fires once and
  is silent on subsequent overruns.  Docstring spells out the
  episode-horizon contract for callers.
- **R3 (low)**: stale "Library entry: one gait-cycle trajectory"
  section header replaced with multi-cycle linear-replay wording.
- Contract test suite: 15/15 pass after the new overrun-warning test.

### v0.20.0-C closeout sweep — C1 + C2 contract execution

Implemented the v0.20.0-C C1 + C2 gate per the contract frozen in
`reference_design.md` after rounds 1-4 of external review.

**Implementation**
- `training/eval/c2_stabilizer.py`: bounded validation-only feedback
  harness with three channels — torso pitch PD → ankle pitch offset
  on stance side (clip ±0.10 rad), torso roll PD → hip roll offset
  on stance side (clip ±0.05 rad), capture-point swing nudge held
  from foot lift to next touchdown (clip ±0.03 m via hip pitch).
  Inputs limited to measured `data.contact` foot/floor pairs and
  time-derived `(sin, cos)(2π·t/cycle_time)`; trajectory annotations
  forbidden.  Forbidden imports from `runtime/` enforced by docstring
  + reviewer.  Sign of pitch / roll application empirically tuned
  (`apply_pitch_sign=-1`, `apply_roll_sign=+1`) via single-bin probe
  before locking and running the closeout matrix.
- `training/eval/view_zmp_in_mujoco.py`: added `--c2-stabilizer` and
  `--seed` flags.  IC perturbation (Q2 outcome): pelvis x/y ±0.02 m,
  yaw ±5°, vx ±0.10 m/s.  Worst-joint RMSE (Q1 amendment) and per-
  channel harness clip-saturation reported in summary.
- `tools/v0200c_closeout.py`: 32-row sweep runner, parses metrics,
  scores against C1 + C2 gates, emits CHANGELOG-ready artifact.

**Closeout artifact (32 rows, all command lines reproducible)**

- harness commit sha: `0d24d2b5d1` (parent of this commit)
- reference library hash: `3b2f79fa89` (`control/zmp/zmp_walk.py`)
- horizon per row: 200 ctrl steps (4.0 s)
- C1 survival budget: `ceil(2.0 · 0.64 / 0.02) = 64` ctrl steps
- seed family: `numpy.random.default_rng(seed)` for seeds [0, 1, 2]

| mode | rows | passed | failed | gate verdict |
|---|---|---|---|---|
| kinematic     |  4 |  4 |  0 | **PASS** |
| fixed-base    |  4 |  4 |  0 | **PASS** |
| C1            | 12 | 12 |  0 | **PASS** |
| C2            | 12 |  0 | 12 | **FAIL** |

C1 (open-loop coherence) sweep — survival in ctrl steps:

| vx | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.10 |  90 |  79 |  90 |
| 0.15 | 117 | 112 | 115 |
| 0.20 | 132 | 161 | 129 |
| 0.25 |  82 | 123 |  85 |

All 12 C1 rows clear the 64-step survival budget and produce ≥ 1 L
+ 1 R real-contact touchdown per replay.  Failures are pitch / roll
balance loss, not q_ref discontinuity or joint runaway.  C1 PASS.

C2 (validation-only stabilizer harness) — closed-loop step length
and ratio per side, harness clip-saturation per channel:

| vx | seed | survived | L step (m) / ratio | R step (m) / ratio | pitch_clip % | roll_clip % | CP_clip % |
|---|---|---|---|---|---|---|---|
| 0.10 | 0 | 109 | -0.035 / -1.08 | -0.019 / -0.60 | 22.9 | **39.4** |  2.8 |
| 0.10 | 1 |  95 | -0.003 / -0.10 | -0.028 / -0.88 | 21.1 | **55.8** |  1.1 |
| 0.10 | 2 | 115 | -0.062 / -1.93 | -0.062 / -1.94 | **30.4** | 20.0 |  2.6 |
| 0.15 | 0 | 154 | -0.023 / -0.48 | -0.024 / -0.50 | 19.5 | **32.5** |  2.6 |
| 0.15 | 1 | 114 | -0.032 / -0.67 | -0.009 / -0.19 | 22.8 | **28.1** |  2.6 |
| 0.15 | 2 | 200 |  0.000 /  0.00 | -0.000 / -0.01 |  7.5 |  5.5 |  2.0 |
| 0.20 | 0 | 163 |  0.012 /  0.18 |  0.009 /  0.13 | 20.9 |  3.1 |  2.5 |
| 0.20 | 1 | 177 | -0.010 / -0.16 | -0.004 / -0.06 |  7.3 | **31.6** |  1.1 |
| 0.20 | 2 | 140 | -0.009 / -0.14 | -0.007 / -0.11 | 19.3 | **31.4** |  2.9 |
| 0.25 | 0 | 129 |  0.003 /  0.03 | -0.008 / -0.10 | 22.5 |  4.7 |  0.8 |
| 0.25 | 1 | 129 | -0.009 / -0.12 | -0.008 / -0.10 | 16.3 | **35.7** |  3.1 |
| 0.25 | 2 | 108 | -0.004 / -0.05 | -0.005 / -0.06 | 10.2 | **44.4** |  0.9 |

C2 fail breakdown (every row fails ≥ 1 metric):
- 12/12 fail step-length gate (`>= 0.03 m`, both sides).
- 12/12 fail realized/cmd ratio gate (`>= 0.5`, both sides).
- 7/12 fail roll_PD clip-saturation (≥ 25 % hard fail).
- 1/12 fails pitch_PD clip-saturation.
- 0/12 fail CP_nudge clip-saturation (channel rarely pinned).

Worst-joint RMSE (Q1 amendment) — knee dominates across all modes
and bins, never exceeds the 0.40 rad per-joint feasibility budget,
but is consistently 80-95 % of it.  Recorded for `v0.20.2` SysID:

| mode | worst-joint RMSE range | dominant joint |
|------|-----------------------|----------------|
| fixed-base | 0.344 – 0.357 | L/R knee_pitch |
| C1         | 0.330 – 0.380 | L/R knee_pitch |
| C2         | 0.322 – 0.383 | L/R knee_pitch |

**Decision per `reference_design.md` v0.20.0-D rule table**

| C1 | C2 | Action |
|---|---|---|
| pass | fail on all bins | stay in C; investigate prior quality (lateral asymmetry, knee feasibility) — stabilizer cannot rescue what is not rescuable |

**v0.20.0-C verdict: NO-GO to D.  Stay in C, fix prior quality.**

What the artifact says about WHY C2 fails (none of these are
stabilizer-tuning issues):

1. **Lateral asymmetry dominates the failure surface.** 7/12 C2
   rows hit the roll_PD hard-fail (≥ 25 %) within the ±0.05 rad
   clip — the harness is pinned the entire replay trying to fight
   a roll bias the prior creates.  The cosine lateral COM
   trajectory in `control/zmp/zmp_walk.py` starts at
   `+lat_amplitude` (toward LEFT), which loads the L hip roll
   first.  This was already flagged in the prior closeout as a
   v0.20.1 concern; the C2 sweep confirms the bound on what any
   bounded correction can rescue.
2. **Open-loop forward propulsion is missing.** Even when survival
   is good (vx=0.15 seed=2: 200 ctrl steps), the realized step is
   essentially 0 (`L=+0.0001 m`, `R=-0.0004 m`).  The IC mismatch
   between standing (vx=0) and the LIPM steady-state (vx ≈ +0.18
   m/s) cannot be closed by ±0.03 m of capture-point swing nudge
   — the CP channel only saturates 1-3 % of the time, meaning it's
   not even using the budget it has.  The propulsion deficit is in
   the reference itself, not in the harness's authority.
3. **Knee tracking holds within budget across the matrix.** The
   knee servo lags ~0.34 rad RMSE in every mode but stays under
   the 0.40 rad per-joint budget.  This remains a v0.20.2 SysID
   concern but does not block C2 directly.

**Recommended follow-up before next C2 attempt** (block until done):

- Lateral COM trajectory: A/B-test sin-based weight shift vs the
  current `+lat_amplitude·cos(πφ)` start, to reduce the L-side
  bias that pins roll_PD.  Re-run only the C2 portion (12 rows)
  to confirm the bias drops below the hard-fail.
- Propulsion: align the kinematic IC closer to LIPM steady-state
  in the viewer's standing init (or accept that a wider IC
  perturbation envelope around +0.18 m/s vx is required to match
  the LIPM frame).  The current ±0.10 m/s window centered on 0
  cannot reach the LIPM target.
- Do **not** loosen any harness clip per the decision rule.  Do
  **not** widen the harness scope.  Do **not** tune gains
  further; the empirical sign is correct and the channel
  authority is at the contract limit.

### v0.20.0-C closeout sweep round 2 — both prior fixes applied

Applied both prior fixes recommended in the round-1 closeout
follow-up section above:

- **Lateral COM amplitude reduced** from `0.5*lat` to `0.25*lat`
  in `control/zmp/zmp_walk.py`, keeping the smooth single-cosine
  shape (one full lateral cycle per gait cycle).  An A/B with a
  per-phase sin pattern (peak-at-mid-stance, two peaks per cycle)
  was rejected because it doubled the lateral frequency and
  worsened C2 roll_PD clip-saturation rather than fixing it.
- **IC vx aligned with LIPM steady-state** in the viewer:
  perturbation now centered on the trajectory's LIPM IC vx
  (read from `traj.pelvis_pos` finite difference at t=0) plus
  `±0.10 m/s` envelope, instead of centered on zero.  Recorded as
  a "post-closeout addendum" under the Q2 entry in
  `reference_design.md`.
- **CP-nudge gain** raised from 0.30 → 0.7 within the ±0.03 m
  hard clip; clip-saturation moved from 0-3 % to 4-7 % so the
  channel actually uses some of its budget.  No clip change.

Re-running the 32-row matrix:

| mode | rows | passed | failed | gate verdict |
|---|---|---|---|---|
| kinematic     |  4 |  4 |  0 | **PASS** |
| fixed-base    |  4 |  4 |  0 | **PASS** |
| C1            | 12 | 10 |  2 | **FAIL** (regressed from 12/12) |
| C2            | 12 |  0 | 12 | **FAIL** (no row clears closed-loop step gate) |

C1 regression — both rows are at the IC alignment edge:

- C1 vx=0.20 seed=1: 50 ctrl steps (under the 64-step C1 budget)
- C1 vx=0.25 seed=1: 18 ctrl steps (catastrophic)

At `vx_cmd=0.25` the LIPM IC vx is `≈ +0.30 m/s`.  Centered
perturbation + worst-seed dvx still gives `+0.20 m/s` of forward
velocity injected into a body still at the standing keyframe pose.
The body face-plants forward inside the standing-to-walking
ramp.  Fix in next iteration: cap the IC vx envelope so it cannot
exceed a fraction of the LIPM steady-state, or move the IC
alignment to cap-by-vx_cmd.

C2 progress (no row passes, but the failure surface moved):

| metric | round 1 (commit 6000177) | round 2 (this commit) |
|---|---|---|
| C2 roll_PD ≥ 25 % hard-fail rows | 7/12 | 1/12 |
| C2 pitch_PD ≥ 25 % hard-fail rows | 1/12 | 6/12 |
| C2 any-joint sat ≥ 25 % hard-fail rows | 5/12 | 1/12 |
| C2 rows surviving 200/200 | 1/12 (vx=0.15 seed=2) | 3/12 (vx=0.15 seed=0,1,2 are 86/72/91; vx=0.25 seed=0,2 are 200/200) |
| best realized/cmd ratio (forward direction) | +0.18 (vx=0.20 s=0) | +0.35 (vx=0.10 s=2) |

Lateral asymmetry is largely fixed (1/12 rows now hit roll_PD
hard-fail vs 7/12 before).  Pitch_PD is now the dominant
bottleneck — the IC alignment lets the body try to walk at LIPM
vx, but the standing-to-walking transient demands more pitch
correction than the ±0.10 rad clip provides.  Survival is
substantially better in many bins (vx=0.25 seed=0 and seed=2
both run the full 200 ctrl steps with realized step ≈ 0), but the
body still cannot sustain forward translation under bounded
feedback.

### Why we are still NO-GO to v0.20.0-D

Per the decision rule (`reference_design.md` v0.20.0-C):

| C1 | C2 | Action |
|---|---|---|
| **fail** | n/a | **fix prior generation; do NOT spend effort on the stabilizer first** |

C1 regressed to 10/12 because the IC alignment overshoots at high
vx, so we hit the top entry of the rule table.  The decision is
unambiguous: the prior needs another iteration before another C2
sweep is even meaningful.

### Concrete next prior-fix queue (in order)

1. **Cap IC vx envelope at high vx_cmd** so vx=0.25 doesn't face-
   plant.  Either: (a) clamp the IC vx center to `min(vx_lipm,
   0.18 m/s)`, or (b) scale the perturbation envelope down with
   vx_cmd.  Cheapest fix for the C1 regression.
2. **From-standing startup transient inside the prior** (the
   long-term right answer).  Modify `_generate_at_step_length`
   in `control/zmp/zmp_walk.py` so cycle 0's COM trajectory
   starts at (x=0, vx=0) and accelerates smoothly into the LIPM
   steady-state by end of cycle 0 or 1.  Mirrors the trick
   ToddlerBot uses (truncate first-cycle warmup) but baked into
   the generator instead of into the validation harness.  This
   removes the standing-to-walking transient from the body's
   closed-loop problem.
3. **Knee feasibility** (deferred to v0.20.2): worst-joint RMSE
   knee remains 0.32-0.40 rad, with one row (vx=0.25 seed=1)
   actually crossing the 0.40 budget.  Not blocking C right now
   but tightens the SysID requirement.

### Implementation in this commit

Files changed (all in this round 2 iteration):

- `control/zmp/zmp_walk.py`: lateral amplitude `0.5*lat → 0.25*lat`,
  cosine pattern preserved (one cycle/gait); inline sin A/B
  documented in the comment block
- `training/eval/view_zmp_in_mujoco.py`: `_apply_ic_perturbation`
  takes `traj` and centers vx on LIPM IC; logging shows both
  vx_center and dvx
- `training/eval/c2_stabilizer.py`: cp_gain `0.30 → 0.7`
- `training/docs/reference_design.md`: post-closeout addendum
  under Q2

The harness contract (hard clips, allowed inputs, forbidden
inputs) is unchanged.  No runtime imports are introduced.

### v0.20.0-C closeout sweep round 3 — IC perturbation split

Round 2 regressed C1 because the LIPM-aligned vx was injected at
the standing keyframe, then the standing-to-walking ramp had to
reconcile a body already moving forward at `+0.20-0.30 m/s` with
joints still in the standing pose.  At high vx_cmd the body face-
planted before the ramp finished.

Fix: split the IC perturbation around the ramp.  Position and yaw
apply BEFORE the ramp (body settles at the perturbed pose with
vx=0); vx applies AFTER the ramp (body is already in the walking
pose, so the LIPM-aligned vx injection no longer transients
through the ramp).  Implemented in
`training/eval/view_zmp_in_mujoco.py` via `_apply_ic_pose` and
`_apply_ic_vx` (replacing the single `_apply_ic_perturbation`).

Re-running the 32-row matrix:

| mode | rows | passed | failed | gate verdict |
|---|---|---|---|---|
| kinematic     |  4 |  4 |  0 | **PASS** |
| fixed-base    |  4 |  4 |  0 | **PASS** |
| C1            | 12 | 12 |  0 | **PASS** (recovered from 10/12) |
| C2            | 12 |  0 | 12 | **FAIL** (no row clears closed-loop step gate) |

C1 fully recovered.  Survival range now 78-200 ctrl steps across
all 12 C1 rows.  Both regressed rows from round 2 are fixed:

- C1 vx=0.20 seed=1: 50 → 98 ctrl steps
- C1 vx=0.25 seed=1: 18 → 200 ctrl steps

C2 progress over the three iterations:

| metric | round 1 | round 2 | round 3 |
|---|---|---|---|
| C2 rows surviving 200/200 | 1/12 | 3/12 | 1/12* |
| C2 rows surviving ≥ C1 budget (64) | 9/12 | 8/12 | **12/12** |
| C2 roll_PD ≥ 25 % hard-fail rows | 7/12 | 1/12 | 3/12 |
| C2 pitch_PD ≥ 25 % hard-fail rows | 1/12 | 6/12 | 4/12 |
| C2 any-joint sat ≥ 25 % hard-fail rows | 5/12 | 1/12 | 1/12 |
| best realized/cmd forward ratio | +0.18 | +0.35 | +0.30 |

\*round-3 200/200 count is lower than round 2 because the split
also widened the "survive a long time" bin: many rows in round 3
land in the 100-190 range instead of either 200 or <100.  The
**all C2 rows now clear the C1 survival budget** is the more
useful metric: the harness consistently keeps the body upright
for 1.5+ gait cycles even at the worst seed.

C2 verdict (per the decision rule): **C1 pass + C2 fail-on-all-
bins → stay in C; investigate prior quality (lateral asymmetry,
knee feasibility) — stabilizer cannot rescue what is not
rescuable.**

### Where the propulsion deficit comes from

Three iterations of incremental prior fixes have eliminated the
lateral asymmetry, the high-vx face-plants, and most of the
roll_PD saturation.  C2 is still 0/12 because **the prior
fundamentally cannot generate net forward momentum from a
zero-vx start**, even with the IC vx alignment helping:

- Body is initialized at the standing keyframe with vx = 0.
- The standing-to-walking ramp settles the body into the first
  walking q_ref frame (50 ctrl steps = 1 s).
- After the ramp, an LIPM-aligned vx is injected.
- But the body is now already a full second past t=0 of the
  prior, and the q_ref pattern from t=0 expects a body at
  `(x = -0.024 m, vx = +0.176 m/s)` — not at the post-ramp
  state.  This residual mismatch consumes the harness's pitch
  budget before any forward stride accumulates.

The cleanest remaining fix is **item 2 of the round-1 queue**:
modify the prior so cycle 0 has a smooth from-standing startup
transient inside the q_ref itself (mirrors ToddlerBot's truncated
first-cycle warmup, baked into `_generate_at_step_length` rather
than into the validation harness).  This is a substantial change
to the prior generator and was not attempted in round 3 — round
3 confirmed that the cheaper IC-split fix lifts C1 fully but
cannot close the C2 propulsion gap on its own.

### Decision point for next iteration

Two viable paths from here:

A. **Implement the from-standing startup transient in the prior**
   (the queued long-term fix).  Rough scope: smooth quintic blend
   of COM trajectory in cycle 0 from (x=0, vx=0) to the LIPM
   steady-state, plus consistent leg-pose interpolation.  This is
   the principled fix and aligns with how ToddlerBot's library
   generator handles startup.  Estimated work: half a day to a
   day, plus regression testing on the kinematic / fixed-base
   gates.

B. **Treat C1 as the v0.20.0-C signoff and move to v0.20.1 with
   PPO carrying the closed-loop work.**  This is closer to
   ToddlerBot's actual practice — they only validate kinematically
   and let PPO handle the rest.  C2 stays in the codebase as a
   future-PPO regression check and as a tool for quantifying
   "how far PPO has to lift" relative to the bounded baseline.
   Risk: gives up the explicit "prior is rescuable by minor
   correction" gate, which was the whole reason the C1+C2 split
   was added.

The choice is a project-level call about how strictly to enforce
the "minor correction" PPO architecture before spending more time
on prior generation.

### v0.20.0-C closeout sweep round 4 — reviewer R1+R2+R3 + true from-rest cycle 0

External review (round 4) flagged three issues that all landed in
this commit:

- **Critical (R1)**: `_generate_at_step_length` documented "cycle 0
  starts from rest" but used the LIPM steady-state IC `(x0_ss,
  vx0_ss)` for *every* cycle including k=0.  Probe at vx=0.15
  showed the stored prior at t=0 had `pelvis_pos[0] = (-0.024,
  +0.013, +0.474)` with `vx ≈ +0.171 m/s` — root cause of the C2
  startup mismatch the previous three rounds were chasing.
  **Fix**: implemented a quintic blend in cycle 0 that actually
  starts at `(x=0, vx=0, ax=0)` and smoothly reaches the cycle-1
  LIPM IC at the cycle boundary
  (`x = 2·sl + x0_ss, vx = vx0_ss, ax = w² · x0_ss`).  Lateral
  COM in cycle 0 uses the same quintic ramp factor scaling the
  cosine pattern, so it starts at `y=0` (matches standing) and
  joins cycle 1's lateral cosine at `+lat_amplitude`.  Cycles
  k>=1 unchanged.  Verified by probe: `pelvis_pos[0] = (0.000,
  0.000, 0.474)`, `vx@0 ≈ 0.000`, `pelvis_pos[T_cycle] ≈
  (2·sl+x0_ss, …)`, `vx@T_cycle ≈ +0.171` (matches LIPM IC).
- **High (R2)**: `tools/v0200c_closeout.py` ignored subprocess
  return codes and hardcoded `mjpython` — on a machine without
  mjpython the runner silently scored 4/32 false passes.
  **Fix**: `_shell` raises `SubprocessFailure` on non-zero rc
  and the runner aborts immediately on a failed row.  Added
  `_detect_runner` + `V0200C_RUNNER` env var so the runner
  falls back to `uv run python` when mjpython is missing.
- **Medium (R3)**: clip-saturation `any` metric was reported as
  `max(channel_counts) / n` instead of the contract's "% of
  steps where any harness output is pinned" (the union of
  per-channel clipped step indices).  Closeout scored
  per-channel hard fails instead of the contract's aggregate.
  **Fix**: `c2_stabilizer` tracks a per-step `_clipped_any`
  counter (union); viewer reports `any (gate)` line; closeout
  scores against the aggregate `any` per
  `reference_design.md`'s C2 metric gate.

Also reverted the round-2 / round-3 IC vx centering on the LIPM
steady-state.  With cycle 0 actually from rest, the standing
keyframe `vx = 0` matches the prior's `vx@t=0`, so the LIPM-IC
injection becomes the OLD bug.  `_apply_ic_vx` now centers on
zero per the original Q2 wording.

Re-running the 32-row matrix with the hardened runner:

| mode | rows | passed | failed | gate verdict |
|---|---|---|---|---|
| kinematic     |  4 |  4 |  0 | **PASS** |
| fixed-base    |  4 |  4 |  0 | **PASS** |
| C1            | 12 | 10 |  2 | **FAIL** (vx=0.25 seed=0,2 at 59,60 ctrl steps — just under 64-step budget) |
| C2            | 12 |  0 | 12 | **FAIL** (no row clears closed-loop step gate) |

C2 progress (true from-rest + correct aggregate metric):

| metric | round 3 | round 4 |
|---|---|---|
| C2 rows surviving 200/200 | 1/12 | **4/12** |
| C2 rows with `any_clip < 25 %` (contract gate) | n/a (mis-measured) | **6/12** |
| C2 best `any_clip` | n/a | 18.0 % (vx=0.15 s=2; vx=0.25 s=2 at 20.5 %) |
| best realized/cmd forward ratio | +0.30 | -0.08 |

The 4 C2 200-survival rows (vx=0.15 s=0,2; vx=0.25 s=0,2) all
have aggregate `any_clip` between 18 % and 21 % — i.e. **the
harness is comfortably inside its budget**.  The remaining
failure on those rows is the closed-loop step length / ratio:
realized step is small negative (−0.006 to −0.023 m).  The
harness is no longer the bottleneck on those rows; the prior
is.

C1 regression at vx=0.25: cycle 0 from-rest is harder than the
old "LIPM-IC at standing pose" pattern (the legs have to do real
work in cycle 0 instead of inheriting forward momentum).  Two
seeds at vx=0.25 fall at 59 / 60 ctrl steps — 5 steps under the
2·cycle_time budget.  Borderline rather than catastrophic.

### Round 5 — geometry calibration probe (reverted)

Investigated whether the residual backward drift came from a
geometry mismatch between the IK and the MJCF: IK assumed
`com_height_m = 0.473` (derived from
`upper_leg + lower_leg + ankle_to_ground` with knee margin) but
MJCF standing keyframe pelvis_z is 0.468 m and the actual
straight-leg pelvis-to-foot drop is 0.406 m (vs the IK's
implicit 0.42).  Tried calibrating `com_height_m = 0.459` so
q_ref[0]'s foot bottom matches the standing keyframe foot z.

**Result: catastrophic regression.**  `safe_max_step_length_m`
re-derives the joint-limit budget from the same `com_height_m`,
so a smaller value tightens the available step length: kinematic
stride capped at 3.4 cm at all vx (ratio 0.43 at vx=0.25), C1
fell to ~60 ctrl steps everywhere, C2 fell to ~90.  **Reverted.**
Added a docstring comment under `com_height_m` explaining the
coupling so the next attempt at calibration also fixes
`safe_max_step_length_m` (or moves the standing keyframe to
match the IK's assumption).

### Round-4 verdict and where the closeout sits

Per the decision rule:

- **C1 fails** (10/12 rows) → "fix prior generation; do NOT
  spend effort on the stabilizer first."  Two cheap candidate
  fixes for the vx=0.25 borderline:
  1. Lengthen the survival budget at high vx_cmd (e.g.
     `2.0 → 2.5 × cycle_time` for the worst seed) — borderline
     justifiable as a contract refinement
  2. Soften the cycle-0 quintic so the legs have a gentler
     rise out of standing (e.g., extend cycle-0 duration by
     0.5×cycle_time before the first stance switch)
- **C2 fails** on closed-loop step gate, but harness budget is
  no longer the bottleneck on the best 4/12 rows.  The remaining
  prior-quality issue is the IK/MJCF geometry mismatch
  documented in round 5: a coordinated fix that adjusts both
  `com_height_m` and `safe_max_step_length_m` (or re-targets
  the MJCF) is the principled next step.

The earlier two-path decision (continue prior iteration vs treat
C1 as signoff) is unchanged in spirit.  Round 4's true from-rest
fix and the round-5 calibration probe sharpened our understanding
of where the gap is: it's a **prior geometry calibration**
problem now, not a "stabilizer needs more authority" problem.

### Round 6 — servo-chain probe + standing keyframe asymmetry fix

Per user request: before committing more time to a geometry
refactor, verify the servo / actuator chain end-to-end (the
v0.19 standing branch hit a "ctrl ordering" bug that wasted
weeks).

**Servo chain verified clean (no bug)**:
- `CtrlOrderMapper._perm_np` correctly maps each policy index
  to the matching MuJoCo actuator (probed by name, all 8 leg
  joints align: policy[0] = `left_hip_pitch` → mj[0] =
  `left_hip_pitch`, etc.).
- `set_all_ctrl(d, ctrl)` writes `ctrl[i]` to `data.ctrl[perm[i]]`;
  read-back via `mj_qpos[perm]` returns `policy_qpos[i]` as the
  qpos of the joint linked to policy actuator `i`.
- Sign convention probed: `ctrl[L_hip_pitch] = +0.5` →
  `L_foot_x = +0.068 m` (forward) after 50 PD steps with pelvis
  pinned, matching the WildRobot convention used in the IK.
- IK ↔ FK alignment probed: across 6 frames of the prior, the
  FK foot world x matches the IK intended foot world x within
  0–9 mm.  Kinematic and fixed-base modes still pass cleanly.

**Real bug found**: standing keyframe contained CAD-export noise
(R_hip_roll = -0.0021 vs L = +0.00005, pelvis_y off by 0.5 mm,
pelvis quat with 0.0009-rad forward pitch) AND the model has
intrinsic mass asymmetry (waist body inertial pos = -0.00188 m
in x → total CoM sits 6 mm AHEAD of mid-feet).  Combined, these
created a sustained backward force: even with `ctrl = qpos`
(zero PD error) the pelvis accelerated backward at ~0.5 m/s².
After 100 physics steps with no walking ctrl: vx = -0.046 m/s.
That bias contaminated every C2 free-floating evidence row in
rounds 1–4.

**Fix**: replaced `assets/v2/keyframes.xml`'s home key with the
settled equilibrium of a clean symmetric pose pitched -0.020 rad
to compensate the inertial offset.  Verified: 1000 physics
steps with `ctrl = qpos` → pelvis drift -0.002 m, |vx| ≤ 0.0001
m/s, z stable at 0.4684 m (vs old keyframe vx -0.046 m/s after
100 steps).  ~100× reduction in baseline backward drift.

Re-running the matrix (round 6, with cleaned keyframe + true
from-rest cycle 0 + correct aggregate clip metric):

| mode | rows | passed | failed | gate verdict |
|---|---|---|---|---|
| kinematic     |  4 |  4 |  0 | **PASS** |
| fixed-base    |  4 |  4 |  0 | **PASS** |
| C1            | 12 | 10 |  2 | **FAIL** (vx=0.25 seed=0,2 still at 59,60 steps — cycle-0 demand) |
| C2            | 12 |  0 | 12 | **FAIL** (closed-loop step gate) |

C2 survival markedly better than round 4 across most seeds
(notably the previously-flaky seed=1 rows now reach 115-125
ctrl steps), but the body **still drifts backward in walking**.
Best forward step ratio is -0.04 (vx=0.25 seed=0 R side) —
better than round 4's worst, still not at the gate.

### Where the residual backward drift is NOT (ruled out by round 6)

- **Servo chain**: name-checked end-to-end, sign convention
  verified by FK probe.  Not the bug.
- **Standing keyframe asymmetry**: was a real ~0.5 m/s² baseline
  bias, now reduced 100× by the clean keyframe.  Not the
  remaining bug.
- **From-rest cycle 0**: round 4's quintic blend confirmed the
  prior actually starts at `(x=0, vx=0)`.  Not the remaining
  bug.

### Where the residual backward drift might be (ranked by evidence)

1. **IK ↔ MJCF leg-geometry mismatch**: probed in round 5 (the
   IK assumes leg reach 0.42 m, MJCF straight-leg drop is 0.406
   m → q_ref puts feet ~7 cm above ground in IK frame; bent-knee
   q_ref still puts feet ~1 cm above ground).  Round-5 attempt
   to lower `com_height_m` broke `safe_max_step_length_m`
   coupling — the proper fix is to recalibrate
   `upper_leg_m + lower_leg_m + ankle_to_ground_m` together
   from MJCF FK probes, then re-derive
   `safe_max_step_length_m` from the new geometry.  This is the
   most likely remaining root cause.
2. **Knee-pitch servo lag**: worst-joint RMSE 0.34 rad with
   max_abs 0.94 rad (~54°) across all modes.  The knee can't
   bend deep enough during swing → foot drag → backward force
   on stance.  Round 6 saw small improvement on knee MAE but
   max_abs still ~0.95 rad.  Tightly coupled to the geometry
   issue (deeper knee bend would be needed if the IK assumed
   shorter legs).
3. **PD gain headroom**: kp=21.1, force-limit ±4 N·m on the
   htd45hServo class.  At the demanded swing accelerations the
   force-limit might cap; not directly probed in this round.

### Honest verdict at end of round 6

The user-suggested debug path eliminated the most plausible
"easy" cause (servo bug) and uncovered + fixed a real but
distinct issue (standing keyframe asymmetry).  The harness is
not the bottleneck on the best 4 rows (any_clip 28-36% — over
the 25% gate but not by much).  The remaining bottleneck is
prior-geometry coupled with knee servo capability, exactly
where round 5 pointed.  A coordinated geometry refit is the
right next step but is not a one-line change — see CHANGELOG
queue at top.

### Files changed in round 6

- `assets/v2/keyframes.xml`: replaced home key with the clean
  equilibrium pose (pelvis pitched -0.020 rad, all single-dof
  joints near 0 except knees ≈ 0.023, settled qpos baked in).

---

## [v0.19.4c] - 2026-04-14: Reference propulsion rework

### Problem
v0.19.4-B achieved nominal walking (461 steps, 15 gait cycles) but only
delivered 25% of commanded forward speed (+0.038 m/s at cmd=0.15) and
30-50% of commanded step amplitude (1-3cm actual vs 5-6cm commanded).
Premature handoff to PPO caused three rounds of reward surgery (v0.19.5,
v0.19.5b, v0.19.5c) that found lean-back and move-less exploits.

Root cause diagnosis (after first run):
- The support gate penalised double-support in SUPPORT_STABILIZE mode,
  collapsing `progression_permission` to zero and preventing real stepping
- But `com_x_planned` was added to the stance foot independently of
  permission, so the torso pitched forward without a recovery step
- Result: robot stood/shuffled while accumulating forward pitch → termination
- Initial propulsion changes (LIPM cosh/sinh, large pelvis pitch, ankle
  push-off) amplified this mismatch rather than fixing it

### Changes

Support gate fix (walking_ref_v2.py):
- **Remove double-support penalty in SUPPORT_STABILIZE**: the contact
  penalty (0.25 instability) is now exempt during both STARTUP and
  SUPPORT_STABILIZE modes, not just STARTUP.  This unblocks
  `progression_permission` for real stepping.
- **Couple COM drive to progression**: `com_x_planned` is now scaled by
  `progression_permission`, so forward torso drive is reduced when stepping
  is suppressed.  Prevents pitch-without-step failure mode.

Config tuning (ppo_walking_v0194c.yaml):
- **Reduced support_health_gain**: 10.0 → 5.0 (less aggressive health
  collapse under mild instability)
- **Raised foothold min scale**: 0.15 → 0.30 (always take a visible step)
- **Raised phase min scale**: 0.05 → 0.15 (phase doesn't freeze)
- **Added swing progress min scale**: 0.00 → 0.10 (swing always progresses)
- **Modest pelvis pitch**: gain 0.015 → 0.03, max 0.02 → 0.04 rad
- **Longer step time**: 0.50 → 0.60s
- **Lower support_release_phase_start**: 0.25 → 0.20

New propulsion capabilities (available but NOT enabled in shipped config):
- **LIPM COM trajectory mode**: `com_trajectory_mode: lipm` (cosh/sinh).
  Available for future use; default remains `linear`.
- **Ankle push-off**: `ankle_pushoff_enabled: true` with configurable
  phase start and max plantarflexion.  Disabled by default.

Diagnostics:
- **Multi-seed handoff gate evaluator**: `training/eval/eval_handoff_gate.py`
- **Raw nominal metrics**: added `debug/loc_ref_foothold_x_raw` and
  `debug/loc_ref_nominal_step_length` for pre-scaling foothold measurement
- **Extended probe traces**: root_roll, lateral_velocity, step_length_m
- **Gate instrumentation fix**: handoff gate uses raw foothold metric,
  not the support-gated swing target

### Quantitative Handoff Gate (must pass before PPO)
1. Multi-seed: ≥80% seeds survive ≥400/500, no seed <300, mean ≥430
2. Forward velocity: ≥0.075 m/s (50% of cmd=0.15)
3. Step generation: commanded ≥0.045m, realized ≥0.03m or ratio ≥0.5
4. Stability: pitch p95 ≤0.20, roll p95 ≤0.12, term fracs ≤0.20
5. Gait quality: stance switches on all passing seeds
6. Lateral drift: direction varies across seeds

### Files Changed
- `control/references/walking_ref_v2.py` — remove double-support penalty in
  SUPPORT_STABILIZE, couple COM drive to progression_permission, add LIPM
  mode and ankle push-off capabilities
- `training/envs/wildrobot_env.py` — wire new config fields, ankle push-off
  through IK adapter, add raw nominal foothold/step-length metrics
- `training/core/metrics_registry.py` — register raw nominal metrics
- `training/eval/eval_handoff_gate.py` — NEW: multi-seed gate evaluator
- `training/eval/eval_loc_ref_probe.py` — add trace fields, raw metric summaries
- `training/configs/training_runtime_config.py` — new EnvConfig fields
- `training/configs/training_config.py` — loader wiring for new fields
- `training/configs/ppo_walking_v0194c.yaml` — NEW: support-gate-fixed config

### Verification
```bash
# Diagnostics: multi-seed nominal probe
JAX_PLATFORMS=cpu uv run python training/eval/eval_handoff_gate.py \
  --config training/configs/ppo_walking_v0194c.yaml \
  --seeds 5 --forward-cmd 0.15 --horizon 500

# Single-seed nominal viewer
JAX_PLATFORMS=cpu uv run mjpython training/eval/visualize_nominal_ref.py \
  --config training/configs/ppo_walking_v0194c.yaml \
  --forward-cmd 0.15 --horizon 500 --print-every 20 --log --record --headless
```

### Training Result (PPO Attempt, Premature)

Run analyzed:
- Offline run: `training/wandb/offline-run-20260414_055804-fam8kwkt` (W&B tags: `v0.19.5`, started `2026-04-14`)
- Checkpoints: `training/checkpoints/ppo_walking_v0195_v00195_20260414_060040-fam8kwkt`

Metrics sanity (important):
- `eval_clean/*` and `eval_push/*` use `teacher_com_velocity_target_mean=0.0` in this run, so they are **standing/recovery** evals, not forward-walking command tracking. Do not pick a “walking” checkpoint based on `eval_clean/success_rate` here.

Walking-rollout summary (iters 10..680, logged cmd is ~0.10 m/s):
- Mean `env/velocity_cmd`: `0.099 m/s`
- Mean `env/forward_velocity`: `0.046 m/s` (about `46%` of cmd)
- Mean `tracking/cmd_vs_achieved_forward`: `0.153 m/s` (fails the `<= 0.075` handoff criterion)
- Mean `term_pitch_frac`: `66%` (dominant termination)
- Mean `debug/step_length_m`: `0.0034 m` (note: diluted by non-touchdown steps; touchdown-only metric not logged here)

Regime split (why this run is not deployable for walking):
- “Fast” phases (e.g. iter 330): `env/forward_velocity=0.232` but `term_pitch_frac=99.8%` and `env/episode_length=66` (forward-dive / pitch-fall).
- “Stable” phases (e.g. iter 680): `env/episode_length=487`, `env/success_rate=92%`, `term_pitch_frac=7.7%` but `env/forward_velocity=0.011` (standing / low-motion basin).

Verdict:
- This PPO run does not demonstrate residual improvements on top of a nominal reference that already owns locomotion, so it does not satisfy the `v0.19.4-C`→`v0.19.5` precondition. Do not treat `checkpoint_680_89128960.pkl` as a walking checkpoint.

### Next Step
If handoff gate passes: resume v0.19.5 PPO with propulsion rewards enabled.
If gate fails after all 5 interventions: escalate per v0.19.5-D fallback plan.

---

## [v0.19.5c] - 2026-04-13: Reward rebalance — propulsion-first

### Problem
v0.19.5 first run found a low-motion exploit: the policy learned to reduce
slip (weighted slip improved +1.39/step from iter 180 to 280) while losing
only -0.06 on forward tracking.  The dominant exploit channel was slip
at -0.15 weight, not alive alone.  Backward lean was a symptom of the
broader "move less = more reward" basin.

### Root Cause Analysis
- `slip: -0.15` was the largest marginal reward improvement for the bad policy
- `alive: 0.5` provided unconditional survival bonus exceeding standing penalty
- `tracking_lin_vel: 4.0` was too weak relative to slip/alive
- Task contract mismatch: `min_velocity: 0.0` + `loc_ref_min_step_length_m: 0.03`
  sent contradictory signals (stop command + nonzero nominal step)
- Existing locomotion rewards (`dense_progress`, `step_progress`) were disabled

### Changes
Task contract:
- `min_velocity`: 0.0 → 0.08 (forward-walking only, no stop)
- `velocity_cmd_min`: 0.05 → 0.08 (align gate with min velocity)

Propulsion rewards (previously disabled, now enabled):
- `dense_progress`: 0.0 → 1.0 (key term that best separates good/bad policies)
- `step_progress`: 0.0 → 0.30
- `foot_place`: 0.0 → 0.10
- `step_event`: 0.0 → 0.02

Rebalanced weights:
- `tracking_lin_vel`: 4.0 → 6.0 (must dominate over low-motion incentives)
- `slip`: -0.15 → -0.05 (was dominant exploit channel)
- `alive`: 0.5 → 0.25 (reduce survival dominance)
- `velocity_standing_penalty`: 0.3 → 0.6
- `orientation`: -0.6 → -0.8
- `pitch_rate`: -0.2 → -0.25
- `base_height`: 0.05 → 0.03
- `height_target`: 0.10 → 0.05
- `m3_pelvis_height_tracking`: 0.20 → 0.15

Removed:
- `backward_lean`: -2.0 → 0.0 (proper rebalance addresses root cause)
- `negative_velocity`: -8.0 → 0.0 (same)

### Stopping Criteria
- By iter 40-60: abort if `term_pitch_frac > 0.8`
- By iter 60-80: require `env/forward_velocity > 0.03`
- By iter 80: require `reward/step_progress > 0` AND `reward/dense_progress > 0.08`
- Checkpoint selection: `eval_clean/success_rate > 0.4` AND
  `env/forward_velocity > 0.05` AND `reward/step_progress > 0`

### Config
- `training/configs/ppo_walking_v0195c.yaml`

---

## [v0.19.5b] - 2026-04-13: Fix backward-lean reward exploit

### Problem
First v0.19.5 PPO run (280 iterations) found a reward exploit:
- PPO learned to lean backward (pitch: +0.025 → -0.142 rad)
- Forward velocity collapsed (+0.038 → -0.040 m/s, walking backward)
- Eval success peaked at 73.7% (iter 180) then collapsed to 0% (iter 280)
- 99% of episodes terminated from pitch (backward falls)

Root cause: `alive` reward at +0.5/step dominated all other signals.
Symmetric `orientation` penalty (pitch²) didn't distinguish backward lean.
Net reward for "lean back and stand" was positive.

### Changes
- `alive`: 0.5 → 0.2 (reduce survival dominance)
- `velocity_standing_penalty`: 0.3 → 0.8 (make standing/backward costly)
- NEW `backward_lean`: -2.0 (asymmetric penalty on negative pitch)
- NEW `negative_velocity`: -8.0 (penalize backward walking when forward commanded)

### Files Changed
- `training/envs/wildrobot_env.py` — add backward_lean and negative_velocity rewards
- `training/configs/ppo_walking_v0195.yaml` — updated weights

### Training Results (first run, before fix)
| Iter | eval_success% | step_length_m | pitch | forward_vel |
|------|--------------|---------------|-------|-------------|
| 1 | 2.5 | 0.002 | +0.025 | +0.038 |
| 180 | 73.7 (peak) | 0.004 | -0.031 | +0.032 |
| 280 | 0.0 | 0.004 | -0.142 | -0.040 |

---

## [v0.19.4] - 2026-04-12: Nominal Walking Reference Achieved ✅

### Summary
First nominal walking reference that produces sustained forward walking
on WildRobot v2.  The robot walks 461 steps (9.2 seconds) at cmd=0.15 m/s
with 15 full gait cycles and 30 stance switches before lateral roll
termination.  Includes M2.5 reference validation, M3.0-A COM trajectory,
and M3.0-B nominal walking probe.

### Key Milestones Resolved

**Ctrl ordering fix (root cause of ALL v0.19.3+ failures):**
- PolicySpec and MuJoCo listed actuators in different orders
- The env wrote PolicySpec-ordered ctrl to MuJoCo's data.ctrl, sending
  joint targets to wrong actuators (e.g. knee bend → hip pitch)
- Fixed via `CtrlOrderMapper` (`training/utils/ctrl_order.py`)
- Regression test: `training/tests/test_actuator_ordering.py`
- Architecture doc: `docs/system_architecture.md`

**DCM COM trajectory for stance-leg forward drive:**
- Phase-proportional COM trajectory shifts stance foot IK target forward
  during walking modes, producing natural hip extension and forward drive
- Linear ramp (not full LIPM cosh/sinh — too aggressive for open-loop)
- Config: `com_trajectory_enabled: true`

**Two-margin support posture system:**
- Support mode: height=0.42, margin=0.020 → knee 36° (more leg reach)
- Walking mode: height=0.42, crouch=0.045 → knee 53° (lower COM, stability)
- The deeper walking posture braces for dynamic phases

**Episode starts from support posture B:**
- Pre-computed via IK pipeline with `debug_force_support_only=True`
- Walking reference starts in SUPPORT_STABILIZE mode (skip startup)
- Config: `start_from_support_posture: true`

### Nominal Walking Probe Results (cmd=0.15, seed=0)
- Survived: 461/500 steps (9.2 seconds)
- Forward speed mean: +0.038 m/s
- Pitch oscillation: ±0.17 rad (10°)
- Gait cycles: ~15 complete cycles, 30 stance switches
- Termination: term/roll (lateral drift accumulation)

### Known Quality Issues (PPO scope)
- **Wobble** (±10° pitch oscillation) — open-loop, no active balance
- **Lateral drift** — per-step asymmetries accumulate → roll termination
- **Small steps** (~1-3cm actual) — servo tracking limits foot displacement
- These are the intended scope for PPO residual corrections

### Config
- Config: `training/configs/ppo_walking_v0193a.yaml`
- Key values: `walking_pelvis_height=0.42, support_margin=0.020,
  walking_crouch=0.045, swing_target_blend=1.0, com_trajectory_enabled=true`

### Files Changed
- `control/references/walking_ref_v2.py` — COM trajectory, rename height field
- `training/envs/wildrobot_env.py` — ctrl ordering, B-start, two-margin IK,
  stance randomization, pending_action init, walking_crouch
- `training/utils/ctrl_order.py` — NEW: CtrlOrderMapper
- `training/tests/test_actuator_ordering.py` — NEW: 5 regression tests
- `docs/system_architecture.md` — ctrl ordering principle (merged)
- `training/eval/visualize_nominal_ref.py` — display sync, --from-keyframe,
  --record auto-generate, ctrl diagnostics fix

### Next Step
v0.19.5: Walking PPO v2 — train residual policy on top of the walking
reference to improve balance (wobble), lateral stability (drift), and
foot tracking (step size).

---

## [v0.17.4t] - 2026-03-26: Bounded Step-Target Teacher Result ✅

### Summary
`v0.17.4t` is the first completed bounded teacher-assisted standing branch.
This branch kept the deployed artifact as the same single policy and added:
- disturbed-window-only teacher activation
- heuristic step-target generation
- first-touchdown `teacher_target_step_xy` reward shaping
- teacher diagnostics for reachability, activation, and agreement

The internal `eval_push` curve plateaued and regressed late, but the best
checkpoint still cleared the real fixed-ladder hard gate. This branch therefore
succeeded, and stronger teacher complexity is not the next move.

### Training Run
- Run: `training/wandb/run-20260324_225425-zn8r3l5g`
- Config: `training/configs/ppo_standing_v0174t.yaml`
- Checkpoints: `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g`

### Best Checkpoint Selection
The clean-standing eval saturates at `100%`, so the standing objective should
still be selected by `eval_push/*`, not `eval_clean/*`.

Best push-eval checkpoint:
- `training/checkpoints/ppo_standing_v0174t_v0174t_20260324_225428-zn8r3l5g/checkpoint_330_43253760.pkl`

Metrics at iter `330`:
- `eval_push/success_rate = 93.75%`
- `eval_push/episode_length = 478.9`
- `eval_push/term_height_low_frac = 6.25%`
- `eval_push/term_pitch_frac = 50.0%`
- `env/success_rate = 83.1%`
- `debug/torque_sat_frac = 6.5%`

Late-run note:
- the clean-saturated final checkpoint at iter `510` is **not** the best
  standing checkpoint
- internal push performance regressed after the iter `330` peak

### Fixed Ladder Result
Measured on `checkpoint_330_43253760.pkl` using the external standing ladder:
- `eval_medium = 93.8%`
- `eval_hard = 70.6%`
- `eval_hard/term_height_low_frac = 29.4%`

Comparison:
- `v0.17.3a`: `eval_medium = 83.1%`, `eval_hard = 51.4%`
- `v0.17.3a-pitchfix`: `eval_medium = 85.4%`, `eval_hard = 51.8%`
- `v0.17.3a-arrest`: `eval_medium = 83.1%`, `eval_hard = 52.2%`
- `v0.17.4t`: `eval_medium = 93.8%`, `eval_hard = 70.6%`

### What Worked
- Teacher stayed bounded to disturbed windows:
  - `teacher/active_frac ≈ 0.12`
  - `teacher/teacher_active_during_push_frac ≈ 1.0`
  - `teacher/teacher_active_during_clean_frac = 0.0`
- Teacher targets were mostly reachable:
  - `teacher/reachable_frac ≈ 0.92`
- Clean standing remained intact:
  - `eval_clean/success_rate = 100%`
- The branch materially improved the true hard-push gate:
  - `eval_hard` improved from the `52.2%` pure-RL plateau to `70.6%`

### Verdict
`v0.17.4t` clears the standing hard gate.

Decision:
- freeze `checkpoint_330_43253760.pkl` as the winning standing-recovery checkpoint
- stop teacher escalation for now
- move next to `v0.17.3b` sim2real hardening and deployment transfer work
- then continue to the walking-on-the-same-policy-stack path

### Main Lesson
The repo now has a complete answer for the standing branch:
- reward-only pure-RL tuning plateaued below the hard gate
- bounded step-target teacher support broke through that plateau
- the deployed controller can stay a single policy

---

## [v0.17.3a-arrest] - 2026-03-24: Recovery-Gated Arrest Rewards Result ⚠️

### Summary
`v0.17.3a-arrest` completed the last planned reward-only standing branch before teacher support. This branch kept the `v0.17.3a` standing recipe and mild `pitchfix` shaping, then added:
- recovery-window gating for step rewards
- explicit post-touchdown arrest rewards
- short-horizon post-touchdown survival reward

The mechanism worked, but the outcome did not break through the hard-push gate.

### Training Run
- Run: `training/wandb/run-20260323_234412-ke6ts9kz`
- Config: `training/configs/ppo_standing_v0173a_arrest.yaml`
- Checkpoints: `training/checkpoints/ppo_standing_v0173a_arrest_v0173a-arrest_20260323_234415-ke6ts9kz`

### Best Internal Push-Eval Checkpoint
The clean-standing eval saturates at `100%`, so the standing objective should be selected by `eval_push/*`, not `eval_clean/*`.

Best push-eval checkpoint:
- `training/checkpoints/ppo_standing_v0173a_arrest_v0173a-arrest_20260323_234415-ke6ts9kz/checkpoint_300_39321600.pkl`

Metrics at iter `300`:
- `eval_push/success_rate = 93.75%`
- `eval_push/episode_length = 478.7`
- `eval_push/term_height_low_frac = 6.25%`
- `eval_push/term_pitch_frac = 46.9%`
- `env/success_rate = 87.6%`
- `debug/torque_sat_frac = 5.5%`

### Fixed Ladder Result
Measured on `checkpoint_300_39321600.pkl` using the external standing ladder:
- `eval_medium = 83.1%`
- `eval_hard = 52.2%`
- `eval_hard/term_height_low_frac = 47.8%`

Comparison:
- `v0.17.3a`: `eval_medium = 83.1%`, `eval_hard = 51.4%`
- `v0.17.3a-pitchfix`: `eval_medium = 85.4%`, `eval_hard = 51.8%`
- `v0.17.3a-arrest`: `eval_medium = 83.1%`, `eval_hard = 52.2%`

### What Worked
- Arrest rewards were active and stable:
  - `reward/arrest_pitch_rate` non-zero
  - `reward/arrest_capture_error` non-zero
  - `reward/post_touchdown_survival ≈ 0.10`
- Recovery gating was genuinely engaged:
  - `debug/recovery_step_gate ≈ 0.13`
  - `debug/post_touchdown_arrest_gate ≈ 0.10`
- `recovery/no_touchdown_frac` was effectively zero at the best point, which means the policy is usually stepping when disturbed.

### Verdict
`v0.17.3a-arrest` confirms that the remaining gap is **not** “the robot never steps.” The policy steps, and the arrest shaping is active, but the step quality still does not improve enough to clear the hard gate.

Decision:
- stop reward-only tuning for this standing branch
- keep `v0.17.3b` deferred; this is still a sim-competence miss, not a transfer miss
- proceed to bounded teacher support in `v0.17.4t`

### Main Lesson
The repo now has evidence for all three pure-RL standing branches:
- `v0.17.3a` improved the baseline materially
- `v0.17.3a-pitchfix` slightly improved hard-push results
- `v0.17.3a-arrest` activated the intended recovery mechanism but still plateaued below the gate

The next justified lever is **bounded step-target teacher support**, not more local reward shaping.

---

## [v0.17.3a-pitchfix] - 2026-03-23: Cheap Sim-Side Pitch-Arrest Probe 🧪

### Summary
Added a minimal follow-up config on top of the best `v0.17.3a` policy recipe before escalating to teacher support. This probe does **not** change the policy contract, observation layout, action mapping, or termination boundary. It only adds small reward-shaping terms intended to reduce unrecoverable pitch momentum after hard pushes in simulation.

### Why This Exists
`v0.17.3a` materially improved the standing policy and beat `v0.17.1` on the fixed ladder, but it still missed the hard gate:
- best checked checkpoint: `checkpoint_210_27525120.pkl`
- `eval_medium = 83.1%`
- `eval_hard = 51.4%`

The remaining gap is still a **simulation competence** gap. `v0.17.3b` remains sim2real hardening and is therefore not the next explanation for the fixed-ladder miss. Before adding teacher support, this branch tries one cheap pure-RL intervention aimed at pitch arrest.

### What Changed
- Added new config: `training/configs/ppo_standing_v0173a_pitchfix.yaml`
- Preserved all successful `v0.17.3a` recipe changes:
  - `action_mapping_id: pos_target_home_v1`
  - `use_relaxed_termination: true`
  - `alive: 15.0`
  - `min_height: 0.15`
  - no domain randomization
  - no action delay
  - no teacher / runtime hierarchy
- Added mild sim-side shaping terms:
  - `pitch_rate: -0.2`
  - `collapse_height: -0.05`
  - `collapse_vz: -0.1`
- Added a `quick_verify` block so `--verify` runs as a real smoke test.

### Training Plan
- Start from the last good `v0.17.3a` checkpoint:
  - `training/checkpoints/ppo_standing_v0173a_v0173a_20260323_011340-0gmualhi/checkpoint_210_27525120.pkl`
- Use resume training as a **screening** run because the policy contract is unchanged.
- If this run materially improves `eval_hard`, rerun from scratch to confirm the gain is not checkpoint luck.
- If it does not move the fixed-ladder result enough, proceed to bounded teacher support in `v0.17.4t`.

### Status
- Config landed
- Config load verified
- Training complete
- Best checked checkpoint:
  - `training/checkpoints/ppo_standing_v0173a_pitchfix_v0173a-pitchfix_20260323_175304-8v4o0hfn/checkpoint_280_36700160.pkl`
- Fixed ladder:
  - `eval_medium = 85.4%`
  - `eval_hard = 51.8%`
- Verdict:
  - small improvement over `v0.17.3a`
  - not enough to clear the hard gate
  - justified one last reward-only branch (`v0.17.3a-arrest`) before teacher support

---

## [v0.17.3a] - 2026-03-23: Recipe-Fixed Pure RL Result ✅

### Summary
`v0.17.3a` is the completed pure-RL recipe-fix standing run. This branch tested the closest Open Duck / OP3-inspired standing recipe adjustments before any teacher or runtime hierarchy:
- home-centered residual action mapping
- relaxed termination
- dominant alive bonus

The result is a clear improvement over `v0.17.1`, but it still misses the true hard-push gate on the external fixed ladder.

### Training Run
- Run: `training/wandb/run-20260323_011338-0gmualhi`
- Config: `training/configs/ppo_standing_v0173a.yaml`
- Best deploy/eval candidate:
  - `training/checkpoints/ppo_standing_v0173a_v0173a_20260323_011340-0gmualhi/checkpoint_210_27525120.pkl`

### Internal Training / Eval Result
- `eval_clean/success_rate = 100%`
- best `eval_push/success_rate = 92.2%` (around iter `200`, nearest saved checkpoint iter `210`)
- final `eval_push/success_rate = 89.2%`
- final `env/success_rate = 82.0%`
- episode length improved substantially versus `v0.17.1`

### Fixed Ladder Result
Measured on `checkpoint_210_27525120.pkl` using the external standing ladder:
- `eval_medium = 83.1%`
- `eval_hard = 51.4%`
- `eval_hard/term_height_low_frac = 48.6%`

For comparison, `v0.17.1` fixed-ladder result was:
- `eval_medium = 71.6%`
- `eval_hard = 39.2%`
- `eval_hard/term_height_low_frac = 60.8%`

### Verdict
`v0.17.3a` is a **real pure-RL improvement**, but it does **not** clear the standing hard gate.

Interpretation:
- quiet standing is solved (`eval_clean = 100%`)
- the RL recipe changes substantially improved push recovery in simulation
- the dominant collapse mode shifted away from the old `term_height_low` dominance
- the remaining gap is still a **simulation competence** gap, not a transfer gap

Decision:
- freeze `v0.17.3a` as the current best pure-RL standing baseline
- do **not** treat `v0.17.3b` as the fix for this sim miss; `v0.17.3b` remains sim2real hardening
- try one cheap sim-side reward follow-up (`v0.17.3a-pitchfix`) before escalating to teacher support

### Why The Best Checkpoint Is Iter 210
- raw internal push eval peaked before the final iteration
- later checkpoints showed worse push success and higher torque stress
- iter `210` is the nearest saved checkpoint to the best internal eval point

### Main Lessons
- recipe alignment mattered; pure PPO was not exhausted in `v0.17.1`
- internal `eval_push/*` is directionally useful but still overstates true fixed-ladder performance
- remaining failures appear more pitch-related than the older straight-collapse pattern
- the next decision should continue to be made off the fixed ladder, not only internal probe metrics

---

## [v0.17.2] - 2026-03-22: Architecture-Pivot Groundwork (Scaffolding) 🧱

### Summary
Landed concrete repository scaffolding for the OCS2 / humanoid-MPC pivot without pretending a complete controller exists. This milestone establishes controller-path selection, model-compatibility interfaces, and compatibility-note placeholders so `v0.17.3+` can proceed without reopening architecture decisions.

### What Landed
- Added top-level `control/` package scaffolding aligned with repo architecture:
  - `control/interfaces.py` (planner/execution debug interface contracts)
  - `control/robot_model/adapter.py` (model artifact compatibility checks/report)
- Added WildRobot compatibility notes:
  - `assets/v2/FRAME_CONVENTIONS.md`
  - `assets/v2/CONTACT_GEOMETRY.md`
- Added config/runtime selection hook:
  - `env.controller_stack` (`ppo` default, `mpc_standing` for bring-up path)
  - v0.17.3 bring-up controller parameters wired through config parsing.

### Deliberately Deferred
- No full OCS2 optimizer integration.
- No URDF exporter implementation.
- No whole-body controller implementation.
- No standing performance claims from this milestone alone.

---

## [v0.17.3] - 2026-03-22: Architecture-Pivot Standing Bring-Up (Scaffold) 🚧

### Summary
Implemented the first truthful standing bring-up slice on top of the OCS2/humanoid-MPC pivot groundwork. This change adds a selectable `mpc_standing` controller path that initializes and runs in the existing MuJoCo/JAX environment, emits inspectable controller/planner debug signals, and preserves the existing PPO/eval infrastructure.

This is **not** a full MPC implementation and does **not** claim hard-push recovery or walking.

### What Landed
- **Controller-path plumbing**
  - Added `env.controller_stack` config selection (`ppo` default, `mpc_standing` bring-up path).
  - Kept legacy paths (`ppo`, `base_ctrl_enabled`, `fsm_enabled`) unchanged.
- **Minimal control scaffolding**
  - Added top-level `control/` modules and a conservative standing controller stub:
    - `control/interfaces.py`
    - `control/mpc/standing.py`
  - The stub computes upright stabilization action deltas in policy-action space and emits planner/controller debug signals.
- **Environment integration**
  - `training/envs/wildrobot_env.py` now routes to `mpc_standing` when selected.
  - Added debug metrics emission for planner active/controller active/targets/support/step request.
  - Extended `WildRobotInfo` state and auto-reset metric preservation for the new debug signals.
- **Config and smoke coverage**
  - Added `training/configs/mpc_standing_v0173.yaml`.
  - Added `tests/test_mpc_standing_v0173.py` covering config load + reset/step + debug metrics presence.

### Deliberately Deferred
- No switched-contact optimization loop.
- No true CoM/footstep horizon optimizer.
- No whole-body controller.
- No hard-push success claims.
- No walking behavior claims.

### Why This Scope
The goal for `v0.17.3` is honest bring-up: controller initialization + inspectability + compatibility with existing training/eval flow. This keeps implementation narrow, testable, and ready for the next milestone (moderate-push hardening, then fixed-ladder validation).

---

## [v0.17.4] - 2026-03-22: Pivot To OCS2 / Humanoid MPC Architecture 🔄

### Summary
After reviewing the completed `v0.17.1` standing baseline and the confidence requirements for a shared standing+walking solution, the `v0.17` mainline pivots away from PPO-only escalation and toward an OCS2 / humanoid MPC architecture. The new main references are:
- OCS2 as the optimal-control foundation
- 1X `wb-humanoid-mpc` as the concrete humanoid implementation reference
- OpenLoong only as a secondary MuJoCo implementation reference

This pivot changes the mainline plan from "improve pure-PPO standing recovery with bounded local controller patches" to "adopt a more established model-based humanoid control stack and validate standing push recovery first."

### Why The Mainline Changed
- `v0.17.1` failed the true hard-push gate:
  - `eval_medium = 71.6%`
  - `eval_hard = 39.2%`
  - dominant failure remained `term_height_low`
- The remaining gap is no longer best understood as a reward-tuning or PPO-stability issue
- The project now wants one architecture for both:
  - standing push recovery
  - nominal walking
- A model-based humanoid MPC stack is a higher-confidence direction for that goal than continued PPO-only escalation

### Mainline Decision
- **New mainline**: OCS2 / humanoid MPC adoption path
- **Primary concrete reference**: 1X `wb-humanoid-mpc`
- **Secondary simulation reference**: OpenLoong Dyn-Control
- **Standing-first validation remains**
- **Walking remains a required follow-on objective on the same stack**

### What This Means For Existing v0.17 Branches
- `v0.17.1` remains the final PPO standing baseline
- `v0.17.3` M3-lite remains a side branch / local experiment, not the mainline direction
- Li-style history and motion-prior investigations are also demoted to side branches unless the architecture pivot stalls

### Next Focus
1. Define the WildRobot adoption plan for OCS2 / humanoid MPC
2. Complete robot-model and control-stack gap analysis
3. Bring up quiet standing on the new stack
4. Re-run standing push recovery against the same fixed ladder
5. Extend the same stack to walking

---

## [v0.17.2] - 2026-03-21: Finalized Recovery Summary Metrics 🔧

### Summary
Reworked standing recovery diagnostics into finalized recovery-summary metrics instead of per-step proxy signals. Recovery bookkeeping now lives in env state, summary metrics are emitted once when a recovery window completes (or the episode terminates mid-window), and rollout aggregation normalizes those summaries by `recovery/completed`. This makes `recovery/first_step_latency`, `recovery/touchdown_count`, `recovery/support_foot_changes`, and `recovery/post_push_velocity` interpretable during training and eval.

### Problems Fixed
1. **first_step_latency**: No longer emitted as a single-step spike and then lost in rollout aggregation.
2. **touchdown_count**: No longer logged as a running per-step accumulator averaged across the rollout.
3. **support_foot_changes**: No longer duplicated touchdown density; now counts actual support-foot alternations inside the recovery window.
4. **post_push_velocity**: Now records the residual horizontal speed at recovery finalization instead of averaging across all recovery steps.
5. **Auto-reset preservation**: Finalized recovery summaries now survive the terminating step and remain visible to the trainer.

### Changes
- **training/envs/env_info.py**
  - Added persistent recovery state: `recovery_active`, `recovery_age`, `recovery_first_step_latency`, `recovery_touchdown_count`, `recovery_support_foot_changes`, `recovery_last_support_foot`
- **training/envs/wildrobot_env.py**
  - Moved recovery bookkeeping out of `_get_reward()` and into `step()`
  - Added finalized summary emission with a fixed 50-step recovery window
  - Added `recovery/completed`
  - Preserved finalized recovery metrics through auto-reset
- **training/core/metrics_registry.py**
  - Switched recovery metrics to summary-event `SUM` reducers
  - Added normalization by `recovery/completed`
- **training/core/experiment_tracking.py**
  - Added zero initialization for `recovery/completed`
- **CHANGELOG.md**
  - Clarified that the fixed eval ladder remains external to the training loop

### Testing
- `tests/test_recovery_metrics_aggregation.py` now validates finalized summary semantics directly:
  - no-touchdown recovery
  - single-touchdown latency
  - alternating vs repeated support foot touchdowns
  - early termination during active recovery
  - rollout normalization by `recovery/completed`
- `tests/test_v0171_boundary_push_eval.py` continues to validate config wiring and env-level metric presence

---

## [v0.17.1] - 2026-03-21: Recipe Baseline at the Boundary ✅ [BUG FIXES APPLIED]

### Summary
`v0.17.1` implements the recipe baseline milestone at the push boundary as defined in `training/docs/standing_training.md`. This milestone focuses on answering one key question: **Can the cleaned standing recipe beat the old v0.14.6 pure-PPO hard-push baseline at 10N x 10?** The implementation provides a boundary-focused training curriculum, periodic `eval_push/*` and `eval_clean/*` checks, and initial recovery-metric plumbing.

**CRITICAL BUG FIXES APPLIED**: Fixed reward sign regression, schema mismatches, recovery metrics crashes, and test validation gaps identified in post-implementation review.

### Final Training Result
- Training run: `training/wandb/run-20260321_123307-0flajxzh`
- Internal probe result: `eval_push/success_rate` rose to `84.4%` with stable PPO and `eval_clean = 100%`
- Best checked fixed-ladder checkpoint: `checkpoint_160_20971520.pkl`
- Fixed ladder result on the branch gate:
  - `eval_medium = 71.6%`
  - `eval_hard = 39.2%`
  - `eval_hard/term_height_low_frac = 60.8%`

### Verdict
`v0.17.1` did **not** beat the real hard-push baseline.

Interpretation:
- it did not beat the old `v0.14.6` result of `60.84%` at `10N x 10`
- the in-training `eval_push/*` probe overstated the true fixed-ladder hard-push performance
- the dominant failure remained collapse through `term_height_low`
- clean standing and PPO stability were not the bottleneck

Decision:
- stop extending `v0.17.1`
- treat it as the clean recipe baseline result
- the immediate M3-lite follow-up was later superseded as the mainline by the `v0.17.4` OCS2 / humanoid MPC architecture pivot

### Key Goals
- **Primary Target**: Beat v0.14.6 baseline (60.84% success at 10N x 10)
- **Exit Criteria**: `eval_easy > 95%`, `eval_medium > 75%`, `eval_hard > 60%`, `term_height_low_frac` reduction
- **Boundary Focus**: Curriculum centered on empirical bracing boundary (5N-10N regime)  
- **Recipe-Driven**: Execute legged_gym/Rudin + Li et al. 2024 Cassie recipes cleanly

### Code Changes
- **New config**: `training/configs/ppo_standing_v0171.yaml`
  - Boundary-focused push curriculum: 3.0N to 10.0N (covers easy->boundary->hard regime)
  - Extended iterations: 300 (up from 200) for curriculum progression
  - FSM/base controller disabled for pure PPO approach
  - Eval frequency: 25 iterations (eval_push/* and eval_clean/* tests only)
- **Recovery metrics computation**: Enhanced `training/envs/wildrobot_env.py`
  - Added `push_schedule` parameter to `_get_reward()` function
  - Implemented recovery metrics computation logic during push recovery periods
  - Tracks first-step latency, touchdown count, support foot changes, post-push velocity
- **Initial metrics update**: Modified `training/core/experiment_tracking.py`
  - Added recovery metrics initialization to `get_initial_env_metrics_jax()`
  - Ensures recovery metrics are present at environment reset
- **Test infrastructure**: Created `tests/test_v0171_boundary_push_eval.py`
  - Validates boundary push curriculum configuration
  - Tests recovery metrics computation and logging  
  - Verifies eval ladder integration and conservative reward stack

### Config Changes
- **Curriculum Design**: Boundary-focused push forces span the known bracing boundary
  - Easy regime: 3-7N (ankle/hip strategy quality)
  - Boundary regime: 7-10N (first reliable recovery steps at capturability boundary)
  - Hard regime: 10N fixed (target to exceed old baseline)
  - Duration: 10 steps (0.2s) matching proven regime
- **Conservative Reward Stack**: No walking reward creep
  - Walking baggage explicitly zeroed: `step_length: 0.0`, `step_progress: 0.0`, etc.
  - Mild stepping auxiliaries at reduced weights: `step_event: 0.12`, `foot_place: 0.30`
  - Core standing terms preserved: base_height, orientation, collapse prevention
- **Priority Metrics**: Push evals prioritized over reward metrics
  - Primary: `eval_hard/success_rate` (beat v0.14.6 baseline)
  - Boundary performance: `eval_medium/success_rate`, `eval_easy/success_rate`
  - Failure mode tracking: `term_height_low_frac`
  - Recovery diagnostics: `recovery/first_step_latency`, etc.

### Recovery Metrics Implementation
- **First-step latency**: Time from push end to first touchdown event  
- **Touchdown count**: Number of touchdown events during recovery period
- **Support foot changes**: Support foot transitions (simplified as touchdown count)
- **Post-push velocity**: Residual horizontal body velocity after recovery
- **Implementation**: Computed during 50-step window after push ends
- **Status**: Registry + computation logic complete, ready for training validation

### Evaluation Ladder Design
- **eval_clean**: No pushes (quiet standing baseline)
- **eval_easy**: 5N x 10 (easy regime, target >95%)
- **eval_medium**: 8N x 10 (boundary regime, target >75%)
- **eval_hard**: 10N x 10 (hard regime, target >60% to beat v0.14.6)
- **eval_hard_long**: 9N x 15 (long impulse stress test)
- **Note**: Training loop currently runs eval_push/* and eval_clean/* only; eval_ladder/* test not wired yet

### What Was Preserved from v0.17.0
- **Standing-only design**: Zero velocity commands, reactive stepping only when needed
- **Observation layout**: `wr_obs_v2` with contact-aware signals and `capture_point_error`
- **Health defaults**: Proven standing thresholds (target_height: 0.44, min_height: 0.40, etc.)
- **Pure PPO approach**: No FSM, no base controller, no motion priors

### What Remains for v0.17.2
This roadmap decision was later superseded by the `v0.17.4` architecture pivot:
- **Current mainline**: OCS2 / humanoid MPC adoption path
- **Demoted side branch**: M3-lite guide-only foot-placement path
- **Reason**: the project now prioritizes a shared standing+walking architecture with higher external confidence

### Target Performance Validation
- **Baseline to beat**: v0.14.6 achieved 59.59% at 200 iters, 60.84% extended at 10N x 10
- **Success metrics**: 
  - `eval_easy/success_rate > 95%` (5N x 10)
  - `eval_medium/success_rate > 75%` (8N x 10)  
  - `eval_hard/success_rate > 60%` (10N x 10) - **primary target**
  - Reduced `term_height_low_frac` (dominant failure mode)
- **Actual result**: missed `eval_medium` and `eval_hard`; branch proceeds to bounded escalation

### Critical Bug Fixes Applied
**Post-implementation review identified several blocking issues that have been fixed:**

1. **Reward Sign Regression (HIGH PRIORITY)** ✅ **FIXED**
   - **Issue**: Core stability penalties accidentally flipped to rewards (`orientation: 2.0`, `collapse_height: 1.0`, `collapse_vz: 0.5`)
   - **Impact**: Would have made larger pitch/roll error and collapse states increase reward instead of penalizing them
   - **Fix**: Corrected to proper penalty signs (`orientation: -2.0`, `collapse_height: -1.0`, `collapse_vz: -0.5`)

2. **Config Schema Mismatch (HIGH PRIORITY)** ✅ **FIXED**
   - **Issue**: Custom networks, observation, and eval blocks not wired to actual TrainingConfig schema
   - **Impact**: Claimed boundary-eval integration was not actually configured at runtime
   - **Fix**: Restructured to use proper nested `ppo.eval` configuration (enabled: true, interval: 25, num_envs: 64, num_steps: 500)

3. **Recovery Metrics Implementation (HIGH PRIORITY)** ✅ **CRASH FIXED**, ❌ **SEMANTICS LIMITED**
   - **Crash fix**: Fixed `root_velocity.linear[:2]` indexing issue that would cause training crashes ✅
   - **Semantic limitation**: Metrics still don't measure exactly what their names claim ❌
   - **Current state**: 
     - `recovery/touchdown_count`: Measures touchdown event density (not actual episode counts)
     - `recovery/first_step_latency`: Averages early touchdown latencies (not first-only) 
     - `recovery/support_foot_changes`: Duplicates touchdown density (not actual transitions)
     - `recovery/post_push_velocity`: Measures correctly ✅
   - **Impact**: Provides useful diagnostic signals for training, but names are misleading
   - **Proper fix**: Would require episode-level state tracking (significant architecture change)

4. **Test Validation Gaps (MEDIUM PRIORITY)** ✅ **FIXED**
   - **Issue**: Tests didn't properly exercise recovery metrics path or validate claimed behavior
   - **Impact**: Critical bugs above went undetected by test suite  
   - **Fix**: Extended test runtime to cover push recovery periods and added semantic validation (checks latency ≤ 5 steps, counts are binary)

### Additional Remaining Issues (Acknowledged Limitations)

5. **Recovery Metrics Semantics (MEDIUM PRIORITY)** ❌ **KNOWN LIMITATION**
   - **Issue**: Metrics don't measure what their names claim due to per-step computation + MEAN aggregation
   - **Status**: Acknowledged limitation documented in code - would require significant architecture changes to fix properly
   - **Impact**: Names are misleading but metrics provide useful training diagnostics
   - **Decision**: Accept limitation for v0.17.1 - focus on training validation rather than perfect metrics
6. **Eval Ladder Integration (MEDIUM PRIORITY)** ❌ **INCOMPLETE**
   - **Issue**: Fixed ladder suites (eval_clean, eval_easy, eval_medium, eval_hard, eval_hard_long) exist in external script but are not integrated into training-time evaluation
   - **Current state**: Training loop only runs generic `eval_push/*` and `eval_clean/*` 
   - **Impact**: Boundary-focused evaluation must be run manually via external script
   - **Decision**: Accept manual evaluation workflow for v0.17.1

### Training Readiness Status  
- **Blocking crashes**: ✅ All fixed (reward signs, schema, Velocity3D indexing)
- **Config verification**: ✅ Loads correctly, training initializes successfully  
- **Recovery diagnostics**: ⚠️ Provide useful signals despite semantic limitations
- **Eval integration**: ⚠️ Manual execution required via external script
- **Ready for training**: ✅ Can answer core milestone question despite limitations

---

## [v0.17.0] - 2026-03-21: Standing Reset - Reactive Push Recovery

### Summary
`v0.17.0` implements the standing-reset branch for reactive push recovery as defined in `training/docs/standing_training.md`. This is a clean, standing-only branch that follows two public recipes: `legged_gym` / Rudin for RL implementation and Li et al. 2024 Cassie for biped behavior/escalation. The branch focuses on standing robustness under external pushes with reactive stepping only when bracing is no longer sufficient.

### Key Design Principles
- **Standing-only**: Not a walking branch - command is always zero velocity
- **Recipe-driven**: Follows established practices rather than ad hoc exploration
- **Conservative**: Builds on proven standing machinery from v0.14.x baseline
- **Reactive stepping**: Steps only when needed, returns to neutral posture after recovery

### Code Changes
- **New config**: `training/configs/ppo_standing_v0170.yaml`
  - Forked from `ppo_standing_push.yaml` (v0.14.9)
  - `fsm_enabled: false`, `base_ctrl_enabled: false` for pure PPO approach
  - Push forces calibrated to bracing boundary: 9-10N where stepping should begin
- **Observation path**: Uses `wr_obs_v2` layout with contact-aware signals
  - Includes `capture_point_error` which encodes heading-local base linear velocity information
  - Provides projected gravity, base angular velocity, joint states, foot contacts
  - NOTE: Explicit base linear velocity is privileged-only, not in actor observation per current design
- **Evaluation infrastructure**: `training/eval/eval_ladder_v0170.py`
  - Fixed eval suites: `eval_clean`, `eval_easy` (5N), `eval_medium` (8N), `eval_hard` (10N), `eval_hard_long` (9N x 15)
  - Standardized evaluation ladder per standing_training.md requirements
- **Recovery metrics**: Added registry entries in `training/core/metrics_registry.py`
  - Registry infrastructure for future implementation
  - `recovery/first_step_latency`: Time from push end to first step
  - `recovery/touchdown_count`: Number of touchdown events after push
  - `recovery/support_foot_changes`: Support foot transition count
  - `recovery/post_push_velocity`: Residual body velocity after recovery

### Config Changes
- **Health defaults**: Kept proven standing thresholds
  - `target_height: 0.44`, `min_height: 0.40`
  - `max_pitch: 0.8`, `max_roll: 0.8`
  - `collapse_height_buffer: 0.03`, `collapse_vz_gate_band: 0.07`
- **Reward weights**: Standing-focused, walking baggage explicitly zeroed
  - **Explicitly disabled**: All walking-specific reward terms set to 0.0 for config self-containment
    - `step_length: 0.0`: Structured step length rewards
    - `step_progress: 0.0`: Touchdown-to-touchdown forward displacement  
    - `dense_progress: 0.0`: Dense forward progress shaping
    - `cycle_progress: 0.0`: Gait-clock-dependent cycle completion
    - `velocity_step_gate: 0.0`: Forward reward gating based on stepping quality
  - **Kept**: Core stability (base_height, orientation, collapse prevention)
  - **Kept**: Quality terms (slip, action_rate, torque, saturation, posture return)
  - **Kept**: Mild stepping auxiliaries at reduced weights
    - `step_event: 0.12` (reduced from 0.25)
    - `foot_place: 0.30` (reduced from 0.65)
- **Push curriculum**: Centered on empirical bracing boundary
  - 9-10N forces where stepping geometrically expected per capture-point analysis
  - Duration: 10 steps (0.2s) matching proven push regime

### What Was Intentionally Removed
- **Walking-specific reward baggage**:
  - `dense_progress`: Dense forward progress rewards
  - `step_progress`: Structured touchdown-to-touchdown forward displacement
  - `step_length`: Structured step length rewards for propulsion
  - `cycle_progress`: Gait-clock-dependent cycle completion
  - `propulsion_gate`: Forward reward gating based on stepping quality
- **Controllers**: FSM and base controller disabled for pure PPO approach
- **Velocity tracking**: No non-zero velocity commands (standing-only)

### Exit Criteria (v0.17.0)
- [x] Code imports and runs
- [x] New config is clearly standing-only
- [x] Actor observation includes contact-aware signals and `capture_point_error`
- [x] Walking-specific reward baggage removed
- [x] Mild need-gated stepping auxiliaries remain at reduced weights
- [x] Fixed evaluation ladder implemented
- [x] Recovery metrics registry infrastructure added
- [ ] Training validation and smoke tests

### What Remains for v0.17.1
- **Training validation**: Run actual training and verify behavior
- **Recovery metrics computation**: Implement actual computation logic in environment (currently registry-only)
- **Observation enhancement**: Consider adding explicit base linear velocity if capture_point_error encoding proves insufficient
- **Baseline establishment**: Reproduce and exceed old v0.14.6 hard-push baseline
- **Target performance**: 
  - `eval_clean > 95%`
  - `eval_medium > 75%`
  - `eval_hard > 60%`
  - Low `term_height_low_frac`

### Next Steps
If v0.17.0 training succeeds, v0.17.1 should establish the recipe baseline at the bracing boundary. If hard-push recovery plateaus, apply bounded escalation:
- **Track A**: Li-style history for timing/event sensitivity
- **Track B**: M3-lite guide path for touchdown geometry quality

---

## [v0.15.12] - 2026-03-16: Contact-driven step-to-step propulsion after `v0.15.11` still converged to pitch-collapse speed

### Summary
`v0.15.11` improved the shape of the run but did not change the terminal basin. It delayed the old `v0.15.10` pitch exploit, but by the end of the `40`-iteration probe it still converged to the same outcome: strong forward speed with `term_pitch_frac = 1.0`, zero train success, and near-dead `eval_clean`. The root cause is now clearer: dense velocity-shaped reward is still the easiest way to earn speed, while `step_length` remains mostly flat and `cycle_progress` stays too weak and too time-based to dominate credit assignment. `v0.15.12` should therefore pivot from velocity-led reward credit to contact-driven step-to-step propulsion.

### Results (v0.15.11)
- Run: `training/wandb/offline-run-20260315_171309-wxzaehsm`
- Checkpoints: `training/checkpoints/ppo_walking_v01511_20260315_171311-wxzaehsm`
- Verdict: delayed collapse, but same final pitch-driven speed basin
- Trajectory:
  - early `20`-iter checkpoint:
    - `env/forward_velocity: 0.037`
    - `env/velocity_cmd: 0.164`
    - `env/velocity_error: 0.153`
    - `env/success_rate: 0.895`
    - `eval_clean/success_rate: 0.489`
    - `term_pitch_frac: 0.105`
  - final `40`-iter checkpoint:
    - `env/forward_velocity: 0.221`
    - `env/velocity_cmd: 0.164`
    - `env/velocity_error: 0.149`
    - `env/success_rate: 0.000`
    - `eval_clean/success_rate: 0.050`
    - `term_pitch_frac: 1.000`
    - `reward/dense_progress: 0.166`
    - `reward/step_length: 0.138`
    - `reward/cycle_progress: 0.017`
    - `debug/propulsion_gate: 0.145`

### Root Cause
- `v0.15.11` proved that posture-gating alone is not enough:
  - it slowed the exploit and improved early stability
  - but it did not change which reward path ultimately dominates
- The structured gait terms still are not the main explanation for speed:
  - `step_length` stays nearly flat as the run accelerates
  - `cycle_progress` remains tiny and sparse
  - the final speed increase still comes mainly from dense forward-progress reward
- The current cycle reward is also poorly aligned to actual gait events:
  - clock-cycle completion is weaker than contact-driven touchdown-to-touchdown progress for this failure mode
- Net conclusion:
  - the next branch must stop paying primarily for instantaneous forward motion
  - and instead pay for forward displacement caused by successful steps

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_step_progress_touchdown_reward()` for structured touchdown-to-touchdown forward displacement credit in heading-local frame
  - track touchdown state in env info (`last_touchdown_root_pos`, `last_touchdown_foot`) so reward is tied to real contact transitions
  - gate `forward_reward` with `compute_propulsion_quality_gate(step_length, step_progress)` and remove dense velocity / cycle terms from the gate path
  - keep `dense_progress` and `cycle_progress` available only as auxiliary reward channels (not gate openers)
  - make the commanded forward reward unlock only from structured propulsion evidence:
    - `step_length`
    - `step_progress`
  - keep `wr_obs_v3` clock and phase-gated clearance, but demote pure clock-cycle reward as the main propulsion signal
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add/parse structured propulsion config fields:
    - `reward_weights.step_progress`
    - `reward_weights.step_progress_target_scale`
    - `reward_weights.step_progress_sigma`
    - `reward_weights.propulsion_gate_step_length_weight`
    - `reward_weights.propulsion_gate_step_progress_weight`
- `training/tests/test_reward_terms.py`
  - add helper-linked tests for:
    - touchdown-to-touchdown forward progress reward semantics
    - propulsion gate depending on structured step evidence instead of dense velocity alone
    - reset / alternating-touchdown behavior for the new `step_progress` term

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.12"`
- keep:
  - `env.actor_obs_layout_id: wr_obs_v3`
  - fresh-run semantics
  - short probe format (`ppo.iterations: 40`)
- change the reward mix toward structured contact-driven propulsion:
  - `dense_progress: 0.0` or tiny support only
  - `step_length`: increase into the primary Stage-A structured term
  - add `step_progress` as a strong reward
  - demote or remove `cycle_progress` as the main propulsion driver
  - keep `foot_place: 0.0` and `step_event` negligible
- narrow command range again for the structured branch:
  - bias toward `0.05-0.15 m/s`
- keep stronger pitch stability shaping from `v0.15.11`

### `v0.15.12` Plan (Completed)
- Keep:
  - zero-baseline forward reward
  - `wr_obs_v3`
  - phase-gated clearance
  - stronger pitch / posture penalties
- Remove dense velocity as the main learning path:
  - take `dense_progress` out of the propulsion gate
  - reduce its weight to near-zero or zero
- Shift to contact-driven step-to-step propulsion:
  - reward touchdown-time `step_length`
  - reward new touchdown-to-touchdown `step_progress`
  - gate commanded forward reward on structured step evidence, not raw speed
- Keep the probe short and decisive:
  - fresh run
  - `ppo.iterations: 40`

### Training Intent
- Convert the branch from velocity-led learning to contact-driven locomotion learning.
- Make the policy earn forward reward because steps move the body forward, not because the body is already moving forward.
- Readout:
  - iter `20`: `env/forward_velocity > 0.03` with `term_pitch_frac` still well below `v0.15.10/11` collapse levels
  - iter `40`: `env/forward_velocity > 0.06` with meaningful `eval_clean` survival and structured reward terms (`step_length`, `step_progress`) clearly active
  - if speed rises again while structured step rewards stay flat, stop and treat the branch as failed

## [v0.15.11] - 2026-03-16: Stability-constrained propulsion after `v0.15.10` broke the low-speed basin by pitching out

### Summary
`v0.15.10` is the first walking branch that truly broke the low-propulsion / stepping-in-place basin. The run reached and exceeded commanded forward velocity, proving that the reward-economics fix worked. But the branch immediately converted that propulsion into a pitch-driven collapse basin: forward speed kept rising while `term_pitch_frac` went to `1.0` and `eval_clean` collapsed. `v0.15.11` should therefore keep the `v0.15.10` reward-economics foundation, but make propulsion pay only when posture remains sufficiently upright.

### Results (v0.15.10)
- Run: `training/wandb/offline-run-20260315_135306-ta8q1b8v`
- Checkpoints: `training/checkpoints/ppo_walking_v01510_20260315_135308-ta8q1b8v`
- Verdict: propulsion unlocked, stability lost
- Basin reached by iter `40-50`:
  - `env/forward_velocity: 0.175 -> 0.215`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.154 -> 0.148`
  - `env/success_rate: 0.010 -> 0.000`
  - `eval_clean/success_rate: 0.021 -> 0.042`
  - `term_pitch_frac: 0.990 -> 1.000`
  - `reward/dense_progress: 0.389 -> 0.456`
  - `reward/step_length: ~0.135` (plateaued)
  - `reward/cycle_progress: ~0.017` (still weak)
  - `debug/propulsion_gate: 0.467 -> 0.525`

### Root Cause
- `v0.15.10` fixed the old reward basin:
  - forward reward no longer overpaid near-standing
  - propulsion gating and dense progress shaping finally created a real path to translation
- But the new dense propulsion path is not posture-constrained enough:
  - `dense_progress` keeps rising as the policy leans harder
  - `step_length` plateaus instead of improving further
  - `cycle_progress` remains too small to be the dominant structured gait signal
- Net result:
  - the policy discovered a fast way to earn propulsion-related reward
  - that fast way is a pitch-driven locomotion exploit, not stable walking

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_smooth_upright_gate()` and use it to posture-gate `dense_progress` with smooth pitch/pitch-rate margins
  - posture-gate commanded forward reward (zero-baseline path) with configurable strength so forward tracking pays far less during pitch collapse
  - tighten propulsion gate economics via weighted structured-vs-dense composition:
    - structured signal: `max(step_length, cycle_progress)`
    - dense support: capped contribution with configurable weight/cap
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add/parse new reward fields:
    - `dense_progress_upright_pitch`
    - `dense_progress_upright_pitch_rate`
    - `dense_progress_upright_sharpness`
    - `forward_upright_gate_strength`
    - `propulsion_gate_dense_weight`
    - `propulsion_gate_structured_weight`
    - `propulsion_gate_dense_cap`
- `training/tests/test_reward_terms.py`
  - extend helper-linked tests for:
    - posture-gated dense progress under excessive pitch/pitch-rate
    - monotonic smooth upright gate behavior
    - propulsion gate no longer dominated by dense-progress-only signal

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.11"` and shorten probe to `ppo.iterations: 40`
- keep `wr_obs_v3` and the `v0.15.10` reward-economics base (zero-baseline + propulsion gating)
- retune for stability-constrained propulsion:
  - lower `dense_progress` weight (`0.8 -> 0.35`)
  - slightly raise structured cycle term (`cycle_progress: 1.0 -> 1.2`)
  - strengthen stability shaping:
    - `orientation: -0.4 -> -0.7`
    - `pitch_rate: -0.25 -> -0.4`
    - `collapse_height: -0.25 -> -0.3`
    - `collapse_vz: -0.15 -> -0.2`
  - add upright/propgate controls:
    - `dense_progress_upright_pitch: 0.22`
    - `dense_progress_upright_pitch_rate: 0.75`
    - `dense_progress_upright_sharpness: 10.0`
    - `forward_upright_gate_strength: 0.85`
    - `propulsion_gate_dense_weight: 0.25`
    - `propulsion_gate_structured_weight: 0.75`
    - `propulsion_gate_dense_cap: 0.35`

### `v0.15.11` Plan (Completed)
- Keep the `v0.15.10` foundation:
  - zero-baseline forward reward
  - propulsion-quality gate
  - relaxed Stage-A step targets
  - `wr_obs_v3`
- Add stability-constrained propulsion:
  - make `dense_progress` posture-aware using smooth pitch / pitch-rate gates
  - optionally make the forward reward itself upright-gated for commanded walking
  - reduce `dense_progress` weight so it cannot dominate by pure leaning
  - increase `orientation` and `pitch_rate` penalties
  - strengthen pre-collapse / anti-pitch shaping if needed
- Tighten what opens the propulsion gate:
  - let `step_length` and `cycle_progress` matter more relative to dense progress
  - dense progress should support propulsion learning, not fully define it
- Keep the probe short:
  - fresh run
  - `ppo.iterations: 40`

### Training Intent
- Preserve the propulsion breakthrough from `v0.15.10`.
- Prevent the new reward path from paying mostly for pitch-driven forward collapse.
- Readout:
  - iter `20`: `env/forward_velocity > 0.03` while `term_pitch_frac < 0.25`
  - iter `40`: `env/forward_velocity > 0.05` and `eval_clean/success_rate` remains meaningfully alive
  - if speed rises only together with pitch collapse again, stop immediately

## [v0.15.10] - 2026-03-15: Reward-economics fix after `v0.15.9` low-propulsion basin

### Summary
`v0.15.9` confirmed that the branch is no longer blocked by missing stepping, missing clock input, or missing propulsion-specific reward terms. It is blocked by reward economics. The dense forward-tracking term still pays a large positive reward even when the robot barely moves, while the new propulsion terms (`step_length`, `cycle_progress`) are too sparse and too small in realized value to compete. PPO therefore converges to safe low-propulsion stepping because that solution is already strongly rewarded.

### Results (v0.15.9)
- Run: `training/wandb/offline-run-20260315_093127-25uho0zk`
- Checkpoints: `training/checkpoints/ppo_walking_v00159_20260315_093129-25uho0zk`
- Verdict: no-go; another safe stepping / near-standing basin
- Best checkpoint by current analyzer: iter `50`
- Key metrics near the end of the probe:
  - `env/forward_velocity: 0.003-0.009`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.170-0.174`
  - `env/success_rate: ~0.89`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: ~0.11`
  - `reward/step_event: ~0.354`
  - `reward/foot_place: ~0.229`
  - `reward/step_length: ~0.003`
  - `reward/cycle_progress: ~0.004`
  - `debug/velocity_step_gate: ~0.936`

### Root Cause
- Dense forward reward is still too generous at low velocity:
  - `reward/forward = exp(-|forward_vel - velocity_cmd| * forward_velocity_scale)`
  - at `velocity_cmd ~= 0.164` and `forward_vel ~= 0`, raw forward reward is still about `0.37`
  - with `tracking_lin_vel: 8.0`, that is about `+3.0` reward before other shaping
- The standing penalty is far too small to offset that basin:
  - `velocity_standing_penalty: 0.5`
- The step gate still opens on generic stepping signals rather than propulsion-quality signals:
  - clearance / periodicity / touchdown are enough to unlock the dense forward reward
- The new propulsion terms are numerically too weak in practice:
  - `step_length` and `cycle_progress` are correct directionally, but their realized values stay around `0.001-0.005`
  - `cycle_progress` is also sparse because it only pays on cycle completion
- Net result:
  - safe stepping with little translation is already highly rewarded
  - real propulsion is too sparse and too hard to discover early

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_zero_baseline_forward_reward()` and apply it to forward tracking so commanded near-standing no longer gets large positive reward
  - add `compute_dense_progress_reward()` for dense heading-local propulsion credit each step
  - add `compute_propulsion_quality_gate()` and gate forward reward on propulsion-quality (`step_length`, dense progress, cycle progress), replacing generic stepping gate economics
  - keep cycle progress heading-local and cycle-complete, but pair it with dense per-step progress shaping
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add and parse `reward_weights.dense_progress`
- `training/tests/test_reward_terms.py`
  - add helper-linked tests for zero-baseline forward reward, propulsion-quality gate, and dense progress semantics

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.10"`
- keep fresh-run, `wr_obs_v3`, and short probe (`ppo.iterations: 60`)
- relax Stage-A propulsion targets:
  - `step_length_target_scale: 0.30 -> 0.15`
  - `step_length_sigma: 0.03 -> 0.06`
  - `cycle_progress_target_scale: 1.0 -> 0.6`
  - `cycle_progress_sigma: 0.06 -> 0.12`
- densify propulsion shaping:
  - `dense_progress: 0.8`
- reduce legacy stepping reward path:
  - `step_event: 0.001`
  - `foot_place: 0.0`

### Training Intent
- Break the reward basin first, not the controller architecture.
- The next probe should answer whether propulsion becomes economically preferable once bad tracking no longer pays well.
- Readout:
  - iter `20`: forward velocity should clearly exceed the `v0.15.9` floor
  - iter `40`: should separate from safe stepping / near-standing
  - iter `60`: if `env/forward_velocity <= 0.02`, treat the branch as failed

## [v0.15.9] - 2026-03-15: Per-step propulsion rewards after `v0.15.8` still stepped in place

### Summary
Analyzes the failed `v0.15.8` run `training/wandb/offline-run-20260314_230225-jgkxxp2w`. Even with `wr_obs_v3` clocked observations, the policy still converged to a stepping-in-place basin: stepping activity was present, but commanded forward translation remained near the floor. `v0.15.9` therefore shifts reward structure from generic stepping to explicit propulsion contracts: touchdown forward step length, per-cycle net forward displacement, and clock-phase-gated clearance.

### Why `v0.15.8` failed
- Clock features improved timing cues but did not change what was being paid for most strongly.
- Existing touchdown/foot-placement terms still allowed high reward from low-propulsion stepping patterns.
- Result: stable stepping without material cycle-to-cycle forward displacement.

### Code Updates
- `training/envs/wildrobot_env.py`
  - add touchdown `step_length` reward tied to `velocity_cmd`
  - add `cycle_progress` reward at clock-cycle completion using net forward displacement
  - phase-gate clearance with `clock_phase_gate_width` so left/right clearance only pays in expected swing windows
  - add metrics and auto-reset preservation for new reward/debug terms
- `training/envs/env_info.py`
  - add `cycle_start_forward_x` to persistent env info state for cycle displacement tracking
- `training/core/metrics_registry.py`
  - register `reward/step_length`, `reward/cycle_progress`, `debug/cycle_complete`, `debug/cycle_forward_delta`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.9"`
- keep fresh-run clocked actor contract:
  - `env.actor_obs_layout_id: wr_obs_v3`
  - `ppo.iterations: 60`
- add phase gate parameter:
  - `env.clock_phase_gate_width: 0.22`
- reduce undirected stepping reward:
  - `reward_weights.step_event: 0.005`
- make propulsion dominate:
  - `reward_weights.step_length: 1.0`
  - `reward_weights.step_length_target_base: 0.035`
  - `reward_weights.step_length_target_scale: 0.30`
  - `reward_weights.step_length_sigma: 0.03`
  - `reward_weights.cycle_progress: 1.2`
  - `reward_weights.cycle_progress_target_scale: 1.0`
  - `reward_weights.cycle_progress_sigma: 0.06`
- keep stability terms broadly aligned with `v0.15.6+` and only modestly retune gait auxiliaries.

### Training Intent
- Keep `wr_obs_v3`, but pay primarily for propulsion per touchdown and per cycle.
- Expected staged readout:
  - iter `20`: forward velocity should move off floor more clearly than `v0.15.8`
  - iter `40`: should visibly break stepping-in-place basin
  - iter `60`: if still near-zero forward velocity, stop and treat branch as failed

## [v0.15.8] - 2026-03-14: Clocked observation after `v0.15.7` exhausted reward-only tuning

### Summary
Analyzes the completed `v0.15.7` run `training/wandb/offline-run-20260314_193228-0y0j38w2` and concludes that reward-only tuning is no longer the main bottleneck. The policy again stayed upright, opened the stepping gate, and produced strong step rewards, yet still failed to generate useful forward motion. By iteration `60`, `debug/velocity_step_gate` remained near `0.96`, `reward/step_event` remained high, and `eval_clean/success_rate` returned to `1.0`, while `env/forward_velocity` was still only about `0.008 m/s` against `env/velocity_cmd ≈ 0.164`. That means the next justified lever is no longer another local reward retune, but an observation change: `wr_obs_v3` adds an explicit gait clock so the actor can coordinate periodic stepping and propulsion more directly.

### Results (v0.15.7)
- Run: `training/wandb/offline-run-20260314_193228-0y0j38w2`
- Checkpoints: `training/checkpoints/ppo_walking_v00157_20260314_193230-0y0j38w2`
- Verdict: reward-only tuning exhausted; still stepping without propulsion
- Best checkpoint: `checkpoint_60_7864320.pkl`
- Key metrics at iter `60`:
  - `env/forward_velocity: 0.008`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.166`
  - `env/success_rate: 0.951`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: 0.049`
  - `reward/step_event: 0.353`
  - `reward/foot_place: 0.205`
  - `debug/velocity_step_gate: 0.961`

### Code Updates
- add policy-contract layout `wr_obs_v3`
- add 4-D gait clock features (`sin/cos` for left and right phase) to JAX and NumPy observation builders
- add `env.clock_stride_period_steps`
- wire clock features through the training env and policy visualization path

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.8"`
- switch actor observation layout:
  - `env.actor_obs_layout_id: wr_obs_v1 -> wr_obs_v3`
  - `env.clock_stride_period_steps: 36`
- keep the aggressive propulsion objective from `v0.15.7`
- increase exploration slightly for the fresh contract change:
  - `ppo.entropy_coef: 0.02 -> 0.03`
- keep the short hard-stop probe:
  - `ppo.iterations: 60`

### Training Intent
- This is a fresh run, not a resume-safe continuation. `wr_obs_v3` changes the policy contract.
- Give the actor an explicit periodic cue now that the reward path can already induce stable stepping.
- Use the same `20` / `40` / `60` readout, but expect a clearer answer faster than the reward-only branches.

## [v0.15.7] - 2026-03-14: Aggressive propulsion rebalance after `v0.15.6` learned to step in place

### Summary
Analyzes the completed `v0.15.6` run `training/wandb/offline-run-20260314_143238-eten6grc` and concludes that the structural stepping fix worked, but it converged to a new local minimum: stable stepping without useful forward propulsion. By iteration `80`, the run maintained excellent survival and low pitch failure while `debug/velocity_step_gate` stayed near `0.96` and stepping rewards stayed high, yet `env/forward_velocity` remained only about `0.003-0.006 m/s` against a commanded `0.124 m/s`. `v0.15.7` therefore becomes an aggressive propulsion probe: pay far less for arbitrary touchdown, pay much more for command-aligned forward foot placement, and increase raw forward-tracking pressure now that the policy can step without collapsing.

### Results (v0.15.6)
- Run: `training/wandb/offline-run-20260314_143238-eten6grc`
- Checkpoints: `training/checkpoints/ppo_walking_v00156_20260314_143239-eten6grc`
- Verdict: trapped in standing / stepping-in-place
- Best checkpoint: `checkpoint_80_10485760.pkl`
- Key metrics at iter `80`:
  - `env/forward_velocity: 0.003`
  - `env/velocity_cmd: 0.124`
  - `env/velocity_error: 0.130`
  - `env/success_rate: 0.985`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: 0.015`
  - `reward/step_event: 0.341`
  - `reward/foot_place: 0.298`
  - `debug/velocity_step_gate: 0.964`

### Code Updates
- add `reward_weights.foot_place_k_cmd_vel`
- make foot-placement forward target depend on `velocity_cmd` as well as current `forward_vel`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.7"`
- keep the short hard-stop probe:
  - `ppo.iterations: 60`
- increase exploration / propulsion pressure:
  - `ppo.entropy_coef: 0.01 -> 0.02`
  - `env.min_velocity: 0.05 -> 0.08`
  - `env.max_velocity: 0.20 -> 0.25`
  - `reward_weights.tracking_lin_vel: 6.0 -> 8.0`
  - `reward_weights.forward_velocity_scale: 5.0 -> 6.0`
  - `reward_weights.velocity_standing_penalty: 0.4 -> 0.5`
  - `reward_weights.velocity_standing_threshold: 0.10 -> 0.12`
  - `reward_weights.velocity_cmd_min: 0.05 -> 0.08`
- aggressively rebalance stepping rewards toward propulsion:
  - `reward_weights.step_event: 0.30 -> 0.05`
  - `reward_weights.foot_place: 0.15 -> 0.35`
  - `reward_weights.foot_place_k_cmd_vel: 0.30`
  - `reward_weights.foot_place_k_fwd_vel: 0.05 -> 0.08`

### Training Intent
- Bias the policy away from stepping-in-place and toward net forward placement per step.
- Keep the run short; by `60` iterations it should be obvious whether the new objective converts stepping into propulsion.
- If this still fails, the next step should be a more invasive observation / curriculum change rather than another small reward retune.

## [v0.15.6] - 2026-03-14: Structural reward fix after `v0.15.5` fell back to standing

### Summary
Analyzes the `v0.15.5` probe run `training/wandb/offline-run-20260314_083700-h51mp8jx` and treats it as confirmation that config-only retuning is insufficient. By iteration `100`, the policy had largely recovered stable eval-clean survival, but forward velocity remained near zero (`~0.016 m/s`), meaning the run reverted to the stand-still basin instead of discovering a stepping gait. Historical review of `v0.15.3` to `v0.15.5` also showed that the basin is already obvious by around `80` iterations. `v0.15.6` therefore makes the smallest structural change that directly targets the failure mode and shortens the probe: forward reward is now gated by stepping engagement, pitch-rate is penalized explicitly, and the already-implemented stepping rewards are activated.

### Results (v0.15.5)
- Run: `training/wandb/offline-run-20260314_083700-h51mp8jx`
- Checkpoints: `training/checkpoints/ppo_walking_v00155_20260314_083702-h51mp8jx`
- Verdict: stable-standing fallback after a brief early lean/fall transient
- Early failure phase: around iter `20`, `env/forward_velocity ~ 0.036` with heavy pitch failure
- Latest inspected point: iter `100`
- Key metrics at iter `100`:
  - `env/forward_velocity: ~0.016`
  - `env/velocity_cmd: ~0.209`
  - `env/velocity_error: ~0.208`
  - `eval_clean/success_rate: 1.000`

### Code Updates
- add explicit `reward_weights.pitch_rate` support and include it in total reward
- gate the forward tracking reward by stepping evidence:
  - single-support contact diversity
  - swing clearance
  - liftoff/touchdown events
- add `debug/velocity_step_gate` and `reward/pitch_rate` to env metrics / registry
- preserve the new metrics through auto-reset

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.6"`
- keep the short structural probe format:
  - `ppo.iterations: 80`
  - `ppo.num_envs: 1024`
  - `ppo.rollout_steps: 128`
- lower the command curriculum to a shuffle-discovery band:
  - `env.min_velocity: 0.12 -> 0.05`
  - `env.max_velocity: 0.30 -> 0.20`
- reduce unconditional velocity forcing and make it contingent on stepping:
  - `reward_weights.tracking_lin_vel: 8.0 -> 6.0`
  - `reward_weights.forward_velocity_scale: 6.0 -> 5.0`
  - `reward_weights.velocity_step_gate: 1.0`
- explicitly penalize pitch-rate drift:
  - `reward_weights.pitch_rate: -0.25`
- activate the stepping reward path already present in the env:
  - `reward_weights.step_event: 0.30`
  - `reward_weights.foot_place: 0.15`
- rebalance the low-speed standing penalty for the new command band:
  - `reward_weights.velocity_standing_penalty: 0.5 -> 0.4`
  - `reward_weights.velocity_standing_threshold: 0.16 -> 0.10`
  - `reward_weights.velocity_cmd_min: 0.10 -> 0.05`

### Training Intent
- Keep the experiment incremental: fix the objective before introducing a clocked observation contract or larger batch geometry.
- Require real stepping engagement before the policy can earn most of its forward reward.
- Use `40` / `60` / `80` as the staged readout, with `80` as the hard go / no-go gate for whether Stage 1b now has a viable stepping path.

### Why Not Clock Yet
- The next lever after `v0.15.5` could have been a clocked observation layout, but that was intentionally deferred in `v0.15.6`.
- At that point the main unresolved question was still whether the reward objective itself could make the policy step, because prior runs were trapped between standing and lean-fall.
- A clock change would have been broader and harder to attribute:
  - new observation contract
  - env observation changes
  - policy / export / eval compatibility changes
- `v0.15.6` was therefore used as the smallest structural test first:
  - if stepping did not emerge, the next step would be clock
  - if stepping did emerge, the next bottleneck could be isolated more cleanly
- Result: `v0.15.6` did induce stable stepping, so the decision to delay clock was justified for diagnosis. The remaining failure after `v0.15.6` is no longer "no timing cue," but "stepping without propulsion."

## [v0.15.5] - 2026-03-14: Conservative gait-emergence reset after pitch-instability regression

### Summary
Analyzes the ongoing `v0.15.4` walking run `training/wandb/offline-run-20260313_212618-c7z4fbv8` and concludes that the policy is still in a non-deployable posture-exploit regime. Relative to `v0.15.3`, `v0.15.4` did increase forward velocity somewhat, but it did so by leaning harder and eventually collapsing eval-clean stability. By iteration `210`, forward velocity rose to about `0.088 m/s` against a commanded `0.273 m/s`, while `term_pitch_frac` rose above `0.81` and `eval_clean/success_rate` fell below `0.09`.

### Results (v0.15.4)
- Run: `training/wandb/offline-run-20260313_212618-c7z4fbv8`
- Checkpoints: `training/checkpoints/ppo_walking_v00154_20260313_212621-c7z4fbv8`
- Verdict: posture exploit with pitch-instability regression, not deployable walking
- Best stable phase: around iter `100`, but still near-zero usable forward speed
- Latest logged point: iter `210`
- Key metrics at iter `210`:
  - `env/forward_velocity: 0.088`
  - `env/velocity_cmd: 0.273`
  - `env/velocity_error: 0.235`
  - `term_pitch_frac: 0.817`
  - `env/success_rate: 0.183`
  - `eval_clean/success_rate: 0.084`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.5"`
- reset Stage 1b toward an earlier gait-emergence regime rather than increasing fixed reward pressure again
- make this a short diagnostic probe instead of a full 300-iteration commitment:
  - `ppo.iterations: 300 -> 120`
  - `checkpoints.interval: 25 -> 20`
- narrow the command curriculum further:
  - `env.min_velocity: 0.10 -> 0.12`
  - `env.max_velocity: 0.45 -> 0.30`
- add a bit more action damping to reduce pitch-diving:
  - `env.action_filter_alpha: 0.2 -> 0.3`
- keep velocity tracking primary, but reduce the fixed forcing from `v0.15.4`:
  - `reward_weights.tracking_lin_vel: 10.0 -> 8.0`
  - `reward_weights.forward_velocity_scale: 8.0 -> 6.0`
  - `reward_weights.velocity_standing_penalty: 0.8 -> 0.5`
  - `reward_weights.velocity_standing_threshold: 0.18 -> 0.16`
- lighten effort penalties aggressively, but only lighten posture / pre-collapse penalties moderately:
  - `reward_weights.orientation: -0.8 -> -0.4`
  - `reward_weights.angular_velocity: -0.12 -> -0.06`
  - `reward_weights.height_target: 0.2 -> 0.10`
  - `reward_weights.collapse_height: -0.4 -> -0.25`
  - `reward_weights.collapse_vz: -0.3 -> -0.15`
  - `reward_weights.torque: -0.0012 -> -0.0007`
  - `reward_weights.saturation: -0.1 -> -0.05`
  - `reward_weights.slip: -0.15 -> -0.05`

### Training Intent
- Start from scratch rather than resume from a checkpoint that already encodes a posture-exploit basin.
- Solve conservative nominal forward translation first.
- Use the first 120 iterations as an explicit go / no-go gate:
  - continue only if forward velocity rises materially without eval-clean pitch collapse
  - otherwise retune again rather than training longer into a bad solution
- Only after stable translation appears should the command range widen and stability / effort penalties tighten again.

## [v0.15.4] - 2026-03-13: Reward rebalance for walking velocity and pitch stability

### Summary
Analyzes the fresh-start walking run `training/wandb/offline-run-20260313_143658-oirn4qg9` and treats it as a non-deployable posture-exploit trajectory rather than a promising gait. By iteration `120`, forward velocity only reached about `0.04 m/s` against a commanded `0.35 m/s`, while pitch terminations rose to about `60%` and eval success collapsed. The fix is to pay substantially more for commanded forward motion, pay less for passive survival, and increase the cost of low-height / pitch-instability drift.

### Results (v0.15.3)
- Run: `training/wandb/offline-run-20260313_143658-oirn4qg9`
- Checkpoints: `training/checkpoints/ppo_walking_v00153_20260313_143700-oirn4qg9`
- Verdict: posture exploit / weak shuffle, not deployable walking
- Latest logged point: iter `120`
- Key metrics at iter `120`:
  - `env/forward_velocity: 0.040`
  - `env/velocity_cmd: 0.348`
  - `env/velocity_error: 0.326`
  - `term_pitch_frac: 0.601`
  - `env/success_rate: 0.399`
  - `eval_clean/success_rate: 0.314`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.4"`
- narrow the command curriculum slightly so the robot first solves a more stable nominal gait:
  - `env.max_velocity: 0.6 -> 0.45`
- reduce flat survival reward:
  - `reward_weights.base_height: 0.1 -> 0.03`
- increase direct pressure to match the commanded speed:
  - `reward_weights.tracking_lin_vel: 8.0 -> 10.0`
  - `reward_weights.forward_velocity_scale: 6.0 -> 8.0`
  - `reward_weights.velocity_standing_penalty: 0.5 -> 0.8`
  - `reward_weights.velocity_standing_threshold: 0.15 -> 0.18`
- strengthen stability / pre-collapse shaping:
  - `reward_weights.orientation: -0.5 -> -0.8`
  - `reward_weights.angular_velocity: -0.1 -> -0.12`
  - `reward_weights.height_target: 0.1 -> 0.2`
  - `reward_weights.collapse_height: -0.2 -> -0.4`
  - `reward_weights.collapse_vz: -0.2 -> -0.3`
- modestly increase effort penalties to discourage the pitchy high-torque solution:
  - `reward_weights.torque: -0.001 -> -0.0012`
  - `reward_weights.saturation: -0.08 -> -0.1`
- slightly relax slip cost so early stepping is not overconstrained:
  - `reward_weights.slip: -0.2 -> -0.15`

### Training Intent
- Force a larger reward gap between standing still and matching the forward command.
- Preserve Stage 1 PPO-only walking while making pitch stability a hard requirement.
- Stop using survival-only eval signals as evidence of walking quality; walking checkpoint selection should prioritize velocity/error plus stability.

## [v0.15.3] - 2026-03-13: Fresh-start walking after standing-resume failure

### Summary
Promotes the walking branch from "resume from standing" to a true fresh-run Stage 1b experiment. `v0.15.2` showed that the standing checkpoint remained too strong a local minimum: forward velocity stayed near zero while pitch failures and torque climbed. That means the standing warm start is blocking gait discovery rather than helping it.

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.3"`
- start from scratch:
  - do **not** resume from the standing checkpoint
- keep controllers off:
  - `env.base_ctrl_enabled: false`
  - `env.fsm_enabled: false`
- keep PPO-only Stage 1b setup:
  - `amp.enabled: false`
  - `env.actor_obs_layout_id: wr_obs_v1`
- reduce action lag for gait emergence:
  - `env.action_filter_alpha: 0.2`
- switch back to standard walking batch geometry:
  - `ppo.num_envs: 1024`
  - `ppo.rollout_steps: 128`
- use standard PPO learning rate for fresh training:
  - `ppo.learning_rate: 3e-4`
- keep the stronger walking incentive from `v0.15.2`:
  - high velocity tracking weight
  - strong standing penalty
  - low flat healthy bonus
  - rollback disabled

### Training Intent
- Learn walking directly instead of trying to deform a standing policy into a gait.
- Use the first `80-120` iterations as the main decision window for forward-velocity emergence.
- Continue to `300` iterations only if the policy shows real translational motion rather than posture/torque exploitation.

## [v0.15.2] - 2026-03-13: Walking breakout from standing local minimum

### Summary
Tightens the Stage 1b walking config after `v0.15.1` stayed near zero forward velocity while preserving perfect standing success. The root cause was that the warm-started standing policy could still earn good reward by standing still: forward tracking was too soft, the standing penalty was too weak, and the `base_height` term was effectively acting as a flat healthy bonus.

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.2"`
- keep warm-start compatibility with the standing checkpoint:
  - `env.actor_obs_layout_id: wr_obs_v1`
  - `env.action_filter_alpha: 0.6`
- increase pressure to move:
  - `reward_weights.tracking_lin_vel: 8.0`
  - `reward_weights.forward_velocity_scale: 6.0`
  - `reward_weights.velocity_standing_penalty: 0.5`
  - `reward_weights.velocity_standing_threshold: 0.15`
  - `reward_weights.velocity_cmd_min: 0.1`
- reduce reward for "just stay healthy":
  - `reward_weights.base_height: 0.1`
  - `reward_weights.height_target: 0.1`
- relax posture-only costs that can block gait emergence:
  - `reward_weights.orientation: -0.5`
  - `reward_weights.angular_velocity: -0.1`
  - `reward_weights.slip: -0.2`
- slightly favor alternating support:
  - `reward_weights.gait_periodicity: 0.15`
- make PPO updates more willing to leave the standing basin:
  - `ppo.learning_rate: 2e-4`
  - `ppo.entropy_coef: 0.01`
  - `ppo.iterations: 300`
- disable rollback for this transition run:
  - `ppo.rollback.enabled: false`

### Training Intent
- Break the warm-started standing policy out of the stand-still solution.
- Accept some temporary drop in standing-style success if it produces actual locomotion.
- Use the first `80-120` iterations as the main decision window for whether forward motion is emerging.

## [v0.15.1] - 2026-03-12: Freeze best standing bundle and prepare Stage 1b walking

### Summary
Freezes the best standing-push pure-PPO policy from `v0.14.6` into a runtime bundle, then switches the default walking config back to the repo's main execution plan: Stage 1b PPO-only walking with no AMP and no stepping controller.

### Frozen Standing Bundle
- Source checkpoint:
  - `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- Matching historical config:
  - `training/configs/ppo_standing_push.yaml` @ commit `d32d11c`
- Exported runtime bundle:
  - `runtime/bundles/standing_push_v0.14.6_ckpt250`

### Walking Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.1"`
- switch back to Stage 1b walking intent:
  - PPO only
  - AMP disabled
  - `env.base_ctrl_enabled: false`
  - `env.fsm_enabled: false`
- move walking config to current asset variant:
  - `env.assets_root: assets/v2`
- keep warm-start compatibility with the frozen standing checkpoint:
  - `env.actor_obs_layout_id: wr_obs_v1`
  - `env.action_filter_alpha: 0.6`
- use a conservative commanded walking range:
  - `env.min_velocity: 0.1`
  - `env.max_velocity: 0.6`
- add modern PPO eval / rollback settings and keep the standing batch geometry for resume stability:
  - `ppo.num_envs: 512`
  - `ppo.rollout_steps: 256`
  - `ppo.iterations: 400`
  - `ppo.eval.enabled: true`
  - `ppo.rollback.enabled: true`

### Training Intent
- Warm-start walking from the best pure standing-push PPO checkpoint instead of continuing the disproven FSM branch.
- Learn nominal foot placement and weight transfer through dense walking rewards rather than sparse recovery events.
- Use this as the Stage 1b baseline before any later robustness or AMP work.

### Run Command
```bash
uv run python training/train.py \
  --config training/configs/ppo_walking.yaml \
  --resume training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl
```

## [v0.14.9] - 2026-03-11: FSM guide on top of actor capture-point observation

### Summary
Reintroduces the M3 foot-placement FSM after both information-first branches plateaued below the `v0.14.6` pure-PPO baseline. `v0.14.7` tested critic-only privileged information and `v0.14.8` tested actor-side capture-point information; neither beat the best pure-PPO result. The next step is to test whether the existing actor signal plus a guide-only FSM can turn step intent into more effective recovery.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.9"`
- keep actor capture-point observation:
  - `env.actor_obs_layout_id: wr_obs_v2`
- re-enable FSM:
  - `env.fsm_enabled: true`
- keep M2 disabled:
  - `env.base_ctrl_enabled: false`
- keep full residual override in every phase:
  - `env.fsm_resid_scale_stance: 1.00`
  - `env.fsm_resid_scale_swing: 1.00`
  - `env.fsm_resid_scale_recover: 1.00`
- keep the calibrated disturbance regime:
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`
- keep critic symmetric:
  - `ppo.critic_privileged_enabled: false`

### Resume Point
- Resume from the best `v0.14.8` checkpoint:
  - `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r/checkpoint_200_26214400.pkl`
- This is resume-safe because `wr_obs_v2` and `action_filter_alpha` are unchanged from `v0.14.8`.

### Training Intent
- Test whether a guide-only FSM improves `eval_push/success_rate` beyond the `v0.14.6` pure-PPO baseline (`59.59%` at 200 iters, `60.84%` extended).
- Test whether the FSM reduces the dominant `term_height_low_frac` failure mode rather than merely increasing step-like activity.
- Preserve the lower pitch-failure and lower torque-stress profile that `v0.14.8` achieved.

### Interpretation Rule
- If `v0.14.9` beats `v0.14.6`, the guide-FSM branch is justified and can be tuned further.
- If `v0.14.9` still fails to beat `v0.14.6`, the current FSM implementation is not adding enough value and should not be tightened further without improving execution accuracy first.

### Results (v0.14.9)
- Run: `training/wandb/offline-run-20260311_143401-qcfon7tw`
- Checkpoints: `training/checkpoints/staged_v0149/run_20260311_095042/ppo_standing_push_v00149_20260311_143403-qcfon7tw`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0149/run_20260311_095042/ppo_standing_push_v00149_20260311_143403-qcfon7tw/checkpoint_390_51118080.pkl`
- Best @ iter 390: `eval_push/success_rate=38.06%`, `eval_push/episode_length=308.1`
- Train @ iter 390: `success=42.43%`, `ep_len=327.7`
- Clean eval @ iter 390: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- FSM verdict: `weakly engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 57.57% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 55.89% |
| debug/torque_sat_frac | 0.60% |
| ppo/approx_kl | 0.0034 |
| ppo/clip_fraction | 0.0727 |
| debug/bc_phase | 0.258 |
| debug/bc_phase_ticks | 98.982 |
| fsm/swing_occupancy | 3.99% |
| reward/step_event | 0.0012 |
| reward/foot_place | 0.0007 |
| debug/need_step | 0.1637 |
| reward/posture | 0.7528 |

### Diagnosis
- `v0.14.9` failed decisively. It underperformed not only the best pure-PPO baseline (`v0.14.6`: `59.59%` at 200 iters, `60.84%` extended), but also the actor-information run it resumed from (`v0.14.8`: `55.41%`).
- The guide-FSM did not convert need-step pressure into meaningful stepping. `debug/need_step` stayed elevated, but swing occupancy was only `3.99%`, and both `reward/step_event` and `reward/foot_place` collapsed toward zero.
- The controller remained mostly `timeout-driven`, which is the same mechanism failure seen in earlier M3 runs. That strongly suggests the current swing execution / touchdown accuracy is still too weak for the FSM to help, even when PPO retains full override authority.
- Pitch and roll failures stayed at zero, and torque stress remained reasonable. The regression is specifically failed push recovery through `term_height_low`, not general training instability.

### Next Step
- Do **not** continue tuning the current FSM branch. This run is strong evidence that the present FSM implementation is not adding useful recovery value.
- Revert to the best pure-PPO baseline for deployment / continuation:
  - `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- If standing-push stepping research continues later, only revisit FSM after improving execution accuracy first, for example:
  - replace joint-heuristic swing tracking with geometrically correct leg IK, or
  - expose desired foot targets and let PPO learn the leg coordination directly
- Until then, treat the current FSM as disproven for this task setting rather than something that needs more threshold tuning.

## [v0.14.8] - 2026-03-10: Actor-information run with capture-point observation

### Summary
Prepares the next information-first standing-push run after v0.14.7 failed to beat the plain PPO baseline. The critic-only privileged path increased stepping-related rewards but did not improve survival. The next step moves the information to the actor by extending the policy observation contract with a 2-D heading-local capture-point error.

### Motivation
- v0.14.7 showed higher `reward/step_event` and `reward/foot_place` than v0.14.6, so the policy is willing to step more.
- Those extra step events still did not reduce `term_height_low`, which suggests the actor still lacks direct information about where stepping would help.
- Before reintroducing FSM, test whether explicit actor-side stepping cues are enough.

### Code Updates
- Add `env.actor_obs_layout_id` to the config/runtime schema.
- Add a new policy-contract layout `wr_obs_v2`.
- Extend actor observations with `capture_point_error` (2-D heading-local approximate CoM-minus-capture-point offset).
- Pass the selected layout through training startup, env creation, visualization, and export policy-spec generation.

### Observation Contract
- `wr_obs_v1`: existing actor observation layout (resume-safe within prior milestones).
- `wr_obs_v2`: `wr_obs_v1` plus:
  - `capture_point_error[0]`
  - `capture_point_error[1]`

`wr_obs_v2` is a contract-breaking change and requires a fresh run.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.8"`
- switch actor layout:
  - `env.actor_obs_layout_id: wr_obs_v2`
- return critic to symmetric input for isolation:
  - `ppo.critic_privileged_enabled: false`
- keep controllers disabled and disturbance regime unchanged:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`

### Training Intent
- Test whether actor-side capture-point information improves step quality and recovery quality.
- Target outcome: beat the v0.14.6 baseline on `eval_push/success_rate` while reducing `term_height_low`.
- If this still fails to beat v0.14.6, only then consider reintroducing FSM as a guide.

### Run Notes
- Start a fresh run. Do not resume from v0.14.7 or earlier checkpoints because `wr_obs_v2` changes `policy_spec_hash`.

### Results (v0.14.8)
- Run: `training/wandb/offline-run-20260310_203609-l1g2jb6r`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=55.41%`, `eval_push/episode_length=366.7`
- Train @ iter 200: `success=55.52%`, `ep_len=366.6`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 44.48% |
| term_pitch_frac | 6.40% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 57.50% |
| debug/torque_sat_frac | 0.71% |
| ppo/approx_kl | 0.0039 |
| ppo/clip_fraction | 0.0993 |
| reward/step_event | 0.0250 |
| reward/foot_place | 0.0212 |
| debug/need_step | 0.1286 |
| reward/posture | 0.9631 |

### Diagnosis
- Actor-side capture-point information did **not** beat the best plain PPO baseline. `eval_push/success_rate` reached `55.41%`, still below `59.59%` in v0.14.6 and `60.84%` in the later pure-PPO extension.
- Compared with v0.14.7, `v0.14.8` substantially reduced `term_pitch_frac` and torque stress, which means the new actor signal improved recovery quality and action quality.
- But those gains did not translate into higher push success. `reward/step_event` and `reward/foot_place` fell back toward the v0.14.6 range, and `term_height_low` remained the dominant failure mode.
- Clean standing stayed perfect, so the limitation is still specifically push recovery under the calibrated hard-push regime.
- Taken together with v0.14.7, the information-first branch has now been tested on both sides:
  - critic-only information (`v0.14.7`)
  - actor-side capture-point information (`v0.14.8`)
  Neither beat the simpler v0.14.6 baseline.

### Updated Next Step
- The next run should move to the **FSM-as-guide** branch.
- Recommended `v0.14.9` plan:
  - keep `wr_obs_v2` actor observations
  - re-enable `env.fsm_enabled: true`
  - keep `env.base_ctrl_enabled: false`
  - keep full residual override:
    - `env.fsm_resid_scale_stance: 1.00`
    - `env.fsm_resid_scale_swing: 1.00`
    - `env.fsm_resid_scale_recover: 1.00`
  - keep the same `10N x 10` push regime
- Resume from the best v0.14.8 checkpoint if you want the fastest test of whether a guide-FSM can add value on top of the current policy. This is resume-safe as long as `wr_obs_v2` and `action_filter_alpha` stay unchanged.
- The success criterion for that next run is straightforward: beat the v0.14.6 baseline on `eval_push/success_rate` without giving up the lower pitch-failure and lower torque-stress profile that v0.14.8 achieved.

---

## [v0.14.7] - 2026-03-10: Information-first standing push run with privileged critic

### Summary
Prepares the next standing-push run as an information-first follow-up to v0.14.6. Pure PPO at `10N x 10` improved over the bracing baseline, but the extended run plateaued around `60.84%` push success and still failed mainly by `term_height_low`. The next experiment keeps FSM off and actor observations unchanged, while giving the critic sim-only disturbance state for better credit assignment.

### Motivation
- v0.14.6 showed non-zero `reward/step_event` and `reward/foot_place`, so stepping pressure is present.
- The extension did not convert that extra stepping activity into sustained survival gains.
- The next missing ingredient is value-function information, not more controller authority.

### Code Updates
- Add `ppo.critic_privileged_enabled` to the training config/runtime schema.
- Add a fixed-size privileged critic vector to `WildRobotInfo` and rollout storage.
- Feed the critic a separate sim-only input while keeping the actor on the existing policy-contract observation.
- Save `critic_privileged_enabled` in checkpoints and reject incompatible resumes.

### Privileged Critic Features
The critic now sees a 12-D sim-only vector:
- heading-local linear velocity `(vx, vy, vz)`
- heading-local angular velocity `(wx, wy, wz)`
- root attitude `(roll, pitch)`
- root height
- active push force `(fx, fy)`
- push-active flag

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.7"`
- keep pure PPO / no controller assistance:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
- keep the calibrated stepping regime:
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`
- enable information-first critic:
  - `ppo.critic_privileged_enabled: true`

### Training Intent
- Test whether better critic information can turn emerging step events into more useful recovery behavior.
- Keep the actor observation contract unchanged so runtime/export interfaces stay stable.
- If this run still plateaus, only then consider reintroducing FSM as a guide with full residual override.

### Run Notes
- Start a fresh run. This is not a resume-safe follow-up to v0.14.6 because the critic network input shape changes.

### Results (v0.14.7)
- Run: `training/wandb/offline-run-20260310_131556-xsg0sih9`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00147_20260310_131558-xsg0sih9`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00147_20260310_131558-xsg0sih9/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=55.86%`, `eval_push/episode_length=367.6`
- Train @ iter 200: `success=44.72%`, `ep_len=342.8`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 54.77% |
| term_pitch_frac | 21.61% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 62.48% |
| debug/torque_sat_frac | 0.94% |
| ppo/approx_kl | 0.0037 |
| ppo/clip_fraction | 0.0865 |
| reward/step_event | 0.0466 |
| reward/foot_place | 0.0382 |
| debug/need_step | 0.1888 |
| reward/posture | 0.9443 |

### Diagnosis
- The critic-only information-first run did **not** beat the plain PPO baseline. `eval_push/success_rate` regressed from `59.59%` in v0.14.6 and `60.84%` in the v0.14.6 extension to `55.86%` here.
- The best checkpoint was the final checkpoint, so v0.14.7 was still improving at the end of 200 iterations, but it was improving toward a worse ceiling than the simpler baseline.
- `reward/step_event` and `reward/foot_place` both increased relative to v0.14.6, which suggests the privileged critic did help push the policy toward more stepping-like behavior.
- That extra stepping activity did **not** translate into better survival. `term_height_low` remains dominant and `term_pitch_frac` rose materially, which points to poor step quality / recovery quality rather than an absence of step attempts.
- Clean standing remained perfect, so the regression is specific to push recovery, not a general loss of balance.

### Updated Next Step
- Do **not** enable FSM yet. This run does not justify turning controller authority back on.
- The next step should stay in the information-first branch, but move the information to the **actor**, not only the critic:
  - add actor-side capture-point or CoM-velocity error features
  - start a fresh run because this changes the policy observation contract
  - keep `env.fsm_enabled=false` and `env.base_ctrl_enabled=false`
- The goal of that next run is to test whether explicit actor-side stepping information can convert the now-higher step-event/foot-place activity into lower `term_height_low` and higher `eval_push/success_rate`.
- Only if that actor-information run also fails to beat the v0.14.6 baseline should FSM come back, and then only as a guide with full residual override.

---

## [v0.14.6] - 2026-03-08: Pure PPO next run at calibrated stepping regime

### Summary
Replaces the default standing-push baseline with a pure-PPO run at the calibrated stepping regime. The v0.13.9 force sweep showed a bracing ceiling of approximately `9N` for `10` control steps, so the next run should increase disturbance difficulty instead of reducing policy authority. FSM and M2 base control are disabled by default; reward-only stepping terms remain enabled.

### Calibration Result
- Baseline checkpoint: `runtime/bundles/standing_push_v0.13.9_ckpt310/checkpoint.pkl`
- Matching training source checkpoint: `training/checkpoints/ppo_standing_push_auto_confirm_20260305_040129_v00139_20260305_040133-j3l4fc1e/checkpoint_310_40632320.pkl`
- Matching historical config: `training/configs/ppo_standing_push.yaml` @ commit `3a1d1ad`
- Force-sweep result:
  - `8N x 10` -> `74.6%`, `ep_len=431.8`
  - `9N x 10` -> `58.0%`, `ep_len=382.9`
  - `10N x 10` -> `47.3%`, `ep_len=346.7`
  - estimated bracing ceiling: `~9N`
- Dominant failure mode above the ceiling: `term_height_low`

### Diagnosis
- The calibrated stepping regime begins around `9N x 10`, not at some much higher force range.
- `9N x 15` remains a useful harder stress test, but it should not be combined with reduced residual authority while the FSM is still inaccurate.
- The next question is whether pure PPO, given full authority and pushes above the bracing ceiling, will discover stepping on its own.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.6"`
- disable controller assistance by default:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
- keep full residual override authority if FSM is later re-enabled:
  - `env.fsm_resid_scale_stance: 1.00`
  - `env.fsm_resid_scale_swing: 1.00`
  - `env.fsm_resid_scale_recover: 1.00`
- move training pushes to the calibrated stepping regime:
  - `env.push_duration_steps: 10` (from `15`)
  - `env.push_force_min: 10.0` (from `3.0`)
  - `env.push_force_max: 10.0` (from `9.0`)

### Results (v0.14.6)
- Staged run root: `training/checkpoints/staged_v0146/run_20260308_234828`
- Final W&B segment: `training/wandb/offline-run-20260309_052819-npc17xbk`
- Final checkpoint dir: `training/checkpoints/staged_v0146/run_20260308_234828/ppo_standing_push_v00146_20260309_052820-npc17xbk`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0146/run_20260308_234828/ppo_standing_push_v00146_20260309_052820-npc17xbk/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=59.59%`, `eval_push/episode_length=379.0`
- Train @ iter 200: `success=57.27%`, `ep_len=377.1`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 42.73% |
| term_pitch_frac | 4.45% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 69.99% |
| debug/torque_sat_frac | 1.70% |
| ppo/approx_kl | 0.0039 |
| ppo/clip_fraction | 0.0923 |
| reward/step_event | 0.0254 |
| reward/foot_place | 0.0222 |
| debug/need_step | 0.1252 |
| reward/posture | 0.9711 |

### Diagnosis
- Pure PPO improved materially over the calibrated `10N x 10` bracing baseline (`47.3% -> 59.6%`), so the run did create meaningful pressure toward recovery strategies beyond in-place bracing.
- The dominant failure mode remains `term_height_low`; pitch failures are secondary and roll is negligible.
- `reward/step_event` and `reward/foot_place` are small but clearly non-zero, which suggests reward-only stepping behavior is emerging rather than the policy relying purely on the old crouch strategy.
- The best checkpoint is the final checkpoint, and eval_push improved through the last segment (`58.2% -> 59.0% -> 59.3% -> 59.6%` from iters `160 -> 170 -> 190 -> 200`), so the run does not look plateaued yet.
- Torque stress rose compared with the softer v0.13.x baselines but remains within a manageable range.

### Next Step
- Do **not** enable FSM yet. The v0.14.6 result does not justify reintroducing controller authority because pure PPO is still improving at the calibrated stepping regime.
- First extend the same pure-PPO run for another `100-150` iterations from `checkpoint_200_26214400.pkl` and check whether `eval_push/success_rate`, `reward/step_event`, and `reward/foot_place` continue to rise.
- If that extended pure-PPO run plateaus below a useful level, keep FSM off and add **information first**:
  - asymmetric actor-critic / privileged critic
  - capture-point or CoM-velocity information without reducing actor authority
- Only after a plateaued information-first run should FSM be re-enabled, and when it comes back it should be as a **guide** with full residual override (`resid_scale=1.0`), not as a controller that constrains the policy.

### Extended Results (+150 iter resume)
- Resume run: `training/wandb/offline-run-20260309_195021-oij0cbcc`
- Resume checkpoints: `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- Best @ iter 250: `eval_push/success_rate=60.84%`, `eval_push/episode_length=383.1`
- Train @ iter 250: `success=56.86%`, `ep_len=368.3`
- Final @ iter 350: `eval_push/success_rate=54.05%`, `eval_push/episode_length=364.5`
- Clean eval remained perfect throughout the extension (`eval_clean/success_rate=100%`)

| Signal | Best @250 | Final @350 |
|---|---:|---:|
| term_height_low_frac | 39.16% | 45.95% |
| term_pitch_frac | 10.49% | 7.43% |
| tracking/max_torque | 68.00% | 66.58% |
| debug/torque_sat_frac | 1.51% | 1.36% |
| ppo/approx_kl | 0.0045 | 0.0027 |
| ppo/clip_fraction | 0.1047 | 0.0485 |
| reward/step_event | 0.0335 | 0.0369 |
| reward/foot_place | 0.0286 | 0.0311 |
| debug/need_step | 0.1523 | 0.1574 |
| reward/posture | 0.9640 | 0.9614 |

### Updated Diagnosis
- The extra `150` iterations produced only a small peak gain over the iter-200 checkpoint (`59.59% -> 60.84%` at iter `250`), then regressed for the remainder of the run.
- `reward/step_event` and `reward/foot_place` continued to rise, but that did **not** translate into sustained push-survival gains. The policy appears to be stepping more often, but not yet stepping well enough to reduce collapse reliably.
- `term_height_low` remains the dominant failure mode, so the mechanism gap is still recovery quality after disturbance, not merely triggering a foot touchdown.
- The run looks plateaued/noisy now. The best checkpoint arrived early in the extension and the later checkpoints fell back toward the pre-resume level.
- A rollback/LR-backoff signal appeared during the late run, which is another sign that simply running longer on the same setup is no longer buying much.

### Updated Next Step
- Do **not** enable FSM yet. Pure PPO beat the bracing baseline, but this extension does not support turning controller authority back on.
- The next run should be **information-first, still with FSM off**:
  - asymmetric actor-critic / privileged critic
  - add capture-point / CoM-velocity / push-vector information to the critic first
  - keep the actor contract unchanged if resume-safety is still desired
- The goal of that run is to see whether better credit assignment can convert the now-nonzero step events into useful foot placement and lower `term_height_low`.
- Only if the information-first run also plateaus should FSM be enabled, and then only as a **guide** with full residual override (`resid_scale=1.0`) and no residual handicapping.

---

## [v0.14.5] - 2026-03-08: M3 committed stepping after phase-aware FSM fixes

### Summary
Promotes the latest controller/debug fixes into the next training baseline. The major change is that M3 stepping signals are now phase-aware and `need_step` no longer shuts off near a height-collapse event. With those fixes in place, the next run should optimize for more useful catch steps: wider lateral placement, stronger swing tracking, and cleaner touchdown-driven step rewards.

### Results (v0.14.4, reinterpreted after analyzer/controller fixes)
- Run: `training/wandb/offline-run-20260308_095026-0vvxftyt`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00144_20260308_095028-0vvxftyt`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00144_20260308_095028-0vvxftyt/checkpoint_260_34078720.pkl`
- Best @ iter 260: `eval_push/success_rate=56.64%`, `eval_push/episode_length=363.1`
- Train @ iter 260: `success=61.10%`, `ep_len=386.5`
- FSM verdict: `weakly engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 38.90% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 45.67% |
| debug/torque_sat_frac | 0.18% |
| ppo/approx_kl | 0.0038 |
| ppo/clip_fraction | 0.0803 |
| debug/bc_phase | 0.139 |
| fsm/swing_occupancy | 13.86% |
| reward/step_event | 0.0070 |
| reward/foot_place | 0.0048 |
| debug/need_step | 0.1223 |
| reward/posture | 0.7603 |

### Diagnosis
- The earlier `0%` swing conclusion was partly a metrics bug; M3 is weakly engaging, not completely idle.
- The main remaining mechanism gap is step quality, not step triggering alone.
- Posture recovery remains decent, but step events are still mostly timeout-driven and too weak to improve hard-push survival.
- The next iteration should favor more decisive lateral catch steps and stronger swing tracking, not just lower thresholds.

### Code Updates
- `need_step` no longer hard-zeros below `min_height`, so the FSM can still attempt a rescue step near collapse.
- `reward/step_event`, `reward/foot_place`, and touchdown debug signals are now tied to the active swing foot during `SWING`.
- New debug metrics:
  - `debug/bc_in_swing`
  - `debug/bc_in_recover`
- FSM analyzer scripts now use explicit swing occupancy instead of misreading averaged `debug/bc_phase`.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.5"`
- make catch steps wider and less constrained:
  - `env.fsm_y_nominal_m: 0.125` (from `0.115`)
  - `env.fsm_k_lat_vel: 0.20` (from `0.15`)
  - `env.fsm_k_roll: 0.12` (from `0.10`)
  - `env.fsm_y_step_outer_m: 0.22` (from `0.20`)
  - `env.fsm_step_max_delta_m: 0.16` (from `0.12`)
- make swing execution more assertive:
  - `env.fsm_swing_height_m: 0.05` (from `0.04`)
  - `env.fsm_swing_x_to_hip_pitch: 0.35` (from `0.30`)
  - `env.fsm_swing_y_to_hip_roll: 0.40` (from `0.30`)
  - `env.fsm_swing_z_to_knee: 0.55` (from `0.40`)
  - `env.fsm_swing_z_to_ankle: 0.25` (from `0.20`)
- strengthen phase-aware stepping rewards moderately:
  - `reward.step_event: 0.25` (from `0.20`)
  - `reward.foot_place: 0.65` (from `0.50`)
  - `reward.foot_place_sigma: 0.14` (from `0.12`)

### Training Intent
This release is meant to answer the next M3 question directly:
- can the controller turn weak FSM engagement into useful touchdown-driven catch steps?
- can wider lateral placement reduce collapse without giving up the current posture recovery gains?
- can hard-push survival recover while maintaining low torque stress?

### Next-Run Success Signals
- `debug/bc_in_swing` rises above the v0.14.4 baseline
- `reward/step_event` and `reward/foot_place` both rise materially
- `debug/bc_phase_ticks` trends down from timeout-like values
- `eval_push/success_rate` recovers toward or above the v0.14.3 result
- `term_height_low_frac` falls without a major increase in torque saturation

---

## [v0.14.4] - 2026-03-08: M3 engagement tuning after first hard-push run

### Summary
Responds to the first real M3 standing-push run, which showed decent posture recovery but almost no meaningful FSM stepping. The next iteration is tuned to make the step state machine engage earlier and to reduce the policy's ability to survive hard pushes by residual-only bracing or waist/arm compensation.

### Results (v0.14.3)
- Run: `training/wandb/offline-run-20260307_235137-un260w81`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00143_20260307_235139-un260w81`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00143_20260307_235139-un260w81/checkpoint_240_31457280.pkl`
- Best @ iter 240: `eval_push/success_rate=63.77%`, `eval_push/episode_length=386.1`
- Train @ iter 240: `success=67.36%`, `ep_len=405.8`
- FSM verdict: `not meaningfully engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 32.64% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 55.97% |
| debug/torque_sat_frac | 0.68% |
| ppo/approx_kl | 0.0038 |
| ppo/clip_fraction | 0.0912 |
| debug/bc_phase | 0.049 |
| debug/bc_phase_ticks | 204.847 |
| fsm/swing_occupancy | 0.00% |
| reward/step_event | 0.0071 |
| reward/foot_place | 0.0046 |
| debug/need_step | 0.1070 |
| reward/posture | 0.7575 |

### Diagnosis
- M3 posture recovery is working better than M3 stepping.
- The dominant failure remains collapse (`term_height_low_frac`), not pitch/roll loss.
- The residual policy still appears able to survive too much of the push without committing to useful `SWING` phases.
- Waist/arm compensation is likely helping mask poor foot placement rather than supporting it.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.4"`
- tune the FSM to engage earlier under hard pushes:
  - `env.fsm_trigger_threshold: 0.30` (from `0.45`)
  - `env.fsm_recover_threshold: 0.15` (from `0.20`)
  - `env.fsm_trigger_hold_ticks: 1` (from `2`)
  - `env.fsm_swing_timeout_ticks: 10` (from `12`)
- reduce residual masking so the step controller must contribute:
  - `env.fsm_resid_scale_stance: 0.55` (from `0.75`)
  - `env.fsm_resid_scale_swing: 0.45` (from `0.60`)
  - `env.fsm_resid_scale_recover: 0.50` (from `0.70`)
- weaken arm compensation so foot placement, not upper-body damping, carries recovery:
  - `env.fsm_arm_need_step_threshold: 0.45` (from `0.35`)
  - `env.fsm_arm_k_roll: 0.06` (from `0.10`)
  - `env.fsm_arm_k_roll_rate: 0.03` (from `0.05`)
  - `env.fsm_arm_k_pitch_rate: 0.02` (from `0.03`)
  - `env.fsm_arm_max_delta_rad: 0.18` (from `0.25`)

### Training Intent
This release is meant to answer a narrower second M3 question:
- can the FSM be made to engage reliably under hard pushes?
- can step events become touchdown-driven instead of timeout-driven?
- can the robot keep the current posture recovery gains while using the feet more decisively?

### Next-Run Success Signals
- `debug/bc_phase` shows clear `SWING` occupancy during disturbed episodes
- `reward/step_event` rises materially above the v0.14.3 baseline
- `reward/foot_place` rises with step events
- `debug/bc_phase_ticks` shortens away from timeout-like values
- `eval_push/success_rate` improves from the v0.14.3 result without a major torque spike

---

## [v0.14.3] - 2026-03-08: M3 training-ready standing-push baseline

### Summary
Promotes the standing-push baseline from "M3 code present" to "M3 training-ready". The default standing-push config now trains with the foot-placement FSM enabled, M2 disabled, hard pushes restored, and residual authority reduced enough that the FSM must do real recovery work instead of being masked by the policy residual.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.3"`
- enable M3 by default:
  - `env.fsm_enabled: true`
  - `env.base_ctrl_enabled: false`
- restore hard-push curriculum for the first real M3 run:
  - `env.push_duration_steps: 15`
  - `env.push_force_max: 9.0`
- reduce residual authority so the FSM must contribute meaningful stepping:
  - `env.fsm_resid_scale_swing: 0.60`
  - `env.fsm_resid_scale_stance: 0.75`
  - `env.fsm_resid_scale_recover: 0.70`

### Training Intent
This release is meant to answer the first M3 question directly:
- does the robot step under hard pushes?
- does the step enlarge support meaningfully instead of fidgeting?
- does the robot return toward upright instead of remaining in a crouched recovery pose?

### First-Run Review Signals
- `debug/bc_phase` should show real occupancy in `SWING`, not only `STANCE`
- `reward/step_event` and `reward/foot_place` should rise in disturbed episodes
- `debug/bc_phase_ticks` should not be dominated by swing timeouts
- `eval_push/survival_rate` should at least match M2, with clearer stepping behavior
- `reward/posture` should recover after the disturbance rather than staying near zero

### Notes
- M2 config fields remain in the file for rollback/comparison, but are disabled in the training-ready default.
- Deprecated gait-style reward terms remain debug-only for now and should be removed after the first validated M3 run if they are not needed.

---

## [v0.14.2] - 2026-03-07: M3 — foot-placement FSM base controller + waist arm damping

### Summary
Implements M3 from `training/docs/step_trait_base_controller_design.md`: an explicit 3-phase step state machine (STANCE / SWING / TOUCHDOWN_RECOVER) replaces the M2 joint-heuristic controller. The FSM uses Raibert-style foot placement and a half-sine swing trajectory plus waist arm-momentum compensation. Disabled by default; enable with `env.m3_enabled: true`.

### New Files
- `training/envs/step_controller.py` (~400 lines): Pure-JAX M3 module.
  - `compute_need_step()` — gate [0,1] from pitch/roll/lateral-vel/pitch-rate
  - `select_swing_foot()` — load-based + lateral bias
  - `compute_step_target()` — Raibert-style target, clamped to step bounds
  - `compute_swing_trajectory()` — smoothstep XY + half-sine Z arc
  - `update_fsm()` — fully vectorised FSM transitions (no Python control flow)
  - `compute_ctrl_base()` — stance uprightness feedback + swing tracking + waist damping
- `tests/test_step_controller.py` — 26 unit tests, **all passing** (1.82s CPU JAX)

### Code Updates
- `training/envs/env_info.py`: 9 new FSM state fields in `WildRobotInfo` (`fsm_phase`, `fsm_swing_foot`, `fsm_phase_ticks`, `fsm_frozen_tx/ty`, `fsm_swing_sx/sy`, `fsm_touch_hold`, `fsm_trigger_hold`)
- `training/configs/training_runtime_config.py`: 25 new `m3_*` fields in `EnvConfig` (`m3_enabled`, trigger/recovery thresholds, hold ticks, nominal step size, Raibert gains, step bounds, swing trajectory params, residual authority per phase, arm gains)
- `training/envs/wildrobot_env.py`:
  - `_make_initial_state()`: FSM fields initialised to 0 (STANCE)
  - `step()`: 3-branch action pipeline (m3 / m2 / passthrough); FSM tuple carried in all branches
  - `new_wr_info` / `preserved_wr_info`: FSM fields in both `lax.cond` branches (pytree safety)
  - `WildRobotEnv._m3_compute_ctrl()`: new ~200-line method wiring FSM into step loop
  - 3 new debug metrics: `debug/bc_phase`, `debug/bc_swing_foot`, `debug/bc_phase_ticks`
- `training/core/metrics_registry.py`: MetricSpec indices 60-62 for new debug metrics
- `training/core/experiment_tracking.py`: initial values added to ENV_METRICS_KEYS and both `get_initial_env_metrics()` functions

### Config Updates (`training/configs/ppo_standing_push.yaml` → v0.14.2)
- `version: "0.14.2"`
- M3 config block added with `m3_enabled: false` and all 25 parameters at design-doc defaults

### Verified
```
26/26 unit tests passing (1.82s CPU JAX)
Smoke test (reset + 5 steps M3=off, reset + 20 steps M3=on): PASSED
  M3 disabled: bc_phase=0.000, reward/step_event flowing
  M3 enabled:  fsm_phase=0 (STANCE), fsm_phase_ticks=20, metrics plumbed correctly
```

### Activation
To activate M3 in training:
```yaml
env:
  m3_enabled: true
  base_ctrl_enabled: false   # M3 replaces M2
```

---

## [v0.14.1] - 2026-03-07: Standing pushes (M2: joint-heuristic base + residual gating)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.14.1"` and enable M2 mixing:
  - `env.base_ctrl_enabled: true`
  - base feedback gains: `env.base_ctrl_*`
  - residual gate: `env.residual_scale_{min,max}` + `env.residual_gate_power`
- `training/configs/ppo_standing_push.yaml`: reduce base/residual authority defaults to lower action clipping (observed high `action_sat` during pushes):
  - lower `env.base_ctrl_*` gains and `env.base_ctrl_action_clip`
  - lower `env.residual_scale_max`

### Code Updates
- `training/configs/training_runtime_config.py`: add `EnvConfig` knobs for M2 base controller + residual gating.
- `training/configs/training_config.py`: parse new `env.base_ctrl_*` and `env.residual_*` fields from YAML.
- `training/envs/wildrobot_env.py`: implement M2 action mixing:
  - `action_applied = action_base(default_pose + conservative pitch/roll feedback) + residual_scale(need_step) * policy_action`
  - residual gate uses the same `need_step` computation as stepping-trait rewards (tilt + lateral velocity + pitch rate)

### Notes
- No policy contract changes (action/obs dims unchanged). Existing checkpoints should still resume; the controller is purely an environment-side action transformation controlled by config.

### Results (smoke + hard confirm eval @500)
- Baseline checkpoint for M3 (handoff): `training/checkpoints/ppo_standing_push_v00141_20260307_182350/checkpoint_210_430080.pkl`
- Hard confirm eval summary (iter 210, eval horizon 500):
  - `eval_push`: success=86.2%, ep_len=456.0, term_h_low=13.8% (dominant failure mode)
  - `eval_clean`: success=99.2%, ep_len=496.7, term_h_low=0.8%

## [v0.13.11] - 2026-03-07: Standing pushes (step trait via touchdown + foot placement)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.11"`.
- `training/configs/ppo_standing_push.yaml`: enable stepping-trait shaping (all gated by `debug/need_step` to avoid marching-in-place):
  - add `reward_weights.gait_periodicity`, `hip_swing`, `knee_swing`, and `_min` thresholds
  - add `reward_weights.step_event` (touchdown reward) and `reward_weights.foot_place` (foot placement reward)
  - add `reward_weights.foot_place_*` coefficients and `reward_weights.step_need_*` gates

### Code Updates
- `training/envs/env_info.py`: add `prev_left_loaded` / `prev_right_loaded` fields for contact transition (touchdown) detection.
- `training/envs/wildrobot_env.py`: add step-trait rewards:
  - `reward/step_event`: touchdown event reward (gated by `need_step`)
  - `reward/foot_place`: Raibert-style foot placement reward at touchdown (gated by `need_step`)
  - log `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right`
- `training/core/experiment_tracking.py`: include new reward/debug keys in reset metrics dict (JAX scan carry stability).
- `training/core/metrics_registry.py`: append new metrics (append-only) for W&B logging: `reward/step_event`, `reward/foot_place`, `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right`.
- `wildrobot/agents/training_loop_agent.py`: heuristic advisor now tunes stepping weights and can relax `env.min_height` (bounded) when height-low dominates, to allow deep-crouch recovery + stepping.

### Milestone M1 implementation summary

This release completes **Milestone M1: Reward-only stepping trait** from `training/docs/step_trait_base_controller_design.md`.

**What's implemented:**
1. **Touchdown detection** — `prev_left_loaded` / `prev_right_loaded` stored in `WildRobotInfo` each step; contact transitions detected as `(~prev_loaded) & loaded` per foot.
2. **Need-to-step gate** — `need_step ∈ [0,1]` computed from `|pitch|/g_pitch + |roll|/g_roll + |lat_vel|/g_lat + |pitch_rate|/g_pr) / 4.0`, then multiplied by `healthy`. Prevents marching-in-place.
3. **Foot placement reward** (Raibert-style) — `exp(-placement_err/sigma²)` at touchdown, gated by `need_step`. Lateral target adjusted by lateral velocity and roll; forward target by forward_vel and pitch.
4. **Config knobs** — all `step_event`, `foot_place`, `foot_place_sigma/k_*`, `step_need_*`, `gait_periodicity`, `hip_swing/knee_swing*` exposed via `RewardWeightsConfig` and parsed from YAML.
5. **Metrics** — `reward/step_event`, `reward/foot_place`, `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right` in registry (indices 55-59) and in initial-metrics dict (zero at reset, correct during step).
6. **No obs/action/contract changes** — `action_filter_alpha` unchanged; policy contract hash stable.

**Verified (CPU JAX smoke):**
```
reward/step_event:   0.1480  (non-zero, gated by need_step)
reward/foot_place:   0.1469  (non-zero at touchdown)
debug/need_step:     0.1480  (non-zero when disturbed)
debug/touchdown_left:  0.0000
debug/touchdown_right: 1.0000  (right foot touchdown from home keyframe)
```

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Resume stepping curriculum from latest best checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume <best_ckpt.pkl>`

**Acceptance criteria (how to verify during training):**
- `debug/need_step > 0` during push disturbances.
- `reward/step_event > 0` and `debug/touchdown_*` become non-zero in pushed runs.
- `reward/foot_place > 0` (not always zero) at touchdown events.
- Existing `eval_push/*`, `eval_clean/*`, and survival metrics remain intact.

## [v0.13.10] - 2026-03-07: Standing pushes (step recovery + posture return)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.10"`.
- `training/configs/ppo_standing_push.yaml`: make stepping recovery more accessible under pushes:
  - `reward_weights.clearance: 0.05 -> 0.10`
  - `reward_weights.flight_phase_penalty: -0.1 -> 0.0`
- `training/configs/ppo_standing_push.yaml`: add posture-return shaping (encourage returning to default pose once upright):
  - `reward_weights.posture: 0.3`
  - `reward_weights.posture_sigma: 0.40`
  - `reward_weights.posture_gate_pitch: 0.35`
  - `reward_weights.posture_gate_roll: 0.35`

### Code Updates
- `training/envs/wildrobot_env.py`: add `reward/posture` shaping term (gated by uprightness) and log `debug/posture_mse`.
- `training/configs/training_runtime_config.py` + `training/configs/training_config.py`: add/parse new posture shaping knobs under `reward_weights`.

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Fine-tune from best v0.13.9 checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume <best_ckpt.pkl>`

## [v0.13.9] - 2026-03-04: Standing pushes (reduce height-low + pitch terminations)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.9"`.
- `training/configs/ppo_standing_push.yaml`: strengthen pre-collapse shaping to reduce `term_height_low_frac`:
  - `env.collapse_height_buffer: 0.02 -> 0.03`
  - `env.collapse_vz_gate_band: 0.05 -> 0.07`
  - `reward_weights.collapse_height: -0.3 -> -0.5`
  - `reward_weights.collapse_vz: -0.2 -> -0.3`
- `training/configs/ppo_standing_push.yaml`: improve recovery to reduce pitch-limit failures:
  - `reward_weights.orientation: -3.0 -> -4.0`
  - keep `env.action_filter_alpha: 0.6` to allow resuming from v0.13.8 checkpoints (part of `policy_spec_hash`)

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Fine-tune from v0.13.8 best checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3/checkpoint_80_10485760.pkl`

---

## [v0.13.8] - 2026-03-04: Standing stability under pushes (collapse prevention + dual eval)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.8"` and tune standing stability settings.
- `training/configs/ppo_standing_push.yaml`: add pre-collapse shaping knobs under `env`:
  - `collapse_height_buffer`, `collapse_height_sigma`, `collapse_vz_gate_band`
- `training/configs/ppo_standing_push.yaml`: add collapse-prevention reward terms under `reward_weights`:
  - `collapse_height`, `collapse_vz`
- `training/configs/ppo_standing_push.yaml`: increase `ppo.kl_lr_backoff_multiplier` (`2.0 -> 3.0`) to avoid immediate KL-triggered LR backoff at iter 1.

### Code Updates
- `training/envs/wildrobot_env.py`: add two pre-collapse penalties:
  - `reward/collapse_height_pen`: quadratic penalty when height drops below `(min_height + buffer)`
  - `reward/collapse_vz_pen`: downward vertical velocity penalty gated near `min_height`
- `training/configs/training_runtime_config.py` + `training/configs/training_config.py`: add/parse new env knobs and reward weights for collapse shaping.
- `training/core/metrics_registry.py`: append `reward/collapse_height_pen` and `reward/collapse_vz_pen` (append-only registry).
- `training/core/experiment_tracking.py`: include new reward terms in schema/init + log filtering for `eval_push/*` and `eval_clean/*`.
- `training/core/training_loop.py` + `training/train.py`: run and log dual deterministic eval passes:
  - `eval_push/*` (pushes enabled, baseline robustness signal)
  - `eval_clean/*` (pushes disabled, clean policy quality signal)
  - rollback/best-checkpoint comparisons are now explicitly driven by `eval_push/success_rate`.
- Torque saturation metrics remain preferred stress signals (`debug/torque_sat_frac`, `debug/torque_abs_max`); action saturation metrics are retained as legacy diagnostics.

### Results (v0.13.8)
- Run: `training/wandb/offline-run-20260303_195741-6y6ufyw3`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3/checkpoint_80_10485760.pkl`
- Best @ iter 80: eval_push/success_rate=98.44%, eval_push/episode_length=495.3
- Clean @ iter 80: eval_clean/success_rate=100.00%, eval_clean/episode_length=500.0
- Train @ iter 80: success=95.24%, ep_len=487.5
- Final @ iter 200: eval_push/success_rate=98.44%, eval_push/episode_length=494.6

| Signal | Value |
|---|---:|
| term_height_low_frac | 4.76% |
| term_pitch_frac | 0.73% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 59.28% |
| debug/torque_sat_frac | 0.81% |
| ppo/approx_kl | 0.0064 |
| ppo/clip_fraction | 0.1733 |

---

## [v0.13.6] - 2026-03-03: Standing robustness (strong pushes + PPO stability guardrails)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump training version to v0.13.6.
- `training/configs/ppo_standing_push.yaml`: enable two-sided height target shaping (`env.height_target_two_sided: true`) to actively reward exploring lower stance near `env.target_height`.
- `training/configs/ppo_standing_push.yaml`: tighten PPO update aggressiveness for stability under pushes:
  - `ppo.learning_rate: 3e-4 → 1e-4`, `ppo.epochs: 4 → 2`, `ppo.entropy_coef: 0.01 → 0.005`
  - `ppo.clip_epsilon: 0.2 → 0.15`
- `training/configs/ppo_standing_push.yaml`: add P0/P1 stability guardrails (PPO KL + schedules + deterministic eval + rollback):
  - `ppo.target_kl`, `ppo.kl_*`, `ppo.lr_schedule_end_factor`, `ppo.entropy_schedule_end_factor`
  - `ppo.eval.*` (periodic deterministic eval)
  - `ppo.rollback.*` (rollback-on-regression with LR backoff)
- `training/configs/training_runtime_config.py`: add `PPOEvalConfig` + `PPORollbackConfig` under `PPOConfig`.
- `training/configs/training_config.py`: parse `ppo.eval` and `ppo.rollback` blocks from YAML.

### Code Updates
- `training/core/training_loop.py`: stabilize PPO minibatch sampling (one permutation per epoch; contiguous minibatches).
- `training/core/training_loop.py`: add in-update KL early-stop (halts remaining minibatches within the current PPO update).
- `training/core/training_loop.py`: add deterministic periodic eval (fixed RNG across eval checkpoints) for monotonic checkpoint comparisons.
- `training/core/training_loop.py`: add rollback-on-regression using eval metrics + LR backoff scale.
- `training/core/training_loop.py`: switch to optax LR schedule inside `optax.adam(...)` (instead of gradient pre-scaling); add `ppo/active_updates` + `ppo/epochs_used` metrics for diagnostics and correct LR logging (including resume).
- `training/train.py`: plumb separate eval reset/step functions (allow different eval batch size).
- `training/core/experiment_tracking.py`: log `ppo/*` and `eval/*` metrics to W&B/offline logs.
- `training/tests/test_training_stability_controls.py`: add unit coverage for the stability controls.

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. `uv run python training/train.py --config training/configs/ppo_standing_push.yaml`

---

## [v0.12.2] - 2026-01-05: Standing robustness (moderate pushes)

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.12.2.
- `training/configs/ppo_standing.yaml`: increase push forces to 2–5 N and keep 6-step pushes.
- `training/configs/training_runtime_config.py`: add IMU noise + latency knobs (training-only).

### Code Updates
- `training/envs/wildrobot_env.py`: apply IMU noise + latency before observation build (training-only).

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
3. `uv run python training/train.py --config training/configs/ppo_standing.yaml`

### Results (Standing PPO, v0.12.2)
- Run: `training/wandb/offline-run-20260108_212208-r9wr2rq5/`
- Checkpoints: `training/checkpoints/ppo_standing_v00122_20260108_212209-r9wr2rq5/`
- Best checkpoint (reward): `training/checkpoints/ppo_standing_v00122_20260108_212209-r9wr2rq5/checkpoint_50_6553600.pkl`
- Run length: 330 iterations (final step: 43,253,760)

| Metric | Value (best @ iter 50) |
|--------|-------------------------|
| Episode Reward | 635.17 |
| Episode Length | 500.0 |
| Forward Velocity (cmd=0.00) | -0.00 m/s |
| Robot Height | 0.468 m |

---

## [v0.12.1] - 2026-01-05: Standing policy_contract Baseline (v0.12.1)

### Contract Migration
- Actor observation/action semantics now flow through `policy_contract` (shared JAX/NumPy implementation + parity tests).
- Resume is now guarded by a `policy_spec_hash` stored in checkpoints to prevent silently resuming with a different policy contract.
- PPO actor initialization is biased to the **home keyframe pose** (prevents early drift/falls before learning).

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.12.1.
- `policy_contract`: remove actor linvel from `wr_obs_v1` (linvel stays privileged-only for reward/metrics).
- `training/configs/*.yaml`: remove `env.linvel_mode` / `env.linvel_dropout_prob` (no actor linvel to mask).
- `training/configs/ppo_standing.yaml`: set `wandb.mode: offline` by default (switch to `online` when desired).
- `training/configs/ppo_standing.yaml`: lower `env.action_filter_alpha` to `0.6` (faster corrections, less lag).

### Notes
- Expected `obs_dim` for v0.12.1 standing: `36` (actor linvel removed from `wr_obs_v1`).
- Fresh runs only: resuming from pre-contract checkpoints is blocked by `policy_spec_hash`.

### Plan
1. Validate assets + environment:
   `uv run python scripts/validate_training_setup.py`
2. Quick smoke test (no resume):
   `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
3. Full standing run from scratch (no resume):
   `uv run python training/train.py --config training/configs/ppo_standing.yaml`

---

## [v0.11.5] - 2026-01-04: Standing Sim2Real Prep (v0.11.5)

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.11.5.
- `training/configs/ppo_standing.yaml`: document resume command for `training/checkpoints/ppo_standing_v00113_final.pkl`.

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing.yaml --resume training/checkpoints/ppo_standing_v00113_final.pkl`

---

## [v0.11.4] - 2026-01-01: Walking Conservative Warm Start (v0.11.4)

### Plan
1. Validate assets + config:
   `uv run python scripts/validate_training_setup.py`
2. Walking warm-start (conservative range):
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume training/checkpoints/ppo_standing_v00113_final.pkl`
3. Full run:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --resume training/checkpoints/ppo_standing_v00113_final.pkl`

### Base Checkpoint
- `training/checkpoints/ppo_standing_v00113_final.pkl` (v0.11.3 standing best @ iter 310)

### Final Results (2026-01-03, run-20260102_220919-xiqcenuh)
- Run: `training/wandb/run-20260102_220919-xiqcenuh/`
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/`
- Iterations: 690 → 880 (115M total steps)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Episode Length | 474 ± 33 steps | > 400 | ✓ PASS |
| Forward Velocity | 0.33 ± 0.01 m/s | 0.5-1.0 m/s | ✗ BELOW |
| Velocity Tracking Error | ~0.03 m/s | < 0.2 m/s | ✓ PASS |
| Robot Height | 0.458 m | ~0.46 m | ✓ PASS |
| Fall Rate | ~5% | < 5% | Borderline |

- Best checkpoints (99%+ survival):
  - `checkpoint_720_94371840.pkl`: reward=498.2, ep_len=500, vel=0.32 m/s
  - `checkpoint_700_91750400.pkl`: reward=495.8, ep_len=500, vel=0.32 m/s
- Highest reward: `checkpoint_730_95682560.pkl` (reward=509.2, ep_len=457, vel=0.34 m/s)

### Analysis (Final, 2026-01-03)

**What Worked:**
1. **Stable Conservative Walking**: Policy learned stable walking at ~0.33 m/s within velocity range [0.1, 0.6].
2. **Upright Posture**: Height maintained at ~0.458m (target 0.46m), height shaping rewards effective.
3. **Good Velocity Tracking**: Tracking error ~0.03 m/s when cmd≈0.35 m/s.

**Issues Identified:**
1. **Velocity Plateau**: Policy converged to ~0.33 m/s, below 0.5 m/s minimum target.
   - Likely cause: conservative reward balance prioritizes stability over speed.
2. **Stochastic Dependency**: Policy requires stochastic sampling for reliable deployment:
   - `--stochastic`: 100% success (5/5 episodes)
   - `--deterministic`: 20% success (1/5 episodes)
   - Cause: Policy learned to use exploration noise for balance recovery.
3. **Yaw Drift**: Episodes that fail often show significant heading drift before falling.
4. **Visualization Bug Fixed**: `prev_action` was incorrectly initialized to zeros instead of default pose action in `visualize_policy.py`, causing early falls.

### Code Fixes (2026-01-03)
- Fixed `visualize_policy.py`: Initialize `prev_action` to `cal.ctrl_to_policy_action(cal.get_ctrl_for_default_pose())` instead of zeros.
- Fixed `cal.py`: Handle both MJX and native MuJoCo data in `_get_geom_contact_force()`.

### Visualization Commands
```bash
# Recommended (stochastic - matches training)
uv run mjpython training/eval/visualize_policy.py \
  --checkpoint training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/checkpoint_720_94371840.pkl \
  --config training/configs/ppo_walking_conservative.yaml \
  --stochastic

# Deterministic (lower success rate)
uv run mjpython training/eval/visualize_policy.py \
  --checkpoint training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/checkpoint_720_94371840.pkl \
  --config training/configs/ppo_walking_conservative.yaml
```

### Next Steps (v0.11.5)
1. **Increase velocity**: Boost `tracking_lin_vel` weight or raise `min_velocity` to push past 0.33 m/s plateau.
2. **Deterministic robustness**: Add entropy regularization or train with periodic deterministic evaluation.
3. **Reduce yaw drift**: Increase `angular_velocity` penalty or add explicit heading tracking reward.
4. **Curriculum learning**: Gradually increase velocity target during training.

### Earlier Results (v0.11.4)

#### Status Update (2026-01-02, Walking PPO Conservative Resume)
- Run: `training/wandb/run-20260102_143302-xmislxx8/` (resumed from iter 590)
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260102_143303-xmislxx8/`
- Resumed from: `training/checkpoints/ppo_walking_conservative_v00114_20260102_083850-lq6k40oo/checkpoint_590_77332480.pkl`
- Best checkpoint (reward): `checkpoint_640_83886080.pkl` (reward=490.65, vel=0.33 @ cmd=0.35, ep_len=468.5)
- Topline (final @690): reward=476.46, ep_len=453.6, success=80.3%, vel=0.32 @ cmd=0.35, vel_err=0.057, max_torque=0.783
- Terminations (final @690): truncated=80.3%, pitch=18.4%, roll=1.3%, height_low/high=0.0%

#### Plan Update (2026-01-03, Height Shaping + Posture)
- Goal: Reduce squat-walk while keeping conservative walking stable.
- (Historical) used sim linvel; actor masking was not applied in this run.
- Change (code): height shaping is now one-sided (no penalty above `env.target_height`) in `training/envs/wildrobot_env.py`.
- Change (config): set `env.target_height: 0.46` (waist/root height) in `training/configs/ppo_walking_conservative.yaml`.

#### Config + Code Updates (post-run)
- Reduced action lag for faster pitch correction: `env.action_filter_alpha: 0.5 → 0.2` in `training/configs/ppo_walking_conservative.yaml`
- Prioritize straight + upright gait:
  - `reward_weights.angular_velocity: -0.05 → -0.2` (stronger yaw-rate penalty)
  - `reward_weights.orientation: -0.5 → -1.0` (stronger pitch/roll penalty)
  - Added `reward_weights.height_target: 0.2` (reduce squat) with `height_target_sigma: 0.05`
- Fix: resolved `NameError: raw_action_abs_max` by explicitly passing `raw_action` into `_get_reward()` for debug metrics (`training/envs/wildrobot_env.py`)

#### Results (2026-01-01, Walking PPO Conservative Warm Start)
- Run: `training/wandb/run-20260101_215853-5e15gkam`
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260101_215854-5e15gkam/`
- Best checkpoint (reward): `checkpoint_430_56360960.pkl` (reward=440.31, vel=0.31 @ cmd=0.35, ep_len=466.4)
- Topline (final @610): reward=431.74, ep_len=474.6, success=88.9%, vel=0.32 @ cmd=0.35, vel_err=0.079, max_torque=0.869
- Notes: Good conservative walking found; next run should continue v0.11.4 (no config changes) from best checkpoint to push success rate toward 95%+

---

## [v0.11.3] - 2025-12-31: Foot Switches + Standing Retrain Plan

### Plan
1. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)
- Switched foot contact signals to boolean foot switches (toe/heel), obs dim now 39
- Removed `*_touch` sites/sensors from `assets/wildrobot.xml`; switch signal derived from geom forces
- Added height target shaping + stance width penalty to discourage squat posture

### Results (Standing PPO, v0.11.3)
- Run: `training/wandb/run-20260101_182859-2wsk4xt7`
- Resumed from: `checkpoint_260_34078720.pkl` (reward=518.32, ep_len=498)
- Best checkpoint: `training/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_310_40632320.pkl`
- Final checkpoint: `training/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_360_47185920.pkl`
- Summary (best @310): reward=574.31, ep_len=500, height=0.454m, vel=-0.02m/s (cmd=0.00), vel_err=0.076, torque~84.9%
- Terminations: 0% across all categories at best window; stable 500-step episodes
- Notes: Ready to warm-start walking with conservative velocity range

---

## [v0.11.2] - 2025-12-31: Upright Standing Retrain Plan

### Plan
1. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)

---

## [v0.11.1] - 2025-12-31: CAL PPO Walking Smoke Test

### Test Plan
1. `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume training/checkpoints/ppo_standing_v0110_170.pkl`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_walking.yaml --checkpoint <path>`

### Results
- Run: `training/wandb/run-20251231_003248-8p2bv838`
- Best checkpoint: `training/checkpoints/ppo_walking_conservative_v00111_20251231_003249-8p2bv838/checkpoint_360_47185920.pkl`
- Summary: reward ~426.79, ep_len ~491, success ~95.4%, vel ~0.33 m/s
- Notes: stable but squat-biased posture inherited from standing checkpoint; hip swing limited

---

## [v0.11.0] - 2025-12-30: Control Abstraction Layer (CAL) - BREAKING CHANGE

### Overview

Major architectural change introducing a **Control Abstraction Layer (CAL)** to decouple high-level flows (training, testing, sim2real) from MuJoCo primitives.

⚠️ **BREAKING CHANGE**: This version requires retraining from scratch. Existing checkpoints are incompatible due to action semantics changes.

### Why CAL?

The previous implementation had critical issues:
1. **Direct MuJoCo Primitive Access**: Multiple layers directly manipulated `data.ctrl`, `data.qpos`, etc.
2. **Ambiguous Actuator Semantics**: Policy outputs [-1, 1] were passed directly to MuJoCo ctrl (which expects radians)
3. **Incorrect Symmetry Assumptions**: Left/right hip joints have mirrored ranges, but code didn't account for this
4. **Training Learning Simulator Quirks**: Policies learned MuJoCo-specific behaviors instead of transferable motor skills

### New Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HIGH-LEVEL LAYER                           │
│  Training Pipeline │ Test Suite │ Sim2Real │ Policy Inference   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTROL ABSTRACTION LAYER (CAL)                    │
│                                                                 │
│  • policy_action_to_ctrl() - normalized [-1,1] → radians        │
│  • physical_angles_to_ctrl() - radians → ctrl (no normalization)│
│  • get_normalized_joint_positions() - for policy observations   │
│  • get_physical_joint_positions() - for logging/debugging       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOW-LEVEL LAYER (MuJoCo)                      │
│                  data.ctrl, data.qpos, data.qvel                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

#### Two-Method API Pattern
- `policy_action_to_ctrl()`: For policy outputs [-1, 1], always applies symmetry correction
- `physical_angles_to_ctrl()`: For GMR/mocap angles (radians), no symmetry correction

#### Symmetry Correction
- `mirror_sign` field handles joints with inverted ranges (e.g., left/right hip pitch)
- Automatic correction ensures symmetric actions produce symmetric motion

#### Sim2Real Ready
- `ControlCommand` dataclass with target_position and target_velocity
- `VelocityProfile` enum for trajectory generation (STEP, LINEAR, TRAPEZOIDAL, S_CURVE)
- Clean separation between simulation and deployment

#### Default Positions from MuJoCo
- Home pose loaded from MuJoCo keyframe at runtime (single source of truth)
- Recommendation: Use `keyframes.xml` for dedicated keyframe management

### New Files

| File | Description |
|------|-------------|
| `training/control/__init__.py` | Export CAL, specs, types |
| `training/control/cal.py` | ControlAbstractionLayer class |
| `training/control/specs.py` | JointSpec, ActuatorSpec, ControlCommand |
| `training/control/types.py` | ActuatorType, VelocityProfile enums |
| `training/tests/test_cal.py` | CAL unit tests |
| `assets/keyframes.xml` | Dedicated keyframe file (home, t_pose) |

### Files Updated

| File | Change |
|------|--------|
| `training/envs/wildrobot_env.py` | Integrate CAL |
| `assets/robot_config.py` | Robot configuration management (moved from training/configs/) |
| `assets/mujoco_robot_config.json` | Add actuated_joints section |
| `assets/post_process.py` | Add generate_actuated_joints_config() |

### Config Schema Update

New `actuated_joints` section in `mujoco_robot_config.json`:

```yaml
actuated_joints:
  - name: left_hip_pitch
    type: position
    class: htd45hServo
    range: [-0.087, 1.571]
    symmetry_pair: right_hip_pitch
    mirror_sign: 1.0
    max_velocity: 10.0
    # NOTE: default_pos loaded from MuJoCo keyframe at runtime
```

### Migration

1. **Retrain from scratch** - existing checkpoints are incompatible
2. Update `mujoco_robot_config.json` with `actuated_joints` section
3. Create `assets/keyframes.xml` with home pose
4. Replace direct MuJoCo access with CAL methods

### Documentation

See `training/docs/CONTROL_ABSTRACTION_LAYER_PROPOSAL.md` for full design document.

### Results (Standing PPO, v0.11.0)
- Config: `training/configs/ppo_standing.yaml`
- Run: `training/wandb/run-20251230_181458-27832na7`
- Best checkpoint: `training/checkpoints/ppo_standing_v00110_20251230_181500-27832na7/checkpoint_170_22282240.pkl`
- Summary: ep_len ~484, height ~0.434 m, success ~89.8%, vel ~0.01 m/s
- Notes: success rate below 95% target; height_low terminations ~10% late in training


## [v0.10.6] - 2025-12-28: Hip/Knee Swing Reward (Stage 1)

### Config Updates
- Added swing-gated hip/knee rewards to encourage leg articulation during swing.
- Added small flight-phase penalty to discourage hopping.
- Kept gait periodicity reward (alternating support).
- Maintained v0.10.5 tuning for velocity/orientation/torque.

### Base Checkpoint (v0.10.5)
- `training/checkpoints/wildrobot_ppo_20251228_205536/checkpoint_520_68157440.pkl`

### Training Results (v0.10.5 → v0.10.6 prep)
- v0.10.5 best: reward ~566.42 (iter 520), vel ~0.63 m/s, ep_len ~452, torque ~97%.
- Gait analysis: alternating support ~79–80%, flight phase ~18%, knees used ~22–27% of range.
- Interpretation: ankle-dominant shuffle persists; add swing-gated knee/hip rewards + flight penalty.

### Files Updated
- `training/configs/ppo_walking.yaml`
- `training/envs/wildrobot_env.py`
- `training/configs/training_config.py`
- `training/configs/training_runtime_config.py`
- `training/core/metrics_registry.py`

---

## [v0.10.5] - 2025-12-28: Gait Shaping Tune (Stage 1)

### Config Updates
- Reduced velocity dominance to allow gait shaping.
- Increased orientation + torque/saturation penalties to curb forward-lean shuffle.
- Added gait periodicity reward (alternating support).
- Shortened default training run to 400 iterations.

### Notes
- Resume-ready PPO run (best at iter 520) still ankle-dominant; used as base for v0.10.6.

---

## [v0.10.4] - 2025-12-28: Velocity Incentive (Stage 1)

### Config Updates
- Strong velocity tracking incentive to escape standing-still local minimum.
- Reduced stability penalties to allow forward motion discovery.

### Results Summary
- Tracking met (vel_err ~0.085, ep_len ~460) but peak torque ~0.98 and forward-lean shuffle.

---

## [v0.10.3] - 2025-12-27: Forward Walking (Stage 1)

### Config Updates
- PPO walking config aligned to 0.5–1.0 m/s command range.
- Enabled velocity tracking metrics and exit criteria logging.

---

## [v0.10.2] - 2025-12-27: Basic Standing (Stage 1)

### Updates
- Added termination diagnostics and reset preservation for accurate logging.
- Standing training readiness (stability-focused PPO).

---

## [v0.10.1] - 2025-12-26: Config Schema Migration

### Updates
- Migrated to unified `TrainingConfig` with Freezable runtime config for JIT.
- Simplified config loading and CLI override flow.

---

## [v0.8.0] - 2024-12-26: Feature Set Refactoring

### Major Change: Single Source of Truth Architecture

Complete refactoring of AMP feature configuration and extraction to establish clear separation of concerns and a single source of truth for feature layout.

### Feature Dropping (Prevent Discriminator Shortcuts)

Added v0.8.0 feature flags to prevent discriminator from exploiting artifact-prone features:

| Flag | Effect | Rationale |
|------|--------|-----------|
| `drop_contacts` | Drop foot contacts (4 dims) | Discrete values, easy to exploit |
| `drop_height` | Drop root height (1 dim) | Policy/reference mismatch early in training |
| `normalize_velocity` | Normalize root_linvel to unit direction | Remove speed as discriminative signal |

```yaml
# ppo_amass_training.yaml
amp:
  feature:
    drop_contacts: false
    drop_height: false
    normalize_velocity: false
```

### Architecture Refactoring

**New Structure:**
```
training/
├── amp/
│   ├── policy_features.py     # Online extraction (JAX) + extract_amp_features_batched
│   └── ref_features.py        # Offline extraction (NumPy) + load_reference_features
├── configs/
│   ├── feature_config.py      # FeatureConfig + _FeatureLayout (SINGLE SOURCE OF TRUTH)
│   ├── training_config.py     # TrainingConfig + RobotConfig
│   └── training_runtime_config.py  # TrainingRuntimeConfig (JIT)
└── training/
    └── trainer_jit.py         # Training loop with normalize_features (training-specific)
```

#### Files Created
| File | Description |
|------|-------------|
| `configs/training_runtime_config.py` | `TrainingRuntimeConfig` class (renamed from `AMPPPOConfigJit`) |

#### Files Moved
| From | To |
|------|-----|
| `amp/feature_config.py` | `configs/feature_config.py` |

#### Files Deleted
| File | Reason |
|------|--------|
| `amp/amp_features.py` | Superseded by `policy_features.py` |
| `common/preproc.py` | Functionality moved to `policy_features.py` and `ref_features.py` |

### Key Changes

#### `_FeatureLayout` Class (Single Source of Truth)
```python
class _FeatureLayout:
    """Centralized feature layout definition."""

    COMPONENT_DEFS = [
        ("joint_pos", "num_joints", False),      # not droppable
        ("joint_vel", "num_joints", False),      # not droppable
        ("root_linvel", "ROOT_LINVEL_DIM", False),
        ("root_angvel", "ROOT_ANGVEL_DIM", False),
        ("root_height", "ROOT_HEIGHT_DIM", True),   # droppable
        ("foot_contacts", "FOOT_CONTACTS_DIM", True), # droppable
    ]
```

#### `TrainingConfig.to_runtime_config()` Method
```python
# Only way to create TrainingRuntimeConfig
runtime_config = training_cfg.to_runtime_config()
```

#### `load_reference_features()` Function
```python
# Consolidated reference data loading in ref_features.py
from training.amp.ref_features import load_reference_features
ref_features = load_reference_features(args.amp_data)
```

#### CLI Overrides Applied to TrainingConfig
```python
# CLI parameters overwrite training_cfg BEFORE to_runtime_config()
if args.iterations is not None:
    training_cfg.iterations = args.iterations
# ... other CLI overrides
config = training_cfg.to_runtime_config()
```

### Removed Code

| Removed | Reason |
|---------|--------|
| `mask_waist` config/code | No waist joint in robot |
| `velocity_filter_alpha` from runtime | Not used in training |
| `ankle_offset` from runtime | Not used in training |
| `RunningMeanStd` class | Only used in tests |
| `create_running_stats()` | Only used in tests |
| `update_running_stats()` | Only used in tests |
| `normalize_features()` in policy_features.py | Duplicate, kept in trainer_jit.py |

### Design Principles Established

1. **Feature extraction** (`policy_features.py`, `ref_features.py`) - extracts raw features
2. **Training normalization** (`trainer_jit.py`) - statistical normalization for training
3. **Single source of truth** - `_FeatureLayout.COMPONENT_DEFS` defines feature ordering
4. **Config-driven** - All drop flags read from `TrainingConfig`

### Config
```yaml
version: "0.8.0"
version_name: "Feature Set Refactoring"
```

### Migration

If upgrading from v0.7.0:
1. Update imports from `amp.feature_config` → `configs.feature_config`
2. Update imports from `amp.amp_features` → `amp.policy_features`
3. Replace `AMPPPOConfigJit` with `TrainingRuntimeConfig`
4. Use `training_cfg.to_runtime_config()` instead of direct instantiation
5. Remove any references to `mask_waist`, `velocity_filter_alpha`, `ankle_offset` in runtime code

### Status
✅ Complete - architecture refactored with single source of truth

---

## [v0.7.0] - 2024-12-25: Physics Reference Data

### Major Change: Physics-Generated Reference Data

Switched from GMR retargeted motions to physics-realized reference data.

**Why**: GMR motions are dynamically infeasible (robot falls immediately). Physics rollout ensures reference data is achievable by the robot.

### Physics Reference Generator

New script: `scripts/generate_physics_reference_dataset.py`

**Features:**
- Full harness mode (Height + XY + Orientation stabilization)
- Physics-derived contacts from MuJoCo `efc_force`
- AMP features computed from realized physics states
- Quality gates (accept/trim/reject based on load support)

**Results:**
- 12 motions processed → All accepted
- 3,407 frames, 68.14s total duration
- Output: `training/data/physics_ref/walking_physics_merged.pkl`

### GMR Script Updates (v0.7.0 + v0.9.3)

**v0.7.0 - Heading-Local Frame:**
```python
# All velocities now in heading-local frame (rotation invariance)
root_lin_vel_heading = world_to_heading_local(root_lin_vel, root_rot)
root_ang_vel_heading = world_to_heading_local(root_ang_vel, root_rot)
```

**v0.7.0 - FK-Based Contacts:**
```python
# Replaced hip-pitch heuristic (84% double stance) with FK
foot_contacts = estimate_foot_contacts_fk(
    dof_pos, root_pos, root_rot, mj_model,
    contact_height_threshold=0.02
)
```

### Config Changes

```yaml
# ppo_amass_training.yaml
version: "0.7.0"
version_name: "Physics Reference Data"

amp:
  # NEW: Physics-generated reference data
  dataset_path: training/data/physics_ref/walking_physics_merged.pkl
  # was: training/data/walking_motions_normalized_vel.pkl
```

### Expected Training Behavior

| Metric | GMR Baseline | Expected (Physics) |
|--------|--------------|-------------------|
| Robot falls | Immediately | Should walk |
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (healthy) |
| Reference feasibility | ~15% load support | 100% (harness) |

---

## [v0.6.6] - 2024-12-24: Feature Parity Fix (Root Linear Velocity)

### Problem: Policy and Reference Features Had Different Root Velocity Semantics

Root cause of `disc_acc = 0.50` (discriminator stuck at chance):

| Feature | Policy Extractor | Reference Generator | Impact |
|---------|------------------|---------------------|--------|
| **Root linear vel** | Normalized to unit direction | Raw velocity (m/s) | **CRITICAL** |

When features have different semantics, the discriminator cannot learn a stable decision boundary → collapses to 0.50 accuracy.

### Root Cause Analysis

**Root Linear Velocity Normalization Mismatch**
```python
# Policy (WRONG - normalized to unit direction):
root_linvel_dir = root_linvel / ||root_linvel||  # Removes magnitude!

# Reference (correct - raw velocity):
root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt  # Keeps magnitude
```

After z-score normalization with reference stats, policy velocities (unit vectors with small variance) become near-constant, removing a major discriminative signal.

### Changes

**Policy Feature Extractor (`amp/amp_features.py`):**
```python
# BEFORE (v0.6.5):
root_linvel_dir = root_linvel / (linvel_norm + 1e-8)  # ❌ Normalized

# AFTER (v0.6.6):
root_linvel_feat = root_linvel  # ✅ Raw velocity in m/s
```

**Waist Masking:** Removed (both policy and reference use 29 dims).

The waist_yaw being constant 0 in AMASS is actually valid signal - the policy will learn to keep waist stable like the reference data.

### Feature Vector Contract (v0.6.6)

Both policy and reference produce identical **29-dim** vectors:

| Index | Feature | Dims |
|-------|---------|------|
| 0-8 | Joint positions | 9 |
| 9-17 | Joint velocities | 9 |
| 18-20 | Root linear velocity (**raw m/s**) | 3 |
| 21-23 | Root angular velocity | 3 |
| 24 | Root height | 1 |
| 25-28 | Foot contacts | 4 |
| **Total** | | **29** |

### No Reference Data Rebuild Needed

This fix only changes the policy extractor. Reference data already uses raw velocities and 29 dims.

### Expected Post-Fix Behavior

| Metric | Before (v0.6.5) | Expected (v0.6.6) |
|--------|-----------------|-------------------|
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (oscillating) |
| `D(real)` | ≈0.5 | 0.7-0.9 |
| `D(fake)` | ≈0.5 | 0.3-0.5 |
| `amp_reward` | ≈0.05 | 0.1-0.4 |

---

## [v0.6.5] - 2024-12-24: Discriminator Middle-Ground + Diagnostic

### Problem: disc_acc stuck at extremes

| Version | `disc_acc` | Problem |
|---------|-----------|---------|
| v0.6.3 | 1.00 | D too strong (overpowered) |
| v0.6.4 | 0.50 | D collapsed (not learning) |

### Changes

**Middle-ground discriminator settings:**
| Parameter | v0.6.3 | v0.6.4 | v0.6.5 |
|-----------|--------|--------|--------|
| `disc_lr` | 1e-4 | 5e-5 | **8e-5** |
| `update_steps` | 3 | 1 | **2** |
| `r1_gamma` | 5.0 | 20.0 | **10.0** |

**Diagnostic settings (test if noise/filter erases distribution):**
| Parameter | Before | v0.6.5 | Rationale |
|-----------|--------|--------|-----------|
| `amp.weight` | 0.3 | **0.5** | Boost AMP signal (was underfeeding) |
| `disc_input_noise_std` | 0.03 | **0.0** | Test if noise erases distribution |
| `velocity_filter_alpha` | 0.5 | **0.0** | Test if filter erases distribution |

### Expected Outcomes

**If disc_acc jumps above 0.50:**
- Noise/filter was erasing the distribution difference
- Re-enable with lower values

**If disc_acc stays at 0.50:**
- Feature mismatch between policy and reference
- Run `diagnose_amp_features.py` to compare features
- Check: joint ordering, pelvis orientation (Z-up vs Y-up), degrees vs radians

### Target

`disc_acc = 0.55-0.75` (healthy adversarial game)

---

## [v0.6.4] - 2024-12-24: Discriminator Balance Fix

### Problem: Discriminator Collapse (disc_acc = 1.00)

Training logs showed classic discriminator collapse:
```
#60-#220: disc_acc=1.00 | amp≈0.02-0.03 | success=0.0%
```

**Root Cause Analysis:**
- `disc_acc = 1.00` means discriminator perfectly classifies real vs policy samples
- This causes `amp_reward → 0` (policy gets no gradient from AMP)
- Policy ignores style and only optimizes task reward → no AMASS transfer
- This is the "perfect expert classifier" failure mode

**Why it happened (previous v0.6.3 config):**
| Parameter | Old Value | Problem |
|-----------|-----------|---------|
| `disc_lr` | 1e-4 | Too high - D learns too fast |
| `update_steps` | 3 | Too many D updates per PPO iter |
| `r1_gamma` | 5.0 | Too weak - not enough gradient penalty |
| `disc_input_noise_std` | 0.02 | Too low - D can overfit to clean features |
| `replay_buffer_ratio` | 0.5 | D learns to perfectly classify old samples |

### Changes

**Config (ppo_amass_training.yaml):**
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `disc_lr` | 1e-4 | 5e-5 | Slow down D learning |
| `update_steps` | 3 | 1 | Fewer D updates per PPO iteration |
| `r1_gamma` | 5.0 | 20.0 | Stronger gradient penalty to smooth D |
| `disc_input_noise_std` | 0.02 | 0.05 | More noise to blur D decision boundary |
| `replay_buffer_ratio` | 0.5 | 0.2 | Less historical samples when D is strong |

### Expected Post-Fix Behavior

| Metric | Before | Expected After |
|--------|--------|----------------|
| `disc_acc` | 1.00 (stuck) | 0.55-0.75 (oscillating) |
| `amp_reward` | ≈0.02 | 0.1-0.4 (rising) |
| `task_r` | Rising fast | May dip initially, then rise |
| `success` | 0.0% | Should become non-zero |

**Key insight:** You want the discriminator to **struggle**, not win. A healthy GAN game keeps D accuracy around 0.55-0.75.

### Future Tuning

Once `disc_acc` stabilizes in 0.55-0.75 range:
- Increase `amp.weight` from 0.3 → 0.5 for stronger style influence
- Monitor for any regression back to 1.00

---

## [v0.6.2] - 2024-12-24: Golden Rule Configuration

### Problem
The v0.6.1 Golden Rule fixes had hardcoded values that should be configurable:
1. **Hardcoded joint indices**: `left_hip_pitch=1`, `left_knee_pitch=3`, etc. were hardcoded instead of derived from `mujoco_robot_config.json`
2. **Magic numbers unexplained**: Contact estimation used unexplained values (0.5, 0.3, 1.0)
3. **Parameters as defaults**: `use_estimated_contacts`, `use_finite_diff_vel`, and contact params had defaults instead of being required from training config

### Changes

#### Configuration (Training Config → Feature Extraction)
| File | Change |
|------|--------|
| `configs/ppo_amass_training.yaml` | Added Golden Rule section with all configurable params |
| `configs/config.py` | Added Golden Rule params to `TrainingConfig` dataclass |

**New Config Options:**
```yaml
amp:
  # v0.6.2: Golden Rule Configuration (Mathematical Parity)
  use_estimated_contacts: true   # Use joint-based contact estimation (matches reference)
  use_finite_diff_vel: true      # Use finite difference velocities (matches reference)
  contact_threshold_angle: 0.1   # Hip pitch threshold for contact detection (rad, ~6°)
  contact_knee_scale: 0.5        # Knee angle at which confidence decreases (rad, ~28°)
  contact_min_confidence: 0.3    # Minimum confidence when hip indicates contact (0-1)
```

#### Feature Extraction (Config-Driven Joint Indices)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added joint indices to `AMPFeatureConfig` (derived from robot_config) |
| `amp/amp_features.py` | `estimate_foot_contacts_from_joints()` uses config indices |
| `amp/amp_features.py` | `extract_amp_features()` now REQUIRES Golden Rule params (no defaults) |
| `amp/amp_features.py` | Added comprehensive docstrings explaining magic numbers |

**AMPFeatureConfig Now Includes:**
```python
class AMPFeatureConfig(NamedTuple):
    # ... existing fields ...
    # v0.6.2: Joint indices for contact estimation (from robot_config)
    left_hip_pitch_idx: int   # Index of left_hip_pitch in joint_pos
    left_knee_pitch_idx: int  # Index of left_knee_pitch in joint_pos
    right_hip_pitch_idx: int  # Index of right_hip_pitch in joint_pos
    right_knee_pitch_idx: int # Index of right_knee_pitch in joint_pos
```

**Magic Numbers Explained:**
```python
def estimate_foot_contacts_from_joints(
    joint_pos, config,
    threshold_angle: float = 0.1,   # ~6° - leg behind body → contact
    knee_scale: float = 0.5,        # ~28° - confidence decreases as knee bends
    min_confidence: float = 0.3,    # 30% - minimum contact when hip indicates contact
):
    # Confidence formula: clip(1.0 - |knee| / knee_scale, min_confidence, 1.0)
    # - knee=0 → confidence=1.0 (fully extended)
    # - knee=0.5 → confidence=0.3 (min, bent knee)
```

#### Training Pipeline (Pass Config Through)
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added Golden Rule params to `AMPPPOConfigJit` |
| `training/trainer_jit.py` | `extract_amp_features_batched()` accepts all params |
| `training/trainer_jit.py` | `make_train_iteration_fn()` passes config values |

#### Diagnostic Script (Config-Driven)
| File | Change |
|------|--------|
| `scripts/diagnose_amp_features.py` | Reads Golden Rule params from training config |
| `scripts/diagnose_amp_features.py` | Prints config values for debugging |

### Config
```yaml
version: "0.6.2"
version_name: "Golden Rule Configuration"
```

### Key Principle
**"No hardcoding, everything from config"** — All parameters that affect feature extraction now come from `ppo_amass_training.yaml` and `mujoco_robot_config.json`. Joint indices are derived from actuator names, not hardcoded.

### Migration
If upgrading from v0.6.1, add these fields to your training config:
```yaml
amp:
  use_estimated_contacts: true
  use_finite_diff_vel: true
  contact_threshold_angle: 0.1
  contact_knee_scale: 0.5
  contact_min_confidence: 0.3
```

### Status
✅ Complete - all Golden Rule parameters are now configurable

---

## [v0.6.0] - 2024-12-23: Spectral Normalization + Policy Replay Buffer

### Problem
Two remaining discriminator stability issues:
1. **Unconstrained Lipschitz constant**: Discriminator gradients could explode, destabilizing training
2. **Catastrophic forgetting**: Discriminator overfits to recent policy samples, forgetting earlier distributions

### Changes

#### Spectral Normalization (Miyato et al., 2018)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Added `SpectralNormDense` layer wrapping Flax `nn.SpectralNorm` |
| `amp/discriminator.py` | `AMPDiscriminator` now uses spectral-normalized layers by default |
| `amp/discriminator.py` | Added `use_spectral_norm` flag for A/B testing (default: `True`) |

**Why Spectral Normalization:**
- Industry standard for GANs (StyleGAN, BigGAN, etc.)
- Controls Lipschitz constant to 1 by normalizing weights by largest singular value
- Prevents gradient explosion without hyperparameter tuning
- More stable than gradient penalties alone

```python
class SpectralNormDense(nn.Module):
    """Dense layer with Spectral Normalization."""
    features: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        dense = nn.Dense(features=self.features, ...)
        spectral_dense = nn.SpectralNorm(dense, collection_name="batch_stats")
        return spectral_dense(x)
```

#### Policy Replay Buffer
| File | Change |
|------|--------|
| `amp/replay_buffer.py` | NEW: `PolicyReplayBuffer` (Python mutations) |
| `amp/replay_buffer.py` | NEW: `JITReplayBuffer` (functional JAX updates) |
| `training/trainer_jit.py` | Added `replay_buffer_size` and `replay_buffer_ratio` to config |
| `training/trainer_jit.py` | Added replay buffer state fields to `TrainingState` |
| `train.py` | Pass replay buffer config from YAML |
| `configs/ppo_amass_training.yaml` | Added replay buffer settings |

**Why Replay Buffer:**
- Stores historical policy features to prevent discriminator from overfitting to current policy
- Mixes `replay_buffer_ratio` (default 50%) historical samples with fresh samples
- JIT-compatible implementation using functional updates for `jax.lax.scan`

```python
# JIT-compatible replay buffer (functional updates)
class JITReplayBuffer:
    @staticmethod
    def add(state: dict, samples: jnp.ndarray) -> dict:
        # Functional update for use in jax.lax.scan
        ...

    @staticmethod
    def sample(state: dict, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        # Sample batch mixing fresh + historical
        ...
```

### Config
```yaml
version: "0.6.0"
version_name: "Spectral Normalization + Policy Replay Buffer"

amp:
  # v0.6.0: Policy Replay Buffer
  replay_buffer_size: 100000
  replay_buffer_ratio: 0.5  # 50% historical samples
```

### Expected Result
- More stable discriminator training (no gradient explosions)
- Smoother `disc_acc` curves (less catastrophic forgetting)
- Better generalization of discriminator across policy evolution

### Status
🔄 Ready to train - building on v0.5.0 foot contact fix

---

## [v0.5.0] - 2024-12-23: Foot Contact Fix + Temporal Context

### Problem
**Critical distribution mismatch bug discovered:** Policy rollouts sent zeros for foot contacts while reference data had 88% non-zero foot contacts. The discriminator learned to trivially distinguish them by checking if foot_contacts == 0, achieving 100% accuracy without learning actual motion quality.

| Source | Foot Contacts |
|--------|---------------|
| Reference Data | 88% non-zero (real contact values) |
| Policy Rollouts (v0.4.x) | 100% zeros (not extracted from sim!) |

**Result:** `disc_acc = 1.00` and `amp_reward ≈ 0` — the "foot contact cheat" bug.

### Root Cause
`foot_contacts` were never extracted from MJX simulation during policy rollouts. The `extract_amp_features()` function silently fell back to zeros when `foot_contacts=None`.

### Changes

#### Foot Contact Extraction (Core Fix)
| File | Change |
|------|--------|
| `envs/wildrobot_env.py` | Extract real foot contacts from MJX contact forces |
| `envs/wildrobot_env.py` | Use explicit config keys (`left_toe`, `left_heel`, etc.) |
| `envs/wildrobot_env.py` | Raise `RuntimeError` if foot geoms not found |
| `envs/wildrobot_env.py` | Made `contact_threshold` and `contact_scale` configurable |
| `training/trainer_jit.py` | Added `foot_contacts` to `Transition` namedtuple |
| `training/trainer_jit.py` | Pass `foot_contacts` through entire training pipeline |
| `amp/amp_features.py` | Require `foot_contacts` (no silent fallback to zeros!) |

#### Asset Updates
| File | Change |
|------|--------|
| `assets/post_process.py` | Renamed geoms: `left_toe/left_heel` (was `left_foot_btm_front/back`) |
| `assets/post_process.py` | Added explicit config keys to `mujoco_robot_config.json` |
| `assets/mujoco_robot_config.json` | Added `left_toe`, `left_heel`, `right_toe`, `right_heel` keys |
| `assets/wildrobot.xml` | Updated geom names via post_process.py |

#### Temporal Context Infrastructure (P3 - Prepared, Not Enabled)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added `TemporalFeatureConfig`, `create_temporal_buffer()` |
| `amp/amp_features.py` | Added `update_temporal_buffer()`, `add_temporal_context_to_reference()` |

#### Testing
| File | Description |
|------|-------------|
| `tests/test_foot_contacts.py` | 8 comprehensive tests for foot contact pipeline |

### Foot Contact Detection
```python
# 4-point model: [left_toe, left_heel, right_toe, right_heel]
# Soft thresholding: tanh(force / scale) for continuous 0-1 values
foot_contacts = jnp.tanh(contact_forces / self._contact_scale)
```

### Key Principle
**"Fail loudly, no silent fallbacks"** — If foot contacts are missing, raise an exception immediately rather than silently using zeros.

### Expected Result
- `disc_acc` should fluctuate around 0.5-0.7 (healthy discrimination)
- `amp_reward` should stay meaningful (0.3-0.8)
- `success_rate` should gradually increase as robot learns

### Status
🔄 Ready to train - foot contacts now correctly extracted from simulation

---

## [v0.4.1] - 2024-12-23: R1 Regularizer + Distribution Metrics

### Problem
WGAN-GP gradient penalty was theoretically inconsistent with LSGAN loss (which uses bounded targets [0, 1]).

### Changes

#### R1 Regularizer (Replaced WGAN-GP)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Replaced WGAN-GP with R1 regularizer |
| `training/trainer_jit.py` | Updated discriminator training to use R1 |
| `configs/ppo_amass_training.yaml` | Renamed `gradient_penalty_weight` → `r1_gamma` |

**Why R1 over WGAN-GP:**
- WGAN-GP uses interpolated samples (designed for Wasserstein critics)
- R1 only penalizes gradient on REAL samples (simpler, faster)
- More theoretically consistent with LSGAN

```python
# R1 Regularization: gradient penalty on REAL samples only
def real_disc_sum(x):
    return jnp.sum(model.apply(params, x, training=True))

grad_real = jax.grad(real_disc_sum)(real_obs)
r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty
```

#### Distribution Metrics
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added `disc_real_mean`, `disc_fake_mean`, `disc_real_std`, `disc_fake_std` |

These metrics help diagnose discriminator behavior:
- `disc_real_mean` should approach 1.0
- `disc_fake_mean` should approach 0.0
- If both are similar, discriminator is not learning separation

### Config
```yaml
amp:
  r1_gamma: 5.0  # R1 regularizer weight (was gradient_penalty_weight)
```

### Status
✅ Implemented and validated

---

## [v0.4.0] - 2024-12-23: Critical Bug Fixes (Reward Formula + Training Order)

### Problem
Two critical bugs identified by external design review:

1. **Reward Formula Bug ("The Participation Trophy"):** Policy received 0.75 reward when discriminator was 100% sure motion was fake.
2. **Training Order Bug ("The Hindsight Bias"):** AMP rewards computed with updated discriminator, not the one active when samples were collected.

### Changes

#### Fix 1: Reward Formula (Clipped Linear)
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| `max(0, 1 - 0.25 * (D(s) - 1)²)` | `clip(D(s), 0, 1)` |
| D=0 (fake) → reward=0.75 ❌ | D=0 (fake) → reward=0.0 ✅ |

| File | Change |
|------|--------|
| `amp/discriminator.py` | Changed reward formula to clipped linear |

#### Fix 2: Training Order
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| Train disc → Compute reward (NEW params) | Compute reward (OLD params) → Train disc |

| File | Change |
|------|--------|
| `training/trainer_jit.py` | Moved AMP reward computation BEFORE discriminator training |

#### Fix 3: Network Size Increase
| Setting | Before | After |
|---------|--------|-------|
| `discriminator_hidden` | `[256, 128]` | `[512, 256]` |

### Expected Result
After these fixes, `disc_acc` should immediately jump from 0.50 to 0.60-0.80 within the first 20 iterations.

### Status
✅ Implemented and validated

---

## [v0.3.1] - 2024-12-23: Config-Driven Pipeline (No Hardcoding)

### Problem
Joint order mismatch between reference data and policy features. GMR's `convert_to_amp_format.py` had **hardcoded** joint order that didn't match MuJoCo qpos order, causing discriminator to compare misaligned features.

| Component | Expected | Bug |
|-----------|----------|-----|
| MuJoCo qpos | `waist_yaw` at index 0 | ✅ Correct |
| `mujoco_robot_config.json` | `waist_yaw` at index 0 | ✅ Correct |
| GMR hardcoded | `left_hip_pitch` at index 0 | ❌ Wrong |

### Changes

#### GMR Pipeline (Config-Driven)
| File | Change |
|------|--------|
| `GMR/scripts/convert_to_amp_format.py` | Added `--robot-config` flag (required), removed all hardcoded joint names/indices |
| `GMR/scripts/batch_convert_to_amp.py` | Added `--robot-config` flag (required), uses `get_joint_names()` |

#### WildRobot Fixes
| File | Change |
|------|--------|
| `amp/amp_features.py` | Fixed NamedTuple field order (defaults must come last) |
| `envs/wildrobot_env.py` | Fixed `_init_qpos` access before initialization, added `_mjx_model` creation |
| `scripts/scp_to_remote.sh` | Exclude cache files (`__pycache__`, `*.pyc`), include `mujoco_robot_config.json` |

#### Cleanup
- Deleted workaround scripts: `fix_reference_data.py`, `reorder_reference_data.py`
- Cleaned `training/data/` - only `walking_motions_normalized_vel.pkl` remains (641KB)
- Removed 14 intermediate/old data files

### Reference Data Regeneration
```bash
# Full pipeline with config-driven joint order
cd ~/projects/GMR
uv run python scripts/batch_retarget_walking.py
uv run python scripts/batch_convert_to_amp.py \
    --robot-config ~/projects/wildrobot/assets/mujoco_robot_config.json

cd ~/projects/wildrobot
uv run python scripts/convert_ref_data_normalized_velocity.py
```

### Validation
```
✅ Joint order: ['waist_yaw', 'left_hip_pitch', ...] matches mujoco_robot_config.json
✅ Features shape: (3407, 29)
✅ Velocity normalized: True
✅ 12 motions, 68.19s total duration
```

### Key Principle
**"Fail fast, no silent defaults"** - Joint order now comes from `mujoco_robot_config.json`, never hardcoded. Missing config raises immediate error.

### Status
🔄 Ready to train - code synced to remote

---

## [v0.3.0] - 2024-12-23: Velocity Normalization

### Problem
Speed mismatch between reference motion data (~0.27 m/s) and commanded velocity (0.5-1.0 m/s). Discriminator penalized policy for walking at correct speed.

### Changes

#### Capability
- **Velocity Direction Normalization**: Root linear velocity normalized to unit direction vector
- **Safe Thresholding**: If speed < 0.1 m/s, use zero vector (handles stationary frames)
- **Feature Validation Script**: `scripts/validate_feature_consistency.py` ensures reference = policy features

#### Files Modified
| File | Change |
|------|--------|
| `amp/amp_features.py` | Safe velocity normalization with 0.1 m/s threshold |
| `scripts/convert_ref_data_normalized_velocity.py` | Reference data conversion script |
| `scripts/validate_feature_consistency.py` | End-to-end feature validation |
| `reference_data_generation.md` | Updated pipeline documentation |

#### Config
```yaml
amp:
  dataset_path: training/data/walking_motions_normalized_vel.pkl
  # Feature dim remains 29 (direction replaces raw velocity)
```

#### Feature Format (29-dim)
| Index | Feature | Notes |
|-------|---------|-------|
| 0-8 | Joint positions | |
| 9-17 | Joint velocities | |
| 18-20 | Root velocity **DIRECTION** | Normalized unit vector (NEW) |
| 21-23 | Root angular velocity | |
| 24 | Root height | |
| 25-28 | Foot contacts | |

### Validation
- ✅ 63.4% moving frames (norm = 1.0)
- ✅ 36.6% stationary frames (norm = 0.0)
- ✅ Reference and policy features identical

### Expected Result
- AMP reward should be smoother (no speed penalty)
- Discriminator learns gait style, not speed matching
- Policy can walk at commanded speed without AMP penalty

### Status
🔄 Ready to train - awaiting results

---

## [v0.2.0] - 2024-12-22: Discriminator Tuning (Middle-Ground)

### Problem
Initial discriminator settings caused disc_acc stuck at 0.97-0.99 (too strong).
Over-correction caused disc_acc collapse to 0.50 (not learning).

### Changes

#### Config Evolution
| Setting | Initial | Over-corrected | Final (Middle-ground) |
|---------|---------|----------------|----------------------|
| `disc_lr` | 1e-4 | 2e-5 | **5e-5** |
| `update_steps` | 3 | 1 | **2** |
| `gradient_penalty_weight` | 5.0 | 20.0 | **15.0** |
| `disc_input_noise_std` | 0.0 | 0.05 | **0.05** |

#### Final Config
```yaml
amp:
  weight: 0.3
  disc_lr: 5e-5
  batch_size: 256
  update_steps: 2
  gradient_penalty_weight: 15.0
  disc_input_noise_std: 0.05
```

### Result
- ✅ Healthy disc_acc oscillation: 0.72-0.95
- ✅ AMP reward stable
- ❌ Speed mismatch still present (addressed in v0.3.0)

### Lessons Learned
- `disc_acc = 0.50` means discriminator collapsed (not learning)
- `disc_acc = 0.97+` means discriminator too strong
- Target: `disc_acc = 0.6-0.8` with healthy oscillation
- Don't over-handicap the discriminator

---

## [v0.1.1] - 2024-12-21: JIT Compilation Fix

### Problem
30-second GPU idle after first training iteration.

### Cause
Second JIT compilation when switching from single iteration to batch mode.

### Fix
Pre-compile both `train_iteration_fn` and `train_batch_fn` before training starts.

#### Files Modified
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Pre-compile JIT functions with dummy call |
| `train.py` | Added PID display for easy termination |

#### Code
```python
# Pre-compile both functions before training
print("Pre-compiling JIT functions...")
_ = train_iteration_fn(state, env_state, ref_buffer_data)
jax.block_until_ready(_)

if train_batch_fn is not None:
    _ = train_batch_fn(state, env_state, ref_buffer_data)
    jax.block_until_ready(_)
```

### Result
- ✅ No more GPU idle after iteration #1
- ✅ Smooth training from start

---

## [v0.1.0] - 2024-12-20: Initial AMP Training Pipeline

### Capability
- PPO + AMP adversarial training
- LSGAN discriminator formulation
- Reference motion from AMASS via GMR retargeting
- W&B logging integration
- Checkpoint save/restore

### Config
```yaml
trainer:
  num_envs: 1024
  rollout_steps: 128
  iterations: 1000
  lr: 3e-4

reward_weights:
  tracking_lin_vel: 5.0
  base_height: 0.3
  action_rate: -0.01

amp:
  weight: 0.3
  discriminator_hidden: [256, 128]
```

### Result
- Training functional but disc_acc issues (see v0.2.0)
- Speed mismatch discovered (see v0.3.0)

---

## Training Results Log

| Date | Version | Iterations | disc_acc | Velocity | Notes |
|------|---------|------------|----------|----------|-------|
| 2024-12-20 | v0.1.0 | 500 | 0.97-0.99 | ~0.3 m/s | Disc too strong |
| 2024-12-22 | v0.2.0 | 300 | 0.72-0.95 | ~0.5 m/s | Healthy oscillation |
| 2024-12-23 | v0.3.0 | TBD | TBD | TBD | Velocity normalized |
| 2025-12-28 | v0.10.4 | 500 | 0.50 | 0.65 m/s | PPO-only; vel_err ~0.085; ep_len ~460; pitch fall ~10-17%; max torque ~0.98 |
| 2025-12-28 | v0.10.5 | 400 | 0.50 | 0.63 m/s | PPO-only resume from v0.10.4; reward ~566; ankle-dominant gait persisted |

---

## Quick Reference

### Discriminator Tuning Guide
| Symptom | disc_acc | Action |
|---------|----------|--------|
| Disc too strong | 0.95+ | ↓ disc_lr, ↓ update_steps, ↑ gradient_penalty |
| Disc collapsed | ~0.50 | ↑ disc_lr, ↑ update_steps, ↓ gradient_penalty |
| Healthy | 0.6-0.8 | Keep current settings |

### Feature Validation
```bash
# Validate reference = policy features
uv run python scripts/validate_feature_consistency.py

# Analyze reference data velocities
uv run python scripts/analyze_reference_velocities.py
```

### Training Commands
```bash
# Start training
cd ~/projects/wildrobot
uv run python training/train.py

# Kill training (PID printed at startup)
kill -9 <PID>
```

---

## Future Improvements (Backlog)

- [ ] Label smoothing for discriminator (defensive measure)
- [ ] Filter stationary frames from reference data if needed
- [ ] Curriculum learning for velocity commands
- [ ] Foot contact detection from simulation
- [ ] Multi-speed reference data augmentation
