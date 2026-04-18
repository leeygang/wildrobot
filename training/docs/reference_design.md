# WildRobot Nominal Reference Design

**Status:** Active design note for `v0.20.x` prior reference validation  
**Last updated:** 2026-04-17

---

## Purpose

This document defines the intended architecture and validation plan for the
WildRobot locomotion prior-reference layer after the failed `v0.19.5`
reward-first PPO attempt.

`v0.20.0` exists because `v0.19.5` showed that residual PPO is not a good
mainline when the base walking reference does not yet own propulsion, step
release, and command tracking strongly enough.

The goal of this document is to make the design explicit enough that:

- the reference architecture has one source of truth
- downstream layers remain adapters, not competing controllers
- validation is repeatable and gate-based
- future refactors reduce code debt rather than add more local patches
- prior selection is explicit rather than implicit

---

## Scope

This document is about:

- locomotion prior / reference architecture
- `walking_ref_v1` / `walking_ref_v2` limitations and lessons
- offline prior / reference library design
- validation of prior assets, short-horizon feasibility, and first-policy
  learning
- `v0.20.0` execution milestones

This document is not about:

- PPO reward tuning
- long-run training strategy
- hardware deployment steps
- full MPC adoption

---

## Main Design Decision

The locomotion stack should follow this contract:

```text
command + robot state
    ->
walking prior / reference
    ->
offline reference library lookup / preview
    ->
nominal targets + preview annotations
    ->
policy
    ->
joint targets
```

The intended ownership is:

- the prior reference owns locomotion semantics
- IK / nominal target shaping owns execution adaptation
- PPO owns correction, robustness, and adaptation only after the prior is
  validated

Active first contract:

- ToddlerBot-style bounded residual policy around library-backed `q_ref`

Planned upgrade compatibility:

- keep the same offline prior asset and preview interface if later policy
  versions move toward a Cassie-style broader-authority contract

Only one layer should own locomotion semantics, and for the active `v0.20`
path it does so at generation time:

- whether progression opens at all
- which gait phase / contact schedule is encoded
- how much forward swing release is planned
- how foothold progression is shaped across the command family

That layer is the offline prior generator, not the runtime adapter.

---

## Pivot Decision

`v0.19.5` did not fail because PPO was configured slightly wrong. It failed
because the walking prior still did not own enough of the locomotion task.

Observed lesson from `v0.19.4` and `v0.19.5`:

- gait existence is not enough
- visible step release is not enough
- the prior must already own meaningful command tracking, step amplitude, and
  multi-seed stability before PPO resumes

So the `v0.20.0` design decision is:

- stop treating reward tuning as the mainline next step
- choose a solid prior family first
- validate that prior visually and metrically before PPO work resumes

---

## Prior Options

### Comparison

| Option | Development cost | Debuggability | Validation difficulty | Generalization upside | Fit for WildRobot v2 | Recommendation |
|---|---|---|---|---|---|---|
| `ZMP` / preview-style flat-ground prior | low-medium | high | medium | medium | good | recommended first prior |
| `ALIP` / capture-point / angular-momentum-aware prior | medium-high | medium | high | high | very good | recommended upgrade path |
| `Placo` / Mini-Duck-style whole-body kinematic prior | medium | medium-low | high | low-medium | mixed | use as a prototyping aid, not first source of truth |

### `ZMP`

Strengths:

- easiest prior to reason about on flat ground
- easiest to debug with COM, foothold, and phase plots
- easiest to verify visually on a position-controlled biped
- lowest implementation risk for stop, resume, and slow forward walk

Weaknesses:

- conservative
- less expressive for dynamic catch steps and momentum-rich locomotion
- likely to saturate before the full adaptation goal is met

WildRobot fit:

- good first fit because WildRobot v2 is small, servo-driven, and lacks active
  ankle roll
- a `ZMP`-shaped prior is easier to keep inside reachable geometry and visible
  stepping than a more aggressive model on day one

### `ALIP`

Strengths:

- better dynamic step-placement logic
- better long-term fit for speed adaptation and disturbance handling
- stronger ceiling for broader locomotion behavior

Weaknesses:

- harder to tune and diagnose than `ZMP`
- easier to inject forward-drive channels that look plausible in plots but are
  unstable in full env dynamics

WildRobot fit:

- best medium-term direction once the first prior family is proven
- especially relevant if `ZMP` clears flat-ground walking but stalls on command
  breadth, push response, or dynamic step placement

### `Placo` / Mini-Duck-style prior

Strengths:

- fast way to author coherent whole-body motion
- useful for visualizing startup, stop, and turn sequences
- useful as a prototyping environment for candidate references

Weaknesses:

- easiest option to over-trust visually
- kinematic coherence does not prove dynamic viability
- harder to make it the single source of truth for support permission and step
  release under real contact dynamics

WildRobot fit:

- useful as a sidecar tool for prototyping and inspection
- not the recommended mainline prior source of truth on current hardware

### Recommendation

For WildRobot `v0.20.x`:

1. start with a `ZMP`-shaped command-conditioned prior family
2. freeze the contract so the same PPO stack can consume an `ALIP` upgrade
   later
3. use `Placo` only for rapid kinematic prototyping and visual authoring
4. do not start from mocap or pure kinematic clips as the mainline walking
   prior

---

## Active `v0.20` Architecture

### Runtime-FSM decision

The runtime walking FSM is deprecated for the active `v0.20` path.

This means:

- no active runtime `support_health -> progression_permission -> mode-transition`
  controller in the deployed walking stack
- no runtime `walking_ref_v2`-style gatekeeper deciding when swing may release
- no split-brain semantics between a reference FSM and a policy/runtime adapter

If old `walking_ref_v1` / `walking_ref_v2` code is reused, it is allowed only
for:

- offline trajectory generation
- historical diagnostics
- side-by-side comparison against the new prior path

It is not the active runtime controller.

### Mainline architecture

The active `v0.20` architecture is:

```text
command
    ->
offline prior generator
    ->
command-indexed reference library
    ->
runtime lookup / interpolation / preview service
    ->
reference preview + nominal `q_ref`
    ->
policy
    ->
joint targets
    ->
runtime / hardware
```

The mainline policy contract is ToddlerBot-style first, Cassie-compatible
later:

- `v0.20.x` first policy: bounded residual around library-backed `q_ref`
- later upgrade path: broader direct policy authority using the same reference
  preview asset

### Layer responsibilities

**Layer 0: Offline prior generation**

Responsibilities:

- generate command-conditioned reference trajectories offline
- compute gait phase, footholds, pelvis / COM targets, and contact annotations
- validate the trajectory family before policy training starts

Allowed implementations:

- `ZMP` / preview-style planner
- later `ALIP` upgrade
- optional `Placo`-assisted authoring or prototyping

Reference starting point:

- algorithm family: Kajita et al. `ZMP` preview control over a cart-table /
  `LIPM` model
- public implementation reference: ToddlerBot
  `toddlerbot.algorithms.zmp_planner` and
  `toddlerbot.algorithms.zmp_walk`
- intended WildRobot pipeline:
  - command bin / footstep plan
  - `ZMP` preview-control COM plan
  - swing-foot trajectory generation
  - IK / nominal joint target generation
  - offline lookup-table export indexed by command

**Layer 1: Reference library asset**

Responsibilities:

- store reference trajectories indexed by command
- support interpolation between nearby command bins
- expose both task-space targets and nominal joint-space targets where useful

This is the active source of locomotion semantics.

Minimum schema per library entry:

- command key:
  - `vx` for the first branch
  - later extension points for `vy` and `yaw_rate`
- timing:
  - `dt`
  - step period / cycle length
  - normalized phase
- nominal targets:
  - `q_ref`
  - optional `dq_ref`
- task-space preview:
  - pelvis pose target
  - COM target
  - left/right foot pose targets
- annotations:
  - stance foot / support side
  - contact state or contact mask
  - stop / resume markers where applicable
- metadata:
  - generation version
  - command-bin bounds
  - feasibility / validation flags

**Layer 2: Runtime reference service**

Responsibilities:

- lookup / interpolate the library at runtime
- expose a short preview window to the policy
- emit current nominal `q_ref`, future targets, phase, and contact annotations

This layer must not contain a new locomotion FSM.

**Layer 3: Policy**

`v0.20` first contract:

- bounded residual policy
- imitation-dominant hybrid reward
- ToddlerBot-style reference-conditioned training

Future upgrade path:

- larger action authority
- longer observation / action history
- Cassie-style reference-guided direct policy if the residual contract
  saturates

**Layer 4: Target adaptation and runtime**

Responsibilities:

- lightweight target shaping only
- actuator-safe clipping
- optional small smoothing or delay compensation
- no new gait semantics

### Single source of truth

For the active `v0.20` path, the single source of truth is:

- the offline reference library asset

Not:

- a runtime FSM
- an adapter-side gating controller
- a PPO policy inventing phase structure

### ToddlerBot-first, Cassie-compatible path

This architecture is chosen specifically to support both phases:

1. **ToddlerBot-first**
   - offline reference asset
   - bounded residual policy around `q_ref`
   - imitation-dominant hybrid reward
2. **Cassie-compatible later**
   - keep the same reference preview asset
   - expand policy authority
   - add longer history and more direct control if the residual contract limits
     adaptation

The upgrade path works only if the prior is a reusable offline asset.
That is why the runtime FSM is removed from the active path instead of being
kept as the mainline controller.

---

## Adapter Design

The active runtime adapter is intentionally simple.

Its job is to convert offline-prior outputs into executable targets without
reintroducing gait semantics.

It should:

- consume the library lookup output directly
- run IK where the library stores task-space targets
- emit joint targets from stored `q_ref` where joint-space targets are already
  available
- clip to actuator and joint limits
- apply lightweight smoothing or rate limiting when required for actuator
  safety
- apply explicit safe fallbacks for unreachable targets

It should not:

- infer or reopen gait phase
- recompute support-health, progression, or release gates
- reinterpret stop / resume semantics
- introduce a second locomotion controller through hidden scaling rules

If runtime shaping exists, it must be:

- mechanical
- safety-oriented
- explicitly bounded and documented
- unable to change the gait family encoded by the offline prior

Historical adapter-side support-scaling logic from the `walking_ref_v2` era now
belongs to
[reference_design_history.md](reference_design_history.md), not to the active
`v0.20` architecture.

---

## Validation Approach

Validation should follow a layered ladder.

### Active `v0.20.0` validation ladder

This is the active validation contract for the prior pivot.

The prior is not considered valid because it "looks reasonable" in a viewer.
For the active offline-prior path, it must pass all five layers:

1. `offline reference asset quality`
2. `reference + IK playback`
3. `short-horizon MuJoCo feasibility`
4. `ToddlerBot policy learning gate`
5. `policy multi-command sweep`

### Layer 1: `offline reference asset quality`

What to verify visually:

- step timing is regular
- alternating step-out is visible
- stop commands settle into quiet standing
- resume commands re-open stepping without a jump
- pelvis / COM preview stays bounded

What to verify metrically:

- commanded foothold mean at `cmd=0.15` is `>= 0.045 m`
- previewed step period stays within `0.45-0.75 s`
- pelvis and COM preview are continuous
- target-rate limits are not violated
- library coverage includes:
  - quiet stand
  - stop / resume
  - low-speed walk
  - nominal command sweep points

### Layer 2: `reference + IK playback`

What to verify visually:

- feet land where the prior intended
- stance switching is clearly visible
- no foot scissoring, severe toe drag, or obvious self-collision posture
- stop, resume, and low-speed stepping remain coherent after IK

What to verify metrically:

- realized nominal touchdown step length mean `>= 0.03 m`
- realized / commanded step ratio `>= 0.5`
- IK reachability stays near `1.0`
- joint targets remain inside configured safety margins
- nominal foot-clearance floor remains positive throughout swing

### Layer 3: `short-horizon MuJoCo feasibility`

This replaces the old "open-loop must walk 400+ steps" gate for the active
offline-prior path.  Uses the existing MuJoCo actuator model for feasibility
checking.  Hardware SysID verification is deferred to `v0.20.2` and gates
sim-to-real deployment, not ZMP library generation.

What to verify visually:

- the MuJoCo actuator model can track the reference for short rollouts without
  immediate geometric collapse
- startup, stop, and one-to-two gait cycles look trackable rather than rigid or
  obviously impossible
- failures should look like tracking limits or contact mismatch, not like an
  incoherent reference

What to verify metrically:

- simulated leg-joint tracking RMSE stays within the configured actuator budget
- repeated target saturation does not dominate the rollout
- no systematic joint-limit hits occur in the validated command range
- short-horizon replay survives at least one full gait cycle without immediate
  pitch-faceplant termination

### Layer 4: `ToddlerBot policy learning gate`

This is the main unblock gate for the active `v0.20` path.

What to verify visually:

- within a short training horizon, the learned policy starts to track the
  reference and produce visible stepping
- the learned gait still looks like the reference family, not a new exploit
- stop / resume behavior remains recognizable

What to verify metrically:

- reference-tracking metrics improve materially from initialization:
  - `ref/joint_track_rmse`
  - `ref/pelvis_pose_error`
  - `ref/foot_site_error`
  - `ref/contact_phase_match`
- by the early smoke horizon, the run achieves all of:
  - `env/forward_velocity >= 0.04 m/s`
  - `eval_walk/episode_length >= 150`
  - `term_pitch_frac < 0.50`
- by the promotion horizon, the run achieves all of:
  - `env/forward_velocity >= 0.075 m/s`
  - `eval_walk/success_rate >= 0.60`
  - `tracking/cmd_vs_achieved_forward <= 0.075 m/s`
  - touchdown step length mean `>= 0.03 m`

### Layer 5: `policy multi-command sweep`

What to verify visually:

- zero command means standing, not creeping
- low command means slow but visible stepping
- increasing command does not immediately collapse into pitch dive
- stop/resume works without a discontinuous posture jump

What to verify metrically:

- zero-command drift `< 0.01 m/s`
- stop latency to quiet posture `< 1.0 s`
- resume-to-first-step latency `< 1.0 s`
- forward-speed response is monotonic non-decreasing over the tested range
- no single command bin is propped up by a qualitatively different exploit gait

### Option-specific validation emphasis

`ZMP`

- strongest on layers 1, 3, and 4
- easiest to validate with COM, foothold, stop/resume, and imitation tracking
  diagnostics

`ALIP`

- requires extra scrutiny on pitch-rate spikes, overspeed, and dynamic
  catch-step plausibility
- only upgrade here after `ZMP` clears the main gate or clearly saturates

`Placo`

- never accept viewer quality alone as validation
- always require full layer-3 through layer-5 checks before trusting it as a
  walking prior

## Execution Milestones For `v0.20.0`

The milestone labels below are the canonical `v0.20.0` sequence and should
match [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md).

### `v0.20.0-A`: Prior contract freeze

Expectation:

- this milestone freezes interface and schema; it does not claim walking
  robustness

Quick visual check:

- the same command-bin replay always produces the same reference trajectory and
  phase evolution
- viewer traces show no discontinuous target jumps

How this advances the end goal:

- removes interface churn so later milestones focus on locomotion quality

Goal:

- define one stable offline-prior contract that both `ZMP` and later `ALIP`
  variants can satisfy
- make the contract ToddlerBot-first and Cassie-compatible later

Required outputs:

- command-indexed library schema
- reference preview window contract
- nominal `q_ref` contract
- task-space preview contract
- phase / contact annotation contract
- stop / resume handling contract

Minimum schema details:

- command key:
  - `vx` in the first branch
  - reserved extension fields for `vy` and `yaw_rate`
- timing:
  - `dt`
  - normalized phase
  - step period / cycle length
- nominal targets:
  - `q_ref`
  - optional `dq_ref`
- task-space targets:
  - pelvis pose
  - COM target
  - left/right foot pose targets
- annotations:
  - stance side
  - contact mask / contact state
  - stop / resume markers
- metadata:
  - generator version
  - command-bin bounds
  - validation and feasibility flags

Exit criteria:

- one documented prior schema exists
- one viewer and one trace logger can consume that schema
- rate limits and bounds are explicitly defined
- active runtime FSM path is marked deprecated in the design docs

### `v0.20.0-B`: `ZMP` prior generator and viewer

Expectation:

- the offline prior should already encode recognizable walking semantics before
  policy learning

Quick visual check:

- walk command shows clear alternating step-out pattern
- stop command settles into quiet standing
- resume restarts stepping without discontinuity

How this advances the end goal:

- proves gait structure is prior-owned, so PPO can remain correction-focused

Goal:

- implement the first production prior as a `ZMP`-shaped offline reference
  library using a concrete, reviewable pipeline

Reference implementation target:

- planning family: Kajita et al. `ZMP` preview control
- public code reference: ToddlerBot
  `toddlerbot.algorithms.zmp_planner` and
  `toddlerbot.algorithms.zmp_walk`
- intended WildRobot pipeline:
  - command bin / footstep plan
  - `ZMP` preview-control COM plan
  - swing-foot trajectory generation
  - IK / nominal joint target generation
  - lookup-table export indexed by command

Visual gate:

- at `cmd=0.15`, feet clearly step out instead of shuffling
- stop command settles to quiet standing
- resume command restarts stepping without a jump

Metric gate:

- commanded foothold mean `>= 0.045 m`
- step period remains within `0.45-0.75 s`
- preview continuity and rate limits pass on command sweeps
- no validated reference trajectory requires systematic joint-limit violation

### `v0.20.0-C`: Reference + IK + short-horizon feasibility validation

Expectation:

- prior trajectories remain trackable after IK and full MuJoCo dynamics on
  short horizons
- a small, bounded validation-only feedback can produce visible forward
  walking — proving the prior is rescuable by a "minor correction" layer
  before PPO is asked to learn that correction

How this advances the end goal:

- verifies the prior is physically usable as an imitation target for PPO
- catches priors whose kinematic motion looks plausible but cannot be
  rescued by realistic closed-loop feedback (the v0.19.5 failure mode)

Status: **proposed split into C1 + C2 — pending external review**.
The original single-gate wording (open-loop "realized step length ≥ 0.03 m"
and "realized/cmd ratio ≥ 0.5" measured in free-floating replay) was
unmeetable because LIPM is unstable open-loop and `v0.20.0` explicitly
forbids a runtime balance controller.  See "Pivot rationale" below.

Goal:

- prove the prior remains coherent after IK and nominal target adaptation
- prove the offline prior is trackable enough in full MuJoCo dynamics to be
  a valid imitation target for PPO
- do not require long-horizon open-loop walking from the offline prior
- do require that a bounded validation-only stabilizer can extract forward
  walking from the prior (C2)

Measurement contract for this gate:

- `control steps` means env control-loop ticks (`ctrl_dt`), not MuJoCo
  substeps
- `footfalls` / `touchdowns` are detected from MuJoCo `data.contact`
  foot-geom / floor-geom pairs, NOT from the kinematic reference
  contact mask (the reference still ticks normally during a fall)
- per-mode usage of metrics:
  - kinematic playback: q_ref correctness only (forward speed, IK
    reachability, joint limits)
  - fixed-base PD (pelvis pinned): per-joint tracking RMSE and
    saturation only — pelvis is held in place, so foot world
    positions do not advance and stride is not measurable here
  - free-floating C1 (no feedback): survival, touchdown counts,
    failure classification only
  - free-floating C2 (validation-only stabilizer): closed-loop
    realized step length, realized/cmd ratio, saturation under
    feedback

#### Pivot rationale (why a C1 + C2 split)

The original single-gate wording mixed open-loop survival metrics with
closed-loop walking metrics in the same free-floating replay.  Two
concrete problems surfaced:

1. **LIPM is unstable open-loop.** Without state feedback, an LIPM
   trajectory falls within 1-3 seconds because the COM diverges from
   the foot.  This is a property of the inverted pendulum, not of the
   prior's quality.  ToddlerBot's reference implementation
   (`toddlerbot.algorithms.zmp_walk` + `walk_env.py`) never validates
   the prior in free-floating physics — the prior is consumed
   kinematically during library generation, and PPO is the
   stabilizer.  Open-loop forward-stride metrics are not meetable for
   any pure offline LIPM/ZMP prior.
2. **Pure kinematic validation does not catch all failure modes.**
   `v0.19.5` shipped a kinematically coherent reference that PPO
   could not stabilize.  Validating only kinematically would have
   passed the same defective prior.

The split resolves both:

- **C1** keeps an open-loop coherence check — no closed-loop
  performance metrics, only that the prior does not corrupt itself
  during the first cycles (no NaN, no joint runaway, no q_ref
  discontinuity, no instant pitch-faceplant).
- **C2** adds a bounded validation-only stabilizer harness that
  measures whether a "minor correction" layer can extract forward
  walking from the prior.  The harness exists ONLY in the validation
  toolchain (`view_zmp_in_mujoco.py --c2-stabilizer`), is forbidden
  from the runtime adapter, and tests the same hypothesis PPO will
  later verify by training: that the prior is rescuable.

D is unblocked only if **both C1 and C2 pass on the matrix** (see
"Decision rule" below).

#### C1 — open-loop coherence

C1 verifies the prior does not corrupt itself before PPO is involved.
It is a strict prior-quality gate, not a closed-loop performance
gate.

Visual gate (C1):

- in **kinematic replay**: touchdown locations match the intended
  stepping pattern, no toe drag / scissoring / pathological posture,
  stop and resume settle cleanly
- in **fixed-base PD** (pelvis pinned): legs visibly swing through
  the q_ref pattern with no servo runaway, oscillation, or growing
  error envelope
- in **free-floating, no feedback**: startup and the first 1-2 gait
  cycles remain coherent before the open-loop fall — failure looks
  like a bounded balance loss (pitch / roll exceeds bound), not
  q_ref breakdown

Metric gate (C1):

- *Reference correctness* (kinematic playback):
  - realized forward speed = commanded vx within numerical noise
  - realized touchdown step length mean `>= 0.03 m` (kinematic stride
    truth)
  - realized / commanded step ratio `>= 0.95`
  - IK reachability `≈ 1.0`; safety margin violations = 0
- *Servo trackability* (fixed-base PD):
  - per-leg-joint tracking RMSE `<= 0.40 rad` per joint, `<= 0.25 rad`
    overall — see open question Q1
  - any-joint saturation `< 5 %` of replayed control steps
  - per-joint MAE stable (within ±20 %) across the command range
- *Open-loop coherence* (free-floating, no feedback):
  - short-horizon env replay survives at least
    `ceil(2.0 * cycle_time / ctrl_dt)` control steps before
    catastrophic pitch termination.  This is a **horizon tied to
    gait period**, not a fixed 45 steps — at the default
    `cycle_time=0.64 s, ctrl_dt=0.02 s` it evaluates to 64 steps
    (1.28 s); at the spec's longest allowed period
    `cycle_time=0.75 s` (see `v0.20.0-B` metric gate) it evaluates
    to 75 steps (1.50 s).  The 2× multiplier guarantees at least
    two full gait cycles regardless of seed phase offset, so the
    "≥ 1 L + 1 R touchdown" requirement below is reachable in every
    replay rather than phase-dependent
  - touchdown progression includes at least one left and one right
    physical foot/floor contact in the same replay (guaranteed by
    the cycle-tied horizon above)
  - failure cause is balance loss (pitch > 0.8 rad or root_z < 0.15
    m), NOT joint runaway, q_ref discontinuity, or NaN
  - no preview wrap discontinuity (linear-replay contract intact —
    enforced by `ReferenceLibrary.get_preview` overrun warning)

#### C2 — validation-only stabilizer harness

C2 adds a small bounded feedback channel inside the validation
viewer to measure realized walking under the prior.  The harness is
NOT a runtime controller and MUST NOT be exported into the runtime
adapter.

Harness scope (HARD bounds — must be enforced by code review):

| Channel | Form | Hard clip | Forbidden inputs |
|---|---|---|---|
| Torso pitch PD | `K_p · pitch_err + K_d · pitch_rate` → ankle pitch offset on stance side | `±0.10 rad` | gait phase from FSM, foothold preview, contact-mask annotations |
| Torso roll PD | same form → hip roll offset on stance side | `±0.05 rad` | foothold preview, contact-mask annotations |
| Capture-point swing nudge | `(x_cp − x_target_swing) · gain` → swing-x offset, applied only at foot lift | `±0.03 m` | timing changes, foothold re-planning, runtime FSM state |

**Allowed signals for "stance side" and "foot lift" identification
(and only these signals)**: (a) measured MuJoCo foot/floor contacts
via `data.contact` geom-pair lookup, mirroring the touchdown
detection used for C1 metrics; (b) time-derived gait phase
`(sin, cos)(2π · t / cycle_time)` where `t` is the replay clock and
`cycle_time` is read from the trajectory metadata.  These are the
same signals the runtime PPO policy will see — measured contact and
phase — so the harness cannot use any source of stance information
that PPO will not also have.  Reading
`traj.stance_foot_id`, `traj.contact_mask`, any FSM annotation, or
any per-step lookup into the stored reference's contact schedule is
forbidden.

Forbidden everywhere: re-reading gait phase to decide what to do,
recomputing footstep timing, scaling by stance side via a runtime
FSM, importing the harness into `runtime/`.  The harness is
instantiated only inside `view_zmp_in_mujoco.py` and `eval_*` paths,
gated by `--c2-stabilizer`.

If the stabilizer ever needs more than the inputs above to make C2
pass, the prior has failed the gate — do not loosen the harness.

Visual gate (C2):

- learned motion (= prior + harness) shows visible forward walking,
  not standing-shuffle, lean-back exploit, or backward drift
- gait still resembles the prior family (alternating stepping
  pattern preserved); harness output stays within hard clips for the
  whole replay
- stop / resume remains recognizable

Metric gate (C2, on the closeout matrix — see Q2/Q3 for sweep
parameters):

- realized touchdown step length mean `>= 0.03 m` (physical contacts)
- realized / commanded step ratio `>= 0.5`
- any-joint saturation: target `< 20 %`, hard fail `>= 25 %` (see
  open question Q3)
- tracking RMSE within the C1 budget above (no relaxation under C2)
- at least one left and one right physical touchdown in each replay
- harness clip-saturation (% of steps where any harness output is
  pinned at its clip): target `< 10 %`, hard fail `>= 25 %` — high
  clip-saturation indicates the prior is being patched, not rescued

#### Decision rule for v0.20.0-D

| C1 | C2 | Action |
|---|---|---|
| pass | pass on all bins | promote to v0.20.0-D |
| pass | partial pass / variance across seeds | stay in C; tune stabilizer gains within the hard clips, or align initial pelvis state with LIPM IC; do NOT relax clips |
| pass | fail on all bins | stay in C; investigate prior quality (lateral asymmetry, knee feasibility) — stabilizer cannot rescue what is not rescuable |
| fail | n/a | fix prior generation; do NOT spend effort on the stabilizer first |

#### Open questions for external review (decide before kickoff)

The four numbers / choices below are intentionally left open for
the external reviewer.  They control how strict the gate is and how
the sweep is parameterized.  Each option is annotated with the
internal recommendation, the external-review verdict (when known),
and the trade-off so the reviewer can pick or override.

**Q1 — Tracking-RMSE budget (per-joint and overall)**

Internal recommendation: **per-joint `≤ 0.40 rad`, overall `≤ 0.25
rad`**.

External-review verdict: **accept**, but treat as a C-only
feasibility budget rather than a signoff that servo tracking is
healthy.  Knee error is already near the edge — closeout artifact
must explicitly print **worst-joint RMSE per (vx bin, seed)** so
the SysID debt remains visible and is carried into `v0.20.2` rather
than hidden by an aggregate "pass".

Trade-off: current fixed-base data shows knee RMSE 0.34-0.36 rad and
overall 0.19-0.20 rad.  At 0.40 / 0.25 the gate just passes today.
At 0.30 / 0.20 the knee fails immediately and v0.20.0-C is blocked
on knee SysID — which is real signal but defers C2 work indefinitely.
Tightening here is conservative; loosening invites a real knee
tracking issue to bleed into v0.20.1 PPO training.

**Q2 — What the 3 seeds randomize**

Internal recommendation: **initial pelvis perturbation** (small ±x /
±y / ±yaw on the standing keyframe).

External-review verdict: **accept**, with one refinement: include a
small **initial forward-velocity perturbation** as part of the same
IC probe.  This directly probes the LIPM-IC mismatch (standing
starts at `vx=0` while the q_ref expects `vx ≈ +0.18 m/s` at the
default command).  Do NOT add domain randomization at C — that
belongs to `v0.20.1` / `v0.20.2`.

Closeout-row IC parameters per seed:
- pelvis position: `±0.02 m` (x, y), `±5°` yaw, uniform
- pelvis forward velocity: `±0.10 m/s` x-component, uniform
- everything else (joints, mass, friction, motor delay) deterministic

Options considered:
1. Initial pelvis perturbation (recommended above) — directly probes
   the LIPM-IC mismatch driving today's open-loop fall, cheapest
2. Initial joint-pos noise (±2° on legs) — tests standing-pose
   variation, less targeted
3. Domain randomization (friction, mass, motor delay) — closest to
   v0.20.1 training conditions, biggest scope, properly belongs to
   later milestones

**Q3 — Saturation hard-fail threshold**

Internal recommendation: **target `< 20 %`, hard fail `>= 25 %`**.

External-review verdict: **accept as written**.

Trade-off: free-floating without C2 currently shows 28-61 %
saturation, almost entirely fall-driven.  If C2 is doing its job the
fall is prevented and saturation should drop into the single digits.
A 25 % hard fail catches the case where C2 only partially stabilizes
(prior stays upright but legs are still pegged at limits).  Looser
(`>= 35 %`) makes C2 mostly redundant with C1 survival.

**Q4 — Should the harness allow any feedback on the swing trajectory
beyond the capture-point nudge?**

Internal recommendation: **no — capture-point only, ±0.03 m clip**.

External-review verdict: **accept**, flagged as the most important
boundary in the section.  Any richer swing adaptation turns C2 from
a validation harness into the runtime controller the design just
deprecated.

#### Closeout matrix and artifact

The closeout sweep covers four measurement modes.  Two are
deterministic baselines (one row per vx); two are seed-randomized
closeout rows (one row per (vx, seed)).  Total **32 rows**, with
the **24 free-floating rows carrying the gate decision**.

| Mode | Rows | Purpose | Seeds | Total |
|------|------|---------|-------|-------|
| Kinematic playback | 1 per vx, deterministic | sanity baseline: q_ref encodes correct forward motion | n/a | 4 |
| Fixed-base PD (pelvis pinned) | 1 per vx, deterministic | servo-trackability baseline: per-joint RMSE and saturation | n/a | 4 |
| Free-floating, no stabilizer (C1) | 3 per vx | open-loop coherence, survival, real-contact touchdowns | seeds 0, 1, 2 | 12 |
| Free-floating, C2 stabilizer (`--c2-stabilizer`) | 3 per vx | closed-loop realized step length, ratio, saturation, harness clip-saturation | seeds 0, 1, 2 | 12 |

Bins: `vx ∈ {0.10, 0.15, 0.20, 0.25}`.  Horizon per row: 200
control steps (4 s at `ctrl_dt=0.02`).  Initial-condition
randomization (Q2 outcome) applies to the 24 free-floating rows
only; the 8 baseline rows are deterministic given the standing
keyframe.

The C1 + C2 gate decision is computed from the 24 free-floating
rows.  The 8 baseline rows confirm prior correctness and servo
trackability; if they fail, fix the prior or servo model before
collecting the free-floating sweep (per the decision rule above).

CHANGELOG block (single contiguous section under the v0.20.0-C
heading) must contain:

- raw matrix table, all 32 rows, one row per (mode, vx, seed)
- per-row metrics: survival ctrl steps, L+R touchdown counts,
  realized step length per side, realized/cmd ratio per side,
  overall RMSE, **worst-joint RMSE with joint name** (Q1 amendment),
  any-joint saturation %, harness clip-saturation % (C2 only)
- exact command lines used to produce each row
- seed list (integers, RNG family used for IC perturbation)
- pass / fail annotation per criterion in both the C1 and C2 metric
  gates, evaluated row-by-row and aggregated
- sha of the validation harness commit and the on-disk reference
  library directory hash

### `v0.20.0-D`: PPO unblock decision

Expectation:

- this is a go/no-go gate, not a behavior-creation stage

Quick visual check:

- unblock only when prior replay already looks like a real walk family and not
  a balance exploit

How this advances the end goal:

- avoids launching PPO on an unready prior and repeating earlier reward-exploit
  loops

PPO is unblocked only if:

- `v0.20.0-A` and `v0.20.0-B` pass
- `v0.20.0-C1` (open-loop coherence) passes on all bins of the
  closeout matrix
- `v0.20.0-C2` (validation-only stabilizer harness) passes on all
  bins of the closeout matrix without exceeding the hard clips on
  the harness channels
- the offline prior visually looks like a real walk family, not a
  balance exploit
- the remaining issues are plausibly correction-layer problems
  (i.e. the things PPO is supposed to learn — touchdown
  stabilization, contact adaptation, servo-delay rejection — and not
  basic gait structure or propulsion)

### `v0.20.1`: ToddlerBot-style PPO

Expectation:

- PPO should improve stability and tracking around the prior, not replace the
  gait structure

Quick visual check:

- learned motion still matches the reference family
- stepping becomes more stable and less fragile, without drifting into a new
  exploit style

How this advances the end goal:

- adds closed-loop correction required for sustained command-conditioned
  walking

Goal:

- train the first policy as a bounded residual, imitation-dominant
  reference-conditioned PPO
- keep the architecture compatible with a later Cassie-style direct-policy
  upgrade, but do not take that step yet

Visual gate:

- by the smoke horizon, the learned policy produces visible reference-like
  stepping rather than standing or inventing a new gait
- stop / resume remains recognizable
- the learned gait still resembles the reference family

Metric gate:

- reference-tracking metrics improve materially from initialization:
  - `ref/joint_track_rmse`
  - `ref/pelvis_pose_error`
  - `ref/foot_site_error`
  - `ref/contact_phase_match`
- by the early smoke horizon:
  - `env/forward_velocity >= 0.04 m/s`
  - `eval_walk/episode_length >= 150`
  - `term_pitch_frac < 0.50`
- by the promotion horizon:
  - `env/forward_velocity >= 0.075 m/s`
  - `eval_walk/success_rate >= 0.60`
  - `tracking/cmd_vs_achieved_forward <= 0.075 m/s`
  - touchdown step length mean `>= 0.03 m`

### `v0.20.2`: SysID verification + policy coverage + transfer hardening

Expectation:

- policy behavior extends beyond one command and the servo model is validated
  for hardware deployment

Quick visual check:

- zero command stands quietly
- low command shows slow but visible stepping
- higher commands increase speed without immediate pitch collapse

How this advances the end goal:

- turns a narrow sim policy into a transferable walking capability

Goal:

- verify the servo model against real hardware before sim-to-real deployment
- make sure the first ToddlerBot-style policy is not only valid at one exact
  operating point

SysID is a hard gate for sim-to-real deployment:

- collect hardware step-response captures (step, hold, chirp) using
  `tools/sysid/run_capture.py`
- fit servo parameters from captures and update the realism profile
- run sim-vs-real comparison via `tools/sim2real_eval/replay_compare.py`
- if fitted parameters differ materially from the existing MJCF model,
  re-run v0.20.0-C feasibility with the corrected model and regenerate the
  ZMP library if needed

SysID metric gate:

- sim-vs-real step-response RMSE per leg joint is within 15% of commanded
  amplitude
- delay and backlash estimates are versioned and stable
- fitted parameters are applied to the MJCF actuator model

Policy coverage visual gate:

- zero command means standing, not creeping
- low command means slow but visible stepping
- stop/resume looks controlled

Metric gate:

- zero-command drift `< 0.01 m/s`
- stop latency `< 1.0 s`
- resume-to-first-step latency `< 1.0 s`
- forward speed response is monotonic non-decreasing over the tested range

### `v0.20.3`: Cassie-upgrade decision gate

Expectation:

- architecture upgrades are triggered by measured bottlenecks, not preference

Quick visual check:

- if residual policy is stable but adaptation-limited, consider Cassie-style
  upgrade
- if current issues are prior-quality issues, continue improving the prior

How this advances the end goal:

- keeps complexity proportional to evidence and preserves forward progress

Goal:

- decide whether the ToddlerBot-style bounded residual contract is sufficient
- move to a Cassie-style broader-authority policy only if the residual contract
  is the limiting factor

Promote to Cassie-style only if:

- the offline prior library is already working
- the imitation-dominant residual policy has cleared `v0.20.2`
- remaining limitations are clearly adaptation / authority limits rather than
  prior-quality limits

If promoted, keep:

- the same offline reference library
- the same preview interface
- the same validation tools

What the first policy should own:

- tracking correction
- disturbance recovery
- contact adaptation
- delay / backlash / model-mismatch robustness

What the first policy should not own:

- basic step release
- core propulsion generation
- stop semantics
- command-conditioned gait structure

---

## Historical Companion

Historical `walking_ref_v1/v2`, `M2.5`, `v0.19.x`, and `v0.19.4` DCM COM
trajectory notes now live in
[reference_design_history.md](reference_design_history.md) so this
document stays focused on the active `v0.20.x` offline-prior path.

## Relationship To Main Walking Plan

This document is a focused design companion to:

- [`walking_training.md`](/home/leeygang/projects/wildrobot/training/docs/walking_training.md)

`walking_training.md` remains the main milestone plan.
This file exists to keep the nominal reference architecture and `M2.5`
validation approach explicit and stable while the locomotion stack is still
being refactored.
