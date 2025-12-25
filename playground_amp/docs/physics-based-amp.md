Below is a concrete roadmap to shift your current pipeline to Option 1, leveraging what you have already built (Step 2A FD parity, robust parity tests, Test 1B diagnostics and failure classifiers). It is written as an engineering execution plan with deliverables, acceptance gates, and minimal risk sequencing.

---

## Goal definition

**Option 1 target state:** the discriminator “real” distribution is drawn from **physics generated reference rollouts** (not raw kinematic AMASS or GMR outputs), and those rollouts are curated by feasibility gates so the reference stays on the robot’s realizable manifold.

---

## Phase 0: Freeze interfaces and invariants (no behavior change)

### Deliverables

1. **Single source AMP feature extractor**

* One function used everywhere:

  * dataset generation
  * training rollout feature extraction
  * parity tests
* Uses your Step 2A FD velocity methods (heading local linvel and angvel).

2. **Reference dataset schema spec**

* Exact keys, shapes, and metadata expected by the discriminator sampler.
* Add versioning fields (feature_version, dt, robot_model_id, contact_mode).

### Acceptance criteria

* Current training still runs with existing reference dataset, no regression.
* Test 1A still passes.

---

## Phase 1: Make Test 1B physically meaningful (feasibility harness hardening)

You already identified the puppet string issue. Convert Test 1B from “keep it alive” to “measure feasibility.”

### Deliverables

1. **Stabilization cap and gating**

* Implement:

  * reduced gains
  * hard cap on |F_stab| (10–15% of mg)
  * enable only when |pitch| and |roll| < 25 degrees
* Log stabilization contribution: mean(|F_stab|)/mg and percent of steps saturated.

2. **Force based validity rules**

* Friction margin computed only when normal force is meaningful:

  * per foot Fn threshold = 5% mg (diagnostic) and 10% mg (high confidence)
* Add summary of frames “unloaded” vs “loaded.”

3. **Feasibility metrics export**

* Per clip CSV/JSON including:

  * mean(sum Fn)/mg
  * slip rate
  * max pitch/roll
  * contact alternation score
  * joint RMSE
  * stabilizer utilization

### Acceptance criteria

* In standing test (static pose), mean(sum Fn)/mg is near 1.0 with stabilizer capped.
* Stabilizer contributes less than 15% mg on average in “upright” segments.

---

## Phase 2: Physics reference generator MVP (core Option 1 shift)

### Deliverables

1. **Script: generate_physics_reference_dataset.py**
   Inputs:

* GMR motion files at 50 Hz (root pose and joint targets)
* MuJoCo model and PD controller config
* sim_dt, control_dt, n_substeps

Outputs:

* physics rollout trajectories (qpos series, optional qvel)
* physics derived contacts (preferred) and normal forces
* AMP features computed from realized physics states using the frozen extractor
* merged dataset file in the exact discriminator format

2. **Quality gate integration**

* Use your feasibility metrics to:

  * accept full clip
  * accept feasible segments only (trim)
  * reject clip
* Write per motion metadata:

  * pass/fail reason
  * trimmed ranges
  * key metrics

### Acceptance criteria

* Dataset loads into the discriminator with zero code changes.
* For accepted segments, mean(sum Fn)/mg ≥ 0.85 and stabilizer contribution is within cap.
* Feature extraction on a saved physics rollout is deterministic (replay and compare within tolerance).

---

## Phase 3: Training switch over (use physics reference as “real”)

### Deliverables

1. **Training config branch**

* A new config option:

  * real dataset path = physics reference dataset
  * disable any older reference normalization statistics tied to raw kinematic reference

2. **Normalization re computation**

* Compute normalization stats only from physics reference features:

  * mean/std per dimension
  * track and store with dataset version

3. **Diagnostic dashboards**
   Add logs you already planned, with two new comparisons:

* real vs fake:

  * contact alternation rate
  * slip proxy if available
* discriminator:

  * AUC, logit saturation, reward percentiles

### Acceptance criteria

Early training sanity:

* D(real) and D(fake) separate within early windows (AUC > 0.6 quickly)
* amp reward distribution is not collapsed near zero
* disc_acc is not pegged at 0.5 for long periods and not saturating to 1.0 instantly

---

## Phase 4: Curriculum and coverage expansion

You already observed that many clips are poor. Curriculum is the pragmatic way to scale.

### Deliverables

1. **Tiered dataset**

* Tier 0 strict gates
* Tier 1 relaxed gates
* Tier 2 experimental
  Store each as separate dataset file or one dataset with tier labels.

2. **Curriculum schedule**

* Start with Tier 0 only
* Mix in Tier 1 gradually as gait stabilizes
* Keep Tier 2 for ablations

3. **Clip level weighting**

* Weight “better” clips higher (lower slip, better alternation, better Fn support)
* Prevent the discriminator from being dominated by marginal clips

### Acceptance criteria

* As tiers expand, discriminator remains stable (no sudden collapse)
* Policy maintains success metrics while style improves

---

## Phase 5: Long term improvements (reduce dependence on PD filtering)

Once Option 1 is stable, you can improve realism by reducing the “PD imprint.”

### Deliverables

1. **Retargeting constraints**

* enforce minimum swing clearance
* step length and velocity caps
* explicit contact alternation constraints

2. **Better controller for reference generation**

* Replace pure PD with:

  * task space foot tracking
  * simple balance controller (COM over support polygon)
    This improves the physics reference quality without changing AMP.

3. **Contact representation refinement**

* Use physics contacts as ground truth for reference
* Use estimator only for policy if needed, but keep parity.

---

## Recommended sequencing from your current state

Given your work:

* Step 2A parity is done
* diagnostics exist
* Test 1B found puppet stabilization issue

You should execute in this order:

1. Phase 1 (harden Test 1B and stabilization cap)
2. Phase 2 (physics dataset generator MVP + gating)
3. Phase 3 (training switchover + normalization recompute)
4. Phase 4 (curriculum and dataset expansion)

Do not invest in complex retargeting constraints until Option 1 training is stable.

---

## Key risk register and mitigations

1. Risk: physics reference generator produces few or no pass segments

* Mitigation: allow trimming to feasible segments and relax gates for Tier 1

2. Risk: reference becomes “PD style,” not human like

* Mitigation: later replace PD controller with a slightly smarter tracker, without changing AMP

3. Risk: training still unstable after switch

* Mitigation: check normalization source, sampling balance, discriminator saturation, and real vs fake feature distribution overlap

---

## Operational definition of success

You have “successfully shifted to Option 1” when:

* the discriminator “real” dataset is physics generated
* training no longer exhibits low amp reward collapse caused by kinematic vs physics mismatch
* adding more motions improves style rather than destabilizing learning

---

If you paste the exact merged dataset schema your discriminator expects (keys and shapes), I can give you a precise output contract for `generate_physics_reference_dataset.py` so you can implement Phase 2 without changing any downstream code.
