# Human-Like Walking Strategy (Concise Roadmap)

## Overview
Goal: Train WildRobot to exhibit stable, efficient, human-like bipedal walking using a staged curriculum that layers locomotion competence, contact pattern shaping, motion style, robustness, and deployment readiness.

We minimize premature complexity early (avoid imitation & adversarial style until basic gait is reliable) and progressively introduce higher-fidelity objectives once stability and forward progression metrics are consistently met.

---
## Phase 0: Foundations (Done / Iterating)
Focus: Forward motion incentive + prevention of stationary local optimum.
Key Tools:
- Velocity tracking (exp + squared) with gating & gentler threshold penalty (decaying).
- Forward velocity bonus (now reduced to avoid jitter).
Adjustments Pending:
- Activate contact rewards (light) to start shaping foot placement.
Success Criteria:
- Mean forward velocity ≥ 0.55 m/s
- Gate active rate ≤ 15%
- Distance walked per eval ≥ 9.5
- Falls (1 - success_rate) < 55%

---
## Phase 1: Contact Pattern Stabilization
Focus: Achieve consistent alternating foot contacts, reduce sliding, maintain upright balance.
Add:
- Foot contact reward (moderate, weight: 5.0)
- Sliding penalty (light, weight: 2.0)
- Optional phase clock (already present) leveraged for alignment metric.
Metrics to Monitor:
- **contact/alternation_ratio**: L→R→L cycle quality (0.0-1.0)
- **contact/left_sliding_vel & contact/right_sliding_vel**: Foot sliding speed
- **forward_velocity**: Maintain Phase 0 baseline (≥0.55 m/s)
- **success_rate**: Episode completion without falls (>55%)
- **contact/both_meet_air_time_threshold**: Proper swing phase (≥0.15s)
Progress Gate to Phase 2 when:
- ✅ Alternation ratio > 0.85 (85%)
- ✅ Average sliding velocity < 0.15 m/s
- ✅ Forward velocity ≥ 0.55 m/s (maintained from Phase 0)
- ✅ Success rate > 55% (falls < 45%)

---
## Phase 2: Motion Reference Integration (Imitation)
Focus: Introduce stylistic targets from mocap / procedural reference.
Add Rewards:
- Joint pose tracking (selected major joints)
- COM velocity & height tracking
- Foot clearance timing (lift → swing → plant)
Anneal imitation weights down as task proficiency rises to avoid overfitting.
Criteria:
- Pose tracking error < threshold (e.g. normalized RMSE < 0.25)
- No degradation in forward velocity (>0.65 m/s maintained)

---
## Phase 3: Style Discriminator (AMP Hybrid)
Focus: Naturalistic motion texture (subtle sway, arm swing, cadence).
Components:
- Discriminator trained on curated “good gait” segments (real or synthetic).
- Style reward blended with task reward (r_total = r_task + w_style * r_style).
Ramp w_style from 0 → target over ~1M env steps.
Monitor:
- Discriminator score convergence
- Diversity vs collapse (entropy, footfall variance)

---
## Phase 4: Energy & Smoothness Refinement
Introduce normalized penalties only after gait stable:
- Joint acceleration (scaled, not raw 50k magnitudes)
- Mechanical power / distance (efficiency)
Tune to lower energetic cost without harming velocity or stability.
Criteria:
- Energy per meter improves ≥15% vs Phase 3 baseline
- No increase in fall rate

---
## Phase 5: Robustness & Domain Randomization
Randomize:
- Friction, mass distribution, minor pushes, sensor noise.
Add recovery reward (return COM over support within N steps).
Graduated schedule increasing variance.
Criteria:
- Performance drop under randomization < 15%
- Recovery success > 80% (push scenarios)

---
## Phase 6: Terrain & Directional Generalization
Extend to slopes, mild irregular terrain, lateral & yaw commands.
Add heading tracking reward, penalize cross-over foot placement.
Criteria:
- Maintain ≥0.6 m/s forward on varied terrain
- Successful directional transitions (yaw change response time ≤ target)

---
## Phase 7: Style Fine-Tuning
Micro-adjust:
- Arm swing amplitude (phase relation to legs)
- Pelvis sway and vertical excursion
- Step length variability (avoid monotony)
Use additional small discriminators or direct kinematic heuristics.

---
## Phase 8: Compression & Deployment
Distill larger policy → smaller network (behavior cloning with KL regularization).
Validate latency + determinism; export ONNX.
Add safety envelopes (joint limits, torque clamps).

---
## Key Metrics Dashboard (Always Track)
- **forward_velocity**: Mean forward velocity (m/s)
- **tracking_gate_active_rate**: % of time velocity tracking is disabled
- **velocity_threshold_scale**: Penalty decay schedule (1.0 → 0.0)
- **contact/alternation_ratio**: Gait cycle quality (0.0-1.0, target >0.85)
- **contact/left_sliding_vel & contact/right_sliding_vel**: Foot sliding (m/s, target <0.15)
- **success_rate**: Episode completion rate (%, target >55%)
- **contact/both_meet_air_time_threshold**: Proper swing phase (0.0-1.0)
- **reward_per_step**: Total reward (watch for plateau or collapse)
- **style_discriminator_score**: (Phase 3+)
- **energy_per_meter**: (Phase 4+)
- **joint_acceleration_norm**: (Phase 4+)
- **recovery_success_rate**: (Phase 5+)

---
## Intervention Guide
**If Gate Active Rate High**: Lower gate velocity or increase gate scale.
**If Velocity Plateau <0.4 m/s**: Increase forward_velocity_bonus slightly OR reduce early penalties (velocity_threshold_penalty weight).
**If Alternation Ratio Stuck <0.70 after 5M steps**: Increase foot_contact reward weight (5.0 → 7.0) OR adjust phase_period for gait timing.
**If Sliding Persists >0.25 m/s**: Increase foot_sliding penalty gradually (2.0 → 3.0 → 5.0) only after forward velocity stable.
**If Catastrophic Forgetting Detected**: Load checkpoint from peak steps and reduce learning rate OR adjust reward weights.
**If Energy Cost Explodes**: Delay acceleration penalty; confirm torque weight not too low (avoid reckless compensation).
**If Style Discriminator Collapses**: Increase entropy bonus temporarily; reduce style weight ramp speed.

---
## Simplified Timeline (Approximate)
- 0–5M steps: Baseline + contact activation
- 5–10M: Contact stable; introduce imitation
- 10–15M: Add style discriminator
- 15–18M: Energy & smoothness refinements
- 18–22M: Robustness randomization
- 22–26M: Terrain & directional generalization
- 26–30M: Fine style + compression

(Adjust based on metric attainment rather than fixed step counts.)

---
## Next Immediate Actions
1. ✅ Contact alternation ratio implemented and tracked in `contact/alternation_ratio` metric
2. ✅ Early stopping protection added to prevent catastrophic forgetting (60-98% compute savings)
3. ✅ Phase 1 config optimized: foot_contact=5.0, 10M steps, phase0 warm-start
4. ⏳ Run Phase 1 training and monitor for success criteria:
   ```bash
   python3 amp/train.py amp/phase1_contact.yaml
   python3 amp/monitor_training.py amp/training_logs/<dir> --watch --force_stop
   ```
5. ⏳ Watch for alternation_ratio >85% and sliding <0.15 m/s to gate Phase 2

---
## Roadmap TODO Checklist
Use this phased checklist; advance only when criteria met.

### Phase 0: Foundations
- [ ] Decay activates (`velocity_threshold_scale < 1.0` after ≥150k env steps)
- [ ] Forward velocity_mean ≥ 0.55 m/s (stable 3 evals)
- [ ] Gate active rate ≤ 0.15
- [ ] Foot contact & sliding metrics non-zero

### Phase 1: Contact Pattern
- [ ] `contact/alternation_ratio` > 0.85 (85%)
- [ ] `contact/left_sliding_vel` & `contact/right_sliding_vel` avg < 0.15 m/s
- [ ] `forward_velocity` ≥ 0.55 m/s (maintained)
- [ ] `success_rate` > 0.55 (fall rate < 0.45)
- [ ] `contact/both_meet_air_time_threshold` > 0.7 (70% of episodes)

### Phase 2: Imitation
- [ ] Pose tracking RMSE < 0.25
- [ ] Forward velocity ≥ 0.65 m/s (no regression)
- [ ] COM height variance within band

### Phase 3: Style (AMP)
- [ ] Discriminator score plateaus (no collapse)
- [ ] Policy entropy above floor
- [ ] Footfall variance preserved

### Phase 4: Efficiency
- [ ] Energy per meter ↓ ≥15% vs Phase 3
- [ ] Acceleration penalty added without >5% velocity loss
- [ ] Fall rate unchanged

### Phase 5: Robustness
- [ ] Randomization performance within 15% of nominal
- [ ] Recovery success ≥80%
- [ ] Style metrics stable

### Phase 6: Terrain & Direction
- [ ] Forward velocity ≥0.6 m/s on varied terrain
- [ ] Yaw response time ≤ target
- [ ] Cross-over foot placements minimized

### Phase 7: Fine Style
- [ ] Arm swing amplitude in target range
- [ ] Pelvis sway & vertical excursion within bands
- [ ] Step length variability > threshold

### Phase 8: Deployment
- [ ] Distilled policy KL < threshold
- [ ] Inference latency < budget
- [ ] ONNX parity tests pass
- [ ] Safety envelopes validated

### Continuous
- [ ] Dashboard metrics updated
- [ ] Alert thresholds configured
*This concise strategy emphasizes staged complexity, metric gates, and adaptive weighting to avoid early overfitting or reward interference.*
