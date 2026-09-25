# WR Morphology Size Factorial

**Purpose:** determine whether the current actuator saturation is caused by
WR's physical length scale, its mass, or their interaction. This replaces the
pose-only stance experiment for that question: changing joint pose did not
change hip spacing, leg length, mass, or inertia.

## Questions and contrasts

The experiment is a matched 2x2 factorial. Every arm keeps the same HTD-45H
model and torque limit.

| Arm | WR lengths | WR mass | Causal comparison |
|---|---:|---:|---|
| `baseline` | 1.000 | 1.000 | Current robot |
| `dimension_only` | ToddlerBot/WR | 1.000 | Dimension effect at fixed mass |
| `mass_only` | 1.000 | ToddlerBot/WR | Mass effect at fixed dimensions |
| `dimension_and_mass` | ToddlerBot/WR | ToddlerBot/WR | Combined TB-sized counterfactual |

The source values are taken from code/model data rather than estimates:

- WR reference leg length: `0.193 + 0.180 = 0.373 m` in
  `control/zmp/zmp_walk.py`.
- ToddlerBot hip-to-ankle-pitch length: `0.2115 m` in
  `~/projects/toddlerbot/toddlerbot/descriptions/toddlerbot_2xm/robot.yml`.
- WR compiled model mass: `4.0371656 kg`.
- ToddlerBot 2xm compiled model mass: `3.76887188 kg`.

This gives a default length scale of about `0.567` and mass scale of about
`0.934`. ToddlerBot is only about 6.6% lighter, so the factorial does not
silently attribute its much shorter legs to mass.

## What is scaled

For the dimension arms, every MJCF body position, inertial offset, collision
primitive, site, mesh, and home free-root translation is uniformly scaled.
Link masses stay fixed in `dimension_only`; inertias scale as `mass * length²`.
The combined arm additionally scales every link mass to the ToddlerBot total.

The reference contract is scaled consistently:

- link lengths, stance width, hip offset, foot clearance, and distance
  thresholds scale with length;
- gait cycle scales with `sqrt(length scale)` (Froude similarity), making the
  default target approximately ToddlerBot's `0.72 s` cycle;
- the 20 ms controller, HTD dynamics, torque limits, action scale, PPO setup,
  and absolute deployment commands remain unchanged;
- the single-support COM-lever normalization and foot-height reward basin are
  rescaled so the reward does not change merely because meters changed.

The generated models preserve the same `nq`, `nv`, `nu`, joint ranges, policy
shape, and actuator force limits. Each training arm starts from the same TB9
actor with a fresh critic and optimizer. Checkpoints are saved every iteration
so the exact final state is retained.

## Important limitation

Uniformly scaling WR while retaining the same HTD actuator is an idealized
causal simulation counterfactual. It answers whether length and mass explain
the load, but it is not a packaging-valid CAD design: real servos, brackets,
wall thicknesses, wiring, and battery dimensions do not scale uniformly.
Only a positive result justifies preparing an actual CAD redesign.

## Runbook

First generate and audit all four variants without training:

```bash
uv run python training/scripts/run_morphology_size_factorial.py \
  --prepare-only
```

The printed manifest contains generated model/config paths, the exact training
commands, compiled mass and geometry checks, and a reference-feasibility audit
at both deployment speeds. It also audits the equal-Froude command for the
shortened arms without adding that unmatched command to policy evaluation.

Run the initial 20-iteration screen on the GPU machine:

```bash
uv run python training/scripts/run_morphology_size_factorial.py
```

Twenty iterations are a screening budget, not a deployment training budget.
Do not extend all arms automatically.

## Decision rule

Ignore lateral drift for this forward-only experiment, but do not relax falls,
orientation, or actuator headroom.

1. **Size is causal** only if `dimension_only` materially reduces the
   support-phase COM lever, bilateral hip-roll RMS, and worst stable saturation
   relative to `baseline` without increasing falls or peak/final tilt.
2. **Mass is causal** only if `mass_only` produces the corresponding reduction.
3. **TB-sized WR is sufficient** only if `dimension_and_mass` reaches `0/64`
   falls at both absolute deployment commands and every actuator has at most
   `5%` stable occupancy above 95% of its model limit.
4. If only the combined arm works, compute the factorial interaction before
   choosing dimensions; neither length nor mass alone is sufficient.
5. If no arm reaches the gate, reject resizing as a sufficient HTD-45H remedy
   and evaluate reduction gearing or higher-capacity hip/knee actuators.

For a positive arm, rerun only it and baseline using three seeds and at least
100 iterations before any CAD decision:

```bash
uv run python training/scripts/run_morphology_size_factorial.py \
  --arms baseline dimension_only \
  --seeds 42 43 44 \
  --iterations 100
```

Replace `dimension_only` with `mass_only` or `dimension_and_mass` according to
the initial result.

## References

- Shi et al., *ToddlerBot*, arXiv:2502.00893.
- ToddlerBot `toddlerbot/descriptions/toddlerbot_2xm/robot.yml`.
- ToddlerBot `toddlerbot/locomotion/walk.gin`.
- Hof, *Scaling gait data to body size*, Gait & Posture 4(3), 1996.
