# WR Lateral Stance: Dynamics Analysis and Pose-Only A/B

**Status:** ready for a fixed-policy simulator A/B using TB11 checkpoint 5.
**Date:** 2026-09-24

## Question Being Tested

The current deployment blocker is insufficient whole-leg actuator headroom,
especially hip roll and knee pitch. Before changing the robot dimensions, test
whether a narrower pose can reduce lateral support demand while preserving the
existing WR structure, mass, actuator model, gait timing, and policy contract.

This experiment does not assume that a wider robot is inherently unstable.
For geometrically and dynamically similar robots, a wider stance can be valid.
The relevant quantity is actuator demand relative to available capacity:

```text
actuator utilization ~= required joint torque / available joint torque

static lateral support moment ~= mass * gravity * COM-to-support-foot distance
```

Joint angular speed determines kinematic timing, but does not remove the
gravity moment required while stationary on one foot. It also does not, by
itself, match mass, rotational inertia, acceleration, or the motor's
torque-versus-speed envelope.

## WR and ToddlerBot Scaling

ToddlerBot uses a `0.037 m` centerline-to-foot distance and a `0.2115 m`
hip-to-ankle length. WR's current canonical stance uses `0.0782 m` and a
`0.373 m` leg length:

```text
ToddlerBot normalized half stance = 0.0370 / 0.2115 = 0.175
WR normalized half stance         = 0.0782 / 0.3730 = 0.210
```

WR is therefore about 20% wider after leg-length normalization. Dynamic time
is already scaled consistently:

```text
WR cycle = 0.72 s * sqrt(0.373 / 0.2115) ~= 0.96 s
```

This means gait timing is not the first variable in this A/B. The experiment
keeps WR's `0.96 s` cycle fixed.

References:

- Shi et al., *ToddlerBot*, arXiv:2502.00893.
- ToddlerBot `toddlerbot/descriptions/toddlerbot_2xm/robot.yml`.
- ToddlerBot `toddlerbot/locomotion/walk.gin`.
- WR `control/zmp/zmp_walk.py`.

## Pose Geometry

Foot separation is an outcome of pelvis/hip spacing, leg length, foot shape,
and joint pose. It is not the same as hip-to-hip width.

The raw WR home has nearly zero hip/ankle roll and approximately `179.4 mm`
foot-center separation. TB11 applies a symmetric correction:

```text
left hip roll    += 0.030 rad
right hip roll   -= 0.030 rad
left ankle roll  -= 0.030 rad
right ankle roll += 0.030 rad
```

The opposite ankle rotation keeps the soles approximately level while the long
legs move both feet inward. The proposed narrow arm uses `0.0635 rad` total
correction, not an additional `0.0635 rad`.

The deterministic MuJoCo forward-kinematics check gives:

| Quantity | TB11 baseline | Pose-only narrow |
|---|---:|---:|
| Total symmetric roll correction | 0.0300 rad / 1.72 deg | 0.0635 rad / 3.64 deg |
| Foot-center separation | 156.4 mm | 130.75 mm |
| Centerline-to-foot distance | 78.2 mm | 65.37 mm |
| Physical inner-foot clearance | 75.4 mm | 48.3 mm |
| Worst quasi-static support moment | 3.135 Nm | 2.627 Nm |
| Utilization of verifier's 4 Nm limit | 78.4% | 65.7% |

The narrow pose passes physical inner-clearance, self-contact, joint-limit,
sole-height, foot-orientation, and static-support gates.

## Correct Interpretation of the Close-Feet Threshold

The historical `0.146 m` `close_feet_threshold` is a training reward
threshold, not a physical collision boundary. It was scaled from ToddlerBot's
`0.060 m` threshold at its `0.074 m` nominal foot separation.

The narrow WR pose has `48.3 mm` of measured physical inner clearance and no
self-contact. It fails only the stale `0.146 m` reward threshold. For the
fixed-policy A/B, each arm preserves ToddlerBot's dimensionless `60/74`
threshold ratio:

```text
baseline reward threshold ~= 0.1564 * (0.060 / 0.074) = 0.1268 m
narrow reward threshold   ~= 0.13075 * (0.060 / 0.074) = 0.1060 m
```

This threshold changes reward bookkeeping only. Reward is not an actor input
and cannot change the deterministic fixed-policy trajectory. A future training
configuration must use the geometry-consistent value instead of the stale
`0.146 m` value.

## Controlled A/B

The runner first performs the static geometry gates, then evaluates both poses
at the slow and fast forward commands. It holds these variables fixed:

- TB11 checkpoint 5 actor
- robot dimensions, mass, and inertias
- HTD-45H simulation and scalar force limit
- `0.96 s` gait cycle
- `home` residual-action contract
- forward commands, seed, episode length, and environment count
- pushes disabled

Only the canonical symmetric roll pose and its matching reference half-width
change. The observation and action dimensions remain unchanged.

Run on the GPU machine:

```bash
RUN=training/checkpoints/ppo_walking_v0210_tb11_reference_geometry_com_lever/ppo_walking_v0210_tb11_reference_geometry_com_lever_v0210-tb11_20260914_082106-mch7tsfr

uv run python training/eval/sweep_stance_pose.py \
  --checkpoint "$RUN/checkpoint_5_102400.pkl" \
  --config "$RUN/training_config.yaml" \
  --num-envs 64 \
  --num-steps 1000 \
  --output-dir _eval/tb11_ckpt5_stance_pose_ab
```

The command produces the four raw evaluation JSON files and a paired
`summary.json` containing falls, orientation, forward/lateral motion,
per-actuator saturation, and bilateral hip/knee RMS torque for both poses.

## Decision Rule

The unchanged actor provides causal evidence, not a final deployment policy.

- **Strong positive:** narrow remains `0/64` at both commands, preserves the
  orientation gates and forward motion, reduces hip-roll RMS/saturation at
  both commands, and does not move comparable demand into knees or ankles.
  Prepare a short five-iteration adaptation around the narrow canonical pose.
- **Policy-mismatch result:** static gates pass, but the unchanged actor falls
  immediately while early torque demand improves. The pose remains
  mechanically plausible, but requires a bounded adaptation A/B before a
  conclusion.
- **Negative:** load does not decrease, or it moves to knees/ankles while
  stability or forward speed regresses. Reject stance narrowing as the main
  solution and prioritize measured mass reduction, gearing, or actuators.

Structural resizing is justified only after this pose-only test and the
measured upper-body mass counterfactual fail to provide sufficient headroom.
