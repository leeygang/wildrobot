# Fine-tuning After a CAD Morphology Update

Use actor-only transfer after changing link geometry, mass, inertia, joint
ranges, or the settled home pose. Do not use `--resume`: the old critic and
optimizer were fitted to the old dynamics, and the policy-spec fingerprint
changes when joint ranges or `home_ctrl_rad` change.

`--finetune-policy` is an alias for `--init-policy`. It loads actor parameters
only and initializes a fresh critic and fresh optimizers under the current
model. The actor observation/action dimensions and network shape must still be
compatible.

## Current Onshape export(9) measurements

The regenerated model remains `nq=24`, `nv=23`, `nu=17`, with one free root
and no wrist joint or actuator. Relative to the previous v2 model:

| Metric | Previous | Current | Change |
|---|---:|---:|---:|
| Total modeled mass | 4.127652 kg | 4.037166 kg | -2.19% |
| Bodies | 23 | 19 | separate wrist bodies removed |
| Left shoulder-roll range | [-175°, 25°] | [-170°, 30°] | +5° reference shift |
| Right shoulder-roll range | [-25°, 175°] | [-30°, 170°] | -5° reference shift |
| Stored home pelvis height | 0.460272 m | 0.459852 m | -0.42 mm |
| Stored home pitch | -1.09° | -2.74° | -1.65° |

The sagittal ZMP link lengths remain 0.193 m + 0.180 m. The ankle-pitch to
ground offset increased by 1 mm, so `ZMPWalkConfig.ankle_to_ground_m` is now
0.062 m. Because leg length and the 20 ms control period are unchanged, retain
the existing 0.96 s gait cycle and command-speed grid. This keeps the existing
WildRobot/ToddlerBot Froude scaling instead of copying ToddlerBot's 0.72 s
cycle directly. The new nominal mass remains within the existing [0.9, 1.1]
mass-randomization envelope, but the redistributed link inertias and new home
pose still require policy adaptation and fresh value learning.

## Standing

The deployed standing checkpoint is `(obs_dim, action_dim) = (59, 17)`, which
matches the current actor shape. First run the short wiring check:

```bash
uv run python training/train.py \
  --config training/configs/ppo_standing_stabilizer_v0227.yaml \
  --finetune-policy runtime/bundles/standing_v0227_ckpt200/checkpoint.pkl \
  --lr 1e-5 \
  --verify
```

Then run a conservative 100-iteration actor fine-tune (13,107,200 environment
steps) with a fresh critic and optimizer:

```bash
uv run python training/train.py \
  --config training/configs/ppo_standing_stabilizer_v0227.yaml \
  --finetune-policy runtime/bundles/standing_v0227_ckpt200/checkpoint.pkl \
  --lr 1e-5 \
  --iterations 100 \
  --checkpoint-dir training/checkpoints/standing_v0227_onshape9_finetune
```

Evaluate candidates against home hold before deployment:

```bash
uv run python training/eval/eval_standing_stabilization.py \
  --checkpoint <new-standing-checkpoint.pkl> \
  --config training/configs/ppo_standing_stabilizer_v0227.yaml \
  --platform gpu \
  --output /tmp/standing_onshape9_eval.json
```

## Walking

The deployed walking teacher is `(1129, 21)`, while the current walking actor
is `(937, 17)`. It cannot be loaded directly. First distill the archived 21D
actor into the current 17D model:

```bash
uv run python training/scripts/distill_walking_21d_to_17d.py \
  --output training/checkpoints/walking_v0210_smoke6_17d_onshape9_seed.pkl
```

Apply the structural gates in
[`17d_walk_policy_migration.md`](17d_walk_policy_migration.md), then verify and
fine-tune with the current morphology:

```bash
uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_smoke6_17d_latency_finetune.yaml \
  --finetune-policy training/checkpoints/walking_v0210_smoke6_17d_onshape9_seed.pkl \
  --verify

uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_smoke6_17d_latency_finetune.yaml \
  --finetune-policy training/checkpoints/walking_v0210_smoke6_17d_onshape9_seed.pkl
```

The walking config already uses the conservative `1e-5` learning rate and
10,240,000 environment steps. Evaluate at least 1000 deterministic steps at
both `(0.065, 0, 0)` and `(0.13, 0, 0)` before exporting a bundle.

## Runtime calibration

The current v2 shoulder-roll servo electrical centers map to MuJoCo -80° on
the left and +80° on the right. Re-run offset calibration after assembling the
new shoulder geometry. In the calibration menu, `z` aligns MuJoCo 0° while `r`
aligns raw servo midpoint 500 to the configured -80°/+80° reference.

## References and ToddlerBot comparison

- ToddlerBot separates compatible checkpoint resume from model loading without
  optimizer reuse in `toddlerbot/locomotion/on_policy_runner.py`; WildRobot's
  actor-only fine-tune follows the same separation while retaining its existing
  PPO implementation.
- ToddlerBot's 0.72 s walking cycle and 20 ms control period are defined in
  `toddlerbot/locomotion/walk.gin`. WildRobot keeps its already normalized
  0.96 s cycle because this CAD update did not change sagittal leg length.
- Rusu et al., [Policy Distillation](https://arxiv.org/abs/1511.06295), supports
  the 21D-to-17D walking actor migration.
- Shi et al., [ToddlerBot](https://arxiv.org/abs/2502.00893), is the reference
  humanoid training/runtime architecture.
- MuJoCo's [`joint/ref`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-ref)
  defines which joint coordinate corresponds to the model's reference pose;
  the physical servo-center calibration remains a separate runtime mapping.
