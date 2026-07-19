# Native 17D Walking Policy Migration

## Contract

- MuJoCo remains 21-actuator. The four wrist joints stay mechanically modeled.
- Walking actor/runtime locomotion I/O uses 17 actuators and a 937-value
  `wr_obs_v8_cmd3d` observation.
- Excluded MuJoCo wrist controls are held at the model home pose by
  `WildRobotEnv._to_mj_ctrl`.
- The separate wrist controller owns the four physical wrist servos. A native
  17D policy neither writes them nor synthesizes their feedback.

This differs from the temporary legacy bridge, where a 21D actor still receives
fixed wrist feedback while runtime hardware I/O contains 17 servos.

## ToddlerBot Comparison

The unchanged parts follow ToddlerBot: a static home action base, RSI, domain
randomization, and 15-frame observation/critic histories. See local
`~/projects/toddlerbot/toddlerbot/locomotion/mjx_env.py` around lines 697-750,
1363-1371, and 2164-2171.

The asynchronous feedback model is an explicit WR-specific divergence.
ToddlerBot uses a 2 Mbps Dynamixel bus (`toddlerbot/sim/real_world.py:58`) and
its training observation code explicitly models encoders as commanded motor
positions (`mjx_env.py:2050-2090`). WR reads actual Hiwonder servos one at a
time at 115200 baud and exposes cached position plus finite-difference velocity.
Copying ToddlerBot's fresh-feedback assumption would therefore preserve a
measured sim-to-real mismatch.

Public basis:

- Rusu et al., *Policy Distillation*, arXiv:1511.06295.
- Tan et al., *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*,
  arXiv:1804.10332 (latency and dynamics randomization).
- Shi et al., *ToddlerBot: Open-Source ML-Compatible Humanoid Platform for
  Loco-Manipulation*, arXiv:2502.00893.

## 1. Distill Without Delay

The distillation script disables domain randomization, IMU noise, pushes, and
joint-feedback sample/hold for both teacher and student. It rolls the 21D
teacher with wrist actions forced to zero, projects every current/history
actuator block to 17D,
initializes the student by selecting the retained teacher input/output weights,
and then trains the 17D actor by deterministic action MSE. Shared hidden layers
are copied exactly; distillation corrects the loss of wrist input channels.

```bash
uv run python training/scripts/distill_walking_21d_to_17d.py \
  --output training/checkpoints/walking_v0210_smoke6_17d_distilled.pkl
```

Defaults collect 1000 steps at each of `(0.065,0,0)` and `(0.13,0,0)`, train
for 100 epochs, and run 1000-step native-17D validation at both commands.

## 2. Accept Structural Parity

Do not fine-tune until the generated JSON report meets all of these:

- Teacher and student have no termination before step 1000 at both commands.
- Validation action RMSE is at most `0.03` and max absolute error at most `0.15`.
- Student final forward displacement has the same sign as the teacher at both
  commands; investigate any displacement magnitude difference above 20%.
- Checkpoint dimensions report `(937, 17)`:

```bash
uv run python -c "from pathlib import Path; from training.exports.export_onnx import get_checkpoint_dims; print(get_checkpoint_dims(Path('training/checkpoints/walking_v0210_smoke6_17d_distilled.pkl')))"
```

These are migration gates, not claims that the distilled actor is ready for
hardware. Record the real full-run metrics before changing the thresholds.

## 3. Feedback Delay Model

`joint_feedback_sample_hold_enabled` reproduces the runtime cache rather than a
uniform whole-observation delay. Each joint receives an episode-randomized
refresh period and phase. Between refreshes the actor sees held position and
velocity; on refresh, velocity is finite-differenced over the actual cache age.

The initial config uses:

- legs: 4-7 control steps (80-140 ms), based on the measured
  `83.6/112.6/126.7 ms` leg max-cache age avg/p95/max;
- waist/shoulders/elbows: 12-24 steps (240-480 ms), provisional around the
  observed approximately 400 ms arm maximum.

The actor does not receive cache age, keeping the observation contract at 937.
Replace these bounds with per-joint age percentiles after the next robot log.

## 4. Domain-Randomized Fine-Tune

Start a fresh PPO state from the distilled actor. `--init-policy` loads only the
actor; it deliberately does not reuse the incompatible 21D value network or
optimizer state.

```bash
uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_smoke6_17d_latency_finetune.yaml \
  --init-policy training/checkpoints/walking_v0210_smoke6_17d_distilled.pkl
```

The fine-tune config keeps smoke6's physics/domain randomization, enables the
asynchronous cache model, and uses 500 iterations at `1e-5`. Evaluate saved
checkpoints for 1000 steps at both commands before bundle export or robot use.
