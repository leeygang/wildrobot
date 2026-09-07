# ToddlerBot-aligned direct PPO (`v0.21.0-tb2`)

## Decision

`v0.21.0-tb2` is a cold-start correction of the `tb1` experiment. It does
not resume, initialize from, or distill any earlier WildRobot policy. The
existing deployment candidate and standing bundle remain unchanged.

The correction keeps one startup pose for physical home, walking frame zero,
and the residual-action base. A zero command is learned by the same locomotion
actor, and Reference State Initialization (RSI) remains disabled. The new
config is
[`ppo_walking_v0210_tb2_rsl_parity.yaml`](../configs/ppo_walking_v0210_tb2_rsl_parity.yaml).

## Source of truth

The implementation was checked against local ToddlerBot commit `f81679b`:

- `toddlerbot/locomotion/ppo_config.py`
- `toddlerbot/locomotion/rsl_rl_config.yml`
- `toddlerbot/locomotion/train_mjx.py`
- `toddlerbot/locomotion/mjx_config.py`
- `toddlerbot/locomotion/mjx_env.py`
- `toddlerbot/locomotion/walk.gin`

Public references:

- Shi et al., [ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation](https://arxiv.org/abs/2502.00893)
- Shi et al., [ToddlerBot 2.0](https://arxiv.org/abs/2601.03607)
- [RSL-RL v2.3.3](https://github.com/leggedrobotics/rsl_rl/tree/v2.3.3)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## Corrections after `tb1`

The completed `tb1` run did not execute the algorithm described by the
current ToddlerBot source closely enough to be a valid parity test. In the
recorded W&B samples, its hard KL early-stop averaged only 4.57 of 64
configured minibatch updates per iteration (about 0.29 of four epochs); the
last logged iteration ran only four updates. It also retained a
squashed state-dependent actor, mirror loss, a WR-specific observation frame,
and duplicated actor inputs in the critic.

`tb2` corrects those contracts:

| Contract | Current ToddlerBot | WildRobot `tb2` |
|---|---|---|
| Learning | Direct PPO from random initialization; time-limit transitions bootstrap with the rollout value | Same |
| PPO implementation | RSL-RL 2.3.3 | JAX implementation of the same update semantics |
| Rollout / epochs | 20 steps / 4 complete epochs | Same; no KL epoch skipping |
| KL schedule | Exact Gaussian KL, bidirectional LR adjustment, target 0.01 | Same thresholds, factor 1.5, and bounds `[1e-5, 1e-2]` |
| Value loss | PPO-clipped squared error, coefficient 0.25 | Same |
| Gradient clipping | One global norm over actor and critic, max 1.0 | Same |
| Actor distribution | Unsquashed Normal, learned global log standard deviation, initial std 0.5 | Same |
| Symmetry | Disabled unless `train_mjx.py --symmetry` is passed | Disabled |
| Actor observation | All motor positions/velocities, leg action, IMU, command, phase; 15 newest-first frames | Same ordering; 17 WR motors and 10 WR leg actions, 840 values |
| Critic observation | Clean actor frame plus reference error, base velocity, actuator force, actual/reference stance; 15 frames | Same ordering; 97 values/frame, 1,455 values total |
| Actor contact input | None | None |
| Zero command | Default motor pose while the time-based gait phase continues | Same in training, native evaluation, and hardware runtime |
| Reset | Reference frame 0 plus structured torso/arm perturbation | Home-backed frame 0 plus equivalent WR joint perturbation and height compensation |
| Action | `default_action + 0.25 * Normal action` | Same before final WR physical joint-limit clipping |

The policy controls ten leg joints because WR has no actuated hip-yaw joints.
It observes all 17 motors, including the fixed waist and arms. Runtime export
records both orders, and the hardware adapter reads all 17 motors while
writing only the ten policy targets; excluded joints remain at home.

## Deliberate hardware and morphology differences

These are retained because copying the numerical ToddlerBot values would not
represent the WR robot:

- WR linear commands and swing geometry use the established `4/3` linear
  scale: gait period `0.72 -> 0.96 s`, linear commands
  `0.10/0.05 -> 0.1333/0.0667 m/s`, and swing height
  `0.04 -> 0.05 m`. The close-feet threshold uses the measured stance-width
  ratio instead: `0.06 / 0.074 * 0.18056 = 0.146 m`.
- Yaw speed is time-normalized `1.0 -> 0.75 rad/s`; reward kernels are scaled
  so normalized command error has ToddlerBot's shape.
- WR retains measured asynchronous servo feedback and its current IMU noise
  model. Those are deployment hardware effects, not alternate task logic.
- WR retains final joint-limit clipping as a hardware safety boundary.
- WR's existing mass randomizer uses multiplicative body uncertainty instead
  of ToddlerBot's robot-specific torso/hand payload categories.

The corrected domain randomizer now covers damping, armature, friction loss,
gain, backlash, encoder noise, and structured reset pose. Unsupported
ToddlerBot servo-curve parameters remain represented by WR's measured
actuator model rather than copied by name.

## Training budget and gates

The first corrected run remains a 50,012,160-transition validation
(`1024 envs x 20 steps x 2442 iterations`). The lower environment count is a
GPU-memory constraint; it is not claimed to reproduce ToddlerBot's final
1-billion-transition budget. Scale only after the corrected learner shows a
healthy curve and passes deterministic evaluation.

Promotion remains stability-first:

- `0/64` falls on the primary 1,000-step deterministic forward screen;
- stable torso tilt mean <= 10 degrees, peak <= 15 degrees, and survivor-final
  <= 10 degrees;
- worst stable per-actuator torque-saturation fraction <= 5%;
- forward tracking passes the existing command-scaled gate.

Zero, backward, lateral, and yaw probes remain report-only for this first
corrected run. They diagnose command coverage without overriding the primary
forward stability decision.

## Launch

Run the smoke verification first on the GPU machine:

```bash
uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml \
  --verify
```

Then start a fresh run, with no `--init-policy` and no `--resume`:

```bash
uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml
```

For the cross-machine loop:

```bash
uv run python wildrobot/agents/autonomous_training_loop.py start \
  --new-run \
  --config training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml \
  --max-cycles 30

uv run python wildrobot/agents/autonomous_training_loop.py run
```

Do not deploy merely because training completes. Require the generated
`post_training_eval_summary.json`, inspect the selected checkpoint visually,
and then use a tethered trial. To exercise zero-command standing through this
single locomotion actor, pass `--disable-zero-cmd-hold-home`; otherwise runtime
intentionally holds the static home pose at zero command.
