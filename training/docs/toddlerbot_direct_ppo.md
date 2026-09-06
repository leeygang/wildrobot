# ToddlerBot-parity direct PPO baseline (`v0.21.0-tb1`)

## Decision

`v0.21.0-tb1` starts a new walking lineage from random policy weights. It does
not resume, initialize from, or distill the `17d*` lineage. The existing
champion and its deployment bundle remain unchanged.

The purpose of this branch is to remove the startup-state ambiguity before
more local reward tuning: the physical home pose, reset pose, walking frame
zero, and residual-action base are one pose. A zero velocity command is the
standing task for the same locomotion actor. Reference State Initialization
(RSI) is disabled.

## Source comparison

The reference implementation is local ToddlerBot commit `f81679b`:

- `~/projects/toddlerbot/toddlerbot/locomotion/walk.gin`
- `~/projects/toddlerbot/toddlerbot/locomotion/mjx_config.py`
- `~/projects/toddlerbot/toddlerbot/locomotion/mjx_env.py`
- `~/projects/toddlerbot/toddlerbot/locomotion/rsl_rl_config.yml`
- Shi et al., [ToddlerBot](https://arxiv.org/abs/2502.00893)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

The corresponding WildRobot config is
[`ppo_walking_v0210_tb1_direct_ppo.yaml`](../configs/ppo_walking_v0210_tb1_direct_ppo.yaml).

| Contract | ToddlerBot | WildRobot `tb1` |
|---|---|---|
| Learning | Direct PPO from scratch | Direct PPO from scratch |
| Startup | Reference frame 0 is default/home | Reference frame 0 is MJCF home |
| Reset frames | Frame 0 | Frame 0; RSI disabled |
| Standing | Zero command on locomotion actor | Zero command on locomotion actor |
| Actor contact input | None | None (`wr_obs_v11_cmd3d_proprio`) |
| Contact in simulation | Rewards and privileged critic | Rewards and privileged critic |
| Controlled joints | 12 leg joints | 10 leg joints; WR has no hip yaw |
| Upper body | Not in walking action mask | Seven joints fixed at MJCF home |
| History | 15 frames | 15-frame proprioceptive history |
| Delay | One control step | One control step |
| PPO | 20-step rollout, 4 epochs, LR 3e-5, KL 0.01 | Same |
| Action scale | 0.25 rad | 0.25 rad |
| Gait period | 0.72 s | 0.96 s, Froude-scaled for WR leg length |
| Linear commands | ±0.10 / ±0.05 m/s | ±0.1333 / ±0.0667 m/s (4/3 scale) |
| Yaw commands | ±1.0 rad/s | ±0.75 rad/s (0.72/0.96 time scale) |

The velocity tracking kernels preserve ToddlerBot's normalized error shape.
For example, ToddlerBot computes yaw tracking as
`exp(-4.0 * yaw_error²)`. Scaling the command range from 1.0 to 0.75 rad/s
therefore gives WR `alpha = 4.0 / 0.75² = 7.111111`; this is not the legacy
WR default of 0.25.

Three deliberate differences remain. WR trains its measured slow servo
feedback sample-and-hold behavior, and its available domain-randomization
model does not yet expose every actuator parameter randomized by ToddlerBot.
WR also retains the existing deployable `wr_obs_v11_cmd3d_proprio` schema: its
policy observes the ten controlled leg joints and uses the current sample plus
15 lagged proprioceptive bundles, whereas ToddlerBot observes all motors in a
15-frame stack. This avoids another hardware policy-contract migration; the
fixed WR waist and arm targets are nevertheless present in simulation physics
and the privileged critic. None of these differences introduces hardware foot
contact into the actor.

## Training budget and gates

The first run is a 50,012,160-transition validation (`1024 envs × 20 steps ×
2442 iterations`). This is long enough to evaluate the new architecture but is
not a claim that WR has reproduced ToddlerBot's final 1B-transition training
budget. Scale this lineage toward 1B only after it shows the expected learning
curve and passes the deterministic safety screen.

Promotion remains stability-first:

- `0/64` falls on the primary 1000-step deterministic screen;
- stable torso tilt mean ≤ 10°, peak ≤ 15°, survivor-final ≤ 10°;
- worst stable per-actuator torque-saturation fraction ≤ 5%;
- forward tracking must pass the existing command-scaled gate.

The configured zero, backward, lateral, and yaw probes are report-only for the
first run. They expose whether the broad ToddlerBot command distribution is
learning without allowing lateral/yaw performance to override the primary
stability decision.

## Launch

Commit and push this branch, update the GPU checkout/service, then start a new
campaign on the Mac without `--init-policy` or `--resume`:

```bash
uv run python wildrobot/agents/autonomous_training_loop.py start \
  --new-run \
  --config training/configs/ppo_walking_v0210_tb1_direct_ppo.yaml \
  --max-cycles 30

uv run python wildrobot/agents/autonomous_training_loop.py run
```

The absence of an initialization option is intentional: this is pure PPO from
random weights. The GPU worker runs the exact committed config. If a failed
attempt produces no usable checkpoint, the loop may restart a corrected direct
PPO config with `start_mode=none`; if it produces a valid checkpoint, normal
continuations should use `--resume` only when the complete contract is unchanged.
The loop freezes the canonical home/frame-zero, no-RSI, leg-only, zero-source-KL
contract and will not enqueue the teacher-recoverability/distillation branch for
this campaign.

Before submitting the full run, the GPU can perform the built-in smoke test:

```bash
uv run python training/train.py \
  --config training/configs/ppo_walking_v0210_tb1_direct_ppo.yaml \
  --verify
```

Do not deploy from this branch merely because training completes. Use the
generated `post_training_eval_summary.json`, then visually inspect the selected
checkpoint before any tethered hardware trial. For zero-command standing with
this single locomotion policy, runtime testing must include
`--disable-zero-cmd-hold-home`; otherwise the runtime intentionally bypasses
the actor and holds the static home pose.
