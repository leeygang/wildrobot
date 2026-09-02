# Training Loop Agent

## Mac ↔ GPU supervised loop

`remote_training_loop.py` is the cross-machine controller. It uses Git for
code/config provenance and rsync only for generated artifacts. Every GPU run is
checked out at an exact commit in a dedicated worktree, launched through a
transient user `systemd` service, and described by an atomic JSON manifest.

The controller deliberately does not push a generated change, launch the next
changed experiment, update `training/CHANGELOG.md`, or deploy hardware without
review.

### Submit the current walking retry

Commit and push the Mac worktree first, then run:

```bash
uv run python wildrobot/agents/remote_training_loop.py submit \
  --config training/configs/ppo_walking_v0210_17d11_native_stance_stage1.yaml \
  --init-policy training/checkpoints/ppo_walking_v0210_17d10_roll_ik_contract/ppo_walking_v0210_17d10_roll_ik_contract_v0210-17d10_20260901_222121-mil2cg1q/checkpoint_14_286720.pkl
```

The default GPU target matches the existing transfer scripts:
`leeygang@linux-pc.local:/home/leeygang/projects/wildrobot`. Override it with
`--host`, `--user`, `--port`, or `--remote-repo`. Use `--dry-run` to print the
exact remote checkout and launch commands without connecting.

The GPU must have Git access to `origin`, an existing repository `.venv`,
`systemd-run --user`, and the initial checkpoint path. Enable user lingering if
the service must survive all SSH sessions. Only one training worker runs at a
time; a second job fails its global GPU lock instead of competing for the
device.

### Monitor, sync, and analyze

```bash
uv run python wildrobot/agents/remote_training_loop.py status
uv run python wildrobot/agents/remote_training_loop.py sync
uv run python wildrobot/agents/remote_training_loop.py analyze
```

Summary sync retrieves the manifest, log, frozen configs, W&B metrics, and
deterministic evaluation JSON. It does not retrieve `.pkl` files. A promoted
checkpoint can be copied only after the authoritative selector passes:

```bash
uv run python wildrobot/agents/remote_training_loop.py sync --selected-checkpoint
```

Local control state and analysis reports live in ignored
`training/remote_jobs/`. W&B and checkpoint summaries are restored to their
normal `training/wandb/` and `training/checkpoints/` locations so the existing
`wildrobot-training-analyze` workflow remains the source of truth.

### Codex scheduled watcher

Use the durable prompt in `CODEX_SCHEDULED_TASK.md` for a project-scoped Codex
scheduled task. Run the task in the local project so it can access the ignored
job state and synchronized artifacts. The Mac must remain powered on with the
ChatGPT desktop app running.

References: [OpenAI scheduled tasks](https://developers.openai.com/codex/automations),
[systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html),
[Git worktrees](https://git-scm.com/docs/git-worktree), and
[ToddlerBot](https://arxiv.org/abs/2502.00893). ToddlerBot similarly snapshots
arguments, training/environment configuration, code state, and checkpoints per
run; WildRobot adds an explicit cross-machine manifest and deterministic
promotion state.

## Legacy single-machine tuning loop

`wildrobot/agents/training_loop_agent.py` automates a **train → eval → tune → resume** loop for
`training/train.py` runs, using W&B **offline** `metrics.jsonl` to decide the next knob change.

## What it does

- Runs training in **short cycles** (`--iters-per-cycle`) starting from `--resume`.
- Reads `training/wandb/offline-run-*/files/metrics.jsonl`.
- Picks the best checkpoint by lexicographically maximizing `(eval_push/success_rate, eval_push/episode_length)`.
- Tunes **resume-safe** knobs to reduce early terminations and improve recovery:
  - `env.collapse_height_buffer`, `env.collapse_vz_gate_band`
  - `env.push_force_max`, `env.push_duration_steps`
  - `reward_weights.collapse_height`, `reward_weights.collapse_vz`
  - `reward_weights.orientation`
  - `reward_weights.clearance`, `reward_weights.flight_phase_penalty`
  - `reward_weights.posture` (if present in your training code/config)
  - `reward_weights.action_rate`, `reward_weights.torque`
- Refuses to change knobs known to break resume safety (e.g. `env.action_filter_alpha`).

## Usage

Example (standing_push):

```bash
UV_CACHE_DIR=/tmp/uv-cache JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache \
uv run python wildrobot/agents/training_loop_agent.py \
  --base-config training/configs/ppo_standing_push.yaml \
  --resume training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3/checkpoint_80_10485760.pkl \
  --iters-per-cycle 20 \
  --max-cycles 10 \
  --probe-eval-steps 200 \
  --confirm-eval-steps 500
```

Notes:
- `eval_*` JIT compilation can be expensive; the agent uses a shorter `--probe-eval-steps` during tuning.
- Final success is only declared on a confirm run when `eval_push/success_rate == 1.0` and `eval_push/episode_length == 500`.
- The agent prints live per-iteration summaries by tailing `metrics.jsonl`; disable with `--no-live-metrics`.

## Push curriculum (hard → very hard)

To empirically ramp pushes, use `--push-stages` (each stage sets `env.push_force_max` and `env.push_duration_steps`).
The agent advances to the next stage only after a confirm run meets the target.

Example:

```bash
uv run python wildrobot/agents/training_loop_agent.py ... \
  --push-stages "9:15,12:15"
```

## Using an LLM advisor (optional)

By default the agent uses `--advisor heuristic` (no external calls).

If you want the config-tuning decision to be suggested by ChatGPT (or an OpenAI-compatible endpoint), use:

```bash
export OPENAI_API_KEY=...  # required
uv run python wildrobot/agents/training_loop_agent.py ... --advisor openai
```

Safety:
- The agent only applies numeric/bool updates under allowed prefixes (e.g. `reward_weights.*`, `env.push_*`).
- It will refuse forbidden keys like `env.action_filter_alpha`.
