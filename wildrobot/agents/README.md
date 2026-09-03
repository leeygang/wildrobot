# Training Loop Agent

## Full autonomous Mac ↔ GPU loop

The full loop uses two manually installed services:

- Ubuntu: `wildrobot-training-gpu.service` polls an SSH-visible queue and runs
  one exact-commit GPU job at a time.
- macOS: `com.wildrobot.autonomous-training` polls the active job, synchronizes
  results, runs the deterministic analyzer, invokes `codex exec` to implement
  and commit the next experiment, pushes it, and queues the next GPU run.

The loop is bounded by `--max-cycles` (default: 20) and
`--max-training-failures`. A completed run that misses deployment gates must
produce one bounded follow-up experiment;
the loop stops only on a deterministically promoted checkpoint, an unexpected
orchestration error, or either configured limit. It can also be manually paused
without cancelling an active GPU job. It
exports and validates a bundle after promotion. It never starts robot hardware.

### Deploy and start the Ubuntu service

Push this automation commit from the Mac, then update the normal GPU checkout:

```bash
# Mac
git push origin main

# Ubuntu
cd ~/projects/wildrobot
git pull --ff-only origin main
uv run python wildrobot/agents/remote_training_loop.py install-gpu-service
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now wildrobot-training-gpu.service
systemctl --user status wildrobot-training-gpu.service
```

The GPU checkout needs an existing `.venv`, Git access to `origin`, and all
source checkpoints referenced by queued jobs.

### Deploy and start the Mac service

Authenticate Codex CLI once, verify non-interactive execution, then install the
LaunchAgent:

```bash
cd ~/projects/wildrobot
codex exec --ephemeral "Reply with OK"
uv run python wildrobot/agents/autonomous_training_loop.py install-mac-service
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.wildrobot.autonomous-training.plist
```

SSH and Git authentication must work without an interactive password prompt
from the LaunchAgent environment.

### Start the first autonomous walking job

If training has already completed outside the queue, adopt that result instead
of launching it again. Before updating the GPU checkout, record the commit that
was used for training:

```bash
# Ubuntu, immediately after the manual training run finishes
cd ~/projects/wildrobot
git rev-parse HEAD
```

Update the GPU checkout and service as described above. Then start the Mac
controller from the completed result:

```bash
cd ~/projects/wildrobot
uv run python wildrobot/agents/autonomous_training_loop.py start \
  --config training/configs/ppo_walking_v0210_17d11_native_stance_stage1.yaml \
  --adopt-completed \
  --training-git-sha <40-character-training-commit> \
  --max-cycles 20

# Stay attached, poll continuously, and stream progress in this terminal.
uv run python wildrobot/agents/autonomous_training_loop.py run
```

`--adopt-completed` discovers the newest W&B run that has `metrics.jsonl` and a
matching checkpoint directory containing both `training_config.yaml` and
`post_training_eval_summary.json`. To avoid selecting by recency, pass the
exact name:

```bash
--adopt-completed offline-run-YYYYMMDD_HHMMSS-RUNID
```

The non-Codex controller creates the job manifest, synchronizes the result, and
runs the deterministic analyzer before invoking Codex. The adoption is rejected
if training-relevant files changed between the declared training commit and the
current automation checkout.

To launch a new GPU run through the queue instead, use:

```bash
uv run python wildrobot/agents/autonomous_training_loop.py start \
  --config training/configs/ppo_walking_v0210_17d11_native_stance_stage1.yaml \
  --init-policy training/checkpoints/ppo_walking_v0210_17d10_roll_ik_contract/ppo_walking_v0210_17d10_roll_ik_contract_v0210-17d10_20260901_222121-mil2cg1q/checkpoint_14_286720.pkl \
  --max-cycles 20
```

For a combined deployment bundle, also pass both
`--standing-checkpoint <checkpoint.pkl>` and
`--standing-config <training_config.yaml>`. Without them, the terminal artifact
is a validated walking-only bundle.

Inspect or stop the controller:

```bash
uv run python wildrobot/agents/autonomous_training_loop.py status
uv run python wildrobot/agents/autonomous_training_loop.py status --last 10
uv run python wildrobot/agents/autonomous_training_loop.py run
uv run python wildrobot/agents/autonomous_training_loop.py stop
tail -f training/remote_jobs/mac-service.log
ssh leeygang@linux-pc.local \
  'journalctl --user -u wildrobot-training-gpu.service -f'
```

`run` is the foreground continuous loop: it polls every 10 seconds, prints the
current cycle, job, and remote status, and streams synchronization, analyzer,
and Codex output while also writing the per-job logs. Pressing Ctrl-C only
detaches the foreground monitor; it does not stop the active loop. Temporary
SSH polling failures are printed and retried. `step` is a single non-blocking
poll used by the LaunchAgent and for debugging. `run` also automatically
reactivates `stopped_error` at its persisted stage and retries after the poll
interval. Configured terminal states (`ready` and cycle/failure limits) are not
overridden. GPU training output is likewise teed to both
`train.log` and the systemd journal shown above.

`stop` pauses only the Mac controller and preserves its durable pipeline stage;
it does not cancel an active GPU training job. If analysis or Codex is already
running, the request is honored at the next durable stage boundary. Wait until
`status` reports `Loop status: paused` before making manual repository changes.
Run the same `run` command later to reactivate and resume that exact stage.

`status` prints the persisted stage, whether that stage belongs to the GPU or
Mac, whether a Mac supervisor currently holds the loop lock, the live or cached
GPU job status, and summaries of the five most recent autonomous cycles. Use
`--last 10` (or another positive count) to change the history length, and
`--json` for machine-readable output.

The Mac state records durable stages: `adopt`, `training`, `analysis`, `fix`,
`push`, `enqueue`, and `export`. Restarting `run` resumes the recorded stage.
Analysis is rerun safely if interrupted. An interrupted Codex fix preserves its
existing diff or commit and starts a recovery invocation; a completed
structured decision is reused. Push is idempotent, and enqueue records its
exact job ID and inputs before contacting the GPU, so a lost SSH response
cannot duplicate a job.

The GPU service also recovers interrupted `dispatching` and `running` jobs. If
the post-training summary was already written, it accepts that completed
result. Otherwise it requeues the same job, reuses its exact-commit worktree,
and restarts from the originally declared `--init-policy` or `--resume`
checkpoint. Partial checkpoints are deliberately not trusted as a new resume
source.

After a process, stage, or machine restart, run the same foreground command
again. It resumes or reactivates the persisted stage automatically:

```bash
uv run python wildrobot/agents/autonomous_training_loop.py run
```

The Mac controller invokes Codex with `--approve-for-me`, which selects the
workspace-write sandbox in the installed CLI. Codex may analyze, edit, test,
and create one local commit;
the controller—not Codex—validates that commit, pushes it, and enqueues the next
job. Automation files and `training/CHANGELOG.md` are protected from autonomous
changes.

References: [Codex non-interactive mode](https://developers.openai.com/codex/non-interactive-mode),
[systemd services](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html),
[Git worktrees](https://git-scm.com/docs/git-worktree), and
[ToddlerBot](https://arxiv.org/abs/2502.00893).

## Manual single-run controller

`remote_training_loop.py` is the cross-machine controller. It uses Git for
code/config provenance and rsync only for generated artifacts. Every GPU run is
checked out at an exact commit in a dedicated worktree and described by an
atomic JSON manifest.

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
exact queued manifest and remote publication commands without connecting.

The GPU must have Git access to `origin`, an existing repository `.venv`,
the installed `wildrobot-training-gpu.service`, and the initial checkpoint
path. Enable user lingering if the service must survive all SSH sessions. The
persistent service dispatches one queued job at a time, and the worker also
holds a global GPU lock to prevent accidental competition for the device.

### Monitor, sync, and analyze

```bash
uv run python wildrobot/agents/remote_training_loop.py status
uv run python wildrobot/agents/remote_training_loop.py sync
uv run python wildrobot/agents/remote_training_loop.py analyze
```

An already-completed manual run can also be adopted into this supervised
single-run controller without enabling the autonomous loop:

```bash
uv run python wildrobot/agents/remote_training_loop.py adopt-completed \
  --config training/configs/ppo_walking_v0210_17d11_native_stance_stage1.yaml \
  --run-name offline-run-YYYYMMDD_HHMMSS-RUNID \
  --training-git-sha <40-character-training-commit>
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

### Optional Codex Desktop watcher

Use the durable prompt in `CODEX_SCHEDULED_TASK.md` for a project-scoped Codex
scheduled task. Run the task in the local project so it can access the ignored
job state and synchronized artifacts. The Mac must remain powered on with the
ChatGPT desktop app running.

References: [OpenAI scheduled tasks](https://developers.openai.com/codex/automations),
[systemd services](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html),
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
