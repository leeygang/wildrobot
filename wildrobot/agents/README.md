# Training Loop Agent

`wildrobot/agents/training_loop_agent.py` automates a **train → eval → tune → resume** loop for
`training/train.py` runs, using W&B **offline** `metrics.jsonl` to decide the next knob change.

## What it does

- Runs training in **short cycles** (`--iters-per-cycle`) starting from `--resume`.
- Reads `training/wandb/offline-run-*/files/metrics.jsonl`.
- Picks the best checkpoint by lexicographically maximizing `(eval_push/success_rate, eval_push/episode_length)`.
- Tunes **resume-safe** knobs to reduce early terminations:
  - `env.collapse_height_buffer`, `env.collapse_vz_gate_band`
  - `reward_weights.collapse_height`, `reward_weights.collapse_vz`
  - `reward_weights.orientation`
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
