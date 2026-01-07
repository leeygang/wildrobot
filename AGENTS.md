# Repository Guidelines

## Active Plan (AI Quickstart)
- Follow the execution plan in `training/docs/learn_first_plan.md`; current focus is Stage 1 (PPO-only walking with task rewards, no AMP). Align experiments and changes with the stated stage gates and metrics.
- Use the commands in `CLAUDE.md` as the canonical entry points (validation, PPO training, visualization). Keep new scripts/configs consistent with that flow.

## Project Structure & Module Organization
- `training/`: Primary training stack (configs, envs, training loops, AMP utilities, visualization). Most changes land here.
- `assets/`: Robot definitions and generated configs; keep `assets/robot_config.yaml` in sync when updating MJCF/CAD assets.
- `scripts/`: Validation and one-off tooling (e.g., parity checks, PD tuning, feature verification).
- `tests/`: Fast regression/consistency tests; `tests/envs/` covers env parity, root-level tests cover AMP features and quaternion math.
- `mujoco-brax/` and `isaac/`: Alternate runtimes/notes; coordinate before modifying these experimental paths.

## Build, Test, and Development Commands
- Environment: `uv sync` to create/update `.venv` (Python 3.12+). Run tools with `uv run ...` to ensure the repo env is used.
- Validate training setup before long runs: `uv run python scripts/validate_training_setup.py`.
- Stage 1 PPO walking (current focus): `uv run python training/train.py --config training/configs/ppo_walking.yaml --no-amp`; add `--verify` for a quick smoke test.
- Visualize a checkpoint: `uv run python training/visualize_policy.py --checkpoint <path>`.
- Fast deterministic checks: `uv run python scripts/run_unit_tests.py`. Full suite: `uv run pytest tests`.

## Coding Style & Naming Conventions
- Python-first codebase; follow PEP 8 (4-space indent, snake_case for functions/vars, PascalCase for classes). Add type hints on new/changed functions.
- Keep modules importable as scripts (tests rely on `sys.path` manipulation); avoid hard-coded working directories.
- Prefer small, pure helpers in `training/utils` and reuse existing math helpers before adding new dependencies.
- When touching configs, document required assets (e.g., regenerated `robot_config.yaml`) and default values inline.

## Testing Guidelines
- Target tests to the area you touch: env logic in `tests/envs/`, AMP feature math in root `tests/`. If a test needs `assets/robot_config.yaml`, regenerate via `cd assets && python post_process.py` before running.
- For new features, add pytest cases mirroring existing patterns; favor deterministic inputs over random seeds.
- Capture expected output shapes/thresholds in assertions rather than printouts; keep prints only for helpful debugging context.

## Commit & Pull Request Guidelines
- Use concise, imperative commits (e.g., `Tighten AMP feature bounds`, `Add PPO smoke test`) and include the stage/version tag if relevant (history uses short release-style summaries).
- PRs should state scope, risk, and how to reproduce tests (commands + key outputs). Link issues or roadmap stages (`v0.10.x`) when applicable.
- Include screenshots or short logs for training/visualization changes; note any required asset regeneration or external data.

## Related Repositories (in `/home/leeygang/projects`)
- `IsaacLab/`: External simulator dependency; keep versions compatible with `assets/usd/` and Isaac scripts.
- `env_isaaclab/`: Environment setup helpers; useful when syncing IsaacLab configs.
- `GMR/`: Reference models/data; align protocol when importing datasets or motion references.
- `amass/`: Motion capture assets; ensure usage complies with data licensing and preprocessing steps.
- `ide_config/`, `gnome-terminal/`: Local tooling/config; adjust only if dev environment changes are required.
