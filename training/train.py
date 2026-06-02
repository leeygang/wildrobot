#!/usr/bin/env python3
"""Entry point for WildRobot training.

This script provides a command-line interface for training walking policies
using PPO.

Usage Examples:

    # v0.20.1 PPO smoke (vx=0.15 only):
    python training/train.py --config training/configs/ppo_walking_v0201_smoke.yaml

    # Standing-branch training (separate from the locomotion path):
    python training/train.py --config training/configs/ppo_standing.yaml

    # CLI overrides:
    python training/train.py --config <yaml> --num-envs 512 --iterations 5000

    # Quick verify (overrides via the config's quick_verify section):
    python training/train.py --config <yaml> --verify

See also:
    - training/docs/walking_training.md: locomotion roadmap (v0.20.x)
    - training/docs/v0201_env_wiring.md: v0.20.1 env design + status
    - training/configs/ppo_standing*.yaml: standing-branch configs

History:
    Pre-v0.20.1 ``ppo_walking*.yaml`` configs were deleted along with the
    v1/v2 walking_ref runtime stack.  Old training-config snapshots remain
    in ``training/checkpoints/*/training_config.yaml`` for reference but
    are not runnable against the current env.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.runtime_env import configure_training_runtime_env

configure_training_runtime_env()

from ml_collections import config_dict

# Default config paths.
# v0.20.1: ppo_walking.yaml was deleted along with the v1/v2 reference
# stack.  Set to the v0.20.1 PPO smoke YAML when Task 3 lands; until
# then ``--config`` is required (no default).
DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).parent / "configs" / "ppo_walking_v0201_smoke.yaml"
DEFAULT_ROBOT_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "v2" / "mujoco_robot_config.json"

# Import config loaders from configs module
from training.configs.training_config import (
    load_robot_config,
    load_training_config,
    RobotConfig,
    TrainingConfig,
    WandbConfig,
)
from training.configs.realism import load_training_realism_profile

# Import checkpoint management
from training.core.checkpoint import (
    list_checkpoints,
    load_checkpoint,
    print_top_checkpoints_summary,
    save_window_best_checkpoint,
)
from training.core.post_training_eval import (
    CheckpointMetricCandidate,
    deterministic_eval_gate,
    rank_checkpoint_candidates,
)

# Import W&B tracker
from training.core.experiment_tracking import (
    REWARD_TERM_KEYS,
    build_wandb_metrics,
    create_training_metrics,
    define_wandb_topline_metrics,
    generate_eval_video,
    generate_job_name,
    save_and_upload_video,
    WandbTracker,
)
from training.policy_spec_utils import build_policy_spec_from_training_config


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    if parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _load_metrics_jsonl_by_iteration(metrics_log_path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """Load metrics.jsonl rows keyed by progress/iteration when available."""
    if metrics_log_path is None or not metrics_log_path.exists():
        return {}
    rows_by_iteration: Dict[int, Dict[str, Any]] = {}
    for raw_line in metrics_log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        iteration = row.get("progress/iteration")
        try:
            if iteration is None:
                continue
            rows_by_iteration[int(iteration)] = row
        except (TypeError, ValueError):
            continue
    return rows_by_iteration


def parse_args():
    """Parse command-line arguments.

    CLI arguments override config file values when explicitly provided.
    """
    parser = argparse.ArgumentParser(
        description="Train WildRobot with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (loaded first, then CLI overrides)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (default: configs/wildrobot_phase3_training.yaml)",
    )

    # Training configuration (CLI overrides config file)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (default from config)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default from config)",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default from config)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default from config)",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=None,
        help="PPO clipping parameter (default from config)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Entropy bonus coefficient (default from config)",
    )

    # Logging and checkpoints
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Log metrics every N iterations (default from config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints (default from config)",
    )

    # Quick test mode
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run quick verification (10 iterations, 4 envs)",
    )

    # Resume training from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., checkpoints/best_checkpoint.pkl)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )

    return parser.parse_args()


# v0.20.1-smoke7: eval-cmd plumbing helpers, extracted from start_training
# so that test_smoke7_eval_cmd_behavior.py can exercise the train.py-level
# wiring decision (not just the env-level reset_for_eval behavior).  The
# decision rule is "if eval_velocity_cmd >= 0, eval rollouts pin cmd
# (reset_for_eval pins; step disables mid-episode resample); else, eval
# behaves like training (sampled cmd at reset, resample fires per the
# configured period)."  See training/docs/walking_training.md
# v0.20.1-smoke7 §.

def is_eval_cmd_pinned(env_cfg) -> bool:
    """Return True iff eval rollouts should pin the command (override active).

    Sentinel: ``eval_velocity_cmd < 0`` means "no override; eval behaves
    like training".  Any non-negative value means "pin eval cmd to this
    value" — used by smoke7+ to keep G4 metrics interpretable when
    training is multi-cmd.
    """
    # v0.21.0 P3 / H3: ``eval_velocity_cmd`` is (vx, vy, wz); sentinel
    # detection reads only the [0]th axis (vx).
    _ecv = env_cfg.eval_velocity_cmd
    _vx = _ecv[0] if hasattr(_ecv, "__len__") else _ecv
    return float(_vx) >= 0.0


def eval_step_kwargs(env_cfg) -> dict:
    """Return the kwargs the factory's eval step closure passes to env.step.

    Pinned mode (``eval_velocity_cmd >= 0``): returns
    ``{"disable_cmd_resample": True}`` so the env's mid-episode
    resample is suppressed and the eval cmd stays fixed across the
    resample boundary.

    Sentinel mode (``eval_velocity_cmd < 0``): returns ``{}`` so eval
    behaves exactly like training — including mid-episode resample if
    ``cmd_resample_steps > 0``.

    Extracted as a separate helper so unit tests can verify the
    pinned-vs-sentinel wiring decision without constructing a full
    env (which costs minutes of JIT compilation).  Catches the
    smoke7-prep1 review's third feedback item: a future regression
    in ``make_eval_env_fns`` could re-break sentinel semantics while
    env-level tests still pass.
    """
    if is_eval_cmd_pinned(env_cfg):
        return {"disable_cmd_resample": True}
    return {}


def _query_visible_gpus() -> list[tuple[str, str]]:
    """Return visible GPU rows as ``(name, display_active)`` tuples.

    Uses ``nvidia-smi`` so the guard can run before any JAX CUDA context
    is created.  Fail-open: if the query is unavailable or malformed,
    return an empty list and leave the config unchanged.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,display_active",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return []

    rows: list[tuple[str, str]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def maybe_apply_desktop_gpu_num_env_guard(training_cfg, args) -> bool:
    """Cap ``ppo.num_envs`` on watchdog-prone single-GeForce Linux hosts.

    NVIDIA documents that single-GPU CUDA+X systems should either keep
    kernels short, disable X, or use a second GPU.  Our local smoke runs
    are often executed on a single GeForce desktop GPU, so a large
    ``num_envs`` can produce long fused JAX kernels and hit
    ``CUDA_ERROR_LAUNCH_TIMEOUT``.  To keep fresh local runs from dying
    late in training, clamp to a conservative local cap unless the user
    explicitly overrides ``--num-envs`` or disables the guard via
    ``WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD=1``.
    """
    if platform.system() != "Linux":
        return False
    if os.environ.get("WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD") == "1":
        return False
    if args.num_envs is not None:
        return False

    env_cap = int(os.environ.get("WR_DESKTOP_GPU_NUM_ENVS_CAP", "1024"))
    if training_cfg.ppo.num_envs <= env_cap:
        return False

    gpu_rows = _query_visible_gpus()
    if len(gpu_rows) != 1:
        return False

    gpu_name, display_active = gpu_rows[0]
    if "geforce" not in gpu_name.lower():
        return False

    original_num_envs = int(training_cfg.ppo.num_envs)
    training_cfg.ppo.num_envs = env_cap
    if training_cfg.ppo.eval.num_envs > env_cap:
        training_cfg.ppo.eval.num_envs = env_cap
    if getattr(training_cfg.ppo.eval, "post_training_num_envs", 0) > env_cap:
        training_cfg.ppo.eval.post_training_num_envs = env_cap

    print("=" * 60)
    print("Desktop GPU num_envs guard applied")
    print(
        f"  GPU: {gpu_name} (display_active={display_active})"
    )
    print(
        f"  num_envs: {original_num_envs} -> {env_cap}"
    )
    print(
        "  Reason: single GeForce Linux host; clamping local JAX batch width "
        "to reduce CUDA launch-timeout risk."
    )
    print(
        "  Override: pass --num-envs explicitly or set "
        "WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD=1"
    )
    print("=" * 60)
    return True


def make_eval_env_fns(env, training_cfg, eval_num_envs: int):
    """Build (step_fn, clean_step_fn, reset_fn) for eval rollouts.

    Honors the ``eval_velocity_cmd`` sentinel.  In pinned mode
    (``eval_velocity_cmd >= 0``):
      - ``reset_fn`` pins cmd via ``env.reset_for_eval``
      - ``step_fn`` / ``clean_step_fn`` pass ``disable_cmd_resample=True``
        so the pin survives mid-episode resample boundaries

    In sentinel mode (``eval_velocity_cmd < 0``):
      - ``reset_fn`` falls back to sampled cmd (``reset_for_eval`` no-op)
      - ``step_fn`` / ``clean_step_fn`` omit the disable flag so eval
        rollouts behave like training (mid-episode resample fires per
        ``cmd_resample_steps``)

    Returns:
        (step_fn, clean_step_fn, reset_fn) — three batched (vmap'd)
        callables matching the signatures expected by the PPO loop.
    """
    # jax is lazy-imported here to match start_training's deferred-import
    # pattern (keeps train.py module-import cheap when only the parser
    # or the factory's pure-Python helpers are needed).
    import jax  # noqa: WPS433

    # Single source of truth for the pinned-vs-sentinel wiring decision.
    # ``eval_step_kwargs`` is unit-tested independently of env construction.
    step_kwargs = eval_step_kwargs(training_cfg.env)
    clean_step_kwargs = {"disable_pushes": True, **step_kwargs}

    def step_fn(state, action):
        """Batched eval step (kwargs determined by eval_step_kwargs)."""
        return jax.vmap(
            lambda s, a: env.step(s, a, **step_kwargs)
        )(state, action)

    def clean_step_fn(state, action):
        """Batched eval step with pushes disabled."""
        return jax.vmap(
            lambda s, a: env.step(s, a, **clean_step_kwargs)
        )(state, action)

    def reset_fn(rng):
        """Batched eval reset.

        Always uses ``reset_for_eval`` — it's a no-op when
        ``eval_velocity_cmd < 0`` (delegates to ``reset``) so the
        sentinel "eval = training" semantics are preserved.
        """
        rngs = jax.random.split(rng, eval_num_envs)
        return jax.vmap(env.reset_for_eval)(rngs)

    return step_fn, clean_step_fn, reset_fn


def start_training(
    training_cfg: "TrainingConfig",
    wandb_tracker: Optional[WandbTracker] = None,
    checkpoint_dir: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
    config_name: Optional[str] = None,
):
    """PPO training entry point.

    Args:
        training_cfg: Training configuration (single source of truth)
        wandb_tracker: Optional W&B tracker for logging
        checkpoint_dir: Directory for saving checkpoints (overrides config if provided)
        resume_checkpoint_path: Path to checkpoint to resume training from
        config_name: Name of the config file (without extension), e.g., "ppo_walking"
    """
    import pickle

    import jax
    import jax.numpy as jnp
    import numpy as np

    from training.configs.training_config import get_robot_config
    from policy_contract.calib import JaxCalibOps

    def _to_yamlable(value: Any) -> Any:
        """Recursively convert config dataclasses into YAML-safe primitives."""
        if is_dataclass(value):
            return {
                f.name: _to_yamlable(getattr(value, f.name))
                for f in fields(value)
                if f.name != "_frozen"
            }
        if isinstance(value, tuple):
            return [_to_yamlable(v) for v in value]
        if isinstance(value, list):
            return [_to_yamlable(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_yamlable(v) for k, v in value.items()}
        return value

    # Check JAX backend
    print(f"\n{'=' * 60}")
    print("JAX Configuration")
    print(f"{'=' * 60}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("  ⚠️  WARNING: JAX is NOT using GPU!")
        print("  Install JAX with CUDA: pip install jax[cuda12]")
    else:
        print("  ✓ GPU detected")
    print(f"{'=' * 60}\n")

    from training.envs.wildrobot_env import WildRobotEnv
    from training.core.training_loop import (
        IterationMetrics,
        train,
        TrainingState,
    )

    # Create environment config and environment
    print("Creating WildRobotEnv...")
    env = WildRobotEnv(config=training_cfg)
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )
    realism_profile = load_training_realism_profile(training_cfg)
    if realism_profile is not None:
        print(
            "✓ Realism profile loaded "
            f"({realism_profile.profile_name}, schema={realism_profile.schema_version})"
        )

    # Determine checkpoint directory
    final_checkpoint_dir = checkpoint_dir or training_cfg.checkpoints.dir

    # Create training job name (for checkpoint subdirectory)
    # Format: {config_name}_v{version}_{timestamp}-{wandb_run_id}
    # Example: ppo_walking_v01005_20251228_205534-uf665cr6
    job_name = generate_job_name(
        version=training_cfg.version,
        config_name=config_name,
        wandb_tracker=wandb_tracker,
        mode_suffix="ppo",
    )
    job_checkpoint_dir = os.path.join(final_checkpoint_dir, job_name)
    os.makedirs(job_checkpoint_dir, exist_ok=True)

    print(f"Training job: {job_name}")
    print(f"Checkpoints will be saved to: {job_checkpoint_dir}")

    # Persist the effective training config next to checkpoints so later tools
    # (for example bundle export) can auto-detect the exact run configuration.
    training_config_snapshot_path = os.path.join(job_checkpoint_dir, "training_config.yaml")
    with open(training_config_snapshot_path, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(
            _to_yamlable(training_cfg),
            f,
            sort_keys=False,
            allow_unicode=False,
        )

    print(
        f"✓ Environment functions created (vmapped for {training_cfg.ppo.num_envs} envs)"
    )

    # Policy contract fingerprint (stored in checkpoints to prevent silent resume drift).
    robot_cfg = get_robot_config()
    policy_spec = build_policy_spec_from_training_config(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )
    policy_spec_dict = policy_spec.to_json_dict()

    # Create vmapped environment functions
    def batched_step_fn(state, action):
        """Batched environment step."""
        return jax.vmap(lambda s, a: env.step(s, a))(state, action)

    def batched_reset_fn(rng):
        """Batched environment reset."""
        rngs = jax.random.split(rng, training_cfg.ppo.num_envs)
        return jax.vmap(env.reset)(rngs)

    eval_num_envs = training_cfg.ppo.eval.num_envs or training_cfg.ppo.num_envs

    # v0.20.1-smoke7: eval-cmd override + resample suppression.
    # The decision rule (pinned vs sentinel) and the resulting step /
    # reset closures are built by ``make_eval_env_fns`` (top of this
    # module) so test_smoke7_eval_cmd_behavior.py can exercise the
    # exact same factory used here without duplicating the wiring.
    (
        batched_eval_step_fn,
        batched_eval_clean_step_fn,
        batched_eval_reset_fn,
    ) = make_eval_env_fns(env, training_cfg, eval_num_envs)

    # Get checkpoint settings from config
    checkpoint_interval = training_cfg.checkpoints.interval
    post_training_eval_enabled = bool(training_cfg.ppo.eval.post_training_enabled)
    post_training_top_k = int(training_cfg.ppo.eval.post_training_top_k or 3)
    post_training_num_envs = int(training_cfg.ppo.eval.post_training_num_envs or 8)
    post_training_num_steps = int(training_cfg.ppo.eval.post_training_num_steps or 500)
    post_training_strict_lateral_drift = bool(
        getattr(training_cfg.ppo.eval, "post_training_strict_lateral_drift", False)
    )
    post_training_checkpoint_label = (
        str(training_cfg.ppo.eval.post_training_checkpoint_label).strip()
        or "eval_promoted"
    )

    # Checkpoint management state (captured by callback closure)
    best_reward = {"value": float("-inf")}
    window_best = {
        "reward": float("-inf"),
        "state": None,
        "metrics": None,
        "iteration": 0,
        "total_steps": 0,
    }
    metrics_log_path = None
    if training_cfg.wandb.enabled:
        run_dir = None
        if wandb_tracker is not None and wandb_tracker._run_dir is not None:
            run_dir = Path(wandb_tracker._run_dir)
        if run_dir is None:
            run_id = wandb_tracker.get_run_id() if wandb_tracker is not None else None
            if run_id is None:
                run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path(training_cfg.wandb.log_dir) / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_log_path = run_dir / "metrics.jsonl"

    # Training callback for W&B logging and checkpoint management
    def callback(
        iteration: int,
        state: TrainingState,
        metrics: IterationMetrics,
        steps_per_sec: float,
    ):
        # W&B logging
        if wandb_tracker is not None:
            wandb_metrics, missing_terms = build_wandb_metrics(
                iteration=iteration,
                metrics=metrics,
                steps_per_sec=steps_per_sec,
                reward_terms=REWARD_TERM_KEYS,
            )
            if iteration == 1 and missing_terms:
                print(f"⚠️ Missing reward terms in env_metrics: {missing_terms}")

            wandb_tracker.log(wandb_metrics, step=int(state.total_steps))
            if metrics_log_path is not None:
                with metrics_log_path.open("a", encoding="utf-8") as f:
                    json.dump(wandb_metrics, f, sort_keys=True)
                    f.write("\n")

        # Skip checkpoint tracking if disabled
        if checkpoint_interval <= 0:
            return

        current_reward = float(metrics.episode_reward)

        # Track best within current checkpoint window
        if current_reward > window_best["reward"]:
            window_best["reward"] = current_reward
            window_best["state"] = jax.device_get(state)
            window_best["metrics"] = metrics
            window_best["iteration"] = iteration
            window_best["total_steps"] = int(state.total_steps)

        # At checkpoint boundary, save the best from this window
        if iteration % checkpoint_interval == 0:
            save_window_best_checkpoint(
                window_best=window_best,
                best_reward=best_reward,
                config=training_cfg,
                checkpoint_dir=job_checkpoint_dir,
                policy_spec=policy_spec_dict,
            )

    # Load checkpoint for resuming if provided
    resume_checkpoint = None
    if resume_checkpoint_path is not None:
        print(f"\nLoading checkpoint for resume: {resume_checkpoint_path}")
        resume_checkpoint = load_checkpoint(resume_checkpoint_path)

    print("\n" + "=" * 60)
    print("Starting PPO training...")
    print("=" * 60 + "\n")

    final_state = train(
        env_step_fn=batched_step_fn,
        env_reset_fn=batched_reset_fn,
        config=training_cfg,
        callback=callback,
        resume_checkpoint=resume_checkpoint,
        eval_env_step_fn=batched_eval_step_fn,
        eval_env_step_fn_no_push=batched_eval_clean_step_fn,
        eval_env_reset_fn=batched_eval_reset_fn,
        # v0.20.1 v3-only env treats `action` as a bounded residual on top
        # of the offline reference (target_q = q_ref + clip(action) * scale).
        # iter-0 must therefore output zero so the rollout starts from
        # bare-q_ref replay — this is the zero-residual invariant
        # in walking_training.md (v0.20.1 §
        # "Smoke policy initialisation") and what the pre-smoke wiring
        # test (tests/test_v0201_env_zero_action.py) verifies.
        # The legacy bias toward default_joint_qpos was only correct for
        # the v0.19.5c standing path where action = absolute pose; under
        # the residual contract it injects a non-zero residual at iter 0.
        policy_init_action=jnp.zeros(env.action_size, dtype=jnp.float32)
        if resume_checkpoint is None
        else None,
    )

    # Save any remaining best checkpoint from the final window
    # This fixes the bug where if training ends at iteration 740 with checkpoint_interval=100,
    # the best checkpoint between 700-740 would not be saved
    if checkpoint_interval > 0:
        save_window_best_checkpoint(
            window_best=window_best,
            best_reward=best_reward,
            config=training_cfg,
            checkpoint_dir=job_checkpoint_dir,
            policy_spec=policy_spec_dict,
        )

    if post_training_eval_enabled:
        print()
        print("=" * 60)
        print("Post-training top-K deterministic eval")
        print("=" * 60)
        checkpoint_rows = list_checkpoints(job_checkpoint_dir)
        if not checkpoint_rows:
            print("No checkpoints found for post-training evaluation.")
        else:
            metrics_rows_by_iteration = _load_metrics_jsonl_by_iteration(metrics_log_path)
            metric_keys_for_ranking = (
                "episode_reward",
                "forward_velocity",
                "episode_length",
                "tracking/cmd_vs_achieved_forward",
                "tracking/step_length_touchdown_event_m",
                "tracking/forward_velocity_cmd_ratio",
            )
            raw_candidates: list[CheckpointMetricCandidate] = []
            used_metrics_jsonl = False
            for iter_num, filename in checkpoint_rows:
                checkpoint_path = os.path.join(job_checkpoint_dir, filename)
                ckpt_data = load_checkpoint(checkpoint_path)
                merged_metrics = dict(ckpt_data.get("metrics", {}) or {})
                metrics_row = metrics_rows_by_iteration.get(int(iter_num), {})
                if metrics_row:
                    used_metrics_jsonl = True
                for key in metric_keys_for_ranking:
                    if _safe_float(merged_metrics.get(key)) is not None:
                        continue
                    row_value = _safe_float(metrics_row.get(key))
                    if row_value is not None:
                        merged_metrics[key] = row_value
                raw_candidates.append(
                    CheckpointMetricCandidate(
                        checkpoint_path=checkpoint_path,
                        iteration=int(iter_num),
                        total_steps=int(ckpt_data.get("total_steps", 0)),
                        metrics=merged_metrics,
                    )
                )

            ranked_candidates, ranking_filter_fallback = rank_checkpoint_candidates(
                raw_candidates,
                top_k=post_training_top_k,
            )
            if not ranked_candidates:
                print("No candidate checkpoints available for post-training evaluation.")
            else:
                if ranking_filter_fallback:
                    print(
                        "  ⚠ train-side hard filters rejected all candidates; "
                        "falling back to score-based ranking on available metrics/reward"
                    )
                if used_metrics_jsonl:
                    print("  ✓ merged metrics.jsonl rows when checkpoint metrics were sparse")
                if any(candidate.used_reward_fallback for candidate in ranked_candidates):
                    print(
                        "  ⚠ ranking used reward fallback for candidates with sparse "
                        "walking metrics"
                    )

                def _fmt(value: Optional[float], spec: str) -> str:
                    return "n/a" if value is None else format(float(value), spec)

                print()
                print("Post-training deterministic eval candidates:")
                print(
                    "rank  checkpoint                      train_score  train_vx  "
                    "train_cmd_err  train_step_len  train_ep_len"
                )
                for rank, candidate in enumerate(ranked_candidates, 1):
                    print(
                        f"{rank:<5} "
                        f"{Path(candidate.checkpoint_path).name:<30} "
                        f"{candidate.train_score:>10.3f}  "
                        f"{_fmt(candidate.train_forward_velocity, '.3f'):>8}  "
                        f"{_fmt(candidate.train_cmd_err, '.3f'):>13}  "
                        f"{_fmt(candidate.train_step_length, '.4f'):>14}  "
                        f"{_fmt(candidate.train_episode_length, '.0f'):>12}"
                    )

                from training.algos.ppo.ppo_core import create_networks, sample_actions
                from training.core.metrics_registry import METRIC_INDEX, METRICS_VEC_KEY
                from training.envs.env_info import PRIVILEGED_OBS_DIM

                actor_activation = str(training_cfg.networks.actor.activation).lower()
                critic_activation = str(training_cfg.networks.critic.activation).lower()
                if actor_activation != critic_activation:
                    raise ValueError(
                        f"actor.activation ({actor_activation!r}) and "
                        f"critic.activation ({critic_activation!r}) must match."
                    )
                critic_obs_dim = (
                    int(PRIVILEGED_OBS_DIM)
                    if bool(training_cfg.ppo.critic_privileged_enabled)
                    else int(policy_spec.model.obs_dim)
                )
                ppo_network = create_networks(
                    obs_dim=int(policy_spec.model.obs_dim),
                    action_dim=int(policy_spec.model.action_dim),
                    policy_hidden_dims=training_cfg.networks.actor.hidden_sizes,
                    value_hidden_dims=training_cfg.networks.critic.hidden_sizes,
                    critic_obs_dim=critic_obs_dim,
                    activation=actor_activation,
                )
                _, post_eval_clean_step_fn, post_eval_reset_fn = make_eval_env_fns(
                    env,
                    training_cfg,
                    post_training_num_envs,
                )

                def _build_rollout_aggregates(
                    rollout: Dict[str, jnp.ndarray],
                ) -> Dict[str, jnp.ndarray]:
                    """Common metric aggregation for both the primary
                    eval and the v0.21.0 probe rollouts.

                    Factored out so the probe path (different reset
                    closure) computes the same set of deploy metrics
                    + the signed lateral / ang_vel_z means used by
                    ``evaluate_lateral_yaw_pass_criterion``."""
                    total_done = jnp.sum(rollout["done"])
                    episode_lengths = (
                        rollout["metrics_vec"][..., METRIC_INDEX["episode_step_count"]]
                        * rollout["done"]
                    )
                    mean_episode_length = jnp.where(
                        total_done > 0.0,
                        jnp.sum(episode_lengths) / total_done,
                        0.0,
                    )
                    mean_episode_reward = jnp.mean(jnp.sum(rollout["reward"], axis=0))

                    def _mean(name: str) -> jnp.ndarray:
                        return jnp.mean(
                            rollout["metrics_vec"][..., METRIC_INDEX[name]]
                        )

                    def _terminal_episode_mean(
                        name: str, abs_aggregate: bool = False
                    ) -> jnp.ndarray:
                        raw = rollout["metrics_vec"][..., METRIC_INDEX[name]]
                        if abs_aggregate:
                            raw = jnp.abs(raw)
                        return jnp.where(
                            total_done > 0.0,
                            jnp.sum(raw * rollout["done"]) / total_done,
                            jnp.float32(0.0),
                        )

                    eps_touchdown = jnp.float32(1e-6)
                    td_left = _mean("tracking/touchdown_rate_left_count")
                    td_right = _mean("tracking/touchdown_rate_right_count")
                    step_left_event = _mean("tracking/step_length_left_event_m")
                    step_right_event = _mean("tracking/step_length_right_event_m")
                    return {
                        "mean_episode_reward": mean_episode_reward,
                        "mean_episode_length": mean_episode_length,
                        "forward_velocity": _mean("forward_velocity"),
                        "cmd_vs_achieved_forward": _mean(
                            "tracking/cmd_vs_achieved_forward"
                        ),
                        "step_length_touchdown_event_m": _mean(
                            "tracking/step_length_touchdown_event_m"
                        ),
                        "lateral_velocity_abs": _mean(
                            "tracking/lateral_velocity_abs"
                        ),
                        # v0.21.0 follow-up — signed siblings consumed
                        # by ``evaluate_lateral_yaw_pass_criterion``.
                        # Per-step MEAN over the rollout (not
                        # per-episode-terminal): the pass criterion
                        # wants average behavior across the rollout,
                        # not the value at episode end.
                        "lateral_velocity_signed_m_s": _mean(
                            "tracking/lateral_velocity_signed_m_s"
                        ),
                        "ang_vel_z_signed_rad_s": _mean(
                            "tracking/ang_vel_z_signed_rad_s"
                        ),
                        "world_x_progress_m": _terminal_episode_mean(
                            "tracking/world_x_progress_m", abs_aggregate=False
                        ),
                        "world_y_drift_signed_m": _terminal_episode_mean(
                            "tracking/world_y_drift_signed_m", abs_aggregate=False
                        ),
                        "world_y_drift_abs_m": _terminal_episode_mean(
                            "tracking/world_y_drift_signed_m", abs_aggregate=True
                        ),
                        "yaw_drift_signed_rad": _terminal_episode_mean(
                            "tracking/yaw_drift_signed_rad", abs_aggregate=False
                        ),
                        "yaw_drift_abs_rad": _terminal_episode_mean(
                            "tracking/yaw_drift_signed_rad", abs_aggregate=True
                        ),
                        "touchdown_rate_left_count": td_left,
                        "touchdown_rate_right_count": td_right,
                        "step_length_left_event_m": step_left_event,
                        "step_length_right_event_m": step_right_event,
                        "step_length_left_per_touchdown_m": (
                            step_left_event
                            / jnp.maximum(td_left, eps_touchdown)
                        ),
                        "step_length_right_per_touchdown_m": (
                            step_right_event
                            / jnp.maximum(td_right, eps_touchdown)
                        ),
                    }

                def _eval_scan_body(policy_params, processor_params):
                    """Closure factory for the per-step scan body.
                    Shared between primary and probe eval paths."""
                    def eval_step(carry, _):
                        env_state, rng_state = carry
                        rng_state, action_rng = jax.random.split(rng_state)
                        action, _, _ = sample_actions(
                            processor_params,
                            policy_params,
                            ppo_network,
                            env_state.obs,
                            action_rng,
                            deterministic=True,
                        )
                        next_env_state = post_eval_clean_step_fn(env_state, action)
                        step_data = {
                            "reward": next_env_state.reward,
                            "done": next_env_state.done,
                            "metrics_vec": next_env_state.metrics[METRICS_VEC_KEY],
                        }
                        return (next_env_state, rng_state), step_data
                    return eval_step

                @jax.jit
                def run_post_training_eval(
                    policy_params: Any,
                    processor_params: Any,
                    eval_rng: jax.Array,
                ) -> Dict[str, jnp.ndarray]:
                    """Deterministic post-training eval rollout.

                    Returns a dict of deploy-facing metrics aggregated
                    over the rollout.  Two aggregation conventions:

                    - Per-step metrics (forward_velocity,
                      cmd_vs_achieved_forward, lateral_velocity_abs,
                      step_length / touchdown event signals): plain
                      MEAN over the (T, N) rollout window.

                    - Cumulative-since-spawn drift metrics
                      (world_x_progress_m, world_y_drift_*,
                      yaw_drift_*): per-episode terminal-step value
                      averaged across completed episodes via
                      ``sum(metric * done) / total_done`` (same
                      pattern as ``mean_episode_length``).  The env
                      emits cumulative drift since the most-recent
                      reset every step, so on the done step the
                      value IS that episode's terminal drift; this
                      aggregation captures the drift of every
                      episode that finishes during the rollout
                      (including envs that auto-reset mid-rollout).

                    For sign-ambiguous drift (yaw, world-y) we also
                    emit ``*_abs_*`` variants computed as
                    ``mean(|terminal drift|)`` over completed
                    episodes.  Those are the values the deploy gate
                    reads — they're invariant under cross-env sign
                    cancellation that would otherwise mask a bad
                    policy (e.g. half the envs drifting +0.5 m and
                    half drifting -0.5 m has signed mean ≈ 0 but
                    abs mean = 0.5 m).
                    """
                    eval_env_state = post_eval_reset_fn(eval_rng)
                    (_, _), rollout = jax.lax.scan(
                        _eval_scan_body(policy_params, processor_params),
                        (eval_env_state, eval_rng),
                        None,
                        length=post_training_num_steps,
                    )
                    return _build_rollout_aggregates(rollout)

                @jax.jit
                def run_post_training_probe_eval(
                    policy_params: Any,
                    processor_params: Any,
                    eval_rng: jax.Array,
                    cmd_override: jax.Array,
                ) -> Dict[str, jnp.ndarray]:
                    """v0.21.0 follow-up — per-probe deterministic eval.

                    Same rollout machinery as ``run_post_training_eval``
                    but the reset uses ``env.reset_for_eval(rng,
                    cmd_override=cmd_override)`` so the YAML-pinned
                    ``eval_velocity_cmd`` is replaced by the probe's
                    explicit (vx, vy, wz).  Steps use
                    ``post_eval_clean_step_fn`` which already passes
                    ``disable_cmd_resample=True``, so the override
                    persists for the full rollout.

                    Used to evaluate the
                    walking_training.md Appendix C lateral / yaw cmd
                    tracking pass criteria at pure-lateral and
                    pure-yaw cmd points (smoke1's primary
                    ``eval_velocity_cmd`` mixes axes and can't isolate
                    either signal).
                    """
                    rngs = jax.random.split(eval_rng, post_training_num_envs)
                    eval_env_state = jax.vmap(
                        lambda r: env.reset_for_eval(r, cmd_override=cmd_override)
                    )(rngs)
                    (_, _), rollout = jax.lax.scan(
                        _eval_scan_body(policy_params, processor_params),
                        (eval_env_state, eval_rng),
                        None,
                        length=post_training_num_steps,
                    )
                    return _build_rollout_aggregates(rollout)

                post_eval_base_rng = jax.random.PRNGKey(
                    training_cfg.seed + training_cfg.ppo.eval.seed_offset + 20_000
                )
                # v0.21.0 P3 / H3: ``eval_velocity_cmd`` is (vx, vy, wz);
                # the post-training gate is a vx-only ratio so we read
                # the [0]th axis.  P6 will add a wider lateral / yaw gate.
                eval_velocity_cmd = float(training_cfg.env.eval_velocity_cmd[0])
                eval_rows: list[dict[str, Any]] = []
                for rank, candidate in enumerate(ranked_candidates, 1):
                    ckpt_data = load_checkpoint(candidate.checkpoint_path)
                    eval_rng = jax.random.fold_in(post_eval_base_rng, int(candidate.iteration))
                    eval_result = run_post_training_eval(
                        ckpt_data["policy_params"],
                        ckpt_data.get("processor_params", ()),
                        eval_rng,
                    )
                    jax.block_until_ready(eval_result["mean_episode_reward"])
                    # Project the JAX dict to Python floats for the
                    # gate + JSON summary.  Any None-equivalent values
                    # (e.g. step_length when episode had no touchdowns)
                    # get coerced to None via _safe_float so the gate's
                    # "step_metric_available" branch fires correctly.
                    step_length_value = _safe_float(
                        eval_result["step_length_touchdown_event_m"]
                    )
                    eval_metrics: Dict[str, Any] = {
                        "mean_reward": float(eval_result["mean_episode_reward"]),
                        "mean_episode_length": float(eval_result["mean_episode_length"]),
                        "forward_velocity": float(eval_result["forward_velocity"]),
                        "cmd_vs_achieved_forward": float(
                            eval_result["cmd_vs_achieved_forward"]
                        ),
                        "step_length_touchdown_event_m": step_length_value,
                        # smoke15 deploy-facing eval metrics — exposed
                        # so the gate (lateral cap) and the JSON summary
                        # can show sideways-walk / heading-drift even
                        # when the G4 forward gates pass.
                        "lateral_velocity_abs": float(
                            eval_result["lateral_velocity_abs"]
                        ),
                        # v0.21.0 follow-up — signed siblings for
                        # ``evaluate_lateral_yaw_pass_criterion``.
                        # On the primary eval these are present for
                        # diagnostic purposes; the probe loop below
                        # is what actually grades them against the
                        # Appendix C pass criteria.
                        "lateral_velocity_signed_m_s": float(
                            eval_result["lateral_velocity_signed_m_s"]
                        ),
                        "ang_vel_z_signed_rad_s": float(
                            eval_result["ang_vel_z_signed_rad_s"]
                        ),
                        "world_x_progress_m": float(
                            eval_result["world_x_progress_m"]
                        ),
                        "world_y_drift_signed_m": float(
                            eval_result["world_y_drift_signed_m"]
                        ),
                        # Cross-env-cancellation-safe abs aggregations
                        # (per-episode |terminal drift| averaged over
                        # completed episodes); the soft gate reads
                        # these, not the signed variants.
                        "world_y_drift_abs_m": float(
                            eval_result["world_y_drift_abs_m"]
                        ),
                        "yaw_drift_signed_rad": float(
                            eval_result["yaw_drift_signed_rad"]
                        ),
                        "yaw_drift_abs_rad": float(
                            eval_result["yaw_drift_abs_rad"]
                        ),
                        "touchdown_rate_left_count": float(
                            eval_result["touchdown_rate_left_count"]
                        ),
                        "touchdown_rate_right_count": float(
                            eval_result["touchdown_rate_right_count"]
                        ),
                        "step_length_left_event_m": float(
                            eval_result["step_length_left_event_m"]
                        ),
                        "step_length_right_event_m": float(
                            eval_result["step_length_right_event_m"]
                        ),
                        "step_length_left_per_touchdown_m": float(
                            eval_result["step_length_left_per_touchdown_m"]
                        ),
                        "step_length_right_per_touchdown_m": float(
                            eval_result["step_length_right_per_touchdown_m"]
                        ),
                    }
                    decision = deterministic_eval_gate(
                        eval_metrics=eval_metrics,
                        eval_velocity_cmd=eval_velocity_cmd,
                        eval_num_steps=post_training_num_steps,
                        strict_lateral_drift=post_training_strict_lateral_drift,
                    )
                    eval_rows.append(
                        {
                            "rank": rank,
                            "checkpoint_path": candidate.checkpoint_path,
                            "checkpoint_name": Path(candidate.checkpoint_path).name,
                            "train_score": candidate.train_score,
                            "train_metrics": {
                                "forward_velocity": candidate.train_forward_velocity,
                                "cmd_vs_achieved_forward": candidate.train_cmd_err,
                                "step_length_touchdown_event_m": candidate.train_step_length,
                                "episode_length": candidate.train_episode_length,
                                "forward_velocity_cmd_ratio": candidate.train_cmd_ratio,
                            },
                            "eval_metrics": eval_metrics,
                            "eval_ratio": decision.forward_velocity_cmd_ratio,
                            "passed": bool(decision.passed),
                            "gates": dict(decision.gates),
                            "fail_reasons": [
                                key for key, ok in decision.gates.items() if not ok
                            ],
                            # smoke15 deploy-facing report-only signals.
                            "soft_signals": dict(decision.soft_signals),
                            "soft_fail_reasons": [
                                key
                                for key, ok in decision.soft_signals.items()
                                if not ok
                            ],
                            "step_metric_available": bool(decision.step_metric_available),
                            "ratio_gate_applied": bool(decision.ratio_gate_applied),
                        }
                    )

                # v0.21.0 follow-up — per-probe deterministic eval for
                # the Appendix C lateral / yaw cmd tracking pass
                # criteria.  One extra rollout per (candidate, probe);
                # each rollout reuses ``post_eval_clean_step_fn`` (which
                # already disables mid-rollout cmd resample), so the
                # probe cmd persists for the full ``post_training_num_steps``.
                # Report-only: probes do NOT block primary promotion
                # (still gated on the G4 forward set against the
                # primary ``eval_velocity_cmd``).  Surfaced in the JSON
                # summary so the smoke promotion decision can be made
                # against the criteria the doc actually writes down.
                probe_cmds: list[Tuple[float, float, float]] = list(
                    getattr(training_cfg.env, "eval_velocity_cmd_probes", ())
                )
                if probe_cmds:
                    print()
                    print(
                        f"Running {len(probe_cmds)} v0.21.0 lateral/yaw "
                        f"probe eval(s) per top-{len(ranked_candidates)} "
                        "candidate..."
                    )
                    from training.core.post_training_eval import (
                        evaluate_lateral_yaw_pass_criterion,
                    )

                    for row in eval_rows:
                        ckpt_data = load_checkpoint(row["checkpoint_path"])
                        probe_results: list[dict[str, Any]] = []
                        for probe_idx, probe_cmd in enumerate(probe_cmds):
                            probe_rng = jax.random.fold_in(
                                jax.random.fold_in(
                                    post_eval_base_rng,
                                    int(row["rank"]) * 1000,
                                ),
                                probe_idx,
                            )
                            probe_cmd_arr = jnp.asarray(
                                probe_cmd, dtype=jnp.float32
                            )
                            probe_result = run_post_training_probe_eval(
                                ckpt_data["policy_params"],
                                ckpt_data.get("processor_params", ()),
                                probe_rng,
                                probe_cmd_arr,
                            )
                            jax.block_until_ready(
                                probe_result["mean_episode_reward"]
                            )
                            probe_eval_metrics: Dict[str, Any] = {
                                "lateral_velocity_signed_m_s": float(
                                    probe_result["lateral_velocity_signed_m_s"]
                                ),
                                "ang_vel_z_signed_rad_s": float(
                                    probe_result["ang_vel_z_signed_rad_s"]
                                ),
                                "lateral_velocity_abs": float(
                                    probe_result["lateral_velocity_abs"]
                                ),
                                "forward_velocity": float(
                                    probe_result["forward_velocity"]
                                ),
                                "mean_episode_length": float(
                                    probe_result["mean_episode_length"]
                                ),
                                "mean_reward": float(
                                    probe_result["mean_episode_reward"]
                                ),
                            }
                            probe_decision = (
                                evaluate_lateral_yaw_pass_criterion(
                                    probe_cmd=probe_cmd,
                                    eval_metrics=probe_eval_metrics,
                                )
                            )
                            probe_results.append(
                                {
                                    "probe_index": probe_idx,
                                    "probe_cmd": list(probe_cmd),
                                    "axis": probe_decision.axis,
                                    "achieved": probe_decision.achieved,
                                    "commanded": probe_decision.commanded,
                                    "signed_ratio": probe_decision.signed_ratio,
                                    "passed": bool(probe_decision.passed),
                                    "skip_reason": probe_decision.skip_reason,
                                    "eval_metrics": probe_eval_metrics,
                                }
                            )
                        row["lateral_yaw_probes"] = probe_results
                        # One-line console summary per candidate so the
                        # human launching the smoke can eyeball
                        # pass/fail across the probes.
                        probe_summary = ", ".join(
                            (
                                f"{p['axis']}@{p['probe_cmd']}={'✓' if p['passed'] else '✗'}"
                                + (
                                    f"({p['signed_ratio']:.2f})"
                                    if p["signed_ratio"] is not None
                                    else ""
                                )
                            )
                            for p in probe_results
                        )
                        print(
                            f"  rank {row['rank']} {row['checkpoint_name']}: "
                            + probe_summary
                        )
                else:
                    # Annotate the summary explicitly — silent absence
                    # of probes is the smoke14 trap (selection looks
                    # forward-only because it IS forward-only).
                    for row in eval_rows:
                        row["lateral_yaw_probes"] = []

                # smoke7 — config-gated 2D acceptance: when strict lateral drift
                # is on AND nonzero-vy probes are configured, the probes BLOCK
                # promotion (they are report-only otherwise).  This completes the
                # acceptance contract: not only "low lateral drift when commanded
                # straight" (the drift hard-gate on the primary cmd) but also
                # "actually tracks the commanded tiny vy" (Appendix C signed
                # ratio >= 0.5 on every non-skipped probe).  Other configs
                # (strict off, or no probes) are unaffected.
                if post_training_strict_lateral_drift and probe_cmds:
                    from training.core.post_training_eval import (
                        lateral_probe_gate_passed,
                    )

                    for row in eval_rows:
                        probes_ok = lateral_probe_gate_passed(
                            row.get("lateral_yaw_probes", [])
                        )
                        row["lateral_probe_gate_passed"] = probes_ok
                        if not probes_ok:
                            row["passed"] = False

                print()
                print("Deterministic eval results:")
                print(
                    "rank  checkpoint                      eval_vx  eval_cmd_err  "
                    "eval_step_len  eval_ep_len  vx/cmd  pass"
                )
                for row in eval_rows:
                    eval_metrics = row["eval_metrics"]
                    ratio_text = (
                        "n/a"
                        if row["eval_ratio"] is None
                        else format(float(row["eval_ratio"]), ".2f")
                    )
                    print(
                        f"{row['rank']:<5} "
                        f"{row['checkpoint_name']:<30} "
                        f"{eval_metrics['forward_velocity']:>7.3f}  "
                        f"{eval_metrics['cmd_vs_achieved_forward']:>12.3f}  "
                        f"{_fmt(eval_metrics['step_length_touchdown_event_m'], '.4f'):>13}  "
                        f"{eval_metrics['mean_episode_length']:>11.0f}  "
                        f"{ratio_text:>6}  "
                        f"{'✓' if row['passed'] else '✗'}"
                    )

                def _eval_select_score(row: dict[str, Any]) -> float:
                    eval_metrics = row["eval_metrics"]
                    step = (
                        float(eval_metrics["step_length_touchdown_event_m"])
                        if eval_metrics["step_length_touchdown_event_m"] is not None
                        else 0.0
                    )
                    return (
                        float(eval_metrics["forward_velocity"])
                        - float(eval_metrics["cmd_vs_achieved_forward"])
                        + step
                        + float(eval_metrics["mean_episode_length"])
                        / float(max(1, post_training_num_steps))
                    )

                passing_rows = [row for row in eval_rows if row["passed"]]
                promoted_checkpoint_path: Optional[str] = None
                selected_row: Optional[dict[str, Any]] = None
                if passing_rows:
                    selected_row = max(passing_rows, key=_eval_select_score)
                    promoted_dir = os.path.join(
                        job_checkpoint_dir,
                        post_training_checkpoint_label,
                    )
                    os.makedirs(promoted_dir, exist_ok=True)
                    promoted_checkpoint_path = os.path.join(
                        promoted_dir,
                        Path(selected_row["checkpoint_path"]).name,
                    )
                    shutil.copy2(selected_row["checkpoint_path"], promoted_checkpoint_path)
                    print(f"  ⭐ Post-training promoted checkpoint: {promoted_checkpoint_path}")
                else:
                    print(
                        "  ⏭ No checkpoint passed deterministic post-training gates; "
                        "no promoted checkpoint written"
                    )

                no_passing_message: Optional[str] = None
                if selected_row is None:
                    # Make the no-passing outcome unambiguous in the
                    # JSON so downstream analyzer / reporting tools
                    # cannot silently fall back to a train-side proxy.
                    no_passing_message = (
                        "No checkpoint passed the deterministic G4 gates "
                        "(forward_velocity, cmd_vs_achieved_forward, "
                        "mean_episode_length, step_length_touchdown_event_m, "
                        "forward_velocity_cmd_ratio).  selected_checkpoint_path is "
                        "null; do NOT promote any checkpoint from this run based on "
                        "training-side proxies like env/episode_length.  See "
                        "top_k_candidates[*].fail_reasons for per-candidate detail."
                    )

                summary_payload = {
                    "selected_checkpoint_path": promoted_checkpoint_path,
                    "selected_rank_before_eval": (
                        None if selected_row is None else int(selected_row["rank"])
                    ),
                    "deterministic_eval_metrics": (
                        None if selected_row is None else selected_row["eval_metrics"]
                    ),
                    "deterministic_eval_gates": (
                        None if selected_row is None else selected_row["gates"]
                    ),
                    # smoke15 — explicit "no candidate passed" message
                    # surfaced both for null and non-null
                    # selected_checkpoint_path (None when a candidate
                    # passes).  The analyzer / downstream reporting
                    # should prefer this summary over train-side
                    # selection heuristics when present.
                    "no_passing_candidate_message": no_passing_message,
                    "deterministic_selection_is_authoritative": True,
                    # v0.21.0 follow-up — record the probe configuration
                    # at the top level so a reader can immediately tell
                    # whether the smoke evaluated the Appendix C
                    # lateral / yaw pass criteria.  Empty list +
                    # ``lateral_yaw_probes_message`` makes the
                    # not-evaluated outcome explicit.
                    "eval_velocity_cmd_probes_configured": [
                        list(p) for p in probe_cmds
                    ],
                    "lateral_yaw_probes_message": (
                        "Lateral / yaw cmd tracking pass criteria "
                        "(walking_training.md Appendix C) were NOT "
                        "evaluated — no probes configured on "
                        "env.eval_velocity_cmd_probes.  Promotion is "
                        "forward-only; do not treat this checkpoint "
                        "as lateral-ready or yaw-ready without "
                        "running probes."
                        if not probe_cmds
                        else (
                            f"Evaluated {len(probe_cmds)} Appendix C probe(s) "
                            "per top-k candidate; see "
                            "top_k_candidates[*].lateral_yaw_probes "
                            "for per-probe pass/fail (>=0.5 signed ratio "
                            "with matching sign)."
                        )
                    ),
                    "top_k_candidates": [
                        {
                            "rank": row["rank"],
                            "checkpoint_path": row["checkpoint_path"],
                            "train_score": row["train_score"],
                            "train_metrics": row["train_metrics"],
                            "eval_metrics": row["eval_metrics"],
                            "eval_ratio": row["eval_ratio"],
                            "passed": row["passed"],
                            "gates": row["gates"],
                            "fail_reasons": row["fail_reasons"],
                            # smoke15 deploy-facing report-only signals.
                            "soft_signals": row["soft_signals"],
                            "soft_fail_reasons": row["soft_fail_reasons"],
                            # v0.21.0 follow-up — per-candidate
                            # Appendix C lateral / yaw probe results.
                            # Empty list when no probes are configured;
                            # the summary metadata below records that
                            # explicitly so a missing-probe outcome is
                            # not silently mistaken for a pass.
                            "lateral_yaw_probes": row.get(
                                "lateral_yaw_probes", []
                            ),
                            "step_metric_available": row["step_metric_available"],
                            "ratio_gate_applied": row["ratio_gate_applied"],
                        }
                        for row in eval_rows
                    ],
                }
                run_summary_path = os.path.join(
                    job_checkpoint_dir,
                    "post_training_eval_summary.json",
                )
                with open(run_summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, indent=2, sort_keys=True)

                if promoted_checkpoint_path is not None:
                    promoted_summary_path = os.path.join(
                        os.path.dirname(promoted_checkpoint_path),
                        "summary.json",
                    )
                    with open(promoted_summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary_payload, f, indent=2, sort_keys=True)

                if wandb_tracker is not None and eval_rows:
                    best_eval_row = max(eval_rows, key=_eval_select_score)
                    best_eval_metrics = best_eval_row["eval_metrics"]
                    wandb_summary = {
                        "post_training_eval/best_forward_velocity": float(
                            best_eval_metrics["forward_velocity"]
                        ),
                        "post_training_eval/best_cmd_vs_achieved_forward": float(
                            best_eval_metrics["cmd_vs_achieved_forward"]
                        ),
                        "post_training_eval/best_step_length_touchdown_event_m": (
                            float(best_eval_metrics["step_length_touchdown_event_m"])
                            if best_eval_metrics["step_length_touchdown_event_m"] is not None
                            else float("nan")
                        ),
                        "post_training_eval/best_mean_episode_length": float(
                            best_eval_metrics["mean_episode_length"]
                        ),
                        "post_training_eval/best_pass": (
                            1.0 if promoted_checkpoint_path is not None else 0.0
                        ),
                        "post_training_eval/evaluated_count": float(len(eval_rows)),
                    }
                    wandb_tracker.log(wandb_summary, step=int(final_state.total_steps))
                    if metrics_log_path is not None:
                        with metrics_log_path.open("a", encoding="utf-8") as f:
                            json.dump(wandb_summary, f, sort_keys=True)
                            f.write("\n")

    # Print best checkpoints summary
    print_top_checkpoints_summary(job_checkpoint_dir)

    return final_state


def override_config_with_cli(training_cfg: "TrainingConfig", args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to training config.

    CLI arguments have highest priority and override both config file values
    and quick_verify settings. Only non-None arguments are applied.

    Args:
        training_cfg: Training configuration to modify in-place
        args: Parsed command-line arguments
    """
    # PPO parameters
    if args.iterations is not None:
        training_cfg.ppo.iterations = args.iterations
    if args.num_envs is not None:
        training_cfg.ppo.num_envs = args.num_envs
    if args.seed is not None:
        training_cfg.seed = args.seed
    if args.lr is not None:
        training_cfg.ppo.learning_rate = args.lr
    if args.gamma is not None:
        training_cfg.ppo.gamma = args.gamma
    if args.clip_epsilon is not None:
        training_cfg.ppo.clip_epsilon = args.clip_epsilon
    if args.entropy_coef is not None:
        training_cfg.ppo.entropy_coef = args.entropy_coef
    if args.log_interval is not None:
        training_cfg.ppo.log_interval = args.log_interval

    # Checkpoint parameters
    if args.checkpoint_dir is not None:
        training_cfg.checkpoints.dir = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        training_cfg.checkpoints.interval = args.checkpoint_interval


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file.
    # v0.20.1: the previous default ``ppo_walking.yaml`` was deleted
    # along with the v1/v2 reference stack.  ``DEFAULT_TRAINING_CONFIG_PATH``
    # now points at ``ppo_walking_v0201_smoke.yaml`` which Task 3
    # (open task #49) hasn't created yet.  If --config is omitted AND
    # the default doesn't exist on disk, fail early with a clear
    # message rather than fall through to a missing file.
    config_path = Path(args.config) if args.config else DEFAULT_TRAINING_CONFIG_PATH
    if not config_path.exists():
        print(
            f"ERROR: Config file not found: {config_path}",
            file=sys.stderr,
        )
        if not args.config:
            print(
                "  --config was omitted and the default "
                f"({DEFAULT_TRAINING_CONFIG_PATH.name}) does not exist on disk.\n"
                "  v0.20.1 cleanup deleted ppo_walking*.yaml; the v0.20.1 PPO\n"
                "  smoke YAML is open task #49.  Pass --config explicitly,\n"
                "  or land the smoke YAML first.",
                file=sys.stderr,
            )
        sys.exit(2)
    print(f"Loading config from: {config_path}")
    training_cfg = load_training_config(config_path)
    training_cfg.config_path = str(config_path)

    # Resolve asset paths from config (variant-aware)
    robot_config_path = Path(training_cfg.env.robot_config_path)
    if not robot_config_path.is_absolute():
        robot_config_path = Path(__file__).parent.parent / robot_config_path
    if not robot_config_path.exists():
        print(f"Warning: Robot config not found at {robot_config_path}")
        print("Generate it via:")
        print("  cd assets/v1 && uv run python ../post_process.py wildrobot.xml")
    else:
        robot_cfg = load_robot_config(robot_config_path)
        print(
            f"Loaded robot config: {robot_cfg.robot_name} (action_dim={robot_cfg.action_dim})"
        )

    # Quick verify mode - apply overrides from config's quick_verify section
    # Priority: CLI > Quick_Verify > Config
    # Apply quick_verify first, then CLI overrides will take precedence
    quick_verify_section = training_cfg.raw_config.get("quick_verify", {})
    quick_verify_enabled = args.verify or quick_verify_section.get("enabled", False)

    if quick_verify_enabled:
        training_cfg.apply_overrides(quick_verify_section)
        print("=" * 60)
        print("VERIFICATION MODE")
        print("Running quick smoke test")
        print("=" * 60)

    # Apply CLI overrides (highest priority - overrides both config and quick_verify)
    override_config_with_cli(training_cfg, args)
    maybe_apply_desktop_gpu_num_env_guard(training_cfg, args)

    # Freeze config after all overrides are applied
    training_cfg.freeze()

    print(f"\n{'=' * 60}")
    print("WildRobot Training")
    print(f"{'=' * 60}")
    print(f"  Version: {training_cfg.version} ({training_cfg.version_name})")
    print(f"  PID: {os.getpid()}  (kill -9 {os.getpid()} to terminate)")
    print(f"  Config: {config_path}")
    print(f"  Iterations: {training_cfg.ppo.iterations}")
    print(f"  Environments: {training_cfg.ppo.num_envs}")
    print(f"  Learning rate: {training_cfg.ppo.learning_rate}")
    print(f"  Seed: {training_cfg.seed}")
    print(f"  Checkpoint dir: {training_cfg.checkpoints.dir}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Initialize W&B tracker
    wandb_cfg = training_cfg.wandb
    wandb_tracker = None

    if wandb_cfg.enabled and not args.verify:
        wandb_tracker = WandbTracker.from_config(
            training_cfg=training_cfg,
            wandb_cfg=wandb_cfg,
        )

        # Define topline metrics with exit criteria targets (Section 3.2.4)
        define_wandb_topline_metrics()

    try:
        # Extract config name from config path (e.g., "ppo_walking" from "ppo_walking.yaml")
        config_name = config_path.stem if config_path else None

        start_training(
            training_cfg=training_cfg,
            wandb_tracker=wandb_tracker,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint_path=args.resume,
            config_name=config_name,
        )

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if args.verify:
            print("\n✅ VERIFICATION PASSED!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Finish W&B tracking
        if wandb_tracker is not None:
            wandb_tracker.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
