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
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    get_top_checkpoints_by_reward,
    load_checkpoint,
    manage_checkpoints,
    print_top_checkpoints_summary,
    save_checkpoint,
    save_checkpoint_from_cpu,
    save_window_best_checkpoint,
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
    return float(env_cfg.eval_velocity_cmd) >= 0.0


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
        # bare-q_ref replay — this is the G6 contract in
        # walking_training.md (v0.20.1 §) and what the pre-smoke wiring
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
