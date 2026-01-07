"""Comprehensive Discriminator Diagnostic Logging for AMP Training.

v0.9.0: Industry-grade diagnostic logging to detect AMP failure modes.

This module provides diagnostic tools to distinguish:
- D blind: inputs too similar, wrong labels, normalization erased signal
- D saturated: too strong, reward collapses
- Reward mapping bug: sigmoid twice, sign error, clipping
- Dataset sampling bug: real frames wrong file or repeated
- Distribution shift: policy feature distribution doesn't resemble reference

Usage:
    from training.amp.discriminator_diagnostics import (
        AMPDiagnostics,
        DiscriminatorDiagnosticState,
    )

    # Initialize
    diag = AMPDiagnostics(config)
    diag_state = diag.init()

    # Log each iteration
    diag_state = diag.log_iteration(
        diag_state,
        logits_real=...,
        logits_fake=...,
        amp_reward=...,
        real_features=...,
        fake_features=...,
    )

    # Check for anomalies (call every N iterations)
    if diag.check_anomalies(diag_state):
        diag.dump_anomaly_batch(diag_state, real_features, fake_features, logits)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from training.amp.policy_features import FeatureConfig


@dataclass
class DiagnosticConfig:
    """Configuration for discriminator diagnostics."""

    # Logging frequency
    log_every_n_iterations: int = 100
    detailed_log_every_n: int = 1000

    # Anomaly detection thresholds
    disc_acc_anomaly_range: Tuple[float, float] = (0.49, 0.51)  # Stuck at 0.5
    amp_reward_anomaly_threshold: float = 0.01  # Very low reward
    saturation_threshold: float = 10.0  # |logit| > this is saturated
    high_prob_threshold: float = 0.99  # prob > this is "certain"
    low_prob_threshold: float = 0.01  # prob < this is "certain"

    # Anomaly persistence
    anomaly_window_size: int = 10  # Check over this many windows
    anomaly_trigger_count: int = 5  # Trigger if anomaly in N of M windows

    # Feature distribution thresholds
    near_constant_std_threshold: float = 1e-4  # Std below this is "constant"

    # Dump configuration
    dump_batch_size: int = 10  # Number of samples to dump on anomaly
    dump_dir: str = "logs/amp_diagnostics"


@dataclass
class DiscriminatorDiagnosticState:
    """State for tracking diagnostics across iterations."""

    # Iteration counter
    iteration: int = 0

    # Rolling statistics
    disc_acc_history: List[float] = field(default_factory=list)
    amp_reward_mean_history: List[float] = field(default_factory=list)
    logit_real_mean_history: List[float] = field(default_factory=list)
    logit_fake_mean_history: List[float] = field(default_factory=list)

    # Saturation counters
    saturation_counts: Dict[str, List[float]] = field(default_factory=dict)

    # Anomaly tracking
    anomaly_flags: List[bool] = field(default_factory=list)
    last_anomaly_dump: int = -1000

    # Feature distribution stats (per block)
    feature_stats: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


class AMPDiagnostics:
    """Comprehensive diagnostic logging for AMP discriminator training."""

    def __init__(
        self,
        amp_config: FeatureConfig,
        diag_config: Optional[DiagnosticConfig] = None,
    ):
        """Initialize diagnostics.

        Args:
            amp_config: AMP feature configuration
            diag_config: Diagnostic configuration (uses defaults if None)
        """
        self.amp_config = amp_config
        self.config = diag_config or DiagnosticConfig()

        # Feature block ranges
        n = amp_config.num_actuated_joints
        self.feature_blocks = {
            "joint_pos": (0, n),
            "joint_vel": (n, 2 * n),
            "root_lin_vel": (2 * n, 2 * n + 3),
            "root_ang_vel": (2 * n + 3, 2 * n + 6),
            "root_height": (2 * n + 6, 2 * n + 7),
            "foot_contacts": (2 * n + 7, 2 * n + 11),
        }

        # Ensure dump directory exists
        Path(self.config.dump_dir).mkdir(parents=True, exist_ok=True)

    def init(self) -> DiscriminatorDiagnosticState:
        """Initialize diagnostic state."""
        state = DiscriminatorDiagnosticState()

        # Initialize saturation counters
        state.saturation_counts = {
            "real_saturated_high": [],
            "real_saturated_low": [],
            "fake_saturated_high": [],
            "fake_saturated_low": [],
            "real_prob_certain": [],
            "fake_prob_certain": [],
        }

        # Initialize feature stats per block
        for block_name in self.feature_blocks:
            state.feature_stats[block_name] = {
                "real_mean": [],
                "real_std": [],
                "fake_mean": [],
                "fake_std": [],
                "near_constant_dims_real": [],
                "near_constant_dims_fake": [],
            }

        return state

    def compute_disc_metrics(
        self,
        logits_real: jnp.ndarray,
        logits_fake: jnp.ndarray,
    ) -> Dict[str, float]:
        """Compute discriminator score separation metrics.

        Args:
            logits_real: Discriminator logits for real samples
            logits_fake: Discriminator logits for fake samples

        Returns:
            Dict with metric values
        """
        # Convert to numpy for easier manipulation
        logits_real = np.asarray(logits_real)
        logits_fake = np.asarray(logits_fake)

        # Logit statistics
        logit_real_mean = float(np.mean(logits_real))
        logit_real_std = float(np.std(logits_real))
        logit_fake_mean = float(np.mean(logits_fake))
        logit_fake_std = float(np.std(logits_fake))

        # Probability statistics (after sigmoid)
        prob_real = 1 / (1 + np.exp(-np.clip(logits_real, -50, 50)))
        prob_fake = 1 / (1 + np.exp(-np.clip(logits_fake, -50, 50)))
        prob_real_mean = float(np.mean(prob_real))
        prob_fake_mean = float(np.mean(prob_fake))

        # Discriminator accuracy
        # D should predict high for real (prob > 0.5) and low for fake (prob < 0.5)
        correct_real = np.mean(prob_real > 0.5)
        correct_fake = np.mean(prob_fake < 0.5)
        disc_acc = float((correct_real + correct_fake) / 2)

        # Saturation metrics
        thresh = self.config.saturation_threshold
        high_thresh = self.config.high_prob_threshold
        low_thresh = self.config.low_prob_threshold

        frac_real_saturated_high = float(np.mean(logits_real > thresh))
        frac_real_saturated_low = float(np.mean(logits_real < -thresh))
        frac_fake_saturated_high = float(np.mean(logits_fake > thresh))
        frac_fake_saturated_low = float(np.mean(logits_fake < -thresh))

        frac_real_prob_certain = float(np.mean(prob_real > high_thresh))
        frac_fake_prob_certain = float(np.mean(prob_fake < low_thresh))

        return {
            "logit_real_mean": logit_real_mean,
            "logit_real_std": logit_real_std,
            "logit_fake_mean": logit_fake_mean,
            "logit_fake_std": logit_fake_std,
            "prob_real_mean": prob_real_mean,
            "prob_fake_mean": prob_fake_mean,
            "disc_acc": disc_acc,
            "frac_real_saturated_high": frac_real_saturated_high,
            "frac_real_saturated_low": frac_real_saturated_low,
            "frac_fake_saturated_high": frac_fake_saturated_high,
            "frac_fake_saturated_low": frac_fake_saturated_low,
            "frac_real_prob_certain": frac_real_prob_certain,
            "frac_fake_prob_certain": frac_fake_prob_certain,
        }

    def compute_amp_reward_stats(
        self,
        amp_reward: jnp.ndarray,
        prob_fake: Optional[jnp.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute AMP reward statistics.

        Args:
            amp_reward: AMP reward values
            prob_fake: Optional discriminator probability for fake samples

        Returns:
            Dict with reward statistics
        """
        amp_reward = np.asarray(amp_reward).flatten()

        stats = {
            "amp_reward_mean": float(np.mean(amp_reward)),
            "amp_reward_std": float(np.std(amp_reward)),
            "amp_reward_min": float(np.min(amp_reward)),
            "amp_reward_max": float(np.max(amp_reward)),
            "amp_reward_p10": float(np.percentile(amp_reward, 10)),
            "amp_reward_p50": float(np.percentile(amp_reward, 50)),
            "amp_reward_p90": float(np.percentile(amp_reward, 90)),
        }

        # Correlation with discriminator probability
        if prob_fake is not None:
            prob_fake = np.asarray(prob_fake).flatten()
            if len(prob_fake) == len(amp_reward) and np.std(prob_fake) > 1e-8:
                correlation = np.corrcoef(amp_reward, prob_fake)[0, 1]
                stats["reward_prob_correlation"] = float(correlation)

        return stats

    def compute_feature_distribution_stats(
        self,
        real_features: jnp.ndarray,
        fake_features: jnp.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute feature distribution statistics per block.

        Args:
            real_features: Real (reference) features
            fake_features: Fake (policy) features

        Returns:
            Dict mapping block names to statistics
        """
        real_features = np.asarray(real_features)
        fake_features = np.asarray(fake_features)

        block_stats = {}

        for block_name, (start, end) in self.feature_blocks.items():
            real_block = real_features[:, start:end]
            fake_block = fake_features[:, start:end]

            # Mean/std per dimension, then average
            real_mean = float(np.mean(real_block))
            real_std = float(np.mean(np.std(real_block, axis=0)))
            fake_mean = float(np.mean(fake_block))
            fake_std = float(np.mean(np.std(fake_block, axis=0)))

            # Count near-constant dimensions
            dim_stds_real = np.std(real_block, axis=0)
            dim_stds_fake = np.std(fake_block, axis=0)
            near_const_real = int(
                np.sum(dim_stds_real < self.config.near_constant_std_threshold)
            )
            near_const_fake = int(
                np.sum(dim_stds_fake < self.config.near_constant_std_threshold)
            )

            # Check for NaN/Inf
            nan_inf_real = int(np.sum(~np.isfinite(real_block)))
            nan_inf_fake = int(np.sum(~np.isfinite(fake_block)))

            block_stats[block_name] = {
                "real_mean": real_mean,
                "real_std": real_std,
                "fake_mean": fake_mean,
                "fake_std": fake_std,
                "near_constant_dims_real": near_const_real,
                "near_constant_dims_fake": near_const_fake,
                "nan_inf_real": nan_inf_real,
                "nan_inf_fake": nan_inf_fake,
            }

        return block_stats

    def log_iteration(
        self,
        state: DiscriminatorDiagnosticState,
        logits_real: jnp.ndarray,
        logits_fake: jnp.ndarray,
        amp_reward: jnp.ndarray,
        real_features: Optional[jnp.ndarray] = None,
        fake_features: Optional[jnp.ndarray] = None,
        motion_ids: Optional[jnp.ndarray] = None,
    ) -> Tuple[DiscriminatorDiagnosticState, Dict[str, Any]]:
        """Log diagnostics for one training iteration.

        Args:
            state: Current diagnostic state
            logits_real: Discriminator logits for real samples
            logits_fake: Discriminator logits for fake samples
            amp_reward: AMP reward values
            real_features: Optional real features for distribution analysis
            fake_features: Optional fake features for distribution analysis
            motion_ids: Optional motion IDs for sampling diagnostics

        Returns:
            Tuple of (updated state, metrics dict)
        """
        state.iteration += 1
        metrics = {}

        # Always compute basic discriminator metrics
        disc_metrics = self.compute_disc_metrics(logits_real, logits_fake)
        metrics.update(disc_metrics)

        # Update history
        state.disc_acc_history.append(disc_metrics["disc_acc"])
        state.logit_real_mean_history.append(disc_metrics["logit_real_mean"])
        state.logit_fake_mean_history.append(disc_metrics["logit_fake_mean"])

        # Update saturation counts
        state.saturation_counts["real_saturated_high"].append(
            disc_metrics["frac_real_saturated_high"]
        )
        state.saturation_counts["fake_saturated_high"].append(
            disc_metrics["frac_fake_saturated_high"]
        )

        # Compute AMP reward stats
        prob_fake = 1 / (1 + np.exp(-np.clip(np.asarray(logits_fake), -50, 50)))
        reward_stats = self.compute_amp_reward_stats(amp_reward, prob_fake)
        metrics.update(reward_stats)

        state.amp_reward_mean_history.append(reward_stats["amp_reward_mean"])

        # Detailed logging (less frequent)
        if (
            state.iteration % self.config.detailed_log_every_n == 0
            and real_features is not None
            and fake_features is not None
        ):
            feature_stats = self.compute_feature_distribution_stats(
                real_features, fake_features
            )
            metrics["feature_distribution"] = feature_stats

            # Update feature stats history
            for block_name, stats in feature_stats.items():
                for stat_name, value in stats.items():
                    if stat_name in state.feature_stats[block_name]:
                        state.feature_stats[block_name][stat_name].append(value)

            # Sampling diagnostics
            if motion_ids is not None:
                motion_ids = np.asarray(motion_ids)
                unique_ids = len(np.unique(motion_ids))
                metrics["sampling"] = {
                    "unique_motion_ids": unique_ids,
                    "total_samples": len(motion_ids),
                    "diversity_ratio": unique_ids / max(1, len(motion_ids)),
                }

        # Check for anomaly
        is_anomaly = self._check_current_anomaly(metrics)
        state.anomaly_flags.append(is_anomaly)

        # Keep history bounded
        max_history = self.config.anomaly_window_size * 10
        if len(state.disc_acc_history) > max_history:
            state.disc_acc_history = state.disc_acc_history[-max_history:]
            state.amp_reward_mean_history = state.amp_reward_mean_history[-max_history:]
            state.anomaly_flags = state.anomaly_flags[-max_history:]

        metrics["is_anomaly"] = is_anomaly

        return state, metrics

    def _check_current_anomaly(self, metrics: Dict[str, Any]) -> bool:
        """Check if current iteration shows anomalous behavior."""
        disc_acc = metrics.get("disc_acc", 0.5)
        amp_reward = metrics.get("amp_reward_mean", 0)

        # Check disc_acc stuck near 0.5
        low, high = self.config.disc_acc_anomaly_range
        if low <= disc_acc <= high:
            return True

        # Check very low AMP reward
        if amp_reward < self.config.amp_reward_anomaly_threshold:
            return True

        return False

    def check_persistent_anomaly(self, state: DiscriminatorDiagnosticState) -> bool:
        """Check if anomaly has persisted across multiple windows.

        Args:
            state: Current diagnostic state

        Returns:
            True if anomaly is persistent
        """
        if len(state.anomaly_flags) < self.config.anomaly_window_size:
            return False

        recent_flags = state.anomaly_flags[-self.config.anomaly_window_size :]
        anomaly_count = sum(recent_flags)

        return anomaly_count >= self.config.anomaly_trigger_count

    def dump_anomaly_batch(
        self,
        state: DiscriminatorDiagnosticState,
        real_features: jnp.ndarray,
        fake_features: jnp.ndarray,
        logits_real: jnp.ndarray,
        logits_fake: jnp.ndarray,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Dump a small batch of data for anomaly investigation.

        Args:
            state: Current diagnostic state
            real_features: Real features batch
            fake_features: Fake features batch
            logits_real: Real logits
            logits_fake: Fake logits
            extra_info: Optional extra information to include

        Returns:
            Path to dump file
        """
        # Prevent too frequent dumps
        if state.iteration - state.last_anomaly_dump < 100:
            return ""

        state.last_anomaly_dump = state.iteration

        n = self.config.dump_batch_size
        real_features = np.asarray(real_features)[:n]
        fake_features = np.asarray(fake_features)[:n]
        logits_real = np.asarray(logits_real)[:n]
        logits_fake = np.asarray(logits_fake)[:n]

        # Extract key dimensions for quick analysis
        n_joints = self.amp_config.num_actuated_joints
        dump_data = {
            "iteration": state.iteration,
            "timestamp": str(np.datetime64("now")),
            "real_features": real_features.tolist(),
            "fake_features": fake_features.tolist(),
            "logits_real": logits_real.tolist(),
            "logits_fake": logits_fake.tolist(),
            # Extract specific components for easy viewing
            "real_linvel": real_features[:, 2 * n_joints : 2 * n_joints + 3].tolist(),
            "fake_linvel": fake_features[:, 2 * n_joints : 2 * n_joints + 3].tolist(),
            "real_angvel": real_features[
                :, 2 * n_joints + 3 : 2 * n_joints + 6
            ].tolist(),
            "fake_angvel": fake_features[
                :, 2 * n_joints + 3 : 2 * n_joints + 6
            ].tolist(),
            # Recent history
            "recent_disc_acc": state.disc_acc_history[-20:],
            "recent_amp_reward": state.amp_reward_mean_history[-20:],
        }

        if extra_info:
            dump_data["extra"] = extra_info

        # Save dump
        dump_path = (
            Path(self.config.dump_dir) / f"anomaly_dump_iter{state.iteration}.json"
        )
        with open(dump_path, "w") as f:
            json.dump(dump_data, f, indent=2)

        print(f"[AMP Diagnostics] Anomaly dump saved to: {dump_path}")
        return str(dump_path)

    def format_summary(
        self,
        state: DiscriminatorDiagnosticState,
        metrics: Dict[str, Any],
    ) -> str:
        """Format a summary string for logging.

        Args:
            state: Current diagnostic state
            metrics: Current metrics

        Returns:
            Formatted summary string
        """
        lines = [
            f"[AMP Diag] iter={state.iteration}",
            f"  D: acc={metrics.get('disc_acc', 0):.3f}, "
            f"real_logit={metrics.get('logit_real_mean', 0):.2f}±{metrics.get('logit_real_std', 0):.2f}, "
            f"fake_logit={metrics.get('logit_fake_mean', 0):.2f}±{metrics.get('logit_fake_std', 0):.2f}",
            f"  R: mean={metrics.get('amp_reward_mean', 0):.4f}, "
            f"p50={metrics.get('amp_reward_p50', 0):.4f}, "
            f"p90={metrics.get('amp_reward_p90', 0):.4f}",
        ]

        if metrics.get("is_anomaly"):
            lines.append("  ⚠️ ANOMALY DETECTED")

        return "\n".join(lines)


def integrate_with_training(
    diag: AMPDiagnostics,
    state: DiscriminatorDiagnosticState,
    logits_real: jnp.ndarray,
    logits_fake: jnp.ndarray,
    amp_reward: jnp.ndarray,
    real_features: Optional[jnp.ndarray] = None,
    fake_features: Optional[jnp.ndarray] = None,
    log_fn: Optional[callable] = None,
) -> DiscriminatorDiagnosticState:
    """Helper function to integrate diagnostics with training loop.

    Example usage in training:
        diag = AMPDiagnostics(amp_config)
        diag_state = diag.init()

        for iteration in range(num_iterations):
            # ... training code ...

            diag_state = integrate_with_training(
                diag, diag_state,
                logits_real, logits_fake, amp_reward,
                real_features, fake_features,
                log_fn=wandb.log  # or tensorboard, etc.
            )

    Args:
        diag: AMPDiagnostics instance
        state: Current diagnostic state
        logits_real: Real logits
        logits_fake: Fake logits
        amp_reward: AMP reward
        real_features: Optional real features
        fake_features: Optional fake features
        log_fn: Optional logging function (e.g., wandb.log)

    Returns:
        Updated diagnostic state
    """
    state, metrics = diag.log_iteration(
        state,
        logits_real,
        logits_fake,
        amp_reward,
        real_features,
        fake_features,
    )

    # Log to external system if provided
    if log_fn is not None and state.iteration % diag.config.log_every_n_iterations == 0:
        log_metrics = {
            f"amp_diag/{k}": v for k, v in metrics.items() if not isinstance(v, dict)
        }
        log_fn(log_metrics)

    # Check for persistent anomaly
    if diag.check_persistent_anomaly(state):
        print(diag.format_summary(state, metrics))
        if real_features is not None and fake_features is not None:
            diag.dump_anomaly_batch(
                state, real_features, fake_features, logits_real, logits_fake
            )

    return state
