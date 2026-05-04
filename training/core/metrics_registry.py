"""Metrics Registry for WildRobot Training.

This module defines a centralized registry for all training metrics.
It enables a "packed array" approach where metrics are stored as a single
fixed-shape tensor, making the rollout fully JIT-compatible with lax.scan.

Design Principles:
1. Single source of truth for metric names and indices
2. Fixed pytree structure for lax.scan compatibility
3. No .get() fallbacks - missing metrics cause hard failures
4. Adding new metrics requires only registry + env changes

Usage:
    from training.core.metrics_registry import (
        METRIC_NAMES,
        METRIC_INDEX,
        build_metrics_vec,
        unpack_metrics,
    )

    # In env: build packed vector
    metrics_vec = build_metrics_vec({
        "forward_velocity": forward_vel,
        "height": height,
        ...
    })

    # In training: unpack for logging
    metrics_dict = unpack_metrics(metrics_mean)
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, NamedTuple

import jax.numpy as jnp


class Reducer(Enum):
    """Aggregation method for metrics across rollout."""
    MEAN = "mean"           # Standard mean over (T, N)
    MAX = "max"             # Max over (T, N) - for peak values
    MIN = "min"             # Min over (T, N) - for lower bounds
    SUM = "sum"             # Sum over (T, N) - for counts
    LAST = "last"           # Last value only - for episode-end metrics


class MetricSpec(NamedTuple):
    """Specification for a single metric."""
    name: str                           # Unique identifier (e.g., "forward_velocity")
    reducer: Reducer = Reducer.MEAN     # How to aggregate over rollout
    log_prefix: str = "env"             # W&B prefix (e.g., "env/forward_velocity")
    topline: bool = False               # Also log as topline/ metric
    description: str = ""               # Documentation


# =============================================================================
# METRIC REGISTRY - Single source of truth
# =============================================================================
# Order matters! Index in this list = index in packed vector.
# When adding metrics, append to end to avoid breaking checkpoints.

METRIC_SPECS: List[MetricSpec] = [
    # =========================================================================
    # Core locomotion metrics (indices 0-2)
    # =========================================================================
    MetricSpec(
        name="forward_velocity",
        reducer=Reducer.MEAN,
        topline=True,
        description="Forward velocity in m/s",
    ),
    MetricSpec(
        name="height",
        reducer=Reducer.MEAN,
        description="Robot base height in m",
    ),
    MetricSpec(
        name="velocity_command",
        reducer=Reducer.MEAN,
        topline=True,
        description="Commanded velocity in m/s",
    ),
    # v0.10.4: Episode step count for accurate ep_len calculation
    MetricSpec(
        name="episode_step_count",
        reducer=Reducer.MEAN,  # Will be weighted by dones in training_loop
        description="Step count within episode (for ep_len)",
    ),

    # =========================================================================
    # v0.10.3: Walking tracking metrics (indices 3-4)
    # =========================================================================
    MetricSpec(
        name="tracking/vel_error",
        reducer=Reducer.MEAN,
        topline=True,
        description="|forward_vel - velocity_cmd| tracking error",
    ),
    MetricSpec(
        name="tracking/max_torque",
        reducer=Reducer.MEAN,
        topline=True,
        description="Peak torque as fraction of limit (0-1)",
    ),
    MetricSpec(
        name="tracking/avg_torque",
        reducer=Reducer.MEAN,
        description="Mean absolute actuator torque (Nm)",
    ),

    # =========================================================================
    # Reward components (indices 5-20)
    # =========================================================================
    MetricSpec(
        name="reward/total",
        reducer=Reducer.MEAN,
        description="Total reward",
    ),
    MetricSpec(
        name="reward/forward",
        reducer=Reducer.MEAN,
        description="Forward velocity tracking reward",
    ),
    MetricSpec(
        name="reward/lateral",
        reducer=Reducer.MEAN,
        description="Lateral velocity penalty",
    ),
    MetricSpec(
        name="reward/healthy",
        reducer=Reducer.MEAN,
        description="Alive/healthy bonus",
    ),
    MetricSpec(
        name="reward/orientation",
        reducer=Reducer.MEAN,
        description="Orientation penalty (pitch² + roll²)",
    ),
    MetricSpec(
        name="reward/angvel",
        reducer=Reducer.MEAN,
        description="Angular velocity penalty",
    ),
    MetricSpec(
        name="reward/height_target",
        reducer=Reducer.MEAN,
        description="Height target reward",
    ),
    MetricSpec(
        name="reward/height_floor",
        reducer=Reducer.MEAN,
        description="Disturbed-window height-floor penalty",
    ),
    MetricSpec(
        name="reward/torque",
        reducer=Reducer.MEAN,
        description="Torque penalty",
    ),
    MetricSpec(
        name="reward/saturation",
        reducer=Reducer.MEAN,
        description="Actuator saturation penalty",
    ),
    MetricSpec(
        name="reward/action_rate",
        reducer=Reducer.MEAN,
        description="Action rate penalty (smoothness)",
    ),
    MetricSpec(
        name="reward/joint_vel",
        reducer=Reducer.MEAN,
        description="Joint velocity penalty",
    ),
    MetricSpec(
        name="reward/slip",
        reducer=Reducer.MEAN,
        description="Foot slip penalty",
    ),
    MetricSpec(
        name="reward/clearance",
        reducer=Reducer.MEAN,
        description="Swing foot clearance reward",
    ),
    # v0.10.4: Standing penalty
    MetricSpec(
        name="reward/standing",
        reducer=Reducer.MEAN,
        description="Standing still penalty (1.0 when |vel| < threshold)",
    ),
    MetricSpec(
        name="reward/gait_periodicity",
        reducer=Reducer.MEAN,
        description="Alternating support reward (left XOR right contact)",
    ),
    MetricSpec(
        name="reward/hip_swing",
        reducer=Reducer.MEAN,
        description="Hip swing amplitude reward",
    ),
    MetricSpec(
        name="reward/knee_swing",
        reducer=Reducer.MEAN,
        description="Knee swing amplitude reward",
    ),
    MetricSpec(
        name="reward/flight_phase",
        reducer=Reducer.MEAN,
        description="Flight phase penalty (both feet unloaded)",
    ),
    MetricSpec(
        name="reward/stance_width_penalty",
        reducer=Reducer.MEAN,
        description="Stance width penalty",
    ),

    # =========================================================================
    # Debug metrics
    # =========================================================================
    MetricSpec(
        name="debug/pitch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Body pitch angle",
    ),
    MetricSpec(
        name="debug/roll",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Body roll angle",
    ),
    MetricSpec(
        name="debug/root_height",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Root body height (meters)",
    ),
    MetricSpec(
        name="debug/root_height_rate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Root height rate of change (m/s), gate target 0.03-0.05",
    ),
    MetricSpec(
        name="debug/forward_vel",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw forward velocity",
    ),
    MetricSpec(
        name="debug/lateral_vel",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw lateral velocity",
    ),
    MetricSpec(
        name="debug/left_force",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left foot contact force",
    ),
    MetricSpec(
        name="debug/right_force",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right foot contact force",
    ),
    MetricSpec(
        name="debug/left_toe_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left toe switch state",
    ),
    MetricSpec(
        name="debug/left_heel_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left heel switch state",
    ),
    MetricSpec(
        name="debug/right_toe_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right toe switch state",
    ),
    MetricSpec(
        name="debug/right_heel_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right heel switch state",
    ),

    # =========================================================================
    # Termination diagnostics (indices 23-30)
    # =========================================================================
    MetricSpec(
        name="term/height_low",
        reducer=Reducer.MEAN,
        description="Height too low termination flag",
    ),
    MetricSpec(
        name="term/height_high",
        reducer=Reducer.MEAN,
        description="Height too high termination flag",
    ),
    MetricSpec(
        name="term/pitch",
        reducer=Reducer.MEAN,
        description="Pitch limit termination flag",
    ),
    MetricSpec(
        name="term/roll",
        reducer=Reducer.MEAN,
        description="Roll limit termination flag",
    ),
    MetricSpec(
        name="term/truncated",
        reducer=Reducer.MEAN,
        description="Successful truncation (max steps reached)",
    ),
    MetricSpec(
        name="term/pitch_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Pitch value at termination",
    ),
    MetricSpec(
        name="term/roll_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Roll value at termination",
    ),
    MetricSpec(
        name="term/height_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Height value at termination",
    ),
    # =========================================================================
    # v0.11.4+: Failure mode debugging (append-only)
    # =========================================================================
    MetricSpec(
        name="debug/pitch_rate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Body pitch rate (heading-local angvel y)",
    ),
    MetricSpec(
        name="debug/velocity_step_gate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Forward reward gate value (v0.15.10 propulsion-quality gate)",
    ),
    MetricSpec(
        name="debug/action_abs_max",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Max |action| across actuators (post-filter)",
    ),
    MetricSpec(
        name="debug/action_sat_frac",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Fraction of actuators with |action|>0.95 (post-filter)",
    ),
    MetricSpec(
        name="debug/raw_action_abs_max",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Max |action| across actuators (pre-filter)",
    ),
    MetricSpec(
        name="debug/raw_action_sat_frac",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Fraction of actuators with |action|>0.95 (pre-filter)",
    ),
    MetricSpec(
        name="debug/torque_abs_max",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Max |torque| as fraction of limit across actuators",
    ),
    MetricSpec(
        name="debug/torque_sat_frac",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Fraction of actuators with |torque|>0.95 of limit",
    ),
    # v0.13.8+: Pre-collapse shaping diagnostics (append-only)
    MetricSpec(
        name="reward/collapse_height_pen",
        reducer=Reducer.MEAN,
        description="Pre-collapse low-height margin penalty",
    ),
    MetricSpec(
        name="reward/collapse_vz_pen",
        reducer=Reducer.MEAN,
        description="Pre-collapse downward vertical velocity penalty",
    ),
    # v0.13.10+: Posture return shaping diagnostics (append-only)
    MetricSpec(
        name="reward/posture",
        reducer=Reducer.MEAN,
        description="Posture return reward (gated exp(-mse/sigma^2))",
    ),
    MetricSpec(
        name="reward/com_velocity_damping",
        reducer=Reducer.MEAN,
        description="Disturbed-window horizontal CoM speed damping reward",
    ),
    MetricSpec(
        name="reward/pitch_rate",
        reducer=Reducer.MEAN,
        description="Pitch-rate penalty",
    ),
    MetricSpec(
        name="reward/penalty_slip_raw",
        reducer=Reducer.MEAN,
        description=(
            "Raw stance-foot slip penalty (Σ stance horizontal vel²); "
            "logged independently of reward_weights.slip so the M1 "
            "fail-mode tree can size the weight before turning it on."
        ),
    ),
    MetricSpec(
        name="reward/penalty_pitch_rate_raw",
        reducer=Reducer.MEAN,
        description=(
            "Raw pitch-rate penalty (gyro_y²); logged independently of "
            "reward_weights.pitch_rate so the M1 fail-mode tree can "
            "size the weight before turning it on."
        ),
    ),
    MetricSpec(
        name="debug/posture_mse",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Mean squared error between joint_pos_rad and default pose (gated in reward)",
    ),
    MetricSpec(
        name="reward/step_event",
        reducer=Reducer.MEAN,
        description="Step event reward (touchdown * step reward gate)",
    ),
    MetricSpec(
        name="reward/foot_place",
        reducer=Reducer.MEAN,
        description="Foot placement reward at touchdown (Raibert-style, gated by step reward gate)",
    ),
    MetricSpec(
        name="debug/need_step",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Need-to-step gate (0..1) used for stepping-only rewards",
    ),
    MetricSpec(
        name="debug/touchdown_left",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left-foot touchdown event flag",
    ),
    MetricSpec(
        name="debug/touchdown_right",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right-foot touchdown event flag",
    ),
    # v0.14.x: M3 foot-placement FSM debug metrics (append-only)
    MetricSpec(
        name="debug/bc_phase",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 base-controller FSM phase (0=STANCE, 1=SWING, 2=TOUCHDOWN_RECOVER)",
    ),
    MetricSpec(
        name="debug/bc_in_swing",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 FSM swing occupancy fraction",
    ),
    MetricSpec(
        name="debug/bc_in_recover",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 FSM touchdown-recover occupancy fraction",
    ),
    MetricSpec(
        name="debug/bc_swing_foot",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 FSM active swing foot (0=left, 1=right)",
    ),
    MetricSpec(
        name="debug/bc_phase_ticks",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 FSM ticks spent in current phase",
    ),
    # v0.15.9: propulsion-per-step and cycle diagnostics (append-only)
    MetricSpec(
        name="reward/step_length",
        reducer=Reducer.MEAN,
        description="Touchdown step-length reward (forward landing relative to body)",
    ),
    MetricSpec(
        name="reward/step_progress",
        reducer=Reducer.MEAN,
        description="Touchdown-to-touchdown structured forward progress reward",
    ),
    MetricSpec(
        name="reward/m3_pelvis_orientation_tracking",
        reducer=Reducer.MEAN,
        description="Pelvis orientation tracking reward to locomotion reference",
    ),
    MetricSpec(
        name="reward/m3_pelvis_height_tracking",
        reducer=Reducer.MEAN,
        description="Pelvis height tracking reward to locomotion reference",
    ),
    MetricSpec(
        name="reward/m3_swing_foot_tracking",
        reducer=Reducer.MEAN,
        description="Swing-foot tracking reward to locomotion reference",
    ),
    MetricSpec(
        name="reward/m3_foothold_consistency",
        reducer=Reducer.MEAN,
        description="Touchdown consistency reward vs desired foothold",
    ),
    MetricSpec(
        name="reward/m3_residual_magnitude",
        reducer=Reducer.MEAN,
        description="Residual action magnitude penalty in joint space",
    ),
    MetricSpec(
        name="reward/m3_excessive_impact",
        reducer=Reducer.MEAN,
        description="Excessive contact impact penalty",
    ),
    MetricSpec(
        name="tracking/cmd_vs_achieved_forward",
        reducer=Reducer.MEAN,
        description="Absolute commanded-vs-achieved forward speed error",
    ),
    MetricSpec(
        name="tracking/loc_ref_phase_progress",
        reducer=Reducer.MEAN,
        description="Locomotion reference intra-step phase progress in [0, 1]",
    ),
    MetricSpec(
        name="tracking/loc_ref_stance_foot",
        reducer=Reducer.MEAN,
        description="Locomotion reference stance foot id (0/1)",
    ),
    MetricSpec(
        name="tracking/loc_ref_mode_id",
        reducer=Reducer.MEAN,
        description="Locomotion reference hybrid mode id",
    ),
    MetricSpec(
        name="tracking/loc_ref_progression_permission",
        reducer=Reducer.MEAN,
        description="Support-conditioned progression permission in [0, 1]",
    ),
    MetricSpec(
        name="tracking/nominal_q_abs_mean",
        reducer=Reducer.MEAN,
        description="Mean absolute nominal q_ref magnitude",
    ),
    MetricSpec(
        name="tracking/residual_q_abs_mean",
        reducer=Reducer.MEAN,
        description="Mean absolute residual delta_q magnitude",
    ),
    MetricSpec(
        name="tracking/residual_q_abs_max",
        reducer=Reducer.MEAN,
        description="Max absolute residual delta_q magnitude",
    ),
    MetricSpec(
        name="tracking/loc_ref_left_reachable",
        reducer=Reducer.MEAN,
        description="Left nominal IK target reachable fraction",
    ),
    MetricSpec(
        name="tracking/loc_ref_right_reachable",
        reducer=Reducer.MEAN,
        description="Right nominal IK target reachable fraction",
    ),
    MetricSpec(
        name="reward/teacher_target_step_xy",
        reducer=Reducer.MEAN,
        description="Teacher XY target agreement reward at first recovery touchdown",
    ),
    MetricSpec(
        name="reward/teacher_step_required",
        reducer=Reducer.MEAN,
        description="Teacher step-required reward on first recovery step-init events",
    ),
    MetricSpec(
        name="reward/teacher_swing_foot",
        reducer=Reducer.MEAN,
        description="Teacher swing-foot agreement reward on first recovery touchdown",
    ),
    MetricSpec(
        name="reward/cycle_progress",
        reducer=Reducer.MEAN,
        description="Per-step-cycle net forward displacement reward",
    ),
    MetricSpec(
        name="debug/cycle_complete",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Cycle completion event flag based on clock_stride_period_steps",
    ),
    MetricSpec(
        name="debug/cycle_forward_delta",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Accumulated heading-local forward displacement within current cycle",
    ),
    MetricSpec(
        name="debug/step_progress_delta",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Forward progress delta between alternating touchdowns",
    ),
    MetricSpec(
        name="debug/step_progress_event",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Alternating touchdown event indicator for step_progress reward",
    ),
    MetricSpec(
        name="reward/dense_progress",
        reducer=Reducer.MEAN,
        description="Dense heading-local forward progress shaping reward",
    ),
    MetricSpec(
        name="debug/propulsion_gate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Propulsion-quality gate applied to forward reward",
    ),
    # v0.17.0+: Recovery metrics for standing-reset branch (append-only)
    # v0.17.1: Corrected semantics with episode-level state tracking
    MetricSpec(
        name="recovery/first_step_latency",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-step latencies from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/touchdown_count",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed touchdown counts from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/support_foot_changes",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed support-foot changes from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/post_push_velocity",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed final horizontal velocities from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/completed",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Number of finalized recovery summaries in the rollout",
    ),
    MetricSpec(
        name="recovery/no_touchdown_frac",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed no-touchdown indicators from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/touchdown_then_fail_frac",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed touchdown-then-fail indicators from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_liftoff_latency",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-liftoff latencies from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_touchdown_latency",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-touchdown latencies from finalized recovery summaries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_step_dx",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-step touchdown x in heading-local frame; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_step_dy",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-step touchdown y in heading-local frame; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_step_target_err_x",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-step x target errors; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_step_target_err_y",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed first-step y target errors; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/pitch_rate_at_push_end",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed pitch-rate snapshots at push end; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/pitch_rate_at_touchdown",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed pitch-rate snapshots at first touchdown; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/pitch_rate_reduction_10t",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed pitch-rate reduction from first touchdown to +10 ticks; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/capture_error_at_push_end",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed capture-error snapshots at push end; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/capture_error_at_touchdown",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed capture-error snapshots at first touchdown; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/capture_error_reduction_10t",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed capture-error reduction from first touchdown to +10 ticks; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/touchdown_to_term_steps",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed steps from first touchdown to failure termination; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/visible_step_rate",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed visible-step indicators (first liftoff followed by first touchdown); normalized by recovery/completed",
    ),
    MetricSpec(
        name="unnecessary_step_rate",
        reducer=Reducer.MEAN,
        log_prefix="eval_clean",
        description="Per-step touchdown event when no push/recovery active and need_step is low",
    ),
    # =========================================================================
    # v0.17.3: Architecture-pivot controller debug metrics
    # =========================================================================
    MetricSpec(
        name="debug/mpc_planner_active",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="MPC standing planner active flag (bring-up scaffold)",
    ),
    MetricSpec(
        name="debug/mpc_controller_active",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="MPC standing controller active flag (bring-up scaffold)",
    ),
    MetricSpec(
        name="debug/mpc_target_com_x",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner target CoM x (debug only)",
    ),
    MetricSpec(
        name="debug/mpc_target_com_y",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner target CoM y (debug only)",
    ),
    MetricSpec(
        name="debug/mpc_target_step_x",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner target step x (debug only)",
    ),
    MetricSpec(
        name="debug/mpc_target_step_y",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner target step y (debug only)",
    ),
    MetricSpec(
        name="debug/mpc_support_state",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner support state indicator",
    ),
    MetricSpec(
        name="debug/mpc_step_requested",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Planner step-request indicator",
    ),
    MetricSpec(
        name="reward/arrest_pitch_rate",
        reducer=Reducer.MEAN,
        description="Post-touchdown pitch-rate arrest reward during recovery",
    ),
    MetricSpec(
        name="reward/arrest_capture_error",
        reducer=Reducer.MEAN,
        description="Post-touchdown capture-error arrest reward during recovery",
    ),
    MetricSpec(
        name="reward/post_touchdown_survival",
        reducer=Reducer.MEAN,
        description="Per-step survival reward after first recovery touchdown",
    ),
    MetricSpec(
        name="teacher/active_frac",
        reducer=Reducer.MEAN,
        log_prefix="teacher",
        description="Fraction of steps where teacher is active",
    ),
    MetricSpec(
        name="teacher/active_count",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of active teacher samples used to normalize teacher means",
    ),
    MetricSpec(
        name="teacher/step_required_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed teacher soft step-required signal; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/step_required_hard_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed hard step-required indicator; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/target_step_x_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed teacher target x; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/target_step_y_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed teacher target y; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/swing_left_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed swing-left indicator; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/target_xy_error",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed touchdown-vs-teacher XY error; normalized by teacher/target_xy_error_events",
    ),
    MetricSpec(
        name="teacher/target_xy_error_events",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of teacher XY touchdown events for error normalization",
    ),
    MetricSpec(
        name="teacher/teacher_active_during_clean_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Teacher-active count during clean (no push/recovery) steps; normalized by teacher/clean_step_count",
    ),
    MetricSpec(
        name="teacher/raw_step_required_during_clean_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Raw (pre-disturbance-gating) teacher step-required sum during clean steps; normalized by teacher/clean_step_count",
    ),
    MetricSpec(
        name="teacher/clean_step_count",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of clean (no push/recovery) samples for clean-window normalization",
    ),
    MetricSpec(
        name="teacher/push_step_count",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of push-active samples for push-window normalization",
    ),
    MetricSpec(
        name="teacher/teacher_active_during_push_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Teacher-active count during push-active steps; normalized by teacher/push_step_count",
    ),
    MetricSpec(
        name="teacher/reachable_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed reachable-target indicator; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/target_step_x_min",
        reducer=Reducer.MIN,
        log_prefix="teacher",
        description="Minimum teacher target x over active samples",
    ),
    MetricSpec(
        name="teacher/target_step_x_max",
        reducer=Reducer.MAX,
        log_prefix="teacher",
        description="Maximum teacher target x over active samples",
    ),
    MetricSpec(
        name="teacher/target_step_y_min",
        reducer=Reducer.MIN,
        log_prefix="teacher",
        description="Minimum teacher target y over active samples",
    ),
    MetricSpec(
        name="teacher/target_step_y_max",
        reducer=Reducer.MAX,
        log_prefix="teacher",
        description="Maximum teacher target y over active samples",
    ),
    MetricSpec(
        name="teacher/target_step_x_sq_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed target x squared; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/target_step_y_sq_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed target y squared; normalized by teacher/active_count",
    ),
    MetricSpec(
        name="teacher/swing_foot",
        reducer=Reducer.LAST,
        log_prefix="teacher",
        description="Latest teacher swing-foot suggestion (-1 none, 0 left, 1 right)",
    ),
    MetricSpec(
        name="teacher/target_step_x_std",
        reducer=Reducer.LAST,
        log_prefix="teacher",
        description="Std dev of teacher target step x over active samples",
    ),
    MetricSpec(
        name="teacher/target_step_y_std",
        reducer=Reducer.LAST,
        log_prefix="teacher",
        description="Std dev of teacher target step y over active samples",
    ),
    MetricSpec(
        name="teacher/step_required_event_rate",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed teacher step-required event gates; normalized by recovery/completed",
    ),
    MetricSpec(
        name="teacher/swing_foot_match_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed teacher swing-foot match indicators; normalized by teacher/swing_foot_match_events",
    ),
    MetricSpec(
        name="teacher/swing_foot_match_events",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of first-touchdown events used for teacher/swing_foot_match_frac normalization",
    ),
    MetricSpec(
        name="visible_step_rate_hard",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Hard-push visible-step proxy; normalized by recovery/completed",
    ),
    MetricSpec(
        name="debug/recovery_step_gate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Gate enabling recovery-window step rewards",
    ),
    MetricSpec(
        name="debug/post_touchdown_arrest_gate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Gate enabling post-touchdown arrest rewards",
    ),
    MetricSpec(
        name="debug/disturbed_reward_gate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Gate enabling disturbed-window reward relaxation",
    ),
    MetricSpec(
        name="debug/effective_orientation_weight",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Effective orientation weight after disturbed-window gating",
    ),
    MetricSpec(
        name="debug/effective_height_target_weight",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Effective height-target weight after disturbed-window gating",
    ),
    MetricSpec(
        name="debug/effective_posture_weight",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Effective posture weight after disturbed-window gating",
    ),
    MetricSpec(
        name="recovery/min_height",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed minimum root heights over finalized recoveries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/max_knee_flex",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed maximum knee flexion magnitudes over finalized recoveries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="recovery/first_step_dist_abs",
        reducer=Reducer.SUM,
        log_prefix="recovery",
        description="Summed absolute first-step distances over finalized recoveries; normalized by recovery/completed",
    ),
    MetricSpec(
        name="reward/teacher_recovery_height",
        reducer=Reducer.MEAN,
        description="Whole-body teacher recovery-height reward",
    ),
    MetricSpec(
        name="reward/teacher_com_velocity_reduction",
        reducer=Reducer.MEAN,
        description="Whole-body teacher CoM-velocity reduction reward",
    ),
    MetricSpec(
        name="teacher/whole_body_active_frac",
        reducer=Reducer.MEAN,
        log_prefix="teacher",
        description="Fraction of steps where whole-body teacher is active",
    ),
    MetricSpec(
        name="teacher/whole_body_active_count",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Count of active whole-body teacher samples for normalization",
    ),
    MetricSpec(
        name="teacher/whole_body_active_during_clean_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Whole-body teacher active count during clean steps; normalized by teacher/clean_step_count",
    ),
    MetricSpec(
        name="teacher/whole_body_active_during_push_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Whole-body teacher active count during push-active steps; normalized by teacher/push_step_count",
    ),
    MetricSpec(
        name="teacher/recovery_height_target_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed whole-body recovery-height targets; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="teacher/recovery_height_error",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed whole-body recovery-height errors; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="teacher/recovery_height_in_band_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed in-band indicators for recovery-height target; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="teacher/com_velocity_target_mean",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed whole-body CoM-velocity targets; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="teacher/com_velocity_error",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed whole-body CoM-velocity errors; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="teacher/com_velocity_target_hit_frac",
        reducer=Reducer.SUM,
        log_prefix="teacher",
        description="Summed CoM-velocity target-hit indicators; normalized by teacher/whole_body_active_count",
    ),
    MetricSpec(
        name="debug/m3_swing_pos_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 swing-foot position error in stance-foot frame",
    ),
    MetricSpec(
        name="debug/m3_swing_vel_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 swing-foot velocity error in stance-foot frame",
    ),
    MetricSpec(
        name="debug/m3_foothold_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 touchdown foothold error in stance-foot frame",
    ),
    MetricSpec(
        name="debug/step_length_m",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Forward distance between successive foot touchdowns (meters)",
    ),
    MetricSpec(
        name="debug/m3_pelvis_orientation_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 pelvis orientation error magnitude",
    ),
    MetricSpec(
        name="debug/m3_pelvis_height_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 pelvis height absolute error",
    ),
    MetricSpec(
        name="debug/m3_impact_force",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="M3 maximum contact impact force",
    ),
    MetricSpec(
        name="debug/loc_ref_speed_scale",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Legacy v1 loc-ref speed scale after overspeed/pitch braking (v2 leaves this at 0)",
    ),
    MetricSpec(
        name="debug/loc_ref_phase_scale",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Legacy v1 loc-ref phase-time scale after overspeed/pitch braking (v2 leaves this at 0)",
    ),
    MetricSpec(
        name="debug/loc_ref_overspeed",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Loc-ref overspeed signal max(fwd-cmd-deadband, 0)",
    ),
    MetricSpec(
        name="debug/loc_ref_support_health",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-conditioned progression health (1=healthy, 0=freeze)",
    ),
    MetricSpec(
        name="debug/loc_ref_support_instability",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support instability signal used by progression gate",
    ),
    MetricSpec(
        name="debug/loc_ref_nominal_vs_applied_q_l1",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Mean |applied_target_q - nominal_q_ref| per step",
    ),
    MetricSpec(
        name="debug/loc_ref_applied_q_abs_mean",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Mean absolute applied target-q magnitude",
    ),
    MetricSpec(
        name="debug/loc_ref_nominal_q_abs_mean",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Mean absolute nominal q_ref magnitude (debug copy)",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_x_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal support-gated swing-foot x target in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_foothold_x_raw",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw nominal foothold x before support scaling (meters)",
    ),
    MetricSpec(
        name="debug/loc_ref_nominal_step_length",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw nominal step length from speed*step_time before DCM/scaling (meters)",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_x_actual",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual swing-foot x in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_x_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Swing-foot x tracking error (actual - target) in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_y_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal support-gated swing-foot y target in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_y_actual",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual swing-foot y in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_y_error",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Swing-foot y tracking error (actual - target) in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_left_foot_y_actual",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual left-foot y in heading-local frame",
    ),
    MetricSpec(
        name="debug/loc_ref_right_foot_y_actual",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual right-foot y in heading-local frame",
    ),
    MetricSpec(
        name="debug/loc_ref_support_width_y_actual",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual lateral foot separation |right_y-left_y| in heading-local frame",
    ),
    MetricSpec(
        name="debug/loc_ref_support_width_y_nominal",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Reference foothold lateral magnitude |next_foothold_y| in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_support_width_y_commanded",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Commanded lateral support width |swing_y_target| in stance frame",
    ),
    MetricSpec(
        name="debug/loc_ref_base_support_y",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Base lateral support component (from next foothold y in stance frame)",
    ),
    MetricSpec(
        name="debug/loc_ref_lateral_release_y",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Lateral release component (swing_y_target - base_support_y)",
    ),
    MetricSpec(
        name="debug/loc_ref_pelvis_roll_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal pelvis roll target from locomotion reference",
    ),
    MetricSpec(
        name="debug/loc_ref_hip_roll_left_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal left hip-roll target after locomotion-reference mapping",
    ),
    MetricSpec(
        name="debug/loc_ref_hip_roll_right_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal right hip-roll target after locomotion-reference mapping",
    ),
    MetricSpec(
        name="debug/loc_ref_pelvis_pitch_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-gated nominal pelvis pitch target",
    ),
    MetricSpec(
        name="debug/loc_ref_root_pitch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual root pitch (heading-local Euler y)",
    ),
    MetricSpec(
        name="debug/loc_ref_root_pitch_rate",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Actual root pitch rate (heading-local angvel y)",
    ),
    MetricSpec(
        name="debug/loc_ref_stance_sagittal_target",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Nominal stance-leg sagittal z target",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_x_scale",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-first scale applied to forward swing-x progression",
    ),
    MetricSpec(
        name="debug/loc_ref_pelvis_pitch_scale",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-first scale applied to nominal pelvis pitch",
    ),
    MetricSpec(
        name="debug/loc_ref_support_gate_active",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-first clamp activation flag",
    ),
    MetricSpec(
        name="debug/loc_ref_hybrid_mode_id",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Hybrid nominal reference mode id",
    ),
    MetricSpec(
        name="debug/loc_ref_progression_permission",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Support-conditioned progression permission",
    ),
    MetricSpec(
        name="debug/loc_ref_swing_x_scale_active",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Active forward swing-x scale used by current loc-ref version",
    ),
    MetricSpec(
        name="debug/loc_ref_phase_scale_active",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Active phase-advance scale/permission used by current loc-ref version",
    ),
    # v0.19.5: reward-term parity with experiment tracking
    MetricSpec(
        name="reward/backward_lean",
        reducer=Reducer.MEAN,
        description="Backward-lean penalty when pitch tilts behind neutral",
    ),
    MetricSpec(
        name="reward/negative_velocity",
        reducer=Reducer.MEAN,
        description="Penalty for moving backward under forward command",
    ),
    # =========================================================================
    # v0.20.1: imitation-dominant residual reward family.
    # See training/configs/training_runtime_config.py RewardWeightsConfig
    # for the per-term weights / sigmas, and training/docs/walking_training.md
    # (v0.20.1 Reward §) for the canonical term list.
    # =========================================================================
    MetricSpec(
        name="reward/ref_q_track",
        reducer=Reducer.MEAN,
        description="Joint-target tracking exp(-alpha*sum((q-q_ref)^2))",
    ),
    MetricSpec(
        name="reward/ref_body_quat_track",
        reducer=Reducer.MEAN,
        description="Pelvis orientation tracking exp(-alpha*angle^2)",
    ),
    MetricSpec(
        name="reward/torso_pos_xy",
        reducer=Reducer.MEAN,
        description="Torso XY tracking exp(-alpha*||torso_xy-ref_xy||^2)",
    ),
    MetricSpec(
        name="reward/lin_vel_z",
        reducer=Reducer.MEAN,
        description=(
            "v0.20.2 smoke6 TB-aligned vertical body velocity tracking: "
            "exp(-alpha*(vz - ref_vz)^2); ref from finite-diff of prior pelvis_pos[2]"
        ),
    ),
    MetricSpec(
        name="reward/ang_vel_xy",
        reducer=Reducer.MEAN,
        description=(
            "v0.20.2 smoke6 TB-aligned body roll/pitch rate tracking: "
            "exp(-alpha*(gyro_x^2 + gyro_y^2)); ref is zero (yaw-stationary prior)"
        ),
    ),
    MetricSpec(
        name="reward/ref_contact_match",
        reducer=Reducer.MEAN,
        description=(
            "TB boolean equality count (smoke6 onward): "
            "sum(stance_mask == ref_stance_mask) in {0, 1, 2}.  "
            "Smoke3-5 used a WR-specific Gaussian; switched to TB form."
        ),
    ),
    MetricSpec(
        name="reward/cmd_forward_velocity_track",
        reducer=Reducer.MEAN,
        description="Forward velocity command tracking",
    ),
    # =========================================================================
    # v0.20.1 ToddlerBot-alignment shaping rewards
    # (walking_training.md Appendix A.3).
    # =========================================================================
    MetricSpec(
        name="reward/alive",
        reducer=Reducer.MEAN,
        description="Strict ToddlerBot survival: 0 while alive, -alive_w*dt on done",
    ),
    MetricSpec(
        name="reward/feet_air_time",
        reducer=Reducer.MEAN,
        description="Σ per-foot air time at touchdown (cmd-gated)",
    ),
    MetricSpec(
        name="reward/feet_clearance",
        reducer=Reducer.MEAN,
        description="Σ per-foot peak swing height at touchdown (cmd-gated)",
    ),
    MetricSpec(
        name="reward/feet_distance",
        reducer=Reducer.MEAN,
        description="Lateral foot-spacing band (torso-frame y)",
    ),
    MetricSpec(
        name="reward/torso_pitch_soft",
        reducer=Reducer.MEAN,
        description="Soft pitch band penalty (replaces hard pitch term)",
    ),
    MetricSpec(
        name="reward/torso_roll_soft",
        reducer=Reducer.MEAN,
        description="Soft roll band penalty (replaces hard roll term)",
    ),
    MetricSpec(
        name="ref/q_track_err_rmse",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        topline=True,
        description="RMSE between actual and reference joint positions (rad)",
    ),
    MetricSpec(
        name="ref/body_quat_err_deg",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description="Geodesic orientation error vs reference (deg)",
    ),
    MetricSpec(
        name="ref/feet_pos_err_l2",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description="L2 foot-pos error (root-relative, summed L+R) in m",
    ),
    MetricSpec(
        name="ref/feet_pos_track_raw",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description="Legacy feet-pos tracking score exp(-200*||e_feet||^2), debug-only",
    ),
    MetricSpec(
        name="ref/torso_pos_xy_err_m",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description="Torso XY position error norm vs reference pelvis (m)",
    ),
    MetricSpec(
        name="ref/lin_vel_z_err_m_s",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description=(
            "v0.20.2 smoke6 |vz - ref_vz| (m/s); paired with reward/lin_vel_z"
        ),
    ),
    MetricSpec(
        name="ref/ang_vel_xy_err_rad_s",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        description=(
            "v0.20.2 smoke6 sqrt(gyro_x^2 + gyro_y^2) (rad/s); paired with "
            "reward/ang_vel_xy (ref is zero for yaw-stationary prior)"
        ),
    ),
    MetricSpec(
        name="ref/contact_phase_match",
        reducer=Reducer.MEAN,
        log_prefix="ref",
        topline=True,
        description="Smooth contact-phase match (per-step value of the reward term)",
    ),
    # =========================================================================
    # v0.20.1 G5 anti-exploit hard-gate metrics (walking_training.md v0.20.1
    # §, line 980).  Per-joint residual magnitudes for the four leg joints
    # the spec calls out, plus the realized-vs-commanded forward velocity
    # ratio.  MEAN aggregation approximates the spec's p50 (close for the
    # symmetric residual distribution we expect under PPO exploration).
    # =========================================================================
    MetricSpec(
        name="tracking/residual_hip_pitch_left_abs",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="|residual_delta_q[left_hip_pitch]| (rad)",
    ),
    MetricSpec(
        name="tracking/residual_hip_pitch_right_abs",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="|residual_delta_q[right_hip_pitch]| (rad)",
    ),
    MetricSpec(
        name="tracking/residual_knee_left_abs",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="|residual_delta_q[left_knee_pitch]| (rad)",
    ),
    MetricSpec(
        name="tracking/residual_knee_right_abs",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="|residual_delta_q[right_knee_pitch]| (rad)",
    ),
    MetricSpec(
        name="tracking/forward_velocity_cmd_ratio",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        topline=True,
        description="forward_velocity / max(|cmd_vx|, 1e-3); G5 gate is 0.6..1.5",
    ),
    MetricSpec(
        name="tracking/step_length_touchdown_event_m",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        topline=True,
        description=(
            "Most-recent touchdown step length in m (carries between events). "
            "Rollout MEAN approximates the G4 'touchdown step length mean ≥ 0.03 m' gate."
        ),
    ),
    # Per-foot stride / swing-time diagnostics (v0.20.1-smoke2).  All values
    # are nonzero only on the touchdown step of that foot; aggregation gives
    # rates.  Per-event mean = ``<value>_event / touchdown_rate``.
    MetricSpec(
        name="tracking/touchdown_rate_left",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="Left-foot touchdown events per ctrl step (rollout mean)",
    ),
    MetricSpec(
        name="tracking/touchdown_rate_right",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description="Right-foot touchdown events per ctrl step (rollout mean)",
    ),
    MetricSpec(
        name="tracking/swing_air_time_left_event_s",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description=(
            "Left-foot swing-air-time at touchdown (s), zeroed elsewhere. "
            "Per-event mean = value / touchdown_rate_left."
        ),
    ),
    MetricSpec(
        name="tracking/swing_air_time_right_event_s",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description=(
            "Right-foot swing-air-time at touchdown (s), zeroed elsewhere. "
            "Per-event mean = value / touchdown_rate_right."
        ),
    ),
    MetricSpec(
        name="tracking/step_length_left_event_m",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description=(
            "Left-foot step length at touchdown (m), zeroed elsewhere. "
            "Per-event mean = value / touchdown_rate_left."
        ),
    ),
    MetricSpec(
        name="tracking/step_length_right_event_m",
        reducer=Reducer.MEAN,
        log_prefix="tracking",
        description=(
            "Right-foot step length at touchdown (m), zeroed elsewhere. "
            "Per-event mean = value / touchdown_rate_right."
        ),
    ),
]

# =============================================================================
# Derived constants
# =============================================================================

# Number of metrics in packed vector
NUM_METRICS: int = len(METRIC_SPECS)

# Ordered list of metric names (for iteration)
METRIC_NAMES: List[str] = [spec.name for spec in METRIC_SPECS]

# Name -> index mapping (for fast lookup)
METRIC_INDEX: Dict[str, int] = {spec.name: i for i, spec in enumerate(METRIC_SPECS)}

# Name -> spec mapping
METRIC_SPEC_BY_NAME: Dict[str, MetricSpec] = {spec.name: spec for spec in METRIC_SPECS}

# Topline metrics (for dashboard)
TOPLINE_METRICS: List[str] = [spec.name for spec in METRIC_SPECS if spec.topline]

# Key used to store packed metrics vector in state.metrics dict
# This follows the same pattern as WR_INFO_KEY for info dict
METRICS_VEC_KEY: str = "wr_metrics_vec"


# =============================================================================
# Helper functions
# =============================================================================

def build_metrics_vec(metrics_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Build packed metrics vector from dict.

    Missing metrics default to zeros for backward compatibility with older
    checkpoints/configs that lack newly added metrics.

    Args:
        metrics_dict: Dict mapping metric names to scalar values.

    Returns:
        Packed vector of shape (NUM_METRICS,) or (N, NUM_METRICS) if batched.
    """
    zeros = jnp.zeros((), dtype=jnp.float32)
    values = [metrics_dict.get(name, zeros) for name in METRIC_NAMES]
    return jnp.stack(values, axis=-1)


def unpack_metrics(metrics_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Unpack metrics vector to dict.

    Args:
        metrics_vec: Packed vector of shape (..., NUM_METRICS).

    Returns:
        Dict mapping metric names to values.
    """
    return {name: metrics_vec[..., i] for i, name in enumerate(METRIC_NAMES)}


def aggregate_metrics(
    metrics_vec: jnp.ndarray,
    dones: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """Aggregate metrics over rollout using registered reducers.

    Args:
        metrics_vec: Shape (T, N, NUM_METRICS) - metrics over rollout.
        dones: Shape (T, N) - episode done flags (unused currently).

    Returns:
        Dict mapping metric names to aggregated scalar values.
    """
    result = {}

    for i, spec in enumerate(METRIC_SPECS):
        values = metrics_vec[..., i]  # (T, N)

        if spec.reducer == Reducer.MEAN:
            result[spec.name] = jnp.mean(values)
        elif spec.reducer == Reducer.MAX:
            result[spec.name] = jnp.max(values)
        elif spec.reducer == Reducer.MIN:
            result[spec.name] = jnp.min(values)
        elif spec.reducer == Reducer.SUM:
            result[spec.name] = jnp.sum(values)
        elif spec.reducer == Reducer.LAST:
            result[spec.name] = values[-1].mean()  # Last timestep, mean over envs
        else:
            result[spec.name] = jnp.mean(values)

    completed = result.get("recovery/completed")
    if completed is not None:
        completed_safe = jnp.maximum(completed, 1.0)
        has_completed = completed > 0.0
        for name in (
            "recovery/first_step_latency",
            "recovery/touchdown_count",
            "recovery/support_foot_changes",
            "recovery/post_push_velocity",
            "recovery/no_touchdown_frac",
            "recovery/touchdown_then_fail_frac",
            "recovery/first_liftoff_latency",
            "recovery/first_touchdown_latency",
            "recovery/first_step_dx",
            "recovery/first_step_dy",
            "recovery/first_step_target_err_x",
            "recovery/first_step_target_err_y",
            "recovery/pitch_rate_at_push_end",
            "recovery/pitch_rate_at_touchdown",
            "recovery/pitch_rate_reduction_10t",
            "recovery/capture_error_at_push_end",
            "recovery/capture_error_at_touchdown",
            "recovery/capture_error_reduction_10t",
            "recovery/touchdown_to_term_steps",
            "recovery/visible_step_rate",
            "recovery/min_height",
            "recovery/max_knee_flex",
            "recovery/first_step_dist_abs",
            "teacher/step_required_event_rate",
            "visible_step_rate_hard",
        ):
            if name in result:
                normalized = result[name] / completed_safe
                result[name] = jnp.where(has_completed, normalized, 0.0)

    teacher_active = result.get("teacher/active_count")
    if teacher_active is not None:
        teacher_active_safe = jnp.maximum(teacher_active, 1e-6)
        for name in (
            "teacher/step_required_mean",
            "teacher/step_required_hard_frac",
            "teacher/target_step_x_mean",
            "teacher/target_step_y_mean",
            "teacher/swing_left_frac",
            "teacher/reachable_frac",
            "teacher/target_step_x_sq_mean",
            "teacher/target_step_y_sq_mean",
        ):
            if name in result:
                result[name] = result[name] / teacher_active_safe

        clean_count = result.get("teacher/clean_step_count")
        if clean_count is not None:
            clean_count_safe = jnp.maximum(clean_count, 1e-6)
            if "teacher/teacher_active_during_clean_frac" in result:
                result["teacher/teacher_active_during_clean_frac"] = (
                    result["teacher/teacher_active_during_clean_frac"] / clean_count_safe
                )
            if "teacher/raw_step_required_during_clean_mean" in result:
                result["teacher/raw_step_required_during_clean_mean"] = (
                    result["teacher/raw_step_required_during_clean_mean"] / clean_count_safe
                )
        push_count = result.get("teacher/push_step_count")
        if push_count is not None and "teacher/teacher_active_during_push_frac" in result:
            push_count_safe = jnp.maximum(push_count, 1e-6)
            result["teacher/teacher_active_during_push_frac"] = (
                result["teacher/teacher_active_during_push_frac"] / push_count_safe
            )
            if "teacher/whole_body_active_during_push_frac" in result:
                result["teacher/whole_body_active_during_push_frac"] = (
                    result["teacher/whole_body_active_during_push_frac"] / push_count_safe
                )
        if clean_count is not None and "teacher/whole_body_active_during_clean_frac" in result:
            clean_count_safe = jnp.maximum(clean_count, 1e-6)
            result["teacher/whole_body_active_during_clean_frac"] = (
                result["teacher/whole_body_active_during_clean_frac"] / clean_count_safe
            )

        if "teacher/target_step_x_min" in result:
            result["teacher/target_step_x_min"] = jnp.where(
                teacher_active > 0.0, result["teacher/target_step_x_min"], 0.0
            )
        if "teacher/target_step_x_max" in result:
            result["teacher/target_step_x_max"] = jnp.where(
                teacher_active > 0.0, result["teacher/target_step_x_max"], 0.0
            )
        if "teacher/target_step_y_min" in result:
            result["teacher/target_step_y_min"] = jnp.where(
                teacher_active > 0.0, result["teacher/target_step_y_min"], 0.0
            )
        if "teacher/target_step_y_max" in result:
            result["teacher/target_step_y_max"] = jnp.where(
                teacher_active > 0.0, result["teacher/target_step_y_max"], 0.0
            )

        if "teacher/target_xy_error" in result:
            err_events = jnp.maximum(result.get("teacher/target_xy_error_events", 0.0), 1e-6)
            result["teacher/target_xy_error"] = result["teacher/target_xy_error"] / err_events
        if "teacher/swing_foot_match_frac" in result:
            swing_events = jnp.maximum(result.get("teacher/swing_foot_match_events", 0.0), 1e-6)
            result["teacher/swing_foot_match_frac"] = (
                result["teacher/swing_foot_match_frac"] / swing_events
            )

        if (
            "teacher/target_step_x_sq_mean" in result
            and "teacher/target_step_x_mean" in result
        ):
            x_var = jnp.maximum(
                result["teacher/target_step_x_sq_mean"]
                - result["teacher/target_step_x_mean"] * result["teacher/target_step_x_mean"],
                0.0,
            )
            result["teacher/target_step_x_std"] = jnp.sqrt(x_var)
        if (
            "teacher/target_step_y_sq_mean" in result
            and "teacher/target_step_y_mean" in result
        ):
            y_var = jnp.maximum(
                result["teacher/target_step_y_sq_mean"]
                - result["teacher/target_step_y_mean"] * result["teacher/target_step_y_mean"],
                0.0,
            )
            result["teacher/target_step_y_std"] = jnp.sqrt(y_var)

    whole_body_active = result.get("teacher/whole_body_active_count")
    if whole_body_active is not None:
        whole_body_active_safe = jnp.maximum(whole_body_active, 1e-6)
        for name in (
            "teacher/recovery_height_target_mean",
            "teacher/recovery_height_error",
            "teacher/recovery_height_in_band_frac",
            "teacher/com_velocity_target_mean",
            "teacher/com_velocity_error",
            "teacher/com_velocity_target_hit_frac",
        ):
            if name in result:
                result[name] = result[name] / whole_body_active_safe

    return result


def format_for_wandb(
    aggregated_metrics: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
    """Format aggregated metrics for W&B logging.

    Applies log_prefix from registry and creates topline duplicates.

    Args:
        aggregated_metrics: Dict from aggregate_metrics().

    Returns:
        Dict with prefixed keys ready for wandb.log().
    """
    result = {}

    for name, value in aggregated_metrics.items():
        spec = METRIC_SPEC_BY_NAME.get(name)
        if spec is None:
            # Unknown metric, log with default prefix
            result[f"env/{name}"] = float(value)
            continue

        # Primary logging with prefix
        result[f"{spec.log_prefix}/{name}"] = float(value)

        # Topline duplicate
        if spec.topline:
            result[f"topline/{name}"] = float(value)

    return result


def validate_metrics_vec(metrics_vec: jnp.ndarray) -> bool:
    """Check for NaN values indicating missing metrics.

    Args:
        metrics_vec: Packed metrics vector.

    Returns:
        True if all values are valid (no NaN).
    """
    return not jnp.any(jnp.isnan(metrics_vec))


def get_missing_metrics(metrics_dict: Dict[str, jnp.ndarray]) -> List[str]:
    """Get list of registered metrics missing from dict.

    Args:
        metrics_dict: Dict to check.

    Returns:
        List of missing metric names.
    """
    return [name for name in METRIC_NAMES if name not in metrics_dict]
