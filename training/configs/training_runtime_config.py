"""Runtime configuration for JIT-compiled training.

This module provides a unified config system where:
1. All configs start as mutable dataclasses for easy CLI overrides
2. Call `freeze()` to create JIT-compatible frozen versions
3. Single class definition - no duplicate mutable/frozen classes

Usage:
    from training.configs.training_config import load_training_config

    config = load_training_config("configs/ppo_walking_v0201_smoke.yaml")

    # Modify via mutable configs
    config.ppo.learning_rate = 1e-4
    config.networks.actor.hidden_sizes = [512, 256]

    # Freeze for JIT (call once before training)
    config.freeze()

    # After freeze, modifications raise FrozenInstanceError
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


class FrozenInstanceError(Exception):
    """Raised when attempting to modify a frozen config instance."""

    pass


class Freezable:
    """Mixin that allows dataclasses to be frozen after initialization.

    After calling freeze(), all attribute assignments raise FrozenInstanceError.
    This makes the config JIT-compatible while allowing initial mutations.
    """

    _frozen: bool = False

    def freeze(self) -> None:
        """Freeze this config and all nested Freezable configs.

        After freezing:
        - Attribute assignment raises FrozenInstanceError
        - The config becomes JIT-compatible
        """
        object.__setattr__(self, "_frozen", True)

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Freezable):
                value.freeze()

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "_frozen", False):
            raise FrozenInstanceError(
                f"Cannot modify '{name}' on frozen config. "
                "Call freeze() only after all modifications are complete."
            )
        object.__setattr__(self, name, value)

    @property
    def is_frozen(self) -> bool:
        """Check if this config is frozen."""
        return getattr(self, "_frozen", False)


# =============================================================================
# Environment Config
# =============================================================================
@dataclass
class EnvConfig(Freezable):
    """Environment configuration."""

    assets_root: str = "assets/v2"
    scene_xml_path: str = "assets/v2/scene_flat_terrain.xml"
    robot_config_path: str = "assets/v2/mujoco_robot_config.json"
    mjcf_path: str = "assets/v2/wildrobot.xml"
    model_path: str = "assets/v2/scene_flat_terrain.xml"
    realism_profile_path: Optional[str] = None

    # Timing
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02

    # Episode
    max_episode_steps: int = 500

    # Health / termination
    target_height: float = 0.45
    height_target_two_sided: bool = False
    min_height: float = 0.20
    collapse_height_buffer: float = 0.02
    collapse_height_sigma: float = 0.03
    collapse_vz_gate_band: float = 0.05
    max_height: float = 0.70
    max_pitch: float = 0.8
    max_roll: float = 0.8
    # Relaxed termination: only terminate on height, not orientation.
    # When true, max_pitch/max_roll are ignored for termination (but still
    # available for reward penalties).
    use_relaxed_termination: bool = False

    # Commands
    min_velocity: float = 0.0
    max_velocity: float = 1.0

    # Contacts
    contact_threshold_force: float = 5.0
    contact_scale: float = 10.0
    foot_switch_threshold: float = 2.0

    # Action filtering (alpha=0 disables filtering)
    action_filter_alpha: float = 0.7
    actor_obs_layout_id: str = "wr_obs_v1"
    # v0.19.3: reference-guided locomotion (M2 nominal + PPO residual)
    loc_ref_enabled: bool = False
    # Reference implementation selector.  v0.20.1 deleted v1/v2; the
    # only supported value is "v3_offline_library", which loads the
    # offline ZMP ReferenceLibrary (requires
    # loc_ref_offline_library_path on disk, or omit the path to build
    # one on-the-fly via ZMPWalkGenerator at the configured
    # loc_ref_offline_command_vx).
    loc_ref_version: str = "v3_offline_library"
    # Residual joint delta scale.  Interpretation depends on
    # loc_ref_residual_mode (see below).
    loc_ref_residual_scale: float = 0.18

    # v0.20.1 G1 / §4.1 — residual interpretation mode.
    #   "absolute"  - v3 default; effective delta_q =
    #                 action * loc_ref_residual_scale (the scale value
    #                 is the rad bound directly).  When set, the
    #                 per-joint override map below replaces the scalar
    #                 per the smoke G1 spec (legs ±0.25 rad, others
    #                 ±0.20 rad).
    #   "half_span" - legacy interpretation (action * scale *
    #                 half_span); kept available so a future
    #                 broader-authority policy can opt in without
    #                 redefining the residual contract.
    loc_ref_residual_mode: str = "absolute"
    # v0.20.1 G1 — optional per-joint residual scale override (joint
    # name -> rad bound).  Only consulted when
    # loc_ref_residual_mode == "absolute".  Joints not in the map
    # fall back to the scalar loc_ref_residual_scale.
    loc_ref_residual_scale_per_joint: Dict[str, float] = field(default_factory=dict)

    # v0.20.1 v3_offline_library — offline ReferenceLibrary source.
    # Set loc_ref_offline_library_path to load a saved library from
    # disk; otherwise the env builds an explicit one-bin library for
    # loc_ref_offline_command_vx at init.  This avoids nearest-neighbor
    # snapping when the operating point is not on the generator's
    # historical default 0.05 m/s grid.  loc_ref_offline_command_vx
    # selects the fixed q_ref trajectory for the v0.20.1 smoke; future
    # true multi-command reference conditioning moves to a stacked-arrays
    # representation per the design note's Q3.
    loc_ref_offline_library_path: Optional[str] = None
    loc_ref_offline_command_vx: float = 0.15
    # Walking reference v1 parameters (kept explicit for conservative M3 tuning).
    loc_ref_step_time_s: float = 0.36
    loc_ref_walking_pelvis_height_m: float = 0.40
    loc_ref_nominal_lateral_foot_offset_m: float = 0.09
    loc_ref_min_step_length_m: float = 0.02
    loc_ref_max_step_length_m: float = 0.14
    loc_ref_max_lateral_step_m: float = 0.14
    loc_ref_swing_height_m: float = 0.04
    loc_ref_pelvis_roll_bias_rad: float = 0.03
    loc_ref_pelvis_pitch_gain: float = 0.08
    loc_ref_max_pelvis_pitch_rad: float = 0.08
    # Blend between nominal speed-based foothold and DCM foothold.
    # 0.0 = pure nominal step length, 1.0 = pure DCM foothold.
    loc_ref_dcm_placement_gain: float = 1.0
    # Conservative nominal-path shaping for M3 reference execution.
    loc_ref_swing_target_blend: float = 0.65
    loc_ref_stance_height_blend: float = 0.25
    # Two-height system: baseline knee bend + extra during walking.
    loc_ref_support_margin_m: float = 0.020
    loc_ref_walking_crouch_extra_m: float = 0.045
    loc_ref_max_swing_x_delta_m: float = 0.04
    loc_ref_max_swing_z_delta_m: float = 0.03
    loc_ref_swing_y_to_hip_roll: float = 0.30
    # Overspeed-aware nominal reference shaping (v0.19.3d).
    loc_ref_overspeed_deadband: float = 0.05
    loc_ref_overspeed_brake_gain: float = 1.5
    loc_ref_overspeed_phase_slowdown_gain: float = 2.5
    loc_ref_overspeed_phase_min_scale: float = 0.2
    loc_ref_pitch_brake_start_rad: float = 0.12
    loc_ref_pitch_brake_gain: float = 1.0
    # v0.19.3e support-first local clamps for nominal forward-drive channels.
    loc_ref_swing_x_brake_pitch_start_rad: float = 0.08
    loc_ref_swing_x_brake_overspeed_deadband: float = 0.03
    loc_ref_swing_x_brake_gain: float = 5.0
    loc_ref_swing_x_min_scale: float = 0.05
    loc_ref_pelvis_pitch_brake_gain: float = 6.0
    loc_ref_pelvis_pitch_min_scale: float = 0.0
    # v0.19.3f: support-conditioned progression gate for nominal stepping.
    loc_ref_support_pitch_rate_start_rad_s: float = 0.8
    loc_ref_support_health_gain: float = 4.0
    loc_ref_support_release_phase_start: float = 0.35
    loc_ref_support_foothold_min_scale: float = 0.35
    loc_ref_support_swing_progress_min_scale: float = 0.0
    loc_ref_support_phase_min_scale: float = 0.15
    # Maximum support-conditioned lateral release delta (added to base support y).
    loc_ref_max_lateral_release_m: float = 0.02
    # v0.19.4-prep: walking_ref_v2 hybrid-state thresholds.
    loc_ref_v2_support_open_threshold: float = 0.60
    loc_ref_v2_support_release_threshold: float = 0.30
    loc_ref_v2_touchdown_phase_min: float = 0.55
    loc_ref_v2_capture_hold_s: float = 0.04
    loc_ref_v2_settle_hold_s: float = 0.04
    loc_ref_v2_support_stabilize_max_foothold_scale: float = 0.35
    loc_ref_v2_post_settle_swing_scale: float = 0.15
    loc_ref_v2_startup_ramp_s: float = 1.20
    loc_ref_v2_startup_pelvis_height_offset_m: float = 0.07
    loc_ref_v2_startup_support_open_health: float = 0.25
    loc_ref_v2_startup_handoff_pitch_max_rad: float = 0.30
    loc_ref_v2_startup_handoff_pitch_rate_max_rad_s: float = 1.50
    loc_ref_v2_startup_readiness_knee_err_good_rad: float = 0.10
    loc_ref_v2_startup_readiness_knee_err_bad_rad: float = 0.45
    loc_ref_v2_startup_readiness_ankle_err_good_rad: float = 0.08
    loc_ref_v2_startup_readiness_ankle_err_bad_rad: float = 0.35
    loc_ref_v2_startup_readiness_pitch_good_rad: float = 0.05
    loc_ref_v2_startup_readiness_pitch_bad_rad: float = 0.25
    loc_ref_v2_startup_readiness_pitch_rate_good_rad_s: float = 0.60
    loc_ref_v2_startup_readiness_pitch_rate_bad_rad_s: float = 2.00
    loc_ref_v2_startup_readiness_min_health: float = 0.20
    loc_ref_v2_startup_progress_min_scale: float = 0.20
    loc_ref_v2_startup_realization_lead_alpha: float = 0.25
    loc_ref_v2_startup_handoff_min_readiness: float = 0.10
    loc_ref_v2_startup_handoff_min_pelvis_realization: float = 0.60
    loc_ref_v2_startup_handoff_min_alpha: float = 0.85
    loc_ref_v2_startup_handoff_timeout_s: float = 1.20
    loc_ref_v2_startup_target_rate_design_rad_s: float = 0.5235987756
    loc_ref_v2_startup_target_rate_hard_cap_rad_s: float = 1.7453292520
    loc_ref_v2_startup_route_w1_alpha: float = 0.25
    loc_ref_v2_startup_route_w2_alpha: float = 0.50
    loc_ref_v2_startup_route_w3_alpha: float = 0.75
    loc_ref_v2_startup_route_w1_scale: float = 0.20
    loc_ref_v2_startup_route_w2_scale: float = 0.45
    loc_ref_v2_startup_route_w3_scale: float = 0.75
    loc_ref_v2_startup_route_w1_support_y_scale: float = 0.25
    loc_ref_v2_startup_route_w2_support_y_scale: float = 0.60
    loc_ref_v2_startup_route_w3_support_y_scale: float = 0.85
    loc_ref_v2_startup_route_w1_pelvis_roll_scale: float = 0.45
    loc_ref_v2_startup_route_w2_pelvis_roll_scale: float = 0.75
    loc_ref_v2_startup_route_w3_pelvis_roll_scale: float = 0.90
    loc_ref_v2_startup_route_w1_pelvis_pitch_scale: float = 0.20
    loc_ref_v2_startup_route_w2_pelvis_pitch_scale: float = 0.50
    loc_ref_v2_startup_route_w3_pelvis_pitch_scale: float = 0.80
    loc_ref_v2_startup_route_w1_pelvis_height_scale: float = 0.15
    loc_ref_v2_startup_route_w2_pelvis_height_scale: float = 0.45
    loc_ref_v2_startup_route_w3_pelvis_height_scale: float = 0.80
    loc_ref_v2_startup_route_w2_min_pelvis_realization: float = 0.30
    loc_ref_v2_startup_route_w3_min_pelvis_realization: float = 0.55
    loc_ref_v2_startup_route_w2_pitch_relax: float = 1.25
    loc_ref_v2_startup_route_w2_pitch_rate_relax: float = 1.25
    loc_ref_v2_support_entry_shaping_window_s: float = 0.12
    loc_ref_v2_support_pelvis_height_offset_m: float = 0.00
    # M2.5: start episodes from the support posture B (squat) instead of
    # keyframe A (standing).  Requires loc_ref_enabled=True and
    # loc_ref_version="v2".
    start_from_support_posture: bool = False
    # M3.0: DCM COM trajectory — let the body fall forward over the stance
    # foot following LIPM dynamics during stance phase.
    com_trajectory_enabled: bool = False
    # v0.19.4-C: COM trajectory mode — "linear" (phase ramp) or "lipm" (cosh/sinh).
    com_trajectory_mode: str = "linear"
    # v0.19.4-C: COM trajectory clipping bounds.
    com_trajectory_max_behind_m: float = 0.01
    com_trajectory_max_ahead_m: float = 0.03
    # v0.19.4-C: ankle push-off during terminal stance.
    ankle_pushoff_enabled: bool = False
    ankle_pushoff_phase_start: float = 0.70
    ankle_pushoff_max_rad: float = 0.15
    # Action mapping: "pos_target_rad_v1" (legacy mid-range center) or
    # "pos_target_home_v1" (home-centered, per-joint span)
    action_mapping_id: str = "pos_target_rad_v1"
    clock_stride_period_steps: int = 36
    clock_phase_gate_width: float = 0.20

    # Controller stack selection:
    # - "ppo": existing policy-only / residual-controller path (default)
    # - "mpc_standing": v0.17.3 standing bring-up stub path
    controller_stack: str = "ppo"

    # v0.17.3 standing bring-up scaffold: conservative placeholder gains.
    # These parameters intentionally expose inspectable planner/controller signals
    # without claiming a full MPC implementation.
    mpc_residual_scale: float = 0.25
    mpc_pitch_kp: float = 0.30
    mpc_pitch_kd: float = 0.06
    mpc_roll_kp: float = 0.30
    mpc_roll_kd: float = 0.06
    mpc_height_kp: float = 0.15
    mpc_action_clip: float = 0.35
    mpc_step_trigger_threshold: float = 0.45

    # v0.17.4t: training-time teacher step targets (T0/T1)
    teacher_enabled: bool = False
    teacher_hard_threshold: float = 0.60
    teacher_target_x_min: float = -0.10
    teacher_target_x_max: float = 0.10
    teacher_target_y_left_min: float = 0.02
    teacher_target_y_left_max: float = 0.06
    teacher_target_y_right_min: float = -0.06
    teacher_target_y_right_max: float = -0.02
    whole_body_teacher_enabled: bool = False
    whole_body_teacher_height_target_min: float = 0.39
    whole_body_teacher_height_target_max: float = 0.42
    whole_body_teacher_height_hard_gate: bool = True
    whole_body_teacher_com_vel_target: float = 0.12
    whole_body_teacher_com_vel_active_speed_min: float = 0.10

    # -------------------------------------------------------------------------
    # M2: Base controller + residual policy (optional)
    #
    # When enabled, the environment applies a simple joint-heuristic "base"
    # action each step, then mixes in the policy output as a residual:
    #   action_applied = action_base + residual_scale(need_step) * action_policy
    #
    # The residual authority is gated by the same "need_step" signal used by the
    # stepping-trait rewards (tilt + lateral velocity + pitch rate).
    # -------------------------------------------------------------------------
    base_ctrl_enabled: bool = False

    # Base feedback gains (in policy-action units per rad / rad/s).
    # These are intentionally conservative; they should stabilize uprightness
    # but not overpower learning.
    base_ctrl_pitch_kp: float = 0.25
    base_ctrl_pitch_kd: float = 0.05
    base_ctrl_roll_kp: float = 0.25
    base_ctrl_roll_kd: float = 0.05

    # Per-joint contribution multipliers for the base controller feedback.
    base_ctrl_hip_pitch_gain: float = 1.0
    base_ctrl_ankle_pitch_gain: float = 0.7
    base_ctrl_hip_roll_gain: float = 1.0

    # Clamp applied base deltas (policy-action units) for safety.
    base_ctrl_action_clip: float = 0.35

    # Residual authority gate (0..1) where:
    #   residual_scale = min + need_step**power * (max - min)
    residual_scale_min: float = 0.30
    residual_scale_max: float = 1.00
    residual_gate_power: float = 1.00

    # -------------------------------------------------------------------------
    # M3: Foot-placement + arms base controller + residual RL
    #
    # When fsm_enabled=True, replaces the M2 joint-heuristic action with a
    # full step state-machine (STANCE / SWING / TOUCHDOWN_RECOVER) that freezes
    # touchdown targets and tracks simple swing-foot trajectories.
    # Arm (waist) damping is applied as a secondary stabiliser.
    #
    # FSM phases: STANCE=0, SWING=1, TOUCHDOWN_RECOVER=2
    # -------------------------------------------------------------------------
    fsm_enabled: bool = False

    # Need-to-step gate widths (shared with M1/M2 rewards when m3 active)
    # (gate signals: same as step_need_* in RewardWeightsConfig)

    # FSM thresholds
    fsm_trigger_threshold: float = 0.45       # need_step > this → start trigger hold
    fsm_recover_threshold: float = 0.20       # need_step < this → RECOVER → STANCE
    fsm_trigger_hold_ticks: int = 2           # consecutive ticks to fire SWING
    fsm_touch_hold_ticks: int = 1             # consecutive loaded ticks = touchdown
    fsm_swing_timeout_ticks: int = 12         # max ticks before forced touchdown

    # Foot placement target geometry (heading-local, metres)
    fsm_x_nominal_m: float = 0.0
    fsm_y_nominal_m: float = 0.115            # lateral half-width for each foot
    fsm_k_lat_vel: float = 0.15
    fsm_k_roll: float = 0.10
    fsm_k_pitch: float = 0.05
    fsm_k_fwd_vel: float = 0.05
    fsm_x_step_min_m: float = -0.08
    fsm_x_step_max_m: float = 0.12
    fsm_y_step_inner_m: float = 0.08
    fsm_y_step_outer_m: float = 0.20
    fsm_step_max_delta_m: float = 0.12        # max step from current foot position

    # Swing trajectory
    fsm_swing_height_m: float = 0.04          # peak swing height (m)
    fsm_swing_height_need_step_mult: float = 0.5  # extra height scale with need_step
    fsm_swing_duration_ticks: int = 10        # swing duration in ctrl ticks

    # Swing-foot tracking gains (policy-action units / metre error)
    fsm_swing_x_to_hip_pitch: float = 0.30   # foot x error → hip pitch delta
    fsm_swing_y_to_hip_roll: float = 0.30    # foot y error → hip roll delta
    fsm_swing_z_to_knee: float = 0.40        # foot z error → knee pitch delta
    fsm_swing_z_to_ankle: float = 0.20       # foot z error → ankle pitch delta

    # Residual authority in M3 (replaces M2 residual_scale_min/max for policy action)
    fsm_resid_scale_swing: float = 0.70      # residual authority during SWING
    fsm_resid_scale_stance: float = 0.85     # residual authority during STANCE
    fsm_resid_scale_recover: float = 0.80    # residual authority during TOUCHDOWN_RECOVER

    # Arm / waist strategy
    fsm_arm_enabled: bool = True
    fsm_arm_need_step_threshold: float = 0.35
    fsm_arm_k_roll: float = 0.10
    fsm_arm_k_roll_rate: float = 0.05
    fsm_arm_k_pitch_rate: float = 0.03
    fsm_arm_max_delta_rad: float = 0.25

    # Disturbance pushes (disabled by default)
    push_enabled: bool = False
    push_start_step_min: int = 20
    push_start_step_max: int = 200
    push_duration_steps: int = 10
    push_force_min: float = 0.0
    push_force_max: float = 0.0
    push_body: str = "waist"
    push_bodies: List[str] = field(default_factory=list)

    # IMU observation noise and latency (training-only)
    imu_gyro_noise_std: float = 0.0  # Additive Gaussian noise (rad/s)
    imu_quat_noise_deg: float = 0.0  # Small rotation noise in degrees (std)
    imu_latency_steps: int = 0  # Number of steps delay to apply to IMU readings
    imu_max_latency_steps: int = 4  # Fixed-size history buffer (max latency)

    # v0.17.3b: Domain randomization (disabled by default)
    domain_randomization_enabled: bool = False
    domain_rand_friction_range: List[float] = field(default_factory=lambda: [0.5, 1.0])
    domain_rand_mass_scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    domain_rand_kp_scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    domain_rand_frictionloss_scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    domain_rand_joint_offset_rad: float = 0.03  # U(-val, val) added to qpos0
    # v0.17.3b: Action delay (disabled by default, 0 = no delay)
    action_delay_steps: int = 0  # 1 = apply prev_action instead of current

    # =========================================================================
    # v0.20.1 ToddlerBot-alignment additions (walking_training.md Appendix A)
    # =========================================================================
    # Multi-command resampling.  Mirrors ToddlerBot's CommandsConfig
    # (toddlerbot/locomotion/mjx_config.py:158-164 + walk.gin:1-12).
    # ``cmd_resample_steps == 0`` disables resampling (episode-constant
    # cmd, the v0.19.x / pre-alignment behavior).  ``cmd_resample_steps
    # > 0`` resamples every N control steps from
    # ``[min_velocity, max_velocity]`` with optional zero/turn chances
    # (zero/turn channels are no-ops in the v0.20.1 single-axis cmd path
    # but the fields exist now so v0.20.4 yaw work doesn't have to
    # extend the schema).  Defaults match ToddlerBot
    # (resample_time=3.0s ⇒ 150 ctrl steps at ctrl_dt=0.02).
    cmd_resample_steps: int = 0
    cmd_zero_chance: float = 0.0  # P(resampled cmd == 0)
    cmd_turn_chance: float = 0.0  # P(resampled cmd == turn) — placeholder
    cmd_deadzone: float = 0.0     # |cmd| below this is zeroed (ToddlerBot 0.05)
    # v0.20.1-smoke7: eval-rollout cmd override.  When training enables
    # multi-cmd sampling (cmd_resample_steps > 0, vx range [min, max]),
    # the eval pass would otherwise sample cmds uniformly across the same
    # range and dilute the G4 promotion-horizon gate (E[cmd] over the
    # range may sit below the 0.075 m/s floor).  Setting
    # ``eval_velocity_cmd >= 0`` overrides the eval reset cmd to a fixed
    # value AND suppresses mid-episode resample during eval, so G4 metrics
    # stay interpretable as "behavior at this specific cmd".
    # Sentinel value -1.0 means "no override; eval samples like training".
    eval_velocity_cmd: float = -1.0

    # ToddlerBot soft-pitch / soft-roll behavior gates
    # (toddlerbot/locomotion/mjx_config.py:110-111).  When these
    # thresholds are paired with the corresponding ``torso_pitch_soft``
    # / ``torso_roll_soft`` weights they become the smooth penalty
    # ToddlerBot uses instead of hard pitch/roll termination.
    torso_pitch_soft_min_rad: float = -0.2
    torso_pitch_soft_max_rad: float = 0.2
    torso_roll_soft_min_rad: float = -0.1
    torso_roll_soft_max_rad: float = 0.1

    # Feet geometry gates for the ``feet_distance`` reward
    # (mjx_config.py:108-109).  Lateral foot spacing in body frame; the
    # reward is exp(-100 * |dist - bound|) penalty outside the band.
    min_feet_y_dist: float = 0.07
    max_feet_y_dist: float = 0.13


# =============================================================================
# PPO Config
# =============================================================================
@dataclass
class PPOConfig(Freezable):
    """PPO algorithm configuration."""

    num_envs: int = 1024
    rollout_steps: int = 128
    iterations: int = 1000

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5

    epochs: int = 4
    num_minibatches: int = 32
    max_grad_norm: float = 0.5

    log_interval: int = 10

    target_kl: float = 0.0
    kl_early_stop_multiplier: float = 1.5
    kl_lr_backoff_multiplier: float = 2.0
    kl_lr_backoff_factor: float = 0.5

    lr_schedule_end_factor: float = 1.0
    entropy_schedule_end_factor: float = 1.0
    critic_privileged_enabled: bool = False

    eval: "PPOEvalConfig" = field(default_factory=lambda: PPOEvalConfig())
    rollback: "PPORollbackConfig" = field(default_factory=lambda: PPORollbackConfig())


@dataclass
class PPOEvalConfig(Freezable):
    """Deterministic periodic evaluation settings."""

    enabled: bool = False
    interval: int = 0
    num_envs: int = 0
    num_steps: int = 0
    deterministic: bool = True
    seed_offset: int = 10_000


@dataclass
class PPORollbackConfig(Freezable):
    """Rollback-on-regression settings driven by eval metrics."""

    enabled: bool = False
    patience: int = 2
    success_rate_drop_threshold: float = 0.05
    lr_factor: float = 0.5


# =============================================================================
# Network Configs (Option A: algorithm-agnostic)
# =============================================================================
@dataclass
class ActorNetworkConfig(Freezable):
    """Actor (policy) network configuration."""

    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    log_std_init: float = -1.0
    min_log_std: float = -5.0
    max_log_std: float = 2.0


@dataclass
class CriticNetworkConfig(Freezable):
    """Critic (value) network configuration."""

    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"


@dataclass
class NetworksConfig(Freezable):
    """All network configurations."""

    actor: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    critic: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)


# =============================================================================
# Reward Configs
# =============================================================================
@dataclass
class RewardWeightsConfig(Freezable):
    """Task reward weights (environment-side)."""

    # Primary objectives
    alive: float = 0.0  # v0.17.3a: constant per-step survival reward
    tracking_lin_vel: float = 2.0
    lateral_velocity: float = -0.5
    base_height: float = 0.5
    orientation: float = -0.5
    angular_velocity: float = -0.05
    pitch_rate: float = 0.0
    # v0.19.5b: asymmetric backward-lean penalty (negative pitch only)
    backward_lean: float = 0.0
    # v0.19.5b: penalize backward walking when commanded forward
    negative_velocity: float = 0.0
    height_target: float = 0.0
    height_target_sigma: float = 0.05
    # Disturbed-window gating for standing recovery branches.
    # Defaults preserve historical behavior unless explicitly overridden in YAML.
    disturbed_height_target_scale: float = 1.0
    disturbed_posture_scale: float = 1.0
    disturbed_orientation: float = -0.5
    height_floor: float = 0.0
    height_floor_threshold: float = 0.20
    height_floor_sigma: float = 0.03
    com_velocity_damping: float = 0.0
    com_velocity_damping_scale: float = 1.0
    collapse_height: float = -0.2
    collapse_vz: float = -0.2

    # Effort and safety
    torque: float = -0.001
    saturation: float = -0.1

    # Smoothness
    action_rate: float = -0.01
    joint_velocity: float = -0.001

    # Foot stability
    slip: float = -0.5
    clearance: float = 0.1
    gait_periodicity: float = 0.0

    # Gait shaping (v0.10.6)
    hip_swing: float = 0.0
    knee_swing: float = 0.0
    hip_swing_min: float = 0.0
    knee_swing_min: float = 0.0
    flight_phase_penalty: float = 0.0
    stance_width_penalty: float = 0.0
    stance_width_target: float = 0.10
    stance_width_sigma: float = 0.05

    # Shaping
    forward_velocity_scale: float = 4.0
    velocity_step_gate: float = 0.0

    # v0.10.4: Standing penalty to discourage velocity=0
    velocity_standing_penalty: float = 0.0  # Penalty for standing still
    velocity_standing_threshold: float = 0.2  # Below this = standing still
    velocity_cmd_min: float = 0.2  # Only apply standing penalty if cmd > this

    # v0.14.x: Posture return shaping (encourage returning to default pose after recovery)
    # - Computed from mean squared error between current joint_pos_rad and default_joint_qpos.
    # - Gated by uprightness (|pitch|, |roll|) and healthy.
    posture: float = 0.0
    posture_sigma: float = 0.35  # radians; larger = weaker pull to default
    posture_gate_pitch: float = 0.35  # radians
    posture_gate_roll: float = 0.35  # radians

    # v0.13.10+: Stepping trait shaping (industry-style: step events + foot placement)
    step_event: float = 0.0
    foot_place: float = 0.0
    foot_place_sigma: float = 0.12  # meters
    foot_place_k_lat_vel: float = 0.15  # y target correction per (m/s) lateral vel
    foot_place_k_roll: float = 0.10  # y target correction per rad roll
    foot_place_k_cmd_vel: float = 0.0  # x target correction per (m/s) commanded forward vel
    foot_place_k_pitch: float = 0.05  # x target correction per rad pitch
    foot_place_k_fwd_vel: float = 0.05  # x target correction per (m/s) forward vel
    # v0.15.9: propulsion-per-step shaping
    step_length: float = 0.0
    step_length_target_base: float = 0.03  # metres
    step_length_target_scale: float = 0.25  # metres per (m/s) velocity_cmd
    step_length_sigma: float = 0.04  # metres
    step_progress: float = 0.0
    step_progress_target_scale: float = 1.0
    step_progress_sigma: float = 0.05  # metres
    arrest_pitch_rate: float = 0.0
    arrest_capture_error: float = 0.0
    post_touchdown_survival: float = 0.0
    arrest_pitch_rate_scale: float = 3.0  # rad/s improvement for unit reward
    arrest_capture_error_scale: float = 0.15  # m improvement for unit reward
    # v0.19.3: conservative reference-guided walking reward family (M3)
    m3_pelvis_orientation_tracking: float = 0.0
    m3_pelvis_height_tracking: float = 0.0
    m3_swing_foot_tracking: float = 0.0
    m3_foothold_consistency: float = 0.0
    m3_residual_magnitude: float = 0.0
    m3_excessive_impact: float = 0.0
    m3_pelvis_orientation_sigma: float = 0.20
    m3_pelvis_height_sigma: float = 0.04
    m3_swing_pos_sigma: float = 0.08
    m3_swing_vel_sigma: float = 0.80
    m3_foothold_sigma: float = 0.10
    m3_impact_force_threshold: float = 40.0
    m3_impact_force_sigma: float = 20.0
    dense_progress: float = 0.0
    dense_progress_upright_pitch: float = 0.25  # radians (50% gate around this pitch)
    dense_progress_upright_pitch_rate: float = 0.90  # rad/s (50% gate around this pitch rate)
    dense_progress_upright_sharpness: float = 10.0  # sigmoid sharpness for upright gate
    forward_upright_gate_strength: float = 1.0  # 0=off, 1=full upright gate on forward reward
    cycle_progress: float = 0.0
    cycle_progress_target_scale: float = 1.0
    cycle_progress_sigma: float = 0.08  # metres
    propulsion_gate_step_length_weight: float = 0.5
    propulsion_gate_step_progress_weight: float = 0.5

    # Need-to-step gate (0..1) for stepping-only rewards to avoid marching in place
    step_need_pitch: float = 0.35  # radians
    step_need_roll: float = 0.35  # radians
    step_need_lat_vel: float = 0.30  # m/s
    step_need_pitch_rate: float = 1.00  # rad/s

    # v0.17.4t: teacher-assisted standing rewards (training-time only)
    teacher_target_step_xy: float = 0.45
    teacher_target_step_xy_sigma: float = 0.06
    teacher_step_required: float = 0.0
    teacher_swing_foot: float = 0.0
    teacher_recovery_height: float = 0.0
    teacher_recovery_height_sigma: float = 0.03
    teacher_com_velocity_reduction: float = 0.0
    teacher_com_velocity_reduction_sigma: float = 0.08
    teacher_knee_flex_min: float = 0.0
    teacher_knee_flex_target: float = 0.30
    teacher_knee_flex_sigma: float = 0.10

    # =========================================================================
    # v0.20.1: imitation-dominant residual reward family
    # =========================================================================
    # Term weights (linear; multiplied against the per-term value).  Defaults
    # are 0 so existing v0.19.5c configs are untouched; the v0.20.1 smoke YAML
    # sets them to the DeepMimic-style ratios cited in v0201_env_wiring.md.
    # See training/docs/walking_training.md (v0.20.1 Reward §) for the term
    # list and walking_training.md G3 for the contact-match definition.
    ref_q_track: float = 0.0
    ref_body_quat_track: float = 0.0
    torso_pos_xy: float = 0.0
    # v0.20.2 smoke6: TB-aligned continuous phase signals from walk.gin.
    # ``lin_vel_z`` tracks vertical pelvis velocity vs the prior's bobbing
    # (finite-diff of stored pelvis_pos[2]); ``ang_vel_xy`` tracks body
    # roll/pitch angular velocity vs the yaw-stationary prior's zero.
    # Together with re-enabled ``ref_contact_match``, they close the gap
    # vs ToddlerBot's full phase-signal recipe (Appendix A.3).  Defaults
    # are 0 so non-smoke configs are unaffected.
    lin_vel_z: float = 0.0
    ang_vel_xy: float = 0.0
    ref_contact_match: float = 0.0
    cmd_forward_velocity_track: float = 0.0

    # DeepMimic Gaussian kernel widths (numerator-α convention,
    # r = exp(-α * sum_of_squares)).  Defaults match ToddlerBot
    # (toddlerbot/locomotion/mjx_config.py:104-118): pos_tracking_sigma=200,
    # rot_tracking_sigma=20, motor_pos sigma=1.0; mujoco_playground g1
    # joystick lin_vel tracking_sigma=0.25 (denominator) ⇒ α=4.0.
    ref_q_track_alpha: float = 1.0
    ref_body_quat_alpha: float = 20.0
    torso_pos_xy_alpha: float = 200.0
    # v0.20.2 smoke6: TB lin_vel_tracking_sigma=200 maps to alpha=200 in
    # our numerator-alpha convention; ToddlerBot ang_vel_tracking_sigma=0.5
    # maps to alpha=0.5.  Both apply directly to err^2 (per axis or summed).
    lin_vel_z_alpha: float = 200.0
    ang_vel_xy_alpha: float = 0.5
    # Note: smoke3-5 used a Gaussian contact-match kernel with this sigma;
    # smoke6 onward uses TB's boolean equality count (see env's
    # _compute_reward_terms ``ref/contact_match`` block) which has no sigma.
    # Field retained at the legacy default in case older v0.20.x configs
    # reference it; the v3 env path does not read it.
    ref_contact_match_sigma: float = 0.5
    cmd_forward_velocity_alpha: float = 4.0

    # NOTE: ``slip`` and ``pitch_rate`` already exist above (legacy v0.19.5x
    # fields).  The v0.20.1 v3 env reuses those slots — see the M1 fail-
    # mode decision tree (walking_training.md v0.20.1 §).  The smoke YAML
    # pins both to 0.0 explicitly; the M1 "balance issue" branch turns
    # them on without an env edit.  Raw penalty values are always logged
    # at ``reward/penalty_slip_raw`` / ``reward/penalty_pitch_rate_raw``.

    # =========================================================================
    # ToddlerBot-aligned shaping additions (walking_training.md Appendix A)
    # =========================================================================
    # Each weight defaults to 0 so existing v0.19.x / v0.20.0 configs are
    # unaffected.  Smoke YAML enables these per A.3.  Reward shapes match
    # toddlerbot/locomotion/{mjx_env,walk_env}.py exactly:
    #
    #   feet_air_time      — Σ (air_time_per_foot * first_contact),
    #                        gated on ||cmd|| > 1e-6.  ToddlerBot weight
    #                        500.0 — large because the term scales with
    #                        seconds-airborne (~0.3-0.5 s per step).
    #   feet_clearance     — Σ (peak_air_z * first_contact), cmd-gated.
    #   feet_distance      — exp band penalty around y-distance in
    #                        torso frame; uses min/max_feet_y_dist.
    #   torso_pitch_soft   — exp(-100*|pitch - bound|) inside
    #                        [pitch_min, pitch_max] band.
    #   torso_roll_soft    — same shape, roll axis.
    feet_air_time: float = 0.0
    feet_clearance: float = 0.0
    feet_distance: float = 0.0
    torso_pitch_soft: float = 0.0
    torso_roll_soft: float = 0.0


@dataclass
class RewardCompositionConfig(Freezable):
    """Reward composition (trainer-side)."""

    task_weight: float = 1.0
    task_reward_clip: Optional[Tuple[float, float]] = None


# =============================================================================
# Checkpoint Config
# =============================================================================
@dataclass
class CheckpointConfig(Freezable):
    """Checkpoint configuration."""

    dir: str = "training/checkpoints"
    interval: int = 10


# =============================================================================
# Logging Configs
# =============================================================================
@dataclass
class WandbConfig(Freezable):
    """W&B experiment tracking configuration."""

    enabled: bool = True
    project: str = "wildrobot"
    mode: str = "online"
    tags: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    name: Optional[str] = None
    log_frequency: int = 10
    log_dir: str = "training/wandb"


@dataclass
class VideoConfig(Freezable):
    """Video generation configuration."""

    enabled: bool = False
    num_videos: int = 1
    episode_length: int = 500
    render_every: int = 2
    width: int = 640
    height: int = 480
    fps: int = 25
    upload_to_wandb: bool = True
    output_subdir: str = "videos"


# =============================================================================
# Main Training Config
# =============================================================================
@dataclass
class TrainingConfig(Freezable):
    """Main training configuration.

    Access pattern examples:
        config.networks.actor.hidden_sizes  # [256, 256, 128]
        config.ppo.learning_rate            # 3e-4
        config.reward_weights.tracking_lin_vel  # 2.0
    """

    # Version (from YAML config)
    version: str = ""
    version_name: str = ""

    # Global seed
    seed: int = 42

    # Composed configs
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    networks: NetworksConfig = field(default_factory=NetworksConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    reward: RewardCompositionConfig = field(default_factory=RewardCompositionConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    # Raw config for additional access (not frozen)
    raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)
    config_path: str | None = None

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply nested overrides from a dict (e.g., quick_verify section).

        The overrides dict follows the same schema as the main config.
        Example: {"ppo": {"num_envs": 4}, "wandb": {"enabled": false}}

        Call this before freeze() to apply overrides.
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                target = getattr(self, key)
                if isinstance(value, dict) and isinstance(target, Freezable):
                    # Recursively apply nested overrides
                    self._apply_nested_overrides(target, value)
                else:
                    # Direct value assignment
                    setattr(self, key, value)

    def _apply_nested_overrides(
        self, target: Freezable, overrides: Dict[str, Any]
    ) -> None:
        """Recursively apply overrides to a nested Freezable config."""
        for key, value in overrides.items():
            if hasattr(target, key):
                current = getattr(target, key)
                if isinstance(value, dict) and isinstance(current, Freezable):
                    self._apply_nested_overrides(current, value)
                else:
                    setattr(target, key, value)
