"""Runtime configuration for JIT-compiled training.

This module provides a unified config system where:
1. All configs start as mutable dataclasses for easy CLI overrides
2. Call `freeze()` to create JIT-compatible frozen versions
3. Single class definition - no duplicate mutable/frozen classes

Usage:
    from training.configs.training_config import load_training_config

    config = load_training_config("configs/ppo_walking.yaml")

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
    clock_stride_period_steps: int = 36
    clock_phase_gate_width: float = 0.20

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
# AMP Discriminator Training Config (NOT architecture)
# =============================================================================
@dataclass
class DiscriminatorTrainingConfig(Freezable):
    """Discriminator training hyperparameters (NOT architecture)."""

    learning_rate: float = 8e-5
    batch_size: int = 256
    updates_per_ppo_update: int = 2
    r1_gamma: float = 10.0
    input_noise_std: float = 0.03


@dataclass
class AMPFeatureConfig(Freezable):
    """AMP feature parity controls."""

    use_finite_diff_vel: bool = True
    use_estimated_contacts: bool = True
    mask_waist: bool = False
    enable_mirror_augmentation: bool = False


@dataclass
class AMPTargetsConfig(Freezable):
    """AMP diagnostic targets for alerts/logging."""

    disc_acc_min: float = 0.55
    disc_acc_max: float = 0.80


@dataclass
class AMPConfig(Freezable):
    """AMP (Adversarial Motion Prior) configuration."""

    enabled: bool = False
    dataset_path: Optional[str] = None
    weight: float = 0.0

    # Discriminator training config
    discriminator: DiscriminatorTrainingConfig = field(
        default_factory=DiscriminatorTrainingConfig
    )

    # Feature parity
    feature_config: AMPFeatureConfig = field(default_factory=AMPFeatureConfig)

    # Diagnostic targets
    targets: AMPTargetsConfig = field(default_factory=AMPTargetsConfig)


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
class DiscriminatorNetworkConfig(Freezable):
    """Discriminator network configuration (architecture only)."""

    hidden_sizes: Tuple[int, ...] = (512, 256)
    activation: str = "relu"


@dataclass
class NetworksConfig(Freezable):
    """All network configurations."""

    actor: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    critic: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    discriminator: DiscriminatorNetworkConfig = field(
        default_factory=DiscriminatorNetworkConfig
    )


# =============================================================================
# Reward Configs
# =============================================================================
@dataclass
class RewardWeightsConfig(Freezable):
    """Task reward weights (environment-side)."""

    # Primary objectives
    tracking_lin_vel: float = 2.0
    lateral_velocity: float = -0.5
    base_height: float = 0.5
    orientation: float = -0.5
    angular_velocity: float = -0.05
    pitch_rate: float = 0.0
    height_target: float = 0.0
    height_target_sigma: float = 0.05
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
    dense_progress: float = 0.0
    cycle_progress: float = 0.0
    cycle_progress_target_scale: float = 1.0
    cycle_progress_sigma: float = 0.08  # metres

    # Need-to-step gate (0..1) for stepping-only rewards to avoid marching in place
    step_need_pitch: float = 0.35  # radians
    step_need_roll: float = 0.35  # radians
    step_need_lat_vel: float = 0.30  # m/s
    step_need_pitch_rate: float = 1.00  # rad/s


@dataclass
class RewardCompositionConfig(Freezable):
    """Reward composition (trainer-side)."""

    task_weight: float = 1.0
    amp_weight: float = 0.0
    task_reward_clip: Optional[Tuple[float, float]] = None
    amp_reward_clip: Optional[Tuple[float, float]] = (0.0, 1.0)


# =============================================================================
# Checkpoint Config
# =============================================================================
@dataclass
class CheckpointConfig(Freezable):
    """Checkpoint configuration."""

    dir: str = "training/checkpoints"
    interval: int = 50


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
        config.amp.discriminator.r1_gamma   # 10.0
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
    amp: AMPConfig = field(default_factory=AMPConfig)
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
