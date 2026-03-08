"""M3 Foot-Placement + Arms Base Controller (Step State Machine).

Pure JAX functions implementing the M3 hybrid controller described in
training/docs/step_trait_base_controller_design.md.

Design goals:
- Pure functions (no Python control flow inside JAX-traced paths).
- lax.cond / lax.scan compatible.
- All state held externally (caller stores in WildRobotInfo FSM fields).
- Operates in policy-action space so CAL symmetry correction is preserved.

FSM phases:
    STANCE             = 0  both feet as support; wait for trigger
    SWING              = 1  one foot in swing; tracking frozen target
    TOUCHDOWN_RECOVER  = 2  blend back to neutral after touchdown

Controller API (called each step):
    update_fsm(...)     -> updated FSM fields
    compute_ctrl_base(...)  -> ctrl_base in policy-action space

Only one swing foot is active at a time.  A step target is frozen when the
controller enters SWING and is not recomputed until the next step cycle.

Arms (waist) are secondary stabilizers gated by need_step.
"""

from __future__ import annotations

import jax
import jax.numpy as jp

# ---------------------------------------------------------------------------
# FSM phase constants (int32 values stored in WildRobotInfo.fsm_phase)
# ---------------------------------------------------------------------------

#: Both feet considered support candidates; upright baseline.
STANCE: int = 0

#: One foot in swing; tracking frozen touchdown target.
SWING: int = 1

#: Blend back to neutral after swing-foot touchdown.
TOUCHDOWN_RECOVER: int = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: jp.ndarray) -> jp.ndarray:
    """Smoothstep: 3t²-2t³, maps [0,1] -> [0,1] with C1 continuity."""
    t = jp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_need_step(
    pitch: jp.ndarray,
    roll: jp.ndarray,
    lateral_vel: jp.ndarray,
    pitch_rate: jp.ndarray,
    healthy: jp.ndarray,
    g_pitch: float,
    g_roll: float,
    g_lat: float,
    g_pr: float,
) -> jp.ndarray:
    """Compute 0..1 need-to-step gate from torso state.

    Same logic as WildRobotEnv._compute_need_step; duplicated here so
    step_controller.py is self-contained and usable without the env.

    Args:
        pitch, roll: Torso Euler angles (rad).
        lateral_vel: Lateral velocity in heading-local frame (m/s).
        pitch_rate: Pitch angular rate (rad/s).
        healthy: 1.0 if robot is in a valid height range, else 0.0.
        g_pitch, g_roll, g_lat, g_pr: Gate widths (positive scalars).

    Returns:
        Scalar in [0, 1]; 0 = no stepping needed.
    """
    g_pitch = jp.maximum(jp.asarray(g_pitch, dtype=jp.float32), 1e-6)
    g_roll = jp.maximum(jp.asarray(g_roll, dtype=jp.float32), 1e-6)
    g_lat = jp.maximum(jp.asarray(g_lat, dtype=jp.float32), 1e-6)
    g_pr = jp.maximum(jp.asarray(g_pr, dtype=jp.float32), 1e-6)
    raw = (
        jp.abs(pitch) / g_pitch
        + jp.abs(roll) / g_roll
        + jp.abs(lateral_vel) / g_lat
        + jp.abs(pitch_rate) / g_pr
    ) / 4.0
    return jp.clip(raw, 0.0, 1.0) * healthy


def select_swing_foot(
    roll: jp.ndarray,
    lateral_vel: jp.ndarray,
    prev_left_loaded: jp.ndarray,
    prev_right_loaded: jp.ndarray,
) -> jp.ndarray:
    """Select which foot to swing (0 = left, 1 = right).

    Prefer swinging the less-loaded foot.  Tie-break uses lateral state:
    positive roll / positive lateral velocity → swing right foot (catch step
    toward falling direction).

    Args:
        roll: Torso roll (rad); positive = tilting to the right.
        lateral_vel: Heading-local lateral velocity (m/s); positive = moving right.
        prev_left_loaded: 1.0 if left foot was loaded last step.
        prev_right_loaded: 1.0 if right foot was loaded last step.

    Returns:
        0 = swing left, 1 = swing right.
    """
    # If one foot is clearly more loaded → swing the other.
    right_more_loaded = prev_right_loaded > prev_left_loaded
    # Tie-break bias: positive lateral state → swing right foot to catch.
    bias_right = (roll + lateral_vel) > 0.0
    # more_loaded_right → swing left (0); else → swing right (1)
    from_load = jp.where(right_more_loaded, jp.int32(0), jp.int32(1))
    from_bias = jp.where(bias_right, jp.int32(1), jp.int32(0))
    similar_load = jp.abs(prev_right_loaded - prev_left_loaded) < 0.3
    return jp.where(similar_load, from_bias, from_load)


def compute_step_target(
    lateral_vel: jp.ndarray,
    roll: jp.ndarray,
    forward_vel: jp.ndarray,
    pitch: jp.ndarray,
    swing_foot: jp.ndarray,
    y_nominal_m: float,
    k_lat_vel: float,
    k_roll: float,
    k_fwd_vel: float,
    k_pitch: float,
    x_nominal_m: float,
    x_step_min_m: float,
    x_step_max_m: float,
    y_step_inner_m: float,
    y_step_outer_m: float,
    current_foot_x: jp.ndarray | None = None,
    current_foot_y: jp.ndarray | None = None,
    step_max_delta_m: float = 99.0,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Compute desired touchdown position in heading-local frame.

    Args:
        lateral_vel: Lateral velocity (m/s, positive=right).
        roll: Torso roll (rad, positive=rightward tilt).
        forward_vel: Forward velocity (m/s).
        pitch: Torso pitch (rad, positive=forward tilt).
        swing_foot: 0 = left, 1 = right.
        y_nominal_m: Nominal lateral half-width (positive for left foot).
        k_lat_vel, k_roll, k_fwd_vel, k_pitch: Gain coefficients.
        x_nominal_m: Nominal fore-aft position (usually 0 for standing).
        x_step_min_m, x_step_max_m: Fore-aft reachable bounds.
        y_step_inner_m, y_step_outer_m: Absolute values of inner/outer y bounds.
        current_foot_x, current_foot_y: Current swing-foot position in
            heading-local frame.  When provided together with step_max_delta_m,
            the target is clamped so that it never moves more than
            step_max_delta_m in either axis relative to the current foot pose.
        step_max_delta_m: Per-axis reach limit from the current foot position
            (metres).  Only used when current_foot_x/y are also supplied.

    Returns:
        (x_des, y_des) in heading-local frame.
    """
    base_y = jp.asarray(y_nominal_m, dtype=jp.float32)
    # left foot = +y side, right foot = -y side
    y_side = jp.where(swing_foot == 0, base_y, -base_y)

    y_corr = (
        jp.asarray(k_lat_vel, dtype=jp.float32) * lateral_vel
        + jp.asarray(k_roll, dtype=jp.float32) * roll
    )
    x_corr = (
        jp.asarray(k_fwd_vel, dtype=jp.float32) * forward_vel
        + jp.asarray(k_pitch, dtype=jp.float32) * pitch
    )

    x_des = jp.asarray(x_nominal_m, dtype=jp.float32) + x_corr
    y_des = y_side + y_corr

    x_des = jp.clip(x_des, x_step_min_m, x_step_max_m)

    # Lateral bounds (signed): left foot should stay in [inner, outer] on +y side
    # right foot should stay in [-outer, -inner] on -y side.
    y_inner = jp.where(swing_foot == 0,
                       jp.asarray(y_step_inner_m, dtype=jp.float32),
                       -jp.asarray(y_step_outer_m, dtype=jp.float32))
    y_outer = jp.where(swing_foot == 0,
                       jp.asarray(y_step_outer_m, dtype=jp.float32),
                       -jp.asarray(y_step_inner_m, dtype=jp.float32))
    y_des = jp.clip(y_des, jp.minimum(y_inner, y_outer), jp.maximum(y_inner, y_outer))

    # Per-step reach clamp: keep target within step_max_delta_m of the current
    # foot position in each axis (L∞ ball).  Only applied when current position
    # is provided; the default step_max_delta_m=99.0 is effectively no-op.
    if current_foot_x is not None and current_foot_y is not None:
        delta = jp.asarray(step_max_delta_m, dtype=jp.float32)
        fx = jp.asarray(current_foot_x, dtype=jp.float32)
        fy = jp.asarray(current_foot_y, dtype=jp.float32)
        x_des = jp.clip(x_des, fx - delta, fx + delta)
        y_des = jp.clip(y_des, fy - delta, fy + delta)

    return x_des, y_des


def compute_swing_trajectory(
    phase_ticks: jp.ndarray,
    swing_start_x: jp.ndarray,
    swing_start_y: jp.ndarray,
    frozen_target_x: jp.ndarray,
    frozen_target_y: jp.ndarray,
    swing_height_m: float,
    swing_duration_ticks: int,
    need_step: jp.ndarray,
    swing_height_need_step_mult: float = 0.5,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
    """Compute desired swing-foot position on the current trajectory.

    Uses smoothstep for XY and a half-sine arc for Z.

    Args:
        phase_ticks: Number of ticks spent in SWING phase (0-indexed).
        swing_start_x/y: Heading-local foot position when swing began.
        frozen_target_x/y: Frozen touchdown target.
        swing_height_m: Peak swing height above ground (m).
        swing_duration_ticks: Total swing duration in control ticks.
        need_step: 0..1 gate (scales swing height slightly).
        swing_height_need_step_mult: Extra height multiplier from need_step.

    Returns:
        (foot_x_des, foot_y_des, foot_z_des) in heading-local frame.
    """
    t = jp.clip(
        phase_ticks.astype(jp.float32) / jp.asarray(swing_duration_ticks, dtype=jp.float32),
        0.0,
        1.0,
    )
    t_smooth = _smoothstep(t)

    foot_x_des = swing_start_x + t_smooth * (frozen_target_x - swing_start_x)
    foot_y_des = swing_start_y + t_smooth * (frozen_target_y - swing_start_y)

    h = jp.asarray(swing_height_m, dtype=jp.float32) * (
        1.0 + jp.asarray(swing_height_need_step_mult, dtype=jp.float32) * need_step
    )
    foot_z_des = h * jp.sin(jp.pi * t)

    return foot_x_des, foot_y_des, foot_z_des


def update_fsm(
    # Current FSM state (flat arrays; all shape=(), matching WildRobotInfo fields)
    phase: jp.ndarray,
    swing_foot: jp.ndarray,
    phase_ticks: jp.ndarray,
    frozen_tx: jp.ndarray,
    frozen_ty: jp.ndarray,
    swing_sx: jp.ndarray,
    swing_sy: jp.ndarray,
    touch_hold: jp.ndarray,
    trigger_hold: jp.ndarray,
    # Current signals
    need_step: jp.ndarray,
    loaded_left: jp.ndarray,
    loaded_right: jp.ndarray,
    left_foot_x: jp.ndarray,
    left_foot_y: jp.ndarray,
    right_foot_x: jp.ndarray,
    right_foot_y: jp.ndarray,
    lateral_vel: jp.ndarray,
    roll: jp.ndarray,
    forward_vel: jp.ndarray,
    pitch: jp.ndarray,
    # Config scalars
    trigger_threshold: float,
    recover_threshold: float,
    trigger_hold_ticks: int,
    touch_hold_ticks: int,
    swing_timeout_ticks: int,
    y_nominal_m: float,
    k_lat_vel: float,
    k_roll: float,
    k_fwd_vel: float,
    k_pitch: float,
    x_nominal_m: float,
    x_step_min_m: float,
    x_step_max_m: float,
    y_step_inner_m: float,
    y_step_outer_m: float,
    step_max_delta_m: float = 99.0,
) -> tuple[
    jp.ndarray, jp.ndarray, jp.ndarray,
    jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray,
    jp.ndarray, jp.ndarray,
]:
    """Advance the step FSM by one control tick.

    All arguments are JAX arrays or Python float/int literals (traced at JIT time).
    No Python control flow is used inside this function.

    Args:
        phase, swing_foot, phase_ticks, ...: Current FSM state.
        need_step, loaded_left, loaded_right, ...: Current sensor readings.
        trigger_threshold, ...: Config scalars (traced at JIT time).
        step_max_delta_m: Maximum per-axis reach from the current swing-foot
            position (metres).  Forwarded to compute_step_target; default 99.0
            is effectively no-op for backward compatibility.

    Returns:
        Tuple of updated FSM arrays in the same order as inputs:
        (new_phase, new_swing_foot, new_phase_ticks,
         new_frozen_tx, new_frozen_ty, new_swing_sx, new_swing_sy,
         new_touch_hold, new_trigger_hold)
    """
    # -----------------------------------------------------------------------
    # 1. Compute candidate new swing foot and target (used when entering SWING)
    # -----------------------------------------------------------------------
    cand_swing_foot = select_swing_foot(roll, lateral_vel, loaded_left, loaded_right)
    # Swing start = current position of the chosen swing foot (needed before
    # compute_step_target so the per-step delta clamp can be applied there).
    cand_swing_sx = jp.where(cand_swing_foot == 0, left_foot_x, right_foot_x)
    cand_swing_sy = jp.where(cand_swing_foot == 0, left_foot_y, right_foot_y)
    cand_target_x, cand_target_y = compute_step_target(
        lateral_vel, roll, forward_vel, pitch,
        cand_swing_foot,
        y_nominal_m, k_lat_vel, k_roll, k_fwd_vel, k_pitch,
        x_nominal_m, x_step_min_m, x_step_max_m, y_step_inner_m, y_step_outer_m,
        current_foot_x=cand_swing_sx,
        current_foot_y=cand_swing_sy,
        step_max_delta_m=step_max_delta_m,
    )

    # -----------------------------------------------------------------------
    # 2. Transition condition: STANCE -> SWING
    # -----------------------------------------------------------------------
    in_stance = phase == STANCE
    # Increment trigger_hold when need_step > threshold in STANCE, else reset.
    new_trigger_hold = jp.where(
        in_stance,
        jp.where(need_step > trigger_threshold,
                 jp.minimum(trigger_hold + jp.int32(1),
                             jp.int32(trigger_hold_ticks + 5)),
                 jp.int32(0)),
        trigger_hold,  # preserve outside STANCE
    )
    trigger_fired = in_stance & (new_trigger_hold >= jp.int32(trigger_hold_ticks))

    # -----------------------------------------------------------------------
    # 3. Transition condition: SWING -> TOUCHDOWN_RECOVER (or timeout)
    # -----------------------------------------------------------------------
    in_swing = phase == SWING
    swing_loaded = jp.where(swing_foot == 0, loaded_left, loaded_right)
    # Increment touch_hold when swing foot is loaded in SWING phase.
    new_touch_hold = jp.where(
        in_swing,
        jp.where(swing_loaded,
                 jp.minimum(touch_hold + jp.int32(1),
                             jp.int32(touch_hold_ticks + 5)),
                 jp.int32(0)),
        jp.int32(0),  # reset outside SWING
    )
    swing_timeout = in_swing & (phase_ticks >= jp.int32(swing_timeout_ticks))
    touchdown = in_swing & (
        (new_touch_hold >= jp.int32(touch_hold_ticks)) | swing_timeout
    )

    # -----------------------------------------------------------------------
    # 4. Transition condition: TOUCHDOWN_RECOVER -> STANCE
    # -----------------------------------------------------------------------
    in_recover = phase == TOUCHDOWN_RECOVER
    recover_done = in_recover & (need_step < jp.asarray(recover_threshold, jp.float32))

    # -----------------------------------------------------------------------
    # 5. Compute new phase (using where, fully vectorised)
    # -----------------------------------------------------------------------
    # Priority: touchdown > trigger_fired > recover_done > stay
    new_phase_from_stance = jp.where(trigger_fired, jp.int32(SWING), jp.int32(STANCE))
    new_phase_from_swing = jp.where(touchdown, jp.int32(TOUCHDOWN_RECOVER), jp.int32(SWING))
    new_phase_from_recover = jp.where(recover_done, jp.int32(STANCE), jp.int32(TOUCHDOWN_RECOVER))

    # Select based on current phase
    is_stance = (phase == STANCE).astype(jp.int32)
    is_swing = (phase == SWING).astype(jp.int32)
    is_recover = (phase == TOUCHDOWN_RECOVER).astype(jp.int32)
    new_phase = (
        is_stance * new_phase_from_stance
        + is_swing * new_phase_from_swing
        + is_recover * new_phase_from_recover
    )

    # -----------------------------------------------------------------------
    # 6. Update swing foot (only on STANCE -> SWING transition)
    # -----------------------------------------------------------------------
    entering_swing = (phase != SWING) & (new_phase == SWING)
    new_swing_foot = jp.where(entering_swing, cand_swing_foot, swing_foot)

    # -----------------------------------------------------------------------
    # 7. Freeze targets on STANCE -> SWING
    # -----------------------------------------------------------------------
    new_frozen_tx = jp.where(entering_swing, cand_target_x, frozen_tx)
    new_frozen_ty = jp.where(entering_swing, cand_target_y, frozen_ty)
    new_swing_sx = jp.where(entering_swing, cand_swing_sx, swing_sx)
    new_swing_sy = jp.where(entering_swing, cand_swing_sy, swing_sy)

    # -----------------------------------------------------------------------
    # 8. Phase tick counter (reset on any phase change)
    # -----------------------------------------------------------------------
    phase_changed = new_phase != phase
    new_phase_ticks = jp.where(phase_changed, jp.int32(1), phase_ticks + jp.int32(1))

    return (
        new_phase.astype(jp.int32),
        new_swing_foot.astype(jp.int32),
        new_phase_ticks.astype(jp.int32),
        new_frozen_tx.astype(jp.float32),
        new_frozen_ty.astype(jp.float32),
        new_swing_sx.astype(jp.float32),
        new_swing_sy.astype(jp.float32),
        new_touch_hold.astype(jp.int32),
        new_trigger_hold.astype(jp.int32),
    )


def compute_ctrl_base(
    # FSM state
    phase: jp.ndarray,
    swing_foot: jp.ndarray,
    phase_ticks: jp.ndarray,
    frozen_tx: jp.ndarray,
    frozen_ty: jp.ndarray,
    swing_sx: jp.ndarray,
    swing_sy: jp.ndarray,
    # Signals
    pitch: jp.ndarray,
    roll: jp.ndarray,
    pitch_rate: jp.ndarray,
    roll_rate: jp.ndarray,
    need_step: jp.ndarray,
    # Foot positions (heading-local)
    left_foot_x: jp.ndarray,
    left_foot_y: jp.ndarray,
    right_foot_x: jp.ndarray,
    right_foot_y: jp.ndarray,
    # Default (home) policy-action (same shape as output)
    default_action: jp.ndarray,
    # Per-actuator index mapping (Python ints or -1 for missing)
    idx_left_hip_pitch: int,
    idx_right_hip_pitch: int,
    idx_left_hip_roll: int,
    idx_right_hip_roll: int,
    idx_left_knee: int,
    idx_right_knee: int,
    idx_left_ankle: int,
    idx_right_ankle: int,
    idx_waist: int,
    # Base stance feedback gains (M2-style uprightness, applied on stance leg)
    base_pitch_kp: float,
    base_pitch_kd: float,
    base_roll_kp: float,
    base_roll_kd: float,
    hip_pitch_gain: float,
    ankle_pitch_gain: float,
    hip_roll_gain: float,
    base_action_clip: float,
    # M3 swing-tracking gains (foot error → joint delta, in policy-action units)
    swing_x_to_hip_pitch: float,
    swing_y_to_hip_roll: float,
    swing_z_to_knee: float,
    swing_z_to_ankle: float,
    # Swing trajectory config
    swing_height_m: float,
    swing_duration_ticks: int,
    swing_height_need_step_mult: float,
    # Arm (waist) momentum strategy
    arm_enabled: bool,
    arm_need_step_threshold: float,
    arm_k_roll: float,
    arm_k_roll_rate: float,
    arm_k_pitch_rate: float,
    arm_max_delta_rad: float,
) -> jp.ndarray:
    """Compute a stabilising base action in policy-action space.

    Combines:
    1. Stance feedback (uprightness correction on hips/ankles - same as M2).
    2. Swing-foot trajectory tracking (heading-local foot pos error → joints).
    3. Arm/waist momentum strategy (secondary roll/pitch-rate damping).

    All computations are in policy-action space [-1, 1] (before residual mix).

    Args:
        phase: FSM phase (STANCE=0, SWING=1, TOUCHDOWN_RECOVER=2).
        swing_foot: 0=left swing, 1=right swing.
        phase_ticks: Ticks in current phase.
        frozen_tx, frozen_ty: Frozen touchdown targets (heading-local).
        swing_sx, swing_sy: Swing start positions (heading-local).
        pitch, roll: Torso Euler angles (rad).
        pitch_rate, roll_rate: Angular rates (rad/s).
        need_step: 0..1 gate.
        left_foot_x/y, right_foot_x/y: Foot positions (heading-local, m).
        default_action: Home-pose action (baseline for deltas, shape=(n_act,)).
        idx_*: Integer actuator indices (-1 = joint not present in this robot).
        base_*: M2-style stance feedback gains.
        swing_*: Swing-tracking gains and trajectory config.
        arm_*: Arm/waist policy config.

    Returns:
        ctrl_base of shape (n_act,) in policy-action space.
    """
    action = default_action

    # -----------------------------------------------------------------------
    # Helper: add delta to a specific actuator (no-op for idx==-1)
    # -----------------------------------------------------------------------
    def _add(action: jp.ndarray, idx: int, delta: jp.ndarray) -> jp.ndarray:
        if idx < 0:
            return action
        return action.at[idx].add(delta)

    # -----------------------------------------------------------------------
    # 1. Uprightness feedback (applied to stance/both legs; same as M2)
    # -----------------------------------------------------------------------
    clip_val = jp.asarray(base_action_clip, dtype=jp.float32)

    pitch_fb = -(
        jp.asarray(base_pitch_kp, dtype=jp.float32) * pitch
        + jp.asarray(base_pitch_kd, dtype=jp.float32) * pitch_rate
    )
    roll_fb = -(
        jp.asarray(base_roll_kp, dtype=jp.float32) * roll
        + jp.asarray(base_roll_kd, dtype=jp.float32) * roll_rate
    )
    pitch_fb = jp.clip(pitch_fb, -clip_val, clip_val)
    roll_fb = jp.clip(roll_fb, -clip_val, clip_val)

    hp_g = jp.asarray(hip_pitch_gain, dtype=jp.float32)
    ap_g = jp.asarray(ankle_pitch_gain, dtype=jp.float32)
    hr_g = jp.asarray(hip_roll_gain, dtype=jp.float32)

    # Apply to both hips/ankles (stance feedback; gated off for swing joint
    # below if swing-tracking is active)
    action = _add(action, idx_left_hip_pitch, hp_g * pitch_fb)
    action = _add(action, idx_right_hip_pitch, hp_g * pitch_fb)
    action = _add(action, idx_left_ankle, ap_g * pitch_fb)
    action = _add(action, idx_right_ankle, ap_g * pitch_fb)
    action = _add(action, idx_left_hip_roll, hr_g * roll_fb)
    action = _add(action, idx_right_hip_roll, hr_g * roll_fb)

    # -----------------------------------------------------------------------
    # 2. Swing-foot trajectory tracking
    # -----------------------------------------------------------------------
    in_swing = (phase == SWING).astype(jp.float32)

    foot_x_des, foot_y_des, foot_z_des = compute_swing_trajectory(
        phase_ticks,
        swing_sx, swing_sy,
        frozen_tx, frozen_ty,
        swing_height_m, swing_duration_ticks,
        need_step, swing_height_need_step_mult,
    )

    # Foot tracking errors (desired - current)
    left_foot_z = jp.zeros(())  # approximate: feet near ground
    right_foot_z = jp.zeros(())

    err_x = jp.where(swing_foot == 0,
                     foot_x_des - left_foot_x,
                     foot_x_des - right_foot_x)
    err_y = jp.where(swing_foot == 0,
                     foot_y_des - left_foot_y,
                     foot_y_des - right_foot_y)
    err_z = jp.where(swing_foot == 0,
                     foot_z_des - left_foot_z,
                     foot_z_des - right_foot_z)

    # Map errors to joint deltas (in policy-action space)
    kx = jp.asarray(swing_x_to_hip_pitch, dtype=jp.float32)
    ky = jp.asarray(swing_y_to_hip_roll, dtype=jp.float32)
    kz_knee = jp.asarray(swing_z_to_knee, dtype=jp.float32)
    kz_ank = jp.asarray(swing_z_to_ankle, dtype=jp.float32)

    swing_hip_pitch_delta = in_swing * kx * err_x
    swing_hip_roll_delta = in_swing * ky * err_y
    swing_knee_delta = in_swing * (-kz_knee * err_z)  # more bend = higher foot
    swing_ankle_delta = in_swing * kz_ank * err_z      # compensate ankle

    # Apply to swing-side joints only (opposite = stance, already handled above)
    left_gate = jp.where(swing_foot == 0, in_swing, 0.0)
    right_gate = jp.where(swing_foot == 1, in_swing, 0.0)

    action = _add(action, idx_left_hip_pitch, left_gate * swing_hip_pitch_delta)
    action = _add(action, idx_right_hip_pitch, right_gate * swing_hip_pitch_delta)
    action = _add(action, idx_left_hip_roll, left_gate * swing_hip_roll_delta)
    action = _add(action, idx_right_hip_roll, right_gate * swing_hip_roll_delta)
    action = _add(action, idx_left_knee, left_gate * swing_knee_delta)
    action = _add(action, idx_right_knee, right_gate * swing_knee_delta)
    action = _add(action, idx_left_ankle, left_gate * swing_ankle_delta)
    action = _add(action, idx_right_ankle, right_gate * swing_ankle_delta)

    # -----------------------------------------------------------------------
    # 3. Arm / waist momentum strategy (secondary stabilizer)
    # -----------------------------------------------------------------------
    if arm_enabled and idx_waist >= 0:
        arm_gate = (need_step > jp.asarray(arm_need_step_threshold, jp.float32)).astype(jp.float32)
        arm_delta_raw = (
            jp.asarray(arm_k_roll, dtype=jp.float32) * roll
            + jp.asarray(arm_k_roll_rate, dtype=jp.float32) * roll_rate
            + jp.asarray(arm_k_pitch_rate, dtype=jp.float32) * pitch_rate
        )
        arm_delta = jp.clip(
            arm_delta_raw * arm_gate,
            -jp.asarray(arm_max_delta_rad, dtype=jp.float32),
            jp.asarray(arm_max_delta_rad, dtype=jp.float32),
        )
        action = _add(action, idx_waist, arm_delta)

    return jp.clip(action, -1.0, 1.0).astype(jp.float32)
