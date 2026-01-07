# SMPLX Body Model Skeleton

SMPLX has 22 body joints (excluding hands/face). Here's the kinematic tree:

## SMPLX Joint Hierarchy

```mermaid
graph TD
    subgraph "SMPLX Body Model (22 Joints)"

        %% Root
        PELVIS["<b>pelvis</b><br/>(0) ROOT"]

        %% Spine chain
        PELVIS --> SPINE1["spine1<br/>(3)"]
        SPINE1 --> SPINE2["spine2<br/>(6)"]
        SPINE2 --> SPINE3["spine3<br/>(9)"]
        SPINE3 --> NECK["neck<br/>(12)"]
        NECK --> HEAD["head<br/>(15)"]

        %% Left leg
        PELVIS --> L_HIP["<b>left_hip</b><br/>(1)"]
        L_HIP --> L_KNEE["<b>left_knee</b><br/>(4)"]
        L_KNEE --> L_ANKLE["<b>left_ankle</b><br/>(7)"]
        L_ANKLE --> L_FOOT["<b>left_foot</b><br/>(10)"]

        %% Right leg
        PELVIS --> R_HIP["<b>right_hip</b><br/>(2)"]
        R_HIP --> R_KNEE["<b>right_knee</b><br/>(5)"]
        R_KNEE --> R_ANKLE["<b>right_ankle</b><br/>(8)"]
        R_ANKLE --> R_FOOT["<b>right_foot</b><br/>(11)"]

        %% Left arm
        SPINE3 --> L_COLLAR["left_collar<br/>(13)"]
        L_COLLAR --> L_SHOULDER["left_shoulder<br/>(16)"]
        L_SHOULDER --> L_ELBOW["left_elbow<br/>(18)"]
        L_ELBOW --> L_WRIST["left_wrist<br/>(20)"]

        %% Right arm
        SPINE3 --> R_COLLAR["right_collar<br/>(14)"]
        R_COLLAR --> R_SHOULDER["right_shoulder<br/>(17)"]
        R_SHOULDER --> R_ELBOW["right_elbow<br/>(19)"]
        R_ELBOW --> R_WRIST["right_wrist<br/>(21)"]
    end

    %% Styling
    style PELVIS fill:#ff6b6b,stroke:#333,stroke-width:3px
    style L_HIP fill:#4ecdc4,stroke:#333,stroke-width:2px
    style L_KNEE fill:#4ecdc4,stroke:#333,stroke-width:2px
    style L_ANKLE fill:#4ecdc4,stroke:#333,stroke-width:2px
    style L_FOOT fill:#4ecdc4,stroke:#333,stroke-width:2px
    style R_HIP fill:#45b7d1,stroke:#333,stroke-width:2px
    style R_KNEE fill:#45b7d1,stroke:#333,stroke-width:2px
    style R_ANKLE fill:#45b7d1,stroke:#333,stroke-width:2px
    style R_FOOT fill:#45b7d1,stroke:#333,stroke-width:2px
```

## SMPLX → WildRobot Mapping

Only the **leg joints** are used for WildRobot retargeting:

```
SMPLX Joint          WildRobot Joint         Notes
─────────────────────────────────────────────────────────────
pelvis          →    root (floating base)    Position + Orientation

left_hip        →    left_hip_pitch          3-DOF → 2-DOF
                     left_hip_roll           (yaw discarded)

left_knee       →    left_knee_pitch         1-DOF

left_ankle      →    left_ankle_pitch        3-DOF → 1-DOF
left_foot                                    (roll/yaw discarded)

right_hip       →    right_hip_pitch         3-DOF → 2-DOF
                     right_hip_roll          (yaw discarded)

right_knee      →    right_knee_pitch        1-DOF

right_ankle     →    right_ankle_pitch       3-DOF → 1-DOF
right_foot                                   (roll/yaw discarded)
```

## Visual Representation

```
                    HEAD (15)
                      │
                    NECK (12)
                      │
    L_WRIST(20)─L_ELBOW(18)─L_SHOULDER(16)─L_COLLAR(13)
                      │                         │
                   SPINE3 (9)───────────────R_COLLAR(14)─R_SHOULDER(17)─R_ELBOW(19)─R_WRIST(21)
                      │
                   SPINE2 (6)
                      │
                   SPINE1 (3)
                      │
                ┌─────┴─────┐
                │  PELVIS   │  ← ROOT (0)
                │    (0)    │
                └─────┬─────┘
               ┌──────┴──────┐
               │             │
          L_HIP (1)     R_HIP (2)
               │             │
          L_KNEE (4)    R_KNEE (5)
               │             │
         L_ANKLE (7)   R_ANKLE (8)
               │             │
          L_FOOT (10)  R_FOOT (11)
               │             │
              ─┴─           ─┴─
            (ground)      (ground)
```

## Key Points for AMP Training

1. **Pelvis is ROOT**: All body motion is relative to pelvis
   - Pelvis position → `root_pos` (3D)
   - Pelvis orientation → `root_rot` (quaternion)
   - Pelvis has NO dedicated joint angle (it's the floating base)

2. **No waist_yaw in SMPLX**: Human torso rotation comes from:
   - Pelvis rotation (main heading)
   - Distributed spine twist (spine1-3)
   - WildRobot's `waist_yaw` was removed since SMPLX can't provide it

3. **Leg joints used**: Only 8 DOFs for locomotion
   - 4 joints per leg (hip_pitch, hip_roll, knee_pitch, ankle_pitch)
   - Arms/spine ignored for walking

4. **Scale factor 0.55**: WildRobot is ~55% of human height (1.8m)

## SMPLX Joint Index Reference

| Index | Joint Name | Parent | Used by WildRobot |
|-------|------------|--------|-------------------|
| 0 | pelvis | - | ✅ (root) |
| 1 | left_hip | pelvis | ✅ |
| 2 | right_hip | pelvis | ✅ |
| 3 | spine1 | pelvis | ❌ |
| 4 | left_knee | left_hip | ✅ |
| 5 | right_knee | right_hip | ✅ |
| 6 | spine2 | spine1 | ❌ |
| 7 | left_ankle | left_knee | ✅ |
| 8 | right_ankle | right_knee | ✅ |
| 9 | spine3 | spine2 | ❌ |
| 10 | left_foot | left_ankle | ✅ |
| 11 | right_foot | right_ankle | ✅ |
| 12 | neck | spine3 | ❌ |
| 13 | left_collar | spine3 | ❌ |
| 14 | right_collar | spine3 | ❌ |
| 15 | head | neck | ❌ |
| 16 | left_shoulder | left_collar | ❌ |
| 17 | right_shoulder | right_collar | ❌ |
| 18 | left_elbow | left_shoulder | ❌ |
| 19 | right_elbow | right_shoulder | ❌ |
| 20 | left_wrist | left_elbow | ❌ |
| 21 | right_wrist | right_elbow | ❌ |
