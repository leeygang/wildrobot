# WildRobot v2 — Model Issues to Fix in Onshape

**Last updated:** 2026-04-26
**Source audit:** Phase 12C closed-loop diagnostic (commits `cc5e8d3` / `381ab3c` / `d3a6989`)
**Onshape document:** [WildRobot](https://cad.onshape.com/documents/ba106cac193c0e2253e1958f/w/3b5db3932a544d9f939ac6a6/e/5c6acb3ff696b03bdf018f3b)

This document records issues discovered in the WildRobot v2 MJCF model that
**originate in the Onshape CAD source** (not in our `onshape-to-robot`
export pipeline or our `assets/post_process.py`). All issues survive the
export-pipeline unchanged and end up in `assets/v2/wildrobot.xml`.

The bugs were surfaced by running closed-loop replay of the ZMP walking
prior under MuJoCo (Phase 10 of `training/docs/reference_architecture_comparison.md`).
The symptom is `left_hip_roll` joint hitting its inner limit (+0.175 rad)
during walking while the symmetric `right_hip_roll` stays well within
range — caused by a small but real left/right mass-moment imbalance from
the model asymmetries.

## TL;DR — what to fix in Onshape

| # | Issue | Severity | Onshape change required |
|---|---|---|---|
| 1 | Right-leg upper-leg LINK is named `knee` instead of mirroring left's `upper_leg` | **HIGH** (load-bearing for closed-loop) | Rename the right-side sub-assembly to `upper_leg_2` (or `right_upper_leg`) so the export produces a proper mirror name |
| 2 | LINK frame positions for left and right upper legs differ by ~0.8 mm | **HIGH** (root cause of `left_hip_roll` saturation) | Add a CAD mirror constraint between left and right upper-leg sub-assembly coordinate frames |
| 3 | Arm chain has the inverse `_2` suffix pattern from legs (right arm un-suffixed, left arm `_2`) | MEDIUM (cosmetic + maintenance hazard) | Standardize: rename arm sub-assemblies so left = `left_X` and right = `right_X` (or both consistent with the leg convention) |
| 4 | Sub-mm asymmetries in body-inertia COM positions throughout the model | LOW (small effect) | Apply CAD mirror constraints to all body-pair sub-assemblies, not just leg/arm |

## Issue 1 — Right-leg LINK named `knee` instead of `upper_leg_2`

### What we see in the export

In `assets/v2/onshape_export/wildrobot.xml` (the raw `onshape-to-robot`
output, before any post-processing):

```xml
<!-- LEFT side -->
<!-- Link upper_leg -->
<body name="upper_leg" pos="-0.0269 -0.000166429 -0.0365475" ...>
  <joint name="left_hip_roll" .../>
  <!-- Part upper_leg -->
  <!-- Part upper_leg_servo_connect -->
  <!-- Part knee_ball_left -->
  <!-- Part knee_ball_mid -->
  <!-- Part knee_ball_right -->
  ...
</body>

<!-- RIGHT side -->
<!-- Link knee -->                                    ← LINK named `knee` (BUG)
<body name="knee"      pos="0.0261  0.000169775 -0.0364583" ...>
  <joint name="right_hip_roll" .../>
  <!-- Part upper_leg_2 -->                           ← contains upper_leg_2 part
  <!-- Part upper_leg_servo_connect_2 -->
  <!-- Part knee_ball_left_2 -->
  <!-- Part knee_ball_mid_2 -->
  <!-- Part knee_ball_right_2 -->
  ...
</body>
```

### Why this is wrong

- The body has the same kinematic role on both sides: it sits between
  `*_hip_roll` and `*_knee_pitch` joints; physically it IS the upper leg
- The PARTs inside both LINKs are equivalent (mirrored as `X` / `X_2`):
  `upper_leg`, `upper_leg_servo_connect`, `knee_ball_left/mid/right`,
  `knee_ball_servo_connect_*`
- LEFT side correctly names the LINK `upper_leg` (matching its primary
  PART)
- RIGHT side names the LINK `knee` — which is **not the name of any PART
  inside it**. The closest is `knee_ball_*` (the bearings at the knee
  end), but there is no standalone `knee` PART
- The CAD assembly tree on the right side groups the upper-leg parts
  under a sub-assembly named `knee`, while the left side groups them
  under a sub-assembly named `upper_leg`

### What to fix in Onshape

In the Onshape feature tree under `right_hip`:
- Find the sub-assembly currently named `knee`
- Verify its contents: should include `upper_leg_2`,
  `upper_leg_servo_connect_2`, `knee_ball_left_2`, `knee_ball_mid_2`,
  `knee_ball_right_2`, `icm45686_2`, `htd_45h_1circle_6`,
  `knee_ball_servo_connect_*_2`, plus a kinematic mate to `right_hip`
- Rename the sub-assembly to **`upper_leg_2`** (matching CAD `_2` mirror
  convention used elsewhere in the right side)
  - Or to `right_upper_leg` if you also want to clean up the broader
    `_2`-suffix convention (see Issue 3)

### How to verify after fix

After re-exporting (`./assets/update_xml.sh --version v2 --export-onshape`):
- `assets/v2/onshape_export/wildrobot.xml` should contain
  `<!-- Link upper_leg_2 -->` (or `<!-- Link right_upper_leg -->`)
  followed by `<body name="upper_leg_2" .../>`
- The `<!-- Link knee -->` comment should not appear

## Issue 2 — Upper-leg LINK frame positions are not exact mirrors

### What we see in the export

```xml
<!-- LEFT body position -->
<body name="upper_leg" pos="-0.0269   -0.000166429 -0.0365475" .../>

<!-- RIGHT body position (current; in mirror should be +0.0269 -0.000166429 -0.0365475) -->
<body name="knee"      pos=" 0.0261    0.000169775 -0.0364583" .../>
```

Component-by-component asymmetry:

| Component | Left value | Expected right (mirror of left) | Actual right value | Asymmetry |
|---|---|---|---|---|
| pos x | −0.0269 | +0.0269 | **+0.0261** | **0.8 mm** |
| pos y | −0.000166429 | −0.000166429 | **+0.000169775** (sign flipped + magnitude differs) | sign + ~3 µm |
| pos z | −0.0365475 | −0.0365475 | **−0.0364583** | 89 µm |

### Why this is wrong

A symmetric robot model should have the right-leg LINK frame at the
exact mirror of the left-leg LINK frame relative to the body centerline.
The 0.8 mm x-axis asymmetry shifts the right hip-roll joint axis 0.8 mm
forward (in body frame) compared to the mirror position. This produces:

- A small but consistent rightward bias in the standing equilibrium
  pose (whole-body COM ends up 0.7 mm right of support polygon center)
- Asymmetric hip_roll joint loading during walking (LEFT hip bears
  more lateral support load → saturates at +0.175 rad inner limit)
- The standing keyframes (`home`, `walk_start`) encode this asymmetry
  in their hip_roll q values: both legs slightly negative instead of
  mirror-symmetric

The asymmetry is small in absolute terms (sub-mm) but is the
**load-bearing root cause** of the `left_hip_roll` saturation observed
in closed-loop replay (Phase 10 corrected diagnostic, commit
`381ab3c`). At the operating dynamics scale, a 0.8 mm geometric bias
produces a 0.18 rad joint deviation that exceeds the position
actuator's tracking authority.

### What to fix in Onshape

The two upper-leg sub-assemblies need to have their CAD coordinate
frames constrained as exact mirrors. Possible Onshape approaches:

**Option A — mirror constraint (recommended):**
1. In the WildRobot top-level assembly, find the right upper-leg
   sub-assembly (currently named `knee`, see Issue 1)
2. Replace the right sub-assembly's mate references with a **CAD mirror
   operation** that takes the left `upper_leg` sub-assembly as the
   source and mirrors it across the body's YZ plane (sagittal plane)
3. This guarantees future edits to the left side automatically
   propagate as a mirrored update to the right side

**Option B — manual frame realignment:**
1. Find the right upper-leg sub-assembly's coordinate-frame definition
2. Set its origin position to be the exact mirror of the left's frame:
   - Left frame at `(−0.0269, −0.000166429, −0.0365475)` relative to
     `left_hip`
   - Right frame should be at `(+0.0269, −0.000166429, −0.0365475)`
     relative to `right_hip` (x-mirrored, y/z preserved)
3. Verify by exporting and checking the position values

Option A is more robust because it uses CAD's mirror-symmetry as a
constraint; future edits stay symmetric without manual re-checking.

### How to verify after fix

After re-exporting:
- Right body's `pos` attribute should be **`0.0269 -0.000166429 -0.0365475`**
  (exact mirror of left's `-0.0269 -0.000166429 -0.0365475`)
- Re-derive the `home` and `walk_start` keyframes (they encode the
  current biased equilibrium and need to be re-settled against the
  symmetric model)
- Re-run `tools/phase10_diagnostic.py --vx 0.15` and verify
  `left_hip_roll` peak saturation drops from 0.250-0.333 to ≤ 0.05

## Issue 3 — Arm chain has inverse `_2` suffix convention from legs

### What we see in the export

```
LINK names in raw export:
  Legs:  upper_leg (LEFT) /  knee (RIGHT)
         lower_leg (LEFT) /  lower_leg_2 (RIGHT)
         foot (LEFT)      /  foot_2 (RIGHT)
  Arms:  shoulder_2 (LEFT) / shoulder (RIGHT)
         upper_arm_2 (LEFT)/ upper_arm (RIGHT)
         fore_arm_2 (LEFT) / fore_arm (RIGHT)
         palm_2 (LEFT)     / palm (RIGHT)
         finger_2 (LEFT)   / finger (RIGHT)
```

### Why this is wrong

The `_2` suffix indicates the auto-mirrored side from CAD. In the leg
chain, the LEFT side has the canonical (un-suffixed) name and the RIGHT
side has `_2`. In the arm chain, **the convention is INVERTED**: the
RIGHT side has the canonical name and the LEFT side has `_2`.

This means the CAD assembly was built with:
- LEGS authored on the LEFT side first, then mirrored to RIGHT
- ARMS authored on the RIGHT side first, then mirrored to LEFT

Two opposite mirroring directions in the same model. This is:
- Confusing for future maintainers (which side is "canonical" depends
  on which chain you're looking at)
- A maintenance hazard (edits to one side might not propagate
  symmetrically to the other in the way expected)
- Cosmetically inconsistent with the foot LINKs (which post-process
  renames to proper `left_foot` / `right_foot`)

### What to fix in Onshape

Pick one canonical convention per body region and apply consistently.
Two options, both acceptable:

**Option A — match the foot convention (rename to `left_X` / `right_X`):**
- Legs: `upper_leg` → `left_upper_leg`, `knee` → `right_upper_leg`,
  `lower_leg` → `left_lower_leg`, `lower_leg_2` → `right_lower_leg`
- Arms: `shoulder_2` → `left_shoulder`, `shoulder` → `right_shoulder`,
  same pattern for `upper_arm`, `fore_arm`, `palm`, `finger`
- Most explicit; mirrors the L/R prefix used for hips and feet

**Option B — keep `_2` suffix but fix the inversion:**
- Either: legs and arms both use no-suffix=LEFT, `_2`=RIGHT (current
  leg convention), so arm sub-assemblies need to be renamed
- Or: legs and arms both use no-suffix=RIGHT, `_2`=LEFT (current
  arm convention), so leg sub-assemblies need to be renamed

Option A is recommended because it's the convention already used for
hips and feet, and is unambiguous in code that consumes body names.

## Issue 4 — Sub-mm asymmetries in body-inertia COM positions

### What we see (from `tools/reference_geometry_parity.py` whole-body audit)

Inertia COM positions for body pairs differ by sub-mm to micrometer
amounts. Examples:

| Body pair | Left COM (x, y, z) | Right COM (x, y, z) | Δx | Δy | Δz |
|---|---|---|---|---|---|
| left_hip / right_hip | (−0.00291, −0.00898, −0.0331) | (+0.00226, −0.00872, −0.0331) | 0.5 mm | 0.3 mm | ~µm |
| left_knee_mimic / right_knee_mimic | (+0.0475, +0.09019, +0.2369) | (+0.0471, −0.09016, +0.2368) | 0.4 mm | 30 µm | ~µm |
| left_foot / right_foot | (+0.02221, +0.09052, +0.01132) | (+0.02160, −0.09003, +0.01131) | 0.6 mm | 0.5 mm | ~µm |

These are CAD floating-point precision artifacts from imperfect mirror
operations. Each individual asymmetry is small but they add up: the
total whole-body COM ends up 0.7 mm to the right of the support
polygon center at standing.

### Why this matters

In isolation, each sub-mm asymmetry is negligible. Aggregated:
- Whole-body COM y bias of 0.7 mm
- Standing equilibrium settles to a slightly tilted posture
- Walking dynamics amplify the bias into asymmetric joint loading

Issue 2 is the largest contributor; Issue 4 contributes the rest.

### What to fix in Onshape

The fundamental fix is to **apply CAD mirror constraints to every L/R
body pair**, not just the upper-leg sub-assembly (Issue 2). This
includes:
- Hip sub-assemblies
- Knee mimic / ankle mimic frames
- Foot sub-assemblies
- All arm sub-assemblies

If the CAD authoring style is to design one side and mirror to the
other, ensure every mirror operation is a true CAD mirror constraint
(maintained by the assembly), not a one-time copy-and-paste.

## Verification checklist after CAD fixes

Once the Onshape model is fixed, re-run the export pipeline and verify:

```bash
# Re-export and post-process
./assets/update_xml.sh --version v2 --export-onshape

# Verify the symmetry programmatically
uv run python tools/phase10_diagnostic.py --vx 0.15

# Expected results after fix:
# - D3+D4 left_hip_roll peak sat rate: 0.333 -> < 0.05
# - D3+D4 verdict: down-cross-correlated-mechanism-unclear
#                  -> uniform-sat or no-significant-sat
# - All P0 / P1A absolute / P2 q-step / P1 survival gates remain PASSING
# - Episode survival in Phase 10 replay: 67-72 -> > 100 steps
```

Also re-derive the standing keyframes against the symmetric model:

```bash
# Re-settle home + walk_start equilibrium poses
# (see assets/v2/keyframes.xml comments for the round-6 / round-7
#  settling procedure documentation)
```

## Workaround until CAD is fixed

While the CAD-side fix is being prepared, a post-process workaround can
be applied in `assets/post_process.py` to:
1. Rename the `knee` body to `right_upper_leg`
2. Override the right-leg body position to be the exact mirror of left

This is documented as a stopgap; the proper fix remains the CAD-side
correction. See Phase 12C closeout in
`training/docs/reference_architecture_comparison.md` for context.

## Appendix: how the bugs were discovered

The asymmetries were surfaced by running closed-loop replay of the ZMP
walking prior under MuJoCo as part of Phase 10 diagnostic in
`training/docs/reference_architecture_comparison.md`. Key findings:

- Per-step joint trace at `vx=0.15` shows `left_hip_roll` saturating
  at +0.175 rad limit while `right_hip_roll` stays at −0.04 rad (well
  within range)
- IK output is symmetric (`left_hip_roll` + `right_hip_roll` q_ref
  sums to ~0 on average), so the planner is not the source
- Standing equilibrium pose has both hip_rolls at slightly negative q
  values (−0.0023 and −0.0041), not mirror-symmetric
- Whole-body COM is offset 0.7 mm to the right of support polygon
  center
- Asymmetry is invariant to `vx` (verified in Phase 9A re-read at
  vx ∈ {0.21, 0.265}, ruling out shuffling-driven causes)
- Tracing the body geometries showed the right-side LINK is named
  `knee` (not the expected `upper_leg_2`) and its position is
  asymmetric to left's `upper_leg` by 0.8 mm in x
- Audit of `assets/v2/onshape_export/wildrobot.xml` (the raw export,
  before any post-processing) confirms both bugs originate in the CAD
  source, not in our pipeline

For full investigation traces and supporting data, see:
- `tools/phase10_diagnostic.py` (the closed-loop diagnostic tool)
- `tools/phase10_diagnostic.json` (per-step traces from the audit)
- `tools/phase10_diagnostic_nonshuffle.json` (Phase 9A re-read)
- Commits `cc5e8d3`, `381ab3c`, `d3a6989`, `72f25d4`, `2cf32d6`,
  `5b634be`, `e0cf165` for the diagnostic implementation, methodology
  fixes, and verdict
