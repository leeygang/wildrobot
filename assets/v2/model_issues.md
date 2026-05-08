# WildRobot v2 — Model Issues to Fix in Onshape

**Last updated:** 2026-05-05 (post v20 ankle_roll merge re-audit)
**Original audit:** Phase 12C closed-loop diagnostic (commits `cc5e8d3` / `381ab3c` / `d3a6989`), 2026-04-26
**Re-audit trigger:** v20 ankle_roll merge (commits `66d5c00`, `731a07d`, `f5906cc`) appears to have re-exported with corrected sub-assembly naming and tighter mirror constraints in the leg chain.
**Onshape document:** [WildRobot](https://cad.onshape.com/documents/ba106cac193c0e2253e1958f/w/3b5db3932a544d9f939ac6a6/e/5c6acb3ff696b03bdf018f3b)

This document records issues discovered in the WildRobot v2 MJCF model that
**originate in the Onshape CAD source** (not in our `onshape-to-robot`
export pipeline or our `assets/post_process.py`). All issues survive the
export-pipeline unchanged and end up in `assets/v2/wildrobot.xml`.

The bugs were originally surfaced by Phase 10 of
`training/docs/reference_architecture_comparison.md`. The original symptom
(`left_hip_roll` saturating to its inner limit during walking) is no
longer reproducible at the in-scope operating point post the v20 merge —
see the "Re-audit results" section below.

## TL;DR — current status (post v20 ankle_roll merge)

| # | Issue | Pre-merge severity | Current status | Onshape change still required? |
|---|---|---|---|---|
| 1 | Right-leg upper-leg LINK named `knee` instead of `upper_leg_2` | HIGH | ✅ **CLOSED** — body now named `upper_leg_2` in raw export | No |
| 2 | Upper-leg LINK frame positions not exact mirrors | HIGH (load-bearing) | ⚠️ **PARTIALLY CLOSED** — y perfectly mirrored, x asymmetry 4× smaller (0.8 mm → 0.2 mm), z roughly unchanged (~95 µm) | Optional — residual is sub-mm and not load-bearing at vx ≤ 0.15 |
| 3 | Arm chain has inverse `_2` suffix convention from legs | MEDIUM (cosmetic) | ❌ **NOT FIXED** — still `shoulder_2` (LEFT) / `shoulder` (RIGHT), inverted from leg convention | Yes (cosmetic + maintenance) |
| 4 | Sub-mm L/R asymmetries in body-inertia COM positions | LOW | ⚠️ **MIXED** — leg chain is now sub-100 µm in x and z; arm chain has 0.5–1 mm in x and up to 11 mm in y-mirror at palm/finger | Yes (especially for arm pairs) |

## Re-audit results — post v20 merge

### Empirical evidence the load-bearing issue is closed

The Phase 10 closed-loop diagnostic re-run on 2026-05-05 (commit pending,
post-`31b764d` ankle-roll-realignment patches):

| Metric | Pre-merge (cited) | Post-merge (re-run) |
|---|---|---|
| `left_hip_roll` peak sat at vx=0.10 | 0.250 (uniform across i_swing 1-9) | **0.091 at i_swing=8 only** |
| `left_hip_roll` peak sat at vx=0.15 | 0.250 (i_swing 7-10) | **0.000** (no joint saturates anywhere) ✓ |
| `left_hip_roll` peak sat at vx=0.20 | 0.333 (i_swing 8-9) | 0.333 (unchanged — out-of-scope vx) |
| `ref_contact_match_frac` at vx=0.15 | 0.611 | 0.745 (+0.134) |
| Phase 10 D3+D4 verdict at vx=0.15 | down-cross-correlated-mechanism-unclear | **no-significant-sat** |

The original "0.8 mm geometric bias produces 0.18 rad joint deviation that
exceeds actuator authority" calculation no longer applies because the
geometric bias is now 0.2 mm (linearly: ≈0.045 rad, well within actuator
authority). Combined with the v20 ankle_roll merge giving the prior a
local foot-flat actuator, lateral support is now distributed across two
joints rather than forced through hip_roll alone.

### Body position re-audit (parent-frame, from `assets/v2/wildrobot.xml`)

#### Issue 2 — upper-leg LINK frame: largely fixed

| Component | Left (`upper_leg`) | Right (`upper_leg_2`) | Pre-merge asymmetry | Post-merge asymmetry |
|---|---|---|---|---|
| pos x | −0.0269 | +0.0267 | 0.8 mm | **0.2 mm** (4× smaller) |
| pos y | −0.000166429 | −0.000166429 | sign-flipped + ~3 µm magnitude | **0** (perfectly mirrored) ✓ |
| pos z | −0.0365475 | −0.0364525 | 89 µm | 95 µm (unchanged) |

The body name is now `upper_leg_2` (was `knee`), so Issue 1 is structurally
closed in the export.

### World-frame body-pair COM symmetry (Issue 4 / 3)

Computed at the `home` keyframe with `mj_forward`. `|Δx|`, `|Δz|` should be
zero for a perfect mirror; `|Δy_mirror|` is `|left_y + right_y|` (zero if
the L/R COMs are symmetric about the sagittal plane).

```
L body name           R body name              |Δx|   |Δy_mirror|     |Δz|
----------------------------------------------------------------------------
LEG CHAIN:
left_hip              right_hip               0.05mm     1.77mm    0.10mm
upper_leg             upper_leg_2             0.07mm     3.47mm    0.10mm
lower_leg             lower_leg_2             0.02mm     1.41mm    0.10mm
left_foot             right_foot              0.08mm     0.34mm    0.01mm

ARM CHAIN (Issue 3 + 4):
shoulder_2 (LEFT)     shoulder    (RIGHT)     1.01mm     1.69mm    0.05mm
upper_arm_2 (LEFT)    upper_arm   (RIGHT)     0.96mm     3.46mm    0.17mm
fore_arm_2  (LEFT)    fore_arm    (RIGHT)     0.78mm     3.68mm    0.19mm
palm_2      (LEFT)    palm        (RIGHT)     0.59mm    10.11mm    0.80mm
finger_2    (LEFT)    finger      (RIGHT)     0.53mm    11.35mm    0.87mm

WHOLE-BODY (4.128 kg total):
Whole-body COM (world):  (-0.00453, -0.00136, +0.33524)
COM y-bias from y=0:     1.36 mm
```

#### Reading

- **Leg chain is now near-symmetric in x and z** (sub-100 µm). The y-mirror
  residual (0.34–3.47 mm) is the largest remaining asymmetry; it does not
  trip Phase 10's saturation gates at vx ≤ 0.15.
- **Arm chain is materially asymmetric** — palm and finger COMs are off
  the y-mirror by ~1 cm. This does not affect locomotion (arms are static
  during walking and contribute to whole-body COM only as a near-constant
  offset), but it is the largest remaining symmetry violation.
- **Whole-body COM y-bias is 1.4 mm** (was 0.7 mm pre-merge per the
  original doc). The leg chain is more symmetric now, so the bias has
  shifted toward arm-driven sources. Static COM bias does not affect
  Phase 10 gates because the closed-loop replay starts from the home
  keyframe which has already settled to this biased equilibrium.

### `assets/validate_model.py` results

The default invocation (post the v0.20.x realignment patch) reports:

```
Checks passed:
  - scene include ↔ robot_config.generated_from
  - toe/heel collision symmetry + mesh_pos sanity
  - left/right body mass + inertia symmetry
  - left/right named collision geom parity
  - default reset contact sanity
  - actuated joint range parity
Model validation OK
```

`left/right body mass + inertia symmetry` PASS confirms that Issue 4 is
within the validate_model tolerance for ALL body pairs (legs and arms).
The arm asymmetries above are real but small enough to clear the parity
band the validator uses.

## Issue 1 — Right-leg LINK named `knee` (CLOSED)

**Status:** CLOSED. The raw export at
`assets/v2/onshape_export/wildrobot.xml:143-144` now has:

```xml
<!-- Link upper_leg_2 -->
<body name="upper_leg_2" pos="0.0267 -0.000166429 -0.0364525" ...>
```

The pre-merge `<!-- Link knee -->` comment is gone. The CAD-side
sub-assembly under `right_hip` was renamed (or auto-mirrored from a
renamed source); the export now produces a proper mirror name.

No further action.

## Issue 2 — Upper-leg LINK frame positions not exact mirrors (PARTIALLY CLOSED)

**Status:** PARTIALLY CLOSED.

Position values in the raw export (and in `assets/v2/wildrobot.xml`):

```xml
<!-- LEFT body position (unchanged) -->
<body name="upper_leg"   pos="-0.0269 -0.000166429 -0.0365475" .../>

<!-- RIGHT body position (post-merge) -->
<body name="upper_leg_2" pos="+0.0267 -0.000166429 -0.0364525" .../>
```

| Component | Pre-merge asymmetry | Post-merge asymmetry | Doc-projected joint deviation |
|---|---|---|---|
| x | 0.8 mm | **0.2 mm** | ~0.045 rad (was 0.18 rad) |
| y | sign-flipped + ~3 µm magnitude | **0** ✓ | n/a |
| z | 89 µm | 95 µm | n/a |

The y-component is now perfectly mirrored — this was the sign-flip the
original audit flagged as the largest violation.

The residual x-asymmetry (0.2 mm) and z-asymmetry (95 µm) are sub-mm.
Linearly scaling the original "0.8 mm → 0.18 rad joint deviation"
calculation gives ~0.045 rad at the current 0.2 mm bias — well within the
position actuator's tracking authority.

### Empirical confirmation

Phase 10 diagnostic at vx=0.15 (post-merge) shows zero joint saturation
across all 8 leg joints. The "load-bearing" qualifier in the original
audit no longer applies at the in-scope operating point.

### Should this be fixed in CAD anyway?

**Optional.** The CAD-side ideal is exact mirror. Practical considerations:

- **For sim-only work at vx ≤ 0.15:** no measurable benefit. Phase 10
  already shows no saturation.
- **For sim2real deployment (v0.20.2+):** worth doing. The hardware
  position actuator stack has less tracking margin than MuJoCo's; a
  residual 0.2 mm bias may resurface as a small but consistent lean
  during real-robot trials.
- **For higher-vx training:** at vx=0.20 the Phase 10 diagnostic still
  shows `left_hip_roll` peak sat 0.333. Whether this is the residual CAD
  asymmetry or a different mechanism (post-merge spawn-z change, new
  collision primitives) is unknown without further diagnostic. The
  remaining x-asymmetry is a candidate.

If fixing in CAD: apply Option A from the original audit (CAD mirror
constraint between left and right upper-leg sub-assemblies, mirrored
across the sagittal plane). After re-export, the right body's `pos`
should be `(+0.0269, -0.000166429, -0.0365475)` exactly.

## Issue 3 — Arm chain inverse `_2` suffix convention (NOT FIXED)

**Status:** NOT FIXED. Current arm naming in `assets/v2/wildrobot.xml`:

```
LEGS:  upper_leg (LEFT) /  upper_leg_2 (RIGHT)   ← left = canonical
       lower_leg (LEFT) /  lower_leg_2 (RIGHT)
       left_foot (LEFT) /  right_foot (RIGHT)    ← post_process renames

ARMS:  shoulder_2  (LEFT) / shoulder  (RIGHT)    ← right = canonical (INVERTED)
       upper_arm_2 (LEFT) / upper_arm (RIGHT)
       fore_arm_2  (LEFT) / fore_arm  (RIGHT)
       palm_2      (LEFT) / palm      (RIGHT)
       finger_2    (LEFT) / finger    (RIGHT)
```

The leg / arm convention inversion documented in the original audit
persists: legs are authored on the LEFT side and mirrored to RIGHT (so
RIGHT has `_2`); arms are authored on the RIGHT side and mirrored to LEFT
(so LEFT has `_2`).

This is cosmetic + maintenance hazard, not a correctness issue. PPO and
the planner reference joints by name (`left_*` / `right_*`), not by body
name, so the inversion does not affect any downstream consumer.

### What to fix in Onshape (unchanged from original audit)

Pick one canonical convention per body region and apply consistently. See
the original `model_issues.md` Issue 3 section for the two options
(Option A: rename to `left_X` / `right_X`; Option B: fix the `_2`
inversion).

**Recommended sequencing:** defer until the next intentional Onshape
authoring pass; not worth a stand-alone CAD change.

## Issue 4 — Sub-mm L/R asymmetries in body-inertia COM positions (MIXED)

**Status:** MIXED.

### Leg chain — near-symmetric (was sub-mm, now sub-100 µm in x and z)

| Body pair | Pre-merge example (orig doc) | Post-merge re-audit |
|---|---|---|
| left_hip / right_hip | (0.5 mm, 0.3 mm, ~µm) | (0.05 mm, 1.77 mm, 0.10 mm) |
| upper_leg / upper_leg_2 | n/a | (0.07 mm, 3.47 mm, 0.10 mm) |
| lower_leg / lower_leg_2 | n/a | (0.02 mm, 1.41 mm, 0.10 mm) |
| left_foot / right_foot | (0.6 mm, 0.5 mm, ~µm) | (0.08 mm, 0.34 mm, 0.01 mm) |

x and z components are at sub-100 µm (CAD floating-point noise level).
The y-mirror residual (0.3–3.5 mm) is larger but is dominated by the
chain's lever arm — small angular asymmetries at the hip get amplified
through the upper_leg's length. The leg chain is no longer the dominant
contributor to whole-body COM bias.

### Arm chain — substantial asymmetries (NOT FIXED)

| Body pair | |Δx| | |Δy_mirror| | |Δz| |
|---|---|---|---|
| shoulder_2 / shoulder | 1.01 mm | 1.69 mm | 0.05 mm |
| upper_arm_2 / upper_arm | 0.96 mm | 3.46 mm | 0.17 mm |
| fore_arm_2 / fore_arm | 0.78 mm | 3.68 mm | 0.19 mm |
| palm_2 / palm | **0.59 mm** | **10.11 mm** | **0.80 mm** |
| finger_2 / finger | **0.53 mm** | **11.35 mm** | **0.87 mm** |

The palm and finger pairs are off the y-mirror by ~1 cm. This does not
affect walking (arms are static during locomotion and contribute as a
near-constant offset to the whole-body COM), but it is the largest
remaining symmetry violation in the model.

### Whole-body COM bias

```
Whole-body COM (world):  (-0.00453, -0.00136, +0.33524)
COM y-bias from y=0:     1.36 mm  (was 0.7 mm pre-merge per orig doc)
```

The whole-body y-bias actually grew slightly (0.7 → 1.4 mm) despite the
leg chain becoming more symmetric. The shift is consistent with
arm-side asymmetries dominating, since the legs are now sub-100 µm in
the x-mirror direction.

This static bias settles into the home keyframe equilibrium and the
closed-loop replay starts from there, so Phase 10 metrics are
insensitive to it. It may matter on real hardware where the standing
controller has less tolerance.

### What to fix in Onshape

Apply CAD mirror constraints to all arm sub-assemblies (Issue 3's
recommended Option A would fix the naming and the constraints in one
change). The leg chain is good enough as-is.

## Workaround status

The post-process workaround section from the original doc (rename `knee`
to `right_upper_leg`, override the right-leg body position) is **no
longer needed**: Issue 1 is closed in the export, and Issue 2's residual
is sub-mm and not load-bearing.

If the residual x-asymmetry on the upper-leg becomes a measurable problem
on hardware (v0.20.2+), the cheapest interim fix is the same workaround
in `assets/post_process.py` — but defer until there's evidence it's
needed.

## Open follow-ups

1. **Phase 10 at vx=0.20 still shows `left_hip_roll` saturation
   (peak 0.333).** Out-of-scope for the current operating point but
   worth investigating before any operating-vx widening (Phase 9). The
   residual 0.2 mm x-asymmetry on the upper-leg, the spawn-z change
   (0.50 → 0.48 m), and the post-process collision primitive changes are
   all candidates.
2. **Whole-body COM y-bias grew (0.7 → 1.4 mm).** Track on hardware
   trials; the arm-pair asymmetries are the likely source.
3. **Arm-chain mirror constraints (Issue 3 + Issue 4-arm).** Defer to
   the next intentional Onshape authoring pass.

## Verification checklist

```bash
# Repeat the symmetry checks anytime the model changes
uv run python assets/validate_model.py

# Re-run the Phase 10 closed-loop diagnostic
uv run python tools/phase10_diagnostic.py --vx 0.10 0.15 0.20

# Expected post-merge baseline (recorded 2026-05-05):
# - vx=0.10: left_hip_roll peak sat = 0.091 at i_swing=8 only
# - vx=0.15: no-significant-sat (all 8 leg joints zero saturation)
# - vx=0.20: left_hip_roll peak sat = 0.333 (still down-cross-correlated)
# - All P0 / P1A absolute / P2 q-step / P1 survival gates at vx ≤ 0.15
#   remain PASSING
```

## Appendix: original audit context

The pre-merge investigation (Phase 10 of
`training/docs/reference_architecture_comparison.md`) is preserved in
git history at commits `cc5e8d3`, `381ab3c`, `d3a6989`, `72f25d4`,
`2cf32d6`, `5b634be`, `e0cf165`. The original `model_issues.md` content
(which described the pre-merge state of Issues 1-4) is recoverable via
`git show 9c8d9ee:assets/v2/model_issues.md`.

This re-audit (2026-05-05) was triggered by a question on whether the v20
ankle_roll merge had implicitly fixed the CAD-side asymmetries that
Phase 12C had opened to investigate. The answer is: largely yes for the
leg chain (Issues 1 + 2), no for the arm chain (Issues 3 + 4-arm).
