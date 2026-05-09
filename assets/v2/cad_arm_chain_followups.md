# WildRobot v2 — Arm-chain CAD follow-ups

**Status:** non-blocking residual — arms are static during walking, so
this does not affect sim training. May matter for hardware deployment
(v0.20.2+) where the resulting whole-body COM bias could surface as a
small standing lean.

**History:** the original `model_issues.md` (now removed) tracked four
CAD-side asymmetries surfaced by Phase 12C closed-loop diagnostic.
Issues 1 (right-leg `knee` body misnamed) and 2 (upper-leg LINK frame
0.8 mm x asymmetry, sign-flipped y) are **fully closed** in the current
export; closeout history lives in CHANGELOG entries
`v0.20.1-ankle-roll-realignment`, `v0.20.1-model-issues-reaudit`, and
`v0.20.1-c1c2-closeout-phase9A`.  The leg chain is now near-perfectly
symmetric (sub-100 µm in x and z; 0 mm in y at parent-local; Phase 10
shows zero saturation at vx=0.15).

This file tracks the remaining arm-chain issues.

---

## Issue 3 — Arm chain inverse `_2` suffix convention

Current naming in `assets/v2/wildrobot.xml`:

| Chain | LEFT | RIGHT | `_2` on |
|---|---|---|---|
| Legs (post Issue 1 fix) | `upper_leg` / `lower_leg` / `left_foot` | `upper_leg_2` / `lower_leg_2` / `right_foot` | RIGHT |
| Arms (unchanged) | `shoulder_2` / `upper_arm_2` / `fore_arm_2` / `palm_2` / `finger_2` | `shoulder` / `upper_arm` / `fore_arm` / `palm` | **LEFT** (inverted) |

**Why this isn't blocking:** PPO and the planner reference joints by
name (`left_*` / `right_*`), not by body name.  The inversion is a
maintenance hazard, not a correctness issue.

**What to fix in Onshape:** pick one canonical convention per body
region and apply consistently.  Recommended: rename to
`left_X` / `right_X` everywhere (matches the leg-chain foot naming
already done by `assets/post_process.py`).  Defer to next intentional
Onshape authoring pass.

## Issue 4 (arm subtree) — Sub-mm to ~1 cm L/R asymmetries

World-frame body-pair COM offsets at the home keyframe:

| Body pair | |Δx| | |Δy_mirror| | |Δz| |
|---|---|---|---|
| shoulder_2 / shoulder | 1.01 mm | 1.69 mm | 0.05 mm |
| upper_arm_2 / upper_arm | 0.96 mm | 3.46 mm | 0.17 mm |
| fore_arm_2 / fore_arm | 0.78 mm | 3.68 mm | 0.19 mm |
| **palm_2 / palm** | **0.59 mm** | **10.11 mm** | **0.80 mm** |
| **finger_2 / finger** | **0.53 mm** | **11.35 mm** | **0.87 mm** |

**Aggregate effect:** whole-body COM y-bias = 1.5 mm (was 0.7 mm
pre-merge — leg chain became more symmetric, so arm asymmetries now
dominate).  Static bias only; does not affect closed-loop walking
gates because the bias settles into the home keyframe equilibrium.

**What to fix in Onshape:** apply CAD mirror constraints to all arm
sub-assemblies.  Same Onshape change as Issue 3 (rename + add mirror
mate constraints in one pass).

## Re-checking after a CAD fix

```bash
# Local-frame body-pair symmetry (parent-local positions)
uv run python -c "
import re
xml = open('assets/v2/onshape_export/wildrobot.xml').read()
for l, r in [('shoulder_2','shoulder'), ('upper_arm_2','upper_arm'), ('fore_arm_2','fore_arm'), ('palm_2','palm'), ('finger_2','finger')]:
    lp = [float(v) for v in re.search(rf'<body name=\"{l}\" pos=\"([^\"]+)\"', xml).group(1).split()]
    rp = [float(v) for v in re.search(rf'<body name=\"{r}\" pos=\"([^\"]+)\"', xml).group(1).split()]
    print(f'{l:<14}{lp}\\n{r:<14}{rp}\\n')"

# World-frame COM symmetry at the home keyframe
uv run python assets/validate_model.py
```

Expected after fix: every arm body pair should have parent-local
positions that are bit-identical mirrors (or as close as the CAD
mirror constraint allows — sub-100 µm tolerance is fine).
