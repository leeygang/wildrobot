# WildRobot v2 — Arm-chain CAD follow-ups

**Status:** non-blocking residual after the 2026-08-23 Onshape refresh. The
wrist servo DOFs and separate palm/finger bodies are gone, but a small mirrored
arm-position mismatch remains and can contribute to standing lean.

**History:** the original `model_issues.md` (now removed) tracked four
CAD-side asymmetries surfaced by Phase 12C closed-loop diagnostic.
Issues 1 (right-leg `knee` body misnamed) and 2 (upper-leg LINK frame
0.8 mm x asymmetry, sign-flipped y) are **fully closed** in the current
export; closeout history lives in CHANGELOG entries
`v0.20.1-ankle-roll-realignment`, `v0.20.1-model-issues-reaudit`, and
`v0.20.1-c1c2-closeout-phase9A`.  The leg chain is now near-perfectly
symmetric (sub-100 µm in x and z; 0 mm in y at parent-local; Phase 10
shows zero saturation at vx=0.15).

This file tracks the remaining arm-chain issue. The generated model passes the
left/right body mass and inertia symmetry checks.

---

## Issue 4 (arm subtree) — Sub-mm to ~5 mm L/R asymmetries

World-frame body-pair COM offsets at the home keyframe:

| Body pair | |Δx| | |Δy_mirror| | |Δz| |
|---|---|---|---|
| left_shoulder / right_shoulder | 0.37 mm | 1.27 mm | 0.17 mm |
| left_upper_arm / right_upper_arm | 0.50 mm | 2.93 mm | 0.39 mm |
| **left_fore_arm / right_fore_arm** | **0.77 mm** | **5.25 mm** | **0.79 mm** |

**Aggregate effect:** whole-body COM y-bias = -1.25 mm at the current home
keyframe (magnitude improved from 1.5 mm). The forearm inertias now include the
fixed hand/wrist geometry, so there are no separate palm/finger body pairs to
audit.

**What remains to fix in Onshape:** tighten the mirror constraints on the
shoulder-to-forearm chain, especially the forearm placement. The wrist-servo
removal itself is complete.

## Re-checking after a CAD fix

```bash
# World-frame COM, mass, inertia, and collision symmetry
uv run python assets/validate_model.py
```

Expected after a follow-up: every arm body pair should be a mirror within the
validator tolerance, with the forearm `|Δy_mirror|` reduced below 1 mm.
