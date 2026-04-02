# Locomotion References (Milestone 0)

This package defines the locomotion-first schema contract introduced in `v0.19.0`.

Milestone 0 scope is intentionally schema-only:
- command space (`forward_speed_mps`, `yaw_rate_rps`)
- reference state container used by training/runtime
- observation contract container (IMU orientation + angular velocity, plus command + reference)
- residual action contract around nominal targets

Milestone `v0.19.2` now adds:

- `walking_ref_v1.py`: forward-only bounded stepping reference
- reduced-order step-placement shaping using LIPM/DCM helpers
- outputs compatible with `LocomotionReferenceState`

Still intentionally deferred to `v0.19.3+`:

- policy integration in env/runtime loops
- yaw reference support
- ALIP upgrade
