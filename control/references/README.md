# Locomotion References (Milestone 0)

This package defines the locomotion-first schema contract introduced in `v0.19.0`.

Milestone 0 scope is intentionally schema-only:
- command space (`forward_speed_mps`, `yaw_rate_rps`)
- reference state container used by training/runtime
- observation contract container (IMU orientation + angular velocity, plus command + reference)
- residual action contract around nominal targets

Non-goals in this milestone:
- no walking reference generator implementation
- no LIPM/DCM logic
- no IK logic
