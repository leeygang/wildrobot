# Adapters (Milestone 0)

This package contains small adapter utilities for locomotion contracts.

For Milestone 0, it only includes schema-level composition helpers and avoids
controller logic.

Milestone `v0.19.2` adds:

- `reference_to_joint_targets.py` to convert locomotion reference + leg IK into bounded nominal `q_ref` targets in actuator order.
