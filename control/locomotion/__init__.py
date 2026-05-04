"""Locomotion controller package.

The v0.19.x ``walking_controller`` and ``nominal_ik_adapter`` modules
were deleted at v0.20.1.  v0.20.x architecture: locomotion semantics
live in the offline ``ReferenceLibrary``
(``control.references.reference_library``) and are consumed at
runtime via ``RuntimeReferenceService``
(``control.references.runtime_reference_service``).  No runtime
walking controller exists in the active path; PPO + the offline
reference together replace it.
"""
