#!/usr/bin/env python3
"""Compute GMR rot_offsets for wildrobot mimic bodies.

This script analyzes the wildrobot MuJoCo model to determine the correct
rot_offset quaternions for each mimic body used in GMR IK matching.

The rot_offset transforms from SMPLX body frame to robot mimic body frame.

SMPLX coordinate system: Y-up, Z-forward, X-right
MuJoCo/Robot coordinate system: Z-up, X-forward, Y-left

The base transform from SMPLX to MuJoCo Z-up is [0.5, -0.5, -0.5, -0.5] (wxyz).

For each robot mimic body, we need to combine:
1. SMPLX to MuJoCo world transform
2. MuJoCo world to robot mimic body local frame transform

rot_offset = mimic_body_world_quat^-1 * smplx_to_mujoco_transform * smplx_body_orientation

Since we want the robot mimic body to match SMPLX body orientation after IK,
and the mimic bodies have their own local orientations, we need to adjust.

Actually, for GMR's offset_human_data():
    updated_quat = (smplx_body_quat * rot_offset)

This means rot_offset should transform the SMPLX body orientation to match
what we want the robot mimic body to look like in world frame.

For a standing pose, pelvis_mimic should be identity (facing forward, upright).
SMPLX pelvis in standing pose is also identity in SMPLX frame.
So rot_offset for pelvis = transform from SMPLX frame to MuJoCo frame = [0.5, -0.5, -0.5, -0.5]

For other mimic bodies that have rotations in the robot model,
we need to account for those rotations.
"""

import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_inverse(q):
    """Inverse of quaternion (wxyz format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    """Multiply two quaternions (wxyz format)."""
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])  # convert to xyzw
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    result = (r1 * r2).as_quat()  # returns xyzw
    return np.array([result[3], result[0], result[1], result[2]])  # convert to wxyz


def analyze_mimic_bodies(xml_path):
    """Analyze mimic body orientations in the wildrobot model."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Put robot in default pose
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Find mimic sites
    mimic_sites = [
        "pelvis_mimic",
        "left_hip_mimic",
        "left_knee_mimic",
        "left_ankle_mimic",
        "left_foot_mimic",
        "right_hip_mimic",
        "right_knee_mimic",
        "right_ankle_mimic",
        "right_foot_mimic",
    ]

    # SMPLX to MuJoCo world transform
    # SMPLX: Y-up, Z-forward, walking in -X direction
    # MuJoCo: Z-up, X-forward
    # Transform: +90° around X (Y->Z), then 180° around Z (-X->+X)
    # Quaternion (wxyz): [0, 0, 0.707107, 0.707107]
    smplx_to_mujoco = np.array([0.0, 0.0, 0.707107, 0.707107])  # wxyz

    print("=" * 70)
    print("Wildrobot Mimic Body Analysis for GMR rot_offset")
    print("=" * 70)
    print()
    print("SMPLX to MuJoCo transform (wxyz): [0.5, -0.5, -0.5, -0.5]")
    print()

    results = {}

    for site_name in mimic_sites:
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except:
            print(f"Site {site_name} not found")
            continue

        # Get site quaternion in world frame from forward kinematics
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        site_rot = R.from_matrix(site_xmat)
        site_quat_xyzw = site_rot.as_quat()  # scipy returns xyzw
        site_quat_wxyz = np.array([site_quat_xyzw[3], site_quat_xyzw[0],
                                   site_quat_xyzw[1], site_quat_xyzw[2]])

        # Get site position in world frame
        site_pos = data.site_xpos[site_id]

        # The rot_offset for GMR should transform SMPLX body orientation
        # to the robot mimic body orientation in world frame.
        #
        # GMR applies: updated_quat = smplx_body_quat * rot_offset
        #
        # For a standing T-pose, SMPLX body quats are identity (in SMPLX frame).
        # We want updated_quat to result in the mimic body matching world Z-up.
        #
        # Since site_quat is already the mimic body's world orientation,
        # and we want SMPLX identity * rot_offset to give us Z-up orientation,
        # rot_offset should be the SMPLX→MuJoCo transform.
        #
        # BUT if the mimic body has a non-identity local orientation in the robot
        # model (defined via the site quat attribute), we need to account for that.

        # Get the site's local quat from the model definition
        site_local_quat = model.site_quat[site_id]  # wxyz

        # If site has non-identity local quat, the world orientation is:
        # site_world_quat = parent_body_world_quat * site_local_quat
        #
        # For GMR, we want the IK solution to make the site match SMPLX orientation.
        # The rot_offset should combine:
        # 1. SMPLX frame to MuJoCo frame
        # 2. Account for site's local rotation relative to its parent body

        # For simplicity, we'll use the base SMPLX→MuJoCo transform but note
        # when sites have non-identity local orientations.

        print(f"{site_name}:")
        print(f"  World position: [{site_pos[0]:.4f}, {site_pos[1]:.4f}, {site_pos[2]:.4f}]")
        print(f"  World quat (wxyz): [{site_quat_wxyz[0]:.6f}, {site_quat_wxyz[1]:.6f}, "
              f"{site_quat_wxyz[2]:.6f}, {site_quat_wxyz[3]:.6f}]")
        print(f"  Local quat (wxyz): [{site_local_quat[0]:.6f}, {site_local_quat[1]:.6f}, "
              f"{site_local_quat[2]:.6f}, {site_local_quat[3]:.6f}]")

        # Check if local quat is identity
        is_identity = np.allclose(site_local_quat, [1, 0, 0, 0], atol=1e-4)
        print(f"  Local is identity: {is_identity}")

        # Compute recommended rot_offset
        # The rot_offset should transform SMPLX body orientation to match
        # what we want the robot mimic body to represent.
        #
        # For bodies with identity local quat (like pelvis_mimic),
        # rot_offset = smplx_to_mujoco = [0.5, -0.5, -0.5, -0.5]
        #
        # For bodies with non-identity local quat, we need to adjust.
        # The adjustment is: rot_offset = site_local_quat^-1 * smplx_to_mujoco * smplx_body_local
        # But since SMPLX body local orientations vary, we'll use a different approach.

        if is_identity:
            recommended_rot_offset = smplx_to_mujoco
        else:
            # For non-identity local quats, we need to "undo" the site's local rotation
            # rot_offset = smplx_to_mujoco * site_local_quat^-1
            # This way: smplx_quat * rot_offset = smplx_quat * smplx_to_mujoco * site_local^-1
            # Which gives us the world orientation adjusted for the site's local frame

            # Actually, let's think about this differently.
            # The GMR IK tries to match robot body orientation to SMPLX body orientation.
            # The rot_offset transforms the SMPLX body orientation BEFORE matching.
            #
            # If robot site has local_quat L, and we want site world orientation = target,
            # then parent_body_quat * L = target
            # parent_body_quat = target * L^-1
            #
            # The IK controls parent_body_quat, not site orientation directly.
            # So we want: parent_body_quat ≈ smplx_body_quat * rot_offset * L^-1
            # But GMR applies rot_offset to smplx_quat, then uses that as target for IK.
            # GMR sets: target = smplx_quat * rot_offset
            # IK makes: robot_body_quat ≈ target
            #
            # So if site is attached to robot_body with local L:
            # site_world ≈ target * L = smplx_quat * rot_offset * L
            #
            # For SMPLX standing pose (smplx_quat = identity in SMPLX frame = Y-up),
            # site_world = rot_offset * L
            #
            # We want site_world = identity (Z-up), so:
            # identity = rot_offset * L
            # rot_offset = L^-1
            #
            # But we also need smplx→mujoco transform, so:
            # site_world_mujoco = smplx_to_mujoco (for smplx identity)
            # We want: smplx_to_mujoco = rot_offset * L
            # rot_offset = smplx_to_mujoco * L^-1

            site_local_inv = quat_inverse(site_local_quat)
            recommended_rot_offset = quat_multiply(smplx_to_mujoco, site_local_inv)

        print(f"  Recommended rot_offset (wxyz): [{recommended_rot_offset[0]:.6f}, "
              f"{recommended_rot_offset[1]:.6f}, {recommended_rot_offset[2]:.6f}, "
              f"{recommended_rot_offset[3]:.6f}]")
        print()

        results[site_name] = {
            "world_pos": site_pos.tolist(),
            "world_quat": site_quat_wxyz.tolist(),
            "local_quat": site_local_quat.tolist(),
            "is_identity": bool(is_identity),
            "recommended_rot_offset": recommended_rot_offset.tolist()
        }

    return results


def generate_ik_config(results):
    """Generate IK config JSON based on analysis."""

    # Map site names to SMPLX body names
    site_to_smplx = {
        "pelvis_mimic": "pelvis",
        "left_hip_mimic": "left_hip",
        "left_knee_mimic": "left_knee",
        "left_ankle_mimic": "left_ankle",
        "left_foot_mimic": "left_foot",
        "right_hip_mimic": "right_hip",
        "right_knee_mimic": "right_knee",
        "right_ankle_mimic": "right_ankle",
        "right_foot_mimic": "right_foot",
    }

    # Default weights
    weights = {
        "pelvis_mimic": (100, 10),  # high position and rotation weight
        "left_hip_mimic": (0, 10),  # rotation only
        "left_knee_mimic": (50, 10),
        "left_ankle_mimic": (50, 10),
        "left_foot_mimic": (1, 0),  # minimal weight
        "right_hip_mimic": (0, 10),
        "right_knee_mimic": (50, 10),
        "right_ankle_mimic": (50, 10),
        "right_foot_mimic": (1, 0),
    }

    ik_match_table = {}
    for site_name, smplx_name in site_to_smplx.items():
        if site_name in results:
            rot_offset = results[site_name]["recommended_rot_offset"]
            pos_weight, rot_weight = weights.get(site_name, (50, 10))

            # Position offset (small adjustments if needed)
            if "foot" in site_name:
                pos_offset = [-0.02, 0.0, 0.0] if "left" in site_name else [0.02, 0.0, 0.0]
            elif "pelvis" in site_name:
                pos_offset = [0.0, 0.0, 0.045]  # slight Z offset
            else:
                pos_offset = [0.0, 0.0, 0.0]

            ik_match_table[site_name] = [
                smplx_name,
                pos_weight,
                rot_weight,
                pos_offset,
                rot_offset
            ]

    config = {
        "robot_root_name": "waist",
        "human_root_name": "pelvis",
        "ground_height": 0.0,
        "human_height_assumption": 1.8,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": {
            "pelvis": 0.440,
            "left_hip": 0.440,
            "right_hip": 0.440,
            "left_knee": 0.428,
            "right_knee": 0.428,
            "left_ankle": 0.440,
            "right_ankle": 0.440,
            "left_foot": 0.30,
            "right_foot": 0.30
        },
        "ik_match_table1": ik_match_table,
        "ik_match_table2": ik_match_table  # Use same for now
    }

    return config


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find wildrobot.xml
    script_dir = Path(__file__).parent
    xml_candidates = [
        script_dir.parent / "assets" / "wildrobot.xml",
        Path.home() / "projects" / "wildrobot" / "assets" / "wildrobot.xml",
    ]

    xml_path = None
    for candidate in xml_candidates:
        if candidate.exists():
            xml_path = str(candidate)
            break

    if xml_path is None:
        print("Could not find wildrobot.xml")
        sys.exit(1)

    print(f"Analyzing: {xml_path}")
    print()

    results = analyze_mimic_bodies(xml_path)

    print("=" * 70)
    print("Generated IK Config")
    print("=" * 70)
    print()

    config = generate_ik_config(results)
    print(json.dumps(config, indent=4))

    # Save to file
    output_path = script_dir.parent / "configs" / "smplx_to_wildrobot_computed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\nSaved to: {output_path}")
