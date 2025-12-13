#!/usr/bin/env python3
"""Analyze joint symmetry and non-orthogonal ranges in WildRobot model"""
import mujoco
import numpy as np

def analyze_joint_ranges(model_path: str):
    """Analyze joint ranges and check for mirror symmetry issues"""
    model = mujoco.MjModel.from_xml_path(model_path)

    print("=" * 80)
    print("JOINT RANGE ANALYSIS - WildRobot Humanoid")
    print("=" * 80)
    print()

    # Group joints by type
    right_joints = []
    left_joints = []
    waist_joints = []

    for i in range(model.njnt):
        # Skip the freejoint (first 7 qpos)
        if i == 0:
            continue

        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]
        type_name = ['free', 'ball', 'slide', 'hinge'][jnt_type]

        if jnt_type != 3:  # Only analyze hinge joints
            continue

        range_min = model.jnt_range[i, 0]
        range_max = model.jnt_range[i, 1]

        if 'right' in jnt_name:
            right_joints.append((jnt_name, range_min, range_max))
        elif 'left' in jnt_name:
            left_joints.append((jnt_name, range_min, range_max))
        elif 'waist' in jnt_name:
            waist_joints.append((jnt_name, range_min, range_max))

    # Print right leg joints
    print("RIGHT LEG JOINTS:")
    print("-" * 80)
    print(f"{'Joint Name':<30} {'Min (rad)':<15} {'Max (rad)':<15} {'Min (deg)':<15} {'Max (deg)':<15}")
    print("-" * 80)
    for name, min_val, max_val in right_joints:
        print(f"{name:<30} {min_val:<15.4f} {max_val:<15.4f} {np.degrees(min_val):<15.2f} {np.degrees(max_val):<15.2f}")
    print()

    # Print left leg joints
    print("LEFT LEG JOINTS:")
    print("-" * 80)
    print(f"{'Joint Name':<30} {'Min (rad)':<15} {'Max (rad)':<15} {'Min (deg)':<15} {'Max (deg)':<15}")
    print("-" * 80)
    for name, min_val, max_val in left_joints:
        print(f"{name:<30} {min_val:<15.4f} {max_val:<15.4f} {np.degrees(min_val):<15.2f} {np.degrees(max_val):<15.2f}")
    print()

    # Print waist joint
    print("WAIST JOINT:")
    print("-" * 80)
    print(f"{'Joint Name':<30} {'Min (rad)':<15} {'Max (rad)':<15} {'Min (deg)':<15} {'Max (deg)':<15}")
    print("-" * 80)
    for name, min_val, max_val in waist_joints:
        print(f"{name:<30} {min_val:<15.4f} {max_val:<15.4f} {np.degrees(min_val):<15.2f} {np.degrees(max_val):<15.2f}")
    print()

    # Analyze symmetry
    print("=" * 80)
    print("SYMMETRY ANALYSIS")
    print("=" * 80)
    print()

    # Match right and left joints
    joint_pairs = [
        ('hip_pitch', 'right_hip_pitch', 'left_hip_pitch'),
        ('hip_roll', 'right_hip_roll', 'left_hip_roll'),
        ('knee_pitch', 'right_knee_pitch', 'left_knee_pitch'),
        ('ankle_pitch', 'right_ankle_pitch', 'left_ankle_pitch'),
        ('foot_roll', 'right_foot_roll', 'left_foot_roll'),
    ]

    for pair_name, right_name, left_name in joint_pairs:
        # Find ranges
        right_range = None
        left_range = None

        for name, min_val, max_val in right_joints:
            if name == right_name:
                right_range = (min_val, max_val)

        for name, min_val, max_val in left_joints:
            if name == left_name:
                left_range = (min_val, max_val)

        if right_range is None or left_range is None:
            print(f"{pair_name}: MISSING JOINT")
            continue

        print(f"\n{pair_name.upper()}:")
        print(f"  Right: [{right_range[0]:>7.4f}, {right_range[1]:>7.4f}] rad = [{np.degrees(right_range[0]):>7.2f}, {np.degrees(right_range[1]):>7.2f}] deg")
        print(f"  Left:  [{left_range[0]:>7.4f}, {left_range[1]:>7.4f}] rad = [{np.degrees(left_range[0]):>7.2f}, {np.degrees(left_range[1]):>7.2f}] deg")

        # Check if orthogonal (symmetric)
        # For true mirror symmetry: right_min = -left_max and right_max = -left_min
        is_orthogonal = np.allclose(right_range[0], -left_range[1], atol=0.001) and \
                       np.allclose(right_range[1], -left_range[0], atol=0.001)

        # Check if identical
        is_identical = np.allclose(right_range[0], left_range[0], atol=0.001) and \
                      np.allclose(right_range[1], left_range[1], atol=0.001)

        if is_orthogonal:
            print(f"  Status: ✅ ORTHOGONAL (Perfect mirror symmetry)")
        elif is_identical:
            print(f"  Status: ⚠️  IDENTICAL (Same ranges, NOT mirrored)")
        else:
            print(f"  Status: ❌ NON-ORTHOGONAL (Asymmetric ranges)")
            print(f"  Expected left for orthogonal: [{-right_range[1]:>7.4f}, {-right_range[0]:>7.4f}] rad")
            print(f"  Difference: min_diff={left_range[0] - (-right_range[1]):.4f}, max_diff={left_range[1] - (-right_range[0]):.4f}")

    print("\n" + "=" * 80)
    print("IMPACT ON TRAINING")
    print("=" * 80)
    print("""
Non-orthogonal (asymmetric) joint ranges impact RL training in several ways:

1. **Policy Asymmetry**
   - Network must learn different action mappings for left/right legs
   - Harder to learn stable gaits (walking requires symmetric patterns)
   - Increases sample complexity (more training time)

2. **Normalization Issues**
   - Action normalization typically assumes [-1, 1] → [min, max]
   - Asymmetric ranges break this assumption
   - May need per-joint normalization scaling

3. **Exploration Difficulty**
   - Symmetric gaits (walking, running) are easier to discover with symmetric ranges
   - Asymmetric ranges bias the policy toward asymmetric behaviors
   - May converge to limping/hopping instead of bipedal walking

4. **Sim2Real Transfer**
   - If physical robot has symmetric hardware but model has asymmetric ranges
   - Creates artificial constraints that don't exist in reality
   - Policy may not transfer well to hardware

RECOMMENDATIONS:

✅ **Best Practice: Orthogonal (Mirrored) Ranges**
   - Right hip_roll: [-0.17, 1.57] → Left hip_roll: [-1.57, 0.17]
   - This is TRUE mirror symmetry (flip sign)

⚠️  **Current Model Status:**
   - Check output above to see which joints are non-orthogonal
   - Consider fixing in CAD/URDF if hardware allows symmetric ranges

🔧 **Workarounds if ranges must be asymmetric:**
   - Use per-joint action normalization
   - Add regularization term to encourage symmetric gaits
   - Use AMP with symmetric motion data
   - Train longer to overcome exploration difficulty
""")

if __name__ == '__main__':
    analyze_joint_ranges('assets/wildrobot.xml')
