"""Test contact alternation ratio tracking."""


def test_alternation_logic():
    """Test the alternation tracking logic with simulated contact patterns."""
    
    # Simulate perfect alternating gait: L → R → L → R → L → R
    # Transitions at steps: 0(L), 5(R), 10(L), 15(R), 20(L), 25(R)
    
    last_contact_foot = 0.0
    alternation_cycles = 0.0
    total_transitions = 0.0
    
    transitions = [
        (1, False),  # Left contact at step 1
        (5, True),   # Right contact at step 5
        (10, False), # Left contact at step 10
        (15, True),  # Right contact at step 15
        (20, False), # Left contact at step 20
        (25, True),  # Right contact at step 25
    ]
    
    for step, is_right in transitions:
        left_transition = not is_right
        right_transition = is_right
        
        # Same logic as in walk_env.py
        new_last_contact = last_contact_foot
        new_cycles = alternation_cycles
        new_transitions = total_transitions
        
        # Left foot contact transition
        if left_transition:
            if last_contact_foot == 2.0:  # Was right, now left
                new_last_contact = 1.0
                new_cycles = new_cycles + 0.5  # Half cycle
            elif last_contact_foot == 0.0:  # First ever
                new_last_contact = 1.0
            new_transitions = new_transitions + 1.0
        
        # Right foot contact transition
        if right_transition:
            if last_contact_foot == 1.0:  # Was left, now right
                new_last_contact = 2.0
                new_cycles = new_cycles + 0.5  # Half cycle
            elif last_contact_foot == 0.0:  # First ever
                new_last_contact = 2.0
            new_transitions = new_transitions + 1.0
        
        last_contact_foot = new_last_contact
        alternation_cycles = new_cycles
        total_transitions = new_transitions
        
        # Calculate ratio
        if new_transitions > 2.0:
            expected_cycles = (new_transitions - 1.0) / 2.0
            alternation_ratio = new_cycles / expected_cycles
        else:
            alternation_ratio = 0.0
        
        print(f"Step {step:2d} ({'R' if is_right else 'L'}): "
              f"last={int(last_contact_foot)} cycles={alternation_cycles:.1f} "
              f"transitions={int(total_transitions)} ratio={alternation_ratio:.2%}")
    
    # Perfect alternation should give ratio of 1.0 (100%)
    print(f"\n✅ Final alternation ratio: {alternation_ratio:.2%}")
    print(f"   Cycles: {alternation_cycles:.1f}, Transitions: {int(total_transitions)}")
    print(f"   Expected: 1.00 (100%)")
    
    assert abs(alternation_ratio - 1.0) < 0.01, f"Expected 100%, got {alternation_ratio:.2%}"
    
    # Test imperfect gait with double contacts: L → L → R → L → R
    print("\n=== Testing imperfect gait (double left contact) ===")
    last_contact_foot = 0.0
    alternation_cycles = 0.0
    total_transitions = 0.0
    
    bad_transitions = [
        (1, False),  # Left
        (3, False),  # Left again (bad!)
        (7, True),   # Right
        (12, False), # Left
        (17, True),  # Right
    ]
    
    for step, is_right in bad_transitions:
        left_transition = not is_right
        right_transition = is_right
        
        new_last_contact = last_contact_foot
        new_cycles = alternation_cycles
        new_transitions = total_transitions
        
        if left_transition:
            if last_contact_foot == 2.0:
                new_last_contact = 1.0
                new_cycles = new_cycles + 0.5
            elif last_contact_foot == 0.0:
                new_last_contact = 1.0
            # Note: If last_contact_foot == 1.0 (left→left), no cycle increment
            new_transitions = new_transitions + 1.0
        
        if right_transition:
            if last_contact_foot == 1.0:
                new_last_contact = 2.0
                new_cycles = new_cycles + 0.5
            elif last_contact_foot == 0.0:
                new_last_contact = 2.0
            new_transitions = new_transitions + 1.0
        
        last_contact_foot = new_last_contact
        alternation_cycles = new_cycles
        total_transitions = new_transitions
        
        if new_transitions > 2.0:
            expected_cycles = (new_transitions - 1.0) / 2.0
            alternation_ratio = new_cycles / expected_cycles
        else:
            alternation_ratio = 0.0
        
        print(f"Step {step:2d} ({'R' if is_right else 'L'}): "
              f"last={int(last_contact_foot)} cycles={alternation_cycles:.1f} "
              f"transitions={int(total_transitions)} ratio={alternation_ratio:.2%}")
    
    print(f"\n⚠️  Final alternation ratio: {alternation_ratio:.2%}")
    print(f"   Cycles: {alternation_cycles:.1f}, Transitions: {int(total_transitions)}")
    print(f"   Expected: <100% (due to L→L bad transition)")
    
    # Should be less than perfect due to double contact
    assert alternation_ratio < 1.0, f"Expected <100%, got {alternation_ratio:.2%}"
    assert alternation_ratio > 0.5, f"Expected >50%, got {alternation_ratio:.2%}"


if __name__ == "__main__":
    test_alternation_logic()
    print("\n✅ All tests passed!")
