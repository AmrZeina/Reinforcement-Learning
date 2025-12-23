# compare_iterations.py
import numpy as np
from Environment import GridWorld
from ValueIteration import value_iteration
from policy_iteration import policy_iteration

def run_comparison():
    cases = [
        (100, 110),
        (10, 100),
        (1, 10),
        (10, 15)
    ]
    
    print("=" * 60)
    print("COMPARISON: VALUE ITERATION vs POLICY ITERATION")
    print("=" * 60)
    
    for R1, R2 in cases:
        env = GridWorld(R1, R2)
        print(f"\nCase: R1 = {R1}, R2 = {R2}")
        print("-" * 60)
        
        # Run Value Iteration
        V_val, policy_val = value_iteration(env)
        env.display_grid("Value Iteration Policy", policy_val, 'policy')
        env.display_grid("Value Iteration Values", V_val, 'value')
        
        # Run Policy Iteration
        V_pol, policy_pol = policy_iteration(env)
        env.display_grid("Policy Iteration Policy", policy_pol, 'policy')
        env.display_grid("Policy Iteration Values", V_pol, 'value')
        
        # Intuition
        print("Intuition:", end=" ")
        if R2 > R1:
            print(f"R2 ({R2}) > R1 ({R1}), so both methods favor moving toward R2.")
        elif R1 > R2:
            print(f"R1 ({R1}) > R2 ({R2}), so both methods favor moving toward R1.")
        else:
            print("R1 and R2 are equal, so proximity and local rewards dominate.")
        print()


run_comparison()
