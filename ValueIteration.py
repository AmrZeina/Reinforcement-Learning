# value_iteration.py
import numpy as np
from Environment import GridWorld

def value_iteration(env, gamma=0.95, epsilon=1e-4):
    """
    Perform value iteration on the given GridWorld environment.
    
    Args:
        env: GridWorld environment
        gamma: Discount factor
        epsilon: Convergence threshold
        
    Returns:
        V: Value function (numpy array)
        policy: Optimal policy (numpy array of strings)
    """
    # Initialize value function to rewards (as specified in assignment)
    V = env.rewards.copy()
    
    while True:
        max_diff = 0    #if we reach a difference of 0 , that is the max and we can not enhance the value fn more
        new_V = np.zeros((env.rows, env.cols))
        
        # For each state
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                best_value = -float('inf')
                
                # For each possible action
                for action in env.actions:
                    expected_value = 0
                    
                    # Get transition probabilities for this intended action
                    probs = env.transition_prob(action)
                    
                    # Sum over all possible actual actions
                    for actual_action, prob in probs.items():
                        next_state = env.move_state(state, actual_action)
                        nr, nc = next_state
                        
                        # Bellman update: R(s') + gamma * V(s')
                        expected_value += prob * (env.get_reward(next_state) + gamma * V[nr, nc])
                    
                    # Track the best value
                    if expected_value > best_value:
                        best_value = expected_value
                
                # Update value for this state
                new_V[r, c] = best_value
                max_diff = max(max_diff, abs(new_V[r, c] - V[r, c]))
        
        # Check for convergence
        V = new_V
        if max_diff < epsilon:
            break
    
    # Extract optimal policy from converged value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy

def extract_policy(env, V, gamma):
    """Extract optimal policy from value function."""
    policy = np.empty((env.rows, env.cols), dtype=str)
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            best_action = None
            best_value = -float('inf')
            
            for action in env.actions:
                expected_value = 0
                probs = env.transition_prob(action)
                
                for actual_action, prob in probs.items():
                    next_state = env.move_state(state, actual_action)
                    nr, nc = next_state
                    expected_value += prob * (env.get_reward(next_state) + gamma * V[nr, nc])
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            policy[r, c] = best_action
    
    return policy

def run_all_cases():
    """Run value iteration for all cases """
    cases = [
        (100, 110),
        (10, 100),
        (1, 10),
        (10, 15)
    ]
    
    print("=" * 60)
    print("VALUE ITERATION RESULTS")
    print("=" * 60)
    
    for R1, R2 in cases:
        # Create environment
        env = GridWorld(R1, R2)
        
        print(f"\nCase: R1 = {R1}, R2 = {R2}")
        print("-" * 40)
        
        # Run value iteration
        V, policy = value_iteration(env)
        
        # Display results
        env.display_grid("Optimal Policy", policy, 'policy')
        env.display_grid("Value Function", V, 'value')
        
        # explanation
        print(f"Intuition: ", end="")
        if R2 > R1:
            print(f"R2 ({R2}) > R1 ({R1}), so policy generally favors moving right/toward R2.")
        elif R1 > R2:
            print(f"R1 ({R1}) > R2 ({R2}), so policy generally favors moving left/toward R1.")
        else:
            print(f"R1 and R2 are equal, so policy depends on proximity and other rewards.")
        print()

if __name__ == "__main__":
    run_all_cases()