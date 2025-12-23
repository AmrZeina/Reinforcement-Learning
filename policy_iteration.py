import numpy as np
import random
from Environment import GridWorld

def policy_iteration(env, gamma=0.95, epsilon=1e-4):  
    """
    Policy Iteration Algorithm to find the optimal policy and value function
    for a GridWorld environment.

    Parameters:
    - env: The environment (GridWorld object)
    - gamma: Discount factor for future rewards
    - epsilon: Threshold for convergence in policy evaluation
    """
    
    # 1) Initialize a random policy
    # Each state gets a random action from the list of possible actions
    policy = np.random.choice(env.actions, size=(env.rows, env.cols)) 
    
    V = env.rewards.copy()

    # 3) Outer loop: Repeat until the policy is stable
    while True:
        # 3a) POLICY EVALUATION: Evaluate the current policy until value function V converges
        while True:
            max_diff = 0  # track the largest change in V across all states
            new_V = V.copy()  # copy current values to update them

            # Go through each state in the grid
            for r in range(env.rows):
                for c in range(env.cols):
                    state = (r, c)         
                    action = policy[r, c]  
                    expected_value = 0    
                    probs = env.transition_prob(action)                    
                    # Calculate expected value for this state
                    for actual_action, prob in probs.items():
                        next_state = env.move_state(state, actual_action)  # resulting state
                        nr, nc = next_state
                        # sum over all possible next states: P * (R + gamma * V(next_state))
                        expected_value += prob * (
                            env.get_reward(next_state) + gamma * V[nr, nc]
                        )

                    # Update the new value function for this state
                    new_V[r, c] = expected_value

                    # Track the maximum change across all states
                    max_diff = max(max_diff, abs(new_V[r, c] - V[r, c]))

            # Update value function after evaluating all states
            V = new_V

            # Check convergence: if max change < epsilon, stop evaluating
            if max_diff < epsilon:
                break

        # 3b) POLICY IMPROVEMENT: Try to improve policy by choosing greedy actions w.r.t current V
        policy_stable = True  # assume policy is stable initially

        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                old_action = policy[r, c]  # save current action to check for changes

                best_action = None
                best_value = -float('inf')  # start with worst possible value

                # Evaluate all possible actions to find the best one
                for action in env.actions:
                    expected_value = 0
                    probs = env.transition_prob(action)

                    for actual_action, prob in probs.items():
                        next_state = env.move_state(state, actual_action)
                        nr, nc = next_state
                        expected_value += prob * (
                            env.get_reward(next_state) + gamma * V[nr, nc]
                        )

                    # Update best action if this action has higher expected value
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action

                # Update policy with the best action found
                policy[r, c] = best_action

                # If any action changes → policy is not stable yet
                if old_action != best_action:
                    policy_stable = False

        # If policy didn't change at all → it is optimal, exit loop
        if policy_stable:
            break

    return V, policy

