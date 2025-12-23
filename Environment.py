import numpy as np

class GridWorld:
    def __init__(self, R1, R2):
        self.rows, self.cols = 5, 5
        self.R1 = R1
        self.R2 = R2
        
        # Define actions
        self.actions = ['U', 'D', 'L', 'R']
        
        # Define movement vectors
        self.moves = {
            'U': (-1, 0), #moving up will decrease the row number by 1 and does not change the column number, so (-1,0)
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
        
        # Initialize reward grid
        self.rewards = self._create_reward_grid()
    
    def _create_reward_grid(self):
        """Create the 5x5 reward grid."""
        fixed_rewards = [
            [self.R1, 1, 0, -1, self.R2],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2]
        ]
        return np.array(fixed_rewards, dtype=float)
    
    def transition_prob(self, intended_action):
        """Return transition probabilities for an intended action."""
        probs = {}
        for a in self.actions:
            if a == intended_action:
                probs[a] = 0.7
            else:
                probs[a] = 0.1
        return probs
    
    def move_state(self, state, action):
        """Apply an action to a state and return the new state."""
        r, c = state
        dr, dc = self.moves[action]
        nr, nc = r + dr, c + dc #new row and new col
        
        # Check boundaries
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return (nr, nc)
        else:
            return (r, c)  # Hit wall, stay in place
    
    def get_reward(self, state):
        """Get the reward for a given state."""
        r, c = state
        return self.rewards[r, c]
    
    def display_grid(self, title, grid, fmt):
        """Display any grid in a readable format."""
        print(f"\n{title}:")
        print("-" * 30)
        for r in range(self.rows):
            for c in range(self.cols):
                if fmt == 'policy':
                    print(f"{grid[r, c]:^4}", end=" ")
                else:  # value
                    print(f"{grid[r, c]:7.2f}", end=" ")
            print()
        print()