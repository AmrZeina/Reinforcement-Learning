import numpy as np

GRID = np.array([[None, 1, 0, -1, None],
        [2, 1, 0, -1, -2],
        [2, 1, 0, -1, -2],
        [2, 1, 0, -1, -2],
        [2, 1, 0, -1, -2]])

TransitionMat = np.array([[0.7, 0.1, 0.1, 0.1],
                          [0.1, 0.7, 0.1, 0.1],
                          [0.1, 0.1, 0.7, 0.1],
                          [0.1, 0.1, 0.1, 0.7]])

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class Env:
    def __init__(self, R1 = None, R2 = None):
        self.grid = GRID
        self.grid[0, 0] = R1
        self.grid[0, 4] = R2

        self.terminal_state = ([0,0],[0,4])
        
    def isTerminal(self, row, col):
        if self.terminal_state.__contains__([row,col]):
            return True
        return False