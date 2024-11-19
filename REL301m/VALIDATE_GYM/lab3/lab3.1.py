import numpy as np
from sklearn.tree import DecisionTreeClassifier

class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4
        self.start_state = (0, 0)
        self.goal_state = (2, 2)

    def step(self, state, action):
        row, col = state
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.grid_size[1] - 1, col + 1)

        next_state = (row, col)
        reward = 0
        if next_state == self.goal_state:
            reward = 1
        return reward, next_state
    

def generate_training_data(grid_world, num_samples):
    X = np.zeros((num_samples, 2))
    y = np.zeros((num_samples, ))

    for i in range(num_samples):
        state = (np.random.randint(grid_world.grid_size[0]),
                 np.random.randint(grid_world.grid_size[1]))
        action = np.random.randint(grid_world.grid_size.num_actions)

        next_state,_ = grid_world.step(state, action)

        X[i] = state
        y[i] = action
    return X, y

grid_world = GridWorld()

num_samples = 1000
X_train, y_train = generate_training_data(grid_world, num_samples)



model = DecisionTreeClassifier()
