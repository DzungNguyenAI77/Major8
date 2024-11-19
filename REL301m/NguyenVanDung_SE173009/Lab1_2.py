import numpy as np


class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4  # Up, Down, Left, Right
        self.rewards = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])

    def get_reward(self, state):
        return self.rewards[state[0], state[1]]


class ValueFunction:
    def __init__(self, grid_size):
        self.values = np.zeros(grid_size)

    def update_value(self, state, new_value):
        self.values[state[0], state[1]] = new_value

    def get_update(self, state):
        return self.values[state[0], state[1]]


# Creat grid world environment
grid_world = GridWorld()

# Creat value function for grid world
value_function = ValueFunction(grid_world.grid_size)

# Initialize value function with reward
for i in range(grid_world.grid_size[0]):
    for j in range(grid_world.grid_size[1]):
        state = (i, j)
        value_function.update_value(state, grid_world.get_reward(state))

print("Initial value function:")
print(value_function.values)
