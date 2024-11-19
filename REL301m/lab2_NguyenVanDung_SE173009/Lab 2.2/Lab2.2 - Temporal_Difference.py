import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.start_state = (0, 0)
        self.goal_state = (2, 2)
        self.num_actions = 4

    def step(self, state, action):
        x, y = state

        if action == 0:  # Up
            next_state = (max(x - 1, 0), y)
        elif action == 1:  # Down
            next_state = (min(x + 1, self.grid_size[0] - 1), y)
        elif action == 2:  # Left
            next_state = (x, max(y - 1, 0))
        elif action == 3:  # Right
            next_state = (x, min(y + 1, self.grid_size[1] - 1))

        reward = -1
        done = False

        if next_state == self.goal_state:
            reward = 0
            done = True

        return next_state, reward, done


def td_learning(grid_world, num_episodes, alpha, gamma):
    values = np.zeros(grid_world.grid_size)

    for _ in range(num_episodes):
        state = grid_world.start_state
        done = False

        while not done:
            action = np.random.choice(grid_world.num_actions)
            next_state, reward, done = grid_world.step(state, action)

            x, y = state
            next_x, next_y = next_state
            values[x, y] += alpha + (reward + gamma * values[next_x, next_y] - values[x, y])

            state = next_state

    return values


grid_world = GridWorld()

num_episodes = 1000
alpha = 0.1
gamma = 0.9
values = td_learning(grid_world, num_episodes, alpha, gamma)

print("Value function: ")
print(values)
