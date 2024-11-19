import random

import numpy as np


class GridWorld:
    def __init__(self):
        self.grid_size = (3, 4)
        self.num_actions = 4
        self.rewards = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        self.start_state = (2, 0)

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
        reward = self.rewards[row, col]
        return next_state, reward


def monte_carlo(grid_world, num_episodes, gamma=1.0):
    returns_sum = np.zeros(grid_world.grid_size)
    returns_count = np.zeros(grid_world.grid_size)
    V = np.zeros(grid_world.grid_size)

    for _ in range(num_episodes):
        episode = generate_episode(grid_world)
        visited_states = set()
        for t, (state, action, reward) in enumerate(episode):
            if state not in visited_states:
                visited_states.add(state)
                G = sum([gamma ** 1 * step[2] for i, step in enumerate(episode[t:])])
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V


def generate_episode(grid_world):
    episode = []
    state = grid_world.start_state
    done = False

    while not done:
        action = random.choice(range(grid_world.num_actions))
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))

        if reward == 1:
            done = True
        else:
            state = next_state

    return episode


grid_world = GridWorld()

num_episodes = 1000
V = monte_carlo(grid_world, num_episodes)

print("Estimate State-Value Function:")
print(V)