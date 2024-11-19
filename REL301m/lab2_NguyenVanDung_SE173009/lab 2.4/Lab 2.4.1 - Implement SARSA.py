import numpy as np
import random


class GridWorld:
    def __init__(self):
        self.grid_size = (3, 4)
        self.num_actions = 4
        self.start_state = (0, 0)
        self.goal_state = (2, 3)

    def step(self, state, action):
        row, col = state
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.grid_size[1] - 1, col + 1)

        next_state = (row, col)
        reward = 0
        if next_state == self.goal_state:
            reward = 1
        return next_state, reward


def epsilon_greedy_policy(Q, state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state])


def sarsa(grid_world, num_episodes, alpha, gamma, epsilon):
    Q = np.zeros((grid_world.grid_size[0], grid_world.grid_size[1], grid_world.num_actions))

    for episode in range(num_episodes):
        state = grid_world.start_state
        action = epsilon_greedy_policy(Q, state, epsilon, grid_world.num_actions)

        while state != grid_world.goal_state:
            next_state, reward = grid_world.step(state, action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, grid_world.num_actions)

            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

    return Q


grid_world = GridWorld()

num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

Q = sarsa(grid_world, num_episodes, alpha, gamma, epsilon)

print("Learned Q-value Function:")
print(Q)
