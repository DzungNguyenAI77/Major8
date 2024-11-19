import numpy as np
from collections import defaultdict
from Grid_World import GridWorld


def off_policy_mc_prediction(grid_world, behavior_policy, target_policy, num_episodes, gamma=1.0):
    V = defaultdict(float)
    C = defaultdict(float)

    for _ in range(num_episodes):
        episode = generate_episode(grid_world, behavior_policy)
        G = 0
        W = 1.0

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state] += W
            V[state] += (W / C[state]) * (G - V[state])
            W *= 1.0 / behavior_policy[state][action]

    return V


def generate_episode(grid_world, behavior_policy):
    episode = []
    state = grid_world.start_state
    while True:
        action = np.random.choice(grid_world.num_actions, p=behavior_policy[state])
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3):
            break
        state = next_state
    return episode


def random_policy(grid_world):
    policy = {}
    for row in range(grid_world.grid_size[0]):
        for col in range(grid_world.grid_size[1]):
            policy[(row, col)] = np.ones(grid_world.num_actions) / grid_world.num_actions
    return policy


if __name__ == "__main__":
    grid_world = GridWorld()
    num_episodes = 1000

    behavior_policy = random_policy(grid_world)
    target_policy = random_policy(grid_world)

    V = off_policy_mc_prediction(grid_world, behavior_policy, target_policy, num_episodes)

    print("Estimated State Values Function:")
    for state, value in V.items():
        print(f"State {state}: {value:.2f}")
