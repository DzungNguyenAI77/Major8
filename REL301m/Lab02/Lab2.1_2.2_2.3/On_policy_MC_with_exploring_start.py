import numpy as np
from collections import defaultdict
from Grid_world_2_1 import GridWorld  


def on_policy_mc_exploring_starts(grid_world, num_episodes, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(grid_world.num_actions))  # Action-value function
    returns_sum = defaultdict(lambda: np.zeros(grid_world.num_actions))
    returns_count = defaultdict(lambda: np.zeros(grid_world.num_actions))

    policy = defaultdict(lambda: np.ones(grid_world.num_actions) / grid_world.num_actions)

    for _ in range(num_episodes):
        episode = generate_episode_exploring_starts(grid_world)
        G = 0
        visited_state_actions = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]

        for state in Q:
            best_action = np.argmax(Q[state])
            policy[state] = np.eye(grid_world.num_actions)[best_action]

    return policy, Q

def generate_episode_exploring_starts(grid_world):
    episode = []
    state = (np.random.randint(0, grid_world.grid_size[0]), np.random.randint(0, grid_world.grid_size[1]))
    action = np.random.randint(0, grid_world.num_actions)
    while True:
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3):  # Terminal state
            break
        state = next_state
        action = np.random.choice(grid_world.num_actions)
    return episode

if __name__ == "__main__":
    grid_world = GridWorld()
    num_episodes = 1000
    V = on_policy_mc_exploring_starts(grid_world, num_episodes)

    print("Estimated State Values Function:")
    for row in range(grid_world.grid_size[0]):
        for col in range(grid_world.grid_size[1]):
            if (row, col) in V:
                print(f"State {row, col}: {V[(row, col)]:.2f}")
            else:
                print(f"State {row, col}: Not visited")
