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
        if action == 0: 
            row = max(0, row - 1)
        elif action == 1:  
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:  
            col = max(0, col - 1)
        elif action == 3: 
            col = min(self.grid_size[1] - 1, col + 1)
        next_state = (row, col)
        reward = self.rewards[row, col]
        return next_state, reward

def generate_episode(grid_world):
    episode = []
    state = grid_world.start_state
    while True:
        action = np.random.choice(grid_world.num_actions) 
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3):  
            break
        state = next_state
    return episode

def monte_carlo(grid_world, num_episodes, gamma=1.0):
    V = np.zeros(grid_world.grid_size)  
    returns_sum = np.zeros(grid_world.grid_size)
    returns_count = np.zeros(grid_world.grid_size)

    for _ in range(num_episodes):
        episode = generate_episode(grid_world)
        G = 0
        states_visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in states_visited:
                states_visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V

grid_world = GridWorld()
num_episodes = 1000
V = monte_carlo(grid_world, num_episodes)

print("Estimated State Values Function:")
print(V)
