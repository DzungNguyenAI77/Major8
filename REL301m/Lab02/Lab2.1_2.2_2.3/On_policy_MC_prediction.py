import numpy as np
from collections import defaultdict
from Grid_world_2_1 import GridWorld  

def on_policy_mc_prediction(grid_world, policy, num_episodes, gamma=1.0):
    V = defaultdict(float)  
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for _ in range(num_episodes):
        episode = generate_episode_policy(grid_world, policy)
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

def generate_episode_policy(grid_world, policy):
    episode = []
    state = grid_world.start_state
    while True:
        action_probs = policy[state]  
        print(f"Current state: {state}, Policy: {action_probs}")  
        action = np.random.choice(len(action_probs), p=action_probs)
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3): 
            break
        state = next_state
    return episode

if __name__ == "__main__":
    grid_world = GridWorld()  
    num_episodes = 1000  

    policy = defaultdict(lambda: np.ones(grid_world.num_actions) / grid_world.num_actions)

    V = on_policy_mc_prediction(grid_world, policy, num_episodes, gamma=1.0)

    print("Estimated State-Value Function (V):")
    for state, value in V.items():
        print(f"State {state}: {value:.2f}")
