import numpy as np
from collections import defaultdict
from Grid_world_2_1 import GridWorld

def off_policy_mc_prediction(grid_world, behavior_policy, target_policy, num_episodes, gamma=1.0):
    V = defaultdict(float)
    C = defaultdict(float)  # Accumulate weights

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
        # Lấy hành động từ chính sách hành vi
        action = np.random.choice(grid_world.num_actions, p=behavior_policy[state]) 
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3):  # Trạng thái kết thúc
            break
        state = next_state
    return episode

def random_policy(grid_world):
    """ Tạo một chính sách ngẫu nhiên cho tất cả các trạng thái """
    policy = {}
    for row in range(grid_world.grid_size[0]):
        for col in range(grid_world.grid_size[1]):
            policy[(row, col)] = np.ones(grid_world.num_actions) / grid_world.num_actions
    return policy

# Hàm chính để thực hiện
if __name__ == "__main__":
    grid_world = GridWorld()
    num_episodes = 1000
    
    behavior_policy = random_policy(grid_world)
    target_policy = random_policy(grid_world)

    V = off_policy_mc_prediction(grid_world, behavior_policy, target_policy, num_episodes)

    print("Estimated State Values Function:")
    for state, value in V.items():
        print(f"State {state}: {value:.2f}")
