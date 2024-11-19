import numpy as np
from collections import defaultdict
from Grid_world_2_1 import GridWorld

def on_policy_mc_control(grid_world, num_episodes, gamma=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(grid_world.num_actions))
    returns_sum = defaultdict(lambda: np.zeros(grid_world.num_actions))
    returns_count = defaultdict(lambda: np.zeros(grid_world.num_actions))

    for _ in range(num_episodes):
        episode = generate_episode_epsilon_greedy(grid_world, Q, epsilon)
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

    policy = defaultdict(lambda: np.zeros(grid_world.num_actions))
    for state in Q:
        best_action = np.argmax(Q[state])
        policy[state] = np.eye(grid_world.num_actions)[best_action]

    return policy, Q

def generate_episode_epsilon_greedy(grid_world, Q, epsilon):
    episode = []
    state = grid_world.start_state
    while True:
        probs = epsilon_greedy_policy(Q[state], epsilon, grid_world.num_actions)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward = grid_world.step(state, action)
        episode.append((state, action, reward))
        if next_state == (2, 3):  # Terminal state
            break
        state = next_state
    return episode

def epsilon_greedy_policy(Q_s, epsilon, num_actions):
    policy_s = np.ones(num_actions) * epsilon / num_actions  
    best_action = np.argmax(Q_s)
    policy_s[best_action] += 1.0 - epsilon  
    return policy_s

if __name__ == "__main__":
    grid_world = GridWorld()
    num_episodes = 1000
    policy, Q = on_policy_mc_control(grid_world, num_episodes)

    print("Estimated State-Action Values Function (Q):")
    for state, actions in Q.items():
        print(f"State {state}: {actions}")

    print("\nDerived Policy:")
    for state, action_probs in policy.items():
        print(f"State {state}: {action_probs}")
