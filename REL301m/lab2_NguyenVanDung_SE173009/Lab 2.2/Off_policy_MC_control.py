import numpy as np
from collections import defaultdict
from Grid_World import GridWorld
from On_policy_MC_control_without_exploring_starts import epsilon_greedy_policy
from On_policy_MC_prediction import generate_episode_policy


def off_policy_mc_control(grid_world, num_episodes, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(grid_world.num_actions))
    C = defaultdict(lambda: np.zeros(grid_world.num_actions))
    target_policy = defaultdict(lambda: np.zeros(grid_world.num_actions))

    for _ in range(num_episodes):
        behavior_policy = defaultdict(
            lambda: np.ones(grid_world.num_actions) / grid_world.num_actions)
        episode = generate_episode_policy(grid_world, behavior_policy)
        G = 0
        W = 1.0

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            best_action = np.argmax(Q[state])
            target_policy[state] = np.eye(grid_world.num_actions)[best_action]

            if action != best_action:
                break
            W *= 1.0 / behavior_policy[state][action]

    return target_policy, Q


if __name__ == "__main__":
    grid_world = GridWorld()
    num_episodes = 1000

    target_policy, Q = off_policy_mc_control(grid_world, num_episodes)

    print("Target Policy:")
    for state in target_policy:
        action = np.argmax(target_policy[state])
        print(f"State {state}: Action {action}")

    print("\nEstimated Action Values (Q):")
    for state, q_values in Q.items():
        print(f"State {state}: {q_values}")