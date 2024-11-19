import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_monte_carlo(episodes, is_training=True, gamma=0.9, epsilon=0.9, render=False):
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human' if render else None)

    if is_training:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    else:
        with open("blackjack_monte.pkl", "rb") as f:
            q = pickle.load(f)

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    rewards_per_episode = np.zeros(episodes)

    rng = np.random.default_rng()

    for i in range(episodes):
        state = env.reset()[0]
        episode = []
        terminated = False

        # Tạo một tập
        while not terminated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = new_state

        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        rewards_per_episode[i] = episode[-1][2]

    env.close()

    sum_reward = np.zeros(episodes)
    for t in range(episodes):
        sum_reward[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(sum_reward)
    plt.title('Reward episode (Monte Carlo - Blackjack)')
    plt.xlabel('Episode')
    plt.ylabel('Sum reward')
    plt.savefig('blackjack_monte.png')
    plt.show()


    if is_training:
        with open("blackjack_monte.pkl", "wb") as f:
            pickle.dump(dict(q), f)


if __name__ == '__main__':
    run_monte_carlo(10000, is_training=True, gamma=0.9, epsilon=0.1, render=False)
