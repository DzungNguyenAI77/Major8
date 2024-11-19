import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True,
                   render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open("frozen_lake8x8.pkl", "rb") as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)    

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards=0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards
    env.close()

    if is_training:
        f = open('frozen_lake8x8.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'frozen_lake8x8.png')


if __name__ == '__main__':
    # số episode muốn chạy
    # is_training: True: train và lưu file pkl
    #              False: không train, lấy file pkl trước đó để chạy
    # render: True: hiện trò chơi
    #         False: không hiện trò chơi
    run(1, is_training=False, render=True)
