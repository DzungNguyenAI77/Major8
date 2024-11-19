import numpy as np


class EpsilonGreedySearch:
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.action_values = np.zeros(num_actions)
        self.actions_counts = np.zeros(num_actions)

    def get_action(self):
        if np.random.rand() < self.epsilon:
            # explore
            action = np.random.randint(0, self.num_actions)
        else:
            # exploit
            action = np.argmax(self.action_values)
        return action

    def update_value(self, action, reward):
        self.actions_counts[action] += 1
        step_size = 1 / self.actions_counts[action]
        self.action_values[action] = self.action_values[action] + step_size * (reward - self.action_values[action])


class MultiArmedBanditEnv:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_action_values = np.random.normal(0, 1, num_arms)

    def get_reward(self, action):
        return np.random.normal(self.true_action_values[action], 1)



# initialize:
num_arms = 5
num_steps = 1000
agent = EpsilonGreedySearch(num_arms)

# Interaction loop
bandit = MultiArmedBanditEnv(num_arms)
total_reward = 0
for step in range(num_steps):
    action = agent.get_action()
    reward = bandit.get_reward(action)
    agent.update_value(action, reward)
    total_reward += reward

print(f"Total reward obtain: {total_reward}")
print(f"Estimate action value: {agent.action_values}")

