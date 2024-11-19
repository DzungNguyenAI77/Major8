import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)

        # Randomly choose  an action for exploration
        # Choose the greedy action for exploitation
    def select_action(self):
        # Exploration
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        # Exploitation
        else:
            action = np.argmax(self.action_values)
        return action

    # Update action-value estimate using incremental update rule
    def update_value(self, action, reward):
        self.action_counts[action] = self.action_counts[action] + 1
        self.action_values[action] = (1 / self.action_counts[action]) * (reward - self.action_values[action])


# Creat simple multi-armed bandit environment
class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_action_values = np.random.normal(0, 1, num_arms)

    def get_reward(self, action):
        return np.random.normal(self.true_action_values[action], 1)


# Initialize
num_arms = 5
num_steps = 1000
agent = EpsilonGreedyAgent(num_arms)

# Interaction loop
bandit = MultiArmedBandit(num_arms)
total_reward = 0
for step in range(num_steps):
    action = agent.select_action()
    reward = bandit.get_reward(action)
    agent.update_value(action, reward)
    total_reward += reward

print("Total reward obtain: ", total_reward)
print("Estimate action value: ", agent.action_values)


