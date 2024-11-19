import numpy as np


class Bandit:
    def __init__(self, true_means):
        self.true_means = true_means

    def pull_arms(self, arm):
        return np.random.normal(self.true_means[arm], 1)


class DistributionModel:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.mean_rewards = np.zeros(num_arms)
        self.variance_rewards = np.ones(num_arms)
        self.counts = np.zeros(num_arms)

    def update_distribution(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]

        old_mean = self.mean_rewards[arm]
        new_mean = old_mean + (reward - old_mean) / n
        self.mean_rewards[arm] = new_mean

        if n > 1:
            old_variance = self.variance_rewards[arm]
            new_variance = ((n - 1) * old_variance + (reward - old_mean) * (reward - new_mean)) / n
            self.variance_rewards[arm] = new_variance


true_means = [1.0, 2.0]

bandit = Bandit(true_means)
distribution_model = DistributionModel(len(true_means))


num_pulls = 1000
for _ in range(num_pulls):
    arm = np.random.randint(len(true_means))
    reward = bandit.pull_arms(arm)
    distribution_model.update_distribution(arm, reward)


print("Updated Distribution Model:")
print(f"Mean Rewards: {distribution_model.mean_rewards}")
print(f"Variance of Rewards: {distribution_model.variance_rewards}")
