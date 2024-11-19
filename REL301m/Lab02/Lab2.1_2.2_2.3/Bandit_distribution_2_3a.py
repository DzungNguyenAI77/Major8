import numpy as np

class Bandit:
    def __init__(self, true_means):
        self.true_means = true_means
        
    def pull_arm(self, arm):
        return np.random.normal(self.true_means[arm], 1)
    

class DistributionModel:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.means_rewards = np.zeros(num_arms)  
        self.variance_rewards = np.ones(num_arms)  
        self.counts = np.zeros(num_arms) 
        
    def update_distribution(self, arm, reward):
        self.counts[arm] += 1
        
        alpha = 1.0 / self.counts[arm]
        self.means_rewards[arm] += alpha * (reward - self.means_rewards[arm])
        
        if self.counts[arm] > 1:
            self.variance_rewards[arm] = ((self.counts[arm] - 1) * self.variance_rewards[arm] + 
                                          (reward - self.means_rewards[arm]) ** 2) / self.counts[arm]

true_means = [1.0, 2.0]

bandit = Bandit(true_means)

distribution_model = DistributionModel(num_arms=2)

num_pulls = 1000

for _ in range(num_pulls):
    arm = np.random.choice(2)  
    
    reward = bandit.pull_arm(arm)
    
    distribution_model.update_distribution(arm, reward)

print("Updated Distribution Model:")
print("Mean Rewards:", distribution_model.means_rewards)
print("Variance Rewards:", distribution_model.variance_rewards)
