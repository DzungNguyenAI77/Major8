import numpy as np 
from sklearn.tree import DecisionTreeClassifier

class Gridworld:
    def __init__(self):
        self.grid_size = (3, 3)
        self.num_actions = 4
        self.start_state = (0, 0)
        self.goal_state = (2, 2)

    def step(self, state, action):
        row, col = state
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.grid_size[1] - 1, col + 1)
        next_state = (row, col)
        reward = 0
        if next_state == self.goal_state:
            reward = 1
        return next_state, reward

    def generate_training_data(self, num_samples):
        X = np.zeros((num_samples, 2))
        Y = np.zeros((num_samples,))
        for i in range(num_samples):
            state = (np.random.randint(self.grid_size[0]),
                     np.random.randint(self.grid_size[1]))
            action = np.random.randint(self.num_actions)
            next_state, _ = self.step(state, action)
            X[i] = state
            Y[i] = action
        return X, Y

grid_world = Gridworld()

num_samples = 1000
X_train, y_train = grid_world.generate_training_data(num_samples)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


def evaluate_policy(grid_world, model):
    state = grid_world.start_state
    total_reward = 0

    while state != grid_world.goal_state:
        state_features = np.array(state).reshape(1, -1)
        action = model.predict(state_features)[0]
        next_state, reward = grid_world.step(state, action)
        total_reward += reward
        state = next_state

    return total_reward

total_reward = evaluate_policy(grid_world, model)
print("Total reward obtained by learned policy:", total_reward)
