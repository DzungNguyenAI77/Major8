import numpy as np

class GridWorld:
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

class ActorCritic:
    def __init__(self, num_actions, alpha_actor, alpha_critic, gamma):
        self.num_actions = num_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.actor_params = np.zeros((3, 3, num_actions))
        self.critic_values = np.zeros((3, 3))

    def select_action(self, state):
        state_row, state_col = state
        action_probs = self.softmax(self.actor_params[state_row, state_col])
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        state_row, state_col = state
        next_state_row, next_state_col = next_state

        td_target = reward + self.gamma * self.critic_values[next_state_row, next_state_col]
        td_error = td_target - self.critic_values[state_row, state_col]

        self.critic_values[state_row, state_col] += self.alpha_critic * td_error

        action_probs = self.softmax(self.actor_params[state_row, state_col])
        self.actor_params[state_row, state_col, action] += self.alpha_actor * td_error * (1 - action_probs[action])
        for a in range(self.num_actions):
            if a != action:
                self.actor_params[state_row, state_col, a] -= self.alpha_actor * td_error * action_probs[a]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


grid_world = GridWorld()

num_actions = 4
alpha_actor = 0.1 
alpha_critic = 0.1
gamma = 0.9
actor_critic_agent = ActorCritic(num_actions, alpha_actor, alpha_critic, gamma)


num_episodes = 1000
for _ in range(num_episodes):
    state = grid_world.start_state
    done = False
    while not done:
        action = actor_critic_agent.select_action(state)
        next_state, reward = grid_world.step(state, action)
        actor_critic_agent.update(state, action, reward, next_state)
        state = next_state
        done = (state == grid_world.goal_state)

total_reward = 0
state = grid_world.start_state
while state != grid_world.goal_state:
    action = actor_critic_agent.select_action(state)
    next_state, reward = grid_world.step(state, action)
    total_reward += reward
    state = next_state

print("Total reward obtained by learned policy:", total_reward)
