import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)  
        self.num_actions = 4  
        self.start_state = (0, 0)  

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
        return next_state 
    

class SampleModel:
    def __init__(self, environment):
        self.environment = environment  

    def simulate_step(self, state, action):
        next_state = self.environment.step(state, action)
        return next_state
    

grid_world = GridWorld()

sample_model = SampleModel(grid_world)

current_state = (0, 0)
action = np.random.choice(grid_world.num_actions)  

next_state = sample_model.simulate_step(current_state, action)

print("Current state:", current_state)
print("Action:", action)
print("Next state:", next_state)
