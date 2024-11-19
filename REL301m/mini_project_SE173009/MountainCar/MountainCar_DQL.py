import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from custom_game.envs import CustomGame

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# MountainCar Deep Q-Learning
class MountainCarDQL():
    learning_rate_a = 0.01         
    discount_factor_g = 0.9           
    network_sync_rate = 50000         
    replay_memory_size = 100000       
    mini_batch_size = 32            
    
    num_divisions = 20

    # Neural Network
    loss_fn = nn.MSELoss()          
    optimizer = None               


    # Train the environment
    def train(self, episodes, render=False):
        env = gym.make('CustomGame-v0', render_mode='human' if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Divide position and velocity into segments
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07
    
        epsilon = 1 
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

  
        rewards_per_episode = []

        
        epsilon_history = []

       
        step_count=0
        goal_reached=False
        best_rewards=-200
            
        for i in range(episodes):
            state = env.reset()[0] 
            terminated = False      

            rewards = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and rewards>-1000):

                # epsilon-greedy
                if random.random() < epsilon:
                    
                    action = env.action_space.sample() # actions: 0=left,1=idle,2=right
                else:
                          
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                
                new_state,reward,terminated,truncated,_ = env.step(action)

               
                rewards += reward

                
                memory.append((state, action, new_state, reward, terminated)) 

                
                state = new_state

                
                step_count+=1

            
            rewards_per_episode.append(rewards)
            if(terminated):
                goal_reached = True

            
            if(i!=0 and i%1000==0):
                print(f'Episode {i} Epsilon {epsilon}')
                                        
                self.plot_progress(rewards_per_episode, epsilon_history)
            
            if rewards>best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                torch.save(policy_dqn.state_dict(), f"cus_mountaincar_dql_{i}.pt")

            # Check if enough experience has been collected
            if len(memory)>self.mini_batch_size and goal_reached:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0                    

        env.close()
    def plot_progress(self, rewards_per_episode, epsilon_history):
        plt.figure(1)
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        
        plt.subplot(122) 
        plt.plot(epsilon_history)
        
        plt.savefig('cus_mountaincar_dql.png')
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state)) 
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts a state (position, velocity) to tensor representation.
    '''
    def state_to_dqn_input(self, state)->torch.Tensor:
        state_p = np.digitize(state[0], self.pos_space)
        state_v = np.digitize(state[1], self.vel_space)
        
        return torch.FloatTensor([state_p, state_v])
    
    # Run the environment with the learned policy
    def test(self, episodes, model_filepath):
        env = gym.make('CustomGame-v0', render_mode='human')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07

        
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()    

        for i in range(episodes):
            state = env.reset()[0]  
            terminated = False      
            truncated = False                  

            while(not terminated and not truncated):  
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

if __name__ == '__main__':

    mountaincar = MountainCarDQL()
    # mountaincar.train(20000, False)
    mountaincar.test(1, "D:/Chuyen_Nganh_8/RLE/VALIDATE_GYM/mini_project/custom_game/cus_mountaincar_dql_15628.pt")