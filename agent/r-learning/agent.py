import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()

        #Flatten 6x7 board into 42 inputs and predict 7 column scores.
        self.network = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,7)
        )

    def forward(self, x):
        return self.network(x)

#replay memory, where each experience is (state,action,reward, next_state, done)
class ReplayMemory:
    def __init__(self, capacity=10000):
        #remove old memories once you hit capacity(we store moves and deque moves once cap is hit)
        self.memory=deque(maxlen=capacity)
        
    def push(self,state,action,reward, next_state, done):
        #save an experience
        self.memory.append((state,action,reward,next_state, done))
        
    def sample(self, batch_size):
        #pull a random batch of experiences to learn from
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
#==============
# AGENT
#==============
class Agent:
    def __init__(self):
        self.model = DQN() #the brain
        self.memory =ReplayMemory() #experience storage
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.loss_fn = nn.MSELoss() #measure how wrong we are
        
        #Epsilon control exploration vs exploitation, so high is random and low is the agent exploiting
        self.epsilon = 1.0 #we start fully random
        self.epsilon_min = 0.05 #never go below 5% random
        self.epsilon_decay = 0.998 # slowly reduce randomness each episode
        self.gamma = 0.95 #how much future reqrds matter(0=none, 1=fully)
        self.batch_size = 64
        
    def select_action(self, state, valid_columns):
        if random.random() < self.epsilon:
            return random.choice(valid_columns)
        
        #Exploit:ask the network what it thinks is best
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor) #get 7 q values
            
        #hide invalid columns by setting their q values to a very low number so the agent never picks a full column
        q_values = q_values.squeeze().numpy()
        masked = np.full(7, -1e9) #start with everything invalid
        for col in valid_columns:
            masked[col] = q_values[col] #only allow valid columns
            
        return int(np.argmax(masked)) #pick the highest q-value
    
    def store(self,state,action,reward, next_state, done):
        self.memory.push(state,action,reward, next_state, done)
        
    def learn(self):
        #dont learn until we have enough experiences to sample from
        if len(self.memory) < self.batch_size:
            return
        
        #pull a random batch of past experiences
        batch = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states      = torch.FloatTensor(np.array([s.flatten() for s in states]))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array([s.flatten() for s in next_states]))
        dones       = torch.FloatTensor(dones)

        # Get the Q-values the model currently predicts for these states
        # .gather() plucks out the Q-value for the action actually taken
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        max_next_q = self.model(next_states).max(1)[0]
        target_q = rewards + self.gamma * max_next_q * (1-dones)
        
         # Calculate how wrong we were and update the network
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()   # clear old gradients
        loss.backward()              # compute new gradients
        self.optimizer.step()        # update weights

        # Decay epsilon so the agent gradually relies more on what it learned
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        
        
        
        
