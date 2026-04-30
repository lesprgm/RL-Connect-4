import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import replayBuffer
import neuralNetwork
import copy
import random
import torch
import torch.optim as optim
import torch.nn as nn
from connect4 import game


DEFAULT_WEIGHTS = {
    "four": 100000,
    "block_four": 95000,
    "three": 120,
    "two": 20,
    "block_three": 140,
    "block_two": 25,
    "center": 6,
}

#policy model used for action selection and learning
policy_model = neuralNetwork.connect4SelfPlayModel()
#Older frozen model used for stable target value estimation
target_model = neuralNetwork.connect4SelfPlayModel()

#Copy the weights from the policy model to the target model
target_model.load_state_dict(policy_model.state_dict())

#Initialized buffer with a maximum size of 10,000 transitions
replay_buffer = replayBuffer.ReplayBuffer(50000)

#An optimizer and loss function for training the policy model from Torch
optimizer = optim.Adam(policy_model.parameters(), lr=0.00001)
criterion = nn.MSELoss()

#Epsilon-greedy parameters
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 0.999

def get_State(board):
    #Get Board state and switch player representation to 1 and -1 for the current player and opponent
    state = copy.deepcopy(board)
    flat_state = [cell for row in state for cell in row]
    return flat_state

#Outer Game loop for multiple games
numGames = 0
min_loss = float('inf')
for games in range(10000):
    #Initialized game Environment
    env = game.ConnectFourGame()
    done = False
    current_loss = 0.0

    #inner Move loop for a single game
    while not done:
        #Get Board state and switch player representation to 1 and -1 for the current player and opponent
        state = get_State(env.board)
        
        #Picking an action(Column) using epsilon-greedy strategy
        if random.random() < epsilonStart:
            action = random.choice(env.available_columns())
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
                q_values = policy_model(state_tensor)
                #check if column is available
                for col in range(q_values.shape[1]):
                    if col not in env.available_columns():
                        q_values[0][col] = float('-inf')  # Set Q-value to -inf for unavailable columns
                action = torch.argmax(q_values).item()
        
        

        #Save players, get players' scores and Take Action
        current_player = env.current_player
        other_player = game.other_player(current_player)
        playerBDScore = game.score_position(env, current_player, DEFAULT_WEIGHTS)
        otherPlayerBDScore = game.score_position(env, other_player, DEFAULT_WEIGHTS)

        env.drop_piece(action)

        #Get next State
        next_state = get_State(env.board)

        #Assign reward based on game outcome
        reward = 0.0
        if env.is_over():
            if (env.is_draw):
                reward += 0.5
            elif(env.winner == current_player):
                reward += 1.0
            else:
                reward += -1.0
            done = True
        else:
            playerADScore = game.score_position(env, current_player, DEFAULT_WEIGHTS)
            otherPlayerADScore = game.score_position(env, other_player, DEFAULT_WEIGHTS)
            Playerscore_diff = playerADScore - playerBDScore
            OtherPlayerScore_diff = otherPlayerADScore - otherPlayerBDScore
            reward -= max(min(OtherPlayerScore_diff / 100000, 0.1), -0.1)
            
            reward += max(min(Playerscore_diff / 100000, 0.1), -0.1)
        
            done = False

        #Store transition in replay buffer
        replay_buffer.add((state, action, reward, next_state, done))

        #Sample a batch of transitions from the replay buffer
        if replay_buffer.size() >= 64:
            batch = replay_buffer.sample(64)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

            #Compute current Q-values using the policy model
            current_q_values = policy_model(states_tensor).gather(1, actions_tensor)

            #Compute target Q-values using the target model
            with torch.no_grad():
                next_q_values = target_model(next_states_tensor)
                max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
                target_q_values = rewards_tensor + (0.95 * max_next_q_values * (1 - dones_tensor))

            #Compute loss and update policy model
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            current_loss = loss.item()
            nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
    #Decay epsilon after each game        
    epsilonStart *= epsilonDecay
    if epsilonStart < epsilonEnd:
        epsilonStart = epsilonEnd

    if numGames > 3000:
        if current_loss <= min_loss:
            min_loss = current_loss
            print(f"New best model with loss {min_loss:.4f}, saving checkpoint...")
            torch.save({'model_state_dict': policy_model.state_dict()}, 'connect4_policy_model.pth')

    #Every 100 games, we update the target model to match the policy model
    numGames += 1
    if numGames % 100 == 0:
        target_model.load_state_dict(policy_model.state_dict())
    
    if numGames % 100 == 0:
        print(f"Game: {numGames}, Epsilon: {epsilonStart:.4f}, Loss: {loss.item():.4f} reward: {reward}")

    
    
torch.save({'model_state_dict': policy_model.state_dict()}, 'connect4_policy_model.pth')

