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
optimizer = optim.Adam(policy_model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

#Epsilon-greedy parameters
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 0.999

def get_State(board, current_player):
    # Get board state from the current player's perspective:
    # empty = 0, current player = 1, opponent = 2
    flat_state = []
    for row in board:
        for cell in row:
            if cell == 0:
                flat_state.append(0)
            elif cell == current_player:
                flat_state.append(1)
            else:
                flat_state.append(2)
    return flat_state


def mask_invalid_q_values(q_values, states_tensor, dones_tensor=None):
    # Columns are valid only if the top cell in that column is empty.
    # Since the flattened state is row-major, indices 0 through 6 are the top row.
    masked_q_values = q_values.clone()

    for row_index in range(masked_q_values.shape[0]):
        if dones_tensor is not None and dones_tensor[row_index].item() == 1.0:
            masked_q_values[row_index] = 0.0
            continue

        valid_columns = []
        for col in range(7):
            if states_tensor[row_index][col].item() == 0:
                valid_columns.append(col)
            else:
                masked_q_values[row_index][col] = float('-inf')

        if len(valid_columns) == 0:
            masked_q_values[row_index] = 0.0

    return masked_q_values


# --- Model evaluation and action selection helpers ---
def choose_model_action(model, board, current_player, valid_columns):
    state = get_State(board, current_player)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)

    masked_q_values = torch.full((7,), float('-inf'))
    for col in valid_columns:
        masked_q_values[col] = q_values[col]

    return int(torch.argmax(masked_q_values).item())


def evaluate_model_against_random(model, games_to_play=100):
    model.eval()
    wins = 0
    losses = 0
    draws = 0

    for game_index in range(games_to_play):
        env = game.ConnectFourGame()
        done = False
        model_player = 1 if game_index % 2 == 0 else 2

        while not done:
            valid_columns = env.available_columns()

            if env.current_player == model_player:
                action = choose_model_action(model, env.board, env.current_player, valid_columns)
            else:
                action = random.choice(valid_columns)

            env.drop_piece(action)

            if env.is_over():
                done = True

        winner = env.winner
        if winner == model_player:
            wins += 1
        elif winner == 0 or winner is None:
            draws += 1
        else:
            losses += 1

    model.train()
    return wins / games_to_play, wins, losses, draws

#Outer Game loop for multiple games
numGames = 0
best_win_rate = -1.0
best_eval_score = -float("inf")
for games in range(10000):
    #Initialized game Environment
    env = game.ConnectFourGame()
    done = False
    current_loss = 0.0

    #inner Move loop for a single game
    while not done:
        # Save the player whose turn it is, then encode the board from that player's perspective.
        current_player = env.current_player
        state = get_State(env.board, current_player)
        
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
        other_player = game.other_player(current_player)
        playerBDScore = game.score_position(env, current_player, DEFAULT_WEIGHTS)
        otherPlayerBDScore = game.score_position(env, other_player, DEFAULT_WEIGHTS)

        env.drop_piece(action)

        # After the move, the next state belongs to the next player to move.
        # Encoding it from env.current_player's perspective prevents player 2 from learning backwards.
        next_state = get_State(env.board, env.current_player)

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

            #Compute target Q-values using the target model.
            # In self-play, the next state is the opponent's turn, so the opponent's best value
            # should be subtracted from the current player's value.
            with torch.no_grad():
                next_q_values = target_model(next_states_tensor)
                next_q_values = mask_invalid_q_values(next_q_values, next_states_tensor, dones_tensor)
                max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
                target_q_values = rewards_tensor - (0.95 * max_next_q_values * (1 - dones_tensor))

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

    if numGames > 3000 and numGames % 500 == 0:
        win_rate, eval_wins, eval_losses, eval_draws = evaluate_model_against_random(
            policy_model,
            games_to_play=100,
        )
        eval_score = eval_wins - eval_losses

        if win_rate > best_win_rate or (win_rate == best_win_rate and eval_score > best_eval_score):
            best_win_rate = win_rate
            best_eval_score = eval_score
            print(
                f"New best gameplay model at game {numGames}: "
                f"win_rate={win_rate:.2f}, wins={eval_wins}, losses={eval_losses}, draws={eval_draws}. "
                "Saving checkpoint..."
            )
            torch.save({
                'model_state_dict': policy_model.state_dict(),
                'win_rate_vs_random': best_win_rate,
                'eval_wins': eval_wins,
                'eval_losses': eval_losses,
                'eval_draws': eval_draws,
                'game_number': numGames,
            }, 'connect4_policy_model.pth')

    #Every 100 games, we update the target model to match the policy model
    numGames += 1
    if numGames % 100 == 0:
        target_model.load_state_dict(policy_model.state_dict())
    
    if numGames % 100 == 0:
        latest_loss = current_loss if replay_buffer.size() >= 64 else 0.0
        print(
            f"Game: {numGames}, Epsilon: {epsilonStart:.4f}, "
            f"Loss: {latest_loss:.4f}, reward: {reward}, best_win_rate: {best_win_rate:.2f}"
        )

    
    
if best_win_rate < 0:
    print("No gameplay-evaluated checkpoint was saved, so saving final model as fallback...")
    torch.save({'model_state_dict': policy_model.state_dict()}, 'connect4_policy_model.pth')
else:
    print(f"Training complete. Best saved model had win_rate_vs_random={best_win_rate:.2f}")
