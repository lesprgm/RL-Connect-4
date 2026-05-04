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
import matplotlib.pyplot as plt


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
#Frozen opponent model used to make self-play less unstable
opponent_model = neuralNetwork.connect4SelfPlayModel()

#Copy the weights from the policy model to the target model and opponent model
target_model.load_state_dict(policy_model.state_dict())
opponent_model.load_state_dict(policy_model.state_dict())
opponent_model.eval()

#Initialized buffer with a maximum size of 10,000 transitions
replay_buffer = replayBuffer.ReplayBuffer(100000)

#An optimizer and loss function for training the policy model from Torch
optimizer = optim.Adam(policy_model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

#Epsilon-greedy parameters
epsilonStart = 1.0
epsilonEnd = 0.05
epsilonDecay = 0.9998

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

def find_immediate_win_or_block(env, current_player, valid_columns):
    opponent = game.other_player(current_player)

    # 1. If current player can win immediately, choose that move.
    for column in valid_columns:
        candidate = env.clone()
        candidate.current_player = current_player
        candidate.drop_piece(column)
        if candidate.is_over() and candidate.winner == current_player:
            return column, "win"

    # 2. If opponent can win immediately, block that move.
    for column in valid_columns:
        candidate = env.clone()
        candidate.current_player = opponent
        candidate.drop_piece(column)
        if candidate.is_over() and candidate.winner == opponent:
            return column, "block"

    return None, None


def move_allows_opponent_win(env, current_player, column):
    opponent = game.other_player(current_player)
    candidate = env.clone()
    candidate.current_player = current_player
    candidate.drop_piece(column)

    if candidate.is_over():
        return False

    opponent_moves = candidate.available_columns()
    for opponent_column in opponent_moves:
        opponent_candidate = candidate.clone()
        opponent_candidate.current_player = opponent
        opponent_candidate.drop_piece(opponent_column)
        if opponent_candidate.is_over() and opponent_candidate.winner == opponent:
            return True

    return False


def choose_training_action(policy_model, env, current_player, state, epsilon):
    valid_columns = env.available_columns()

    tactical_action, tactical_reason = find_immediate_win_or_block(
        env,
        current_player,
        valid_columns,
    )

    # If there is a direct win or block, use it during training.
    # This stores tactical examples in replay memory so the model learns them.
    if tactical_action is not None:
        return tactical_action, tactical_reason

    safe_columns = []
    for column in valid_columns:
        if not move_allows_opponent_win(env, current_player, column):
            safe_columns.append(column)

    candidate_columns = safe_columns if len(safe_columns) > 0 else valid_columns

    # Epsilon-greedy exploration, but prefer safe exploratory moves.
    if random.random() < epsilon:
        return random.choice(candidate_columns), "explore_safe"

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_model(state_tensor).squeeze(0)

    masked_q_values = torch.full((7,), float('-inf'))
    for column in candidate_columns:
        masked_q_values[column] = q_values[column]

    return int(torch.argmax(masked_q_values).item()), "model_safe"


# --- Tactical/blocking action selection for tactical agent ---
def choose_tactical_blocking_action(env, current_player):
    valid_columns = env.available_columns()

    tactical_action, tactical_reason = find_immediate_win_or_block(
        env,
        current_player,
        valid_columns,
    )
    if tactical_action is not None:
        return tactical_action, f"tactical_{tactical_reason}"

    safe_columns = []
    for column in valid_columns:
        if not move_allows_opponent_win(env, current_player, column):
            safe_columns.append(column)

    candidate_columns = safe_columns if len(safe_columns) > 0 else valid_columns

    # Prefer center columns because they create more possible connect-four lines.
    column_preference = [3, 2, 4, 1, 5, 0, 6]
    for column in column_preference:
        if column in candidate_columns:
            return column, "tactical_center_safe"

    return random.choice(candidate_columns), "tactical_random_safe"


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


def evaluate_tactical_accuracy(model):
    model.eval()
    correct = 0
    total = 0

    test_columns = [0, 3, 6]

    for current_player in [1, 2]:
        opponent = game.other_player(current_player)

        for column in test_columns:
            # Test 1: model should take an immediate vertical win.
            env = game.ConnectFourGame()
            env.current_player = current_player
            env.board[5][column] = current_player
            env.board[4][column] = current_player
            env.board[3][column] = current_player

            prediction = choose_model_action(
                model,
                env.board,
                current_player,
                env.available_columns(),
            )
            if prediction == column:
                correct += 1
            total += 1

            # Test 2: model should block opponent's immediate vertical win.
            env = game.ConnectFourGame()
            env.current_player = current_player
            env.board[5][column] = opponent
            env.board[4][column] = opponent
            env.board[3][column] = opponent

            prediction = choose_model_action(
                model,
                env.board,
                current_player,
                env.available_columns(),
            )
            if prediction == column:
                correct += 1
            total += 1

    model.train()
    return correct / total if total > 0 else 0.0, correct, total

#Outer Game loop for multiple games
numGames = 0
best_win_rate = -1.0
current_win_rate = -1.0
current_tactical_accuracy = -1.0
current_combined_score = -1.0
best_combined_score = -float("inf")
training_log = []
for games in range(50000):
    #Initialized game Environment
    env = game.ConnectFourGame()
    done = False
    current_loss = 0.0

    # Opponent mix:
    # 35% live self-play, 30% frozen older model, 25% tactical/blocking agent, 10% random agent.
    opponent_roll = random.random()
    if opponent_roll < 0.35:
        opponent_type = "self"
        opponent_player = None
    elif opponent_roll < 0.65:
        opponent_type = "frozen"
        opponent_player = 2 if games % 2 == 0 else 1
    elif opponent_roll < 0.90:
        opponent_type = "tactical"
        opponent_player = 2 if games % 2 == 0 else 1
    else:
        opponent_type = "random"
        opponent_player = 2 if games % 2 == 0 else 1

    #inner Move loop for a single game
    while not done:
        # Save the player whose turn it is, then encode the board from that player's perspective.
        current_player = env.current_player
        state = get_State(env.board, current_player)

        # Pick an action based on the selected opponent mix.
        # Only store replay transitions for moves chosen by the live policy model.
        train_this_transition = True

        if opponent_type == "self" or current_player != opponent_player:
            action, action_reason = choose_training_action(
                policy_model,
                env,
                current_player,
                state,
                epsilonStart,
            )
        elif opponent_type == "frozen":
            train_this_transition = False
            action, action_reason = choose_training_action(
                opponent_model,
                env,
                current_player,
                state,
                0.0,
            )
        elif opponent_type == "tactical":
            train_this_transition = False
            action, action_reason = choose_tactical_blocking_action(env, current_player)
        else:
            train_this_transition = False
            action = random.choice(env.available_columns())
            action_reason = "random_opponent"

        allowed_opponent_win = move_allows_opponent_win(env, current_player, action)

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

        # Tactical reward shaping teaches the model why certain moves matter.
        if action_reason == "win":
            reward += 2.0
        elif action_reason == "block":
            reward += 1.0
        elif action_reason in ["model_safe", "explore_safe"]:
            reward += 0.05

        if allowed_opponent_win:
            reward -= 1.5

        #Store transition in replay buffer
        if train_this_transition:
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
        tactical_accuracy, tactical_correct, tactical_total = evaluate_tactical_accuracy(policy_model)

        current_win_rate = win_rate
        current_tactical_accuracy = tactical_accuracy
        current_combined_score = (0.40 * current_win_rate) + (0.60 * current_tactical_accuracy)

        print(
            f"Current evaluation at game {numGames}: "
            f"win_rate={current_win_rate:.2f}, wins={eval_wins}, losses={eval_losses}, draws={eval_draws}, "
            f"tactical_accuracy={current_tactical_accuracy:.2f} ({tactical_correct}/{tactical_total}), "
            f"combined_score={current_combined_score:.2f}"
        )

        if current_combined_score > best_combined_score:
            best_combined_score = current_combined_score
            best_win_rate = win_rate
            print(
                f"New best combined model at game {numGames}: "
                f"win_rate={win_rate:.2f}, tactical_accuracy={tactical_accuracy:.2f}, "
                f"combined_score={current_combined_score:.2f}. Saving checkpoint..."
            )
            torch.save({
                'model_state_dict': policy_model.state_dict(),
                'win_rate_vs_random': best_win_rate,
                'tactical_accuracy': current_tactical_accuracy,
                'combined_score': best_combined_score,
                'eval_wins': eval_wins,
                'eval_losses': eval_losses,
                'eval_draws': eval_draws,
                'game_number': numGames,
            }, 'connect4_policy_model.pth')

    #Every 100 games, we update the target model to match the policy model
    numGames += 1
    if numGames % 500 == 0:
        target_model.load_state_dict(policy_model.state_dict())

    if numGames % 2000 == 0:
        opponent_model.load_state_dict(policy_model.state_dict())
        opponent_model.eval()
    
    if numGames % 100 == 0:
        latest_loss = current_loss if replay_buffer.size() >= 64 else 0.0

        training_log.append({
            "game": numGames,
            "epsilon": epsilonStart,
            "loss": latest_loss,
            "reward": reward,
            "current_win_rate": current_win_rate,
            "best_win_rate": best_win_rate,
            "current_tactical_accuracy": current_tactical_accuracy,
            "current_combined_score": current_combined_score,
            "best_combined_score": best_combined_score,
            "opponent_type": opponent_type,
            "last_action_reason": action_reason,
        })

        print(
            f"Game: {numGames}, Opponent: {opponent_type}, Epsilon: {epsilonStart:.4f}, "
            f"Loss: {latest_loss:.4f}, reward: {reward}, "
            f"current_win_rate: {current_win_rate:.2f}, best_win_rate: {best_win_rate:.2f}, "
            f"tactical_accuracy: {current_tactical_accuracy:.2f}, combined_score: {current_combined_score:.2f}"
        )

    
    

if best_win_rate < 0:
    print("No gameplay-evaluated checkpoint was saved, so saving final model as fallback...")
    torch.save({'model_state_dict': policy_model.state_dict()}, 'connect4_policy_model.pth')
else:
    print(
        f"Training complete. Best saved model had win_rate_vs_random={best_win_rate:.2f} "
        f"and combined_score={best_combined_score:.2f}"
    )


# Save training plots
if len(training_log) > 0:
    games = [row["game"] for row in training_log]
    losses = [row["loss"] for row in training_log]
    epsilons = [row["epsilon"] for row in training_log]
    rewards = [row["reward"] for row in training_log]

    plt.figure()
    plt.plot(games, losses)
    plt.xlabel("Game")
    plt.ylabel("Loss")
    plt.title("Self-Play Training Loss")
    plt.savefig("training_loss.png")
    plt.close()

    plt.figure()
    plt.plot(games, epsilons)
    plt.xlabel("Game")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay During Training")
    plt.savefig("epsilon_decay.png")
    plt.close()

    plt.figure()
    plt.plot(games, rewards)
    plt.xlabel("Game")
    plt.ylabel("Reward")
    plt.title("Reward During Training")
    plt.savefig("training_reward.png")
    plt.close()

    valid_current_win_rate_games = []
    valid_current_win_rates = []
    valid_best_win_rate_games = []
    valid_best_win_rates = []

    for row in training_log:
        if row["current_win_rate"] >= 0:
            valid_current_win_rate_games.append(row["game"])
            valid_current_win_rates.append(row["current_win_rate"])

        if row["best_win_rate"] >= 0:
            valid_best_win_rate_games.append(row["game"])
            valid_best_win_rates.append(row["best_win_rate"])

    if len(valid_current_win_rates) > 0:
        plt.figure()
        plt.plot(valid_current_win_rate_games, valid_current_win_rates)
        plt.xlabel("Game")
        plt.ylabel("Current Win Rate vs Random")
        plt.title("Current Gameplay Win Rate Over Time")
        plt.savefig("current_win_rate.png")
        plt.close()

    if len(valid_best_win_rates) > 0:
        plt.figure()
        plt.plot(valid_best_win_rate_games, valid_best_win_rates)
        plt.xlabel("Game")
        plt.ylabel("Best Win Rate vs Random")
        plt.title("Best Gameplay Win Rate")
        plt.savefig("best_win_rate.png")
        plt.close()

    valid_tactical_games = []
    valid_tactical_accuracies = []
    valid_combined_games = []
    valid_combined_scores = []

    for row in training_log:
        if row["current_tactical_accuracy"] >= 0:
            valid_tactical_games.append(row["game"])
            valid_tactical_accuracies.append(row["current_tactical_accuracy"])

        if row["current_combined_score"] >= 0:
            valid_combined_games.append(row["game"])
            valid_combined_scores.append(row["current_combined_score"])

    if len(valid_tactical_accuracies) > 0:
        plt.figure()
        plt.plot(valid_tactical_games, valid_tactical_accuracies)
        plt.xlabel("Game")
        plt.ylabel("Tactical Accuracy")
        plt.title("Tactical Accuracy Over Time")
        plt.savefig("tactical_accuracy.png")
        plt.close()

    if len(valid_combined_scores) > 0:
        plt.figure()
        plt.plot(valid_combined_games, valid_combined_scores)
        plt.xlabel("Game")
        plt.ylabel("Combined Score")
        plt.title("Combined Evaluation Score Over Time")
        plt.savefig("combined_score.png")
        plt.close()

    print("Training plots saved as PNG files.")
else:
    print("No training log entries were recorded, so no plots were created.")
