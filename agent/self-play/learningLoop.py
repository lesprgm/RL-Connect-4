import os
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.environ.get(
    "SELF_PLAY_CHECKPOINT_PATH",
    os.path.join(SCRIPT_DIR, "connect4_policy_model.pth"),
)

import replayBuffer
import neuralNetwork
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


def load_model_weights_if_available(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return True


def save_model_checkpoint(model, checkpoint_path, metadata=None):
    payload = {"model_state_dict": model.state_dict()}
    if metadata is not None:
        payload.update(metadata)
    torch.save(payload, checkpoint_path)


def copy_model_weights(source_model, target_model, eval_mode=False):
    target_model.load_state_dict(source_model.state_dict())
    if eval_mode:
        target_model.eval()


def import_pyplot_for_plots():
    # Plotting is useful after training, but it should not be required to start RL.
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "rl-connect4-matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed, so training plots were skipped.")
        return None
    return plt


#policy model used for action selection and learning
policy_model = neuralNetwork.connect4SelfPlayModel()
#Older frozen model used for stable target value estimation
target_model = neuralNetwork.connect4SelfPlayModel()
#Frozen opponent model used to make self-play less unstable
opponent_model = neuralNetwork.connect4SelfPlayModel()
#Stable baseline used only for checkpoint evaluation. This keeps model saving based
#on gameplay against a fixed RL opponent instead of only beating random moves.
evaluation_opponent_model = neuralNetwork.connect4SelfPlayModel()

loaded_existing_checkpoint = load_model_weights_if_available(policy_model, CHECKPOINT_PATH)
if loaded_existing_checkpoint:
    print(f"Loaded existing self-play checkpoint from {CHECKPOINT_PATH}")

#Copy the weights from the policy model to the target model and opponent models
copy_model_weights(policy_model, target_model)
copy_model_weights(policy_model, opponent_model, eval_mode=True)
copy_model_weights(policy_model, evaluation_opponent_model, eval_mode=True)

#Initialized buffer with a maximum size of 10,000 transitions
replay_buffer = replayBuffer.ReplayBuffer(100000)

#An optimizer and loss function for training the policy model from Torch
optimizer = optim.Adam(policy_model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

#Epsilon-greedy parameters
epsilonStart = 1.0
epsilonEnd = 0.05
epsilonDecay = 0.9998
gamma = 0.95

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


def choose_q_action(model, state, valid_columns):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)

    masked_q_values = torch.full((7,), float('-inf'))
    for col in valid_columns:
        masked_q_values[col] = q_values[col]

    return int(torch.argmax(masked_q_values).item())


def choose_model_action(model, board, current_player, valid_columns):
    state = get_State(board, current_player)
    return choose_q_action(model, state, valid_columns)


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

    # explore with a random legal move or exploit the Q-values learned by the model.
    if random.random() < epsilon:
        return random.choice(valid_columns), "explore"

    return choose_q_action(policy_model, state, valid_columns), "model"


def choose_opponent_setup(game_index):
    opponent_roll = random.random()
    if opponent_roll < 0.45:
        return "self", None

    opponent_player = 2 if game_index % 2 == 0 else 1
    if opponent_roll < 0.80:
        return "frozen", opponent_player
    return "random", opponent_player


def choose_game_action(opponent_type, opponent_player, env, current_player, state, epsilon):
    if opponent_type == "self" or current_player != opponent_player:
        action, action_reason = choose_training_action(
            policy_model,
            env,
            current_player,
            state,
            epsilon,
        )
        return action, action_reason, True

    if opponent_type == "frozen":
        action, action_reason = choose_training_action(
            opponent_model,
            env,
            current_player,
            state,
            0.0,
        )
        return action, action_reason, False

    return random.choice(env.available_columns()), "random_opponent", False


def evaluate_checkpoint_candidate(model, baseline_opponent, games_to_play=100):
    model.eval()
    baseline_opponent.eval()

    wins = 0
    losses = 0
    draws = 0

    for game_index in range(games_to_play):
        env = game.ConnectFourGame()
        model_player = 1 if game_index % 2 == 0 else 2

        while not env.is_over():
            valid_columns = env.available_columns()

            if env.current_player == model_player:
                action = choose_model_action(model, env.board, env.current_player, valid_columns)
            else:
                action = choose_model_action(baseline_opponent, env.board, env.current_player, valid_columns)

            env.drop_piece(action)

        winner = env.winner
        if winner == model_player:
            wins += 1
        elif winner == 0 or winner is None:
            draws += 1
        else:
            losses += 1

    model.train()
    baseline_opponent.eval()
    return {
        "score": (wins + 0.5 * draws) / games_to_play,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def checkpoint_metadata(eval_results, game_number):
    return {
        "checkpoint_gameplay_score": eval_results["score"],
        "eval_wins": eval_results["wins"],
        "eval_losses": eval_results["losses"],
        "eval_draws": eval_results["draws"],
        "game_number": game_number,
    }


def save_line_plot(plt, x_values, y_values, ylabel, title, filename):
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel("Game")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(SCRIPT_DIR, filename))
    plt.close()

#Outer Game loop for multiple games
total_games = int(os.environ.get("SELF_PLAY_NUM_GAMES", "10000"))
eval_start_game = int(os.environ.get("SELF_PLAY_EVAL_START_GAME", "3000"))
eval_interval = int(os.environ.get("SELF_PLAY_EVAL_INTERVAL", "500"))
eval_games = int(os.environ.get("SELF_PLAY_EVAL_GAMES", "100"))
numGames = 0
best_checkpoint_score = -1.0
best_eval_tiebreak_score = -float("inf")
checkpoint_saved = False
training_log = []
for games in range(total_games):
    #Initialized game Environment
    env = game.ConnectFourGame()
    current_loss = 0.0
    pending_policy_transitions = {}

    # 45% live self-play, 35% frozen older RL model, 20% random agent.
    opponent_type, opponent_player = choose_opponent_setup(games)

    #inner Move loop for a single game
    while not env.is_over():
        # Save the player whose turn it is, then encode the board from that player's perspective.
        current_player = env.current_player

        # If this player gets another turn, the previous policy move did not
        # lead to an immediate terminal loss, so the normal transition can enter replay.
        if current_player in pending_policy_transitions:
            replay_buffer.add(pending_policy_transitions.pop(current_player))

        state = get_State(env.board, current_player)

        action, action_reason, train_this_transition = choose_game_action(
            opponent_type,
            opponent_player,
            env,
            current_player,
            state,
            epsilonStart,
        )

        allowed_opponent_win = move_allows_opponent_win(env, current_player, action)

        #Save players, get players' scores and Take Action
        other_player = game.other_player(current_player)
        playerBDScore = game.score_position(env, current_player, DEFAULT_WEIGHTS)
        otherPlayerBDScore = game.score_position(env, other_player, DEFAULT_WEIGHTS)

        env.drop_piece(action)
        done = env.is_over()

        # After the move, the next state belongs to the next player to move.
        # Encoding it from env.current_player's perspective prevents player 2 from learning backwards.
        next_state = get_State(env.board, env.current_player)

        #Assign reward based on game outcome
        reward = 0.0
        if done:
            if (env.is_draw):
                reward += 0.5
            elif(env.winner == current_player):
                reward += 1.0
            else:
                # Defensive fallback. With the current game API, the player who
                # just moved cannot immediately lose; delayed loss credit is
                # handled below when an opponent's later move wins.
                reward += -1.0
        else:
            playerADScore = game.score_position(env, current_player, DEFAULT_WEIGHTS)
            otherPlayerADScore = game.score_position(env, other_player, DEFAULT_WEIGHTS)
            Playerscore_diff = playerADScore - playerBDScore
            OtherPlayerScore_diff = otherPlayerADScore - otherPlayerBDScore
            reward -= max(min(OtherPlayerScore_diff / 100000, 0.1), -0.1)
            
            reward += max(min(Playerscore_diff / 100000, 0.1), -0.1)

        # Reward shaping remains reinforcement learning because the model receives
        # feedback after taking an action instead of being given a labeled answer.
        if action_reason in ["model", "explore"]:
            reward += 0.02

        if allowed_opponent_win:
            reward -= 1.5

        #Store transition in replay buffer. Non-terminal policy moves are held
        #until the player gets another turn or the opponent ends the game.
        if train_this_transition:
            if done:
                replay_buffer.add((state, action, reward, next_state, done))
            else:
                pending_policy_transitions[current_player] = (state, action, reward, next_state, done)

        if done:
            if env.is_draw:
                for _, pending_transition in list(pending_policy_transitions.items()):
                    old_state, old_action, old_reward, old_next_state, _ = pending_transition
                    replay_buffer.add((old_state, old_action, old_reward + 0.5, old_next_state, True))
                pending_policy_transitions.clear()
            elif env.winner is not None:
                losing_player = game.other_player(env.winner)
                if losing_player in pending_policy_transitions:
                    old_state, old_action, old_reward, old_next_state, _ = pending_policy_transitions.pop(losing_player)
                    replay_buffer.add((old_state, old_action, old_reward - 1.0, old_next_state, True))

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
                target_q_values = rewards_tensor - (gamma * max_next_q_values * (1 - dones_tensor))

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

    if numGames > eval_start_game and numGames % eval_interval == 0:
        eval_results = evaluate_checkpoint_candidate(
            policy_model,
            evaluation_opponent_model,
            games_to_play=eval_games,
        )
        checkpoint_score = eval_results["score"]
        eval_tiebreak_score = eval_results["wins"] - eval_results["losses"]

        if checkpoint_score > best_checkpoint_score or (
            checkpoint_score == best_checkpoint_score
            and eval_tiebreak_score > best_eval_tiebreak_score
        ):
            best_checkpoint_score = checkpoint_score
            best_eval_tiebreak_score = eval_tiebreak_score
            print(
                f"New best RL model at game {numGames}: "
                "Saving checkpoint..."
            )
            save_model_checkpoint(
                policy_model,
                CHECKPOINT_PATH,
                checkpoint_metadata(eval_results, numGames),
            )
            checkpoint_saved = True

    #Every 100 games, we update the target model to match the policy model
    numGames += 1
    if numGames % 500 == 0:
        copy_model_weights(policy_model, target_model)

    if numGames % 2000 == 0:
        copy_model_weights(policy_model, opponent_model, eval_mode=True)
    
    if numGames % 100 == 0:
        latest_loss = current_loss if replay_buffer.size() >= 64 else 0.0

        training_log.append({
            "game": numGames,
            "epsilon": epsilonStart,
            "loss": latest_loss,
            "reward": reward,
        })

        print(
            f"Game: {numGames}, Opponent: {opponent_type}, Epsilon: {epsilonStart:.4f}, "
            f"Loss: {latest_loss:.4f}, reward: {reward}"
        )

    
    

if not checkpoint_saved:
    if loaded_existing_checkpoint:
        print("No improved gameplay-evaluated checkpoint was saved; existing checkpoint was left unchanged.")
    else:
        print("No gameplay-evaluated checkpoint was saved, so saving final model as fallback...")
        save_model_checkpoint(policy_model, CHECKPOINT_PATH)
else:
    print("Training complete. Best checkpoint was saved.")


# Save training plots
if len(training_log) > 0:
    plt = import_pyplot_for_plots()

    if plt is not None:
        games = [row["game"] for row in training_log]
        base_plots = [
            ("loss", "Loss", "Self-Play Training Loss", "training_loss.png"),
            ("epsilon", "Epsilon", "Epsilon Decay During Training", "epsilon_decay.png"),
            ("reward", "Reward", "Reward During Training", "training_reward.png"),
        ]
        for metric_name, ylabel, title, filename in base_plots:
            save_line_plot(
                plt,
                games,
                [row[metric_name] for row in training_log],
                ylabel,
                title,
                filename,
            )

        print("Training plots saved as PNG files.")
else:
    print("No training log entries were recorded, so no plots were created.")
