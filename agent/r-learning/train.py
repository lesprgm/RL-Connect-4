import numpy as np
import random
import matplotlib.pyplot as plt
from agent import Agent

# SEMI-RANDOM OPPONENT
# This opponent plays randomly most of the time but will take a winning move or block your agent if it sees one.

def get_valid_columns(board):
    # A column is valid if the top row (row 0) is still empty
    return [col for col in range(7) if board[0][col] == 0]

def drop_piece(board, col, piece):
    # because of gravity find the lowest empty row in the column and place the piece there
    for row in range(5, -1, -1): 
        if board[row][col] == 0:
            board[row][col] = piece
            return row 

def check_win(board, piece):
    # Check all 4 directions for 4 in a row

    # Horizontal
    for row in range(6):
        for col in range(4):
            if all(board[row][col + i] == piece for i in range(4)):
                return True

    # Vertical
    for row in range(3):  # only need to go up to row 2
        for col in range(7):
            if all(board[row + i][col] == piece for i in range(4)):
                return True

    # Diagonal (down-right)
    for row in range(3):
        for col in range(4):
            if all(board[row + i][col + i] == piece for i in range(4)):
                return True

    # Diagonal (down-left)
    for row in range(3):
        for col in range(3, 7):
            if all(board[row + i][col - i] == piece for i in range(4)):
                return True

    return False

def semi_random_move(board, opponent_piece, agent_piece):
    valid_cols = get_valid_columns(board)

    # First check if opponent can win then it take that move
    for col in valid_cols:
        temp_board = board.copy()
        drop_piece(temp_board, col, opponent_piece)
        if check_win(temp_board, opponent_piece):
            return col

    # Then check if agent is about to win then we block it
    for col in valid_cols:
        temp_board = board.copy()
        drop_piece(temp_board, col, agent_piece)
        if check_win(temp_board, agent_piece):
            return col

    # Otherwise play randomly
    return random.choice(valid_cols)


def mid_game_reward(board):
    reward = 0.0

    # reward controlling center column (most powerful position in Connect 4)
    center_col = board[:, 3]
    reward += 0.05 * np.count_nonzero(center_col == 1)  # agent pieces in center
    reward -= 0.05 * np.count_nonzero(center_col == 2)  # opponent pieces in center

    # reward having 3 in a row (close to winning)
    if has_n_in_a_row(board, 1, 3):
        reward += 0.3
    
    # penalize opponent having 3 in a row (dangerous)
    if has_n_in_a_row(board, 2, 3):
        reward -= 0.4

    return reward

def has_n_in_a_row(board, piece, n):
    # Horizontal
    for row in range(6):
        for col in range(7 - n + 1):
            if all(board[row][col + i] == piece for i in range(n)):
                return True
    # Vertical
    for row in range(6 - n + 1):
        for col in range(7):
            if all(board[row + i][col] == piece for i in range(n)):
                return True
    # Diagonal down-right
    for row in range(6 - n + 1):
        for col in range(7 - n + 1):
            if all(board[row + i][col + i] == piece for i in range(n)):
                return True
    # Diagonal down-left
    for row in range(6 - n + 1):
        for col in range(n - 1, 7):
            if all(board[row + i][col - i] == piece for i in range(n)):
                return True
    return False

# Returns the result so we can track win rates over time

def play_episode(agent):
    board = np.zeros((6, 7), dtype=int)
    done = False
    result = None  

    while not done:

        # AGENT'S TURN 
        valid_cols = get_valid_columns(board)

        if not valid_cols:
            result = "draw"
            break

        # Agent picks a column based on current Q-values (or randomly if exploring)
        action = agent.select_action(board, valid_cols)
        state = board.copy()  # save state BEFORE the move for learning later

        drop_piece(board, action, 1)  # 1 = agent's piece

        if check_win(board, 1):
            # Agent won — store this experience with a big positive reward
            agent.store(state, action, reward=1.0, next_state=board.copy(), done=1)
            result = "win"
            done = True
            break

        # OPPONENT'S TURN
        valid_cols = get_valid_columns(board)

        if not valid_cols:
            result = "draw"
            # No reward for draw — it's neutral
            agent.store(state, action, reward=0.0, next_state=board.copy(), done=1)
            break

        opp_action = semi_random_move(board, opponent_piece=2, agent_piece=1)
        drop_piece(board, opp_action, 2)  # 2 = opponent's piece

        if check_win(board, 2):
            # Opponent won — store this experience with a negative reward
            agent.store(state, action, reward=-1.0, next_state=board.copy(), done=1)
            result = "loss"
            done = True
            break

        # Game is still going — small neutral reward, keep learning
        # next_state is the board AFTER the opponent moved
        agent.store(state, action, reward=mid_game_reward(board), next_state=board.copy(), done=0)

    # Learn from a random batch of past experiences
    agent.learn()

    return result


# TRAINING LOOP

def train(episodes=100000):
    agent = Agent()

    wins   = 0
    losses = 0
    draws  = 0
    win_rates = []

    print("Training started...\n")

    for episode in range(1, episodes + 1):
        result = play_episode(agent)

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1

        # Every 500 episodes print a progress update
        if episode % 500 == 0:
            win_rate = wins / 500 * 100
            win_rates.append(win_rate)
            print(f"Episode {episode:>6} | "
                  f"Wins: {wins:>3} | "
                  f"Losses: {losses:>3} | "
                  f"Draws: {draws:>3} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
            wins = losses = draws = 0  

    print("\nTraining complete.")

    plt.figure(figsize=(10, 5))
    plt.plot(win_rates)
    plt.title("Agent Win Rate Over Training (per 500 episodes)")
    plt.xlabel("Check-in (every 500 episodes)")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("win_rate.png")  # saves a chart as an image
    plt.show()
    print("Win rate chart saved as win_rate.png")

    return agent

if __name__ == "__main__":
    trained_agent = train(episodes=50000)