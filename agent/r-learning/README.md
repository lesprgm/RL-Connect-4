# R-Learning Agent (Semi-Random RL)

A Deep Q-Network (DQN) agent that learns to play Connect Four through self-play against a semi-random opponent.

What changed recently (the two big improvements you see reflected in the code):
- Target network: a frozen copy of the online network is used to compute stable TD targets. The target network is updated periodically (every 500 episodes) via `update_target()`.
- Learn step cap: per-episode learning now performs a small, capped number of gradient updates (2) instead of a single update. This keeps training time predictable while still learning effectively from replay.

## How It Works

The agent uses a neural network (42 inputs → 128 → 128 → 7 outputs) to estimate Q-values for each column given the board state. Training uses:

- **Epsilon-greedy exploration** — starts fully random (`ε=1.0`), decays by the schedule to a minimum of `0.05`.
- **Experience replay** — stores 100,000 past transitions `(s, a, r, s', done)` (the memory size was increased to improve sample diversity), samples random batches of 64 for training.
- **TD learning** — `target = r + γ·max(Q_target(s', a'))` with `γ=0.95`, minimized via MSE loss and Adam optimizer. The max is taken from the frozen target network.
- **Dense rewards** — a small shaping signal (center control, three-in-a-row threats) in addition to terminal rewards.

The opponent plays randomly but opportunistically takes/block winning moves, providing a curriculum as the agent improves.

## Training

Run from the project root:

```bash
python3 agent/r-learning/train.py
```

Default: 50,000 episodes. Prints progress every 500 episodes and saves:

- agent/r-learning/best_dqn.pth — checkpoint with the best win rate
- agent/r-learning/win_rate.png — win rate plot over training

## Playing Against the Agent

```bash
python3 run.py
```

Open `http://127.0.0.1:8000` and select **Human vs Semi-Random RL Agent**.

## Files

| File | Purpose |
|------|---------|
| `agent.py` | DQN network, ReplayMemory, and Agent class with `select_action`, `store`, and `learn` |
| `train.py` | Training loop, semi-random opponent, reward functions, episode runner |
| `README.md` | This readme (updated with the latest changes) |
