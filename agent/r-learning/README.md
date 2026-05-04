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

## Known Limitations

- **Plateaus around ~78% win rate** — the agent peaks against the semi-random opponent but doesn't break past it. More episodes alone won't help much; the bottleneck is architectural and algorithmic.
- **Q-value overestimation** — standard DQN uses `max(Q(s', a'))` for the target, which tends to overestimate action values, especially in noisy environments. This can make the agent overly confident in mediocre moves.
- **Weak against strategic opponents** — the agent only trains against the semi-random opponent (takes wins, blocks wins, otherwise random). It has never seen a minimax-style or lookahead opponent, so it struggles against agents that plan ahead.
- **Small network** — 128→128→7 is enough to beat random play but likely too small to capture deeper positional patterns (multi-move threats, forced sequences).
- **No opponent diversity** — training against a single opponent type means the agent overfits to that opponent's weaknesses. It doesn't generalize well to different play styles.

## Future Work

- **Double DQN** — decouple action selection from evaluation in the Bellman target to reduce Q-value overestimation. This is a small code change (one extra line in `learn()`) that typically gives a meaningful win-rate boost.
- **Larger or deeper network** — try 42→256→128→7 or add a third hidden layer. More capacity lets the network represent more complex board patterns.
- **Mixed training opponents** — alternate between semi-random, a minimax agent (depth 2–3), and earlier versions of the agent itself. This prevents overfitting to one style.
- **Reward shaping improvements** — the current mid-game reward is simple (center control + three-in-a-row). Adding penalties for giving the opponent a winning threat, or bonuses for creating double-threats (forks), could speed up learning.
- **Prioritized experience replay** — sample more from transitions with high TD error (where the network was most surprised), instead of uniform random sampling. This focuses learning on the most informative experiences.
- **Self-play training loop** — retrain the agent against its own past checkpoints (like the self-play agent in `agent/self-play/`), so the opponent gradually strengthens as the agent improves. This is how strong game-playing agents are typically built.
