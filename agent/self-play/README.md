# Self-Play Agent

This folder contains the self-play reinforcement learning training loop, neural network definition, replay buffer, and saved checkpoint used by the Connect Four app.

The training code is in `learningLoop.py`. The deployed agent is loaded through `connect4/agents.py`.

## Board Encoding

The self-play model sees the board from the current player's perspective:

```text
0 = empty cell
1 = current player
2 = opponent
```

The 6x7 board is flattened into 42 inputs. This lets one model play as either Player 1 or Player 2 without treating one color as permanently "self".

## Neural Network

Defined in `neuralNetwork.py`.

```text
42 inputs -> 256 hidden units -> 128 hidden units -> 7 outputs
```

Each output is a Q-value for one Connect Four column.

## Training Loop

Run from the project root:

```bash
./venv/bin/python agent/self-play/learningLoop.py
```

The loop defaults to 50,000 games. These environment variables can override the defaults:

```bash
SELF_PLAY_NUM_GAMES=50000
SELF_PLAY_EVAL_START_GAME=3000
SELF_PLAY_EVAL_INTERVAL=500
SELF_PLAY_EVAL_GAMES=100
SELF_PLAY_CHECKPOINT_PATH=connect4_policy_model.pth
```

`SELF_PLAY_CHECKPOINT_PATH` is resolved relative to `agent/self-play/` unless you provide an absolute path.

Example short smoke run:

```bash
SELF_PLAY_NUM_GAMES=5 SELF_PLAY_EVAL_START_GAME=10000 \
./venv/bin/python agent/self-play/learningLoop.py
```

## RL Setup

The training loop uses:

- **Policy model**: the model being optimized.
- **Target model**: a periodically synced copy used for Q-learning targets.
- **Frozen opponent model**: a periodically synced copy used as a more stable opponent.
- **Semi-random DQN opponent**: the trained teammate model from `agent/r-learning/best_dqn.pth`.
- **Replay buffer**: stores policy transitions and samples batches for training.

If `connect4_policy_model.pth` already exists, training starts from that checkpoint. Otherwise, training starts from random initialization.

## Opponent Mix

Each training game chooses one opponent type:

```text
40% live self-play
30% semi-random DQN opponent
30% frozen older self-play model
```

Only moves made by the live policy model are stored as policy transitions. Frozen-model and semi-random DQN opponent moves are not stored as if they were policy decisions.

## Action Selection During Training

Training action selection is epsilon-greedy:

```text
epsilon chance: random legal move
otherwise: highest legal Q-value from the model
```

The current training loop does not use direct supervised tactical examples. It also does not force immediate wins or blocks during training. Tactical behavior is expected to come from rewards and game outcomes.

## Rewards

The training loop uses RL rewards and reward shaping:

- terminal win: positive reward
- terminal draw: small positive reward
- terminal loss credit is applied to the earlier policy move that allowed the opponent to win
- board-position shaping through `connect4.game.score_position`
- penalty when a move allows the opponent an immediate win

The Bellman target uses the self-play zero-sum form:

```text
target = reward - gamma * opponent_best_next_q
```

This is used because after the current player moves, the next state belongs to the opponent.

## Checkpoint Saving

Every `SELF_PLAY_EVAL_INTERVAL` games after `SELF_PLAY_EVAL_START_GAME`, the current policy is evaluated against the semi-random DQN opponent.

The checkpoint is saved only if this fixed-baseline gameplay score improves:

```text
score = wins + 0.5 * draws
```

normalized by the number of evaluation games.

Saved checkpoint:

```text
agent/self-play/connect4_policy_model.pth
```

If no improved checkpoint is found and an existing checkpoint was loaded, the existing checkpoint is left unchanged. If no checkpoint existed, the final model is saved as a startup fallback so the app has a self-play checkpoint to load.

## Training Outputs

If `matplotlib` is available, the loop saves:

```text
training_loss.png
epsilon_decay.png
training_reward.png
```

Plotting is optional. Missing `matplotlib` will not prevent training from running.

## Deployed Agent Behavior

The app loads the checkpoint through `connect4/agents.py`.

The deployed `SelfPlayAgent` currently uses a small tactical safety layer before asking the neural network:

1. play an immediate winning move if one exists
2. block an immediate opponent win if one exists
3. otherwise use the self-play neural network

This safety layer is not part of the RL training update. It is inference-time Connect Four logic used to avoid obvious one-move losses.

## Evaluation Against Teammate Agent

The project-level `evaluate_agents.py` compares `SelfPlayAgent` with `SemiRandomRLAgent`.

`SemiRandomRLAgent` currently uses:

1. immediate win
2. immediate block
3. 15% random legal move
4. otherwise the teammate DQN move

The 15% random move makes repeated evaluations produce varied games instead of replaying the same deterministic game traces.

The evaluator reports:

- self-play wins
- teammate wins
- draws
- wins by agent and seat
- Player 1 vs Player 2 win rates
- number of unique game traces

This matters because a 50/50 agent result can be misleading if Player 1 wins every game regardless of which agent starts.

## Files

| File | Purpose |
|---|---|
| `learningLoop.py` | Main RL training loop, opponent mix, reward shaping, checkpoint evaluation, checkpoint saving, and plot generation. |
| `neuralNetwork.py` | Defines the `connect4SelfPlayModel` network. |
| `replayBuffer.py` | Experience replay buffer. |
| `connect4_policy_model.pth` | Saved self-play checkpoint loaded by `connect4/agents.py`. |
| `README.md` | Documentation for the current self-play setup. |

## Current Limitations

- The deployed tactical safety layer improves play, but it is not pure neural-network inference.
- Stronger performance as Player 2 likely requires retraining with better opponents and reward pressure against immediate threats.
