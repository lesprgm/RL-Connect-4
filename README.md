# RL-Connect-4

A lightweight Connect Four web app with four playable modes:

- `human_vs_human`
- `human_vs_genetic_algorithm`
- `human_vs_self_play`
- `human_vs_semi_random_rl`

The RL mode loads the best saved checkpoint from `artifacts/rl/best_dqn.pth`.

## Run

```bash
python3 run.py
```

Then open `http://127.0.0.1:8000`.

## Train the RL agent

```bash
python3 agent/r-learning/train.py
```

That script now saves:

- `artifacts/rl/best_dqn.pth`
- `artifacts/rl/win_rate.png`

## API

- `GET /api/modes` lists supported game modes.
- `POST /api/games` creates a new game with JSON body `{"mode": "human_vs_self_play"}`.
- `GET /api/games/<session_id>` returns the current game state.
- `POST /api/games/<session_id>/move` submits a move with JSON body `{"column": 3}`.
