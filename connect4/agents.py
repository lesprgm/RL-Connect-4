from __future__ import annotations

import math
import random
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import List

import numpy as np
import torch

from .game import ConnectFourGame, other_player, score_position


ROWS = 6
COLUMNS = 7
ROOT = Path(__file__).resolve().parent.parent
RL_AGENT_PATH = ROOT / "agent" / "r-learning" / "agent.py"
Self_Play_Model_PATH = ROOT / "agent" / "self-play" / "neuralNetwork.py"
Self_Play_Weight_PATH = ROOT / "agent" / "self-play" / "connect4_policy_model.pth"


DEFAULT_WEIGHTS = {
    "four": 100000,
    "block_four": 95000,
    "three": 120,
    "two": 20,
    "block_three": 140,
    "block_two": 25,
    "center": 6,
}


def _load_rl_agent_module():
    spec = importlib.util.spec_from_file_location("rl_agent_module", RL_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RL agent module from {RL_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _load_self_play_model():
    spec = importlib.util.spec_from_file_location("self_play_model_module", Self_Play_Model_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load self-play model module from {Self_Play_Model_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rl_agent_module = _load_rl_agent_module()
DQN = _rl_agent_module.DQN

_self_play_model_module = _load_self_play_model()
connect4SelfPlayModel = _self_play_model_module.connect4SelfPlayModel


def encode_board(board: List[List[int]], agent_piece: int, opponent_piece: int) -> np.ndarray:
    encoded = np.zeros((ROWS, COLUMNS), dtype=np.float32)
    for row in range(ROWS):
        for column in range(COLUMNS):
            cell = board[row][column]
            if cell == agent_piece:
                encoded[row][column] = 1.0
            elif cell == opponent_piece:
                encoded[row][column] = 2.0
    return encoded.reshape(-1)


def choose_best_move(
    model: DQN,
    board: List[List[int]],
    valid_columns: List[int],
    agent_piece: int,
    opponent_piece: int,
) -> int:
    state_tensor = torch.from_numpy(
        encode_board(board, agent_piece=agent_piece, opponent_piece=opponent_piece)
    ).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0).detach().cpu().numpy()
    masked = np.full(COLUMNS, -1e9, dtype=np.float32)
    for column in valid_columns:
        masked[column] = q_values[column]
    return int(np.argmax(masked))


def load_checkpoint(checkpoint_path: Path | str, model_class) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload

    model = model_class()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def save_checkpoint(checkpoint_path: Path | str, model: DQN) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


class BaseAgent:
    name = "base"

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        raise NotImplementedError

    def _valid_moves(self, game: ConnectFourGame) -> List[int]:
        return game.available_columns()


@dataclass
class GeneticAlgorithmAgent(BaseAgent):
    name = "genetic_algorithm"
    seed: int = 7

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.weights = DEFAULT_WEIGHTS.copy()
        self.weights["three"] += rng.randint(-15, 20)
        self.weights["two"] += rng.randint(-3, 5)
        self.weights["block_three"] += rng.randint(-10, 10)
        self.weights["center"] += rng.randint(0, 4)

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        best_score = -math.inf
        best_move = self._valid_moves(game)[0]
        for column in self._valid_moves(game):
            candidate = game.clone()
            candidate.drop_piece(column)
            score = score_position(candidate, player, self.weights)
            if score > best_score:
                best_score = score
                best_move = column
        return best_move


@dataclass
class SemiRandomRLAgent(BaseAgent):
    name = "semi_random_rl"
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_path is None:
            self.checkpoint_path = Path(__file__).resolve().parent.parent / "artifacts" / "rl" / "best_dqn.pth"
        self.model = load_checkpoint(self.checkpoint_path, DQN)

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        moves = self._valid_moves(game)
        if len(moves) == 1:
            return moves[0]
        return choose_best_move(
            self.model,
            game.board,
            moves,
            agent_piece=player,
            opponent_piece=other_player(player),
        )


@dataclass
class SelfPlayAgent(BaseAgent):
    name = "self_play"
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_path is None:
            self.checkpoint_path = Self_Play_Weight_PATH
        self.model = load_checkpoint(self.checkpoint_path, connect4SelfPlayModel)

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        moves = self._valid_moves(game)
        if len(moves) == 1:
            return moves[0]
        return choose_best_move(
            self.model,
            game.board,
            moves,
            agent_piece=player,
            opponent_piece=other_player(player),
        )

def build_agent(mode: str) -> BaseAgent | None:
    if mode == "human_vs_human":
        return None
    if mode == "human_vs_genetic_algorithm":
        return GeneticAlgorithmAgent()
    if mode == "human_vs_self_play":
        return SelfPlayAgent()
    if mode == "human_vs_semi_random_rl":
        return SemiRandomRLAgent()
    raise ValueError(f"Unsupported mode: {mode}")
