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


_rl_agent_module = _load_rl_agent_module()
DQN = _rl_agent_module.DQN


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


def load_checkpoint(checkpoint_path: Path | str) -> DQN:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported RL checkpoint format: {path}")

    model = DQN()
    model.load_state_dict(payload["model_state_dict"])
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
        self.model = load_checkpoint(self.checkpoint_path)

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
    depth: int = 4

    def __post_init__(self) -> None:
        self.weights = {
            "four": 100000,
            "block_four": 100000,
            "three": 180,
            "two": 32,
            "block_three": 200,
            "block_two": 36,
            "center": 8,
        }

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        score, move = self._minimax(game, self.depth, -math.inf, math.inf, True, player)
        if move is None:
            return self._valid_moves(game)[0]
        return move

    def _minimax(
        self,
        game: ConnectFourGame,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        player: int,
    ) -> tuple[float, int | None]:
        moves = game.available_columns()
        opponent = other_player(player)

        if game.winner == player:
            return 1_000_000 + depth, None
        if game.winner == opponent:
            return -1_000_000 - depth, None
        if game.is_draw:
            return 0, None
        if depth == 0:
            return score_position(game, player, self.weights), None

        ordered_moves = sorted(moves, key=lambda move: abs(3 - move))
        if maximizing:
            best_value = -math.inf
            best_move = ordered_moves[0]
            for move in ordered_moves:
                candidate = game.clone()
                candidate.drop_piece(move)
                value, _ = self._minimax(candidate, depth - 1, alpha, beta, False, player)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_value, best_move

        best_value = math.inf
        best_move = ordered_moves[0]
        for move in ordered_moves:
            candidate = game.clone()
            candidate.drop_piece(move)
            value, _ = self._minimax(candidate, depth - 1, alpha, beta, True, player)
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, best_value)
            if alpha >= beta:
                break
        return best_value, best_move


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
