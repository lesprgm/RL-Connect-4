from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .game import ConnectFourGame, other_player, score_position


DEFAULT_WEIGHTS = {
    "four": 100000,
    "block_four": 95000,
    "three": 120,
    "two": 20,
    "block_three": 140,
    "block_two": 25,
    "center": 6,
}


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
    epsilon: float = 0.25
    seed: int = 21

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.weights = {
            "four": 100000,
            "block_four": 90000,
            "three": 115,
            "two": 16,
            "block_three": 135,
            "block_two": 20,
            "center": 5,
        }

    def choose_move(self, game: ConnectFourGame, player: int) -> int:
        moves = self._valid_moves(game)
        if len(moves) == 1:
            return moves[0]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(moves)

        scored_moves = []
        for column in moves:
            candidate = game.clone()
            candidate.drop_piece(column)
            value = score_position(candidate, player, self.weights)
            scored_moves.append((value, column))
        scored_moves.sort(reverse=True)
        return scored_moves[0][1]


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
