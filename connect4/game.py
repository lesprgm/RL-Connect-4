from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2
CONNECT_N = 4


def other_player(player: int) -> int:
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


@dataclass
class ConnectFourGame:
    board: List[List[int]] = field(
        default_factory=lambda: [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
    )
    current_player: int = PLAYER_ONE
    winner: Optional[int] = None
    is_draw: bool = False
    last_move: Optional[Tuple[int, int]] = None
    move_count: int = 0

    def clone(self) -> "ConnectFourGame":
        return ConnectFourGame(
            board=[row[:] for row in self.board],
            current_player=self.current_player,
            winner=self.winner,
            is_draw=self.is_draw,
            last_move=self.last_move,
            move_count=self.move_count,
        )

    def available_columns(self) -> List[int]:
        return [column for column in range(COLUMNS) if self.board[0][column] == EMPTY]

    def is_over(self) -> bool:
        return self.winner is not None or self.is_draw

    def drop_piece(self, column: int) -> Tuple[int, int]:
        if self.is_over():
            raise ValueError("The game is already over.")
        if column < 0 or column >= COLUMNS:
            raise ValueError("Column is out of range.")
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][column] == EMPTY:
                self.board[row][column] = self.current_player
                self.last_move = (row, column)
                self.move_count += 1
                if self._is_winning_move(row, column, self.current_player):
                    self.winner = self.current_player
                elif not self.available_columns():
                    self.is_draw = True
                else:
                    self.current_player = other_player(self.current_player)
                return row, column
        raise ValueError("Column is full.")

    def _count_direction(
        self, row: int, column: int, player: int, row_step: int, col_step: int
    ) -> int:
        count = 0
        r = row + row_step
        c = column + col_step
        while 0 <= r < ROWS and 0 <= c < COLUMNS and self.board[r][c] == player:
            count += 1
            r += row_step
            c += col_step
        return count

    def _is_winning_move(self, row: int, column: int, player: int) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for row_step, col_step in directions:
            total = 1
            total += self._count_direction(row, column, player, row_step, col_step)
            total += self._count_direction(row, column, player, -row_step, -col_step)
            if total >= CONNECT_N:
                return True
        return False

    def to_dict(self, mode: str) -> dict:
        return {
            "mode": mode,
            "board": self.board,
            "current_player": self.current_player,
            "winner": self.winner,
            "is_draw": self.is_draw,
            "is_over": self.is_over(),
            "last_move": self.last_move,
            "available_columns": self.available_columns(),
            "move_count": self.move_count,
        }


def iter_windows(board: Sequence[Sequence[int]]) -> Iterable[Tuple[int, int, int, int]]:
    for row in range(ROWS):
        for column in range(COLUMNS - 3):
            yield tuple(board[row][column + offset] for offset in range(CONNECT_N))
    for row in range(ROWS - 3):
        for column in range(COLUMNS):
            yield tuple(board[row + offset][column] for offset in range(CONNECT_N))
    for row in range(ROWS - 3):
        for column in range(COLUMNS - 3):
            yield tuple(board[row + offset][column + offset] for offset in range(CONNECT_N))
    for row in range(ROWS - 3):
        for column in range(3, COLUMNS):
            yield tuple(board[row + offset][column - offset] for offset in range(CONNECT_N))


def evaluate_window(window: Sequence[int], player: int, weights: dict) -> int:
    opponent = other_player(player)
    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(EMPTY)

    if player_count == 4:
        return weights["four"]
    if opponent_count == 4:
        return -weights["block_four"]
    if player_count == 3 and empty_count == 1:
        return weights["three"]
    if player_count == 2 and empty_count == 2:
        return weights["two"]
    if opponent_count == 3 and empty_count == 1:
        return -weights["block_three"]
    if opponent_count == 2 and empty_count == 2:
        return -weights["block_two"]
    return 0


def score_position(game: ConnectFourGame, player: int, weights: dict) -> int:
    score = 0
    center_column = [game.board[row][COLUMNS // 2] for row in range(ROWS)]
    score += center_column.count(player) * weights.get("center", 0)
    for window in iter_windows(game.board):
        score += evaluate_window(list(window), player, weights)
    return score
