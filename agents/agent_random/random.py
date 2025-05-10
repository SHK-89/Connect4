import math
from random import random

import numpy as np

from game_utils import BoardPiece, PlayerAction, SavedState, PLAYER1, PLAYER2, NO_PLAYER, BOARD_COLS, BOARD_ROWS, \
    check_end_state, GameState, apply_player_action

'''def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # Choose a valid, non-full column randomly and return it as `action`
    return action, saved_state'''




def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Agent using Negamax with alpha-beta pruning for Connect Four.
    """
    DEPTH = 4

    def evaluate_window(window: list[BoardPiece], player: BoardPiece) -> int:
        opp = PLAYER1 if player == PLAYER2 else PLAYER2
        if window.count(player) == 4:
            return 100
        elif window.count(player) == 3 and window.count(NO_PLAYER) == 1:
            return 5
        elif window.count(player) == 2 and window.count(NO_PLAYER) == 2:
            return 2
        if window.count(opp) == 3 and window.count(NO_PLAYER) == 1:
            return -4
        return 0

    def heuristic_score(b: np.ndarray, player: BoardPiece) -> int:
        score = 0
        # Score center column
        center_array = list(b[:, BOARD_COLS // 2])
        score += center_array.count(player) * 3

        # Horizontal
        for row in range(BOARD_ROWS):
            row_array = list(b[row, :])
            for col in range(BOARD_COLS - 3):
                window = row_array[col : col + 4]
                score += evaluate_window(window, player)

        # Vertical
        for c in range(BOARD_COLS):
            col_array = list(b[:, c])
            for r in range(BOARD_ROWS - 3):
                window = col_array[r : r + 4]
                score += evaluate_window(window, player)

        # Positive diagonal
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = [b[r + i, c + i] for i in range(4)]
                score += evaluate_window(window, player)

        # Negative diagonal
        for r in range(BOARD_ROWS - 3):
            for c in range(BOARD_COLS - 3):
                window = [b[r + 3 - i, c + i] for i in range(4)]
                score += evaluate_window(window, player)

        return score

    def negamax(b: np.ndarray, current_player: BoardPiece, depth: int, alpha: float, beta: float) -> float:
        state = check_end_state(b, current_player)
        if state == GameState.IS_WIN:
            return math.inf
        elif state == GameState.IS_DRAW:
            return 0
        if depth == 0:
            return heuristic_score(b, current_player)

        max_score = -math.inf
        for col in range(BOARD_COLS):
            # check valid move
            if b[BOARD_ROWS - 1, col] != NO_PLAYER:
                continue
            nb = b.copy()
            apply_player_action(nb, col, current_player)
            score = -negamax(nb, PLAYER1 + PLAYER2 - current_player, depth - 1, -beta, -alpha)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score

    # Get all valid moves
    valid_moves = [c for c in range(BOARD_COLS) if board[BOARD_ROWS - 1, c] == NO_PLAYER]
    best_score = -math.inf
    best_move = random.choice(valid_moves)

    for col in valid_moves:
        nb = board.copy()
        apply_player_action(nb, col, player)
        score = -negamax(nb, PLAYER1 + PLAYER2 - player, DEPTH - 1, -math.inf, math.inf)
        if score > best_score:
            best_score = score
            best_move = col

    return PlayerAction(best_move), saved_state