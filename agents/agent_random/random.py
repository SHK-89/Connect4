import numpy as np
import math
import importlib

# Ensure we import the standard library random, not this module
_std_random = importlib.import_module("random")

from game_utils import (
    BoardPiece,
    PlayerAction,
    SavedState,
    apply_player_action,
    check_end_state,
    GameState,
    BOARD_COLS,
    BOARD_ROWS,
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
)

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Agent using Negamax with alpha-beta pruning for Connect Four.

    This function avoids conflicts with a local random.py by importing the
    standard library random module via importlib.
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
        for r in range(BOARD_ROWS):
            row_array = list(b[r, :])
            for c in range(BOARD_COLS - 3):
                window = row_array[c : c + 4]
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
    # Use the stdlib random.choice to avoid conflicts
    best_move = _std_random.choice(valid_moves)

    for col in valid_moves:
        nb = board.copy()
        apply_player_action(nb, col, player)
        score = -negamax(nb, PLAYER1 + PLAYER2 - player, DEPTH - 1, -math.inf, math.inf)
        if score > best_score:
            best_score = score
            best_move = col

    return PlayerAction(best_move), saved_state




def generate_move_random_MCTS(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Agent using Monte Carlo Tree Search (MCTS) for Connect Four.
    Each move runs a fixed number of playouts to estimate the best column.
    """
    # MCTS parameters
    ITERATIONS = 1000  # number of simulations
    EXPLORATION_COEF = math.sqrt(2)

    class Node:
        __slots__ = ("state", "player", "parent", "children", "wins", "visits", "untried_moves")

        def __init__(self, state: np.ndarray, player: BoardPiece, parent=None):
            self.state = state
            self.player = player
            self.parent = parent
            self.children = []  # list of Node
            self.wins = 0
            self.visits = 0
            # legal moves from this state
            self.untried_moves = [c for c in range(BOARD_COLS) if state[BOARD_ROWS - 1, c] == NO_PLAYER]

        def uct_select_child(self):
            # pick child with highest UCT value
            log_parent_visits = math.log(self.visits)
            def uct(c):
                return (c.wins / c.visits) + EXPLORATION_COEF * math.sqrt(log_parent_visits / c.visits)
            return max(self.children, key=uct)

        def expand(self):
            # expand by creating a new child for one untried move
            move = self.untried_moves.pop()
            new_state = self.state.copy()
            apply_player_action(new_state, move, self.player)
            next_player = PLAYER1 if self.player == PLAYER2 else PLAYER2
            child = Node(new_state, next_player, parent=self)
            self.children.append(child)
            return child

        def update(self, result_player: BoardPiece):
            self.visits += 1
            if result_player == self.player:
                self.wins += 1

    def simulate(state: np.ndarray, player_to_move: BoardPiece) -> BoardPiece:
        # play random until terminal
        b = state.copy()
        current = player_to_move
        while True:
            st = check_end_state(b, current)
            if st == GameState.IS_WIN:
                # previous player won
                return PLAYER1 if current == PLAYER2 else PLAYER2
            if st == GameState.IS_DRAW:
                return None
            # random playout
            valid = [c for c in range(BOARD_COLS) if b[BOARD_ROWS - 1, c] == NO_PLAYER]
            move = _std_random.choice(valid)
            apply_player_action(b, move, current)
            current = PLAYER1 if current == PLAYER2 else PLAYER2

    # root of search tree
    root = Node(board, player)

    for _ in range(ITERATIONS):
        node = root
        # 1) Selection
        while node.untried_moves == [] and node.children:
            node = node.uct_select_child()
        # 2) Expansion
        if node.untried_moves:
            node = node.expand()
        # 3) Simulation
        result = simulate(node.state, node.player)
        # 4) Backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

    # pick the move with highest visit count
    best_child = max(root.children, key=lambda c: c.visits)
    # determine which column that was
    for col in range(BOARD_COLS):
        # compare states
        st = board.copy()
        apply_player_action(st, col, player)
        if np.array_equal(st, best_child.state):
            return PlayerAction(col), saved_state

    # Fallback
    valid_moves = [c for c in range(BOARD_COLS) if board[BOARD_ROWS - 1, c] == NO_PLAYER]
    return PlayerAction(_std_random.choice(valid_moves)), saved_state