import numpy as np
import math
import random

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

# Module-level parameters, overrideable for testing
ITERATIONS = 1000
EXPLORATION_COEF = math.sqrt(2)
RANDOM = random

class MCTSNode:
    __slots__ = (
        "state", "player", "parent", "children", "wins", "visits", "untried_moves"
    )

    def __init__(self, state: np.ndarray, player: BoardPiece, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        # legal moves from this state
        self.untried_moves = [
            c for c in range(BOARD_COLS) if state[BOARD_ROWS - 1, c] == NO_PLAYER
        ]

    def uct_value(self, total_simulation: int) -> float:
        if self.visits == 0:
            return float('inf')
        win_rate = self.wins / self.visits
        exploration = EXPLORATION_COEF * math.sqrt(math.log(total_simulation) / self.visits)
        return win_rate + exploration

    def select_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.uct_value(self.visits))

    def expand(self) -> 'MCTSNode':
        move = self.untried_moves.pop()
        new_state = self.state.copy()
        apply_player_action(new_state, move, self.player)
        next_player = PLAYER1 if self.player == PLAYER2 else PLAYER2
        child = MCTSNode(new_state, next_player, parent=self)
        self.children.append(child)
        return child

    def update(self, result_player: BoardPiece | None):
        self.visits += 1
        if result_player == self.player:
            self.wins += 1


def simulate(state: np.ndarray, player_to_move: BoardPiece) -> BoardPiece | None:
    """
    Play a random playout until terminal, returning the winner or None for draw.
    """
    b = state.copy()
    current = player_to_move
    while True:
        current_state = check_end_state(b, current)
        if current_state == GameState.IS_WIN:
            # previous player won
            return PLAYER1 if current == PLAYER2 else PLAYER2
        if current_state == GameState.IS_DRAW:
            return None
        # random playout
        valid = [c for c in range(BOARD_COLS) if b[BOARD_ROWS - 1, c] == NO_PLAYER]
        move = RANDOM.choice(valid)
        apply_player_action(b, move, current)
        current = PLAYER1 if current == PLAYER2 else PLAYER2


def generate_move_random(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: SavedState | None,
) -> tuple[PlayerAction, SavedState | None]:
    """
    Monte Carlo Tree Search on a fixed number of iterations.
    """
    root = MCTSNode(board, player)

    for _ in range(ITERATIONS):
        node = root
        # 1) Selection
        while not node.untried_moves and node.children:
            node = node.select_child()
        # 2) Expansion
        if node.untried_moves:
            node = node.expand()
        # 3) Simulation
        result = simulate(node.state, node.player)
        # 4) Backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

    # choose the child with highest visit count
    best_child = max(root.children, key=lambda c: c.visits)

    # find the move leading to best_child
    for col in range(BOARD_COLS):
        trial = board.copy()
        apply_player_action(trial, col, player)
        if np.array_equal(trial, best_child.state):
            return PlayerAction(col), saved_state

    # fallback
    valid_moves = [c for c in range(BOARD_COLS) if board[BOARD_ROWS - 1, c] == NO_PLAYER]
    return PlayerAction(RANDOM.choice(valid_moves)), saved_state

# Alias for compatibility
random_agent = generate_move_random