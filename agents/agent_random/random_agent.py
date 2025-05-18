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
EXPLORATION_COEF = math.sqrt(2) # UCT exploration coefficient
#RANDOM = random

class MCTSNode:
    """
        A node in the Monte Carlo Tree Search (MCTS) tree.

        Attributes:
            state (np.ndarray): the current board state.
            player (BoardPiece): the player who will make the next move.
            parent (MCTSNode | None): the parent node of this node, or None if this is the root node.
            children (list[MCTSNode]): the child nodes of this node.
            wins (int): the number of wins for this node.
            visits (int): the number of visits to this node.
            untried_moves (list[int]): the legal moves from this state that have not yet been tried.
    """
    __slots__ = (
        "state", "player", "parent", "children", "wins", "visits", "untried_moves"
    )

    def __init__(self, state: np.ndarray, player: BoardPiece, parent=None):
        """
            Initialize a new MCTSNode.

            Parameters:
                state (np.ndarray): the current board state.
                player (BoardPiece): the player who will make the next move.
                parent (MCTSNode | None): the parent node of this node, or None if this is the root node.
        """
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
        """
            Calculate the Upper Confidence Bound for Trees (UCT) value of this node.

            Parameters:
                total_simulation (int): the total number of simulations performed.

            Returns:
                float: the UCT value of this node.
        """
        if self.visits == 0:
            return float('inf')
        win_rate = self.wins / self.visits
        exploration = EXPLORATION_COEF * math.sqrt(math.log(total_simulation) / self.visits)
        return win_rate + exploration

    def select_child(self) -> 'MCTSNode':
        """
            Select the child node with the highest UCT value.

            Returns:
                MCTSNode: the child node with the highest UCT value.
        """
        return max(self.children, key=lambda c: c.uct_value(self.visits))

    def expand(self) -> 'MCTSNode':
        """
            Expand this node by selecting one of its untried moves and creating a child node.

            Returns:
                MCTSNode: the newly created child node.
        """

        move = self.untried_moves.pop()
        new_state = self.state.copy()
        apply_player_action(new_state, move, self.player)
        next_player = PLAYER1 if self.player == PLAYER2 else PLAYER2
        child = MCTSNode(new_state, next_player, parent=self)
        self.children.append(child)
        return child

    def update(self, result_player: BoardPiece | None):
        """
            Update this node's statistics after a simulation.

            Parameters:
                result_player (BoardPiece | None): the player who won the simulation, or None for a draw.

            Side-effects:
                Increments `visits` by 1. If `result_player` matches this node's player, also increments `wins` by 1.
        """
        self.visits += 1
        if result_player == self.player:
            self.wins += 1


def simulate(state: np.ndarray, player_to_move: BoardPiece) -> BoardPiece | None:
    """
        Play a random playout from the given board state until the game ends.

        Parameters:
            state (np.ndarray): the current board state.
            player_to_move (BoardPiece): which player will make the next move.

        Returns:
            BoardPiece | None: the winning player, or None if the playout ends in a draw.
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
        move = random.choice(valid)
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
    if ITERATIONS <= 0:
        # Fallback to a random valid move
        valid_moves = [col for col in range(board.shape[1]) if board[0, col] == NO_PLAYER]
        if not valid_moves:
            raise ValueError("No valid moves available.")
        action = random.choice(valid_moves)
        return action, saved_state

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
    return PlayerAction(random.choice(valid_moves)), saved_state

# Alias for compatibility
random_agent = generate_move_random