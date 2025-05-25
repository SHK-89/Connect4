import numpy as np
import math
from game_utils import ( BoardPiece,  BOARD_COLS, BOARD_ROWS,NO_PLAYER, PLAYER1, PLAYER2,apply_player_action, check_end_state, GameState)

EXPLORATION_COEF = math.sqrt(2)  # UCT exploration coefficient

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
            col for col in range(BOARD_COLS) if state[BOARD_ROWS - 1, col] == NO_PLAYER
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
