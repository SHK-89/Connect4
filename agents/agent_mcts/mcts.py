import numpy as np
import math
import random

from agents.agent_mcts.node import MCTSNode

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

EXPLORATION_COEF = math.sqrt(2)  # UCT exploration coefficient

class MCTSAgent:
    #__slots__ = ("iterations")
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations

    def simulate(self, state: np.ndarray, player_to_move: BoardPiece) -> BoardPiece | None:
        """
               Play a random playout from the given board state until the game ends.

               Parameters:
                   state (np.ndarray): the current board state.
                   player_to_move (BoardPiece): which player will make the next move.

               Returns:
                   BoardPiece | None: the winning player, or None if the playout ends in a draw.
        """

        simulated_state = state.copy()
        current = player_to_move
        while True:
            current_state = check_end_state(simulated_state, current)
            if current_state == GameState.IS_WIN:
                return PLAYER1 if current == PLAYER2 else PLAYER2
            if current_state == GameState.IS_DRAW:
                return None
            valid = [col for col in range(BOARD_COLS) if simulated_state[BOARD_ROWS - 1, col] == NO_PLAYER]
            move = random.choice(valid)
            apply_player_action(simulated_state, move, current)
            current = PLAYER1 if current == PLAYER2 else PLAYER2

    def generate_move(self, board: np.ndarray, player: BoardPiece, saved_state: SavedState | None) -> tuple[PlayerAction, SavedState | None]:
        """
        Monte Carlo Tree Search on a fixed number of iterations.
        Parameters:
            board (np.ndarray): the current game board.
            player (BoardPiece): the player who will make the next move.
            saved_state (SavedState | None): any saved state from previous moves.
        Returns:
            tuple[PlayerAction, SavedState | None]: the chosen action and the updated saved state.
        """

        valid_moves = [col for col in range(board.shape[1]) if board[0, col] == NO_PLAYER]
        if not valid_moves:
            raise ValueError("No valid moves available.")

        if self.iterations <= 0:
            # Fallback to random move if no iterations are set
            action = int(random.choice(valid_moves))
            return action, saved_state

        root = MCTSNode(board, player)

        for _ in range(self.iterations):
            node = root
            # 1) Selection
            while not node.untried_moves and node.children:
                node = node.select_child()
            # 2) Expansion
            if node.untried_moves:
                node = node.expand()
            # 3) Simulation
            result = self.simulate(node.state, node.player)
            # 4) Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

        # choose the child with highest visit count
        best_child = max(root.children, key=lambda c: c.visits)

        # find the move leading to best_child
        for col in valid_moves:
            trial = board.copy()
            apply_player_action(trial, col, player)
            if np.array_equal(trial, best_child.state):
                return PlayerAction(col), saved_state

        # fallback
        return PlayerAction(random.choice(valid_moves)), saved_state

