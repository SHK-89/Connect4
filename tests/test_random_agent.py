import numpy as np
import pytest
from agents.agent_random.random_agent import generate_move_random
from game_utils import  PlayerAction, SavedState, NO_PLAYER, BOARD_COLS, BOARD_ROWS

create_empty_board = lambda: np.full((BOARD_ROWS, BOARD_COLS), NO_PLAYER, dtype=int)
def test_generate_move_random_valid_move():
    board = create_empty_board()
    action, state = generate_move_random(board, NO_PLAYER, None)
    assert isinstance(action, PlayerAction), "Action should be of type PlayerAction"
    assert 0 <= action < BOARD_COLS, "Action should be within valid column range"


def test_generate_move_random_no_valid_moves():
    board = np.ones((BOARD_ROWS, BOARD_COLS), dtype=int)  # Fill the board
    with pytest.raises(ValueError, match="No valid moves available."):
        generate_move_random(board, NO_PLAYER, None)


def test_generate_move_random_preserves_saved_state():
    board = create_empty_board()
    dummy_state = SavedState()
    action, returned_state = generate_move_random(board, NO_PLAYER, dummy_state)
    assert returned_state is dummy_state, "Saved state should remain unchanged"


def test_generate_move_random_randomness(monkeypatch):
    board = create_empty_board()
    valid_moves = [col for col in range(BOARD_COLS)]
    monkeypatch.setattr('numpy.random.choice', lambda _: valid_moves[0])  # Force first column
    action, _ = generate_move_random(board, NO_PLAYER, None)
    assert action == valid_moves[0], "Action should match the forced random choice"