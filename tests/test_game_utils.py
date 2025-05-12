import numpy as np
import pytest
from game_utils import (
    BOARD_COLS,
    BOARD_ROWS,
    initialize_game_state,
    pretty_print_board,
    string_to_board,
    apply_player_action,
    connected_four,
    check_end_state,
    GameState,
    check_move_status,
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    MoveStatus,
    PlayerAction, INDEX_HIGHEST_ROW, INDEX_LOWEST_ROW
)
def get_empty_board():
    return initialize_game_state()


def test_initialize_game():
    board = get_empty_board()
    assert board.shape == (BOARD_ROWS, BOARD_COLS)
    assert np.all(board == NO_PLAYER)
def test_initialize_game_state():
    board = get_empty_board()
    assert board.shape == (BOARD_ROWS, BOARD_COLS)
    assert np.all(board == NO_PLAYER)

def test_pretty_print_to_string_to_board():
    board = get_empty_board()
    pretty_print = pretty_print_board(board)
    reconstructed_board = string_to_board(pretty_print)
    assert np.array_equal(board, reconstructed_board)


def test_apply_player_action_valid():
    board = get_empty_board()
    action = PlayerAction(0)
    apply_player_action(board, action, PLAYER1)
    assert board[0, 0] == PLAYER1

def test_apply_player_action_full():
    board = get_empty_board()
    action = PlayerAction(0)

    # Fill column 0 to make it full
    for row in range(INDEX_LOWEST_ROW, INDEX_HIGHEST_ROW + 1):
        board[row, 0] = PLAYER1  # Mark the column as full for PLAYER1

    # Try applying another action in the full column (should raise an exception)
    with pytest.raises(ValueError):
        apply_player_action(board, action, PLAYER2)

def test_apply_player_action_upper_bound():
    board = get_empty_board()
    action = PlayerAction(BOARD_COLS)  # Out of bounds, invalid action
    with pytest.raises(ValueError):
        apply_player_action(board, action, PLAYER1)

def test_apply_player_action_lower_bound():
    board = get_empty_board()
    action = PlayerAction(-1)  # Invalid action, lower than 0
    with pytest.raises(ValueError):
        apply_player_action(board, action, PLAYER1)

def test_connected_four_horizontal():
    board = get_empty_board()
    for col in range(4):
        board[0, col] = PLAYER1
    assert connected_four(board, PLAYER1) == True

def test_connected_four_vertical():
    board = get_empty_board()
    for row in range(4):
        board[row, 0] = PLAYER1
    assert connected_four(board, PLAYER1) == True

def test_connected_four_diagonal():
    board = get_empty_board()
    for i in range(4):
        board[i, i] = PLAYER1
    assert connected_four(board, PLAYER1) == True

def test_connected_four_diagonal_reverse():
    board = get_empty_board()
    for i in range(4):
        board[3 - i, i] = PLAYER1
    assert connected_four(board, PLAYER1) == True
def test_connected_four_no_connection():
    board = get_empty_board()
    for col in range(4):
        board[0, col] = PLAYER1
    assert connected_four(board, PLAYER2) == False
def test_connected_four_full_board():
    board = get_empty_board()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row, col] = PLAYER1
    assert connected_four(board, PLAYER1) == True
def test_connected_four_empty_board():
    board = get_empty_board()
    assert connected_four(board, PLAYER1) == False
def test_check_end_state_win():
    board = get_empty_board()
    for col in range(4):
        board[0, col] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN
def test_check_end_state_draw():
    board = get_empty_board()

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            # Fill the board randomly or in a way that no player wins
            board[row, col] = PLAYER1 if (row + col) % 2 == 0 else PLAYER2
    print(board)
    # Check for a draw when the board is full
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW
def test_check_end_state_still_playing():
    board = get_empty_board()
    for col in range(3):
        board[0, col] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

def test_check_move_status_valid():
    board = get_empty_board()
    action = PlayerAction(0)
    status = check_move_status(board, action)
    assert status == MoveStatus.IS_VALID
def test_check_move_status_upper_bound():
    board = get_empty_board()
    action = PlayerAction(BOARD_COLS)  # Out of bounds
    status = check_move_status(board, action)
    assert status == MoveStatus.OUT_OF_BOUNDS
def test_check_move_status_lower_bound():
    board = get_empty_board()
    action = PlayerAction(-1)  # Negative index
    status = check_move_status(board, action)
    assert status == MoveStatus.OUT_OF_BOUNDS
def test_check_move_status_column_full():
    board = get_empty_board()
    action = PlayerAction(0)

    # Fill column 0 to make it full
    for row in range(INDEX_LOWEST_ROW, INDEX_HIGHEST_ROW + 1):
        board[row, 0] = PLAYER1  # Mark the column as full for PLAYER1

    # Check move status for the full column
    status = check_move_status(board, action)
    assert status == MoveStatus.FULL_COLUMN
def test_check_move_status_wrong_type():
    board = get_empty_board()
    action = "invalid_action"  # Wrong type
    status = check_move_status(board, action)
    assert status == MoveStatus.WRONG_TYPE
def test_check_move_status_empty_board():
    board = get_empty_board()
    action = PlayerAction(0)  # Valid action
    status = check_move_status(board, action)
    assert status == MoveStatus.IS_VALID