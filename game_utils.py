from typing import Callable, Optional, Any
from enum import Enum
import numpy as np

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros(BOARD_SHAPE, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_str = '|==============|\n'
    numbered_line= '|0 1 2 3 4 5 6 |'
    for row in range(INDEX_HIGHEST_ROW, INDEX_LOWEST_ROW - 1, -1):
        board_str += '| '
        for col in range(BOARD_COLS):
            piece = board[row, col]
            if piece == NO_PLAYER:
                board_str += NO_PLAYER_PRINT + ' '
            elif piece == PLAYER1:
                board_str += PLAYER1_PRINT + ' '
            elif piece == PLAYER2:
                board_str += PLAYER2_PRINT + ' '
        board_str += '|\n'
    board_str += '|==============|\n'
    board_str += numbered_line + '\n'
    return board_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    board = np.zeros(BOARD_SHAPE, dtype=BoardPiece)
    lines = pp_board.strip().split('\n')[1:-2]  # Skip the first and last two lines
    for i, line in enumerate(lines):
        for j, char in enumerate(line[1:-1].strip().split()):
            if char == PLAYER1_PRINT:
                board[i, j] = PLAYER1
            elif char == PLAYER2_PRINT:
                board[i, j] = PLAYER2
            else:
                board[i, j] = NO_PLAYER
    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input
    board should be modified in place, such that it's not necessary to return
    something.
    """
    # Find the lowest open row in the specified column
    for row in range(INDEX_LOWEST_ROW, INDEX_HIGHEST_ROW + 1):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player
            return
    raise ValueError(f"Column {action} is full. Cannot apply player action.")


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """

    # Check horizontal
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS - 3):
            if all(board[row, col + i] == player for i in range(4)):
                return True

    # Check vertical
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 3):
            if all(board[row + i, col] == player for i in range(4)):
                return True

    # Check diagonal (bottom-left to top-right)
    for row in range(BOARD_ROWS - 3):
        for col in range(BOARD_COLS - 3):
            if all(board[row + i, col + i] == player for i in range(4)):
                return True

    # Check diagonal (top-left to bottom-right)
    for row in range(3, BOARD_ROWS):
        for col in range(BOARD_COLS - 3):
            if all(board[row - i, col + i] == player for i in range(4)):
                return True

    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move
    or not, and if not, why.
    The provided column must be of the correct type (PlayerAction).
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    if not isinstance(column, PlayerAction):
        return MoveStatus.WRONG_TYPE
    if column < 0 or column >= BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS
    if board[INDEX_HIGHEST_ROW, column] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN
    return MoveStatus.IS_VALID
