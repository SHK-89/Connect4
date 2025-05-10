import numpy as np
from game_utils import BoardPiece, NO_PLAYER, BOARD_SHAPE, PLAYER1, PLAYER2, pretty_print_board, apply_player_action, connected_four, check_end_state, check_move_status
from game_utils import initialize_game_state
import unittest

class test_game_utils(unittest.TestCase):
    def test_initialize_game_state(self):

        game_state = initialize_game_state()

        assert isinstance(game_state, np.ndarray)
        assert game_state.dtype == BoardPiece
        assert game_state.shape == BOARD_SHAPE
        assert np.all(game_state == NO_PLAYER)
        self.assertTrue(np.all(game_state == 0), "The board is not correctly initialized to 0.")

    def test_pretty_print_board(self):
        # Test with an empty board
        board = initialize_game_state()
        expected_output = (
            "|==============|\n"    
            "|              |\n"
            "|              |\n"
            "|              |\n"
                
            "|              |\n"
            "|              |\n"
            "|              |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)
        # Test with a board with some pieces
        board[0, 0] = PLAYER1
        board[1, 1] = PLAYER2
        board[2, 2] = PLAYER1
        expected_output = (
            "|==============|\n"
            "|              |\n"
            "|              |\n"
            "|    X         |\n"
            "|              |\n"
            "|              |\n"
            "|        O     |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)
        # Test with a board with all pieces
        board = np.array([
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER]
        ], dtype=BoardPiece)
        expected_output = (
            "|==============|\n"
            "|X O X O X O X |\n"
            "|O X O X O X O |\n"
            "|X O X O X O X |\n"
            "|O X O X O X O |\n"
            "|X O X O X O X |\n"
            "|O              |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)
        # Test with a board with invalid pieces
        board[0, 0] = 3
        board[1, 1] = 4
        board[2, 2] = 5
        expected_output = (
            "|==============|\n"
            "|              |\n"
            "|              |\n"
            "|    3         |\n"
            "|              |\n"
            "|              |\n"
            "|        4     |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)
        # Test with a board with negative pieces
        board[0, 0] = -1
        board[1, 1] = -2
        board[2, 2] = -3
        expected_output = (
            "|==============|\n"
            "|              |\n"
            "|              |\n"
            "|    -1        |\n"
            "|              |\n"
            "|              |\n"
            "|        -2    |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)
        # Test with a board with all pieces negative
        board = np.array([
            [-1, -2, -1, -2, -1, -2, -1],
            [-2, -1, -2, -1, -2, -1, -2],
            [-1, -2, -1, -2, -1, -2, -1],
            [-2, -1, -2, -1, -2, -1, -2],
            [-1, -2, -1, -2, -1, -2, -1],
            [-2, -1, -2, -1, -2, -1, 0],
            [0]
        ], dtype=BoardPiece)
        expected_output = (
            "|==============|\n"
            "|X O X O X O X |\n"
            "|O X O X O X O |\n"
            "|X O X O X O X |\n"
            "|O X O X O X O |\n"
            "|X O X O X O X |\n"
            "|O              |\n"
            "|==============|\n"
            "|0 1 2 3 4 5 6 |\n"
        )
        self.assertEqual(pretty_print_board(board), expected_output)

    def test_apply_player_action(self):
        # Test with an empty board
        board = initialize_game_state()
        apply_player_action(board, 0, PLAYER1)
        expected_board = np.array([
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertTrue(np.array_equal(board[0], expected_board[0]), "The player action was not applied correctly.")
        # Test with a full column
        board = np.array([
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        with self.assertRaises(ValueError):
            apply_player_action(board, 0, PLAYER1)


    def test_connect_4(self):
        # Test with a horizontal connect 4
        board = np.array([
            [PLAYER1, PLAYER1, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertTrue(connected_four(board), "Horizontal connect 4 not detected.")

        # Test with a vertical connect 4
        board = np.array([
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        ], dtype=BoardPiece)
        self.assertTrue(connected_four(board), "Vertical connect 4 not detected.")

        # Test with a diagonal connect 4
        board = np.array([
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER1],
        ], dtype=BoardPiece)
        self.assertTrue(connected_four(board), "Diagonal connect 4 not detected.")

        # Test with a diagonal connect 4
        board = np.array([
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
        ], dtype=BoardPiece)
        self.assertTrue(connected_four(board), "Diagonal connect 4 not detected.")



        # Test with no connect 4
        board = np.array([
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER2, PLAYER1],
        ], dtype=BoardPiece)
        self.assertFalse(connected_four(board), "False positive for connect 4 detected.")
        # Test with an empty board
        board = np.array([
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertFalse(connected_four(board), "False positive for connect 4 detected.")

    def test_check_end_state(self):
        # Test with a horizontal connect 4
        board = np.array([
            [PLAYER1, PLAYER1, PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertEqual(check_end_state(board), "Player 1 wins", "Horizontal connect 4 not detected.")

        # Test with a vertical connect 4
        board = np.array([
            [PLAYER1, PLAYER2, PLAYER2],
            [PLAYER1, PLAYER2],
            [PLAYER1],
            [PLAYER1]
        ], dtype=BoardPiece)
        self.assertEqual(check_end_state(board), "Player 1 wins", "Vertical connect 4 not detected.")

        # Test with a diagonal connect 4
        board = np.array([
            [NO_PLAYER, PLAYER2],
            [NO_PLAYER],
            [PLAYER2]
        ], dtype=BoardPiece)
        self.assertEqual(check_end_state(board), "Player 2 wins", "Diagonal connect 4 not detected.")

        # Test with no connect 4
        board = np.array([
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertEqual(check_end_state(board), "No winner", "False positive for connect 4 detected.")

    def test_check_move_statue(self):
        # Test with a valid move
        board = np.array([
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER],
            [NO_PLAYER, NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER1],
            [NO_PLAYER, PLAYER2],
            [PLAYER1]
        ], dtype=BoardPiece)
        self.assertEqual(check_move_status(board), "Valid move", "Valid move not detected.")

        # Test with an invalid move
        board = np.array([
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [PLAYER1] * BOARD_SHAPE[0],
            [PLAYER2] * BOARD_SHAPE[0],
            [NO_PLAYER]
        ], dtype=BoardPiece)
        self.assertEqual(check_move_status(board), "Invalid move", "Invalid move not detected.")