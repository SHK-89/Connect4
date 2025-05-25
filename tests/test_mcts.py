import math
import time
import numpy as np
import pytest
import importlib

from agents.agent_mcts.node import MCTSNode
from agents.agent_mcts.mcts import MCTSAgent
from agents.agent_random.random_agent import generate_move_random
from game_utils import BOARD_ROWS, BOARD_COLS, PLAYER1, PLAYER2, apply_player_action, check_end_state, GameState, \
    SavedState, PlayerAction, NO_PLAYER


def create_empty_board():
    # Empty board with NO_PLAYER represented by zeros
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)


def create_mcts_agent(iterations=1000) -> MCTSAgent:
    return MCTSAgent(iterations=iterations)

def test_simulate_detects_win(monkeypatch):
    # Create board where PLAYER1 has three in a row in column 0 and can win next
    board = create_empty_board()
    agent = create_mcts_agent()
    for _ in range(3):
        apply_player_action(board, 0, PLAYER1)
    # Now monkeypatch RANDOM.choice to always pick column 0
    monkeypatch.setattr('random.choice', lambda valid: 0)
    result = agent.simulate(board, PLAYER1)
    # After placing the 4th piece, PLAYER1 wins
    assert result == PLAYER1, "simulate should return PLAYER1 as winner when win condition met"


def test_simulate_detects_draw_on_full_board():
    board = create_empty_board()
    agent= create_mcts_agent()
    # Fill board alternating players without making 4 in a row
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS):
            # Alternate piece by column+row to avoid 4 in a row in any direction
            player = PLAYER1 if (col + row) % 2 == 0 else PLAYER2
            apply_player_action(board, col, player)
    # Initial check_end_state should detect draw before any move
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW
    result = agent.simulate(board, PLAYER1)
    assert result is None, "simulate should return None for a draw on a full board"


def test_simulate_random_playout(monkeypatch):
    board = create_empty_board()
    agent = create_mcts_agent()
    monkeypatch.setattr('random.choice', lambda valid: valid[0])
    result = agent.simulate(board, PLAYER1)
    # The result should be either PLAYER1, PLAYER2, or None
    assert result in (PLAYER1, PLAYER2, None)


def test_generate_move_random_fallback(monkeypatch):
    board = create_empty_board()
    monkeypatch.setattr('agents.agent_mcts.mcts.MCTSAgent.iterations', 0)
    monkeypatch.setattr('numpy.random.choice', lambda valid: valid[0])
    action, state = generate_move_random(board, PLAYER1, None)
    # The action should be the first column (0)
    assert isinstance(action, PlayerAction)
    assert int(action) == 0
    # saved_state stays None
    assert state is None


def test_generate_move_random_runs_with_positive_iterations(monkeypatch):
    board = create_empty_board()
    # Use just 1 iteration so it terminates quickly
    monkeypatch.setattr('agents.agent_mcts.mcts.MCTSAgent.iterations', 1)
    # Force simulate to return current player immediately
    monkeypatch.setattr('agents.agent_mcts.mcts.MCTSAgent.simulate', lambda state, player: player)
    action, state = generate_move_random(board, PLAYER1, None)
    # After one expansion the only popped move is last column
    expected_col = BOARD_COLS - 1
    assert isinstance(action, PlayerAction)
    assert action == expected_col
    # saved_state stays None
    assert state is None


def test_selects_expanded_move(monkeypatch):
    board = create_empty_board()
    # Create an MCTSAgent with 1 iteration
    agent = create_mcts_agent(iterations=1)
    # Make simulate fast and deterministic: always return current player (no effect backprop other than wins)
    monkeypatch.setattr('agents.agent_mcts.mcts.MCTSAgent.simulate', lambda self, state, player: player)
    # Force random.choice to always return the last column
    monkeypatch.setattr('numpy.random.choice', lambda valid: valid[-1])
    # Run agent
    action, state = agent.generate_move(board, PLAYER1, None)
    # After one expansion, the popped column is BOARD_COLS-1
    expected_col = BOARD_COLS - 1
    assert int(action) == expected_col, f"Expected expanded move {expected_col}, got {action}"
    assert state is None


def test_saved_state_preserved(monkeypatch):
    board = create_empty_board()
    # Zero iterations to fallback
    monkeypatch.setattr('agents.agent_mcts.mcts.MCTSAgent.iterations', 0)
    monkeypatch.setattr('numpy.random.choice', lambda valid: valid[0])
    dummy_state = SavedState()
    action, returned_state = generate_move_random(board, PLAYER1, dummy_state)

    # Ensure the returned state is the same as the dummy state
    assert returned_state is dummy_state
    assert isinstance(returned_state, SavedState)

    # Ensure the action is valid
    valid_moves = [col for col in range(board.shape[1]) if board[0, col] == NO_PLAYER]
    assert action in valid_moves
def test_single_valid_move():
    board = create_empty_board()
    # Fill all columns except the last one
    for col in range(BOARD_COLS - 1):
        for row in range(BOARD_ROWS):
            apply_player_action(board, col, PLAYER1 if (col + row) % 2 == 0 else PLAYER2)

    # Create an MCTSAgent
    agent = create_mcts_agent(iterations=10)

    # Run the agent to generate a move
    action, state = agent.generate_move(board, PLAYER1, None)

    # Ensure the action is the only valid column
    expected_col = BOARD_COLS - 1
    assert action == expected_col, f"Expected move {expected_col}, got {action}"
    assert state is None

def test_performance_with_large_iterations():
    board = create_empty_board()
    # Create an MCTSAgent with a large number of iterations
    large_iterations = 10000
    agent = create_mcts_agent(iterations=large_iterations)

    start_time = time.time()
    # Run the agent to generate a move
    action, state = agent.generate_move(board, PLAYER1, None)
    end_time = time.time()

    # Ensure the action is valid
    valid_moves = [col for col in range(board.shape[1]) if board[0, col] == NO_PLAYER]
    assert action in valid_moves, f"Action {action} is not a valid move."

    # Print the execution time for performance analysis
    execution_time = end_time - start_time
    print(f"Execution time for {large_iterations} iterations: {execution_time:.2f} seconds")
