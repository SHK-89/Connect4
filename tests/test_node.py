import math

import numpy as np
import pytest

from agents.agent_mcts.node import MCTSNode
from agents.agent_mcts.node import EXPLORATION_COEF
from game_utils import BOARD_ROWS, BOARD_COLS, PLAYER1, PLAYER2


def create_empty_board():
    # Empty board with NO_PLAYER represented by zeros
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)


board = create_empty_board()
parent = object()
player = PLAYER1
node = MCTSNode(state=board, player=PLAYER1, parent=parent)


def create_node(parent=parent) -> MCTSNode:
    board = create_empty_board()
    node = MCTSNode(state=board, player=player, parent=parent)
    return node


def test_state_player_and_parent_are_recorded():
    node = create_node(parent=parent)
    assert np.array_equal(node.state, board)
    assert node.player == PLAYER1
    assert node.parent is parent


def test_children_wins_and_visits_start_empty_or_zero():
    node = create_node()
    assert node.children == []
    assert node.wins == 0
    assert node.visits == 0


def test_untried_moves_on_empty_board_covers_all_columns():
    node = create_node()
    assert sorted(node.untried_moves) == list(range(BOARD_COLS))


def test_should_not_mutate_untried_moves():
    node = create_node()
    original_moves = node.untried_moves.copy()
    node.untried_moves.pop()
    new_node = MCTSNode(state=board, player=PLAYER1, parent=parent)
    assert new_node.untried_moves == original_moves, "Untried moves should not be affected by other nodes"


def test_uct_value_infinite_when_unvisited():
    node = create_node()
    node.visits = 0
    # Any total_simulation value, should return infinity
    assert node.uct_value(total_simulation=0) == float('inf')
    assert node.uct_value(total_simulation=100) == float('inf')


def test_uct_value_computation_with_visits():
    node = create_node()
    # Set wins and visits to known values
    node.visits = 10
    node.wins = 4
    total_sim = 100
    # Manual computation
    win_rate = node.wins / node.visits
    exploration = EXPLORATION_COEF * math.sqrt(math.log(total_sim) / node.visits)
    expected = win_rate + exploration

    result = node.uct_value(total_simulation=total_sim)
    assert pytest.approx(result, rel=1e-6) == expected


def test_uct_value_edge_case_total_sim_one():
    node = create_node()
    node.visits = 5
    node.wins = 2
    # total_simulation=1 yields log(1)=0, exploration term zero
    expected = node.wins / node.visits
    assert pytest.approx(node.uct_value(total_simulation=1), rel=1e-6) == expected


def test_select_child_prefers_unvisited():
    parent = create_node()
    # Create two child nodes
    child1 = MCTSNode(state=board, player=PLAYER2, parent=parent)
    child2 = MCTSNode(state=board, player=PLAYER2, parent=parent)
    # Simulate child1 visited
    child1.visits = 1
    child1.wins = 1
    # child2 unvisited (visits=0)
    # Attach children
    parent.children = [child1, child2]
    parent.visits = 1  # total_sim passed to uct_value
    selected = parent.select_child()
    # child2 has visits=0 => uct_value = inf, should be selected
    assert selected is child2, "select_child should prefer unvisited child with infinite UCT value"


def test_select_child_prefers_higher_uct():
    parent = create_node()
    # Create two visited child nodes
    child1 = MCTSNode(state=board, player=PLAYER2, parent=parent)
    child2 = MCTSNode(state=board, player=PLAYER2, parent=parent)
    # Set visits and wins
    child1.visits = 10
    child1.wins = 9  # high win rate
    child2.visits = 10
    child2.wins = 1  # low win rate
    parent.children = [child1, child2]
    parent.visits = 20  # total simulations
    # Compute UCT values manually
    uct1 = child1.wins / child1.visits + EXPLORATION_COEF * np.sqrt(np.log(parent.visits) / child1.visits)
    uct2 = child2.wins / child2.visits + EXPLORATION_COEF * np.sqrt(np.log(parent.visits) / child2.visits)
    assert uct1 > uct2
    selected = parent.select_child()
    assert selected is child1, "select_child should choose child with higher UCT value"


def test_expand_creates_child_and_updates_untried_moves():
    node = create_node()
    # Override untried_moves to a known small list
    node.untried_moves = [2, 5]
    original_moves = node.untried_moves.copy()
    # Expand once
    child = node.expand()

    # The popped move is the last element of original_moves
    popped_move = original_moves[-1]
    # Parent's untried_moves should have removed the popped move
    assert node.untried_moves == original_moves[:-1]

    # The child should be appended to parent's children
    assert child in node.children
    assert len(node.children) == 1

    # Child's parent should be the node
    assert child.parent is node

    # Child's player should be the opposite
    expected_player = PLAYER2 if node.player == PLAYER1 else PLAYER1
    assert child.player == expected_player

    # Child's state should differ from the original by exactly one move
    diff = child.state - board
    # All differences should be either 0 or the player's value
    move_positions = np.argwhere(diff != 0)
    assert move_positions.shape[0] == 1, "Exactly one new piece should be placed"
    r, c = move_positions[0]
    assert c == popped_move, "The new piece should be in the popped column"
    assert child.state[r, c] == node.player, "The placed piece should belong to the node's player"


def test_expand_until_exhausted_untried_moves():
    node = create_node()
    # Limit to two moves
    node.untried_moves = [0, 1]
    # Expand twice
    first = node.expand()
    second = node.expand()
    assert len(node.children) == 2
    assert node.untried_moves == []
    # Further expand should error due to no untried moves
    with pytest.raises(IndexError):
        node.expand()


def test_update_increments_visits_always():
    node = create_node()
    assert node.visits == 0
    node.update(result_player=None)
    assert node.visits == 1
    node.update(result_player=PLAYER2)
    assert node.visits == 2
    node.update(result_player=PLAYER1)
    assert node.visits == 3


def test_update_increments_wins_only_on_match():
    node = create_node()
    assert node.wins == 0
    # Non-matching player yields no win
    node.update(result_player=PLAYER2)
    assert node.wins == 0
    # Matching player yields win
    node.update(result_player=PLAYER1)
    assert node.wins == 1
    # None result yields no win
    node.update(result_player=None)
    assert node.wins == 1
    # Another matching
    node.update(result_player=PLAYER1)
    assert node.wins == 2


def test_update_multiple_nodes_independent():
    node1 = MCTSNode(state=create_empty_board(), player=PLAYER1)
    node2 = MCTSNode(state=create_empty_board(), player=PLAYER2)
    node1.update(result_player=PLAYER1)

    node2.update(result_player=PLAYER1)  # result doesn't match node2's player
    assert node1.wins == 1, "node1 should record its own win"
    assert node2.wins == 0, "node2 should not record win when result doesn't match its player"
    assert node1.visits == 1
    assert node2.visits == 1
