from collections import deque
from typing import Optional, List, Deque

from utils.board import State, OBJECTIVE_STATE


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state = state
        self.parent = parent

    def is_objective(self) -> bool:
        return self.state == OBJECTIVE_STATE

    def get_next_states(self) -> List[State]:
        return list(map(lambda pos: self.state.move_space(pos), self.state.get_possible_positions()))

    def get_child_nodes(self) -> List['Node']:
        return list(map(lambda state: Node(state, self), self.get_next_states()))

    def has_parent(self) -> bool:
        return self.parent is not None

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return self.state.__repr__()


def print_tree(deepest_node: Node) -> None:
    tree: Deque[Node] = deque([deepest_node])

    while deepest_node.has_parent():
        deepest_node = deepest_node.parent
        tree.append(deepest_node)

    while not tree:
        print(tree.pop())
