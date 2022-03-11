from collections import deque
from typing import Set, Deque, Optional

from utils.board import State
from utils.node import Node


def bpp(init_state: State) -> Optional[Node]:
    visited: Set[Node] = set()
    border: Deque[Node] = deque()

    border.append(Node(init_state, None))

    objective_node = None

    while border:
        current_node = border.pop()

        if current_node not in visited:
            visited.add(current_node)

        if current_node.is_objective():
            objective_node = current_node
            break

        not_visited_nodes = list(filter(lambda node: node not in visited, current_node.get_child_nodes()))

        for n in not_visited_nodes:
            border.append(n)

    return objective_node
