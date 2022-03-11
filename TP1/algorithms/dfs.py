from collections import deque
from typing import Set, Deque, Iterable

from algorithms.config import Config
from algorithms.stats import Stats
from utils.board import State
from utils.node import Node


def dfs(init_state: State, stats: Stats, config: Config) -> Iterable[Node]:
    visited: Set[Node] = set()
    border: Deque[Node] = deque()

    border.append(Node(init_state, None))

    while border:
        current_node = border.pop()

        if current_node not in visited:
            visited.add(current_node)

        if current_node.is_objective():
            return current_node.get_tree()

        not_visited_nodes = list(filter(lambda node: node not in visited, current_node.get_child_nodes()))

        for n in not_visited_nodes:
            border.append(n)
