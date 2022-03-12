from collections import deque, Iterable
from typing import Set, Deque

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.node import Node


def bfs(init_state: State, stats: Stats, config: Config) -> Iterable[Node]:
    visited: Set[Node] = set()
    border: Deque[Node] = deque()

    border.append(Node(init_state, None))

    while border:
        current_node = border.popleft()

        if current_node not in visited:
            stats.explored_nodes_count += 1
            visited.add(current_node)

        if current_node.is_objective():
            stats.objective_distance = current_node.depth
            return current_node.get_tree()

        not_visited_nodes = list(filter(lambda node: node not in visited, current_node.get_child_nodes()))

        for n in not_visited_nodes:
            border.append(n)
