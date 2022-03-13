import time
from collections import deque
from typing import Iterable, Set, Deque

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.node import HeuristicNode, Node


def hls(init_state: State, stats: Stats, config: Config) -> Iterable[Node]:
    visited: Set[Node] = set()
    border: Deque[HeuristicNode] = deque()

    stats.start_time = time.process_time()

    border.append(HeuristicNode(init_state, None, config.heuristic))

    while border:
        current_node = border.pop()

        if current_node not in visited:
            stats.explored_nodes_count += 1
            visited.add(current_node)

        if current_node.is_objective():
            stats.end_time = time.process_time()
            stats.objective_distance = current_node.depth
            stats.objective_cost = current_node.depth
            stats.border_nodes_count = len(border)
            stats.objective_found = True
            return current_node.get_tree()

        not_visited_nodes = list(filter(lambda node: node not in visited, current_node.get_child_nodes()))

        not_visited_nodes.sort(reverse=True)

        border.extend(not_visited_nodes)

    stats.end_time = time.process_time()
    stats.objective_found = False
    return []
