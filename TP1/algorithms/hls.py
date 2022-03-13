import heapq
import time
from typing import Iterable, Set, Deque

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.node import HeuristicNode, Node


def hls(init_state: State, stats: Stats, config: Config) -> Iterable[Node]:
    visited: Set[Node] = set()
    border = []

    stats.start_time = time.process_time()

    border.append(HeuristicNode(init_state, None, config.heuristic))
    heapq.heapify(border)

    while border:
        current_node = heapq.heappop(border)

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

        border = list(filter(lambda node: node not in visited, current_node.get_child_nodes()))

        heapq.heapify(border)

    stats.end_time = time.process_time()
    stats.objective_found = False
    return []
