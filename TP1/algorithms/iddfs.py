import time
from collections import deque
from typing import Set, Deque, Optional, Iterable, Dict

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.node import Node

DEFAULT_LIMIT = 20


def iddfs(init_state: State, stats: Stats, config: Config) -> Iterable[Node]:
    limit = config.limit if config.limit else DEFAULT_LIMIT
    border: Deque[Node] = deque()

    root_node = Node(init_state, None)

    limit_nodes: Deque[Node] = deque()
    limit_nodes.append(root_node)

    visited_nodes_depth: Dict[Node, int] = dict()
    visited_nodes_depth[root_node] = 0

    stats.start_time = time.process_time()

    while limit_nodes:
        limit_node = limit_nodes.pop()

        max_depth = limit_node.depth + limit

        border.append(limit_node)

        while border:

            current_node = border.pop()

            if current_node.depth >= max_depth:
                limit_nodes.append(current_node)
                continue

            visited_nodes_depth[current_node] = current_node.depth

            if current_node.is_objective():
                stats.end_time = time.process_time()
                stats.objective_distance = current_node.depth
                stats.objective_cost = current_node.depth
                stats.border_nodes_count = len(border)
                stats.objective_found = True
                return current_node.get_tree()

            not_visited_nodes = list(
                filter(lambda node: node not in visited_nodes_depth.keys() or node.depth < visited_nodes_depth[node],
                       current_node.get_child_nodes()))

            stats.explored_nodes_count += 1

            for n in not_visited_nodes:
                border.append(n)

    stats.end_time = time.process_time()
    stats.objective_found = False
    return []
