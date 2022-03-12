import heapq
import time
from typing import Iterable, Set, Deque

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.node import HeuristicNode, Node


def hls(init_state: State, stats: Stats, config: Config) -> Iterable[HeuristicNode]:
    visited: Set[Node] = set()
    border = []

    stats.start_time = time.process_time()

    border.append(HeuristicNode(init_state, None, config.heuristic))
    heapq.heapify(border)

    while border:
        current_node = border.pop(0)

        if current_node.is_objective():
            return current_node.get_tree()

        border.append(current_node.get_child_nodes())
