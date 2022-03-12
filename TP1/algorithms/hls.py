from collections import Iterable

from sortedcontainers import SortedList

from algorithms.config import Config
from algorithms.stats import Stats
from utils.board import State
from utils.node import HeuristicNode


def hls(init_state: State, stats: Stats, config: Config) -> Iterable[HeuristicNode]:
    border = SortedList([HeuristicNode(init_state, None, config.heuristic)])

    while border:
        current_node = border.pop(0)

        if current_node.is_objective():
            return current_node.get_tree()

        border.append(current_node.get_child_nodes())
