from typing import Optional

from algorithms.exceptions import InvalidAlgorithmException, InvalidConfigLimitException, MissingHeuristicException


class Config:
    def __init__(self, algorithm: str, limit: Optional[str] = None, heuristic: Optional[str] = None):

        self.algorithm = ALGORITHMS.get(algorithm)
        self.algorithm_str = algorithm
        if not self.algorithm:
            raise InvalidAlgorithmException()

        if limit is not None:
            if need_limit(algorithm):
                try:
                    limit = int(limit)
                    if limit <= 0:
                        raise InvalidConfigLimitException()
                except Exception as e:
                    raise InvalidConfigLimitException()

        self.limit = limit

        if heuristic is not None:
            self.heuristic = HEURISTICS.get(heuristic)
            if is_informed(algorithm) and not self.heuristic:
                raise MissingHeuristicException()

        self.heuristic_str = heuristic


def is_informed(algorithm: str) -> bool:
    return algorithm in ["a_star", "hls", "hgs"]


def need_limit(algorithm: str) -> bool:
    return algorithm == "iddfs"


from algorithms.a_star import a_star
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.hgs import hgs
from algorithms.hls import hls
from algorithms.iddfs import iddfs
from utils.heuristics import manhattan_distance, hamming_distance, overestimated_manhattan_distance

HEURISTICS = {
    "manhattan": manhattan_distance,
    "hamming": hamming_distance,
    "overestimated": overestimated_manhattan_distance
}

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
    "hgs": hgs,
    "hls": hls,
    "a_star": a_star,
}
