from typing import Optional

from algorithms.exceptions import InvalidAlgorithmException, InvalidConfigLimitException, MissingHeuristicException


class Config:
    def __init__(self, algorithm: str, limit: Optional[str], heuristic: Optional[str]):

        self.algorithm = ALGORITHMS.get(algorithm)
        if not self.algorithm:
            raise InvalidAlgorithmException()

        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise InvalidConfigLimitException()
            self.limit = limit

        if heuristic is not None:
            self.heuristic = HEURISTICS.get(heuristic)
            if is_informed(algorithm) and not self.heuristic:
                raise MissingHeuristicException()


def is_informed(algorithm: str) -> bool:
    return algorithm in ["a*", "hls", "hgs"]


from algorithms.a_star import a_star
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.hgs import hgs
from algorithms.hls import hls
from utils.heuristics import manhattan_distance, hamming_distance, overestimated_manhattan_distance

HEURISTICS = {
    "manhattan": manhattan_distance,
    "hamming": hamming_distance,
    "overestimated": overestimated_manhattan_distance
}

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": dfs,
    "hgs": hgs,
    "hls": hls,
    "a*": a_star,
}
