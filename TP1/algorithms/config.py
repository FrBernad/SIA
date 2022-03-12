from typing import Optional, Callable, Iterable

from algorithms.a_star import a_star
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.exceptions import InvalidConfigLimitException, InvalidAlgorithmException, InvalidHeuristicException, \
    MissingHeuristicException
from algorithms.hgs import hgs
from algorithms.iddfs import iddfs

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
    "hgs": hgs,
    "hls": hgs,
    "a*": a_star,
}

HEURISTICS = {
    "manhattan": manhattan,
    "humming": humming,
    "overestimated": overestimated
}


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
