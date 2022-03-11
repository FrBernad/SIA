from typing import Optional, Callable, Iterable

from algorithms.stats import Stats
from utils.board import State
from utils.node import Node


class Config:
    def __init__(self, algorithm: Callable[[State, Stats, 'Config'], Iterable[Node]], limit: Optional[int] = None):
        self.algorithm = algorithm
        self.limit = limit
