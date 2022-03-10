from typing import List


class Board:

    def __init__(self, board: List[List[int]]):
        self.board = board

    def __eq__(self, other):
        return isinstance(other, Board) and self.board == other.board


def board_generator() -> List[List[int]]:
    return [
        [5, 8, 3],
        [2, 1, 7],
        [9, 6, 10]
    ]
