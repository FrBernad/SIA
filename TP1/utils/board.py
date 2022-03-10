from enum import Enum
from random import choice
from typing import List, NamedTuple

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

Position = NamedTuple('Position', [('i', int), ('j', int)])


class Movements(Enum):
    UP = Position(0, -1)
    DOWN = Position(0, 1)
    LEFT = Position(-1, 0)
    RIGHT = Position(1, 0)

    def add_pos(self, pos: Position) -> Position:
        return Position(pos.i + self.value.i, pos.j + self.value.j)


def get_possible_positions(state: 'State') -> List[Position]:
    space_pos = state.get_space_position()
    possible_pos = []

    if (p := Movements.add_pos(Movements.UP, space_pos)).j >= 0:
        possible_pos.append(p)

    if (p := Movements.add_pos(Movements.DOWN, space_pos)).j < BOARD_SIZE:
        possible_pos.append(p)

    if (p := Movements.add_pos(Movements.LEFT, space_pos)).i >= 0:
        possible_pos.append(p)

    if (p := Movements.add_pos(Movements.RIGHT, space_pos)).i < BOARD_SIZE:
        possible_pos.append(p)

    return possible_pos


class State:

    def __init__(self, state: NDArray[int]):
        self.state = state
        self.space_pos = self.get_space_position()

    @staticmethod
    def move_space(state: 'State', new_pos: Position) -> 'State':
        space_pos = state.get_space_position()
        state = ndarray.copy(state.state)
        state[space_pos.j][space_pos.i], state[new_pos.j][new_pos.i] = \
            state[new_pos.j][new_pos.i], state[space_pos.j][space_pos.i]
        return State(state)

    @staticmethod
    def generate(shuffle_count: int = 100) -> 'State':
        rand_state = OBJECTIVE_STATE

        for _ in range(shuffle_count):
            possible_pos = get_possible_positions(rand_state)
            rand_pos = choice(possible_pos)
            rand_state = State.move_space(rand_state, rand_pos)

        return rand_state

    def get_space_position(self) -> Position:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.state[i][j] == 0:
                    return Position(j, i)

    def __eq__(self, other):
        return isinstance(other, State) and self.state == other.state

    def __str__(self):
        return f'{self.state[0]}\n{self.state[1]}\n{self.state[2]}\n'


BOARD_SIZE = 3

OBJECTIVE_STATE: State = State(
    np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ],
        int
    )
)
