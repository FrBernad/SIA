from typing import Optional

from utils.board import State, OBJECTIVE_STATE, get_space_position


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state = state
        self.parent = parent

    def is_objective(self) -> bool:
        return self.state == OBJECTIVE_STATE

    def get_next_states(self):
        space_pos = get_space_position(self.state)


