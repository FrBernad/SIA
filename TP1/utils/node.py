from typing import Optional, List

from utils.board import State, OBJECTIVE_STATE


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state = state
        self.parent = parent

    def is_objective(self) -> bool:
        return self.state == OBJECTIVE_STATE

    def get_next_states(self) -> List[State]:
        return list(map(lambda pos: self.state.move_space(pos), self.state.get_possible_positions()))
