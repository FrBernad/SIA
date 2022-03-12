from collections import deque
from typing import Optional, List, Deque, Iterable, Callable

from utils.board import State, OBJECTIVE_STATE


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state = state
        self.parent = parent
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def is_objective(self) -> bool:
        return self.state == OBJECTIVE_STATE

    def get_next_states(self) -> List[State]:
        return list(map(lambda pos: self.state.move_space(pos), self.state.get_possible_positions()))

    def get_child_nodes(self) -> List['Node']:
        return list(map(lambda state: Node(state, self), self.get_next_states()))

    def get_tree(self) -> Iterable['Node']:
        deepest_node = self
        tree: Deque[Node] = deque([deepest_node])

        while deepest_node.has_parent():
            deepest_node = deepest_node.parent
            tree.append(deepest_node)

        tree.reverse()

        return tree

    def has_parent(self) -> bool:
        return self.parent is not None

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return self.state.__repr__()


class HeuristicNode(Node):
    def __init__(self, state: State, parent: Optional['HeuristicNode'], heuristic: Callable[[State], int]):
        super().__init__(state, parent)
        self.heuristic = heuristic
        self.heuristic_value = self.heuristic(self.state)

    def get_child_nodes(self) -> List['HeuristicNode']:
        return list(map(lambda state: HeuristicNode(state, self, self.heuristic), self.get_next_states()))

    def is_objective(self) -> bool:
        return self.heuristic_value == 0

    def __eq__(self, other):
        return isinstance(other, HeuristicNode) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __lt__(self, other):
        return self.heuristic_value < other.heuristic_value


def plot_graph(tree: Iterable[Node]) -> None:
    pass
    # graph = nx.Graph()
    #
    # graph.add_nodes_from(map(lambda n: n.__str__(), tree))
    #
    # for node in tree:
    #     if node.has_parent():
    #         graph.add_edge(node.parent, node)
    #
    # plt.plot()
    # nx.draw(graph, with_labels=True, font_weight='bold')
    #
    # plt.show()
    # print(graph)
