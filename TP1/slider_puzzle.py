from algorithms.bpp import bpp
from utils.board import State
from utils.node import print_tree

if __name__ == '__main__':
    state = State.generate()

    objective_node = bpp(state)

    if objective_node:
        print('Found!')
        print_tree(objective_node)
    else:
        print('Not Found')
