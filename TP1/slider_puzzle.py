from typing import Callable, Iterable, Dict

import yaml

from algorithms.a_star import a_star
from algorithms.bfs import bfs
from algorithms.config import Config
from algorithms.dfs import dfs
from algorithms.hgs import hgs
from algorithms.iddfs import iddfs
from algorithms.stats import Stats
from utils.board import State
from utils.node import print_tree, Node

CONFIG_FILE = 'config.yaml'

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
    "hgs": hgs,
    "hls": hgs,
    "a*": a_star,
}


def _get_config() -> Config:
    with open(CONFIG_FILE) as config_file:
        config = yaml.safe_load(config_file)["config"]
        return Config(ALGORITHMS[config["algorithm"]], config["limit"])


def main():
    init_state = State.generate()

    config = _get_config()

    stats = Stats()

    tree = config.algorithm(init_state, stats, config)

    print_tree(tree)


if __name__ == '__main__':
    main()
