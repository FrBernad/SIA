import sys

import yaml

from algorithms.stats import Stats
from utils.board import State
from utils.node import plot_graph
from config import Config

CONFIG_FILE = 'config.yaml'


def _get_config(config_file: str) -> 'Config':
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        return Config(config.get("algorithm"), config.get("limit"), config.get("heuristic"))


def main(config_file: str):
    init_state = State.generate()

    config = _get_config(config_file)

    stats = Stats()

    tree = config.algorithm(init_state, stats, config)

    for n in tree:
        print(n)

    print(stats)

    plot_graph(tree)


# Run as python3 slider_puzzle.py [config_file_path]
if __name__ == '__main__':
    argv = sys.argv

    config_file = CONFIG_FILE
    if len(argv) > 1:
        config_file = argv[1]

    try:
        main(config_file)
    except OSError:
        print("Error opening config file.")
    except yaml.YAMLError:
        print("Error parsing config file.")
    except Exception as e:
        print(e)
