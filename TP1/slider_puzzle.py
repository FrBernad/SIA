import yaml

from algorithms.stats import Stats
from utils.board import State
from utils.node import plot_graph
from config import Config

CONFIG_FILE = 'config.yaml'


def _get_config() -> 'Config':
    with open(CONFIG_FILE) as config_file:
        config = yaml.safe_load(config_file)["config"]
        return Config(config.get("algorithm"), config.get("limit"), config.get("heuristic"))


def main():
    init_state = State.generate()

    config = _get_config()

    stats = Stats()

    tree = config.algorithm(init_state, stats, config)

    for n in tree:
        print(n)

    print(stats)

    plot_graph(tree)

if __name__ == '__main__':
    # try:
    main()
    # except OSError:
    #     print("Error opening config.yaml file.")
    # except yaml.YAMLError:
    #     print("Error parsing config.yaml file.")
    # except Exception as e:
    #     print(e)
