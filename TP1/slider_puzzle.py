import sys

import yaml

from algorithms.stats import Stats
from config import Config
from utils.board import State
from utils.results import generate_solution_yaml

CONFIG_FILE = 'config.yaml'
OUTPUT_FILE = 'solution.yaml'


def _get_config(config_file: str) -> 'Config':
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        return Config(config.get("algorithm"), config.get("limit"), config.get("heuristic"))


def _generate_solution_file(solution, sol_file):
    with open(sol_file, 'w') as f:
        yaml.dump(solution, f, sort_keys=False, default_flow_style=False)


def main(config_file: str, output_file: str):
    config = _get_config(config_file)

    stats = Stats()

    tree = config.algorithm(State.generate(), stats, config)

    _generate_solution_file(generate_solution_yaml(tree, stats, config), output_file)


# Run as pipenv run slider_puzzle.py [config_file_path] [output_file_path]
if __name__ == '__main__':
    argv = sys.argv

    config_file = CONFIG_FILE
    output_file = OUTPUT_FILE
    if len(argv) > 1:
        config_file = argv[1]

    if len(argv) > 2:
        output_file = argv[2]

    try:
        main(config_file, output_file)
    except OSError:
        print("Error opening config file.")
    except yaml.YAMLError:
        print("Error writing or reading from yaml file.")
    except Exception as e:
        print(e)
