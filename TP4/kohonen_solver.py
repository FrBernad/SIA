import sys

from algorithms.kohonen import Kohonen
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_input_csv


def kohonen_solver(config_file: str):
    print('--- WELCOME TO THE KOHONEN NETWORK SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    if config.input_file is None:
        config.input_file = 'data/europe.csv'

    print(f'\tparsing input file: {config.input_file}')
    input_values = parse_input_csv(config.input_file)

    print(f'\tGenerating Kohonen Network...')
    kohonen_network = Kohonen(input_values, config.kohonen)

    print(f'\tTraining network...')
    kohonen_network.train()

    print(f'\tFinished!')


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'KOHONEN')

    config_file = arguments['config_file']

    try:
        kohonen_solver(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
