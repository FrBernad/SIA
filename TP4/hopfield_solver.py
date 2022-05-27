import sys

from numpy import array

from algorithms.hopfield import Hopfield
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_letters


def hopfield_solver(config_file: str):
    print('--- WELCOME TO THE HOPFIELD NETWORK SOLVER ---')

    print(f'\tparsing config file...')
    config = get_config(config_file)

    if config.input_file is None:
        config.input_file = 'data/font.txt'

    print(f'\tparsing input file: {config.input_file}')
    input_values = array(list(parse_letters(config.input_file).values()))

    print(f'\tGenerating Hopfield Network...')
    hopfield_network = Hopfield(config.hopfield, input_values[1:4])

    print(f'\tTraining network...')
    results = hopfield_network.train(input_values[1])

    print(f'\tFinished!')


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:], 'HOPFIELD')

    config_file = arguments['config_file']

    try:
        hopfield_solver(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
