import sys

import plotly.graph_objects as go
from numpy.typing import NDArray

from algorithms.perceptrons import MultiLayerPerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values, parse_nums


def ej3(config_path: str):
    print('--- WELCOME TO THE EJ3 PROBLEM SOLVER ---')

    print('parsing config file...')
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej1-xor-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej1-xor-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values: NDArray
    if 'xor' in training_values.input:
        input_values = parse_training_values(training_values.input)
    else:
        input_values = parse_nums(training_values.input)

    print(f'parsing output file: {training_values.input}')
    output_values = parse_output_values(training_values.output)

    print(f'Generating multilayer perceptron...')
    perceptron = MultiLayerPerceptron(input_values, [6, 6], output_values, config.perceptron.settings)

    print(f'Predicting results...')
    results = perceptron.train()
    print(f'Finished!')
    results.print()

    if config.plot:
        fig = go.Figure(
            go.Scatter(
                y=perceptron.plot['e'],
            )
            ,
            {
                'title': f'Error',
            }
        )
        fig.show()


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])

    config_file = arguments['config_file']

    try:
        ej3(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
