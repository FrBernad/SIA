import sys

import plotly.graph_objects as go
from numpy import copy

from algorithms.perceptrons import NonLinearPerceptron, SimplePerceptron, LinearPerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values


def _plot_errors(perceptron: SimplePerceptron):
    if perceptron.normalize:
        fig = go.Figure(
            go.Scatter(
                y=perceptron.plot['e_normalized'],
            )
            ,
            {
                'title': f'Error Normalized',
            }
        )
        fig.show()

        fig = go.Figure(
            go.Scatter(
                y=perceptron.plot['e_denormalized'],
            )
            ,
            {
                'title': f'Error Denormalized',
            }
        )
        fig.show()

        # fig = go.Figure(
        #     [
        #         go.Scatter(
        #             y=perceptron.activation_function(perceptron.x[:, 0]),
        #             x=perceptron.x[:, 0],
        #             name=f'Logistic Function'
        #         ),
        #         go.Scatter(
        #             y=perceptron.y[:, 0],
        #             x=perceptron.x[:, 0],
        #             name=f'Real Function'
        #         )
        #     ]
        # )
        # fig.show()

    else:
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


def ej2(config_path: str):
    print('--- WELCOME TO THE EJ2 PROBLEM SOLVER ---')

    print('parsing config file...')
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej2-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    perceptron: SimplePerceptron

    if config.perceptron.type == "linear":
        print(f'Generating linear perceptron...')
        perceptron = LinearPerceptron(input_values, copy(output_values), config.perceptron.settings)
        print(f'Predicting results...')
        results = perceptron.train()
        print(f'Finished!')
        results.print(remove_ws=True)

    else:
        print(f'Generating non-linear perceptron...')
        perceptron = NonLinearPerceptron(input_values, copy(output_values), config.perceptron.settings)
        print(f'Predicting results...')
        results = perceptron.train()
        print(f'Finished!')
        results.print(remove_ws=True)

    if config.plot:
        _plot_errors(perceptron)


if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])

    config_file = arguments['config_file']

    try:
        ej2(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)

    # x = []
    # y = []
    # for i in range(200):
    #     x.append(i)
    #     y.append(2 * i)
    # print('\n'.join(numpy.array(x).__str__().split()))
    # print('\n'.join(numpy.array(y).__str__().split()))
