import random
import sys

import numpy
import plotly.graph_objects as go

from algorithms.perceptrons import NonLinearPerceptron, SimplePerceptron, LinearPerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values


def ej2(config_path: str):
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej2-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej2-output.txt'

    input_values = parse_training_values(training_values.input)
    output_values = parse_output_values(training_values.output)

    perceptron: SimplePerceptron

    if config.perceptron.type == "linear":
        perceptron = LinearPerceptron(input_values, output_values, config.perceptron.settings)
        perceptron.train()

    else:
        perceptron = NonLinearPerceptron(input_values, output_values, config.perceptron.settings)
        perceptron.train()

    # FIXME: VER DE REESCALAR EL ERROR
    if config.plot:
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
