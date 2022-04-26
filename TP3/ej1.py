import sys

from numpy import arange
import plotly.graph_objects as go

from algorithms.perceptrons import NonLinearPerceptron, SimplePerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values


def ej1(config_path: str):
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej1-and-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej1-and-output.txt'

    input_values = parse_training_values(training_values.input)
    output_values = parse_output_values(training_values.output)

    perceptron = SimplePerceptron(input_values, output_values, config.perceptron.settings)

    perceptron.train()

    if config.plot:
        figures = [go.Scatter(
            x=perceptron.x[:, 1],
            y=perceptron.x[:, 2],
            mode="markers",
            marker=dict(
                size=15,
                color=
                ((perceptron.x[:, 1] == 1) & (perceptron.x[:, 2] == 1)).astype(int) if "and" in training_values.input
                else ((perceptron.x[:, 1] == 1) ^ (perceptron.x[:, 2] == 1)).astype(int),
                colorscale=[[0, 'red'], [1, 'black']]
            )
        )]

        # Generate hyperplane
        x_vals = arange(-1, 2)
        w = perceptron.w
        figures.append(
            go.Scatter(
                x=x_vals,
                y=(-w[1] / w[2]) * x_vals - w[0] / w[2],
            )
        )

        fig = go.Figure(
            figures
            ,
            {
                'title': f'{"And" if "and" in training_values.input else "Xor"}',
                'showlegend': False
            }
        )
        fig.show()

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
        ej1(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
