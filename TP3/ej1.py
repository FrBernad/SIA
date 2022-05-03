import sys
from itertools import groupby

import plotly.graph_objects as go
from numpy import arange, ones, array_equal

from algorithms.perceptrons import SimplePerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values


def ej1(config_path: str):
    print('--- WELCOME TO THE EJ1 PROBLEM SOLVER ---')

    print('parsing config file...')
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej1-and-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej1-and-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.input}')
    output_values = parse_output_values(training_values.output)

    print(f'Generating simple perceptron...')
    perceptron = SimplePerceptron(input_values, output_values, config.perceptron.settings)

    print(f'Predicting results...')
    results = perceptron.train()
    print(f'Finished!')
    results.print(remove_ws=True)

    # Generate dots
    if config.plot:
        dots = go.Scatter(
            x=perceptron.x[:, 0],
            y=perceptron.x[:, 1],
            mode="markers",
            marker=dict(
                size=15,
                color=
                ((perceptron.x[:, 0] == 1) & (perceptron.x[:, 1] == 1)).astype(int) if "and" in training_values.input
                else ((perceptron.x[:, 0] == 1) ^ (perceptron.x[:, 1] == 1)).astype(int),
                colorscale=[[0, 'red'], [1, 'black']]
            )
        )

        # Generate hyperplane frames
        x_vals = arange(-1, 2)

        w_values = perceptron.plot['w']
        not_repeated_w = []
        for i in range(len(w_values)):
            if i == len(w_values) - 1 or not array_equal(w_values[i], w_values[i + 1]):
                not_repeated_w.append(w_values[i])

        frames = list(map(lambda w: go.Frame(
            data=[
                dots,
                go.Scatter(
                    x=x_vals,
                    y=(-w[0] / w[1]) * x_vals - w[2] / w[1],
                )
            ]
        ), not_repeated_w))

        figures = [dots, go.Scatter(
            x=x_vals,
            y=(-not_repeated_w[0][0] / not_repeated_w[0][1]) * x_vals - not_repeated_w[0][2] / not_repeated_w[0][1],
        )]

        fig = go.Figure(
            data=figures,
            layout=go.Layout(
                xaxis=dict(range=[-4, 4], autorange=False),
                yaxis=dict(range=[-4, 4], autorange=False),
                title=f'{"And" if "and" in training_values.input else "Xor"}',
                showlegend=False,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None,
                                  {
                                      "frame": {
                                          "duration": 300, "redraw": False
                                      },
                                      "fromcurrent": True,
                                      "transition": {
                                          "duration": 300,
                                          "easing": "quadratic-in-out"
                                      }

                                  }
                                  ]
                        )
                    ]
                )]
            ),
            frames=frames
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
