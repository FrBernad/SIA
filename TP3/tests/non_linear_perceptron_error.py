import plotly.graph_objects as go
from numpy import copy, mean, std, concatenate

from algorithms.perceptrons import LinearPerceptron, NonLinearPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values

if __name__ == "__main__":
    print('--- Welcome to non linear perceptron error plot ---')

    print('parsing config file...')
    config = get_config("../config.yaml")

    training_values = config.training_values

    training_values.input = '../training_values/ej2-linear-input.txt'
    training_values.output = '../training_values/ej2-linear-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    errors = []

    for i in range(20):
        perceptron = NonLinearPerceptron(input_values, copy(output_values), config.perceptron.settings)
        print(f'Predicting results {i + 1}')
        results = perceptron.train()
        errors.append(perceptron.plot['e_denormalized'])

    mean_errors = mean(errors, axis=0)
    print(f'Min error: {mean_errors.min()}')
    std_errors = std(errors, axis=0)
    fig = go.Figure(
        [
            go.Scatter(
                y=mean_errors,
            ),
            go.Scatter(
                x=list(range(len(mean_errors))) + list(range(len(mean_errors)))[::-1],
                y=concatenate((mean_errors + std_errors, (mean_errors - std_errors)[::-1])),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        ]
        ,
        {
            'title': f'Error Denormalized',
        }
    )
    fig.show()
