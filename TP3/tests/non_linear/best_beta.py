import plotly.graph_objects as go
from numpy import copy, mean, std, concatenate

from algorithms.perceptrons import NonLinearPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values

if __name__ == "__main__":
    print('--- Welcome to non linear perceptron error plot ---')

    print('parsing config file...')
    config = get_config("../../config.yaml")

    training_values = config.training_values

    training_values.input = '../../training_values/ej2-input.txt'
    training_values.output = '../../training_values/ej2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    errors = []
    betas = [0.01, 0.05, 0.1, 0.5, 1, 1.5]
    for beta in betas:
        config.perceptron.settings.b = beta
        perceptron = NonLinearPerceptron(input_values, copy(output_values), config.perceptron.settings)
        print(f'Predicting results beta {beta}')
        results = perceptron.train()
        errors.append(perceptron.plot['e_denormalized'][-1])

    fig = go.Figure(
        [
            go.Scatter(
                y=errors,
                x=betas
            ),
        ]
        ,
        {
            'title': f'Error Denormalized per beta',
        }
    )
    fig.show()
