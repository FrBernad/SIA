import plotly.graph_objects as go
from numpy import mean, array

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_output_values, parse_nums

if __name__ == "__main__":
    print('--- Welcome to multilayer perceptron parity learning rate errors plot ---')

    print('parsing config file...')
    config = get_config("../../config.yaml")

    training_values = config.training_values

    training_values.input = '../../training_values/ej3-2-input.txt'
    training_values.output = '../../training_values/ej3-2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_nums(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    scatters = []

    for lr in [0.005, 0.01, 0.1, 0.8, 'variable']:
        errors = []
        if lr == 'variable':
            config.perceptron.settings.learning_rate = 0.01
            config.perceptron.settings.variable = True
        else:
            config.perceptron.settings.learning_rate = lr
        for i in range(5):
            print(f'Predicting results {i + 1} - Learning rate {lr}')
            perceptron = MultiLayerPerceptron(input_values, [36], output_values, config.perceptron.settings)
            results = perceptron.train()
            errors.append(perceptron.plot['e'])

        mean_errors = mean(array(errors), axis=0)
        print(f'Min error: {mean_errors.min()}')
        scatters.append(
            go.Scatter(
                y=mean_errors,
                name=f"{lr}"
            ),
        )

    fig = go.Figure(
        scatters,
        {
            'title': f'Error per learning rate and beta',
        }
    )

    fig.show()
