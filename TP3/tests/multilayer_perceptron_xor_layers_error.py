import plotly.graph_objects as go
from numpy import copy, mean, std, concatenate

from algorithms.perceptrons import LinearPerceptron, MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values

if __name__ == "__main__":
    print('--- Welcome to multilayer perceptron xor layers error plot ---')

    print('parsing config file...')
    config = get_config("../config.yaml")

    training_values = config.training_values

    training_values.input = '../training_values/ej1-xor-input.txt'
    training_values.output = '../training_values/ej1-xor-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    scatters = []
    for layers in [[3], [3, 3, 3], [30]]:
        errors = []
        for i in range(1):
            perceptron = MultiLayerPerceptron(input_values, layers, output_values, config.perceptron.settings)
            print(f'Predicting results {i + 1} - layers {layers}')
            results = perceptron.train()
            results.print()
            errors.append(perceptron.plot['e'])
        mean_errors = mean(errors, axis=0)
        print(f'Min error: {mean_errors.min()}')
        scatters.append(
            go.Scatter(
                y=mean_errors,
                name=f'{layers}'
            ),
        )

    fig = go.Figure(
        scatters
        ,
        {
            'title': f'Error per layers',
        }
    )
    fig.show()
