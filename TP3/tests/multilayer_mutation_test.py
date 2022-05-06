from random import random

from numpy import array, reshape, flip, copy
from numpy.random import randint, choice
from plotly.subplots import make_subplots

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_output_values, parse_nums


def _mutate_input_value(input_value, mutated_amount):
    mutated_input = copy(input_value)
    indexes = choice(range(len(input_value) - 1), size=mutated_amount, replace=False)
    for i in indexes:
        mutated_input[i] = 1 if mutated_input[i] == 0 else 0

    return mutated_input


if __name__ == "__main__":
    print('Welcome to the multilayer mutation perceptron test')
    config = get_config('../config.yaml')

    training_values = config.training_values
    training_values.input = '../training_values/ej3-2-input.txt'
    training_values.output = '../training_values/ej3-3-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_nums(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    print(f'Generating multilayer perceptron...')
    perceptron = MultiLayerPerceptron(input_values, [36], output_values, config.perceptron.settings)

    print(f'Training perceptron...')
    results = perceptron.train()
    print(f'Finished!')
    results.print()

    mutations = [0, 5, 15, 35]
    for i in range(len(input_values)):
        bitmaps = []
        prediction_errors = []
        for mutated_amount in mutations:
            mutated_value = _mutate_input_value(input_values[i], mutated_amount)
            bitmaps.append(mutated_value)

            print(f"Input: {i} - {mutated_amount} mutations")

            predicted_output = perceptron.predict(mutated_value)

            print(f'Expected: {output_values[i]}')
            print(f'Predicted: {predicted_output}')

            error = perceptron.calculate_error(array([mutated_value]), array([output_values[i]]))
            prediction_errors.append(error)
            print(f'Error: {error}\n')

        fig = make_subplots(
            rows=2, cols=5,
            subplot_titles=tuple(map(lambda m: f'{m} mutations', mutations)),
            specs=[[{}, {}, {}, {}, {}], [{'type': 'table', "colspan": 2}, None, None, None, None]],
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        for k in range(1, len(mutations) + 1):
            fig.add_heatmap(
                z=flip(reshape(bitmaps[k - 1][:-1], (7, 5)), axis=0),
                showscale=False,
                row=1, col=k, colorscale='Greys',
            )

        fig.add_table(
            columnwidth=[1, 1.5],
            header=dict(
                values=["Mutations", "Prediction Error"],
                align=['center', 'center']
            ),
            cells=dict(
                values=array([mutations, prediction_errors]),
                align=['center', 'center'],
                font_size=12,
                format=[None, ".6f"],
                height=30
            ),
            row=2, col=1
        )

        fig.show()
