from random import random

from numpy import array

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_output_values, parse_nums


def _mutate_input_values(input_values):
    for i in range(len(input_values)):
        for j in range(len(input_values[i]) - 1):
            if random() < 0.02:
                input_values[i][j] = 1 if input_values[i][j] == 0 else 0


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
    perceptron = MultiLayerPerceptron(input_values, [31], output_values, config.perceptron.settings)

    print(f'Predicting results...')
    results = perceptron.train()
    print(f'Finished!')
    results.print()

    _mutate_input_values(input_values)
    for i in range(len(input_values)):
        print(f"Input: {i}")
        predicted_output = perceptron.predict(input_values[i])
        print(f'Expected: {output_values[i]}')
        print(f'Predicted: {predicted_output}')
        print(f'Error: {perceptron.calculate_error(array([input_values[i]]), array([output_values[i]]))}\n')
