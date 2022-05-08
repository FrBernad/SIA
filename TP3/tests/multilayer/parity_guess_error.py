import math

import plotly.graph_objects as go
from numpy import mean, std, concatenate, array
from sklearn.model_selection import train_test_split

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_output_values, parse_nums

if __name__ == "__main__":
    print('--- Welcome to multilayer perceptron parity guess error ---')

    print('parsing config file...')
    config = get_config("../../config.yaml")

    training_values = config.training_values

    training_values.input = '../../training_values/ej3-2-input.txt'
    training_values.output = '../../training_values/ej3-2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_nums(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    testing_guess = []
    testing_errors = []
    for rd in range(100):
        print(f"Round {rd + 1}")

        # SPLIT VALUES
        train_input, test_input, train_output, test_output = train_test_split(input_values, output_values,
                                                                              train_size=0.9)
        perceptron = MultiLayerPerceptron(train_input, [36], train_output, config.perceptron.settings)
        results = perceptron.train()

        print("Predicted - Expected outputs")

        predicted_output = perceptron.predict(test_input[0])
        print(f'{" ".join(predicted_output[0].__str__().split())}')
        print(f'{" ".join(test_output[0][0].__str__().split())}')

        hit = 1 if math.isclose(test_output[0][0], predicted_output[0], abs_tol=0.7) else 0
        testing_guess.append(hit)

        testing_errors.append((test_output[0][0] - predicted_output[0]))

    print(mean(array(testing_guess)))
    print(mean(array(testing_errors)))
    print(std(array(testing_errors)))
