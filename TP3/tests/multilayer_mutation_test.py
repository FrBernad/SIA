from itertools import chain
from random import random
import plotly.graph_objects as go

from numpy import array, split, copy, mean, std
from numpy.random import randint

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_output_values, parse_nums


def _mutate_input_values(input_values):
    mutated_numbers = []
    for i in range(len(input_values)):
        for j in range(len(input_values[i]) - 1):
            if random() < 0.02:
                mutated_numbers.append(i)
                input_values[i][j] = 1 if input_values[i][j] == 0 else 0

    return mutated_numbers


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

    print(f'Mutated values: {_mutate_input_values(input_values)}')

    for i in range(len(input_values)):
        print(f"Input: {i}")
        predicted_output = perceptron.predict(input_values[i])
        print(f'Expected: {output_values[i]}')
        print(f'Predicted: {predicted_output}')
        print(f'Error: {perceptron.calculate_error(array([input_values[i]]), array([output_values[i]]))}\n')

    # mse_percentages_training_errors = []
    # mse_percentages_training_std = []
    # mse_percentages_prediction_errors = []
    # mse_percentages_prediction_std = []
    # percentages = []
    #
    # for percentage in range(1, 10):
    #     print(f"\n--- Cross Validation Training {percentage * 10}% - Testing {(10 - percentage) * 10}% ---\n")
    #     percentages.append(f'{percentage * 10}%')
    #
    #     mse_training_errors = []
    #     mse_prediction_errors = []
    #
    #     for rd in range(1):
    #         print(f"Round {rd + 1}")
    #         # SHUFFLE
    #         for i in range(200):
    #             rand_index_1 = randint(0, len(input_values))
    #             rand_index_2 = randint(0, len(input_values))
    #             aux = copy(input_values[rand_index_1])
    #             input_values[rand_index_1] = copy(input_values[rand_index_2])
    #             input_values[rand_index_2] = aux
    #             aux = copy(output_values[rand_index_1])
    #             output_values[rand_index_1] = copy(output_values[rand_index_2])
    #             output_values[rand_index_2] = aux
    #
    #         # SPLIT VALUES
    #         training_input_parts = split(input_values, 10)
    #         training_output_parts = split(output_values, 10)
    #
    #         i = 0
    #         print("Calculating")
    #         while i + percentage <= len(training_input_parts):
    #             print(f"{i}")
    #             training_input = []
    #             training_output = []
    #             for k in range(i, i + percentage):
    #                 training_input.extend(training_input_parts[k])
    #                 training_output.extend(training_output_parts[k])
    #
    #             training_input = array(training_input)
    #             training_output = array(training_output)
    #
    #             testing_input = []
    #             testing_output = []
    #             for k in chain(range(0, i), range(i + percentage, len(training_input_parts))):
    #                 testing_input.extend(training_input_parts[k])
    #                 testing_output.extend(training_output_parts[k])
    #             testing_input = array(testing_input)
    #             testing_output = array(testing_output)
    #
    #             perceptron = MultiLayerPerceptron(training_input, [6, 6], training_output, config.perceptron.settings)
    #             results = perceptron.train()
    #
    #             testing_error = perceptron.plot['e'][-1]
    #             mse_training_errors.append(testing_error)
    #             print(f"Training values MSE: {testing_error}")
    #
    #             print("Predicted - Expected outputs")
    #             for j in range(len(testing_input)):
    #                 print(f'{perceptron.predict(testing_input[j])}')
    #                 print(f'{testing_output[j]}')
    #
    #             prediction_error = perceptron.calculate_error(testing_input, testing_output)
    #             mse_prediction_errors.append(prediction_error)
    #             print(f"Testing values MSE: {prediction_error}\n")
    #
    #             i += 1
    #
    #         print("")
    #
    #     mse_percentages_training_errors.append(mean(mse_training_errors))
    #     mse_percentages_training_std.append(std(mse_training_errors))
    #
    #     mse_percentages_prediction_errors.append(mean(mse_prediction_errors))
    #     mse_percentages_prediction_std.append(std(mse_prediction_errors))
    #
    # fig = go.Figure(
    #     [
    #         go.Scatter(
    #             x=percentages,
    #             y=mse_percentages_training_errors,
    #             name=f'Training MSE',
    #             error_y=dict(
    #                 type='data',
    #                 array=mse_percentages_training_std,
    #                 visible=True
    #             )
    #         ),
    #         go.Scatter(
    #             x=percentages,
    #             y=mse_percentages_prediction_errors,
    #             name=f'Testing MSE',
    #             error_y=dict(
    #                 type='data',
    #                 array=mse_percentages_prediction_std,
    #                 visible=True
    #             )
    #         )
    #     ]
    #     ,
    #     {
    #         'title': f'Training and testing MSE per training selection size percentage',
    #         'xaxis_title': "Percentage",
    #         'yaxis_title': "MSE",
    #
    #     }
    # )
    # fig.show()
