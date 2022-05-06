import math
from itertools import chain

import plotly.graph_objects as go
from numpy import array, mean, std, copy, split, concatenate
from numpy.random import randint
from sklearn.model_selection import train_test_split

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_nums, parse_output_values

if __name__ == "__main__":
    print('Welcome to the multilayer perceptron parity cross validation test')
    config = get_config('../config.yaml')

    training_values = config.training_values
    training_values.input = '../training_values/ej3-2-input.txt'
    training_values.output = '../training_values/ej3-2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_nums(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    print("SHUFFLING VALUES")
    train_input, test_input, train_output, test_output = train_test_split(input_values, output_values,
                                                                          train_size=0.9)
    input_parts = split(concatenate((train_input, test_input)), 10)
    output_parts = split(concatenate((train_output, test_output)), 10)

    mse_training = []
    testing_guess = []

    for i in range(len(input_parts)):

        print(f"Round {i + 1}")

        training_input = []
        training_output = []
        testing_input = input_parts[i]
        testing_output = output_parts[i]

        for k in chain(range(0, i), range(i + 1, len(input_parts))):
            training_input.extend(input_parts[k])
            training_output.extend(output_parts[k])

        training_input = array(training_input)
        training_output = array(training_output)

        perceptron = MultiLayerPerceptron(training_input, [6, 6], training_output, config.perceptron.settings)
        results = perceptron.train()

        training_error = perceptron.plot['e'][-1]
        mse_training.append(training_error)
        print(f"Training values MSE: {training_error}")

        print("Predicted - Expected outputs")
        predicted_testing = perceptron.predict(testing_input[0])

        print(f'{" ".join(predicted_testing.__str__().split())}')
        print(f'{" ".join(testing_output.__str__().split())}')

        hit = 1 if math.isclose(predicted_testing[0], testing_output[0], abs_tol=0.5) else 0
        testing_guess.append(hit)
        print(f"Testing values guess: {hit}\n")

    mse_training.append(std(mse_training))
    mse_training.append(mean(mse_training))
    testing_guess.append(std(testing_guess))
    testing_guess.append(mean(testing_guess))
    col_0 = list(range(1, 11))
    col_0.append("STD")
    col_0.append("MEAN")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Group", "Training MSE", "Testing Guess"],
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(
                    values=[col_0, mse_training, testing_guess],
                    fill_color='lavender',
                    format=[None, ".5f", ".5f", ".5f", ".5f", ".5f"],
                    align='left')
            )
        ]
    )

    fig.show()
