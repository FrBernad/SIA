from itertools import chain

from numpy import split, array, mean, std
from numpy.random import randint

from algorithms.perceptrons import NonLinearPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values
import plotly.graph_objects as go

if __name__ == "__main__":
    print('Welcome to the non_linear perceptron test')
    config = get_config('../config.yaml')

    training_values = config.training_values
    training_values.input = '../training_values/ej2-input.txt'
    training_values.output = '../training_values/ej2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    mse_percentages_training_errors = []
    mse_percentages_training_std = []
    mse_percentages_prediction_errors = []
    mse_percentages_prediction_std = []
    percentages = []

    for percentage in range(1, 10):
        print(f"\n--- Cross Validation Training {percentage * 10}% - Testing {(10 - percentage) * 10}% ---\n")
        percentages.append(f'{percentage * 10}%')

        mse_training_errors = []
        mse_prediction_errors = []

        for rd in range(2):
            print(f"Round {rd + 1}")
            # SHUFFLE
            # print("Shuffling values")
            for i in range(200):
                rand_index_1 = randint(0, len(input_values))
                rand_index_2 = randint(0, len(input_values))
                input_values[rand_index_1], input_values[rand_index_2] = input_values[rand_index_2], input_values[
                    rand_index_1]
                output_values[rand_index_1], output_values[rand_index_2] = output_values[rand_index_2], output_values[
                    rand_index_1]

            # SPLIT VALUES
            # Voy a buscar el conjunto de prueba que me da el menor error para mi conjunto de practica
            training_input_parts = split(input_values, 10)
            training_output_parts = split(output_values, 10)

            i = 0
            print("Calculating")
            while i + percentage <= len(training_input_parts):
                print(f"{i} ", end=" ")
                training_input = []
                training_output = []
                for k in range(i, i + percentage):
                    training_input.extend(training_input_parts[k])
                    training_output.extend(training_output_parts[k])

                training_input = array(training_input)
                training_output = array(training_output)

                testing_input = []
                testing_output = []
                for k in chain(range(0, i), range(i + percentage, len(training_input_parts))):
                    testing_input.extend(training_input_parts[k])
                    testing_output.extend(training_output_parts[k])
                testing_input = array(testing_input)
                testing_output = array(testing_output)

                perceptron = NonLinearPerceptron(training_input, training_output, config.perceptron.settings)
                results = perceptron.train()

                testing_error = perceptron.plot['e_denormalized'][-1]
                mse_training_errors.append(testing_error)
                # print(f"Training values MSE: {testing_error}")

                # print("Predicted - Expected outputs")
                predicted_testing = perceptron.predict(testing_input)
                # print(f'\n{" ".join(predicted_testing.__str__().split())}')
                # print(f'{" ".join(testing_output.__str__().split())}')

                prediction_error = perceptron.error(predicted_testing, testing_output)[0]
                mse_prediction_errors.append(prediction_error)
                # print(f"Testing values MSE: {prediction_error}\n")

                i += 1

            print("")

        mse_percentages_training_errors.append(mean(mse_training_errors))
        mse_percentages_training_std.append(std(mse_training_errors))

        mse_percentages_prediction_errors.append(mean(mse_prediction_errors))
        mse_percentages_prediction_std.append(std(mse_prediction_errors))

    fig = go.Figure(
        [
            go.Scatter(
                x=percentages,
                y=mse_percentages_training_errors,
                name=f'Training MSE',
                error_y=dict(
                    type='data',
                    array=mse_percentages_training_std,
                    visible=True
                )
            ),
            go.Scatter(
                x=percentages,
                y=mse_percentages_prediction_errors,
                name=f'Testing MSE',
                error_y=dict(
                    type='data',
                    array=mse_percentages_prediction_errors,
                    visible=True
                )
            )
        ]
        ,
        {
            'title': f'Training and testing MSE per training selection size percentage',
            'xaxis_title': "Percentage",
            'yaxis_title': "MSE",

        }
    )
    fig.show()
