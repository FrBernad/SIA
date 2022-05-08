import plotly.graph_objects as go
from numpy import mean, std
from sklearn.model_selection import train_test_split

from algorithms.perceptrons import NonLinearPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values

if __name__ == "__main__":
    print('Welcome to the non_linear best training percentage perceptron test')
    config = get_config('../../config.yaml')

    training_values = config.training_values
    training_values.input = '../../training_values/ej2-input.txt'
    training_values.output = '../../training_values/ej2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_training_values(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    mse_percentages_training = []
    std_percentages_training_mse = []
    mse_percentages_testing = []
    std_percentages_testing_mse = []
    percentages = []

    for percentage in range(1, 10):
        print(f"\n--- Training {percentage * 10}% - Testing {(10 - percentage) * 10}% ---\n")
        percentages.append(f'{percentage * 10}%')

        mse_training_errors = []
        mse_testing_errors = []

        for rd in range(10):
            print(f"Round {rd + 1}")

            # SPLIT VALUES
            train_input, test_input, train_output, test_output = train_test_split(input_values, output_values,
                                                                                  train_size=percentage / 10)

            perceptron = NonLinearPerceptron(train_input, train_output, config.perceptron.settings)
            results = perceptron.train()

            training_error = perceptron.plot['e_denormalized'][-1]
            mse_training_errors.append(training_error)
            # print(f"Training values MSE: {training_error}")

            print("Predicted - Expected outputs")

            predicted_output = perceptron.predict(test_input)
            print(f'\n{" ".join(predicted_output.__str__().split())}')
            print(f'{" ".join(test_output.__str__().split())}')

            testing_error = perceptron.calculate_error(predicted_output, test_output)
            mse_testing_errors.append(testing_error)
            # print(f"Testing values MSE: {testing_error}\n")

        mse_percentages_training.append(mean(mse_training_errors))
        std_percentages_training_mse.append(std(mse_training_errors))

        mse_percentages_testing.append(mean(mse_testing_errors))
        std_percentages_testing_mse.append(std(mse_testing_errors))

    fig = go.Figure(
        [
            go.Scatter(
                x=percentages,
                y=mse_percentages_training,
                name=f'Training MSE',
                error_y=dict(
                    type='data',
                    array=std_percentages_training_mse,
                    visible=True
                )
            ),
            go.Scatter(
                x=percentages,
                y=mse_percentages_testing,
                name=f'Testing MSE',
                error_y=dict(
                    type='data',
                    array=std_percentages_testing_mse,
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
