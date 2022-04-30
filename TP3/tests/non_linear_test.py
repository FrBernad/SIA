from numpy import split, array, mean
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

    if config.perceptron.type == "non_linear":
        errors = []
        min_errors = []
        best_error = []
        prediction_errors = []
        for j in range(6):
            print(f"\n\n\nCross Validation number {j + 1}")
            for i in range(50):
                rand_index_1 = randint(0, len(input_values))
                rand_index_2 = randint(0, len(input_values))
                input_values[rand_index_1], input_values[rand_index_2] = input_values[rand_index_2], input_values[
                    rand_index_1]
                output_values[rand_index_1], output_values[rand_index_2] = output_values[rand_index_2], output_values[
                    rand_index_1]

            # Voy a buscar el conjunto de prueba que me da el menor error para mi conjunto de practica
            training_input_parts = split(input_values, 10)
            training_output_parts = split(output_values, 10)
            best_prediction_error = 500
            for i in range(len(training_input_parts)):
                training_input = []
                training_output = []
                practice_input = training_input_parts[i]
                practice_output = training_output_parts[i]
                for k in range(len(training_input_parts)):
                    if k != i:
                        training_input.extend(training_input_parts[k])
                        training_output.extend(training_output_parts[k])

                perceptron = NonLinearPerceptron(array(training_input), array(training_output),
                                                 config.perceptron.settings)
                results = perceptron.train()
                e = array(perceptron.plot['e_denormalized'])
                errors.append(e)
                min_errors.append(e.min())
                print(f"Training min denormalized error: {e.min()}")
                print(f"Training min normalized error: {array(perceptron.plot['e_normalized']).min()}")

                predicted_practice = perceptron.predict(practice_input)
                print(f'{" ".join(perceptron.predict(practice_input).__str__().split())}')
                print(f'{" ".join(practice_output.__str__().split())}')
                prediction_error = perceptron.error(predicted_practice, practice_output)[0]
                prediction_errors.append(prediction_error)
                print(
                    f"Predicted values mean cuadratic error: {prediction_error}\n")
                if prediction_error < best_prediction_error:
                    best_error = e
                    best_prediction_error = prediction_error

        errors = array(errors)
        fig = go.Figure(
            [
                go.Scatter(
                    y=mean(errors, axis=0),
                    name=f'Error Medio',
                ),
                go.Scatter(
                    y=best_error,
                    name=f'Error asociado a mejor metrica',
                )
            ]
            ,
            {
                'title': f'Error Denormalized',
            }
        )
        fig.show()

        fig = go.Figure(
            [
                go.Scatter(
                    x=min_errors,
                    y=prediction_errors,
                    mode='markers',
                    marker={'size': 15}
                )
            ]
            ,
            {
                'title': f'Error Denormalized',
                'xaxis_title': "Error",
                'yaxis_title': "Metric",

            }
        )
        fig.show()
