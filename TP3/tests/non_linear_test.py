from numpy import split, array
from numpy.random import randint

from algorithms.perceptrons import NonLinearPerceptron
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values

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
        for i in range(50):
            rand_index_1 = randint(0, len(input_values))
            rand_index_2 = randint(0, len(input_values))
            input_values[rand_index_1], input_values[rand_index_2] = input_values[rand_index_2], input_values[
                rand_index_1]
            output_values[rand_index_1], output_values[rand_index_2] = output_values[rand_index_2], output_values[
                rand_index_1]

        training_input_parts = split(input_values, 10)
        training_output_parts = split(output_values, 10)
        for j in range(len(training_input_parts)):
            training_input = []
            training_output = []
            practice_input = training_input_parts[j]
            practice_output = training_output_parts[j]
            for k in range(len(training_input_parts)):
                if k != j:
                    training_input.extend(training_input_parts[k])
                    training_output.extend(training_output_parts[k])

            perceptron = NonLinearPerceptron(array(training_input), array(training_output), config.perceptron.settings)
            results = perceptron.train()
            print(f"Training min denormalized error: {array(perceptron.plot['e_denormalized']).min()}")
            print(f"Training min normalized error: {array(perceptron.plot['e_normalized']).min()}")
            predicted_practice = perceptron.predict(practice_input)
            print(f"Training min denormalized error: {perceptron.error(predicted_practice, practice_output)}\n")
            # print(f'{" ".join(perceptron.predict(practice_input).__str__().split())}')
            # print(f'{" ".join(practice_output.__str__().split())}')
