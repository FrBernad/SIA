from numpy import split, array
from numpy.random import randint

from algorithms.perceptrons import MultiLayerPerceptron
from utils.config import get_config
from utils.parser_utils import parse_nums, parse_output_values

if __name__ == "__main__":
    print('Welcome to the multilayer perceptron parity test')
    config = get_config('../config.yaml')

    training_values = config.training_values
    training_values.input = '../training_values/ej3-2-input.txt'
    training_values.output = '../training_values/ej3-2-output.txt'

    print(f'parsing input file: {training_values.input}')
    input_values = parse_nums(training_values.input)
    print(f'parsing output file: {training_values.output}')
    output_values = parse_output_values(training_values.output)

    print("Shuffling values...")
    for i in range(50):
        rand_index_1 = randint(0, len(input_values))
        rand_index_2 = randint(0, len(input_values))
        input_values[rand_index_1], input_values[rand_index_2] = input_values[rand_index_2], input_values[
            rand_index_1]
        output_values[rand_index_1], output_values[rand_index_2] = output_values[rand_index_2], output_values[
            rand_index_1]

    training_input_parts = split(input_values, 10)
    training_output_parts = split(output_values, 10)

    for i in range(len(training_input_parts)):
        training_input = []
        training_output = []
        practice_input = training_input_parts[i]
        practice_output = training_output_parts[i]
        for k in range(len(training_input_parts)):
            if k != i:
                training_input.extend(training_input_parts[k])
                training_output.extend(training_output_parts[k])

        print(f"Training round {i}")
        perceptron = MultiLayerPerceptron(array(training_input), [6], array(training_output),
                                          config.perceptron.settings)
        perceptron.train()
        predicted_outputs = []
        for j in range(len(practice_input)):
            predicted_outputs.append(perceptron.predict(practice_input[j]))
            print(f'Expected: \n{practice_output[j]}')
            print(f'Predicted: \n{predicted_outputs[-1]}')
        print(f'Error: {perceptron.calculate_error(array(practice_input), array(practice_output))}\n')
