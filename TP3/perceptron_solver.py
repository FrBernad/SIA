from algorithms.activation_functions import ACTIVATION_FUNCTIONS
from algorithms.delta_w_functions import DELTA_W_FUNCTIONS
from algorithms.simple_perceptron import simple_perceptron_algorithm
from utils.parser_utils import parse_training_values, parse_output_values


def main():
    training_values = parse_training_values('data/ej1-and-training.txt')
    output_values = parse_output_values('data/ej1-and-output.txt')

    simple_perceptron_algorithm(training_values, output_values,
                                ACTIVATION_FUNCTIONS['simple_perceptron'], DELTA_W_FUNCTIONS['simple_perceptron'],
                                len(training_values[0]) - 1, len(training_values), 0.01, 5000)


if __name__ == "__main__":
    main()
