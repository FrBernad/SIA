import sys

from algorithms.perceptrons import NonLinearPerceptron, SimplePerceptron
from utils.argument_parser import parse_arguments
from utils.config import get_config
from utils.parser_utils import parse_training_values, parse_output_values


def ej2(config_path: str):
    config = get_config(config_path)

    training_values = config.training_values

    if not training_values or training_values.input is None:
        training_values.input = 'training_values/ej2-input.txt'
    if not training_values or training_values.output is None:
        training_values.output = 'training_values/ej2-output.txt'

    input_values = parse_training_values(training_values.input)
    output_values = parse_output_values(training_values.output)

    if config.perceptron.type == "linear":
        linear_perceptron = NonLinearPerceptron(input_values, output_values, config.perceptron.settings)
        linear_perceptron.algorithm()

    elif config.perceptron.type == "non_linear":
        non_linear_perceptron = NonLinearPerceptron(input_values, output_values, config.perceptron.settings)
        non_linear_perceptron.algorithm()


if __name__ == "__main__":

    arguments = parse_arguments(sys.argv[1:])

    config_file = arguments['config_file']

    try:
        ej2(config_file)
    except FileNotFoundError as e:
        print("File not found")
        print(e)
    except OSError:
        print("Error occurred.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
