import yaml
from pydantic import ValidationError

from algorithms.activation_functions import ACTIVATION_FUNCTIONS
from algorithms.delta_w_functions import DELTA_W_FUNCTIONS
from algorithms.simple_perceptron import simple_perceptron_algorithm
from utils.config import Config
from utils.parser_utils import parse_training_values, parse_output_values


def _get_config(config_file: str) -> Config:
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        try:
            return Config(**config)
        except ValidationError as e:
            print(e.json())


def main():
    training_values = parse_training_values('training_values/ej2-input.txt')
    output_values = parse_output_values('training_values/ej2-output.txt')

    config = _get_config('config.yaml')

    simple_perceptron_algorithm(training_values, output_values,
                                ACTIVATION_FUNCTIONS[config.perceptron],
                                DELTA_W_FUNCTIONS[config.perceptron],
                                len(training_values[0]) - 1, len(training_values),
                                config.perceptron.settings)


if __name__ == "__main__":
    main()
