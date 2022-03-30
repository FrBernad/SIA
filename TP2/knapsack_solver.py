import csv
import sys
from typing import List, Callable

import yaml

from algorithms.algorithm import genetic_algorithm
from utils.argument_parser import parse_arguments
from utils.backpack import Backpack, Element, generate_random_population
from utils.config import Config

DATA_FILE_DELIMITER = ' '
BENEFIT = 'benefit'
WEIGHT = 'weight'
MAX_CAPACITY = 'max_capacity'
MAX_WEIGHT = 'max_weight'
DATA_FILE_ELEMENTS_FIELDS = [BENEFIT, WEIGHT]
DATA_FILE_BACKPACK_FIELDS = [MAX_CAPACITY, MAX_WEIGHT]


def _get_backpack_data(data_file: str, fitness_function: Callable) -> Backpack:
    backpack_elements = _get_backpack_elements(data_file)

    with open(data_file) as df:
        csv_reader = csv.DictReader(df, delimiter=DATA_FILE_DELIMITER, fieldnames=DATA_FILE_BACKPACK_FIELDS)

        backpack_data = csv_reader.__next__()

        return Backpack(int(backpack_data[MAX_CAPACITY]), int(backpack_data[MAX_WEIGHT]),
                        fitness_function, backpack_elements)


def _get_backpack_elements(data_file: str) -> List[Element]:
    with open(data_file) as df:
        csv_reader = csv.DictReader(df, delimiter=DATA_FILE_DELIMITER, fieldnames=DATA_FILE_ELEMENTS_FIELDS)

        csv_reader.__next__()

        elements = []

        for element in csv_reader:
            elements.append(Element(int(element[BENEFIT]), int(element[WEIGHT])))

        return elements


def _get_config(config_file: str) -> Config:
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        return Config.generate(config)


def main(data_file: str, config_file: str, output_file: str):
    config = _get_config(config_file)
    backpack = _get_backpack_data(data_file, config.fitness_function)

    first_generation = generate_random_population(backpack, config.initial_population_size)

    stats = genetic_algorithm(first_generation, backpack, config.couple_selection_method,
                              config.crossover_method_config.method, config.mutation_method_config.method,
                              config.selection_method_config.method, config)

    print(stats.end_condition)
    print(len(stats.get_best_solutions_stats()))


if __name__ == '__main__':
    arguments = parse_arguments(sys.argv[1:])

    config_file = arguments['config_file']
    data_file = arguments['data_file']
    output_file = arguments['output_file']

    try:
        main(data_file, config_file, output_file)
    except OSError:
        print("Error opening config file.")
    except Exception as e:
        print(e)