import csv
import sys
from typing import List

import yaml

from algorithms.algorithm import genetic_algorithm
from utils.argument_parser import parse_arguments
from utils.chromosome_factory import ChromosomeFactory
from utils.config import Config
from utils.knapsack import Knapsack, Element
from utils.results import generate_solution_yaml

DATA_FILE_DELIMITER = ' '
BENEFIT = 'benefit'
WEIGHT = 'weight'
MAX_CAPACITY = 'max_capacity'
MAX_WEIGHT = 'max_weight'
DATA_FILE_ELEMENTS_FIELDS = [BENEFIT, WEIGHT]
DATA_FILE_KNAPSACK_FIELDS = [MAX_CAPACITY, MAX_WEIGHT]


def _get_knapsack_data(data_file: str) -> Knapsack:
    knapsack_elements = _get_knapsack_elements(data_file)

    with open(data_file) as df:
        csv_reader = csv.DictReader(df, delimiter=DATA_FILE_DELIMITER, fieldnames=DATA_FILE_KNAPSACK_FIELDS)

        knapsack_data = csv_reader.__next__()

        return Knapsack(int(knapsack_data[MAX_CAPACITY]), int(knapsack_data[MAX_WEIGHT]), knapsack_elements)


def _get_knapsack_elements(data_file: str) -> List[Element]:
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


def _generate_solution_file(solution, sol_file):
    with open(sol_file, 'w') as f:
        yaml.dump(solution, f, sort_keys=False, default_flow_style=False)


def main(data_file: str, config_file: str, output_file: str):
    print('--- WELCOME TO THE 0-1 KNAPSACK PROBLEM SOLVER ---')

    print('parsing config file...')
    config = _get_config(config_file)

    print('generating generation 0...')
    knapsack = _get_knapsack_data(data_file)

    chromosome_factory = ChromosomeFactory(knapsack, config.fitness_function)
    first_generation = chromosome_factory.generate_random_population(config.initial_population_size)

    print('calculating solution...')
    stats = genetic_algorithm(first_generation, chromosome_factory, config.couple_selection_method,
                              config.crossover_method_config.method, config.mutation_method_config.method,
                              config.selection_method_config.method, config)

    print('solution calculated...')

    print('generating output file...')
    _generate_solution_file(generate_solution_yaml(stats, config), output_file)
    print(f'\nSolution generated, find it inside {output_file} file')


if __name__ == '__main__':
    arguments = parse_arguments(sys.argv[1:])

    config_file = arguments['config_file']
    data_file = arguments['data_file']
    output_file = arguments['output_file']

    try:
        main(data_file, config_file, output_file)
    except OSError:
        print("Error opening config file.")
    except KeyboardInterrupt:
        print('Program interrupted by user.')
    except Exception as e:
        print(e)
