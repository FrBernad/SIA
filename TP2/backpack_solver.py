import csv
from typing import List

from algorithms.algorithm import genetic_algorithm
from algorithms.couple_selection import rand_couple_selection
from algorithms.crossover import simple_crossover, multiple_crossover
from algorithms.mutation import random_mutation
from algorithms.selection import elite_selection
from utils.backpack import Backpack, Element, generate_random_population

CONFIG_FILE = 'config.yaml'
DATA_FILE = 'Mochila100Elementos.txt'
DATA_FILE_DELIMITER = ' '
BENEFIT = 'benefit'
WEIGHT = 'weight'
MAX_CAPACITY = 'max_capacity'
MAX_WEIGHT = 'max_weight'
DATA_FILE_ELEMENTS_FIELDS = [BENEFIT, WEIGHT]
DATA_FILE_BACKPACK_FIELDS = [MAX_CAPACITY, MAX_WEIGHT]


def _get_backpack_data(data_file: str) -> Backpack:
    backpack_elements = _get_backpack_elements(data_file)

    with open(data_file) as df:
        csv_reader = csv.DictReader(df, delimiter=DATA_FILE_DELIMITER, fieldnames=DATA_FILE_BACKPACK_FIELDS)

        backpack_data = csv_reader.__next__()

        return Backpack(int(backpack_data[MAX_CAPACITY]), int(backpack_data[MAX_WEIGHT]), backpack_elements)


def _get_backpack_elements(data_file: str) -> List[Element]:
    with open(data_file) as df:
        csv_reader = csv.DictReader(df, delimiter=DATA_FILE_DELIMITER, fieldnames=DATA_FILE_ELEMENTS_FIELDS)

        csv_reader.__next__()

        elements = []

        for element in csv_reader:
            elements.append(Element(int(element[BENEFIT]), int(element[WEIGHT])))

        return elements


# def _get_config(config_file: str) -> 'Config':
#     with open(config_file) as cf:
#         config = yaml.safe_load(cf)["config"]
#         return Config(config.get("algorithm"), config.get("limit"), config.get("heuristic"))


def main(data_file: str, config_file: str):
    # config = _get_config(config_file)
    backpack = _get_backpack_data(data_file)

    first_generation = generate_random_population(backpack.max_capacity)
    genetic_algorithm(first_generation, backpack, rand_couple_selection,
                      multiple_crossover, random_mutation, elite_selection)


if __name__ == '__main__':
    config_file = CONFIG_FILE
    data_file = DATA_FILE

    try:
        main(data_file, config_file)
    except OSError:
        print("Error opening config file.")
    except Exception as e:
        print(e)
