import random
from random import sample
from typing import List

from utils.backpack import Population, Backpack, Chromosome
from utils.config import SelectionMethodConfig

DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT = 4


def elitism_selection(
        population: Population, backpack: Backpack,
        population_size: int, config: SelectionMethodConfig
) -> Population:
    return sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0:population_size]


def roulette_wheel_selection(population: Population, backpack: Backpack, config: SelectionMethodConfig):
    pass


# FIXME: la seleccion esta mal
def rank_selection(
        population: Population, backpack: Backpack,
        population_size: int,
        config: SelectionMethodConfig
):
    sorted_population = sorted(population,
                               key=lambda chromosome: backpack.calculate_fitness(chromosome),
                               reverse=True)

    population_len = len(population)
    population_weight = []

    for i in range(population_len):
        population_weight.append((population_len - (i + 1)) / population_len)

    # FIXME Sacar 1000 hardcodeado
    return random.choices(population=sorted_population, weights=population_weight, k=1000)


def boltzmann_selection(population: Population, backpack: Backpack, config: SelectionMethodConfig):
    pass


def truncated_selection(population: Population, backpack: Backpack, config: SelectionMethodConfig) -> Population:
    population_len = len(population)
    truncation_index = population_len - config.truncation_size

    temp = sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0: truncation_index]

    return sample(tuple(temp), k=config.sample_size)


def tournament_selection(population: Population, backpack: Backpack, selection_size: int) -> Population:
    new_population = []
    for i in range(selection_size):
        couples = sample(tuple(population), k=DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT)

        first_pick = _tournament_picker(backpack, couples[0], couples[1])

        second_pick = _tournament_picker(backpack, couples[2], couples[3])

        new_population.append(_tournament_picker(backpack, first_pick, second_pick))

    return new_population


def _tournament_picker(backpack: Backpack, first: Chromosome, second: Chromosome) -> Chromosome:
    u = random.uniform(0.5, 1)
    r = random.random()

    first_fitness = backpack.calculate_fitness(first)
    second_fitness = backpack.calculate_fitness(second)

    if r < u:
        if max(first_fitness, second_fitness) == first_fitness:
            return first
        else:
            return second
    else:
        if min(first_fitness, second_fitness) == first_fitness:
            return first
        else:
            return second


def roulette_wheel_selection(population: Population, backpack: Backpack, selection_size: int) -> Population:
    selected_ones = set()
    max_val: float = sum(backpack.calculate_fitness(chromosome) for chromosome in population)
    fitneses_values = list()

    for chrm in population:
        fitneses_values.append(backpack.calculate_fitness(chrm))

    while len(selected_ones) < selection_size:
        selected_ones.add(roulette_algorithm(population, max_val, fitneses_values))

    return list(selected_ones)


def roulette_algorithm(population: Population, max_val: float, fitnesses_values: List[int]) -> Chromosome:
    current: int = 0
    pick: float = random.uniform(0, max_val)
    for chromosome in population:
        current += fitnesses_values[population.index(chromosome)]
        if current > pick:
            return chromosome


SELECTION_METHODS = {
    'elitism_selection': elitism_selection,
    'roulette_wheel_selection': roulette_wheel_selection,
    'rank_selection': rank_selection,
    'tournament_selection': tournament_selection,
    'boltzmann_selection': boltzmann_selection,
    'truncated_selection': truncated_selection
}
