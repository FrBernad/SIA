from random import sample, choices, uniform, random
from typing import List

from utils.backpack import Population, Backpack, Chromosome
from utils.config import SelectionMethodConfig

DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT = 4


def elitism_selection(
        population: Population,
        backpack: Backpack,
        selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    return sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0:selection_size]


def rank_selection(
        population: Population,
        backpack: Backpack,
        selection_size: int,
        config: SelectionMethodConfig
):
    sorted_population = sorted(population,
                               key=lambda chromosome: backpack.calculate_fitness(chromosome),
                               reverse=True)

    population_len = len(population)
    accum_prob = [1]

    for i in range(population_len):
        accum_prob.append((population_len - (i + 1)) / population_len)

    return _rank_accumulated_selection(sorted_population, selection_size, accum_prob)

def _rank_accumulated_selection(
        population: Population,
        selection_size: int,
        accum_probabilities: List[float]
) -> Population:
    new_population = set()
    # 0,5 0.54 0.6 .55
    while len(new_population) < selection_size:
        r = random()

        for i in range(len(population)):
            if accum_probabilities[i] >= r > accum_probabilities[i + 1]:
                new_population.add(population[i])
                break

    return list(new_population)

def boltzmann_selection(population: Population, backpack: Backpack, config: SelectionMethodConfig):
    pass


def truncated_selection(
        population: Population,
        backpack: Backpack,
        config: SelectionMethodConfig
) -> Population:
    population_len = len(population)
    truncation_index = population_len - config.truncation_size

    temp = sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0: truncation_index]

    return sample(tuple(temp), k=population_len)


def tournament_selection(
        population: Population,
        backpack: Backpack,
        selection_size: int
) -> Population:
    new_population = []
    for i in range(selection_size):
        couples = sample(tuple(population), k=DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT)

        first_pick = _tournament_picker(backpack, couples[0], couples[1])

        second_pick = _tournament_picker(backpack, couples[2], couples[3])

        new_population.append(_tournament_picker(backpack, first_pick, second_pick))

    return new_population


def _tournament_picker(backpack: Backpack, first: Chromosome, second: Chromosome) -> Chromosome:
    u = uniform(0.5, 1)
    r = random()

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


def roulette_wheel_selection(
        population: Population,
        backpack: Backpack, selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    fitness = list(map(lambda chromosome: backpack.calculate_fitness(chromosome), population))
    total_fitness = sum(fitness)

    relative_fitness = list(map(lambda f: f / total_fitness, fitness))

    return _roulette_accumulated_selection(population, selection_size, _calculate_accumulated_probability(relative_fitness))


def _roulette_accumulated_selection(
        population: Population,
        selection_size: int,
        accum_probabilities: List[float]
) -> Population:
    new_population = set()

    while len(new_population) < selection_size:
        r = random()

        for i in range(len(population)):
            if accum_probabilities[i] < r <= accum_probabilities[i + 1]:
                new_population.add(population[i])
                break

    return list(new_population)


def _calculate_accumulated_probability(probabilities: List[float]) -> List[float]:
    accum = 0
    accum_values = [accum]
    for p in probabilities:
        accum += p
        accum_values.append(accum)

    return accum_values


def roulette_algorithm(population: Population, max_val: float, fitnesses_values: List[int]) -> Chromosome:
    current: int = 0
    pick: float = uniform(0, max_val)
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
