from math import exp
from random import sample, choices, uniform, random
from typing import List

from utils.chromosome_factory import Population, Chromosome
from utils.config import SelectionMethodConfig

DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT = 4


def elitism_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    return sorted(population,
                  key=lambda chromosome: chromosome.fitness,
                  reverse=True
                  )[0:selection_size]


def roulette_wheel_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    fitness = list(map(lambda chromosome: chromosome.fitness, population))
    total_fitness = sum(fitness)

    relative_fitness = list(map(lambda f: f / total_fitness, fitness))

    return _roulette_accumulated_selection(population, selection_size,
                                           _calculate_accumulated_probability(relative_fitness))


def rank_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
):
    sorted_population = sorted(population,
                               key=lambda chromosome: chromosome.fitness,
                               reverse=True)

    population_len = len(population)
    accum_prob = [1]

    for i in range(population_len):
        accum_prob.append((population_len - (i + 1)) / population_len)

    # FIXME: esto esta raro hay q ver q suban el arreglo a slack
    return _rank_accumulated_selection(sorted_population, selection_size, accum_prob)


def tournament_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    new_population = set()

    while len(new_population) < selection_size:
        couples = sample(tuple(population), k=DEFAULT_TOURNAMENT_CHROMOSOME_AMOUNT)
        u = uniform(0, 1)
        r = random()

        first_pick = _tournament_picker(couples[0], couples[1], u, r)

        second_pick = _tournament_picker(couples[2], couples[3], u, r)
        winner = _tournament_picker(first_pick, second_pick, u, r)

        new_population.add(winner)

    return list(new_population)


def boltzmann_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
):
    fitness = list(map(lambda chromosome: chromosome.fitness / 100, population))
    tc = config.Tc
    t0 = config.T0
    k = config.k

    temp: float = tc + (t0 - tc) * exp(-k * generation_count)
    ve_num_values = list(map(lambda f: exp(f / temp), fitness))

    ve = list(map(lambda num: num / sum(ve_num_values[:ve_num_values.index(num) + 1]), ve_num_values))
    total_ve = sum(ve)

    relative_ve = list(map(lambda v: v / total_ve, ve))

    return _roulette_accumulated_selection(population, selection_size, _calculate_accumulated_probability(relative_ve))


def truncated_selection(
        population: Population,
        generation_count: int,
        selection_size: int,
        config: SelectionMethodConfig
) -> Population:
    truncated_population = sorted(population,
                                  key=lambda chromosome: chromosome.fitness,
                                  )[config.truncation_size - 1::]

    return sample(truncated_population, k=selection_size)


def _tournament_picker(first: Chromosome, second: Chromosome, u: float, r: float) -> Chromosome:
    first_fitness = first.fitness
    second_fitness = second.fitness

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


def _rank_accumulated_selection(
        population: Population,
        selection_size: int,
        accum_probabilities: List[float]
) -> Population:
    new_population = set()

    while len(new_population) < selection_size:
        r = random()

        for i in range(len(population)):
            if accum_probabilities[i] >= r > accum_probabilities[i + 1]:
                new_population.add(population[i])
                break

    return list(new_population)


SELECTION_METHODS = {
    'elitism_selection': elitism_selection,
    'roulette_wheel_selection': roulette_wheel_selection,
    'rank_selection': rank_selection,
    'tournament_selection': tournament_selection,
    'boltzmann_selection': boltzmann_selection,
    'truncated_selection': truncated_selection
}
