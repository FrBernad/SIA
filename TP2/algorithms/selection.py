import random
from random import sample

from utils.backpack import Population, Backpack, Chromosome

CHROMOSOME_AMOUNT = 4


def elite_selection(population: Population, backpack: Backpack, selection_size: int) -> Population:
    return sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0:selection_size]


def truncation_selection(population: Population, backpack: Backpack, truncation_size: int,
                         selection_size: int) -> Population:
    population_len = len(population)
    truncation_index = population_len - truncation_size

    temp = sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0: truncation_index]

    return sample(tuple(temp), k=selection_size)


def tournament_selection(population: Population, backpack: Backpack, selection_size: int) -> Population:
    new_population = []
    for i in range(selection_size):
        couples = sample(tuple(population), k=CHROMOSOME_AMOUNT)

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
