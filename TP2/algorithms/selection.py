from random import sample

from utils.backpack import Population, Backpack


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
