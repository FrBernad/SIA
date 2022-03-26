from itertools import islice

from utils.backpack import Population, Backpack


def elite_selection(population: Population, backpack: Backpack, selection_size: int) -> Population:
    return sorted(population,
                  key=lambda chromosome: backpack.calculate_fitness(chromosome),
                  reverse=True
                  )[0:selection_size]
