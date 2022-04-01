from random import sample, choices
from typing import Tuple

from utils.chromosome_factory import Population, Chromosome


def rand_couple_selection(population: Population) -> Tuple[Chromosome, ...]:
    return tuple(sample(population, k=2))


def fitness_couple_selection(population: Population) -> Tuple[Chromosome, ...]:
    return tuple(choices(population, k=2, weights=list(map(lambda chr: chr.fitness, population))))


COUPLE_SELECTION_METHODS = {
    'rand_couple_selection': rand_couple_selection,
    'fitness_couple_selection': fitness_couple_selection
}
