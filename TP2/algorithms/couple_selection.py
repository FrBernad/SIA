from random import sample
from typing import Tuple

from utils.backpack import Population, Chromosome


def rand_couple_selection(population: Population) -> Tuple[Chromosome, Chromosome]:
    return tuple(sample(tuple(population), k=2))


def fitness_couple_selection(population: Population) -> Tuple[Chromosome, Chromosome]:
    # FIXME:
    pass


COUPLE_SELECTION_METHODS = {
    'rand_couple_selection': rand_couple_selection,
    'fitness_couple_selection': fitness_couple_selection
}
