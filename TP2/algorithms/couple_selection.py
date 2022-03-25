from random import choices
from typing import Tuple

from utils.backpack import Population, Chromosome


def rand_couple_selection(population: Population) -> Tuple[Chromosome, Chromosome]:
    return tuple(choices(population, k=2))