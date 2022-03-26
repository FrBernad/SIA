from random import sample
from typing import Tuple

from utils.backpack import Population, Chromosome


def rand_couple_selection(population: Population) -> Tuple[Chromosome, Chromosome]:
    return tuple(sample(tuple(population), k=2))
