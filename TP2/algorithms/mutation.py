from random import random

from utils.backpack import Chromosome

DEFAULT_PROBABILITY = 0.004


def random_mutation(
        chromosome: Chromosome,
        probability: float = DEFAULT_PROBABILITY
) -> Chromosome:
    return tuple(map(lambda allele: not allele if random() < probability else allele, list(chromosome)))
