from random import random

from utils.backpack import Chromosome

DEFAULT_PROBABILITY = 0.005


def random_mutation(
        chromosome: Chromosome,
        probability: float = DEFAULT_PROBABILITY
) -> Chromosome:
    return tuple(map(lambda allele: allele if random() < probability else not allele, list(chromosome)))
